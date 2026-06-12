//! `IrGraph` -> serialize `Graph` converter (round-trip elimination, Phase 1).
//!
//! Builds the serialize `Graph` directly from the post-pass `IrGraph`, mirroring
//! the Python-sourced node loop in `ingest_fx_graph_with_prefix` but reading
//! `IrNode` fields instead of Python attributes -- so serialization can skip the
//! flush-to-Python + re-ingest round-trip entirely. Gated behind
//! `ARM_SERIALIZE_FROM_IR` during bring-up and validated by running the full test
//! battery with the flag on (the battery is the equality check -- no separate
//! harness).
//!
//! NOT yet handled here (see round-trip Phase 2/3 tasks): TensorStore population
//! from `IrNode.tensor`, and the control-flow submodule hierarchy. The
//! post-conversion fixups (`fixup_reshape_shapes` / `fixup_quantized_dtypes` /
//! `preserve_scalar_shape` marking) still run after this, exactly as after ingest.
#![allow(dead_code)] // Phase 1: not yet wired into the active serialize path.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::fx_control_flow_metadata::{NodeOp, QParam};
use crate::fx_ir::{IrArg, IrGraph, MetaValue, NodeId as IrNodeId, NodeMeta as IrNodeMeta};
use crate::graph::ir::{Arg, DType, Node, NodeId, NodeMeta, OpTarget, OpType, QuantArgs, TensorInfo};
use crate::graph::Graph;
use crate::ingest::target_map::parse_op_target;
use crate::tensor_data::parse_torch_dtype;

/// Build a serialize `Graph` from a post-pass `IrGraph`. Handles one flat graph;
/// the control-flow submodule hierarchy is assembled by the caller (Phase 3).
pub fn graph_from_ir(ir: &IrGraph) -> Graph {
    let mut graph = Graph::new();
    let mut idmap: HashMap<IrNodeId, NodeId> = HashMap::new();

    // Pass 1: allocate serialize nodes (args resolved in pass 2) and build the
    // IrNodeId -> serialize-NodeId map.
    for ir_node in ir.nodes.iter().filter(|n| n.alive) {
        let op = map_op(ir_node.op);
        let target = if op == OpType::CallFunction {
            parse_op_target(&ir_node.target)
        } else {
            OpTarget::Other(ir_node.target.clone())
        };
        let node = Node {
            op,
            target,
            args: SmallVec::new(),
            kwargs: Vec::new(),
            meta: map_meta(&ir_node.meta),
            users: SmallVec::new(),
            name: ir_node.name.clone(),
        };
        let nid = graph.nodes.insert(node);
        idmap.insert(ir_node.id, nid);
        graph.order.push(nid);
        graph.node_names.push((nid, ir_node.name.clone()));
        // Classify by the ORIGINAL NodeOp: only true placeholders are runtime
        // graph inputs. GetAttr nodes map to OpType::Placeholder too, but they
        // are CONSTANTS (their data lives in the TensorStore by name) and must
        // NOT join `inputs`, or the serializer emits them as runtime inputs and
        // the input binding misaligns (-> ref-model segfault).
        match ir_node.op {
            NodeOp::Placeholder => graph.inputs.push(nid),
            NodeOp::Output => graph.output = Some(nid),
            _ => {}
        }
    }

    // Pass 2: resolve args/kwargs through the id map.
    for ir_node in ir.nodes.iter().filter(|n| n.alive) {
        let Some(&nid) = idmap.get(&ir_node.id) else {
            continue;
        };
        let args: SmallVec<[Arg; 4]> = ir_node.args.iter().map(|a| map_arg(a, &idmap)).collect();
        let kwargs: Vec<(String, Arg)> = ir_node
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), map_arg(v, &idmap)))
            .collect();
        if let Some(node) = graph.nodes.get_mut(nid) {
            node.args = args;
            node.kwargs = kwargs;
        }
    }

    // The IrGraph's node Vec is in pass-INSERTION order: passes append new nodes
    // (transposes, rescales) at the end rather than at their topological
    // position, so the Vec is not topologically ordered. The serializer (and the
    // ref model) require topological order, so re-sort `order` by def-use.
    graph.order = topo_order(&graph);
    graph.rebuild_user_lists();
    graph
}

/// Topological order of the graph's nodes by def-use (each node after the nodes
/// its args reference). Iterative DFS post-order -- no recursion, so it cannot
/// itself stack-overflow on deep graphs.
fn topo_order(graph: &Graph) -> Vec<NodeId> {
    let mut visited: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(graph.order.len());
    for &start in &graph.order {
        if visited.contains(&start) {
            continue;
        }
        let mut stack: Vec<(NodeId, bool)> = vec![(start, false)];
        while let Some((nid, post)) = stack.pop() {
            if post {
                result.push(nid);
                continue;
            }
            if !visited.insert(nid) {
                continue;
            }
            stack.push((nid, true));
            if let Some(n) = graph.nodes.get(nid) {
                for dep in Graph::collect_input_ids(n) {
                    if !visited.contains(&dep) {
                        stack.push((dep, false));
                    }
                }
            }
        }
    }
    result
}

/// Env-gated structural dump for comparing a converted graph against the
/// Python-ingested one (ARM_FROM_IR_DEBUG=1).
pub fn debug_dump(graph: &Graph, label: &str) {
    eprintln!(
        "=== {label}: {} nodes inputs={:?} output={:?} ===",
        graph.order.len(),
        graph.inputs,
        graph.output,
    );
    for &nid in &graph.order {
        if let Some(n) = graph.nodes.get(nid) {
            eprintln!(
                "  {:?} {} op={:?} target={:?} args={:?} val={:?}:{:?} dimorder={:?}",
                nid,
                n.name,
                n.op,
                n.target,
                n.args,
                n.meta.val.shape_slice(),
                n.meta.val.dtype,
                n.meta.tosa_dim_order,
            );
        }
    }
}

fn map_op(op: NodeOp) -> OpType {
    match op {
        NodeOp::Placeholder | NodeOp::GetAttr => OpType::Placeholder,
        NodeOp::Output => OpType::Output,
        NodeOp::CallFunction | NodeOp::Other => OpType::CallFunction,
    }
}

fn map_arg(a: &IrArg, idmap: &HashMap<IrNodeId, NodeId>) -> Arg {
    match a {
        IrArg::Node(id) => idmap.get(id).map(|&n| Arg::Node(n)).unwrap_or(Arg::None),
        IrArg::Int(v) => Arg::Int(*v),
        IrArg::Float(v) => Arg::Float(*v),
        IrArg::Bool(v) => Arg::Bool(*v),
        IrArg::Str(s) => Arg::String(s.clone()),
        IrArg::Dtype(s) => Arg::Dtype(parse_torch_dtype(s)),
        IrArg::None => Arg::None,
        IrArg::List(items) => map_list(items, idmap),
    }
}

/// serialize `Arg` lists are typed (`NodeList`/`IntList`/`FloatList`), unlike the
/// general `IrArg::List`. Classify by element type.
fn map_list(items: &[IrArg], idmap: &HashMap<IrNodeId, NodeId>) -> Arg {
    if items.is_empty() {
        return Arg::IntList(SmallVec::new());
    }
    if items.iter().all(|x| matches!(x, IrArg::Node(_))) {
        let ids = items
            .iter()
            .filter_map(|x| match x {
                IrArg::Node(id) => idmap.get(id).copied(),
                _ => None,
            })
            .collect();
        return Arg::NodeList(ids);
    }
    if items.iter().all(|x| matches!(x, IrArg::Int(_) | IrArg::Bool(_))) {
        let v = items
            .iter()
            .map(|x| match x {
                IrArg::Int(i) => *i,
                IrArg::Bool(b) => *b as i64,
                _ => 0,
            })
            .collect();
        return Arg::IntList(v);
    }
    if items.iter().all(|x| matches!(x, IrArg::Float(_) | IrArg::Int(_))) {
        let v = items
            .iter()
            .map(|x| match x {
                IrArg::Float(f) => *f,
                IrArg::Int(i) => *i as f64,
                _ => 0.0,
            })
            .collect();
        return Arg::FloatList(v);
    }
    // Mixed/optional lists (e.g. symint lists with Nones) are rare in lowered
    // graphs; empty for now, refine if the battery surfaces one.
    Arg::IntList(SmallVec::new())
}

fn map_meta(m: &IrNodeMeta) -> NodeMeta {
    let val = match &m.val {
        Some(ft) => TensorInfo::new(&ft.shape, parse_torch_dtype(&ft.dtype)),
        None => TensorInfo::new(&[], DType::Float32),
    };
    let mut meta = NodeMeta::new(val);
    meta.input_qparams = qparam_entries(&m.input_qparams);
    meta.output_qparams = qparam_entries(&m.output_qparams);
    meta.tosa_dim_order = m
        .tosa_dim_order
        .as_ref()
        .map(|v| v.iter().map(|&x| x as u8).collect());
    meta.tosa_special_dtype = special_dtype_from_extras(m);
    meta.arm_control_flow_parent_output_user = matches!(
        m.extras.get("arm_control_flow_parent_output_user"),
        Some(MetaValue::Bool(true))
    );
    meta
}

fn qparam_entries<A>(m: &Option<HashMap<usize, QParam>>) -> SmallVec<A>
where
    A: smallvec::Array<Item = (u32, QuantArgs)>,
{
    let Some(map) = m else {
        return SmallVec::new();
    };
    let mut entries: Vec<(&usize, &QParam)> = map.iter().collect();
    entries.sort_by_key(|(idx, _)| **idx);
    entries
        .into_iter()
        .map(|(idx, qp)| (*idx as u32, qparam_to_quantargs(qp)))
        .collect()
}

fn qparam_to_quantargs(qp: &QParam) -> QuantArgs {
    QuantArgs {
        scale: qp.scale,
        zero_point: qp.zp,
        dtype: parse_torch_dtype(&qp.dtype_str),
        qmin: qp.qmin,
        qmax: qp.qmax,
    }
}

fn special_dtype_from_extras(m: &IrNodeMeta) -> Option<DType> {
    match m.extras.get("tosa_special_dtype") {
        Some(MetaValue::Str(s)) => {
            let s = s.to_uppercase();
            if s.contains("48") {
                Some(DType::Int48)
            } else if s.contains('4') {
                Some(DType::Int4)
            } else {
                None
            }
        }
        _ => None,
    }
}
