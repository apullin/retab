//! `IrGraph` -> serialize `Graph` converter.
//!
//! Builds the serialize `Graph` directly from the post-pass `IrGraph`; control-flow
//! submodule graphs are assembled from their own stashed IRs by the caller.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::fx_control_flow_metadata::{NodeOp, QParam};
use crate::fx_ep_effects::{FxEpEffect, FxEpEffectBatch};
use crate::fx_ir::{IrArg, IrGraph, MetaValue, NodeId as IrNodeId, NodeMeta as IrNodeMeta};
use crate::graph::ir::{
    Arg, DType, Node, NodeId, NodeMeta, OpTarget, OpType, QuantArgs, TensorInfo,
};
use crate::graph::Graph;
use crate::ingest::target_map::parse_op_target;
use crate::store::TensorStore;
use crate::tensor_data::parse_torch_dtype;

/// Build a serialize `Graph` from a post-pass `IrGraph`, inserting the graph's
/// constant tensor data into `store`. Handles one flat graph; the control-flow
/// submodule hierarchy is assembled by the caller (Phase 3).
pub fn graph_from_ir(ir: &IrGraph, store: &mut TensorStore) -> Graph {
    graph_from_ir_ext(ir, store, false)
}

/// Convert a control-flow region from IR while preserving unused runtime
/// placeholders. A while-cond block may not read every loop-carried tensor, but
/// its TOSA block signature still has to match the WHILE_LOOP input list.
pub fn graph_from_ir_preserve_placeholders(ir: &IrGraph, store: &mut TensorStore) -> Graph {
    graph_from_ir_ext(ir, store, true)
}

fn graph_from_ir_ext(
    ir: &IrGraph,
    store: &mut TensorStore,
    preserve_dead_placeholders: bool,
) -> Graph {
    // Node-ids referenced by any alive node. The EP-effect constant folding runs
    // at flush time on the Python graph and removes now-dead constant/param
    // placeholders (per-channel scale/zp, pre-fold weights) -- the IrGraph still
    // carries them. Drop the unreferenced ones here, matching ingest's
    // dead-placeholder skip, so the converted graph has no dangling inputs.
    let mut referenced: std::collections::HashSet<IrNodeId> = std::collections::HashSet::new();
    for n in ir.nodes.iter().filter(|n| n.alive) {
        for a in &n.args {
            collect_ir_refs(a, &mut referenced);
        }
        for v in n.kwargs.values() {
            collect_ir_refs(v, &mut referenced);
        }
    }

    // Control-flow submodule references: a GetAttr to a region (no tensor data)
    // is serialized as a named region string (Arg::String), not a graph node --
    // matching ingest, which skips these get_attr nodes and turns their uses into
    // the branch/body name on the COND_IF / WHILE_LOOP op.
    let mut submodule_map: HashMap<IrNodeId, String> = HashMap::new();
    let cf_dbg = std::env::var("ARM_FROM_IR_DEBUG").is_ok();
    for n in ir.nodes.iter().filter(|n| n.alive) {
        if cf_dbg && n.name.contains("submodule") {
            eprintln!(
                "[SUBMOD] {} op={:?} target={} tensor={}",
                n.name,
                n.op,
                n.target,
                n.tensor.is_some()
            );
        }
        if !matches!(
            parse_op_target(&n.target),
            OpTarget::CondIf | OpTarget::WhileLoop
        ) {
            continue;
        }
        // The branch/region args of a control-flow op reference region get_attr
        // nodes; emit them as String region names, not graph nodes.
        for a in &n.args {
            if let IrArg::Node(id) = a {
                if let Some(m) = ir.nodes.iter().find(|x| x.id == *id) {
                    if matches!(m.op, NodeOp::GetAttr) || m.name.contains("submodule") {
                        submodule_map.insert(*id, m.name.clone());
                    }
                }
            }
        }
    }

    let mut graph = Graph::new();
    let mut idmap: HashMap<IrNodeId, NodeId> = HashMap::new();

    // Pass 1: allocate serialize nodes (args resolved in pass 2) and build the
    // IrNodeId -> serialize-NodeId map.
    for ir_node in ir.nodes.iter().filter(|n| n.alive) {
        if submodule_map.contains_key(&ir_node.id) {
            continue; // emitted as a String region reference, not a node
        }
        let placeholder_like = matches!(ir_node.op, NodeOp::Placeholder | NodeOp::GetAttr);
        let preserve_placeholder =
            preserve_dead_placeholders && matches!(ir_node.op, NodeOp::Placeholder);
        if placeholder_like && !preserve_placeholder && !referenced.contains(&ir_node.id) {
            continue; // dead constant/param placeholder
        }
        // Phase 2: the TensorStore is populated from the IrGraph's own NodeTensor
        // data, not a re-read of the Python graph -- so the unfolded constants the
        // converted graph references (e.g. raw `b__frozen_param0` weights) resolve.
        if let Some(t) = &ir_node.tensor {
            store.insert(ir_node.name.clone(), t.data.clone());
        }
        let op = map_op(ir_node.op);
        let target = if op == OpType::CallFunction {
            let parsed = parse_op_target(&ir_node.target);
            // Mirror ingest: aten.convolution with transposed=True (arg 6) is a
            // transpose-conv, not a plain conv2d.
            if matches!(parsed, OpTarget::Conv2d)
                && ir_node.target.contains("aten.convolution")
                && matches!(ir_node.args.get(6), Some(IrArg::Bool(true)))
            {
                OpTarget::ConvTranspose2d
            } else {
                parsed
            }
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
        // graph inputs. GetAttr nodes map to OpType::Placeholder too but are
        // CONSTANTS (data in the TensorStore by name), so they must NOT join
        // `inputs`, or the serializer emits them as runtime inputs and the input
        // binding misaligns.
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
        let args: SmallVec<[Arg; 4]> = ir_node
            .args
            .iter()
            .map(|a| map_arg(a, &idmap, &submodule_map))
            .collect();
        let kwargs: Vec<(String, Arg)> = ir_node
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), map_arg(v, &idmap, &submodule_map)))
            .collect();
        if let Some(node) = graph.nodes.get_mut(nid) {
            node.args = args;
            node.kwargs = kwargs;
        }
    }

    // Fold constant-fed pure reshapes into materialized constants. Ingest does
    // this via the FuseConstantNode EP-effect; without it the converted graph
    // keeps a RESHAPE on a constant weight, which TOSA rejects for sub-byte INT4
    // ("could not allocate op RESHAPE input=(INT4 ...)").
    fold_constant_reshapes(&mut graph, store);
    // Folding leaves the original constants (e.g. the pre-reshape weight) with no
    // users -- drop them so we don't emit duplicate CONST tensors.
    remove_dead_constants(&mut graph, preserve_dead_placeholders);

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

/// Pure (numel-preserving, no data reorder) reshape-like ops -- safe to fold a
/// constant input through by reinterpreting the same bytes at a new shape.
fn is_pure_reshape_target(t: &OpTarget) -> bool {
    matches!(
        t,
        OpTarget::View
            | OpTarget::Reshape
            | OpTarget::TosaReshape
            | OpTarget::Squeeze
            | OpTarget::Unsqueeze
            | OpTarget::Flatten
    )
}

/// Replace each pure reshape-like node whose single input is a constant (data in
/// `store`) with a materialized constant placeholder of the reshaped data,
/// dropping the RESHAPE op. Iterates to fold constant chains.
fn fold_constant_reshapes(graph: &mut Graph, store: &mut TensorStore) {
    for _ in 0..4 {
        let mut changed = false;
        for nid in graph.order.clone() {
            let (input_id, out_shape, name) = {
                let Some(n) = graph.nodes.get(nid) else {
                    continue;
                };
                if !is_pure_reshape_target(&n.target) {
                    continue;
                }
                let Some(Arg::Node(input_id)) = n.args.first() else {
                    continue;
                };
                (*input_id, n.meta.val.shape_slice().to_vec(), n.name.clone())
            };
            let Some(input_name) = graph.nodes.get(input_id).map(|n| n.name.clone()) else {
                continue;
            };
            let Some(src) = store.get(&input_name) else {
                continue; // input is not a constant
            };
            let src_numel: i64 = src.shape.iter().product();
            let out_numel: i64 = out_shape.iter().product();
            if out_shape.is_empty() || src_numel != out_numel {
                continue; // not a pure (numel-preserving) reshape
            }
            store.insert(
                name.clone(),
                crate::store::TensorData {
                    data: src.data.clone(),
                    shape: out_shape,
                    dtype: src.dtype,
                },
            );
            if let Some(n) = graph.nodes.get_mut(nid) {
                n.op = OpType::Placeholder; // now a constant (data in store)
                n.target = OpTarget::Other(name);
                n.args = SmallVec::new();
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
}

/// Apply the pipeline's accumulated EP-effects to the converted graph + store,
/// Rust-natively. The IrGraph is captured PRE-effects (the effects are otherwise
/// applied to the Python EP at flush time); this reproduces them so the graph
/// matches the final lowered form. Most effects carry their precomputed data, so
/// no Python computation is needed.
///
/// Handled here: MaterializeConstantNode / MaterializeZeroInputConstantNode
/// (replace a computed node with its precomputed constant) and
/// DeleteConstantPlaceholder. The remaining variants (table-op insertion, the
/// compute-in-Python FuseConstantNode/InsertTableOp) are added incrementally.
pub fn apply_effects(graph: &mut Graph, store: &mut TensorStore, effects: &FxEpEffectBatch) {
    apply_effects_ext(graph, store, effects, false);
}

pub fn apply_effects_preserve_placeholders(
    graph: &mut Graph,
    store: &mut TensorStore,
    effects: &FxEpEffectBatch,
) {
    apply_effects_ext(graph, store, effects, true);
}

fn apply_effects_ext(
    graph: &mut Graph,
    store: &mut TensorStore,
    effects: &FxEpEffectBatch,
    preserve_graph_inputs: bool,
) {
    if effects.effects.is_empty() {
        return;
    }
    let name_map: HashMap<String, NodeId> = graph
        .order
        .iter()
        .filter_map(|&nid| graph.nodes.get(nid).map(|n| (n.name.clone(), nid)))
        .collect();
    let mut applied = false;
    for effect in &effects.effects {
        match effect {
            FxEpEffect::MaterializeConstantNode(e) => {
                applied |= materialize_const(
                    graph,
                    store,
                    &name_map,
                    &e.node_name,
                    &e.data,
                    &e.shape,
                    e.dtype,
                );
            }
            FxEpEffect::MaterializeZeroInputConstantNode(e) => {
                applied |= materialize_const(
                    graph,
                    store,
                    &name_map,
                    &e.node_name,
                    &e.data,
                    &e.shape,
                    e.dtype,
                );
            }
            FxEpEffect::DeleteConstantPlaceholder(e) => {
                if let Some(&nid) = name_map.get(&e.placeholder_name) {
                    store.buffers.remove(&e.placeholder_name);
                    remove_node(graph, nid);
                    applied = true;
                }
            }
            FxEpEffect::InsertPrecomputedTableOp(e) => {
                applied |= apply_precomputed_table(graph, store, &name_map, e);
            }
            FxEpEffect::FuseConstantNode(e) => {
                applied |= retarget_to_fused_const(graph, store, &name_map, &e.node_name);
            }
            FxEpEffect::MaterializeZeroInputConstant(e) => {
                applied |= retarget_to_fused_const(graph, store, &name_map, &e.node_name);
            }
            FxEpEffect::CastInt64InputToInt32(e) => {
                applied |= cast_buffer_int64_to_int32(store, &e.placeholder_name);
            }
            FxEpEffect::InsertTableOp(e) => {
                crate::fold_fallback_log(&format!("InsertTableOp node={}", e.node_name));
            }
        }
    }
    if applied {
        remove_dead_constants(graph, preserve_graph_inputs);
        graph.order = topo_order(graph);
        graph.rebuild_user_lists();
    }
}

/// Replace a unary math node (e.g. `tan`, `sigmoid`) with a TOSA TABLE op fed by
/// the precomputed table constant. Mirrors the Python `_apply_table_op` for the
/// precomputed path. INT8 table -> INT8 output; INT16 -> INT32 output with a
/// trailing rescale of 2^rescale_lshift.
fn apply_precomputed_table(
    graph: &mut Graph,
    store: &mut TensorStore,
    name_map: &HashMap<String, NodeId>,
    e: &crate::fx_ep_effects::FxInsertPrecomputedTableOpEffect,
) -> bool {
    let Some(&nid) = name_map.get(&e.node_name) else {
        return false;
    };
    let (input_arg, out_shape, dim_order) = {
        let Some(n) = graph.nodes.get(nid) else {
            return false;
        };
        let Some(input_arg) = n.args.first().cloned() else {
            return false;
        };
        (input_arg, n.meta.val, n.meta.tosa_dim_order.clone())
    };

    // Table constant node.
    let table_name = format!("{}_table", e.node_name);
    store.insert(
        table_name.clone(),
        crate::store::TensorData {
            data: e.data.clone(),
            shape: e.shape.clone(),
            dtype: e.table_dtype,
        },
    );
    let table_meta = NodeMeta::new(TensorInfo::new(&e.shape, e.table_dtype));
    let table_node = Node {
        op: OpType::Placeholder,
        target: OpTarget::Other(table_name.clone()),
        args: SmallVec::new(),
        kwargs: Vec::new(),
        meta: table_meta,
        users: SmallVec::new(),
        name: table_name.clone(),
    };
    let table_nid = graph.nodes.insert(table_node);
    graph.order.push(table_nid);
    graph.node_names.push((table_nid, table_name));

    // INT8 TABLE returns INT8; INT16 TABLE returns INT32.
    let table_out_dtype = if e.input.dtype == DType::Int16 {
        DType::Int32
    } else {
        e.input.dtype
    };
    let mut table_val = out_shape;
    table_val.dtype = table_out_dtype;

    let mut table_args: SmallVec<[Arg; 4]> = SmallVec::new();
    table_args.push(input_arg);
    table_args.push(Arg::Node(table_nid));
    if let Some(n) = graph.nodes.get_mut(nid) {
        n.target = OpTarget::TosaTable;
        n.args = table_args;
        n.meta.val = table_val;
        n.meta.tosa_dim_order = dim_order.clone();
    }

    // INT16 TABLE produces INT32; a trailing rescale by 2^rescale_lshift aligns
    // it (staying INT32 -- the eventual INT16 quantization is done by a
    // downstream rescale, matching the Python _apply_table_op path). Insert it
    // after the table and rewire the table's users.
    if e.input.dtype == DType::Int16 {
        let scale = 2.0_f64.powi(e.rescale_lshift as i32);
        let rescale_name = format!("{}_table_rescale", e.node_name);
        // The rescale's destination dtype is the effect's carried output dtype
        // (INT32 for the 32-bit table variant, INT16 for 16a8w).
        let out_dtype = e.output.dtype;
        let mut r_args: SmallVec<[Arg; 4]> = SmallVec::new();
        r_args.push(Arg::Node(nid));
        r_args.push(Arg::Dtype(out_dtype));
        let mut scales: SmallVec<[f64; 4]> = SmallVec::new();
        scales.push(scale);
        r_args.push(Arg::FloatList(scales));
        r_args.push(Arg::Int(0));
        r_args.push(Arg::Int(0));
        let mut r_val = out_shape;
        r_val.dtype = out_dtype;
        let mut r_meta = NodeMeta::new(r_val);
        r_meta.tosa_dim_order = dim_order;
        let r_node = Node {
            op: OpType::CallFunction,
            target: OpTarget::TosaRescale,
            args: r_args,
            kwargs: Vec::new(),
            meta: r_meta,
            users: SmallVec::new(),
            name: rescale_name.clone(),
        };
        let r_nid = graph.nodes.insert(r_node);
        graph.order.push(r_nid);
        graph.node_names.push((r_nid, rescale_name));
        rewire_node_uses(graph, nid, r_nid);
    }
    true
}

/// Redirect every consumer of `old` to `new` (used after inserting a node in the
/// middle of a def-use chain). Skips `new` itself so its own input is kept.
fn rewire_node_uses(graph: &mut Graph, old: NodeId, new: NodeId) {
    for nid in graph.order.clone() {
        if nid == new {
            continue;
        }
        if let Some(n) = graph.nodes.get_mut(nid) {
            for a in n.args.iter_mut() {
                replace_arg_node(a, old, new);
            }
            for (_, a) in n.kwargs.iter_mut() {
                replace_arg_node(a, old, new);
            }
        }
    }
}

fn replace_arg_node(a: &mut Arg, old: NodeId, new: NodeId) {
    match a {
        Arg::Node(id) if *id == old => *id = new,
        Arg::NodeList(ids) => {
            for id in ids.iter_mut() {
                if *id == old {
                    *id = new;
                }
            }
        }
        _ => {}
    }
}

/// Replace a computed node with its precomputed constant: data -> store, node
/// -> constant placeholder (no args). Its now-dead inputs are dropped later by
/// remove_dead_constants.
fn materialize_const(
    graph: &mut Graph,
    store: &mut TensorStore,
    name_map: &HashMap<String, NodeId>,
    name: &str,
    data: &[u8],
    shape: &[i64],
    dtype: DType,
) -> bool {
    let Some(&nid) = name_map.get(name) else {
        return false;
    };
    // TOSA has no int64; mirror ConvertInt64ConstOpsToInt32Pass by downcasting an
    // int64 const to int32 when every value fits the int32 range.
    let (data, dtype) = match (dtype, downcast_int64_le_to_int32(data)) {
        (DType::Int64, Some(narrowed)) => (narrowed, DType::Int32),
        _ => (data.to_vec(), dtype),
    };
    store.insert(
        name.to_string(),
        crate::store::TensorData {
            data: data.clone(),
            shape: shape.to_vec(),
            dtype,
        },
    );
    if let Some(n) = graph.nodes.get_mut(nid) {
        n.op = OpType::Placeholder;
        n.target = OpTarget::Other(name.to_string());
        n.args = SmallVec::new();
        n.meta.val = TensorInfo::new(shape, dtype);
    }
    true
}

/// Prefix every node name and store key of a converted submodule graph with
/// `{region}::` (matching ingest's qualify_name), so control-flow submodule graphs
/// don't collide with the parent or each other in the shared TensorStore. The
/// serializer keys a placeholder's constant data by node.name, so names and store
/// keys must move together. Intra-graph references are by NodeId, so renaming is
/// safe. Building block for the round-trip Phase 4c converter hierarchy assembly.
pub fn prefix_graph(graph: &mut Graph, store: &mut TensorStore, prefix: &str) {
    if prefix.is_empty() {
        return;
    }
    let mut keys_to_move: Vec<String> = Vec::new();
    for &nid in &graph.order {
        if let Some(n) = graph.nodes.get_mut(nid) {
            if store.buffers.contains_key(&n.name) {
                keys_to_move.push(n.name.clone());
            }
            for arg in n.args.iter_mut() {
                prefix_region_arg(arg, prefix);
            }
            for (_, arg) in n.kwargs.iter_mut() {
                prefix_region_arg(arg, prefix);
            }
            n.name = format!("{prefix}{}", n.name);
        }
    }
    for (_, name) in graph.node_names.iter_mut() {
        *name = format!("{prefix}{name}");
    }
    for key in keys_to_move {
        if let Some(data) = store.buffers.remove(&key) {
            store.buffers.insert(format!("{prefix}{key}"), data);
        }
    }
}

fn prefix_region_arg(arg: &mut Arg, prefix: &str) {
    match arg {
        Arg::String(name) if !name.contains("::") => {
            *name = format!("{prefix}{name}");
        }
        _ => {}
    }
}

/// Diagnostic (ARM_FOLD_FALLBACK_FILE): report constant placeholder nodes that the
/// serializer will emit as CONST but whose data is absent from the store -- i.e.
/// the IR-derived store is missing a constant the Python re-ingest would have had.
pub fn report_missing_consts(graph: &Graph, store: &TensorStore) {
    let inputs: std::collections::HashSet<NodeId> = graph.inputs.iter().copied().collect();
    for &nid in &graph.order {
        let Some(n) = graph.nodes.get(nid) else {
            continue;
        };
        if n.op != OpType::Placeholder || inputs.contains(&nid) {
            continue;
        }
        if !store.buffers.contains_key(&n.name) {
            crate::fold_fallback_log(&format!(
                "MISSING_CONST name={} target={:?} dtype={:?} shape={:?}",
                n.name, n.target, n.meta.val.dtype, n.meta.val.shape
            ));
        }
    }
}

/// Apply a CastInt64InputToInt32 effect to the store: the IrPass already set the
/// placeholder's meta dtype to int32, but `graph_from_ir` copied the raw int64
/// buffer bytes. Re-encode them as int32 so the store matches the node dtype.
fn cast_buffer_int64_to_int32(store: &mut TensorStore, name: &str) -> bool {
    let Some(data) = store.get(name).cloned() else {
        return false;
    };
    if data.dtype != DType::Int64 {
        return false;
    }
    let Some(narrowed) = downcast_int64_le_to_int32(&data.data) else {
        return false;
    };
    store.insert(
        name.to_string(),
        crate::store::TensorData {
            data: narrowed,
            shape: data.shape,
            dtype: DType::Int32,
        },
    );
    true
}

/// Re-encode little-endian int64 bytes as little-endian int32, or `None` if any
/// value exceeds the int32 range (matching ConvertInt64ConstOpsToInt32Pass, which
/// leaves out-of-range constants as int64).
fn downcast_int64_le_to_int32(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() % 8 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(8) {
        let v = i64::from_le_bytes(chunk.try_into().ok()?);
        let narrowed = i32::try_from(v).ok()?;
        out.extend_from_slice(&narrowed.to_le_bytes());
    }
    Some(out)
}

/// FuseConstantNode / MaterializeZeroInputConstant compute their result in
/// Python at flush time and materialize it as a `{node}_fused_const` buffer. The
/// reused ingest store already holds that data, so retarget the node to it (copy
/// the data under the node's own name) instead of recomputing. (For a future
/// store-from-IrGraph-only Phase 4, the computation would be ported to Rust.)
fn retarget_to_fused_const(
    graph: &mut Graph,
    store: &mut TensorStore,
    name_map: &HashMap<String, NodeId>,
    node_name: &str,
) -> bool {
    let folded_name = format!("{node_name}_fused_const");
    let found = store.get(&folded_name).is_some();
    crate::fold_fallback_log(&format!(
        "retarget_to_fused_const found_python_data={found}"
    ));
    let Some(data) = store.get(&folded_name).cloned() else {
        return false;
    };
    let Some(&nid) = name_map.get(node_name) else {
        return false;
    };
    store.insert(node_name.to_string(), data.clone());
    if let Some(n) = graph.nodes.get_mut(nid) {
        n.op = OpType::Placeholder;
        n.target = OpTarget::Other(node_name.to_string());
        n.args = SmallVec::new();
        n.meta.val = TensorInfo::new(&data.shape, data.dtype);
    }
    true
}

fn remove_node(graph: &mut Graph, nid: NodeId) {
    graph.nodes.remove(nid);
    graph.order.retain(|&x| x != nid);
    graph.inputs.retain(|&x| x != nid);
    graph.node_names.retain(|&(x, _)| x != nid);
}

/// Remove unreferenced placeholder/constant nodes (e.g. constants left dead by
/// fold_constant_reshapes), so we don't emit duplicate CONST tensors.
fn remove_dead_constants(graph: &mut Graph, preserve_graph_inputs: bool) {
    let mut referenced: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for &nid in &graph.order {
        if let Some(n) = graph.nodes.get(nid) {
            for id in Graph::collect_input_ids(n) {
                referenced.insert(id);
            }
        }
    }
    if let Some(out) = graph.output {
        referenced.insert(out);
    }
    let dead: Vec<NodeId> = graph
        .order
        .iter()
        .copied()
        .filter(|&nid| {
            if preserve_graph_inputs && graph.inputs.contains(&nid) {
                return false;
            }
            graph
                .nodes
                .get(nid)
                .is_some_and(|n| n.op == OpType::Placeholder && !referenced.contains(&nid))
        })
        .collect();
    for nid in dead {
        graph.nodes.remove(nid);
        graph.order.retain(|&x| x != nid);
        graph.inputs.retain(|&x| x != nid);
        graph.node_names.retain(|&(x, _)| x != nid);
    }
}

fn collect_ir_refs(a: &IrArg, out: &mut std::collections::HashSet<IrNodeId>) {
    match a {
        IrArg::Node(id) => {
            out.insert(*id);
        }
        IrArg::List(items) => {
            for it in items {
                collect_ir_refs(it, out);
            }
        }
        _ => {}
    }
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

fn map_arg(
    a: &IrArg,
    idmap: &HashMap<IrNodeId, NodeId>,
    submodule_map: &HashMap<IrNodeId, String>,
) -> Arg {
    match a {
        IrArg::Node(id) => {
            if let Some(name) = submodule_map.get(id) {
                Arg::String(name.clone())
            } else {
                idmap.get(id).map(|&n| Arg::Node(n)).unwrap_or(Arg::None)
            }
        }
        IrArg::Int(v) => Arg::Int(*v),
        IrArg::Float(v) => Arg::Float(*v),
        IrArg::Bool(v) => Arg::Bool(*v),
        IrArg::Str(s) => Arg::String(s.clone()),
        IrArg::Dtype(s) => Arg::Dtype(parse_torch_dtype(s)),
        IrArg::None => Arg::None,
        IrArg::List(items) => map_list(items, idmap, submodule_map),
    }
}

/// serialize `Arg` lists are typed (`NodeList`/`IntList`/`FloatList`), unlike the
/// general `IrArg::List`. Classify by element type.
fn map_list(
    items: &[IrArg],
    idmap: &HashMap<IrNodeId, NodeId>,
    _submodule_map: &HashMap<IrNodeId, String>,
) -> Arg {
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
    if items
        .iter()
        .all(|x| matches!(x, IrArg::Int(_) | IrArg::Bool(_)))
    {
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
    if items
        .iter()
        .all(|x| matches!(x, IrArg::Float(_) | IrArg::Int(_)))
    {
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
