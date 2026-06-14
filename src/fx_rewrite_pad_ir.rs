//! Pure Rust IrPass version of master RewritePadPass — non-constant modes only.
//!
//! TOSA `PAD` is constant-only, so reflect/replicate/circular `aten.pad.default`
//! must be decomposed into `slice_copy` + `cat`. For each padded dim (innermost
//! outward) we gather `left` and `right` width-1 slices at mode-specific indices
//! and concatenate them around the original tensor:
//!   replicate: left=slice@0,           right=slice@(size-1)
//!   circular:  left=slice@(size-left+i),right=slice@i
//!   reflect:   left=slice@(left-i),     right=slice@(size-2-i)
//! Constant-mode pad is left to the serializer's existing handling.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta};
use crate::fx_target::is_pad_target;
use crate::runner_logging::log_line;

#[allow(clippy::too_many_arguments)]
fn add_slice(
    graph: &mut IrGraph,
    id: NodeId,
    base_meta: &NodeMeta,
    dtype: &str,
    name: String,
    x_id: NodeId,
    x_shape: &[i64],
    dim: usize,
    idx: i64,
    prev: &mut Option<NodeId>,
) -> NodeId {
    let mut out_shape = x_shape.to_vec();
    out_shape[dim] = 1;
    // slice_copy.Tensor(x, dim, idx, idx+1) — width-1 slice keeping the dim.
    let n = graph.add_anchored_call_function(
        id,
        name,
        "aten.slice_copy.Tensor",
        vec![
            IrArg::Node(x_id),
            IrArg::Int(dim as i64),
            IrArg::Int(idx),
            IrArg::Int(idx + 1),
        ],
        base_meta.cloned_with_val(out_shape, dtype),
    );
    if let Some(p) = *prev {
        graph.anchor_after(n, p);
    }
    *prev = Some(n);
    n
}

/// Pad `x` along `dim` by (left, right) per `mode`; returns (new node, new shape).
#[allow(clippy::too_many_arguments)]
fn pad_along_dim(
    graph: &mut IrGraph,
    id: NodeId,
    base_meta: &NodeMeta,
    dtype: &str,
    name_prefix: &str,
    counter: &mut usize,
    x_id: NodeId,
    x_shape: &[i64],
    dim: usize,
    left: i64,
    right: i64,
    mode: &str,
    prev: &mut Option<NodeId>,
) -> (NodeId, Vec<i64>) {
    if left == 0 && right == 0 {
        return (x_id, x_shape.to_vec());
    }
    let size = x_shape[dim];

    let mut members: Vec<IrArg> = Vec::new();
    for i in 0..left {
        let idx = match mode {
            "replicate" => 0,
            "circular" => size - left + i,
            _ => left - i, // reflect (mode validated by caller)
        };
        let s = add_slice(
            graph,
            id,
            base_meta,
            dtype,
            format!("{}_slice{}", name_prefix, *counter),
            x_id,
            x_shape,
            dim,
            idx,
            prev,
        );
        *counter += 1;
        members.push(IrArg::Node(s));
    }
    members.push(IrArg::Node(x_id));
    for i in 0..right {
        let idx = match mode {
            "replicate" => size - 1,
            "circular" => i,
            _ => size - 2 - i, // reflect (mode validated by caller)
        };
        let s = add_slice(
            graph,
            id,
            base_meta,
            dtype,
            format!("{}_slice{}", name_prefix, *counter),
            x_id,
            x_shape,
            dim,
            idx,
            prev,
        );
        *counter += 1;
        members.push(IrArg::Node(s));
    }

    let mut out_shape = x_shape.to_vec();
    out_shape[dim] = size + left + right;
    let cat = graph.add_anchored_call_function(
        id,
        format!("{}_cat{}", name_prefix, *counter),
        "aten.cat.default",
        vec![IrArg::List(members), IrArg::Int(dim as i64)],
        base_meta.cloned_with_val(out_shape.clone(), dtype),
    );
    *counter += 1;
    if let Some(p) = *prev {
        graph.anchor_after(cat, p);
    }
    *prev = Some(cat);
    (cat, out_shape)
}

fn rewrite_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if !is_pad_target(&node.target) || node.args.len() < 2 {
        return false;
    }
    // args: (input, pad_list, mode, value) — mode may be positional OR a kwarg.
    let mode = match node.args.get(2) {
        Some(IrArg::Str(s)) => s.clone(),
        _ => match node.kwargs.get("mode") {
            Some(IrArg::Str(s)) => s.clone(),
            _ => "constant".to_string(),
        },
    };
    if !matches!(mode.as_str(), "reflect" | "replicate" | "circular") {
        return false; // constant handled by the serializer
    }
    let input_id = match node.args[0].as_node() {
        Some(n) => n,
        None => return false,
    };
    let pad: Vec<i64> = match node.args[1].as_int_list() {
        Some(p) if p.len() % 2 == 0 => p,
        _ => return false,
    };
    let node_name = node.name.clone();
    let orig_meta = node.meta.clone();
    let dtype = match orig_meta.val_dtype() {
        Some(d) => d.to_string(),
        None => return false,
    };
    let input_shape = match graph.nodes[input_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let rank = input_shape.len();
    if pad.len() / 2 > rank {
        return false;
    }
    // Validate up front so we never emit a partial chain (reflect requires pad < size).
    for (pair_idx, chunk) in pad.chunks(2).enumerate() {
        let dim = rank - 1 - pair_idx;
        let size = input_shape[dim];
        if mode == "reflect" && (chunk[0] >= size || chunk[1] >= size) {
            return false;
        }
    }

    let mut cur_id = input_id;
    let mut cur_shape = input_shape;
    let mut prev: Option<NodeId> = None;
    let mut counter: usize = 0;
    // F.pad pad pairs run innermost dim outward.
    for (pair_idx, chunk) in pad.chunks(2).enumerate() {
        let (left, right) = (chunk[0], chunk[1]);
        let dim = rank - 1 - pair_idx;
        let (nid, nshape) = pad_along_dim(
            graph,
            id,
            &orig_meta,
            &dtype,
            &node_name,
            &mut counter,
            cur_id,
            &cur_shape,
            dim,
            left,
            right,
            &mode,
            &mut prev,
        );
        cur_id = nid;
        cur_shape = nshape;
    }

    if cur_id == input_id {
        return false; // nothing emitted (all-zero pad)
    }
    graph.replace_and_erase(id, cur_id);
    true
}

pub struct IrRewritePadPass;

impl IrPass for IrRewritePadPass {
    fn name(&self) -> &str {
        "IrRewritePad"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if rewrite_one(graph, id) {
                modified = true;
            }
        }
        if modified {
            log_line(format_args!(
                "[IrRewritePad] decomposed non-constant pad to slice+cat"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeOp;
    use crate::fx_target::{is_cat_target, is_slice_copy_tensor_target};

    fn named<'a>(graph: &'a IrGraph, name: &str) -> &'a crate::fx_ir::IrNode {
        graph
            .nodes
            .iter()
            .find(|n| n.alive && n.name == name)
            .unwrap_or_else(|| panic!("missing node {name}"))
    }

    fn pad_graph(in_shape: &[i64], pad: &[i64], mode: &str) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            vec![],
            NodeMeta::tensor(in_shape, "torch.int8"),
        );
        // output shape: last dim grows by pad[0]+pad[1] (single-dim pad)
        let mut out = in_shape.to_vec();
        *out.last_mut().unwrap() += pad[0] + pad[1];
        let pad_args = vec![
            IrArg::Node(x),
            IrArg::List(pad.iter().map(|&v| IrArg::Int(v)).collect()),
            IrArg::Str(mode.into()),
        ];
        let p = graph.add_node(
            NodeOp::CallFunction,
            "pad".into(),
            "aten.pad.default".into(),
            pad_args,
            NodeMeta::tensor(out.as_slice(), "torch.int8"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::Node(p)],
            NodeMeta::default(),
        );
        (graph, p)
    }

    #[test]
    fn reflect_pad_last_dim() {
        // x [1,4] reflect pad (2,2) on last dim -> [1,8].
        let (mut graph, pad) = pad_graph(&[1, 4], &[2, 2], "reflect");
        assert!(IrRewritePadPass.run(&mut graph));
        assert!(!graph.nodes[pad].alive);
        // 2 left + 2 right = 4 slice_copy nodes + 1 cat.
        let slices: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.alive && is_slice_copy_tensor_target(&n.target))
            .collect();
        assert_eq!(slices.len(), 4);
        // reflect left indices: left-i for i in 0..2 -> [2,1]; right: size-2-i -> [2,3].
        let cat = named(&graph, "pad_cat4");
        assert!(is_cat_target(&cat.target));
        assert_eq!(cat.meta.val_shape().unwrap(), &[1, 8]);
        // cat list has 5 members (2 left + x + 2 right)
        if let IrArg::List(items) = &cat.args[0] {
            assert_eq!(items.len(), 5);
        } else {
            panic!("cat arg0 not a list");
        }
        assert_eq!(named(&graph, "output").args[0].as_node(), Some(cat.id));
    }

    #[test]
    fn circular_pad_last_dim() {
        // x [1,4] circular pad (1,1) -> [1,6]; left=slice@(size-1)=3, right=slice@0.
        let (mut graph, _p) = pad_graph(&[1, 4], &[1, 1], "circular");
        assert!(IrRewritePadPass.run(&mut graph));
        let slices: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.alive && is_slice_copy_tensor_target(&n.target))
            .collect();
        assert_eq!(slices.len(), 2);
    }

    #[test]
    fn skips_constant_mode() {
        let (mut graph, _p) = pad_graph(&[1, 4], &[2, 2], "constant");
        assert!(!IrRewritePadPass.run(&mut graph));
    }
}
