//! Pure Rust IrPass version of RewriteHighRankSingletonPermutePass
//! (executorch master, backends/arm/_passes/rewrite_high_rank_singleton_permute_pass.py).
//!
//! TOSA/Vela is fragile around direct rank>4 TRANSPOSE. When a high-rank permute
//! has singleton input dims, rewrite it so the permute itself happens at <= rank 4:
//!   view_copy(input -> reduced shape with singleton dims removed)
//!   permute(reduced rank)            [skipped when the reduced permutation is identity]
//!   view_copy(reduced output -> full output shape)
//! This keeps semantics unchanged while reducing the permute rank.
//!
//! NOTE: currently DISABLED in fx_pipeline.rs — the Docker op-suite caught it
//! regressing conv3d / multihead-attention / maxpool-dilation. Kept for a future fix.
#![allow(dead_code)]

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp,
};
use crate::runner_logging::log_line;

fn is_permute(target: &str) -> bool {
    target.contains("permute.default") || target.contains("permute_copy.default")
}

/// Clone `base` (preserving qparams, as the upstream pass reuses the permute's
/// meta on every rewritten node) and stamp a new val shape.
fn meta_with_shape(base: &NodeMeta, shape: &[i64], dtype: &str) -> NodeMeta {
    let mut meta = base.clone();
    meta.set_val(FakeTensorMeta {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
    });
    if let Some(iqp) = base.input_qparams.clone() {
        meta.set_input_qparams(iqp);
    }
    if let Some(oqp) = base.output_qparams.clone() {
        meta.set_output_qparams(oqp);
    }
    meta
}

fn rewrite_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if !is_permute(&node.target) || node.args.len() < 2 {
        return false;
    }
    let target = node.target.clone();
    let node_name = node.name.clone();
    let orig_meta = node.meta.clone();
    let input_id = match node.args[0].as_node() {
        Some(n) => n,
        None => return false,
    };
    let permutation = match node.args[1].as_int_list() {
        Some(p) => p,
        None => return false,
    };

    let input_shape = match graph.nodes[input_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let output_shape = match orig_meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = match orig_meta.val_dtype() {
        Some(d) => d.to_string(),
        None => return false,
    };
    let rank = input_shape.len();
    if rank <= 4 || output_shape.len() != rank || permutation.len() != rank {
        return false;
    }

    // Normalize permutation (resolve any negative axes).
    let r = rank as i64;
    let norm: Vec<usize> = permutation.iter().map(|&d| ((d % r + r) % r) as usize).collect();

    let singleton: Vec<usize> = (0..rank).filter(|&a| input_shape[a] == 1).collect();
    if singleton.is_empty() {
        return false;
    }
    let non_singleton: Vec<usize> = (0..rank).filter(|a| !singleton.contains(a)).collect();
    let reduced_rank = non_singleton.len();
    if reduced_rank > 4 {
        return false;
    }

    // Permutation restricted to the non-singleton axes, remapped to reduced indices.
    let reduced_perm: Vec<usize> = norm
        .iter()
        .filter_map(|axis| non_singleton.iter().position(|a| a == axis))
        .collect();
    // Valid only if the surviving axes form a complete permutation of the reduced rank.
    let mut sorted = reduced_perm.clone();
    sorted.sort_unstable();
    if sorted != (0..reduced_rank).collect::<Vec<usize>>() {
        return false;
    }

    let reduced_input_shape: Vec<i64> = non_singleton.iter().map(|&a| input_shape[a]).collect();

    // 1. view_copy(input -> reduced shape)
    let reduced_input = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_reduce_view", node_name),
        "aten.view_copy.default".to_string(),
        vec![IrArg::Node(input_id), i64_list(&reduced_input_shape)],
        meta_with_shape(&orig_meta, &reduced_input_shape, &dtype),
    );
    graph.anchor_to(reduced_input, id);

    // 2. permute(reduced rank), skipped when the reduced permutation is identity.
    let identity: Vec<usize> = (0..reduced_rank).collect();
    let reduced_output = if reduced_perm == identity {
        reduced_input
    } else {
        let permuted_shape: Vec<i64> = reduced_perm.iter().map(|&p| reduced_input_shape[p]).collect();
        let perm_list: Vec<i64> = reduced_perm.iter().map(|&p| p as i64).collect();
        let pnode = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_reduce_permute", node_name),
            target,
            vec![IrArg::Node(reduced_input), i64_list(&perm_list)],
            meta_with_shape(&orig_meta, &permuted_shape, &dtype),
        );
        graph.anchor_to(pnode, id);
        graph.anchor_after(pnode, reduced_input);
        pnode
    };

    // 3. view_copy(reduced output -> full output shape)
    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_restore_view", node_name),
        "aten.view_copy.default".to_string(),
        vec![IrArg::Node(reduced_output), i64_list(&output_shape)],
        meta_with_shape(&orig_meta, &output_shape, &dtype),
    );
    graph.anchor_after(result, reduced_output);
    graph.replace_and_erase(id, result);
    true
}

// Currently disabled in fx_pipeline.rs (regressed conv3d/multihead/maxpool on the
// Docker gate); kept for a future fix. See the pipeline note.
#[allow(dead_code)]
pub struct IrRewriteHighRankSingletonPermutePass;

impl IrPass for IrRewriteHighRankSingletonPermutePass {
    fn name(&self) -> &str {
        "IrRewriteHighRankSingletonPermute"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();
        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if rewrite_one(graph, id) {
                modified = true;
            }
        }
        if modified {
            log_line(format_args!(
                "[IrRewriteHighRankSingletonPermute] reduced rank>4 singleton permute"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    fn named<'a>(graph: &'a IrGraph, name: &str) -> &'a crate::fx_ir::IrNode {
        graph
            .nodes
            .iter()
            .find(|n| n.alive && n.name == name)
            .unwrap_or_else(|| panic!("missing node {name}"))
    }

    /// rank-5 permute with a leading singleton; non-singleton rank is 4.
    fn rank5_graph(perm: &[i64], in_shape: &[i64], out_shape: &[i64]) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            meta(in_shape, "torch.int8"),
        );
        let perm_id = graph.add_node(
            NodeOp::CallFunction,
            "perm".to_string(),
            "aten.permute_copy.default".to_string(),
            vec![IrArg::Node(x), i64_list(perm)],
            meta(out_shape, "torch.int8"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(perm_id)],
            NodeMeta::default(),
        );
        (graph, perm_id)
    }

    #[test]
    fn rewrites_rank5_singleton_permute() {
        // in [1,2,3,4,5], perm [0,2,1,4,3] -> out [1,3,2,5,4].
        let (mut graph, perm) =
            rank5_graph(&[0, 2, 1, 4, 3], &[1, 2, 3, 4, 5], &[1, 3, 2, 5, 4]);
        assert!(IrRewriteHighRankSingletonPermutePass.run(&mut graph));
        assert!(!graph.nodes[perm].alive);

        // reduced view squeezes the singleton: [2,3,4,5].
        let rv = named(&graph, "perm_reduce_view");
        assert_eq!(rv.target, "aten.view_copy.default");
        assert_eq!(rv.meta.val_shape().unwrap(), &[2, 3, 4, 5]);

        // reduced permute is [1,0,3,2] over rank 4 -> [3,2,5,4].
        let rp = named(&graph, "perm_reduce_permute");
        assert!(rp.target.contains("permute"));
        assert_eq!(rp.args[1].as_int_list().unwrap(), vec![1, 0, 3, 2]);
        assert_eq!(rp.meta.val_shape().unwrap(), &[3, 2, 5, 4]);

        // restore view returns to the full output shape and feeds the output.
        let restore = named(&graph, "perm_restore_view");
        assert_eq!(restore.meta.val_shape().unwrap(), &[1, 3, 2, 5, 4]);
        assert_eq!(restore.args[0].as_node(), Some(rp.id));
        assert_eq!(named(&graph, "output").args[0].as_node(), Some(restore.id));
    }

    #[test]
    fn skips_rank4_permute() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            meta(&[1, 2, 3, 4], "torch.int8"),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "perm".to_string(),
            "aten.permute_copy.default".to_string(),
            vec![IrArg::Node(x), i64_list(&[0, 2, 1, 3])],
            meta(&[1, 3, 2, 4], "torch.int8"),
        );
        assert!(!IrRewriteHighRankSingletonPermutePass.run(&mut graph));
    }

    #[test]
    fn skips_rank5_without_singletons() {
        // rank 5, no singleton dims -> reduced rank would be 5 (> 4), so untouched.
        let (mut graph, _perm) =
            rank5_graph(&[0, 2, 1, 4, 3], &[2, 2, 3, 4, 5], &[2, 3, 2, 5, 4]);
        assert!(!IrRewriteHighRankSingletonPermutePass.run(&mut graph));
    }
}
