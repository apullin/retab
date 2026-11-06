//! Port of rc4's `RewriteHighRankSingletonPermutePass`
//! (`backends/arm/_passes/rewrite_high_rank_singleton_permute_pass.py`).
//!
//! A permute of rank above 4 whose input has singleton dims becomes
//! `view(drop the singleton dims) -> permute(reduced rank) -> view(output shape)`
//! when the reduced rank is at most 4. When the reduced permutation is the
//! identity, only the two views remain.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::is_permute_target;
use crate::layout::ctx::LayoutCtx;

const VIEW_COPY: &str = "aten.view_copy.default";

pub(crate) struct IrRewriteHighRankSingletonPermutePass;

impl IrPass for IrRewriteHighRankSingletonPermutePass {
    fn name(&self) -> &str {
        "IrRewriteHighRankSingletonPermute"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let plans: Vec<(NodeId, Plan)> = graph
            .live_call_function_ids()
            .into_iter()
            .filter_map(|id| plan(graph, id).map(|plan| (id, plan)))
            .collect();
        if plans.is_empty() {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        for (id, plan) in plans {
            rewrite(&mut ctx, id, plan);
        }
        ctx.finish();
        true
    }
}

/// The rewrite of one permute, as `call_operator` computes it (`:51-114`).
#[derive(Debug, PartialEq)]
struct Plan {
    reduced_input_shape: Vec<i64>,
    /// `None` when the reduced permutation is the identity (`:99-100`).
    reduced_permutation: Option<Vec<i64>>,
    output_shape: Vec<i64>,
}

fn plan(graph: &IrGraph, id: NodeId) -> Option<Plan> {
    let node = graph.node(id);
    if !is_permute_target(&node.target) || node.args.len() < 2 {
        return None;
    }
    let input = node.args[0].as_node()?;
    let permutation = node.args[1].as_int_list()?;
    let input_shape = graph.node_shape_vec(input)?;
    let output_shape = graph.node_shape_vec(id)?;
    let rank = input_shape.len();
    if rank <= 4 || output_shape.len() != rank {
        return None;
    }

    // Python's `dim % rank` is never negative.
    let normalized: Vec<usize> = permutation
        .iter()
        .map(|&dim| dim.rem_euclid(rank as i64) as usize)
        .collect();
    let non_singleton_axes: Vec<usize> = (0..rank).filter(|&axis| input_shape[axis] != 1).collect();
    if non_singleton_axes.len() == rank {
        return None;
    }
    let reduced_rank = non_singleton_axes.len();
    if reduced_rank > 4 {
        return None;
    }

    let reduced_permutation: Vec<i64> = normalized
        .iter()
        .filter_map(|axis| non_singleton_axes.iter().position(|kept| kept == axis))
        .map(|index| index as i64)
        .collect();
    let identity: Vec<i64> = (0..reduced_rank as i64).collect();
    let mut sorted = reduced_permutation.clone();
    sorted.sort_unstable();
    if sorted != identity {
        return None;
    }

    Some(Plan {
        reduced_input_shape: non_singleton_axes.iter().map(|&axis| input_shape[axis]).collect(),
        reduced_permutation: (reduced_permutation != identity).then_some(reduced_permutation),
        output_shape,
    })
}

/// Replace the permute with the view chain. Every new node gets the permute's
/// meta with its own shape, as `super().call_operator(..., meta)` does, and the
/// chain takes the permute's place in the graph order, as the retrace does.
fn rewrite(ctx: &mut LayoutCtx<'_>, id: NodeId, plan: Plan) {
    let node = ctx.node(id);
    let input = node.args[0].as_node().expect("planned permutes read a node");
    let target = node.target.clone();
    let kwargs = node.kwargs.clone();
    let name = node.name.clone();
    let base = node.meta.clone();
    let stamp = base.tensor_stamp_same_dtype_or_float32();

    let output = ctx.inserting_before(id, |ctx| {
        let reduced_input = ctx.create_node(
            NodeOp::CallFunction,
            format!("{name}_drop_unit_dims"),
            VIEW_COPY,
            vec![IrArg::Node(input), i64_list_arg(&plan.reduced_input_shape)],
            HashMap::new(),
            stamp.preserve_qparams(plan.reduced_input_shape.clone()),
        );
        let reduced_output = match &plan.reduced_permutation {
            None => reduced_input,
            Some(permutation) => {
                let shape: Vec<i64> = permutation
                    .iter()
                    .map(|&axis| plan.reduced_input_shape[axis as usize])
                    .collect();
                ctx.create_node(
                    NodeOp::CallFunction,
                    format!("{name}_reduced"),
                    target,
                    vec![IrArg::Node(reduced_input), i64_list_arg(permutation)],
                    kwargs,
                    stamp.preserve_qparams(shape),
                )
            }
        };
        ctx.create_node(
            NodeOp::CallFunction,
            format!("{name}_restore_rank"),
            VIEW_COPY,
            vec![IrArg::Node(reduced_output), i64_list_arg(&plan.output_shape)],
            HashMap::new(),
            stamp.preserve_qparams(plan.output_shape.clone()),
        )
    });
    ctx.replace_all_uses_with(id, output);
    ctx.erase_node(id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    const PERMUTE: &str = "aten.permute_copy.default";

    fn run(g: &mut LayoutGraph) -> bool {
        IrRewriteHighRankSingletonPermutePass.run(g.graph_mut())
    }

    #[test]
    fn rank5_permute_with_a_singleton_becomes_view_permute_view() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4, 5]);
        let p = g.permute(x, &[0, 4, 1, 2, 3]);
        g.output(&[p]);

        assert!(run(&mut g));

        g.assert_targets(&[VIEW_COPY, PERMUTE, VIEW_COPY]);
        g.assert_permute_dims(&[&[3, 0, 1, 2]]);
        let views: Vec<NodeId> = g
            .graph()
            .live_call_function_ids()
            .into_iter()
            .filter(|&id| g.graph().node(id).target == VIEW_COPY)
            .collect();
        assert_eq!(g.shape(views[0]), &[2, 3, 4, 5]);
        assert_eq!(g.shape(views[1]), &[1, 5, 2, 3, 4]);
    }

    #[test]
    fn an_identity_reduced_permutation_leaves_two_views() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 1, 3, 1, 4]);
        // Moving only singleton axes keeps the non-singleton order.
        let p = g.permute(x, &[1, 0, 3, 2, 4]);
        g.output(&[p]);

        assert!(run(&mut g));

        g.assert_targets(&[VIEW_COPY, VIEW_COPY]);
    }

    #[test]
    fn negative_dims_are_normalized() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4, 5]);
        let p = g.permute(x, &[0, -1, 1, 2, 3]);
        g.output(&[p]);

        assert!(run(&mut g));

        g.assert_permute_dims(&[&[3, 0, 1, 2]]);
    }

    #[test]
    fn rank4_and_no_singleton_and_reduced_rank_above_4_are_left_alone() {
        for (shape, permutation) in [
            (&[1, 2, 3, 4][..], &[0, 3, 1, 2][..]),
            (&[2, 3, 4, 5, 6][..], &[0, 4, 1, 2, 3][..]),
            (&[1, 2, 3, 4, 5, 6][..], &[0, 5, 1, 2, 3, 4][..]),
        ] {
            let mut g = LayoutGraph::new();
            let x = g.input("x", shape);
            let p = g.permute(x, permutation);
            g.output(&[p]);
            assert!(!run(&mut g), "{shape:?}");
            g.assert_targets(&[PERMUTE]);
        }
    }

    #[test]
    fn new_nodes_keep_the_permute_meta_and_take_its_place() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4, 5]);
        let before = g.call("aten.relu.default", vec![node(x)]);
        let p = g.permute(before, &[0, 4, 1, 2, 3]);
        let after = g.call("aten.relu.default", vec![node(p)]);
        g.output(&[after]);
        g.graph_mut()
            .node_mut(p)
            .meta
            .set_both_qparams(qparam(0.5, 3, "torch.int8"));

        assert!(run(&mut g));

        g.assert_targets(&[
            "aten.relu.default",
            VIEW_COPY,
            PERMUTE,
            VIEW_COPY,
            "aten.relu.default",
        ]);
        for id in g.graph().live_call_function_ids() {
            if g.graph().node(id).target != "aten.relu.default" {
                assert!(g.graph().node(id).meta.output_qparam().is_some());
            }
        }
    }
}
