//! Port of rc4's `FuseViewCopyTransform` (`backends/transforms/fuse_view_copy.py`),
//! which Arm runs as `FuseViewCopyTransformPass` with no extra ops or filters
//! (`backends/arm/_passes/fuse_view_copy_transform_pass.py`,
//! `arm_pass_manager.py:725,744`).
//!
//! A view followed by a single-user chain of views and unary elementwise ops
//! takes the last view's shape, and the later views are bypassed:
//!
//! ```text
//! view(s0) -> view(s1) -> abs -> view(s2) -> sqrt -> view(s3) -> users
//! view(s3) -> abs -> sqrt -> users
//! ```
//!
//! Then dead code is eliminated from the whole graph, as rc4 does.

use crate::fx_ir::{IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::{has_target_op, LayoutCtx};
use crate::layout::shape::{mark_contiguous, refreshed_meta};
use crate::layout::pending::is_pending_table;

const VIEW_COPY: &str = "aten.view_copy.default";

/// `UNARY_ELEMENTWISE_OPS` (`:43-68`), with RETAB's spellings of the dim-order ops.
const UNARY_ELEMENTWISE_OPS: &[&str] = &[
    "aten.alias_copy.default",
    "aten.clone.default",
    "dim_order_ops._clone_dim_order.default",
    "_clone_dim_order.default",
    "aten._to_copy.default",
    "dim_order_ops._to_dim_order_copy.default",
    "_to_dim_order_copy.default",
    "quantized_decomposed.quantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_tensor.default",
    "aten.abs.default",
    "aten.clamp.default",
    "aten.ceil.default",
    "aten.floor.default",
    "aten.neg.default",
    "aten.relu.default",
    "aten.round.default",
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten.sqrt.default",
    "aten.tanh.default",
    "aten.sign.default",
    "aten.reciprocal.default",
    "aten.gelu.default",
    "aten.rsqrt.default",
    "aten.exp.default",
    "aten.log.default",
];

/// rc4 has already turned a quantized table op into a TABLE (see
/// `layout::pending`), and TABLE is not in the fusible set.
fn is_fusible(ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
    let node = ctx.node(id);
    node.op == NodeOp::CallFunction
        && (node.target == VIEW_COPY || UNARY_ELEMENTWISE_OPS.contains(&node.target.as_str()))
        && !is_pending_table(node)
}

pub(crate) struct IrFuseViewCopyTransformPass;

impl IrPass for IrFuseViewCopyTransformPass {
    fn name(&self) -> &str {
        "IrFuseViewCopyTransform"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        if !has_target_op(graph, |node| node.target == VIEW_COPY) {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let modified = merge_view_copy_chains(&mut ctx);
        if modified {
            ctx.eliminate_dead_code();
            ctx.finish();
        }
        modified
    }
}

/// `_merge_view_copy_chains` (`:142-196`). Shapes are integer lists in RETAB,
/// so `_shape_is_available_at` always holds, and Arm passes no target filters.
fn merge_view_copy_chains(ctx: &mut LayoutCtx<'_>) -> bool {
    let views: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| {
            let node = ctx.node(id);
            node.op == NodeOp::CallFunction && node.target == VIEW_COPY
        })
        .collect();
    let mut modified = false;
    for start in views {
        let (retained, removed) = find_view_copy_chain(ctx, start);
        let Some(&last) = removed.last() else {
            continue;
        };
        modified = true;
        let final_shape = ctx.node(last).args[1].clone();
        let input = ctx.node(start).args[0].clone();
        ctx.set_args(start, vec![input, final_shape]);
        for view in removed {
            let source = ctx.node(view).args[0]
                .as_node()
                .expect("a chained view reads its chain predecessor");
            ctx.replace_all_uses_with(view, source);
        }
        // The retrace re-derives the first view's value from its new size, which
        // may hold a -1 or be empty, and the unary ops after it up to the last
        // removed view take that shape; their dtypes don't change.
        let start_meta = refreshed_meta(ctx.graph(), start).expect("views have a shape rule");
        let shape = start_meta
            .tensor_facts()
            .map(|facts| facts.shape.clone())
            .expect("a refreshed view has a value");
        *ctx.meta_mut(start) = start_meta;
        for id in retained.into_iter().filter(|&id| id != start) {
            let meta = ctx.meta_mut(id);
            let dtype = meta.val_dtype().unwrap_or("torch.float32").to_owned();
            meta.set_tensor_facts(TensorFacts::new(shape.clone(), dtype));
            mark_contiguous(meta);
        }
    }
    modified
}

/// `_find_view_copy_chain` (`:91-112`): the views after `start` that can be
/// removed, and the nodes kept before the last of them (`start` included).
fn find_view_copy_chain(ctx: &LayoutCtx<'_>, start: NodeId) -> (Vec<NodeId>, Vec<NodeId>) {
    let mut retained = vec![start];
    let mut since_last_view = Vec::new();
    let mut removed = Vec::new();
    let mut end = start;
    loop {
        if !is_fusible(ctx, end) {
            break;
        }
        let Some(user) = ctx.graph().single_live_user(end) else {
            break;
        };
        if !is_fusible(ctx, user) {
            break;
        }
        if ctx.node(user).target == VIEW_COPY {
            removed.push(user);
            retained.append(&mut since_last_view);
        } else {
            // Unary ops after the last removed view keep their shape.
            since_last_view.push(user);
        }
        end = user;
    }
    if removed.is_empty() {
        retained.clear();
    }
    (retained, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph) -> bool {
        IrFuseViewCopyTransformPass.run(g.graph_mut())
    }

    /// The docstring example (`:24-36`).
    #[test]
    fn views_through_a_unary_chain_fuse_into_the_first() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let v0 = g.view(x, &[6, 4]);
        let v1 = g.view(v0, &[24]);
        let abs = g.call("aten.abs.default", vec![node(v1)]);
        let v2 = g.view(abs, &[4, 6]);
        let sqrt = g.call("aten.sqrt.default", vec![node(v2)]);
        let v3 = g.view(sqrt, &[2, 12]);
        let user0 = g.call("aten.neg.default", vec![node(v3)]);
        let user1 = g.call("aten.relu.default", vec![node(v3)]);
        g.output(&[user0, user1]);
        let names = [v0, abs, sqrt, user0, user1].map(|id| g.name(id));

        assert!(run(&mut g));

        let [v0, abs, sqrt, user0, user1] = names.map(|name| g.id(&name));
        g.assert_targets(&[
            VIEW_COPY,
            "aten.abs.default",
            "aten.sqrt.default",
            "aten.neg.default",
            "aten.relu.default",
        ]);
        assert_eq!(g.graph().node(v0).args[1].as_int_list(), Some(vec![2, 12]));
        for id in [v0, abs, sqrt, user0, user1] {
            assert_eq!(g.shape(id), &[2, 12], "{}", g.graph().node(id).name);
        }
        assert_eq!(g.graph().node(user0).args[0].as_node(), Some(sqrt));
        assert_eq!(g.graph().node(user1).args[0].as_node(), Some(sqrt));
    }

    #[test]
    fn a_fork_stops_the_chain() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v0 = g.view(x, &[6]);
        let relu = g.call("aten.relu.default", vec![node(v0)]);
        let neg = g.call("aten.neg.default", vec![node(v0)]);
        let v1 = g.view(relu, &[3, 2]);
        g.output(&[v1, neg]);

        assert!(!run(&mut g));
    }

    #[test]
    fn a_view_with_several_users_is_still_bypassed() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v0 = g.view(x, &[6]);
        let relu = g.call("aten.relu.default", vec![node(v0)]);
        let v1 = g.view(relu, &[3, 2]);
        let a = g.call("aten.neg.default", vec![node(v1)]);
        let b = g.call("aten.abs.default", vec![node(v1)]);
        g.output(&[a, b]);

        assert!(run(&mut g));

        g.assert_targets(&[VIEW_COPY, "aten.relu.default", "aten.neg.default", "aten.abs.default"]);
        assert_eq!(g.shape(relu), &[3, 2]);
    }

    #[test]
    fn unary_ops_after_the_last_view_keep_their_shape() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v0 = g.view(x, &[6]);
        let v1 = g.view(v0, &[3, 2]);
        let relu = g.call("aten.relu.default", vec![node(v1)]);
        let add = g.call("aten.add.Tensor", vec![node(relu), node(relu)]);
        g.output(&[add]);

        assert!(run(&mut g));

        g.assert_targets(&[VIEW_COPY, "aten.relu.default", "aten.add.Tensor"]);
        assert_eq!(g.shape(v0), &[3, 2]);
        assert_eq!(g.shape(relu), &[3, 2]);
    }

    #[test]
    fn non_unary_ops_stop_the_chain_and_dead_code_goes() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v0 = g.view(x, &[6]);
        let add = g.call("aten.add.Tensor", vec![node(v0), node(v0)]);
        let v1 = g.view(add, &[3, 2]);
        let dead = g.call("aten.neg.default", vec![node(x)]);
        g.output(&[v1]);
        let _ = dead;

        // Nothing fuses, so rc4 does not eliminate dead code either.
        assert!(!run(&mut g));
        assert_eq!(g.count("aten.neg.default"), 1);
    }
}
