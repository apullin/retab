//! Port of rc4's `CanonicalizeViewCopyPermutePass`
//! (`backends/transforms/canonicalize_view_copy_permute_pass.py`) as Arm
//! configures it (`backends/arm/_passes/canonicalize_view_copy_permute_pass.py`):
//! only `permute_copy` counts as a permute.
//!
//! For each linear chain of views and permutes it removes identity views and
//! permutes, turns permutes that only move size-1 dims into views, fuses
//! adjacent views and adjacent permutes, and sinks a lone singleton-inserting
//! view below its permute. While a chain stays longer than two it swaps the
//! first swappable view/permute pair and fuses again; each swap makes two ops
//! of the same kind adjacent, so the chain shrinks. Nodes are rewritten in
//! place (they keep their names and meta; only the value is refreshed), and
//! dead code is eliminated from the whole graph if anything changed.

use std::collections::HashSet;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::layout::ctx::LayoutCtx;
use crate::layout::dim_maps::{is_permutation, normalize_dim, normalize_dims, ViewMap};
use crate::layout::shape::{int_list, int_list_arg, refreshed_meta};

const VIEW: &str = "aten.view_copy.default";
const PERMUTE: &str = "aten.permute_copy.default";

pub(crate) struct IrCanonicalizeViewCopyPermutePass;

impl IrPass for IrCanonicalizeViewCopyPermutePass {
    fn name(&self) -> &str {
        "IrCanonicalizeViewCopyPermute"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        if !graph
            .live_call_function_ids()
            .into_iter()
            .any(|id| is_target(graph, id))
        {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let mut modified = false;
        for mut chain in collect_chains(&ctx) {
            loop {
                modified |= fuse_sequential_ops(&mut ctx, &mut chain);
                if chain.len() <= 2 {
                    break;
                }
                // `call` (`:58-86`): swap the first pair that can be swapped.
                let swap = (0..chain.len() - 1).find_map(|index| {
                    maybe_swap_args(&ctx, chain[index], chain[index + 1])
                        .map(|args| (index, args))
                });
                let Some((index, (op1_arg, op2_arg))) = swap else {
                    break;
                };
                let (op1, op2) = (chain[index], chain[index + 1]);
                let input = input_of(&ctx, op1);
                let op1_target = ctx.node(op1).target.clone();
                let op2_target = ctx.node(op2).target.clone();
                set_node_op(&mut ctx, op1, &op2_target, input, &op1_arg);
                set_node_op(&mut ctx, op2, &op1_target, op1, &op2_arg);
                modified = true;
            }
        }
        if modified {
            ctx.eliminate_dead_code();
            ctx.finish();
        }
        modified
    }
}

fn is_target(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    node.op == NodeOp::CallFunction && (node.target == VIEW || node.target == PERMUTE)
}

fn is_view(ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
    ctx.node(id).target == VIEW
}

fn is_permute(ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
    ctx.node(id).target == PERMUTE
}

fn input_of(ctx: &LayoutCtx<'_>, id: NodeId) -> NodeId {
    ctx.node(id).args[0]
        .as_node()
        .expect("views and permutes read a node")
}

fn shape(ctx: &LayoutCtx<'_>, id: NodeId) -> Vec<i64> {
    ctx.graph()
        .node_shape_vec(id)
        .expect("views, permutes and their inputs carry a value")
}

fn permute_dims(ctx: &LayoutCtx<'_>, id: NodeId) -> Vec<i64> {
    int_list(&ctx.node(id).args[1]).unwrap_or_default()
}

/// `_collect_chains` (`:100-136`): linear single-user chains of views and
/// permutes, each starting at the first unclaimed node in graph order.
fn collect_chains(ctx: &LayoutCtx<'_>) -> Vec<Vec<NodeId>> {
    let nodes: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| is_target(ctx.graph(), id))
        .collect();
    let mut remaining: HashSet<NodeId> = nodes.iter().copied().collect();
    let mut chains = Vec::new();
    for node in nodes {
        if !remaining.remove(&node) {
            continue;
        }
        let mut chain = vec![node];
        let mut current = node;
        while let Some(user) = ctx.graph().single_live_user(current) {
            // A view or permute has one input, so a user can't already belong to
            // another chain; rc4 raises if it does.
            if !is_target(ctx.graph(), user) || !remaining.remove(&user) {
                break;
            }
            chain.push(user);
            current = user;
        }
        chains.push(chain);
    }
    chains
}

/// `_fuse_sequential_ops` (`:138-213`): fuse and simplify until nothing changes.
fn fuse_sequential_ops(ctx: &mut LayoutCtx<'_>, chain: &mut Vec<NodeId>) -> bool {
    let mut any_changed = false;
    loop {
        let mut changed = false;
        let mut index = 0;
        while index < chain.len() {
            let node = chain[index];
            let input = input_of(ctx, node);

            if is_view(ctx, node) && shape(ctx, input) == shape(ctx, node) {
                remove_node(ctx, chain, index, input);
                changed = true;
                continue;
            }

            if is_permute(ctx, node) {
                let input_shape = shape(ctx, input);
                let rank = input_shape.len();
                let mut dims = permute_dims(ctx, node);
                if dims.iter().any(|&dim| dim < 0 || dim >= rank as i64) {
                    let Ok(normalized) = dims
                        .iter()
                        .map(|&dim| normalize_dim(dim, rank).map(|dim| dim as i64))
                        .collect::<Result<Vec<_>, _>>()
                    else {
                        // rc4 asserts on a dim outside the rank.
                        index += 1;
                        continue;
                    };
                    dims = normalized;
                    set_node_op(ctx, node, PERMUTE, input, &dims);
                    changed = true;
                }
                if dims.iter().enumerate().all(|(index, &dim)| dim == index as i64) {
                    remove_node(ctx, chain, index, input);
                    changed = true;
                    continue;
                }
                if is_singleton_permutation(&input_shape, &dims) {
                    let output_shape = shape(ctx, node);
                    set_node_op(ctx, node, VIEW, input, &output_shape);
                    changed = true;
                    continue;
                }
            }

            if fuse_pair(ctx, chain, index) {
                changed = true;
                continue;
            }
            index += 1;
        }
        if !changed {
            return any_changed;
        }
        any_changed = true;
    }
}

/// `_fuse_pair` (`:215-249`).
fn fuse_pair(ctx: &mut LayoutCtx<'_>, chain: &mut Vec<NodeId>, index: usize) -> bool {
    if index + 1 >= chain.len() {
        return false;
    }
    let (node, next) = (chain[index], chain[index + 1]);
    let input = input_of(ctx, node);

    if sink_singleton_view(ctx, chain, index) {
        return true;
    }
    if is_view(ctx, node) && is_view(ctx, next) {
        let next_shape = shape(ctx, next);
        set_node_op(ctx, node, VIEW, input, &next_shape);
        remove_node(ctx, chain, index + 1, node);
        return true;
    }
    if is_permute(ctx, node) && is_permute(ctx, next) {
        let dims = permute_dims(ctx, node);
        // Python indexes `dims` with the next permute's raw dims, so a negative
        // one counts from the end.
        let Some(fused) = permute_dims(ctx, next)
            .iter()
            .map(|&dim| {
                let index = if dim < 0 { dim + dims.len() as i64 } else { dim };
                usize::try_from(index).ok().and_then(|index| dims.get(index).copied())
            })
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        set_node_op(ctx, node, PERMUTE, input, &fused);
        remove_node(ctx, chain, index + 1, node);
        return true;
    }
    false
}

/// `_sink_singleton_view` (`:251-282`): `view(S).permute(P)` becomes
/// `permute(P').view(S')` for a lone pair whose view only inserts unit dims.
fn sink_singleton_view(ctx: &mut LayoutCtx<'_>, chain: &[NodeId], index: usize) -> bool {
    if chain.len() != 2 {
        return false;
    }
    let (view, permute) = (chain[index], chain[index + 1]);
    let input = input_of(ctx, view);
    if !is_view(ctx, view)
        || !is_permute(ctx, permute)
        || !only_inserts_singletons(&shape(ctx, input), &shape(ctx, view))
    {
        return false;
    }
    let Some((permute_arg, view_arg)) = view_permute_swap(ctx, view, permute) else {
        return false;
    };
    set_node_op(ctx, view, PERMUTE, input, &permute_arg);
    set_node_op(ctx, permute, VIEW, view, &view_arg);
    true
}

/// `_maybe_swap_args` (`:284-303`).
fn maybe_swap_args(ctx: &LayoutCtx<'_>, op1: NodeId, op2: NodeId) -> Option<(Vec<i64>, Vec<i64>)> {
    if is_permute(ctx, op1) && is_view(ctx, op2) {
        return permute_view_swap(ctx, op1, op2);
    }
    if is_view(ctx, op1) && is_permute(ctx, op2) {
        return view_permute_swap(ctx, op1, op2);
    }
    None
}

/// `_permute_view_swap` (`:305-339`): `permute(P).view(S)` as
/// `view(S').permute(P')`.
fn permute_view_swap(ctx: &LayoutCtx<'_>, permute: NodeId, view: NodeId) -> Option<(Vec<i64>, Vec<i64>)> {
    let x_shape = shape(ctx, input_of(ctx, permute));
    let permute_dims = normalize_dims(&permute_dims(ctx, permute), x_shape.len()).ok()?;
    let view_map = view_map(ctx, view)?;
    let permuted_axis = inverse(&permute_dims);
    let target_axis_order = view_map
        .map_permutation(&to_i64(&permuted_axis))
        .ok()
        .flatten()?;
    let view_shape_before_permute = target_axis_order
        .iter()
        .map(|&axis| view_map.target_shape()[axis])
        .collect();
    Some((view_shape_before_permute, to_i64(&inverse(&target_axis_order))))
}

/// `_view_permute_swap` (`:341-364`): `view(S).permute(P)` as
/// `permute(P').view(S')`, where `S'` is the permute's output shape.
fn view_permute_swap(ctx: &LayoutCtx<'_>, view: NodeId, permute: NodeId) -> Option<(Vec<i64>, Vec<i64>)> {
    let view_map = view_map(ctx, view)?;
    let mapped = view_map
        .map_permutation_inverse(&permute_dims(ctx, permute))
        .ok()
        .flatten()?;
    Some((to_i64(&mapped), shape(ctx, permute)))
}

/// `ViewMap(view_node)`, or `None` where rc4 catches an `AssertionError`
/// (`_view_map`, `:396-401`) or the map is not valid.
fn view_map(ctx: &LayoutCtx<'_>, view: NodeId) -> Option<ViewMap> {
    let source = shape(ctx, input_of(ctx, view));
    let target = int_list(&ctx.node(view).args[1])?;
    ViewMap::from_shapes(&source, &target)
        .ok()
        .filter(ViewMap::is_valid_map)
}

/// `_only_inserts_singletons` (`:372-381`).
fn only_inserts_singletons(input_shape: &[i64], output_shape: &[i64]) -> bool {
    let non_unit = |shape: &[i64]| shape.iter().copied().filter(|&dim| dim != 1).collect::<Vec<_>>();
    output_shape.len() > input_shape.len() && non_unit(output_shape) == non_unit(input_shape)
}

/// `_is_singleton_permutation` (`:383-394`): the permutation keeps the
/// non-singleton axes in order.
fn is_singleton_permutation(shape: &[i64], permutation: &[i64]) -> bool {
    let rank = shape.len();
    let Ok(normalized) = normalize_dims(permutation, rank) else {
        return false;
    };
    if !is_permutation(&normalized, rank) {
        return false;
    }
    let non_singleton: Vec<usize> = (0..rank).filter(|&axis| shape[axis] != 1).collect();
    let permuted: Vec<usize> = normalized
        .into_iter()
        .filter(|&axis| shape[axis] != 1)
        .collect();
    permuted == non_singleton
}

/// `_remove_node` (`:409-421`).
fn remove_node(ctx: &mut LayoutCtx<'_>, chain: &mut Vec<NodeId>, index: usize, replacement: NodeId) {
    let node = chain[index];
    assert_ne!(node, replacement);
    ctx.replace_all_uses_with(node, replacement);
    ctx.erase_node(node);
    chain.remove(index);
}

/// `_set_node_op` (`:423-433`): retarget `node` in place and refresh its value
/// as `refresh_permute_view_meta` does; the rest of its meta stays.
fn set_node_op(ctx: &mut LayoutCtx<'_>, node: NodeId, target: &str, input: NodeId, arg: &[i64]) {
    ctx.set_target(node, target);
    ctx.set_args(node, vec![IrArg::Node(input), int_list_arg(arg)]);
    let meta = refreshed_meta(ctx.graph(), node).expect("views and permutes have a shape rule");
    *ctx.meta_mut(node) = meta;
}

fn inverse(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; permutation.len()];
    for (index, &axis) in permutation.iter().enumerate() {
        inverse[axis] = index;
    }
    inverse
}

fn to_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&dim| dim as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{PERMUTE, VIEW};
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    fn run(g: &mut LayoutGraph) -> bool {
        IrCanonicalizeViewCopyPermutePass.run(g.graph_mut())
    }

    #[test]
    fn identity_views_and_permutes_go() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let v = g.view(x, &[2, 3, 4]);
        let p = g.permute(v, &[0, 1, 2]);
        let relu = g.call("aten.relu.default", vec![node(p)]);
        g.output(&[relu]);

        assert!(run(&mut g));

        g.assert_targets(&["aten.relu.default"]);
    }

    #[test]
    fn a_permute_of_only_unit_dims_becomes_a_view() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 1, 3]);
        let p = g.permute(x, &[1, 0, 2]);
        g.output(&[p]);
        let name = g.name(p);

        assert!(run(&mut g));

        let p = g.id(&name);
        g.assert_targets(&[VIEW]);
        assert_eq!(g.graph().node(p).args[1].as_int_list(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn negative_permute_dims_are_normalized() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let p = g.permute(x, &[0, -1, -2]);
        g.output(&[p]);

        assert!(run(&mut g));

        g.assert_permute_dims(&[&[0, 2, 1]]);
    }

    #[test]
    fn adjacent_views_and_adjacent_permutes_fuse_into_the_first() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let v0 = g.view(x, &[6, 4]);
        let v1 = g.view(v0, &[24]);
        let y = g.input("y", &[2, 3, 4, 5]);
        let p0 = g.permute(y, &[0, 2, 3, 1]);
        let p1 = g.permute(p0, &[1, 0, 2, 3]);
        g.output(&[v1, p1]);
        let names = [v0, p0].map(|id| g.name(id));

        assert!(run(&mut g));

        let [v0, p0] = names.map(|name| g.id(&name));
        g.assert_targets(&[VIEW, PERMUTE]);
        assert_eq!(g.shape(v0), &[24]);
        assert_eq!(g.graph().node(p0).args[1].as_int_list(), Some(vec![2, 0, 3, 1]));
        assert_eq!(g.shape(p0), &[4, 2, 5, 3]);
    }

    #[test]
    fn a_lone_singleton_view_sinks_below_its_permute() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v = g.view(x, &[2, 1, 3]);
        let p = g.permute(v, &[2, 1, 0]);
        g.output(&[p]);
        let names = [v, p].map(|id| g.name(id));

        assert!(run(&mut g));

        // The nodes swap roles in place.
        let [v, p] = names.map(|name| g.id(&name));
        assert_eq!(g.graph().node(v).target, PERMUTE);
        assert_eq!(g.graph().node(v).args[1].as_int_list(), Some(vec![1, 0]));
        assert_eq!(g.graph().node(p).target, VIEW);
        assert_eq!(g.shape(p), &[3, 1, 2]);
    }

    #[test]
    fn a_swap_lets_a_three_op_chain_fuse() {
        // view -> permute -> view has no adjacent pair to fuse until the first
        // view and the permute swap.
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let v0 = g.view(x, &[6, 4]);
        let p = g.permute(v0, &[1, 0]);
        let v1 = g.view(p, &[4, 6, 1]);
        g.output(&[v1]);

        assert!(run(&mut g));

        // The swap brings the two views together, and they fuse.
        assert_eq!(g.count(VIEW) + g.count(PERMUTE), 2);
        let output = g.graph().live_output_ids()[0];
        let last = match &g.graph().node(output).args[0] {
            IrArg::List(items) => items[0].as_node().unwrap(),
            IrArg::Node(id) => *id,
            _ => unreachable!(),
        };
        assert_eq!(g.shape(last), &[4, 6, 1]);
    }

    #[test]
    fn a_fork_ends_a_chain() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let p0 = g.permute(x, &[1, 0]);
        let p1 = g.permute(p0, &[1, 0]);
        let relu = g.call("aten.relu.default", vec![node(p0)]);
        g.output(&[p1, relu]);

        // p0 has two users, so p0 and p1 are separate chains and nothing fuses.
        assert!(!run(&mut g));
    }

    #[test]
    fn rewritten_nodes_keep_their_meta() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let p0 = g.permute(x, &[0, 2, 1]);
        let p1 = g.permute(p0, &[2, 0, 1]);
        g.output(&[p1]);
        g.graph_mut()
            .node_mut(p0)
            .meta
            .set_single_output_qparam(qparam(0.5, 1, "torch.int8"));
        let name = g.name(p0);

        assert!(run(&mut g));

        let p0 = g.id(&name);
        assert!(g.graph().node(p0).meta.output_qparam().is_some());
        assert_eq!(g.graph().node(p0).args[1].as_int_list(), Some(vec![1, 0, 2]));
    }

    #[test]
    fn helpers_match_rc4() {
        assert!(only_inserts_singletons(&[2, 3], &[2, 1, 3]));
        assert!(!only_inserts_singletons(&[2, 3], &[3, 2, 1]));
        assert!(!only_inserts_singletons(&[2, 1, 3], &[2, 3]));
        assert!(is_singleton_permutation(&[2, 1, 3], &[1, 0, 2]));
        assert!(!is_singleton_permutation(&[2, 1, 3], &[2, 1, 0]));
        assert!(!is_singleton_permutation(&[2, 3], &[0, 0]));
    }
}
