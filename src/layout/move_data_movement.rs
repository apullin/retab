//! Port of rc4's `MoveDataMovementOpsToSmallerDtypePass`
//! (`backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py`),
//! which runs after the permute/view propagation (`arm_pass_manager.py:731`).
//!
//! A permute (or a dim-order copy) next to a per-tensor RESCALE moves to the
//! side with the smaller element type, so it copies fewer bytes:
//!
//! ```text
//! x:i8 -> RESCALE:i32 -> permute:i32      ->   x:i8 -> permute:i8 -> RESCALE:i32
//! x:i32 -> permute:i32 -> RESCALE:i8      ->   x:i32 -> RESCALE:i8 -> permute:i8
//! ```
//!
//! A permute with several users is split: each narrowing RESCALE user gets a
//! copy after it, and the original goes once it has no users.
//!
//! [`move_data_movement_ops_to_smaller_dtype`] is the pass body inside a
//! [`LayoutCtx`]; [`IrMoveDataMovementOpsToSmallerDtypePass`] is the
//! standalone pass.
//!
//! **Order.** rc4 rescans the graph from the top after every rewrite
//! (`:31-65`, the `break`s), so it always rewrites the first node in graph
//! order whose move applies, and checks the move before a RESCALE producer
//! ahead of the move after a RESCALE user. That is quadratic. The port keeps
//! the same sequence of rewrites with a heap of the nodes whose move applies
//! ("ready" nodes), keyed by graph position: the minimum is the node rc4's
//! rescan would find. Whether a node is ready depends only on the node, its
//! input, that input's input when it is a RESCALE, and its users, so after a
//! rewrite only the nodes next to the ones it changed are re-checked
//! ([`affected_nodes`]). The heap drops entries whose node moved or stopped
//! being ready, and is rebuilt when the node order relabels its keys. Cost:
//! O(N) to seed, then O(log N + local degree) per rewrite.
//!
//! **Values.** rc4 sets some values during the loop and leaves others stale
//! until its closing `ExportPass` retrace (`:67-70`): a moved or split
//! dim-order copy keeps its old value (`refresh_permute_view_meta` only
//! refreshes views and permutes), and a split RESCALE keeps its old shape.
//! The decisions read element sizes from those values, so the port sets
//! exactly what rc4 sets, then re-derives the values the retrace would for the
//! nodes the pass touched and for anything downstream whose value then
//! changes (see [`retrace`]).
//!
//! **Spellings.** RETAB writes `_clone_dim_order` and `_to_dim_order_copy`
//! with or without their namespace, and emits `aten._to_copy.default` where
//! rc4 emits `_to_dim_order_copy` (`pointwise::is_transparent_copy`); all count
//! as data movement. Users are taken in graph order, where rc4 takes them in
//! `node.users` insertion order; which narrowing RESCALE a split serves first
//! changes only node names.
//!
//! **Unsound rewrites.** rc4 moves a permute before a RESCALE that has other
//! users too, which rewires them to the permuted value, and moves a casting
//! `_to_dim_order_copy` across a RESCALE, which changes the dtype its users
//! see. The port does the same. Where a rewired user comes before the
//! permute, rc4's lint raises and the port panics. Otherwise rc4's retrace
//! re-derives the values downstream by fake execution; the port re-derives
//! them with `layout::shape`'s rules ([`retraced_facts`]), and where it has no
//! rule, or rc4's retrace would raise on a shape it can't derive, it keeps the
//! old value. The replay fixtures and a 2529-case fuzz run match rc4 in every
//! case, including the 28 and 135 where rc4 raises.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::fx_node::{all_input_nodes, element_size, is_per_tensor_rescale, is_rescale};
use crate::layout::pointwise::{is_pointwise, is_transparent_copy};
use crate::layout::shape::{infer_facts, is_layout_transform, mark_contiguous, refreshed_meta, ShapeError};

const PERMUTE_COPY: &str = "aten.permute_copy.default";

/// `_DATA_MOVEMENT_OPS` (`:23-27`) in RETAB's spellings (see the module docs).
fn is_data_movement(node: &IrNode) -> bool {
    node.alive
        && node.op == NodeOp::CallFunction
        && (node.target == PERMUTE_COPY || is_transparent_copy(&node.target))
}

/// The standalone pass (`MoveDataMovementOpsToSmallerDtypePass()`,
/// `arm_pass_manager.py:731`).
pub(crate) struct IrMoveDataMovementOpsToSmallerDtypePass;

impl IrPass for IrMoveDataMovementOpsToSmallerDtypePass {
    fn name(&self) -> &str {
        "IrMoveDataMovementOpsToSmallerDtype"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let modified = move_data_movement_ops_to_smaller_dtype(&mut ctx);
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// The rewrite that applies to a data-movement node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Move {
    /// `_move_before_rescale` (`:100-118`): the node's input is a RESCALE
    /// that widens.
    BeforeRescale(NodeId),
    /// `_move_after_rescale` (`:120-137`) when the RESCALE is the node's only
    /// user, else `_split_after_rescale` (`:139-158`).
    AfterRescale(NodeId),
}

/// `MoveDataMovementOpsToSmallerDtypePass.call(graph_module)` (`:29-71`)
/// inside `ctx`; returns rc4's `modified`.
pub(crate) fn move_data_movement_ops_to_smaller_dtype(ctx: &mut LayoutCtx<'_>) -> bool {
    let mut ready = ReadyNodes::new(ctx);
    let movers: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| is_data_movement(ctx.node(id)))
        .collect();
    for id in movers {
        ready.update(ctx, id);
    }
    let mut touched: Vec<NodeId> = Vec::new();
    while let Some((node, chosen)) = ready.pop(ctx) {
        let Changed {
            moved,
            rewired,
            users_changed,
        } = match chosen {
            Move::BeforeRescale(rescale) => move_before_rescale(ctx, node, rescale),
            Move::AfterRescale(rescale) if ctx.graph().live_user_ids(node).len() == 1 => {
                move_after_rescale(ctx, node, rescale)
            }
            Move::AfterRescale(rescale) => split_after_rescale(ctx, node, rescale),
        };
        let changed: Vec<NodeId> = moved.iter().chain(&rewired).copied().collect();
        touched.extend(moved);
        for id in affected_nodes(ctx.graph(), &changed, &users_changed) {
            ready.update(ctx, id);
        }
        #[cfg(test)]
        ready.assert_complete(ctx);
    }
    if touched.is_empty() {
        return false;
    }
    // `graph_module.graph.lint()` (`:68`): a RESCALE whose other user came
    // before the permute now feeds it from below.
    assert!(
        ctx.is_topological(),
        "MoveDataMovementOpsToSmallerDtype: a node is used before it has been defined \
         (rc4's Graph.lint raises here)"
    );
    retrace(ctx, touched);
    true
}

/// Which move applies to `node`, as rc4's loop body decides (`:35-64`).
fn plan(ctx: &LayoutCtx<'_>, node: NodeId) -> Option<Move> {
    let graph = ctx.graph();
    let data_movement = graph.node(node);
    if !is_data_movement(data_movement) {
        return None;
    }
    let inputs = all_input_nodes(data_movement);
    let [producer] = inputs[..] else {
        return None;
    };
    if is_rescale(&graph.node(producer).target) {
        let smaller_side = *all_input_nodes(graph.node(producer))
            .first()
            .expect("IndexError: a RESCALE reads a node");
        if uses_wider_dtype(graph, node, producer, smaller_side) {
            return Some(Move::BeforeRescale(producer));
        }
    }
    let mut users = graph.live_user_ids(node);
    ctx.order().sort(&mut users);
    users
        .into_iter()
        .find(|&user| is_rescale(&graph.node(user).target) && uses_wider_dtype(graph, node, user, user))
        .map(Move::AfterRescale)
}

/// `_uses_wider_dtype` (`:73-87`): the RESCALE is per tensor and the data
/// movement's elements are wider than `smaller_side`'s.
fn uses_wider_dtype(graph: &IrGraph, data_movement: NodeId, rescale: NodeId, smaller_side: NodeId) -> bool {
    if !is_per_tensor_rescale(graph, rescale) {
        return false;
    }
    let size = |id: NodeId| graph.node_dtype(id).and_then(element_size);
    matches!((size(data_movement), size(smaller_side)), (Some(wide), Some(narrow)) if wide > narrow)
}

/// The nodes a rewrite changed.
struct Changed {
    /// The data movement and RESCALE nodes it rewired and gave values to (or
    /// left stale): what rc4's retrace re-derives.
    moved: Vec<NodeId>,
    /// Users that now read a different node holding the same value.
    rewired: Vec<NodeId>,
    /// Nodes of which only the user list changed.
    users_changed: Vec<NodeId>,
}

/// `_move_before_rescale` (`:100-118`): `q -> rescale -> node -> users`
/// becomes `q -> node -> rescale -> users`.
fn move_before_rescale(ctx: &mut LayoutCtx<'_>, node: NodeId, rescale: NodeId) -> Changed {
    let producer = all_input_nodes(ctx.node(rescale))[0];
    let movement_users = ctx.graph().live_user_ids(node);
    ctx.replace_input_with(node, rescale, producer);
    ctx.replace_input_with(rescale, producer, node);
    for &user in &movement_users {
        ctx.replace_input_with(user, node, rescale);
    }
    ctx.move_after(rescale, node);

    let old_movement = facts(ctx.graph(), node);
    let producer_dtype = facts(ctx.graph(), producer).dtype;
    set_facts(ctx, node, TensorFacts::new(old_movement.shape.clone(), producer_dtype));
    set_facts(ctx, rescale, old_movement);
    Changed {
        moved: vec![node, rescale],
        rewired: movement_users,
        users_changed: vec![producer],
    }
}

/// `_move_after_rescale` (`:120-137`): `p -> node -> rescale -> users`
/// becomes `p -> rescale -> node -> users`.
fn move_after_rescale(ctx: &mut LayoutCtx<'_>, node: NodeId, rescale: NodeId) -> Changed {
    let producer = all_input_nodes(ctx.node(node))[0];
    let rescale_users = ctx.graph().live_user_ids(rescale);
    ctx.replace_input_with(rescale, node, producer);
    ctx.replace_input_with(node, producer, rescale);
    for &user in &rescale_users {
        ctx.replace_input_with(user, rescale, node);
    }
    ctx.move_after(node, rescale);

    let old_rescale_dtype = facts(ctx.graph(), rescale).dtype;
    let producer_shape = facts(ctx.graph(), producer).shape;
    set_facts(ctx, rescale, TensorFacts::new(producer_shape, old_rescale_dtype));
    refresh_permute_view_meta(ctx, node);
    Changed {
        moved: vec![node, rescale],
        rewired: rescale_users,
        users_changed: vec![producer],
    }
}

/// `_split_after_rescale` (`:139-158`): `rescale` reads `node`'s input, and a
/// copy of `node` after it serves its users. The RESCALE's value keeps its old
/// shape until the retrace, and so does the copy's, derived from it.
fn split_after_rescale(ctx: &mut LayoutCtx<'_>, node: NodeId, rescale: NodeId) -> Changed {
    let producer = all_input_nodes(ctx.node(node))[0];
    let rescale_users = ctx.graph().live_user_ids(rescale);
    ctx.replace_input_with(rescale, node, producer);

    let original = ctx.node(node);
    let target = original.target.clone();
    let args: Vec<IrArg> = std::iter::once(IrArg::Node(rescale))
        .chain(original.args.iter().skip(1).cloned())
        .collect();
    let kwargs = original.kwargs.clone();
    let meta = original.meta.clone();
    let name = target.replace('.', "_");
    let branch = ctx.inserting_after(rescale, |ctx| {
        ctx.create_node(NodeOp::CallFunction, name, target, args, kwargs, meta)
    });
    refresh_permute_view_meta(ctx, branch);
    for &user in &rescale_users {
        ctx.replace_input_with(user, rescale, branch);
    }
    if !ctx.graph().has_live_users(node) {
        ctx.erase_node(node);
    }
    Changed {
        moved: vec![rescale, branch],
        rewired: rescale_users,
        users_changed: vec![node, producer],
    }
}

fn facts(graph: &IrGraph, id: NodeId) -> TensorFacts {
    graph
        .node(id)
        .meta
        .tensor_facts()
        .cloned()
        .unwrap_or_else(|| panic!("KeyError: 'val' on {}", graph.node(id).name))
}

fn set_facts(ctx: &mut LayoutCtx<'_>, id: NodeId, facts: TensorFacts) {
    let meta = ctx.meta_mut(id);
    meta.set_tensor_facts(facts);
    mark_contiguous(meta);
}

/// `refresh_permute_view_meta` (backends/transforms/permute_view_meta.py:22-56)
/// for the targets this pass moves: a permute takes its input's dtype at the
/// permuted shape; a dim-order copy is left alone.
fn refresh_permute_view_meta(ctx: &mut LayoutCtx<'_>, id: NodeId) {
    if ctx.node(id).target != PERMUTE_COPY {
        return;
    }
    let meta = refreshed_meta(ctx.graph(), id)
        .unwrap_or_else(|error| panic!("cannot refresh a moved permute: {error}"));
    *ctx.meta_mut(id) = meta;
}

/// Data-movement nodes whose move may have changed. A node's move reads the
/// node (its input, value and users), its input (target, args, inputs and
/// special dtype), that input's input when it is a RESCALE (value and special
/// dtype) and each user (target, args, inputs, special dtype and value). So
/// for a node in `changed` the candidates are itself, its inputs (it is their
/// user), its users (it is their input) and the users of its RESCALE users;
/// for a node in `users_changed`, only itself.
fn affected_nodes(graph: &IrGraph, changed: &[NodeId], users_changed: &[NodeId]) -> Vec<NodeId> {
    let mut nodes: Vec<NodeId> = users_changed.to_vec();
    for &id in changed {
        nodes.push(id);
        if !graph.node(id).alive {
            continue;
        }
        nodes.extend(all_input_nodes(graph.node(id)));
        for user in graph.live_user_ids(id) {
            nodes.push(user);
            if is_rescale(&graph.node(user).target) {
                nodes.extend(graph.live_user_ids(user));
            }
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    nodes.retain(|&id| is_data_movement(graph.node(id)));
    nodes
}

/// rc4's closing retrace (`:69`) for what this pass changed: one sweep in
/// graph order over the touched nodes and whatever downstream reads them,
/// re-deriving each value ([`retraced_facts`]). A touched node's users are
/// always visited, since they may now read a different node; another node's
/// users only when its value changed. A downstream node without a local rule
/// keeps its value. The sweep moves no node, so order keys stay valid.
fn retrace(ctx: &mut LayoutCtx<'_>, touched: Vec<NodeId>) {
    let touched: HashSet<NodeId> = touched.into_iter().filter(|&id| !ctx.is_erased(id)).collect();
    let mut pending: BinaryHeap<Reverse<(u64, NodeId)>> = BinaryHeap::new();
    let mut queued: HashSet<NodeId> = HashSet::new();
    let push = |ctx: &LayoutCtx<'_>, id: NodeId, pending: &mut BinaryHeap<_>, queued: &mut HashSet<NodeId>| {
        if ctx.node(id).op == NodeOp::CallFunction && queued.insert(id) {
            let key = ctx.order().key(id).expect("a live node is placed");
            pending.push(Reverse((key, id)));
        }
    };
    for &id in &touched {
        push(ctx, id, &mut pending, &mut queued);
    }
    while let Some(Reverse((_, id))) = pending.pop() {
        let is_touched = touched.contains(&id);
        let facts = match retraced_facts(ctx.graph(), id) {
            Ok(facts) => facts,
            Err(error) if is_touched => panic!("rc4's retrace raises here: {error}"),
            Err(_) => continue,
        };
        let changed = ctx.node(id).meta.tensor_facts() != Some(&facts);
        if changed {
            set_facts(ctx, id, facts);
        }
        if changed || is_touched {
            for user in ctx.graph().live_user_ids(id) {
                push(ctx, user, &mut pending, &mut queued);
            }
        }
    }
}

/// Pointwise targets whose integer result torch makes floating point (true
/// division); RETAB records that float, which `layout::shape`'s
/// dtype-preserving elementwise rule doesn't give.
const TRUE_DIVISION: &[&str] = &["aten.div.Tensor", "aten.div.Scalar", "aten.div_.Tensor"];

/// The value fake execution gives `id` in rc4's retrace. Views, permutes,
/// dim-order copies and RESCALEs take their dtype from an input or a dtype
/// argument, and pointwise ops other than true division promote their
/// operands' dtypes, as `layout::shape` derives them; everything else keeps
/// the dtype RETAB records, which differs from torch's on purpose for some
/// ops (`layout::shape`'s module docs), and only its shape is re-derived.
/// The derived dtypes equal the recorded ones unless rc4's move was unsound:
/// a `_to_dim_order_copy` that casts moves across a RESCALE and changes the
/// dtype its users see.
fn retraced_facts(graph: &IrGraph, id: NodeId) -> Result<TensorFacts, ShapeError> {
    let mut facts = infer_facts(graph, id)?;
    let node = graph.node(id);
    let target = node.target.as_str();
    let derives_dtype = is_layout_transform(target)
        || is_transparent_copy(target)
        || is_rescale(target)
        || (is_pointwise(target) && !TRUE_DIVISION.contains(&target));
    if !derives_dtype {
        if let Some(dtype) = node.meta.val_dtype() {
            facts.dtype = dtype.to_owned();
        }
    }
    Ok(facts)
}

#[cfg(test)]
thread_local! {
    static CHECK_READY_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Switch the per-rewrite ready-set check of test builds on or off for this
/// thread; it costs O(N) per rewrite. Returns the old setting.
#[cfg(test)]
pub(crate) fn set_check_ready_set(enabled: bool) -> bool {
    CHECK_READY_SET.with(|check| check.replace(enabled))
}

/// The ready nodes, and a min-heap of their graph positions.
struct ReadyNodes {
    heap: BinaryHeap<Reverse<(u64, NodeId)>>,
    ready: HashSet<NodeId>,
    /// `NodeOrder::relabel_count` when the heap's keys were read.
    relabels: usize,
}

impl ReadyNodes {
    fn new(ctx: &LayoutCtx<'_>) -> Self {
        Self {
            heap: BinaryHeap::new(),
            ready: HashSet::new(),
            relabels: ctx.order().relabel_count(),
        }
    }

    /// Re-check `id`; a ready node gets a heap entry at its current position.
    fn update(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) {
        if plan(ctx, id).is_some() {
            self.ready.insert(id);
            let key = ctx.order().key(id).expect("a live node is placed");
            self.heap.push(Reverse((key, id)));
        } else {
            self.ready.remove(&id);
        }
    }

    /// The first ready node in graph order, and its move.
    fn pop(&mut self, ctx: &LayoutCtx<'_>) -> Option<(NodeId, Move)> {
        if ctx.order().relabel_count() != self.relabels {
            self.relabels = ctx.order().relabel_count();
            self.heap = self
                .ready
                .iter()
                .map(|&id| Reverse((ctx.order().key(id).expect("a ready node is placed"), id)))
                .collect();
        }
        while let Some(Reverse((key, id))) = self.heap.pop() {
            if !self.ready.contains(&id) || ctx.order().key(id) != Some(key) {
                continue;
            }
            if let Some(chosen) = plan(ctx, id) {
                return Some((id, chosen));
            }
            self.ready.remove(&id);
        }
        None
    }

    /// Test builds check after every rewrite that the ready set is exactly
    /// what rc4's rescan would see, so a missed neighbour fails loudly.
    /// Timing tests switch the check off ([`set_check_ready_set`]).
    #[cfg(test)]
    fn assert_complete(&self, ctx: &LayoutCtx<'_>) {
        if !CHECK_READY_SET.with(std::cell::Cell::get) {
            return;
        }
        for id in ctx.order().iter() {
            assert_eq!(
                plan(ctx, id).is_some(),
                self.ready.contains(&id),
                "the ready set is stale at {}",
                ctx.node(id).name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_meta::MetaValue;
    use crate::layout::fx_node::SPECIAL_DTYPE_KEY;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph) -> bool {
        IrMoveDataMovementOpsToSmallerDtypePass.run(g.graph_mut())
    }

    fn input_of(g: &LayoutGraph, id: NodeId) -> NodeId {
        g.graph().node(id).arg_node(0).expect("a node input")
    }

    /// A chain of permutes ahead of a narrowing RESCALE: the last permute
    /// moves first, which makes the one before it ready, as rc4's rescan
    /// from the top finds.
    #[test]
    fn a_permute_chain_moves_below_a_narrowing_rescale() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
        let p1 = g.permute(x, &[0, 2, 3, 1]);
        let p2 = g.permute(p1, &[0, 3, 1, 2]);
        let r = g.rescale(p2, "torch.int8", 0.5);
        g.output(&[r]);

        assert!(run(&mut g));

        g.assert_targets(&[RESCALE, PERMUTE, PERMUTE]);
        let calls: Vec<NodeId> = g
            .graph()
            .topological_live_node_ids()
            .into_iter()
            .filter(|&id| g.graph().node(id).op == NodeOp::CallFunction)
            .collect();
        assert_eq!(g.shape(calls[0]), &[1, 2, 3, 4]);
        assert_eq!(g.dtype(calls[1]), "torch.int8");
        assert_eq!(g.dtype(calls[2]), "torch.int8");
        assert_eq!(g.shape(calls[2]), &[1, 2, 3, 4]);
    }

    /// A moved clone keeps its stale int32 value during the loop (rc4 only
    /// refreshes views and permutes): it moves below the int8 RESCALE, then
    /// still looks wider than the int16 one and moves below it, then above it
    /// again (`_move_before_rescale` finally gives it int8). rc4 ends the
    /// same way (checked in the rc4 venv).
    #[test]
    fn a_moved_clone_decides_on_its_stale_value() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], "torch.int32");
        let clone = g.call(CLONE, vec![node(x)]);
        let narrow = g.rescale(clone, "torch.int8", 0.5);
        let widen = g.rescale(narrow, "torch.int16", 0.5);
        g.output(&[widen]);
        let clone = g.name(clone);

        assert!(run(&mut g));

        g.assert_targets(&[RESCALE, CLONE, RESCALE]);
        assert_eq!(g.dtype(g.id(&clone)), "torch.int8");
    }

    #[test]
    fn per_channel_and_special_dtype_rescales_block_the_move() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
        let p = g.permute(x, &[0, 2, 3, 1]);
        let r = g.call(
            RESCALE,
            vec![
                node(p),
                IrArg::Dtype("torch.int8".into()),
                IrArg::List(vec![IrArg::Float(1.0), IrArg::Float(2.0)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
        );
        let y = g.typed_input("y", &[1, 2, 3, 4], "torch.int32");
        let q = g.permute(y, &[0, 2, 3, 1]);
        let s = g.rescale(q, "torch.int8", 0.5);
        g.graph_mut()
            .node_mut(q)
            .meta
            .set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("INT48".into()));
        g.output(&[r, s]);

        assert!(!run(&mut g));
    }

    /// A permute with two narrowing RESCALE users is split: one copy per
    /// RESCALE, each after it, and the original goes.
    #[test]
    fn a_shared_permute_splits_per_narrowing_rescale() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
        let p = g.permute(x, &[0, 2, 3, 1]);
        let r1 = g.rescale(p, "torch.int8", 1.0);
        let r2 = g.rescale(p, "torch.int8", 2.0);
        g.output(&[r1, r2]);

        assert!(run(&mut g));

        g.assert_targets(&[RESCALE, PERMUTE, RESCALE, PERMUTE]);
        let output = g.graph().output_node().unwrap();
        let IrArg::List(results) = &output.args[0] else {
            panic!("a list output");
        };
        for result in results {
            let permute = result.as_node().unwrap();
            assert_eq!(g.dtype(permute), "torch.int8");
            assert_eq!(g.shape(permute), &[1, 3, 4, 2]);
            let rescale = input_of(&g, permute);
            assert_eq!(g.shape(rescale), &[1, 2, 3, 4]);
            assert_eq!(input_of(&g, rescale), g.id("x"));
        }
    }

    /// rc4's lint raises when the RESCALE's other user sits between it and
    /// the permute that moves above it.
    #[test]
    #[should_panic(expected = "used before it has been defined")]
    fn a_user_between_the_rescale_and_the_permute_fails_like_rc4() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int8");
        let r = g.rescale(x, "torch.int32", 0.5);
        let neg = g.call(NEG, vec![node(r)]);
        let p = g.permute(r, &[0, 2, 3, 1]);
        g.output(&[p, neg]);
        run(&mut g);
    }

    /// When the RESCALE's other user comes after the permute, rc4 rewires it
    /// to the permuted value and its retrace re-derives the user's shape.
    #[test]
    fn a_later_user_of_the_rescale_is_rewired_like_rc4() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int8");
        let r = g.rescale(x, "torch.int32", 0.5);
        let p = g.permute(r, &[0, 2, 3, 1]);
        let neg = g.call(NEG, vec![node(r)]);
        g.output(&[p, neg]);
        let neg = g.name(neg);

        assert!(run(&mut g));

        g.assert_targets(&[PERMUTE, RESCALE, NEG]);
        assert_eq!(g.shape(g.id(&neg)), &[1, 3, 4, 2]);
        assert_eq!(g.dtype(g.id(&neg)), "torch.int32");
    }
}
