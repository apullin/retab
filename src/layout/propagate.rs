//! Port of rc4's permute/view propagation passes,
//! `PropagateViewCopyPermuteDownPass` and `PropagateViewCopyPermuteUpPass`
//! (`backends/transforms/propagate_view_copy_permute_pass.py`), with Arm's TOSA
//! overrides (`TosaPropagationOverrides`,
//! `backends/arm/_passes/propagate_view_copy_permute_pass.py`), constructed as
//! `arm_pass_manager.py:726-727` constructs them:
//! `PropagateViewCopyPermuteDownPass(compile_spec, exported_program)`, then the
//! Up pass (APU-159). Line references without a file are to the shared pass.
//!
//! The Down pass sinks each permute or view as far as it can go, the Up pass
//! hoists it as far as it can go, and the two meet where copies cancel:
//!
//! ```text
//! Down:  x -> permute -> relu -> neg -> y      ->   x -> relu -> neg -> permute -> y
//! Up:    x -> relu -> neg -> permute -> y      ->   x -> permute -> relu -> neg -> y
//! ```
//!
//! A copy crosses pointwise ops (torch `Tag.pointwise`), transparent copies,
//! keepdim reductions and slices, with their dims remapped through
//! `layout::dim_maps`; Up also swaps views across slices and reductions and
//! splits a cat whose users are all the same permute; Down also splits a
//! permute over a fork, or moves it past a single-exit pointwise region. Fork
//! splits and region moves are planned in full before the graph changes; a
//! walk assigns each crossed node's swapped args as it advances.
//!
//! # The driver (`:106-139`)
//!
//! Fuse horizontally (Up: FuseDuplicateUsers; Down:
//! FuseIdenticalInputTransforms), fuse vertically (RemovePermutesAroundElementwiseTosaOps
//! when an exported program is given, then CanonicalizeViewCopyPermute), then
//! repeat: sweep every live copy, Up in graph order and Down in reverse; when a
//! sweep changes nothing, fuse again, and stop when the fusions change nothing
//! either. The fusions are the `layout` ports of those passes, run in this
//! pass's [`LayoutCtx`].
//!
//! # TOSA overrides
//!
//! [`Overrides::Tosa`]: only keepdim `mean.dim` is a reduction (a non-keepdim
//! one panics, as rc4 raises); a per-tensor RESCALE is elementwise and any
//! RESCALE is transparent (a rank-changing view can't cross it); TABLE is a
//! multi-input elementwise op behind a shape guard (no `[N, 1, 1, C]` operand
//! with `N > 1`, MLBEDSW-11805); a permute never moves where it or a node it
//! would cross reads or holds INT48 data; a SCATTER user pins its operand.
//!
//! **Pending TABLEs.** rc4 has replaced quantized table ops with
//! `TABLE(x, table_constant)` by phase 6, RETAB still holds the aten op
//! (`layout::pending`). A pending table is treated as that TABLE, including its
//! table constant, which RETAB has no node for: the constant is a second input
//! that only the table uses and that is never layout-invariant
//! ([`Propagator::has_table_constant`]). So Up never hoists past a pending
//! table's input (two inputs end the walk, `:250-256`), and a pending table
//! makes a pointwise region unsafe (`:1209-1222`) and can't be crossed by a
//! fork split's elementwise rule (`:1320-1336`). A 16-bit table that needs a
//! left shift is a TABLE producing int32 followed by a per-tensor RESCALE in
//! rc4 (insert_table_ops.py:400-417), and the walks treat the two differently:
//! the RESCALE is transparent, so a rank-changing view moving down crosses the
//! TABLE and stops at the RESCALE, parked between the two. So every entry
//! point that walks gives each such table rc4's two nodes first
//! (`layout::pending::split_table_rescales`: the pending table holds the
//! TABLE's int32 value and a real RESCALE node follows it) and folds back the
//! ones whose RESCALE still follows alone at the end (`merge_table_rescales`),
//! where the replay generator restores rc4's two nodes as one. A table that
//! stays split marks a node rc4 parked inside it.
//!
//! # Differences from rc4
//!
//! - **No re-trace.** rc4 re-traces the whole graph with `ExportPass` after
//!   its first vertical fusion and at the end (`_retrace`, `:159-162`), and
//!   every fusion pass re-traces when it changes the graph. In a sound
//!   rewrite the values that change are exactly those rc4's local refresh
//!   sets (`:164-192`, crossed nodes and the moved copy), plus the new nodes'
//!   values it sets explicitly, so the port refreshes those ([`refresh_node`]).
//!   At each re-trace point it eliminates dead code (the driver's), returns
//!   every node's users to graph order (below) and re-derives the users of the
//!   refreshed nodes, which only an unsound rewrite changes
//!   ([`Propagator::retrace_stale`]). Node names are kept where the re-trace
//!   renames.
//! - **A cat's repeated input.** `_maybe_split_upwards_cat_fanout` rebuilds a
//!   cat from `all_input_nodes`, which lists a repeated input once, so rc4
//!   turns `cat([x, x])` into `cat([permute(x)])` and changes the result. The
//!   port keeps every occurrence, each reading the input's one new permute.
//! - **The walk's args after a region move.** Down's advance assigns the
//!   swapped args as it crosses each node (`:234-235`), before it knows how
//!   the walk ends. When the fork after a crossed mean or slice can't be split,
//!   rc4 falls back to a region move planned from the original permute
//!   (`:1036-1039`), which keeps those args: the crossed node then reads the
//!   moved permute's output with dims meant for its input. Its users change
//!   shape, or rc4's next re-trace raises (a channel split after a clamp cuts
//!   W instead of C). Where the kept dims change what a crossed node computes
//!   ([`same_op_args`]: a keepdim mean over the same axes, or over other
//!   size-1 axes, computes the same), the port takes the walk's assignments
//!   back before the region move ([`WalkWrites::undo`]), so the crossed nodes
//!   read what they did, and the replays list rc4's cases as known
//!   divergences; elsewhere it keeps rc4's graph.
//! - **`IrArg::None` dims.** RETAB's importer reads `[]` as `IrArg::None`, so a
//!   reduction's dims of `None` read as `[]` (every dim), where rc4 sees a
//!   Python `None` and declines.
//!
//! # User order
//!
//! FX keeps each node's users in insertion order: every args update moves the
//! updated node to the end of its inputs' users, and a re-trace rebuilds the
//! graph in node order. rc4 iterates `node.users` in the walks (Down's next
//! nodes and the fork and region searches, the rewires in `_move_node`,
//! `_apply_fork_split` and the cat split) and, through the driver, in
//! RemovePermutesAroundElementwiseTosaOps, which runs after sweeps that rewired
//! nodes and before any re-trace, so the order the sweeps leave decides which
//! of two regions sharing a permute it rewrites. The port reads users in that
//! order everywhere rc4 iterates them (`LayoutCtx::users_in_fx_order`),
//! assigns args where rc4 does even when they don't change (each assignment
//! reorders), and resets the order where rc4 re-traces: at `_retrace` and
//! after each fusion pass that changed the graph ([`Propagator::retrace_point`]).
//! A failed walk changes nothing, and its outcome doesn't depend on the order
//! (a fork plan is all-or-nothing, and the region search rejects a node on
//! every path), so the memo can still skip it; a node whose users reorder gets
//! a new version anyway.
//!
//! # Complexity
//!
//! rc4 rescans every node on every sweep and re-traces the whole graph between
//! steps. Here a sweep visits only the live copies (kept in a list rebuilt
//! after each fusion round and extended with the copies the sweep creates),
//! ordered once per sweep: O(T log T) for T copies. A walk that failed is
//! memoized with the ctx's attempt memo under the version stamps of every node
//! it read ([`Reads`]), and is retried only after one of them changes, so an
//! unchanged region costs one memo check per sweep. A walk costs the nodes it
//! passes and their neighbours; a region search costs the region and its
//! boundary, and rc4's `enumerate(graph.nodes)` order index (`:1108-1111`) is a
//! sort by the ctx's order keys. Reading a node's k users in FX order sorts
//! them by their stamps, O(k log k). The fusions cost what their ports cost:
//! an O(N) scan each, with memos that skip unchanged nodes across rounds. A
//! re-trace point restamps every node once, O(N), and re-derives only the
//! users of the nodes refreshed since the last one, going further only where
//! a value changed. Splitting and folding 16-bit tables is an O(N) scan per
//! pass. In all, O(R·N + I·T·log T) plus the rewrites and the retried walks,
//! for R fusion rounds and I sweeps; rc4 is O((R + I)·N) re-traces on top of
//! the same work.
//! `propagate_scaling_tests` times both passes on 50-60k-node graphs at
//! 0.06-0.4 s in a release test build on a loaded machine (load average
//! about 50 on 12 cores), 4.5-6 times the time of graphs a quarter the size.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::canonicalize::{canonicalize_view_copy_permute, is_singleton_permutation};
use crate::layout::ctx::{LayoutCtx, RuleId};
use crate::layout::dim_maps::{PermuteMap, ViewMap};
use crate::layout::dtype::{is_float_dtype, value_dtype};
use crate::layout::fuse_duplicate_users::{fuse_duplicate_users, TableStage, TOSA_EXCLUDED_TARGETS};
use crate::layout::fuse_identical_input_transforms::{
    fuse_identical_input_transforms, is_layout_invariant, FuseIdenticalInputTransforms, DELEGATION_TAG_META_KEY,
};
use crate::layout::fx_node::{all_input_nodes, is_per_tensor_rescale, is_rescale, rc4_target, special_dtype};
use crate::layout::pending::{is_pending_table, merge_table_rescales, split_table_rescales};
use crate::layout::pointwise::{is_pointwise, is_transparent_copy};
use crate::layout::remove_permutes::{remove_permutes_in_ctx, DeadCode, RemovePermutesConfig, RemovePermutesMemo};
use crate::layout::shape::{has_local_rule, int_list, mark_contiguous, refreshed_meta};
use crate::permutation::dims_to_i64;

/// `_VIEW_TARGET` (`:59`).
pub(crate) const VIEW_TARGET: &str = "aten.view_copy.default";
/// `_VIEW_DEFAULT_TARGET` (`:60`).
pub(crate) const VIEW_DEFAULT_TARGET: &str = "aten.view.default";
/// `_PERMUTE_TARGET` (`:61`), the only permute target Arm's passes use.
pub(crate) const PERMUTE_TARGET: &str = "aten.permute_copy.default";
const MEAN_DIM: &str = "aten.mean.dim";
const SUM_DIM_INT_LIST: &str = "aten.sum.dim_IntList";
const SLICE_COPY: &str = "aten.slice_copy.Tensor";
const CAT: &str = "aten.cat.default";
/// `exir_ops.backend.tosa.TABLE.default` and `.SCATTER.default`, as
/// `rc4_target` spells them.
const TABLE: &str = "backend.tosa.TABLE.default";
const SCATTER: &str = "backend.tosa.SCATTER.default";

/// Which way copies move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// `PropagateViewCopyPermuteUpPass` (`:526-886`).
    Up,
    /// `PropagateViewCopyPermuteDownPass` (`:889-1418`).
    Down,
}

/// Which answers the extension points give.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Overrides {
    /// The shared pass's defaults, as rc4's shared tests construct it.
    #[cfg(test)]
    Shared,
    /// `TosaPropagationOverrides`, Arm's passes.
    Tosa,
}

/// A replacement `blocks_moving` (`:428-441`), as rc4's tests subclass the
/// pass: `(graph, moving_node, frontier, next_nodes)`.
pub(crate) type BlocksMovingHook = fn(&IrGraph, NodeId, NodeId, &[NodeId]) -> bool;

/// How a propagation pass is constructed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PropagateConfig {
    pub(crate) direction: Direction,
    pub(crate) overrides: Overrides,
    /// Whether an `exported_program` was passed. With the TOSA overrides it
    /// makes `make_fusion_pass` return RemovePermutesAroundElementwiseTosaOps
    /// (backends/arm/_passes/propagate_view_copy_permute_pass.py:90-93).
    pub(crate) exported_program: bool,
    /// A test subclass's `blocks_moving`, which replaces the configured one.
    pub(crate) blocks_moving_hook: Option<BlocksMovingHook>,
}

impl PropagateConfig {
    /// The pass as `arm_pass_manager.py:726-727` constructs it.
    pub(crate) const fn arm(direction: Direction) -> Self {
        Self {
            direction,
            overrides: Overrides::Tosa,
            exported_program: true,
            blocks_moving_hook: None,
        }
    }

    /// Arm's pass constructed without arguments, as rc4's graph-level tests
    /// do (`PropagateViewCopyPermuteDownPass()`): no fusion pass.
    #[cfg(test)]
    pub(crate) const fn arm_without_program(direction: Direction) -> Self {
        Self {
            direction,
            overrides: Overrides::Tosa,
            exported_program: false,
            blocks_moving_hook: None,
        }
    }

    /// The shared pass constructed without arguments.
    #[cfg(test)]
    pub(crate) const fn shared(direction: Direction) -> Self {
        Self {
            direction,
            overrides: Overrides::Shared,
            exported_program: false,
            blocks_moving_hook: None,
        }
    }

    fn class_name(&self) -> &'static str {
        match self.direction {
            Direction::Up => "PropagateViewCopyPermuteUpPass",
            Direction::Down => "PropagateViewCopyPermuteDownPass",
        }
    }

    /// The memo rule for a walk that changed nothing; one per configuration,
    /// so passes sharing a ctx never read each other's records.
    fn rule(&self) -> RuleId {
        match (self.direction, self.overrides, self.blocks_moving_hook.is_some()) {
            (_, _, true) => RuleId("propagate_view_copy_permute.hooked"),
            (Direction::Up, Overrides::Tosa, false) => RuleId("propagate_view_copy_permute.up.tosa"),
            (Direction::Down, Overrides::Tosa, false) => RuleId("propagate_view_copy_permute.down.tosa"),
            #[cfg(test)]
            (Direction::Up, Overrides::Shared, false) => RuleId("propagate_view_copy_permute.up.shared"),
            #[cfg(test)]
            (Direction::Down, Overrides::Shared, false) => RuleId("propagate_view_copy_permute.down.shared"),
        }
    }
}

/// The Arm pass at `arm_pass_manager.py:726` (Down) or `:727` (Up), or any
/// other construction for tests.
pub(crate) struct IrPropagateViewCopyPermutePass {
    pub(crate) config: PropagateConfig,
}

impl IrPropagateViewCopyPermutePass {
    /// `PropagateViewCopyPermuteDownPass(self.compile_spec, exported_program)`.
    pub(crate) const ARM_DOWN: Self = Self {
        config: PropagateConfig::arm(Direction::Down),
    };
    /// `PropagateViewCopyPermuteUpPass(self.compile_spec, exported_program)`.
    pub(crate) const ARM_UP: Self = Self {
        config: PropagateConfig::arm(Direction::Up),
    };
}

impl IrPass for IrPropagateViewCopyPermutePass {
    fn name(&self) -> &str {
        match self.config.direction {
            Direction::Up => "IrPropagateViewCopyPermuteUp",
            Direction::Down => "IrPropagateViewCopyPermuteDown",
        }
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let modified = propagate_view_copy_permute(&mut ctx, &self.config);
        if modified {
            ctx.finish();
        }
        modified
    }
}

#[cfg(test)]
thread_local! {
    /// Which rewrite branches ran on this thread, for the replay coverage test.
    pub(crate) static COVERAGE: std::cell::RefCell<HashMap<&'static str, usize>> =
        std::cell::RefCell::new(HashMap::new());
    /// Whether a sweep re-runs each walk the memo skips, to check it fails
    /// (on in test builds; the scaling tests turn it off).
    pub(crate) static CHECK_SKIPPED_WALKS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// `_retrace` calls (`:159-162`) on this thread, for rc4's retrace-count test.
    pub(crate) static RETRACES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// `_refresh_propagation_meta` calls on this thread.
    pub(crate) static LOCAL_REFRESHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Walks started (`_propagate` calls the sweeps made) on this thread, for
    /// the scaling tests.
    pub(crate) static WALKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// When set, the driver logs the graph with each node's users in
    /// torch.fx's order after each sweep and around each fusion pass, the
    /// points tests/fixtures/propagate_review/trace_driver.py logs in rc4.
    pub(crate) static TRACE: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

/// Log `ctx` under `label` in `TRACE` (test builds only).
#[inline]
fn trace(_ctx: &LayoutCtx<'_>, _label: impl FnOnce() -> String) {
    #[cfg(test)]
    TRACE.with(|trace| {
        if let Some(log) = trace.borrow_mut().as_mut() {
            log.push(format!("{}\n{}", _label(), describe_users(_ctx)));
        }
    });
}

/// Each node in order with its args and its users in torch.fx's order, by
/// position, as trace_driver.py prints rc4's graph.
#[cfg(test)]
fn describe_users(ctx: &LayoutCtx<'_>) -> String {
    let order: Vec<NodeId> = ctx.order().iter().collect();
    let position: HashMap<NodeId, usize> = order.iter().enumerate().map(|(index, &id)| (id, index)).collect();
    fn arg_text(arg: &IrArg, position: &HashMap<NodeId, usize>) -> String {
        match arg {
            IrArg::Node(id) => format!("%{}", position[id]),
            IrArg::List(items) => format!(
                "[{}]",
                items.iter().map(|item| arg_text(item, position)).collect::<Vec<_>>().join(", ")
            ),
            IrArg::Float(value) => format!("{value:?}"),
            IrArg::Dtype(dtype) => dtype.clone(),
            other => python_repr(other),
        }
    }
    let mut lines = Vec::new();
    for &id in &order {
        let node = ctx.node(id);
        if node.op == NodeOp::Output {
            continue;
        }
        let (target, args) = match node.op {
            NodeOp::CallFunction => (
                node.target.clone(),
                node.args.iter().map(|arg| arg_text(arg, &position)).collect::<Vec<_>>(),
            ),
            _ => ("placeholder".to_string(), Vec::new()),
        };
        let users = ctx.users_in_fx_order(id);
        let mut names: Vec<String> = users
            .iter()
            .filter(|&&user| ctx.node(user).op != NodeOp::Output)
            .map(|user| format!("'%{}'", position[user]))
            .collect();
        if users.iter().any(|&user| ctx.node(user).op == NodeOp::Output) {
            names.push("'out'".to_string());
        }
        lines.push(format!("  %{} {target}({}) users=[{}]", position[&id], args.join(", "), names.join(", ")));
    }
    lines.join("\n")
}

/// `PropagateViewCopyPermutePass.call(graph_module)` (`:106-139`) inside `ctx`;
/// returns rc4's `modified`. 16-bit pending tables take rc4's two-node shape
/// for the pass ([`with_table_rescales`]).
pub(crate) fn propagate_view_copy_permute(ctx: &mut LayoutCtx<'_>, config: &PropagateConfig) -> bool {
    with_table_rescales(ctx, config, |ctx| drive(ctx, config))
}

/// The driver proper (`:106-139`).
fn drive(ctx: &mut LayoutCtx<'_>, config: &PropagateConfig) -> bool {
    let mut propagator = Propagator::new(config);
    let mut modified = propagator.fuse_horizontal(ctx);
    let vertical = propagator.fuse_vertical(ctx);
    modified |= vertical;
    if vertical {
        propagator.retrace(ctx);
    }
    propagator.collect_transforms(ctx);
    loop {
        let swept = propagator.propagate_iteration(ctx);
        trace(ctx, || format!("after sweep modified={}", if swept { "True" } else { "False" }));
        if swept {
            modified = true;
            continue;
        }
        let mut iteration_modified = propagator.fuse_horizontal(ctx);
        iteration_modified |= propagator.fuse_vertical(ctx);
        modified |= iteration_modified;
        if !iteration_modified {
            break;
        }
        propagator.collect_transforms(ctx);
    }
    if modified {
        propagator.retrace(ctx);
    }
    modified
}

/// One sweep, `_propagate_iteration` (`:141-152`), without the fusions: rc4's
/// tests call it directly.
#[cfg(test)]
pub(crate) fn propagate_iteration(ctx: &mut LayoutCtx<'_>, config: &PropagateConfig) -> bool {
    with_table_rescales(ctx, config, |ctx| {
        let mut propagator = Propagator::new(config);
        propagator.collect_transforms(ctx);
        propagator.propagate_iteration(ctx)
    })
}

/// `_propagate(node)` (`:238-282`) for one copy, which rc4's tests call
/// directly.
#[cfg(test)]
pub(crate) fn propagate_node(ctx: &mut LayoutCtx<'_>, config: &PropagateConfig, node: NodeId) -> bool {
    with_table_rescales(ctx, config, |ctx| {
        let mut propagator = Propagator::new(config);
        let mut reads = Reads::default();
        propagator.propagate(ctx, node, &mut reads)
    })
}

/// Run `f` with every 16-bit pending table in rc4's shape, a TABLE and its own
/// RESCALE node (`layout::pending::split_table_rescales`; Arm's passes only,
/// as only they know TABLE), then fold back each table whose RESCALE still
/// follows it alone (`merge_table_rescales`). A table left split marks a node
/// rc4 parked between its TABLE and RESCALE. Returns `f`'s result: the split
/// and the fold are representation changes, not rewrites rc4 reports. The
/// fold after a single sweep or walk restamps the RESCALE's users, so a test
/// calling those again in the same ctx sees them later in their inputs' users
/// than rc4 would.
fn with_table_rescales(ctx: &mut LayoutCtx<'_>, config: &PropagateConfig, f: impl FnOnce(&mut LayoutCtx<'_>) -> bool) -> bool {
    if config.overrides != Overrides::Tosa {
        return f(ctx);
    }
    if !split_table_rescales(ctx).is_empty() {
        cover("table.split");
    }
    let modified = f(ctx);
    if merge_table_rescales(ctx) > 0 {
        cover("table.kept_split");
    }
    modified
}

/// `_move_node(node, frontier, previous_frontier)`, which one of rc4's tests
/// calls directly.
#[cfg(test)]
pub(crate) fn move_node(
    ctx: &mut LayoutCtx<'_>,
    config: &PropagateConfig,
    node: NodeId,
    frontier: NodeId,
    previous_frontier: NodeId,
) {
    Propagator::new(config).move_node(ctx, node, frontier, previous_frontier);
}

/// `is_swappable(node)`, which panics for a non-keepdim reduction under the
/// TOSA overrides, as rc4 raises; rc4's tests call it directly.
#[cfg(test)]
pub(crate) fn is_swappable(ctx: &LayoutCtx<'_>, config: &PropagateConfig, node: NodeId) -> bool {
    Propagator::new(config).is_swappable(ctx, node, &mut Reads::default())
}

/// Count a branch in `COVERAGE` (test builds only).
#[inline]
fn cover(_branch: &'static str) {
    #[cfg(test)]
    COVERAGE.with(|coverage| *coverage.borrow_mut().entry(_branch).or_default() += 1);
}

/// The nodes a walk read. A walk that fails changes nothing and depends on
/// nothing else, so it fails again until one of them changes version.
#[derive(Default)]
struct Reads(Vec<NodeId>);

impl Reads {
    fn add(&mut self, id: NodeId) {
        self.0.push(id);
    }

    fn extend(&mut self, ids: impl IntoIterator<Item = NodeId>) {
        self.0.extend(ids);
    }
}

/// The arg assignments one walk made as it advanced (`:234-235`), in order,
/// with each assigned node's args from before the walk.
#[derive(Default)]
struct WalkWrites {
    order: Vec<NodeId>,
    before: HashMap<NodeId, Vec<IrArg>>,
    /// The args each assignment wrote, for [`Self::redo`].
    written: Vec<Vec<IrArg>>,
}

impl WalkWrites {
    /// `node.args = args`: rc4 assigns even unchanged args, which moves the
    /// node to the end of its inputs' users.
    fn set_args(&mut self, ctx: &mut LayoutCtx<'_>, id: NodeId, args: Vec<IrArg>) {
        self.before.entry(id).or_insert_with(|| ctx.node(id).args.clone());
        self.order.push(id);
        self.written.push(args.clone());
        ctx.set_args(id, args);
    }

    /// Whether some assigned node now computes something else than before the
    /// walk ([`same_op_args`]): a mean over the same axes listed in another
    /// order, or over size-1 axes besides, is the same op, and rc4's region
    /// move then keeps a sound graph, which the port keeps too.
    fn changed_meaning(&self, ctx: &LayoutCtx<'_>) -> bool {
        self.before
            .iter()
            .any(|(&id, before)| !same_op_args(ctx.graph(), id, before, &ctx.node(id).args))
    }

    /// Give every assigned node its args from before the walk again,
    /// assigning in the walk's order so each node's place in its inputs'
    /// users is the one rc4's assignments gave it.
    fn undo(&self, ctx: &mut LayoutCtx<'_>) {
        for &id in &self.order {
            ctx.set_args(id, self.before[&id].clone());
        }
    }

    /// Assign the walk's args again after an [`Self::undo`].
    fn redo(&self, ctx: &mut LayoutCtx<'_>) {
        for (&id, args) in self.order.iter().zip(&self.written) {
            ctx.set_args(id, args.clone());
        }
    }
}

/// A Python dim argument as `_dim_arg` (`:98-104`) reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DimArg {
    Int(i64),
    List(Vec<i64>),
}

impl DimArg {
    fn dims(&self) -> Vec<i64> {
        match self {
            Self::Int(dim) => vec![*dim],
            Self::List(dims) => dims.clone(),
        }
    }

    /// `mapped[0] if isinstance(args, int) else mapped`.
    fn rebuild(&self, mapped: &[usize]) -> IrArg {
        match self {
            Self::Int(_) => IrArg::Int(mapped[0] as i64),
            Self::List(_) => i64_list_arg(&dims_to_i64(mapped)),
        }
    }
}

/// `_dim_arg(arg)`: an int, a sequence of ints, or nothing. A bool is an int
/// in Python; `IrArg::None` is RETAB's import spelling of `[]` (see the
/// module docs).
fn dim_arg(arg: &IrArg) -> Option<DimArg> {
    match arg {
        IrArg::Int(dim) => Some(DimArg::Int(*dim)),
        IrArg::Bool(dim) => Some(DimArg::Int(i64::from(*dim))),
        IrArg::None => Some(DimArg::List(Vec::new())),
        IrArg::List(_) => Some(DimArg::List(
            int_list(arg).unwrap_or_else(|| panic!("TypeError: dims {arg:?} hold a non-int")),
        )),
        _ => None,
    }
}

/// One branch of a planned fork split (`_ForkBranchSplit`, `:35-39`).
struct ForkBranchSplit {
    next_node: NodeId,
    source_shape: Vec<i64>,
    /// The swapped `(node_args, next_node_args)` for a reduction or slice.
    arg_update: Option<(Vec<IrArg>, Vec<IrArg>)>,
}

/// A planned region move (`_PointwiseRegionMove`, `:42-46`).
struct PointwiseRegionMove {
    nodes_and_shapes: Vec<(NodeId, Vec<i64>)>,
    exit_producer: NodeId,
    exit_user: NodeId,
}

/// The next nodes of a frontier (`_get_next_nodes`), with a pending table's
/// table constant, which only exists in rc4.
struct NextNodes {
    nodes: Vec<NodeId>,
    table_constant: bool,
}

impl NextNodes {
    fn len(&self) -> usize {
        self.nodes.len() + usize::from(self.table_constant)
    }
}

struct Propagator<'c> {
    config: &'c PropagateConfig,
    /// Every live copy, and maybe some ids that stopped being one.
    transforms: Vec<NodeId>,
    remove_permutes_memo: RemovePermutesMemo,
    /// Users of the nodes the walks refreshed since the last re-trace, whose
    /// values rc4 leaves as they are until then ([`Self::retrace_stale`]).
    stale: Vec<NodeId>,
}

impl<'c> Propagator<'c> {
    fn new(config: &'c PropagateConfig) -> Self {
        Self {
            config,
            transforms: Vec::new(),
            remove_permutes_memo: RemovePermutesMemo::default(),
            stale: Vec::new(),
        }
    }

    fn up(&self) -> bool {
        self.config.direction == Direction::Up
    }

    fn tosa(&self) -> bool {
        self.config.overrides == Overrides::Tosa
    }

    // -----------------------------------------------------------------------
    // The driver
    // -----------------------------------------------------------------------

    /// Collect the live copies in graph order: rc4's sweep visits only nodes
    /// whose target is in `self._targets` (`:149`).
    fn collect_transforms(&mut self, ctx: &LayoutCtx<'_>) {
        self.transforms = ctx
            .order()
            .iter()
            .filter(|&id| self.is_target(ctx, id))
            .collect();
    }

    /// `n.target in self._targets` (`:93-96`), for a call.
    fn is_target(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let graph = ctx.graph();
        let node = graph.node(id);
        node.alive
            && node.op == NodeOp::CallFunction
            && matches!(node.target.as_str(), VIEW_TARGET | VIEW_DEFAULT_TARGET | PERMUTE_TARGET)
    }

    /// `_propagate_iteration` (`:141-153`) over a snapshot of the copies in
    /// `_propagation_order` (`:155-156`, `:907-910`). Walks the memo says
    /// would fail again are skipped: a failed walk changes nothing, so the
    /// sweep is rc4's.
    fn propagate_iteration(&mut self, ctx: &mut LayoutCtx<'_>) -> bool {
        let mut snapshot: Vec<NodeId> = self
            .transforms
            .iter()
            .copied()
            .filter(|&id| self.is_target(ctx, id))
            .collect();
        snapshot.sort_unstable();
        snapshot.dedup();
        ctx.order().sort(&mut snapshot);
        self.transforms.clone_from(&snapshot);
        if !self.up() {
            snapshot.reverse();
        }
        let rule = self.config.rule();
        let mut modified = false;
        for node in snapshot {
            if !self.is_target(ctx, node) || !ctx.graph().has_live_users(node) {
                continue;
            }
            if !ctx.should_retry(rule, node) {
                cover("memo.skip");
                #[cfg(test)]
                if CHECK_SKIPPED_WALKS.with(std::cell::Cell::get) {
                    assert!(
                        !self.propagate(ctx, node, &mut Reads::default()),
                        "the memo skipped a walk from {} that changes the graph",
                        ctx.node(node).name
                    );
                }
                continue;
            }
            #[cfg(test)]
            WALKS.with(|walks| walks.set(walks.get() + 1));
            let mut reads = Reads::default();
            if self.propagate(ctx, node, &mut reads) {
                modified = true;
            } else {
                ctx.record_attempt(rule, node, reads.0);
            }
        }
        modified
    }

    /// `_retrace` (`:159-162`): dead-code elimination, then the re-trace,
    /// which puts every node's users back in graph order; see the module docs.
    fn retrace(&mut self, ctx: &mut LayoutCtx<'_>) {
        #[cfg(test)]
        RETRACES.with(|retraces| retraces.set(retraces.get() + 1));
        ctx.eliminate_dead_code();
        self.retrace_point(ctx);
    }

    /// What an `ExportPass` re-trace does that later steps can see: every
    /// node's users return to graph order (`LayoutCtx::retraced`), and the
    /// values the walks left stale are re-derived ([`Self::retrace_stale`]).
    /// rc4 re-traces at `_retrace` and at the end of each fusion pass that
    /// changed the graph.
    fn retrace_point(&mut self, ctx: &mut LayoutCtx<'_>) {
        ctx.retraced();
        self.retrace_stale(ctx);
    }

    /// The part of an `ExportPass` re-trace that re-derives values: rc4's
    /// local refresh sets only the crossed nodes and the moved copy
    /// (`:164-192`), and a sound rewrite leaves every other value as it was,
    /// so at each re-trace point the users of the refreshed nodes are
    /// re-derived in graph order, and their users when a value changes; a
    /// value torch would reject panics, as rc4's re-trace raises. The one
    /// unsound rewrite known, rc4's region move keeping the walk's args
    /// (`:234-235`, `:1036-1039`), is taken back before it happens
    /// ([`WalkWrites::undo`]), so this is a guard that finds nothing in the
    /// replays. A node without a local rule keeps its value: only an already
    /// broken rewrite reaches one.
    fn retrace_stale(&mut self, ctx: &mut LayoutCtx<'_>) {
        if self.stale.is_empty() {
            return;
        }
        let mut heap: BinaryHeap<Reverse<(u64, NodeId)>> = BinaryHeap::new();
        let mut queued: HashSet<NodeId> = HashSet::new();
        let push = |ctx: &LayoutCtx<'_>, heap: &mut BinaryHeap<Reverse<(u64, NodeId)>>, queued: &mut HashSet<NodeId>, id: NodeId| {
            if let Some(key) = ctx.order().key(id) {
                if queued.insert(id) {
                    heap.push(Reverse((key, id)));
                }
            }
        };
        for id in std::mem::take(&mut self.stale) {
            push(ctx, &mut heap, &mut queued, id);
        }
        // Only meta changes below, so the order keys stay valid.
        while let Some(Reverse((_, id))) = heap.pop() {
            let node = ctx.node(id);
            if ctx.is_erased(id) || node.op != NodeOp::CallFunction {
                continue;
            }
            if !has_local_rule(&node.target) && !is_table(ctx.graph(), id) {
                continue;
            }
            let Some(old) = node.meta.tensor_facts().cloned() else {
                continue;
            };
            let meta = refreshed_node_meta(ctx.graph(), id);
            if meta.tensor_facts() == Some(&old) {
                continue;
            }
            cover("retrace.stale_value");
            *ctx.meta_mut(id) = meta;
            for user in ctx.graph().live_user_ids(id) {
                push(ctx, &mut heap, &mut queued, user);
            }
        }
    }

    /// `fuse_horizontal`: Up runs `FuseDuplicateUsersPass` with
    /// `duplicate_user_fusion_exclusions()` (`:569-576`; TOSA's are
    /// `TOSA_EXCLUDED_TARGETS`, backends/arm/_passes/propagate_view_copy_permute_pass.py:37-38),
    /// Down `FuseIdenticalInputTransformsPass(permute_targets=...)` (`:921-928`).
    /// Either re-traces when it changed the graph
    /// (backends/transforms/fuse_duplicate_users_pass.py:147-151,
    /// backends/transforms/fuse_identical_input_transforms_pass.py:200-204).
    fn fuse_horizontal(&mut self, ctx: &mut LayoutCtx<'_>) -> bool {
        trace(ctx, || format!("before {}", if self.up() { "fdu" } else { "fiit" }));
        let modified = match self.config.direction {
            Direction::Up => {
                let excluded: &[&str] = if self.tosa() { TOSA_EXCLUDED_TARGETS } else { &[] };
                fuse_duplicate_users(ctx, excluded, TableStage::OwnTableConstants)
            }
            Direction::Down => fuse_identical_input_transforms(ctx, &FuseIdenticalInputTransforms::DRIVER),
        };
        if modified {
            self.retrace_point(ctx);
        }
        modified
    }

    /// `fuse_vertical` (`:301-316`): the fusion pass, then
    /// `CanonicalizeViewCopyPermutePass(self._permute_targets)`. Each
    /// re-traces when it changed the graph
    /// (backends/transforms/remove_permutes_around_elementwise_ops.py:453,
    /// backends/transforms/canonicalize_view_copy_permute_pass.py:93-100);
    /// `remove_permutes_in_ctx` resets the user order itself.
    fn fuse_vertical(&mut self, ctx: &mut LayoutCtx<'_>) -> bool {
        let mut modified = false;
        if self.tosa() && self.config.exported_program {
            trace(ctx, || "before remove_permutes".to_string());
            let removed = remove_permutes_in_ctx(
                ctx,
                &RemovePermutesConfig::tosa(),
                &mut self.remove_permutes_memo,
                DeadCode::Full,
            );
            if removed {
                self.retrace_point(ctx);
            }
            modified |= removed;
        }
        trace(ctx, || "before canonicalize".to_string());
        if canonicalize_view_copy_permute(ctx) {
            self.retrace_point(ctx);
            modified = true;
        }
        modified
    }

    // -----------------------------------------------------------------------
    // One walk
    // -----------------------------------------------------------------------

    /// `_propagate(node)` (`:238-282`).
    fn propagate(&mut self, ctx: &mut LayoutCtx<'_>, node: NodeId, reads: &mut Reads) -> bool {
        let mut frontier = node;
        let mut previous_frontier: Option<NodeId> = None;
        let mut path = vec![node];
        let mut moved = false;
        let mut writes = WalkWrites::default();
        reads.add(node);
        while let Some(next_nodes) = self.validated_next_nodes(ctx, node, frontier, reads) {
            if next_nodes.len() > 1 {
                if !self.maybe_split_fork(ctx, node, frontier, previous_frontier, &next_nodes.nodes, reads, &mut writes) {
                    if next_nodes.table_constant {
                        cover("up.stop.table_constant");
                    }
                    break;
                }
                self.refresh_propagation_meta(ctx, node, &path);
                return true;
            }
            let next_node = next_nodes.nodes[0];
            if self.advance_through_next_node(ctx, node, next_node, reads, &mut writes) {
                previous_frontier = Some(frontier);
                frontier = next_node;
                path.push(next_node);
                moved = true;
                continue;
            }
            // Concats are a special case since they branch the graph.
            if self.maybe_split_upwards_cat_fanout(ctx, node, next_node, reads) {
                cover("up.cat_fanout");
                self.refresh_propagation_meta(ctx, node, &path);
                return true;
            }
            break;
        }
        if !moved {
            return false;
        }
        let previous_frontier = previous_frontier.expect("a moved walk has a previous frontier");
        cover(if self.up() { "up.move" } else { "down.move" });
        self.move_node(ctx, node, frontier, previous_frontier);
        self.refresh_propagation_meta(ctx, node, &path);
        true
    }

    /// `_validated_next_nodes` (`:200-219`).
    fn validated_next_nodes(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        frontier: NodeId,
        reads: &mut Reads,
    ) -> Option<NextNodes> {
        let graph = ctx.graph();
        let next_nodes = self.next_nodes(ctx, frontier, reads);
        if next_nodes.len() == 0 {
            cover("endpoint");
            let op = graph.node(frontier).op;
            assert!(
                matches!(op, NodeOp::Placeholder | NodeOp::Output),
                "AssertionError: {} reached an endpoint node which is not a placeholder or output: {}",
                self.config.class_name(),
                graph.node(frontier).name
            );
            return None;
        }
        if !self.can_cross_next_nodes(ctx, frontier, &next_nodes, reads) {
            return None;
        }
        if self.blocks_moving(ctx, node, frontier, &next_nodes.nodes, reads) {
            cover("blocks_moving");
            return None;
        }
        Some(next_nodes)
    }

    /// `_get_next_nodes`: Up, the frontier's inputs (`:578-579`), with a
    /// pending table's constant; Down, its users in torch.fx's order
    /// (`:930-931`).
    fn next_nodes(&self, ctx: &LayoutCtx<'_>, frontier: NodeId, reads: &mut Reads) -> NextNodes {
        let graph = ctx.graph();
        reads.add(frontier);
        match self.config.direction {
            Direction::Up => NextNodes {
                nodes: all_input_nodes(graph.node(frontier)),
                table_constant: self.has_table_constant(ctx, frontier),
            },
            Direction::Down => NextNodes {
                nodes: fx_users(ctx, frontier, reads),
                table_constant: false,
            },
        }
    }

    /// `_can_cross_next_nodes`: Down accepts (`:333-336`); Up needs no
    /// blocking user of the frontier and every next node used by the frontier
    /// alone (`:584-592`). A table constant's only user is its table.
    fn can_cross_next_nodes(&self, ctx: &LayoutCtx<'_>, frontier: NodeId, next_nodes: &NextNodes, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        if !self.up() {
            return true;
        }
        let users = graph.live_user_ids(frontier);
        reads.extend(users.iter().copied());
        if users.iter().any(|&user| self.blocks_crossing(ctx, user)) {
            cover("up.blocks_crossing");
            return false;
        }
        next_nodes.nodes.iter().all(|&next_node| {
            reads.add(next_node);
            graph.users(next_node).iter().all(|&user| user == frontier)
        })
    }

    /// `_advance_through_next_node` (`:221-236`). rc4 assigns both arg tuples
    /// (`:234-235`) even when they are unchanged, which moves both nodes to
    /// the end of their inputs' users; `writes` logs the assignments so a
    /// region move can take them back ([`Self::maybe_split_fork`]).
    fn advance_through_next_node(
        &mut self,
        ctx: &mut LayoutCtx<'_>,
        node: NodeId,
        next_node: NodeId,
        reads: &mut Reads,
        writes: &mut WalkWrites,
    ) -> bool {
        reads.add(next_node);
        if !self.is_swappable(ctx, next_node, reads) {
            return false;
        }
        let Some((node_args, next_args)) = self.maybe_swap_args(ctx, node, next_node, reads) else {
            return false;
        };
        writes.set_args(ctx, node, node_args);
        writes.set_args(ctx, next_node, next_args);
        true
    }

    /// `_maybe_swap_args` (`:374-388`).
    fn maybe_swap_args(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        next_node: NodeId,
        reads: &mut Reads,
    ) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        match graph.node(node).target.as_str() {
            PERMUTE_TARGET => match self.config.direction {
                Direction::Up => self.up_swap_permute_args(ctx, node, next_node, reads),
                Direction::Down => self.down_swap_permute_args(ctx, node, next_node, reads),
            },
            VIEW_TARGET | VIEW_DEFAULT_TARGET => match self.config.direction {
                Direction::Up => self.up_swap_view_args(ctx, node, next_node, reads),
                Direction::Down => self.down_swap_view_args(ctx, node, next_node, reads),
            },
            target => panic!("ValueError: Unexpected node target {target} in {}", self.config.class_name()),
        }
    }

    // -----------------------------------------------------------------------
    // Extension points and their TOSA answers
    // -----------------------------------------------------------------------

    /// `is_transparent` (`:419-426`); TOSA adds RESCALE
    /// (backends/arm/_passes/propagate_view_copy_permute_pass.py:95-99). A
    /// 16-bit pending table's RESCALE is its own node by the time a walk runs
    /// (see the module docs).
    fn is_transparent(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let graph = ctx.graph();
        let target = graph.node(id).target.as_str();
        if is_transparent_copy(target) {
            return true;
        }
        self.tosa() && is_rescale(target)
    }

    /// `is_elementwise` (`:460-470`); TOSA's (`:125-130`): a per-tensor
    /// RESCALE and any TABLE. `_is_per_tensor_rescale` (`:132-144`) reads the
    /// RESCALE's input, which joins `reads`.
    fn is_elementwise(&self, ctx: &LayoutCtx<'_>, id: NodeId, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        let node = graph.node(id);
        if self.tosa() {
            if is_rescale(&node.target) {
                reads.add(id);
                reads.extend(all_input_nodes(node));
                return is_per_tensor_rescale(graph, id);
            }
            if is_table(graph, id) {
                return true;
            }
        }
        if node.op != NodeOp::CallFunction {
            return false;
        }
        if self.is_transparent(ctx, id) {
            return true;
        }
        // rc4 holds a pending table as a TABLE, which carries no torch tags.
        !is_pending_table(node) && is_pointwise(&node.target)
    }

    /// `is_multi_input_elementwise` (`:428-430`); TOSA's TABLE (`:119-120`).
    fn is_multi_input_elementwise(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let graph = ctx.graph();
        self.tosa() && is_table(graph, id)
    }

    /// `blocks_crossing` (`:432-434`); TOSA's SCATTER (`:122-123`).
    fn blocks_crossing(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let graph = ctx.graph();
        self.tosa() && rc4_target(&graph.node(id).target) == SCATTER
    }

    /// `blocks_moving` (`:436-447`); TOSA's INT48 rule (`:47-57`). The
    /// configured hook, a test subclass's override, replaces both.
    fn blocks_moving(&self, ctx: &LayoutCtx<'_>, moving: NodeId, frontier: NodeId, next_nodes: &[NodeId], reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        if let Some(hook) = self.config.blocks_moving_hook {
            reads.add(frontier);
            reads.extend(next_nodes.iter().copied());
            return hook(graph, moving, frontier, next_nodes);
        }
        if !self.tosa() || graph.node(moving).target != PERMUTE_TARGET {
            return false;
        }
        std::iter::once(frontier)
            .chain(next_nodes.iter().copied())
            .any(|candidate| node_or_inputs_are_int48(graph, candidate, reads))
    }

    /// `tolerates_shape_after_move` (`:449-457`); TOSA's TABLE guard
    /// (`:59-79`), also for pending tables.
    fn tolerates_shape_after_move(&self, ctx: &LayoutCtx<'_>, next_node: NodeId, shape: Option<&[i64]>) -> bool {
        let graph = ctx.graph();
        if !self.tosa() || !is_table(graph, next_node) {
            return true;
        }
        let unsafe_shape = shape.is_some_and(|shape| shape.len() == 4 && shape[0] > 1 && shape[1] == 1 && shape[2] == 1);
        if unsafe_shape {
            cover("table_guard");
        }
        !unsafe_shape
    }

    /// `is_swappable` (`:472-476`); TOSA's raises for a reduction that isn't
    /// keepdim (`:101-115`), as Arm normalizes reductions first.
    fn is_swappable(&self, ctx: &LayoutCtx<'_>, next_node: NodeId, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        reads.add(next_node);
        let node = graph.node(next_node);
        if self.tosa() && self.is_reduction(&node.target) {
            let keep_dim = keep_dim_arg(graph, next_node);
            if !is_true(keep_dim) {
                panic!(
                    "RuntimeError: {} expects keep_dim=True for reduction ops to simplify propagation logic, got {} for node {}.",
                    self.config.class_name(),
                    keep_dim.map_or("None".to_string(), python_repr),
                    node.name
                );
            }
        }
        self.is_elementwise(ctx, next_node, reads) || self.is_arg_update_target(&node.target)
    }

    /// `target in self._REDUCTION_TARGETS` (`:68-71`; TOSA's, `:40-42`).
    fn is_reduction(&self, target: &str) -> bool {
        target == MEAN_DIM || (!self.tosa() && target == SUM_DIM_INT_LIST)
    }

    /// `target in self._ARG_UPDATE_TARGETS` (`:72-75`; TOSA's, `:43-46`).
    fn is_arg_update_target(&self, target: &str) -> bool {
        self.is_reduction(target) || target == SLICE_COPY
    }

    /// Whether `id` has a table constant only rc4 holds: a pending table's
    /// second input (see the module docs).
    fn has_table_constant(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let graph = ctx.graph();
        is_pending_table(graph.node(id))
    }

    /// `_shape_seen_by_next_node`: Up, the copy's output (`:536-539`); Down,
    /// its input (`:902-905`).
    fn shape_seen_by_next_node<'g>(&self, graph: &'g IrGraph, moving: NodeId, reads: &mut Reads) -> Option<&'g [i64]> {
        match self.config.direction {
            Direction::Up => graph.node_shape(moving),
            Direction::Down => {
                let input = graph.node(moving).args.first().and_then(IrArg::as_node)?;
                reads.add(input);
                graph.node_shape(input)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Elementwise crossings
    // -----------------------------------------------------------------------

    /// `_can_move_through_elementwise` (`:478-523`), and Up's override
    /// (`:594-616`).
    fn can_move_through_elementwise(&self, ctx: &LayoutCtx<'_>, moving: NodeId, next_node: NodeId, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        if self.base_can_move_through_elementwise(ctx, moving, next_node, reads) {
            return true;
        }
        if !self.up() {
            return false;
        }
        if graph.node(moving).target != PERMUTE_TARGET || !self.is_elementwise(ctx, next_node, reads) {
            return false;
        }
        let Some(rank) = graph.node_shape(next_node).map(<[i64]>::len) else {
            return false;
        };
        let one = self.layout_dependent_input_count(ctx, next_node, rank, reads) == 1;
        if one {
            cover("up.one_layout_dependent_input");
        }
        one
    }

    fn base_can_move_through_elementwise(&self, ctx: &LayoutCtx<'_>, moving: NodeId, next_node: NodeId, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        reads.add(next_node);
        if !self.is_elementwise(ctx, next_node, reads)
            || (all_input_nodes(graph.node(next_node)).len() != 1
                && !self.is_multi_input_elementwise(ctx, next_node))
            || graph.node_shape(next_node).is_none()
        {
            return false;
        }
        if graph.node(moving).target == VIEW_TARGET {
            let view_map = view_map(graph, moving, reads);
            if !view_map.is_valid_map() {
                return false;
            }
            if self.is_transparent(ctx, next_node) && view_map.source_rank() != view_map.target_rank() {
                cover("view.transparent_rank_change");
                return false;
            }
        }
        let shape = self.shape_seen_by_next_node(graph, moving, reads);
        if !self.tolerates_shape_after_move(ctx, next_node, shape) {
            return false;
        }
        if graph.node(moving).target != PERMUTE_TARGET {
            return true;
        }
        let source_rank = match graph.node(moving).args.get(1).and_then(dim_arg) {
            Some(DimArg::List(dims)) => dims.len(),
            _ => return false,
        };
        graph.node_shape(next_node).is_some_and(|shape| shape.len() == source_rank)
    }

    /// How many of `id`'s inputs aren't layout-invariant at `rank`, counting
    /// a pending table's constant, which never is.
    fn layout_dependent_input_count(&self, ctx: &LayoutCtx<'_>, id: NodeId, rank: usize, reads: &mut Reads) -> usize {
        let graph = ctx.graph();
        let inputs = all_input_nodes(graph.node(id));
        reads.extend(inputs.iter().copied());
        inputs
            .iter()
            .filter(|&&input| !is_layout_invariant(graph, input, rank))
            .count()
            + usize::from(self.has_table_constant(ctx, id))
    }

    /// `_can_split_through_elementwise` (`:1336-1350`).
    fn can_split_through_elementwise(&self, ctx: &LayoutCtx<'_>, frontier: NodeId, next_node: NodeId, reads: &mut Reads) -> bool {
        let graph = ctx.graph();
        if !self.is_elementwise(ctx, next_node, reads) {
            return false;
        }
        let Some(rank) = graph.node_shape(frontier).map(<[i64]>::len) else {
            return false;
        };
        let inputs = all_input_nodes(graph.node(next_node));
        reads.extend(inputs.iter().copied());
        let split = !self.has_table_constant(ctx, next_node)
            && inputs
                .iter()
                .all(|&input| input == frontier || is_layout_invariant(graph, input, rank));
        if split {
            cover("down.split_through_elementwise");
        }
        split
    }

    /// `_propagation_input` (`:541-558`): the one input a copy moving up
    /// passes to, the layout-invariant inputs aside.
    fn propagation_input(&self, ctx: &LayoutCtx<'_>, node: NodeId, reads: &mut Reads) -> Option<NodeId> {
        let graph = ctx.graph();
        let mut inputs = all_input_nodes(graph.node(node));
        reads.extend(inputs.iter().copied());
        let multi = self.is_multi_input_elementwise(ctx, node);
        if inputs.len() > 1 && !multi {
            let rank = graph.node_shape(node)?.len();
            inputs.retain(|&input| !is_layout_invariant(graph, input, rank));
        }
        if !inputs.is_empty() && (inputs.len() == 1 || multi) {
            return Some(inputs[0]);
        }
        None
    }

    /// `_remap_permute_for_broadcast_rank` (`:390-410`): the permute's args
    /// for leading axes broadcasting adds or removes.
    fn remap_permute_for_broadcast_rank(&self, ctx: &LayoutCtx<'_>, node: NodeId, new_rank: usize) -> Option<Vec<IrArg>> {
        let graph = ctx.graph();
        let permute_map = permute_map(graph, node);
        let old_rank = permute_map.permute_dims().len();
        let args = &graph.node(node).args;
        if old_rank == new_rank {
            return Some(args.clone());
        }
        let retained_axes: Vec<i64> = (old_rank.abs_diff(new_rank)..old_rank.max(new_rank))
            .map(|axis| axis as i64)
            .collect();
        let dims = if old_rank < new_rank {
            cover("broadcast.lift");
            permute_map.lift_to_rank(&retained_axes, new_rank)
        } else {
            cover("broadcast.project");
            permute_map.project_to_axes(&retained_axes, true)
        }?;
        Some(with_arg(args, 1, i64_list_arg(&dims_to_i64(&dims))))
    }

    // -----------------------------------------------------------------------
    // Swaps moving up
    // -----------------------------------------------------------------------

    /// Up's `_maybe_swap_permute_args` (`:618-672`).
    fn up_swap_permute_args(&self, ctx: &LayoutCtx<'_>, node: NodeId, next_node: NodeId, reads: &mut Reads) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if self.is_elementwise(ctx, next_node, reads) {
            if !self.can_move_through_elementwise(ctx, node, next_node, reads) {
                return None;
            }
            graph.node_shape(next_node)?;
            let producer = self.propagation_input(ctx, next_node, reads)?;
            let producer_rank = graph.node_shape(producer)?.len();
            let node_args = self.remap_permute_for_broadcast_rank(ctx, node, producer_rank)?;
            cover("up.permute.elementwise");
            return Some((node_args, next.args.clone()));
        }
        let permute_map = permute_map(graph, node);
        let args = dim_arg(next_arg(graph, next_node, 1))?;
        let keep_dim = keep_dim_arg(graph, next_node);
        if self.is_reduction(&next.target) && !is_true(keep_dim) {
            let input = *all_input_nodes(next).first().unwrap_or_else(|| panic!("IndexError: {} reads no node", next.name));
            reads.add(input);
            let input_rank = graph.node_shape(input)?.len();
            if input_rank == 0 {
                return None;
            }
            let mut reduction_dims = args.dims();
            if reduction_dims.is_empty() {
                reduction_dims = (0..input_rank as i64).collect();
            }
            let reduction_dims: Vec<i64> = reduction_dims.iter().map(|dim| dim.rem_euclid(input_rank as i64)).collect();
            let retained_axes: Vec<i64> = (0..input_rank as i64).filter(|dim| !reduction_dims.contains(dim)).collect();
            let dims = permute_map.lift_to_rank(&retained_axes, input_rank)?;
            let mapped: Vec<usize> = reduction_dims
                .iter()
                .map(|&dim| {
                    dims.iter()
                        .position(|&axis| axis as i64 == dim)
                        .unwrap_or_else(|| panic!("ValueError: {dim} is not in list"))
                })
                .collect();
            let node_args = with_arg(&graph.node(node).args, 1, i64_list_arg(&dims_to_i64(&dims)));
            let next_args = with_arg(&next.args, 1, args.rebuild(&mapped));
            cover("up.permute.reduction_dropped");
            return Some((node_args, next_args));
        }
        let mapped = permute_map
            .map_dims(&args.dims())
            .unwrap_or_else(|error| panic!("AssertionError: {error:?}"));
        cover(if next.target == SLICE_COPY { "up.permute.slice" } else { "up.permute.reduction" });
        Some((graph.node(node).args.clone(), with_arg(&next.args, 1, args.rebuild(&mapped))))
    }

    /// Up's `_maybe_swap_view_args` (`:674-697`).
    fn up_swap_view_args(&self, ctx: &LayoutCtx<'_>, node: NodeId, next_node: NodeId, reads: &mut Reads) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if self.is_elementwise(ctx, next_node, reads) {
            if !self.can_move_through_elementwise(ctx, node, next_node, reads) {
                return None;
            }
            cover("up.view.elementwise");
            return Some((graph.node(node).args.clone(), next.args.clone()));
        }
        let view_map = view_map(graph, node, reads);
        let inputs = all_input_nodes(next);
        if !view_map.is_valid_map() || inputs.len() != 1 {
            return None;
        }
        reads.add(inputs[0]);
        let input_shape = graph
            .node_shape_vec(inputs[0])
            .unwrap_or_else(|| panic!("KeyError: 'val' on {}", graph.node(inputs[0]).name));
        let new_shape = view_map.remap_target_shape(&input_shape);
        if self.is_reduction(&next.target) {
            return self.up_swap_reduction_view_args(ctx, node, next_node, &view_map, &input_shape);
        }
        if next.target == SLICE_COPY {
            return self.up_swap_slice_view_args(ctx, node, next_node, &view_map, &input_shape, new_shape);
        }
        None
    }

    /// `_maybe_swap_reduction_view_args` (`:699-716`).
    fn up_swap_reduction_view_args(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        next_node: NodeId,
        view_map: &ViewMap,
        input_shape: &[i64],
    ) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if next.args.len() <= 2 || !is_true(next.args.get(2)) {
            return None;
        }
        let reduction_dims = dim_arg(&next.args[1]).unwrap_or_else(|| panic!("TypeError: reduction dims {:?}", next.args[1]));
        let (new_shape, new_dims) = view_map.map_reduction_after_view(input_shape, &reduction_dims.dims())?;
        cover("up.view.reduction");
        let next_args = with_arg(&next.args, 1, i64_list_arg(&dims_to_i64(&new_dims)));
        Some((view_args(graph, node, &new_shape), next_args))
    }

    /// `_maybe_swap_slice_view_args` (`:718-772`).
    fn up_swap_slice_view_args(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        next_node: NodeId,
        view_map: &ViewMap,
        input_shape: &[i64],
        new_shape: Option<Vec<i64>>,
    ) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if next.args.len() < 4 {
            return None;
        }
        let slice_dim = python_int(&next.args[1]);
        let start = python_int(&next.args[2]);
        let end = python_int(&next.args[3]);
        let step = match next.args.get(4) {
            None => Some(1),
            Some(arg) => python_int(arg),
        };
        // `remap_unit_slice` rejects a non-int bound or a step other than 1.
        let unit_slice_swap = match (slice_dim, step) {
            (Some(slice_dim), Some(step)) => view_map
                .remap_unit_slice(input_shape, slice_dim, start, end, step)
                .unwrap_or_else(|error| panic!("AssertionError: {error:?}")),
            _ => None,
        };
        if let Some((new_shape, new_dim, new_start, new_end)) = unit_slice_swap {
            if !valid_slice_interval(&new_shape, new_dim as i64, Some(new_start), Some(new_end)) {
                return None;
            }
            cover("up.view.unit_slice");
            let mut next_args = vec![next.args[0].clone(), IrArg::Int(new_dim as i64), IrArg::Int(new_start), IrArg::Int(new_end)];
            next_args.extend(next.args.iter().skip(4).cloned());
            return Some((view_args(graph, node, &new_shape), next_args));
        }
        let new_shape = new_shape?;
        let slice_dim = slice_dim.unwrap_or_else(|| panic!("TypeError: slice dim {:?}", next.args[1]));
        let mapped_dim = map_slice_dim(view_map, slice_dim)?;
        if !valid_slice_interval(&new_shape, mapped_dim as i64, start, end) {
            return None;
        }
        let next_args = with_arg(&next.args, 1, IrArg::Int(mapped_dim as i64));
        cover("up.view.slice");
        Some((view_args(graph, node, &new_shape), next_args))
    }

    // -----------------------------------------------------------------------
    // Swaps moving down
    // -----------------------------------------------------------------------

    /// Down's `_maybe_swap_permute_args` (`:933-979`).
    fn down_swap_permute_args(&self, ctx: &LayoutCtx<'_>, node: NodeId, next_node: NodeId, reads: &mut Reads) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if self.is_elementwise(ctx, next_node, reads) {
            if !self.can_move_through_elementwise(ctx, node, next_node, reads) {
                return None;
            }
            let next_rank = graph.node_shape(next_node)?.len();
            let node_args = self.remap_permute_for_broadcast_rank(ctx, node, next_rank)?;
            cover("down.permute.elementwise");
            return Some((node_args, next.args.clone()));
        }
        let permute_map = permute_map(graph, node);
        let rank = permute_map.permute_dims().len();
        let args = dim_arg(next_arg(graph, next_node, 1))?;
        let keep_dim = keep_dim_arg(graph, next_node);
        let reduces = self.is_reduction(&next.target) && !is_true(keep_dim);
        let mut reduction_dims = args.dims();
        if reduces && reduction_dims.is_empty() {
            reduction_dims = (0..rank as i64).collect();
        }
        let mapped = permute_map
            .map_dims_inverse(&reduction_dims)
            .unwrap_or_else(|error| panic!("AssertionError: {error:?}"));
        let next_args = with_arg(&next.args, 1, args.rebuild(&mapped));
        cover(if next.target == SLICE_COPY {
            "down.permute.slice"
        } else if reduces {
            "down.permute.reduction_dropped"
        } else {
            "down.permute.reduction"
        });
        let mut node_args = graph.node(node).args.clone();
        if reduces {
            let retained_axes: Vec<i64> = (0..rank).filter(|dim| !mapped.contains(dim)).map(|dim| dim as i64).collect();
            let dims = permute_map.project_to_axes(&retained_axes, false)?;
            node_args = with_arg(&node_args, 1, i64_list_arg(&dims_to_i64(&dims)));
        }
        Some((node_args, next_args))
    }

    /// Down's `_maybe_swap_view_args` (`:981-1011`).
    fn down_swap_view_args(&self, ctx: &LayoutCtx<'_>, node: NodeId, next_node: NodeId, reads: &mut Reads) -> Option<(Vec<IrArg>, Vec<IrArg>)> {
        let graph = ctx.graph();
        let next = graph.node(next_node);
        if self.is_elementwise(ctx, next_node, reads) {
            if !self.can_move_through_elementwise(ctx, node, next_node, reads) {
                return None;
            }
            cover("down.view.elementwise");
            return Some((graph.node(node).args.clone(), next.args.clone()));
        }
        let view_map = view_map(graph, node, reads);
        if !view_map.is_valid_map() {
            return None;
        }
        let (new_dims, output_shape) = if self.is_reduction(&next.target) {
            if next.args.len() <= 2 || !is_true(next.args.get(2)) {
                return None;
            }
            let dims = dim_arg(&next.args[1]).unwrap_or_else(|| panic!("TypeError: reduction dims {:?}", next.args[1]));
            let (new_dims, output_shape) = view_map
                .map_reduction_before_view(&dims.dims())
                .unwrap_or_else(|error| panic!("AssertionError: {error:?}"))?;
            cover("down.view.reduction");
            (i64_list_arg(&dims_to_i64(&new_dims)), output_shape)
        } else if next.target == SLICE_COPY {
            let dim = python_int(next_arg(graph, next_node, 1)).unwrap_or_else(|| panic!("TypeError: slice dim"));
            let new_dims = view_map
                .map_dim_inverse(&[dim])
                .unwrap_or_else(|error| panic!("AssertionError: {error:?}"))?;
            if new_dims.len() != 1 {
                return None;
            }
            let output_shape = graph
                .node_shape_vec(next_node)
                .unwrap_or_else(|| panic!("KeyError: 'val' on {}", next.name));
            cover("down.view.slice");
            (IrArg::Int(new_dims[0] as i64), output_shape)
        } else {
            return None;
        };
        Some((view_args(graph, node, &output_shape), with_arg(&next.args, 1, new_dims)))
    }

    // -----------------------------------------------------------------------
    // Forks moving down
    // -----------------------------------------------------------------------

    /// `_maybe_split_fork`: Up has none (`:359-369`); Down splits the permute
    /// over the branches, or moves it past a pointwise region (`:1015-1042`).
    ///
    /// The region move is planned from the original permute, so the walk's
    /// crossings are not part of it, but rc4 keeps the args its advance
    /// already assigned (`:234-235`): a slice or mean the walk crossed then
    /// reads the moved permute's output with dims meant for its input, a
    /// wrong graph wherever those dims change what it computes (the module
    /// docs). The port then takes the assignments back first
    /// ([`WalkWrites::undo`]), so the crossed nodes read what they read
    /// before, and plans again on that graph.
    #[allow(clippy::too_many_arguments)]
    fn maybe_split_fork(
        &mut self,
        ctx: &mut LayoutCtx<'_>,
        node: NodeId,
        frontier: NodeId,
        previous_frontier: Option<NodeId>,
        next_nodes: &[NodeId],
        reads: &mut Reads,
        writes: &mut WalkWrites,
    ) -> bool {
        if self.up() {
            return false;
        }
        if frontier != node && previous_frontier.is_none() {
            return false;
        }
        if let Some((producer, branch_splits)) = self.plan_fork_split(ctx, node, frontier, next_nodes, reads) {
            cover(if frontier == node { "down.fork_split" } else { "down.fork_split.after_crossing" });
            if branch_splits.iter().any(|split| split.arg_update.is_some()) {
                cover("down.fork_split.arg_update");
            }
            self.apply_fork_split(ctx, node, frontier, producer, branch_splits);
            return true;
        }
        let Some(mut plan) = self.plan_pointwise_region_move(ctx, node, reads) else {
            return false;
        };
        if writes.changed_meaning(ctx) {
            cover("down.region_move.undo");
            writes.undo(ctx);
            match self.plan_pointwise_region_move(ctx, node, &mut Reads::default()) {
                Some(replanned) => plan = replanned,
                None => {
                    // Only a crossing that rewrote the permute's own dims
                    // (the shared pass's non-keepdim reductions) can change
                    // the plan; go on as rc4 does without the region move.
                    writes.redo(ctx);
                    return false;
                }
            }
        }
        cover("down.region_move");
        self.apply_pointwise_region_move(ctx, node, plan);
        true
    }

    /// `_plan_fork_split` (`:1265-1318`).
    fn plan_fork_split(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        frontier: NodeId,
        next_nodes: &[NodeId],
        reads: &mut Reads,
    ) -> Option<(NodeId, Vec<ForkBranchSplit>)> {
        let graph = ctx.graph();
        let normalized_dims = self.normalized_permute_dims(ctx, node, reads)?;
        let producer = all_input_nodes(graph.node(node))[0];
        let rank = normalized_dims.len();
        let mut branch_splits = Vec::new();
        for &next_node in next_nodes {
            reads.add(next_node);
            let arg_update = if self.can_move_through_elementwise(ctx, node, next_node, reads)
                || self.can_split_through_elementwise(ctx, frontier, next_node, reads)
            {
                None
            } else if self.is_swappable(ctx, next_node, reads) {
                Some(self.maybe_swap_args(ctx, node, next_node, reads)?)
            } else {
                return None;
            };
            let next_shape = graph.node_shape(next_node)?;
            if next_shape.len() != rank {
                return None;
            }
            let source_shape = source_shape(next_shape, &normalized_dims);
            branch_splits.push(ForkBranchSplit {
                next_node,
                source_shape,
                arg_update,
            });
        }
        Some((producer, branch_splits))
    }

    /// The permute's dims normalized once, when it is a valid permute with
    /// one tensor input: the shared start of `_plan_fork_split` (`:1283-1294`)
    /// and `_pointwise_region_permute_dims` (`:1119-1138`).
    fn normalized_permute_dims(&self, ctx: &LayoutCtx<'_>, node: NodeId, reads: &mut Reads) -> Option<Vec<usize>> {
        let graph = ctx.graph();
        let n = graph.node(node);
        let inputs = all_input_nodes(n);
        if n.target != PERMUTE_TARGET || inputs.len() != 1 {
            return None;
        }
        reads.add(inputs[0]);
        graph.node_shape(inputs[0])?;
        let Some(DimArg::List(dims)) = n.args.get(1).and_then(dim_arg) else {
            return None;
        };
        let rank = dims.len() as i64;
        let normalized: Vec<i64> = dims.iter().map(|&dim| if dim >= 0 { dim } else { dim + rank }).collect();
        let mut sorted = normalized.clone();
        sorted.sort_unstable();
        if !sorted.iter().copied().eq(0..rank) {
            return None;
        }
        Some(normalized.into_iter().map(|dim| dim as usize).collect())
    }

    /// `_apply_fork_split` (`:1338-1391`).
    fn apply_fork_split(
        &mut self,
        ctx: &mut LayoutCtx<'_>,
        node: NodeId,
        frontier: NodeId,
        producer: NodeId,
        branch_splits: Vec<ForkBranchSplit>,
    ) {
        if frontier != node {
            let original_user = ctx.users_in_fx_order(node)[0];
            ctx.replace_input_with(original_user, node, producer);
        }
        let node_target = ctx.node(node).target.clone();
        let node_args = ctx.node(node).args.clone();
        let node_kwargs = ctx.node(node).kwargs.clone();
        for split in branch_splits {
            let next_node = split.next_node;
            let old_next_meta = ctx.node(next_node).meta.clone();
            if frontier == node {
                ctx.replace_input_with(next_node, node, producer);
            }
            if let Some((_, next_args)) = &split.arg_update {
                let branch_input = if frontier == node { IrArg::Node(producer) } else { next_args[0].clone() };
                ctx.set_args(next_node, with_arg(next_args, 0, branch_input));
            }
            set_shape(ctx, next_node, split.source_shape);
            let mut transform_args = vec![IrArg::Node(next_node)];
            match &split.arg_update {
                None => transform_args.extend(node_args.iter().skip(1).cloned()),
                Some((swapped_node_args, _)) => transform_args.extend(swapped_node_args.iter().skip(1).cloned()),
            }
            let branch_transform = ctx.inserting_after(next_node, |ctx| {
                ctx.create_node(
                    NodeOp::CallFunction,
                    node_target.replace('.', "_"),
                    node_target.clone(),
                    transform_args,
                    node_kwargs.clone(),
                    old_next_meta,
                )
            });
            self.transforms.push(branch_transform);
            for user in ctx.users_in_fx_order(next_node) {
                if user != branch_transform {
                    ctx.replace_input_with(user, next_node, branch_transform);
                }
            }
        }
        if !ctx.graph().has_live_users(node) {
            ctx.erase_node(node);
        }
    }

    /// `_plan_pointwise_region_move` (`:1044-1117`).
    fn plan_pointwise_region_move(&self, ctx: &LayoutCtx<'_>, node: NodeId, reads: &mut Reads) -> Option<PointwiseRegionMove> {
        let graph = ctx.graph();
        let normalized_dims = self.normalized_permute_dims(ctx, node, reads)?;
        let rank = normalized_dims.len();
        let (region, exit_edges) = self.discover_pointwise_region(ctx, node, &normalized_dims, reads)?;
        if region.is_empty() {
            return None;
        }
        if exit_edges.iter().any(|&(parent, _)| parent == node) {
            return None;
        }
        let exits: HashSet<(NodeId, NodeId)> = exit_edges
            .into_iter()
            .filter(|(parent, user)| region.contains(parent) && !region.contains(user))
            .collect();
        if exits.len() != 1 {
            cover("down.region.exits");
            return None;
        }
        if !self.pointwise_region_inputs_are_safe(ctx, &region, node, rank, reads) {
            cover("down.region.unsafe_input");
            return None;
        }
        let mut ordered: Vec<NodeId> = region.into_iter().collect();
        ctx.order().sort(&mut ordered);
        let nodes_and_shapes = ordered
            .into_iter()
            .map(|region_node| {
                let shape = graph.node_shape(region_node).expect("region nodes hold tensors");
                (region_node, source_shape(shape, &normalized_dims))
            })
            .collect();
        let (exit_producer, exit_user) = exits.into_iter().next().expect("one exit");
        Some(PointwiseRegionMove {
            nodes_and_shapes,
            exit_producer,
            exit_user,
        })
    }

    /// `_discover_pointwise_region` (`:1140-1169`): the elementwise nodes the
    /// permute can move past and the edges leaving them; `None` where a path
    /// rejects a node another path accepted. Users are visited in torch.fx's
    /// order, as rc4 iterates `node.users`.
    #[allow(clippy::type_complexity)]
    fn discover_pointwise_region(
        &self,
        ctx: &LayoutCtx<'_>,
        node: NodeId,
        normalized_dims: &[usize],
        reads: &mut Reads,
    ) -> Option<(HashSet<NodeId>, HashSet<(NodeId, NodeId)>)> {
        let mut region: HashSet<NodeId> = HashSet::new();
        let mut pending: Vec<(NodeId, NodeId)> = fx_users(ctx, node, reads)
            .into_iter()
            .map(|user| (node, user))
            .collect();
        let mut exit_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
        let mut exit_users: HashSet<NodeId> = HashSet::new();
        while let Some((parent, candidate)) = pending.pop() {
            reads.add(candidate);
            if region.contains(&candidate) {
                if self.blocks_moving(ctx, node, parent, &[candidate], reads) {
                    return None;
                }
                continue;
            }
            if !self.can_include_in_pointwise_region(ctx, node, parent, candidate, normalized_dims, reads) {
                exit_edges.insert((parent, candidate));
                exit_users.insert(candidate);
                continue;
            }
            if exit_users.contains(&candidate) {
                cover("down.region.path_conflict");
                return None;
            }
            region.insert(candidate);
            pending.extend(
                fx_users(ctx, candidate, reads)
                    .into_iter()
                    .map(|user| (candidate, user)),
            );
        }
        Some((region, exit_edges))
    }

    /// `_can_include_in_pointwise_region` (`:1171-1198`).
    fn can_include_in_pointwise_region(
        &self,
        ctx: &LayoutCtx<'_>,
        moving: NodeId,
        parent: NodeId,
        candidate: NodeId,
        normalized_dims: &[usize],
        reads: &mut Reads,
    ) -> bool {
        let graph = ctx.graph();
        if !self.is_elementwise(ctx, candidate, reads) {
            return false;
        }
        let Some(shape) = graph.node_shape(candidate) else {
            return false;
        };
        if shape.len() != normalized_dims.len() {
            return false;
        }
        if self.blocks_moving(ctx, moving, parent, &[candidate], reads) {
            return false;
        }
        let source = source_shape(shape, normalized_dims);
        self.tolerates_shape_after_move(ctx, candidate, Some(&source))
    }

    /// `_pointwise_region_inputs_are_safe` (`:1209-1222`), with a pending
    /// table's constant as an input from outside.
    fn pointwise_region_inputs_are_safe(
        &self,
        ctx: &LayoutCtx<'_>,
        region: &HashSet<NodeId>,
        moving: NodeId,
        rank: usize,
        reads: &mut Reads,
    ) -> bool {
        let graph = ctx.graph();
        region.iter().all(|&region_node| {
            let inputs = all_input_nodes(graph.node(region_node));
            reads.extend(inputs.iter().copied());
            !self.has_table_constant(ctx, region_node)
                && inputs.iter().all(|&input| {
                    input == moving || region.contains(&input) || is_layout_invariant(graph, input, rank)
                })
        })
    }

    /// `_apply_pointwise_region_move` (`:1224-1266`).
    fn apply_pointwise_region_move(&mut self, ctx: &mut LayoutCtx<'_>, node: NodeId, plan: PointwiseRegionMove) {
        let producer = all_input_nodes(ctx.node(node))[0];
        let old_exit_meta = ctx.node(plan.exit_producer).meta.clone();
        for (region_node, source_shape) in plan.nodes_and_shapes {
            ctx.replace_input_with(region_node, node, producer);
            set_shape(ctx, region_node, source_shape);
        }
        let n = ctx.node(node);
        let target = n.target.clone();
        let mut args = vec![IrArg::Node(plan.exit_producer)];
        args.extend(n.args.iter().skip(1).cloned());
        let kwargs = n.kwargs.clone();
        let mut meta = n.meta.clone();
        if let Some(facts) = old_exit_meta.tensor_facts() {
            meta.set_tensor_facts(facts.clone());
        }
        if let Some(dim_order) = old_exit_meta.val_dim_order() {
            meta.set_val_dim_order(dim_order.to_vec());
        }
        match old_exit_meta.extras.get(DELEGATION_TAG_META_KEY) {
            Some(tag) => {
                meta.extras.insert(DELEGATION_TAG_META_KEY.to_string(), tag.clone());
            }
            None => {
                meta.extras.remove(DELEGATION_TAG_META_KEY);
            }
        }
        let moved_permute = ctx.inserting_after(plan.exit_producer, |ctx| {
            ctx.create_node(NodeOp::CallFunction, target.replace('.', "_"), target, args, kwargs, meta)
        });
        self.transforms.push(moved_permute);
        ctx.replace_input_with(plan.exit_user, plan.exit_producer, moved_permute);
        if !ctx.graph().has_live_users(node) {
            ctx.erase_node(node);
        }
    }

    // -----------------------------------------------------------------------
    // Cat fan-outs moving up
    // -----------------------------------------------------------------------

    /// Up's `_maybe_split_upwards_cat_fanout` (`:824-886`): every user of the
    /// cat is the same permute, which only moves size-1 dims of each input, so
    /// each input gets its own permute and the cat's dim is remapped. A
    /// repeated input keeps every occurrence (see the module docs).
    fn maybe_split_upwards_cat_fanout(&mut self, ctx: &mut LayoutCtx<'_>, node: NodeId, next_node: NodeId, reads: &mut Reads) -> bool {
        if !self.up() {
            return false;
        }
        let graph = ctx.graph();
        if graph.node(node).target != PERMUTE_TARGET || graph.node(next_node).target != CAT {
            return false;
        }
        let cat_users = fx_users(ctx, next_node, reads);
        if cat_users.is_empty() {
            return false;
        }
        reads.extend(cat_users.iter().copied());
        if !cat_users.iter().all(|&user| graph.node(user).target == PERMUTE_TARGET) {
            return false;
        }
        let permute_args: Vec<Option<DimArg>> = cat_users
            .iter()
            .map(|&user| graph.node(user).args.get(1).and_then(dim_arg))
            .collect();
        let Some(DimArg::List(permutation)) = permute_args[0].clone() else {
            return false;
        };
        if !permute_args
            .iter()
            .all(|args| args.as_ref() == Some(&DimArg::List(permutation.clone())))
        {
            return false;
        }
        let cat = graph.node(next_node);
        let cat_dim = match cat.args.get(1).or_else(|| cat.kwargs.get("dim")) {
            None => 0,
            Some(IrArg::Int(dim)) => *dim,
            Some(IrArg::Bool(dim)) => i64::from(*dim),
            Some(_) => return false,
        };
        // rc4 sets the dim positionally and keeps a `dim` kwarg, so its next
        // re-trace calls `cat` with the dim twice and raises.
        assert!(
            !cat.kwargs.contains_key("dim"),
            "TypeError: cat() got multiple values for argument 'dim'"
        );
        let new_cat_dim = permute_map(graph, node)
            .map_dims(&[cat_dim])
            .unwrap_or_else(|error| panic!("AssertionError: {error:?}"))[0];
        let cat_inputs = all_input_nodes(cat);
        reads.extend(cat_inputs.iter().copied());
        let shapes: Vec<Vec<i64>> = cat_inputs
            .iter()
            .map(|&input| {
                graph
                    .node_shape_vec(input)
                    .unwrap_or_else(|| panic!("KeyError: 'val' on {}", graph.node(input).name))
            })
            .collect();
        if !shapes.iter().all(|shape| is_singleton_permutation(shape, &permutation)) {
            return false;
        }
        let tensors = cat.args.first().cloned().unwrap_or(IrArg::None);
        let mut rest_args: Vec<IrArg> = cat.args.iter().skip(2).cloned().collect();
        // `next_node.meta = dict(node.meta)` (`:883`): the cat takes the permute's
        // meta, whose val holds rc4's dtype for the permute. RETAB's may not
        // (design F2): a quantized compute node keeps a float32 val and its dtype
        // in qparams, and a copy follows its producer's val, so a qparam-less
        // permute of such a node holds float32 and takes its dtype from its input
        // (`value_dtype` rule 4). The cat takes the permute's qparams along, which
        // still give it the permute's dtype (rules 2-3), but it is no copy: rule 4
        // is lost, and the cat would serialize as float32. So in that case it
        // takes the permute's dtype into its val, which keeps `value_dtype` across
        // the move, as rc4's dict copy keeps the val's dtype.
        let mut node_meta = graph.node(node).meta.clone();
        let float_without_qparams = node_meta.val_dtype().is_some_and(is_float_dtype)
            && node_meta.output_qparam().is_none()
            && node_meta.input_qparam().is_none();
        if float_without_qparams {
            if let Some(dtype) = value_dtype(graph, node) {
                node_meta.set_val_dtype_for_existing_shape(&dtype);
            }
        }
        let cat_users_list = cat_users;

        let mut new_inputs: HashMap<NodeId, NodeId> = HashMap::new();
        for (input, shape) in cat_inputs.iter().zip(&shapes) {
            let mut meta = ctx.node(*input).meta.clone();
            let dtype = meta.val_dtype().unwrap_or("torch.float32").to_owned();
            let output_shape: Vec<i64> = permutation
                .iter()
                .map(|&dim| shape[dim.rem_euclid(shape.len().max(1) as i64) as usize])
                .collect();
            meta.set_tensor_facts(TensorFacts::new(output_shape, dtype));
            mark_contiguous(&mut meta);
            let permute = ctx.inserting_before(next_node, |ctx| {
                ctx.call_function(
                    PERMUTE_TARGET.replace('.', "_"),
                    PERMUTE_TARGET,
                    vec![IrArg::Node(*input), i64_list_arg(&permutation)],
                    meta,
                )
            });
            self.transforms.push(permute);
            new_inputs.insert(*input, permute);
        }
        let new_tensors = match tensors {
            IrArg::List(items) => IrArg::List(
                items
                    .iter()
                    .map(|item| match item.as_node() {
                        Some(input) => IrArg::Node(new_inputs[&input]),
                        None => item.clone(),
                    })
                    .collect(),
            ),
            other => other,
        };
        let mut cat_args = vec![new_tensors, IrArg::Int(new_cat_dim as i64)];
        cat_args.append(&mut rest_args);
        ctx.set_args(next_node, cat_args);
        *ctx.meta_mut(next_node) = node_meta;
        for &cat_user in &cat_users_list {
            ctx.replace_all_uses_with(cat_user, next_node);
        }
        for &cat_user in &cat_users_list {
            if !ctx.graph().has_live_users(cat_user) {
                ctx.erase_node(cat_user);
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Moves
    // -----------------------------------------------------------------------

    /// `_move_node`: Up puts the copy before the frontier (`:796-822`), Down
    /// after it (`:1405-1418`).
    fn move_node(&mut self, ctx: &mut LayoutCtx<'_>, node: NodeId, frontier: NodeId, previous_frontier: NodeId) {
        match self.config.direction {
            Direction::Up => {
                let original_input = all_input_nodes(ctx.node(node))[0];
                let (producer, frontier_user) = if ctx.node(frontier).op == NodeOp::Placeholder {
                    (frontier, previous_frontier)
                } else {
                    let producer = self
                        .propagation_input(ctx, frontier, &mut Reads::default())
                        .unwrap_or_else(|| {
                            panic!("AssertionError: Expected one propagation input for {}", ctx.node(frontier).target)
                        });
                    (producer, frontier)
                };
                ctx.replace_input_with(node, original_input, producer);
                ctx.replace_input_with(frontier_user, producer, node);
                for user in ctx.users_in_fx_order(node) {
                    if user != frontier_user {
                        ctx.replace_input_with(user, node, original_input);
                    }
                }
                ctx.move_before(node, frontier_user);
            }
            Direction::Down => {
                let original_user = ctx.users_in_fx_order(node)[0];
                let producer = all_input_nodes(ctx.node(node))[0];
                let output = ctx.node(frontier).op == NodeOp::Output;
                let frontier_input = if output { previous_frontier } else { frontier };
                let frontier_users = ctx.users_in_fx_order(frontier_input);
                ctx.replace_input_with(original_user, node, producer);
                ctx.replace_input_with(node, producer, frontier_input);
                for user in frontier_users {
                    if user != node {
                        ctx.replace_input_with(user, frontier_input, node);
                    }
                }
                if output {
                    ctx.move_before(node, frontier);
                } else {
                    ctx.move_after(node, frontier);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Values
    // -----------------------------------------------------------------------

    /// `_refresh_propagation_meta`: Up refreshes the copy, then the crossed
    /// nodes in reverse (`:560-567`); Down the crossed nodes in order, then the
    /// copy (`:912-919`).
    fn refresh_propagation_meta(&mut self, ctx: &mut LayoutCtx<'_>, moving: NodeId, path: &[NodeId]) {
        #[cfg(test)]
        LOCAL_REFRESHES.with(|count| count.set(count.get() + 1));
        match self.config.direction {
            Direction::Up => {
                refresh_node(ctx, moving);
                for &node in path[1..].iter().rev() {
                    refresh_node(ctx, node);
                }
            }
            Direction::Down => {
                for &node in &path[1..] {
                    refresh_node(ctx, node);
                }
                refresh_node(ctx, moving);
            }
        }
        for &node in path.iter().chain(std::iter::once(&moving)) {
            if !ctx.is_erased(node) {
                self.stale.extend(ctx.graph().live_user_ids(node));
            }
        }
    }
}

/// `_refresh_node_meta` (`:164-192`): a node with input nodes gets the value
/// its operands give, with `layout::shape`'s rules; a TABLE keeps its dtype
/// and takes its input's shape, as the TABLE fake kernel gives it. A value
/// rc4's fake execution would reject panics, as rc4 raises.
pub(crate) fn refresh_node(ctx: &mut LayoutCtx<'_>, id: NodeId) {
    if ctx.is_erased(id) || all_input_nodes(ctx.node(id)).is_empty() {
        return;
    }
    let meta = refreshed_node_meta(ctx.graph(), id);
    *ctx.meta_mut(id) = meta;
}

/// `id`'s meta with the value its operands give (see [`refresh_node`]).
fn refreshed_node_meta(graph: &IrGraph, id: NodeId) -> crate::fx_ir::NodeMeta {
    if is_table(graph, id) {
        let input = all_input_nodes(graph.node(id))[0];
        let shape = graph
            .node_shape_vec(input)
            .unwrap_or_else(|| panic!("KeyError: 'val' on {}", graph.node(input).name));
        let mut meta = graph.node(id).meta.clone();
        let dtype = meta.val_dtype().unwrap_or("torch.int8").to_owned();
        meta.set_tensor_facts(TensorFacts::new(shape, dtype));
        mark_contiguous(&mut meta);
        meta
    } else {
        refreshed_meta(graph, id).unwrap_or_else(|error| {
            let node = graph.node(id);
            panic!("cannot refresh {} ({}): {error}", node.name, node.target)
        })
    }
}

/// `id`'s value shape becomes `shape`, keeping its dtype:
/// `val.new_empty(shape)`.
fn set_shape(ctx: &mut LayoutCtx<'_>, id: NodeId, shape: Vec<i64>) {
    let meta = ctx.meta_mut(id);
    let dtype = meta.val_dtype().unwrap_or("torch.float32").to_owned();
    meta.set_tensor_facts(TensorFacts::new(shape, dtype));
    mark_contiguous(meta);
}

/// A TABLE: rc4's node, or a pending table (`layout::pending`).
fn is_table(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    rc4_target(&node.target) == TABLE || is_pending_table(node)
}

/// `_node_or_inputs_are_int48` (backends/arm/_passes/propagate_view_copy_permute_pass.py:81-88).
fn node_or_inputs_are_int48(graph: &IrGraph, id: NodeId, reads: &mut Reads) -> bool {
    reads.add(id);
    if special_dtype(&graph.node(id).meta) == Some("INT48") {
        cover("int48");
        return true;
    }
    let inputs = all_input_nodes(graph.node(id));
    reads.extend(inputs.iter().copied());
    inputs
        .iter()
        .any(|&input| special_dtype(&graph.node(input).meta) == Some("INT48"))
}

/// `id.users` in torch.fx's order (`LayoutCtx::users_in_fx_order`); reading
/// them is reading `id`, whose version changes when they reorder.
fn fx_users(ctx: &LayoutCtx<'_>, id: NodeId, reads: &mut Reads) -> Vec<NodeId> {
    reads.add(id);
    ctx.users_in_fx_order(id)
}

/// `PermuteMap(node)`, which raises for dims that aren't a permutation.
fn permute_map(graph: &IrGraph, node: NodeId) -> PermuteMap {
    let n = graph.node(node);
    let dims = match n.args.get(1).and_then(dim_arg) {
        Some(DimArg::List(dims)) => dims,
        _ => panic!("AssertionError: {} has no permute dims", n.name),
    };
    PermuteMap::new(&dims).unwrap_or_else(|error| panic!("ValueError: Invalid permute dims: {error:?}"))
}

/// `ViewMap(node)` (backends/transforms/dim_maps.py:258-271), which asserts a
/// `view_copy` with a tensor input and raises for a bad view shape.
fn view_map(graph: &IrGraph, node: NodeId, reads: &mut Reads) -> ViewMap {
    let n = graph.node(node);
    assert!(
        n.target == VIEW_TARGET,
        "AssertionError: ViewMap needs a view_copy, got {}",
        n.target
    );
    let input = n
        .args
        .first()
        .and_then(IrArg::as_node)
        .unwrap_or_else(|| panic!("AssertionError: {} reads no node", n.name));
    reads.add(input);
    let source = graph
        .node_shape_vec(input)
        .unwrap_or_else(|| panic!("AssertionError: {} has no tensor value", graph.node(input).name));
    let target = n
        .args
        .get(1)
        .and_then(int_list)
        .unwrap_or_else(|| panic!("TypeError: view shape of {}", n.name));
    ViewMap::from_shapes(&source, &target).unwrap_or_else(|error| panic!("AssertionError: {error:?}"))
}

/// `next_node.args[index]`, which raises `IndexError` when absent.
fn next_arg(graph: &IrGraph, node: NodeId, index: usize) -> &IrArg {
    let n = graph.node(node);
    n.args
        .get(index)
        .unwrap_or_else(|| panic!("IndexError: {} has no args[{index}]", n.name))
}

/// `next_node.args[2] if len(next_node.args) > 2 else next_node.kwargs.get("keepdim")`.
fn keep_dim_arg(graph: &IrGraph, node: NodeId) -> Option<&IrArg> {
    let n = graph.node(node);
    n.args.get(2).or_else(|| n.kwargs.get("keepdim"))
}

/// `arg is True`.
fn is_true(arg: Option<&IrArg>) -> bool {
    matches!(arg, Some(IrArg::Bool(true)))
}

/// Whether `lhs` and `rhs`, two arg lists of `id`, make the same op: the same
/// args, or for a mean, sum or slice the same args but its dims, where those
/// dims do the same to its input. A keepdim reduction is the same over axes
/// that differ only in size-1 axes of its input (`[0, 2, -1]` and `[3, 0]`
/// for an input of shape `[2, 3, 1, 4]`); a slice is the same on the same
/// axis, or when it keeps the whole of either axis.
fn same_op_args(graph: &IrGraph, id: NodeId, lhs: &[IrArg], rhs: &[IrArg]) -> bool {
    if same_args(lhs, rhs) {
        return true;
    }
    let target = graph.node(id).target.as_str();
    if !matches!(target, MEAN_DIM | SUM_DIM_INT_LIST | SLICE_COPY) || lhs.len() != rhs.len() || lhs.len() < 2 {
        return false;
    }
    if !lhs.iter().zip(rhs).enumerate().all(|(index, (a, b))| index == 1 || same_arg(a, b)) {
        return false;
    }
    let Some(shape) = lhs[0].as_node().and_then(|input| graph.node_shape(input)) else {
        return false;
    };
    if shape.is_empty() {
        return false;
    }
    let rank = shape.len() as i64;
    let axes = |arg: &IrArg| -> Option<std::collections::BTreeSet<usize>> {
        Some(dim_arg(arg)?.dims().iter().map(|dim| dim.rem_euclid(rank) as usize).collect())
    };
    let (Some(a), Some(b)) = (axes(&lhs[1]), axes(&rhs[1])) else {
        return false;
    };
    if a == b {
        return true;
    }
    if target == SLICE_COPY {
        let whole = |axis: usize| {
            lhs.get(2).and_then(python_int) == Some(0)
                && lhs.get(3).and_then(python_int).is_some_and(|end| end >= shape[axis])
                && lhs.get(4).is_none_or(|step| python_int(step) == Some(1))
        };
        let only = |axes: &std::collections::BTreeSet<usize>| (axes.len() == 1).then(|| *axes.iter().next().expect("one axis"));
        return matches!((only(&a), only(&b)), (Some(x), Some(y)) if whole(x) && whole(y));
    }
    if !is_true(keep_dim_arg(graph, id)) {
        return false;
    }
    let wide = |axes: &std::collections::BTreeSet<usize>| -> std::collections::BTreeSet<usize> {
        axes.iter().copied().filter(|&axis| shape[axis] != 1).collect()
    };
    wide(&a) == wide(&b)
}

/// Whether two argument lists are the same, value for value (floats by bits).
fn same_args(lhs: &[IrArg], rhs: &[IrArg]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(lhs, rhs)| same_arg(lhs, rhs))
}

fn same_arg(lhs: &IrArg, rhs: &IrArg) -> bool {
    match (lhs, rhs) {
        (IrArg::Node(a), IrArg::Node(b)) => a == b,
        (IrArg::Int(a), IrArg::Int(b)) => a == b,
        (IrArg::Float(a), IrArg::Float(b)) => a.to_bits() == b.to_bits(),
        (IrArg::Bool(a), IrArg::Bool(b)) => a == b,
        (IrArg::Str(a), IrArg::Str(b)) | (IrArg::Dtype(a), IrArg::Dtype(b)) => a == b,
        (IrArg::List(a), IrArg::List(b)) => same_args(a, b),
        (IrArg::None, IrArg::None) => true,
        _ => false,
    }
}

/// An int argument, as `isinstance(arg, int)` reads it.
fn python_int(arg: &IrArg) -> Option<i64> {
    match arg {
        IrArg::Int(value) => Some(*value),
        IrArg::Bool(value) => Some(i64::from(*value)),
        _ => None,
    }
}

/// Python's `repr` of an argument, for rc4's error messages.
fn python_repr(arg: &IrArg) -> String {
    match arg {
        IrArg::Bool(true) => "True".to_string(),
        IrArg::Bool(false) => "False".to_string(),
        IrArg::None => "None".to_string(),
        IrArg::Int(value) => value.to_string(),
        other => format!("{other:?}"),
    }
}

/// `args` with `args[index]` replaced.
fn with_arg(args: &[IrArg], index: usize, value: IrArg) -> Vec<IrArg> {
    let mut args = args.to_vec();
    args[index] = value;
    args
}

/// `(*node.args[:1], shape)` for a view.
fn view_args(graph: &IrGraph, node: NodeId, shape: &[i64]) -> Vec<IrArg> {
    vec![graph.node(node).args[0].clone(), i64_list_arg(shape)]
}

/// `_source_shape` (`:1200-1207`): the shape before a permute by
/// `normalized_dims` whose output has `shape`.
fn source_shape(shape: &[i64], normalized_dims: &[usize]) -> Vec<i64> {
    let mut source = vec![1; normalized_dims.len()];
    for (output_axis, &source_axis) in normalized_dims.iter().enumerate() {
        source[source_axis] = shape[output_axis];
    }
    source
}

/// `_map_slice_dim` (`:774-791`).
fn map_slice_dim(view_map: &ViewMap, slice_dim: i64) -> Option<usize> {
    let new_dims = view_map
        .map_source_dims_to_target_axes(&[slice_dim])
        .unwrap_or_else(|error| panic!("AssertionError: {error:?}"))?;
    if new_dims.len() != 1 {
        return None;
    }
    let new_dim = new_dims[0];
    let normalized_slice_dim = slice_dim.rem_euclid(view_map.source_rank() as i64) as usize;
    let source_to_target_axes = view_map
        .source_to_target_axes()
        .unwrap_or_else(|error| panic!("AssertionError: {error:?}"));
    let target_source_axes = ViewMap::source_axes_for_target_axis(new_dim, &source_to_target_axes);
    if target_source_axes.iter().any(|&axis| axis != normalized_slice_dim) {
        return None;
    }
    Some(new_dim)
}

/// `_valid_slice_interval` (`:793-801`): `0 <= start < end <= shape[dim]`,
/// false for a non-int bound.
fn valid_slice_interval(shape: &[i64], dim: i64, start: Option<i64>, end: Option<i64>) -> bool {
    assert!(!shape.is_empty(), "ZeroDivisionError: integer modulo by zero");
    let dim = dim.rem_euclid(shape.len() as i64) as usize;
    match (start, end) {
        (Some(start), Some(end)) => 0 <= start && start < end && end <= shape[dim],
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph, config: PropagateConfig) -> bool {
        IrPropagateViewCopyPermutePass { config }.run(g.graph_mut())
    }

    #[test]
    fn down_sinks_a_permute_below_unary_ops() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let p = g.permute(x, &[0, 2, 3, 1]);
        let r = g.call(RELU, vec![node(p)]);
        let n = g.call(NEG, vec![node(r)]);
        g.output(&[n]);
        assert!(run(&mut g, PropagateConfig::arm_without_program(Direction::Down)));
        g.assert_targets(&[RELU, NEG, PERMUTE]);
        assert_eq!(g.shape(g.id("relu")), [1, 2, 3, 4]);
        assert_eq!(g.shape(g.id("neg")), [1, 2, 3, 4]);
        assert_eq!(g.shape(g.id("permute_copy")), [1, 3, 4, 2]);
    }

    /// A grouped conv: IrRewriteConv's per-group outputs are
    /// int8 NHWC values, their cat is a quantized compute node RETAB keeps as
    /// float32 with int8 qparams (design F2), and FuseIdenticalInputTransforms
    /// has sunk the groups' NCHW permutes below it as one permute, whose val it
    /// refreshed from the cat: float32, without qparams. Up splits the cat
    /// fan-out (`:824-886`) and gives the cat that permute's meta, which holds
    /// int8 in rc4. The cat must keep int8 (from the permute's input,
    /// `value_dtype` rule 4), or the serializer types its CONCAT float32, which
    /// the INT profile rejects (test_conv2d.py groups_bias).
    #[test]
    fn up_keeps_a_split_cats_storage_dtype() {
        let mut g = LayoutGraph::new();
        let r0 = g.typed_input("r0", &[1, 5, 5, 1], "torch.int8");
        let r1 = g.typed_input("r1", &[1, 5, 5, 1], "torch.int8");
        let cat = g.cat(&[r0, r1], 3);
        let int8 = crate::fx_ir::QParam::per_tensor(0.0134, 12, -128, 127, "torch.int8".to_string());
        let meta = &mut g.graph_mut().node_mut(cat).meta;
        meta.set_val_dtype_for_existing_shape("torch.float32");
        meta.set_input_qparams_from([(0, int8.clone())]);
        meta.set_single_output_qparam(int8);
        let p = g.permute(cat, &[0, 3, 1, 2]);
        assert_eq!(g.dtype(p), "torch.float32");
        g.output(&[p]);
        assert!(run(&mut g, PropagateConfig::arm_without_program(Direction::Up)));
        // The driver's canonicalization turns the cat inputs' singleton permutes
        // into views, as rc4's does.
        g.assert_targets(&[VIEW, VIEW, super::CAT]);
        let cat = g.id("cat");
        assert_eq!(g.shape(cat), [1, 2, 5, 5]);
        assert_eq!(g.dtype(cat), "torch.int8");
        assert!(g.graph().node(cat).meta.output_qparam().is_none(), "rc4's permute meta");
        for input in all_input_nodes(g.graph().node(cat)) {
            assert_eq!(g.dtype(input), "torch.int8");
        }
    }

    /// An A16W8 matmul (test_matmul.py `double_input_randn_rand_1d_4d`): the
    /// int16 product is INT48, as rc4's RewriteMatmulPass marks it, so its
    /// RESCALE is not elementwise (`_is_per_tensor_rescale` compares the special
    /// dtypes) and Up keeps the bmm decomposition's output view after it.
    /// Hoisted, the view would reshape the packed INT48 accumulator, which the
    /// TOSA reference model rejects. An unmarked (int8) product lets it cross.
    #[test]
    fn up_keeps_a_view_after_the_rescale_of_an_int48_product() {
        for (int48, crosses) in [(true, false), (false, true)] {
            let mut g = LayoutGraph::new();
            let a = g.typed_input("a", &[8, 1, 5], "torch.int16");
            let b = g.typed_input("b", &[8, 5, 3], "torch.int16");
            let bmm = g.add_unrefreshed("aten.bmm.default", vec![node(a), node(b)]);
            let meta = &mut g.graph_mut().node_mut(bmm).meta;
            meta.set_tensor_facts(TensorFacts::new(vec![8, 1, 3], "torch.int32"));
            if int48 {
                crate::layout::dtype::set_special_dtype(meta, crate::layout::dtype::SpecialDtype::Int48);
            }
            let rescale = g.rescale(bmm, "torch.int16", 0.5);
            let view = g.view(rescale, &[2, 4, 3]);
            g.output(&[view]);
            let modified = run(&mut g, PropagateConfig::arm_without_program(Direction::Up));
            assert_eq!(modified, crosses, "int48={int48}");
            let expected: &[&str] = if crosses {
                &["aten.bmm.default", VIEW, RESCALE]
            } else {
                &["aten.bmm.default", RESCALE, VIEW]
            };
            g.assert_targets(expected);
        }
    }

    #[test]
    fn up_hoists_a_permute_above_unary_ops() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let r = g.call(RELU, vec![node(x)]);
        let n = g.call(NEG, vec![node(r)]);
        let p = g.permute(n, &[0, 2, 3, 1]);
        g.output(&[p]);
        assert!(run(&mut g, PropagateConfig::arm_without_program(Direction::Up)));
        g.assert_targets(&[PERMUTE, RELU, NEG]);
        assert_eq!(g.shape(g.id("relu")), [1, 3, 4, 2]);
    }

    /// The undo before a region move runs only where the walk's args change
    /// what a crossed mean or slice computes.
    #[test]
    fn same_op_args_reads_dims_as_axes_of_the_input() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 1, 4]);
        let mean = g.call(MEAN, vec![node(x), ints(&[0, 2, -1]), IrArg::Bool(true)]);
        let part = g.slice(x, 1, 0, 2);
        let whole = g.slice(x, 1, 0, 3);
        let graph = g.graph();
        let with_dims = |id: NodeId, dims: IrArg| with_arg(&graph.node(id).args, 1, dims);
        let mean_args = graph.node(mean).args.clone();
        // The same axes in another order, and size-1 axis 2 dropped.
        assert!(same_op_args(graph, mean, &mean_args, &with_dims(mean, ints(&[3, 0, 2]))));
        assert!(same_op_args(graph, mean, &mean_args, &with_dims(mean, ints(&[3, 0]))));
        assert!(!same_op_args(graph, mean, &mean_args, &with_dims(mean, ints(&[1, 2, 3]))));
        // A slice on the same axis, or keeping the whole of both axes.
        let part_args = graph.node(part).args.clone();
        assert!(same_op_args(graph, part, &part_args, &with_dims(part, IrArg::Int(-3))));
        assert!(!same_op_args(graph, part, &part_args, &with_dims(part, IrArg::Int(0))));
        let whole_args = graph.node(whole).args.clone();
        assert!(same_op_args(graph, whole, &whole_args, &with_dims(whole, IrArg::Int(2))));
        assert!(!same_op_args(graph, whole, &whole_args, &with_dims(whole, IrArg::Int(3))));
    }

    #[test]
    fn helpers_match_rc4() {
        assert_eq!(source_shape(&[1, 3, 4, 2], &[0, 2, 3, 1]), [1, 2, 3, 4]);
        assert!(valid_slice_interval(&[4, 5], 1, Some(0), Some(5)));
        assert!(!valid_slice_interval(&[4, 5], -1, Some(0), Some(6)));
        assert!(!valid_slice_interval(&[4, 5], 0, None, Some(2)));
        assert_eq!(dim_arg(&IrArg::None), Some(DimArg::List(Vec::new())));
        assert_eq!(dim_arg(&IrArg::Bool(true)), Some(DimArg::Int(1)));
        assert_eq!(dim_arg(&IrArg::Float(1.0)), None);
    }
}
