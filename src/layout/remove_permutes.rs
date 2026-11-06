//! Port of rc4's `RemovePermutesAroundElementwiseOps`
//! (`backends/transforms/remove_permutes_around_elementwise_ops.py`) and of
//! the Arm subclass rc4's propagation passes run as their fusion pass,
//! `RemovePermutesAroundElementwiseTosaOps`
//! (`backends/arm/_passes/remove_permutes_around_elementwise_tosa_ops.py`,
//! constructed by `TosaPropagationOverrides.make_fusion_pass`,
//! `backends/arm/_passes/propagate_view_copy_permute_pass.py:90-93`), with the
//! helpers they use from `backends/transforms/permute_pass_utils.py` and
//! `backends/transforms/channels_last_layout.py`.
//!
//! The pass looks for regions of elementwise ops entered through a permute P
//! and left through its inverse, deletes both permutes and makes the region
//! compute in the layout before P:
//!
//! ```text
//! x -> permute(P) -> add(., c) -> relu -> permute(P^-1) -> y
//! x -> add(., permute(c, P^-1)) -> relu -> y
//! ```
//!
//! Dim arguments in the region are remapped (cat, keepdim reductions, slice,
//! constant pads, squeeze/unsqueeze views and the repeat_interleave triple),
//! constants get a compensating permute (or a view when their rank is lower),
//! numel-1 inputs stay as they are, other incoming permutes with the same
//! permutation are bypassed too, and an outgoing permute that is not the
//! inverse absorbs the removed one instead.
//!
//! # Faithfulness
//!
//! The detection sweep, the region search and every rewrite follow rc4 line by
//! line; each function names the rc4 lines it mirrors. rc4's region search
//! defers nested visits onto a stack (`visit`, `:457-485`), which is not a
//! recursive DFS, and that order decides which permutation a node reached on
//! two paths gets; [`Detector::visit`] keeps the same stack discipline. torch.fx
//! iterates `node.users` in insertion order, which after a retrace is graph
//! order, so users are visited in graph order. (rc4's propagation passes do not
//! retrace between their propagation steps, so when they call this pass FX's
//! order can differ; that changes the result only for a node a search reaches
//! on two paths with different permutations.) `all_input_nodes` is positional
//! args, then kwargs sorted by name (RETAB keeps kwargs in a map).
//!
//! Where RETAB's IR differs from rc4's at this point of the pipeline:
//!
//! - **TABLE.** rc4 has already replaced quantized table ops with
//!   `TABLE(x, table)` (`layout::pending`). The TOSA subclass drops the table's
//!   constant edge (`:45-54`), so a pending table differs from its TABLE only in
//!   the visit's constant-rank check on that rank-1 table input, which fails a
//!   rank-0 region (`:631-637`); [`Detector::visit_one`] applies that check.
//! - **Constants.** rc4's subclass reads parameters, buffers and lifted
//!   constants from the graph signature (`:30-43`); RETAB marks them with
//!   tensor data. The base rule (`get_attr`, `b_`/`p_`/`c_` placeholder names,
//!   `full`) is kept as is (`:672-688`).
//! - **The retrace.** rc4 ends with `ExportPass.call`, which re-derives every
//!   node's value (`:451-453`). The port re-derives only the values a rewrite
//!   can change, in graph order: the region, the views it remaps, the nodes it
//!   creates and the permutes it rewrites, then any node whose input value
//!   changed (see [`refresh`]). Dtypes stay as recorded (`layout::shape`),
//!   except a RESCALE's, which is its dtype argument as in rc4's fake kernel. A
//!   value torch would reject panics, as rc4's retrace raises. The retrace also
//!   renames call nodes after their targets; RETAB keeps the names.
//! - **Edge order.** rc4 iterates `edges_in` and `constant_edges_in`, sets of
//!   node pairs, in an order that depends on object addresses, and the order
//!   can matter. Each compensating node goes directly after its constant
//!   (`:850`), so the nodes for one constant feeding several region nodes land
//!   in reverse iteration order. And `replace_input_with` rewrites every use,
//!   so when one bypassed permute feeds another into the same region node,
//!   bypassing the downstream one first lets the upstream bypass rewrite both
//!   uses (`:832-838`). The port iterates both sets in graph order of their
//!   pairs, which also gives the correct rewrite in the second case; the replay
//!   generator pins rc4 to the same order.
//!
//! # Complexity
//!
//! rc4 rescans the whole graph and re-traces it on every call, and its
//! propagation passes call it on every fusion round. Per call, the port scans
//! the node order once for permutes (O(N)), and a region search costs the nodes
//! it reaches. Searches that cannot change are skipped:
//!
//! - a search that failed is skipped for the rest of the call when it starts
//!   at the same node with the same permutation (its outcome is fixed: the graph
//!   does not change before the rewrites, and nodes claimed by an earlier region
//!   only add failures);
//! - [`RemovePermutesMemo`] carries failed searches across calls that share a
//!   `LayoutCtx`, keyed by (permute, user) and invalidated through the ctx's
//!   version stamps of every node the search read (searches the per-call
//!   caches skipped share the read set of their proof, checked once per call);
//! - a failed search over a *closed simple* component (permutable nodes that
//!   neither change rank nor are numel-1, with every traversable neighbour
//!   inside) fails from every start inside it with the same permutation, so
//!   its failure is cached per component (see [`ComponentCache`]).
//!
//! The rewrite touches only the region and its boundary, the value refresh only
//! the nodes whose value can change, and dead code goes in one sweep (O(N),
//! as rc4's `eliminate_dead_code`) or, inside a caller that keeps the graph free
//! of dead code, only from the nodes the call stopped using ([`DeadCode`]).
//! Region searches that start outside closed simple components can still revisit
//! a region once per entry permute, as rc4 does.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrNode, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::fx_target::{classify_ir_target, IrTargetKind};
use crate::layout::ctx::{is_impure_call, LayoutCtx};
use crate::layout::pending::{is_pending_table, TABLE_OPS};
use crate::layout::pointwise::is_pointwise;
use crate::layout::shape::{
    has_local_rule, infer_facts, int_list, mark_contiguous, reduce_shape, refreshed_meta,
    ShapeError,
};

/// `PERMUTE_COPY_TARGETS` (`backends/transforms/channels_last_layout.py:14-18`).
const PERMUTE_COPY_TARGETS: &[&str] = &[
    "aten.permute_copy.default",
    "channels_last.permute_copy.default",
];
const PERMUTE_COPY: &str = "aten.permute_copy.default";
/// `_VIEW_OPS` (`:125-128`).
const VIEW_OPS: &[&str] = &["aten.view_copy.default", "aten.view.default"];
const VIEW_COPY: &str = "aten.view_copy.default";
/// `_PAD_OPS` (`:154-157`).
const PAD_OPS: &[&str] = &["aten.constant_pad_nd.default", "aten.pad.default"];
const PAD_DEFAULT: &str = "aten.pad.default";
/// The reductions whose `keepdim` gates permutability and whose dims are
/// remapped (`:712-719`, `:804-810`).
const REDUCTIONS: &[&str] = &[
    "aten.amax.default",
    "aten.amin.default",
    "aten.mean.dim",
    "aten.sum.dim_IntList",
];
const CAT: &str = "aten.cat.default";
const SLICE_COPY: &str = "aten.slice_copy.Tensor";
const UNSQUEEZE_COPY: &str = "aten.unsqueeze_copy.default";
const EXPAND_COPY: &str = "aten.expand_copy.default";
const FULL: &str = "aten.full.default";
const QUANTIZE_PER_TENSOR: &str = "quantized_decomposed.quantize_per_tensor.default";
const DEQUANTIZE_PER_TENSOR: &str = "quantized_decomposed.dequantize_per_tensor.default";

/// The base `_permutable_ops` (`:84-107`): the quantize/dequantize pair is
/// registered in rc4's environment, so the `try` always adds it.
const BASE_PERMUTABLE_OPS: &[&str] = &[
    "aten.add.Tensor",
    "aten.amax.default",
    "aten.amin.default",
    "aten.mul.Tensor",
    "aten.sub.Tensor",
    "aten.hardtanh.default",
    "aten.clamp.default",
    CAT,
    "aten.constant_pad_nd.default",
    "aten.mean.dim",
    PAD_DEFAULT,
    "aten.sum.dim_IntList",
    SLICE_COPY,
    QUANTIZE_PER_TENSOR,
    DEQUANTIZE_PER_TENSOR,
];

/// `target_filters` (`:79-81`, `:110-116`): a per-target predicate that can
/// veto permutability. It must read only the node it is given.
pub(crate) type TargetFilter = fn(&IrGraph, NodeId) -> bool;

/// How the pass is constructed: the base pass's `extra_permutable_ops` and
/// `target_filters`, and whether the TOSA subclass's overrides apply.
#[derive(Clone, Default)]
pub(crate) struct RemovePermutesConfig {
    /// `RemovePermutesAroundElementwiseTosaOps`: TABLE ops, RESCALE and TABLE
    /// are permutable (`:19-26`), graph-signature constants count as constants
    /// (`:40-43`) and TABLE's constant edges are dropped (`:45-54`).
    pub(crate) tosa: bool,
    pub(crate) extra_permutable_ops: Vec<&'static str>,
    pub(crate) target_filters: Vec<(&'static str, TargetFilter)>,
    /// Test hook for rc4's `RankMismatchPass`, which overrides
    /// `get_permutation` for one node (named here).
    #[cfg(test)]
    pub(crate) permutation_override: Option<(String, Vec<usize>)>,
}

impl RemovePermutesConfig {
    /// `RemovePermutesAroundElementwiseOps()`.
    pub(crate) fn base() -> Self {
        Self::default()
    }

    /// `RemovePermutesAroundElementwiseTosaOps(exported_program)`, as
    /// `TosaPropagationOverrides.make_fusion_pass` builds it.
    pub(crate) fn tosa() -> Self {
        Self {
            tosa: true,
            ..Self::default()
        }
    }

    /// `node.target in self._permutable_ops` (`:711`).
    fn is_permutable_op(&self, target: &str) -> bool {
        BASE_PERMUTABLE_OPS.contains(&target)
            || self
                .extra_permutable_ops
                .iter()
                .any(|&extra| same_target(extra, target))
            || (self.tosa
                && (TABLE_OPS.binary_search(&target).is_ok()
                    || classify_ir_target(target) == IrTargetKind::TosaRescale
                    || is_tosa_table_target(target)))
    }

    fn target_filter(&self, target: &str) -> Option<TargetFilter> {
        self.target_filters
            .iter()
            .find(|(filtered, _)| same_target(filtered, target))
            .map(|&(_, filter)| filter)
    }
}

/// RETAB spells TOSA dialect targets `backend.tosa.X` or `tosa.X`.
fn same_target(lhs: &str, rhs: &str) -> bool {
    let canonical = |target: &str| {
        target
            .strip_prefix("backend.")
            .unwrap_or(target)
            .to_string()
    };
    lhs == rhs || canonical(lhs) == canonical(rhs)
}

fn is_tosa_table_target(target: &str) -> bool {
    matches!(target, "backend.tosa.TABLE.default" | "tosa.TABLE.default")
}

/// What to do about dead code once the pass changed the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadCode {
    /// `graph_module.graph.eliminate_dead_code()` (`:451`): every dead node goes.
    Full,
    /// Only the nodes this call stopped using, and what only they used. The
    /// same result as `Full` when the graph had no dead code before the call,
    /// which a caller that eliminates dead code after every change can promise.
    Local,
}

/// rc4's pass as constructed for the Arm pipeline.
pub(crate) struct IrRemovePermutesAroundElementwiseTosaOpsPass;

impl IrPass for IrRemovePermutesAroundElementwiseTosaOpsPass {
    fn name(&self) -> &str {
        "IrRemovePermutesAroundElementwiseTosaOps"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        run_standalone(graph, &RemovePermutesConfig::tosa())
    }
}

/// rc4's base pass with an explicit configuration, for the base pass's tests.
pub(crate) struct IrRemovePermutesAroundElementwiseOpsPass {
    pub(crate) config: RemovePermutesConfig,
}

impl IrPass for IrRemovePermutesAroundElementwiseOpsPass {
    fn name(&self) -> &str {
        if self.config.tosa {
            "IrRemovePermutesAroundElementwiseTosaOps"
        } else {
            "IrRemovePermutesAroundElementwiseOps"
        }
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        run_standalone(graph, &self.config)
    }
}

fn run_standalone(graph: &mut IrGraph, config: &RemovePermutesConfig) -> bool {
    if !graph
        .live_call_function_ids()
        .into_iter()
        .any(|id| is_permute_copy(graph.node(id)))
    {
        return false;
    }
    let mut ctx = LayoutCtx::new(graph);
    let modified = remove_permutes_in_ctx(
        &mut ctx,
        config,
        &mut RemovePermutesMemo::default(),
        DeadCode::Full,
    );
    if modified {
        ctx.finish();
    }
    modified
}

/// One call of rc4's `call` (`:418-455`) inside a caller's `LayoutCtx`: find
/// every region, rewrite those still current, then eliminate dead code and
/// refresh the values the rewrites changed. Returns whether anything changed.
///
/// A caller that runs the pass repeatedly on one ctx (rc4's propagation passes
/// run it on every fusion round) passes the same `memo` each time so searches
/// whose inputs have not changed are skipped.
pub(crate) fn remove_permutes_in_ctx(
    ctx: &mut LayoutCtx<'_>,
    config: &RemovePermutesConfig,
    memo: &mut RemovePermutesMemo,
    dead_code: DeadCode,
) -> bool {
    let found = find_subgraphs(ctx, config, memo);
    let mut rewrite = Rewrite::default();
    let mut modified = false;
    for mut subgraph in found {
        if permute_subgraph(ctx, config, &mut subgraph, &mut rewrite) {
            modified = true;
        }
    }
    if !modified {
        return false;
    }
    match dead_code {
        DeadCode::Full => {
            ctx.eliminate_dead_code();
        }
        DeadCode::Local => eliminate_dead_from(ctx, rewrite.unused.iter().copied()),
    }
    refresh(ctx, &rewrite);
    true
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

type Perm = Rc<[usize]>;

#[cfg(test)]
thread_local! {
    /// Region-search steps (`_visit` calls) on this thread, for scaling tests.
    static SEARCH_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Whether test builds re-run each skipped search to check it fails.
    static CHECK_SKIPPED_SEARCHES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// A matched repeat_interleave: `(dim, scale, expand_node, view_node)`
/// (`:61-65`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Triple {
    dim: usize,
    scale: i64,
    expand: NodeId,
    view: NodeId,
}

/// A set that iterates in insertion order.
#[derive(Debug, Clone)]
struct OrderedSet<T> {
    items: Vec<T>,
    seen: HashSet<T>,
}

impl<T> Default for OrderedSet<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            seen: HashSet::new(),
        }
    }
}

impl<T: Copy + Eq + Hash> OrderedSet<T> {
    fn insert(&mut self, item: T) -> bool {
        let fresh = self.seen.insert(item);
        if fresh {
            self.items.push(item);
        }
        fresh
    }

    fn contains(&self, item: &T) -> bool {
        self.seen.contains(item)
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let seen = &mut self.seen;
        self.items.retain(|item| {
            let kept = keep(item);
            if !kept {
                seen.remove(item);
            }
            kept
        });
    }
}

/// `RemovePermutesAroundElementwiseOps.Subgraph` (`:36-74`).
#[derive(Debug, Clone)]
struct Subgraph {
    start_permute: Perm,
    end_permute: Perm,
    nodes: OrderedSet<NodeId>,
    edges_in: OrderedSet<(NodeId, NodeId)>,
    edges_out: OrderedSet<(NodeId, NodeId)>,
    /// `(producer, permute, new permutation)`; a permute has one producer, so
    /// `(producer, permute)` identifies an entry.
    edges_out_to_update: Vec<(NodeId, NodeId, Vec<usize>)>,
    constant_edges_in: OrderedSet<(NodeId, NodeId)>,
    node_end_permute: HashMap<NodeId, Perm>,
    node_start_permute: HashMap<NodeId, Perm>,
    /// Keyed by head, in insertion order.
    interleaves: Vec<(NodeId, Triple)>,
    view_shape_overrides: HashMap<NodeId, Vec<i64>>,
    sink_edges_in: OrderedSet<(NodeId, NodeId)>,
}

impl Subgraph {
    fn new(start_permute: Perm, end_permute: Perm) -> Self {
        Self {
            start_permute,
            end_permute,
            nodes: OrderedSet::default(),
            edges_in: OrderedSet::default(),
            edges_out: OrderedSet::default(),
            edges_out_to_update: Vec::new(),
            constant_edges_in: OrderedSet::default(),
            node_end_permute: HashMap::new(),
            node_start_permute: HashMap::new(),
            interleaves: Vec::new(),
            view_shape_overrides: HashMap::new(),
            sink_edges_in: OrderedSet::default(),
        }
    }

    /// `subgraph.node_start_permute.get(node, subgraph.start_permute)`.
    fn start_of(&self, node: NodeId) -> Perm {
        self.node_start_permute
            .get(&node)
            .cloned()
            .unwrap_or_else(|| self.start_permute.clone())
    }

    /// `subgraph.node_end_permute.get(node, subgraph.end_permute)`.
    fn end_of(&self, node: NodeId) -> Perm {
        self.node_end_permute
            .get(&node)
            .cloned()
            .unwrap_or_else(|| self.end_permute.clone())
    }

    fn add_edge_out_to_update(&mut self, producer: NodeId, permute: NodeId, perm: Vec<usize>) {
        if !self
            .edges_out_to_update
            .iter()
            .any(|entry| (entry.0, entry.1, &entry.2) == (producer, permute, &perm))
        {
            self.edges_out_to_update.push((producer, permute, perm));
        }
    }
}

/// A deferred `visit` call (`:465-474`).
struct Pending {
    node: NodeId,
    end: Perm,
    start: Perm,
}

/// rc4's detection loop (`:419-443`).
fn find_subgraphs(
    ctx: &LayoutCtx<'_>,
    config: &RemovePermutesConfig,
    memo: &mut RemovePermutesMemo,
) -> Vec<Subgraph> {
    let mut detector = Detector::new(config);
    let mut found = Vec::new();
    let mut processed: HashSet<NodeId> = HashSet::new();
    // Searches that failed without meeting a claimed node, by start node and
    // permutation: the same search fails again for the rest of the call.
    let mut failed_starts: HashMap<(NodeId, Perm), ReadSet> = HashMap::new();
    // Whether each memo read set met so far is still current.
    let mut current: HashMap<usize, (ReadSet, bool)> = HashMap::new();
    let permutes: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| is_permute_copy(ctx.node(id)))
        .collect();
    for permute in permutes {
        let Some(start) = detector.permutation(ctx, permute) else {
            continue;
        };
        let start: Perm = start.into();
        let end: Perm = inverse(&start).into();
        for user in users(ctx, permute) {
            if !detector.is_node_permutable(ctx, user)
                && detector.interleave_triple(ctx, user).is_none()
            {
                continue;
            }
            // A search known to fail is skipped; one proven by this call's
            // caches is remembered with the read set of the proof.
            let skipped = if let Some(reads) = failed_starts.get(&(user, start.clone())) {
                Some(Some(reads.clone()))
            } else if memo.known_failure(ctx, permute, user, &mut current) {
                Some(None)
            } else {
                detector.components.known_failure(user, &start).map(Some)
            };
            if let Some(proof) = skipped {
                // Test builds check that the skipped search does fail.
                #[cfg(test)]
                if CHECK_SKIPPED_SEARCHES.with(std::cell::Cell::get) {
                    let mut check = Subgraph::new(start.clone(), end.clone());
                    assert!(
                        !detector.visit(ctx, user, &mut check, &processed),
                        "a skipped search from {} succeeds",
                        ctx.node(user).name
                    );
                }
                if let Some(reads) = proof {
                    memo.record(permute, user, reads);
                }
                continue;
            }
            let mut subgraph = Subgraph::new(start.clone(), end.clone());
            if detector.visit(ctx, user, &mut subgraph, &processed) {
                processed.extend(subgraph.nodes.iter().copied());
                found.push(subgraph);
                continue;
            }
            if detector.hit_processed {
                continue;
            }
            detector.reads.push(permute);
            let reads = stamp(ctx, detector.reads.iter().copied());
            memo.record(permute, user, reads.clone());
            failed_starts.insert((user, start.clone()), reads);
            if detector.simple {
                let search_reads = std::mem::take(&mut detector.reads);
                detector.cache_component_failure(ctx, user, &start, &search_reads);
            }
        }
    }
    found
}

/// The region search and the node predicates it uses, with rc4's per-call
/// caches (`:117-123`).
struct Detector<'c> {
    config: &'c RemovePermutesConfig,
    sq_unsq_cache: HashMap<NodeId, bool>,
    interleave_cache: HashMap<NodeId, Option<Triple>>,
    components: ComponentCache,
    /// Every node the current search read.
    reads: Vec<NodeId>,
    /// Whether the current search met a node an earlier region claimed.
    hit_processed: bool,
    /// Whether every node the current search added is simple (see
    /// [`ComponentCache`]).
    simple: bool,
}

impl<'c> Detector<'c> {
    fn new(config: &'c RemovePermutesConfig) -> Self {
        Self {
            config,
            sq_unsq_cache: HashMap::new(),
            interleave_cache: HashMap::new(),
            components: ComponentCache::default(),
            reads: Vec::new(),
            hit_processed: false,
            simple: true,
        }
    }

    fn permutation(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Vec<usize>> {
        permutation_of(self.config, ctx, id)
    }

    /// `visit` (`:457-485`): nested visits are deferred and run from a stack,
    /// each node's in the order `_visit` asked for them.
    fn visit(
        &mut self,
        ctx: &LayoutCtx<'_>,
        start: NodeId,
        subgraph: &mut Subgraph,
        processed: &HashSet<NodeId>,
    ) -> bool {
        self.reads.clear();
        self.hit_processed = false;
        self.simple = true;
        let mut pending = vec![Pending {
            node: start,
            end: subgraph.end_permute.clone(),
            start: subgraph.start_permute.clone(),
        }];
        let mut deferred = Vec::new();
        while let Some(item) = pending.pop() {
            if !self.visit_one(ctx, item, subgraph, processed, &mut deferred) {
                return false;
            }
            pending.extend(deferred.drain(..).rev());
        }
        true
    }

    /// `_visit` (`:487-650`), with each nested `visit` pushed onto `deferred`.
    fn visit_one(
        &mut self,
        ctx: &LayoutCtx<'_>,
        item: Pending,
        subgraph: &mut Subgraph,
        processed: &HashSet<NodeId>,
        deferred: &mut Vec<Pending>,
    ) -> bool {
        let Pending {
            node,
            end: current_end,
            start: current_start,
        } = item;
        #[cfg(test)]
        SEARCH_STEPS.with(|steps| steps.set(steps.get() + 1));
        if subgraph.nodes.contains(&node) {
            return true;
        }
        self.note_reads(ctx, node);
        if processed.contains(&node) {
            self.hit_processed = true;
            return false;
        }
        let triple = self.interleave_triple(ctx, node);
        if triple.is_none() && !self.is_node_permutable(ctx, node) {
            return false;
        }
        let squeeze_view = triple.is_none() && self.is_squeeze_unsqueeze_view(ctx, node);
        if triple.is_some() {
            let Some(input) = first_arg_node(ctx.node(node)) else {
                return false;
            };
            let Some(in_shape) = concrete_shape(ctx, input) else {
                return false;
            };
            if current_start.len() != in_shape.len() {
                return false;
            }
        } else if !squeeze_view {
            // A broadcast can change rank under a permutation (`:518-527`).
            if let Some(shape) = ctx.graph().node_shape(node) {
                if shape.len() != current_start.len() {
                    return false;
                }
            }
        }
        if triple.is_some() || squeeze_view || numel(ctx, node) == Some(1) {
            self.simple = false;
        }
        subgraph.nodes.insert(node);
        subgraph.node_end_permute.insert(node, current_end.clone());
        subgraph
            .node_start_permute
            .insert(node, current_start.clone());

        let mut users_source = node;
        if let Some(triple) = triple {
            // `_absorb_interleave` (`:652-670`).
            if processed.contains(&triple.expand) || processed.contains(&triple.view) {
                self.hit_processed = true;
                return false;
            }
            for interior in [triple.expand, triple.view] {
                subgraph.nodes.insert(interior);
                subgraph
                    .node_end_permute
                    .insert(interior, current_end.clone());
                subgraph
                    .node_start_permute
                    .insert(interior, current_start.clone());
            }
            subgraph.interleaves.push((node, triple));
            users_source = triple.view;
        }

        let (downstream_end, downstream_start) = if squeeze_view {
            let Some(adapted) = self.adapt_permute_across_view(ctx, &current_start, node) else {
                return false;
            };
            let adapted: Perm = adapted.into();
            (Perm::from(inverse(&adapted)), adapted)
        } else {
            (current_end.clone(), current_start.clone())
        };

        // Downstream (`:565-607`).
        let single_user = live_user_count(ctx, node) == 1;
        for user in users(ctx, users_source) {
            self.reads.push(user);
            let user_node = ctx.node(user);
            if is_permute_target(&user_node.target) {
                let user_perm = self.permutation(ctx, user);
                let boundary_shape = if squeeze_view && single_user {
                    self.singleton_view_boundary_shape(ctx, node, &current_start, user)
                } else {
                    None
                };
                if let Some(shape) = boundary_shape {
                    subgraph.view_shape_overrides.insert(node, shape);
                    subgraph.edges_out.insert((users_source, user));
                } else if user_perm.as_deref() == Some(&*downstream_end) {
                    subgraph.edges_out.insert((users_source, user));
                } else {
                    // A non-matching permute absorbs the start permute.
                    let Some(user_perm) =
                        user_perm.filter(|perm| perm.len() == downstream_start.len())
                    else {
                        return false;
                    };
                    let composed = user_perm.iter().map(|&d| downstream_start[d]).collect();
                    subgraph.add_edge_out_to_update(users_source, user, composed);
                }
            } else if user_node.op == NodeOp::Output {
                return false;
            } else if self.is_permutation_sink_view(ctx, user) {
                continue;
            } else {
                deferred.push(Pending {
                    node: user,
                    end: downstream_end.clone(),
                    start: downstream_start.clone(),
                });
            }
        }

        // Upstream (`:609-648`).
        for input in all_input_nodes(ctx.node(node)) {
            self.reads.push(input);
            let input_node = ctx.node(input);
            if VIEW_OPS.contains(&input_node.target.as_str()) {
                // The sink check reads the view's input value.
                self.reads.extend(first_arg_node(input_node));
            }
            if is_permute_target(&input_node.target) {
                if self.permutation(ctx, input).as_deref() != Some(&*current_start) {
                    return false;
                }
                subgraph.edges_in.insert((input, node));
            } else if numel(ctx, input) == Some(1) {
                // Layout-invariant: it broadcasts the same under any permutation.
                continue;
            } else if subgraph.nodes.contains(&input) {
                continue;
            } else if self.is_permutation_sink_view(ctx, input) {
                let remapped = remapped_sink_shape(ctx, input, &current_start);
                let Some(remapped) = remapped.filter(|_| live_user_count(ctx, input) == 1) else {
                    return false;
                };
                subgraph.view_shape_overrides.insert(input, remapped);
                subgraph.sink_edges_in.insert((input, node));
            } else if self.is_constant(ctx, input) {
                // A rank implies a val, so rc4's `const_rank < permute_rank and
                // val is None` branch (`:638-639`) cannot fire.
                let Some(const_rank) = ctx.graph().node_shape(input).map(<[i64]>::len) else {
                    return false;
                };
                if const_rank > current_end.len() {
                    return false;
                }
                subgraph.constant_edges_in.insert((input, node));
            } else {
                deferred.push(Pending {
                    node: input,
                    end: current_end.clone(),
                    start: current_start.clone(),
                });
            }
        }

        // rc4's TABLE reads a rank-1 table constant after its input
        // (`insert_table_ops.py:392-401`), and the constant check above rejects
        // it in a rank-0 region; the subclass drops the edge otherwise.
        if self.config.tosa && is_pending_table(ctx.node(node)) && current_end.is_empty() {
            return false;
        }
        true
    }

    /// Record what the predicates read about `id`: its args, meta and users,
    /// its inputs' values, and the expand and view an interleave match looks
    /// at below it.
    fn note_reads(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) {
        note_reads_of(ctx, id, &mut self.reads);
    }

    /// `is_node_permutable` (`:705-723`).
    fn is_node_permutable(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let node = ctx.node(id);
        if node.op != NodeOp::CallFunction {
            // rc4 compares a placeholder's name with the op targets.
            return false;
        }
        let target = node.target.as_str();
        if PAD_OPS.contains(&target) && !is_constant_pad(node) {
            return false;
        }
        if let Some(filter) = self.config.target_filter(target) {
            if !filter(ctx.graph(), id) {
                return false;
            }
        }
        if self.config.is_permutable_op(target) {
            if REDUCTIONS.contains(&target) {
                return get_bool_arg(node, "keepdim", 2, false);
            }
            return true;
        }
        if self.is_squeeze_unsqueeze_view(ctx, id) {
            return true;
        }
        is_pointwise(target)
    }

    /// `_is_squeeze_unsqueeze_view` and its check (`:177-195`).
    fn is_squeeze_unsqueeze_view(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        if let Some(&cached) = self.sq_unsq_cache.get(&id) {
            return cached;
        }
        let result = match view_shapes(ctx, id) {
            None => false,
            Some((in_shape, out_shape)) => {
                if out_shape.len() > in_shape.len() {
                    find_extra_ones(&out_shape, &in_shape).is_some()
                } else if in_shape.len() > out_shape.len() {
                    find_extra_ones(&in_shape, &out_shape).is_some()
                } else {
                    false
                }
            }
        };
        self.sq_unsq_cache.insert(id, result);
        result
    }

    /// `_is_permutation_sink_view` (`:197-215`): a reshape of a tensor with at
    /// most one non-unit dim walks the same elements under any layout.
    fn is_permutation_sink_view(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let node = ctx.node(id);
        if !VIEW_OPS.contains(&node.target.as_str()) {
            return false;
        }
        let input = first_arg_node(node)
            .unwrap_or_else(|| panic!("view {} reads no node (rc4 asserts)", node.name));
        let shape = ctx.graph().node_shape(input).unwrap_or_else(|| {
            panic!(
                "{} has no value (rc4 raises KeyError reading it)",
                ctx.node(input).name
            )
        });
        shape.iter().filter(|&&extent| extent != 1).count() <= 1
    }

    /// `_singleton_view_boundary_shape` (`:236-272`): compose a layout pair
    /// across a view that only inserts unit dims.
    fn singleton_view_boundary_shape(
        &self,
        ctx: &LayoutCtx<'_>,
        view: NodeId,
        start_permute: &[usize],
        end_permute_node: NodeId,
    ) -> Option<Vec<i64>> {
        let shapes = view_shapes(ctx, view);
        let end_dims = self.permutation(ctx, end_permute_node);
        let end_shape = concrete_shape(ctx, end_permute_node);
        let ((in_shape, out_shape), end_dims, end_shape) = (shapes?, end_dims?, end_shape?);
        if start_permute.len() != in_shape.len() || end_dims.len() != out_shape.len() {
            return None;
        }
        let inserted = find_extra_ones(&out_shape, &in_shape)?;
        let mut labels: Vec<Option<usize>> = start_permute.iter().copied().map(Some).collect();
        for index in inserted {
            labels.insert(index, None);
        }
        let restored: Vec<usize> = end_dims.iter().filter_map(|&index| labels[index]).collect();
        if !restored.iter().copied().eq(0..in_shape.len()) {
            return None;
        }
        let inverse = inverse(start_permute);
        let unpermuted_input: Vec<i64> = inverse.iter().map(|&index| in_shape[index]).collect();
        find_extra_ones(&end_shape, &unpermuted_input)?;
        Some(end_shape)
    }

    /// `_interleave_triple` and `_match_interleave_triple` (`:304-367`):
    /// `unsqueeze(dim + 1) -> expand_copy(scale) -> view_copy(merge)`.
    fn interleave_triple(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Triple> {
        if let Some(&cached) = self.interleave_cache.get(&id) {
            return cached;
        }
        let triple = match_interleave_triple(ctx, id);
        self.interleave_cache.insert(id, triple);
        triple
    }

    /// `_adapt_permute_across_view` (`:369-416`): carry a permutation from a
    /// squeeze/unsqueeze view's input rank to its output rank.
    fn adapt_permute_across_view(
        &self,
        ctx: &LayoutCtx<'_>,
        permute: &[usize],
        id: NodeId,
    ) -> Option<Vec<usize>> {
        let (in_shape, out_shape) = view_shapes(ctx, id)?;
        if permute.len() != in_shape.len() {
            return None;
        }
        if out_shape.len() > in_shape.len() {
            let positions = find_extra_ones(&out_shape, &in_shape)?;
            let mut new_perm = permute.to_vec();
            for index in positions {
                for x in &mut new_perm {
                    if *x >= index {
                        *x += 1;
                    }
                }
                new_perm.insert(index, index);
            }
            return Some(new_perm);
        }
        if in_shape.len() > out_shape.len() {
            let positions = find_extra_ones(&in_shape, &out_shape)?;
            if positions.last().is_some_and(|&last| last >= permute.len()) {
                return None;
            }
            let squeezed: HashSet<usize> = positions.iter().map(|&p| permute[p]).collect();
            return Some(
                permute
                    .iter()
                    .filter(|x| !squeezed.contains(x))
                    .map(|&x| x - squeezed.iter().filter(|&&v| v < x).count())
                    .collect(),
            );
        }
        None
    }

    /// `_is_constant` (`:672-688`), with the subclass's graph-signature
    /// constants (`remove_permutes_around_elementwise_tosa_ops.py:40-43`),
    /// which RETAB marks with tensor data.
    fn is_constant(&self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        let node = ctx.node(id);
        match node.op {
            NodeOp::GetAttr => true,
            NodeOp::Placeholder => {
                ["b_", "p_", "c_"]
                    .iter()
                    .any(|prefix| node.target.starts_with(prefix))
                    || (self.config.tosa && node.tensor.is_some())
            }
            NodeOp::CallFunction => node.target == FULL,
            _ => false,
        }
    }

    /// After a search over simple nodes failed at `start`, cache the failure
    /// for its whole component if the component is closed (see
    /// [`ComponentCache`]), with what a search over the component reads.
    fn cache_component_failure(
        &mut self,
        ctx: &LayoutCtx<'_>,
        start: NodeId,
        perm: &Perm,
        search_reads: &[NodeId],
    ) {
        if let Some((index, mut reads)) = self.closed_simple_component(ctx, start) {
            reads.extend_from_slice(search_reads);
            self.components
                .failures
                .insert((index, perm.clone()), stamp(ctx, reads));
        }
    }

    /// The closed simple component holding `start` and the nodes a search
    /// over it reads, or `None` when a search from `start` can reach a node
    /// that is permutable but not simple.
    fn closed_simple_component(
        &mut self,
        ctx: &LayoutCtx<'_>,
        start: NodeId,
    ) -> Option<(usize, Vec<NodeId>)> {
        if let Some(&index) = self.components.component_index.get(&start) {
            return Some((index, self.components.reads[index].clone()));
        }
        if self.components.open.contains(&start) {
            return None;
        }
        let mut members = vec![start];
        let mut seen: HashSet<NodeId> = HashSet::from([start]);
        let mut reads = Vec::new();
        let mut index = 0;
        while index < members.len() {
            let node = members[index];
            index += 1;
            if !self.is_simple(ctx, node) {
                // Every member reaches `node` back, so none is in a closed
                // simple component either.
                self.components.open.extend(members);
                return None;
            }
            note_reads_of(ctx, node, &mut reads);
            let mut neighbours = Vec::new();
            for user in users(ctx, node) {
                let user_node = ctx.node(user);
                if is_permute_target(&user_node.target)
                    || user_node.op == NodeOp::Output
                    || self.is_permutation_sink_view(ctx, user)
                {
                    continue;
                }
                neighbours.push(user);
            }
            for input in all_input_nodes(ctx.node(node)) {
                let input_node = ctx.node(input);
                if VIEW_OPS.contains(&input_node.target.as_str()) {
                    reads.extend(first_arg_node(input_node));
                }
                if is_permute_target(&input_node.target)
                    || numel(ctx, input) == Some(1)
                    || self.is_permutation_sink_view(ctx, input)
                    || self.is_constant(ctx, input)
                {
                    continue;
                }
                neighbours.push(input);
            }
            for neighbour in neighbours {
                if seen.contains(&neighbour) {
                    continue;
                }
                // A neighbour that fails the permutability check fails every
                // search that reaches it, so it does not break the closure.
                if self.interleave_triple(ctx, neighbour).is_none()
                    && !self.is_node_permutable(ctx, neighbour)
                {
                    note_reads_of(ctx, neighbour, &mut reads);
                    continue;
                }
                seen.insert(neighbour);
                members.push(neighbour);
            }
        }
        let index = self.components.reads.len();
        for &member in &members {
            self.components.component_index.insert(member, index);
        }
        self.components.reads.push(reads.clone());
        Some((index, reads))
    }

    /// A permutable node whose searches keep their permutation and that every
    /// neighbour's search reaches back: not a squeeze/unsqueeze view, not an
    /// interleave head and not numel-1.
    fn is_simple(&mut self, ctx: &LayoutCtx<'_>, id: NodeId) -> bool {
        self.interleave_triple(ctx, id).is_none()
            && self.is_node_permutable(ctx, id)
            && !self.is_squeeze_unsqueeze_view(ctx, id)
            && numel(ctx, id) != Some(1)
    }
}

/// Failed searches over closed simple components, for one call.
///
/// A node is simple when it is permutable, not a squeeze/unsqueeze view, not an
/// interleave head and not numel-1. A search through simple nodes keeps its
/// permutation, and between simple nodes every edge it follows is followed
/// back: a user reaches its input unless the input is a permute, numel-1, a
/// sink view or a constant, none of which a simple node is, and an input
/// reaches its user unless the user is a permute, the output or a sink view,
/// none of which a simple node is either. So in a component of simple nodes
/// with no permutable non-simple neighbour, a search from any member reaches
/// every member with the same permutation, and the checks it makes at each
/// member depend only on that permutation. If one search fails there, every
/// search from a member with the same permutation fails (nodes claimed by an
/// earlier region only add failures).
#[derive(Default)]
struct ComponentCache {
    component_index: HashMap<NodeId, usize>,
    /// What a search over each component reads.
    reads: Vec<Vec<NodeId>>,
    /// Failed (component, permutation) pairs, with the reads that prove it.
    failures: HashMap<(usize, Perm), ReadSet>,
    /// Nodes whose component reaches a permutable node that is not simple.
    open: HashSet<NodeId>,
}

impl ComponentCache {
    /// The proof that a search from `start` with `perm` fails, if one is known.
    fn known_failure(&self, start: NodeId, perm: &Perm) -> Option<ReadSet> {
        let &index = self.component_index.get(&start)?;
        self.failures.get(&(index, perm.clone())).cloned()
    }
}

/// The nodes a failed search read, each with its version stamp at the time,
/// sorted by node. Searches proven to fail by the same reads share one.
type ReadSet = Rc<[(NodeId, u64)]>;

fn stamp(ctx: &LayoutCtx<'_>, reads: impl IntoIterator<Item = NodeId>) -> ReadSet {
    let mut stamped: Vec<(NodeId, u64)> = reads.into_iter().map(|id| (id, ctx.version(id))).collect();
    stamped.sort_unstable();
    stamped.dedup();
    stamped.into()
}

/// Push what the predicates read about `id`: its args, meta and users, its
/// inputs' values, and the expand and view an interleave match looks at
/// below it.
fn note_reads_of(ctx: &LayoutCtx<'_>, id: NodeId, reads: &mut Vec<NodeId>) {
    reads.push(id);
    reads.extend(all_input_nodes(ctx.node(id)));
    let users = ctx.graph().live_user_ids(id);
    if let [expand] = users[..] {
        if ctx.node(expand).target == EXPAND_COPY {
            reads.extend(ctx.graph().live_user_ids(expand));
        }
    }
    reads.extend(users);
}

/// Failed searches remembered across calls on one `LayoutCtx`.
///
/// A search from `(permute, user)` that failed without meeting a node claimed
/// by an earlier region fails again as long as no node it read has changed:
/// claimed nodes can only add failures. The ctx bumps a node's version when its
/// args, meta or users change, so a read set of version stamps detects any
/// change that could alter the search.
#[derive(Default)]
pub(crate) struct RemovePermutesMemo {
    failed: HashMap<(NodeId, NodeId), ReadSet>,
}

impl RemovePermutesMemo {
    /// Whether the search from `(permute, user)` is known to fail. A read set
    /// shared by several searches is checked once per call: `current` holds
    /// each one met so far (keeping it alive, so its address stays unique).
    fn known_failure(
        &self,
        ctx: &LayoutCtx<'_>,
        permute: NodeId,
        user: NodeId,
        current: &mut HashMap<usize, (ReadSet, bool)>,
    ) -> bool {
        let Some(reads) = self.failed.get(&(permute, user)) else {
            return false;
        };
        let key = Rc::as_ptr(reads).cast::<(NodeId, u64)>() as usize;
        current
            .entry(key)
            .or_insert_with(|| {
                let fresh = reads
                    .iter()
                    .all(|&(id, version)| ctx.version(id) == version);
                (reads.clone(), fresh)
            })
            .1
    }

    fn record(&mut self, permute: NodeId, user: NodeId, reads: ReadSet) {
        self.failed.insert((permute, user), reads);
    }

    /// How many failed searches are remembered.
    pub(crate) fn len(&self) -> usize {
        self.failed.len()
    }
}

/// `_match_interleave_triple` (`:312-367`).
fn match_interleave_triple(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Triple> {
    let pos = inserted_unit_dim(ctx, id)?;
    if pos == 0 || live_user_count(ctx, id) != 1 {
        return None;
    }
    let dim = pos - 1;
    let expand = ctx.graph().live_user_ids(id)[0];
    if ctx.node(expand).target != EXPAND_COPY || live_user_count(ctx, expand) != 1 {
        return None;
    }
    let unsq_shape = concrete_shape(ctx, id)?;
    let size = get_arg(ctx.node(expand), "size", 1)?;
    let IrArg::List(items) = size else {
        return None;
    };
    if items.len() != unsq_shape.len() {
        return None;
    }
    let size: Vec<i64> = items.iter().map(IrArg::as_int).collect::<Option<_>>()?;
    if size
        .iter()
        .enumerate()
        .any(|(k, &s)| k != pos && s != -1 && s != unsq_shape[k])
    {
        return None;
    }
    let scale = size[pos];
    if scale < 1 {
        return None;
    }
    let view = ctx.graph().live_user_ids(expand)[0];
    if !VIEW_OPS.contains(&ctx.node(view).target.as_str()) {
        return None;
    }
    let input = first_arg_node(ctx.node(id))?;
    let mut merged = concrete_shape(ctx, input)?;
    merged[dim] *= scale;
    if concrete_shape(ctx, view)? != merged {
        return None;
    }
    Some(Triple {
        dim,
        scale,
        expand,
        view,
    })
}

/// `_inserted_unit_dim` (`:274-302`): where an unsqueeze, or a view that only
/// adds one unit dim, puts the new dim.
fn inserted_unit_dim(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<usize> {
    let node = ctx.node(id);
    let is_unsqueeze = node.target == UNSQUEEZE_COPY;
    if !is_unsqueeze && !VIEW_OPS.contains(&node.target.as_str()) {
        return None;
    }
    let input = first_arg_node(node)?;
    let in_shape = concrete_shape(ctx, input)?;
    let out_shape = concrete_shape(ctx, id)?;
    if out_shape.len() != in_shape.len() + 1 {
        return None;
    }
    if is_unsqueeze {
        let dim = get_int_arg(node, "dim", 1, None);
        let pos = if dim >= 0 {
            dim
        } else {
            dim + out_shape.len() as i64
        };
        if pos < 0 || pos >= out_shape.len() as i64 || out_shape[pos as usize] != 1 {
            return None;
        }
        return Some(pos as usize);
    }
    match find_extra_ones(&out_shape, &in_shape) {
        Some(positions) if positions.len() == 1 => Some(positions[0]),
        _ => None,
    }
}

/// `_find_extra_ones` (`:159-175`): the positions in `longer` whose removal,
/// all of them unit dims, leaves `shorter`.
fn find_extra_ones(longer: &[i64], shorter: &[i64]) -> Option<Vec<usize>> {
    let mut extra = Vec::new();
    let mut j = 0;
    for (i, &dim) in longer.iter().enumerate() {
        if j < shorter.len() && dim == shorter[j] {
            j += 1;
            continue;
        }
        if dim != 1 {
            return None;
        }
        extra.push(i);
    }
    (j == shorter.len()).then_some(extra)
}

/// `_remapped_sink_shape` (`:217-234`): a sink's shape in the layout before
/// the start permute.
fn remapped_sink_shape(ctx: &LayoutCtx<'_>, sink: NodeId, start_permute: &[usize]) -> Option<Vec<i64>> {
    let out_shape = concrete_shape(ctx, sink)?;
    if out_shape.len() != start_permute.len() {
        return None;
    }
    if out_shape.iter().filter(|&&extent| extent != 1).count() > 1 {
        return None;
    }
    Some(
        inverse(start_permute)
            .iter()
            .map(|&index| out_shape[index])
            .collect(),
    )
}

/// `_view_shapes` (`:141-152`).
fn view_shapes(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<(Vec<i64>, Vec<i64>)> {
    let node = ctx.node(id);
    if !VIEW_OPS.contains(&node.target.as_str()) {
        return None;
    }
    let input = first_arg_node(node)?;
    Some((concrete_shape(ctx, input)?, concrete_shape(ctx, id)?))
}

/// `_is_constant_pad` (`:725-741`).
fn is_constant_pad(node: &IrNode) -> bool {
    if node.args.len() < 2 {
        return false;
    }
    let pad_len = match &node.args[1] {
        IrArg::List(items) => items.len(),
        // FX import's spelling of `[]`.
        IrArg::None => 0,
        _ => return false,
    };
    if !pad_len.is_multiple_of(2) {
        return false;
    }
    if node.target == PAD_DEFAULT {
        let mode = node.args.get(2).or_else(|| node.kwargs.get("mode"));
        return match mode {
            None | Some(IrArg::None) => true,
            Some(IrArg::Str(mode)) => mode == "constant",
            Some(_) => false,
        };
    }
    true
}

/// `self.get_permutation(node)`, through the test override of rc4's
/// `RankMismatchPass`.
fn permutation_of(config: &RemovePermutesConfig, ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Vec<usize>> {
    #[cfg(test)]
    if let Some((name, perm)) = &config.permutation_override {
        if ctx.node(id).name == *name {
            return Some(perm.clone());
        }
    }
    #[cfg(not(test))]
    let _ = config;
    get_permutation(ctx.node(id))
}

/// `get_permutation` (`:1041-1059`): the normalized permutation, or `None`
/// when it is not one. rc4 raises on a `dims` that is not a list; RETAB's IR
/// never builds one, and it reads as no permutation here.
fn get_permutation(node: &IrNode) -> Option<Vec<usize>> {
    let raw = if node.args.len() >= 2 {
        int_list(&node.args[1])?
    } else {
        let dims = node.kwargs.get("dims").or_else(|| node.kwargs.get("dim"))?;
        int_list(dims)?
    };
    let rank = raw.len() as i64;
    let normalized: Vec<i64> = raw
        .iter()
        .map(|&d| if d < 0 { d + rank } else { d })
        .collect();
    if normalized.iter().any(|&d| d < 0 || d >= rank) {
        return None;
    }
    let mut sorted = normalized.clone();
    sorted.sort_unstable();
    if !sorted.iter().copied().eq(0..rank) {
        return None;
    }
    Some(normalized.iter().map(|&d| d as usize).collect())
}

/// `is_permute_copy` (`backends/transforms/channels_last_layout.py:21-22`).
fn is_permute_copy(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction && is_permute_target(&node.target)
}

/// `node.target in PERMUTE_COPY_TARGETS`.
fn is_permute_target(target: &str) -> bool {
    PERMUTE_COPY_TARGETS.contains(&target)
}

fn inverse(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (index, &dim) in perm.iter().enumerate() {
        inverse[dim] = index;
    }
    inverse
}

fn first_arg_node(node: &IrNode) -> Option<NodeId> {
    node.args.first().and_then(IrArg::as_node)
}

/// `_concrete_shape` (`:130-139`); RETAB shapes are always concrete.
fn concrete_shape(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Vec<i64>> {
    ctx.graph().node_shape_vec(id)
}

fn numel(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<i64> {
    ctx.graph()
        .node_shape(id)
        .map(|shape| shape.iter().product())
}

fn live_user_count(ctx: &LayoutCtx<'_>, id: NodeId) -> usize {
    ctx.graph().live_user_ids(id).len()
}

/// `node.users`, in graph order: torch.fx keeps users in insertion order,
/// which is graph order after a retrace.
fn users(ctx: &LayoutCtx<'_>, id: NodeId) -> Vec<NodeId> {
    let mut users = ctx.graph().live_user_ids(id);
    ctx.order().sort(&mut users);
    users
}

/// `node.all_input_nodes`: positional args, then kwargs by name, each node
/// once.
fn all_input_nodes(node: &IrNode) -> Vec<NodeId> {
    let mut ids = Vec::new();
    for arg in &node.args {
        arg.collect_node_ids(&mut ids);
    }
    let mut kwargs: Vec<(&String, &IrArg)> = node.kwargs.iter().collect();
    kwargs.sort_by(|lhs, rhs| lhs.0.cmp(rhs.0));
    for (_, arg) in kwargs {
        arg.collect_node_ids(&mut ids);
    }
    let mut seen = HashSet::new();
    ids.retain(|&id| seen.insert(id));
    ids
}

/// `get_arg` (`backends/transforms/permute_pass_utils.py:87-116`): a keyword
/// wins, then the positional arg at the schema `index`; `None` stands for the
/// schema default.
fn get_arg<'a>(node: &'a IrNode, name: &str, index: usize) -> Option<&'a IrArg> {
    node.kwargs.get(name).or_else(|| node.args.get(index))
}

/// `get_arg(node, name, int)`, which raises when the value is not an int.
fn get_int_arg(node: &IrNode, name: &str, index: usize, default: Option<i64>) -> i64 {
    match get_arg(node, name, index) {
        Some(IrArg::Int(value)) => *value,
        None => default.unwrap_or_else(|| {
            panic!("get_arg: {} has no {name} (rc4 raises)", node.name)
        }),
        Some(other) => panic!(
            "get_arg: expected <class 'int'> for '{name}' of {}, got {other:?}",
            node.name
        ),
    }
}

/// `get_arg(node, name, bool)`, which raises when the value is not a bool.
fn get_bool_arg(node: &IrNode, name: &str, index: usize, default: bool) -> bool {
    match get_arg(node, name, index) {
        Some(IrArg::Bool(value)) => *value,
        None => default,
        Some(other) => panic!(
            "get_arg: expected <class 'bool'> for '{name}' of {}, got {other:?}",
            node.name
        ),
    }
}

/// `set_arg` (`backends/transforms/permute_pass_utils.py:119-143`): update the
/// keyword if the node passes one, else the positional arg at the schema
/// `index` if the node has it, else add the keyword.
fn set_arg(ctx: &mut LayoutCtx<'_>, id: NodeId, name: &str, index: usize, value: IrArg) {
    let node = ctx.node(id);
    if node.kwargs.contains_key(name) || index >= node.args.len() {
        ctx.set_kwarg(id, name, value);
    } else {
        ctx.set_arg(id, index, value);
    }
}

/// `Node.update_arg(index, value)`, which raises past the last positional arg.
fn update_arg(ctx: &mut LayoutCtx<'_>, id: NodeId, index: usize, value: IrArg) {
    let node = ctx.node(id);
    if index >= node.args.len() {
        panic!(
            "update_arg: {} has no positional arg {index} (rc4 raises IndexError)",
            node.name
        );
    }
    ctx.set_arg(id, index, value);
}

/// Python's `perm[index]`, which counts a negative index from the end and
/// raises past either end.
fn py_index(perm: &[usize], index: i64) -> usize {
    let len = perm.len() as i64;
    let wrapped = if index < 0 { index + len } else { index };
    if wrapped < 0 || wrapped >= len {
        panic!("list index {index} out of range for {perm:?} (rc4 raises IndexError)");
    }
    perm[wrapped as usize]
}

fn usize_list_arg(values: &[usize]) -> IrArg {
    IrArg::List(values.iter().map(|&v| IrArg::Int(v as i64)).collect())
}

// ---------------------------------------------------------------------------
// Rewriting
// ---------------------------------------------------------------------------

/// What the rewrites of one call changed, for the dead-code sweep and the
/// value refresh.
#[derive(Default)]
struct Rewrite {
    /// rc4's `_bypassed_permutes` (`:123`).
    bypassed: HashSet<NodeId>,
    /// Nodes whose value must be re-derived: region nodes, remapped sinks, new
    /// nodes and rewritten permutes.
    seeds: Vec<NodeId>,
    /// `(user, input)` pairs where `user` now reads `input` in place of a
    /// node whose value was this.
    seen_values: HashMap<(NodeId, NodeId), Option<TensorFacts>>,
    /// Nodes that lost users: the bypassed permutes.
    unused: Vec<NodeId>,
}

/// `permute_subgraph` (`:753-885`), after the subclass drops TABLE's constant
/// edges (`remove_permutes_around_elementwise_tosa_ops.py:45-54`).
fn permute_subgraph(
    ctx: &mut LayoutCtx<'_>,
    config: &RemovePermutesConfig,
    subgraph: &mut Subgraph,
    rewrite: &mut Rewrite,
) -> bool {
    if config.tosa {
        subgraph
            .constant_edges_in
            .retain(|&(_, user)| !is_tosa_table_target(&ctx.node(user).target));
    }
    if !subgraph_edges_are_current(ctx, config, subgraph, &rewrite.bypassed) {
        return false;
    }
    // Folding an end permute pays only if some permute goes away.
    if !subgraph.edges_out_to_update.is_empty() && !removes_a_permute(ctx, subgraph) {
        return false;
    }

    // The interleave triples' ranks (`:764-777`).
    let mut interleave_nodes = HashSet::new();
    for &(head, triple) in &subgraph.interleaves {
        interleave_nodes.extend([head, triple.expand, triple.view]);
        let perm = subgraph.start_of(head);
        let Some(input) = first_arg_node(ctx.node(head)) else {
            return false;
        };
        match concrete_shape(ctx, input) {
            Some(shape) if shape.len() == perm.len() => {}
            _ => return false,
        }
    }
    // Each view's permutation must match its input's rank (`:779-792`).
    for &node in subgraph.nodes.iter() {
        if interleave_nodes.contains(&node) || !VIEW_OPS.contains(&ctx.node(node).target.as_str())
        {
            continue;
        }
        let perm = subgraph.start_of(node);
        if let Some(rank) = first_arg_node(ctx.node(node))
            .and_then(|input| ctx.graph().node_shape(input))
            .map(<[i64]>::len)
        {
            if perm.len() != rank {
                return false;
            }
        }
    }

    // Dim arguments first, while the inputs and values are still the old ones
    // (`:794-819`).
    let nodes: Vec<NodeId> = subgraph.nodes.iter().copied().collect();
    for &node in &nodes {
        if interleave_nodes.contains(&node) {
            continue;
        }
        let perm = subgraph.start_of(node);
        let target = ctx.node(node).target.clone();
        if target == CAT || target == SLICE_COPY {
            update_dim(ctx, node, &perm);
        } else if REDUCTIONS.contains(&target.as_str()) {
            update_reduction_dim(ctx, node, &perm);
        } else if PAD_OPS.contains(&target.as_str()) {
            update_pad(ctx, node, &perm);
        } else if VIEW_OPS.contains(&target.as_str()) {
            match subgraph.view_shape_overrides.get(&node) {
                Some(shape) => update_arg(ctx, node, 1, i64_list_arg(shape)),
                None => update_view_copy(ctx, node, &perm),
            }
        }
    }
    rewrite.seeds.extend(nodes.iter().copied());

    for &(sink, _) in subgraph.sink_edges_in.iter() {
        let shape = subgraph.view_shape_overrides[&sink].clone();
        update_arg(ctx, sink, 1, i64_list_arg(&shape));
        rewrite.seeds.push(sink);
    }

    for &(head, triple) in &subgraph.interleaves {
        update_interleave(ctx, head, triple, &subgraph.start_of(head));
    }

    // Skip the incoming permutes (`:831-838`), in graph order of (permute,
    // user): `replace_input_with` rewrites every use, so when one bypassed
    // permute feeds another into the same node, the order decides what the
    // node reads.
    let edges_in = in_graph_order(ctx, subgraph.edges_in.iter().copied());
    for (input, out) in edges_in {
        rewrite.bypassed.insert(input);
        let source = first_arg_node(ctx.node(input))
            .or_else(|| ctx.node(input).kwargs.get("input").and_then(IrArg::as_node))
            .unwrap_or_else(|| panic!("permute {} reads no node", ctx.node(input).name));
        ctx.replace_input_with(out, input, source);
        rewrite.unused.push(input);
    }

    // Compensate the constant inputs (`:840-873`), in graph order of
    // (constant, user).
    let constant_edges = in_graph_order(ctx, subgraph.constant_edges_in.iter().copied());
    for (constant, user) in constant_edges {
        let const_shape = ctx.graph().node_shape_vec(constant);
        let node_end_perm = subgraph.end_of(user);
        let permute_rank = node_end_perm.len();
        let new_node = match const_shape {
            Some(shape) if shape.len() == permute_rank => {
                let name = format!("{}_permuted", ctx.node(constant).name);
                ctx.inserting_after(constant, |ctx| {
                    ctx.call_function(
                        name,
                        PERMUTE_COPY,
                        vec![IrArg::Node(constant), usize_list_arg(&node_end_perm)],
                        NodeMeta::default(),
                    )
                })
            }
            Some(shape) if shape.len() < permute_rank => {
                let mut padded = vec![1; permute_rank - shape.len()];
                padded.extend_from_slice(&shape);
                let target_shape: Vec<i64> = node_end_perm.iter().map(|&d| padded[d]).collect();
                let target_shape = target_shape[permute_rank - shape.len()..].to_vec();
                let name = format!("{}_viewed", ctx.node(constant).name);
                ctx.inserting_after(constant, |ctx| {
                    ctx.call_function(
                        name,
                        VIEW_COPY,
                        vec![IrArg::Node(constant), i64_list_arg(&target_shape)],
                        NodeMeta::default(),
                    )
                })
            }
            _ => continue,
        };
        ctx.replace_input_with(user, constant, new_node);
        rewrite.seeds.push(new_node);
    }

    // Skip the outgoing permutes (`:875-878`).
    for &(input, out) in subgraph.edges_out.iter() {
        let value = ctx.node(out).meta.tensor_facts().cloned();
        for user in ctx.replace_all_uses_with(out, input) {
            rewrite.seen_values.insert((user, input), value.clone());
        }
        rewrite.unused.push(out);
    }

    // Rewrite the outgoing permutes that stay (`:880-883`).
    for (_, out, perm) in &subgraph.edges_out_to_update {
        set_arg(ctx, *out, "dims", 1, usize_list_arg(perm));
        rewrite.seeds.push(*out);
    }
    true
}

/// `edges` sorted by the graph positions of their nodes: the order the replay
/// generator pins rc4's address-ordered edge sets to (see the module docs).
fn in_graph_order(
    ctx: &LayoutCtx<'_>,
    edges: impl Iterator<Item = (NodeId, NodeId)>,
) -> Vec<(NodeId, NodeId)> {
    let key = |id| ctx.order().key(id).expect("region edges join placed nodes");
    let mut keyed: Vec<(u64, u64, NodeId, NodeId)> = edges
        .map(|(from, to)| (key(from), key(to), from, to))
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, _, from, to)| (from, to)).collect()
}

/// `_subgraph_edges_are_current` (`:887-929`): whether an earlier rewrite
/// invalidated this candidate.
fn subgraph_edges_are_current(
    ctx: &LayoutCtx<'_>,
    config: &RemovePermutesConfig,
    subgraph: &Subgraph,
    bypassed: &HashSet<NodeId>,
) -> bool {
    for &(input, out) in subgraph.edges_in.iter() {
        if !is_permute_target(&ctx.node(input).target)
            || !all_input_nodes(ctx.node(out)).contains(&input)
            || permutation_of(config, ctx, input).as_deref() != Some(&*subgraph.start_of(out))
        {
            return false;
        }
    }
    let outgoing = subgraph.edges_out.iter().copied().chain(
        subgraph
            .edges_out_to_update
            .iter()
            .map(|(input, out, _)| (*input, *out)),
    );
    for (input, out) in outgoing {
        if !is_permute_target(&ctx.node(out).target)
            || !ctx.graph().live_user_ids(input).contains(&out)
            || bypassed.contains(&out)
        {
            return false;
        }
    }
    for &(constant, user) in subgraph.constant_edges_in.iter() {
        if !all_input_nodes(ctx.node(user)).contains(&constant) {
            return false;
        }
    }
    for &(sink, user) in subgraph.sink_edges_in.iter() {
        if !all_input_nodes(ctx.node(user)).contains(&sink) || live_user_count(ctx, sink) != 1 {
            return false;
        }
    }
    for &(head, triple) in &subgraph.interleaves {
        let head_users = ctx.graph().live_user_ids(head);
        let expand_users = ctx.graph().live_user_ids(triple.expand);
        if head_users.len() != 1
            || expand_users.len() != 1
            || !head_users.contains(&triple.expand)
            || !expand_users.contains(&triple.view)
        {
            return false;
        }
    }
    true
}

/// `_removes_a_permute` (`:743-751`).
fn removes_a_permute(ctx: &LayoutCtx<'_>, subgraph: &Subgraph) -> bool {
    if !subgraph.edges_out.is_empty() {
        return true;
    }
    subgraph.edges_in.iter().any(|&(permute, _)| {
        ctx.graph()
            .live_user_ids(permute)
            .into_iter()
            .all(|user| subgraph.edges_in.contains(&(permute, user)))
    })
}

/// `update_cat` and `update_slice_copy` (`:965-967`, `:983-985`).
fn update_dim(ctx: &mut LayoutCtx<'_>, id: NodeId, perm: &[usize]) {
    let dim = get_int_arg(ctx.node(id), "dim", 1, Some(0));
    set_arg(ctx, id, "dim", 1, IrArg::Int(py_index(perm, dim) as i64));
}

/// `update_reduction_dim` (`:969-981`). FX import spells `[]` as `None`, so a
/// `None` dim reads as the empty list.
fn update_reduction_dim(ctx: &mut LayoutCtx<'_>, id: NodeId, perm: &[usize]) {
    let node = ctx.node(id);
    let rank = perm.len() as i64;
    let remap = |dim: i64| {
        if rank == 0 {
            panic!("integer modulo by zero remapping {} (rc4 raises)", node.name);
        }
        perm[dim.rem_euclid(rank) as usize] as i64
    };
    let value = match get_arg(node, "dim", 1) {
        Some(IrArg::Int(dim)) => IrArg::Int(remap(*dim)),
        None | Some(IrArg::None) => IrArg::List(Vec::new()),
        Some(arg) => {
            let dims = int_list(arg)
                .unwrap_or_else(|| panic!("reduction dims of {} are not ints", node.name));
            IrArg::List(dims.into_iter().map(|dim| IrArg::Int(remap(dim))).collect())
        }
    };
    set_arg(ctx, id, "dim", 1, value);
}

/// `update_pad` (`:987-1003`): move each dim's pad pair to the dim it lands
/// on, and drop trailing zero pairs.
fn update_pad(ctx: &mut LayoutCtx<'_>, id: NodeId, perm: &[usize]) {
    let node = ctx.node(id);
    let pad = int_list(&node.args[1])
        .unwrap_or_else(|| panic!("pad of {} is not a list of ints", node.name));
    let rank = perm.len();
    let mut pad_pairs = vec![[0i64; 2]; rank];
    for (pair_index, pair_start) in (0..pad.len()).step_by(2).enumerate() {
        let dim = rank as i64 - 1 - pair_index as i64;
        let slot = if dim < 0 { dim + rank as i64 } else { dim };
        if slot < 0 || slot >= rank as i64 {
            panic!("pad of {} is longer than its rank (rc4 raises IndexError)", node.name);
        }
        pad_pairs[slot as usize] = [pad[pair_start], pad[pair_start + 1]];
    }
    let inverse = inverse(perm);
    let mut remapped: Vec<i64> = Vec::with_capacity(2 * rank);
    for dim in (0..rank).rev() {
        remapped.extend_from_slice(&pad_pairs[inverse[dim]]);
    }
    while remapped.len() > 2 && remapped[remapped.len() - 2..] == [0, 0] {
        remapped.truncate(remapped.len() - 2);
    }
    update_arg(ctx, id, 1, i64_list_arg(&remapped));
}

/// `update_view_copy` (`:1005-1039`): a squeeze/unsqueeze view's shape in the
/// un-permuted layout.
fn update_view_copy(ctx: &mut LayoutCtx<'_>, id: NodeId, perm: &[usize]) {
    let Some((in_shape, out_shape)) = view_shapes(ctx, id) else {
        return;
    };
    let inverse = inverse(perm);
    let unpermuted: Vec<i64> = inverse.iter().map(|&index| in_shape[index]).collect();
    if out_shape.len() > in_shape.len() {
        let Some(positions) = find_extra_ones(&out_shape, &in_shape) else {
            return;
        };
        let mut new_shape = unpermuted;
        for index in positions {
            new_shape.insert(index, 1);
        }
        update_arg(ctx, id, 1, i64_list_arg(&new_shape));
    } else if in_shape.len() > out_shape.len() {
        let Some(positions) = find_extra_ones(&in_shape, &out_shape) else {
            return;
        };
        if positions.last().is_some_and(|&last| last >= perm.len()) {
            return;
        }
        let mut removed: Vec<usize> = positions.iter().map(|&p| perm[p]).collect();
        removed.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut new_shape = unpermuted;
        for index in removed {
            new_shape.remove(index);
        }
        update_arg(ctx, id, 1, i64_list_arg(&new_shape));
    }
}

/// `update_interleave` (`:931-963`): rebuild the triple's shape arguments at
/// the un-permuted dim.
fn update_interleave(ctx: &mut LayoutCtx<'_>, head: NodeId, triple: Triple, perm: &[usize]) {
    let input = first_arg_node(ctx.node(head)).expect("an interleave head reads a node");
    let in_shape = concrete_shape(ctx, input).expect("validated before the rewrite");
    let inverse = inverse(perm);
    let unpermuted: Vec<i64> = inverse.iter().map(|&index| in_shape[index]).collect();
    let target_dim = perm[triple.dim];

    if ctx.node(head).target == UNSQUEEZE_COPY {
        set_arg(ctx, head, "dim", 1, IrArg::Int(target_dim as i64 + 1));
    } else {
        let mut unsqueezed = unpermuted.clone();
        unsqueezed.insert(target_dim + 1, 1);
        set_arg(ctx, head, "size", 1, i64_list_arg(&unsqueezed));
    }
    let mut expand_size = unpermuted.clone();
    expand_size.insert(target_dim + 1, triple.scale);
    set_arg(ctx, triple.expand, "size", 1, i64_list_arg(&expand_size));
    let mut merged = unpermuted;
    merged[target_dim] *= triple.scale;
    set_arg(ctx, triple.view, "size", 1, i64_list_arg(&merged));
}

// ---------------------------------------------------------------------------
// Dead code and values
// ---------------------------------------------------------------------------

/// Erase the `candidates` nothing uses, and then whatever only they used.
fn eliminate_dead_from(ctx: &mut LayoutCtx<'_>, candidates: impl IntoIterator<Item = NodeId>) {
    let mut stack: Vec<NodeId> = candidates.into_iter().collect();
    while let Some(id) = stack.pop() {
        let node = ctx.node(id);
        if ctx.is_erased(id)
            || ctx.graph().has_live_users(id)
            || !matches!(node.op, NodeOp::CallFunction | NodeOp::GetAttr)
            || (node.op == NodeOp::CallFunction && is_impure_call(&node.target))
        {
            continue;
        }
        let inputs = all_input_nodes(node);
        ctx.erase_node(id);
        stack.extend(inputs);
    }
}

/// rc4's closing retrace (`:453`), for the values a rewrite can change.
///
/// Every value is re-derived from its operands as fake execution would, in
/// graph order: the seeds always, and any other node once the value of one of
/// its inputs, as that node sees it, has changed. A user rewired off a
/// bypassed permute saw the permute's value before. In a correct rewrite only
/// the seeds and the sink views below the region are recomputed.
fn refresh(ctx: &mut LayoutCtx<'_>, rewrite: &Rewrite) {
    let mut queue: BinaryHeap<Reverse<(u64, NodeId)>> = BinaryHeap::new();
    let mut queued: HashSet<NodeId> = HashSet::new();
    // Only calls have values to re-derive; the output node has none.
    let push = |ctx: &LayoutCtx<'_>,
                queue: &mut BinaryHeap<Reverse<(u64, NodeId)>>,
                queued: &mut HashSet<NodeId>,
                id: NodeId| {
        if !ctx.is_erased(id) && ctx.node(id).op == NodeOp::CallFunction && queued.insert(id) {
            let key = ctx.order().key(id).expect("live nodes are placed");
            queue.push(Reverse((key, id)));
        }
    };
    for &id in &rewrite.seeds {
        push(ctx, &mut queue, &mut queued, id);
    }
    let mut rewired_inputs: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &(user, input) in rewrite.seen_values.keys() {
        rewired_inputs.entry(input).or_default().push(user);
    }
    // A rewired input that is not itself refreshed keeps its value; its new
    // users must still compare it with what they saw.
    for (&input, rewired) in &rewired_inputs {
        if queued.contains(&input) || ctx.is_erased(input) {
            continue;
        }
        let value = ctx.node(input).meta.tensor_facts().cloned();
        for &user in rewired {
            if rewrite.seen_values[&(user, input)] != value {
                push(ctx, &mut queue, &mut queued, user);
            }
        }
    }
    while let Some(Reverse((_, id))) = queue.pop() {
        if ctx.is_erased(id) {
            continue;
        }
        let before = ctx.node(id).meta.tensor_facts().cloned();
        let meta = retraced_meta(ctx.graph(), id).unwrap_or_else(|error| {
            let node = ctx.node(id);
            panic!(
                "RemovePermutesAroundElementwiseOps: rc4's retrace fails at {} ({}): {error}",
                node.name, node.target
            )
        });
        let after = meta.tensor_facts().cloned();
        *ctx.meta_mut(id) = meta;
        for user in ctx.graph().live_user_ids(id) {
            let seen = match rewrite.seen_values.get(&(user, id)) {
                Some(seen) => seen,
                None => &before,
            };
            if *seen != after {
                push(ctx, &mut queue, &mut queued, user);
            }
        }
    }
}

/// `id`'s meta with its value re-derived from its operands, as rc4's retrace
/// does: [`refreshed_meta`] where `layout::shape` has a rule, else the rules
/// for the other targets a region can hold. Qparams stay, and so do recorded
/// dtypes (see `layout::shape`), except a RESCALE's, which is its dtype
/// argument.
fn retraced_meta(graph: &IrGraph, id: NodeId) -> Result<NodeMeta, ShapeError> {
    let node = graph.node(id);
    if classify_ir_target(&node.target) == IrTargetKind::TosaRescale {
        // RESCALE's value dtype is its dtype argument: rc4's fake kernel
        // returns `torch.empty_like(x, dtype=dtype)`
        // (backends/arm/tosa/dialect/ops/conversion_ops.py:213).
        let mut meta = refreshed_meta(graph, id)?;
        meta.set_tensor_facts(infer_facts(graph, id)?);
        mark_contiguous(&mut meta);
        return Ok(meta);
    }
    if has_local_rule(&node.target) || all_input_nodes(node).is_empty() {
        return refreshed_meta(graph, id);
    }
    let input = |index: usize, name: &'static str| -> Result<&TensorFacts, ShapeError> {
        get_arg(node, name, index)
            .and_then(IrArg::as_node)
            .and_then(|input| graph.node(input).meta.tensor_facts())
            .ok_or(ShapeError::MissingOperand(name))
    };
    let target = node.target.as_str();
    let (shape, input_dtype) = if REDUCTIONS.contains(&target) {
        let source = input(0, "self")?;
        let dims = match get_arg(node, "dim", 1) {
            None | Some(IrArg::None) => Vec::new(),
            Some(IrArg::Int(dim)) => vec![*dim],
            Some(arg) => int_list(arg).ok_or(ShapeError::InvalidArg("reduction dims"))?,
        };
        let keepdim = match get_arg(node, "keepdim", 2) {
            None => false,
            Some(IrArg::Bool(keepdim)) => *keepdim,
            Some(_) => return Err(ShapeError::InvalidArg("reduction keepdim")),
        };
        (reduce_shape(&source.shape, &dims, keepdim)?, &source.dtype)
    } else if PAD_OPS.contains(&target) {
        let source = input(0, "self")?;
        let pad = node
            .args
            .get(1)
            .and_then(int_list)
            .ok_or(ShapeError::InvalidArg("pad"))?;
        (pad_shape(&source.shape, &pad)?, &source.dtype)
    } else if target == UNSQUEEZE_COPY {
        let source = input(0, "self")?;
        let dim = match get_arg(node, "dim", 1) {
            Some(IrArg::Int(dim)) => *dim,
            _ => return Err(ShapeError::InvalidArg("unsqueeze dim")),
        };
        let rank = source.shape.len() as i64 + 1;
        let axis = if dim < 0 { dim + rank } else { dim };
        if !(0..rank).contains(&axis) {
            return Err(ShapeError::InvalidArg("unsqueeze dim out of range"));
        }
        let mut shape = source.shape.clone();
        shape.insert(axis as usize, 1);
        (shape, &source.dtype)
    } else if target == EXPAND_COPY {
        let source = input(0, "self")?;
        let size = get_arg(node, "size", 1)
            .and_then(int_list)
            .ok_or(ShapeError::InvalidArg("expand size"))?;
        (expand_shape(&source.shape, &size)?, &source.dtype)
    } else if matches!(target, QUANTIZE_PER_TENSOR | DEQUANTIZE_PER_TENSOR)
        || TABLE_OPS.binary_search(&target).is_ok()
    {
        let source = input(0, "input")?;
        (source.shape.clone(), &source.dtype)
    } else if is_tosa_table_target(target) {
        let source = input(0, "input1")?;
        (source.shape.clone(), &source.dtype)
    } else {
        return Err(ShapeError::Unsupported(node.target.clone()));
    };
    let dtype = node
        .meta
        .val_dtype()
        .map_or_else(|| input_dtype.clone(), str::to_owned);
    let mut meta = node.meta.clone();
    meta.set_tensor_facts(TensorFacts::new(shape, dtype));
    mark_contiguous(&mut meta);
    Ok(meta)
}

/// `constant_pad_nd(self, pad)`: pad pairs run from the last dim backwards.
fn pad_shape(shape: &[i64], pad: &[i64]) -> Result<Vec<i64>, ShapeError> {
    if !pad.len().is_multiple_of(2) || pad.len() / 2 > shape.len() {
        return Err(ShapeError::InvalidArg("pad length"));
    }
    let mut out = shape.to_vec();
    for (pair, values) in pad.chunks(2).enumerate() {
        let axis = shape.len() - 1 - pair;
        out[axis] += values[0] + values[1];
        if out[axis] < 0 {
            return Err(ShapeError::InvalidArg("pad makes a dim negative"));
        }
    }
    Ok(out)
}

/// `expand_copy(self, size)`: sizes align from the right, `-1` keeps a dim,
/// and only unit dims grow.
fn expand_shape(shape: &[i64], size: &[i64]) -> Result<Vec<i64>, ShapeError> {
    if size.len() < shape.len() {
        return Err(ShapeError::InvalidArg("expand size is shorter than the input"));
    }
    let lead = size.len() - shape.len();
    size.iter()
        .enumerate()
        .map(|(index, &extent)| {
            if index < lead {
                return if extent < 0 {
                    Err(ShapeError::InvalidArg("a new expand dim cannot be -1"))
                } else {
                    Ok(extent)
                };
            }
            let existing = shape[index - lead];
            match extent {
                -1 => Ok(existing),
                _ if existing == extent || existing == 1 => Ok(extent),
                _ => Err(ShapeError::Incompatible("expand of a non-unit dim")),
            }
        })
        .collect()
}

/// rc4's `_is_constant` for `id` under `config`, for the upstream test of the
/// subclass's per-call constant names.
#[cfg(test)]
pub(crate) fn is_constant_in(graph: &mut IrGraph, config: &RemovePermutesConfig, id: NodeId) -> bool {
    let ctx = LayoutCtx::new(graph);
    Detector::new(config).is_constant(&ctx, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::{node, LayoutGraph, ADD, NEG, PERMUTE, RELU, SIGMOID};

    #[test]
    fn find_extra_ones_matches_rc4() {
        assert_eq!(find_extra_ones(&[1, 4, 1, 8], &[4, 8]), Some(vec![0, 2]));
        assert_eq!(find_extra_ones(&[4, 1, 8], &[4, 8]), Some(vec![1]));
        assert_eq!(find_extra_ones(&[4, 2, 8], &[4, 8]), None);
        assert_eq!(find_extra_ones(&[4, 8], &[4, 8]), Some(vec![]));
        assert_eq!(find_extra_ones(&[1, 1], &[1]), Some(vec![1]));
        assert_eq!(find_extra_ones(&[4], &[4, 8]), None);
    }

    #[test]
    fn permutations_normalize_like_rc4() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            vec![],
            NodeMeta::tensor(vec![2, 3, 4], "torch.float32"),
        );
        let permute = |dims: IrArg| IrNode {
            id: 1,
            op: NodeOp::CallFunction,
            name: "p".into(),
            target: PERMUTE_COPY.into(),
            args: vec![IrArg::Node(x), dims],
            kwargs: HashMap::new(),
            meta: NodeMeta::default(),
            tensor: None,
            alive: true,
        };
        assert_eq!(
            get_permutation(&permute(i64_list_arg(&[0, -1, 1]))),
            Some(vec![0, 2, 1])
        );
        assert_eq!(get_permutation(&permute(i64_list_arg(&[0, 0, 1]))), None);
        assert_eq!(get_permutation(&permute(i64_list_arg(&[0, 3, 1]))), None);
        assert_eq!(get_permutation(&permute(IrArg::None)), Some(vec![]));
    }

    /// Turns off, on this thread, the checks test builds make that would make
    /// a scaling test quadratic: the user index after every edit and the
    /// re-run of every skipped search. Restored on drop.
    struct Unchecked {
        users: bool,
        skips: bool,
    }

    impl Unchecked {
        fn new() -> Self {
            Self {
                users: crate::fx_ir::set_check_users_after_edit(false),
                skips: CHECK_SKIPPED_SEARCHES.with(|check| check.replace(false)),
            }
        }
    }

    impl Drop for Unchecked {
        fn drop(&mut self) {
            CHECK_SKIPPED_SEARCHES.with(|check| check.set(self.skips));
            crate::fx_ir::set_check_users_after_edit(self.users);
        }
    }

    /// Region-search steps and wall time of `run` on this thread.
    fn measured(run: impl FnOnce() -> bool) -> (bool, usize, std::time::Duration) {
        SEARCH_STEPS.with(|steps| steps.set(0));
        let started = std::time::Instant::now();
        let modified = run();
        let elapsed = started.elapsed();
        (modified, SEARCH_STEPS.with(std::cell::Cell::get), elapsed)
    }

    /// A residual stream of `blocks` adds, each fed by its own permute with
    /// the same permutation, whose last add reaches the graph output: every
    /// search fails at the output after walking the stream. rc4 walks it once
    /// per permute; the component cache walks it once.
    fn residual_stream(blocks: usize) -> LayoutGraph {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let mut stream = g.permute(x, &[0, 2, 3, 1]);
        stream = g.call(RELU, vec![node(stream)]);
        for block in 0..blocks {
            let y = g.input(&format!("y{block}"), &[1, 3, 4, 5]);
            let branch = g.permute(y, &[0, 2, 3, 1]);
            stream = g.call(ADD, vec![node(stream), node(branch)]);
        }
        g.output(&[stream]);
        g
    }

    #[test]
    fn a_failing_component_is_searched_once() {
        let _unchecked = Unchecked::new();
        for blocks in [500, 2000] {
            let mut g = residual_stream(blocks);
            let (modified, steps, _) = measured(|| {
                IrRemovePermutesAroundElementwiseTosaOpsPass.run(g.graph_mut())
            });
            assert!(!modified);
            // One walk of the stream (each add is reached from its user and
            // its input) plus one closure, not one walk per permute.
            assert!(
                steps <= 3 * blocks + 10,
                "{steps} search steps for {blocks} blocks"
            );
        }
    }

    /// `blocks` independent regions `x -> permute -> relu -> neg -> permute^-1`
    /// whose results a chain of matmuls (not permutable) collects, so no node
    /// gathers more than two operands: rewiring a node costs its operand count,
    /// in `IrGraph` as in torch.fx.
    fn independent_regions(blocks: usize) -> LayoutGraph {
        let mut g = LayoutGraph::new();
        let mut acc = g.input("acc", &[3, 3]);
        for block in 0..blocks {
            let x = g.input(&format!("x{block}"), &[3, 3]);
            let p = g.permute(x, &[1, 0]);
            let r = g.call(RELU, vec![node(p)]);
            let n = g.call(NEG, vec![node(r)]);
            let exit = g.permute(n, &[1, 0]);
            acc = g.add_unrefreshed("aten.mm.default", vec![node(acc), node(exit)]);
            g.graph_mut().node_mut(acc).meta = NodeMeta::tensor(vec![3, 3], "torch.float32");
        }
        g.output(&[acc]);
        g
    }

    #[test]
    fn independent_regions_are_searched_once_each() {
        let _unchecked = Unchecked::new();
        for blocks in [1000, 4000] {
            let mut g = independent_regions(blocks);
            let (modified, steps, _) = measured(|| {
                IrRemovePermutesAroundElementwiseTosaOpsPass.run(g.graph_mut())
            });
            assert!(modified);
            assert_eq!(g.count(PERMUTE), 0);
            assert_eq!(steps, 2 * blocks);
        }
    }

    /// Wall time on growing graphs, for the record:
    /// `cargo test --release --lib remove_permutes_timing -- --ignored --nocapture`.
    /// On an M-series laptop (2026-11) both families scale linearly: 30k, 120k
    /// and 480k nodes of independent regions take about 50, 215 and 900 ms;
    /// 15k, 60k and 240k nodes of failing residual stream 13, 66 and 260 ms.
    #[test]
    #[ignore = "timing report"]
    fn remove_permutes_timing() {
        let _unchecked = Unchecked::new();
        for blocks in [5_000, 20_000, 80_000] {
            let mut regions = independent_regions(blocks);
            let (_, steps, elapsed) = measured(|| {
                IrRemovePermutesAroundElementwiseTosaOpsPass.run(regions.graph_mut())
            });
            let nodes = regions.graph().nodes.len();
            println!("independent regions: {nodes} nodes, {steps} steps, {elapsed:?}");
            let mut stream = residual_stream(blocks);
            let (_, steps, elapsed) = measured(|| {
                IrRemovePermutesAroundElementwiseTosaOpsPass.run(stream.graph_mut())
            });
            let nodes = stream.graph().nodes.len();
            println!("failing residual stream: {nodes} nodes, {steps} steps, {elapsed:?}");
        }
    }

    #[test]
    fn a_deep_region_is_searched_without_recursion() {
        let _unchecked = Unchecked::new();
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3]);
        let mut chain = g.permute(x, &[0, 2, 1]);
        for _ in 0..50_000 {
            chain = g.call("aten.hardtanh.default", vec![node(chain)]);
        }
        let out = g.permute(chain, &[0, 2, 1]);
        g.output(&[out]);
        let (modified, steps, _) =
            measured(|| IrRemovePermutesAroundElementwiseTosaOpsPass.run(g.graph_mut()));
        assert!(modified);
        assert_eq!(g.count(PERMUTE), 0);
        assert_eq!(steps, 50_000);
    }

    /// A second call on an unchanged ctx skips every search the first one
    /// failed.
    #[test]
    fn the_memo_skips_unchanged_failed_searches() {
        let _unchecked = Unchecked::new();
        let mut g = residual_stream(50);
        let config = RemovePermutesConfig::tosa();
        let graph = g.graph_mut();
        let mut ctx = LayoutCtx::new(graph);
        let mut memo = RemovePermutesMemo::default();
        let (first, first_steps, _) =
            measured(|| remove_permutes_in_ctx(&mut ctx, &config, &mut memo, DeadCode::Local));
        let (second, second_steps, _) =
            measured(|| remove_permutes_in_ctx(&mut ctx, &config, &mut memo, DeadCode::Local));
        assert!(!first && !second);
        assert!(first_steps > 0);
        assert_eq!(second_steps, 0);
    }

    fn run_with(config: RemovePermutesConfig, g: &mut LayoutGraph) -> bool {
        IrRemovePermutesAroundElementwiseOpsPass { config }.run(g.graph_mut())
    }

    /// A `get_attr` tensor is a constant (`:677-678`) and gets its
    /// compensating permute directly after it, as rc4 does (checked with rc4
    /// on the same graph; the replay format has no `get_attr`).
    #[test]
    fn a_get_attr_constant_gets_a_permute_after_it() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let w = g.graph_mut().add_node(
            NodeOp::GetAttr,
            "frozen_w".to_string(),
            "frozen_w".to_string(),
            Vec::new(),
            NodeMeta::tensor(vec![1, 4, 5, 3], "torch.float32"),
        );
        let p = g.permute(x, &[0, 2, 3, 1]);
        let add = g.call(ADD, vec![node(p), node(w)]);
        let q = g.permute(add, &[0, 3, 1, 2]);
        g.output(&[q]);
        let names = [x, w, add].map(|id| g.name(id));

        assert!(run_with(RemovePermutesConfig::base(), &mut g));

        let [x, w, add] = names.map(|name| g.id(&name));
        g.assert_targets(&[PERMUTE, ADD]);
        let permute = g.graph().node(add).args[1].as_node().expect("a permuted constant");
        assert_eq!(g.graph().node(permute).args[0].as_node(), Some(w));
        assert_eq!(g.graph().node(permute).arg_int_list(1), Some(vec![0, 3, 1, 2]));
        assert_eq!(g.graph().node(add).args[0].as_node(), Some(x));
        let order = g.graph().topological_live_node_ids();
        let at = |id| order.iter().position(|&o| o == id).expect("live");
        assert_eq!(at(permute), at(w) + 1);
        assert_eq!(g.shape(add), &[1, 3, 4, 5]);
    }

    /// RETAB spells the TOSA dialect `backend.tosa.X` and `tosa.X`; RESCALE is
    /// permutable in the subclass either way.
    #[test]
    fn both_rescale_spellings_are_permutable() {
        for target in ["backend.tosa.RESCALE.default", "tosa.RESCALE.default"] {
            let mut g = LayoutGraph::new();
            let x = g.typed_input("x", &[1, 3, 4, 5], "torch.int32");
            let p = g.permute(x, &[0, 2, 3, 1]);
            let args = vec![
                node(p),
                IrArg::Dtype("torch.int8".to_string()),
                IrArg::List(vec![IrArg::Float(0.5)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ];
            let r = g.call(target, args);
            let q = g.permute(r, &[0, 3, 1, 2]);
            g.output(&[q]);
            let rescale = g.name(r);

            assert!(run_with(RemovePermutesConfig::tosa(), &mut g), "{target}");

            g.assert_targets(&[target]);
            assert_eq!(g.shape(g.id(&rescale)), &[1, 3, 4, 5]);
            assert_eq!(g.dtype(g.id(&rescale)), "torch.int8");
        }
    }

    fn rank0_sigmoid_region(qparams: bool) -> LayoutGraph {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[]);
        let p = g.permute(x, &[]);
        let s = g.call(SIGMOID, vec![node(p)]);
        if qparams {
            let qparam = crate::fx_ir::QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string());
            g.graph_mut().node_mut(s).meta.set_both_qparams(qparam);
        }
        let q = g.permute(s, &[]);
        g.output(&[q]);
        g
    }

    /// RETAB-only: a quantized table op is the TABLE rc4 has inserted by now,
    /// whose rank-1 table fails a rank-0 region in the subclass. Without
    /// qparams, or in the base pass, the op is just elementwise.
    #[test]
    fn a_pending_table_fails_a_rank0_region() {
        assert!(!run_with(RemovePermutesConfig::tosa(), &mut rank0_sigmoid_region(true)));
        assert!(run_with(RemovePermutesConfig::tosa(), &mut rank0_sigmoid_region(false)));
        assert!(run_with(RemovePermutesConfig::base(), &mut rank0_sigmoid_region(true)));
    }

    /// `hardswish` is not tagged pointwise, so only the subclass, which lists
    /// the TABLE ops (`:19-26`), crosses it.
    #[test]
    fn hardswish_is_permutable_only_in_the_subclass() {
        let build = || {
            let mut g = LayoutGraph::new();
            let x = g.input("x", &[1, 3, 4, 5]);
            let p = g.permute(x, &[0, 2, 3, 1]);
            let h = g.add_unrefreshed("aten.hardswish.default", vec![node(p)]);
            g.graph_mut().node_mut(h).meta = NodeMeta::tensor(vec![1, 4, 5, 3], "torch.float32");
            let q = g.permute(h, &[0, 3, 1, 2]);
            g.output(&[q]);
            g
        };
        assert!(!run_with(RemovePermutesConfig::base(), &mut build()));
        let mut g = build();
        assert!(run_with(RemovePermutesConfig::tosa(), &mut g));
        g.assert_targets(&["aten.hardswish.default"]);
    }

    #[test]
    fn pad_and_expand_shapes() {
        assert_eq!(pad_shape(&[1, 3, 4], &[1, 2]), Ok(vec![1, 3, 7]));
        assert_eq!(pad_shape(&[1, 3, 4], &[0, 0, 1, 0]), Ok(vec![1, 4, 4]));
        assert!(pad_shape(&[3], &[0, 0, 1, 1]).is_err());
        assert_eq!(expand_shape(&[2, 1, 3], &[-1, 4, 3]), Ok(vec![2, 4, 3]));
        assert_eq!(expand_shape(&[3], &[2, 3]), Ok(vec![2, 3]));
        assert!(expand_shape(&[2, 3], &[4, 3]).is_err());
    }
}

// Adversarial review tests (APU-158 review), with access to the private caches.
#[cfg(test)]
#[path = "remove_permutes_review_tests.rs"]
mod review_tests;
