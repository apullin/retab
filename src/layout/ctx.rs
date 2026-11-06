//! The mutation context the ported rc4 layout passes run in (APU-135 P0).
//!
//! rc4's permute/view propagation
//! (backends/transforms/propagate_view_copy_permute_pass.py and its helpers)
//! edits a torch.fx graph in place. It creates nodes at an insert point, moves
//! nodes with `Node.prepend`/`Node.append`, rewires inputs and erases nodes,
//! and it sweeps and breaks ties in `graph.nodes` order. `LayoutCtx` gives the
//! Rust ports the same model over an `IrGraph`:
//!
//! - `NodeOrder` is the `graph.nodes` linked list with torch.fx's insertion
//!   semantics. Gapped `u64` keys play the role of FX's `Node._sort_key`, so
//!   order comparisons are O(1); a gap that runs out is fixed by spreading out
//!   a small window of keys.
//! - Every mutation made through the ctx bumps per-node version stamps.
//! - An attempt memo records the node versions a failed rewrite read, so a
//!   worklist driver retries it only after one of them changes.
//!
//! `LayoutCtx::finish` writes the order back by renumbering the graph
//! (`IrGraph::reorder_nodes`). Position anchors alone would not be enough: they
//! steer `IrGraph::topological_live_node_ids`, but the IR-to-serialize
//! conversion (`ingest::from_ir::graph_from_ir`) never reads them; it seeds its
//! def-use sort (`topo_order`) with node-id order.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeId, NodeMeta, NodeOp};

/// Link sentinel: no previous or next node.
const NIL: NodeId = NodeId::MAX;
/// Key of a node outside the order; placed keys are at least 1.
const UNPLACED: u64 = 0;
/// Key spacing for a seeded order and for nodes added at either end.
const SEED_GAP: u64 = 1 << 32;
/// A relabel widens its window until it can space keys at least this far apart.
const MIN_RELABEL_GAP: u64 = 1 << 16;

fn non_nil(id: NodeId) -> Option<NodeId> {
    (id != NIL).then_some(id)
}

/// torch.fx's `graph.nodes` linked list (`_node_list`, torch/fx/graph.py:291-314
/// in torch 2.15) over node ids.
///
/// `insert_before` and `insert_after` are `Node.prepend` and `Node.append`
/// (torch/fx/node.py:427-452). A node that is already placed is unlinked
/// first, so the same calls create and move nodes, and FX's self-moves are
/// no-ops. Keys strictly increase along the list. Inserting between two
/// neighbours bisects their gap; when no gap is left, a window around the
/// insertion point is spread out evenly, doubling until the spacing reaches
/// `MIN_RELABEL_GAP`.
#[derive(Debug, Clone)]
pub(crate) struct NodeOrder {
    prev: Vec<NodeId>,
    next: Vec<NodeId>,
    key: Vec<u64>,
    head: NodeId,
    tail: NodeId,
    len: usize,
    relabels: usize,
}

impl NodeOrder {
    /// An order holding `ids` in sequence.
    pub(crate) fn new(ids: &[NodeId]) -> Self {
        let mut order = Self {
            prev: Vec::new(),
            next: Vec::new(),
            key: Vec::new(),
            head: NIL,
            tail: NIL,
            len: 0,
            relabels: 0,
        };
        for &id in ids {
            debug_assert!(!order.contains(id), "node {id} is listed twice");
            order.push_back(id);
        }
        order
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn contains(&self, id: NodeId) -> bool {
        self.key.get(id).is_some_and(|&key| key != UNPLACED)
    }

    pub(crate) fn first(&self) -> Option<NodeId> {
        non_nil(self.head)
    }

    pub(crate) fn last(&self) -> Option<NodeId> {
        non_nil(self.tail)
    }

    /// The node after `id`: `Node.next` (torch/fx/node.py:404-414), with
    /// `None` for the root sentinel.
    pub(crate) fn next(&self, id: NodeId) -> Option<NodeId> {
        non_nil(self.next[self.placed(id)])
    }

    /// The node before `id`: `Node.prev` (torch/fx/node.py:415-425), with
    /// `None` for the root sentinel.
    pub(crate) fn prev(&self, id: NodeId) -> Option<NodeId> {
        non_nil(self.prev[self.placed(id)])
    }

    /// `id`'s order key. Keys increase along the order; an insertion or move
    /// may relabel other nodes, so compare keys only between edits.
    pub(crate) fn key(&self, id: NodeId) -> Option<u64> {
        self.contains(id).then(|| self.key[id])
    }

    /// Compare two placed nodes by position, like FX's `_sort_key`
    /// comparisons and the `enumerate(graph.nodes)` index in
    /// backends/transforms/propagate_view_copy_permute_pass.py:1108-1111.
    pub(crate) fn cmp(&self, a: NodeId, b: NodeId) -> Ordering {
        self.key[self.placed(a)].cmp(&self.key[self.placed(b)])
    }

    pub(crate) fn is_before(&self, a: NodeId, b: NodeId) -> bool {
        self.cmp(a, b) == Ordering::Less
    }

    /// Sort placed nodes by position: `sorted(nodes, key=graph_order.__getitem__)`
    /// (backends/transforms/propagate_view_copy_permute_pass.py:1111).
    pub(crate) fn sort(&self, ids: &mut [NodeId]) {
        ids.sort_unstable_by_key(|&id| self.key[self.placed(id)]);
    }

    /// Nodes in order: `iter(graph.nodes)`.
    pub(crate) fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::successors(self.first(), move |&id| non_nil(self.next[id]))
    }

    /// Nodes in reverse order: `reversed(graph.nodes)`.
    pub(crate) fn iter_rev(&self) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::successors(self.last(), move |&id| non_nil(self.prev[id]))
    }

    /// A snapshot, like `list(graph.nodes)`.
    pub(crate) fn to_vec(&self) -> Vec<NodeId> {
        self.iter().collect()
    }

    /// How many relabels the order has needed so far.
    pub(crate) fn relabel_count(&self) -> usize {
        self.relabels
    }

    /// Place `id` directly before `anchor`: `anchor.prepend(id)`
    /// (torch/fx/node.py:427-441). A no-op when `id == anchor`.
    pub(crate) fn insert_before(&mut self, id: NodeId, anchor: NodeId) {
        self.placed(anchor);
        if id == anchor {
            return;
        }
        self.remove(id);
        let prev = self.prev[anchor];
        self.link_between(id, prev, anchor);
    }

    /// Place `id` directly after `anchor`: `anchor.append(id)`, which FX
    /// implements as `anchor.next.prepend(id)` (torch/fx/node.py:443-452).
    pub(crate) fn insert_after(&mut self, id: NodeId, anchor: NodeId) {
        match non_nil(self.next[self.placed(anchor)]) {
            Some(next) => self.insert_before(id, next),
            None => self.push_back(id),
        }
    }

    /// Place `id` last: `Graph._root.prepend(id)`, FX's default insert point
    /// (torch/fx/graph.py:1459).
    pub(crate) fn push_back(&mut self, id: NodeId) {
        self.remove(id);
        let tail = self.tail;
        self.link_between(id, tail, NIL);
    }

    /// Place `id` first: `Graph._root.append(id)`, the insert point of
    /// `inserting_before(None)` (torch/fx/graph.py:1736-1737).
    pub(crate) fn push_front(&mut self, id: NodeId) {
        match self.first() {
            Some(head) => self.insert_before(id, head),
            None => self.push_back(id),
        }
    }

    /// Unlink `id`, as `Graph.erase_node` does (torch/fx/graph.py:1677-1716).
    /// Returns whether it was placed.
    pub(crate) fn remove(&mut self, id: NodeId) -> bool {
        if !self.contains(id) {
            return false;
        }
        let (prev, next) = (self.prev[id], self.next[id]);
        if prev == NIL {
            self.head = next;
        } else {
            self.next[prev] = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.prev[next] = prev;
        }
        self.prev[id] = NIL;
        self.next[id] = NIL;
        self.key[id] = UNPLACED;
        self.len -= 1;
        true
    }

    fn placed(&self, id: NodeId) -> NodeId {
        assert!(self.contains(id), "node {id} is not in the node order");
        id
    }

    /// Link the unplaced `id` between the adjacent `prev` and `next`, either
    /// of which may be `NIL` at an end of the list.
    fn link_between(&mut self, id: NodeId, prev: NodeId, next: NodeId) {
        if id >= self.key.len() {
            self.prev.resize(id + 1, NIL);
            self.next.resize(id + 1, NIL);
            self.key.resize(id + 1, UNPLACED);
        }
        self.key[id] = self.key_between(prev, next);
        self.prev[id] = prev;
        self.next[id] = next;
        if prev == NIL {
            self.head = id;
        } else {
            self.next[prev] = id;
        }
        if next == NIL {
            self.tail = id;
        } else {
            self.prev[next] = id;
        }
        self.len += 1;
    }

    fn key_between(&mut self, prev: NodeId, next: NodeId) -> u64 {
        if let Some(key) = self.free_key(prev, next) {
            return key;
        }
        self.relabel_around(prev, next);
        self.free_key(prev, next)
            .expect("a relabel leaves room between neighbours")
    }

    fn free_key(&self, prev: NodeId, next: NodeId) -> Option<u64> {
        let low = non_nil(prev).map_or(0, |id| self.key[id]);
        let high = non_nil(next).map_or(u64::MAX, |id| self.key[id]);
        let room = high - low;
        if room < 2 {
            return None;
        }
        Some(match (prev == NIL, next == NIL) {
            (false, false) => low + room / 2,
            (_, true) => low + (room / 2).min(SEED_GAP),
            (true, false) => high - (room / 2).min(SEED_GAP),
        })
    }

    /// Spread out the keys of a window around the gap between `prev` and
    /// `next`, doubling the window until its keys can be `MIN_RELABEL_GAP`
    /// apart or it covers the whole list.
    fn relabel_around(&mut self, prev: NodeId, next: NodeId) {
        let (mut first, mut last) = match (non_nil(prev), non_nil(next)) {
            (Some(prev), Some(next)) => (prev, next),
            (Some(id), None) | (None, Some(id)) => (id, id),
            (None, None) => unreachable!("an empty order always has room"),
        };
        let mut count: u64 = if first == last { 1 } else { 2 };
        loop {
            let low = non_nil(self.prev[first]).map_or(0, |id| self.key[id]);
            let high = non_nil(self.next[last]).map_or(u64::MAX, |id| self.key[id]);
            let gap = ((high - low) / (count + 1)).min(SEED_GAP);
            let whole = self.prev[first] == NIL && self.next[last] == NIL;
            if gap >= MIN_RELABEL_GAP || (whole && gap >= 2) {
                let mut key = low;
                let mut id = first;
                loop {
                    key += gap;
                    self.key[id] = key;
                    if id == last {
                        break;
                    }
                    id = self.next[id];
                }
                self.relabels += 1;
                return;
            }
            assert!(!whole, "node order keys exhausted at {} nodes", self.len);
            let target = count * 2;
            while count < target && (self.prev[first] != NIL || self.next[last] != NIL) {
                if self.prev[first] != NIL {
                    first = self.prev[first];
                    count += 1;
                }
                if count < target && self.next[last] != NIL {
                    last = self.next[last];
                    count += 1;
                }
            }
        }
    }
}

/// torch.fx's `Node.is_impure` for a call (`torch/_library/utils.py`
/// `is_impure`, torch 2.15), on RETAB's target strings: a mutable schema
/// (an in-place `name_` op or an `out` variant), an op in
/// `torch/fx/node.py`'s `_side_effectful_functions`, or a seeded random op.
pub(crate) fn is_impure_call(target: &str) -> bool {
    const SIDE_EFFECTFUL: &[&str] = &[
        "aten._assert_async.msg",
        "aten._assert_scalar.default",
        "aten._assert_tensor_metadata.default",
        "aten.sym_constrain_range.default",
        "aten.sym_constrain_range_for_size.default",
        "prims._sink_tokens.default",
    ];
    const SEEDED_RANDOM: &[&str] = &[
        "bernoulli",
        "multinomial",
        "native_dropout",
        "normal",
        "poisson",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
    ];
    let mut parts = target.split('.');
    let (Some(_namespace), Some(name)) = (parts.next(), parts.next()) else {
        return false;
    };
    let overload = parts.next().unwrap_or("default");
    SIDE_EFFECTFUL.contains(&target)
        || (name.ends_with('_') && !name.trim_end_matches('_').is_empty())
        || overload == "out"
        || overload.ends_with("_out")
        || SEEDED_RANDOM.contains(&name)
}

/// Identifies a rewrite rule in the attempt memo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RuleId(pub(crate) &'static str);

/// The `(node, version)` pairs a failed attempt read, sorted by node.
type ReadSet = Box<[(NodeId, u64)]>;

/// Where `LayoutCtx::create_node` places new nodes, after torch.fx's insert
/// points (torch/fx/graph.py:1459 and 1718-1766).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InsertPoint {
    /// The default, `Graph._root.prepend`: each node goes last.
    End,
    /// `inserting_before(None)`, which is `Graph._root.append`: each node goes
    /// first, so a sequence of nodes ends up in reverse creation order.
    Start,
    /// `inserting_before(n)`, which is `n.prepend`: each node goes directly
    /// before `n`, so a sequence keeps its creation order.
    Before(NodeId),
    /// `inserting_after(n)`, which is `n.append`: each node goes directly
    /// after `n`, so a sequence ends up in reverse creation order.
    After(NodeId),
}

/// The mutation context the ported rc4 layout passes run in.
///
/// It borrows the graph for the whole pass, and every edit goes through it so
/// the order, the version stamps and the graph stay in step. Nothing reaches
/// the graph's order until `finish`.
pub(crate) struct LayoutCtx<'g> {
    graph: &'g mut IrGraph,
    order: NodeOrder,
    versions: Vec<u64>,
    insert_point: InsertPoint,
    attempts: HashMap<(RuleId, NodeId), ReadSet>,
    /// Per base name, the lowest suffix `unique_name` may still find free.
    name_suffixes: RefCell<HashMap<String, u64>>,
}

impl<'g> LayoutCtx<'g> {
    /// Seed the order from `IrGraph::topological_live_node_ids` (fx_ir.rs),
    /// which honours the position anchors RETAB's passes leave.
    pub(crate) fn new(graph: &'g mut IrGraph) -> Self {
        let order = NodeOrder::new(&graph.topological_live_node_ids());
        let versions = vec![0; graph.nodes.len()];
        Self {
            graph,
            order,
            versions,
            insert_point: InsertPoint::End,
            attempts: HashMap::new(),
            name_suffixes: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn graph(&self) -> &IrGraph {
        self.graph
    }

    pub(crate) fn node(&self, id: NodeId) -> &IrNode {
        self.graph.node(id)
    }

    pub(crate) fn order(&self) -> &NodeOrder {
        &self.order
    }

    pub(crate) fn is_erased(&self, id: NodeId) -> bool {
        !self.graph.node(id).alive
    }

    /// `id`'s version stamp. It increases whenever, through this ctx, the
    /// node's args, kwargs, target or meta change, it is moved or erased, or
    /// a node starts, stops or rewrites a use of it. Inserting an unrelated
    /// node next to it does not count.
    pub(crate) fn version(&self, id: NodeId) -> u64 {
        self.versions.get(id).copied().unwrap_or(0)
    }

    pub(crate) fn insert_point(&self) -> InsertPoint {
        self.insert_point
    }

    /// Set the insert point until changed again, as calling
    /// `graph.inserting_before(n)` outside a `with` does (torch/fx/graph.py:
    /// 1718-1766). Returns the previous insert point.
    pub(crate) fn set_insert_point(&mut self, point: InsertPoint) -> InsertPoint {
        if let InsertPoint::Before(anchor) | InsertPoint::After(anchor) = point {
            self.order.placed(anchor);
        }
        std::mem::replace(&mut self.insert_point, point)
    }

    /// Run `f` with `point` as the insert point and restore the previous one
    /// afterwards, like `with graph.inserting_before(n):` (`_InsertPoint`,
    /// torch/fx/graph.py:274-288).
    pub(crate) fn with_insert_point<R>(
        &mut self,
        point: InsertPoint,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let saved = self.set_insert_point(point);
        let result = f(self);
        self.insert_point = saved;
        result
    }

    /// `with graph.inserting_before(anchor):` (torch/fx/graph.py:1718-1741),
    /// as at backends/transforms/propagate_view_copy_permute_pass.py:873.
    pub(crate) fn inserting_before<R>(
        &mut self,
        anchor: NodeId,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.with_insert_point(InsertPoint::Before(anchor), f)
    }

    /// `with graph.inserting_after(anchor):` (torch/fx/graph.py:1743-1766),
    /// as at backends/transforms/propagate_view_copy_permute_pass.py:1379.
    pub(crate) fn inserting_after<R>(
        &mut self,
        anchor: NodeId,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.with_insert_point(InsertPoint::After(anchor), f)
    }

    /// Add a node at the insert point: `Graph.create_node`
    /// (torch/fx/graph.py:1587-1663). The nodes it reads gain a user. Like FX's
    /// `_graph_namespace.create_name`, a taken name gets a `_N` suffix: names
    /// key `IrGraph`'s name index and become TOSA tensor names.
    pub(crate) fn create_node(
        &mut self,
        op: NodeOp,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let name = self.unique_name(name.into());
        let id = self
            .graph
            .add_node_with_kwargs(op, name, target.into(), args, kwargs, meta);
        match self.insert_point {
            InsertPoint::End => self.order.push_back(id),
            InsertPoint::Start => self.order.push_front(id),
            InsertPoint::Before(anchor) => self.order.insert_before(id, anchor),
            InsertPoint::After(anchor) => self.order.insert_after(id, anchor),
        }
        self.bump_inputs_of(id);
        id
    }

    /// `name`, or the first of `name_1`, `name_2`, ... that no node has.
    ///
    /// The graph's name index only grows, so a suffix found taken stays taken;
    /// remembering where each base's scan stopped keeps a run of nodes with one
    /// name (rc4 names every node it creates after its target) at amortized
    /// O(1) per name instead of O(nodes so far), as FX's `_Namespace` counters do.
    pub(crate) fn unique_name(&self, name: String) -> String {
        if !self.graph.contains_name(&name) {
            return name;
        }
        let mut suffixes = self.name_suffixes.borrow_mut();
        let next = suffixes.entry(name.clone()).or_insert(1);
        loop {
            let candidate = format!("{name}_{next}");
            if !self.graph.contains_name(&candidate) {
                return candidate;
            }
            *next += 1;
        }
    }

    /// `graph.call_function(target, args)` at the insert point.
    pub(crate) fn call_function(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        self.create_node(
            NodeOp::CallFunction,
            name,
            target,
            args,
            HashMap::new(),
            meta,
        )
    }

    /// Move `id` directly before `anchor`: `anchor.prepend(id)`
    /// (torch/fx/node.py:427-441), as at
    /// backends/transforms/propagate_view_copy_permute_pass.py:822.
    pub(crate) fn move_before(&mut self, id: NodeId, anchor: NodeId) {
        self.move_with(id, |order| order.insert_before(id, anchor));
    }

    /// Move `id` directly after `anchor`: `anchor.append(id)`
    /// (torch/fx/node.py:443-452), as at
    /// backends/transforms/propagate_view_copy_permute_pass.py:1418.
    pub(crate) fn move_after(&mut self, id: NodeId, anchor: NodeId) {
        self.move_with(id, |order| order.insert_after(id, anchor));
    }

    /// Replace `id`'s args: `node.args = ...` (torch/fx/node.py:467).
    pub(crate) fn set_args(&mut self, id: NodeId, args: Vec<IrArg>) {
        self.bump_inputs_of(id);
        self.graph.set_args(id, args);
        self.bump_inputs_of(id);
        self.bump(id);
    }

    /// Replace one positional arg: `Node.update_arg` (torch/fx/node.py:515-527),
    /// which raises for an index past the end.
    pub(crate) fn set_arg(&mut self, id: NodeId, index: usize, value: IrArg) {
        assert!(
            index < self.graph.node(id).args.len(),
            "set_arg: node {id} has no arg {index}"
        );
        self.bump_inputs_of(id);
        self.graph.set_arg(id, index, value);
        self.bump_inputs_of(id);
        self.bump(id);
    }

    /// Set a keyword arg: `Node.update_kwarg` (torch/fx/node.py:557).
    pub(crate) fn set_kwarg(&mut self, id: NodeId, key: impl Into<String>, value: IrArg) {
        self.bump_inputs_of(id);
        self.graph.set_kwarg(id, key, value);
        self.bump_inputs_of(id);
        self.bump(id);
    }

    /// Replace `id`'s kwargs: `node.kwargs = ...` (torch/fx/node.py:490), as at
    /// backends/transforms/fuse_identical_input_transforms_pass.py:241.
    pub(crate) fn set_kwargs(&mut self, id: NodeId, kwargs: HashMap<String, IrArg>) {
        self.bump_inputs_of(id);
        self.graph.node_mut(id).kwargs = kwargs;
        self.graph.sync_node_users(id);
        self.bump_inputs_of(id);
        self.bump(id);
    }

    /// Retarget `id` in place, as at
    /// backends/transforms/canonicalize_view_copy_permute_pass.py:433.
    pub(crate) fn set_target(&mut self, id: NodeId, target: impl Into<String>) {
        self.graph.node_mut(id).target = target.into();
        self.bump(id);
    }

    /// Mutable access to `id`'s meta; counts as a change whether or not the
    /// caller writes.
    pub(crate) fn meta_mut(&mut self, id: NodeId) -> &mut NodeMeta {
        self.bump(id);
        &mut self.graph.node_mut(id).meta
    }

    /// Point `user`'s uses of `old` at `new`: `Node.replace_input_with`
    /// (torch/fx/node.py:868-884). Returns whether anything changed.
    pub(crate) fn replace_input_with(&mut self, user: NodeId, old: NodeId, new: NodeId) -> bool {
        if old == new {
            return false;
        }
        let inputs = self.input_ids(user);
        if !self.graph.replace_input_use(user, old, new) {
            return false;
        }
        for input in inputs {
            self.bump(input);
        }
        self.bump_inputs_of(user);
        self.bump(user);
        true
    }

    /// Point every live use of `old` at `new`: `Node.replace_all_uses_with`
    /// (torch/fx/node.py:712-756). Returns the rewired users in id order.
    pub(crate) fn replace_all_uses_with(&mut self, old: NodeId, new: NodeId) -> Vec<NodeId> {
        if old == new {
            return Vec::new();
        }
        let users = self.graph.live_user_ids(old);
        if users.is_empty() {
            return users;
        }
        for &user in &users {
            self.bump_inputs_of(user);
            self.bump(user);
        }
        self.graph.replace_all_uses(old, new);
        self.bump(new);
        users
    }

    /// Erase a node nothing uses: `Graph.erase_node` (torch/fx/graph.py:
    /// 1677-1716), which raises while the node has users and ignores a node
    /// already erased.
    pub(crate) fn erase_node(&mut self, id: NodeId) {
        if self.is_erased(id) {
            return;
        }
        assert!(
            !self.graph.has_live_users(id),
            "erase_node: node {id} still has live users"
        );
        self.bump_inputs_of(id);
        self.graph.erase(id);
        self.order.remove(id);
        self.bump(id);
    }

    /// `graph.eliminate_dead_code()` as rc4 calls it (e.g.
    /// backends/transforms/propagate_view_copy_permute_pass.py:160), kept in
    /// step with the order: unused nodes go unless they are impure
    /// ([`is_impure_call`]).
    pub(crate) fn eliminate_dead_code(&mut self) -> bool {
        if !self
            .graph
            .eliminate_dead_code_keeping(|node| node.op == NodeOp::CallFunction && is_impure_call(&node.target))
        {
            return false;
        }
        let erased: Vec<NodeId> = self
            .order
            .iter()
            .filter(|&id| !self.graph.node(id).alive)
            .collect();
        for id in erased {
            self.order.remove(id);
            self.bump_inputs_of(id);
            self.bump(id);
        }
        true
    }

    /// Record a failed attempt of `rule` at `anchor` that read `reads`; the
    /// anchor always counts as read. rc4 re-traces the whole graph between
    /// sweeps (backends/transforms/propagate_view_copy_permute_pass.py:106-162);
    /// the memo lets the incremental port skip rewrites whose inputs have not
    /// changed instead.
    pub(crate) fn record_attempt(
        &mut self,
        rule: RuleId,
        anchor: NodeId,
        reads: impl IntoIterator<Item = NodeId>,
    ) {
        let mut stamped: Vec<(NodeId, u64)> = std::iter::once(anchor)
            .chain(reads)
            .map(|id| (id, self.version(id)))
            .collect();
        stamped.sort_unstable();
        stamped.dedup();
        self.attempts
            .insert((rule, anchor), stamped.into_boxed_slice());
    }

    /// Whether `rule` should run at `anchor`: true when no attempt is recorded
    /// or a node it read has changed version since, which includes being
    /// erased. A rule must therefore list every node whose args, meta, users
    /// or position it inspected.
    pub(crate) fn should_retry(&self, rule: RuleId, anchor: NodeId) -> bool {
        self.attempts.get(&(rule, anchor)).is_none_or(|reads| {
            reads
                .iter()
                .any(|&(id, version)| self.version(id) != version)
        })
    }

    pub(crate) fn forget_attempt(&mut self, rule: RuleId, anchor: NodeId) {
        self.attempts.remove(&(rule, anchor));
    }

    /// Whether every node comes after the live nodes it reads, which
    /// `Graph.lint` requires (torch/fx/graph.py:2612) and `finish` needs.
    pub(crate) fn is_topological(&self) -> bool {
        self.order.iter().all(|id| {
            self.input_ids(id)
                .into_iter()
                .all(|input| !self.order.contains(input) || self.order.is_before(input, id))
        })
    }

    /// Write the order back: renumber the graph so its node ids follow the
    /// order (`IrGraph::reorder_nodes`). Afterwards `topological_live_node_ids`
    /// and `ingest::from_ir::graph_from_ir` both emit the live nodes in this
    /// order, and later passes anchor new nodes against it as they would
    /// against a freshly imported graph. Returns the old-to-new id map.
    pub(crate) fn finish(self) -> Vec<NodeId> {
        debug_assert_eq!(self.order.len(), self.graph.live_node_count());
        debug_assert!(self.is_topological(), "node order is not topological");
        let order = self.order.to_vec();
        self.graph.reorder_nodes(&order)
    }

    fn move_with(&mut self, id: NodeId, place: impl FnOnce(&mut NodeOrder)) {
        let neighbours = (self.order.prev(id), self.order.next(id));
        place(&mut self.order);
        if (self.order.prev(id), self.order.next(id)) != neighbours {
            self.bump(id);
        }
    }

    fn input_ids(&self, id: NodeId) -> Vec<NodeId> {
        let node = self.graph.node(id);
        let mut ids = Vec::new();
        for arg in node.args.iter().chain(node.kwargs.values()) {
            arg.collect_node_ids(&mut ids);
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn bump_inputs_of(&mut self, id: NodeId) {
        for input in self.input_ids(id) {
            self.bump(input);
        }
    }

    fn bump(&mut self, id: NodeId) {
        if id >= self.graph.nodes.len() {
            return;
        }
        if id >= self.versions.len() {
            self.versions.resize(self.graph.nodes.len(), 0);
        }
        self.versions[id] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::TensorStore;
    use crate::test_ir::{f32_call, f32_meta, f32_placeholder};

    const RELU: &str = "aten.relu.default";

    /// Deterministic SplitMix64 for the property tests.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// FX's node list as a plain vector, the reference model for `NodeOrder`.
    struct VecOrder(Vec<NodeId>);

    impl VecOrder {
        fn position(&self, id: NodeId) -> usize {
            self.0.iter().position(|&x| x == id).unwrap()
        }

        fn remove(&mut self, id: NodeId) {
            self.0.retain(|&x| x != id);
        }

        fn prepend(&mut self, anchor: NodeId, id: NodeId) {
            if id != anchor {
                self.remove(id);
                let at = self.position(anchor);
                self.0.insert(at, id);
            }
        }

        fn append(&mut self, anchor: NodeId, id: NodeId) {
            match self.0.get(self.position(anchor) + 1).copied() {
                Some(next) => self.prepend(next, id),
                None => {
                    self.remove(id);
                    self.0.push(id);
                }
            }
        }
    }

    fn assert_consistent(order: &NodeOrder) {
        let forward = order.to_vec();
        let mut backward: Vec<NodeId> = order.iter_rev().collect();
        backward.reverse();
        assert_eq!(forward, backward);
        assert_eq!(forward.len(), order.len());
        let keys: Vec<u64> = forward.iter().map(|&id| order.key(id).unwrap()).collect();
        assert!(keys.iter().all(|&key| key != UNPLACED));
        assert!(
            keys.windows(2).all(|pair| pair[0] < pair[1]),
            "keys must strictly increase along the order"
        );
    }

    #[test]
    fn node_order_queries_follow_the_list() {
        let order = NodeOrder::new(&[4, 0, 2]);
        assert_eq!(order.to_vec(), [4, 0, 2]);
        assert_eq!(order.iter_rev().collect::<Vec<_>>(), [2, 0, 4]);
        assert_eq!(
            (order.first(), order.last(), order.len()),
            (Some(4), Some(2), 3)
        );
        assert_eq!((order.prev(4), order.next(4)), (None, Some(0)));
        assert_eq!((order.prev(2), order.next(2)), (Some(0), None));
        assert!(order.is_before(4, 2) && !order.is_before(2, 0));
        assert_eq!(order.cmp(0, 0), Ordering::Equal);
        assert!(!order.contains(1) && !order.contains(99) && order.key(1).is_none());
        let mut ids = [2, 4, 0];
        order.sort(&mut ids);
        assert_eq!(ids, [4, 0, 2]);
        assert_consistent(&order);
        assert!(NodeOrder::new(&[]).is_empty());
    }

    /// The FX no-op cases: `n.prepend(n)`, `n.append(n)`, `n.append(n.next)`
    /// and `_root.append(head)` (torch/fx/node.py:427-452).
    #[test]
    fn node_order_self_and_neighbour_moves_are_noops() {
        let mut order = NodeOrder::new(&[0, 1, 2]);
        order.insert_before(1, 1);
        order.insert_after(1, 1);
        order.insert_after(1, 0);
        order.insert_before(0, 1);
        order.push_front(0);
        order.push_back(2);
        order.insert_after(2, 2);
        assert_eq!(order.to_vec(), [0, 1, 2]);
        assert!(order.remove(1));
        assert!(!order.remove(1));
        assert_eq!(order.to_vec(), [0, 2]);
        order.push_front(1);
        assert_eq!(order.to_vec(), [1, 0, 2]);
        assert_consistent(&order);
    }

    /// Random creates, moves and removals, with clustered inserts that exhaust
    /// key gaps, must match the vector model and keep keys increasing.
    #[test]
    fn node_order_matches_a_vector_model_under_random_edits() {
        for seed in 1..=6u64 {
            let mut rng = SplitMix64(seed);
            let base: Vec<NodeId> = (0..16).collect();
            let mut order = NodeOrder::new(&base);
            let mut model = VecOrder(base);
            let mut fresh = 16;
            let hot = 5;
            for step in 0..2500 {
                let live = &model.0;
                let anchor = live[rng.below(live.len())];
                let existing = live[rng.below(live.len())];
                let id = if rng.below(3) == 0 {
                    existing
                } else {
                    fresh += 1;
                    fresh - 1
                };
                match rng.below(8) {
                    0 | 1 => {
                        order.insert_before(id, anchor);
                        model.prepend(anchor, id);
                    }
                    2 | 3 => {
                        order.insert_after(id, anchor);
                        model.append(anchor, id);
                    }
                    4 if order.contains(hot) => {
                        order.insert_after(fresh, hot);
                        model.append(hot, fresh);
                        fresh += 1;
                    }
                    5 if order.contains(hot) => {
                        order.insert_before(fresh, hot);
                        model.prepend(hot, fresh);
                        fresh += 1;
                    }
                    6 => {
                        if rng.below(2) == 0 {
                            order.push_front(id);
                            model.0.retain(|&x| x != id);
                            model.0.insert(0, id);
                        } else {
                            order.push_back(id);
                            model.remove(id);
                            model.0.push(id);
                        }
                    }
                    _ if model.0.len() > 2 && existing != hot => {
                        assert!(order.remove(existing));
                        model.remove(existing);
                    }
                    _ => {}
                }
                assert_eq!(order.to_vec(), model.0, "seed {seed} step {step}");
                if step % 100 == 0 {
                    assert_consistent(&order);
                }
            }
            assert_consistent(&order);
            assert!(order.relabel_count() > 0, "seed {seed} never relabelled");
        }
    }

    /// The worst cases for bisection: thousands of nodes inserted directly
    /// after one node, directly before the first node and at the end.
    #[test]
    fn node_order_hot_spot_inserts_stay_ordered() {
        const N: usize = 4000;
        let mut order = NodeOrder::new(&[0, 1]);
        let mut model = VecOrder(vec![0, 1]);
        for i in 0..N {
            let (after, before, end) = (2 + 3 * i, 3 + 3 * i, 4 + 3 * i);
            order.insert_after(after, 0);
            model.append(0, after);
            order.push_front(before);
            model.0.insert(0, before);
            order.push_back(end);
            model.0.push(end);
        }
        assert_eq!(order.to_vec(), model.0);
        assert_consistent(&order);
        // Each relabel buys many bisections, so relabels stay rare.
        assert!(
            order.relabel_count() < N / 4,
            "{} relabels for {} inserts",
            order.relabel_count(),
            3 * N
        );
    }

    fn names(graph: &IrGraph, ids: impl IntoIterator<Item = NodeId>) -> String {
        ids.into_iter()
            .map(|id| graph.node(id).name.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn ctx_names(ctx: &LayoutCtx<'_>) -> String {
        names(ctx.graph(), ctx.order().iter())
    }

    /// After `finish`, node-id order, `topological_live_node_ids` and
    /// `graph_from_ir` must all produce `expected`.
    fn assert_written_back(graph: &IrGraph, expected: &str) {
        assert_eq!(
            names(graph, graph.live_node_ids()),
            expected,
            "node-id order"
        );
        assert_eq!(
            names(graph, graph.topological_live_node_ids()),
            expected,
            "topological_live_node_ids"
        );
        assert_eq!(converted_names(graph), expected, "graph_from_ir");
    }

    fn converted_names(graph: &IrGraph) -> String {
        let converted = crate::ingest::from_ir::graph_from_ir(graph, &mut TensorStore::new());
        converted
            .order
            .iter()
            .map(|&nid| converted.nodes[nid].name.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Build a graph from a spec like `graph_from` in
    /// tests/fixtures/torch_fx_insert_order.py, e.g. `"x; a=x; c=[x a]; output=c"`:
    /// argument-free names are placeholders, `[..]` is one list argument and
    /// `output` is the output node.
    fn build(spec: &str) -> (IrGraph, HashMap<String, NodeId>) {
        let mut graph = IrGraph::empty();
        let mut ids: HashMap<String, NodeId> = HashMap::new();
        for item in spec.split(';').map(str::trim) {
            let (name, args) = match item.split_once('=') {
                Some((name, args)) => (name.trim(), Some(args.trim())),
                None => (item, None),
            };
            let node_args = |text: &str| -> Vec<IrArg> {
                text.split_whitespace()
                    .map(|input| IrArg::Node(ids[input]))
                    .collect()
            };
            let id = match args {
                None => f32_placeholder(&mut graph, name, [1]),
                Some(args) => {
                    let args = match args.strip_prefix('[').and_then(|a| a.strip_suffix(']')) {
                        Some(list) => vec![IrArg::List(node_args(list))],
                        None => node_args(args),
                    };
                    if name == "output" {
                        graph.add_node(
                            NodeOp::Output,
                            "output".into(),
                            "output".into(),
                            args,
                            NodeMeta::default(),
                        )
                    } else {
                        f32_call(&mut graph, name, RELU, args, [1])
                    }
                }
            };
            ids.insert(name.to_string(), id);
        }
        (graph, ids)
    }

    fn create_named(ctx: &mut LayoutCtx<'_>, ids: &mut HashMap<String, NodeId>, new: &[&str]) {
        for name in new {
            let id = ctx.call_function(*name, RELU, Vec::new(), f32_meta([1]));
            ids.insert((*name).to_string(), id);
        }
    }

    const FX_INSERT_ORDER: &str = include_str!("../../tests/fixtures/torch_fx_insert_order.txt");

    /// Replays tests/fixtures/torch_fx_insert_order.txt, which
    /// tests/fixtures/torch_fx_insert_order.py generates by running the same
    /// op scripts through torch.fx (torch 2.15.0): every checkpoint must match
    /// FX's `graph.nodes` order, and so must the IR order after `finish`.
    #[test]
    fn ctx_replays_torch_fx_insert_order_vectors() {
        let mut scenarios = 0;
        for block in FX_INSERT_ORDER.split("\nscenario ").skip(1) {
            let mut lines = block.lines();
            let title = lines.next().unwrap();
            let base = lines.next().unwrap().strip_prefix("base ").unwrap();
            let mut graph = IrGraph::empty();
            let mut ids = HashMap::new();
            for name in base.split_whitespace() {
                ids.insert(
                    name.to_string(),
                    f32_call(&mut graph, name, RELU, vec![], [1]),
                );
            }
            let mut ctx = LayoutCtx::new(&mut graph);
            let mut expected = String::new();
            for line in lines {
                let (op, rest) = line.split_once(' ').unwrap_or((line, ""));
                let words: Vec<&str> = rest.split_whitespace().collect();
                match op {
                    "before" => {
                        let anchor = ids[words[0]];
                        ctx.inserting_before(anchor, |ctx| {
                            create_named(ctx, &mut ids, &words[1..])
                        });
                    }
                    "after" => {
                        let anchor = ids[words[0]];
                        ctx.inserting_after(anchor, |ctx| create_named(ctx, &mut ids, &words[1..]));
                    }
                    "start" => ctx.with_insert_point(InsertPoint::Start, |ctx| {
                        create_named(ctx, &mut ids, &words)
                    }),
                    "end" => create_named(&mut ctx, &mut ids, &words),
                    "prepend" => ctx.move_before(ids[words[1]], ids[words[0]]),
                    "append" => ctx.move_after(ids[words[1]], ids[words[0]]),
                    "erase" => ctx.erase_node(ids[words[0]]),
                    "check" => {
                        expected = words.join(" ");
                        assert_eq!(ctx_names(&ctx), expected, "{title}: before `{line}`");
                    }
                    other => panic!("unknown fixture op {other}"),
                }
            }
            assert!(!expected.is_empty(), "{title} has no checkpoints");
            assert_eq!(ctx.insert_point(), InsertPoint::End);
            ctx.finish();
            assert_written_back(&graph, &expected);
            scenarios += 1;
        }
        assert_eq!(scenarios, 2);
    }

    /// `unique_name` picks the first free suffix, gaps included, and keeps
    /// doing so as the names it hands out are taken.
    #[test]
    fn unique_names_take_the_first_free_suffix() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2]);
        for name in ["n", "n_1", "n_3"] {
            f32_call(&mut graph, name, RELU, vec![IrArg::Node(x)], [2]);
        }
        let mut ctx = LayoutCtx::new(&mut graph);
        assert_eq!(ctx.unique_name("fresh".into()), "fresh");
        assert_eq!(ctx.unique_name("n".into()), "n_2");
        // Asking again without taking it returns the same name.
        assert_eq!(ctx.unique_name("n".into()), "n_2");
        let names: Vec<String> = (0..4)
            .map(|_| {
                let id = ctx.call_function("n", RELU, vec![IrArg::Node(x)], f32_meta(vec![2]));
                ctx.node(id).name.clone()
            })
            .collect();
        assert_eq!(names, ["n_2", "n_4", "n_5", "n_6"]);
    }

    /// Nested insertion scopes and permanent or moving insert points; expected
    /// orders from `nested()` in tests/fixtures/torch_fx_insert_order.py
    /// (torch 2.15.0).
    #[test]
    fn ctx_insert_points_match_torch_fx_scopes() {
        let base = || {
            let mut graph = IrGraph::empty();
            let mut ids = HashMap::new();
            for name in ["a", "b", "c", "d"] {
                ids.insert(
                    name.to_string(),
                    f32_call(&mut graph, name, RELU, vec![], [1]),
                );
            }
            (graph, ids)
        };

        let (mut graph, mut ids) = base();
        let mut ctx = LayoutCtx::new(&mut graph);
        let (a, c) = (ids["a"], ids["c"]);
        ctx.inserting_after(a, |ctx| {
            create_named(ctx, &mut ids, &["p0"]);
            ctx.inserting_before(c, |ctx| create_named(ctx, &mut ids, &["q0", "q1"]));
            create_named(ctx, &mut ids, &["p1"]);
        });
        assert_eq!(ctx_names(&ctx), "a p1 p0 b q0 q1 c d", "nested_scopes");

        let (mut graph, mut ids) = base();
        let mut ctx = LayoutCtx::new(&mut graph);
        let (a, c) = (ids["a"], ids["c"]);
        ctx.inserting_after(a, |ctx| {
            create_named(ctx, &mut ids, &["p0"]);
            ctx.move_after(c, a);
            create_named(ctx, &mut ids, &["p1"]);
        });
        assert_eq!(ctx_names(&ctx), "a p1 c p0 b d", "after_anchor_next_moves");

        let (mut graph, mut ids) = base();
        let mut ctx = LayoutCtx::new(&mut graph);
        let (b, d) = (ids["b"], ids["d"]);
        ctx.inserting_before(b, |ctx| {
            create_named(ctx, &mut ids, &["p0"]);
            ctx.move_after(b, d);
            create_named(ctx, &mut ids, &["p1"]);
        });
        assert_eq!(ctx_names(&ctx), "a p0 c d p1 b", "before_anchor_moves");

        let (mut graph, mut ids) = base();
        let mut ctx = LayoutCtx::new(&mut graph);
        let b = ids["b"];
        assert_eq!(
            ctx.set_insert_point(InsertPoint::After(b)),
            InsertPoint::End
        );
        create_named(&mut ctx, &mut ids, &["p0", "p1"]);
        assert_eq!(ctx_names(&ctx), "a b p1 p0 c d", "permanent_insert_point");
        assert_eq!(ctx.insert_point(), InsertPoint::After(b));
    }

    /// PropagateViewCopyPermuteUpPass._move_node
    /// (backends/transforms/propagate_view_copy_permute_pass.py:797-822). The
    /// expected orders come from the same sequence on a torch.fx graph
    /// (torch_fx_insert_order.py --embedded: up_move_node_frontier_b and
    /// up_move_node_frontier_placeholder).
    #[test]
    fn ctx_mirrors_rc4_up_move_node() {
        for (frontier_is_placeholder, expected) in
            [(false, "x a p b output"), (true, "x p a b output")]
        {
            let (mut graph, id) = build("x; a=x; b=a; p=b; output=p");
            let mut ctx = LayoutCtx::new(&mut graph);
            let (node, original_input) = (id["p"], id["b"]);
            let (producer, frontier_user) = if frontier_is_placeholder {
                (id["x"], id["a"])
            } else {
                (id["a"], id["b"])
            };
            ctx.replace_input_with(node, original_input, producer);
            ctx.replace_input_with(frontier_user, producer, node);
            for user in ctx.graph().live_user_ids(node) {
                if user != frontier_user {
                    ctx.replace_input_with(user, node, original_input);
                }
            }
            ctx.move_before(node, frontier_user);
            assert_eq!(ctx_names(&ctx), expected);
            ctx.finish();
            assert_written_back(&graph, expected);
        }
    }

    /// PropagateViewCopyPermuteDownPass._move_node
    /// (backends/transforms/propagate_view_copy_permute_pass.py:1393-1418);
    /// torch.fx orders from down_move_node_frontier_a and
    /// down_move_node_frontier_output.
    #[test]
    fn ctx_mirrors_rc4_down_move_node() {
        for (frontier_is_output, expected) in [(false, "x a p b output"), (true, "x a b p output")]
        {
            let (mut graph, id) = build("x; p=x; a=p; b=a; output=b");
            let mut ctx = LayoutCtx::new(&mut graph);
            let node = id["p"];
            let original_user = id["a"];
            let producer = id["x"];
            let (frontier, frontier_input) = if frontier_is_output {
                (id["output"], id["b"])
            } else {
                (id["a"], id["a"])
            };
            let frontier_users = ctx.graph().live_user_ids(frontier_input);
            ctx.replace_input_with(original_user, node, producer);
            ctx.replace_input_with(node, producer, frontier_input);
            for user in frontier_users {
                if user != node {
                    ctx.replace_input_with(user, frontier_input, node);
                }
            }
            if ctx.node(frontier).op == NodeOp::Output {
                ctx.move_before(node, frontier);
            } else {
                ctx.move_after(node, frontier);
            }
            assert_eq!(ctx_names(&ctx), expected);
            ctx.finish();
            assert_written_back(&graph, expected);
        }
    }

    /// PropagateViewCopyPermuteDownPass._apply_fork_split with the frontier at
    /// the permute (backends/transforms/propagate_view_copy_permute_pass.py:
    /// 1338-1391); torch.fx order from down_fork_split.
    #[test]
    fn ctx_mirrors_rc4_down_fork_split() {
        let (mut graph, id) = build("x; p=x; a=p; b=p; output=[a b]");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (node, producer) = (id["p"], id["x"]);
        for next_node in [id["a"], id["b"]] {
            ctx.replace_input_with(next_node, node, producer);
            let name = format!("t_{}", ctx.node(next_node).name);
            let branch = ctx.inserting_after(next_node, |ctx| {
                ctx.call_function(name, RELU, vec![IrArg::Node(next_node)], f32_meta([1]))
            });
            for user in ctx.graph().live_user_ids(next_node) {
                if user != branch {
                    ctx.replace_input_with(user, next_node, branch);
                }
            }
        }
        if !ctx.graph().has_live_users(node) {
            ctx.erase_node(node);
        }
        let expected = "x a t_a b t_b output";
        assert_eq!(ctx_names(&ctx), expected);
        ctx.finish();
        assert_written_back(&graph, expected);
    }

    /// PropagateViewCopyPermuteUpPass._maybe_split_upwards_cat_fanout
    /// (backends/transforms/propagate_view_copy_permute_pass.py:824-890);
    /// torch.fx order from up_cat_fanout.
    #[test]
    fn ctx_mirrors_rc4_up_cat_fanout() {
        let (mut graph, id) = build("x1; x2; c=[x1 x2]; p=c; output=p");
        let mut ctx = LayoutCtx::new(&mut graph);
        let next_node = id["c"];
        let cat_users = ctx.graph().live_user_ids(next_node);
        let new_inputs: Vec<IrArg> = [id["x1"], id["x2"]]
            .into_iter()
            .map(|input| {
                let name = format!("q_{}", ctx.node(input).name);
                let permute = ctx.inserting_before(next_node, |ctx| {
                    ctx.call_function(name, RELU, vec![IrArg::Node(input)], f32_meta([1]))
                });
                IrArg::Node(permute)
            })
            .collect();
        ctx.set_args(next_node, vec![IrArg::List(new_inputs)]);
        for &cat_user in &cat_users {
            ctx.replace_all_uses_with(cat_user, next_node);
        }
        for cat_user in cat_users {
            if !ctx.graph().has_live_users(cat_user) {
                ctx.erase_node(cat_user);
            }
        }
        let expected = "x1 x2 q_x1 q_x2 c output";
        assert_eq!(ctx_names(&ctx), expected);
        ctx.finish();
        assert_written_back(&graph, expected);
    }

    /// MoveDataMovementOpsToSmallerDtypePass._move_before_rescale and
    /// _move_after_rescale
    /// (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:
    /// 100-137); torch.fx orders from move_before_rescale and
    /// move_after_rescale.
    #[test]
    fn ctx_mirrors_rc4_rescale_swaps() {
        let (mut graph, id) = build("x; r=x; p=r; u=p; output=u");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (data_movement, rescale, producer) = (id["p"], id["r"], id["x"]);
        let movement_users = ctx.graph().live_user_ids(data_movement);
        ctx.replace_input_with(data_movement, rescale, producer);
        ctx.replace_input_with(rescale, producer, data_movement);
        for user in movement_users {
            ctx.replace_input_with(user, data_movement, rescale);
        }
        ctx.move_after(rescale, data_movement);
        assert_eq!(ctx_names(&ctx), "x p r u output");
        ctx.finish();
        assert_written_back(&graph, "x p r u output");

        let (mut graph, id) = build("x; p=x; r=p; u=r; output=u");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (data_movement, rescale, producer) = (id["p"], id["r"], id["x"]);
        let rescale_users = ctx.graph().live_user_ids(rescale);
        ctx.replace_input_with(rescale, data_movement, producer);
        ctx.replace_input_with(data_movement, producer, rescale);
        for user in rescale_users {
            ctx.replace_input_with(user, rescale, data_movement);
        }
        ctx.move_after(data_movement, rescale);
        assert_eq!(ctx_names(&ctx), "x r p u output");
        ctx.finish();
        assert_written_back(&graph, "x r p u output");
    }

    /// MoveDataMovementOpsToSmallerDtypePass._split_after_rescale
    /// (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:
    /// 139-158); torch.fx order from split_after_rescale.
    #[test]
    fn ctx_mirrors_rc4_split_after_rescale() {
        let (mut graph, id) = build("x; p=x; r=p; v=p; u=r; output=[u v]");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (data_movement, rescale, producer) = (id["p"], id["r"], id["x"]);
        let rescale_users = ctx.graph().live_user_ids(rescale);
        ctx.replace_input_with(rescale, data_movement, producer);
        let branch = ctx.inserting_after(rescale, |ctx| {
            ctx.call_function("branch", RELU, vec![IrArg::Node(rescale)], f32_meta([1]))
        });
        for user in rescale_users {
            ctx.replace_input_with(user, rescale, branch);
        }
        if !ctx.graph().has_live_users(data_movement) {
            ctx.erase_node(data_movement);
        }
        let expected = "x p r branch v u output";
        assert_eq!(ctx_names(&ctx), expected);
        ctx.finish();
        assert_written_back(&graph, expected);
    }

    fn versions(ctx: &LayoutCtx<'_>) -> Vec<u64> {
        (0..ctx.graph().nodes.len())
            .map(|id| ctx.version(id))
            .collect()
    }

    /// Names of the nodes whose version changed since `before`.
    fn bumped(ctx: &LayoutCtx<'_>, before: &[u64]) -> String {
        let changed = (0..ctx.graph().nodes.len())
            .filter(|&id| before.get(id).copied().unwrap_or(0) != ctx.version(id));
        names(ctx.graph(), changed)
    }

    #[test]
    fn versions_track_each_kind_of_mutation() {
        let (mut graph, id) = build("x; y; a=x; b=a y; c=b; z=x; output=[c z]");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (x, y, a, b, c, z) = (id["x"], id["y"], id["a"], id["b"], id["c"], id["z"]);

        let before = versions(&ctx);
        ctx.inserting_after(a, |ctx| {
            ctx.call_function("n", RELU, vec![IrArg::Node(a)], f32_meta([1]))
        });
        assert_eq!(
            bumped(&ctx, &before),
            "a",
            "a new user; neighbours untouched"
        );

        let before = versions(&ctx);
        ctx.set_args(b, vec![IrArg::Node(x), IrArg::Node(x)]);
        assert_eq!(bumped(&ctx, &before), "x y a b", "old and new inputs");

        let before = versions(&ctx);
        ctx.set_arg(b, 1, IrArg::Node(y));
        assert_eq!(bumped(&ctx, &before), "x y b");

        let before = versions(&ctx);
        ctx.set_kwarg(c, "other", IrArg::Node(y));
        assert_eq!(bumped(&ctx, &before), "y b c");

        let before = versions(&ctx);
        ctx.meta_mut(c);
        assert_eq!(bumped(&ctx, &before), "c");

        let before = versions(&ctx);
        ctx.set_target(c, "aten.neg.default");
        assert_eq!(bumped(&ctx, &before), "c");

        let before = versions(&ctx);
        assert!(!ctx.replace_input_with(c, a, x), "c does not read a");
        assert!(!ctx.replace_input_with(c, b, b));
        assert_eq!(bumped(&ctx, &before), "");
        assert!(ctx.replace_input_with(c, b, a));
        assert_eq!(
            bumped(&ctx, &before),
            "y a b c",
            "user, old, new and kwargs"
        );

        let before = versions(&ctx);
        assert_eq!(ctx.replace_all_uses_with(z, y), [id["output"]]);
        assert_eq!(bumped(&ctx, &before), "y c z output");

        let before = versions(&ctx);
        ctx.erase_node(z);
        assert_eq!(
            bumped(&ctx, &before),
            "x z",
            "the erased node and its input"
        );
        ctx.erase_node(z);
        assert_eq!(bumped(&ctx, &before), "x z", "erasing twice is a no-op");

        let before = versions(&ctx);
        ctx.move_before(c, a);
        assert_eq!(bumped(&ctx, &before), "c", "only the moved node");
        let before = versions(&ctx);
        ctx.move_before(c, a);
        ctx.move_after(c, y);
        assert_eq!(bumped(&ctx, &before), "", "moves that keep the position");
    }

    #[test]
    fn eliminate_dead_code_keeps_order_and_versions_in_step() {
        let (mut graph, id) = build("x; a=x; b=a; c=x; output=c");
        let mut ctx = LayoutCtx::new(&mut graph);
        let before = versions(&ctx);
        assert!(ctx.eliminate_dead_code());
        assert_eq!(ctx_names(&ctx), "x c output");
        assert_eq!(
            bumped(&ctx, &before),
            "x a b",
            "erased nodes and their inputs"
        );
        assert!(ctx.is_erased(id["a"]) && ctx.is_erased(id["b"]));
        assert!(!ctx.eliminate_dead_code());
        ctx.finish();
        assert_written_back(&graph, "x c output");
    }

    const RULE: RuleId = RuleId("test.rule");
    const OTHER_RULE: RuleId = RuleId("test.other_rule");

    #[test]
    fn attempt_memo_retries_only_after_a_read_node_changes() {
        let (mut graph, id) = build("x; a=x; b=a; c=b; d=x; output=[c d]");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (x, a, b, c, d) = (id["x"], id["a"], id["b"], id["c"], id["d"]);
        assert!(ctx.should_retry(RULE, b), "never attempted");

        ctx.record_attempt(RULE, b, [a, c, a]);
        assert!(!ctx.should_retry(RULE, b));
        assert!(
            ctx.should_retry(OTHER_RULE, b),
            "rules are memoized separately"
        );

        ctx.meta_mut(d);
        ctx.inserting_after(b, |ctx| {
            ctx.call_function("n", RELU, vec![IrArg::Node(x)], f32_meta([1]))
        });
        assert!(!ctx.should_retry(RULE, b), "edits to unread nodes");

        ctx.meta_mut(c);
        assert!(ctx.should_retry(RULE, b), "a read node changed");

        ctx.record_attempt(RULE, b, [a]);
        ctx.set_target(b, "aten.neg.default");
        assert!(
            ctx.should_retry(RULE, b),
            "the anchor always counts as read"
        );

        ctx.record_attempt(RULE, b, [a]);
        ctx.inserting_after(a, |ctx| {
            ctx.call_function("m", RELU, vec![IrArg::Node(a)], f32_meta([1]))
        });
        assert!(ctx.should_retry(RULE, b), "a read node gained a user");

        ctx.replace_all_uses_with(d, x);
        ctx.record_attempt(RULE, b, [d]);
        assert!(!ctx.should_retry(RULE, b));
        ctx.erase_node(d);
        assert!(ctx.should_retry(RULE, b), "a read node was erased");

        ctx.record_attempt(RULE, b, [d]);
        assert!(
            !ctx.should_retry(RULE, b),
            "a node already erased when read does not force retries"
        );

        ctx.forget_attempt(RULE, b);
        assert!(ctx.should_retry(RULE, b));
    }

    /// A pass appended `n` and anchored it before `a`, as RETAB's rewrite
    /// cursor does. `topological_live_node_ids` honours the anchor but
    /// `graph_from_ir` ignores it, so the two disagree until a ctx writes its
    /// order back.
    #[test]
    fn finish_makes_conversion_follow_the_ctx_order() {
        let (mut graph, id) = build("x; a=x; b=a; output=b");
        let n = f32_call(&mut graph, "n", RELU, vec![IrArg::Node(id["x"])], [1]);
        graph.anchor_to(n, id["a"]);
        graph.set_args(
            id["output"],
            vec![IrArg::List(vec![IrArg::Node(id["b"]), IrArg::Node(n)])],
        );
        assert_eq!(
            names(&graph, graph.topological_live_node_ids()),
            "x n a b output"
        );
        assert_eq!(converted_names(&graph), "x a b n output");

        let ctx = LayoutCtx::new(&mut graph);
        assert_eq!(ctx_names(&ctx), "x n a b output");
        ctx.finish();
        assert_written_back(&graph, "x n a b output");

        // Later passes anchor against the renumbered graph as usual.
        let b = graph.by_name("b").unwrap().id;
        let x = graph.by_name("x").unwrap().id;
        let late = graph.rewrite_cursor(b).call_function(
            "late",
            RELU,
            vec![IrArg::Node(x)],
            f32_meta([1]),
        );
        assert_eq!(late, graph.nodes.len() - 1);
        assert_eq!(
            names(&graph, graph.topological_live_node_ids()),
            "x n a late b output"
        );
    }

    #[test]
    fn finish_renumbers_nodes_references_and_users() {
        let (mut graph, id) = build("x; a=x; b=a; c=b; output=c");
        let mut ctx = LayoutCtx::new(&mut graph);
        let (x, a, b, c) = (id["x"], id["a"], id["b"], id["c"]);
        let n = ctx.inserting_before(a, |ctx| {
            ctx.call_function("n", RELU, vec![IrArg::Node(x)], f32_meta([1]))
        });
        ctx.replace_input_with(a, x, n);
        ctx.replace_all_uses_with(b, a);
        ctx.erase_node(b);
        ctx.set_kwarg(c, "other", IrArg::Node(n));
        assert_eq!(ctx_names(&ctx), "x n a c output");
        let new_ids = ctx.finish();

        assert_written_back(&graph, "x n a c output");
        assert_eq!(new_ids.len(), graph.nodes.len());
        assert_eq!(
            [x, n, a, c, id["output"]].map(|old| new_ids[old]),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(new_ids[b], 5, "dead nodes follow the live ones");
        assert!(graph
            .nodes
            .iter()
            .enumerate()
            .all(|(index, node)| node.id == index));
        assert!(!graph.node(5).alive && graph.node(5).name == "b");
        for (name, new_id) in ["x", "n", "a", "c", "output", "b"].into_iter().zip(0..) {
            assert_eq!(graph.by_name(name).map(|node| node.id), Some(new_id));
        }
        assert_eq!(graph.node(2).args[0].as_node(), Some(1), "a reads n");
        assert_eq!(graph.node(3).args[0].as_node(), Some(2), "c reads a");
        assert_eq!(graph.node(3).kwargs["other"].as_node(), Some(1));
        assert_eq!(graph.live_user_ids(1), [2, 3]);
        assert_eq!(graph.users(0), [1]);
        assert_eq!(graph.users(2), [3]);
    }

    #[test]
    fn is_topological_rejects_a_user_before_its_input() {
        let (mut graph, id) = build("x; a=x; b=a; output=b");
        let mut ctx = LayoutCtx::new(&mut graph);
        assert!(ctx.is_topological());
        ctx.move_before(id["b"], id["a"]);
        assert!(!ctx.is_topological());
        ctx.move_after(id["b"], id["a"]);
        assert!(ctx.is_topological());
    }

    #[test]
    #[should_panic(expected = "still has live users")]
    fn erase_node_rejects_a_node_with_live_users() {
        let (mut graph, id) = build("x; a=x; output=a");
        LayoutCtx::new(&mut graph).erase_node(id["a"]);
    }

    #[test]
    #[should_panic(expected = "is not in the node order")]
    fn insert_point_anchored_on_an_erased_node_panics() {
        let (mut graph, id) = build("x; a=x; b=x; output=b");
        let mut ctx = LayoutCtx::new(&mut graph);
        let a = id["a"];
        ctx.set_insert_point(InsertPoint::Before(a));
        ctx.erase_node(a);
        ctx.call_function("n", RELU, vec![], f32_meta([1]));
    }

    #[test]
    #[should_panic(expected = "missing from the order")]
    fn reorder_nodes_requires_every_live_node() {
        let (mut graph, id) = build("x; a=x; output=a");
        graph.reorder_nodes(&[id["x"], id["output"]]);
    }

    #[test]
    #[should_panic(expected = "dead or listed twice")]
    fn reorder_nodes_rejects_a_repeated_node() {
        let (mut graph, id) = build("x; a=x; output=a");
        graph.reorder_nodes(&[id["x"], id["a"], id["a"], id["output"]]);
    }
}
