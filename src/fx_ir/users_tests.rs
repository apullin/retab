//! Tests for `IrGraph`'s incremental user index.
//!
//! Seeded random edits are checked against a from-scratch rebuild and against the
//! user bookkeeping the graph had before the index became incremental (RETAB main
//! @ 5c911c8). An ignored test times `set_arg` on a 20k-node graph both ways.

use std::time::{Duration, Instant};

use super::{
    build_user_edges, collect_users_from_arg, IrArg, IrGraph, NodeId, NodeMeta, NodeOp,
    CHECK_USERS_AFTER_EDIT,
};

/// SplitMix64, so the tests need no RNG crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn percent(&mut self, chance: u64) -> bool {
        self.next_u64() % 100 < chance
    }
}

/// The user cache as `IrGraph` kept it before the index became incremental.
/// `set_args`, `set_arg` (index in range), `replace_input_use` (when it changed
/// something) and `eliminate_dead_code` (when it erased something) rebuilt it from
/// scratch. `add_node` appended the new id to the lists of the nodes in its args.
/// `replace_all_uses` moved the old id's list onto the end of the new id's list.
/// Nothing else touched it.
struct OldUserCache(Vec<Vec<NodeId>>);

impl OldUserCache {
    fn rebuild(&mut self, graph: &IrGraph) {
        self.0 = build_user_edges(&graph.nodes);
    }

    /// Mirror `add_node`; call it before the graph creates node `id`.
    fn add_node(&mut self, id: NodeId, args: &[IrArg]) {
        for arg in args {
            collect_users_from_arg(arg, id, &mut self.0);
        }
        self.0.push(Vec::new());
    }

    fn replace_all_uses(&mut self, old: NodeId, new: NodeId) {
        let moved = std::mem::take(&mut self.0[old]);
        for user in moved {
            if !self.0[new].contains(&user) {
                self.0[new].push(user);
            }
        }
    }
}

/// `live_user_ids` before the index: a scan over every live node.
fn scanned_live_users(graph: &IrGraph, id: NodeId) -> Vec<NodeId> {
    graph
        .live_nodes()
        .filter(|node| {
            node.args.iter().any(|arg| arg.contains_node(id))
                || node.kwargs.values().any(|arg| arg.contains_node(id))
        })
        .map(|node| node.id)
        .collect()
}

/// `live_getitem_users` before the index.
fn scanned_getitem_users(graph: &IrGraph, id: NodeId) -> Vec<(NodeId, i64)> {
    graph
        .live_nodes()
        .filter(|node| node.is_getitem())
        .filter(|node| node.first_node_arg() == Some(id))
        .map(|node| (node.id, node.arg_int_or(1, 0)))
        .collect()
}

/// The index matches a from-scratch rebuild, `users(id)` matches the old cache for
/// every id, and the queries for `ids` match the old full scans.
fn assert_old_answers(graph: &IrGraph, old: &OldUserCache, ids: &[NodeId]) {
    graph.assert_users_consistent();
    assert_eq!(old.0.len(), graph.nodes.len());
    for (id, cached) in old.0.iter().enumerate() {
        assert_eq!(graph.users(id), cached.as_slice(), "users({id})");
    }
    for &id in ids {
        let scanned = scanned_live_users(graph, id);
        assert_eq!(graph.live_user_ids(id), scanned, "live_user_ids({id})");
        assert_eq!(graph.has_live_users(id), !scanned.is_empty());
        assert_eq!(
            graph.single_live_user(id),
            (scanned.len() == 1).then(|| scanned[0])
        );
        let call_users: Vec<NodeId> = scanned
            .iter()
            .copied()
            .filter(|&user| graph.nodes[user].op == NodeOp::CallFunction)
            .collect();
        assert_eq!(graph.live_call_function_users(id), call_users);
        assert_eq!(
            graph.live_getitem_users(id),
            scanned_getitem_users(graph, id)
        );
    }
}

const GETITEM: &str = "<built-in function getitem>";
const KWARG_KEYS: [&str; 3] = ["other", "dtype", "self"];

/// Mostly an existing id; now and then the next id (a self-reference when given to
/// `add_node`) or the one after it (a forward reference).
fn random_ref(rng: &mut Rng, node_count: usize) -> NodeId {
    match rng.below(60) {
        0 => node_count,
        1 => node_count + 1,
        _ => rng.below(node_count),
    }
}

fn random_arg(rng: &mut Rng, node_count: usize, depth: usize) -> IrArg {
    match rng.below(10) {
        0..=4 => IrArg::Node(random_ref(rng, node_count)),
        5 | 6 if depth < 2 => IrArg::List(
            (0..rng.below(4))
                .map(|_| random_arg(rng, node_count, depth + 1))
                .collect(),
        ),
        7 => IrArg::Int(rng.below(3) as i64),
        8 => IrArg::Float(1.5),
        _ => IrArg::None,
    }
}

fn random_args(rng: &mut Rng, node_count: usize) -> Vec<IrArg> {
    (0..1 + rng.below(3))
        .map(|_| random_arg(rng, node_count, 0))
        .collect()
}

fn random_kwargs(rng: &mut Rng, node_count: usize) -> std::collections::HashMap<String, IrArg> {
    (0..1 + rng.below(2))
        .map(|_| {
            let key = KWARG_KEYS[rng.below(KWARG_KEYS.len())].to_string();
            (key, random_arg(rng, node_count, 0))
        })
        .collect()
}

/// A random node to add: a call (mostly), a getitem, an output or an "other" node.
fn random_node(rng: &mut Rng, node_count: usize) -> (NodeOp, String, Vec<IrArg>) {
    match rng.below(10) {
        0..=5 => (
            NodeOp::CallFunction,
            "aten.add.Tensor".to_string(),
            random_args(rng, node_count),
        ),
        6 | 7 => (
            NodeOp::CallFunction,
            GETITEM.to_string(),
            vec![
                IrArg::Node(rng.below(node_count)),
                IrArg::Int(rng.below(3) as i64),
            ],
        ),
        8 => (
            NodeOp::Output,
            "output".to_string(),
            random_args(rng, node_count),
        ),
        _ => (
            NodeOp::Other,
            "other".to_string(),
            random_args(rng, node_count),
        ),
    }
}

/// Apply one random edit to the graph and mirror it in the old cache.
fn random_edit(rng: &mut Rng, graph: &mut IrGraph, old: &mut OldUserCache) {
    let node_count = graph.nodes.len();
    let name = format!("n{node_count}");
    match rng.below(100) {
        0..=15 => {
            let (op, target, args) = random_node(rng, node_count);
            old.add_node(node_count, &args);
            graph.add_node(op, name, target, args, NodeMeta::default());
        }
        16..=21 => {
            let (op, target, args) = random_node(rng, node_count);
            let kwargs = random_kwargs(rng, node_count);
            old.add_node(node_count, &args);
            graph.add_node_with_kwargs(op, name, target, args, kwargs, NodeMeta::default());
        }
        22..=37 => {
            let id = rng.below(node_count);
            // One past the end now and then: the old code ignored that edit.
            let index = rng.below(graph.nodes[id].args.len() + 1);
            let value = random_arg(rng, node_count, 0);
            let in_range = index < graph.nodes[id].args.len();
            graph.set_arg(id, index, value);
            if in_range {
                old.rebuild(graph);
            }
        }
        38..=42 => {
            let id = rng.below(node_count);
            let args = random_args(rng, node_count);
            graph.set_args(id, args);
            old.rebuild(graph);
        }
        43..=48 => {
            let id = rng.below(node_count);
            let key = KWARG_KEYS[rng.below(KWARG_KEYS.len())];
            let value = random_arg(rng, node_count, 0);
            graph.set_kwarg(id, key, value);
        }
        49..=51 => {
            let id = rng.below(node_count);
            graph.remove_kwarg(id, KWARG_KEYS[rng.below(KWARG_KEYS.len())]);
        }
        52..=56 => {
            // An in-place edit through `node_mut`, then a resync.
            let id = rng.below(node_count);
            let value = random_arg(rng, node_count, 0);
            let arg_count = graph.nodes[id].args.len();
            if arg_count > 0 && rng.percent(60) {
                let index = rng.below(arg_count);
                graph.node_mut(id).args[index] = value;
            } else {
                graph
                    .node_mut(id)
                    .kwargs
                    .insert("inplace".to_string(), value);
            }
            graph.sync_node_users(id);
        }
        57..=66 => {
            let user = rng.below(node_count);
            let operands = graph.operands[user].clone();
            let old_id = if !operands.is_empty() && rng.percent(80) {
                operands[rng.below(operands.len())]
            } else {
                rng.below(node_count)
            };
            let new_id = random_ref(rng, node_count);
            if graph.replace_input_use(user, old_id, new_id) {
                old.rebuild(graph);
            }
        }
        67..=72 => {
            let (old_id, new_id) = (rng.below(node_count), rng.below(node_count));
            graph.replace_all_uses(old_id, new_id);
            old.replace_all_uses(old_id, new_id);
        }
        73..=80 => {
            let (old_id, new_id) = (rng.below(node_count), rng.below(node_count));
            graph.replace_and_erase(old_id, new_id);
            old.replace_all_uses(old_id, new_id);
        }
        81..=84 => {
            let (old_id, new_id) = (rng.below(node_count), rng.below(node_count));
            let excluded = rng.below(node_count);
            if graph.replace_live_uses_except(old_id, new_id, excluded) {
                old.rebuild(graph);
            }
        }
        85..=96 => graph.erase(rng.below(node_count)),
        97 => {
            graph.rebuild_users();
            old.rebuild(graph);
        }
        _ => {
            // Dead-code elimination indexes every referenced id, so it cannot run
            // while a forward reference is outstanding (nor could it before).
            if graph.pending_users.is_empty() && graph.eliminate_dead_code() {
                old.rebuild(graph);
            }
        }
    }
}

/// A random starting graph: a few placeholders, random calls, and an output.
fn random_graph(rng: &mut Rng) -> (IrGraph, OldUserCache) {
    let mut graph = IrGraph::empty();
    let mut old = OldUserCache(Vec::new());
    for index in 0..3 + rng.below(4) {
        old.add_node(graph.nodes.len(), &[]);
        graph.add_node(
            NodeOp::Placeholder,
            format!("x{index}"),
            format!("x{index}"),
            Vec::new(),
            NodeMeta::default(),
        );
    }
    for _ in 0..10 + rng.below(20) {
        let node_count = graph.nodes.len();
        let args: Vec<IrArg> = (0..1 + rng.below(3))
            .map(|_| IrArg::Node(rng.below(node_count)))
            .collect();
        old.add_node(node_count, &args);
        graph.add_node(
            NodeOp::CallFunction,
            format!("n{node_count}"),
            "aten.add.Tensor".to_string(),
            args,
            NodeMeta::default(),
        );
    }
    let last = graph.nodes.len() - 1;
    let args = vec![IrArg::List(vec![IrArg::Node(last)])];
    old.add_node(graph.nodes.len(), &args);
    graph.add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        args,
        NodeMeta::default(),
    );
    (graph, old)
}

/// Thousands of seeded random edits keep the index equal to a from-scratch rebuild,
/// `users(id)` equal to the old cache, and `live_user_ids` and friends equal to the
/// old full scans (checked for a sample of ids after each edit and for every id after
/// every 20th).
#[test]
fn random_edits_keep_user_queries_identical_to_the_old_bookkeeping() {
    let mut edits = 0;
    for seed in 0..24 {
        let mut rng = Rng(0x5EED_0000 + seed);
        let (mut graph, mut old) = random_graph(&mut rng);
        for step in 0..400 {
            random_edit(&mut rng, &mut graph, &mut old);
            // Ids up to two past the end cover forward references.
            let end = graph.nodes.len() + 3;
            let ids: Vec<NodeId> = if step % 20 == 0 {
                (0..end).collect()
            } else {
                (0..6).map(|_| rng.below(end)).collect()
            };
            assert_old_answers(&graph, &old, &ids);
            edits += 1;
        }
        let end = graph.nodes.len() + 3;
        assert_old_answers(&graph, &old, &(0..end).collect::<Vec<_>>());
    }
    assert_eq!(edits, 9_600);
}

#[test]
fn users_keeps_erased_users_until_an_old_rebuild_point() {
    let mut graph = IrGraph::empty();
    let x = graph.add_node(
        NodeOp::Placeholder,
        "x".into(),
        "x".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    let a = graph.add_call_function(
        "a",
        "aten.neg.default",
        vec![IrArg::Node(x)],
        NodeMeta::default(),
    );
    let b = graph.add_call_function(
        "b",
        "aten.abs.default",
        vec![IrArg::Node(x)],
        NodeMeta::default(),
    );

    graph.erase(a);
    assert_eq!(graph.users(x), &[a, b]);
    assert_eq!(graph.live_user_ids(x), vec![b]);

    // Any in-range `set_arg` rebuilt the old cache.
    graph.set_arg(b, 0, IrArg::Node(x));
    assert_eq!(graph.users(x), &[b]);
}

#[test]
fn users_lists_replace_all_uses_moves_after_the_existing_users() {
    // x feeds a no-op n and a later node v; z reads n.
    let mut graph = IrGraph::empty();
    let x = graph.add_node(
        NodeOp::Placeholder,
        "x".into(),
        "x".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    let n = graph.add_call_function(
        "n",
        "aten.clone.default",
        vec![IrArg::Node(x)],
        NodeMeta::default(),
    );
    let z = graph.add_call_function(
        "z",
        "aten.neg.default",
        vec![IrArg::Node(n)],
        NodeMeta::default(),
    );
    let v = graph.add_call_function(
        "v",
        "aten.abs.default",
        vec![IrArg::Node(x)],
        NodeMeta::default(),
    );

    graph.replace_and_erase(n, x);
    assert_eq!(graph.users(x), &[n, v, z]);
    assert_eq!(graph.live_user_ids(x), vec![z, v]);
    assert_eq!(graph.users(n), &[] as &[NodeId]);

    graph.set_args(v, vec![IrArg::Node(x)]);
    assert_eq!(graph.users(x), &[z, v]);
}

#[test]
fn users_ignores_kwargs_references_until_an_old_rebuild_point() {
    let mut graph = IrGraph::empty();
    let x = graph.add_node(
        NodeOp::Placeholder,
        "x".into(),
        "x".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    let y = graph.add_node(
        NodeOp::Placeholder,
        "y".into(),
        "y".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("other".to_string(), IrArg::Node(y));
    let add = graph.add_call_function_with_kwargs(
        "add",
        "aten.add.Tensor",
        vec![IrArg::Node(x)],
        kwargs,
        NodeMeta::default(),
    );
    assert_eq!(graph.live_user_ids(y), vec![add]);
    assert_eq!(graph.users(y), &[] as &[NodeId]);

    graph.remove_kwarg(add, "other");
    assert!(graph.live_user_ids(y).is_empty());
    assert_eq!(graph.users(y), &[] as &[NodeId]);

    graph.set_kwarg(add, "other", IrArg::Node(y));
    graph.set_arg(add, 0, IrArg::Node(x));
    assert_eq!(graph.users(y), &[add]);
}

#[test]
fn forward_and_self_references_match_a_rebuild() {
    let mut graph = IrGraph::empty();
    let x = graph.add_node(
        NodeOp::Placeholder,
        "x".into(),
        "x".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    // `a` reads itself and the node after it.
    let a_id = graph.nodes.len();
    let a = graph.add_call_function(
        "a",
        "aten.add.Tensor",
        vec![IrArg::Node(a_id), IrArg::Node(a_id + 1)],
        NodeMeta::default(),
    );
    assert_eq!(graph.live_user_ids(a), vec![a]);
    assert_eq!(graph.live_user_ids(a + 1), vec![a]);
    // The old cache skipped both references when `a` was added.
    assert_eq!(graph.users(a), &[] as &[NodeId]);

    let b = graph.add_call_function(
        "b",
        "aten.neg.default",
        vec![IrArg::Node(x)],
        NodeMeta::default(),
    );
    assert_eq!(b, a + 1);
    assert!(graph.pending_users.is_empty());
    assert_eq!(graph.live_user_ids(b), vec![a]);
    assert_eq!(graph.users(b), &[] as &[NodeId]);

    graph.set_arg(b, 0, IrArg::Node(x));
    assert_eq!(graph.users(a), &[a]);
    assert_eq!(graph.users(b), &[a]);
}

/// Switches off the per-edit index check while it lives.
struct UncheckedEdits;

impl UncheckedEdits {
    fn new() -> Self {
        CHECK_USERS_AFTER_EDIT.with(|check| check.set(false));
        Self
    }
}

impl Drop for UncheckedEdits {
    fn drop(&mut self) {
        CHECK_USERS_AFTER_EDIT.with(|check| check.set(true));
    }
}

/// Node `i` reads nodes `i - 1` and `i / 2`.
fn chain_graph(node_count: usize) -> IrGraph {
    let mut graph = IrGraph::empty();
    graph.add_node(
        NodeOp::Placeholder,
        "x".into(),
        "x".into(),
        Vec::new(),
        NodeMeta::default(),
    );
    for id in 1..node_count {
        graph.add_call_function(
            format!("n{id}"),
            "aten.add.Tensor",
            vec![IrArg::Node(id - 1), IrArg::Node(id / 2)],
            NodeMeta::default(),
        );
    }
    graph
}

/// Point random args at random earlier nodes with `set_arg`, and ask for the new
/// operand's users after each edit. Returns the total user count seen.
fn rewire_and_query(graph: &mut IrGraph, edits: usize, seed: u64) -> usize {
    let mut rng = Rng(seed);
    let mut seen = 0;
    for _ in 0..edits {
        let user = 1 + rng.below(graph.nodes.len() - 1);
        let operand = rng.below(user);
        graph.set_arg(user, rng.below(2), IrArg::Node(operand));
        seen += graph.live_user_ids(operand).len();
    }
    seen
}

/// The same workload done the old way: `set_arg` rebuilt every user list and
/// `live_user_ids` scanned every node.
fn rewire_and_query_with_rebuilds(graph: &mut IrGraph, edits: usize, seed: u64) -> usize {
    let mut rng = Rng(seed);
    let mut seen = 0;
    for _ in 0..edits {
        let user = 1 + rng.below(graph.nodes.len() - 1);
        let operand = rng.below(user);
        graph.nodes[user].args[rng.below(2)] = IrArg::Node(operand);
        graph.users = build_user_edges(&graph.nodes);
        seen += scanned_live_users(graph, operand).len();
    }
    seen
}

fn per_edit(total: Duration, edits: usize) -> f64 {
    total.as_secs_f64() * 1e6 / edits as f64
}

/// `cargo test --lib set_arg_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing printout"]
fn set_arg_timing_on_a_20k_node_graph() {
    const NODES: usize = 20_000;
    const EDITS: usize = 2_000;
    let _unchecked = UncheckedEdits::new();

    let mut rebuilt = chain_graph(NODES);
    let start = Instant::now();
    let rebuilt_seen = rewire_and_query_with_rebuilds(&mut rebuilt, EDITS, 7);
    let rebuilt_time = start.elapsed();

    let mut incremental = chain_graph(NODES);
    let start = Instant::now();
    let incremental_seen = rewire_and_query(&mut incremental, EDITS, 7);
    let incremental_time = start.elapsed();
    assert_eq!(incremental_seen, rebuilt_seen);
    incremental.assert_users_consistent();

    let long_edits = 100 * EDITS;
    let start = Instant::now();
    rewire_and_query(&mut incremental, long_edits, 11);
    let long_time = start.elapsed();
    incremental.assert_users_consistent();

    println!(
        "{NODES} nodes, {EDITS} set_arg + live_user_ids ({incremental_seen} users seen): \
         rebuild-and-scan {:.3} s ({:.1} us/edit), incremental {:.4} s ({:.2} us/edit), {:.0}x; \
         incremental {long_edits} edits {:.3} s ({:.2} us/edit)",
        rebuilt_time.as_secs_f64(),
        per_edit(rebuilt_time, EDITS),
        incremental_time.as_secs_f64(),
        per_edit(incremental_time, EDITS),
        rebuilt_time.as_secs_f64() / incremental_time.as_secs_f64(),
        long_time.as_secs_f64(),
        per_edit(long_time, long_edits),
    );
}
