//! Large-graph checks for the propagation passes (APU-159).
//!
//! The passes run on synthetic graphs built from repeated blocks that give
//! them work in every block. The test-build checks that cost O(N) per edit
//! (the user index, and re-running the walks the memo skips) are switched
//! off, so the timings show the passes' own cost. The plain tests check the
//! results at a size where a quadratic pass would be slow; the ignored one
//! prints timings at two sizes four times apart, where a linear pass takes
//! about four times as long and a quadratic one sixteen.

use std::time::{Duration, Instant};

use crate::fx_ir::{set_check_users_after_edit, IrPass, NodeId, NodeMeta};
use crate::layout::propagate::{IrPropagateViewCopyPermutePass, CHECK_SKIPPED_WALKS, WALKS};
use crate::layout::remove_permutes::CHECK_SKIPPED_SEARCHES;
use crate::layout::testing::*;

const MM: &str = "aten.mm.default";

/// Switches the test-build checks that cost O(N) per step off while it
/// lives: the user index after every edit, and re-running each walk (and each
/// RemovePermutes search) a memo skips.
struct UncheckedEdits(bool, bool, bool);

impl UncheckedEdits {
    fn new() -> Self {
        Self(
            set_check_users_after_edit(false),
            CHECK_SKIPPED_WALKS.with(|check| check.replace(false)),
            CHECK_SKIPPED_SEARCHES.with(|check| check.replace(false)),
        )
    }
}

impl Drop for UncheckedEdits {
    fn drop(&mut self) {
        set_check_users_after_edit(self.0);
        CHECK_SKIPPED_WALKS.with(|check| check.set(self.1));
        CHECK_SKIPPED_SEARCHES.with(|check| check.set(self.2));
    }
}

fn mm(g: &mut LayoutGraph, x: NodeId, w: NodeId, shape: &[i64]) -> NodeId {
    let id = g.add_unrefreshed(MM, vec![node(x), node(w)]);
    g.graph_mut().node_mut(id).meta = NodeMeta::tensor(shape.to_vec(), "torch.float32");
    id
}

/// Per block: `permute(prev) -> relu -> neg -> mm(., w)`. Down moves each
/// permute two steps, to the matmul; five nodes per block.
fn sinking_graph(blocks: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut prev = g.input("x", &[3, 3]);
    for _ in 0..blocks {
        let w = g.input("w", &[3, 3]);
        let p = g.permute(prev, &[1, 0]);
        let r = g.call(RELU, vec![node(p)]);
        let n = g.call(NEG, vec![node(r)]);
        prev = mm(&mut g, n, w, &[3, 3]);
    }
    g.output(&[prev]);
    g
}

/// Per block: `neg(prev) -> relu -> permute -> mm(., w)`. Up moves each
/// permute two steps, to the matmul; five nodes per block.
fn hoisting_graph(blocks: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut prev = g.input("x", &[3, 3]);
    for _ in 0..blocks {
        let w = g.input("w", &[3, 3]);
        let n = g.call(NEG, vec![node(prev)]);
        let r = g.call(RELU, vec![node(n)]);
        let p = g.permute(r, &[1, 0]);
        prev = mm(&mut g, p, w, &[3, 3]);
    }
    g.output(&[prev]);
    g
}

/// Per block: a permute forking into `abs` and `neg`, rejoined by an add and
/// closed by a matmul. Down splits the permute over the fork, then sinks the
/// two identical branch permutes below the add; six nodes per block.
fn fork_graph(blocks: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut prev = g.input("x", &[4, 4]);
    for _ in 0..blocks {
        let w = g.input("w", &[4, 4]);
        let p = g.permute(prev, &[1, 0]);
        let a = g.call(ABS, vec![node(p)]);
        let n = g.call(NEG, vec![node(p)]);
        let add = g.call(ADD, vec![node(a), node(n)]);
        prev = mm(&mut g, add, w, &[4, 4]);
    }
    g.output(&[prev]);
    g
}

/// Many copies that can't move (each feeds a matmul right away) beside one
/// long chain whose permute takes a sweep per fused pair: the memo keeps the
/// stuck copies at one check per sweep.
fn stuck_copies_and_a_long_chain(stuck: usize, chain: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut outputs = Vec::new();
    for _ in 0..stuck {
        let x = g.input("x", &[3, 3]);
        let w = g.input("w", &[3, 3]);
        let p = g.permute(x, &[1, 0]);
        outputs.push(mm(&mut g, p, w, &[3, 3]));
    }
    let mut prev = g.input("y", &[2, 3, 4]);
    for step in 0..chain {
        prev = g.call(RELU, vec![node(prev)]);
        if step % 2 == 0 {
            prev = g.permute(prev, &[0, 2, 1]);
        }
    }
    outputs.push(prev);
    g.output(&outputs);
    g
}

fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

fn run(g: &mut LayoutGraph, pass: &IrPropagateViewCopyPermutePass) -> bool {
    pass.run(g.graph_mut())
}

#[test]
fn down_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 3000;
    let mut g = sinking_graph(blocks);
    assert!(run(&mut g, &IrPropagateViewCopyPermutePass::ARM_DOWN));
    // Every permute now sits right before its matmul.
    let graph = g.graph();
    for id in graph.live_call_function_ids() {
        if graph.node(id).target == PERMUTE {
            let user = graph.single_live_user(id).expect("one user");
            assert_eq!(graph.node(user).target, MM);
        }
    }
    assert_eq!(g.count(PERMUTE), blocks);
}

#[test]
fn up_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 3000;
    let mut g = hoisting_graph(blocks);
    assert!(run(&mut g, &IrPropagateViewCopyPermutePass::ARM_UP));
    // Every permute now reads the previous matmul (or the input) directly.
    let graph = g.graph();
    for id in graph.live_call_function_ids() {
        if graph.node(id).target == PERMUTE {
            let input = graph.node(id).args[0].as_node().unwrap();
            assert!(matches!(graph.node(input).target.as_str(), MM | "x"), "{}", graph.node(input).target);
        }
    }
    assert_eq!(g.count(PERMUTE), blocks);
}

#[test]
fn down_splits_forks_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 1000;
    let mut g = fork_graph(blocks);
    assert!(run(&mut g, &IrPropagateViewCopyPermutePass::ARM_DOWN));
    // Each block's permute was split over the fork and the two copies sank
    // below the add as one: abs, neg, add, permute, mm.
    let block = [ABS, NEG, ADD, PERMUTE, MM];
    let expected: Vec<&str> = std::iter::repeat_n(block, blocks).flatten().collect();
    assert_eq!(g.targets(), expected);
}

/// The stuck copies are walked once and then skipped on each later sweep.
#[test]
fn stuck_copies_are_walked_once() {
    let _unchecked = UncheckedEdits::new();
    let stuck = 2000;
    let mut g = stuck_copies_and_a_long_chain(stuck, 40);
    WALKS.with(|walks| walks.set(0));
    run(&mut g, &IrPropagateViewCopyPermutePass::ARM_UP);
    let walks = WALKS.with(std::cell::Cell::get);
    assert!(walks < stuck + 400, "{walks} walks for {stuck} stuck copies");
}

#[test]
#[ignore = "diagnostic: prints timings at two sizes; run with --ignored --nocapture"]
fn propagate_scaling_report() {
    let _unchecked = UncheckedEdits::new();
    for blocks in [2_500usize, 10_000] {
        for (label, build, pass) in [
            ("Down, sinking chains", sinking_graph as fn(usize) -> LayoutGraph, &IrPropagateViewCopyPermutePass::ARM_DOWN),
            ("Up, hoisting chains", hoisting_graph, &IrPropagateViewCopyPermutePass::ARM_UP),
            ("Down, forks", fork_graph, &IrPropagateViewCopyPermutePass::ARM_DOWN),
            ("Up, forks", fork_graph, &IrPropagateViewCopyPermutePass::ARM_UP),
        ] {
            let mut g = build(blocks);
            let nodes = g.graph().live_node_count();
            WALKS.with(|walks| walks.set(0));
            let (_, time) = timed(|| run(&mut g, pass));
            println!(
                "{label}, {nodes} nodes: {time:?}, {} walks",
                WALKS.with(std::cell::Cell::get)
            );
        }
        let mut g = stuck_copies_and_a_long_chain(blocks, 200);
        let nodes = g.graph().live_node_count();
        WALKS.with(|walks| walks.set(0));
        let (_, time) = timed(|| run(&mut g, &IrPropagateViewCopyPermutePass::ARM_UP));
        println!(
            "Up, stuck copies and a long chain, {nodes} nodes: {time:?}, {} walks",
            WALKS.with(std::cell::Cell::get)
        );
    }
}

#[test]
#[ignore = "diagnostic: times the driver's pieces on the hoisting graph"]
fn propagate_phase_probe() {
    use crate::layout::canonicalize::canonicalize_view_copy_permute;
    use crate::layout::ctx::LayoutCtx;
    use crate::layout::fuse_duplicate_users::{fuse_duplicate_users, TableStage, TOSA_EXCLUDED_TARGETS};
    use crate::layout::propagate::{propagate_iteration, Direction, PropagateConfig};
    use crate::layout::remove_permutes::{remove_permutes_in_ctx, DeadCode, RemovePermutesConfig, RemovePermutesMemo};
    let _unchecked = UncheckedEdits::new();
    for blocks in [2_500usize, 10_000] {
        let mut g = hoisting_graph(blocks);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        let (_, fdu) = timed(|| fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, TableStage::OwnTableConstants));
        let mut memo = RemovePermutesMemo::default();
        let (_, rp) = timed(|| remove_permutes_in_ctx(&mut ctx, &RemovePermutesConfig::tosa(), &mut memo, DeadCode::Full));
        let (_, canon) = timed(|| canonicalize_view_copy_permute(&mut ctx));
        let config = PropagateConfig::arm(Direction::Up);
        let (_, sweep1) = timed(|| propagate_iteration(&mut ctx, &config));
        let (_, sweep2) = timed(|| propagate_iteration(&mut ctx, &config));
        let (_, fdu2) = timed(|| fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, TableStage::OwnTableConstants));
        let (rp2_modified, rp2) = timed(|| remove_permutes_in_ctx(&mut ctx, &RemovePermutesConfig::tosa(), &mut memo, DeadCode::Full));
        println!("rp2 modified {rp2_modified}, {} permutes", ctx.order().iter().filter(|&id| ctx.node(id).target == PERMUTE).count());
        let (_, canon2) = timed(|| canonicalize_view_copy_permute(&mut ctx));
        let (_, finish) = timed(|| ctx.finish());
        println!("{blocks} blocks: fdu {fdu:?} rp {rp:?} canon {canon:?} sweep {sweep1:?} sweep2 {sweep2:?} fdu2 {fdu2:?} rp2 {rp2:?} canon2 {canon2:?} finish {finish:?}");
    }
}
