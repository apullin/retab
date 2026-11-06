//! Large-graph checks for FuseDuplicateUsers, FuseIdenticalInputTransforms
//! and MoveDataMovementOpsToSmallerDtype (APU-160).
//!
//! Each pass runs on synthetic graphs built from repeated blocks that give it
//! work in every block. The test-build checks that cost O(N) per edit (the
//! user index and MoveDataMovement's ready set) are switched off, so the
//! timings show the passes' own cost. The plain tests check the results at a
//! size where a quadratic pass would be slow; the ignored one prints timings
//! at two sizes four times apart, where a linear pass takes about four times
//! as long and a quadratic one sixteen.

use std::time::{Duration, Instant};

use crate::fx_ir::{set_check_users_after_edit, IrPass};
use crate::layout::ctx::LayoutCtx;
use crate::layout::fuse_duplicate_users::{fuse_duplicate_users, IrFuseDuplicateUsersPass, TOSA_EXCLUDED_TARGETS};
use crate::layout::fuse_identical_input_transforms::{
    fuse_identical_input_transforms, FuseIdenticalInputTransforms, IrFuseIdenticalInputTransformsPass,
};
use crate::layout::move_data_movement::{
    move_data_movement_ops_to_smaller_dtype, set_check_ready_set, IrMoveDataMovementOpsToSmallerDtypePass,
};
use crate::layout::testing::*;

/// Switches the O(N)-per-edit test checks off while it lives.
struct UncheckedEdits(bool, bool);

impl UncheckedEdits {
    fn new() -> Self {
        Self(set_check_users_after_edit(false), set_check_ready_set(false))
    }
}

impl Drop for UncheckedEdits {
    fn drop(&mut self) {
        set_check_users_after_edit(self.0);
        set_check_ready_set(self.1);
    }
}

/// Per block: `y = neg(prev)`, two identical `add(y, c)`, a `relu` on each
/// (duplicates once the adds fuse) and their product. Six nodes per block
/// (seven without `shared`); the pass fuses two of them. With `shared`, one
/// `c` serves every block, so its user list grows with the graph.
fn duplicate_users_graph(blocks: usize, shared: bool) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let shared_c = g.input("c", &[4]);
    let mut prev = g.input("x", &[4]);
    for _ in 0..blocks {
        let c = if shared { shared_c } else { g.input("c", &[4]) };
        let y = g.call(NEG, vec![node(prev)]);
        let a0 = g.call(ADD, vec![node(y), node(c)]);
        let a1 = g.call(ADD, vec![node(y), node(c)]);
        let r0 = g.call(RELU, vec![node(a0)]);
        let r1 = g.call(RELU, vec![node(a1)]);
        prev = g.call(MUL, vec![node(r0), node(r1)]);
    }
    g.output(&[prev]);
    g
}

/// Per block: `permute(prev) + permute(y)` with identical permutes, and a
/// permute back. Four nodes per block (five without `shared`); the pass sinks
/// one permute pair. With `shared`, one `y` serves every block.
fn identical_transforms_graph(blocks: usize, shared: bool) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let shared_y = g.input("y", &[2, 3]);
    let mut prev = g.input("x", &[2, 3]);
    for _ in 0..blocks {
        let y = if shared { shared_y } else { g.input("y", &[2, 3]) };
        let p1 = g.permute(prev, &[1, 0]);
        let p2 = g.permute(y, &[1, 0]);
        let sum = g.call(ADD, vec![node(p1), node(p2)]);
        prev = g.permute(sum, &[1, 0]);
    }
    g.output(&[prev]);
    g
}

/// Per block: a permute between an int32 value and a narrowing RESCALE, then
/// a widening one. Three nodes per block; every permute moves at least once,
/// and each rescan from the top in rc4 would cost O(N).
fn data_movement_graph(blocks: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut prev = g.typed_input("x", &[2, 3, 4], "torch.int32");
    for _ in 0..blocks {
        let p = g.permute(prev, &[2, 0, 1]);
        let narrow = g.rescale(p, "torch.int8", 0.5);
        prev = g.rescale(narrow, "torch.int32", 2.0);
    }
    g.output(&[prev]);
    g
}

fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

#[test]
fn fuse_duplicate_users_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 4000;
    let mut g = duplicate_users_graph(blocks, false);
    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));
    assert_eq!((g.count(ADD), g.count(RELU)), (blocks, blocks));
}

#[test]
fn fuse_identical_input_transforms_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 3000;
    let mut g = identical_transforms_graph(blocks, false);
    assert!(IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));
    // One sunk permute and the permute back per block.
    assert_eq!(g.count(PERMUTE), 2 * blocks);
    assert_eq!(g.count(ADD), blocks);
}

#[test]
fn move_data_movement_on_a_large_graph() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 3000;
    let mut g = data_movement_graph(blocks);
    assert!(IrMoveDataMovementOpsToSmallerDtypePass.run(g.graph_mut()));
    // Every permute ends up on int8 data.
    let graph = g.graph();
    for id in graph.live_call_function_ids() {
        if graph.node(id).target == PERMUTE {
            assert_eq!(graph.node_dtype(id), Some("torch.int8"));
        }
    }
}

/// The memo makes a repeated ctx call, as the propagation driver makes after
/// every sweep, cheap: nothing it read has changed.
#[test]
fn repeated_ctx_calls_find_nothing_new() {
    let _unchecked = UncheckedEdits::new();
    let mut g = duplicate_users_graph(500, true);
    let mut ctx = LayoutCtx::new(g.graph_mut());
    assert!(fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
    assert!(!fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
    ctx.finish();
    let mut g = identical_transforms_graph(500, true);
    let mut ctx = LayoutCtx::new(g.graph_mut());
    assert!(fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER));
    assert!(!fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER));
    ctx.finish();
}

#[test]
#[ignore = "diagnostic: prints timings at two sizes; run with --ignored --nocapture"]
fn fuse_helpers_scaling_report() {
    let _unchecked = UncheckedEdits::new();
    let mut rows: Vec<String> = Vec::new();
    for shared in [false, true] {
        for blocks in [5_000usize, 20_000] {
            let label = if shared { "one shared input" } else { "inputs per block" };
            let mut g = duplicate_users_graph(blocks, shared);
            let nodes = g.graph().live_node_count();
            let (_, pass) = timed(|| IrFuseDuplicateUsersPass.run(g.graph_mut()));
            let mut g = duplicate_users_graph(blocks, shared);
            let mut ctx = LayoutCtx::new(g.graph_mut());
            fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS);
            let (_, again) = timed(|| fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
            rows.push(format!(
                "FuseDuplicateUsers, {label}, {nodes} nodes: pass {pass:?}, repeated ctx call {again:?}"
            ));

            let mut g = identical_transforms_graph(blocks, shared);
            let nodes = g.graph().live_node_count();
            let (_, pass) = timed(|| IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));
            let mut g = identical_transforms_graph(blocks, shared);
            let mut ctx = LayoutCtx::new(g.graph_mut());
            fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER);
            let (_, again) =
                timed(|| fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER));
            rows.push(format!(
                "FuseIdenticalInputTransforms, {label}, {nodes} nodes: pass {pass:?}, repeated ctx call {again:?}"
            ));

            if !shared {
                let mut g = data_movement_graph(blocks);
                let nodes = g.graph().live_node_count();
                let (_, pass) = timed(|| IrMoveDataMovementOpsToSmallerDtypePass.run(g.graph_mut()));
                rows.push(format!("MoveDataMovementOpsToSmallerDtype, {nodes} nodes: pass {pass:?}"));
            }
        }
    }
    for row in rows {
        println!("{row}");
    }
}

/// At a fixed size, time grows with the rewrites, not with rewrites times
/// nodes as rc4's rescan from the top does.
#[test]
#[ignore = "diagnostic: MoveDataMovement time against the number of rewrites at a fixed size"]
fn move_data_movement_rewrite_probe() {
    let _unchecked = UncheckedEdits::new();
    let blocks = 40_000;
    for moving in [1_000usize, 4_000, 16_000, 40_000] {
        let mut g = LayoutGraph::new();
        let mut prev = g.typed_input("x", &[2, 3, 4], "torch.int32");
        for block in 0..blocks {
            let p = g.permute(prev, &[2, 0, 1]);
            let narrow = g.rescale(p, if block < moving { "torch.int8" } else { "torch.int32" }, 0.5);
            prev = g.rescale(narrow, "torch.int32", 2.0);
        }
        g.output(&[prev]);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        let (_, pass) = timed(|| move_data_movement_ops_to_smaller_dtype(&mut ctx));
        println!("MDM {} nodes, {moving} moving blocks: pass {pass:?}", 3 * blocks + 2);
    }
}
