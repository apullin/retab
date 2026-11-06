//! Adversarial review tests for the remove-permutes port (APU-158 review).
//!
//! A child module of `remove_permutes`, so the tests can reach the port's
//! private caches. Tests marked `#[ignore = "BUG: ..."]` demonstrate an issue
//! and fail when run with `--ignored`; each names the rc4 evidence it encodes.

use super::*;
use crate::layout::replay;
use crate::layout::testing::{node, LayoutGraph, ADD, PERMUTE, RELU};

const REVIEW_FIXTURE: &str =
    include_str!("../../tests/fixtures/remove_permutes_review/remove_permutes_review.txt");

/// rc4 raises TypeError iterating a `None` reduction dim; see
/// `a_none_reduction_dim_fails_like_rc4`.
const NONE_DIM_CASE: &str = "sum_with_dim_none_keepdim";

/// The review cases (tests/fixtures/remove_permutes_review) match rc4, except
/// the `None` reduction dim that RETAB's importer cannot tell from `[]`.
#[test]
fn review_cases_match_rc4() {
    replay::check_known(
        &IrRemovePermutesAroundElementwiseTosaOpsPass,
        REVIEW_FIXTURE,
        &[(
            NONE_DIM_CASE,
            "rc4 raises TypeError on dim=None; the port reads None as [] (FX import's spelling)",
        )],
    );
}

/// `sum.dim_IntList(x, None, True)` in a region: rc4's `update_reduction_dim`
/// iterates `None` and raises TypeError (recorded in the review fixture); the
/// port reads `IrArg::None` as `[]` and rewrites the region. rc4's
/// DecomposeSumPass and RETAB's IrConvertSumPass both turn multi-dim sums into
/// single-dim ones before phase 6, so the pipeline should not deliver this.
#[test]
#[ignore = "BUG: a None reduction dim is rewritten as [] where rc4 raises TypeError (minor; unreachable after DecomposeSumPass/IrConvertSumPass)"]
fn a_none_reduction_dim_fails_like_rc4() {
    let results = replay::replay(
        &IrRemovePermutesAroundElementwiseTosaOpsPass,
        REVIEW_FIXTURE,
        replay::Options::default(),
    );
    let result = results
        .iter()
        .find(|result| result.name == NONE_DIM_CASE)
        .expect("the case is in the review fixture");
    assert!(result.mismatch.is_none(), "{}", result.report);
}

/// rc4 iterates `node.users` in FX insertion order. rc4's propagation passes
/// call this pass between propagation steps without a retrace, so after a step
/// rewires an input the order is no longer graph order. Here U1 = relu(y) is
/// rewired to read P (as a propagation step would, inside the caller's ctx):
/// P's FX users become [U2, U1] while graph order stays U1 < U2. The two
/// regions share E1 (U1's exit, U2's entry), so whichever is rewritten first
/// invalidates the other. rc4 (tests/fixtures/remove_permutes_review/user_order.py):
/// with users [relu, add] it leaves 2 permutes; with [add, relu] it rewrites
/// U2's region and leaves 1 permute: `relu(P)`, `add(x, relu)`. No node is
/// reached on two paths. The port visits users in the order the ctx tracks
/// (`LayoutCtx::users_in_fx_order`) and matches rc4.
#[test]
fn a_rewired_user_order_decides_which_region_is_rewritten() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[4, 4]);
    let y = g.input("y", &[4, 4]);
    let p = g.permute(x, &[1, 0]);
    let u1 = g.call(RELU, vec![node(y)]);
    let e1 = g.permute(u1, &[1, 0]);
    let u2 = g.call(ADD, vec![node(p), node(e1)]);
    let e2 = g.permute(u2, &[1, 0]);
    g.output(&[e2]);
    let names = [x, p, u1, u2].map(|id| g.name(id));
    {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        // The propagation step: U1 now reads P, and FX appends U1 to P's users.
        assert!(ctx.replace_input_with(u1, y, p));
        let modified = remove_permutes_in_ctx(
            &mut ctx,
            &RemovePermutesConfig::tosa(),
            &mut RemovePermutesMemo::default(),
            DeadCode::Full,
        );
        assert!(modified);
        ctx.finish();
    }
    let [x, p, u1, u2] = names.map(|name| g.id(&name));
    assert_eq!(g.count(PERMUTE), 1, "rc4 leaves one permute");
    let args = |id| g.graph().node(id).args.iter().filter_map(IrArg::as_node).collect::<Vec<_>>();
    assert_eq!(args(u2), [x, u1], "rc4: add(x, relu)");
    assert_eq!(args(u1), [p], "rc4: relu(P)");
}

/// The same graph built with U1 reading P from the start, so P's users are
/// in graph order, [relu, add]: rc4 (user_order.py, `rewire=False`) rewrites
/// U1's region first, which takes E1 away from U2's, and leaves 2 permutes:
/// `relu(x)`, `add(P, relu)` and U2's exit.
#[test]
fn users_in_graph_order_rewrite_the_first_region() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[4, 4]);
    g.input("y", &[4, 4]);
    let p = g.permute(x, &[1, 0]);
    let u1 = g.call(RELU, vec![node(p)]);
    let e1 = g.permute(u1, &[1, 0]);
    let u2 = g.call(ADD, vec![node(p), node(e1)]);
    let e2 = g.permute(u2, &[1, 0]);
    g.output(&[e2]);
    let names = [x, p, u1, u2].map(|id| g.name(id));
    assert!(IrRemovePermutesAroundElementwiseTosaOpsPass.run(g.graph_mut()));
    let [x, p, u1, u2] = names.map(|name| g.id(&name));
    assert_eq!(g.count(PERMUTE), 2, "rc4 leaves two permutes");
    let args = |id| g.graph().node(id).args.iter().filter_map(IrArg::as_node).collect::<Vec<_>>();
    assert_eq!(args(u1), [x], "rc4: relu(x)");
    assert_eq!(args(u2), [p, u1], "rc4: add(P, relu)");
}

/// A residual stream like `tests::residual_stream`, with a squeeze/unsqueeze
/// view pair at its head: the views are permutable but not simple, so no
/// component is closed and the component cache never applies. Every search
/// fails at the output after walking the rest of the stream.
fn open_residual_stream(blocks: usize) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let mut stream = g.permute(x, &[0, 2, 3, 1]);
    stream = g.call(RELU, vec![node(stream)]);
    let up = g.view(stream, &[1, 4, 5, 3, 1]);
    stream = g.view(up, &[1, 4, 5, 3]);
    for block in 0..blocks {
        let y = g.input(&format!("y{block}"), &[1, 3, 4, 5]);
        let branch = g.permute(y, &[0, 2, 3, 1]);
        stream = g.call(ADD, vec![node(stream), node(branch)]);
    }
    g.output(&[stream]);
    g
}

/// One call on `open_residual_stream(blocks)`: search steps, memo read-set
/// entries and wall time, with the test builds' per-edit checks off.
fn measure_open_stream(blocks: usize) -> (usize, usize, std::time::Duration) {
    let checks = (
        crate::fx_ir::set_check_users_after_edit(false),
        CHECK_SKIPPED_SEARCHES.with(|check| check.replace(false)),
    );
    let mut g = open_residual_stream(blocks);
    let config = RemovePermutesConfig::tosa();
    let mut ctx = LayoutCtx::new(g.graph_mut());
    let mut memo = RemovePermutesMemo::default();
    SEARCH_STEPS.with(|steps| steps.set(0));
    let started = std::time::Instant::now();
    let modified = remove_permutes_in_ctx(&mut ctx, &config, &mut memo, DeadCode::Local);
    let elapsed = started.elapsed();
    let steps = SEARCH_STEPS.with(std::cell::Cell::get);
    let entries = memo.entries();
    CHECK_SKIPPED_SEARCHES.with(|check| check.set(checks.1));
    crate::fx_ir::set_check_users_after_edit(checks.0);
    assert!(!modified);
    (steps, entries, elapsed)
}

/// On this stream each of the N searches reads O(N) nodes before it fails,
/// as in rc4. When every failed search kept its whole read set (in the
/// per-call cache and the cross-call memo), memory grew quadratically on top
/// of rc4's quadratic walk, where rc4 keeps nothing: 500, 1000, 2000 and
/// 4000 blocks held 0.25M, 1.0M, 4.0M and 16.0M read-set entries (up to 244
/// MiB). Searches with more than `EXACT_READS_CAP` reads now share one
/// union, and the same streams hold 22,759, 23,759, 25,759 and 29,759
/// entries for the same 0.13M to 8.0M search steps
/// (`open_stream_scaling_report` prints the release table).
#[test]
fn failed_search_memory_grows_linearly_on_an_open_residual_stream() {
    let (_, small, _) = measure_open_stream(250);
    let (_, large, _) = measure_open_stream(500);
    assert!(
        large <= 5 * small / 2,
        "{small} -> {large} memo read-set entries when the stream doubles"
    );
}

/// Scaling report for the open residual stream:
/// `cargo test --release --lib open_stream_scaling_report -- --ignored --nocapture`.
#[test]
#[ignore = "scaling report"]
fn open_stream_scaling_report() {
    for blocks in [500, 1000, 2000, 4000] {
        let (steps, entries, elapsed) = measure_open_stream(blocks);
        println!(
            "open residual stream, {blocks} blocks: {steps} search steps, {entries} memo read-set entries ({} MiB), {elapsed:?}",
            entries * std::mem::size_of::<(NodeId, u64)>() / (1 << 20)
        );
    }
}

/// Steps of each of two calls of the pass on one ctx over
/// `open_residual_stream(blocks)`, with `between` run on the ctx between them.
/// Test builds re-run every skipped search to check that it fails.
fn two_calls_on_an_open_stream(
    blocks: usize,
    between: impl FnOnce(&mut LayoutCtx<'_>, &[NodeId]),
) -> (usize, usize) {
    let users = crate::fx_ir::set_check_users_after_edit(false);
    let mut g = open_residual_stream(blocks);
    let stream: Vec<NodeId> = g.graph().topological_live_node_ids();
    let config = RemovePermutesConfig::tosa();
    let mut ctx = LayoutCtx::new(g.graph_mut());
    let mut memo = RemovePermutesMemo::default();
    let mut run = |ctx: &mut LayoutCtx<'_>| {
        SEARCH_STEPS.with(|steps| steps.set(0));
        assert!(!remove_permutes_in_ctx(ctx, &config, &mut memo, DeadCode::Local));
        SEARCH_STEPS.with(std::cell::Cell::get)
    };
    let first = run(&mut ctx);
    between(&mut ctx, &stream);
    let second = run(&mut ctx);
    crate::fx_ir::set_check_users_after_edit(users);
    (first, second)
}

/// The searches that read more than `EXACT_READS_CAP` nodes share the union,
/// which proves them all again on an unchanged ctx.
#[test]
fn the_union_skips_an_unchanged_open_stream() {
    let (first, second) = two_calls_on_an_open_stream(300, |_, _| {});
    assert!(first > 300 * 150, "{first} steps");
    assert_eq!(second, 0);
}

/// Changing a node only the search from the head read (the squeeze view
/// pair is upstream of every other start) invalidates the union, so every
/// search it proves runs again; the searches near the output keep their own
/// read sets and stay skipped.
#[test]
fn a_changed_union_node_reruns_only_the_union_searches() {
    let (first, second) = two_calls_on_an_open_stream(300, |ctx, stream| {
        let view = stream
            .iter()
            .copied()
            .find(|&id| ctx.node(id).target == "aten.view_copy.default")
            .expect("the head view");
        ctx.meta_mut(view);
    });
    assert!(second > 0, "the union searches run again");
    assert!(second < first, "{second} of {first} steps: the small searches stay skipped");
}
