//! rc4 replays of `RemovePermutesAroundElementwiseTosaOps` (APU-158): every
//! case of tests/fixtures/layout_replay/remove_permutes_around_elementwise.txt
//! must match what the real rc4 pass did (see `replay` for what is compared),
//! except the divergences listed with their reasons. So must every case of
//! tests/fixtures/layout_replay/remove_permutes_user_order.txt, whose graphs
//! have users out of graph order.

use crate::fx_ir::{IrGraph, NodeId};
use crate::layout::ctx::LayoutCtx;
use crate::layout::remove_permutes::{
    remove_permutes_in_ctx, DeadCode, IrRemovePermutesAroundElementwiseTosaOpsPass,
    RemovePermutesConfig, RemovePermutesMemo,
};
use crate::layout::replay::{self, DimOrders, Options};

const FIXTURE: &str =
    include_str!("../../tests/fixtures/layout_replay/remove_permutes_around_elementwise.txt");

const USER_ORDER_FIXTURE: &str =
    include_str!("../../tests/fixtures/layout_replay/remove_permutes_user_order.txt");

/// The pass inside a `LayoutCtx`, as rc4's propagation passes run it, after
/// touching the case's nodes through the ctx: `set_args` with the same args
/// moves each to the end of its inputs' users, as `node.args = node.args`
/// does in torch.fx.
struct TouchedInCtx {
    /// Replay the touches; without them users stay in graph order.
    touch: bool,
}

impl replay::Port for TouchedInCtx {
    fn name(&self) -> &str {
        "remove_permutes_in_ctx"
    }

    fn run(&self, graph: &mut IrGraph, touches: &[NodeId]) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        if self.touch {
            for &id in touches {
                let args = ctx.node(id).args.clone();
                ctx.set_args(id, args);
            }
        }
        let modified = remove_permutes_in_ctx(
            &mut ctx,
            &RemovePermutesConfig::tosa(),
            &mut RemovePermutesMemo::default(),
            DeadCode::Full,
        );
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// Every case of the user-order fixture matches rc4: hand cases around two
/// regions that share a boundary permute (`shared_boundary` in the
/// generator) and 600 random graphs whose call nodes were touched in a
/// random order before rc4 ran.
#[test]
fn remove_permutes_on_reordered_users_matches_rc4() {
    replay::check_port_known(&TouchedInCtx { touch: true }, USER_ORDER_FIXTURE, &[]);
}

/// The user-order fixture depends on user order: with users left in graph
/// order, the hand case `touched_first_user_goes_last` and dozens of random
/// cases differ from rc4 (the generator measured 51 random graphs whose rc4
/// result changes with their touches).
#[test]
fn the_user_order_fixture_depends_on_user_order() {
    let results = replay::replay_port(
        &TouchedInCtx { touch: false },
        USER_ORDER_FIXTURE,
        Options::default(),
    );
    let differing: Vec<&str> = results
        .iter()
        .filter(|result| result.mismatch.is_some())
        .map(|result| result.name.as_str())
        .collect();
    assert!(differing.contains(&"touched_first_user_goes_last"), "{differing:?}");
    assert!(!differing.contains(&"users_in_graph_order"), "{differing:?}");
    assert!(differing.len() > 40, "only {} cases depend on their user order", differing.len());
}

/// rc4 runs this pass after `InsertTableOpsPass`, so a quantized table op is
/// a `TABLE(x, table)` there, and the visit's constant check on the rank-1
/// table rejects a rank-0 region (`table_in_a_rank0_region_fails` replays
/// exactly that). RETAB still holds the op itself (`layout::pending`) and
/// applies the same check; the generator hands rc4 the op, which has no table
/// input, so rc4 rewrites this region and RETAB does not.
const KNOWN: &[(&str, &str)] = &[(
    "pending_table_in_a_rank0_region",
    "rc4's TABLE reads a rank-1 table that fails a rank-0 region; RETAB checks a pending TABLE the same way",
)];

#[test]
fn remove_permutes_around_elementwise_matches_rc4() {
    replay::check_known(&IrRemovePermutesAroundElementwiseTosaOpsPass, FIXTURE, KNOWN);
}

/// rc4 eliminates dead code only when it rewrote something, and so does the
/// port, so the dead nodes match too.
#[test]
fn remove_permutes_dead_nodes_match_rc4() {
    let options = Options {
        dead_nodes: true,
        ..Options::default()
    };
    let results = replay::replay(&IrRemovePermutesAroundElementwiseTosaOpsPass, FIXTURE, options);
    let failures: Vec<&str> = results
        .iter()
        .filter(|result| result.mismatch.is_some())
        .map(|result| result.name.as_str())
        .filter(|name| !KNOWN.iter().any(|(known, _)| known == name))
        .collect();
    assert!(failures.is_empty(), "dead-node mismatches: {failures:?}");
}

/// Every value rc4's retrace leaves is contiguous, as the port's refresh
/// leaves the values it re-derives, so all dim orders match.
#[test]
fn remove_permutes_dim_orders_match_rc4() {
    let options = Options {
        dim_orders: DimOrders::All,
        ..Options::default()
    };
    let results = replay::replay(&IrRemovePermutesAroundElementwiseTosaOpsPass, FIXTURE, options);
    let failures: Vec<&str> = results
        .iter()
        .filter(|result| result.mismatch.is_some())
        .map(|result| result.name.as_str())
        .filter(|name| !KNOWN.iter().any(|(known, _)| known == name))
        .collect();
    assert!(failures.is_empty(), "dim-order mismatches: {failures:?}");
}

/// Replays a fixture written elsewhere, such as a large fuzzing run:
///
/// ```text
/// cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py \
///     --pass remove_permutes_around_elementwise --random 5000 --out /tmp/replay
/// RETAB_REPLAY_DIR=/tmp/replay cargo test --lib remove_permutes_replay_external -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: replays $RETAB_REPLAY_DIR/remove_permutes_around_elementwise.txt"]
fn remove_permutes_replay_external() {
    let dir = std::env::var("RETAB_REPLAY_DIR").expect("set RETAB_REPLAY_DIR");
    let path = std::path::Path::new(&dir).join("remove_permutes_around_elementwise.txt");
    let text = std::fs::read_to_string(&path).expect("read the external fixture");
    let results = replay::replay(&IrRemovePermutesAroundElementwiseTosaOpsPass, &text, Options::default());
    let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
    println!(
        "{}: {} of {} cases differ ({})",
        path.display(),
        failures.len(),
        results.len(),
        replay::class_counts(failures.iter().copied())
    );
    for failure in failures.iter().take(5) {
        println!("{}", failure.report);
    }
    let names: Vec<&str> = failures.iter().map(|result| result.name.as_str()).collect();
    println!("all differing cases: {}", names.join(" "));
}

/// Replays a user-order fixture written elsewhere through the ctx port:
///
/// ```text
/// cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py \
///     --pass remove_permutes_user_order --random 2000 --out /tmp/replay
/// RETAB_REPLAY_DIR=/tmp/replay cargo test --lib remove_permutes_user_order_replay_external -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: replays $RETAB_REPLAY_DIR/remove_permutes_user_order.txt"]
fn remove_permutes_user_order_replay_external() {
    let dir = std::env::var("RETAB_REPLAY_DIR").expect("set RETAB_REPLAY_DIR");
    let path = std::path::Path::new(&dir).join("remove_permutes_user_order.txt");
    let text = std::fs::read_to_string(&path).expect("read the external fixture");
    let results = replay::replay_port(&TouchedInCtx { touch: true }, &text, Options::default());
    let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
    println!(
        "{}: {} of {} cases differ ({})",
        path.display(),
        failures.len(),
        results.len(),
        replay::class_counts(failures.iter().copied())
    );
    for failure in failures.iter().take(5) {
        println!("{}", failure.report);
    }
    assert!(failures.is_empty());
}

/// rc4's propagation passes call the pass again on the graph it returned.
/// A second call that shares the first call's ctx and memo (and so skips
/// the searches the memo proves still fail; test builds re-run each one to
/// check) must leave the graph a fresh second call leaves. The fixture's
/// searches are small, so the memo's union and compaction (`Proofs`) are
/// forced too: every failed search into the union (cap 0), and every read
/// set, or all but the smallest, merged into it at the end of a call (budget
/// 0 or 8).
#[test]
fn a_second_call_with_the_memo_matches_a_fresh_one() {
    use crate::layout::remove_permutes::with_memo_limits;

    second_call_with_the_memo_matches_a_fresh_one();
    for (cap, budget) in [(0, usize::MAX), (usize::MAX, 0), (usize::MAX, 8)] {
        with_memo_limits(cap, budget, second_call_with_the_memo_matches_a_fresh_one);
    }
}

fn second_call_with_the_memo_matches_a_fresh_one() {
    use crate::fx_ir::IrPass;

    let config = RemovePermutesConfig::tosa();
    let mut remembered = 0;
    for case in replay::parse(FIXTURE).expect("the fixture parses") {
        if case.raises.is_some() || case.name == "pending_table_in_a_rank0_region" {
            continue;
        }
        let mut shared = replay::build(&case.before);
        {
            let mut ctx = LayoutCtx::new(&mut shared);
            let mut memo = RemovePermutesMemo::default();
            remove_permutes_in_ctx(&mut ctx, &config, &mut memo, DeadCode::Full);
            remove_permutes_in_ctx(&mut ctx, &config, &mut memo, DeadCode::Local);
            remembered += memo.len();
            ctx.finish();
        }
        let mut fresh = replay::build(&case.before);
        IrRemovePermutesAroundElementwiseTosaOpsPass.run(&mut fresh);
        IrRemovePermutesAroundElementwiseTosaOpsPass.run(&mut fresh);
        let (shared, fresh) = (
            replay::snapshot(&shared).expect("a valid graph"),
            replay::snapshot(&fresh).expect("a valid graph"),
        );
        let options = Options {
            dead_nodes: true,
            ..Options::default()
        };
        if let Err(mismatch) = replay::compare(&fresh, &shared, options) {
            panic!(
                "case {}: the memoized second call differs: {mismatch:?}\nfresh:\n{fresh}shared:\n{shared}",
                case.name
            );
        }
    }
    assert!(remembered > 0, "no search was remembered across calls");
}

/// Inside a caller that keeps the graph free of dead code, `DeadCode::Local`
/// (erase only what the call stopped using) leaves what rc4's full sweep
/// leaves.
#[test]
fn local_dead_code_matches_the_full_sweep_on_graphs_without_dead_code() {
    let config = RemovePermutesConfig::tosa();
    let mut compared = 0;
    for case in replay::parse(FIXTURE).expect("the fixture parses") {
        let already_dead = case.before.nodes.iter().any(|node| node.dead);
        if already_dead || case.raises.is_some() || !case.modified {
            continue;
        }
        let mut graph = replay::build(&case.before);
        {
            let mut ctx = LayoutCtx::new(&mut graph);
            let modified = remove_permutes_in_ctx(
                &mut ctx,
                &config,
                &mut RemovePermutesMemo::default(),
                DeadCode::Local,
            );
            if modified {
                ctx.finish();
            }
        }
        let snapshot = replay::snapshot(&graph).expect("a valid graph");
        let options = Options {
            dead_nodes: true,
            ..Options::default()
        };
        if KNOWN.iter().any(|(known, _)| *known == case.name) {
            continue;
        }
        replay::compare(&case.after, &snapshot, options).unwrap_or_else(|mismatch| {
            panic!("case {}: local dead-code removal differs: {mismatch:?}", case.name)
        });
        compared += 1;
    }
    assert!(compared > 50, "only {compared} cases compared");
}
