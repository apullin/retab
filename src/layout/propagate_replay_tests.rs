//! rc4 replays of the propagation passes (APU-159): every case of
//! `propagate_view_copy_permute_down.txt` and `_up.txt` in
//! tests/fixtures/layout_replay must match what the real rc4 pass did, built
//! as `arm_pass_manager.py:726-727` builds it, including the `modified` flag
//! and the dead nodes; see `replay` for what else is compared. The generator
//! turns quantized table ops into TABLE ops before the pass and back after
//! (`PropagationPass`), so the fixtures show rc4's decisions in RETAB's form.

use crate::layout::propagate::IrPropagateViewCopyPermutePass;
use crate::layout::replay::{self, Options};

const DOWN: &str = include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_down.txt");
const UP: &str = include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_up.txt");

/// The default comparison plus the `modified` flag and dead nodes: rc4 removes
/// dead code only where a step changed something, which decides what later
/// steps see.
const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: true,
    modified: true,
};

/// `_maybe_split_upwards_cat_fanout` rebuilds the cat from
/// `all_input_nodes`, which lists a repeated input once, so rc4 turns
/// `cat([x, x])` into `cat([view(x)])` and halves the result. The port keeps
/// every occurrence (see `layout::propagate`).
const REPEATED_CAT_INPUT: &str = "rc4 drops a repeated cat input when it splits a cat fan-out, which changes the result";

#[test]
fn propagate_down_matches_rc4() {
    replay::check_known_with(&IrPropagateViewCopyPermutePass::ARM_DOWN, DOWN, &[], WITH_FLAG);
}

#[test]
fn propagate_up_matches_rc4() {
    replay::check_known_with(
        &IrPropagateViewCopyPermutePass::ARM_UP,
        UP,
        &[("cat_with_a_repeated_input", REPEATED_CAT_INPUT)],
        WITH_FLAG,
    );
}

/// Per-pass statistics under the default comparison and with the flag,
/// for a fuzzing session: `cargo test --lib propagate_replay_report -- --ignored --nocapture`.
#[test]
#[ignore = "report"]
fn propagate_replay_report() {
    for (pass, fixture) in [
        (&IrPropagateViewCopyPermutePass::ARM_DOWN, DOWN),
        (&IrPropagateViewCopyPermutePass::ARM_UP, UP),
    ] {
        for (label, options) in [("default", Options::default()), ("flag+dead", WITH_FLAG)] {
            let results = replay::replay(pass, fixture, options);
            let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
            println!(
                "{} [{label}]: {} of {} differ ({})",
                crate::fx_ir::IrPass::name(pass),
                failures.len(),
                results.len(),
                replay::class_counts(failures.iter().copied())
            );
            if label == "flag+dead" {
                for failure in failures.iter().take(6) {
                    println!("{}", failure.report);
                }
                let names: Vec<&str> = failures.iter().map(|result| result.name.as_str()).collect();
                println!("all: {}", names.join(" "));
            }
        }
    }
}

/// The rewrite branches the Arm passes have, each of which the two fixtures
/// must reach at least once. Three can't be reached with the TOSA overrides
/// and are covered by the ported shared tests instead: the shared pass's
/// non-keepdim reductions (`*.reduction_dropped`), a region path that one
/// path rejects and another accepts (`down.region.path_conflict`; INT48 is
/// the only TOSA rule that depends on the path, and it blocks at the copy
/// before any region search), and lifting a permute to a higher broadcast
/// rank (`broadcast.lift`; a crossed op's output never has a lower rank than
/// its producer).
const TOSA_BRANCHES: &[&str] = &[
    "blocks_moving",
    "broadcast.project",
    "down.fork_split",
    "down.fork_split.after_crossing",
    "down.fork_split.arg_update",
    "down.move",
    "down.permute.elementwise",
    "down.permute.reduction",
    "down.permute.slice",
    "down.region.exits",
    "down.region.unsafe_input",
    "down.region_move",
    "down.split_through_elementwise",
    "down.view.elementwise",
    "down.view.reduction",
    "down.view.slice",
    "endpoint",
    "int48",
    "memo.skip",
    "retrace.stale_value",
    "table_guard",
    "up.blocks_crossing",
    "up.cat_fanout",
    "up.move",
    "up.one_layout_dependent_input",
    "up.permute.elementwise",
    "up.permute.reduction",
    "up.permute.slice",
    "up.stop.table_constant",
    "up.view.elementwise",
    "up.view.reduction",
    "up.view.slice",
    "up.view.unit_slice",
    "view.transparent_rank_change",
];

fn coverage_of_both_fixtures() -> std::collections::BTreeMap<&'static str, usize> {
    use crate::layout::propagate::COVERAGE;
    COVERAGE.with(|coverage| coverage.borrow_mut().clear());
    for (pass, fixture) in [
        (&IrPropagateViewCopyPermutePass::ARM_DOWN, DOWN),
        (&IrPropagateViewCopyPermutePass::ARM_UP, UP),
    ] {
        replay::replay(pass, fixture, WITH_FLAG);
    }
    COVERAGE.with(|coverage| coverage.borrow().iter().map(|(&branch, &count)| (branch, count)).collect())
}

#[test]
fn replays_reach_every_tosa_branch() {
    let coverage = coverage_of_both_fixtures();
    let missing: Vec<&str> = TOSA_BRANCHES
        .iter()
        .copied()
        .filter(|branch| !coverage.contains_key(branch))
        .collect();
    assert!(missing.is_empty(), "branches no replay case reaches: {missing:?}\ncoverage: {coverage:?}");
}

#[test]
#[ignore = "report"]
fn propagate_replay_coverage_report() {
    for (branch, count) in coverage_of_both_fixtures() {
        println!("{branch:40} {count}");
    }
}

/// Replays propagation fixtures written elsewhere, such as a large fuzzing
/// run that is too big to commit:
///
/// ```text
/// cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py \
///     --pass propagate_view_copy_permute_down --random 3000 --seed 7 --out /tmp/replay
/// RETAB_REPLAY_DIR=/tmp/replay cargo test --lib propagate_replay_external -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: replays the propagation fixtures in $RETAB_REPLAY_DIR"]
fn propagate_replay_external() {
    let dir = std::env::var("RETAB_REPLAY_DIR").expect("set RETAB_REPLAY_DIR");
    for (pass, file) in [
        (&IrPropagateViewCopyPermutePass::ARM_DOWN, "propagate_view_copy_permute_down.txt"),
        (&IrPropagateViewCopyPermutePass::ARM_UP, "propagate_view_copy_permute_up.txt"),
    ] {
        let path = std::path::Path::new(&dir).join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            println!("{}: skipped", path.display());
            continue;
        };
        let results = replay::replay(pass, &text, WITH_FLAG);
        let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
        println!(
            "{}: {} of {} differ ({})",
            path.display(),
            failures.len(),
            results.len(),
            replay::class_counts(failures.iter().copied())
        );
        for failure in failures.iter().take(4) {
            println!("{}", failure.report);
        }
        let names: Vec<&str> = failures.iter().map(|result| result.name.as_str()).collect();
        println!("all: {}", names.join(" "));
    }
}
