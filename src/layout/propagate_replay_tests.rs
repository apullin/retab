//! rc4 replays of the propagation passes (APU-159): every case of
//! `propagate_view_copy_permute_down.txt`, `_up.txt` and `_down_up.txt` (the
//! Down pass, then the Up pass, as phase 6 runs them) in
//! tests/fixtures/layout_replay must match what the real rc4 passes did,
//! built as `arm_pass_manager.py:726-727` builds them, including the
//! `modified` flag and the dead nodes; see `replay` for what else is
//! compared. The generator turns quantized table ops into TABLE ops before
//! the passes and back after (`PropagationPass`), so the fixtures show rc4's
//! decisions in RETAB's form, a 16-bit table split from its RESCALE where rc4
//! keeps them apart (`layout::pending`).

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::propagate::{Direction, IrPropagateViewCopyPermutePass, PropagateConfig};
use crate::layout::replay::{self, Options};

const DOWN: &str = include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_down.txt");
const UP: &str = include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_up.txt");
const DOWN_UP: &str = include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_down_up.txt");
const DOWN_WITHOUT_PROGRAM: &str =
    include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_down_without_program.txt");
const UP_WITHOUT_PROGRAM: &str =
    include_str!("../../tests/fixtures/layout_replay/propagate_view_copy_permute_up_without_program.txt");

/// The Arm Down pass, then the Arm Up pass, as phase 6 runs them
/// (`arm_pass_manager.py:726-727`).
pub(crate) struct DownThenUp;

impl IrPass for DownThenUp {
    fn name(&self) -> &str {
        "IrPropagateViewCopyPermuteDown+Up"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let down = IrPropagateViewCopyPermutePass::ARM_DOWN.run(graph);
        let up = IrPropagateViewCopyPermutePass::ARM_UP.run(graph);
        down | up
    }
}

/// Arm's passes built without arguments, as rc4's graph-level tests build
/// them: no fusion pass.
const ARM_DOWN_WITHOUT_PROGRAM: IrPropagateViewCopyPermutePass = IrPropagateViewCopyPermutePass {
    config: PropagateConfig::arm_without_program(Direction::Down),
};
const ARM_UP_WITHOUT_PROGRAM: IrPropagateViewCopyPermutePass = IrPropagateViewCopyPermutePass {
    config: PropagateConfig::arm_without_program(Direction::Up),
};

/// The default comparison plus the `modified` flag and dead nodes: rc4 removes
/// dead code only where a step changed something, which decides what later
/// steps see.
const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: true,
    float32_activations: false,
    modified: true,
};

/// `_maybe_split_upwards_cat_fanout` rebuilds the cat from
/// `all_input_nodes`, which lists a repeated input once, so rc4 turns
/// `cat([x, x])` into `cat([view(x)])` and halves the result. The port keeps
/// every occurrence (see `layout::propagate`).
const REPEATED_CAT_INPUT: &str = "rc4 drops a repeated cat input when it splits a cat fan-out, which changes the result";

/// rc4's Down walk assigns the swapped args as it crosses a mean or a slice
/// (`:234-235`), and when the fork after it can't be split it falls back to a
/// region move planned from the original permute (`:1036-1039`), which keeps
/// those args: the crossed node then reads the moved permute's output with
/// dims meant for its input. Where that changes what the node computes, the
/// port takes the assignments back first (`layout::propagate`'s module docs),
/// so its graph keeps the outputs' shapes where rc4's re-trace raises (a view
/// no longer fits) or its outputs change shape. Fuzzing found the same in 1
/// of 1,500 of the generator's random graphs run Down then Up
/// (fuzz_down_up.py 1500 7, random_544: a slice after a relu).
const REGION_MOVE_KEEPS_WALK_ARGS: &str =
    "rc4's region-move fallback keeps the args the walk assigned to a crossed mean or slice, a wrong graph; the port takes them back";

/// The Down cases where rc4's region-move fallback breaks the graph.
const DOWN_KNOWN: &[(&str, &str)] = &[
    ("advance_then_region_move_breaks_a_view", REGION_MOVE_KEEPS_WALK_ARGS),
    ("advance_then_region_move_reshapes_users", REGION_MOVE_KEEPS_WALK_ARGS),
    ("channel_split_after_clamp", REGION_MOVE_KEEPS_WALK_ARGS),
];

#[test]
fn propagate_down_matches_rc4() {
    replay::check_known_with(&IrPropagateViewCopyPermutePass::ARM_DOWN, DOWN, DOWN_KNOWN, WITH_FLAG);
}

/// The Down pass, then the Up pass, on the same cases: the Up pass meets what
/// Down left, such as a view parked between a 16-bit TABLE and its RESCALE.
#[test]
fn propagate_down_then_up_matches_rc4() {
    let mut known = DOWN_KNOWN.to_vec();
    known.push(("cat_with_a_repeated_input", REPEATED_CAT_INPUT));
    replay::check_known_with(&DownThenUp, DOWN_UP, &known, WITH_FLAG);
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

/// The same cases through the passes built without an exported program, so
/// with no RemovePermutes round: the walks meet every permute the fusion pass
/// would have cancelled first.
#[test]
fn propagate_down_without_a_program_matches_rc4() {
    replay::check_known_with(&ARM_DOWN_WITHOUT_PROGRAM, DOWN_WITHOUT_PROGRAM, DOWN_KNOWN, WITH_FLAG);
}

#[test]
fn propagate_up_without_a_program_matches_rc4() {
    replay::check_known_with(
        &ARM_UP_WITHOUT_PROGRAM,
        UP_WITHOUT_PROGRAM,
        &[("cat_with_a_repeated_input", REPEATED_CAT_INPUT)],
        WITH_FLAG,
    );
}

/// Per-pass statistics under the default comparison and with the flag,
/// for a fuzzing session: `cargo test --lib propagate_replay_report -- --ignored --nocapture`.
#[test]
#[ignore = "report"]
fn propagate_replay_report() {
    for (pass, fixture) in replayed() {
        for (label, options) in [("default", Options::default()), ("flag+dead", WITH_FLAG)] {
            let results = replay::replay(pass, fixture, options);
            let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
            println!(
                "{} [{label}]: {} of {} differ ({})",
                pass.name(),
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

/// The rewrite branches the Arm passes have, each of which the fixtures must
/// reach at least once. Three can't be reached with the TOSA overrides and
/// are covered by the ported shared tests instead: the shared pass's
/// non-keepdim reductions (`*.reduction_dropped`), a region path that one
/// path rejects and another accepts (`down.region.path_conflict`; INT48 is
/// the only TOSA rule that depends on the path, and it blocks at the copy
/// before any region search), and lifting a permute to a higher broadcast
/// rank (`broadcast.lift`; a crossed op's output never has a lower rank than
/// its producer). A re-trace point that finds a stale value
/// (`retrace.stale_value`) was reached only through rc4's region-move
/// fallback keeping the walk's args, which the port no longer copies; it
/// stays as a guard (`Propagator::retrace_stale`).
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
    "down.region_move.undo",
    "down.split_through_elementwise",
    "down.view.elementwise",
    "down.view.reduction",
    "down.view.slice",
    "endpoint",
    "int48",
    "memo.skip",
    "table.kept_split",
    "table.split",
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

/// Every committed fixture with the port that replays it.
fn replayed() -> [(&'static dyn IrPass, &'static str); 5] {
    [
        (&IrPropagateViewCopyPermutePass::ARM_DOWN, DOWN),
        (&IrPropagateViewCopyPermutePass::ARM_UP, UP),
        (&DownThenUp, DOWN_UP),
        (&ARM_DOWN_WITHOUT_PROGRAM, DOWN_WITHOUT_PROGRAM),
        (&ARM_UP_WITHOUT_PROGRAM, UP_WITHOUT_PROGRAM),
    ]
}

fn coverage_of_the_fixtures() -> std::collections::BTreeMap<&'static str, usize> {
    use crate::layout::propagate::COVERAGE;
    COVERAGE.with(|coverage| coverage.borrow_mut().clear());
    for (pass, fixture) in replayed() {
        replay::replay(pass, fixture, WITH_FLAG);
    }
    COVERAGE.with(|coverage| coverage.borrow().iter().map(|(&branch, &count)| (branch, count)).collect())
}

#[test]
fn replays_reach_every_tosa_branch() {
    let coverage = coverage_of_the_fixtures();
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
    for (branch, count) in coverage_of_the_fixtures() {
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
        (&ARM_DOWN_WITHOUT_PROGRAM, "propagate_view_copy_permute_down_without_program.txt"),
        (&ARM_UP_WITHOUT_PROGRAM, "propagate_view_copy_permute_up_without_program.txt"),
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
