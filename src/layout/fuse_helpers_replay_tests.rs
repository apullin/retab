//! rc4 replays of FuseDuplicateUsers, FuseIdenticalInputTransforms and
//! MoveDataMovementOpsToSmallerDtype (APU-160). Every case of each fixture in
//! tests/fixtures/layout_replay must match what the real rc4 pass did,
//! including the `modified` flag, which the propagation driver loops on; see
//! `replay` for what else is compared.

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::ctx::LayoutCtx;
use crate::layout::fuse_duplicate_users::IrFuseDuplicateUsersPass;
use crate::layout::fuse_identical_input_transforms::{
    fuse_identical_input_transforms, FuseIdenticalInputTransforms, IrFuseIdenticalInputTransformsPass,
};
use crate::layout::move_data_movement::IrMoveDataMovementOpsToSmallerDtypePass;
use crate::layout::replay::{self, Options};

const FUSE_DUPLICATE_USERS: &str =
    include_str!("../../tests/fixtures/layout_replay/fuse_duplicate_users.txt");
const FUSE_IDENTICAL_INPUT_TRANSFORMS: &str =
    include_str!("../../tests/fixtures/layout_replay/fuse_identical_input_transforms.txt");
const FUSE_IDENTICAL_INPUT_TRANSFORMS_DRIVER: &str =
    include_str!("../../tests/fixtures/layout_replay/fuse_identical_input_transforms_driver.txt");
const MOVE_DATA_MOVEMENT: &str =
    include_str!("../../tests/fixtures/layout_replay/move_data_movement_ops_to_smaller_dtype.txt");

/// The default comparison plus the `modified` flag.
const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: false,
    modified: true,
};

/// rc4 fuses two users that share one NaN float object (`math.nan`, or one
/// `float("nan")` passed twice): Python compares the object to itself before
/// calling `==`. RETAB's IR has no object identity, so the port never fuses
/// on a NaN argument; separate NaN objects match (`separate_nan_scalars_never_fuse`).
const SHARED_NAN_OBJECT: &str = "rc4 fuses users that share one NaN float object, which the IR can't tell apart from two";

#[test]
fn fuse_duplicate_users_matches_rc4() {
    replay::check_known_with(
        &IrFuseDuplicateUsersPass,
        FUSE_DUPLICATE_USERS,
        &[("one_shared_nan_object_fuses", SHARED_NAN_OBJECT)],
        WITH_FLAG,
    );
}

#[test]
fn fuse_identical_input_transforms_matches_rc4() {
    replay::check_known_with(
        &IrFuseIdenticalInputTransformsPass,
        FUSE_IDENTICAL_INPUT_TRANSFORMS,
        &[],
        WITH_FLAG,
    );
}

/// The propagation driver's entry point:
/// `FuseIdenticalInputTransformsPass(permute_targets=...).call(graph_module)`
/// (backends/transforms/propagate_view_copy_permute_pass.py:921-928), with no
/// exported program and no `ArmOpTargetedPass` gate, on a `LayoutCtx`.
struct DriverFuseIdenticalInputTransforms;

impl IrPass for DriverFuseIdenticalInputTransforms {
    fn name(&self) -> &str {
        "DriverFuseIdenticalInputTransforms"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let modified = fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER);
        if modified {
            ctx.finish();
        }
        modified
    }
}

#[test]
fn fuse_identical_input_transforms_driver_entry_matches_rc4() {
    replay::check_known_with(
        &DriverFuseIdenticalInputTransforms,
        FUSE_IDENTICAL_INPUT_TRANSFORMS_DRIVER,
        &[],
        WITH_FLAG,
    );
}

#[test]
fn move_data_movement_ops_to_smaller_dtype_matches_rc4() {
    replay::check_known_with(
        &IrMoveDataMovementOpsToSmallerDtypePass,
        MOVE_DATA_MOVEMENT,
        &[],
        WITH_FLAG,
    );
}

/// Where rc4's MoveDataMovementOpsToSmallerDtype raises, it is always its
/// closing `graph.lint()` rejecting a RESCALE user that now comes before the
/// RESCALE, and the port panics for that reason too.
#[test]
fn move_data_movement_fails_like_rc4_for_the_same_reason() {
    let cases = replay::parse(MOVE_DATA_MOVEMENT).unwrap();
    let raising: Vec<&replay::Case> = cases.iter().filter(|case| case.raises.is_some()).collect();
    assert!(raising.len() >= 10, "{} raising cases", raising.len());
    for case in raising {
        assert_eq!(case.raises.as_deref(), Some("RuntimeError"), "{}", case.name);
        let mut graph = replay::build(&case.before);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            IrMoveDataMovementOpsToSmallerDtypePass.run(&mut graph)
        }))
        .expect_err("the port panics where rc4 raises");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            message.contains("used before it has been defined"),
            "{}: {message}",
            case.name
        );
    }
}

/// Each fixture holds the structured cases plus at least 300 accepted random
/// cases, and enough of them rewrite the graph to exercise the pass.
#[test]
fn fixtures_cover_their_passes() {
    for (fixture, minimum_random, minimum_modified) in [
        (FUSE_DUPLICATE_USERS, 300, 200),
        (FUSE_IDENTICAL_INPUT_TRANSFORMS, 300, 120),
        (MOVE_DATA_MOVEMENT, 300, 150),
    ] {
        let cases = replay::parse(fixture).unwrap();
        let random = cases.iter().filter(|case| case.name.starts_with("random_")).count();
        let modified = cases.iter().filter(|case| case.modified).count();
        assert!(random >= minimum_random, "{random} random cases");
        assert!(modified >= minimum_modified, "{modified} modified cases");
    }
    let driver = replay::parse(FUSE_IDENTICAL_INPUT_TRANSFORMS_DRIVER).unwrap();
    assert!(driver.iter().filter(|case| case.modified).count() > 20);
}

/// Statistics for the notes; with RETAB_REPLAY_DIR set, replays fixtures
/// written there instead (a larger fuzzing run, say):
///
/// ```text
/// cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py \
///     --pass fuse_duplicate_users --random 3000 --seed 7 --out /tmp/replay
/// RETAB_REPLAY_DIR=/tmp/replay cargo test --lib fuse_helpers_replay_report -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: prints replay statistics; run with --ignored --nocapture"]
fn fuse_helpers_replay_report() {
    let passes: [(&dyn IrPass, &str, &str); 4] = [
        (&IrFuseDuplicateUsersPass, "fuse_duplicate_users.txt", FUSE_DUPLICATE_USERS),
        (
            &IrFuseIdenticalInputTransformsPass,
            "fuse_identical_input_transforms.txt",
            FUSE_IDENTICAL_INPUT_TRANSFORMS,
        ),
        (
            &DriverFuseIdenticalInputTransforms,
            "fuse_identical_input_transforms_driver.txt",
            FUSE_IDENTICAL_INPUT_TRANSFORMS_DRIVER,
        ),
        (
            &IrMoveDataMovementOpsToSmallerDtypePass,
            "move_data_movement_ops_to_smaller_dtype.txt",
            MOVE_DATA_MOVEMENT,
        ),
    ];
    let dir = std::env::var("RETAB_REPLAY_DIR").ok();
    for (pass, file, committed) in passes {
        let text = match &dir {
            Some(dir) => match std::fs::read_to_string(std::path::Path::new(dir).join(file)) {
                Ok(text) => text,
                Err(error) => {
                    println!("{file}: skipped, {error}");
                    continue;
                }
            },
            None => committed.to_string(),
        };
        let results = replay::replay(pass, &text, WITH_FLAG);
        let matched = results.iter().filter(|result| result.mismatch.is_none()).count();
        let raises = replay::parse(&text)
            .unwrap()
            .iter()
            .filter(|case| case.raises.is_some())
            .count();
        println!(
            "{} ({file}): {matched}/{} match, {raises} where rc4 raises; mismatches: {}",
            pass.name(),
            results.len(),
            replay::class_counts(results.iter())
        );
        for result in results.iter().filter(|result| result.mismatch.is_some()).take(3) {
            println!("{}", result.report);
        }
    }
}
