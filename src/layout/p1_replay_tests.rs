//! rc4 replays of the P1 layout passes (APU-135): every case of each fixture
//! in tests/fixtures/layout_replay must match what the real rc4 pass did; see
//! `replay` for what is compared.

use crate::fx_ir::IrPass;
use crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass;
use crate::layout::high_rank_singleton_permute::IrRewriteHighRankSingletonPermutePass;
use crate::layout::match_arg_ranks::IrMatchArgRanksPass;
use crate::layout::normalize_delegate_io::IrNormalizeDelegateIOLayoutPass;
use crate::layout::replay::{self, DimOrders, Options};

const NORMALIZE_DELEGATE_IO: &str =
    include_str!("../../tests/fixtures/layout_replay/normalize_delegate_io.txt");
const MATCH_ARG_RANKS: &str =
    include_str!("../../tests/fixtures/layout_replay/match_arg_ranks.txt");
const HIGH_RANK_SINGLETON_PERMUTE: &str =
    include_str!("../../tests/fixtures/layout_replay/high_rank_singleton_permute.txt");
const FUSE_VIEW_COPY: &str = include_str!("../../tests/fixtures/layout_replay/fuse_view_copy.txt");
const CANONICALIZE_VIEW_COPY_PERMUTE: &str =
    include_str!("../../tests/fixtures/layout_replay/canonicalize_view_copy_permute.txt");

#[test]
fn normalize_delegate_io_matches_rc4() {
    replay::check(&IrNormalizeDelegateIOLayoutPass, NORMALIZE_DELEGATE_IO);
}

#[test]
fn match_arg_ranks_matches_rc4() {
    replay::check(&IrMatchArgRanksPass, MATCH_ARG_RANKS);
}

#[test]
fn high_rank_singleton_permute_matches_rc4() {
    replay::check(
        &IrRewriteHighRankSingletonPermutePass,
        HIGH_RANK_SINGLETON_PERMUTE,
    );
}

#[test]
fn fuse_view_copy_matches_rc4() {
    replay::check_known(
        &IrFuseViewCopyTransformPass,
        FUSE_VIEW_COPY,
        FUSE_VIEW_COPY_KNOWN,
    );
}

const FUSE_RANK0: &str = "BUG: the last view of the chain has rank 0";
const FUSE_INFERRED: &str = "BUG: the last view of the chain infers a dim (-1)";

/// FuseViewCopy cases that still differ from rc4, by bug; see the two
/// ignored tests below. Delete entries as the port is fixed.
const FUSE_VIEW_COPY_KNOWN: &[(&str, &str)] = &[
    ("final_view_to_rank0", FUSE_RANK0),
    ("random_18", FUSE_RANK0),
    ("random_28", FUSE_RANK0),
    ("random_32", FUSE_RANK0),
    ("random_75", FUSE_RANK0),
    ("random_128", FUSE_RANK0),
    ("random_165", FUSE_RANK0),
    ("random_196", FUSE_RANK0),
    ("final_view_with_an_inferred_dim", FUSE_INFERRED),
    ("random_6", FUSE_INFERRED),
    ("random_95", FUSE_INFERRED),
    ("random_129", FUSE_INFERRED),
    ("random_130", FUSE_INFERRED),
    ("random_145", FUSE_INFERRED),
    ("random_148", FUSE_INFERRED),
    ("random_163", FUSE_INFERRED),
    ("random_192", FUSE_INFERRED),
    ("random_259", FUSE_INFERRED),
];

/// rc4 gives the first view and the unary ops before the last removed view
/// the shape that view computes: `node.args = (node.args[0], final_shape)`
/// (fuse_view_copy.py:190-192), then the retrace (:207) re-runs view_copy. The
/// port parses the last view's shape argument instead
/// (fuse_view_copy.rs:98-116); RETAB's importer turns `[]` into `IrArg::None`
/// (`resolve_arg`, fx_ir.rs:1949), so `as_int_list` fails, the loop `continue`s
/// and the retained nodes keep their old shapes ([1] where rc4 has []).
#[test]
#[ignore = "BUG: a fused view chain whose last view has rank 0 keeps its old shapes"]
fn fuse_view_copy_rank0_final_view() {
    replay::check_cases(
        &IrFuseViewCopyTransformPass,
        FUSE_VIEW_COPY,
        &["final_view_to_rank0"],
    );
}

/// The same code path: a last view of `[-1, 8]` sets the retained nodes'
/// shapes to [-1, 8] where rc4's retrace computes [3, 8].
#[test]
#[ignore = "BUG: a fused view chain whose last view has a -1 dim records -1 as a size"]
fn fuse_view_copy_inferred_final_dim() {
    replay::check_cases(
        &IrFuseViewCopyTransformPass,
        FUSE_VIEW_COPY,
        &["final_view_with_an_inferred_dim"],
    );
}

/// rc4's retrace gives every node it creates the val its op returns
/// (`self.tracer.set_metadata(res_proxy.node, res_data)`, exir/pass_base.py:690),
/// so the views and the reduced permute have identity dim orders of their own
/// rank. The port builds their meta with `stamp.preserve_qparams`
/// (high_rank_singleton_permute.rs:117,132,142), which clones the permute's
/// meta, dim order included: the rank-4 view in `rank5_with_a_singleton`
/// carries [0, 1, 2, 3, 4]. Nothing reads it in explicit mode yet.
#[test]
#[ignore = "BUG: views replacing a high-rank permute keep its dim order of the old rank"]
fn high_rank_singleton_permute_new_nodes_keep_the_old_rank_dim_order() {
    replay::check_cases_with(
        &IrRewriteHighRankSingletonPermutePass,
        HIGH_RANK_SINGLETON_PERMUTE,
        &["rank5_with_a_singleton"],
        Options {
            dim_orders: DimOrders::All,
            ..Options::default()
        },
    );
}

/// rc4's retrace recomputes the val of every node a fusion reshapes. The port
/// sets their tensor facts only (fuse_view_copy.rs:112-116), so in
/// `docstring_example` abs goes from [24] to [2, 12] and keeps dim order [0].
#[test]
#[ignore = "BUG: nodes a view fusion reshapes keep their dim order of the old rank"]
fn fuse_view_copy_reshaped_nodes_keep_the_old_rank_dim_order() {
    replay::check_cases_with(
        &IrFuseViewCopyTransformPass,
        FUSE_VIEW_COPY,
        &["docstring_example"],
        Options {
            dim_orders: DimOrders::All,
            ..Options::default()
        },
    );
}

// The Rust canonicalizer doesn't exist yet; its fixture is ready:
// #[test]
// fn canonicalize_view_copy_permute_matches_rc4() {
//     replay::check(&IrCanonicalizeViewCopyPermutePass, CANONICALIZE_VIEW_COPY_PERMUTE);
// }

#[test]
fn every_fixture_parses_and_covers_its_pass() {
    // Over a quarter of each fixture's cases add (or remove) nodes of the target.
    for (fixture, target, adds) in [
        (NORMALIZE_DELEGATE_IO, "aten.permute_copy.default", true),
        (MATCH_ARG_RANKS, "aten.view_copy.default", true),
        (HIGH_RANK_SINGLETON_PERMUTE, "aten.view_copy.default", true),
        (FUSE_VIEW_COPY, "aten.view_copy.default", false),
        (
            CANONICALIZE_VIEW_COPY_PERMUTE,
            "aten.permute_copy.default",
            false,
        ),
    ] {
        let cases = replay::parse(fixture).unwrap();
        assert!(cases.len() > 300, "{} cases", cases.len());
        let count = |graph: &replay::Graph| {
            graph
                .nodes
                .iter()
                .filter(
                    |node| matches!(&node.op, replay::Op::Call { target: t, .. } if t == target),
                )
                .count()
        };
        let rewritten = cases
            .iter()
            .filter(|case| {
                let (before, after) = (count(&case.before), count(&case.after));
                if adds {
                    after > before
                } else {
                    after < before
                }
            })
            .count();
        assert!(
            rewritten > cases.len() / 4,
            "only {rewritten} of {} cases change the {target} count",
            cases.len()
        );
    }
}

/// The four P1 ports with their fixture file names.
fn p1_passes() -> [(&'static dyn IrPass, &'static str); 4] {
    [
        (
            &IrNormalizeDelegateIOLayoutPass,
            "normalize_delegate_io.txt",
        ),
        (&IrMatchArgRanksPass, "match_arg_ranks.txt"),
        (
            &IrRewriteHighRankSingletonPermutePass,
            "high_rank_singleton_permute.txt",
        ),
        (&IrFuseViewCopyTransformPass, "fuse_view_copy.txt"),
    ]
}

fn fixture(file: &str) -> &'static str {
    match file {
        "normalize_delegate_io.txt" => NORMALIZE_DELEGATE_IO,
        "match_arg_ranks.txt" => MATCH_ARG_RANKS,
        "high_rank_singleton_permute.txt" => HIGH_RANK_SINGLETON_PERMUTE,
        "fuse_view_copy.txt" => FUSE_VIEW_COPY,
        _ => panic!("no fixture {file}"),
    }
}

/// Print one replay's statistics, the smallest example of each mismatch
/// class and, with RETAB_REPLAY_VERBOSE set, every mismatch.
fn print_replay(pass: &dyn IrPass, fixture: &str, mode: &str, options: Options) {
    let results = replay::replay(pass, fixture, options);
    let matched = results
        .iter()
        .filter(|result| result.mismatch.is_none())
        .count();
    let flags = results
        .iter()
        .filter(|result| {
            result
                .retab_modified
                .is_some_and(|flag| flag != result.rc4_modified)
        })
        .count();
    println!(
        "{} [{mode}]: {matched}/{} match; mismatches: {}; modified flag differs in {flags}",
        pass.name(),
        results.len(),
        replay::class_counts(results.iter())
    );
    let mut failures: Vec<_> = results
        .iter()
        .filter(|result| result.mismatch.is_some())
        .collect();
    failures.sort_by_key(|result| result.report.len());
    let mut shown = std::collections::BTreeSet::new();
    for failure in &failures {
        let mismatch = failure.mismatch.as_ref().unwrap();
        if shown.insert(mismatch.class) {
            println!("  smallest {} example:\n{}", mismatch.class, failure.report);
        }
    }
    if std::env::var_os("RETAB_REPLAY_VERBOSE").is_some() {
        for failure in &failures {
            let mismatch = failure.mismatch.as_ref().unwrap();
            println!("  {}: {} {}", failure.name, mismatch.class, mismatch.detail);
        }
    }
}

/// Per-pass statistics for the APU-135 notes: the default comparison, with
/// placeholder or all dim orders, and with dead nodes.
#[test]
#[ignore = "diagnostic: prints replay statistics; run with --ignored --nocapture"]
fn replay_report() {
    let dim_orders = |dim_orders| Options {
        dim_orders,
        ..Options::default()
    };
    let modes = [
        ("default", Options::default()),
        (
            "placeholder dim orders",
            dim_orders(DimOrders::Placeholders),
        ),
        ("all dim orders", dim_orders(DimOrders::All)),
        (
            "dead nodes",
            Options {
                dead_nodes: true,
                ..Options::default()
            },
        ),
    ];
    for (pass, file) in p1_passes() {
        for (mode, options) in modes {
            print_replay(pass, fixture(file), mode, options);
        }
    }
}

/// Replays fixtures written elsewhere, such as a large fuzzing run that is
/// too big to commit:
///
/// ```text
/// cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py \
///     --random 3000 --seed 7 --out /tmp/replay
/// RETAB_REPLAY_DIR=/tmp/replay cargo test --lib replay_external -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: replays the fixtures in $RETAB_REPLAY_DIR"]
fn replay_external() {
    let dir = std::env::var("RETAB_REPLAY_DIR").expect("set RETAB_REPLAY_DIR");
    for (pass, file) in p1_passes() {
        let path = std::path::Path::new(&dir).join(file);
        match std::fs::read_to_string(&path) {
            Ok(text) => print_replay(pass, &text, &path.display().to_string(), Options::default()),
            Err(error) => println!("{}: skipped, {error}", path.display()),
        }
    }
}
