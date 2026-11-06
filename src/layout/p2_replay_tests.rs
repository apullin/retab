//! rc4 replays for the P2 layout passes (APU-135 S1).
//!
//! The fixtures are captured in rc4's own ArmPassManager pipeline while it
//! lowers small quantized models (see the generator's docstring): the graphs
//! RewriteConvPass, RemoveGetItemPass, RewriteMaxPool2dPass,
//! RewriteUpsamplePass and RewriteAvgPool2dPass (alone and with the
//! ComputeConstantOpsAOTPass after it) receive and return. The tests below pin
//! what the fixtures hold, so a regeneration that drops a case or loses a
//! detail fails here. Each port has a replay hook below: IrRewriteConv's (S3,
//! and S4's A16W8 and U55 fixtures) are enabled, the pool and upsample ports'
//! (S7) are still commented out. The tests that pin the S1 review's findings
//! follow them.

use std::collections::BTreeSet;

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::replay::{self, Case, Data, DimOrders, Graph, Op, Options, QValue, Val};
use crate::layout::rewrite_conv::IrRewriteConvPass;

const REWRITE_CONV: &str = include_str!("../../tests/fixtures/layout_replay/rewrite_conv.txt");
const REWRITE_CONV_A16W8: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8.txt");
const REWRITE_CONV_A16W8_U55: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8_u55.txt");
const REMOVE_GETITEM: &str = include_str!("../../tests/fixtures/layout_replay/remove_getitem.txt");
const REWRITE_MAX_POOL2D: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_max_pool2d.txt");
const REWRITE_UPSAMPLE: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_upsample.txt");
const REWRITE_AVG_POOL2D: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_avg_pool2d.txt");
const REWRITE_AVG_POOL2D_AOT: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_avg_pool2d_aot.txt");

const CONVOLUTION: &str = "aten.convolution.default";
const TOSA_CONVS: &[&str] = &[
    "tosa.CONV2D.default",
    "tosa.DEPTHWISE_CONV2D.default",
    "tosa.TRANSPOSE_CONV2D.default",
    "tosa.CONV3D.default",
];
const MAX_POOL: &str = "aten.max_pool2d.default";
const MAX_POOL_WITH_INDICES: &str = "aten.max_pool2d_with_indices.default";
const GETITEM: &str = "<built-in function getitem>";

/// Every P2 fixture: name, text, the targets its before graphs hold and the
/// targets rc4 rewrites them to.
fn fixtures() -> [(&'static str, &'static str, &'static [&'static str], &'static [&'static str]);
       8] {
    [
        ("rewrite_conv", REWRITE_CONV, &[CONVOLUTION], TOSA_CONVS),
        ("rewrite_conv_a16w8", REWRITE_CONV_A16W8, &[CONVOLUTION], TOSA_CONVS),
        (
            "rewrite_conv_a16w8_u55",
            REWRITE_CONV_A16W8_U55,
            &[CONVOLUTION],
            TOSA_CONVS,
        ),
        (
            "remove_getitem",
            REMOVE_GETITEM,
            &[MAX_POOL_WITH_INDICES],
            &[MAX_POOL],
        ),
        (
            "rewrite_max_pool2d",
            REWRITE_MAX_POOL2D,
            &[MAX_POOL, MAX_POOL_WITH_INDICES],
            &["tosa.MAX_POOL2D.default"],
        ),
        (
            "rewrite_upsample",
            REWRITE_UPSAMPLE,
            &[
                "aten.upsample_nearest2d.vec",
                "aten.upsample_bilinear2d.vec",
            ],
            &["tosa.RESIZE.default"],
        ),
        (
            "rewrite_avg_pool2d",
            REWRITE_AVG_POOL2D,
            &["aten.avg_pool2d.default"],
            &["tosa.AVG_POOL2D.default"],
        ),
        (
            "rewrite_avg_pool2d_aot",
            REWRITE_AVG_POOL2D_AOT,
            &["aten.avg_pool2d.default"],
            &["tosa.AVG_POOL2D.default"],
        ),
    ]
}

fn cases(fixture: &str) -> Vec<Case> {
    replay::parse(fixture).unwrap_or_else(|error| panic!("bad replay fixture: {error}"))
}

fn case<'a>(cases: &'a [Case], name: &str) -> &'a Case {
    cases
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

fn target(node: &replay::Node) -> Option<&str> {
    match &node.op {
        Op::Call { target, .. } => Some(target),
        _ => None,
    }
}

fn args(node: &replay::Node) -> &[replay::Arg] {
    match &node.op {
        Op::Call { args, .. } => args,
        _ => &[],
    }
}

fn count(graph: &Graph, targets: &[&str]) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| target(node).is_some_and(|t| targets.contains(&t)))
        .count()
}

fn nodes_of<'a>(graph: &'a Graph, wanted: &'a str) -> impl Iterator<Item = &'a replay::Node> + 'a {
    graph
        .nodes
        .iter()
        .filter(move |node| target(node) == Some(wanted))
}

fn node_arg<'a>(graph: &'a Graph, node: &replay::Node, index: usize) -> &'a replay::Node {
    match args(node).get(index) {
        Some(replay::Arg::Node(position)) => &graph.nodes[*position],
        other => panic!("argument {index} is {other:?}, not a node"),
    }
}

fn qvalue(map: &Option<replay::QMap>, key: usize) -> Option<&QValue> {
    map.as_ref()?
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, value)| value.as_ref())
}

fn dtype(node: &replay::Node) -> Option<&str> {
    match &node.val {
        Val::Tensor { dtype, .. } => Some(dtype),
        _ => None,
    }
}

#[test]
fn every_p2_fixture_parses_in_format_2() {
    for (name, fixture, _, _) in fixtures() {
        let cases = cases(fixture);
        assert!(cases.len() >= 8, "{name}: {} cases", cases.len());
        let mut names = BTreeSet::new();
        for case in &cases {
            assert_eq!(case.format, replay::FORMAT, "{name}/{}", case.name);
            assert!(names.insert(case.name.as_str()), "{name}: {} twice", case.name);
            // Every captured pass rewrote its graph.
            assert!(case.modified, "{name}/{}", case.name);
        }
    }
}

/// rc4's P2 passes erase what they replace and receive dead-free graphs, so
/// the replay compares format-2 graphs whole (`replay::compare_case`): a node
/// a port leaves dead is a mismatch. That needs rc4's graphs to hold none; if
/// a regeneration brings one in, a port would have to leave the same node
/// behind, so decide then whether the comparison should drop it instead.
#[test]
fn p2_fixtures_hold_no_dead_nodes() {
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            for (graph, label) in [(&case.before, "before"), (&case.after, "after")] {
                let dead: Vec<usize> = (0..graph.nodes.len())
                    .filter(|&position| graph.nodes[position].dead)
                    .collect();
                assert!(
                    dead.is_empty(),
                    "{name}/{} ({label}): dead {dead:?}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn p2_fixtures_cover_their_passes() {
    for (name, fixture, sources, results) in fixtures() {
        for case in cases(fixture) {
            let label = format!("{name}/{}", case.name);
            assert!(count(&case.before, sources) > 0, "{label}: nothing to rewrite");
            assert_eq!(count(&case.after, sources), 0, "{label}: left a source op");
            assert!(count(&case.after, results) > 0, "{label}: no rewritten op");
        }
    }
}

/// The case lists in the P2 design's §5 table (S3, S4, S7), with risk 6's
/// float32 scale factors and the conv inputs SizeAdjustInputPass slices
/// (design C7, F5). "Shared dw weights" isn't on it: no INT graph reaches
/// RewriteConvPass with a shared weight (see
/// `review_shared_conv_weights_are_unreachable_in_int`).
#[test]
fn p2_fixtures_cover_the_design_case_lists() {
    let names = |fixture: &str| -> BTreeSet<String> {
        cases(fixture).into_iter().map(|case| case.name).collect()
    };
    let conv = names(REWRITE_CONV);
    for model in [
        "conv2d",
        "conv2d_nobias",
        "conv2d_s2p1",
        "conv2d_s3p2",
        "conv2d_s2p0",
        "conv2d_1x1_s2",
        "conv2d_dilation2",
        "dw2d",
        "dw2d_nobias",
        "dw2d_kx1",
        "grouped2d",
        "tconv2d",
        "tconv2d_nobias",
        "conv1d",
        "conv1d_nobias",
        "conv1d_dw",
        "conv1d_s2p0",
        "conv3d",
        "linear",
        "linear_in1",
        "conv2d_cin1",
        "conv1d_cin1",
        "conv3d_cin1",
        "two_convs_no_bias",
        "two_convs_one_input",
        "cumsum",
    ] {
        for quant in ["pt", "pc"] {
            assert!(conv.contains(&format!("{model}_{quant}")), "{model}_{quant}");
        }
    }
    for model in ["conv2d", "dw2d", "conv1d", "linear"] {
        assert!(conv.contains(&format!("{model}_a8w4")), "{model}_a8w4");
    }
    let a16w8 = names(REWRITE_CONV_A16W8);
    let u55 = names(REWRITE_CONV_A16W8_U55);
    for model in [
        "conv2d",
        "conv_add",
        "conv1d_add",
        "conv_add_sub",
        "permuted_int32",
        "mixed_chain_conv",
        "mixed_chain_dw",
        "mixed_chain_avg_pool",
        "mixed_chain_max_pool",
        "mixed_tconv",
    ] {
        assert!(a16w8.contains(&format!("{model}_a16w8")), "{model}_a16w8");
        assert!(u55.contains(&format!("{model}_a16w8_u55")), "{model}_a16w8_u55");
    }
    assert!(a16w8.contains("mixed_conv3d_a16w8"));
    let upsample = names(REWRITE_UPSAMPLE);
    for case in [
        "nearest_size_5to8_pc",
        "nearest_scale2_pc",
        "nearest_scale1_5_pc",
        "nearest_scale1_6_pc",
        "nearest_scale2_5_pc",
        "nearest_scale_third_pc",
        "nearest_scale1_6_f32_pc",
        "nearest_scale_third_f32_pc",
        "bilinear_scale2_pc",
        "bilinear_scale2_align_corners_pc",
        "bilinear_size_5to9_align_corners_pc",
        "bilinear_scale1_5_pc",
        "nearest_scale2_a16w8",
        "bilinear_scale2_a16w8",
    ] {
        assert!(upsample.contains(case), "{case}");
    }
    let max_pool = names(REWRITE_MAX_POOL2D);
    for case in names(REMOVE_GETITEM) {
        assert!(max_pool.contains(&case), "{case}");
        assert!(max_pool.contains(&format!("{case}_with_indices")), "{case}");
    }
    assert_eq!(names(REWRITE_AVG_POOL2D), names(REWRITE_AVG_POOL2D_AOT));
}

#[test]
fn p2_before_graphs_hold_importable_data() {
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            for node in &case.before.nodes {
                let constant = matches!(
                    node.op,
                    Op::Placeholder {
                        constant: Some(_),
                        ..
                    } | Op::GetAttr { .. }
                );
                if constant {
                    assert!(
                        matches!(node.data, Some(Data::Bytes(_))),
                        "{name}/{}: a before-graph constant without bytes",
                        case.name
                    );
                }
            }
        }
    }
}

/// Weight qparams are per-channel in the `_pc`, A8W4 and A16W8 cases and
/// per-tensor in the `_pt` ones; RewriteConvPass's TOSA ops inherit them.
/// DecomposeCumsumPass's all-ones kernel is per-tensor (scale 1.0) either way.
#[test]
fn conv_fixtures_record_real_qparams() {
    for fixture in [REWRITE_CONV, REWRITE_CONV_A16W8, REWRITE_CONV_A16W8_U55] {
        for case in cases(fixture) {
            let per_channel = !case.name.ends_with("_pt") && !case.name.starts_with("cumsum_");
            for conv in nodes_of(&case.before, CONVOLUTION) {
                let input = qvalue(&conv.q_in, 0).expect("input qparam");
                let weight = qvalue(&conv.q_in, 1).expect("weight qparam");
                let output = qvalue(&conv.q_out, 0).expect("output qparam");
                assert!(!input.per_channel && !output.per_channel, "{}", case.name);
                assert_eq!(weight.per_channel, per_channel, "{}", case.name);
                if per_channel {
                    let w = node_arg(&case.before, conv, 1);
                    let Val::Tensor { shape, .. } = &w.val else {
                        panic!("weight without a tensor val");
                    };
                    let channels = if args(conv)[6] == replay::Arg::Bool(true) {
                        // Transposed: IOHW, per-channel along O (axis 1).
                        assert_eq!(weight.axis, 1, "{}", case.name);
                        shape[1]
                    } else {
                        assert_eq!(weight.axis, 0, "{}", case.name);
                        shape[0]
                    };
                    assert_eq!(weight.scales.len() as i64, channels, "{}", case.name);
                }
            }
            for tosa in case
                .after
                .nodes
                .iter()
                .filter(|node| target(node).is_some_and(|t| TOSA_CONVS.contains(&t)))
            {
                assert!(qvalue(&tosa.q_in, 0).is_some(), "{}", case.name);
                assert!(qvalue(&tosa.q_out, 0).is_some(), "{}", case.name);
                assert_eq!(dtype(tosa), Some("torch.int32"), "{}", case.name);
            }
        }
    }
}

/// A8W4 weights carry INT4 into the rewritten weights; A16W8 TOSA convs and
/// their biases are INT48; the U55 subset separates shared accumulators with
/// `do_not_fuse_duplicate` clones and the generic profile doesn't.
#[test]
fn conv_fixtures_record_special_dtypes_and_flags() {
    let special = |graph: &Graph, node: &replay::Node, index: usize| {
        node_arg(graph, node, index).special.clone()
    };
    for case in cases(REWRITE_CONV) {
        let a8w4 = case.name.ends_with("_a8w4");
        for tosa in case
            .after
            .nodes
            .iter()
            .filter(|node| target(node).is_some_and(|t| TOSA_CONVS.contains(&t)))
        {
            let expected = a8w4.then(|| "INT4".to_string());
            assert_eq!(special(&case.after, tosa, 1), expected, "{}", case.name);
            assert_eq!(tosa.special, None, "{}", case.name);
        }
        assert!(!case.after.nodes.iter().any(|node| node.no_fuse));
    }
    for (fixture, u55) in [(REWRITE_CONV_A16W8, false), (REWRITE_CONV_A16W8_U55, true)] {
        let mut flagged = 0;
        for case in cases(fixture) {
            for tosa in case
                .after
                .nodes
                .iter()
                .filter(|node| target(node).is_some_and(|t| TOSA_CONVS.contains(&t)))
            {
                assert_eq!(tosa.special.as_deref(), Some("INT48"), "{}", case.name);
                assert_eq!(
                    special(&case.after, tosa, 2).as_deref(),
                    Some("INT48"),
                    "{} bias",
                    case.name
                );
            }
            flagged += case.after.nodes.iter().filter(|node| node.no_fuse).count();
        }
        assert_eq!(flagged > 0, u55);
    }
    let u55 = cases(REWRITE_CONV_A16W8_U55);
    let mixed = case(&u55, "mixed_chain_conv_a16w8_u55");
    let clones: Vec<&replay::Node> = mixed.after.nodes.iter().filter(|node| node.no_fuse).collect();
    assert_eq!(clones.len(), 2);
    assert_eq!(clones[0].op, clones[1].op, "the clones read the same operands");
}

/// RemoveGetItemPass gives MAX_POOL2D the input qparams of
/// `max_pool2d_with_indices` and drops the getitem's output qparams
/// (critique A3); the `_with_indices` replay cases undo it with the same maps.
#[test]
fn max_pool_fixtures_record_remove_getitem() {
    let removed = cases(REMOVE_GETITEM);
    let rewritten = cases(REWRITE_MAX_POOL2D);
    for case in &removed {
        for pool in nodes_of(&case.before, MAX_POOL_WITH_INDICES) {
            assert!(matches!(pool.val, Val::Other(_)), "{}", case.name);
        }
        for getitem in nodes_of(&case.before, GETITEM) {
            assert!(qvalue(&getitem.q_out, 0).is_some(), "{}", case.name);
        }
        for pool in nodes_of(&case.after, MAX_POOL) {
            assert!(qvalue(&pool.q_in, 0).is_some(), "{}", case.name);
            assert_eq!(pool.q_out, Some(vec![]), "{}", case.name);
        }
        let base = self::case(&rewritten, &case.name);
        let undone = self::case(&rewritten, &format!("{}_with_indices", case.name));
        assert_eq!(base.after, undone.after);
        assert_eq!(count(&undone.before, &[MAX_POOL]), 0);
        assert_eq!(
            count(&undone.before, &[MAX_POOL_WITH_INDICES]),
            count(&base.before, &[MAX_POOL])
        );
        for getitem in nodes_of(&undone.before, GETITEM) {
            let pool = node_arg(&undone.before, getitem, 0);
            assert_eq!(target(pool), Some(MAX_POOL_WITH_INDICES));
            assert_eq!(args(getitem)[1], replay::Arg::Int(0));
            // The getitem keeps the output qparams RemoveGetItemPass drops.
            assert_eq!(
                qvalue(&getitem.q_out, 0),
                nodes_of(&case.before, GETITEM)
                    .next()
                    .and_then(|source| qvalue(&source.q_out, 0))
            );
        }
    }
}

/// CONST_SHAPE operands are SHAPE-typed non-tensor vals; an int16 bilinear
/// RESIZE is INT48 and an int8 bilinear one int32 with a RESCALE after it.
#[test]
fn upsample_fixture_records_shapes_and_resize_dtypes() {
    for case in cases(REWRITE_UPSAMPLE) {
        for shape in nodes_of(&case.after, "tosa.CONST_SHAPE.default") {
            assert_eq!(shape.special.as_deref(), Some("SHAPE"), "{}", case.name);
            assert!(matches!(shape.val, Val::Other(_)), "{}", case.name);
        }
        for resize in nodes_of(&case.after, "tosa.RESIZE.default") {
            let bilinear = case.name.starts_with("bilinear");
            let int16 = case.name.ends_with("_a16w8");
            let expected = match (bilinear, int16) {
                (true, false) => "torch.int32",
                (_, true) => "torch.int16",
                (false, false) => "torch.int8",
            };
            assert_eq!(dtype(resize), Some(expected), "{}", case.name);
            assert_eq!(
                resize.special.as_deref(),
                (bilinear && int16).then_some("INT48"),
                "{}",
                case.name
            );
            assert_eq!(
                count(&case.after, &["tosa.RESCALE.default"]),
                usize::from(bilinear),
                "{}",
                case.name
            );
        }
    }
}

/// ComputeConstantOpsAOTPass quantizes both zero points with the pool's
/// output qparams (critique A7): the constants hold `q(zp)`, not `zp`.
#[test]
fn avg_pool_aot_fixture_records_quantized_zero_points() {
    let quantize = |value: i64, q: &QValue| -> i64 {
        let scaled = (value as f64 / q.scales[0]).round_ties_even() as i64;
        (scaled + q.zps[0]).clamp(q.qmin, q.qmax)
    };
    let value = |node: &replay::Node| -> i64 {
        match &node.data {
            Some(Data::Bytes(bytes)) if dtype(node) == Some("torch.int8") => {
                i64::from(bytes[0] as i8)
            }
            Some(Data::Bytes(bytes)) => i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
            _ => panic!("zero point without data"),
        }
    };
    for case in cases(REWRITE_AVG_POOL2D_AOT) {
        assert_eq!(count(&case.after, &["aten.full.default"]), 0, "{}", case.name);
        for pool in nodes_of(&case.after, "tosa.AVG_POOL2D.default") {
            let input = qvalue(&pool.q_in, 0).expect("input qparam");
            let output = qvalue(&pool.q_out, 0).expect("output qparam");
            let input_zp = node_arg(&case.after, pool, 1);
            let output_zp = node_arg(&case.after, pool, 2);
            assert!(matches!(input_zp.op, Op::Placeholder { .. }), "{}", case.name);
            assert_eq!(value(input_zp), quantize(input.zps[0], output), "{}", case.name);
            assert_eq!(value(output_zp), quantize(output.zps[0], output), "{}", case.name);
        }
    }
    for case in cases(REWRITE_AVG_POOL2D) {
        for pool in nodes_of(&case.after, "tosa.AVG_POOL2D.default") {
            let input_zp = node_arg(&case.after, pool, 1);
            assert_eq!(target(input_zp), Some("aten.full.default"), "{}", case.name);
        }
    }
}

/// The import and snapshot round trip every captured graph: data (int64
/// narrowed and widened back), per-channel qparams, special dtypes, flags and
/// tuple vals.
#[test]
fn p2_fixtures_round_trip_through_the_importer() {
    let strict = Options {
        dim_orders: DimOrders::All,
        dead_nodes: true,
        float32_activations: false,
    };
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            for (graph, label) in [(&case.before, "before"), (&case.after, "after")] {
                let snapshot = replay::snapshot(&replay::build(graph)).unwrap();
                if let Err(mismatch) = replay::compare(graph, &snapshot, strict) {
                    panic!("{name}/{} ({label}): {mismatch:?}", case.name);
                }
            }
        }
    }
}

#[test]
fn p2_fixtures_print_and_reparse() {
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            let text = format!(
                "format 2\ncase {}\nbefore\n{}after modified=true\n{}end\n",
                case.name, case.before, case.after
            )
            .replace("    ", "");
            let reparsed = replay::parse(&text)
                .unwrap_or_else(|error| panic!("{name}/{}: {error}", case.name));
            assert_eq!(reparsed[0].before, case.before, "{name}/{}", case.name);
            assert_eq!(reparsed[0].after, case.after, "{name}/{}", case.name);
        }
    }
}

/// Only quantized activations import as float32: the TOSA ops' accumulators,
/// RESCALEs and every placeholder keep rc4's dtype.
#[test]
fn p2_imports_float_only_quantized_activations() {
    let conv = cases(REWRITE_CONV);
    let imported = replay::retab_dtypes(&case(&conv, "conv2d_pc").before);
    let dtypes: Vec<Option<&str>> = imported.nodes.iter().map(dtype).collect();
    let conv_position = imported
        .nodes
        .iter()
        .position(|node| target(node) == Some(CONVOLUTION))
        .unwrap();
    for (position, node) in imported.nodes.iter().enumerate() {
        let expected = if position == conv_position {
            "torch.float32"
        } else {
            dtype(&case(&conv, "conv2d_pc").before.nodes[position]).unwrap()
        };
        assert_eq!(dtypes[position], Some(expected), "%{position} {:?}", node.op);
    }
    for fixture in [REWRITE_CONV_A16W8, REWRITE_UPSAMPLE, REWRITE_AVG_POOL2D] {
        for case in cases(fixture) {
            let imported = replay::retab_dtypes(&case.before);
            for (rc4, retab) in case.before.nodes.iter().zip(&imported.nodes) {
                if dtype(rc4) != dtype(retab) {
                    assert!(matches!(rc4.op, Op::Call { .. }), "{}", case.name);
                    assert!(
                        matches!(dtype(rc4), Some("torch.int8" | "torch.int16")),
                        "{}",
                        case.name
                    );
                }
                if matches!(dtype(rc4), Some("torch.int32" | "torch.float32")) {
                    assert_eq!(dtype(rc4), dtype(retab), "{}", case.name);
                }
            }
        }
    }
}

/// What a perfect port produces: rc4's after graph, imported like a before
/// graph. Every case must match it, and keep matching wherever RETAB leaves a
/// quantized activation float32 (a node downstream of a rewrite keeps its
/// float32 val, since RETAB passes don't retrace).
#[test]
fn a_port_that_reproduces_rc4_matches_every_case() {
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            let oracle: IrGraph = replay::build(&replay::retab_dtypes(&case.after));
            let mut actual = replay::snapshot(&oracle).unwrap();
            if let Err(mismatch) = replay::compare_case(&case, &actual, Options::default()) {
                panic!("{name}/{}: {mismatch:?}", case.name);
            }
            for (node, rc4) in actual.nodes.iter_mut().zip(&case.after.nodes) {
                let moves_data = target(rc4).is_some_and(|t| {
                    t.starts_with("aten.permute_copy")
                        || t.starts_with("aten.view_copy")
                        || t.starts_with("aten.slice_copy")
                });
                let quantized = matches!(dtype(rc4), Some("torch.int8" | "torch.int16"))
                    && (moves_data
                        || [&rc4.q_in, &rc4.q_out]
                            .iter()
                            .any(|map| qvalue(map, 0).is_some_and(|q| Some(q.dtype.as_str()) == dtype(rc4))));
                if quantized {
                    if let Val::Tensor { dtype, .. } = &mut node.val {
                        *dtype = "torch.float32".to_string();
                    }
                }
            }
            if let Err(mismatch) = replay::compare_case(&case, &actual, Options::default()) {
                panic!("{name}/{} with float32 activations: {mismatch:?}", case.name);
            }
        }
    }
}

/// A pass that changes nothing never matches a case: every capture rewrote
/// something the comparison sees.
#[test]
fn a_port_that_does_nothing_matches_no_case() {
    struct Noop;

    impl IrPass for Noop {
        fn name(&self) -> &str {
            "Noop"
        }

        fn run(&self, _graph: &mut IrGraph) -> bool {
            false
        }
    }

    for (name, fixture, _, _) in fixtures() {
        for result in replay::replay(&Noop, fixture, Options::default()) {
            assert!(result.mismatch.is_some(), "{name}/{}", result.name);
        }
    }
}

// ---------------------------------------------------------------------------
// Replay hooks: enable each once its port lands (APU-135 P2). See `replay`
// for what is compared and how rc4's int8 activations stand for RETAB's
// float32 ones. Format 2 compares whole graphs, dead nodes included, so a port
// must erase what it rewrites: the aten conv (design §2.3 step 9), §2.8(c)'s
// unused squeeze view, RESCALE and permute, U55's deduplicated heads, and the
// `_with_indices` pool with its getitem (critique B3).
// ---------------------------------------------------------------------------

// S3 (APU-155): IrRewriteConv on INT8 (per-tensor and per-channel) and A8W4.
#[test]
fn rewrite_conv_matches_rc4() {
    replay::check(
        &IrRewriteConvPass {
            is_u55_subset: false,
            in_region: false,
        },
        REWRITE_CONV,
    );
}

// S4 (APU-155): A16W8 INT48 branching, on the generic profile and on U55.
#[test]
fn rewrite_conv_a16w8_matches_rc4() {
    replay::check(
        &IrRewriteConvPass {
            is_u55_subset: false,
            in_region: false,
        },
        REWRITE_CONV_A16W8,
    );
}

#[test]
fn rewrite_conv_a16w8_u55_matches_rc4() {
    replay::check(
        &IrRewriteConvPass {
            is_u55_subset: true,
            in_region: false,
        },
        REWRITE_CONV_A16W8_U55,
    );
}

// S7 (APU-156): IrRewriteMaxPool2d also does RemoveGetItemPass's work; the
// `_with_indices` cases start from RETAB's max_pool2d_with_indices + getitem,
// with the edge args (trailing defaults omitted).
// #[test]
// fn rewrite_max_pool2d_matches_rc4() {
//     replay::check(&IrRewriteMaxPool2dPass, REWRITE_MAX_POOL2D);
// }

// S7 (APU-156): IrRewriteUpsample.
// #[test]
// fn rewrite_upsample_matches_rc4() {
//     replay::check(&IrRewriteUpsamplePass, REWRITE_UPSAMPLE);
// }

// S7 (APU-156): IrRewriteAvgPool2d. Replay REWRITE_AVG_POOL2D if the port
// emits `aten.full` zero points for IrFuseZeroInputConstantsEffectPass to
// materialize (critique B2), REWRITE_AVG_POOL2D_AOT if it creates the
// quantized zero-point constants itself.
// #[test]
// fn rewrite_avg_pool2d_matches_rc4() {
//     replay::check(&IrRewriteAvgPool2dPass, REWRITE_AVG_POOL2D);
// }

// ---------------------------------------------------------------------------
// Adversarial review of S1 (APU-135): each test below pins one gap the review
// found between the fixtures or the harness and the P2 design, since fixed.
// ---------------------------------------------------------------------------

/// RETAB keeps the edge graph's `max_pool2d_with_indices` args up to
/// IrRewriteMaxPool2d: IrNormalizeMaxPool2dInputRank only lifts the input,
/// IrDecomposeMaxPoolDilation skips undilated pools, and IrSizeAdjustInput only
/// replaces arg 0. Those are the args RemoveGetItemPass receives, and the
/// `_with_indices` cases, which start from RETAB's form, carry them, with the
/// omitted stride, padding, dilation and ceil_mode RETAB presents, not the
/// args rc4's DecomposeMaxPool2dPass spells out (`0, 1, False`) before
/// RewriteMaxPool2dPass.
#[test]
fn review_with_indices_pools_keep_the_edge_args() {
    let rewritten = cases(REWRITE_MAX_POOL2D);
    let mut omitted = 0;
    for source in cases(REMOVE_GETITEM) {
        let edge = nodes_of(&source.before, MAX_POOL_WITH_INDICES)
            .next()
            .expect("the edge pool");
        let undone = case(&rewritten, &format!("{}_with_indices", source.name));
        let pool = nodes_of(&undone.before, MAX_POOL_WITH_INDICES)
            .next()
            .expect("the undone pool");
        assert_eq!(&args(pool)[1..], &args(edge)[1..], "{}", undone.name);
        omitted += usize::from(args(pool).len() < 6);
    }
    assert!(omitted > 0, "every pool spells out all six args");
}

/// SizeAdjustInputPass slices a conv's input when the stride leaves a
/// remainder larger than the padding (k3 s2 p0 on an even size, or a 1x1
/// stride-2 projection), so RewriteConvPass sees `slice -> conv` and emits
/// slice -> permute -> view -> CONV2D for conv1d (design C7: "Replay fixtures
/// must use the RCP-output order"), and the port must read the sliced val
/// (F5). grouped2d's slices don't count: they are DecomposeGroupedConvPass's
/// channel slices, along dim 1; nor does cumsum's, which follows its conv.
#[test]
fn review_conv_fixture_replays_size_adjust_slices() {
    let spatially_sliced = |case: &Case| {
        nodes_of(&case.before, CONVOLUTION).any(|conv| {
            let x = node_arg(&case.before, conv, 0);
            target(x) == Some("aten.slice_copy.Tensor")
                && matches!(args(x).get(1), Some(replay::Arg::Int(dim)) if *dim >= 2)
        })
    };
    let conv = cases(REWRITE_CONV);
    for kind in ["conv1d", "conv2d"] {
        assert!(
            conv.iter()
                .any(|case| case.name.starts_with(kind) && spatially_sliced(case)),
            "no {kind} case has a sliced input"
        );
    }
    let conv1d = case(&conv, "conv1d_s2p0_pt");
    let tosa = nodes_of(&conv1d.after, "tosa.CONV2D.default")
        .next()
        .expect("the CONV2D");
    let view = node_arg(&conv1d.after, tosa, 0);
    let permute = node_arg(&conv1d.after, view, 0);
    let slice = node_arg(&conv1d.after, permute, 0);
    assert_eq!(
        [target(slice), target(permute), target(view)],
        [
            Some("aten.slice_copy.Tensor"),
            Some("aten.permute_copy.default"),
            Some("aten.view_copy.default"),
        ]
    );
}

/// The quantizer freezes one quantized weight per conv, so no INT graph
/// reaches RewriteConvPass with two convs on one weight placeholder, not even
/// for one nn.Parameter read by two convs on the same input. rc4's
/// DWConvsModule (one depthwise conv module used twice, the
/// `dw_shared_weights` cases before the review) reads two placeholders with
/// equal data, like any two convs. The design's §2.6 shared-weight handling
/// ("the second lands before the first") is therefore FP-only, and RETAB
/// lowers INT only; a DSL unit test is the place to pin that order if S3 wants
/// it. Should a future rc4 or torchao share the placeholder, this fails and
/// the case covers §2.6 after all.
#[test]
fn review_shared_conv_weights_are_unreachable_in_int() {
    let conv = cases(REWRITE_CONV);
    assert!(!conv
        .iter()
        .any(|case| case.name.starts_with("dw_shared_weights")));
    for name in ["dw_reused_module_pt", "dw_reused_module_pc"] {
        let case = case(&conv, name);
        let weights: Vec<usize> = nodes_of(&case.before, CONVOLUTION)
            .map(|conv| match &args(conv)[1] {
                replay::Arg::Node(position) => *position,
                other => panic!("{name}: weight {other:?}"),
            })
            .collect();
        assert_eq!(weights.len(), 2, "{name}");
        assert_ne!(weights[0], weights[1], "{name}: the convs share a weight");
        assert_eq!(
            case.before.nodes[weights[0]].data, case.before.nodes[weights[1]].data,
            "{name}: the weights hold different data"
        );
    }
}

/// Design risk 6 asks for nsimplify(tolerance=1e-6) parity on nearest scale
/// factors, with 1.6 as the float32 value a quantized export can carry
/// (1.6000000238; rewrite_upsample.py:101-108). The `_f32` cases hold
/// float32-rounded 1.6 and 1/3, which only a tolerance that loose recovers as
/// 8/5 and 1/3 (1e-12 gives 13421773/8388608 and 11184811/33554432); the
/// other scale factors are within 1e-15 of their rationals.
#[test]
fn review_upsample_covers_float32_rounded_scale_factors() {
    let upsample = cases(REWRITE_UPSAMPLE);
    for (name, factor, scale) in [
        ("nearest_scale1_6_f32_pc", 1.6f32, [16, 10, 16, 10]),
        ("nearest_scale_third_f32_pc", 1.0f32 / 3.0, [2, 6, 2, 6]),
    ] {
        let case = case(&upsample, name);
        let rounded = replay::Arg::Float(f64::from(factor));
        let node = nodes_of(&case.before, "aten.upsample_nearest2d.vec")
            .next()
            .expect("the upsample");
        assert_eq!(
            args(node).get(2),
            Some(&replay::Arg::List(vec![rounded.clone(), rounded])),
            "{name}"
        );
        // rc4's RESIZE scale: [2·num, 2·den] per axis of the recovered ratio.
        let resize = nodes_of(&case.after, "tosa.RESIZE.default")
            .next()
            .expect("the RESIZE");
        let expected: Vec<replay::Arg> = scale.iter().map(|&v| replay::Arg::Int(v)).collect();
        assert_eq!(
            args(node_arg(&case.after, resize, 1)),
            &[replay::Arg::List(expected)],
            "{name}"
        );
    }
}

/// At the P2 slots RETAB's qparams come from IrFoldQdq, whose per-channel
/// QParam keeps the first scale and zero point in `scale` and `zp`
/// (`resolve_per_channel_qparam` in src/fx_fold_qdq_ir.rs). The replay
/// imports them the same way, not the way `QParam::from_py` reads an rc4
/// QuantArgs (1.0 and 0), a path RETAB's pipeline never takes for these
/// graphs.
#[test]
fn review_per_channel_qparams_import_like_ir_fold_qdq() {
    let conv = cases(REWRITE_CONV);
    let graph = replay::import(case(&conv, "conv2d_pc"));
    let id = graph
        .live_call_function_ids()
        .into_iter()
        .find(|&id| graph.node(id).target == CONVOLUTION)
        .expect("the conv");
    let weight = graph.node(id).meta.input_qparam_at(1).expect("weight qparam");
    assert!(weight.is_per_channel());
    assert_eq!(
        (weight.scale, weight.zp),
        (weight.per_channel_scales[0], weight.per_channel_zps[0])
    );
}

/// No P2 graph holds a dead node (`p2_fixtures_hold_no_dead_nodes`): rc4's P2
/// passes erase what they replace. So `replay::compare_case`, which the hooks'
/// `replay::check` uses, compares format-2 graphs whole, and a port that never
/// erases the rewritten aten conv (design §2.3 step 9), or leaves §2.8(c)'s
/// unused squeeze view, RESCALE and permute, fails. Nothing downstream removes
/// such a node reliably (LayoutCtx::finish doesn't DCE; ingest drops dead
/// calls only when an effect applied), and in explicit mode a leftover aten
/// conv would reach the X5 guard and silently send the graph to stock.
#[test]
fn review_p2_hooks_reject_leftover_dead_nodes() {
    use crate::fx_ir::{IrArg, NodeMeta, NodeOp};

    let conv = cases(REWRITE_CONV);
    let case = case(&conv, "conv2d_pt");
    // rc4's after graph, as a port that reproduces it would leave it...
    let mut graph = replay::build(&replay::retab_dtypes(&case.after));
    let clean = replay::snapshot(&graph).unwrap();
    assert_eq!(
        replay::compare_case(case, &clean, Options::default()),
        Ok(())
    );
    // ...plus the original convolution, never erased.
    let id = |name: &str| IrArg::Node(graph.by_name(name).expect(name).id);
    let ints = |values: &[i64]| IrArg::List(values.iter().map(|&v| IrArg::Int(v)).collect());
    let args = vec![
        id("quantized_decomposed_quantize_per_tensor_default"),
        id("b__frozen_param0"),
        id("b__frozen_param1"),
        ints(&[1, 1]),
        ints(&[1, 1]),
        ints(&[1, 1]),
        IrArg::Bool(false),
        ints(&[0, 0]),
        IrArg::Int(1),
    ];
    graph.add_node(
        NodeOp::CallFunction,
        "aten_convolution_default".to_string(),
        CONVOLUTION.to_string(),
        args,
        NodeMeta::tensor(vec![1, 4, 8, 8], "torch.float32"),
    );
    let actual = replay::snapshot(&graph).unwrap();
    let leftover: Vec<&replay::Node> = nodes_of(&actual, CONVOLUTION).collect();
    assert!(leftover.len() == 1 && leftover[0].dead);
    // `replay::check` compares with `Options::default()`.
    assert!(
        replay::compare_case(case, &actual, Options::default()).is_err(),
        "a leftover aten conv matched rc4's after graph"
    );
}

/// RETAB's IR never holds rc4's `tosa.X` spelling at the P2 slots: the INT32
/// RESCALEs IrRewriteConv retargets (design §2.8) come from
/// IrInsertInt32Rescale, which builds `backend.tosa.RESCALE.default`
/// (`IrGraph::add_rescale`), and the P2 rewrites create
/// `backend.tosa.MAX_POOL2D`/`AVG_POOL2D`. The replay imports format-2 before
/// graphs in that spelling (rc4's `tosa.RESCALE.default` in every A16W8 case,
/// `tosa.MAX_POOL2D`/`tosa.AVG_POOL2D` in the mixed chains), so a port that
/// matches target strings literally sees what production hands it. The
/// comparison treats the two spellings as one target.
#[test]
fn review_before_graph_tosa_targets_import_in_retab_spelling() {
    let a16w8 = cases(REWRITE_CONV_A16W8);
    let graph = replay::import(case(&a16w8, "conv_add_a16w8"));
    let rescales: Vec<String> = graph
        .live_call_function_ids()
        .into_iter()
        .map(|id| graph.node(id).target.clone())
        .filter(|target| target.ends_with("RESCALE.default"))
        .collect();
    assert_eq!(rescales.len(), 3, "{rescales:?}");
    for target in &rescales {
        assert_eq!(target, "backend.tosa.RESCALE.default");
    }
    for (name, fixture, _, _) in fixtures() {
        for case in cases(fixture) {
            let graph = replay::import(&case);
            for id in graph.live_call_function_ids() {
                let target = &graph.node(id).target;
                assert!(
                    !target.starts_with("tosa."),
                    "{name}/{}: {target}",
                    case.name
                );
            }
        }
    }
}
