//! rc4 replays for the P2 layout passes (APU-135 S1).
//!
//! The fixtures are captured in rc4's own ArmPassManager pipeline while it
//! lowers small quantized models (see the generator's docstring): the graphs
//! RewriteConvPass, RemoveGetItemPass, RewriteMaxPool2dPass,
//! RewriteUpsamplePass and RewriteAvgPool2dPass (alone and with the
//! ComputeConstantOpsAOTPass after it) receive and return. The tests below pin
//! what the fixtures hold, so a regeneration that drops a case or loses a
//! detail fails here. The ports don't exist yet: each has a commented-out
//! replay hook at the end of this file for S3, S4 and S7 to enable.

use std::collections::BTreeSet;

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::replay::{self, Case, Data, DimOrders, Graph, Op, Options, QValue, Val};

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

/// The case lists in the P2 design's §5 table (S3, S4, S7).
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
        "conv3d",
        "linear",
        "linear_in1",
        "conv2d_cin1",
        "conv1d_cin1",
        "conv3d_cin1",
        "two_convs_no_bias",
        "two_convs_one_input",
        "dw_shared_weights",
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
// float32 ones.
// ---------------------------------------------------------------------------

// S3 (APU-155): IrRewriteConv on INT8 (per-tensor and per-channel) and A8W4.
// #[test]
// fn rewrite_conv_matches_rc4() {
//     replay::check(&IrRewriteConvPass { is_u55_subset: false, in_region: false }, REWRITE_CONV);
// }

// S4 (APU-155): A16W8 INT48 branching, on the generic profile and on U55.
// #[test]
// fn rewrite_conv_a16w8_matches_rc4() {
//     replay::check(&IrRewriteConvPass { is_u55_subset: false, in_region: false }, REWRITE_CONV_A16W8);
// }
// #[test]
// fn rewrite_conv_a16w8_u55_matches_rc4() {
//     replay::check(&IrRewriteConvPass { is_u55_subset: true, in_region: false }, REWRITE_CONV_A16W8_U55);
// }

// S7 (APU-156): IrRewriteMaxPool2d also does RemoveGetItemPass's work; the
// `_with_indices` cases start from RETAB's max_pool2d_with_indices + getitem.
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
