//! Tests for `IrRewriteConvPass`'s A16W8 branches and U55 separation (APU-135
//! P2 S4, APU-155): ports of rc4's own tests, then the design's S4 cases
//! (§2.8 with critique A4) that rc4's tests leave out.
//!
//! The graphs are rc4's: the programs rc4's tests build
//! (`rewrite_conv_rc4_tests.txt`, which records RewriteConvPass's result for
//! each), the pipeline-captured A16W8 fixtures (`rewrite_conv_a16w8*.txt`, which
//! `p2_replay_tests` replays whole), and, for the helpers rc4 tests on
//! hand-built FX graphs, the same graphs built by hand.
//!
//! rc4's tests (`backends/arm/test/passes/test_rewrite_conv_pass.py`):
//!
//! | rc4 test | port |
//! |---|---|
//! | `test_rewrite_conv_a16w8_mixed_consumers_restore_int16` (:446-541, six models) | [`rc4_test_rewrite_conv_a16w8_mixed_consumers_restore_int16`]: each model's rc4-test program and its pipeline-captured replay |
//! | `..._rescale_signature_includes_positional_unsigned_flags` (:544) | [`rc4_test_rewrite_conv_rescale_signature_includes_positional_unsigned_flags`] |
//! | `..._without_convolution_does_not_require_context` (:555) | `rewrite_conv_tests::rc4_test_rewrite_conv_without_convolution_does_not_require_context` (S3) |
//! | `..._a16w8_unknown_accumulator_user_is_unchanged` (:565) | [`rc4_test_rewrite_conv_a16w8_unknown_accumulator_user_is_unchanged`] |
//! | `..._a16w8_multi_consumer_rescale_is_not_deduplicated` (:587) | [`rc4_test_rewrite_conv_a16w8_multi_consumer_rescale_is_not_deduplicated`] |
//! | `..._a16w8_deduplication_uses_graph_order` (:605) | [`rc4_test_rewrite_conv_a16w8_deduplication_uses_graph_order`] |
//! | `..._a16w8_unknown_order_user_is_unchanged` (:629) | [`rc4_test_rewrite_conv_a16w8_unknown_order_user_is_unchanged`] |
//! | `..._a16w8_u55_separates_distinct_output_rescales` (:650) | [`rc4_test_rewrite_conv_a16w8_u55_separates_distinct_output_rescales`] |
//! | `..._a16w8_preserves_int32_for_int32_consumers` (:690) | [`rc4_test_rewrite_conv_a16w8_preserves_int32_for_int32_consumers`] |
//! | `test_rewrite_conv1d_a16w8_narrows_instead_of_forking_int32` (:723) | [`rc4_test_rewrite_conv1d_a16w8_narrows_instead_of_forking_int32`] |
//! | `test_rewrite_conv1d_a16w8_shares_one_layout_boundary` (:757) | [`rc4_test_rewrite_conv1d_a16w8_shares_one_layout_boundary`] |
//! | `..._a16w8_preserves_int32_after_permute` (:786) | [`rc4_test_rewrite_conv_a16w8_preserves_int32_after_permute`] |
//! | `test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries` (:828-881, both) | `rewrite_conv_tests::rc4_test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries` (S3) |
//! | `..._vgf_quant_infers_quantized_bias_dtype_from_inputs` (:911, structure only) | `rewrite_conv_tests::rc4_test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs` (S3) |
//! | `..._int16_bias_lowers_to_single_tosa_conv` (:933-972, four modules) | [`rc4_test_rewrite_conv_int16_bias_lowers_to_single_tosa_conv`] |
//!
//! Not ported: `test_rewrite_conv_a16w8_mixed_consumers_lowers_on_u55` (:677)
//! lowers through Vela, which is the lab's (design §5, S4). The six models of
//! :446-541 first check that a whole TOSA lowering succeeds
//! (`_lower_a16w8_to_tosa`); here the pipeline-captured cases stand for that
//! lowering's RewriteConvPass step, and
//! `serialize::explicit_rewrite_conv_tests` serializes their results.

use std::collections::{BTreeSet, HashMap};

use crate::fx_fuse_duplicate_users_ir::node_signature_from;
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, QParam};
use crate::fx_meta::TensorFacts;
use crate::graph::ir::DType;
use crate::layout::ctx::LayoutCtx;
use crate::layout::dtype::{set_special_dtype, special_dtype, SpecialDtype};
use crate::layout::replay::{self, Case};
use crate::layout::rewrite_conv::{
    combine_rescale_scales, deduplicate_a16w8_output_rescales, graph_order,
    separate_u55_a16w8_output_rescales, IrRewriteConvPass,
};
use crate::pipeline_config::PipelineConfig;
use crate::store::TensorData;
use crate::test_ir::{buffer, call, placeholder, qparam};

const RC4_TESTS: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_rc4_tests.txt");
const REWRITE_CONV_A16W8: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8.txt");
const REWRITE_CONV_A16W8_U55: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8_u55.txt");

const CONVOLUTION: &str = "aten.convolution.default";
const PERMUTE: &str = "aten.permute_copy.default";
const VIEW: &str = "aten.view_copy.default";
const ADD: &str = "aten.add.Tensor";
const RESCALE: &str = "backend.tosa.RESCALE.default";
const CONV2D: &str = "backend.tosa.CONV2D.default";
const DEPTHWISE: &str = "backend.tosa.DEPTHWISE_CONV2D.default";
const TRANSPOSE_CONV2D: &str = "backend.tosa.TRANSPOSE_CONV2D.default";
const CONV3D: &str = "backend.tosa.CONV3D.default";
const TOSA_CONVS: [&str; 4] = [CONV2D, DEPTHWISE, TRANSPOSE_CONV2D, CONV3D];
const INT16: &str = "torch.int16";
const INT32: &str = "torch.int32";
const DO_NOT_FUSE: &str = "do_not_fuse_duplicate";

/// The pass for rc4's TOSA spec `spec`, as the pipeline builds it.
fn pass_for(spec: &str) -> IrRewriteConvPass {
    IrRewriteConvPass::new(&PipelineConfig::from_tosa_spec(spec), false)
}

fn generic() -> IrRewriteConvPass {
    pass_for("TOSA-1.0+INT+int16")
}

fn u55() -> IrRewriteConvPass {
    pass_for("TOSA-1.0+INT+int16+int4+u55")
}

fn case(fixture: &str, name: &str) -> Case {
    replay::parse(fixture)
        .expect("a valid fixture")
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

fn imported(fixture: &str, name: &str) -> IrGraph {
    replay::import(&case(fixture, name))
}

/// rc4's graph `name` of `fixture`, rewritten by `pass` without a fallback;
/// the result must be rc4's whole graph.
fn rewritten_with(pass: &IrRewriteConvPass, fixture: &str, name: &str) -> IrGraph {
    replay::check_cases(pass, fixture, &[name]);
    let mut graph = imported(fixture, name);
    crate::stock_fallback::take();
    assert!(pass.run(&mut graph), "{name}");
    assert_eq!(crate::stock_fallback::take(), None, "{name}");
    graph
}

/// Live nodes calling `target`, in graph order.
fn calls(graph: &IrGraph, target: &str) -> Vec<NodeId> {
    graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| graph.node(id).op == NodeOp::CallFunction && graph.node(id).target == target)
        .collect()
}

fn single_call(graph: &IrGraph, target: &str) -> NodeId {
    let found = calls(graph, target);
    assert_eq!(found.len(), 1, "one {target}");
    found[0]
}

fn tosa_convs(graph: &IrGraph) -> Vec<NodeId> {
    graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| TOSA_CONVS.contains(&graph.node(id).target.as_str()))
        .collect()
}

fn users(graph: &IrGraph, id: NodeId) -> Vec<NodeId> {
    graph.live_user_ids(id)
}

fn shape(graph: &IrGraph, id: NodeId) -> Vec<i64> {
    graph.node_shape_vec(id).expect("a tensor value")
}

fn position(graph: &IrGraph, id: NodeId) -> usize {
    graph
        .topological_live_node_ids()
        .iter()
        .position(|&live| live == id)
        .expect("a live node")
}

fn is_rescale_to(graph: &IrGraph, id: NodeId, dtype: &str) -> bool {
    let node = graph.node(id);
    node.target == RESCALE && matches!(&node.args[1], IrArg::Dtype(arg) if arg == dtype)
}

fn floats(arg: &IrArg) -> Vec<f64> {
    match arg {
        IrArg::List(items) => items
            .iter()
            .map(|item| item.as_numeric_f64().expect("a number"))
            .collect(),
        other => panic!("{other:?} is not a list"),
    }
}

fn weight_scales(weight: &QParam) -> Vec<f64> {
    if weight.is_per_channel() {
        weight.per_channel_scales.clone()
    } else {
        vec![weight.scale]
    }
}

/// rc4's `_get_expected_int32_scales` (test_rewrite_conv_pass.py:227-268), on
/// the graph RewriteConvPass receives: for each direct INT32 RESCALE of a
/// conv, `activation * weight / output * user` per output channel, derived
/// independently of the pass.
fn expected_int32_scales(graph: &IrGraph) -> Vec<Vec<f64>> {
    let mut expected = Vec::new();
    for conv in calls(graph, CONVOLUTION) {
        let meta = &graph.node(conv).meta;
        for user in users(graph, conv) {
            if !is_rescale_to(graph, user, INT32) {
                continue;
            }
            let activation = meta.input_qparam_at(0).expect("an activation qparam").scale;
            let weights = weight_scales(meta.input_qparam_at(1).expect("a weight qparam"));
            let output = meta.output_qparam().expect("an output qparam");
            let rescale = graph.node(user);
            assert_eq!(rescale.args[3].as_int(), Some(output.zp));
            let user_scales = floats(&rescale.args[2]);
            assert!(user_scales.len() == 1 || user_scales.len() == weights.len());
            expected.push(
                weights
                    .iter()
                    .zip(user_scales.iter().cycle())
                    .map(|(weight, user)| activation * weight / output.scale * user)
                    .collect(),
            );
        }
    }
    expected
}

/// The nodes a list arg (the output's) holds, in order.
fn node_list(arg: &IrArg) -> Vec<NodeId> {
    match arg {
        IrArg::List(items) => items
            .iter()
            .map(|item| item.as_node().expect("a node"))
            .collect(),
        other => panic!("{other:?} is not a list"),
    }
}

/// `actual == pytest.approx(expected)`: relative 1e-6, absolute 1e-12.
fn approx(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= (1e-6 * e.abs()).max(1e-12))
}

// ---------------------------------------------------------------------------
// rc4's tests on rc4's lowering programs
// ---------------------------------------------------------------------------

/// `test_rewrite_conv_a16w8_mixed_consumers_restore_int16` (:446-541): in each
/// of the six models, exactly one TOSA conv of the model's kind feeds two
/// RESCALEs, one to int16 for the following op and one to int32 for the
/// residual add, and the int32 one's scales are the conv's and the add's
/// RESCALE's multiplied. rc4 checks the program its test builds
/// (`_rewrite_a16w8_convs`); the pipeline-captured case of each model, which
/// `p2_replay_tests` replays with the rest, is checked too.
#[test]
fn rc4_test_rewrite_conv_a16w8_mixed_consumers_restore_int16() {
    for (model, pipeline_case, target) in [
        ("conv", "mixed_chain_conv_a16w8", CONV2D),
        ("depthwise_conv", "mixed_chain_dw_a16w8", CONV2D),
        ("avg_pool", "mixed_chain_avg_pool_a16w8", CONV2D),
        ("max_pool", "mixed_chain_max_pool_a16w8", CONV2D),
        ("conv3d", "mixed_conv3d_a16w8", CONV3D),
        ("transpose_conv2d", "mixed_tconv_a16w8", TRANSPOSE_CONV2D),
    ] {
        let test_case = format!("a16w8_mixed_consumers_{model}");
        for (fixture, name) in [
            (RC4_TESTS, test_case.as_str()),
            (REWRITE_CONV_A16W8, pipeline_case),
        ] {
            let expected = expected_int32_scales(&imported(fixture, name));
            assert!(!expected.is_empty(), "{name}");
            let graph = rewritten_with(&generic(), fixture, name);
            let mixed: Vec<NodeId> = calls(&graph, target)
                .into_iter()
                .filter(|&conv| {
                    let conv_users = users(&graph, conv);
                    let dtypes: BTreeSet<&str> = conv_users
                        .iter()
                        .filter(|&&user| graph.node(user).target == RESCALE)
                        .filter_map(|&user| match &graph.node(user).args[1] {
                            IrArg::Dtype(dtype) => Some(dtype.as_str()),
                            _ => None,
                        })
                        .collect();
                    conv_users.len() == 2 && dtypes == BTreeSet::from([INT16, INT32])
                })
                .collect();
            assert_eq!(mixed.len(), 1, "{name}");
            let int32 = users(&graph, mixed[0])
                .into_iter()
                .find(|&user| is_rescale_to(&graph, user, INT32))
                .expect("the INT32 RESCALE");
            let scales = floats(&graph.node(int32).args[2]);
            assert!(
                expected.iter().any(|expected| approx(&scales, expected)),
                "{name}: {scales:?} is none of {expected:?}"
            );
        }
    }
}

/// `test_rewrite_conv_a16w8_u55_separates_distinct_output_rescales` (:650):
/// on the U55 subset the mixed chain's projection conv is cloned, so there is
/// one TOSA conv more than on the generic profile and each feeds exactly one
/// RESCALE. Checked on the test's program and on the pipeline-captured pair.
#[test]
fn rc4_test_rewrite_conv_a16w8_u55_separates_distinct_output_rescales() {
    for (fixture, generic_case, u55_fixture, u55_case) in [
        (
            RC4_TESTS,
            "a16w8_mixed_consumers_conv",
            RC4_TESTS,
            "a16w8_mixed_consumers_conv_u55",
        ),
        (
            REWRITE_CONV_A16W8,
            "mixed_chain_conv_a16w8",
            REWRITE_CONV_A16W8_U55,
            "mixed_chain_conv_a16w8_u55",
        ),
    ] {
        let conv_targets = |graph: &IrGraph| -> Vec<NodeId> {
            let mut convs = calls(graph, CONV2D);
            convs.extend(calls(graph, DEPTHWISE));
            convs
        };
        let generic_graph = rewritten_with(&generic(), fixture, generic_case);
        let u55_graph = rewritten_with(&u55(), u55_fixture, u55_case);
        let generic_convs = conv_targets(&generic_graph);
        let u55_convs = conv_targets(&u55_graph);
        assert_eq!(u55_convs.len(), generic_convs.len() + 1, "{u55_case}");
        for &conv in &u55_convs {
            let conv_users = users(&u55_graph, conv);
            assert_eq!(conv_users.len(), 1, "{u55_case}");
            assert_eq!(u55_graph.node(conv_users[0]).target, RESCALE, "{u55_case}");
        }
    }
}

/// `test_rewrite_conv_a16w8_preserves_int32_for_int32_consumers` (:690): a
/// conv read only by an INT32 add keeps exactly one RESCALE, to int32,
/// directly on the TOSA conv, with the combined scales; its int16 path goes.
#[test]
fn rc4_test_rewrite_conv_a16w8_preserves_int32_for_int32_consumers() {
    for (fixture, name) in [
        (RC4_TESTS, "a16w8_preserves_int32_for_int32_consumers"),
        (REWRITE_CONV_A16W8, "conv_add_a16w8"),
    ] {
        let expected = expected_int32_scales(&imported(fixture, name));
        let graph = rewritten_with(&generic(), fixture, name);
        let direct: Vec<NodeId> = calls(&graph, RESCALE)
            .into_iter()
            .filter(|&rescale| {
                is_rescale_to(&graph, rescale, INT32)
                    && graph.node(rescale).arg_node(0).is_some_and(|input| {
                        matches!(graph.node(input).target.as_str(), CONV2D | DEPTHWISE)
                    })
            })
            .collect();
        assert_eq!(direct.len(), 1, "{name}");
        assert_eq!(expected.len(), 1, "{name}");
        let scales = floats(&graph.node(direct[0]).args[2]);
        assert!(approx(&scales, &expected[0]), "{name}: {scales:?}");
        // The combination is exactly rc4's, `((s_in * s_w) / s_out) * s_user`.
        assert_eq!(scales, expected[0], "{name}");
    }
}

/// `test_rewrite_conv1d_a16w8_narrows_instead_of_forking_int32` (:723): an
/// A16W8 conv1d read by an INT32 add does not fork: its only user is the
/// RESCALE to its int16 output, then the `[1, 8, 4]` squeeze view and the
/// `[1, 4, 8]` output permute.
#[test]
fn rc4_test_rewrite_conv1d_a16w8_narrows_instead_of_forking_int32() {
    for (pass, fixture, name) in [
        (
            generic(),
            RC4_TESTS,
            "conv1d_a16w8_narrows_instead_of_forking_int32",
        ),
        (generic(), REWRITE_CONV_A16W8, "conv1d_add_a16w8"),
        (u55(), REWRITE_CONV_A16W8_U55, "conv1d_add_a16w8_u55"),
    ] {
        let graph = rewritten_with(&pass, fixture, name);
        let conv = single_call(&graph, CONV2D);
        let forked = calls(&graph, RESCALE)
            .into_iter()
            .filter(|&rescale| {
                is_rescale_to(&graph, rescale, INT32)
                    && graph.node(rescale).arg_node(0) == Some(conv)
            })
            .count();
        assert_eq!(forked, 0, "{name}");
        let [boundary_rescale] = users(&graph, conv)[..] else {
            panic!("{name}: the conv has one user");
        };
        assert_eq!(graph.node(boundary_rescale).target, RESCALE, "{name}");
        let [squeeze_view] = users(&graph, boundary_rescale)[..] else {
            panic!("{name}: the RESCALE has one user");
        };
        assert_eq!(graph.node(squeeze_view).target, VIEW, "{name}");
        assert_eq!(shape(&graph, squeeze_view), [1, 8, 4], "{name}");
        let [boundary_permute] = users(&graph, squeeze_view)[..] else {
            panic!("{name}: the view has one user");
        };
        assert_eq!(graph.node(boundary_permute).target, PERMUTE, "{name}");
        assert_eq!(shape(&graph, boundary_permute), [1, 4, 8], "{name}");
    }
}

/// `test_rewrite_conv1d_a16w8_shares_one_layout_boundary` (:757): an A16W8
/// conv1d read by an amax and a subtraction keeps one boundary RESCALE and
/// one `[1, 4, 8]` output permute.
#[test]
fn rc4_test_rewrite_conv1d_a16w8_shares_one_layout_boundary() {
    for (pass, fixture, name) in [
        (
            generic(),
            RC4_TESTS,
            "conv1d_a16w8_shares_one_layout_boundary",
        ),
        (
            generic(),
            REWRITE_CONV_A16W8,
            "conv1d_shared_consumers_a16w8",
        ),
        (
            u55(),
            REWRITE_CONV_A16W8_U55,
            "conv1d_shared_consumers_a16w8_u55",
        ),
    ] {
        let graph = rewritten_with(&pass, fixture, name);
        let conv = single_call(&graph, CONV2D);
        let boundary_rescales = users(&graph, conv)
            .into_iter()
            .filter(|&user| graph.node(user).target == RESCALE)
            .count();
        assert_eq!(boundary_rescales, 1, "{name}");
        let output_permutes = calls(&graph, PERMUTE)
            .into_iter()
            .filter(|&permute| shape(&graph, permute) == [1, 4, 8])
            .count();
        assert_eq!(output_permutes, 1, "{name}");
    }
}

/// `test_rewrite_conv_a16w8_preserves_int32_after_permute` (:786): an INT32
/// consumer behind a permute of the conv gets exactly one widened path:
/// RESCALE to int32 of the TOSA conv, its layout permute, the source permute's
/// copy and the consumer's RESCALE to int32.
#[test]
fn rc4_test_rewrite_conv_a16w8_preserves_int32_after_permute() {
    for (pass, fixture, name) in [
        (generic(), RC4_TESTS, "a16w8_preserves_int32_after_permute"),
        (generic(), REWRITE_CONV_A16W8, "permuted_int32_a16w8"),
        (u55(), REWRITE_CONV_A16W8_U55, "permuted_int32_a16w8_u55"),
    ] {
        let graph = rewritten_with(&pass, fixture, name);
        let mut widened_paths = 0;
        for rescale in calls(&graph, RESCALE) {
            let node = graph.node(rescale);
            if !is_rescale_to(&graph, rescale, INT32)
                || node
                    .arg_node(0)
                    .is_none_or(|input| graph.node(input).target != CONV2D)
            {
                continue;
            }
            for layout_permute in users(&graph, rescale) {
                if graph.node(layout_permute).target != PERMUTE {
                    continue;
                }
                for source_permute in users(&graph, layout_permute) {
                    if graph.node(source_permute).target != PERMUTE {
                        continue;
                    }
                    widened_paths += users(&graph, source_permute)
                        .into_iter()
                        .filter(|&consumer| is_rescale_to(&graph, consumer, INT32))
                        .count();
                }
            }
        }
        assert_eq!(widened_paths, 1, "{name}");
    }
}

/// `test_rewrite_conv_int16_bias_lowers_to_single_tosa_conv` (:933-972): an
/// A16W8 conv of each kind with a bias becomes one TOSA conv, with no add,
/// and its bias is INT48.
#[test]
fn rc4_test_rewrite_conv_int16_bias_lowers_to_single_tosa_conv() {
    for (name, target) in [
        ("int16_bias_conv2d", CONV2D),
        ("int16_bias_depthwise_conv2d", DEPTHWISE),
        ("int16_bias_conv3d", CONV3D),
        ("int16_bias_transpose_conv2d", TRANSPOSE_CONV2D),
    ] {
        let graph = rewritten_with(&generic(), RC4_TESTS, name);
        let conv = single_call(&graph, target);
        assert!(calls(&graph, ADD).is_empty(), "{name}");
        let bias = graph.node(conv).arg_node(2).expect("a bias");
        assert_eq!(
            special_dtype(&graph.node(bias).meta),
            Some(SpecialDtype::Int48),
            "{name}"
        );
    }
}

/// Every case of `rewrite_conv_rc4_tests.txt`, S3's and S4's, replays whole
/// under the pass its test's TOSA spec gives (the `_u55` case on the U55
/// subset).
#[test]
fn every_rc4_test_program_matches_rc4() {
    let names: Vec<String> = replay::parse(RC4_TESTS)
        .expect("a valid fixture")
        .into_iter()
        .map(|case| case.name)
        .collect();
    assert!(names.len() >= 21, "{names:?}");
    for name in &names {
        let pass = if name.ends_with("_u55") {
            u55()
        } else {
            generic()
        };
        replay::check_cases(&pass, RC4_TESTS, &[name.as_str()]);
    }
}

// ---------------------------------------------------------------------------
// rc4's tests of the U55 helpers on hand-built graphs
// ---------------------------------------------------------------------------

/// A hand-built accumulator: an int32 `[1, 8, 8, 4]` placeholder standing for
/// a TOSA conv, as rc4's helper tests use one.
fn accumulator(graph: &mut IrGraph) -> NodeId {
    placeholder(graph, "accumulator", [1, 8, 8, 4], INT32)
}

/// rc4's `_add_a16w8_rescale_head` (test_rewrite_conv_pass.py:304-323):
/// `RESCALE(accumulator, int16, [1.0], 0, 0, *flags)` and its NCHW permute.
fn add_head(graph: &mut IrGraph, accumulator: NodeId, flags: &[bool]) -> (NodeId, NodeId) {
    let index = graph.nodes.len();
    let mut args = vec![
        IrArg::Node(accumulator),
        IrArg::Dtype(INT16.to_string()),
        IrArg::List(vec![IrArg::Float(1.0)]),
        IrArg::Int(0),
        IrArg::Int(0),
    ];
    args.extend(flags.iter().map(|&flag| IrArg::Bool(flag)));
    let rescale = call(
        graph,
        &format!("rescale_{index}"),
        RESCALE,
        args,
        [1, 8, 8, 4],
        INT16,
    );
    let permute = call(
        graph,
        &format!("permute_{index}"),
        PERMUTE,
        vec![
            IrArg::Node(rescale),
            crate::fx_ir::i64_list_arg(&[0, 3, 1, 2]),
        ],
        [1, 4, 8, 8],
        INT16,
    );
    (rescale, permute)
}

/// `graph.output((...))`.
fn add_output(graph: &mut IrGraph, values: &[NodeId]) -> NodeId {
    graph.add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        vec![IrArg::List(
            values.iter().map(|&id| IrArg::Node(id)).collect(),
        )],
        NodeMeta::default(),
    )
}

/// The live nodes and each one's users, for "the graph is unchanged".
fn structure(graph: &IrGraph) -> Vec<(NodeId, Vec<NodeId>)> {
    graph
        .topological_live_node_ids()
        .into_iter()
        .map(|id| (id, graph.live_user_ids(id)))
        .collect()
}

/// `test_rewrite_conv_rescale_signature_includes_positional_unsigned_flags`
/// (:544): two heads that differ only in a trailing positional unsigned flag
/// sign differently, so the U55 step never merges them.
#[test]
fn rc4_test_rewrite_conv_rescale_signature_includes_positional_unsigned_flags() {
    let mut graph = IrGraph::empty();
    let accumulator = accumulator(&mut graph);
    let (signed_rescale, _) = add_head(&mut graph, accumulator, &[false, false]);
    let (unsigned_rescale, _) = add_head(&mut graph, accumulator, &[false, true]);
    assert_ne!(
        node_signature_from(&graph, signed_rescale, 1),
        node_signature_from(&graph, unsigned_rescale, 1)
    );
    // Beyond rc4's test: equal heads sign alike whatever their inputs.
    let other = placeholder(&mut graph, "other", [1, 8, 8, 4], INT32);
    let (same_flags, _) = add_head(&mut graph, other, &[false, false]);
    assert_eq!(
        node_signature_from(&graph, signed_rescale, 1),
        node_signature_from(&graph, same_flags, 1)
    );
}

/// `test_rewrite_conv_a16w8_unknown_accumulator_user_is_unchanged` (:565): a
/// user that is not a RESCALE makes the deduplication return None and change
/// nothing.
#[test]
fn rc4_test_rewrite_conv_a16w8_unknown_accumulator_user_is_unchanged() {
    let mut graph = IrGraph::empty();
    let accumulator = accumulator(&mut graph);
    let (_, first_permute) = add_head(&mut graph, accumulator, &[]);
    let (_, second_permute) = add_head(&mut graph, accumulator, &[]);
    let unexpected_user = call(
        &mut graph,
        "neg",
        "aten.neg.default",
        vec![IrArg::Node(accumulator)],
        [1, 8, 8, 4],
        INT32,
    );
    add_output(
        &mut graph,
        &[first_permute, second_permute, unexpected_user],
    );
    let before = structure(&graph);
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    let result = deduplicate_a16w8_output_rescales(&mut ctx, accumulator, &node_order);
    assert_eq!(result, None);
    assert!(ctx.is_topological());
    drop(ctx);
    assert_eq!(structure(&graph), before);
}

/// `test_rewrite_conv_a16w8_multi_consumer_rescale_is_not_deduplicated`
/// (:587): a RESCALE with a second user is no head, so it stays, with both
/// its users, though it signs like the first head.
#[test]
fn rc4_test_rewrite_conv_a16w8_multi_consumer_rescale_is_not_deduplicated() {
    let mut graph = IrGraph::empty();
    let accumulator = accumulator(&mut graph);
    let (first_rescale, first_permute) = add_head(&mut graph, accumulator, &[]);
    let (second_rescale, second_permute) = add_head(&mut graph, accumulator, &[]);
    let output = add_output(&mut graph, &[first_permute, second_permute, second_rescale]);
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    let result = deduplicate_a16w8_output_rescales(&mut ctx, accumulator, &node_order);
    assert_eq!(result, Some(vec![first_rescale, second_rescale]));
    assert!(ctx.is_topological());
    drop(ctx);
    let mut second_users = graph.live_user_ids(second_rescale);
    second_users.sort_unstable();
    assert_eq!(second_users, [second_permute, output]);
}

/// `test_rewrite_conv_a16w8_deduplication_uses_graph_order` (:605): of two
/// equal heads the one earlier in the graph survives even when the
/// accumulator's users list the other first. rc4 makes its users order
/// disagree with the graph by retargeting the earlier head after building the
/// later one; RETAB lists users by id, so here the later head is built first
/// and the earlier one inserted before it.
#[test]
fn rc4_test_rewrite_conv_a16w8_deduplication_uses_graph_order() {
    let mut graph = IrGraph::empty();
    let accumulator = accumulator(&mut graph);
    let (late_rescale, late_permute) = add_head(&mut graph, accumulator, &[]);
    let output = add_output(&mut graph, &[late_permute]);
    let mut ctx = LayoutCtx::new(&mut graph);
    let rescale_args = ctx.node(late_rescale).args.clone();
    let rescale_meta = ctx.node(late_rescale).meta.clone();
    let early_rescale = ctx.inserting_before(late_rescale, |ctx| {
        ctx.call_function("early_rescale", RESCALE, rescale_args, rescale_meta)
    });
    let permute_meta = ctx.node(late_permute).meta.clone();
    let permute_dims = ctx.node(late_permute).args[1].clone();
    let early_permute = ctx.inserting_after(early_rescale, |ctx| {
        ctx.call_function(
            "early_permute",
            PERMUTE,
            vec![IrArg::Node(early_rescale), permute_dims],
            permute_meta,
        )
    });
    ctx.set_args(
        output,
        vec![IrArg::List(vec![
            IrArg::Node(early_permute),
            IrArg::Node(late_permute),
        ])],
    );
    let node_order = graph_order(ctx.order());
    assert_eq!(
        ctx.graph().live_user_ids(accumulator),
        [late_rescale, early_rescale]
    );
    assert!(node_order[&early_rescale] < node_order[&late_rescale]);

    let result = deduplicate_a16w8_output_rescales(&mut ctx, accumulator, &node_order);
    assert_eq!(result, Some(vec![early_rescale]));
    assert!(ctx.is_erased(late_rescale));
    assert!(ctx.is_erased(late_permute));
    assert!(ctx.is_topological());
    assert_eq!(
        node_list(&ctx.node(output).args[0]),
        [early_permute, early_permute]
    );
}

/// `test_rewrite_conv_a16w8_unknown_order_user_is_unchanged` (:629): a user
/// the graph order taken before the U55 step doesn't know makes the
/// deduplication return None and change nothing.
#[test]
fn rc4_test_rewrite_conv_a16w8_unknown_order_user_is_unchanged() {
    let mut graph = IrGraph::empty();
    let accumulator = accumulator(&mut graph);
    let (_, first_permute) = add_head(&mut graph, accumulator, &[]);
    let node_order = graph_order(LayoutCtx::new(&mut graph).order());
    let (_, second_permute) = add_head(&mut graph, accumulator, &[]);
    add_output(&mut graph, &[first_permute, second_permute]);
    let before = structure(&graph);
    let mut ctx = LayoutCtx::new(&mut graph);
    let result = deduplicate_a16w8_output_rescales(&mut ctx, accumulator, &node_order);
    assert_eq!(result, None);
    assert!(ctx.is_topological());
    drop(ctx);
    assert_eq!(structure(&graph), before);
}

// ---------------------------------------------------------------------------
// The design's S4 cases beyond rc4's tests
// ---------------------------------------------------------------------------

/// `_combine_rescale_scales` (RCP:511-531): elementwise products, a single
/// factor standing for every channel, and rc4's two errors.
#[test]
fn rescale_scales_combine_like_rc4() {
    assert_eq!(combine_rescale_scales(&[2.0], &[3.0]), Ok(vec![6.0]));
    assert_eq!(
        combine_rescale_scales(&[2.0, 4.0], &[0.5]),
        Ok(vec![1.0, 2.0])
    );
    assert_eq!(
        combine_rescale_scales(&[0.5], &[2.0, 4.0, 6.0]),
        Ok(vec![1.0, 2.0, 3.0])
    );
    assert_eq!(
        combine_rescale_scales(&[1.0, 2.0], &[3.0, 4.0]),
        Ok(vec![3.0, 8.0])
    );
    let error = combine_rescale_scales(&[1.0, 2.0], &[1.0, 2.0, 3.0]).unwrap_err();
    assert!(
        error.contains("incompatible scale counts: 2 and 3"),
        "{error}"
    );
    let error = combine_rescale_scales(&[], &[1.0]).unwrap_err();
    assert!(error.contains("empty"), "{error}");
}

/// RCP:705-748: a direct INT32 user keeps its place and kwargs, takes exactly
/// five args (`int32_user.args = (...)` drops trailing flags), the combined
/// scales, input zero point 0 and its own output zero point, and holds the
/// accumulator's NHWC shape in int32. Its users read a permute directly after
/// it that gives back the conv's NCHW shape.
#[test]
fn direct_int32_users_are_retargeted_in_place() {
    let mut graph = imported(REWRITE_CONV_A16W8, "conv_add_a16w8");
    let conv = single_call(&graph, CONVOLUTION);
    let user = users(&graph, conv)
        .into_iter()
        .find(|&user| is_rescale_to(&graph, user, INT32))
        .expect("the add's INT32 RESCALE");
    let user_name = graph.node(user).name.clone();
    let [add] = users(&graph, user)[..] else {
        panic!("the RESCALE feeds the add");
    };
    let add_name = graph.node(add).name.clone();
    let conv_shape = shape(&graph, conv);
    graph.set_arg(user, 4, IrArg::Int(3));
    let mut args = graph.node(user).args.clone();
    args.extend([IrArg::Bool(false), IrArg::Bool(false)]);
    graph.set_args(user, args);
    graph.set_kwarg(
        user,
        "rounding_mode",
        IrArg::Str("SINGLE_ROUND".to_string()),
    );

    crate::stock_fallback::take();
    assert!(generic().run(&mut graph));
    assert_eq!(crate::stock_fallback::take(), None);
    let user = graph.by_name(&user_name).expect("the RESCALE stays").id;
    let tosa = single_call(&graph, CONV2D);
    let node = graph.node(user);
    assert_eq!(node.arg_node(0), Some(tosa));
    assert_eq!(node.args.len(), 5);
    assert_eq!(node.args[3].as_int(), Some(0));
    assert_eq!(node.args[4].as_int(), Some(3));
    assert_eq!(
        node.kwargs.get("rounding_mode").and_then(IrArg::as_str),
        Some("SINGLE_ROUND")
    );
    assert_eq!(shape(&graph, user), shape(&graph, tosa));
    assert_eq!(graph.node_dtype(user), Some(INT32));
    let [permute] = users(&graph, user)[..] else {
        panic!("the RESCALE feeds its permute only");
    };
    assert_eq!(graph.node(permute).target, PERMUTE);
    assert_eq!(position(&graph, permute), position(&graph, user) + 1);
    assert_eq!(shape(&graph, permute), conv_shape);
    assert_eq!(graph.node_dtype(permute), Some(INT32));
    let add = graph.by_name(&add_name).expect("the add").id;
    assert_eq!(graph.node(add).arg_node(0), Some(permute));
    // Every conv user moved to INT32, so the int16 path is gone.
    assert!(calls(&graph, RESCALE)
        .into_iter()
        .all(|rescale| !is_rescale_to(&graph, rescale, INT16)
            || graph.node(rescale).arg_node(0) != Some(tosa)));
}

/// RCP:750-816: a permute of the conv that other nodes read too stays, on the
/// int16 path, while its INT32 RESCALE reads the widened copy.
#[test]
fn a_shared_permute_user_stays_on_the_int16_path() {
    let mut graph = imported(REWRITE_CONV_A16W8, "permuted_int32_a16w8");
    let conv = calls(&graph, CONVOLUTION)[0];
    let permute = users(&graph, conv)
        .into_iter()
        .find(|&user| graph.node(user).target == PERMUTE)
        .expect("the permute user");
    let permute_name = graph.node(permute).name.clone();
    let output = graph.output_node().expect("an output").id;
    let mut outputs = match &graph.node(output).args[0] {
        IrArg::List(items) => items.clone(),
        other => panic!("output {other:?}"),
    };
    outputs.push(IrArg::Node(permute));
    graph.set_args(output, vec![IrArg::List(outputs)]);

    crate::stock_fallback::take();
    assert!(generic().run(&mut graph));
    assert_eq!(crate::stock_fallback::take(), None);
    let permute = graph.by_name(&permute_name).expect("the permute stays").id;
    let source = graph.node(permute).arg_node(0).expect("an input");
    // The int16 output permute of the producer's RESCALE to int16.
    let narrowed = graph.node(source).arg_node(0).expect("the RESCALE");
    assert!(is_rescale_to(&graph, narrowed, INT16));
    let [output] = users(&graph, permute)[..] else {
        panic!("only the output reads the permute now");
    };
    assert_eq!(graph.node(output).op, NodeOp::Output);
    // The INT32 RESCALE reads the copy of the permute on the widened branch.
    let copy = graph
        .by_name(&format!("{permute_name}_int32"))
        .expect("the copy")
        .id;
    let [int32] = users(&graph, copy)[..] else {
        panic!("the copy feeds the INT32 RESCALE");
    };
    assert!(is_rescale_to(&graph, int32, INT32));
    assert_eq!(
        graph.node(copy).arg_int_list(1),
        graph.node(permute).arg_int_list(1)
    );
    let widened = graph
        .node(graph.node(copy).arg_node(0).expect("the layout permute"))
        .arg_node(0)
        .expect("the widened RESCALE");
    assert!(is_rescale_to(&graph, widened, INT32));
    // The copy is `create_node(from_node=permute)`: emptied qparam maps.
    let meta = &graph.node(copy).meta;
    assert!(meta.input_qparams().is_some_and(|map| map.is_empty()));
    assert!(meta.output_qparams().is_some_and(|map| map.is_empty()));
}

/// A TOSA conv with INT48, qparams and one int16 and three int32 RESCALE
/// users, built by hand for the U55 step: an int16 head, two equal int32
/// heads, and an int32 RESCALE the output reads directly besides its permute
/// (no head).
struct SeparationGraph {
    graph: IrGraph,
    tosa: NodeId,
}

fn separation_graph() -> SeparationGraph {
    let mut graph = IrGraph::empty();
    let x = placeholder(&mut graph, "x", [1, 8, 8, 4], INT16);
    let weight = buffer(
        &mut graph,
        "weight",
        TensorData {
            data: vec![1; 16],
            shape: vec![4, 1, 1, 4],
            dtype: DType::Int8,
        },
    );
    graph
        .node_mut(weight)
        .meta
        .set_tensor_facts(TensorFacts::new(vec![4, 1, 1, 4], "torch.int8"));
    let bias = buffer(
        &mut graph,
        "bias",
        TensorData {
            data: vec![0; 16],
            shape: vec![4],
            dtype: DType::Int32,
        },
    );
    graph
        .node_mut(bias)
        .meta
        .set_tensor_facts(TensorFacts::new(vec![4], INT32));
    let mut meta = NodeMeta::tensor(vec![1, 8, 8, 4], INT32);
    meta.set_input_qparams_from([
        (0, qparam(0.25, 0, INT16)),
        (1, qparam(0.5, 0, "torch.int8")),
    ]);
    meta.set_single_output_qparam(qparam(0.125, 0, INT16));
    set_special_dtype(&mut meta, SpecialDtype::Int48);
    let ints = crate::fx_ir::i64_list_arg;
    let tosa = graph.add_node(
        NodeOp::CallFunction,
        "conv_tosa".to_string(),
        CONV2D.to_string(),
        vec![
            IrArg::Node(x),
            IrArg::Node(weight),
            IrArg::Node(bias),
            ints(&[1, 1]),
            ints(&[0, 0, 0, 0]),
            ints(&[1, 1]),
        ],
        meta,
    );
    let mut outputs = Vec::new();
    let mut last_rescale = None;
    for (index, (dtype, scale)) in [(INT16, 1.0), (INT32, 2.0), (INT32, 2.0), (INT32, 3.0)]
        .into_iter()
        .enumerate()
    {
        let rescale = call(
            &mut graph,
            &format!("rescale_{index}"),
            RESCALE,
            vec![
                IrArg::Node(tosa),
                IrArg::Dtype(dtype.to_string()),
                IrArg::List(vec![IrArg::Float(scale)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
            [1, 8, 8, 4],
            dtype,
        );
        outputs.push(call(
            &mut graph,
            &format!("permute_{index}"),
            PERMUTE,
            vec![IrArg::Node(rescale), ints(&[0, 3, 1, 2])],
            [1, 4, 8, 8],
            dtype,
        ));
        last_rescale = Some(rescale);
    }
    outputs.extend(last_rescale);
    add_output(&mut graph, &outputs);
    SeparationGraph { graph, tosa }
}

/// RCP:633-663 with critique A4: after the equal heads merge, every RESCALE
/// left after the first reads its own clone of the TOSA conv, head or not,
/// inserted directly before it with the conv's args, qparams and INT48; the
/// conv and its clones are marked `do_not_fuse_duplicate`.
#[test]
fn u55_clones_the_conv_for_every_rescale_after_the_first() {
    let SeparationGraph { mut graph, tosa } = separation_graph();
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    separate_u55_a16w8_output_rescales(&mut ctx, tosa, &node_order);
    assert!(ctx.is_topological());
    ctx.finish();

    // The second int32 head merged into the first: its permute's users moved.
    let live = |name: &str| {
        graph
            .by_name(name)
            .filter(|node| node.alive)
            .map(|node| node.id)
    };
    assert_eq!(live("rescale_2"), None);
    assert_eq!(live("permute_2"), None);
    let output = graph.output_node().expect("an output");
    let permute_1 = live("permute_1").expect("the first int32 head's permute");
    assert_eq!(
        node_list(&output.args[0]),
        [
            live("permute_0").unwrap(),
            permute_1,
            permute_1,
            live("permute_3").unwrap(),
            live("rescale_3").unwrap(),
        ]
    );

    // Three RESCALEs left: the conv keeps the first, two clones feed the rest.
    let convs = tosa_convs(&graph);
    assert_eq!(convs.len(), 3);
    let original = live("conv_tosa").expect("the conv");
    assert_eq!(convs[0], original);
    for (conv, rescale) in convs.iter().zip(["rescale_0", "rescale_1", "rescale_3"]) {
        let rescale = live(rescale).expect("a RESCALE");
        assert_eq!(users(&graph, *conv), [rescale]);
        let node = graph.node(*conv);
        assert_eq!(node.meta.extra_bool(DO_NOT_FUSE), Some(true));
        assert_eq!(special_dtype(&node.meta), Some(SpecialDtype::Int48));
        // The same target, args and kwargs.
        assert_eq!(
            node_signature_from(&graph, *conv, 0),
            node_signature_from(&graph, original, 0)
        );
        assert_eq!(
            node.meta.input_qparam_at(1),
            graph.node(original).meta.input_qparam_at(1)
        );
        assert_eq!(
            node.meta.output_qparam(),
            graph.node(original).meta.output_qparam()
        );
        if *conv != original {
            assert_eq!(position(&graph, *conv) + 1, position(&graph, rescale));
        }
    }
}

/// RCP:633-650: the U55 step leaves an accumulator alone when it has one
/// user, when a user is not a RESCALE, or when merging leaves one RESCALE.
#[test]
fn u55_leaves_single_users_unknown_users_and_merged_heads_alone() {
    // One user.
    let SeparationGraph { mut graph, tosa } = separation_graph();
    let output = graph.output_node().expect("an output").id;
    let permute_0 = graph.by_name("permute_0").unwrap().id;
    graph.set_args(output, vec![IrArg::List(vec![IrArg::Node(permute_0)])]);
    for index in 1..4 {
        let permute = graph.by_name(&format!("permute_{index}")).unwrap().id;
        graph.erase(permute);
        let rescale = graph.by_name(&format!("rescale_{index}")).unwrap().id;
        graph.erase(rescale);
    }
    let before = structure(&graph);
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    separate_u55_a16w8_output_rescales(&mut ctx, tosa, &node_order);
    drop(ctx);
    assert_eq!(structure(&graph), before);
    assert_eq!(graph.node(tosa).meta.extra_bool(DO_NOT_FUSE), None);

    // A user that is not a RESCALE.
    let SeparationGraph { mut graph, tosa } = separation_graph();
    let output = graph.output_node().expect("an output").id;
    let mut outputs = match &graph.node(output).args[0] {
        IrArg::List(items) => items.clone(),
        other => panic!("output {other:?}"),
    };
    outputs.push(IrArg::Node(tosa));
    graph.set_args(output, vec![IrArg::List(outputs)]);
    let before = structure(&graph);
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    separate_u55_a16w8_output_rescales(&mut ctx, tosa, &node_order);
    drop(ctx);
    assert_eq!(structure(&graph), before);
    assert_eq!(graph.node(tosa).meta.extra_bool(DO_NOT_FUSE), None);

    // Two equal heads: they merge, and one RESCALE is left, so no clone.
    let SeparationGraph { mut graph, tosa } = separation_graph();
    let output = graph.output_node().expect("an output").id;
    let permute_1 = graph.by_name("permute_1").unwrap().id;
    let permute_2 = graph.by_name("permute_2").unwrap().id;
    graph.set_args(
        output,
        vec![IrArg::List(vec![
            IrArg::Node(permute_1),
            IrArg::Node(permute_2),
        ])],
    );
    for name in ["permute_0", "rescale_0", "permute_3", "rescale_3"] {
        let id = graph.by_name(name).unwrap().id;
        graph.erase(id);
    }
    let mut ctx = LayoutCtx::new(&mut graph);
    let node_order = graph_order(ctx.order());
    separate_u55_a16w8_output_rescales(&mut ctx, tosa, &node_order);
    ctx.finish();
    assert_eq!(tosa_convs(&graph).len(), 1);
    let tosa = tosa_convs(&graph)[0];
    assert_eq!(users(&graph, tosa).len(), 1);
    assert_eq!(graph.node(tosa).meta.extra_bool(DO_NOT_FUSE), None);
}

/// The generic profile never separates: two equal INT32 heads survive, as
/// rc4's do (A1 §9: two TRANSPOSEs remain), and nothing is marked.
#[test]
fn the_generic_profile_keeps_duplicate_heads() {
    let graph = rewritten_with(&generic(), REWRITE_CONV_A16W8, "conv_add_sub_a16w8");
    let tosa = single_call(&graph, CONV2D);
    let heads = users(&graph, tosa);
    assert_eq!(heads.len(), 2);
    assert_eq!(
        node_signature_from(&graph, heads[0], 1),
        node_signature_from(&graph, heads[1], 1)
    );
    assert!(graph
        .live_nodes()
        .all(|node| node.meta.extra_bool(DO_NOT_FUSE).is_none()));
    // U55 merges them into one head, and one RESCALE needs no clone.
    let graph = rewritten_with(&u55(), REWRITE_CONV_A16W8_U55, "conv_add_sub_a16w8_u55");
    let tosa = single_call(&graph, CONV2D);
    assert_eq!(users(&graph, tosa).len(), 1);
    assert_eq!(graph.node(tosa).meta.extra_bool(DO_NOT_FUSE), None);
}

/// The U55 flag comes from the TOSA spec (`PipelineConfig::from_tosa_spec`,
/// rc4's `TosaSpecification.is_U55_subset`).
#[test]
fn the_u55_step_follows_the_tosa_spec() {
    assert!(u55().is_u55_subset);
    assert!(!generic().is_u55_subset);
    assert!(!pass_for("TOSA-1.0+INT+int16+int4+cf").is_u55_subset);
    let region = IrRewriteConvPass::new(
        &PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+u55"),
        true,
    );
    assert!(region.is_u55_subset && region.in_region);
}

/// Each U55 clone gets a fresh name, so none can pick up a lowering effect
/// keyed by an old one (design F9).
#[test]
fn u55_clones_take_fresh_names() {
    let graph = rewritten_with(&u55(), REWRITE_CONV_A16W8_U55, "mixed_chain_conv_a16w8_u55");
    let mut names = HashMap::new();
    for node in graph.live_nodes() {
        assert!(
            names.insert(node.name.clone(), node.id).is_none(),
            "{}",
            node.name
        );
    }
    let clones: Vec<&str> = tosa_convs(&graph)
        .into_iter()
        .map(|conv| graph.node(conv).name.as_str())
        .filter(|name| name.ends_with("_clone"))
        .collect();
    assert_eq!(clones.len(), 1, "{clones:?}");
}

// ---------------------------------------------------------------------------
// Review of APU-155 S4
// ---------------------------------------------------------------------------

/// `permuted_int32_a16w8_u55` (`A16W8PermutedInt32Consumer` on the U55 subset)
/// at `[1, 4, 257, 257]`: the shapes scaled so the conv's source permute fails
/// U55's worst-case TRANSPOSE rule (257 * 257 > 65536). The 1x1 convs keep
/// their weights, so only values change.
fn large_permuted_int32_u55() -> IrGraph {
    let fixture = REWRITE_CONV_A16W8_U55.replace("[1, 4, 8, 8]", "[1, 4, 257, 257]");
    let case = replay::parse(&fixture)
        .expect("a valid fixture")
        .into_iter()
        .find(|case| case.name == "permuted_int32_a16w8_u55")
        .expect("the case");
    replay::import(&case)
}

/// Review of APU-155 S4 (minor; design F8/F9 family). On U55, which edge
/// permutes `IrDecomposePermuteForU55` splits is decided in Python, by node
/// name, on the edge program (`_u55_permutes_to_split`,
/// `PipelineConfig::u55_split_permutes`). When every user of a conv's source
/// permute is an INT32 RESCALE, the A16W8 branching erases that permute and
/// gives its consumers a copy named `{permute}_int32` on the widened branch,
/// so a split decided for the edge permute reaches no live node and is
/// dropped silently, as F9 describes for effects keyed by an erased name.
/// Before S4, S3 sent these graphs to stock, which splits the permute; rc4
/// runs DecomposePermuteForU55Pass (`arm_pass_manager.py:736`) on the graph
/// after RewriteConvPass, so it judges the copy itself. F8 accepts that
/// permutes the rewrite creates are never split in P2, but the module docs
/// don't say that an edge permute already listed for splitting loses its
/// split. The test passes once the listed permute's split reaches the
/// widened copy (for example, by carrying its membership in
/// `u55_split_permutes` to the copy), or once such a conv falls back.
#[test]
#[ignore = "BUG: on U55 the A16W8 branching erases a source permute listed in \
            u55_split_permutes and its {permute}_int32 copy is never split"]
fn a_u55_split_listed_source_permute_keeps_its_split() {
    let mut graph = large_permuted_int32_u55();
    let producer = calls(&graph, CONVOLUTION)[0];
    let source_permute = users(&graph, producer)
        .into_iter()
        .find(|&user| graph.node(user).target == PERMUTE)
        .expect("the source permute");
    let name = graph.node(source_permute).name.clone();
    let mut config = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+int4+u55");
    config.u55_split_permutes.insert(name.clone());
    let split = || crate::fx_decompose_permute_for_u55_ir::IrDecomposePermuteForU55Pass {
        split_permutes: config.u55_split_permutes.clone(),
    };
    const CAT: &str = "aten.cat.default";

    // Control: without the rewrite, the listed permute is split into sliced
    // permutes and a cat.
    let mut unrewritten = large_permuted_int32_u55();
    assert!(split().run(&mut unrewritten), "the permute is split-worthy");
    assert_eq!(calls(&unrewritten, CAT).len(), 1);

    crate::stock_fallback::take();
    assert!(IrRewriteConvPass::new(&config, false).run(&mut graph));
    assert_eq!(crate::stock_fallback::take(), None);
    // The INT32 RESCALE the permute fed now reads the widened copy.
    let copy = graph
        .by_name(&format!("{name}_int32"))
        .filter(|node| node.alive)
        .map(|node| node.id)
        .expect("the widened copy of the source permute");
    assert!(users(&graph, copy)
        .into_iter()
        .any(|user| is_rescale_to(&graph, user, INT32)));
    split().run(&mut graph);
    assert_eq!(
        calls(&graph, CAT).len(),
        1,
        "the split decided for {name} reaches no live permute: {name} is {}",
        if graph.by_name(&name).is_some_and(|node| node.alive) {
            "live"
        } else {
            "erased"
        }
    );
}
