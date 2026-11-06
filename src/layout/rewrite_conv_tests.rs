//! Tests for `IrRewriteConvPass` (APU-135 P2 S3, APU-155): ports of rc4's own
//! RewriteConvPass tests, then the design's S3 list (§5): every kind, the
//! static pads, zero biases, INT4, Cin = 1, Linear, grouped pieces, constant
//! convs, regions, and each stock fallback.
//!
//! The graphs are rc4's: cases of the pipeline-captured fixtures
//! (`rewrite_conv*.txt`, which `p2_replay_tests` replays whole) and of
//! `rewrite_conv_rc4_tests.txt`, which records the programs rc4's pass tests
//! build and the graph RewriteConvPass returns for each. A test that needs a
//! graph rc4 never produces (a fallback's trigger) edits one of them.
//!
//! rc4's tests (`backends/arm/test/passes/test_rewrite_conv_pass.py`):
//!
//! | rc4 test | port |
//! |---|---|
//! | `test_rewrite_conv_tosa_FP` (:438) | [`rc4_test_rewrite_conv_tosa_fp`] |
//! | `..._without_convolution_does_not_require_context` (:555) | [`rc4_test_rewrite_conv_without_convolution_does_not_require_context`] |
//! | `test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries` (:828-881, both) | [`rc4_test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries`] |
//! | `..._vgf_quant_infers_quantized_bias_dtype_from_inputs` (:911) | [`rc4_test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs`] |
//! | `..._dynamic_keeps_static_padding_when_symbolic_remainder_is_zero` (:975) | [`rc4_test_rewrite_conv_keeps_static_padding_when_the_remainder_is_zero`], on the test's static example shape |
//! | the three static `_adjust_pad_if_needed` tests (:1002-1027) | [`rc4_test_rewrite_conv_adjust_pad_if_needed_static`] |
//!
//! Not ported: the A16W8 tests (:446-827, :933-972) are S4's; the six
//! symbolic `_adjust_pad_if_needed` tests (:1029-1141) need SymInts, which
//! RETAB's static IR has none of; `:884` tests FoldAndAnnotateQParamsPass and
//! `:899` a whole VGF lowering, not this pass.

use std::collections::HashMap;

use crate::fx_ir::{
    IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp, NodeTensor, PlaceholderTensorBinding, QParam,
    TensorSource,
};
use crate::fx_meta::TensorFacts;
use crate::graph::ir::DType;
use crate::layout::dtype::{special_dtype, SpecialDtype};
use crate::layout::replay::{self, Case};
use crate::layout::rewrite_conv::{adjust_pad_if_needed, IrRewriteConvPass};
use crate::lowering_effects::LoweringInputKind;
use crate::store::TensorData;

const REWRITE_CONV: &str = include_str!("../../tests/fixtures/layout_replay/rewrite_conv.txt");
const REWRITE_CONV_A16W8: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8.txt");
const REWRITE_CONV_A16W8_U55: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8_u55.txt");
const RC4_TESTS: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_rc4_tests.txt");

const CONVOLUTION: &str = "aten.convolution.default";
const PERMUTE: &str = "aten.permute_copy.default";
const VIEW: &str = "aten.view_copy.default";
const CLAMP: &str = "aten.clamp.default";
const RESCALE: &str = "backend.tosa.RESCALE.default";
const CONV2D: &str = "backend.tosa.CONV2D.default";
const DEPTHWISE: &str = "backend.tosa.DEPTHWISE_CONV2D.default";
const TRANSPOSE_CONV2D: &str = "backend.tosa.TRANSPOSE_CONV2D.default";
const CONV3D: &str = "backend.tosa.CONV3D.default";
const TOSA_CONVS: [&str; 4] = [CONV2D, DEPTHWISE, TRANSPOSE_CONV2D, CONV3D];

fn parent_pass() -> IrRewriteConvPass {
    IrRewriteConvPass {
        is_u55_subset: false,
        in_region: false,
    }
}

fn region_pass() -> IrRewriteConvPass {
    IrRewriteConvPass {
        is_u55_subset: false,
        in_region: true,
    }
}

fn case(fixture: &str, name: &str) -> Case {
    replay::parse(fixture)
        .expect("a valid fixture")
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

/// rc4's before graph of `name`, as RETAB's importer builds it.
fn imported(fixture: &str, name: &str) -> IrGraph {
    replay::import(&case(fixture, name))
}

/// Run `pass`; returns what it returned and the stock fallback it requested.
fn run_with(pass: &IrRewriteConvPass, graph: &mut IrGraph) -> (bool, Option<String>) {
    crate::stock_fallback::take();
    let modified = pass.run(graph);
    (modified, crate::stock_fallback::take())
}

/// Run the parent pass, which must rewrite without a fallback.
fn rewrite(graph: &mut IrGraph) {
    let (modified, fallback) = run_with(&parent_pass(), graph);
    assert_eq!(fallback, None);
    assert!(modified);
}

fn rewritten(fixture: &str, name: &str) -> IrGraph {
    let mut graph = imported(fixture, name);
    rewrite(&mut graph);
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

/// The live TOSA convolutions, in graph order.
fn tosa_convs(graph: &IrGraph) -> Vec<NodeId> {
    graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| TOSA_CONVS.contains(&graph.node(id).target.as_str()))
        .collect()
}

fn named(graph: &IrGraph, name: &str) -> NodeId {
    graph
        .live_nodes()
        .find(|node| node.name == name)
        .unwrap_or_else(|| panic!("no live node {name}"))
        .id
}

fn arg(graph: &IrGraph, id: NodeId, index: usize) -> NodeId {
    graph
        .node(id)
        .arg_node(index)
        .unwrap_or_else(|| panic!("{} arg {index} is not a node", graph.node(id).name))
}

fn shape(graph: &IrGraph, id: NodeId) -> Vec<i64> {
    graph.node_shape_vec(id).expect("a tensor value")
}

fn ints(graph: &IrGraph, id: NodeId, index: usize) -> Vec<i64> {
    graph.node(id).arg_int_list(index).expect("an int list")
}

fn position(graph: &IrGraph, id: NodeId) -> usize {
    graph
        .topological_live_node_ids()
        .iter()
        .position(|&live| live == id)
        .expect("a live node")
}

fn data(graph: &IrGraph, id: NodeId) -> &TensorData {
    &graph.node(id).tensor.as_ref().expect("a constant").data
}

fn is_parameter(node: &IrNode) -> bool {
    matches!(
        node.tensor.as_ref().map(|tensor| &tensor.source),
        Some(TensorSource::ExportedProgramInput(
            PlaceholderTensorBinding {
                input_kind: LoweringInputKind::Parameter,
                ..
            }
        ))
    )
}

/// A reference permutation of `data`'s elements, for checking the rewritten
/// weights independently of `permute_tensor_data`: element `out[i]` takes
/// `in[j]` where `j`'s coordinates are `i`'s, read through `dims`.
fn reference_permute(data: &TensorData, dims: &[usize]) -> Vec<u8> {
    let shape: Vec<usize> = data.shape.iter().map(|&dim| dim as usize).collect();
    let out_shape: Vec<usize> = dims.iter().map(|&dim| shape[dim]).collect();
    let size = data.dtype.storage_size_bytes();
    let count: usize = shape.iter().product();
    let mut out = Vec::with_capacity(data.data.len());
    for index in 0..count {
        let mut rest = index;
        let mut coords = vec![0; shape.len()];
        for axis in (0..out_shape.len()).rev() {
            coords[dims[axis]] = rest % out_shape[axis];
            rest /= out_shape[axis];
        }
        let source = coords
            .iter()
            .zip(&shape)
            .fold(0, |flat, (&coord, &extent)| flat * extent + coord);
        out.extend_from_slice(&data.data[source * size..(source + 1) * size]);
    }
    out
}

/// Each use of `id` as `(user name, arg index)`: names survive the
/// renumbering a run does, ids don't (see `output_permutes_keep_the_conv_values`).
fn uses_of(graph: &IrGraph, id: NodeId) -> Vec<(String, usize)> {
    graph
        .live_user_ids(id)
        .into_iter()
        .flat_map(|user| {
            let node = graph.node(user);
            node.args
                .iter()
                .enumerate()
                .filter(move |(_, arg)| arg.contains_node(id))
                .map(move |(index, _)| (node.name.clone(), index))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// rc4's tests
// ---------------------------------------------------------------------------

/// `test_rewrite_conv_tosa_FP`: rc4 runs DWConvsModule, one depthwise conv
/// module used twice, through RewriteConvPass alone in FP and compares the
/// program with eager. The numerics rest on the rewritten weights, which are
/// checked here element by element besides rc4's whole graph. Both convs read
/// one weight and one bias: each gets its own HWCM copy of the weight, the
/// second's inserted first (design §2.6), and both keep the shared bias.
#[test]
fn rc4_test_rewrite_conv_tosa_fp() {
    replay::check_cases(
        &parent_pass(),
        RC4_TESTS,
        &["tosa_fp_dw_convs_shared_weights"],
    );
    let mut graph = imported(RC4_TESTS, "tosa_fp_dw_convs_shared_weights");
    let source = named(&graph, "p_seq_2_weight");
    let source_data = data(&graph, source).clone();
    assert_eq!(source_data.shape, [6, 1, 2, 2]);
    rewrite(&mut graph);

    let convs = calls(&graph, DEPTHWISE);
    assert_eq!(convs.len(), 2);
    assert!(calls(&graph, CONVOLUTION).is_empty());
    let weights: Vec<NodeId> = convs.iter().map(|&conv| arg(&graph, conv, 1)).collect();
    assert_ne!(weights[0], weights[1]);
    assert!(position(&graph, weights[1]) < position(&graph, weights[0]));
    let expected = reference_permute(&source_data, &[2, 3, 0, 1]);
    for &weight in &weights {
        let weight_data = data(&graph, weight);
        assert_eq!(weight_data.shape, [2, 2, 6, 1]);
        assert_eq!(weight_data.dtype, DType::Float32);
        assert_eq!(weight_data.data, expected);
        assert!(is_parameter(graph.node(weight)), "the source's kind");
    }
    let bias = named(&graph, "p_seq_2_bias");
    assert_eq!(arg(&graph, convs[0], 2), bias);
    assert_eq!(arg(&graph, convs[1], 2), bias);
    // FP: no RESCALE, and the TOSA ops compute in float32.
    assert!(calls(&graph, RESCALE).is_empty());
    for &conv in &convs {
        assert_eq!(graph.node_dtype(conv), Some("torch.float32"));
    }
}

/// `test_rewrite_conv_without_convolution_does_not_require_context`: a graph
/// without a convolution is not modified.
#[test]
fn rc4_test_rewrite_conv_without_convolution_does_not_require_context() {
    replay::check_cases(&parent_pass(), RC4_TESTS, &["no_convolution"]);
    for pass in [parent_pass(), region_pass()] {
        let mut graph = imported(RC4_TESTS, "no_convolution");
        assert_eq!(run_with(&pass, &mut graph), (false, None));
    }
}

/// `test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries` (both cases):
/// a Conv1d's TOSA conv reads a `[1, 1, 8, 4]` view of a `[0, 2, 1]` input
/// permute and an OHWI or HWCM weight, and its output goes through a squeeze
/// view and a `[0, 2, 1]` permute back to the conv's shape.
#[test]
fn rc4_test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries() {
    for (name, target, weight_shape, output_shape) in [
        (
            "conv1d_atomic_rank3_layout_boundaries",
            CONV2D,
            [6, 1, 3, 4],
            [1, 6, 8],
        ),
        (
            "conv1d_dw_atomic_rank3_layout_boundaries",
            DEPTHWISE,
            [1, 3, 4, 2],
            [1, 8, 8],
        ),
    ] {
        replay::check_cases(&parent_pass(), RC4_TESTS, &[name]);
        let graph = rewritten(RC4_TESTS, name);
        let conv = single_call(&graph, target);
        let input_view = arg(&graph, conv, 0);
        assert_eq!(graph.node(input_view).target, VIEW, "{name}");
        let input_permute = arg(&graph, input_view, 0);
        assert_eq!(graph.node(input_permute).target, PERMUTE, "{name}");
        assert_eq!(ints(&graph, input_permute, 1), [0, 2, 1], "{name}");
        assert_eq!(shape(&graph, input_view), [1, 1, 8, 4], "{name}");
        assert_eq!(shape(&graph, arg(&graph, conv, 1)), weight_shape, "{name}");
        let output_view = calls(&graph, VIEW)
            .into_iter()
            .find(|&view| graph.node(view).arg_node(0) == Some(conv))
            .expect("the output view reads the conv");
        let [output_permute] = graph.live_user_ids(output_view)[..] else {
            panic!("{name}: the output view has one user");
        };
        assert_eq!(graph.node(output_permute).target, PERMUTE, "{name}");
        assert_eq!(ints(&graph, output_permute, 1), [0, 2, 1], "{name}");
        assert_eq!(shape(&graph, output_permute), output_shape, "{name}");
    }
}

/// `test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs`: the
/// VGF-quantized TinyConvReluCat whose first conv has no bias gets exactly one
/// `*_bias` placeholder, and it is int32.
#[test]
fn rc4_test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs() {
    replay::check_cases(
        &parent_pass(),
        RC4_TESTS,
        &["vgf_quant_no_bias_conv_relu_cat"],
    );
    let graph = rewritten(RC4_TESTS, "vgf_quant_no_bias_conv_relu_cat");
    let biases: Vec<&IrNode> = graph
        .live_nodes()
        .filter(|node| node.op == NodeOp::Placeholder && node.name.ends_with("_bias"))
        .collect();
    assert_eq!(biases.len(), 1);
    assert_eq!(biases[0].meta.val_dtype(), Some("torch.int32"));
}

/// `test_rewrite_conv_dynamic_keeps_static_padding_when_symbolic_remainder_is_zero`
/// on the test's example shape: k3 s3 p0 on 9x12 leaves no remainder, so the
/// CONV2D's pad stays `[0, 0, 0, 0]`, all plain ints. The test's point, that a
/// symbolic remainder known to be zero is not kept symbolic, has no
/// counterpart in RETAB's static IR.
#[test]
fn rc4_test_rewrite_conv_keeps_static_padding_when_the_remainder_is_zero() {
    replay::check_cases(
        &parent_pass(),
        RC4_TESTS,
        &["static_padding_remainder_zero"],
    );
    let graph = rewritten(RC4_TESTS, "static_padding_remainder_zero");
    let conv = single_call(&graph, CONV2D);
    assert_eq!(ints(&graph, conv, 4), [0, 0, 0, 0]);
}

/// The static `_adjust_pad_if_needed` tests (`:1002-1027`).
#[test]
fn rc4_test_rewrite_conv_adjust_pad_if_needed_static() {
    // ..._static_allows_negative_padding_until_later_validation
    let error = adjust_pad_if_needed(6, 2, 3, 0, 1).unwrap_err();
    assert!(error.contains("SizeAdjustInputPass"), "{error}");
    // ..._static_positive_padding_stays_non_negative
    assert_eq!(adjust_pad_if_needed(8, 2, 3, 2, 1), Ok(1));
    // ..._static_exact_remainder_matches_pad
    assert_eq!(adjust_pad_if_needed(6, 1, 3, 1, 1), Ok(0));
}

// ---------------------------------------------------------------------------
// Kinds, layouts and weights
// ---------------------------------------------------------------------------

/// Each kind's target, input permute, weight layout and output permute (design
/// §2.4), on rc4's graphs. rc4's depthwise rule makes Cin = 1 convs depthwise
/// in 1-D and 2-D, Linear(1, Co) and cumsum's kernel included, while a rank-5
/// one stays CONV3D. The weight bytes are the source's, permuted by the
/// kind's order (and reshaped for HWCM and conv1d), checked against a
/// reference permutation; replaying rc4's graphs checks them against rc4's.
#[test]
fn every_kind_takes_rc4s_target_layout_and_weight() {
    struct Kind {
        case: &'static str,
        target: &'static str,
        pre: &'static [i64],
        weight: &'static [i64],
        weight_dims: &'static [usize],
        post: &'static [i64],
    }
    let nhwc: &[i64] = &[0, 2, 3, 1];
    let nchw: &[i64] = &[0, 3, 1, 2];
    let nlc: &[i64] = &[0, 2, 1];
    #[rustfmt::skip]
    let kinds = [
        Kind { case: "conv2d_pt", target: CONV2D, pre: nhwc, weight: &[4, 3, 3, 3], weight_dims: &[0, 2, 3, 1], post: nchw },
        Kind { case: "dw2d_pt", target: DEPTHWISE, pre: nhwc, weight: &[3, 3, 4, 2], weight_dims: &[2, 3, 0, 1], post: nchw },
        Kind { case: "dw2d_kx1_pt", target: DEPTHWISE, pre: nhwc, weight: &[3, 1, 4, 2], weight_dims: &[2, 3, 0, 1], post: nchw },
        Kind { case: "tconv2d_pt", target: TRANSPOSE_CONV2D, pre: nhwc, weight: &[4, 3, 3, 3], weight_dims: &[1, 2, 3, 0], post: nchw },
        Kind { case: "conv1d_pt", target: CONV2D, pre: nlc, weight: &[6, 1, 3, 4], weight_dims: &[0, 2, 1], post: nlc },
        Kind { case: "conv1d_dw_pt", target: DEPTHWISE, pre: nlc, weight: &[1, 3, 4, 2], weight_dims: &[1, 2, 0], post: nlc },
        Kind { case: "conv3d_pt", target: CONV3D, pre: &[0, 2, 3, 4, 1], weight: &[5, 3, 3, 3, 3], weight_dims: &[0, 2, 3, 4, 1], post: &[0, 4, 1, 2, 3] },
        Kind { case: "linear_pt", target: CONV2D, pre: nhwc, weight: &[4, 1, 1, 8], weight_dims: &[0, 2, 3, 1], post: nchw },
        Kind { case: "conv2d_cin1_pt", target: DEPTHWISE, pre: nhwc, weight: &[3, 3, 1, 4], weight_dims: &[2, 3, 0, 1], post: nchw },
        Kind { case: "conv1d_cin1_pt", target: DEPTHWISE, pre: nlc, weight: &[1, 3, 1, 4], weight_dims: &[1, 2, 0], post: nlc },
        Kind { case: "conv3d_cin1_pt", target: CONV3D, pre: &[0, 2, 3, 4, 1], weight: &[4, 3, 3, 3, 1], weight_dims: &[0, 2, 3, 4, 1], post: &[0, 4, 1, 2, 3] },
        Kind { case: "linear_in1_pt", target: DEPTHWISE, pre: nhwc, weight: &[1, 1, 1, 4], weight_dims: &[2, 3, 0, 1], post: nchw },
        Kind { case: "cumsum_pt", target: DEPTHWISE, pre: nhwc, weight: &[4, 1, 1, 1], weight_dims: &[2, 3, 0, 1], post: nchw },
    ];
    for kind in kinds {
        let label = kind.case;
        let mut graph = imported(REWRITE_CONV, kind.case);
        let conv = single_call(&graph, CONVOLUTION);
        let source_data = data(&graph, arg(&graph, conv, 1)).clone();
        let conv_shape = shape(&graph, conv);
        rewrite(&mut graph);

        let tosa = single_call(&graph, kind.target);
        assert_eq!(tosa_convs(&graph), [tosa], "{label}");
        let mut input = arg(&graph, tosa, 0);
        if kind.pre.len() == 3 {
            assert_eq!(graph.node(input).target, VIEW, "{label}");
            input = arg(&graph, input, 0);
        }
        assert_eq!(graph.node(input).target, PERMUTE, "{label}");
        assert_eq!(ints(&graph, input, 1), kind.pre, "{label}");

        let weight = arg(&graph, tosa, 1);
        let weight_data = data(&graph, weight);
        assert_eq!(weight_data.shape, kind.weight, "{label}");
        assert_eq!(
            weight_data.data,
            reference_permute(&source_data, kind.weight_dims),
            "{label}"
        );
        assert_eq!(graph.node_shape(weight), Some(kind.weight), "{label}");

        let rescale = single_call(&graph, RESCALE);
        assert_eq!(arg(&graph, rescale, 0), tosa, "{label}");
        let sole_user = |id: NodeId| {
            let [user] = graph.live_user_ids(id)[..] else {
                panic!("{label}: {} has one user", graph.node(id).name);
            };
            user
        };
        let mut output = sole_user(rescale);
        if kind.pre.len() == 3 {
            assert_eq!(graph.node(output).target, VIEW, "{label}");
            output = sole_user(output);
        }
        assert_eq!(graph.node(output).target, PERMUTE, "{label}");
        assert_eq!(ints(&graph, output, 1), kind.post, "{label}");
        assert_eq!(shape(&graph, output), conv_shape, "{label}");
        // The accumulator is int32 whatever its inherited int8 output qparam
        // says (critique A9); the RESCALE narrows it to the output's int8.
        assert_eq!(graph.node_dtype(tosa), Some("torch.int32"), "{label}");
        assert_eq!(graph.node_dtype(rescale), Some("torch.int8"), "{label}");
    }
}

/// Transposed convs take `[-p, -p + op]` per axis as their TOSA out_pad and
/// keep the stride (RCP:948-985); the rest take rc4's duplicated pads with the
/// trailing one trimmed by the stride remainder (design §2.3, A1 §7.1), and a
/// conv1d's go on the W axis behind `[0, 0]`. SizeAdjustInputPass's slices
/// (s2p0) leave no remainder, and the pass reads the sliced shape.
#[test]
fn pads_follow_rc4s_adjustment() {
    for (name, target, pad_index, pad) in [
        ("conv2d_pt", CONV2D, 4, &[1, 1, 1, 1][..]),
        ("conv2d_s2p1_pt", CONV2D, 4, &[1, 0, 1, 0]),
        ("conv2d_s3p2_pt", CONV2D, 4, &[2, 0, 2, 2]),
        ("conv2d_s2p0_pt", CONV2D, 4, &[0, 0, 0, 0]),
        ("conv2d_dilation2_pt", CONV2D, 4, &[2, 2, 2, 2]),
        ("conv3d_pt", CONV3D, 4, &[1, 1, 1, 0, 1, 0]),
        ("conv1d_pt", CONV2D, 4, &[0, 0, 1, 1]),
        ("conv1d_s2p0_pt", CONV2D, 4, &[0, 0, 0, 0]),
        ("cumsum_pt", DEPTHWISE, 4, &[3, 3, 0, 0]),
        ("tconv2d_pt", TRANSPOSE_CONV2D, 3, &[-1, 0, -1, 0]),
    ] {
        let graph = rewritten(REWRITE_CONV, name);
        let tosa = single_call(&graph, target);
        assert_eq!(ints(&graph, tosa, pad_index), pad, "{name}");
    }
    let graph = rewritten(REWRITE_CONV, "conv1d_pt");
    let conv = single_call(&graph, CONV2D);
    assert_eq!(ints(&graph, conv, 3), [1, 2], "conv1d stride on W");
    assert_eq!(ints(&graph, conv, 5), [1, 1]);
    let graph = rewritten(REWRITE_CONV, "conv2d_s2p0_pt");
    let input = arg(&graph, single_call(&graph, CONV2D), 0);
    assert_eq!(shape(&graph, input), [1, 7, 7, 3], "the sliced input");
    let graph = rewritten(REWRITE_CONV, "tconv2d_pt");
    assert_eq!(
        ints(&graph, single_call(&graph, TRANSPOSE_CONV2D), 4),
        [2, 2]
    );
}

/// A conv without a bias gets int32 zeros as a parameter placeholder inserted
/// before the first placeholder (RCP:160-195), so two such convs leave their
/// biases first, in reverse conv order.
#[test]
fn zero_biases_go_first_in_reverse_conv_order() {
    let graph = rewritten(REWRITE_CONV, "two_convs_no_bias_pt");
    let convs = calls(&graph, CONV2D);
    assert_eq!(convs.len(), 2);
    let biases: Vec<NodeId> = convs.iter().map(|&conv| arg(&graph, conv, 2)).collect();
    let placeholders = graph.live_placeholder_ids();
    let order: Vec<NodeId> = graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|id| placeholders.contains(id))
        .collect();
    assert_eq!(order[..2], [biases[1], biases[0]]);
    for (bias, channels) in biases.into_iter().zip([4, 5]) {
        let node = graph.node(bias);
        assert!(node.name.ends_with("_bias"), "{}", node.name);
        assert!(is_parameter(node));
        let bias_data = data(&graph, bias);
        assert_eq!(
            (bias_data.dtype, bias_data.shape.as_slice()),
            (DType::Int32, &[channels][..])
        );
        assert!(bias_data.data.iter().all(|&byte| byte == 0));
        assert_eq!(special_dtype(&node.meta), None, "INT48 is A16W8's");
    }
}

/// A8W4: the rewritten weight carries the source's INT4 marker (RCP:377-378)
/// and stays one int8 byte per element (packing happens at serialization).
#[test]
fn int4_weights_keep_their_marker() {
    for name in ["conv2d_a8w4", "dw2d_a8w4", "conv1d_a8w4", "linear_a8w4"] {
        let graph = rewritten(REWRITE_CONV, name);
        let [tosa] = tosa_convs(&graph)[..] else {
            panic!("{name}: one TOSA conv");
        };
        let weight = graph.node(arg(&graph, tosa, 1));
        assert_eq!(
            special_dtype(&weight.meta),
            Some(SpecialDtype::Int4),
            "{name}"
        );
        assert_eq!(data(&graph, weight.id).dtype, DType::Int8, "{name}");
        assert_eq!(special_dtype(&graph.node(tosa).meta), None, "{name}");
    }
}

/// IrDecomposeGroupedConv's pieces rewrite like any conv: each gets its own
/// input permute, OHWI weight after its slice constant, and RESCALE with its
/// group's per-channel scales; the cat reads both output permutes.
#[test]
fn grouped_pieces_rewrite_independently() {
    let graph = rewritten(REWRITE_CONV, "grouped2d_pc");
    let convs = calls(&graph, CONV2D);
    assert_eq!(convs.len(), 2);
    let cat = single_call(&graph, "aten.cat.default");
    let Some(IrArg::List(items)) = graph.node(cat).args.first() else {
        panic!("cat tensors");
    };
    let inputs: Vec<NodeId> = items.iter().filter_map(IrArg::as_node).collect();
    for (piece, &conv) in convs.iter().enumerate() {
        let weight = arg(&graph, conv, 1);
        assert_eq!(shape(&graph, weight), [3, 3, 3, 2]);
        let source = named(
            &graph,
            &format!("aten_slice_copy_tensor_{}_fused_const", piece + 2),
        );
        assert_eq!(position(&graph, weight), position(&graph, source) + 1);
        let [rescale] = graph.live_user_ids(conv)[..] else {
            panic!("one RESCALE per piece");
        };
        let Some(IrArg::List(scales)) = graph.node(rescale).args.get(2) else {
            panic!("RESCALE scales");
        };
        assert_eq!(scales.len(), 3);
        let [output] = graph.live_user_ids(rescale)[..] else {
            panic!("one output permute per piece");
        };
        assert_eq!(inputs[piece], output);
    }
}

/// Every replacement keeps the value of the conv it replaces: each user's
/// argument has the conv's shape afterwards (critique B10's internal check).
/// A user that is itself a conv reads it through its TOSA op's input permute.
#[test]
fn output_permutes_keep_the_conv_values() {
    for case in replay::parse(REWRITE_CONV).expect("a valid fixture") {
        let mut graph = replay::import(&case);
        let mut expected = Vec::new();
        for conv in calls(&graph, CONVOLUTION) {
            let conv_shape = shape(&graph, conv);
            for (user, index) in uses_of(&graph, conv) {
                let is_conv = graph.node(named(&graph, &user)).target == CONVOLUTION;
                expected.push((user, index, is_conv, conv_shape.clone()));
            }
        }
        assert!(!expected.is_empty(), "{}", case.name);
        rewrite(&mut graph);
        for (user, index, is_conv, conv_shape) in expected {
            if is_conv {
                let tosa = named(&graph, &format!("{user}_tosa"));
                let mut input = arg(&graph, tosa, 0);
                if graph.node(input).target == VIEW {
                    input = arg(&graph, input, 0);
                }
                let replacement = arg(&graph, input, 0);
                assert_eq!(graph.node(replacement).target, PERMUTE, "{}", case.name);
                assert_eq!(shape(&graph, replacement), conv_shape, "{}", case.name);
                continue;
            }
            let user_id = named(&graph, &user);
            let replacement = match &graph.node(user_id).args[index] {
                IrArg::Node(id) => *id,
                IrArg::List(items) => items
                    .iter()
                    .filter_map(IrArg::as_node)
                    .find(|&id| graph.node(id).target == PERMUTE)
                    .expect("a permute in the list"),
                other => panic!("{}: {other:?}", case.name),
            };
            assert_eq!(graph.node(replacement).target, PERMUTE, "{}", case.name);
            assert_eq!(shape(&graph, replacement), conv_shape, "{}", case.name);
        }
    }
}

/// Created nodes take rc4's meta: the TOSA op inherits the conv's qparams
/// (output qparams included), the permutes and RESCALE get emptied maps, and
/// every value is contiguous; names derive from the conv's (critique B12).
#[test]
fn created_nodes_take_rc4s_meta_and_names() {
    let mut graph = imported(REWRITE_CONV, "conv2d_pc");
    let conv = single_call(&graph, CONVOLUTION);
    let conv_name = graph.node(conv).name.clone();
    let conv_meta = graph.node(conv).meta.clone();
    rewrite(&mut graph);
    let tosa = single_call(&graph, CONV2D);
    let meta = &graph.node(tosa).meta;
    assert_eq!(
        meta.input_qparams().map(|map| map.cloned_pairs()),
        conv_meta.input_qparams().map(|map| map.cloned_pairs())
    );
    assert_eq!(meta.output_qparam(), conv_meta.output_qparam());
    for id in calls(&graph, PERMUTE)
        .into_iter()
        .chain(calls(&graph, RESCALE))
    {
        let meta = &graph.node(id).meta;
        assert!(meta.input_qparams().is_some_and(|map| map.is_empty()));
        assert!(meta.output_qparams().is_some_and(|map| map.is_empty()));
    }
    for id in graph.topological_live_node_ids() {
        let meta = &graph.node(id).meta;
        if let Some(rank) = meta.rank() {
            if graph.node(id).name.starts_with(&conv_name) {
                assert_eq!(
                    meta.val_dim_order(),
                    Some(&(0..rank).collect::<Vec<_>>()[..])
                );
            }
        }
    }
    let names: Vec<&str> = graph
        .topological_live_node_ids()
        .into_iter()
        .map(|id| graph.node(id).name.as_str())
        .filter(|name| name.starts_with(conv_name.as_str()))
        .collect();
    let suffixes: Vec<String> = names
        .iter()
        .map(|name| name[conv_name.len()..].to_string())
        .collect();
    assert_eq!(
        suffixes,
        ["_weight_ohwi", "_nhwc", "_tosa", "_rescale", "_nchw"]
    );
}

// ---------------------------------------------------------------------------
// Quantization details
// ---------------------------------------------------------------------------

/// RCP:481-488: `(s_in * s_w[i]) / s_out` per output channel, with exactly
/// this association, and the output zero point.
#[test]
fn rescale_scales_combine_in_rc4s_order() {
    for name in ["conv2d_pt", "conv2d_pc", "tconv2d_pc", "dw2d_pc"] {
        let mut graph = imported(REWRITE_CONV, name);
        let conv = single_call(&graph, CONVOLUTION);
        let meta = graph.node(conv).meta.clone();
        rewrite(&mut graph);
        let input = meta.input_qparam_at(0).expect("activation").scale;
        let weight = meta.input_qparam_at(1).expect("weight");
        let output = meta.output_qparam().expect("output");
        let weight_scales = if weight.is_per_channel() {
            weight.per_channel_scales.clone()
        } else {
            vec![weight.scale]
        };
        let expected: Vec<f64> = weight_scales
            .iter()
            .map(|&scale| (input * scale) / output.scale)
            .collect();
        let rescale = graph.node(single_call(&graph, RESCALE));
        assert_eq!(float_list(&rescale.args[2]), expected, "{name}");
        assert_eq!(rescale.args[3].as_int(), Some(0), "{name}");
        assert_eq!(rescale.args[4].as_int(), Some(output.zp), "{name}");
    }
}

/// `_get_effective_output_qparams` (RCP:420-450): a conv without output
/// qparams takes those of a sole `clamp` or `view_copy` user.
#[test]
fn output_qparams_fall_back_to_a_sole_clamp_or_view_user() {
    for target in [CLAMP, VIEW] {
        let mut graph = imported(REWRITE_CONV, "conv2d_pt");
        let conv = single_call(&graph, CONVOLUTION);
        let output_qparam = graph.node(conv).meta.output_qparam().cloned().expect("oq");
        let mut moved = output_qparam.clone();
        moved.scale *= 2.0;
        moved.zp = 7;
        graph.node_mut(conv).meta.clear_output_qparams();
        let conv_shape = shape(&graph, conv);
        let args = if target == CLAMP {
            vec![IrArg::Node(conv), IrArg::Int(-128), IrArg::Int(127)]
        } else {
            vec![IrArg::Node(conv), crate::fx_ir::i64_list_arg(&conv_shape)]
        };
        let mut meta = graph.node(conv).meta.clone();
        meta.clear_input_qparams();
        meta.set_single_output_qparam(moved.clone());
        let user = graph.add_call_function("activation", target, args, meta);
        let output = graph.output_node().expect("output").id;
        graph.replace_input_use(output, conv, user);
        rewrite(&mut graph);
        let rescale = graph.node(single_call(&graph, RESCALE));
        assert_eq!(rescale.args[4].as_int(), Some(7), "{target}");
        let tosa = graph.node(single_call(&graph, CONV2D));
        let expected = (tosa.meta.input_qparam_at(0).unwrap().scale
            * tosa.meta.input_qparam_at(1).unwrap().scale)
            / moved.scale;
        assert_eq!(float_list(&rescale.args[2]), [expected], "{target}");
        // The TOSA op inherits the conv's maps as they are: no output qparams.
        assert!(tosa.meta.output_qparams().is_none(), "{target}");
    }
}

fn float_list(arg: &IrArg) -> Vec<f64> {
    match arg {
        IrArg::List(items) => items.iter().filter_map(IrArg::as_float).collect(),
        other => panic!("not a float list: {other:?}"),
    }
}

/// `_get_effective_conv_input_qparams` (RCP:452-462): a conv missing its
/// activation qparam takes input qparam 0 of its input node, here Linear's
/// input permute, and the rewrite is the one the complete qparams give.
#[test]
fn a_missing_activation_qparam_comes_from_the_input_node() {
    let complete = rewritten(REWRITE_CONV, "linear_pt");
    let mut graph = imported(REWRITE_CONV, "linear_pt");
    let conv = single_call(&graph, CONVOLUTION);
    let input = arg(&graph, conv, 0);
    let activation = graph
        .node(conv)
        .meta
        .input_qparam_at(0)
        .cloned()
        .expect("iq 0");
    let mut qparams = graph.node(conv).meta.cloned_input_qparams().expect("iq");
    qparams = qparams
        .cloned_pairs()
        .into_iter()
        .filter(|(index, _)| *index != 0)
        .collect();
    graph.node_mut(conv).meta.set_input_qparams(qparams);
    graph
        .node_mut(input)
        .meta
        .set_single_input_qparam(activation.clone());
    rewrite(&mut graph);
    let tosa = graph.node(single_call(&graph, CONV2D));
    assert_eq!(tosa.meta.input_qparam_at(0), Some(&activation));
    let rescale = |graph: &IrGraph| float_list(&graph.node(single_call(graph, RESCALE)).args[2]);
    assert_eq!(rescale(&graph), rescale(&complete));
}

/// A16W8 convs without INT32 consumers need none of S4's branching (design
/// §2.8): the TOSA conv and its bias (the conv's own or the zero one) are
/// INT48 (RCP:414-418, :856), the RESCALE narrows to int16, and rc4's graph
/// matches on the generic profile and on U55.
#[test]
fn a16w8_without_int32_consumers_matches_rc4() {
    let models = [
        "conv2d",
        "dw2d",
        "tconv2d",
        "conv3d",
        "conv1d",
        "linear",
        "conv1d_add",
        "conv1d_shared_consumers",
    ];
    let generic: Vec<String> = models
        .iter()
        .map(|model| format!("{model}_a16w8"))
        .collect();
    let names: Vec<&str> = generic.iter().map(String::as_str).collect();
    replay::check_cases(&parent_pass(), REWRITE_CONV_A16W8, &names);
    let u55: Vec<String> = models
        .iter()
        .filter(|model| **model != "conv3d")
        .map(|model| format!("{model}_a16w8_u55"))
        .collect();
    let names: Vec<&str> = u55.iter().map(String::as_str).collect();
    let u55_pass = IrRewriteConvPass {
        is_u55_subset: true,
        in_region: false,
    };
    replay::check_cases(&u55_pass, REWRITE_CONV_A16W8_U55, &names);

    // Without its bias the conv gets INT48 zeros.
    let mut graph = imported(REWRITE_CONV_A16W8, "conv2d_a16w8");
    let conv = single_call(&graph, CONVOLUTION);
    graph.set_arg(conv, 2, IrArg::None);
    rewrite(&mut graph);
    let tosa = single_call(&graph, CONV2D);
    assert_eq!(
        special_dtype(&graph.node(tosa).meta),
        Some(SpecialDtype::Int48)
    );
    let bias = graph.node(arg(&graph, tosa, 2));
    assert!(bias.name.ends_with("_bias"));
    assert_eq!(special_dtype(&bias.meta), Some(SpecialDtype::Int48));
    assert_eq!(data(&graph, bias.id).dtype, DType::Int32);
    let rescale = graph.node(single_call(&graph, RESCALE));
    assert!(matches!(&rescale.args[1], IrArg::Dtype(dtype) if dtype == "torch.int16"));
}

/// RCP:1260-1266: an A16W8 conv nothing reads drops its INT16 path, leaving
/// the TOSA op alone.
#[test]
fn an_unused_a16w8_conv_drops_its_int16_path() {
    let mut graph = imported(REWRITE_CONV_A16W8, "conv2d_a16w8");
    let conv = single_call(&graph, CONVOLUTION);
    let input = arg(&graph, conv, 0);
    let output = graph.output_node().expect("output").id;
    graph.set_args(output, vec![IrArg::List(vec![IrArg::Node(input)])]);
    rewrite(&mut graph);
    let tosa = single_call(&graph, CONV2D);
    assert!(!graph.has_live_users(tosa));
    assert!(calls(&graph, RESCALE).is_empty());
    assert_eq!(calls(&graph, PERMUTE).len(), 1, "only the input permute");
}

// ---------------------------------------------------------------------------
// Convs left alone, regions and stock fallbacks
// ---------------------------------------------------------------------------

fn constant_tensor(name: &str, data: TensorData) -> NodeTensor {
    NodeTensor {
        source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
            input_kind: LoweringInputKind::Buffer,
            target_name: name.to_string(),
        }),
        data,
    }
}

fn zeros(shape: &[i64], dtype: DType) -> TensorData {
    let count: i64 = shape.iter().product();
    TensorData {
        data: vec![0; count as usize * dtype.storage_size_bytes()],
        shape: shape.to_vec(),
        dtype,
    }
}

/// Design §2.2: rc4 folds a conv whose inputs are all constants before
/// RewriteConvPass (`arm_pass_manager.py:700`); the port leaves one for
/// RETAB's constant folding, or for the serializer's aten-conv guard to hand
/// to stock.
#[test]
fn constant_convs_are_left_alone() {
    let mut graph = imported(REWRITE_CONV, "conv2d_pt");
    let conv = single_call(&graph, CONVOLUTION);
    let input = arg(&graph, conv, 0);
    let input_shape = shape(&graph, input);
    graph.node_mut(input).tensor = Some(constant_tensor("x", zeros(&input_shape, DType::Int8)));
    let before = replay::snapshot(&graph).expect("a valid graph");
    assert_eq!(run_with(&parent_pass(), &mut graph), (false, None));
    assert_eq!(replay::snapshot(&graph).expect("a valid graph"), before);
}

/// Turn constant placeholder `id` into a region constant: a `get_attr` node
/// with graph-module data.
fn make_region_constant(graph: &mut IrGraph, id: NodeId) {
    let node = graph.node_mut(id);
    node.op = NodeOp::GetAttr;
    node.tensor.as_mut().expect("a constant").source = TensorSource::GraphModuleAttr;
}

/// Every region conv falls back until S8 (see `control_flow_hazard`), even one
/// whose weight and bias are region constants (`get_attr` nodes, design F4),
/// and leaves the region as it was.
#[test]
fn region_convs_fall_back_until_s8() {
    for name in ["conv2d_pt", "conv2d_nobias_pt", "cumsum_pt"] {
        let mut graph = imported(REWRITE_CONV, name);
        let conv = single_call(&graph, CONVOLUTION);
        for constant in graph.node(conv).args[1..3]
            .iter()
            .filter_map(IrArg::as_node)
            .collect::<Vec<_>>()
        {
            make_region_constant(&mut graph, constant);
        }
        let before = replay::snapshot(&graph).expect("a valid graph");
        let (modified, reason) = run_with(&region_pass(), &mut graph);
        assert!(!modified, "{name}");
        let reason = reason.expect("a fallback");
        assert!(reason.contains("APU-135 S8"), "{name}: {reason}");
        assert!(reason.contains("boundary contracts"), "{name}: {reason}");
        assert_eq!(replay::snapshot(&graph).expect("a valid graph"), before);
    }
}

/// Why region convs fall back: the output contract's qparam lookup
/// (`extract_effective_qparam_ir`) walks the rewrite's qparam-less output
/// permute to its RESCALE and reverse engineers it as a rescale of the TOSA
/// conv's inherited output qparam. With a per-tensor weight that gives
/// `s_out^2 / (s_in * s_w)` instead of the conv's `s_out`, so a cond branch
/// ending in such a conv would get a wrong output RESCALE. When S8 fixes the
/// lookup this test fails: lift the region fallback then.
#[test]
fn boundary_contracts_misread_the_rewrites_rescale() {
    let mut graph = imported(REWRITE_CONV, "conv2d_pt");
    let conv = single_call(&graph, CONVOLUTION);
    let meta = graph.node(conv).meta.clone();
    rewrite(&mut graph);
    let output = graph.output_node().expect("output");
    let Some(IrArg::List(results)) = output.args.first() else {
        panic!("output list");
    };
    let permute = results[0].as_node().expect("a node");
    let read = crate::control_flow::extract_effective_qparam_ir(&graph, permute).expect("a qparam");
    let input = meta.input_qparam_at(0).unwrap().scale;
    let weight = meta.input_qparam_at(1).unwrap().scale;
    let output_qparam = meta.output_qparam().unwrap();
    let multiplier = (input * weight) / output_qparam.scale;
    assert_eq!(read.scale, output_qparam.scale / multiplier);
    assert_ne!(read.scale, output_qparam.scale);
}

/// A parent conv reading a cond output falls back until S8: the cond's output
/// contract takes its target qparam from the getitem's direct consumers
/// (`infer_output_qparams_map_ir`), and the rewrite's input permute has none.
/// A getitem carrying its own output qparams is unaffected.
#[test]
fn convs_reading_control_flow_outputs_fall_back_until_s8() {
    for getitem_qparams in [false, true] {
        let mut graph = imported(REWRITE_CONV, "conv2d_pt");
        let conv = single_call(&graph, CONVOLUTION);
        let input = arg(&graph, conv, 0);
        let input_shape = shape(&graph, input);
        let cond = crate::test_ir::call_with_meta(
            &mut graph,
            "cond",
            "torch.ops.higher_order.cond",
            vec![IrArg::Node(input)],
            crate::fx_meta::NodeMeta::default(),
        );
        let mut meta = crate::fx_meta::NodeMeta::tensor(input_shape, "torch.int8");
        if getitem_qparams {
            meta.set_single_output_qparam(int8(0.1, 0));
        }
        let getitem = crate::test_ir::call_with_meta(
            &mut graph,
            "getitem",
            "<built-in function getitem>",
            vec![IrArg::Node(cond), IrArg::Int(0)],
            meta,
        );
        assert!(graph.node(cond).is_control_flow() && graph.node(getitem).is_getitem());
        graph.set_arg(conv, 0, IrArg::Node(getitem));
        let (modified, reason) = run_with(&parent_pass(), &mut graph);
        if getitem_qparams {
            assert_eq!((modified, reason), (true, None));
        } else {
            assert!(!modified);
            let reason = reason.expect("a fallback");
            assert!(reason.contains("control-flow output getitem"), "{reason}");
        }
    }
}

/// One stock fallback's trigger: an edit of rc4's graph `case` of `fixture`,
/// and a fragment of the reason the pass gives.
struct Fallback {
    label: &'static str,
    fixture: &'static str,
    case: &'static str,
    region: bool,
    edit: fn(&mut IrGraph, NodeId),
    reason: &'static str,
}

/// Each case rc4 raises for (design §2.11) or the port cannot rewrite
/// faithfully (critique B5, B10; design §2.2, §2.8, §2.9) requests a stock
/// fallback and leaves the graph as it was.
#[test]
fn rc4s_errors_and_unported_cases_fall_back_to_stock() {
    let fallbacks = [
        Fallback {
            label: "transposed conv1d (RCP:940-943)",
            fixture: REWRITE_CONV,
            case: "conv1d_pt",
            region: false,
            edit: |graph, conv| graph.set_arg(conv, 6, IrArg::Bool(true)),
            reason: "Only 2D transpose convolutions",
        },
        Fallback {
            label: "dilated transposed conv (RCP:944-947)",
            fixture: REWRITE_CONV,
            case: "tconv2d_pt",
            region: false,
            edit: |graph, conv| graph.set_arg(conv, 5, crate::fx_ir::i64_list_arg(&[2, 2])),
            reason: "with dilation are not supported",
        },
        Fallback {
            label: "grouped CONV3D (RCP:147-158)",
            fixture: REWRITE_CONV,
            case: "conv3d_pt",
            region: false,
            edit: |graph, conv| graph.set_arg(conv, 8, IrArg::Int(3)),
            reason: "grouped (groups=3)",
        },
        Fallback {
            label: "grouped conv IrDecomposeGroupedConv declined (B5)",
            fixture: REWRITE_CONV,
            case: "dw2d_pt",
            region: false,
            edit: |graph, conv| graph.set_arg(conv, 8, IrArg::Int(2)),
            reason: "rc4 decomposes before RewriteConvPass",
        },
        Fallback {
            label: "grouped transposed conv (B5)",
            fixture: REWRITE_CONV,
            case: "tconv2d_pt",
            region: false,
            edit: |graph, conv| graph.set_arg(conv, 8, IrArg::Int(3)),
            reason: "grouped (groups=3) transposed",
        },
        Fallback {
            label: "stride remainder above the pad (RCP:126-129)",
            fixture: REWRITE_CONV,
            case: "conv2d_s2p0_pt",
            region: false,
            edit: |graph, conv| {
                let unsliced = named(graph, "quantized_decomposed_quantize_per_tensor_default");
                graph.set_arg(conv, 0, IrArg::Node(unsliced));
            },
            reason: "SizeAdjustInputPass",
        },
        Fallback {
            label: "runtime weight (RCP:243-251)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let weight = arg(graph, conv, 1);
                graph.node_mut(weight).tensor = None;
            },
            reason: "is not a constant placeholder",
        },
        Fallback {
            label: "get_attr weight in the parent (get_constant_placeholder_kind)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let weight = arg(graph, conv, 1);
                make_region_constant(graph, weight);
            },
            reason: "is not a constant placeholder",
        },
        Fallback {
            label: "int8 bias of a quantized conv (conv2d.py:46-50)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let bias = arg(graph, conv, 2);
                graph.node_mut(bias).tensor = Some(constant_tensor("b", zeros(&[4], DType::Int8)));
                graph
                    .node_mut(bias)
                    .meta
                    .set_tensor_facts(TensorFacts::new(vec![4], "torch.int8"));
            },
            reason: "torch.int8 bias for a torch.int32 accumulator",
        },
        Fallback {
            label: "float weight of a quantized conv (conv2d.py:42-46, B10)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let weight = arg(graph, conv, 1);
                graph.node_mut(weight).tensor =
                    Some(constant_tensor("w", zeros(&[4, 3, 3, 3], DType::Float32)));
                graph
                    .node_mut(weight)
                    .meta
                    .set_tensor_facts(TensorFacts::new(vec![4, 3, 3, 3], "torch.float32"));
            },
            reason: "torch.float32 weight",
        },
        Fallback {
            label: "no output qparams (RCP:434-438)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| graph.node_mut(conv).meta.clear_output_qparams(),
            reason: "No output quantization parameter",
        },
        Fallback {
            label: "no weight qparam (RCP:471-474, B10)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let kept: Vec<(usize, QParam)> = graph
                    .node(conv)
                    .meta
                    .sorted_input_qparams()
                    .into_iter()
                    .filter(|(index, _)| *index != 1)
                    .collect();
                graph.node_mut(conv).meta.set_input_qparams_from(kept);
            },
            reason: "no weight qparam",
        },
        Fallback {
            label: "per-channel activation qparam (get_scale_per_tensor)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                let mut activation = graph.node(conv).meta.input_qparam_at(0).cloned().unwrap();
                activation.per_channel_scales = vec![activation.scale; 3];
                activation.per_channel_zps = vec![activation.zp; 3];
                graph.node_mut(conv).meta.set_input_qparam_at(0, activation);
            },
            reason: "activation qparam is per-channel",
        },
        Fallback {
            label: "a float conv reading an int8 input (conv2d.py:17-88)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                graph.node_mut(conv).meta.clear_input_qparams();
                graph.node_mut(conv).meta.clear_output_qparams();
            },
            reason: "float conv reading a torch.int8 input",
        },
        Fallback {
            label: "a raw conv2d form (design §2.2)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| graph.node_mut(conv).target = "aten.conv2d.default".to_string(),
            reason: "aten.conv2d.default reaches RewriteConvPass's slot",
        },
        Fallback {
            label: "an output shape the rewrite cannot give (B10)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: false,
            edit: |graph, conv| {
                graph
                    .node_mut(conv)
                    .meta
                    .set_tensor_facts(TensorFacts::new(vec![1, 4, 8, 9], "torch.float32"));
            },
            reason: "the rewrite would give [1, 4, 8, 8]",
        },
        Fallback {
            label: "a region weight captured from the parent (S8)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: true,
            edit: |graph, conv| {
                let bias = arg(graph, conv, 2);
                make_region_constant(graph, bias);
            },
            reason: "captured from the parent graph",
        },
        Fallback {
            label: "a region bias captured from the parent (S8)",
            fixture: REWRITE_CONV,
            case: "conv2d_pt",
            region: true,
            edit: |graph, conv| {
                let weight = arg(graph, conv, 1);
                make_region_constant(graph, weight);
            },
            reason: "region conv biases need the explicit captured_conv",
        },
        Fallback {
            label: "A16W8 INT32 consumers (S4)",
            fixture: REWRITE_CONV_A16W8,
            case: "conv_add_a16w8",
            region: false,
            edit: |_, _| {},
            reason: "APU-135 S4",
        },
        Fallback {
            label: "A16W8 INT32 consumers behind a permute (S4)",
            fixture: REWRITE_CONV_A16W8,
            case: "permuted_int32_a16w8",
            region: false,
            edit: |_, _| {},
            reason: "APU-135 S4",
        },
    ];
    for fallback in fallbacks {
        let label = fallback.label;
        let mut graph = imported(fallback.fixture, fallback.case);
        let convs = calls(&graph, CONVOLUTION);
        (fallback.edit)(&mut graph, convs[0]);
        let before = replay::snapshot(&graph).expect("a valid graph");
        let pass = if fallback.region {
            region_pass()
        } else {
            parent_pass()
        };
        let (modified, reason) = run_with(&pass, &mut graph);
        let reason = reason.unwrap_or_else(|| panic!("{label}: no fallback"));
        assert!(reason.starts_with("IrRewriteConv: "), "{label}: {reason}");
        assert!(reason.contains(fallback.reason), "{label}: {reason}");
        assert!(!modified, "{label}");
        assert_eq!(
            replay::snapshot(&graph).expect("a valid graph"),
            before,
            "{label}"
        );
    }
}

/// rc4 raises at the first conv it cannot rewrite; the port requests the
/// fallback there and stops, keeping the convs before it rewritten and the
/// graph valid for the passes that still run before the request is honoured.
#[test]
fn a_fallback_stops_after_the_convs_before_it() {
    // A16W8MixedConsumerChain: expand and depthwise feed clamps, the
    // projection and the following conv feed INT32 RESCALEs.
    let mut graph = imported(REWRITE_CONV_A16W8, "mixed_chain_conv_a16w8");
    let convs = calls(&graph, CONVOLUTION);
    assert_eq!(convs.len(), 4);
    let names: Vec<String> = convs
        .iter()
        .map(|&conv| graph.node(conv).name.clone())
        .collect();
    let (modified, reason) = run_with(&parent_pass(), &mut graph);
    assert!(modified);
    let reason = reason.expect("a fallback");
    assert!(
        reason.contains(&format!("IrRewriteConv: {}: ", names[2])),
        "{reason}"
    );
    let left: Vec<NodeId> = names[2..].iter().map(|name| named(&graph, name)).collect();
    assert_eq!(calls(&graph, CONVOLUTION), left);
    assert_eq!(calls(&graph, CONV2D).len(), 1);
    assert_eq!(calls(&graph, DEPTHWISE).len(), 1);
    replay::snapshot(&graph).expect("a valid graph");
}

/// A graph with several fallbacks reports the first conv's reason only.
#[test]
fn the_first_fallback_reason_wins() {
    let mut graph = imported(REWRITE_CONV, "two_convs_no_bias_pt");
    let convs = calls(&graph, CONVOLUTION);
    for &conv in &convs {
        graph.node_mut(conv).meta.clear_output_qparams();
    }
    let first = graph.node(convs[0]).name.clone();
    let (modified, reason) = run_with(&parent_pass(), &mut graph);
    assert!(!modified);
    assert!(reason
        .expect("a fallback")
        .contains(&format!("IrRewriteConv: {first}: ")));
    assert_eq!(calls(&graph, CONVOLUTION).len(), 2);
}

/// The weight's kind and position follow the source (RCP:365-376): a
/// parameter source gives a parameter weight right after it.
#[test]
fn weights_take_their_sources_kind_and_follow_it() {
    let graph = rewritten(REWRITE_CONV, "cumsum_pt");
    let tosa = single_call(&graph, DEPTHWISE);
    let weight = arg(&graph, tosa, 1);
    let source = named(&graph, "aten_cumsum_default_kernel");
    assert!(is_parameter(graph.node(source)));
    assert!(is_parameter(graph.node(weight)));
    assert_eq!(position(&graph, weight), position(&graph, source) + 1);
    // The source stays, unused, as rc4 leaves it; ingest drops it.
    assert!(!graph.has_live_users(source));
}

/// Created names never collide with a taken one, an erased node's included
/// (design F9): a second run's names get `_N` suffixes rather than reusing.
#[test]
fn created_names_are_unique() {
    let mut graph = imported(REWRITE_CONV, "conv2d_pt");
    let conv = single_call(&graph, CONVOLUTION);
    let conv_name = graph.node(conv).name.clone();
    // Occupy the names the rewrite would take.
    for suffix in ["_tosa", "_rescale", "_nhwc", "_nchw", "_weight_ohwi"] {
        let meta = crate::fx_meta::NodeMeta::tensor(vec![1], "torch.int8");
        graph.add_node(
            NodeOp::Placeholder,
            format!("{conv_name}{suffix}"),
            format!("{conv_name}{suffix}"),
            Vec::new(),
            meta,
        );
    }
    rewrite(&mut graph);
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for node in graph.live_nodes() {
        *counts.entry(node.name.as_str()).or_default() += 1;
    }
    assert!(counts.values().all(|&count| count == 1));
    let tosa = single_call(&graph, CONV2D);
    assert_eq!(graph.node(tosa).name, format!("{conv_name}_tosa_1"));
}

// ---------------------------------------------------------------------------
// After RETAB's own passes
// ---------------------------------------------------------------------------

/// A per-tensor int8 qparam.
fn int8(scale: f64, zp: i64) -> QParam {
    QParam::per_tensor(scale, zp, -128, 127, "torch.int8".to_string())
}

/// A per-channel qparam on axis 0.
fn per_channel(scales: &[f64], dtype: &str, (qmin, qmax): (i64, i64)) -> QParam {
    QParam {
        scale: scales[0],
        zp: 0,
        qmin,
        qmax,
        dtype_str: dtype.to_string(),
        per_channel_scales: scales.to_vec(),
        per_channel_zps: vec![0; scales.len()],
        axis: 0,
    }
}

/// Distinct int8 bytes for a constant of `shape`.
fn ramp(shape: &[i64], dtype: DType) -> TensorData {
    let count: i64 = shape.iter().product();
    let size = dtype.storage_size_bytes();
    let data = (0..count as usize * size)
        .map(|index| (index * 37 % 251) as u8)
        .collect();
    TensorData {
        data,
        shape: shape.to_vec(),
        dtype,
    }
}

/// A constant placeholder, as RETAB imports a frozen parameter.
fn constant(graph: &mut IrGraph, name: &str, data: TensorData) -> NodeId {
    let shape = data.shape.clone();
    let dtype = data.dtype.torch_str();
    let id = crate::test_ir::placeholder(graph, name, shape, dtype);
    graph.node_mut(id).tensor = Some(constant_tensor(name, data));
    id
}

/// Run the passes between RETAB's IR import and IrRewriteConv that `graph`
/// needs, then IrRewriteConv; the weights reach it as constants.
fn rewrite_after(graph: &mut IrGraph, upstream: &dyn IrPass) {
    crate::stock_fallback::take();
    assert!(upstream.run(graph), "{}", upstream.name());
    assert!(crate::layout::constants::IrMaterializeConstantsPass { in_region: false }.run(graph));
    assert_eq!(crate::stock_fallback::take(), None);
    rewrite(graph);
}

/// A grouped conv: IrDecomposeGroupedConv slices it per group, with the
/// group's per-channel qparams; IrMaterializeConstants folds the weight and
/// bias slices to constants; each piece then rewrites to a CONV2D whose OHWI
/// weight is its slice of the source, permuted, and whose RESCALE takes the
/// group's scales.
#[test]
fn grouped_convs_after_retabs_decomposition() {
    let mut graph = IrGraph::empty();
    let x = crate::test_ir::placeholder(&mut graph, "x", [1, 4, 8, 8], "torch.int8");
    let weight_data = ramp(&[6, 2, 3, 3], DType::Int8);
    let weight = constant(&mut graph, "w", weight_data.clone());
    let bias = constant(&mut graph, "b", ramp(&[6], DType::Int32));
    let weight_scales = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06];
    let mut meta = crate::fx_meta::NodeMeta::tensor([1, 6, 8, 8], "torch.float32");
    meta.set_input_qparams_from([
        (0, int8(0.5, 3)),
        (1, per_channel(&weight_scales, "torch.int8", (-127, 127))),
        (
            2,
            per_channel(
                &weight_scales.map(|scale| scale * 0.5),
                "torch.int32",
                (-2147483647, 2147483647),
            ),
        ),
    ]);
    meta.set_single_output_qparam(int8(0.25, -2));
    let conv = crate::test_ir::call_with_meta(
        &mut graph,
        "conv",
        CONVOLUTION,
        vec![
            IrArg::Node(x),
            IrArg::Node(weight),
            IrArg::Node(bias),
            crate::fx_ir::i64_list_arg(&[1, 1]),
            crate::fx_ir::i64_list_arg(&[1, 1]),
            crate::fx_ir::i64_list_arg(&[1, 1]),
            IrArg::Bool(false),
            crate::fx_ir::i64_list_arg(&[0, 0]),
            IrArg::Int(2),
        ],
        meta,
    );
    crate::test_ir::output(&mut graph, conv);
    rewrite_after(
        &mut graph,
        &crate::fx_decompose_grouped_conv_ir::IrDecomposeGroupedConvPass {
            layout: crate::pipeline_config::LayoutMode::Explicit,
        },
    );

    let convs = calls(&graph, CONV2D);
    assert_eq!(convs.len(), 2);
    assert!(calls(&graph, CONVOLUTION).is_empty());
    for (group, &tosa) in convs.iter().enumerate() {
        let piece = TensorData {
            data: weight_data.data[group * 54..(group + 1) * 54].to_vec(),
            shape: vec![3, 2, 3, 3],
            dtype: DType::Int8,
        };
        let rewritten_weight = data(&graph, arg(&graph, tosa, 1));
        assert_eq!(rewritten_weight.shape, [3, 3, 3, 2]);
        assert_eq!(
            rewritten_weight.data,
            reference_permute(&piece, &[0, 2, 3, 1])
        );
        let bias = data(&graph, arg(&graph, tosa, 2));
        assert_eq!(bias.shape, [3]);
        let [rescale] = graph.live_user_ids(tosa)[..] else {
            panic!("one RESCALE per group");
        };
        let expected: Vec<f64> = weight_scales[group * 3..group * 3 + 3]
            .iter()
            .map(|&scale| (0.5 * scale) / 0.25)
            .collect();
        assert_eq!(float_list(&graph.node(rescale).args[2]), expected);
        assert_eq!(graph.node(rescale).args[4].as_int(), Some(-2));
    }
}

/// Linear: IrDecomposeLinear's 1x1 conv reads `view_copy(weight)`, which
/// IrMaterializeConstants folds; the rewrite then gives rc4's OHWI
/// `[Co, 1, 1, Ci]` weight, and Linear(1, Co) the depthwise HWCM one.
#[test]
fn linears_after_retabs_decomposition() {
    for (inputs, outputs, target, weight_shape) in [
        (8, 4, CONV2D, [4, 1, 1, 8]),
        (1, 4, DEPTHWISE, [1, 1, 1, 4]),
    ] {
        let mut graph = IrGraph::empty();
        let x = crate::test_ir::placeholder(&mut graph, "x", [2, inputs], "torch.int8");
        let weight_data = ramp(&[outputs, inputs], DType::Int8);
        let weight = constant(&mut graph, "w", weight_data.clone());
        let bias = constant(&mut graph, "b", ramp(&[outputs], DType::Int32));
        let mut meta = crate::fx_meta::NodeMeta::tensor([2, outputs], "torch.float32");
        meta.set_input_qparams_from([
            (0, int8(0.5, 3)),
            (
                1,
                QParam::per_tensor(0.02, 0, -127, 127, "torch.int8".into()),
            ),
            (
                2,
                QParam::per_tensor(0.01, 0, -2147483647, 2147483647, "torch.int32".into()),
            ),
        ]);
        meta.set_single_output_qparam(int8(0.25, -2));
        let linear = crate::test_ir::call_with_meta(
            &mut graph,
            "linear",
            "aten.linear.default",
            vec![IrArg::Node(x), IrArg::Node(weight), IrArg::Node(bias)],
            meta,
        );
        crate::test_ir::output(&mut graph, linear);
        rewrite_after(
            &mut graph,
            &crate::fx_decompose_linear_ir::IrDecomposeLinearPass,
        );

        let tosa = single_call(&graph, target);
        let rewritten_weight = data(&graph, arg(&graph, tosa, 1));
        assert_eq!(
            rewritten_weight.shape, weight_shape,
            "Linear({inputs}, {outputs})"
        );
        // [Co, Ci] -> [Co, Ci, 1, 1] -> OHWI or HWCM keeps the element order.
        assert_eq!(rewritten_weight.data, weight_data.data);
        assert_eq!(arg(&graph, tosa, 2), named(&graph, "b"));
    }
}

/// cumsum: IrDecomposeCumsum's `aten.full` kernel becomes a constant at
/// IrMaterializeConstants (ComputeConstantOpsAOT's share, quantized with its
/// own int8 qparam), and the conv rewrites to rc4's Cin = 1 depthwise form
/// with a zero bias.
#[test]
fn cumsum_after_retabs_decomposition() {
    let mut graph = IrGraph::empty();
    let x = crate::test_ir::placeholder(&mut graph, "x", [1, 4, 5], "torch.int8");
    let mut meta = crate::fx_meta::NodeMeta::tensor([1, 4, 5], "torch.float32");
    meta.set_single_input_qparam(int8(0.5, 3));
    meta.set_single_output_qparam(int8(0.75, -1));
    let cumsum = crate::test_ir::call_with_meta(
        &mut graph,
        "cumsum",
        "aten.cumsum.default",
        vec![IrArg::Node(x), IrArg::Int(1)],
        meta,
    );
    crate::test_ir::output(&mut graph, cumsum);
    rewrite_after(
        &mut graph,
        &crate::fx_decompose_cumsum_ir::IrDecomposeCumsumPass,
    );

    let tosa = single_call(&graph, DEPTHWISE);
    let weight = data(&graph, arg(&graph, tosa, 1));
    assert_eq!(
        (weight.shape.as_slice(), weight.dtype),
        (&[4, 1, 1, 1][..], DType::Int8)
    );
    assert_eq!(weight.data, [1, 1, 1, 1]);
    assert_eq!(ints(&graph, tosa, 4), [3, 3, 0, 0]);
    let bias = data(&graph, arg(&graph, tosa, 2));
    assert_eq!(
        (bias.shape.as_slice(), bias.dtype),
        (&[1][..], DType::Int32)
    );
    let rescale = graph.node(single_call(&graph, RESCALE));
    assert_eq!(float_list(&rescale.args[2]), [(0.5 * 1.0) / 0.75]);
}
