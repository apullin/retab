//! End-to-end checks of `IrRewriteConvPass` (APU-135 P2 S3, S4) through explicit-mode
//! ingest and serialization: rc4's captured graphs (`rewrite_conv*.txt`), rewritten
//! by the pass, serialize to the TOSA rc4's form implies. Each TOSA conv reads its
//! pre-permuted weight as given (no second permutation, design B1) and declares the
//! val's shape with the accumulator's dtype (G1, A9); the RESCALE reads the
//! accumulator; the only layout ops are the rewrite's TRANSPOSEs (and conv1d's
//! RESHAPEs), with no peephole folding them away (G5); and no node carries a dim
//! order or serializes at a shape other than its val's (D1). A16W8 convs whose
//! INT48 accumulators branch to int32 consumers, U55 clones included, serialize too.

use super::collection::{check_explicit_layout_invariants, collect_regions, SerializedGraphItems};
use super::tosa_types::{Attribute, DType, Op, TosaTensor};
use super::visitors;
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeOp};
use crate::graph::Graph;
use crate::layout::replay;
use crate::layout::rewrite_conv::IrRewriteConvPass;
use crate::pipeline_config::PipelineConfig;
use crate::store::TensorStore;

const REWRITE_CONV: &str = include_str!("../../tests/fixtures/layout_replay/rewrite_conv.txt");
const REWRITE_CONV_A16W8: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8.txt");
const REWRITE_CONV_A16W8_U55: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8_u55.txt");

const TOSA_CONVS: [&str; 4] = [
    "backend.tosa.CONV2D.default",
    "backend.tosa.DEPTHWISE_CONV2D.default",
    "backend.tosa.TRANSPOSE_CONV2D.default",
    "backend.tosa.CONV3D.default",
];

/// rc4's before graph of `name`, rewritten by the pass.
fn rewritten(fixture: &str, name: &str) -> IrGraph {
    let pass = IrRewriteConvPass {
        is_u55_subset: false,
        in_region: false,
    };
    rewritten_with(&pass, fixture, name)
}

/// rc4's before graph of `name`, rewritten by `pass`.
fn rewritten_with(pass: &IrRewriteConvPass, fixture: &str, name: &str) -> IrGraph {
    let case = replay::parse(fixture)
        .expect("a valid fixture")
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    let mut graph = replay::import(&case);
    crate::stock_fallback::take();
    assert!(pass.run(&mut graph), "{name}");
    assert_eq!(crate::stock_fallback::take(), None, "{name}");
    graph
}

/// Ingest and serialize `ir` as `ingest_and_serialize` does in explicit mode.
fn serialize_explicit(ir: &IrGraph) -> Result<SerializedGraphItems, String> {
    let mut store = TensorStore::new();
    let mut graph: Graph = crate::ingest::from_ir::graph_from_ir(ir, &mut store);
    crate::ingest::finalize::finalize_converted_graph(&mut graph);
    graph.rebuild_user_lists();
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    // D1: no dim order, and every node serializes at its val's shape.
    check_explicit_layout_invariants("main", &graph, &store)?;
    let mut regions = collect_regions(&[("main".to_string(), graph)], &store)?;
    Ok(regions.remove(0).items)
}

fn tensor<'a>(items: &'a SerializedGraphItems, name: &str) -> &'a TosaTensor {
    items
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor named {name}"))
}

fn count(items: &SerializedGraphItems, op: Op) -> usize {
    items.operators.iter().filter(|o| o.op == op).count()
}

fn i32s(shape: &[i64]) -> Vec<i32> {
    shape.iter().map(|&dim| dim as i32).collect()
}

/// Every kind of rewritten conv serializes: weights as rewritten, accumulators at
/// their vals' shapes and dtypes, one RESCALE per conv reading its accumulator, and
/// exactly the rewrite's layout ops. A8W4 weights are INT4, A16W8 accumulators INT48.
#[test]
fn rewritten_convs_serialize_in_explicit_mode() {
    for (fixture, name, op, accumulator) in [
        (REWRITE_CONV, "conv2d_pc", Op::Conv2d, DType::Int32),
        (REWRITE_CONV, "conv2d_nobias_pt", Op::Conv2d, DType::Int32),
        (REWRITE_CONV, "conv2d_s2p0_pc", Op::Conv2d, DType::Int32),
        (REWRITE_CONV, "dw2d_pc", Op::DepthwiseConv2d, DType::Int32),
        (
            REWRITE_CONV,
            "dw2d_kx1_pc",
            Op::DepthwiseConv2d,
            DType::Int32,
        ),
        (
            REWRITE_CONV,
            "tconv2d_pc",
            Op::TransposeConv2d,
            DType::Int32,
        ),
        (REWRITE_CONV, "conv1d_pc", Op::Conv2d, DType::Int32),
        (
            REWRITE_CONV,
            "conv1d_dw_pt",
            Op::DepthwiseConv2d,
            DType::Int32,
        ),
        (REWRITE_CONV, "conv3d_pc", Op::Conv3d, DType::Int32),
        (REWRITE_CONV, "linear_pt", Op::Conv2d, DType::Int32),
        (
            REWRITE_CONV,
            "linear_in1_pt",
            Op::DepthwiseConv2d,
            DType::Int32,
        ),
        (
            REWRITE_CONV,
            "conv2d_cin1_pc",
            Op::DepthwiseConv2d,
            DType::Int32,
        ),
        (REWRITE_CONV, "conv3d_cin1_pt", Op::Conv3d, DType::Int32),
        (
            REWRITE_CONV,
            "two_convs_no_bias_pt",
            Op::Conv2d,
            DType::Int32,
        ),
        (REWRITE_CONV, "cumsum_pt", Op::DepthwiseConv2d, DType::Int32),
        (REWRITE_CONV, "conv2d_a8w4", Op::Conv2d, DType::Int32),
        (
            REWRITE_CONV,
            "tconv2d_nobias_a8w4",
            Op::TransposeConv2d,
            DType::Int32,
        ),
        (REWRITE_CONV_A16W8, "conv2d_a16w8", Op::Conv2d, DType::Int48),
        (
            REWRITE_CONV_A16W8,
            "tconv2d_a16w8",
            Op::TransposeConv2d,
            DType::Int48,
        ),
    ] {
        let mut ir = rewritten(fixture, name);
        if name == "cumsum_pt" {
            // The replay import types cumsum's slice float32, after the float32
            // conv it read (`replay::retab_dtypes`); RETAB's IrDecomposeCumsum
            // gives it the cumsum input's dtype, int8 here. Finalize settles a
            // float32 view or permute from its producer, but not a slice.
            let slice = ir
                .live_nodes()
                .find(|node| node.target == "aten.slice_copy.Tensor")
                .expect("cumsum's slice")
                .id;
            ir.node_mut(slice)
                .meta
                .set_val_dtype_for_existing_shape("torch.int8");
        }
        let items = serialize_explicit(&ir).unwrap_or_else(|error| panic!("{name}: {error}"));
        let convs: Vec<_> = ir
            .topological_live_node_ids()
            .into_iter()
            .filter(|&id| {
                let node = ir.node(id);
                node.op == NodeOp::CallFunction && TOSA_CONVS.contains(&node.target.as_str())
            })
            .collect();
        assert_eq!(count(&items, op), convs.len(), "{name}");
        assert_eq!(count(&items, Op::Rescale), convs.len(), "{name}");
        for &conv in &convs {
            let node = ir.node(conv);
            let operator = items
                .operators
                .iter()
                .find(|o| o.op == op && o.outputs.first() == Some(&node.name))
                .unwrap_or_else(|| panic!("{name}: no TOSA op for {}", node.name));
            // B1: the weight serializes as IrRewriteConv laid it out.
            let weight = ir.node(node.arg_node(1).expect("a weight"));
            let weight_tensor = tensor(&items, &operator.inputs[1]);
            assert_eq!(
                weight_tensor.shape,
                i32s(weight.meta.val_shape().expect("a weight value")),
                "{name}"
            );
            let int4 = name.ends_with("_a8w4");
            assert_eq!(
                weight_tensor.dtype,
                if int4 { DType::Int4 } else { DType::Int8 },
                "{name}"
            );
            if !int4 {
                let bytes = &weight.tensor.as_ref().expect("a constant").data.data;
                assert_eq!(weight_tensor.data.as_ref(), Some(bytes), "{name}");
            }
            // G1, A9: the val's shape and the accumulator's dtype.
            let output = tensor(&items, &operator.outputs[0]);
            assert_eq!(
                output.shape,
                i32s(node.meta.val_shape().expect("a value")),
                "{name}"
            );
            assert_eq!(output.dtype, accumulator, "{name}");
            let bias = tensor(&items, &operator.inputs[2]);
            assert_eq!(bias.dtype, accumulator, "{name}: the bias");
            let rescale = items
                .operators
                .iter()
                .find(|o| o.op == Op::Rescale && o.inputs.first() == operator.outputs.first())
                .unwrap_or_else(|| panic!("{name}: no RESCALE reads {}", node.name));
            let int16 = accumulator == DType::Int48;
            assert_eq!(
                tensor(&items, &rescale.outputs[0]).dtype,
                if int16 { DType::Int16 } else { DType::Int8 },
                "{name}"
            );
        }
        let rank3 = name.starts_with("conv1d");
        let transposes = ir
            .live_nodes()
            .filter(|node| node.target == "aten.permute_copy.default")
            .count();
        assert_eq!(count(&items, Op::Transpose), transposes, "{name}");
        if rank3 {
            assert_eq!(count(&items, Op::Reshape), 2, "{name}");
        }
    }
}

/// A16W8 convs with INT32 consumers (APU-135 P2 S4, design §2.8) serialize in
/// explicit mode, standing in for the whole lowering rc4's
/// `test_rewrite_conv_a16w8_mixed_consumers_restore_int16` runs first
/// (`_lower_a16w8_to_tosa`): every TOSA conv, U55 clones included, declares an
/// INT48 accumulator; each RESCALE of one reads it as INT48 (scale32 off) and
/// narrows to the int16 or int32 its IR node asks for; on U55 each accumulator
/// feeds exactly one operator; and the only TRANSPOSEs are the graph's
/// permutes.
#[test]
fn a16w8_branches_serialize_in_explicit_mode() {
    let generic =
        IrRewriteConvPass::new(&PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16"), false);
    let u55 = IrRewriteConvPass::new(
        &PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+int4+u55"),
        false,
    );
    let cases = [
        (REWRITE_CONV_A16W8, "conv_add_a16w8"),
        (REWRITE_CONV_A16W8, "conv_add_sub_a16w8"),
        (REWRITE_CONV_A16W8, "permuted_int32_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_chain_conv_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_chain_dw_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_chain_avg_pool_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_chain_max_pool_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_conv3d_a16w8"),
        (REWRITE_CONV_A16W8, "mixed_tconv_a16w8"),
        (REWRITE_CONV_A16W8_U55, "conv_add_sub_a16w8_u55"),
        (REWRITE_CONV_A16W8_U55, "permuted_int32_a16w8_u55"),
        (REWRITE_CONV_A16W8_U55, "mixed_chain_conv_a16w8_u55"),
        (REWRITE_CONV_A16W8_U55, "mixed_chain_avg_pool_a16w8_u55"),
        (REWRITE_CONV_A16W8_U55, "mixed_tconv_a16w8_u55"),
    ];
    for (fixture, name) in cases {
        let on_u55 = name.ends_with("_u55");
        let ir = rewritten_with(if on_u55 { &u55 } else { &generic }, fixture, name);
        let items = serialize_explicit(&ir).unwrap_or_else(|error| panic!("{name}: {error}"));
        let convs: Vec<_> = ir
            .topological_live_node_ids()
            .into_iter()
            .filter(|&id| {
                let node = ir.node(id);
                node.op == NodeOp::CallFunction && TOSA_CONVS.contains(&node.target.as_str())
            })
            .collect();
        let conv_ops = [
            Op::Conv2d,
            Op::DepthwiseConv2d,
            Op::TransposeConv2d,
            Op::Conv3d,
        ];
        let mut widened = 0;
        for &conv in &convs {
            let node = ir.node(conv);
            let operator = items
                .operators
                .iter()
                .find(|o| conv_ops.contains(&o.op) && o.outputs.first() == Some(&node.name))
                .unwrap_or_else(|| panic!("{name}: no TOSA op for {}", node.name));
            assert_eq!(
                tensor(&items, &operator.outputs[0]).dtype,
                DType::Int48,
                "{name}"
            );
            let readers: Vec<_> = items
                .operators
                .iter()
                .filter(|o| o.inputs.contains(&node.name))
                .collect();
            if on_u55 {
                assert_eq!(readers.len(), 1, "{name}: {} feeds one operator", node.name);
            }
            for user in ir.live_user_ids(conv) {
                let rescale = ir.node(user);
                assert_eq!(rescale.target, "backend.tosa.RESCALE.default", "{name}");
                let narrowed = match &rescale.args[1] {
                    IrArg::Dtype(dtype) if dtype == "torch.int16" => DType::Int16,
                    IrArg::Dtype(dtype) if dtype == "torch.int32" => DType::Int32,
                    other => panic!("{name}: a RESCALE to {other:?}"),
                };
                let operator = readers
                    .iter()
                    .find(|o| o.op == Op::Rescale && o.outputs.first() == Some(&rescale.name))
                    .unwrap_or_else(|| panic!("{name}: no RESCALE op for {}", rescale.name));
                assert_eq!(
                    tensor(&items, &operator.inputs[3]).dtype,
                    DType::Int48,
                    "{name}"
                );
                assert_eq!(
                    tensor(&items, &operator.outputs[0]).dtype,
                    narrowed,
                    "{name}"
                );
                let Attribute::Rescale(attribute) = &operator.attribute else {
                    panic!("{name}: expected a RESCALE attribute");
                };
                assert!(!attribute.scale32, "{name}");
                widened += usize::from(narrowed == DType::Int32);
            }
        }
        assert!(
            widened > 0,
            "{name}: no RESCALE to int32 reads an accumulator"
        );
        let transposes = ir
            .live_nodes()
            .filter(|node| node.target == "aten.permute_copy.default")
            .count();
        assert_eq!(count(&items, Op::Transpose), transposes, "{name}");
    }
}

/// An A8W4 transposed conv's RESCALE carries rc4's 3 input-channel scales for its
/// 4-channel accumulator (see `rescales_take_one_scale_per_weight_scale`) and
/// serializes as rc4's `_build_rescale` does (`op_tosa_rescale.py:134-150, 301`):
/// one multiplier and one shift per scale, `per_channel` because there are several.
/// That is the RESCALE TOSA's reference model rejects for stock too.
#[test]
fn a8w4_transposed_conv_rescales_serialize_one_multiplier_per_scale() {
    let ir = rewritten(REWRITE_CONV, "tconv2d_nobias_a8w4");
    let items = serialize_explicit(&ir).unwrap_or_else(|error| panic!("{error}"));
    let conv = items
        .operators
        .iter()
        .find(|o| o.op == Op::TransposeConv2d)
        .expect("a TRANSPOSE_CONV2D");
    let accumulator = tensor(&items, &conv.outputs[0]);
    assert_eq!(
        (accumulator.shape.as_slice(), accumulator.dtype),
        (&[1, 12, 12, 4][..], DType::Int32)
    );
    let rescale = items
        .operators
        .iter()
        .find(|o| o.op == Op::Rescale && o.inputs.first() == conv.outputs.first())
        .expect("a RESCALE reading the accumulator");
    assert_eq!(tensor(&items, &rescale.inputs[1]).shape, [3], "multipliers");
    assert_eq!(tensor(&items, &rescale.inputs[2]).shape, [3], "shifts");
    let Attribute::Rescale(attribute) = &rescale.attribute else {
        panic!("expected a RESCALE attribute");
    };
    assert!(attribute.per_channel);
    assert!(attribute.scale32);
    let output = tensor(&items, &rescale.outputs[0]);
    assert_eq!(
        (output.shape.as_slice(), output.dtype),
        (&[1, 12, 12, 4][..], DType::Int8)
    );
}
