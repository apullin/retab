//! End-to-end checks of `IrRewriteConvPass` (APU-135 P2 S3) through explicit-mode
//! ingest and serialization: rc4's captured graphs (`rewrite_conv*.txt`), rewritten
//! by the pass, serialize to the TOSA rc4's form implies. Each TOSA conv reads its
//! pre-permuted weight as given (no second permutation, design B1) and declares the
//! val's shape with the accumulator's dtype (G1, A9); the RESCALE reads the
//! accumulator; the only layout ops are the rewrite's TRANSPOSEs (and conv1d's
//! RESHAPEs), with no peephole folding them away (G5).

use super::collection::{collect_regions, SerializedGraphItems};
use super::tosa_types::{DType, Op, TosaTensor};
use super::visitors;
use crate::fx_ir::{IrGraph, IrPass, NodeOp};
use crate::graph::Graph;
use crate::layout::replay;
use crate::layout::rewrite_conv::IrRewriteConvPass;
use crate::pipeline_config::LayoutMode;
use crate::store::TensorStore;

const REWRITE_CONV: &str = include_str!("../../tests/fixtures/layout_replay/rewrite_conv.txt");
const REWRITE_CONV_A16W8: &str =
    include_str!("../../tests/fixtures/layout_replay/rewrite_conv_a16w8.txt");

const TOSA_CONVS: [&str; 4] = [
    "backend.tosa.CONV2D.default",
    "backend.tosa.DEPTHWISE_CONV2D.default",
    "backend.tosa.TRANSPOSE_CONV2D.default",
    "backend.tosa.CONV3D.default",
];

/// rc4's before graph of `name`, rewritten by the pass.
fn rewritten(fixture: &str, name: &str) -> IrGraph {
    let case = replay::parse(fixture)
        .expect("a valid fixture")
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    let mut graph = replay::import(&case);
    crate::stock_fallback::take();
    let pass = IrRewriteConvPass {
        is_u55_subset: false,
        in_region: false,
    };
    assert!(pass.run(&mut graph), "{name}");
    assert_eq!(crate::stock_fallback::take(), None, "{name}");
    graph
}

/// Ingest and serialize `ir` as `ingest_and_serialize` does in explicit mode.
fn serialize_explicit(ir: &IrGraph) -> Result<SerializedGraphItems, String> {
    let mut store = TensorStore::new();
    let mut graph: Graph = crate::ingest::from_ir::graph_from_ir(ir, &mut store);
    graph.layout = LayoutMode::Explicit;
    crate::ingest::finalize::finalize_converted_graph(&mut graph);
    graph.rebuild_user_lists();
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
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
