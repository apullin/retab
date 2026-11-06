//! The explicit-mode pool and upsample rewrites (APU-156, P2 design S7) end to
//! end: each test builds the IR RETAB's pipeline holds at the rewrite's slot,
//! runs the rewrite (and, for the avg pool's zero points, the
//! IrMaterializeConstants that follows it at rc4's ComputeConstantOpsAOT slot),
//! then ingests and serializes in explicit mode as `explicit_layout_tests`
//! does. S2's tests hand-build rc4-shaped graphs; these check that what the
//! passes actually leave serializes to rc4's TOSA.

use super::collection::{collect_regions, SerializedGraphItems};
use super::tosa_types::{Attribute, DType, Op, ResizeMode, TosaOperator, TosaTensor};
use super::visitors;
use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeMeta, QParam};
use crate::graph::ir::DType as IrDType;
use crate::graph::Graph;
use crate::layout::constants::IrMaterializeConstantsPass;
use crate::layout::rewrite_avg_pool2d::IrRewriteAvgPool2dPass;
use crate::layout::rewrite_max_pool2d::IrRewriteMaxPool2dPass;
use crate::layout::rewrite_upsample::IrRewriteUpsamplePass;
use crate::pipeline_config::LayoutMode;
use crate::store::TensorStore;
use crate::test_ir::{call, call_with_meta, output, placeholder};

const INT8: &str = "torch.int8";
const INT16: &str = "torch.int16";
const FLOAT32: &str = "torch.float32";

fn qp(scale: f64, zp: i64, dtype: &str) -> QParam {
    let (qmin, qmax) = IrDType::from_torch_str(dtype)
        .quantized_integer_bounds()
        .unwrap();
    QParam::per_tensor(scale, zp, qmin, qmax, dtype.to_string())
}

fn serialize_explicit(ir: &IrGraph) -> SerializedGraphItems {
    let mut store = TensorStore::new();
    let mut graph: Graph = crate::ingest::from_ir::graph_from_ir(ir, &mut store);
    graph.layout = LayoutMode::Explicit;
    crate::ingest::finalize::finalize_converted_graph(&mut graph);
    graph.rebuild_user_lists();
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    let mut regions = collect_regions(&[("main".to_string(), graph)], &store).unwrap();
    regions.remove(0).items
}

fn tensor<'a>(items: &'a SerializedGraphItems, name: &str) -> &'a TosaTensor {
    items
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor named {name}"))
}

fn ops_of(items: &SerializedGraphItems, op: Op) -> Vec<&TosaOperator> {
    items.operators.iter().filter(|o| o.op == op).collect()
}

fn single_op(items: &SerializedGraphItems, op: Op) -> &TosaOperator {
    let ops = ops_of(items, op);
    assert_eq!(ops.len(), 1, "expected one {op:?}");
    ops[0]
}

fn shape_values(items: &SerializedGraphItems, name: &str) -> Vec<i64> {
    let shape = items
        .shapes
        .iter()
        .find(|shape| shape.name == name)
        .unwrap_or_else(|| panic!("no shape named {name}"));
    let (values, rest) = shape.data.as_chunks::<8>();
    assert!(rest.is_empty());
    values.iter().copied().map(i64::from_le_bytes).collect()
}

/// Each TRANSPOSE's output is its input's shape under its permutation, and the
/// two layout TRANSPOSEs are rc4's NHWC pair.
fn assert_nhwc_transposes(items: &SerializedGraphItems) {
    let transposes = ops_of(items, Op::Transpose);
    assert_eq!(transposes.len(), 2);
    for (transpose, perms) in transposes.iter().zip([[0, 2, 3, 1], [0, 3, 1, 2]]) {
        let Attribute::Transpose(attribute) = &transpose.attribute else {
            panic!("expected a TRANSPOSE attribute");
        };
        assert_eq!(attribute.perms, perms);
        let input = &tensor(items, &transpose.inputs[0]).shape;
        let expected: Vec<i32> = perms.iter().map(|&dim| input[dim as usize]).collect();
        assert_eq!(tensor(items, &transpose.outputs[0]).shape, expected);
    }
}

/// `x -> <target>(x, args...) -> output`, the target holding RETAB's float32
/// val for its quantized output and `qparam` on both maps.
fn quantized_graph(
    input: &[i64],
    dtype: &str,
    target: &str,
    args: Vec<IrArg>,
    val: &[i64],
    qparam: QParam,
) -> (IrGraph, NodeId) {
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", input.to_vec(), dtype);
    let mut all_args = vec![IrArg::Node(x)];
    all_args.extend(args);
    let mut meta = NodeMeta::tensor(val.to_vec(), FLOAT32);
    meta.set_both_qparams(qparam);
    let id = call_with_meta(&mut ir, "op", target, all_args, meta);
    output(&mut ir, id);
    (ir, id)
}

#[test]
fn rewritten_avg_pool_serializes_with_materialized_zero_points() {
    // rc4 quantizes each zero point with the pool's output qparams (critique
    // A7): zp -4 at scale 0.05 becomes round(-4 / 0.05) - 4 = -84.
    let (mut ir, _) = quantized_graph(
        &[1, 4, 9, 9],
        INT8,
        "aten.avg_pool2d.default",
        vec![
            i64_list_arg(&[3, 3]),
            i64_list_arg(&[2, 2]),
            i64_list_arg(&[1, 1]),
            IrArg::Bool(false),
            IrArg::Bool(false),
        ],
        &[1, 4, 5, 5],
        qp(0.05, -4, INT8),
    );
    assert!(IrRewriteAvgPool2dPass.run(&mut ir));
    assert!(IrMaterializeConstantsPass { in_region: false }.run(&mut ir));
    let items = serialize_explicit(&ir);

    let pool = single_op(&items, Op::AvgPool2d);
    let Attribute::AvgPool2d(attribute) = &pool.attribute else {
        panic!("expected an AVG_POOL2D attribute");
    };
    assert_eq!(attribute.kernel, [3, 3]);
    assert_eq!(attribute.stride, [2, 2]);
    assert_eq!(attribute.pad, [1, 1, 1, 1]);
    assert_eq!(attribute.acc_type, DType::Int32);
    assert_eq!(pool.inputs.len(), 3);
    for zp in &pool.inputs[1..] {
        let zp = tensor(&items, zp);
        assert_eq!(zp.dtype, DType::Int8);
        assert_eq!(zp.data.as_deref(), Some(&[-84i8 as u8][..]));
    }
    let pooled = tensor(&items, &pool.outputs[0]);
    assert_eq!(
        (pooled.shape.as_slice(), pooled.dtype),
        (&[1, 5, 5, 4][..], DType::Int8)
    );
    assert_nhwc_transposes(&items);
    let out = tensor(&items, &items.output_names[0]);
    assert_eq!(
        (out.shape.as_slice(), out.dtype),
        (&[1, 4, 5, 5][..], DType::Int8)
    );
}

#[test]
fn rewritten_max_pool_with_indices_serializes() {
    // RETAB's form: the tuple-valued pool with input qparams, its getitem with
    // the val and output qparams (critique A3/B3).
    let q = qp(0.05, -4, INT8);
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", vec![1, 4, 9, 9], INT8);
    let pool = call_with_meta(
        &mut ir,
        "pool",
        "aten.max_pool2d_with_indices.default",
        vec![
            IrArg::Node(x),
            i64_list_arg(&[3, 3]),
            i64_list_arg(&[2, 2]),
            i64_list_arg(&[1, 1]),
        ],
        NodeMeta::default(),
    );
    ir.node_mut(pool).meta.set_single_input_qparam(q.clone());
    let getitem = call(
        &mut ir,
        "getitem",
        "<built-in function getitem>",
        vec![IrArg::Node(pool), IrArg::Int(0)],
        vec![1, 4, 5, 5],
        FLOAT32,
    );
    ir.node_mut(getitem).meta.set_single_output_qparam(q);
    output(&mut ir, getitem);
    assert!(IrRewriteMaxPool2dPass.run(&mut ir));
    let items = serialize_explicit(&ir);

    let pool = single_op(&items, Op::MaxPool2d);
    let Attribute::MaxPool2d(attribute) = &pool.attribute else {
        panic!("expected a MAX_POOL2D attribute");
    };
    assert_eq!(attribute.kernel, [3, 3]);
    assert_eq!(attribute.stride, [2, 2]);
    assert_eq!(attribute.pad, [1, 1, 1, 1]);
    let pooled = tensor(&items, &pool.outputs[0]);
    assert_eq!(
        (pooled.shape.as_slice(), pooled.dtype),
        (&[1, 5, 5, 4][..], DType::Int8)
    );
    assert_nhwc_transposes(&items);
    let out = tensor(&items, &items.output_names[0]);
    assert_eq!(
        (out.shape.as_slice(), out.dtype),
        (&[1, 4, 5, 5][..], DType::Int8)
    );
}

#[test]
fn rewritten_nearest_upsample_serializes_rc4_parameters() {
    let (mut ir, _) = quantized_graph(
        &[1, 3, 5, 5],
        INT8,
        "aten.upsample_nearest2d.vec",
        vec![i64_list_arg(&[8, 8]), IrArg::None],
        &[1, 3, 8, 8],
        qp(0.05, -4, INT8),
    );
    assert!(IrRewriteUpsamplePass.run(&mut ir));
    let items = serialize_explicit(&ir);
    let resize = single_op(&items, Op::Resize);
    let Attribute::Resize(attribute) = &resize.attribute else {
        panic!("expected a RESIZE attribute");
    };
    assert_eq!(attribute.mode, ResizeMode::Nearest);
    assert_eq!(shape_values(&items, &resize.inputs[1]), [16, 10, 16, 10]);
    assert_eq!(shape_values(&items, &resize.inputs[2]), [-8, -8]);
    assert_eq!(shape_values(&items, &resize.inputs[3]), [-2, -2]);
    let resized = tensor(&items, &resize.outputs[0]);
    assert_eq!(
        (resized.shape.as_slice(), resized.dtype),
        (&[1, 8, 8, 3][..], DType::Int8)
    );
    assert!(ops_of(&items, Op::Rescale).is_empty());
    assert_nhwc_transposes(&items);
    let out = tensor(&items, &items.output_names[0]);
    assert_eq!(
        (out.shape.as_slice(), out.dtype),
        (&[1, 3, 8, 8][..], DType::Int8)
    );
}

#[test]
fn rewritten_bilinear_upsamples_serialize_their_accumulators() {
    // int8: an int32 RESIZE (design F10) into a RESCALE by 1 / (4 * 4) that
    // reads int32 (critique A8/B4); int16: INT48 and a 16-bit RESCALE.
    for (dtype, accumulator, scale32) in [(INT8, DType::Int32, true), (INT16, DType::Int48, false)]
    {
        let zp = if dtype == INT16 { 0 } else { -4 };
        let (mut ir, _) = quantized_graph(
            &[1, 3, 4, 4],
            dtype,
            "aten.upsample_bilinear2d.vec",
            vec![
                IrArg::None,
                IrArg::Bool(false),
                IrArg::List(vec![IrArg::Float(2.0), IrArg::Float(2.0)]),
            ],
            &[1, 3, 8, 8],
            qp(0.05, zp, dtype),
        );
        assert!(IrRewriteUpsamplePass.run(&mut ir));
        let items = serialize_explicit(&ir);
        let activation = visitors::ir_dtype_to_tosa(IrDType::from_torch_str(dtype));
        let resize = single_op(&items, Op::Resize);
        assert_eq!(shape_values(&items, &resize.inputs[1]), [4, 2, 4, 2]);
        assert_eq!(shape_values(&items, &resize.inputs[2]), [-1, -1]);
        assert_eq!(shape_values(&items, &resize.inputs[3]), [1, 1]);
        assert_eq!(
            tensor(&items, &resize.outputs[0]).dtype,
            accumulator,
            "{dtype}"
        );
        let rescale = single_op(&items, Op::Rescale);
        assert_eq!(rescale.inputs[0], resize.outputs[0]);
        assert_eq!(
            tensor(&items, &rescale.inputs[3]).dtype,
            accumulator,
            "{dtype}"
        );
        let Attribute::Rescale(attribute) = &rescale.attribute else {
            panic!("expected a RESCALE attribute");
        };
        assert_eq!(attribute.scale32, scale32, "{dtype}");
        assert_eq!(
            tensor(&items, &rescale.outputs[0]).dtype,
            activation,
            "{dtype}"
        );
        assert_nhwc_transposes(&items);
        let out = tensor(&items, &items.output_names[0]);
        assert_eq!(
            (out.shape.as_slice(), out.dtype),
            (&[1, 3, 8, 8][..], activation)
        );
    }
}
