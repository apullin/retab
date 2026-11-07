//! Serializer tests for the explicit layout model (APU-135 P2 design S2). Each test hand-builds
//! the graph one of rc4's channels-last rewrites leaves (RewriteConvPass,
//! RewriteMaxPool2dPass, RewriteAvgPool2dPass, RewriteUpsamplePass), ingests it the way
//! `ingest_and_serialize` does, and checks the TOSA against what rc4 emits for it.
//!
//! RETAB's IR keeps quantized compute vals float32 and types them from qparams at
//! ingest (design F2). The nodes these rewrites create carry rc4's dtypes instead: a TOSA
//! conv is int32 and inherits the aten conv's qparams, and its layout permutes have none.

use super::collection::{check_explicit_layout_invariants, collect_regions, SerializedGraphItems};
use super::tosa_types::{Attribute, DType, Op, ResizeMode, TosaOperator, TosaTensor};
use super::visitors;
use crate::fx_ir::{
    i64_list_arg, IrArg, IrGraph, MetaValue, NodeId, NodeTensor, PlaceholderTensorBinding, QParam,
    QParamMap, TensorSource,
};
use crate::graph::ir::{Arg, DType as IrDType, NodeMeta, OpTarget, OpType, TensorInfo};
use crate::graph::Graph;
use crate::lowering_effects::LoweringInputKind;
use crate::store::{TensorData, TensorStore};
use crate::test_ir::{call, call_default, output, placeholder};
use smallvec::smallvec;

const INT8: &str = "torch.int8";
const INT16: &str = "torch.int16";
const INT32: &str = "torch.int32";
const FLOAT32: &str = "torch.float32";

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

fn qp(scale: f64, zp: i64, dtype: &str) -> QParam {
    let (qmin, qmax) = IrDType::from_torch_str(dtype)
        .quantized_integer_bounds()
        .unwrap();
    QParam::per_tensor(scale, zp, qmin, qmax, dtype.to_string())
}

fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + 3) as u8).collect()
}

fn numel(shape: &[i64]) -> usize {
    shape.iter().product::<i64>() as usize
}

fn permuted(shape: &[i64], dims: &[i64]) -> Vec<i64> {
    dims.iter().map(|&dim| shape[dim as usize]).collect()
}

fn i32s(shape: &[i64]) -> Vec<i32> {
    shape.iter().map(|&dim| dim as i32).collect()
}

/// A frozen constant input.
fn constant(
    graph: &mut IrGraph,
    name: &str,
    shape: &[i64],
    dtype: IrDType,
    data: Vec<u8>,
) -> NodeId {
    let id = placeholder(graph, name, shape.to_vec(), dtype.torch_str());
    graph.node_mut(id).tensor = Some(NodeTensor {
        source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
            input_kind: LoweringInputKind::Buffer,
            target_name: name.to_string(),
        }),
        data: TensorData {
            data,
            shape: shape.to_vec(),
            dtype,
        },
    });
    id
}

fn mark_special(graph: &mut IrGraph, id: NodeId, special: &str) {
    graph
        .node_mut(id)
        .meta
        .set_extra("tosa_special_dtype", MetaValue::Str(special.to_string()));
}

/// A layout permute with the permuted val and no qparams, as rc4's conv and upsample
/// rewrites create it. (The pool rewrites copy the pool's qparams onto theirs; the
/// serializer types the permute the same either way.)
fn permute(graph: &mut IrGraph, name: &str, input: NodeId, dims: &[i64], dtype: &str) -> NodeId {
    let shape = permuted(graph.node_shape(input).unwrap(), dims);
    call(
        graph,
        name,
        "aten.permute_copy.default",
        vec![IrArg::Node(input), i64_list_arg(dims)],
        shape,
        dtype,
    )
}

fn const_shape(graph: &mut IrGraph, name: &str, values: &[i64]) -> NodeId {
    let id = call_default(
        graph,
        name,
        "backend.tosa.CONST_SHAPE.default",
        vec![i64_list_arg(values)],
    );
    mark_special(graph, id, "SHAPE");
    id
}

#[allow(clippy::too_many_arguments)]
fn rescale(
    graph: &mut IrGraph,
    name: &str,
    input: NodeId,
    dtype: &str,
    scale: f64,
    input_zp: i64,
    output_zp: i64,
    shape: &[i64],
) -> NodeId {
    call(
        graph,
        name,
        "backend.tosa.RESCALE.default",
        vec![
            IrArg::Node(input),
            IrArg::Dtype(dtype.to_string()),
            IrArg::List(vec![IrArg::Float(scale)]),
            IrArg::Int(input_zp),
            IrArg::Int(output_zp),
        ],
        shape.to_vec(),
        dtype,
    )
}

fn list(values: &[i64]) -> IrArg {
    i64_list_arg(values)
}

/// The dims that move a channels-last tensor of `rank` to channels-first, and back.
fn layout_dims(rank: usize) -> (Vec<i64>, Vec<i64>) {
    let rank = rank as i64;
    let to_first = [0, rank - 1].into_iter().chain(1..rank - 1).collect();
    let to_last = std::iter::once(0)
        .chain(2..rank)
        .chain(std::iter::once(1))
        .collect();
    (to_first, to_last)
}

/// One RewriteConvPass result: `x -> permute -> <TOSA conv>(x', W', b, params..) ->
/// RESCALE -> permute`, with the rewritten conv's val, inherited qparams and (A16W8) the
/// INT48 marks.
struct ConvCase {
    target: &'static str,
    /// The TOSA conv's (channels-last) input.
    input: &'static [i64],
    /// The pre-permuted weight: OHWI, HWCM or ODHWI.
    weight: &'static [i64],
    /// The arguments after (x', W', b).
    params: Vec<IrArg>,
    /// The TOSA conv's val.
    val: &'static [i64],
    activation: &'static str,
}

impl ConvCase {
    fn build(&self) -> IrGraph {
        let (to_first, to_last) = layout_dims(self.input.len());
        let mut ir = IrGraph::empty();
        let x = placeholder(
            &mut ir,
            "x",
            permuted(self.input, &to_first),
            self.activation,
        );
        let x_nhwc = permute(&mut ir, "x_nhwc", x, &to_last, self.activation);
        let w = constant(
            &mut ir,
            "w",
            self.weight,
            IrDType::Int8,
            ramp(numel(self.weight)),
        );
        let channels = *self.val.last().unwrap();
        let b = constant(
            &mut ir,
            "b",
            &[channels],
            IrDType::Int32,
            (0..channels as i32).flat_map(i32::to_le_bytes).collect(),
        );
        let mut args = vec![IrArg::Node(x_nhwc), IrArg::Node(w), IrArg::Node(b)];
        args.extend(self.params.iter().cloned());
        let conv = call(&mut ir, "conv", self.target, args, self.val.to_vec(), INT32);
        let int16 = self.activation == INT16;
        let (input_zp, output_zp) = if int16 { (0, 0) } else { (-3, 5) };
        let meta = &mut ir.node_mut(conv).meta;
        meta.set_input_qparams_from([
            (0, qp(0.05, input_zp, self.activation)),
            (1, qp(0.02, 0, INT8)),
        ]);
        meta.set_single_output_qparam(qp(0.1, output_zp, self.activation));
        if int16 {
            mark_special(&mut ir, conv, "INT48");
            mark_special(&mut ir, b, "INT48");
        }
        let rescaled = rescale(
            &mut ir,
            "rescale",
            conv,
            self.activation,
            0.01,
            0,
            output_zp,
            self.val,
        );
        let out = permute(&mut ir, "out", rescaled, &to_first, self.activation);
        output(&mut ir, out);
        ir
    }

    /// The checks every conv kind shares: weights and vals serialize as given, the
    /// accumulator feeds the RESCALE, and only the two layout TRANSPOSEs remain.
    fn check<'a>(&self, items: &'a SerializedGraphItems, op: Op) -> &'a TosaOperator {
        // B1: no second weight permutation.
        let weight = tensor(items, "w");
        assert_eq!(weight.shape, i32s(self.weight));
        assert_eq!(weight.data.as_deref(), Some(&ramp(numel(self.weight))[..]));

        let conv = single_op(items, op);
        assert_eq!(&conv.inputs[..3], ["x_nhwc", "w", "b"]);
        assert_eq!(conv.inputs.len(), 5);
        let activation = visitors::ir_dtype_to_tosa(IrDType::from_torch_str(self.activation));
        assert_eq!(tensor(items, &conv.inputs[3]).dtype, activation);
        assert_eq!(tensor(items, &conv.inputs[4]).dtype, DType::Int8);
        // G1 and A9: the val's shape, the accumulator's dtype (not the inherited int8
        // output qparam).
        let accumulator = if self.activation == INT16 {
            DType::Int48
        } else {
            DType::Int32
        };
        let conv_output = tensor(items, &conv.outputs[0]);
        assert_eq!(conv_output.shape, i32s(self.val));
        assert_eq!(conv_output.dtype, accumulator);

        // X1 and G3: the RESCALE reads the accumulator at the val's shape.
        let rescale = single_op(items, Op::Rescale);
        assert_eq!(rescale.inputs[0], conv.outputs[0]);
        assert_eq!(tensor(items, &rescale.inputs[3]).dtype, accumulator);
        let rescaled = tensor(items, "rescale");
        assert_eq!(rescaled.shape, i32s(self.val));
        assert_eq!(rescaled.dtype, activation);
        let Attribute::Rescale(attribute) = &rescale.attribute else {
            panic!("expected a RESCALE attribute");
        };
        assert_eq!(attribute.scale32, accumulator != DType::Int48);

        // B2, G4 and G5: the output permute keeps the val, with no peephole and no
        // inferred output-layout TRANSPOSE.
        let (to_first, _) = layout_dims(self.input.len());
        assert_eq!(ops_of(items, Op::Transpose).len(), 2);
        assert_eq!(
            tensor(items, "out").shape,
            i32s(&permuted(self.val, &to_first))
        );
        assert_eq!(items.output_names, ["out"]);
        conv
    }
}

// ---------------------------------------------------------------------------
// Ingest, serialization and lookups
// ---------------------------------------------------------------------------

/// Ingest `ir` as `ingest_and_serialize` does: convert, then finalize.
fn ingest(ir: &IrGraph) -> (Graph, TensorStore) {
    let mut store = TensorStore::new();
    let mut graph = crate::ingest::from_ir::graph_from_ir(ir, &mut store);
    crate::ingest::finalize::finalize_converted_graph(&mut graph);
    graph.rebuild_user_lists();
    (graph, store)
}

fn try_serialize_graph(graph: Graph, store: &TensorStore) -> Result<SerializedGraphItems, String> {
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    let mut regions = collect_regions(&[("main".to_string(), graph)], store)?;
    Ok(regions.remove(0).items)
}

fn serialize_explicit(ir: &IrGraph) -> SerializedGraphItems {
    let (graph, store) = ingest(ir);
    try_serialize_graph(graph, &store).unwrap()
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

/// Every operand that names a shape is declared by an earlier CONST_SHAPE.
fn assert_shape_operands_declared(items: &SerializedGraphItems) {
    let mut declared = std::collections::HashSet::new();
    for op in &items.operators {
        for input in &op.inputs {
            if items.shapes.iter().any(|shape| &shape.name == input) {
                assert!(declared.contains(input), "{input} used before CONST_SHAPE");
            }
        }
        if op.op == Op::ConstShape {
            declared.extend(op.outputs.iter().cloned());
        }
    }
    for shape in &items.shapes {
        assert!(
            declared.contains(&shape.name),
            "{} is never declared",
            shape.name
        );
    }
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

#[test]
fn conv2d_serializes_ohwi_weights_val_shapes_and_pads_as_given() {
    // rc4's CONV2D for Conv2d(3, 8, (3, 5), padding=(1, 2)) on [1, 3, 6, 10]. OHWI
    // [8, 3, 5, 3]: the dim-order serializer permuted it again to [8, 5, 3, 3] (B1).
    let case = ConvCase {
        target: "backend.tosa.CONV2D.default",
        input: &[1, 6, 10, 3],
        weight: &[8, 3, 5, 3],
        params: vec![list(&[1, 1]), list(&[1, 1, 2, 2]), list(&[1, 1])],
        val: &[1, 6, 10, 8],
        activation: INT8,
    };
    let items = serialize_explicit(&case.build());
    let conv = case.check(&items, Op::Conv2d);
    let Attribute::Conv2d(attribute) = &conv.attribute else {
        panic!("expected a CONV2D attribute");
    };
    assert_eq!(attribute.pad, [1, 1, 2, 2]);
    assert_eq!(attribute.stride, [1, 1]);
    assert_eq!(attribute.dilation, [1, 1]);
    assert_eq!(attribute.acc_type, DType::Int32);
    assert_eq!(tensor(&items, "b").dtype, DType::Int32);
}

#[test]
fn depthwise_hwcm_weights_keep_their_layout_and_the_val_shape() {
    // KW == 1 weights matched the dim-order serializer's OIHW guess (B3): [3, 1, 4, 2]
    // would give a [1, 7, 5, 3] output, [1, 1, 4, 2] a [1, 2, 4, 1] one. [4, 1, 2, 2]
    // (KH % C == 0) was also re-permuted as OIHW by its weight plan (B1). IrRewriteConv
    // makes KW == 1 weights routinely through the Cin = 1 depthwise rule (design §2.4,
    // Risk 4): cumsum's [K, 1, 1, 1] and Linear(in=1)'s [1, 1, 1, Co]. Dim-order logic got
    // both wrong: [5, 1, 1, 1] became [1, 1, 1, 5] with a [3, 13, 4, 5] output, and
    // [1, 1, 1, 6] became [1, 6, 1, 1] with a [1, 4, 6, 1] output. Each case is the
    // DEPTHWISE_CONV2D rc4 emits for the module in its comment.
    for (input, weight, params, val) in [
        // Conv2d(4, 8, (3, 1), groups=4, stride=2, padding=(1, 0)) on [1, 4, 16, 11]
        (
            &[1, 16, 11, 4][..],
            &[3, 1, 4, 2][..],
            vec![list(&[2, 2]), list(&[1, 0, 0, 0]), list(&[1, 1])],
            &[1, 8, 6, 8][..],
        ),
        // Conv2d(4, 8, 1, groups=4) on [1, 4, 5, 5]
        (
            &[1, 5, 5, 4][..],
            &[1, 1, 4, 2][..],
            vec![list(&[1, 1]), list(&[0, 0, 0, 0]), list(&[1, 1])],
            &[1, 5, 5, 8][..],
        ),
        // Conv2d(2, 4, (4, 1), groups=2, padding=(2, 0)) on [1, 2, 8, 8]; at stride 1 the
        // pad needs no adjustment.
        (
            &[1, 8, 8, 2][..],
            &[4, 1, 2, 2][..],
            vec![list(&[1, 1]), list(&[2, 2, 0, 0]), list(&[1, 1])],
            &[1, 9, 8, 4][..],
        ),
        // torch.cumsum(x, dim=2) on [1, 3, 5, 4], whose input is reshaped to [3, 5, 4, 1]
        (
            &[3, 5, 4, 1][..],
            &[5, 1, 1, 1][..],
            vec![list(&[1, 1]), list(&[4, 4, 0, 0]), list(&[1, 1])],
            &[3, 9, 4, 1][..],
        ),
        // Linear(1, 6) on [4, 1], whose input is reshaped to [1, 1, 4, 1]
        (
            &[1, 1, 4, 1][..],
            &[1, 1, 1, 6][..],
            vec![list(&[1, 1]), list(&[0, 0, 0, 0]), list(&[1, 1])],
            &[1, 1, 4, 6][..],
        ),
    ] {
        let case = ConvCase {
            target: "backend.tosa.DEPTHWISE_CONV2D.default",
            input,
            weight,
            params: params.clone(),
            val,
            activation: INT8,
        };
        let items = serialize_explicit(&case.build());
        let conv = case.check(&items, Op::DepthwiseConv2d);
        let Attribute::DepthwiseConv2d(attribute) = &conv.attribute else {
            panic!("expected a DEPTHWISE_CONV2D attribute");
        };
        // G2: rc4's already-adjusted pad goes through untouched.
        let Some(IrArg::List(pad)) = params.get(1) else {
            unreachable!()
        };
        let pad: Vec<i32> = pad
            .iter()
            .map(|value| match value {
                IrArg::Int(value) => *value as i32,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(attribute.pad, pad, "{weight:?}");
        assert_eq!(attribute.acc_type, DType::Int32);
    }
}

#[test]
fn conv3d_is_an_int32_accumulator_feeding_its_rescale() {
    // X1: without TosaConv3d in the accumulator lists, ingest types CONV3D int8 from its
    // inherited output qparam and its RESCALE declares an int8 input zero point.
    let case = ConvCase {
        target: "backend.tosa.CONV3D.default",
        input: &[1, 4, 6, 6, 2],
        weight: &[4, 3, 3, 3, 2],
        params: vec![
            list(&[1, 1, 1]),
            list(&[1, 1, 1, 1, 1, 1]),
            list(&[1, 1, 1]),
        ],
        val: &[1, 4, 6, 6, 4],
        activation: INT8,
    };
    let ir = case.build();
    let (graph, store) = ingest(&ir);
    let conv = graph
        .order
        .iter()
        .filter_map(|&id| graph.node(id))
        .find(|node| node.name == "conv")
        .unwrap();
    assert_eq!(conv.meta.val.dtype, IrDType::Int32);
    let items = try_serialize_graph(graph, &store).unwrap();
    let conv = case.check(&items, Op::Conv3d);
    let Attribute::Conv3d(attribute) = &conv.attribute else {
        panic!("expected a CONV3D attribute");
    };
    assert_eq!(attribute.pad, [1, 1, 1, 1, 1, 1]);
    assert_eq!(attribute.acc_type, DType::Int32);
}

#[test]
fn transpose_conv2d_serializes_rc4_arguments() {
    // IOHW [6, 5, 3, 3] permuted (1, 2, 3, 0) to OHWI; padding 1 becomes out_pad -1.
    let case = ConvCase {
        target: "backend.tosa.TRANSPOSE_CONV2D.default",
        input: &[1, 4, 4, 6],
        weight: &[5, 3, 3, 6],
        params: vec![list(&[-1, -1, -1, -1]), list(&[2, 2])],
        val: &[1, 7, 7, 5],
        activation: INT8,
    };
    let items = serialize_explicit(&case.build());
    let conv = case.check(&items, Op::TransposeConv2d);
    let Attribute::TransposeConv2d(attribute) = &conv.attribute else {
        panic!("expected a TRANSPOSE_CONV2D attribute");
    };
    assert_eq!(attribute.out_pad, [-1, -1, -1, -1]);
    assert_eq!(attribute.stride, [2, 2]);
    assert_eq!(attribute.acc_type, DType::Int32);
}

#[test]
fn int16_conv_accumulates_in_int48_and_rescales_with_16_bit_multipliers() {
    let case = ConvCase {
        target: "backend.tosa.CONV2D.default",
        input: &[1, 4, 4, 2],
        weight: &[3, 1, 1, 2],
        params: vec![list(&[1, 1]), list(&[0, 0, 0, 0]), list(&[1, 1])],
        val: &[1, 4, 4, 3],
        activation: INT16,
    };
    let items = serialize_explicit(&case.build());
    let conv = case.check(&items, Op::Conv2d);
    let Attribute::Conv2d(attribute) = &conv.attribute else {
        panic!("expected a CONV2D attribute");
    };
    assert_eq!(attribute.acc_type, DType::Int48);
    let bias = tensor(&items, "b");
    assert_eq!(bias.dtype, DType::Int48);
    assert_eq!(bias.data.as_ref().map(Vec::len), Some(3 * 6));
    let rescale = single_op(&items, Op::Rescale);
    assert_eq!(tensor(&items, &rescale.inputs[1]).dtype, DType::Int16);
}

#[test]
fn conv1d_chain_reshapes_around_its_conv2d() {
    // rc4's conv1d form, for Conv1d(3, 5, 3, padding=1) on [1, 3, 10]: permute and view
    // the input to [N, 1, L, C], CONV2D with an OHWI [O, 1, K, I] weight and pad
    // [0, 0, p, p], RESCALE, then view back to [N, Lout, O] and permute.
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [1, 3, 10], INT8);
    let x_nlc = permute(&mut ir, "x_nlc", x, &[0, 2, 1], INT8);
    let x_nhwc = call(
        &mut ir,
        "x_nhwc",
        "aten.view_copy.default",
        vec![IrArg::Node(x_nlc), list(&[1, 1, 10, 3])],
        [1, 1, 10, 3],
        INT8,
    );
    let weight = [5, 1, 3, 3];
    let w = constant(&mut ir, "w", &weight, IrDType::Int8, ramp(numel(&weight)));
    let b = constant(
        &mut ir,
        "b",
        &[5],
        IrDType::Int32,
        (0..5i32).flat_map(i32::to_le_bytes).collect(),
    );
    let conv = call(
        &mut ir,
        "conv",
        "backend.tosa.CONV2D.default",
        vec![
            IrArg::Node(x_nhwc),
            IrArg::Node(w),
            IrArg::Node(b),
            list(&[1, 1]),
            list(&[0, 0, 1, 1]),
            list(&[1, 1]),
        ],
        [1, 1, 10, 5],
        INT32,
    );
    let meta = &mut ir.node_mut(conv).meta;
    meta.set_input_qparams_from([(0, qp(0.05, -3, INT8)), (1, qp(0.02, 0, INT8))]);
    meta.set_single_output_qparam(qp(0.1, 5, INT8));
    let rescaled = rescale(&mut ir, "rescale", conv, INT8, 0.01, 0, 5, &[1, 1, 10, 5]);
    let y_nlc = call(
        &mut ir,
        "y_nlc",
        "aten.view_copy.default",
        vec![IrArg::Node(rescaled), list(&[1, 10, 5])],
        [1, 10, 5],
        INT8,
    );
    let y = permute(&mut ir, "y", y_nlc, &[0, 2, 1], INT8);
    output(&mut ir, y);

    let items = serialize_explicit(&ir);
    let weight_tensor = tensor(&items, "w");
    assert_eq!(weight_tensor.shape, i32s(&weight));
    assert_eq!(
        weight_tensor.data.as_deref(),
        Some(&ramp(numel(&weight))[..])
    );
    let conv = single_op(&items, Op::Conv2d);
    assert_eq!(&conv.inputs[..3], ["x_nhwc", "w", "b"]);
    let Attribute::Conv2d(attribute) = &conv.attribute else {
        panic!("expected a CONV2D attribute");
    };
    assert_eq!(attribute.pad, [0, 0, 1, 1]);
    assert_eq!(attribute.stride, [1, 1]);
    assert_eq!(attribute.dilation, [1, 1]);
    let conv_output = tensor(&items, &conv.outputs[0]);
    assert_eq!(conv_output.shape, [1, 1, 10, 5]);
    assert_eq!(conv_output.dtype, DType::Int32);
    let rescale = single_op(&items, Op::Rescale);
    assert_eq!(rescale.inputs[0], conv.outputs[0]);
    assert_eq!(tensor(&items, &rescale.inputs[3]).dtype, DType::Int32);
    assert_eq!(tensor(&items, "rescale").shape, [1, 1, 10, 5]);
    assert_eq!(tensor(&items, "rescale").dtype, DType::Int8);

    let reshapes = ops_of(&items, Op::Reshape);
    assert_eq!(reshapes.len(), 2);
    assert_eq!(reshapes[0].inputs[0], "x_nlc");
    assert_eq!(reshapes[1].inputs[0], "rescale");
    assert_eq!(ops_of(&items, Op::Transpose).len(), 2);
    assert_transposes_consistent(&items);
    for (name, shape) in [
        ("x_nlc", &[1, 10, 3][..]),
        ("x_nhwc", &[1, 1, 10, 3][..]),
        ("y_nlc", &[1, 10, 5][..]),
        ("y", &[1, 5, 10][..]),
    ] {
        assert_eq!(tensor(&items, name).shape, shape, "{name}");
    }
    assert_eq!(items.output_names, ["y"]);
}

// ---------------------------------------------------------------------------
// Pools
// ---------------------------------------------------------------------------

/// `x -> permute -> <pool> -> permute -> output`, as rc4's pool rewrites leave it.
fn pool_chain(
    target: &str,
    extra: impl FnOnce(&mut IrGraph) -> Vec<IrArg>,
    val: &[i64],
) -> IrGraph {
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [1, 4, 9, 9], INT8);
    let x_nhwc = permute(&mut ir, "x_nhwc", x, &[0, 2, 3, 1], INT8);
    let mut args = vec![IrArg::Node(x_nhwc)];
    args.extend(extra(&mut ir));
    let pool = call(&mut ir, "pool", target, args, val.to_vec(), INT8);
    ir.node_mut(pool).meta.set_both_qparams(qp(0.05, -4, INT8));
    let out = permute(&mut ir, "out", pool, &[0, 3, 1, 2], INT8);
    output(&mut ir, out);
    ir
}

#[test]
fn max_pool2d_reads_rc4_positional_arguments() {
    // X2: the TOSA visitor read kwargs only, so the kernel fell back to [2, 2].
    let mut ir = pool_chain(
        "backend.tosa.MAX_POOL2D.default",
        |_| vec![list(&[3, 3]), list(&[2, 2]), list(&[1, 1, 1, 1])],
        &[1, 5, 5, 4],
    );
    for input_qparams_only in [false, true] {
        if input_qparams_only {
            // From `max_pool2d_with_indices` + getitem, rc4's RemoveGetItemPass keeps the
            // pool node's meta, which has no output qparams (critique A3/B3).
            let pool = ir.by_name("pool").unwrap().id;
            ir.node_mut(pool).meta.set_output_qparams(QParamMap::new());
        }
        let items = serialize_explicit(&ir);
        let pool = single_op(&items, Op::MaxPool2d);
        let Attribute::MaxPool2d(attribute) = &pool.attribute else {
            panic!("expected a MAX_POOL2D attribute");
        };
        assert_eq!(attribute.kernel, [3, 3]);
        assert_eq!(attribute.stride, [2, 2]);
        assert_eq!(attribute.pad, [1, 1, 1, 1]);
        assert_eq!(pool.inputs, ["x_nhwc"]);
        assert_eq!(tensor(&items, "pool").shape, [1, 5, 5, 4]);
        assert_eq!(tensor(&items, "pool").dtype, DType::Int8);
        assert_eq!(tensor(&items, "out").shape, [1, 4, 5, 5]);
    }
}

#[test]
fn avg_pool2d_reads_rc4_arguments_and_zero_point_operands() {
    // X3: the zero points are graph constants, and kernel/stride/pad/acc_type are
    // arguments 3-6. rc4 materializes each zero point as `full((1,), zp)` carrying the
    // pool's meta, and ComputeConstantOpsAOTPass quantizes it with the pool's output
    // qparams: zp -4 at scale 0.05 becomes round(-4 / 0.05) - 4 = -84.
    // B1: the TRANSPOSE that consumes AVG_POOL2D sees the val, not a shape recomputed from
    // the zero-point operands ([1, 7, 7, 4]).
    const ZP: u8 = -84i8 as u8;
    let ir = pool_chain(
        "backend.tosa.AVG_POOL2D.default",
        |ir| {
            let input_zp = constant(ir, "input_zp", &[1], IrDType::Int8, vec![ZP]);
            let output_zp = constant(ir, "output_zp", &[1], IrDType::Int8, vec![ZP]);
            vec![
                IrArg::Node(input_zp),
                IrArg::Node(output_zp),
                list(&[3, 3]),
                list(&[2, 2]),
                list(&[1, 1, 1, 1]),
                IrArg::Dtype(INT32.to_string()),
            ]
        },
        &[1, 5, 5, 4],
    );
    let items = serialize_explicit(&ir);
    let pool = single_op(&items, Op::AvgPool2d);
    let Attribute::AvgPool2d(attribute) = &pool.attribute else {
        panic!("expected an AVG_POOL2D attribute");
    };
    assert_eq!(attribute.kernel, [3, 3]);
    assert_eq!(attribute.stride, [2, 2]);
    assert_eq!(attribute.pad, [1, 1, 1, 1]);
    assert_eq!(attribute.acc_type, DType::Int32);
    assert_eq!(pool.inputs, ["x_nhwc", "input_zp", "output_zp"]);
    for zp in ["input_zp", "output_zp"] {
        assert_eq!(tensor(&items, zp).data.as_deref(), Some(&[ZP][..]));
    }
    assert_eq!(tensor(&items, "pool").shape, [1, 5, 5, 4]);
    assert_eq!(tensor(&items, "pool").dtype, DType::Int8);
    assert_eq!(tensor(&items, "out").shape, [1, 4, 5, 5]);
    assert_eq!(items.tensors.len(), 6, "no serializer-made zero points");
}

#[test]
fn views_of_avg_pools_need_no_layout_transpose() {
    // G6: an AVG_POOL2D view already works in NHWC coordinates.
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [1, 4, 3, 5], INT8);
    let x_nhwc = permute(&mut ir, "x_nhwc", x, &[0, 2, 3, 1], INT8);
    let input_zp = constant(&mut ir, "input_zp", &[1], IrDType::Int8, vec![0]);
    let output_zp = constant(&mut ir, "output_zp", &[1], IrDType::Int8, vec![0]);
    let pool = call(
        &mut ir,
        "pool",
        "backend.tosa.AVG_POOL2D.default",
        vec![
            IrArg::Node(x_nhwc),
            IrArg::Node(input_zp),
            IrArg::Node(output_zp),
            list(&[3, 1]),
            list(&[1, 1]),
            list(&[0, 0, 0, 0]),
            IrArg::Dtype(INT32.to_string()),
        ],
        [1, 1, 5, 4],
        INT8,
    );
    ir.node_mut(pool).meta.set_both_qparams(qp(0.05, 0, INT8));
    let view = call(
        &mut ir,
        "view",
        "aten.view_copy.default",
        vec![IrArg::Node(pool), list(&[1, 5, 4])],
        [1, 5, 4],
        INT8,
    );
    output(&mut ir, view);
    let items = serialize_explicit(&ir);
    assert_eq!(single_op(&items, Op::Transpose).outputs, ["x_nhwc"]);
    let reshape = single_op(&items, Op::Reshape);
    assert_eq!(reshape.inputs[0], "pool");
    assert_eq!(tensor(&items, "view").shape, [1, 5, 4]);
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

/// `x -> permute -> RESIZE(scale, offset, border) [-> RESCALE] -> permute`, as
/// RewriteUpsamplePass leaves it. RESIZE inherits the upsample's qparams.
fn resize_chain(
    mode: &str,
    activation: &str,
    size: i64,
    out: i64,
    params: [&[i64]; 3],
    resize_dtype: &str,
    rescale_by: Option<f64>,
) -> IrGraph {
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [1, 3, size, size], activation);
    let x_nhwc = permute(&mut ir, "x_nhwc", x, &[0, 2, 3, 1], activation);
    let scale = const_shape(&mut ir, "scale", params[0]);
    let offset = const_shape(&mut ir, "offset", params[1]);
    let border = const_shape(&mut ir, "border", params[2]);
    let resize = call(
        &mut ir,
        "resize",
        "backend.tosa.RESIZE.default",
        vec![
            IrArg::Node(x_nhwc),
            IrArg::Node(scale),
            IrArg::Node(offset),
            IrArg::Node(border),
        ],
        [1, out, out, 3],
        resize_dtype,
    );
    ir.set_kwarg(resize, "resize_mode", IrArg::Str(mode.to_string()));
    let zp = if activation == INT16 { 0 } else { -4 };
    ir.node_mut(resize)
        .meta
        .set_both_qparams(qp(0.05, zp, activation));
    let last = match rescale_by {
        Some(scale) => rescale(
            &mut ir,
            "rescale",
            resize,
            activation,
            scale,
            0,
            0,
            &[1, out, out, 3],
        ),
        None => resize,
    };
    if activation == INT16 && mode == "bilinear" {
        mark_special(&mut ir, resize, "INT48");
    }
    let y = permute(&mut ir, "y", last, &[0, 3, 1, 2], activation);
    output(&mut ir, y);
    ir
}

#[test]
fn nearest_resize_serializes_rc4_parameters() {
    // X4: rc4's nearest 5 -> 8 is scale [16, 10], offset -8, border -2 per axis; the
    // aten upsample emitter recomputes offset -3, border 3. A float32 val (RETAB's usual
    // typing of quantized compute nodes) takes the inherited qparam's int8.
    for resize_dtype in [INT8, FLOAT32] {
        let ir = resize_chain(
            "nearest",
            INT8,
            5,
            8,
            [&[16, 10, 16, 10], &[-8, -8], &[-2, -2]],
            resize_dtype,
            None,
        );
        let items = serialize_explicit(&ir);
        let resize = single_op(&items, Op::Resize);
        assert_eq!(resize.inputs, ["x_nhwc", "scale", "offset", "border"]);
        assert_eq!(shape_values(&items, "scale"), [16, 10, 16, 10]);
        assert_eq!(shape_values(&items, "offset"), [-8, -8]);
        assert_eq!(shape_values(&items, "border"), [-2, -2]);
        let Attribute::Resize(attribute) = &resize.attribute else {
            panic!("expected a RESIZE attribute");
        };
        assert_eq!(attribute.mode, ResizeMode::Nearest);
        assert_eq!(tensor(&items, "resize").shape, [1, 8, 8, 3]);
        assert_eq!(
            tensor(&items, "resize").dtype,
            DType::Int8,
            "{resize_dtype}"
        );
        assert!(ops_of(&items, Op::Rescale).is_empty());
        assert_eq!(tensor(&items, "y").shape, [1, 3, 8, 8]);
        assert_eq!(tensor(&items, "y").dtype, DType::Int8, "{resize_dtype}");
        // No orphaned or duplicated parameter shapes.
        assert_eq!(ops_of(&items, Op::ConstShape).len(), 3);
        assert_shape_operands_declared(&items);
    }
}

#[test]
fn int8_bilinear_resize_stays_int32_into_its_rescale() {
    // X4, critique A8/B4: align_corners 5 -> 9 is scale [4, 2], offset 0, border 0 per
    // axis. The RESIZE inherits an int8 output qparam, yet produces int32, and its
    // RESCALE by 1 / (4 * 4) must read int32.
    let ir = resize_chain(
        "bilinear",
        INT8,
        5,
        9,
        [&[4, 2, 4, 2], &[0, 0], &[0, 0]],
        INT32,
        Some(1.0 / 16.0),
    );
    let (graph, store) = ingest(&ir);
    let resize = graph
        .order
        .iter()
        .filter_map(|&id| graph.node(id))
        .find(|node| node.name == "resize")
        .unwrap();
    assert_eq!(resize.meta.val.dtype, IrDType::Int32);
    let items = try_serialize_graph(graph, &store).unwrap();

    let resize = single_op(&items, Op::Resize);
    assert_eq!(resize.inputs[0], "x_nhwc");
    // Offset and border hold the same values, so they share one pooled CONST_SHAPE.
    for (input, values) in
        resize.inputs[1..]
            .iter()
            .zip([&[4, 2, 4, 2][..], &[0, 0][..], &[0, 0][..]])
    {
        assert_eq!(shape_values(&items, input), values);
    }
    assert_shape_operands_declared(&items);
    let Attribute::Resize(attribute) = &resize.attribute else {
        panic!("expected a RESIZE attribute");
    };
    assert_eq!(attribute.mode, ResizeMode::Bilinear);
    assert_eq!(tensor(&items, "resize").shape, [1, 9, 9, 3]);
    assert_eq!(tensor(&items, "resize").dtype, DType::Int32);

    let rescale = single_op(&items, Op::Rescale);
    assert_eq!(rescale.inputs[0], "resize");
    assert_eq!(tensor(&items, &rescale.inputs[3]).dtype, DType::Int32);
    let Attribute::Rescale(attribute) = &rescale.attribute else {
        panic!("expected a RESCALE attribute");
    };
    assert!(attribute.scale32);
    assert_eq!(tensor(&items, "rescale").dtype, DType::Int8);
    assert_eq!(tensor(&items, "y").shape, [1, 3, 9, 9]);
}

#[test]
fn int16_bilinear_resize_is_int48_into_its_rescale() {
    let ir = resize_chain(
        "bilinear",
        INT16,
        4,
        8,
        [&[4, 2, 4, 2], &[-1, -1], &[1, 1]],
        INT16,
        Some(1.0 / 16.0),
    );
    let items = serialize_explicit(&ir);
    assert_eq!(tensor(&items, "resize").dtype, DType::Int48);
    let rescale = single_op(&items, Op::Rescale);
    assert_eq!(tensor(&items, &rescale.inputs[3]).dtype, DType::Int48);
    let Attribute::Rescale(attribute) = &rescale.attribute else {
        panic!("expected a RESCALE attribute");
    };
    assert!(!attribute.scale32);
    assert_eq!(tensor(&items, "rescale").dtype, DType::Int16);
}

// ---------------------------------------------------------------------------
// The aten spatial-op guard
// ---------------------------------------------------------------------------

#[test]
fn aten_spatial_ops_send_explicit_graphs_to_stock() {
    // X5 and critique B11: aten convs, pools and upsamples have no visitor.
    for target in [
        "aten.convolution.default",
        "aten.conv1d.default",
        "aten.conv_transpose2d.input",
        "aten.max_pool2d.default",
        "aten.max_pool2d_with_indices.default",
        "aten.avg_pool2d.default",
        "aten.adaptive_avg_pool2d.default",
        "aten.upsample_nearest2d.vec",
        "aten.upsample_bilinear2d.vec",
    ] {
        let mut ir = IrGraph::empty();
        let x = placeholder(&mut ir, "x", [1, 2, 4, 4], INT8);
        let op = call(
            &mut ir,
            "op",
            target,
            vec![IrArg::Node(x)],
            [1, 2, 4, 4],
            INT8,
        );
        output(&mut ir, op);
        let (graph, store) = ingest(&ir);
        let error = try_serialize_graph(graph, &store).err().unwrap();
        assert!(
            error.starts_with("RETAB_STOCK_FALLBACK: aten spatial op in explicit layout: 'op' ("),
            "{target}: {error}"
        );
    }
}

#[test]
fn explicit_mode_keeps_the_layout_transposes_a_peephole_would_cancel() {
    // G5: no post-emission peephole cancels the inverse pair; the IR's layout passes own
    // the TRANSPOSEs.
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [1, 2, 3, 4], FLOAT32);
    let nhwc = permute(&mut ir, "nhwc", x, &[0, 2, 3, 1], FLOAT32);
    let nchw = permute(&mut ir, "nchw", nhwc, &[0, 3, 1, 2], FLOAT32);
    let neg = call(
        &mut ir,
        "neg",
        "aten.neg.default",
        vec![IrArg::Node(nchw)],
        [1, 2, 3, 4],
        FLOAT32,
    );
    output(&mut ir, neg);
    let explicit = serialize_explicit(&ir);
    assert_eq!(ops_of(&explicit, Op::Transpose).len(), 2);
    assert_eq!(single_op(&explicit, Op::Negate).inputs[0], "nchw");
}

#[test]
fn explicit_main_outputs_get_no_inferred_layout_transpose() {
    // G4: no output TRANSPOSE is inferred when a rank-4 output's shape and val disagree;
    // IrNormalizeDelegateIOLayout owns the boundary.
    let mut graph = Graph::new();
    let mut input = |name: &str| {
        graph.create_node(
            OpType::Placeholder,
            OpTarget::Other(name.into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[1, 2, 3, 4], IrDType::Float32)),
            name.into(),
        )
    };
    let (x, y) = (input("x"), input("y"));
    let add = graph.create_node(
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(x), Arg::Node(y)],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 3, 4, 2], IrDType::Float32)),
        "add".into(),
    );
    graph.create_node(
        OpType::Output,
        OpTarget::Other("output".into()),
        smallvec![Arg::Node(add)],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[], IrDType::Float32)),
        "output".into(),
    );
    graph.inputs = vec![x, y];
    let explicit = try_serialize_graph(graph, &TensorStore::new()).unwrap();
    assert_eq!(explicit.output_names, ["add"]);
    assert!(ops_of(&explicit, Op::Transpose).is_empty());
}

#[test]
fn rank4_getitems_and_tosa_convs_keep_their_val_shapes() {
    // B2: rank-4 getitems and TOSA conv/pool vals are TOSA shapes, not logical NCHW ones
    // to reorder to [d0, d2, d3, d1].
    let mut graph = Graph::new();
    let x = graph.create_node(
        OpType::Placeholder,
        OpTarget::Other("x".into()),
        smallvec![],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[2], IrDType::Int8)),
        "x".into(),
    );
    let getitem = graph.create_node(
        OpType::CallFunction,
        OpTarget::Other("getitem".into()),
        smallvec![Arg::Node(x), Arg::Int(0)],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 8, 6, 5], IrDType::Int8)),
        "getitem".into(),
    );
    let pool = graph.create_node(
        OpType::CallFunction,
        OpTarget::TosaMaxPool2d,
        smallvec![
            Arg::Node(getitem),
            Arg::IntList(smallvec![1, 1]),
            Arg::IntList(smallvec![1, 1]),
            Arg::IntList(smallvec![0, 0, 0, 0])
        ],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 8, 6, 5], IrDType::Int8)),
        "pool".into(),
    );
    let store = TensorStore::new();
    visitors::shape_utils::clear_shape_cache();
    for id in [getitem, pool] {
        let node = graph.node(id).unwrap();
        assert_eq!(
            visitors::actual_tensor_shape(&graph, &store, node),
            vec![1, 8, 6, 5],
            "{}",
            node.name
        );
    }
}

#[test]
fn explicit_mode_drops_the_singleton_transposes_of_select_and_sum() {
    // G7: the dim-order serializer's TRANSPOSEs here only moved the removed size-1 dim;
    // with no peephole to relabel them as RESHAPEs, none may be emitted.
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [2, 3, 4], INT32);
    let select = call(
        &mut ir,
        "select",
        "aten.select_copy.int",
        vec![IrArg::Node(x), IrArg::Int(0), IrArg::Int(1)],
        [3, 4],
        INT32,
    );
    let sum = call(
        &mut ir,
        "sum",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(x), list(&[0]), IrArg::Bool(false)],
        [3, 4],
        INT32,
    );
    let out = output(&mut ir, select);
    ir.set_args(
        out,
        vec![IrArg::List(vec![IrArg::Node(select), IrArg::Node(sum)])],
    );
    let explicit = serialize_explicit(&ir);
    assert!(ops_of(&explicit, Op::Transpose).is_empty());
    assert_eq!(tensor(&explicit, "select").shape, [3, 4]);
    assert_eq!(tensor(&explicit, "sum").shape, [3, 4]);
}

#[test]
fn explicit_invariant_check_reports_stale_vals() {
    // D1, run under RETAB_DEBUG.
    let mut graph = Graph::new();
    let x = graph.create_node(
        OpType::Placeholder,
        OpTarget::Other("x".into()),
        smallvec![],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 4, 4, 3], IrDType::Int8)),
        "x".into(),
    );
    graph.create_node(
        OpType::CallFunction,
        OpTarget::Permute,
        smallvec![Arg::Node(x), Arg::IntList(smallvec![0, 2, 3, 1])],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 4, 3, 4], IrDType::Int8)),
        "nhwc".into(),
    );
    let store = TensorStore::new();
    visitors::shape_utils::clear_shape_cache();
    assert_eq!(
        check_explicit_layout_invariants("main", &graph, &store),
        Ok(())
    );

    // IrSizeAdjustInput's slices keep the unsliced val (design F5).
    graph.create_node(
        OpType::CallFunction,
        OpTarget::Slice,
        smallvec![Arg::Node(x), Arg::Int(1), Arg::Int(0), Arg::Int(2)],
        Vec::new(),
        NodeMeta::new(TensorInfo::new(&[1, 4, 4, 3], IrDType::Int8)),
        "slice".into(),
    );
    visitors::shape_utils::clear_shape_cache();
    let error = check_explicit_layout_invariants("main", &graph, &store).unwrap_err();
    assert_eq!(
        error,
        "explicit layout invariant broken in region 'main': 'slice' (Slice) has val \
         [1, 4, 4, 3] but serializes as [1, 2, 4, 3]"
    );
}

// ---------------------------------------------------------------------------
// Review findings
// ---------------------------------------------------------------------------

/// Every TRANSPOSE declares the shape its perms make of its input.
fn assert_transposes_consistent(items: &SerializedGraphItems) {
    for op in ops_of(items, Op::Transpose) {
        let Attribute::Transpose(attribute) = &op.attribute else {
            panic!("expected a TRANSPOSE attribute");
        };
        let input = &tensor(items, &op.inputs[0]).shape;
        let expected: Vec<i32> = attribute.perms.iter().map(|&p| input[p as usize]).collect();
        assert_eq!(
            tensor(items, &op.outputs[0]).shape,
            expected,
            "TRANSPOSE {:?} of {input:?} by {:?}",
            op.outputs,
            attribute.perms
        );
    }
}

/// APU-135 P3: a 16-bit pending table that a layout pass left split (`layout::pending`).
/// rc4's InsertTableOpsPass gives a 16-bit table that needs a left shift two nodes, a
/// TABLE returning int32 and a RESCALE by `2**lshift` (insert_table_ops.py:400-417), and
/// rc4's Down pass can park a rank-changing view between them. RETAB holds the two in the
/// pending table and splits them for the propagation passes; one that stays split has its
/// RESCALE as a node of its own after the view. Ingest must then emit the TABLE alone, or
/// the shift happens twice: once in the RESCALE ingest appends to every 16-bit TABLE and
/// once in the RESCALE node.
#[test]
fn a_split_16_bit_table_serializes_its_table_alone() {
    use crate::fx_ir::EffectfulIrPass;
    use crate::layout::ctx::LayoutCtx;
    use crate::layout::pending::{has_split_rescale, split_table_rescales};

    // x -> sigmoid, a 16-bit table that needs a shift of -7 (input 1/1024, output 1/32768).
    let mut ir = IrGraph::empty();
    let x = placeholder(&mut ir, "x", [2, 3], INT16);
    let sigmoid = call(
        &mut ir,
        "sigmoid",
        "aten.sigmoid.default",
        vec![IrArg::Node(x)],
        [2, 3],
        FLOAT32,
    );
    ir.node_mut(sigmoid)
        .meta
        .set_unary_input_output_qparams(qp(1.0 / 1024.0, 0, INT16), qp(1.0 / 32768.0, 0, INT16));
    output(&mut ir, sigmoid);
    let effects = crate::fx_insert_table_ops_ir::IrInsertTableOpsEffectPass
        .run(&mut ir)
        .effects;

    // The split, then the view the Down pass parks between the TABLE and its RESCALE.
    {
        let mut ctx = LayoutCtx::new(&mut ir);
        assert_eq!(split_table_rescales(&mut ctx), [sigmoid]);
        let rescale = ctx.order().next(sigmoid).expect("the RESCALE follows the table");
        let view = ctx.inserting_after(sigmoid, |ctx| {
            ctx.call_function(
                "sigmoid_view",
                "aten.view_copy.default",
                vec![IrArg::Node(sigmoid), i64_list_arg(&[6])],
                crate::fx_ir::NodeMeta::tensor(vec![6], INT32),
            )
        });
        ctx.replace_input_with(rescale, sigmoid, view);
        ctx.meta_mut(rescale).set_tensor_facts(crate::fx_meta::TensorFacts::new(vec![6], INT16));
        ctx.finish();
    }
    assert!(has_split_rescale(ir.node(sigmoid)));

    let mut store = TensorStore::new();
    let mut graph = crate::ingest::from_ir::graph_from_ir(&ir, &mut store);
    crate::ingest::from_ir::apply_effects(&mut graph, &mut store, &effects);
    crate::ingest::finalize::finalize_converted_graph(&mut graph);
    graph.rebuild_user_lists();
    let items = try_serialize_graph(graph, &store).unwrap();

    let table = single_op(&items, Op::Table);
    let table_out = tensor(&items, &table.outputs[0]);
    assert_eq!((table_out.shape.as_slice(), table_out.dtype), (&[2, 3][..], DType::Int32));
    let reshape = single_op(&items, Op::Reshape);
    assert_eq!(reshape.inputs[0], table.outputs[0]);
    let rescale = single_op(&items, Op::Rescale);
    assert_eq!(rescale.inputs[0], reshape.outputs[0]);
    let out = tensor(&items, &rescale.outputs[0]);
    assert_eq!((out.shape.as_slice(), out.dtype), (&[6][..], DType::Int16));
}
