// Per-op TOSA emission visitors.
//
// Each function takes the graph node, its metadata, and the tensor store, and
// returns the TOSA operator(s) + tensor(s) that should be added to the basic
// block being built.

mod arg_utils;
mod arithmetic;
mod broadcast;
mod const_utils;
mod control_flow;
mod dtype_quant;
mod elementwise;
mod emit_result;
mod graph_utils;
mod indexing;
mod layout;
mod math;
mod matmul;
mod nn;
mod reductions;
mod shape_utils;
mod tensor_ops;

use super::tosa_generated::{
    AbsAttribute, AddAttribute, ArithmeticRightShiftAttribute, Attribute, AvgPool2dAttribute,
    BitwiseAndAttribute, BitwiseNotAttribute, BitwiseOrAttribute, BitwiseXorAttribute,
    CastAttribute, CeilAttribute, ClampAttribute, ConcatAttribute, ConstAttribute,
    ConstShapeAttribute, Conv2dAttribute, Conv3dAttribute, CosAttribute, DType,
    DepthwiseConv2dAttribute, EqualAttribute, ErfAttribute, ExpAttribute, FloorAttribute,
    GatherAttribute, GreaterAttribute, GreaterEqualAttribute, LogAttribute, LogicalAndAttribute,
    LogicalLeftShiftAttribute, LogicalNotAttribute, LogicalOrAttribute, LogicalXorAttribute,
    MatMulAttribute, MaxPool2dAttribute, MaximumAttribute, MinimumAttribute, MulAttribute,
    NanPropagationMode, NegateAttribute, Op, PadAttribute, PowAttribute, ReciprocalAttribute,
    ReduceAnyAttribute, ReduceMaxAttribute, ReduceMinAttribute, ReduceSumAttribute,
    RescaleAttribute, ReshapeAttribute, ResizeAttribute, ResizeMode, RoundingMode, RsqrtAttribute,
    ScatterAttribute, SelectAttribute, SigmoidAttribute, SinAttribute, SliceAttribute,
    SubAttribute, TableAttribute, TanhAttribute, TosaOperator, TosaShape, TosaTensor,
    TransposeAttribute, TransposeConv2dAttribute,
};
use crate::graph::ir::{Arg, Node, NodeId, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::TensorStore;
use arg_utils::*;
use arithmetic::{
    emit_add, emit_aten_add, emit_aten_clamp, emit_aten_mul, emit_aten_squeeze, emit_aten_sub,
    emit_aten_unsqueeze, emit_clamp, emit_mul, emit_sub,
};
use broadcast::*;
use const_utils::*;
use control_flow::emit_cond_if;
pub(crate) use dtype_quant::ir_dtype_to_tosa;
use dtype_quant::{
    accumulator_output_dtype, actual_tensor_dtype, clamp_bound_bytes_from_arg,
    compute_multiplier_and_shift_with_width, conv_accumulator_dtype, encode_zp_for_dtype,
    has_only_tosa_rescale_users, matmul_input_dtype, node_input_zero_point, node_output_dtype,
    node_output_zero_point, node_tensor_dtype, normalize_rescale_zp, passthrough_input_dtype,
};
use elementwise::{
    emit_abs, emit_arithmetic_right_shift, emit_aten_eq, emit_aten_ge, emit_aten_gt, emit_aten_le,
    emit_aten_lt, emit_bitwise_and, emit_bitwise_not, emit_bitwise_or, emit_bitwise_xor, emit_cast,
    emit_cos, emit_equal, emit_erf, emit_exp, emit_greater, emit_greater_equal, emit_log,
    emit_logical_and, emit_logical_left_shift, emit_logical_not, emit_logical_or, emit_logical_xor,
    emit_maximum, emit_minimum, emit_pow, emit_reciprocal, emit_rsqrt, emit_sigmoid, emit_sin,
    emit_table, emit_tanh,
};
use emit_result::EmitResult;
use graph_utils::*;
use indexing::{
    emit_gather, emit_index_select, emit_scatter, emit_select, emit_select_copy, emit_slice,
};
use layout::*;
use math::{
    emit_aten_div, emit_aten_expm1, emit_aten_floor_divide, emit_aten_leaky_relu, emit_aten_log1p,
    emit_aten_sign, emit_aten_sqrt, emit_ceil, emit_floor, emit_full_like, emit_negate, emit_round,
};
use matmul::emit_matmul;
use nn::{
    emit_aten_conv2d, emit_aten_max_pool2d, emit_avg_pool2d, emit_conv2d, emit_conv3d,
    emit_depthwise_conv2d, emit_max_pool2d, emit_pad, emit_rescale, emit_transpose_conv2d,
};
use reductions::{
    emit_aten_amax, emit_aten_amin, emit_aten_any, emit_aten_mean, emit_aten_sum, emit_reduce_sum,
};
use shape_utils::*;
use tensor_ops::{
    emit_aten_permute, emit_aten_reshape, emit_concat, emit_const, emit_expand, emit_full,
    emit_repeat, emit_reshape, emit_resize, emit_scalar_tensor, emit_transpose,
};

/// Dispatch to the correct per-op visitor based on the node's target.
pub fn emit_node(
    graph: &Graph,
    node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> Option<EmitResult> {
    match &node.target {
        OpTarget::TosaConst => Some(emit_const(graph, node_id, node, store)),
        OpTarget::Full => Some(emit_full(graph, node_id, node, store)),
        OpTarget::ScalarTensor => Some(emit_scalar_tensor(graph, node_id, node, store)),
        OpTarget::TosaConv2d => Some(emit_conv2d(graph, node_id, node, store)),
        OpTarget::TosaConv3d => Some(emit_conv3d(graph, node_id, node, store)),
        OpTarget::TosaDepthwiseConv2d => Some(emit_depthwise_conv2d(graph, node_id, node, store)),
        OpTarget::TosaRescale => Some(emit_rescale(graph, node_id, node, store)),
        OpTarget::TosaResize => Some(emit_resize(graph, node_id, node, store)),
        OpTarget::TosaMaxPool2d => Some(emit_max_pool2d(graph, node_id, node, store)),
        OpTarget::TosaTranspose => Some(emit_transpose(graph, node_id, node, store)),
        OpTarget::TosaReshape => Some(emit_reshape(graph, node_id, node, store)),
        OpTarget::TosaClamp => Some(emit_clamp(graph, node_id, node, store)),
        OpTarget::TosaAdd => Some(emit_add(graph, node_id, node, store)),
        OpTarget::TosaSub => Some(emit_sub(graph, node_id, node, store)),
        OpTarget::TosaMul => Some(emit_mul(graph, node_id, node, store)),
        OpTarget::TosaAvgPool2d => Some(emit_avg_pool2d(graph, node_id, node, store)),
        OpTarget::TosaReduceSum => Some(emit_reduce_sum(graph, node_id, node, store)),
        OpTarget::Any => Some(emit_aten_any(graph, node_id, node, store)),
        OpTarget::Amax => Some(emit_aten_amax(graph, node_id, node, store)),
        OpTarget::Amin => Some(emit_aten_amin(graph, node_id, node, store)),
        OpTarget::TosaTable => Some(emit_table(graph, node_id, node, store)),
        OpTarget::TosaSigmoid => Some(emit_sigmoid(graph, node_id, node, store)),
        OpTarget::TosaTanh => Some(emit_tanh(graph, node_id, node, store)),
        OpTarget::TosaExp => Some(emit_exp(graph, node_id, node, store)),
        OpTarget::TosaLog => Some(emit_log(graph, node_id, node, store)),
        OpTarget::TosaNeg | OpTarget::TosaNegate => Some(emit_negate(graph, node_id, node, store)),
        OpTarget::TosaAbs => Some(emit_abs(graph, node_id, node, store)),
        OpTarget::TosaCeil => Some(emit_ceil(graph, node_id, node, store)),
        OpTarget::TosaFloor => Some(emit_floor(graph, node_id, node, store)),
        OpTarget::TosaReciprocal => Some(emit_reciprocal(graph, node_id, node, store)),
        OpTarget::TosaErf => Some(emit_erf(graph, node_id, node, store)),
        OpTarget::TosaCos => Some(emit_cos(graph, node_id, node, store)),
        OpTarget::TosaSin => Some(emit_sin(graph, node_id, node, store)),
        OpTarget::TosaRsqrt => Some(emit_rsqrt(graph, node_id, node, store)),
        OpTarget::TosaMatmul => Some(emit_matmul(graph, node_id, node, store)),
        OpTarget::TosaConcat => Some(emit_concat(graph, node_id, node, store)),
        OpTarget::TosaPad => Some(emit_pad(graph, node_id, node, store)),
        OpTarget::TosaSlice => Some(emit_slice(graph, node_id, node, store)),
        OpTarget::TosaGather => Some(emit_gather(graph, node_id, node, store)),
        OpTarget::TosaScatter => Some(emit_scatter(graph, node_id, node, store)),
        OpTarget::TosaSelect => Some(emit_select(graph, node_id, node, store)),
        OpTarget::TosaCast => Some(emit_cast(graph, node_id, node, store)),
        OpTarget::TosaEqual => Some(emit_equal(graph, node_id, node, store)),
        OpTarget::TosaGreater => Some(emit_greater(graph, node_id, node, store)),
        OpTarget::TosaGreaterEqual => Some(emit_greater_equal(graph, node_id, node, store)),
        OpTarget::TosaTransposeConv2d => Some(emit_transpose_conv2d(graph, node_id, node, store)),
        OpTarget::CondIf => Some(emit_cond_if(graph, node_id, node, store)),

        // ATen fallbacks: handle ops that FX-direct passes didn't rewrite
        // to TOSA form. This allows lower_to_tosa_v2 (hybrid pipeline) to
        // produce valid TOSA output even when not all passes run.
        OpTarget::Conv2d | OpTarget::DepthwiseConv2d => {
            Some(emit_aten_conv2d(graph, node_id, node, store))
        }
        OpTarget::MaxPool2d => Some(emit_aten_max_pool2d(graph, node_id, node, store)),
        OpTarget::Clamp | OpTarget::Hardtanh | OpTarget::Relu => {
            Some(emit_aten_clamp(graph, node_id, node, store))
        }
        OpTarget::Expand => Some(emit_expand(graph, node_id, node, store)),
        OpTarget::Repeat => Some(emit_repeat(graph, node_id, node, store)),
        OpTarget::Reshape | OpTarget::Flatten | OpTarget::View => {
            Some(emit_aten_reshape(graph, node_id, node, store))
        }
        OpTarget::Permute => Some(emit_aten_permute(graph, node_id, node, store)),
        OpTarget::To => Some(emit_cast(graph, node_id, node, store)),
        OpTarget::Add => Some(emit_aten_add(graph, node_id, node, store)),
        OpTarget::Sub => Some(emit_aten_sub(graph, node_id, node, store)),
        OpTarget::Mul => Some(emit_aten_mul(graph, node_id, node, store)),
        OpTarget::AvgPool2d => Some(emit_avg_pool2d(graph, node_id, node, store)),
        OpTarget::UpsampleNearest2d | OpTarget::UpsampleBilinear2d => {
            Some(emit_resize(graph, node_id, node, store))
        }
        OpTarget::Sigmoid => Some(emit_sigmoid(graph, node_id, node, store)),
        OpTarget::Tanh => Some(emit_tanh(graph, node_id, node, store)),
        OpTarget::Exp => Some(emit_exp(graph, node_id, node, store)),
        OpTarget::Log => Some(emit_log(graph, node_id, node, store)),
        OpTarget::Neg => Some(emit_negate(graph, node_id, node, store)),
        OpTarget::Abs => Some(emit_abs(graph, node_id, node, store)),
        OpTarget::Ceil => Some(emit_ceil(graph, node_id, node, store)),
        OpTarget::Floor => Some(emit_floor(graph, node_id, node, store)),
        OpTarget::Round => Some(emit_round(graph, node_id, node, store)),
        OpTarget::Reciprocal => Some(emit_reciprocal(graph, node_id, node, store)),
        OpTarget::Rsqrt => Some(emit_rsqrt(graph, node_id, node, store)),
        OpTarget::Matmul | OpTarget::Bmm => Some(emit_matmul(graph, node_id, node, store)),
        OpTarget::Cat => Some(emit_concat(graph, node_id, node, store)),
        OpTarget::Pad => Some(emit_pad(graph, node_id, node, store)),
        OpTarget::Slice => Some(emit_slice(graph, node_id, node, store)),
        OpTarget::Select => Some(emit_select(graph, node_id, node, store)),
        OpTarget::SelectCopy => Some(emit_select_copy(graph, node_id, node, store)),
        OpTarget::IndexSelect => Some(emit_index_select(graph, node_id, node, store)),
        OpTarget::Where => Some(emit_select(graph, node_id, node, store)),
        OpTarget::Sum => Some(emit_aten_sum(graph, node_id, node, store)),
        OpTarget::ConvTranspose2d => Some(emit_transpose_conv2d(graph, node_id, node, store)),

        OpTarget::BitwiseNot => Some(emit_bitwise_not(graph, node_id, node, store)),
        OpTarget::BitwiseAnd => Some(emit_bitwise_and(graph, node_id, node, store)),
        OpTarget::BitwiseOr => Some(emit_bitwise_or(graph, node_id, node, store)),
        OpTarget::BitwiseXor => Some(emit_bitwise_xor(graph, node_id, node, store)),
        OpTarget::BitwiseLeftShift => Some(emit_logical_left_shift(graph, node_id, node, store)),
        OpTarget::BitwiseRightShift => {
            Some(emit_arithmetic_right_shift(graph, node_id, node, store))
        }

        OpTarget::LogicalNot => Some(emit_logical_not(graph, node_id, node, store)),
        OpTarget::LogicalAnd => Some(emit_logical_and(graph, node_id, node, store)),
        OpTarget::LogicalOr => Some(emit_logical_or(graph, node_id, node, store)),
        OpTarget::LogicalXor => Some(emit_logical_xor(graph, node_id, node, store)),

        OpTarget::Gt => Some(emit_aten_gt(graph, node_id, node, store)),
        OpTarget::Ge => Some(emit_aten_ge(graph, node_id, node, store)),
        OpTarget::Lt => Some(emit_aten_lt(graph, node_id, node, store)),
        OpTarget::Le => Some(emit_aten_le(graph, node_id, node, store)),
        OpTarget::Eq => Some(emit_aten_eq(graph, node_id, node, store)),

        OpTarget::Sin => Some(emit_sin(graph, node_id, node, store)),
        OpTarget::Cos => Some(emit_cos(graph, node_id, node, store)),

        OpTarget::Mean => Some(emit_aten_mean(graph, node_id, node, store)),
        OpTarget::Unsqueeze => Some(emit_aten_unsqueeze(graph, node_id, node, store)),
        OpTarget::Squeeze => Some(emit_aten_squeeze(graph, node_id, node, store)),
        OpTarget::Div => Some(emit_aten_div(graph, node_id, node, store)),
        OpTarget::Sqrt => Some(emit_aten_sqrt(graph, node_id, node, store)),
        OpTarget::Pow => Some(emit_pow(graph, node_id, node, store)),
        OpTarget::FullLike => Some(emit_full_like(graph, node_id, node, store)),
        OpTarget::Sign => Some(emit_aten_sign(graph, node_id, node, store)),
        OpTarget::Expm1 => Some(emit_aten_expm1(graph, node_id, node, store)),
        OpTarget::Log1p => Some(emit_aten_log1p(graph, node_id, node, store)),
        OpTarget::FloorDivide => Some(emit_aten_floor_divide(graph, node_id, node, store)),
        OpTarget::LeakyRelu => Some(emit_aten_leaky_relu(graph, node_id, node, store)),

        OpTarget::TosaMaximum => Some(emit_maximum(graph, node_id, node, store)),
        OpTarget::TosaMinimum => Some(emit_minimum(graph, node_id, node, store)),
        OpTarget::TosaReduceMax => Some(emit_aten_amax(graph, node_id, node, store)),
        OpTarget::TosaReduceMin => Some(emit_aten_amin(graph, node_id, node, store)),
        OpTarget::TosaReduceAny => Some(emit_aten_any(graph, node_id, node, store)),

        OpTarget::Zeros => Some(emit_full(graph, node_id, node, store)),

        OpTarget::Other(target) if target.contains("getitem") => Some(EmitResult::default()),

        _ => {
            if std::env::var("TOSA_ENGINE_DEBUG").is_ok() {
                eprintln!(
                    "[emit_node] SKIPPED unhandled target: {:?} (node '{}')",
                    node.target, node.name
                );
            }
            None
        }
    }
}
