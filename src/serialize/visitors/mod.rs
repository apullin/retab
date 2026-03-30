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
mod emitter;
mod graph_utils;
mod indexing;
mod layout;
mod math;
mod matmul;
mod nn;
mod reductions;
mod shape_utils;
mod tensor_ops;

pub use emitter::*;

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

fn build_emitter_table() -> std::collections::HashMap<OpTarget, EmitFn> {
    let mut t: std::collections::HashMap<OpTarget, EmitFn> = std::collections::HashMap::new();

    // TOSA-lowered ops
    t.insert(OpTarget::TosaConst, emit_const);
    t.insert(OpTarget::Full, emit_full);
    t.insert(OpTarget::ScalarTensor, emit_scalar_tensor);
    t.insert(OpTarget::TosaConv2d, emit_conv2d);
    t.insert(OpTarget::TosaConv3d, emit_conv3d);
    t.insert(OpTarget::TosaDepthwiseConv2d, emit_depthwise_conv2d);
    t.insert(OpTarget::TosaRescale, emit_rescale);
    t.insert(OpTarget::TosaResize, emit_resize);
    t.insert(OpTarget::TosaMaxPool2d, emit_max_pool2d);
    t.insert(OpTarget::TosaTranspose, emit_transpose);
    t.insert(OpTarget::TosaReshape, emit_reshape);
    t.insert(OpTarget::TosaClamp, emit_clamp);
    t.insert(OpTarget::TosaAdd, emit_add);
    t.insert(OpTarget::TosaSub, emit_sub);
    t.insert(OpTarget::TosaMul, emit_mul);
    t.insert(OpTarget::TosaAvgPool2d, emit_avg_pool2d);
    t.insert(OpTarget::TosaReduceSum, emit_reduce_sum);
    t.insert(OpTarget::Any, emit_aten_any);
    t.insert(OpTarget::Amax, emit_aten_amax);
    t.insert(OpTarget::Amin, emit_aten_amin);
    t.insert(OpTarget::TosaTable, emit_table);
    t.insert(OpTarget::TosaSigmoid, emit_sigmoid);
    t.insert(OpTarget::TosaTanh, emit_tanh);
    t.insert(OpTarget::TosaExp, emit_exp);
    t.insert(OpTarget::TosaLog, emit_log);
    t.insert(OpTarget::TosaNeg, emit_negate);
    t.insert(OpTarget::TosaNegate, emit_negate);
    t.insert(OpTarget::TosaAbs, emit_abs);
    t.insert(OpTarget::TosaCeil, emit_ceil);
    t.insert(OpTarget::TosaFloor, emit_floor);
    t.insert(OpTarget::TosaReciprocal, emit_reciprocal);
    t.insert(OpTarget::TosaErf, emit_erf);
    t.insert(OpTarget::TosaCos, emit_cos);
    t.insert(OpTarget::TosaSin, emit_sin);
    t.insert(OpTarget::TosaRsqrt, emit_rsqrt);
    t.insert(OpTarget::TosaMatmul, emit_matmul);
    t.insert(OpTarget::TosaConcat, emit_concat);
    t.insert(OpTarget::TosaPad, emit_pad);
    t.insert(OpTarget::TosaSlice, emit_slice);
    t.insert(OpTarget::TosaGather, emit_gather);
    t.insert(OpTarget::TosaScatter, emit_scatter);
    t.insert(OpTarget::TosaSelect, emit_select);
    t.insert(OpTarget::TosaCast, emit_cast);
    t.insert(OpTarget::TosaEqual, emit_equal);
    t.insert(OpTarget::TosaGreater, emit_greater);
    t.insert(OpTarget::TosaGreaterEqual, emit_greater_equal);
    t.insert(OpTarget::TosaTransposeConv2d, emit_transpose_conv2d);
    t.insert(OpTarget::CondIf, emit_cond_if);

    // ATen fallbacks: handle ops that FX-direct passes didn't rewrite
    // to TOSA form.
    t.insert(OpTarget::Conv2d, emit_aten_conv2d);
    t.insert(OpTarget::DepthwiseConv2d, emit_aten_conv2d);
    t.insert(OpTarget::MaxPool2d, emit_aten_max_pool2d);
    t.insert(OpTarget::Clamp, emit_aten_clamp);
    t.insert(OpTarget::Hardtanh, emit_aten_clamp);
    t.insert(OpTarget::Relu, emit_aten_clamp);
    t.insert(OpTarget::Expand, emit_expand);
    t.insert(OpTarget::Repeat, emit_repeat);
    t.insert(OpTarget::Reshape, emit_aten_reshape);
    t.insert(OpTarget::Flatten, emit_aten_reshape);
    t.insert(OpTarget::View, emit_aten_reshape);
    t.insert(OpTarget::Permute, emit_aten_permute);
    t.insert(OpTarget::To, emit_cast);
    t.insert(OpTarget::Add, emit_aten_add);
    t.insert(OpTarget::Sub, emit_aten_sub);
    t.insert(OpTarget::Mul, emit_aten_mul);
    t.insert(OpTarget::AvgPool2d, emit_avg_pool2d);
    t.insert(OpTarget::UpsampleNearest2d, emit_resize);
    t.insert(OpTarget::UpsampleBilinear2d, emit_resize);
    t.insert(OpTarget::Sigmoid, emit_sigmoid);
    t.insert(OpTarget::Tanh, emit_tanh);
    t.insert(OpTarget::Exp, emit_exp);
    t.insert(OpTarget::Log, emit_log);
    t.insert(OpTarget::Neg, emit_negate);
    t.insert(OpTarget::Abs, emit_abs);
    t.insert(OpTarget::Ceil, emit_ceil);
    t.insert(OpTarget::Floor, emit_floor);
    t.insert(OpTarget::Round, emit_round);
    t.insert(OpTarget::Reciprocal, emit_reciprocal);
    t.insert(OpTarget::Rsqrt, emit_rsqrt);
    t.insert(OpTarget::Matmul, emit_matmul);
    t.insert(OpTarget::Bmm, emit_matmul);
    t.insert(OpTarget::Cat, emit_concat);
    t.insert(OpTarget::Pad, emit_pad);
    t.insert(OpTarget::Slice, emit_slice);
    t.insert(OpTarget::Select, emit_select);
    t.insert(OpTarget::SelectCopy, emit_select_copy);
    t.insert(OpTarget::IndexSelect, emit_index_select);
    t.insert(OpTarget::Where, emit_select);
    t.insert(OpTarget::Sum, emit_aten_sum);
    t.insert(OpTarget::ConvTranspose2d, emit_transpose_conv2d);

    t.insert(OpTarget::BitwiseNot, emit_bitwise_not);
    t.insert(OpTarget::BitwiseAnd, emit_bitwise_and);
    t.insert(OpTarget::BitwiseOr, emit_bitwise_or);
    t.insert(OpTarget::BitwiseXor, emit_bitwise_xor);
    t.insert(OpTarget::BitwiseLeftShift, emit_logical_left_shift);
    t.insert(OpTarget::BitwiseRightShift, emit_arithmetic_right_shift);

    t.insert(OpTarget::LogicalNot, emit_logical_not);
    t.insert(OpTarget::LogicalAnd, emit_logical_and);
    t.insert(OpTarget::LogicalOr, emit_logical_or);
    t.insert(OpTarget::LogicalXor, emit_logical_xor);

    t.insert(OpTarget::Gt, emit_aten_gt);
    t.insert(OpTarget::Ge, emit_aten_ge);
    t.insert(OpTarget::Lt, emit_aten_lt);
    t.insert(OpTarget::Le, emit_aten_le);
    t.insert(OpTarget::Eq, emit_aten_eq);

    t.insert(OpTarget::Sin, emit_sin);
    t.insert(OpTarget::Cos, emit_cos);

    t.insert(OpTarget::Mean, emit_aten_mean);
    t.insert(OpTarget::Unsqueeze, emit_aten_unsqueeze);
    t.insert(OpTarget::Squeeze, emit_aten_squeeze);
    t.insert(OpTarget::Div, emit_aten_div);
    t.insert(OpTarget::Sqrt, emit_aten_sqrt);
    t.insert(OpTarget::Pow, emit_pow);
    t.insert(OpTarget::FullLike, emit_full_like);
    t.insert(OpTarget::Sign, emit_aten_sign);
    t.insert(OpTarget::Expm1, emit_aten_expm1);
    t.insert(OpTarget::Log1p, emit_aten_log1p);
    t.insert(OpTarget::FloorDivide, emit_aten_floor_divide);
    t.insert(OpTarget::LeakyRelu, emit_aten_leaky_relu);

    t.insert(OpTarget::TosaMaximum, emit_maximum);
    t.insert(OpTarget::TosaMinimum, emit_minimum);
    t.insert(OpTarget::TosaReduceMax, emit_aten_amax);
    t.insert(OpTarget::TosaReduceMin, emit_aten_amin);
    t.insert(OpTarget::TosaReduceAny, emit_aten_any);

    t.insert(OpTarget::Zeros, emit_full);

    t
}

// Cached emitter dispatch table, built once per thread.
thread_local! {
    static EMITTER_TABLE: std::collections::HashMap<OpTarget, EmitFn> = build_emitter_table();
}

/// Dispatch to the correct per-op visitor based on the node's target.
pub fn emit_node(
    graph: &Graph,
    node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> Option<EmitResult> {
    // Handle the `getitem` special case first (dynamic Other variant).
    if let OpTarget::Other(target) = &node.target {
        if target.contains("getitem") {
            return Some(EmitResult::default());
        }
    }

    EMITTER_TABLE.with(|table| {
        if let Some(emitter) = table.get(&node.target) {
            let ctx = EmitContext {
                graph,
                node_id,
                node,
                store,
            };
            Some(emitter(&ctx))
        } else {
            if std::env::var("TOSA_ENGINE_DEBUG").is_ok() {
                eprintln!(
                    "[emit_node] SKIPPED unhandled target: {:?} (node '{}')",
                    node.target, node.name
                );
            }
            None
        }
    })
}
