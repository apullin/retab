use crate::graph::ir::OpTarget;

pub(crate) fn has_bool_output(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Eq
            | OpTarget::Gt
            | OpTarget::Ge
            | OpTarget::Lt
            | OpTarget::Le
            | OpTarget::TosaEqual
            | OpTarget::TosaGreater
            | OpTarget::TosaGreaterEqual
            | OpTarget::LogicalNot
            | OpTarget::LogicalAnd
            | OpTarget::LogicalOr
            | OpTarget::LogicalXor
    )
}

pub(crate) fn is_layout_dtype_passthrough(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Reshape
            | OpTarget::View
            | OpTarget::Permute
            | OpTarget::Flatten
            | OpTarget::Squeeze
            | OpTarget::Unsqueeze
            | OpTarget::Expand
            | OpTarget::Contiguous
            | OpTarget::TosaReshape
            | OpTarget::TosaTranspose
            | OpTarget::TosaIdentity
    )
}

pub(crate) fn is_accumulator_output(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::ConvTranspose2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::Linear
            | OpTarget::Addmm
            | OpTarget::Matmul
            | OpTarget::Bmm
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaMatmul
    )
}

pub(crate) fn is_shape_preserving_unary(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Relu
            | OpTarget::Hardtanh
            | OpTarget::Clamp
            | OpTarget::LeakyRelu
            | OpTarget::Sigmoid
            | OpTarget::Tanh
            | OpTarget::Exp
            | OpTarget::Erf
            | OpTarget::Log
            | OpTarget::Neg
            | OpTarget::Reciprocal
            | OpTarget::Rsqrt
            | OpTarget::Abs
            | OpTarget::Floor
            | OpTarget::Ceil
            | OpTarget::To
            | OpTarget::Sqrt
            | OpTarget::Log1p
            | OpTarget::Expm1
            | OpTarget::Sign
            | OpTarget::Round
            | OpTarget::Sin
            | OpTarget::Cos
            | OpTarget::TosaSigmoid
            | OpTarget::TosaTanh
            | OpTarget::TosaExp
            | OpTarget::TosaLog
            | OpTarget::TosaAbs
            | OpTarget::TosaReciprocal
            | OpTarget::TosaErf
            | OpTarget::TosaCos
            | OpTarget::TosaSin
            | OpTarget::TosaRsqrt
            | OpTarget::TosaIdentity
    )
}

pub(crate) fn is_qdq_shape_passthrough(target: &OpTarget) -> bool {
    match target {
        OpTarget::Other(name) => {
            name.contains("quantize_per_tensor")
                || name.contains("dequantize_per_tensor")
                || name.contains("dequantize_per_channel")
        }
        _ => matches!(
            target,
            OpTarget::TosaRescale
                | OpTarget::QuantizePerTensor
                | OpTarget::DequantizePerTensor
                | OpTarget::DequantizePerChannel
        ),
    }
}

pub(crate) fn is_transpose_shape_op(target: &OpTarget) -> bool {
    matches!(target, OpTarget::TosaTranspose | OpTarget::Permute)
}

pub(crate) fn is_slice_shape_op(target: &OpTarget) -> bool {
    matches!(target, OpTarget::Slice | OpTarget::TosaSlice)
}

pub(crate) fn is_sum_reduction(target: &OpTarget) -> bool {
    matches!(target, OpTarget::Sum | OpTarget::TosaReduceSum)
}

pub(crate) fn is_aten_sum(target: &OpTarget) -> bool {
    matches!(target, OpTarget::Sum)
}

pub(crate) fn is_max_pool(target: &OpTarget) -> bool {
    matches!(target, OpTarget::MaxPool2d | OpTarget::TosaMaxPool2d)
}

pub(crate) fn is_avg_pool(target: &OpTarget) -> bool {
    matches!(target, OpTarget::AvgPool2d | OpTarget::TosaAvgPool2d)
}

/// TOSA-dialect ops whose shape the serializer recomputes (or reorders) in dim-order
/// mode. In explicit layout mode their val is the TOSA shape, so the serializer reads
/// it as-is (P2 design §4.2 G1, critique B1).
///
/// TRANSPOSE is not one of them. No explicit pass creates one (permutes stay
/// `aten.permute_copy`, design §1c), so the only TRANSPOSEs an explicit graph can hold come
/// from dim-order logic (IrToTosaMemoryFormat), whose vals are not TOSA shapes. Recomputing
/// a TRANSPOSE's shape from its input is right for both kinds of val.
pub(crate) fn has_explicit_tosa_shape_val(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaConv3d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaMaxPool2d
            | OpTarget::TosaAvgPool2d
            | OpTarget::TosaResize
    )
}

/// Aten conv, pool and upsample ops. Their visitors build the NHWC view from dim
/// orders, so in explicit layout mode they must have been rewritten to TOSA ops
/// before serialization (P2 design X5, critique B11).
pub(crate) fn is_aten_spatial_layout_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv1d
            | OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::ConvTranspose2d
            | OpTarget::MaxPool2d
            | OpTarget::AvgPool2d
            | OpTarget::AdaptiveAvgPool2d
            | OpTarget::UpsampleNearest2d
            | OpTarget::UpsampleBilinear2d
    )
}

pub(crate) fn uses_nhwc_layout_shape(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::MaxPool2d
            | OpTarget::AvgPool2d
            | OpTarget::TosaMaxPool2d
            | OpTarget::TosaAvgPool2d
    )
}

pub(crate) fn is_shape_literal_reshape(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaReshape | OpTarget::Reshape | OpTarget::View
    )
}

pub(crate) fn is_matmul_shape_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Matmul | OpTarget::Bmm | OpTarget::TosaMatmul
    )
}

pub(crate) fn is_conv_weight_source(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaConv3d
            | OpTarget::TosaDepthwiseConv2d
    )
}

pub(crate) fn is_weight_layout_passthrough(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaTranspose
            | OpTarget::Permute
            | OpTarget::Reshape
            | OpTarget::TosaReshape
            | OpTarget::View
    )
}
