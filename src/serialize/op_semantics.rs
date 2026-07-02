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
            | OpTarget::TosaFullyConnected
            | OpTarget::TosaMatmul
    )
}

pub(crate) fn is_shape_preserving_unary(target: &OpTarget) -> bool {
    if matches!(
        target,
        OpTarget::Other(name) if name.contains("logical_not") || name.contains("bitwise_not")
    ) {
        return true;
    }

    matches!(
        target,
        OpTarget::Relu
            | OpTarget::Hardtanh
            | OpTarget::Clamp
            | OpTarget::LeakyRelu
            | OpTarget::Sigmoid
            | OpTarget::Tanh
            | OpTarget::Exp
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
            | OpTarget::LogicalNot
            | OpTarget::BitwiseNot
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

pub(crate) fn is_tosa_transpose(target: &OpTarget) -> bool {
    matches!(target, OpTarget::TosaTranspose)
}

pub(crate) fn is_transpose_shape_op(target: &OpTarget) -> bool {
    matches!(target, OpTarget::TosaTranspose | OpTarget::Permute)
}

pub(crate) fn is_slice_shape_op(target: &OpTarget) -> bool {
    matches!(target, OpTarget::Slice | OpTarget::TosaSlice)
}

pub(crate) fn is_axis_reduction(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Sum
            | OpTarget::Mean
            | OpTarget::Amax
            | OpTarget::Amin
            | OpTarget::Any
            | OpTarget::TosaReduceSum
            | OpTarget::TosaReduceMax
            | OpTarget::TosaReduceMin
            | OpTarget::TosaReduceAny
    )
}

pub(crate) fn is_binary_broadcast(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Add
            | OpTarget::Sub
            | OpTarget::Mul
            | OpTarget::Div
            | OpTarget::Remainder
            | OpTarget::Pow
            | OpTarget::BitwiseAnd
            | OpTarget::BitwiseOr
            | OpTarget::BitwiseXor
            | OpTarget::BitwiseLeftShift
            | OpTarget::BitwiseRightShift
            | OpTarget::Eq
            | OpTarget::Gt
            | OpTarget::Ge
            | OpTarget::Lt
            | OpTarget::Le
            | OpTarget::LogicalAnd
            | OpTarget::LogicalOr
            | OpTarget::LogicalXor
            | OpTarget::TosaAdd
            | OpTarget::TosaSub
            | OpTarget::TosaMul
            | OpTarget::TosaMaximum
            | OpTarget::TosaMinimum
            | OpTarget::TosaEqual
            | OpTarget::TosaGreater
            | OpTarget::TosaGreaterEqual
    )
}

pub(crate) fn is_max_pool(target: &OpTarget) -> bool {
    matches!(target, OpTarget::MaxPool2d | OpTarget::TosaMaxPool2d)
}

pub(crate) fn is_avg_pool(target: &OpTarget) -> bool {
    matches!(target, OpTarget::AvgPool2d | OpTarget::TosaAvgPool2d)
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
            | OpTarget::UpsampleNearest2d
            | OpTarget::UpsampleBilinear2d
            | OpTarget::TosaResize
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_logical_not_is_shape_preserving() {
        assert!(is_shape_preserving_unary(&OpTarget::Other(
            "<EdgeOpOverload: aten.logical_not.default>".to_string(),
        )));
    }
}
