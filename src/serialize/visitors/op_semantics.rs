use crate::graph::ir::OpTarget;

pub(super) fn has_bool_output(target: &OpTarget) -> bool {
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

pub(super) fn is_layout_dtype_passthrough(target: &OpTarget) -> bool {
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

pub(super) fn is_accumulator_output(target: &OpTarget) -> bool {
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

pub(super) fn is_shape_preserving_unary(target: &OpTarget) -> bool {
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
