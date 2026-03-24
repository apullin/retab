use crate::graph::ir::OpTarget;

/// Map a Python target string (from `str(node.target)`) to an `OpTarget` variant.
///
/// ATen targets look like `"aten.conv2d.default"`, quantized decomposed targets
/// look like `"quantized_decomposed.quantize_per_tensor.default"`. We match on
/// substrings so we tolerate different overload suffixes.
pub fn parse_op_target(target_str: &str) -> OpTarget {
    // ---------------------------------------------------------------
    // TOSA ops — check first so they don't get caught by ATen matches.
    // FX-direct passes (fx_helpers.py) create nodes with targets like
    // "tosa.RESCALE.default", "tosa.rescale.default", or "tosa::rescale".
    // The ExecuTorch exir_ops variant uses "backend.tosa.RESCALE.default".
    // We match case-insensitively on the substring after "tosa".
    // ---------------------------------------------------------------
    if target_str.contains("tosa") {
        let lower = target_str.to_ascii_lowercase();
        if lower.contains("rescale") {
            return OpTarget::TosaRescale;
        }
        if lower.contains("depthwise_conv2d") {
            return OpTarget::TosaDepthwiseConv2d;
        }
        if lower.contains("transpose_conv2d") || lower.contains("transposeconv2d") {
            return OpTarget::TosaTransposeConv2d;
        }
        if lower.contains("conv2d") {
            return OpTarget::TosaConv2d;
        }
        if lower.contains("conv3d") {
            return OpTarget::TosaConv3d;
        }
        if lower.contains("transpose") {
            return OpTarget::TosaTranspose;
        }
        if lower.contains("max_pool2d") || lower.contains("maxpool2d") {
            return OpTarget::TosaMaxPool2d;
        }
        if lower.contains("avg_pool2d") || lower.contains("avgpool2d") {
            return OpTarget::TosaAvgPool2d;
        }
        if lower.contains("reshape") {
            return OpTarget::TosaReshape;
        }
        if lower.contains("clamp") {
            return OpTarget::TosaClamp;
        }
        if lower.contains("matmul") {
            return OpTarget::TosaMatmul;
        }
        if lower.contains("fully_connected") || lower.contains("fullyconnected") {
            return OpTarget::TosaFullyConnected;
        }
        if lower.contains("reduce_sum") || lower.contains("reducesum") {
            return OpTarget::TosaReduceSum;
        }
        if lower.contains("reduce_max") || lower.contains("reducemax") {
            return OpTarget::TosaReduceMax;
        }
        if lower.contains("reduce_min") || lower.contains("reducemin") {
            return OpTarget::TosaReduceMin;
        }
        if lower.contains("concat") {
            return OpTarget::TosaConcat;
        }
        if lower.contains("slice") {
            return OpTarget::TosaSlice;
        }
        if lower.contains("table") {
            return OpTarget::TosaTable;
        }
        if lower.contains("pad") {
            return OpTarget::TosaPad;
        }
        if lower.contains("gather") {
            return OpTarget::TosaGather;
        }
        if lower.contains("scatter") {
            return OpTarget::TosaScatter;
        }
        if lower.contains("select") {
            return OpTarget::TosaSelect;
        }
        if lower.contains("cast") {
            return OpTarget::TosaCast;
        }
        if lower.contains("equal") {
            return OpTarget::TosaEqual;
        }
        if lower.contains("greater_equal") || lower.contains("greaterequal") {
            return OpTarget::TosaGreaterEqual;
        }
        if lower.contains("greater") {
            return OpTarget::TosaGreater;
        }
        if lower.contains("sigmoid") {
            return OpTarget::TosaSigmoid;
        }
        if lower.contains("tanh") {
            return OpTarget::TosaTanh;
        }
        if lower.contains("exp") {
            return OpTarget::TosaExp;
        }
        if lower.contains("log") {
            return OpTarget::TosaLog;
        }
        if lower.contains("negate") || lower.contains("neg") {
            return OpTarget::TosaNegate;
        }
        if lower.contains("abs") {
            return OpTarget::TosaAbs;
        }
        if lower.contains("ceil") {
            return OpTarget::TosaCeil;
        }
        if lower.contains("floor") {
            return OpTarget::TosaFloor;
        }
        if lower.contains("reciprocal") {
            return OpTarget::TosaReciprocal;
        }
        if lower.contains("erf") {
            return OpTarget::TosaErf;
        }
        if lower.contains("rsqrt") {
            return OpTarget::TosaRsqrt;
        }
        if lower.contains("cos") {
            return OpTarget::TosaCos;
        }
        if lower.contains("sin") {
            return OpTarget::TosaSin;
        }
        if lower.contains("add") {
            return OpTarget::TosaAdd;
        }
        if lower.contains("sub") {
            return OpTarget::TosaSub;
        }
        if lower.contains("mul") {
            return OpTarget::TosaMul;
        }
        if lower.contains("const") {
            return OpTarget::TosaConst;
        }
        // Unknown TOSA op — still tag it as Other but with the tosa prefix
        return OpTarget::Other(target_str.to_string());
    }

    // Quantized decomposed ops — check before aten to avoid false positives
    if target_str.contains("quantized_decomposed.quantize_per_tensor") {
        return OpTarget::QuantizePerTensor;
    }
    if target_str.contains("quantized_decomposed.dequantize_per_tensor") {
        return OpTarget::DequantizePerTensor;
    }
    if target_str.contains("quantized_decomposed.dequantize_per_channel") {
        return OpTarget::DequantizePerChannel;
    }

    // ATen ops — order matters for ops whose names are substrings of others
    // (e.g. "conv2d" vs "conv_transpose2d")
    if target_str.contains("aten.conv_transpose2d") {
        return OpTarget::ConvTranspose2d;
    }
    if target_str.contains("aten.conv2d") || target_str.contains("aten.convolution") {
        return OpTarget::Conv2d;
    }
    if target_str.contains("aten.relu") {
        return OpTarget::Relu;
    }
    if target_str.contains("aten.add.") {
        return OpTarget::Add;
    }
    if target_str.contains("aten.sub.") {
        return OpTarget::Sub;
    }
    if target_str.contains("aten.mul.") {
        return OpTarget::Mul;
    }
    if target_str.contains("aten.div.") {
        return OpTarget::Div;
    }
    if target_str.contains("aten.permute") {
        return OpTarget::Permute;
    }
    if target_str.contains("aten.reshape") {
        return OpTarget::Reshape;
    }
    if target_str.contains("aten.full.") {
        return OpTarget::Full;
    }
    if target_str.contains("aten.avg_pool2d") {
        return OpTarget::AvgPool2d;
    }
    if target_str.contains("aten.max_pool2d") {
        return OpTarget::MaxPool2d;
    }
    if target_str.contains("aten.sum.") {
        return OpTarget::Sum;
    }
    if target_str.contains("aten.rsqrt") {
        return OpTarget::Rsqrt;
    }
    if target_str.contains("aten.layer_norm") {
        return OpTarget::LayerNorm;
    }
    if target_str.contains("aten.batch_norm") || target_str.contains("aten._native_batch_norm") {
        return OpTarget::BatchNorm;
    }
    if target_str.contains("aten.linear") {
        return OpTarget::Linear;
    }
    if target_str.contains("aten.matmul") || target_str.contains("aten.mm.") {
        return OpTarget::Matmul;
    }
    if target_str.contains("aten.bmm") {
        return OpTarget::Bmm;
    }
    if target_str.contains("aten.addmm") {
        return OpTarget::Addmm;
    }
    if target_str.contains("aten.softmax") || target_str.contains("aten._softmax") {
        return OpTarget::Softmax;
    }
    if target_str.contains("aten.gelu") {
        return OpTarget::Gelu;
    }
    if target_str.contains("aten.sigmoid") {
        return OpTarget::Sigmoid;
    }
    if target_str.contains("aten.tanh") {
        return OpTarget::Tanh;
    }
    if target_str.contains("aten.clamp") {
        return OpTarget::Clamp;
    }
    if target_str.contains("aten.hardtanh") {
        return OpTarget::Hardtanh;
    }
    if target_str.contains("aten.mean") {
        return OpTarget::Mean;
    }
    if target_str.contains("aten.cat") {
        return OpTarget::Cat;
    }
    if target_str.contains("aten.slice_copy") || target_str.contains("aten.slice") {
        return OpTarget::Slice;
    }
    if target_str.contains("aten.select_copy")
        || target_str.contains("aten.select")
        || target_str.contains("aten.index_select")
    {
        return OpTarget::Select;
    }
    if target_str.contains("aten.unsqueeze") {
        return OpTarget::Unsqueeze;
    }
    if target_str.contains("aten.squeeze") {
        return OpTarget::Squeeze;
    }
    if target_str.contains("aten.flatten") {
        return OpTarget::Flatten;
    }
    if target_str.contains("aten.constant_pad_nd") || target_str.contains("aten.pad") {
        return OpTarget::Pad;
    }
    if target_str.contains("aten.embedding") {
        return OpTarget::Embedding;
    }
    if target_str.contains("aten.exp.") || target_str.contains("aten.exp2") {
        return OpTarget::Exp;
    }
    if target_str.contains("aten.log.") || target_str.contains("aten.log2") {
        return OpTarget::Log;
    }
    if target_str.contains("aten.neg.") {
        return OpTarget::Neg;
    }
    if target_str.contains("aten.reciprocal") {
        return OpTarget::Reciprocal;
    }
    if target_str.contains("aten.abs.") {
        return OpTarget::Abs;
    }
    if target_str.contains("aten.pow.") {
        return OpTarget::Pow;
    }
    if target_str.contains("aten.floor.") {
        return OpTarget::Floor;
    }
    if target_str.contains("aten.ceil.") {
        return OpTarget::Ceil;
    }
    if target_str.contains("aten.expand") {
        return OpTarget::Expand;
    }
    if target_str.contains("aten.contiguous") {
        return OpTarget::Contiguous;
    }
    if target_str.contains("aten.view") || target_str.contains("aten._unsafe_view") {
        return OpTarget::View;
    }
    if target_str.contains("aten.to.")
        || target_str.contains("aten._to_copy")
        || target_str.contains("aten.clone")
        || target_str.contains("dim_order_ops._to_dim_order_copy")
        || target_str.contains("dim_order_ops._clone_dim_order")
    {
        return OpTarget::To;
    }
    if target_str.contains("aten.group_norm") {
        return OpTarget::GroupNorm;
    }
    if target_str.contains("aten.amax") {
        return OpTarget::Amax;
    }
    if target_str.contains("aten.amin") {
        return OpTarget::Amin;
    }
    if target_str.contains("aten.var.") {
        return OpTarget::Var;
    }
    if target_str.contains("aten.full_like") {
        return OpTarget::FullLike;
    }
    if target_str.contains("aten.where") {
        return OpTarget::Where;
    }
    if target_str.contains("aten.floor_divide") {
        return OpTarget::FloorDivide;
    }
    if target_str.contains("aten.gather") {
        return OpTarget::Other("aten.gather".into());
    }
    if target_str.contains("aten.arange") {
        return OpTarget::Arange;
    }
    if target_str.contains("aten.leaky_relu") {
        return OpTarget::LeakyRelu;
    }
    if target_str.contains("aten.sqrt") {
        return OpTarget::Sqrt;
    }
    if target_str.contains("aten.log1p") {
        return OpTarget::Log1p;
    }
    if target_str.contains("aten.expm1") {
        return OpTarget::Expm1;
    }
    if target_str.contains("aten.remainder") {
        return OpTarget::Remainder;
    }
    if target_str.contains("aten.sign") {
        return OpTarget::Sign;
    }
    if target_str.contains("aten.round") {
        return OpTarget::Round;
    }

    // Bitwise ops
    if target_str.contains("aten.bitwise_not") {
        return OpTarget::BitwiseNot;
    }
    if target_str.contains("aten.bitwise_and") {
        return OpTarget::BitwiseAnd;
    }
    if target_str.contains("aten.bitwise_or") {
        return OpTarget::BitwiseOr;
    }
    if target_str.contains("aten.bitwise_xor") {
        return OpTarget::BitwiseXor;
    }
    if target_str.contains("aten.bitwise_left_shift") {
        return OpTarget::BitwiseLeftShift;
    }
    if target_str.contains("aten.bitwise_right_shift") {
        return OpTarget::BitwiseRightShift;
    }

    // Logical ops
    if target_str.contains("aten.logical_not") {
        return OpTarget::LogicalNot;
    }
    if target_str.contains("aten.logical_and") {
        return OpTarget::LogicalAnd;
    }
    if target_str.contains("aten.logical_or") {
        return OpTarget::LogicalOr;
    }

    // Comparison ops — order matters: gt before ge to avoid aten.gt matching aten.ge
    if target_str.contains("aten.gt.") || target_str.contains("aten.gt_") {
        return OpTarget::Gt;
    }
    if target_str.contains("aten.ge.") || target_str.contains("aten.ge_") {
        return OpTarget::Ge;
    }
    if target_str.contains("aten.lt.") || target_str.contains("aten.lt_") {
        return OpTarget::Lt;
    }
    if target_str.contains("aten.le.") || target_str.contains("aten.le_") {
        return OpTarget::Le;
    }
    if target_str.contains("aten.eq.") || target_str.contains("aten.eq_") {
        return OpTarget::Eq;
    }

    // Trig / math ops
    if target_str.contains("aten.sin.") {
        return OpTarget::Sin;
    }
    if target_str.contains("aten.cos.") {
        return OpTarget::Cos;
    }
    if target_str.contains("aten.tan.") {
        return OpTarget::Tan;
    }
    if target_str.contains("aten.asin.") {
        return OpTarget::Asin;
    }
    if target_str.contains("aten.acos.") {
        return OpTarget::Acos;
    }
    if target_str.contains("aten.atan.") {
        return OpTarget::Atan;
    }
    if target_str.contains("aten.sinh.") {
        return OpTarget::Sinh;
    }
    if target_str.contains("aten.cosh.") {
        return OpTarget::Cosh;
    }
    if target_str.contains("aten.asinh.") {
        return OpTarget::Asinh;
    }
    if target_str.contains("aten.acosh.") {
        return OpTarget::Acosh;
    }
    if target_str.contains("aten.elu.") || target_str.contains("aten.elu_") {
        return OpTarget::Elu;
    }
    if target_str.contains("aten.hardswish") {
        return OpTarget::Hardswish;
    }
    if target_str.contains("aten.silu") {
        return OpTarget::Silu;
    }

    // Additional ops
    if target_str.contains("aten._adaptive_avg_pool2d") || target_str.contains("aten.adaptive_avg_pool2d") {
        return OpTarget::AdaptiveAvgPool2d;
    }
    if target_str.contains("aten.upsample_nearest2d") {
        return OpTarget::UpsampleNearest2d;
    }
    if target_str.contains("aten.conv1d") {
        return OpTarget::Conv1d;
    }
    if target_str.contains("aten.scalar_tensor") {
        return OpTarget::ScalarTensor;
    }
    if target_str.contains("aten.zeros.") {
        return OpTarget::Zeros;
    }
    if target_str.contains("aten.repeat.") {
        return OpTarget::Repeat;
    }
    if target_str.contains("aten.cumsum") {
        return OpTarget::Cumsum;
    }
    if target_str.contains("aten.erfinv") {
        return OpTarget::Erfinv;
    }
    if target_str.contains("aten.unfold_copy") || target_str.contains("aten.unfold.") {
        return OpTarget::UnfoldCopy;
    }
    if target_str.contains("aten.split_with_sizes_copy") || target_str.contains("aten.split_with_sizes.") {
        return OpTarget::SplitWithSizesCopy;
    }
    if target_str.contains("aten.index.Tensor") {
        return OpTarget::IndexTensor;
    }
    if target_str.contains("aten.any.") || target_str.contains("aten.any_") {
        return OpTarget::Any;
    }

    OpTarget::Other(target_str.to_string())
}
