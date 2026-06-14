use crate::graph::ir::OpTarget;

type TargetRule = (&'static [&'static str], OpTarget);

const CONTROL_FLOW_RULES: &[(&str, OpTarget)] = &[
    ("cond", OpTarget::CondIf),
    ("higher_order.cond", OpTarget::CondIf),
    ("torch.ops.higher_order.cond", OpTarget::CondIf),
    ("while_loop", OpTarget::WhileLoop),
    ("higher_order.while_loop", OpTarget::WhileLoop),
    ("torch.ops.higher_order.while_loop", OpTarget::WhileLoop),
];

const TOSA_RULES: &[TargetRule] = &[
    (&["rescale"], OpTarget::TosaRescale),
    (&["resize"], OpTarget::TosaResize),
    (&["depthwise_conv2d"], OpTarget::TosaDepthwiseConv2d),
    (
        &["transpose_conv2d", "transposeconv2d"],
        OpTarget::TosaTransposeConv2d,
    ),
    (&["conv2d"], OpTarget::TosaConv2d),
    (&["conv3d"], OpTarget::TosaConv3d),
    (&["transpose"], OpTarget::TosaTranspose),
    (&["max_pool2d", "maxpool2d"], OpTarget::TosaMaxPool2d),
    (&["avg_pool2d", "avgpool2d"], OpTarget::TosaAvgPool2d),
    (&["reshape"], OpTarget::TosaReshape),
    (&["clamp"], OpTarget::TosaClamp),
    (&["matmul"], OpTarget::TosaMatmul),
    (
        &["fully_connected", "fullyconnected"],
        OpTarget::TosaFullyConnected,
    ),
    (&["reduce_sum", "reducesum"], OpTarget::TosaReduceSum),
    (&["reduce_max", "reducemax"], OpTarget::TosaReduceMax),
    (&["reduce_min", "reducemin"], OpTarget::TosaReduceMin),
    (&["concat"], OpTarget::TosaConcat),
    (&["slice"], OpTarget::TosaSlice),
    (&["table"], OpTarget::TosaTable),
    (&["pad"], OpTarget::TosaPad),
    (&["gather"], OpTarget::TosaGather),
    (&["scatter"], OpTarget::TosaScatter),
    (&["select"], OpTarget::TosaSelect),
    (&["cast"], OpTarget::TosaCast),
    (
        &["greater_equal", "greaterequal"],
        OpTarget::TosaGreaterEqual,
    ),
    (&["greater"], OpTarget::TosaGreater),
    (&["equal"], OpTarget::TosaEqual),
    (&["sigmoid"], OpTarget::TosaSigmoid),
    (&["tanh"], OpTarget::TosaTanh),
    (&["exp"], OpTarget::TosaExp),
    (&["log"], OpTarget::TosaLog),
    (&["negate", "neg"], OpTarget::TosaNegate),
    (&["abs"], OpTarget::TosaAbs),
    (&["ceil"], OpTarget::TosaCeil),
    (&["floor"], OpTarget::TosaFloor),
    (&["reciprocal"], OpTarget::TosaReciprocal),
    (&["erf"], OpTarget::TosaErf),
    (&["rsqrt"], OpTarget::TosaRsqrt),
    (&["cos"], OpTarget::TosaCos),
    (&["sin"], OpTarget::TosaSin),
    (&["add"], OpTarget::TosaAdd),
    (&["sub"], OpTarget::TosaSub),
    (&["mul"], OpTarget::TosaMul),
    (&["const"], OpTarget::TosaConst),
];

const QUANTIZED_DECOMPOSED_RULES: &[TargetRule] = &[
    (
        &["quantized_decomposed.quantize_per_tensor"],
        OpTarget::QuantizePerTensor,
    ),
    (
        &["quantized_decomposed.dequantize_per_tensor"],
        OpTarget::DequantizePerTensor,
    ),
    (
        &["quantized_decomposed.dequantize_per_channel"],
        OpTarget::DequantizePerChannel,
    ),
];

const ATEN_RULES: &[TargetRule] = &[
    (&["aten.conv_transpose2d"], OpTarget::ConvTranspose2d),
    (&["aten.conv2d", "aten.convolution"], OpTarget::Conv2d),
    (&["aten.relu"], OpTarget::Relu),
    (&["aten.add."], OpTarget::Add),
    (&["aten.sub."], OpTarget::Sub),
    (&["aten.mul."], OpTarget::Mul),
    (&["aten.div."], OpTarget::Div),
    (&["aten.permute"], OpTarget::Permute),
    (&["aten.reshape"], OpTarget::Reshape),
    (&["aten.full."], OpTarget::Full),
    (&["aten.avg_pool2d"], OpTarget::AvgPool2d),
    (&["aten.max_pool2d"], OpTarget::MaxPool2d),
    (&["aten.sum."], OpTarget::Sum),
    (&["aten.rsqrt"], OpTarget::Rsqrt),
    (&["aten.layer_norm"], OpTarget::LayerNorm),
    (
        &["aten.batch_norm", "aten._native_batch_norm"],
        OpTarget::BatchNorm,
    ),
    (&["aten.linear"], OpTarget::Linear),
    (&["aten.matmul", "aten.mm."], OpTarget::Matmul),
    (&["aten.bmm"], OpTarget::Bmm),
    (&["aten.addmm"], OpTarget::Addmm),
    (&["aten.softmax", "aten._softmax"], OpTarget::Softmax),
    (&["aten.gelu"], OpTarget::Gelu),
    (&["aten.sigmoid"], OpTarget::Sigmoid),
    (&["aten.tanh"], OpTarget::Tanh),
    (&["aten.maximum"], OpTarget::TosaMaximum),
    (&["aten.minimum"], OpTarget::TosaMinimum),
    (&["aten.clamp"], OpTarget::Clamp),
    (&["aten.hardtanh"], OpTarget::Hardtanh),
    (&["aten.mean"], OpTarget::Mean),
    (&["aten.cat"], OpTarget::Cat),
    (&["aten.slice_copy", "aten.slice"], OpTarget::Slice),
    (&["aten.index_select"], OpTarget::IndexSelect),
    (&["aten.select_copy", "aten.select"], OpTarget::SelectCopy),
    (&["aten.unsqueeze"], OpTarget::Unsqueeze),
    (&["aten.squeeze"], OpTarget::Squeeze),
    (&["aten.flatten"], OpTarget::Flatten),
    (&["aten.constant_pad_nd", "aten.pad"], OpTarget::Pad),
    (&["aten.embedding"], OpTarget::Embedding),
    (&["aten.exp.", "aten.exp2"], OpTarget::Exp),
    (&["aten.log.", "aten.log2"], OpTarget::Log),
    (&["aten.neg."], OpTarget::Neg),
    (&["aten.reciprocal"], OpTarget::Reciprocal),
    (&["aten.abs."], OpTarget::Abs),
    (&["aten.pow."], OpTarget::Pow),
    (&["aten.floor."], OpTarget::Floor),
    (&["aten.ceil."], OpTarget::Ceil),
    (&["aten.expand"], OpTarget::Expand),
    (&["aten.contiguous"], OpTarget::Contiguous),
    (&["aten.view", "aten._unsafe_view"], OpTarget::View),
    (
        &[
            "aten.to.",
            "aten._to_copy",
            "aten.clone",
            "dim_order_ops._to_dim_order_copy",
            "dim_order_ops._clone_dim_order",
        ],
        OpTarget::To,
    ),
    (&["aten.group_norm"], OpTarget::GroupNorm),
    (&["aten.amax"], OpTarget::Amax),
    (&["aten.amin"], OpTarget::Amin),
    (&["aten.var."], OpTarget::Var),
    (&["aten.fill_"], OpTarget::FullLike),
    (&["aten.copy_"], OpTarget::Copy),
    (&["aten.eye"], OpTarget::Eye),
    (&["aten.full_like"], OpTarget::FullLike),
    (&["aten.where"], OpTarget::Where),
    (&["aten.floor_divide"], OpTarget::FloorDivide),
    (&["aten.gather"], OpTarget::TosaGather),
    (&["aten.arange"], OpTarget::Arange),
    (&["aten.leaky_relu"], OpTarget::LeakyRelu),
    (&["aten.sqrt"], OpTarget::Sqrt),
    (&["aten.log1p"], OpTarget::Log1p),
    (&["aten.expm1"], OpTarget::Expm1),
    (&["aten.remainder"], OpTarget::Remainder),
    (&["aten.sign"], OpTarget::Sign),
    (&["aten.round"], OpTarget::Round),
    (&["aten.bitwise_not"], OpTarget::BitwiseNot),
    (&["aten.bitwise_and"], OpTarget::BitwiseAnd),
    (&["aten.bitwise_or"], OpTarget::BitwiseOr),
    (&["aten.bitwise_xor"], OpTarget::BitwiseXor),
    (&["aten.bitwise_left_shift"], OpTarget::BitwiseLeftShift),
    (&["aten.bitwise_right_shift"], OpTarget::BitwiseRightShift),
    (&["aten.logical_not"], OpTarget::LogicalNot),
    (&["aten.logical_and"], OpTarget::LogicalAnd),
    (&["aten.logical_or"], OpTarget::LogicalOr),
    (&["aten.logical_xor"], OpTarget::LogicalXor),
    (&["aten.gt.", "aten.gt_"], OpTarget::Gt),
    (&["aten.ge.", "aten.ge_"], OpTarget::Ge),
    (&["aten.lt.", "aten.lt_"], OpTarget::Lt),
    (&["aten.le.", "aten.le_"], OpTarget::Le),
    (&["aten.eq.", "aten.eq_"], OpTarget::Eq),
    (&["aten.sin."], OpTarget::Sin),
    (&["aten.cos."], OpTarget::Cos),
    (&["aten.tan."], OpTarget::Tan),
    (&["aten.asin."], OpTarget::Asin),
    (&["aten.acos."], OpTarget::Acos),
    (&["aten.atan."], OpTarget::Atan),
    (&["aten.sinh."], OpTarget::Sinh),
    (&["aten.cosh."], OpTarget::Cosh),
    (&["aten.asinh."], OpTarget::Asinh),
    (&["aten.acosh."], OpTarget::Acosh),
    (&["aten.elu.", "aten.elu_"], OpTarget::Elu),
    (&["aten.hardswish"], OpTarget::Hardswish),
    (&["aten.silu"], OpTarget::Silu),
    (
        &["aten._adaptive_avg_pool2d", "aten.adaptive_avg_pool2d"],
        OpTarget::AdaptiveAvgPool2d,
    ),
    (&["aten.upsample_nearest2d"], OpTarget::UpsampleNearest2d),
    (&["aten.upsample_bilinear2d"], OpTarget::UpsampleBilinear2d),
    (&["aten.conv1d"], OpTarget::Conv1d),
    (&["aten.scalar_tensor"], OpTarget::ScalarTensor),
    (&["aten.zeros."], OpTarget::Zeros),
    (&["aten.repeat."], OpTarget::Repeat),
    (&["aten.cumsum"], OpTarget::Cumsum),
    (&["aten.erfinv"], OpTarget::Erfinv),
    (&["aten.unfold_copy", "aten.unfold."], OpTarget::UnfoldCopy),
    (
        &["aten.split_with_sizes_copy", "aten.split_with_sizes."],
        OpTarget::SplitWithSizesCopy,
    ),
    (&["aten.index.Tensor"], OpTarget::IndexTensor),
    (&["aten.any.", "aten.any_"], OpTarget::Any),
];

/// Map a Python target string (from `str(node.target)`) to an `OpTarget` variant.
///
/// ATen targets look like `"aten.conv2d.default"`, quantized decomposed targets
/// look like `"quantized_decomposed.quantize_per_tensor.default"`. We match on
/// substrings so we tolerate different overload suffixes.
pub fn parse_op_target(target_str: &str) -> OpTarget {
    if let Some(target) = exact_match(target_str, CONTROL_FLOW_RULES) {
        return target;
    }

    // TOSA ops must be checked first so they do not get caught by ATen matches.
    if target_str.contains("tosa") {
        return parse_tosa_target(target_str);
    }

    if let Some(target) = match_rules(target_str, QUANTIZED_DECOMPOSED_RULES) {
        return target;
    }

    // slice_scatter is not serializable by the generic slice visitor.
    if target_str.contains("aten.slice_scatter") {
        return OpTarget::Other(target_str.to_string());
    }

    // Keep erf separate from erfinv; both lower through different visitors.
    if target_str.contains("aten.erf.") && !target_str.contains("erfinv") {
        return OpTarget::TosaErf;
    }

    match_rules(target_str, ATEN_RULES).unwrap_or_else(|| OpTarget::Other(target_str.to_string()))
}

fn parse_tosa_target(target_str: &str) -> OpTarget {
    let lower = target_str.to_ascii_lowercase();
    match_rules(&lower, TOSA_RULES).unwrap_or_else(|| OpTarget::Other(target_str.to_string()))
}

fn exact_match(target_str: &str, rules: &[(&str, OpTarget)]) -> Option<OpTarget> {
    rules
        .iter()
        .find_map(|(needle, target)| (*needle == target_str).then(|| target.clone()))
}

fn match_rules(target_str: &str, rules: &[TargetRule]) -> Option<OpTarget> {
    rules
        .iter()
        .find_map(|(needles, target)| contains_any(target_str, needles).then(|| target.clone()))
}

fn contains_any(target_str: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| target_str.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_order_sensitive_tosa_targets() {
        assert_eq!(
            parse_op_target("backend.tosa.GREATER_EQUAL.default"),
            OpTarget::TosaGreaterEqual
        );
        assert_eq!(
            parse_op_target("backend.tosa.TRANSPOSE_CONV2D.default"),
            OpTarget::TosaTransposeConv2d
        );
    }

    #[test]
    fn parses_order_sensitive_aten_targets() {
        assert_eq!(
            parse_op_target("aten.conv_transpose2d.input"),
            OpTarget::ConvTranspose2d
        );
        assert_eq!(
            parse_op_target("aten.slice_scatter.default"),
            OpTarget::Other("aten.slice_scatter.default".to_string())
        );
        assert_eq!(parse_op_target("aten.erf.default"), OpTarget::TosaErf);
        assert_eq!(parse_op_target("aten.erfinv.default"), OpTarget::Erfinv);
    }

    #[test]
    fn parses_control_flow_by_exact_symbol() {
        assert_eq!(parse_op_target("cond"), OpTarget::CondIf);
        assert_eq!(
            parse_op_target("torch.ops.higher_order.while_loop"),
            OpTarget::WhileLoop
        );
        assert_eq!(
            parse_op_target("some.wrapper.cond"),
            OpTarget::Other("some.wrapper.cond".to_string())
        );
    }
}
