use crate::graph::ir::OpTarget;

const CONTROL_FLOW_TARGETS: &[(&str, OpTarget)] = &[
    ("cond", OpTarget::CondIf),
    ("higher_order.cond", OpTarget::CondIf),
    ("torch.ops.higher_order.cond", OpTarget::CondIf),
    ("while_loop", OpTarget::WhileLoop),
    ("higher_order.while_loop", OpTarget::WhileLoop),
    ("torch.ops.higher_order.while_loop", OpTarget::WhileLoop),
];

const TOSA_TARGETS: &[(&str, OpTarget)] = &[
    ("backend.tosa.INTDIV.default", OpTarget::TosaIntDiv),
    ("backend.tosa.RESCALE.default", OpTarget::TosaRescale),
    ("backend.tosa.RESIZE.default", OpTarget::TosaResize),
    (
        "backend.tosa.DEPTHWISE_CONV2D.default",
        OpTarget::TosaDepthwiseConv2d,
    ),
    (
        "backend.tosa.TRANSPOSE_CONV2D.default",
        OpTarget::TosaTransposeConv2d,
    ),
    ("backend.tosa.CONV2D.default", OpTarget::TosaConv2d),
    ("backend.tosa.CONV3D.default", OpTarget::TosaConv3d),
    ("backend.tosa.MAX_POOL2D.default", OpTarget::TosaMaxPool2d),
    ("backend.tosa.AVG_POOL2D.default", OpTarget::TosaAvgPool2d),
    ("backend.tosa.RESHAPE.default", OpTarget::TosaReshape),
    ("backend.tosa.CLAMP.default", OpTarget::TosaClamp),
    ("backend.tosa.MATMUL.default", OpTarget::TosaMatmul),
    ("backend.tosa.REDUCE_SUM.default", OpTarget::TosaReduceSum),
    ("backend.tosa.REDUCE_MAX.default", OpTarget::TosaReduceMax),
    ("backend.tosa.REDUCE_MIN.default", OpTarget::TosaReduceMin),
    ("backend.tosa.CONCAT.default", OpTarget::TosaConcat),
    ("backend.tosa.SLICE.default", OpTarget::TosaSlice),
    ("backend.tosa.TABLE.default", OpTarget::TosaTable),
    ("backend.tosa.PAD.default", OpTarget::TosaPad),
    ("backend.tosa.IDENTITY.default", OpTarget::TosaIdentity),
    ("backend.tosa.CONST_SHAPE.default", OpTarget::TosaConstShape),
    ("backend.tosa.GATHER.default", OpTarget::TosaGather),
    ("backend.tosa.SCATTER.default", OpTarget::TosaScatter),
    ("backend.tosa.SELECT.default", OpTarget::TosaSelect),
    ("backend.tosa.CAST.default", OpTarget::TosaCast),
    (
        "backend.tosa.GREATER_EQUAL.default",
        OpTarget::TosaGreaterEqual,
    ),
    ("backend.tosa.GREATER.default", OpTarget::TosaGreater),
    ("backend.tosa.EQUAL.default", OpTarget::TosaEqual),
    ("backend.tosa.SIGMOID.default", OpTarget::TosaSigmoid),
    ("backend.tosa.TANH.default", OpTarget::TosaTanh),
    ("backend.tosa.EXP.default", OpTarget::TosaExp),
    ("backend.tosa.LOG.default", OpTarget::TosaLog),
    ("backend.tosa.NEGATE.default", OpTarget::TosaNegate),
    ("backend.tosa.ABS.default", OpTarget::TosaAbs),
    ("backend.tosa.CEIL.default", OpTarget::TosaCeil),
    ("backend.tosa.FLOOR.default", OpTarget::TosaFloor),
    ("backend.tosa.RECIPROCAL.default", OpTarget::TosaReciprocal),
    ("backend.tosa.ERF.default", OpTarget::TosaErf),
    ("backend.tosa.RSQRT.default", OpTarget::TosaRsqrt),
    ("backend.tosa.COS.default", OpTarget::TosaCos),
    ("backend.tosa.SIN.default", OpTarget::TosaSin),
    ("backend.tosa.ADD.default", OpTarget::TosaAdd),
    ("backend.tosa.SUB.default", OpTarget::TosaSub),
    ("backend.tosa.MUL.default", OpTarget::TosaMul),
    ("backend.tosa.CONST.default", OpTarget::TosaConst),
    ("backend.tosa.MAXIMUM.default", OpTarget::TosaMaximum),
    ("backend.tosa.MINIMUM.default", OpTarget::TosaMinimum),
];

const QUANTIZED_DECOMPOSED_TARGETS: &[(&str, OpTarget)] = &[
    (
        "quantized_decomposed.quantize_per_tensor.default",
        OpTarget::QuantizePerTensor,
    ),
    (
        "quantized_decomposed.quantize_per_tensor.tensor",
        OpTarget::QuantizePerTensor,
    ),
    (
        "quantized_decomposed.quantize_per_tensor.tensor2",
        OpTarget::QuantizePerTensor,
    ),
    (
        "quantized_decomposed.dequantize_per_tensor.default",
        OpTarget::DequantizePerTensor,
    ),
    (
        "quantized_decomposed.dequantize_per_tensor.tensor",
        OpTarget::DequantizePerTensor,
    ),
    (
        "quantized_decomposed.dequantize_per_tensor.tensor2",
        OpTarget::DequantizePerTensor,
    ),
    (
        "quantized_decomposed.dequantize_per_channel.default",
        OpTarget::DequantizePerChannel,
    ),
];

const ATEN_TARGETS: &[(&str, OpTarget)] = &[
    ("aten.conv_transpose2d.input", OpTarget::ConvTranspose2d),
    ("aten.conv2d.default", OpTarget::Conv2d),
    ("aten.conv2d.padding", OpTarget::Conv2d),
    ("aten.convolution.default", OpTarget::Conv2d),
    ("aten.argmax.default", OpTarget::ArgMax),
    ("aten.relu.default", OpTarget::Relu),
    ("aten.add.Tensor", OpTarget::Add),
    ("aten.add.Scalar", OpTarget::Add),
    ("aten.sub.Tensor", OpTarget::Sub),
    ("aten.sub.Scalar", OpTarget::Sub),
    ("aten.mul.Tensor", OpTarget::Mul),
    ("aten.mul.Scalar", OpTarget::Mul),
    ("aten.div.Tensor", OpTarget::Div),
    ("aten.div.Scalar", OpTarget::Div),
    ("aten.div.Tensor_mode", OpTarget::Div),
    ("aten.div.Scalar_mode", OpTarget::Div),
    ("aten.permute.default", OpTarget::Permute),
    ("aten.permute_copy.default", OpTarget::Permute),
    ("aten.reshape.default", OpTarget::Reshape),
    ("aten.full.default", OpTarget::Full),
    ("aten.avg_pool2d.default", OpTarget::AvgPool2d),
    ("aten.max_pool2d.default", OpTarget::MaxPool2d),
    ("aten.max_pool2d_with_indices.default", OpTarget::MaxPool2d),
    ("aten.sum.default", OpTarget::Sum),
    ("aten.sum.dim_IntList", OpTarget::Sum),
    ("aten.rsqrt.default", OpTarget::Rsqrt),
    ("aten.layer_norm.default", OpTarget::LayerNorm),
    ("aten.batch_norm.default", OpTarget::BatchNorm),
    ("aten._native_batch_norm_legit.default", OpTarget::BatchNorm),
    (
        "aten._native_batch_norm_legit.no_stats",
        OpTarget::BatchNorm,
    ),
    (
        "aten._native_batch_norm_legit_no_training.default",
        OpTarget::BatchNorm,
    ),
    ("aten.linear.default", OpTarget::Linear),
    ("aten.matmul.default", OpTarget::Matmul),
    ("aten.mm.default", OpTarget::Matmul),
    ("aten.bmm.default", OpTarget::Bmm),
    ("aten.addmm.default", OpTarget::Addmm),
    ("aten.softmax.int", OpTarget::Softmax),
    ("aten._softmax.default", OpTarget::Softmax),
    ("aten.gelu.default", OpTarget::Gelu),
    ("aten.sigmoid.default", OpTarget::Sigmoid),
    ("aten.tanh.default", OpTarget::Tanh),
    ("aten.maximum.default", OpTarget::Maximum),
    ("aten.minimum.default", OpTarget::Minimum),
    ("aten.erf.default", OpTarget::Erf),
    ("aten.clamp.default", OpTarget::Clamp),
    ("aten.clamp.Tensor", OpTarget::Clamp),
    ("aten.hardtanh.default", OpTarget::Hardtanh),
    ("aten.mean.dim", OpTarget::Mean),
    ("aten.mean.default", OpTarget::Mean),
    ("aten.cat.default", OpTarget::Cat),
    ("aten.slice_copy.Tensor", OpTarget::Slice),
    ("aten.slice.Tensor", OpTarget::Slice),
    ("aten.flip.default", OpTarget::Flip),
    ("aten.index_select.default", OpTarget::IndexSelect),
    ("aten.select_copy.int", OpTarget::SelectCopy),
    ("aten.select.int", OpTarget::SelectCopy),
    ("aten.unsqueeze.default", OpTarget::Unsqueeze),
    ("aten.unsqueeze_copy.default", OpTarget::Unsqueeze),
    ("aten.squeeze.default", OpTarget::Squeeze),
    ("aten.squeeze.dim", OpTarget::Squeeze),
    ("aten.squeeze.dims", OpTarget::Squeeze),
    ("aten.squeeze_copy.default", OpTarget::Squeeze),
    ("aten.squeeze_copy.dim", OpTarget::Squeeze),
    ("aten.squeeze_copy.dims", OpTarget::Squeeze),
    ("aten.flatten.using_ints", OpTarget::Flatten),
    ("aten.constant_pad_nd.default", OpTarget::Pad),
    ("aten.pad.default", OpTarget::Pad),
    ("aten.embedding.default", OpTarget::Embedding),
    ("aten.exp.default", OpTarget::Exp),
    ("aten.log.default", OpTarget::Log),
    ("aten.neg.default", OpTarget::Neg),
    ("aten.reciprocal.default", OpTarget::Reciprocal),
    ("aten.abs.default", OpTarget::Abs),
    ("aten.pow.Tensor_Scalar", OpTarget::Pow),
    ("aten.pow.Tensor_Tensor", OpTarget::Pow),
    ("aten.floor.default", OpTarget::Floor),
    ("aten.ceil.default", OpTarget::Ceil),
    ("aten.expand.default", OpTarget::Expand),
    ("aten.expand_copy.default", OpTarget::Expand),
    ("aten.contiguous.default", OpTarget::Contiguous),
    ("aten.view.default", OpTarget::View),
    ("aten.view_copy.default", OpTarget::View),
    ("aten._unsafe_view.default", OpTarget::View),
    ("aten.to.dtype", OpTarget::To),
    ("aten.to.device", OpTarget::To),
    ("aten.to.other", OpTarget::To),
    ("aten._to_copy.default", OpTarget::To),
    ("aten.clone.default", OpTarget::To),
    ("dim_order_ops._to_dim_order_copy.default", OpTarget::To),
    ("dim_order_ops._clone_dim_order.default", OpTarget::To),
    ("aten.group_norm.default", OpTarget::GroupNorm),
    ("aten.amax.default", OpTarget::Amax),
    ("aten.amin.default", OpTarget::Amin),
    ("aten.var.correction", OpTarget::Var),
    ("aten.var.dim", OpTarget::Var),
    ("aten.fill_.Scalar", OpTarget::FullLike),
    ("aten.fill_.Tensor", OpTarget::FullLike),
    ("aten.full_like.default", OpTarget::FullLike),
    ("aten.copy_.default", OpTarget::Copy),
    ("aten.eye.default", OpTarget::Eye),
    ("aten.eye.m", OpTarget::Eye),
    ("aten.where.self", OpTarget::Where),
    ("aten.where.ScalarOther", OpTarget::Where),
    ("aten.where.ScalarSelf", OpTarget::Where),
    ("aten.where.Scalar", OpTarget::Where),
    ("aten.floor_divide.default", OpTarget::FloorDivide),
    ("aten.floor_divide.Scalar", OpTarget::FloorDivide),
    ("aten.gather.default", OpTarget::Gather),
    ("aten.arange.default", OpTarget::Arange),
    ("aten.arange.start", OpTarget::Arange),
    ("aten.arange.start_step", OpTarget::Arange),
    ("aten.leaky_relu.default", OpTarget::LeakyRelu),
    ("aten.sqrt.default", OpTarget::Sqrt),
    ("aten.log1p.default", OpTarget::Log1p),
    ("aten.expm1.default", OpTarget::Expm1),
    ("aten.remainder.Scalar", OpTarget::Remainder),
    ("aten.remainder.Tensor", OpTarget::Remainder),
    ("aten.sign.default", OpTarget::Sign),
    ("aten.round.default", OpTarget::Round),
    ("aten.bitwise_not.default", OpTarget::BitwiseNot),
    ("aten.bitwise_and.Tensor", OpTarget::BitwiseAnd),
    ("aten.bitwise_and.Scalar", OpTarget::BitwiseAnd),
    ("aten.bitwise_or.Tensor", OpTarget::BitwiseOr),
    ("aten.bitwise_or.Scalar", OpTarget::BitwiseOr),
    ("aten.bitwise_xor.Tensor", OpTarget::BitwiseXor),
    ("aten.bitwise_xor.Scalar", OpTarget::BitwiseXor),
    ("aten.bitwise_left_shift.Tensor", OpTarget::BitwiseLeftShift),
    (
        "aten.bitwise_left_shift.Tensor_Scalar",
        OpTarget::BitwiseLeftShift,
    ),
    (
        "aten.bitwise_left_shift.Scalar_Tensor",
        OpTarget::BitwiseLeftShift,
    ),
    (
        "aten.bitwise_right_shift.Tensor",
        OpTarget::BitwiseRightShift,
    ),
    (
        "aten.bitwise_right_shift.Tensor_Scalar",
        OpTarget::BitwiseRightShift,
    ),
    (
        "aten.bitwise_right_shift.Scalar_Tensor",
        OpTarget::BitwiseRightShift,
    ),
    ("aten.logical_not.default", OpTarget::LogicalNot),
    ("aten.logical_and.default", OpTarget::LogicalAnd),
    ("aten.logical_or.default", OpTarget::LogicalOr),
    ("aten.logical_xor.default", OpTarget::LogicalXor),
    ("aten.gt.Tensor", OpTarget::Gt),
    ("aten.gt.Scalar", OpTarget::Gt),
    ("aten.gt_.Tensor", OpTarget::Gt),
    ("aten.gt_.Scalar", OpTarget::Gt),
    ("aten.ge.Tensor", OpTarget::Ge),
    ("aten.ge.Scalar", OpTarget::Ge),
    ("aten.ge_.Tensor", OpTarget::Ge),
    ("aten.ge_.Scalar", OpTarget::Ge),
    ("aten.lt.Tensor", OpTarget::Lt),
    ("aten.lt.Scalar", OpTarget::Lt),
    ("aten.lt_.Tensor", OpTarget::Lt),
    ("aten.lt_.Scalar", OpTarget::Lt),
    ("aten.le.Tensor", OpTarget::Le),
    ("aten.le.Scalar", OpTarget::Le),
    ("aten.le_.Tensor", OpTarget::Le),
    ("aten.le_.Scalar", OpTarget::Le),
    ("aten.eq.Tensor", OpTarget::Eq),
    ("aten.eq.Scalar", OpTarget::Eq),
    ("aten.eq_.Tensor", OpTarget::Eq),
    ("aten.eq_.Scalar", OpTarget::Eq),
    ("aten.sin.default", OpTarget::Sin),
    ("aten.cos.default", OpTarget::Cos),
    ("aten.tan.default", OpTarget::Tan),
    ("aten.asin.default", OpTarget::Asin),
    ("aten.acos.default", OpTarget::Acos),
    ("aten.atan.default", OpTarget::Atan),
    ("aten.sinh.default", OpTarget::Sinh),
    ("aten.cosh.default", OpTarget::Cosh),
    ("aten.asinh.default", OpTarget::Asinh),
    ("aten.acosh.default", OpTarget::Acosh),
    ("aten.elu.default", OpTarget::Elu),
    ("aten.elu_.default", OpTarget::Elu),
    ("aten.hardswish.default", OpTarget::Hardswish),
    ("aten.silu.default", OpTarget::Silu),
    (
        "aten._adaptive_avg_pool2d.default",
        OpTarget::AdaptiveAvgPool2d,
    ),
    (
        "aten.adaptive_avg_pool2d.default",
        OpTarget::AdaptiveAvgPool2d,
    ),
    (
        "aten.upsample_nearest2d.default",
        OpTarget::UpsampleNearest2d,
    ),
    ("aten.upsample_nearest2d.vec", OpTarget::UpsampleNearest2d),
    (
        "aten.upsample_bilinear2d.default",
        OpTarget::UpsampleBilinear2d,
    ),
    ("aten.upsample_bilinear2d.vec", OpTarget::UpsampleBilinear2d),
    ("aten.conv1d.default", OpTarget::Conv1d),
    ("aten.conv1d.padding", OpTarget::Conv1d),
    ("aten.scalar_tensor.default", OpTarget::ScalarTensor),
    ("aten.zeros.default", OpTarget::Zeros),
    ("aten.repeat.default", OpTarget::Repeat),
    ("aten.cumsum.default", OpTarget::Cumsum),
    ("aten.erfinv.default", OpTarget::Erfinv),
    ("aten.unfold_copy.default", OpTarget::UnfoldCopy),
    ("aten.unfold.default", OpTarget::UnfoldCopy),
    (
        "aten.split_with_sizes_copy.default",
        OpTarget::SplitWithSizesCopy,
    ),
    (
        "aten.split_with_sizes.default",
        OpTarget::SplitWithSizesCopy,
    ),
    ("aten.index.Tensor", OpTarget::IndexTensor),
    ("aten.any.default", OpTarget::Any),
    ("aten.any.dim", OpTarget::Any),
    ("aten.any.dims", OpTarget::Any),
];

/// Map a normalized FX target to its operator, retaining unknown names for diagnostics.
pub fn parse_op_target(target_str: &str) -> OpTarget {
    exact_match(target_str, CONTROL_FLOW_TARGETS)
        .or_else(|| parse_tosa_target(target_str))
        .or_else(|| exact_match(target_str, QUANTIZED_DECOMPOSED_TARGETS))
        .or_else(|| exact_match(target_str, ATEN_TARGETS))
        .unwrap_or_else(|| OpTarget::Other(target_str.to_string()))
}

fn parse_tosa_target(target_str: &str) -> Option<OpTarget> {
    TOSA_TARGETS.iter().find_map(|(symbol, target)| {
        // BackendOpOverload stringifies without the synthetic "backend." prefix.
        (*symbol == target_str || symbol.strip_prefix("backend.") == Some(target_str))
            .then(|| target.clone())
    })
}

fn exact_match(target_str: &str, targets: &[(&str, OpTarget)]) -> Option<OpTarget> {
    targets
        .iter()
        .find_map(|(symbol, target)| (*symbol == target_str).then(|| target.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_inventory(fixture: &str, count: usize) {
        let mut symbols = std::collections::HashSet::new();
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let (symbol, expected) = line.split_once('\t').expect("target and classification");
            assert!(symbols.insert(symbol), "duplicate target: {symbol}");
            let actual = parse_op_target(symbol);
            if expected == "Other" {
                assert_eq!(actual, OpTarget::Other(symbol.to_string()), "{symbol}");
            } else {
                assert_eq!(format!("{actual:?}"), expected, "{symbol}");
            }

            let raw =
                format!("<EdgeOpOverload: {symbol}>: schema = op(Tensor transposed) -> Tensor");
            let normalized = crate::fx_ir::normalize_fx_target(&raw);
            assert_eq!(normalized, symbol);
            assert_eq!(parse_op_target(&normalized), actual);
            if let Some(imported_symbol) = symbol.strip_prefix("backend.tosa.") {
                let imported_symbol = format!("tosa.{imported_symbol}");
                let imported_target = if expected == "Other" {
                    OpTarget::Other(imported_symbol.clone())
                } else {
                    actual
                };
                assert_eq!(parse_op_target(&imported_symbol), imported_target);
            }
        }
        assert_eq!(symbols.len(), count);
    }

    #[test]
    fn classifies_all_rc4_arm_edge_targets() {
        assert_inventory(
            include_str!("../../tests/fixtures/rc4_arm_edge_targets.tsv"),
            197,
        );
    }

    #[test]
    fn classifies_all_90_tosa_targets() {
        assert_inventory(
            include_str!("../../tests/fixtures/rc4_tosa_targets.tsv"),
            90,
        );
    }

    #[test]
    fn keeps_aten_and_tosa_targets_distinct() {
        for (symbol, aten, tosa) in [
            ("maximum", OpTarget::Maximum, OpTarget::TosaMaximum),
            ("minimum", OpTarget::Minimum, OpTarget::TosaMinimum),
            ("erf", OpTarget::Erf, OpTarget::TosaErf),
            ("gather", OpTarget::Gather, OpTarget::TosaGather),
        ] {
            assert_eq!(parse_op_target(&format!("aten.{symbol}.default")), aten);
            assert_eq!(
                parse_op_target(&format!(
                    "backend.tosa.{}.default",
                    symbol.to_ascii_uppercase()
                )),
                tosa
            );
        }
    }

    #[test]
    fn rejects_unknown_names_namespaces_and_overloads() {
        for fixture in [
            include_str!("../../tests/fixtures/rc4_arm_edge_targets.tsv"),
            include_str!("../../tests/fixtures/rc4_tosa_targets.tsv"),
        ] {
            for line in fixture.lines().filter(|line| !line.starts_with('#')) {
                let (symbol, _) = line.split_once('\t').unwrap();
                for unknown in [format!("renamed.{symbol}"), format!("{symbol}.renamed")] {
                    assert_eq!(
                        parse_op_target(&unknown),
                        OpTarget::Other(unknown.clone()),
                        "{unknown}"
                    );
                }
            }
        }
        for unknown in [
            "aten.adaptive_max_pool2d.default",
            "aten.rsub.Tensor",
            "aten.bilinear.default",
            "aten.embedding_bag.default",
            "aten.exp2.default",
            "aten.log2.default",
            "backend.tosa.FULLY_CONNECTED.default",
            "backend.tosa.NEG.default",
            "backend.tosa.conv2d.default",
            "backend.tosa.CONV2D.out",
        ] {
            assert_eq!(
                parse_op_target(unknown),
                OpTarget::Other(unknown.to_string()),
                "{unknown}"
            );
        }
    }

    #[test]
    fn rejects_near_miss_tosa_targets() {
        for target in [
            "backend.tosa.LOGICAL_LEFT_SHIFT.default",
            "backend.tosa.CONST_SHAPE.out",
            "backend.tosa.MAX_POOL2D_ADAPTIVE.default",
            "backend.tosa.CONV2D_BLOCK_SCALED.default",
        ] {
            assert_eq!(
                parse_op_target(target),
                OpTarget::Other(target.to_string()),
                "{target}"
            );
        }
    }

    #[test]
    fn parses_exact_tosa_targets() {
        assert_eq!(
            parse_op_target("backend.tosa.GREATER_EQUAL.default"),
            OpTarget::TosaGreaterEqual
        );
        assert_eq!(
            parse_op_target("backend.tosa.TRANSPOSE_CONV2D.default"),
            OpTarget::TosaTransposeConv2d
        );
        assert_eq!(
            parse_op_target("backend.tosa.INTDIV.default"),
            OpTarget::TosaIntDiv
        );
    }

    #[test]
    fn parses_exact_aten_targets() {
        assert_eq!(
            parse_op_target("aten.conv_transpose2d.input"),
            OpTarget::ConvTranspose2d
        );
        assert_eq!(
            parse_op_target("aten.slice_scatter.default"),
            OpTarget::Other("aten.slice_scatter.default".to_string())
        );
        assert_eq!(parse_op_target("aten.erf.default"), OpTarget::Erf);
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
