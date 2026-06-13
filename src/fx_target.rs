#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IrTargetKind {
    AtenConv,
    AtenLinear,
    AtenRelu,
    AtenHardtanh,
    QuantizePerTensor,
    QuantizePerChannel,
    DequantizePerTensor,
    DequantizePerChannel,
    TosaRescale,
    AtenMatmul,
    AtenMm,
    AtenBmm,
    AtenAddmm,
    Cond,
    WhileLoop,
    GetItem,
    Other,
}

pub(crate) fn classify_ir_target(target: &str) -> IrTargetKind {
    if matches!(
        target,
        "cond" | "higher_order.cond" | "torch.ops.higher_order.cond"
    ) {
        return IrTargetKind::Cond;
    }
    if matches!(
        target,
        "while_loop" | "higher_order.while_loop" | "torch.ops.higher_order.while_loop"
    ) {
        return IrTargetKind::WhileLoop;
    }
    if target.contains("getitem") {
        return IrTargetKind::GetItem;
    }

    if target.contains("quantized_decomposed.dequantize_per_channel")
        || target.contains("dequantize_per_channel")
    {
        return IrTargetKind::DequantizePerChannel;
    }
    if target.contains("quantized_decomposed.dequantize_per_tensor")
        || target.contains("dequantize_per_tensor")
    {
        return IrTargetKind::DequantizePerTensor;
    }
    if target.contains("quantized_decomposed.quantize_per_channel")
        || (target.contains("quantize_per_channel") && !target.contains("dequantize"))
    {
        return IrTargetKind::QuantizePerChannel;
    }
    if target.contains("quantized_decomposed.quantize_per_tensor")
        || (target.contains("quantize_per_tensor") && !target.contains("dequantize"))
    {
        return IrTargetKind::QuantizePerTensor;
    }

    if target.contains("tosa") && target.to_ascii_lowercase().contains("rescale") {
        return IrTargetKind::TosaRescale;
    }

    if target.contains("convolution.default") || target.contains("conv2d.default") {
        return IrTargetKind::AtenConv;
    }
    if target.contains("linear.default") {
        return IrTargetKind::AtenLinear;
    }
    if target.contains("relu.default") {
        return IrTargetKind::AtenRelu;
    }
    if target.contains("hardtanh.default") {
        return IrTargetKind::AtenHardtanh;
    }
    if target.contains("aten.addmm") {
        return IrTargetKind::AtenAddmm;
    }
    if target.contains("aten.bmm") {
        return IrTargetKind::AtenBmm;
    }
    if target.contains("aten.mm.") {
        return IrTargetKind::AtenMm;
    }
    if target.contains("aten.matmul") {
        return IrTargetKind::AtenMatmul;
    }

    IrTargetKind::Other
}

pub(crate) fn is_zero_input_constant_target(target: &str) -> bool {
    target.contains("aten.full.default")
        || target.contains("aten.arange")
        || target.contains("aten.linspace")
        || target.contains("aten.eye")
}

pub(crate) fn is_shape_only_constant_target(target: &str) -> bool {
    target.contains("aten.reshape.default")
        || target.contains("aten.view_copy.default")
        || target.contains("aten.unsqueeze_copy.default")
        || target.contains("aten.permute_copy.default")
}

pub(crate) fn is_view_like_target(target: &str) -> bool {
    target.contains("view_copy")
        || target.contains("flatten")
        || target.contains("reshape")
        || target.contains("aten.view.")
        || target.contains("aten._unsafe_view")
}

pub(crate) fn is_permute_target(target: &str) -> bool {
    target.contains("permute_copy") || target.contains("aten.permute.")
}

pub(crate) fn is_layout_passthrough_target(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    target.contains("view")
        || target.contains("reshape")
        || target.contains("transpose")
        || target.contains("permute")
        || target.contains("squeeze")
        || target.contains("unsqueeze")
        || target.contains("flatten")
        || target.contains("contiguous")
}

pub(crate) fn is_rank_changing_layout_target(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    target.contains("view")
        || target.contains("reshape")
        || target.contains("flatten")
        || target.contains("squeeze")
        || target.contains("unsqueeze")
}

pub(crate) fn is_transpose_conv2d_target(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    target.contains("transpose_conv2d") || target.contains("transposeconv2d")
}

pub(crate) fn is_conv_bias_annotation_target(target: &str) -> bool {
    target.contains("convolution.default")
        || target.contains("conv2d.default")
        || target.contains("conv3d.default")
        || target.contains("conv_transpose2d")
        || target.contains("transpose_conv2d")
        || target.contains("transposeconv2d")
        || target.contains("backend.tosa.CONV")
        || target.contains("backend.tosa.DEPTHWISE_CONV")
        || target.contains("backend.tosa.TRANSPOSE_CONV2D")
}

pub(crate) fn is_raw_aten_convolution_target(target: &str) -> bool {
    target.to_ascii_lowercase().contains("aten.convolution")
}

pub(crate) fn is_conv_accumulator_target(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    if target.contains("tosa") {
        return target.contains("conv2d")
            || target.contains("depthwise_conv2d")
            || target.contains("conv3d");
    }
    target.contains("aten.convolution") || target.contains("aten.conv2d")
}

pub(crate) fn is_size_adjust_target(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    if is_transpose_conv2d_target(&target) {
        return false;
    }
    target.contains("convolution.default")
        || target.contains("conv2d")
        || target.contains("conv3d")
        || target.contains("max_pool2d")
        || target.contains("avg_pool2d")
}

pub(crate) fn is_split_target(target: &str) -> bool {
    target.contains("split_with_sizes_copy") || target.contains("split_copy")
}

pub(crate) fn is_tril_target(target: &str) -> bool {
    target.contains("aten.tril")
        || target.contains("aten_tril")
        || target.contains("aten.tril_copy")
        || target.contains("aten_tril_copy")
}

pub(crate) fn is_layer_norm_target(target: &str) -> bool {
    target.contains("layer_norm") || target.contains("native_layer_norm")
}

pub(crate) fn is_var_target(target: &str) -> bool {
    target.contains("var.correction")
        || target.contains("var_correction")
        || target.contains("var.dim")
        || target.contains("var_dim")
}

pub(crate) fn is_var_dim_target(target: &str) -> bool {
    target.contains("var.dim")
}

pub(crate) fn is_plain_avg_pool2d_target(target: &str) -> bool {
    target.contains("avg_pool2d") && !target.contains("adaptive")
}

pub(crate) fn is_as_strided_copy_target(target: &str) -> bool {
    target.contains("as_strided_copy.default")
}

pub(crate) fn is_pad_target(target: &str) -> bool {
    target.contains("aten.pad.default")
}

pub(crate) fn is_pixel_shuffle_target(target: &str) -> bool {
    target.contains("pixel_shuffle") && !target.contains("pixel_unshuffle")
}

pub(crate) fn is_pixel_unshuffle_target(target: &str) -> bool {
    target.contains("pixel_unshuffle")
}

pub(crate) fn is_squeeze_target(target: &str) -> bool {
    !target.contains("unsqueeze")
        && (target.contains("squeeze_copy")
            || target.contains("squeeze.dim")
            || target.contains("squeeze.default")
            || target.contains("squeeze.dims"))
}

pub(crate) fn is_unsqueeze_target(target: &str) -> bool {
    target.contains("unsqueeze_copy") || target.contains("unsqueeze.default")
}

pub(crate) fn is_dim_transpose_target(target: &str) -> bool {
    target.contains("transpose_copy") || target.contains("transpose.int")
}

pub(crate) fn is_index_put_family_target(target: &str) -> bool {
    target.contains("index_put")
}

pub(crate) fn is_index_put_default_target(target: &str) -> bool {
    target.contains("index_put.default") || target.contains("index_put_.default")
}

pub(crate) fn is_where_target(target: &str) -> bool {
    target.contains("where.self") || target.contains("aten_where")
}

pub(crate) fn is_clamp_target(target: &str) -> bool {
    target.contains("clamp.default")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_quantize_from_dequantize() {
        assert_eq!(
            classify_ir_target("quantized_decomposed.quantize_per_tensor.default"),
            IrTargetKind::QuantizePerTensor
        );
        assert_eq!(
            classify_ir_target("quantized_decomposed.quantize_per_channel.default"),
            IrTargetKind::QuantizePerChannel
        );
        assert_eq!(
            classify_ir_target("quantized_decomposed.dequantize_per_tensor.default"),
            IrTargetKind::DequantizePerTensor
        );
        assert_eq!(
            classify_ir_target("quantized_decomposed.dequantize_per_channel.default"),
            IrTargetKind::DequantizePerChannel
        );
    }

    #[test]
    fn recognizes_tosa_rescale_case_insensitively() {
        assert_eq!(
            classify_ir_target("backend.tosa.RESCALE.default"),
            IrTargetKind::TosaRescale
        );
        assert_eq!(
            classify_ir_target("tosa.rescale.default"),
            IrTargetKind::TosaRescale
        );
    }

    #[test]
    fn recognizes_activation_and_producer_targets() {
        assert_eq!(
            classify_ir_target("aten.relu.default"),
            IrTargetKind::AtenRelu
        );
        assert_eq!(
            classify_ir_target("aten.hardtanh.default"),
            IrTargetKind::AtenHardtanh
        );
        assert_eq!(
            classify_ir_target("aten.convolution.default"),
            IrTargetKind::AtenConv
        );
        assert_eq!(
            classify_ir_target("aten.linear.default"),
            IrTargetKind::AtenLinear
        );
    }

    #[test]
    fn distinguishes_matmul_overloads() {
        assert_eq!(
            classify_ir_target("aten.matmul.default"),
            IrTargetKind::AtenMatmul
        );
        assert_eq!(classify_ir_target("aten.mm.default"), IrTargetKind::AtenMm);
        assert_eq!(
            classify_ir_target("aten.bmm.default"),
            IrTargetKind::AtenBmm
        );
        assert_eq!(
            classify_ir_target("aten.addmm.default"),
            IrTargetKind::AtenAddmm
        );
    }

    #[test]
    fn recognizes_exact_control_flow_symbols() {
        assert_eq!(classify_ir_target("cond"), IrTargetKind::Cond);
        assert_eq!(
            classify_ir_target("torch.ops.higher_order.while_loop"),
            IrTargetKind::WhileLoop
        );
        assert_eq!(
            classify_ir_target("operator.getitem"),
            IrTargetKind::GetItem
        );
    }

    #[test]
    fn recognizes_constant_target_families() {
        assert!(is_zero_input_constant_target("aten.full.default"));
        assert!(is_zero_input_constant_target("aten.arange.start"));
        assert!(is_zero_input_constant_target("aten.linspace.default"));
        assert!(is_zero_input_constant_target("aten.eye.default"));
        assert!(is_shape_only_constant_target("aten.reshape.default"));
        assert!(is_shape_only_constant_target("aten.view_copy.default"));
        assert!(is_shape_only_constant_target("aten.unsqueeze_copy.default"));
        assert!(is_shape_only_constant_target("aten.permute_copy.default"));
        assert!(is_view_like_target("aten._unsafe_view.default"));
        assert!(is_permute_target("aten.permute.default"));
        assert!(!is_shape_only_constant_target("aten.slice_copy.Tensor"));
    }

    #[test]
    fn recognizes_layout_target_families() {
        assert!(is_layout_passthrough_target("aten.transpose.int"));
        assert!(is_layout_passthrough_target("aten.contiguous.default"));
        assert!(is_rank_changing_layout_target("aten.flatten.using_ints"));
        assert!(!is_rank_changing_layout_target("aten.transpose.int"));
    }

    #[test]
    fn recognizes_convolution_target_families() {
        assert!(is_conv_bias_annotation_target("aten.conv3d.default"));
        assert!(is_conv_bias_annotation_target(
            "backend.tosa.TRANSPOSE_CONV2D.default"
        ));
        assert!(is_raw_aten_convolution_target("aten.convolution.default"));
        assert!(is_conv_accumulator_target("backend.tosa.CONV2D.default"));
        assert!(is_conv_accumulator_target(
            "backend.tosa.DEPTHWISE_CONV2D.default"
        ));
        assert!(is_conv_accumulator_target("aten.conv2d.padding"));
        assert!(is_size_adjust_target("aten.avg_pool2d.default"));
        assert!(!is_size_adjust_target("aten.transpose_conv2d.input"));
    }

    #[test]
    fn recognizes_decomposition_target_families() {
        assert!(is_split_target("aten.split_with_sizes_copy.default"));
        assert!(is_split_target("aten.split_copy.Tensor"));
        assert!(is_tril_target("aten.tril_copy.default"));
        assert!(is_layer_norm_target("aten.native_layer_norm.default"));
        assert!(is_var_target("aten.var.correction"));
        assert!(is_var_dim_target("aten.var.dim"));
        assert!(is_plain_avg_pool2d_target("aten.avg_pool2d.default"));
        assert!(!is_plain_avg_pool2d_target(
            "aten.adaptive_avg_pool2d.default"
        ));
        assert!(is_as_strided_copy_target("aten.as_strided_copy.default"));
        assert!(is_pad_target("aten.pad.default"));
        assert!(is_pixel_shuffle_target("aten.pixel_shuffle.default"));
        assert!(!is_pixel_shuffle_target("aten.pixel_unshuffle.default"));
        assert!(is_pixel_unshuffle_target("aten.pixel_unshuffle.default"));
        assert!(is_squeeze_target("aten.squeeze.dim"));
        assert!(is_unsqueeze_target("aten.unsqueeze.default"));
        assert!(is_dim_transpose_target("aten.transpose.int"));
    }

    #[test]
    fn recognizes_index_mask_and_clamp_targets() {
        assert!(is_index_put_family_target("aten.index_put.default"));
        assert!(is_index_put_family_target("aten.index_put_.default"));
        assert!(is_index_put_default_target("aten.index_put.default"));
        assert!(is_index_put_default_target("aten.index_put_.default"));
        assert!(!is_index_put_default_target("aten.index.Tensor"));
        assert!(is_where_target("aten.where.self"));
        assert!(is_where_target("aten_where.self"));
        assert!(is_clamp_target("aten.clamp.default"));
        assert!(!is_clamp_target("aten.clamp.Tensor"));
    }
}
