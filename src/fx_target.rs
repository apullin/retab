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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroInputConstantTarget {
    Full,
    Arange,
    Linspace,
    Eye,
    ScalarTensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArangeTargetKind {
    Start,
    StartStep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TableOpTargetKind {
    Unary(&'static str),
    Gelu,
    PowTensorScalar,
    RemainderScalar,
    LeakyRelu,
    Softplus,
    Elu(EluTargetKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EluTargetKind {
    Elu,
    Selu,
    Celu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShapeOnlyConstantTarget {
    Reshape,
    Unsqueeze,
    Permute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoopTarget {
    PassthroughFirstArg,
    CopySourceArg,
    DtypeCheckedPassthrough,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SizeAdjustTargetKind {
    Conv,
    Pool(PoolSizeAdjustKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PoolSizeAdjustKind {
    Max,
    Avg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerNormTargetKind {
    Plain,
    NativeOrOther,
}

pub(crate) const BATCH_NORM_TARGETS: &[&str] = &[
    "aten.batch_norm.default",
    "aten.native_batch_norm.default",
    "aten._native_batch_norm_legit_no_training.default",
    "aten._native_batch_norm_legit.no_stats",
];

pub(crate) const GROUP_NORM_TARGETS: &[&str] =
    &["aten.group_norm.default", "aten.native_group_norm.default"];

pub(crate) fn classify_ir_target(target: &str) -> IrTargetKind {
    match target {
        "cond" | "higher_order.cond" | "torch.ops.higher_order.cond" => IrTargetKind::Cond,
        "while_loop" | "higher_order.while_loop" | "torch.ops.higher_order.while_loop" => {
            IrTargetKind::WhileLoop
        }
        "getitem" | "operator.getitem" | "<built-in function getitem>" => IrTargetKind::GetItem,
        "quantized_decomposed.dequantize_per_channel.default" => IrTargetKind::DequantizePerChannel,
        "quantized_decomposed.dequantize_per_tensor.default"
        | "quantized_decomposed.dequantize_per_tensor.tensor"
        | "quantized_decomposed.dequantize_per_tensor.tensor2" => IrTargetKind::DequantizePerTensor,
        "quantized_decomposed.quantize_per_channel.default" => IrTargetKind::QuantizePerChannel,
        "quantized_decomposed.quantize_per_tensor.default"
        | "quantized_decomposed.quantize_per_tensor.tensor"
        | "quantized_decomposed.quantize_per_tensor.tensor2" => IrTargetKind::QuantizePerTensor,
        "backend.tosa.RESCALE.default" | "tosa.RESCALE.default" => IrTargetKind::TosaRescale,
        "aten.convolution.default" | "aten.conv2d.default" => IrTargetKind::AtenConv,
        "aten.linear.default" => IrTargetKind::AtenLinear,
        "aten.relu.default" => IrTargetKind::AtenRelu,
        "aten.hardtanh.default" => IrTargetKind::AtenHardtanh,
        "aten.addmm.default" => IrTargetKind::AtenAddmm,
        "aten.bmm.default" => IrTargetKind::AtenBmm,
        "aten.mm.default" => IrTargetKind::AtenMm,
        "aten.matmul.default" => IrTargetKind::AtenMatmul,
        _ => IrTargetKind::Other,
    }
}

pub(crate) fn zero_input_constant_target(target: &str) -> Option<ZeroInputConstantTarget> {
    match target {
        "aten.full.default" => Some(ZeroInputConstantTarget::Full),
        "aten.arange.default" | "aten.arange.start" | "aten.arange.start_step" => {
            Some(ZeroInputConstantTarget::Arange)
        }
        "aten.linspace.default" => Some(ZeroInputConstantTarget::Linspace),
        "aten.eye.default" | "aten.eye.m" => Some(ZeroInputConstantTarget::Eye),
        "aten.scalar_tensor.default" => Some(ZeroInputConstantTarget::ScalarTensor),
        _ => None,
    }
}

pub(crate) fn is_to_copy_target(target: &str) -> bool {
    matches!(
        target,
        "aten._to_copy.default"
            | "dim_order_ops._to_dim_order_copy.default"
            | "_to_dim_order_copy.default"
    )
}

pub(crate) fn is_reciprocal_target(target: &str) -> bool {
    target == "aten.reciprocal.default"
}

pub(crate) fn is_zero_input_constant_target(target: &str) -> bool {
    zero_input_constant_target(target).is_some()
}

pub(crate) fn is_fold_qdq_constant_target(target: &str) -> bool {
    matches!(
        target,
        "aten.full_like.default"
            | "aten.full.default"
            | "aten.arange.start_step"
            | "aten.eye.default"
            | "aten.linspace.default"
            | "aten.scalar_tensor.default"
    )
}

pub(crate) fn arange_target_kind(target: &str) -> Option<ArangeTargetKind> {
    match target {
        "aten.arange.start_step" => Some(ArangeTargetKind::StartStep),
        "aten.arange.start" => Some(ArangeTargetKind::Start),
        _ => None,
    }
}

const SIMPLE_TABLE_TARGETS: &[(&str, &str)] = &[
    ("aten.rsqrt.default", "rsqrt"),
    ("aten.ceil.default", "ceil"),
    ("aten.floor.default", "floor"),
    ("aten.round.default", "round"),
    ("aten.reciprocal.default", "reciprocal"),
    ("aten.exp.default", "exp"),
    ("aten.log.default", "log"),
    ("aten.log10.default", "log10"),
    ("aten.hardsigmoid.default", "hardsigmoid"),
    ("aten.sigmoid.default", "sigmoid"),
    ("aten.atanh.default", "atanh"),
    ("aten.asinh.default", "asinh"),
    ("aten.acosh.default", "acosh"),
    ("aten.atan.default", "atan"),
    ("aten.tanh.default", "tanh"),
    ("aten.silu.default", "silu"),
    ("aten.erf.default", "erf"),
    ("aten.sin.default", "sin"),
    ("aten.cos.default", "cos"),
    ("aten.sinh.default", "sinh"),
    ("aten.cosh.default", "cosh"),
    ("aten.tan.default", "tan"),
    ("aten.acos.default", "acos"),
    ("aten.asin.default", "asin"),
    ("aten.expm1.default", "expm1"),
    ("aten.hardswish.default", "hardswish"),
    ("aten.erfinv.default", "erfinv"),
    ("aten.log1p.default", "log1p"),
    ("aten.sqrt.default", "sqrt"),
];

pub(crate) fn table_op_target_kind(target: &str) -> Option<TableOpTargetKind> {
    match target {
        "aten.softplus.default" => Some(TableOpTargetKind::Softplus),
        "aten.gelu.default" => Some(TableOpTargetKind::Gelu),
        "aten.pow.Tensor_Scalar" => Some(TableOpTargetKind::PowTensorScalar),
        "aten.remainder.Scalar" => Some(TableOpTargetKind::RemainderScalar),
        "aten.leaky_relu.default" => Some(TableOpTargetKind::LeakyRelu),
        _ => SIMPLE_TABLE_TARGETS
            .iter()
            .find_map(|(symbol, kind)| {
                (*symbol == target).then_some(TableOpTargetKind::Unary(kind))
            })
            .or_else(|| elu_target_kind(target).map(TableOpTargetKind::Elu)),
    }
}

pub(crate) fn elu_target_kind(target: &str) -> Option<EluTargetKind> {
    match target {
        "aten.selu.default" | "aten.selu_.default" => Some(EluTargetKind::Selu),
        "aten.celu.default" | "aten.celu_.default" => Some(EluTargetKind::Celu),
        "aten.elu.default" | "aten.elu_.default" => Some(EluTargetKind::Elu),
        _ => None,
    }
}

pub(crate) fn shape_only_constant_target(target: &str) -> Option<ShapeOnlyConstantTarget> {
    match target {
        "aten.reshape.default" | "aten.view_copy.default" => Some(ShapeOnlyConstantTarget::Reshape),
        "aten.unsqueeze_copy.default" => Some(ShapeOnlyConstantTarget::Unsqueeze),
        "aten.permute_copy.default" => Some(ShapeOnlyConstantTarget::Permute),
        _ => None,
    }
}

pub(crate) fn is_shape_only_constant_target(target: &str) -> bool {
    shape_only_constant_target(target).is_some()
}

pub(crate) fn noop_target(target: &str) -> Option<NoopTarget> {
    match target {
        "aten.copy.default" => Some(NoopTarget::CopySourceArg),
        "dim_order_ops._to_dim_order_copy.default"
        | "_to_dim_order_copy.default"
        | "aten._to_copy.default" => Some(NoopTarget::DtypeCheckedPassthrough),
        "aten.clone.default"
        | "dim_order_ops._clone_dim_order.default"
        | "_clone_dim_order.default"
        | "aten.alias_copy.default"
        | "alias_copy.default"
        | "aten.detach_copy.default"
        | "detach_copy.default" => Some(NoopTarget::PassthroughFirstArg),
        _ => None,
    }
}

pub(crate) fn is_view_like_target(target: &str) -> bool {
    matches!(
        target,
        "aten.view_copy.default"
            | "aten.flatten.using_ints"
            | "aten.reshape.default"
            | "aten.view.default"
            | "aten._unsafe_view.default"
    )
}

pub(crate) fn is_permute_target(target: &str) -> bool {
    matches!(target, "aten.permute_copy.default" | "aten.permute.default")
}

pub(crate) fn is_tosa_transpose_target(target: &str) -> bool {
    matches!(
        target,
        "backend.tosa.TRANSPOSE.default" | "tosa.TRANSPOSE.default"
    )
}

pub(crate) fn is_boundary_rescale_target(target: &str) -> bool {
    matches!(
        classify_ir_target(target),
        IrTargetKind::TosaRescale | IrTargetKind::QuantizePerTensor
    )
}

pub(crate) fn is_layout_passthrough_target(target: &str) -> bool {
    is_rank_changing_layout_target(target)
        || is_permute_target(target)
        || is_dim_transpose_target(target)
        || is_tosa_transpose_target(target)
        || matches!(
            target,
            "aten.contiguous.default" | "backend.tosa.IDENTITY.default" | "tosa.IDENTITY.default"
        )
}

pub(crate) fn is_rank_changing_layout_target(target: &str) -> bool {
    is_view_like_target(target)
        || is_squeeze_target(target)
        || is_unsqueeze_target(target)
        || matches!(
            target,
            "backend.tosa.RESHAPE.default" | "tosa.RESHAPE.default"
        )
}

pub(crate) fn is_conv1d_default_target(target: &str) -> bool {
    target == "aten.conv1d.default"
}

pub(crate) fn is_conv2d_default_target(target: &str) -> bool {
    target == "aten.conv2d.default"
}

pub(crate) fn is_conv3d_default_target(target: &str) -> bool {
    target == "aten.conv3d.default"
}

pub(crate) fn is_aten_convolution_default_target(target: &str) -> bool {
    target == "aten.convolution.default"
}

pub(crate) fn is_raw_aten_convolution_target(target: &str) -> bool {
    matches!(
        target,
        "aten.convolution.default" | "edge__ops_aten_convolution.default"
    )
}

pub(crate) fn size_adjust_target_kind(target: &str) -> Option<SizeAdjustTargetKind> {
    if is_aten_convolution_default_target(target)
        || is_conv2d_default_target(target)
        || is_conv3d_default_target(target)
        || matches!(
            target,
            "aten.conv2d.padding"
                | "aten.conv3d.padding"
                | "backend.tosa.CONV2D.default"
                | "tosa.CONV2D.default"
                | "backend.tosa.DEPTHWISE_CONV2D.default"
                | "tosa.DEPTHWISE_CONV2D.default"
                | "backend.tosa.CONV3D.default"
                | "tosa.CONV3D.default"
        )
    {
        return Some(SizeAdjustTargetKind::Conv);
    }
    if is_max_pool2d_target(target)
        || matches!(
            target,
            "backend.tosa.MAX_POOL2D.default" | "tosa.MAX_POOL2D.default"
        )
    {
        return Some(SizeAdjustTargetKind::Pool(PoolSizeAdjustKind::Max));
    }
    if is_plain_avg_pool2d_target(target)
        || matches!(
            target,
            "backend.tosa.AVG_POOL2D.default" | "tosa.AVG_POOL2D.default"
        )
    {
        return Some(SizeAdjustTargetKind::Pool(PoolSizeAdjustKind::Avg));
    }
    None
}

pub(crate) fn is_split_target(target: &str) -> bool {
    matches!(
        target,
        "aten.split_with_sizes_copy.default" | "aten.split_copy.Tensor" | "aten.split_copy.sizes"
    )
}

pub(crate) fn is_slice_copy_tensor_target(target: &str) -> bool {
    target == "aten.slice_copy.Tensor"
}

pub(crate) fn is_slice_tensor_target(target: &str) -> bool {
    matches!(target, "aten.slice_copy.Tensor" | "aten.slice.Tensor")
}

pub(crate) fn is_tosa_slice_target(target: &str) -> bool {
    matches!(target, "backend.tosa.SLICE.default" | "tosa.SLICE.default")
}

pub(crate) fn is_slice_scatter_target(target: &str) -> bool {
    target == "aten.slice_scatter.default"
}

#[cfg(test)]
pub(crate) fn is_cat_target(target: &str) -> bool {
    target == "aten.cat.default"
}

pub(crate) fn is_unfold_copy_target(target: &str) -> bool {
    target == "aten.unfold_copy.default"
}

pub(crate) fn is_gather_target(target: &str) -> bool {
    target == "aten.gather.default"
}

pub(crate) fn is_tril_target(target: &str) -> bool {
    matches!(
        target,
        "aten.tril.default"
            | "aten.tril_copy.default"
            | "aten_tril.default"
            | "aten_tril_copy.default"
    )
}

pub(crate) fn is_embedding_target(target: &str) -> bool {
    target == "aten.embedding.default"
}

pub(crate) fn is_logit_target(target: &str) -> bool {
    matches!(target, "aten.logit.default" | "aten.logit_.default")
}

pub(crate) fn is_index_select_target(target: &str) -> bool {
    target == "aten.index_select.default"
}

pub(crate) fn is_index_tensor_target(target: &str) -> bool {
    target == "aten.index.Tensor"
}

pub(crate) fn is_cumsum_target(target: &str) -> bool {
    matches!(target, "aten.cumsum.default" | "aten.cumsum_.default")
}

pub(crate) fn is_layer_norm_target(target: &str) -> bool {
    layer_norm_target_kind(target).is_some()
}

pub(crate) fn layer_norm_target_kind(target: &str) -> Option<LayerNormTargetKind> {
    match target {
        "aten.native_layer_norm.default" => Some(LayerNormTargetKind::NativeOrOther),
        "aten.layer_norm.default" => Some(LayerNormTargetKind::Plain),
        _ => None,
    }
}

pub(crate) fn layer_norm_param_view_target(target: &str) -> Option<&'static str> {
    match layer_norm_target_kind(target)? {
        LayerNormTargetKind::Plain => Some("aten.reshape.default"),
        LayerNormTargetKind::NativeOrOther => Some("aten.view_copy.default"),
    }
}

pub(crate) fn is_var_target(target: &str) -> bool {
    matches!(target, "aten.var.correction" | "aten.var.dim")
}

pub(crate) fn is_var_dim_target(target: &str) -> bool {
    target == "aten.var.dim"
}

pub(crate) fn is_plain_avg_pool2d_target(target: &str) -> bool {
    target == "aten.avg_pool2d.default"
}

pub(crate) fn is_adaptive_avg_pool2d_target(target: &str) -> bool {
    matches!(
        target,
        "aten.adaptive_avg_pool2d.default" | "aten._adaptive_avg_pool2d.default"
    )
}

pub(crate) fn is_as_strided_copy_target(target: &str) -> bool {
    target == "aten.as_strided_copy.default"
}

pub(crate) fn is_pad_target(target: &str) -> bool {
    matches!(target, "aten.pad.default" | "aten.constant_pad_nd.default")
}

pub(crate) fn is_pixel_shuffle_target(target: &str) -> bool {
    target == "aten.pixel_shuffle.default"
}

pub(crate) fn is_pixel_unshuffle_target(target: &str) -> bool {
    target == "aten.pixel_unshuffle.default"
}

pub(crate) fn is_squeeze_target(target: &str) -> bool {
    matches!(
        target,
        "aten.squeeze_copy.default"
            | "aten.squeeze_copy.dim"
            | "aten.squeeze_copy.dims"
            | "aten.squeeze.dim"
            | "aten.squeeze.default"
            | "aten.squeeze.dims"
    )
}

pub(crate) fn is_unsqueeze_target(target: &str) -> bool {
    matches!(
        target,
        "aten.unsqueeze_copy.default" | "aten.unsqueeze.default"
    )
}

pub(crate) fn is_dim_transpose_target(target: &str) -> bool {
    matches!(target, "aten.transpose_copy.int" | "aten.transpose.int")
}

pub(crate) fn is_index_put_family_target(target: &str) -> bool {
    matches!(
        target,
        "aten.index_put.default" | "aten.index_put_.default" | "aten._index_put_impl_.default"
    )
}

pub(crate) fn is_index_put_default_target(target: &str) -> bool {
    matches!(target, "aten.index_put.default" | "aten.index_put_.default")
}

pub(crate) fn is_where_target(target: &str) -> bool {
    matches!(target, "aten.where.self" | "aten_where.self")
}

pub(crate) fn is_masked_fill_target(target: &str) -> bool {
    matches!(
        target,
        "aten.masked_fill.Scalar"
            | "aten.masked_fill.Tensor"
            | "aten.masked_fill_.Scalar"
            | "aten.masked_fill_.Tensor"
    )
}

pub(crate) fn is_select_scatter_target(target: &str) -> bool {
    target == "aten.select_scatter.default"
}

pub(crate) fn is_clamp_target(target: &str) -> bool {
    target == "aten.clamp.default"
}

pub(crate) fn is_abs_target(target: &str) -> bool {
    target == "aten.abs.default"
}

pub(crate) fn is_comparison_target(target: &str) -> bool {
    matches!(
        target,
        "aten.eq.Tensor"
            | "aten.ge.Tensor"
            | "aten.gt.Tensor"
            | "aten.le.Tensor"
            | "aten.lt.Tensor"
    )
}

pub(crate) fn is_minmax_target(target: &str) -> bool {
    matches!(target, "aten.maximum.default" | "aten.minimum.default")
}

pub(crate) fn is_add_sub_target(target: &str) -> bool {
    matches!(target, "aten.add.Tensor" | "aten.sub.Tensor")
}

pub(crate) fn add_sub_alpha_result_target(target: &str) -> Option<&'static str> {
    match target {
        "aten.sub.Tensor" => Some("aten.sub.Tensor"),
        "aten.add.Tensor" => Some("aten.add.Tensor"),
        _ => None,
    }
}

pub(crate) fn is_mul_target(target: &str) -> bool {
    target == "aten.mul.Tensor"
}

pub(crate) fn is_sum_dim_int_list_target(target: &str) -> bool {
    target == "aten.sum.dim_IntList"
}

pub(crate) fn is_mean_target(target: &str) -> bool {
    matches!(target, "aten.mean.dim" | "aten.mean.default")
}

pub(crate) fn is_full_like_target(target: &str) -> bool {
    target == "aten.full_like.default"
}

pub(crate) fn is_batch_norm_target(target: &str) -> bool {
    BATCH_NORM_TARGETS.contains(&target)
}

pub(crate) fn not_equal_replacement_target(target: &str) -> Option<&'static str> {
    match target {
        "aten.ne.Scalar" => Some("aten.eq.Scalar"),
        "aten.ne.Tensor" | "aten.ne_.Tensor" => Some("aten.eq.Tensor"),
        _ => None,
    }
}

pub(crate) fn scalar_shift_replacement_target(target: &str) -> Option<&'static str> {
    match target {
        "aten.__lshift__.Scalar" => Some("aten.bitwise_left_shift.Tensor"),
        "aten.__rshift__.Scalar" => Some("aten.bitwise_right_shift.Tensor"),
        _ => None,
    }
}

pub(crate) fn is_repeat_interleave_self_int_target(target: &str) -> bool {
    target == "aten.repeat_interleave.self_int"
}

pub(crate) fn is_max_pool2d_target(target: &str) -> bool {
    matches!(
        target,
        "aten.max_pool2d.default" | "aten.max_pool2d_with_indices.default"
    )
}

pub(crate) fn is_max_pool2d_with_indices_target(target: &str) -> bool {
    target == "aten.max_pool2d_with_indices.default"
}

pub(crate) fn group_norm_reshape_target(target: &str) -> Option<&'static str> {
    match target {
        "aten.native_group_norm.default" => Some("aten.view_copy.default"),
        "aten.group_norm.default" => Some("aten.reshape.default"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrapped_and_extended_targets_in_all_predicates() {
        let predicates: &[fn(&str) -> bool] = &[
            is_zero_input_constant_target,
            is_fold_qdq_constant_target,
            is_shape_only_constant_target,
            is_view_like_target,
            is_permute_target,
            is_tosa_transpose_target,
            is_boundary_rescale_target,
            is_layout_passthrough_target,
            is_rank_changing_layout_target,
            is_conv1d_default_target,
            is_conv2d_default_target,
            is_conv3d_default_target,
            is_aten_convolution_default_target,
            is_raw_aten_convolution_target,
            is_split_target,
            is_slice_copy_tensor_target,
            is_slice_tensor_target,
            is_slice_scatter_target,
            is_cat_target,
            is_unfold_copy_target,
            is_gather_target,
            is_tril_target,
            is_embedding_target,
            is_logit_target,
            is_index_select_target,
            is_index_tensor_target,
            is_cumsum_target,
            is_layer_norm_target,
            is_var_target,
            is_var_dim_target,
            is_plain_avg_pool2d_target,
            is_adaptive_avg_pool2d_target,
            is_as_strided_copy_target,
            is_pad_target,
            is_pixel_shuffle_target,
            is_pixel_unshuffle_target,
            is_squeeze_target,
            is_unsqueeze_target,
            is_dim_transpose_target,
            is_index_put_family_target,
            is_index_put_default_target,
            is_where_target,
            is_to_copy_target,
            is_reciprocal_target,
            is_masked_fill_target,
            is_select_scatter_target,
            is_clamp_target,
            is_abs_target,
            is_comparison_target,
            is_minmax_target,
            is_add_sub_target,
            is_mul_target,
            is_sum_dim_int_list_target,
            is_mean_target,
            is_full_like_target,
            is_batch_norm_target,
            is_repeat_interleave_self_int_target,
            is_max_pool2d_target,
            is_max_pool2d_with_indices_target,
        ];
        for fixture in [
            include_str!("../tests/fixtures/rc4_arm_edge_targets.tsv"),
            include_str!("../tests/fixtures/rc4_tosa_targets.tsv"),
        ] {
            for line in fixture.lines().filter(|line| !line.starts_with('#')) {
                let (symbol, _) = line.split_once('\t').unwrap();
                for unknown in [format!("renamed.{symbol}"), format!("{symbol}.renamed")] {
                    for predicate in predicates {
                        assert!(!predicate(&unknown), "{unknown}");
                    }
                    assert_eq!(
                        classify_ir_target(&unknown),
                        IrTargetKind::Other,
                        "{unknown}"
                    );
                    assert_eq!(zero_input_constant_target(&unknown), None, "{unknown}");
                    assert_eq!(arange_target_kind(&unknown), None, "{unknown}");
                    assert_eq!(table_op_target_kind(&unknown), None, "{unknown}");
                    assert_eq!(elu_target_kind(&unknown), None, "{unknown}");
                    assert_eq!(shape_only_constant_target(&unknown), None, "{unknown}");
                    assert_eq!(noop_target(&unknown), None, "{unknown}");
                    assert_eq!(size_adjust_target_kind(&unknown), None, "{unknown}");
                    assert_eq!(layer_norm_target_kind(&unknown), None, "{unknown}");
                    assert_eq!(layer_norm_param_view_target(&unknown), None, "{unknown}");
                    assert_eq!(add_sub_alpha_result_target(&unknown), None, "{unknown}");
                    assert_eq!(not_equal_replacement_target(&unknown), None, "{unknown}");
                    assert_eq!(scalar_shift_replacement_target(&unknown), None, "{unknown}");
                    assert_eq!(group_norm_reshape_target(&unknown), None, "{unknown}");
                }
            }
        }
    }

    #[test]
    fn rejects_near_miss_linear_targets() {
        assert_eq!(
            classify_ir_target("aten.bilinear.default"),
            IrTargetKind::Other
        );
        assert_eq!(
            classify_ir_target("aten.upsample_bilinear2d.default"),
            IrTargetKind::Other
        );
    }

    #[test]
    fn rejects_near_miss_sub_targets() {
        assert!(!is_add_sub_target("aten.rsub.Tensor"));
        assert_eq!(add_sub_alpha_result_target("aten.rsub.Tensor"), None);
    }

    #[test]
    fn rejects_near_miss_pool_targets() {
        assert!(!is_max_pool2d_target("aten.adaptive_max_pool2d.default"));
        assert!(!is_max_pool2d_with_indices_target(
            "aten.adaptive_max_pool2d_with_indices.default"
        ));
        assert_eq!(
            size_adjust_target_kind("aten.adaptive_max_pool2d.default"),
            None
        );
        assert_eq!(
            size_adjust_target_kind("aten.adaptive_avg_pool2d.default"),
            None
        );
    }

    fn assert_cases<T>(classify: impl Fn(&str) -> T, cases: &[(&str, T)])
    where
        T: Clone + std::fmt::Debug + PartialEq,
    {
        for (target, expected) in cases {
            assert_eq!(classify(target), expected.clone(), "{target}");
        }
    }

    fn assert_true(predicate: impl Fn(&str) -> bool, cases: &[&str]) {
        for target in cases {
            assert!(predicate(target), "{target}");
        }
    }

    fn assert_false(predicate: impl Fn(&str) -> bool, cases: &[&str]) {
        for target in cases {
            assert!(!predicate(target), "{target}");
        }
    }

    #[test]
    fn separates_quantize_from_dequantize() {
        assert_cases(
            classify_ir_target,
            &[
                (
                    "quantized_decomposed.quantize_per_tensor.default",
                    IrTargetKind::QuantizePerTensor,
                ),
                (
                    "quantized_decomposed.quantize_per_channel.default",
                    IrTargetKind::QuantizePerChannel,
                ),
                (
                    "quantized_decomposed.dequantize_per_tensor.default",
                    IrTargetKind::DequantizePerTensor,
                ),
                (
                    "quantized_decomposed.dequantize_per_channel.default",
                    IrTargetKind::DequantizePerChannel,
                ),
            ],
        );
    }

    #[test]
    fn recognizes_exact_tosa_rescale_symbols() {
        assert_cases(
            classify_ir_target,
            &[
                ("backend.tosa.RESCALE.default", IrTargetKind::TosaRescale),
                ("tosa.RESCALE.default", IrTargetKind::TosaRescale),
            ],
        );
    }

    #[test]
    fn recognizes_activation_and_producer_targets() {
        assert_cases(
            classify_ir_target,
            &[
                ("aten.relu.default", IrTargetKind::AtenRelu),
                ("aten.leaky_relu.default", IrTargetKind::Other),
                ("aten.relu_.default", IrTargetKind::Other),
                ("aten.hardtanh.default", IrTargetKind::AtenHardtanh),
                ("aten.convolution.default", IrTargetKind::AtenConv),
                ("aten.linear.default", IrTargetKind::AtenLinear),
            ],
        );
    }

    #[test]
    fn distinguishes_matmul_overloads() {
        assert_cases(
            classify_ir_target,
            &[
                ("aten.matmul.default", IrTargetKind::AtenMatmul),
                ("aten.mm.default", IrTargetKind::AtenMm),
                ("aten.bmm.default", IrTargetKind::AtenBmm),
                ("aten.addmm.default", IrTargetKind::AtenAddmm),
            ],
        );
    }

    #[test]
    fn recognizes_exact_control_flow_symbols() {
        assert_cases(
            classify_ir_target,
            &[
                ("cond", IrTargetKind::Cond),
                ("torch.ops.higher_order.while_loop", IrTargetKind::WhileLoop),
                ("operator.getitem", IrTargetKind::GetItem),
            ],
        );
    }

    #[test]
    fn recognizes_constant_target_families() {
        assert_cases(
            zero_input_constant_target,
            &[
                ("aten.full.default", Some(ZeroInputConstantTarget::Full)),
                ("aten.arange.start", Some(ZeroInputConstantTarget::Arange)),
                (
                    "aten.linspace.default",
                    Some(ZeroInputConstantTarget::Linspace),
                ),
                ("aten.eye.default", Some(ZeroInputConstantTarget::Eye)),
                (
                    "aten.scalar_tensor.default",
                    Some(ZeroInputConstantTarget::ScalarTensor),
                ),
            ],
        );
        assert_cases(
            arange_target_kind,
            &[
                ("aten.arange.start", Some(ArangeTargetKind::Start)),
                ("aten.arange.start_step", Some(ArangeTargetKind::StartStep)),
            ],
        );
        assert_cases(
            shape_only_constant_target,
            &[
                (
                    "aten.reshape.default",
                    Some(ShapeOnlyConstantTarget::Reshape),
                ),
                (
                    "aten.view_copy.default",
                    Some(ShapeOnlyConstantTarget::Reshape),
                ),
                (
                    "aten.unsqueeze_copy.default",
                    Some(ShapeOnlyConstantTarget::Unsqueeze),
                ),
                (
                    "aten.permute_copy.default",
                    Some(ShapeOnlyConstantTarget::Permute),
                ),
                ("aten.slice_copy.Tensor", None),
            ],
        );
        assert!(is_zero_input_constant_target("aten.eye.default"));
        assert!(is_zero_input_constant_target("aten.scalar_tensor.default"));
        assert!(is_to_copy_target("aten._to_copy.default"));
        assert!(is_to_copy_target(
            "dim_order_ops._to_dim_order_copy.default"
        ));
        assert!(!is_to_copy_target("aten.copy.default"));
        assert!(is_reciprocal_target("aten.reciprocal.default"));
        assert!(!is_reciprocal_target("aten.rsqrt.default"));
        assert_true(
            is_fold_qdq_constant_target,
            &["aten.scalar_tensor.default", "aten.full_like.default"],
        );
        assert_false(is_fold_qdq_constant_target, &["aten.reshape.default"]);
        assert!(is_shape_only_constant_target("aten.permute_copy.default"));
        assert!(is_view_like_target("aten._unsafe_view.default"));
        assert!(is_permute_target("aten.permute.default"));
    }

    #[test]
    fn recognizes_table_op_target_families() {
        assert_cases(
            table_op_target_kind,
            &[
                ("aten.exp.default", Some(TableOpTargetKind::Unary("exp"))),
                ("aten.sin.default", Some(TableOpTargetKind::Unary("sin"))),
                ("aten.gelu.default", Some(TableOpTargetKind::Gelu)),
                (
                    "aten.pow.Tensor_Scalar",
                    Some(TableOpTargetKind::PowTensorScalar),
                ),
                (
                    "aten.remainder.Scalar",
                    Some(TableOpTargetKind::RemainderScalar),
                ),
                (
                    "aten.elu.default",
                    Some(TableOpTargetKind::Elu(EluTargetKind::Elu)),
                ),
                (
                    "aten.selu.default",
                    Some(TableOpTargetKind::Elu(EluTargetKind::Selu)),
                ),
                (
                    "aten.celu.default",
                    Some(TableOpTargetKind::Elu(EluTargetKind::Celu)),
                ),
                (
                    "aten.leaky_relu.default",
                    Some(TableOpTargetKind::LeakyRelu),
                ),
                ("aten.softplus.default", Some(TableOpTargetKind::Softplus)),
                ("aten.relu.default", None),
                ("aten.add.Tensor", None),
            ],
        );
    }

    #[test]
    fn recognizes_noop_target_families() {
        assert_cases(
            noop_target,
            &[
                ("aten.clone.default", Some(NoopTarget::PassthroughFirstArg)),
                (
                    "_clone_dim_order.default",
                    Some(NoopTarget::PassthroughFirstArg),
                ),
                ("alias_copy.default", Some(NoopTarget::PassthroughFirstArg)),
                ("detach_copy.default", Some(NoopTarget::PassthroughFirstArg)),
                ("aten.copy.default", Some(NoopTarget::CopySourceArg)),
                (
                    "_to_dim_order_copy.default",
                    Some(NoopTarget::DtypeCheckedPassthrough),
                ),
                (
                    "aten._to_copy.default",
                    Some(NoopTarget::DtypeCheckedPassthrough),
                ),
                ("aten.as_strided_copy.default", None),
            ],
        );
    }

    #[test]
    fn recognizes_layout_target_families() {
        assert_true(
            is_layout_passthrough_target,
            &[
                "aten.transpose.int",
                "aten.contiguous.default",
                "backend.tosa.RESHAPE.default",
                "tosa.RESHAPE.default",
            ],
        );
        assert!(is_tosa_transpose_target("backend.tosa.TRANSPOSE.default"));
        assert!(is_rank_changing_layout_target("aten.flatten.using_ints"));
        assert_false(is_rank_changing_layout_target, &["aten.transpose.int"]);
        assert_false(
            is_layout_passthrough_target,
            &[
                "backend.tosa.TRANSPOSE_CONV2D.default",
                "backend.tosa.CONV2D_BLOCK_SCALED.default",
            ],
        );
    }

    #[test]
    fn recognizes_convolution_target_families() {
        assert_true(
            is_raw_aten_convolution_target,
            &[
                "aten.convolution.default",
                "edge__ops_aten_convolution.default",
            ],
        );
        assert!(is_aten_convolution_default_target(
            "aten.convolution.default"
        ));
        assert!(is_conv1d_default_target("aten.conv1d.default"));
        assert!(is_conv2d_default_target("aten.conv2d.default"));
        assert!(is_conv3d_default_target("aten.conv3d.default"));
        assert_cases(
            size_adjust_target_kind,
            &[
                ("aten.conv2d.default", Some(SizeAdjustTargetKind::Conv)),
                (
                    "aten.avg_pool2d.default",
                    Some(SizeAdjustTargetKind::Pool(PoolSizeAdjustKind::Avg)),
                ),
                (
                    "aten.max_pool2d.default",
                    Some(SizeAdjustTargetKind::Pool(PoolSizeAdjustKind::Max)),
                ),
                ("aten.transpose_conv2d.input", None),
            ],
        );
    }

    #[test]
    fn recognizes_decomposition_target_families() {
        assert_true(
            is_split_target,
            &[
                "aten.split_with_sizes_copy.default",
                "aten.split_copy.Tensor",
            ],
        );
        assert!(is_slice_copy_tensor_target("aten.slice_copy.Tensor"));
        assert_true(
            is_slice_tensor_target,
            &["aten.slice_copy.Tensor", "aten.slice.Tensor"],
        );
        assert_false(is_slice_tensor_target, &["aten.slice_scatter.default"]);
        assert!(is_slice_scatter_target("aten.slice_scatter.default"));
        assert_true(is_cat_target, &["aten.cat.default"]);
        assert_true(is_unfold_copy_target, &["aten.unfold_copy.default"]);
        assert_true(is_gather_target, &["aten.gather.default"]);
        assert_true(is_tril_target, &["aten.tril_copy.default"]);
        assert_true(is_embedding_target, &["aten.embedding.default"]);
        assert_true(is_logit_target, &["aten.logit.default"]);
        assert_true(is_index_select_target, &["aten.index_select.default"]);
        assert_true(is_index_tensor_target, &["aten.index.Tensor"]);
        assert_true(is_cumsum_target, &["aten.cumsum.default"]);
        assert!(is_layer_norm_target("aten.native_layer_norm.default"));
        assert_cases(
            layer_norm_target_kind,
            &[
                ("aten.layer_norm.default", Some(LayerNormTargetKind::Plain)),
                (
                    "aten.native_layer_norm.default",
                    Some(LayerNormTargetKind::NativeOrOther),
                ),
            ],
        );
        assert_cases(
            layer_norm_param_view_target,
            &[
                ("aten.layer_norm.default", Some("aten.reshape.default")),
                (
                    "aten.native_layer_norm.default",
                    Some("aten.view_copy.default"),
                ),
            ],
        );
        assert_true(is_var_target, &["aten.var.correction"]);
        assert_true(is_var_dim_target, &["aten.var.dim"]);
        assert!(is_plain_avg_pool2d_target("aten.avg_pool2d.default"));
        assert_false(
            is_plain_avg_pool2d_target,
            &["aten.adaptive_avg_pool2d.default"],
        );
        assert_true(
            is_adaptive_avg_pool2d_target,
            &["aten.adaptive_avg_pool2d.default"],
        );
        assert_true(is_as_strided_copy_target, &["aten.as_strided_copy.default"]);
        assert_true(
            is_pad_target,
            &["aten.pad.default", "aten.constant_pad_nd.default"],
        );
        assert!(is_pixel_shuffle_target("aten.pixel_shuffle.default"));
        assert_false(is_pixel_shuffle_target, &["aten.pixel_unshuffle.default"]);
        assert!(is_pixel_unshuffle_target("aten.pixel_unshuffle.default"));
        assert_true(is_squeeze_target, &["aten.squeeze.dim"]);
        assert_true(is_unsqueeze_target, &["aten.unsqueeze.default"]);
        assert_true(is_dim_transpose_target, &["aten.transpose.int"]);
    }

    #[test]
    fn recognizes_index_mask_and_clamp_targets() {
        assert_true(
            is_index_put_family_target,
            &["aten.index_put.default", "aten.index_put_.default"],
        );
        assert_true(
            is_index_put_default_target,
            &["aten.index_put.default", "aten.index_put_.default"],
        );
        assert_false(is_index_put_default_target, &["aten.index.Tensor"]);
        assert_true(is_where_target, &["aten.where.self", "aten_where.self"]);
        assert_true(is_masked_fill_target, &["aten.masked_fill.Scalar"]);
        assert_true(is_select_scatter_target, &["aten.select_scatter.default"]);
        assert_true(is_clamp_target, &["aten.clamp.default"]);
        assert_false(is_clamp_target, &["aten.clamp.Tensor"]);
    }

    #[test]
    fn recognizes_arithmetic_target_families() {
        assert!(is_abs_target("aten.abs.default"));
        assert_true(
            is_comparison_target,
            &[
                "aten.eq.Tensor",
                "aten.ge.Tensor",
                "aten.gt.Tensor",
                "aten.le.Tensor",
                "aten.lt.Tensor",
            ],
        );
        assert_true(
            is_minmax_target,
            &["aten.maximum.default", "aten.minimum.default"],
        );
        assert_true(is_add_sub_target, &["aten.add.Tensor", "aten.sub.Tensor"]);
        assert_cases(
            add_sub_alpha_result_target,
            &[
                ("aten.add.Tensor", Some("aten.add.Tensor")),
                ("aten.sub.Tensor", Some("aten.sub.Tensor")),
            ],
        );
        assert_true(is_mul_target, &["aten.mul.Tensor"]);
        assert_true(is_sum_dim_int_list_target, &["aten.sum.dim_IntList"]);
        assert_true(is_mean_target, &["aten.mean.dim", "aten.mean.default"]);
        assert_true(is_full_like_target, &["aten.full_like.default"]);
        assert_false(is_add_sub_target, &["aten.add.Scalar"]);
        assert_cases(
            not_equal_replacement_target,
            &[
                ("aten.ne.Scalar", Some("aten.eq.Scalar")),
                ("aten.ne.Tensor", Some("aten.eq.Tensor")),
                ("aten.ne_.Tensor", Some("aten.eq.Tensor")),
            ],
        );
        assert_cases(
            scalar_shift_replacement_target,
            &[
                (
                    "aten.__lshift__.Scalar",
                    Some("aten.bitwise_left_shift.Tensor"),
                ),
                (
                    "aten.__rshift__.Scalar",
                    Some("aten.bitwise_right_shift.Tensor"),
                ),
            ],
        );
        assert_true(
            is_repeat_interleave_self_int_target,
            &["aten.repeat_interleave.self_int"],
        );
        assert_true(is_max_pool2d_target, &["aten.max_pool2d.default"]);
        assert_true(
            is_max_pool2d_with_indices_target,
            &["aten.max_pool2d_with_indices.default"],
        );
        assert_cases(
            group_norm_reshape_target,
            &[
                ("aten.group_norm.default", Some("aten.reshape.default")),
                (
                    "aten.native_group_norm.default",
                    Some("aten.view_copy.default"),
                ),
            ],
        );
        assert_false(
            is_max_pool2d_with_indices_target,
            &["aten.max_pool2d.default"],
        );
    }
}
