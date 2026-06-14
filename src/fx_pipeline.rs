use crate::fx_convert_clamp_ir;
use crate::fx_convert_full_like_ir;
use crate::fx_convert_split_to_slice_ir;
use crate::fx_convert_sum_ir;
use crate::fx_decompose_addmm_ir;
use crate::fx_decompose_as_strided_copy_ir;
use crate::fx_decompose_cumsum_ir;
use crate::fx_decompose_logit_ir;
use crate::fx_decompose_masked_fill_ir;
use crate::fx_decompose_ne_ir;
use crate::fx_decompose_pixel_shuffle_ir;
use crate::fx_decompose_repeat_interleave_ir;
use crate::fx_decompose_tril_ir;
use crate::fx_expand_where_mask_ir;
use crate::fx_remove_noop_ir;
use crate::fx_replace_inf_values_ir;
use crate::fx_replace_scalar_with_tensor_ir;
use crate::fx_rewrite_squeeze_ir;

/// Build the lowering pipeline over the persistent IrGraph.
pub fn build_hybrid_pipeline(
    _exported_program: Option<pyo3::PyObject>,
) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Effectful, Ir};

    vec![
        // -- Early decompositions (before fold_qdq) --
        Ir(Box::new(
            crate::fx_annotate_output_dim_order_ir::IrAnnotateOutputDimOrderPass,
        )),
        Ir(Box::new(crate::fx_decompose_ops_ir::IrDecomposeOpsPass)),
        Ir(Box::new(
            crate::fx_decorate_fp_int_cast_ir::IrDecorateFpToIntCastPass,
        )),
        // -- Pre-fold --
        Ir(Box::new(
            crate::fx_fuse_quant_activation_ir::IrFuseQuantActivationPass,
        )),
        Ir(Box::new(fx_convert_clamp_ir::IrConvertClampPass)),
        Ir(Box::new(
            crate::fx_decompose_layernorm_ir::IrDecomposeLayerNormPass,
        )),
        Ir(Box::new(crate::fx_decompose_var_ir::IrDecomposeVarPass)),
        // -- Fold Q/DQ + rescale --
        Ir(Box::new(crate::fx_fold_qdq_ir::IrFoldQdqPass)),
        Ir(Box::new(
            crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_linear_ir::IrDecomposeLinearPass,
        )),
        Ir(Box::new(
            crate::fx_conv1d_unsqueeze_ir::IrConv1dUnsqueezePass,
        )),
        Ir(Box::new(
            crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass,
        )),
        Ir(Box::new(
            crate::fx_insert_int32_rescale_ir::IrInsertInt32RescalePass,
        )),
        Ir(Box::new(crate::fx_decompose_quant_ir::IrDecomposeQuantPass)),
        // -- Post-fold: long run of IrPasses (no Python roundtrip) --
        Ir(Box::new(
            fx_convert_split_to_slice_ir::IrConvertSplitToSlicePass,
        )),
        Ir(Box::new(
            fx_replace_inf_values_ir::IrReplaceInfAndLimitValuesPass,
        )),
        Ir(Box::new(
            fx_decompose_masked_fill_ir::IrDecomposeMaskedFillPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_embedding_ir::IrDecomposeEmbeddingPass,
        )),
        Ir(Box::new(
            crate::fx_canonicalize_gather_ir::IrCanonicalizeGatherPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_index_tensor_ir::IrDecomposeIndexTensorPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_index_select_ir::IrDecomposeIndexSelectPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_unfold_ir::IrDecomposeUnfoldPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_select_scatter_ir::IrDecomposeSelectScatterPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_slice_scatter_ir::IrDecomposeSliceScatterPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_strided_slice_ir::IrDecomposeStridedSlicePass,
        )),
        Ir(Box::new(fx_expand_where_mask_ir::IrExpandWhereMaskPass)),
        Ir(Box::new(fx_decompose_tril_ir::IrDecomposeTrilPass)),
        Ir(Box::new(fx_decompose_logit_ir::IrDecomposeLogitPass)),
        Ir(Box::new(fx_decompose_ne_ir::IrDecomposeNotEqualPass)),
        Ir(Box::new(
            fx_replace_scalar_with_tensor_ir::IrReplaceScalarWithTensorPass,
        )),
        Ir(Box::new(fx_decompose_addmm_ir::IrDecomposeAddmmPass)),
        Ir(Box::new(fx_decompose_cumsum_ir::IrDecomposeCumsumPass)),
        Ir(Box::new(
            fx_decompose_repeat_interleave_ir::IrDecomposeRepeatInterleavePass,
        )),
        Ir(Box::new(
            fx_decompose_pixel_shuffle_ir::IrDecomposePixelShufflePass,
        )),
        Ir(Box::new(
            fx_decompose_as_strided_copy_ir::IrDecomposeAsStridedCopyPass,
        )),
        Ir(Box::new(
            crate::fx_quantize_clamp_args_ir::IrQuantizeClampArgsPass,
        )),
        Ir(Box::new(fx_convert_full_like_ir::IrConvertFullLikePass)),
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::matmul_only(),
        )),
        Ir(Box::new(
            crate::fx_decompose_adaptive_avgpool_ir::IrDecomposeAdaptiveAvgPoolPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_avgpool_pad_ir::IrDecomposeAvgPoolPadPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_maxpool_dilation_ir::IrDecomposeMaxPoolDilationPass,
        )),
        Ir(Box::new(crate::fx_decompose_var_ir::IrDecomposeVarPass)),
        Ir(Box::new(crate::fx_decompose_mean_ir::IrDecomposeMeanPass)),
        Ir(Box::new(fx_convert_sum_ir::IrConvertSumPass)),
        Effectful(Box::new(
            crate::fx_insert_table_ops_ir::IrInsertTableOpsEffectPass,
        )),
        // -- ATen -> TOSA --
        Ir(Box::new(
            crate::fx_decompose_grouped_conv_ir::IrDecomposeGroupedConvPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::with_data_slices(),
        )),
        Ir(Box::new(
            crate::fx_annotate_conv_bias_ir::IrAnnotateConvBiasDtypePass,
        )),
        Ir(Box::new(
            crate::fx_size_adjust_input_ir::IrSizeAdjustInputPass,
        )),
        Ir(Box::new(fx_rewrite_squeeze_ir::IrRewriteSqueezePass)),
        Ir(Box::new(crate::fx_rewrite_pad_ir::IrRewritePadPass)),
        // -- Postprocessing --
        Effectful(Box::new(
            crate::fx_cast_int64_ir::IrCastInt64ToInt32EffectPass,
        )),
        Ir(Box::new(
            crate::fx_normalize_index_put_none_indices_ir::IrNormalizeIndexPutNoneIndicesPass,
        )),
        Ir(Box::new(
            crate::fx_normalize_index_put_bool_mask_ir::IrNormalizeIndexPutBoolMaskPass,
        )),
        Ir(Box::new(
            crate::fx_rewrite_index_put_ir::IrRewriteIndexPutPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass,
        )),
        Ir(Box::new(
            crate::fx_memory_format_ir::IrToTosaMemoryFormatPass,
        )),
        Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        Ir(Box::new(
            crate::fx_rewrite_matmul_ir::IrRewriteMatmulRescalePass,
        )),
        Ir(Box::new(
            crate::fx_insert_conv_rescale_ir::IrInsertConvAccumulatorRescalePass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::matmul_only(),
        )),
    ]
}

/// IR-native control-flow submodule pipelines: transforms run as IrPasses on a
/// persistent submodule IrGraph instead of Python FxPasses.
fn rescale_steps() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    vec![
        Ir(Box::new(
            crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass,
        )),
        Ir(Box::new(
            crate::fx_insert_int32_rescale_ir::IrInsertInt32RescalePass,
        )),
    ]
}

fn submodule_fold_and_fuse() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Effectful, Ir};
    vec![
        Effectful(Box::new(
            crate::fx_insert_table_ops_ir::IrInsertTableOpsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::matmul_only(),
        )),
        Ir(Box::new(crate::fx_remove_noop_ir::IrRemoveNoopPass)),
    ]
}

pub fn build_ir_while_submodule_pipeline() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    // Mirror the main hybrid pipeline ordering: rescale once (after fold), before
    // table/layout. The IR rescale passes are not idempotent, so they run once.
    let mut steps: Vec<crate::fx_ir::PipelineStep> = vec![
        Ir(Box::new(crate::control_flow::IrStripWhileBoundaryQPass)),
        Ir(Box::new(crate::fx_fold_qdq_ir::IrFoldQdqPass)),
        Ir(Box::new(
            crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass,
        )),
    ];
    steps.extend(rescale_steps());
    steps.extend(submodule_fold_and_fuse());
    steps.push(Ir(Box::new(
        crate::fx_memory_format_ir::IrToTosaMemoryFormatPass,
    )));
    // Loop-carried output alignment runs last, once the producer scales are final.
    steps.push(Ir(Box::new(
        crate::control_flow::IrInsertCarriedOutputRescalePass,
    )));
    steps
}

pub fn build_ir_cond_submodule_pipeline() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    let mut steps: Vec<crate::fx_ir::PipelineStep> = vec![
        Ir(Box::new(crate::fx_fold_qdq_ir::IrFoldQdqPass)),
        Ir(Box::new(
            crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass,
        )),
    ];
    steps.extend(rescale_steps());
    steps.extend(submodule_fold_and_fuse());
    steps.push(Ir(Box::new(
        crate::fx_rewrite_matmul_ir::IrRewriteMatmulRescalePass,
    )));
    steps.push(Ir(Box::new(
        crate::fx_memory_format_ir::IrToTosaMemoryFormatPass,
    )));
    steps
}
