use crate::fx_control_flow_metadata;
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
use crate::fx_fold_qdq;
use crate::fx_fuse_constant_args;
use crate::fx_fuse_duplicate_users;
use crate::fx_insert_rescale;
use crate::fx_insert_table_ops;
use crate::fx_memory_format;
use crate::fx_pass::FxPass;
use crate::fx_replace_inf_values_ir;
use crate::fx_replace_scalar_with_tensor_ir;
use crate::fx_remove_noop_ir;
use crate::fx_rewrite_matmul;
use crate::fx_rewrite_squeeze_ir;

struct ExportedProgramClones {
    fold_qdq: Option<pyo3::PyObject>,
    fuse: Option<pyo3::PyObject>,
}

impl ExportedProgramClones {
    fn new(exported_program: Option<pyo3::PyObject>) -> Self {
        pyo3::Python::with_gil(|py| {
            let fuse = exported_program.as_ref().map(|value| value.clone_ref(py));
            Self {
                fold_qdq: exported_program,
                fuse,
            }
        })
    }
}

/// Build the hybrid pipeline: IrPasses run on persistent IrGraph, FxPasses
/// cause a flush/reimport cycle. Same pass ordering as the legacy FX path.
pub fn build_hybrid_pipeline(
    _exported_program: Option<pyo3::PyObject>,
) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Ir, IrEp};

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
        IrEp(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEpPass::matmul_only(),
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
        IrEp(Box::new(
            crate::fx_insert_table_ops_ir::IrInsertTableOpsEpPass,
        )),
        // -- ATen -> TOSA --
        Ir(Box::new(
            crate::fx_decompose_grouped_conv_ir::IrDecomposeGroupedConvPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEpPass::with_data_slices(),
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
        IrEp(Box::new(crate::fx_cast_int64_ir::IrCastInt64ToInt32EpPass)),
        Ir(Box::new(
            crate::fx_normalize_index_put_none_indices_ir::IrNormalizeIndexPutNoneIndicesPass,
        )),
        Ir(Box::new(
            crate::fx_normalize_index_put_bool_mask_ir::IrNormalizeIndexPutBoolMaskPass,
        )),
        Ir(Box::new(
            crate::fx_rewrite_index_put_ir::IrRewriteIndexPutPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEpPass,
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
        IrEp(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEpPass::matmul_only(),
        )),
    ]
}

/// Build the FX pipeline for while_loop submodule graphs.
///
/// Mirrors the cond pipeline but uses Carried rescale mode: output[i] must
/// match placeholder[i] because while_loop body outputs feed back as inputs.
pub fn build_fx_while_submodule_pipeline(
    exported_program: Option<pyo3::PyObject>,
) -> Vec<Box<dyn FxPass>> {
    let ep = ExportedProgramClones::new(exported_program);
    vec![
        // Strip boundary Q nodes from placeholders before fold_qdq.
        // Loop-carried values are already in the placeholder's domain.
        Box::new(fx_control_flow_metadata::FxWhileStripBoundaryQPass),
        // Q/DQ folding + rescale
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(ep.fold_qdq)),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::carried()),
        // TABLE lowering for quantized math ops
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        // Cleanup
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(ep.fuse)),
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_remove_noop_ir::IrRemoveNoopPass,
        )),
        // Second rescale pass
        Box::new(fx_insert_rescale::FxInsertRescalePass::carried()),
        // Layout
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        // Final rescale: fix chains broken by TABLE/layout transposes
        Box::new(fx_insert_rescale::FxInsertRescalePass::carried()),
    ]
}

/// Extended submodule pipeline for cond branches.
/// Includes fold_qdq + the core passes. Decomposition passes are excluded
/// because they create new ops that break submodule boundary contracts.
/// IR-native control-flow submodule pipelines (round-trip elimination Phase 4c):
/// the same transforms as build_fx_*_submodule_pipeline, but as IrPasses on a
/// persistent submodule IrGraph instead of Python FxPasses. FxInsertRescale maps
/// to IrInsertQdqRescale + IrInsertInt32Rescale; the loop-carried variant prepends
/// IrStampWhileBodyOutputQParams. Used via run_hybrid_pipeline_ext under ARM_CF_IR.
fn rescale_steps() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    vec![
        Ir(Box::new(crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass)),
        Ir(Box::new(crate::fx_insert_int32_rescale_ir::IrInsertInt32RescalePass)),
    ]
}

fn submodule_fold_and_fuse() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Ir, IrEp};
    vec![
        IrEp(Box::new(crate::fx_insert_table_ops_ir::IrInsertTableOpsEpPass)),
        IrEp(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEpPass,
        )),
        IrEp(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEpPass::matmul_only(),
        )),
        Ir(Box::new(crate::fx_remove_noop_ir::IrRemoveNoopPass)),
    ]
}

pub fn build_ir_while_submodule_pipeline() -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    // Mirror the main hybrid pipeline ordering: rescale once (after fold), before
    // table/layout. The IR rescale passes are not idempotent, so they run once.
    let mut steps: Vec<crate::fx_ir::PipelineStep> = vec![
        Ir(Box::new(
            crate::fx_control_flow_metadata::IrStripWhileBoundaryQPass,
        )),
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
        crate::fx_control_flow_metadata::IrInsertCarriedOutputRescalePass,
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

pub fn build_fx_cond_submodule_pipeline(
    exported_program: Option<pyo3::PyObject>,
) -> Vec<Box<dyn FxPass>> {
    let ep = ExportedProgramClones::new(exported_program);
    vec![
        // Q/DQ folding + rescale (the core quantization passes)
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(ep.fold_qdq)),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        // TABLE lowering for quantized math ops
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        // Cleanup
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(ep.fuse)),
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_remove_noop_ir::IrRemoveNoopPass,
        )),
        // Matmul rescale + second rescale pass
        Box::new(fx_rewrite_matmul::FxRewriteMatmulRescalePass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        // Layout
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        // Final rescale repair: re-run after TABLE + memory format to fix
        // rescale chains that were broken by TABLE node insertion or
        // layout transposes (especially for buffer/captured args).
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
    ]
}
