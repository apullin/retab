use crate::fx_annotate_output_dim_order;
use crate::fx_canonicalize_gather;
use crate::fx_cast_int64;
use crate::fx_conv1d_unsqueeze;
use crate::fx_control_flow_metadata;
use crate::fx_convert_clamp;
use crate::fx_convert_full_like;
use crate::fx_convert_split_to_slice;
use crate::fx_convert_sum;
use crate::fx_decompose_adaptive_avgpool;
use crate::fx_decompose_addmm;
use crate::fx_decompose_as_strided_copy;
use crate::fx_decompose_cumsum;
use crate::fx_decompose_grouped_conv;
use crate::fx_decompose_index_select;
use crate::fx_decompose_index_tensor;
use crate::fx_decompose_linear;
use crate::fx_decompose_logit;
use crate::fx_decompose_masked_fill;
use crate::fx_decompose_maxpool_dilation;
use crate::fx_decompose_mean;
use crate::fx_decompose_ops;
use crate::fx_decompose_pixel_shuffle;
use crate::fx_decompose_quant;
use crate::fx_decompose_repeat_interleave;
use crate::fx_decompose_select_scatter;
use crate::fx_decompose_strided_slice;
use crate::fx_decompose_tril;
use crate::fx_decompose_unfold;
use crate::fx_expand_where_mask;
use crate::fx_fold_qdq;
use crate::fx_fuse_constant_args;
use crate::fx_fuse_duplicate_users;
use crate::fx_fuse_equal_placeholders;
use crate::fx_fuse_quant_activation;
use crate::fx_insert_rescale;
use crate::fx_insert_table_ops;
use crate::fx_memory_format;
use crate::fx_pass::FxPass;
use crate::fx_quantize_clamp_args;
use crate::fx_remove_noop;
use crate::fx_rewrite_conv;
use crate::fx_rewrite_index_put;
use crate::fx_rewrite_matmul;
use crate::fx_convert_clamp_ir;
use crate::fx_convert_full_like_ir;
use crate::fx_convert_split_to_slice_ir;
use crate::fx_convert_sum_ir;
use crate::fx_decompose_addmm_ir;
use crate::fx_decompose_as_strided_copy_ir;
use crate::fx_decompose_cumsum_ir;
use crate::fx_decompose_logit_ir;
use crate::fx_decompose_masked_fill_ir;
use crate::fx_decompose_pixel_shuffle_ir;
use crate::fx_decompose_repeat_interleave_ir;
use crate::fx_decompose_tril_ir;
use crate::fx_expand_where_mask_ir;
use crate::fx_remove_noop_ir;
use crate::fx_rewrite_squeeze_ir;
use crate::fx_size_adjust_input;

struct ExportedProgramClones {
    fold_qdq: Option<pyo3::PyObject>,
    fuse: Option<pyo3::PyObject>,
    fuse_grouped: Option<pyo3::PyObject>,
    cast: Option<pyo3::PyObject>,
    fuse_placeholders: Option<pyo3::PyObject>,
    fuse_late: Option<pyo3::PyObject>,
}

impl ExportedProgramClones {
    fn new(fold_qdq: Option<pyo3::PyObject>) -> Self {
        pyo3::Python::with_gil(|py| {
            let clone_ep =
                |ep: &Option<pyo3::PyObject>| ep.as_ref().map(|value| value.clone_ref(py));
            let fuse = clone_ep(&fold_qdq);
            let fuse_grouped = clone_ep(&fold_qdq);
            let cast = clone_ep(&fold_qdq);
            let fuse_placeholders = clone_ep(&fold_qdq);
            let fuse_late = clone_ep(&fold_qdq);
            Self {
                fold_qdq,
                fuse,
                fuse_grouped,
                cast,
                fuse_placeholders,
                fuse_late,
            }
        })
    }
}

/// Build the full FX-direct lowering pipeline in execution order.
///
/// The ordering mirrors the Python `ArmPassManager._tosa_pipeline()`:
///
///   Pre-fold:
///    1. Fuse quantized activations (relu between conv and Q)
///    2. Convert relu/hardtanh to clamp (before Q/DQ fold)
///
///   Fold Q/DQ + rescale:
///    3. **Rust** fold Q/DQ and annotate QuantArgs namedtuples in node.meta
///    4. Fuse duplicate users (deduplicate identical call_function nodes)
///    5. Decompose linear into reshape+conv2d+reshape
///    6. Unsqueeze conv1d to conv2d + view envelope
///    7. **Rust** insert rescales (DQ->Q pairs + INT32 elementwise rescales)
///    8. Decompose remaining Q/DQ nodes into primitive arithmetic
///
///   Post-fold decompositions:
///    9. Quantize clamp min/max args to match output quantization scale
///   10. Fuse constant args (evaluate ops with all-constant inputs)
///   11. Convert sum to chained TOSA REDUCE_SUM ops
///   12. Insert TABLE ops (quantised lookup tables for rsqrt, exp, etc.)
///
///   ATen -> TOSA conversion:
///   13. Rewrite conv2d to TOSA CONV2D/DEPTHWISE_CONV2D
///
///   Postprocessing/cleanup (matches Python ordering):
///   14. Cast int64 buffers to int32
///   15. Fuse equal placeholders (dedup identical constant tensors)
///   16. Annotate TOSA memory format (dim_order, spatial_rank, transposes)
///   17. Remove no-op nodes (identity, alias_copy, detach_copy, etc.)
///   18. Final rescale insertion (InsertRescalePass)
///   19. Late fold of generated zero-input FULL nodes only
pub fn build_fx_pipeline(exported_program: Option<pyo3::PyObject>) -> Vec<Box<dyn FxPass>> {
    let ExportedProgramClones {
        fold_qdq,
        fuse,
        fuse_grouped,
        cast,
        fuse_placeholders,
        fuse_late,
    } = ExportedProgramClones::new(exported_program);

    vec![
        // -- Early decompositions (before fold_qdq) --
        Box::new(fx_annotate_output_dim_order::FxAnnotateOutputDimOrderPass),
        // Decompose atanh, hardsigmoid, stack, unbind into primitive ops
        // that TOSA can represent. (masked_fill handled post-fold below.)
        Box::new(fx_decompose_ops::FxDecomposeOpsPass),
        // -- Pre-fold --
        Box::new(fx_fuse_quant_activation::FxFuseQuantizedActivationPass),
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_convert_clamp_ir::IrConvertClampPass,
        )),
        // -- Fold Q/DQ + rescale --
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(fold_qdq)),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_decompose_linear::FxDecomposeLinearPass),
        Box::new(fx_conv1d_unsqueeze::FxConv1dUnsqueezePass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        Box::new(fx_decompose_quant::FxDecomposeQuantNodesPass),
        // -- Post-fold decompositions (matches Python ordering) --
        // Convert split_with_sizes_copy → multiple slice_copy ops
        // Python runs ConvertSplitToSlicePass post-fold (line 281)
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_convert_split_to_slice_ir::IrConvertSplitToSlicePass,
        )),
        // Decompose masked_fill -> where(mask, quantized_full, x).  Runs after
        // fold_qdq so the fill scalar can be quantized with output qparams.
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_masked_fill_ir::IrDecomposeMaskedFillPass)),
        // Mirror the Python Arm backend's gather canonicalization:
        // reshape/permute/cast into backend.tosa.GATHER-compatible form.
        Box::new(fx_canonicalize_gather::FxCanonicalizeGatherPass),
        // Lower supported single-index index.Tensor cases directly to
        // backend.tosa.GATHER before ingestion/serialization.
        Box::new(fx_decompose_index_tensor::FxDecomposeIndexTensorPass),
        // Mirror the Python Arm backend's index_select lowering:
        // view + unsqueeze + expand + backend.tosa.GATHER + view.
        Box::new(fx_decompose_index_select::FxDecomposeIndexSelectToGatherPass),
        // Mirror the Python Arm backend's unfold_copy lowering:
        // view + arange/index-build + backend.tosa.GATHER + view/permute.
        // Runs before constant folding so the arange/index scaffolding
        // collapses into constants instead of leaking unsupported ops.
        Box::new(fx_decompose_unfold::FxDecomposeUnfoldToGatherPass),
        // Mirror the Python Arm backend's select_scatter lowering:
        // arange + view + eq + unsqueeze + expand + where.
        Box::new(fx_decompose_select_scatter::FxDecomposeSelectScatterPass),
        // Mirror the Python Arm backend's strided slice_copy lowering:
        // unit-step slice + optional pad + view + slice + view.
        Box::new(fx_decompose_strided_slice::FxDecomposeStridedSlicePass),
        // Expand lower-rank where/select masks to the full data rank so the
        // direct TOSA SELECT emission sees shape-aligned inputs.
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_expand_where_mask_ir::IrExpandWhereMaskPass,
        )),
        // Mirror the Python Arm backend's tril lowering:
        // arange + view + add(diagonal) + ge + where / bitwise_and.
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_tril_ir::IrDecomposeTrilPass)),
        // Decompose logit -> clamp + mul + add + reciprocal + log
        // Python runs DecomposeLogitPass post-fold (line 285 in arm_pass_manager.py)
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_decompose_logit_ir::IrDecomposeLogitPass,
        )),
        // Decompose addmm -> mm + mul + add
        // Python runs DecomposeAddmmPass post-fold (line 297 in arm_pass_manager.py)
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_decompose_addmm_ir::IrDecomposeAddmmPass,
        )),
        // Decompose cumsum into chained slice_copy + add + cat
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_cumsum_ir::IrDecomposeCumsumPass)),
        // Decompose repeat_interleave(self_int) into unsqueeze+repeat+view
        // so direct lowering reuses the same repeat grammar as the Python path.
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_repeat_interleave_ir::IrDecomposeRepeatInterleavePass)),
        // Decompose pixel_shuffle/pixel_unshuffle into view+permute+view
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_pixel_shuffle_ir::IrDecomposePixelShufflePass)),
        // Replace contiguous zero-offset as_strided_copy with view_copy,
        // matching the Python Arm pass and avoiding a raw unsupported ATen op.
        Box::new(crate::fx_ir::IrPassAdapter::new(crate::fx_decompose_as_strided_copy_ir::IrDecomposeAsStridedCopyPass)),
        Box::new(fx_quantize_clamp_args::FxQuantizeClampArgsPass),
        // Convert full_like -> full with explicit shape
        // Python runs ConvertFullLikeToFullPass in scalars section (line 330)
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_convert_full_like_ir::IrConvertFullLikePass,
        )),
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(fuse)),
        // Decompose adaptive_avg_pool2d → avg_pool2d (or slice+pool+cat)
        Box::new(fx_decompose_adaptive_avgpool::FxDecomposeAdaptiveAvgPoolPass),
        // Decompose dilated max_pool2d into pad + space-to-batch + plain pool
        // before size adjustment and memory-format annotation.
        Box::new(fx_decompose_maxpool_dilation::FxDecomposeMaxPoolDilationPass),
        // Decompose mean.dim into sum*mul with keepdim=True + reshape
        Box::new(fx_decompose_mean::FxDecomposeMeanPass),
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_convert_sum_ir::IrConvertSumPass,
        )),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        // -- ATen -> TOSA conversion --
        // Decompose grouped convolutions (groups>1, not depthwise) into
        // per-group slice_copy -> conv(groups=1) -> cat before TOSA rewrite.
        Box::new(fx_decompose_grouped_conv::FxDecomposeGroupedConvPass),
        // Grouped-conv decomposition creates fresh slice_copy nodes over
        // constant weights and biases. Fold them immediately so each per-group
        // conv sees real constant tensors instead of a shared base weight.
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(
            fuse_grouped,
        )),
        Box::new(fx_rewrite_conv::FxRewriteConvPass),
        // Adjust input sizes for conv/pool to yield exact integer output dims
        Box::new(fx_size_adjust_input::FxSizeAdjustInputPass),
        // Rewrite squeeze/unsqueeze to view_copy with explicit shape,
        // and transpose_copy to permute_copy with full permutation.
        // Runs after conv rewriting so shapes are stable.
        Box::new(crate::fx_ir::IrPassAdapter::new(
            fx_rewrite_squeeze_ir::IrRewriteSqueezePass,
        )),
        // -- Postprocessing/cleanup (Python ordering) --
        Box::new(fx_cast_int64::FxCastInt64ToInt32Pass::new(cast)),
        Box::new(fx_rewrite_index_put::FxRewriteIndexPutPass),
        Box::new(fx_fuse_equal_placeholders::FxFuseEqualPlaceholdersPass::new(fuse_placeholders)),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        Box::new(crate::fx_ir::IrPassAdapter::new(
            crate::fx_remove_noop_ir::IrRemoveNoopPass,
        )),
        // Matmul rescale MUST run BEFORE InsertRescalePass: it inserts a
        // RESCALE(INT32→INT8) after bmm to bring the accumulator output back
        // to the quantized dtype.  InsertRescalePass then correctly creates
        // INT8→INT32 upscales for downstream arithmetic ops (sub, add, etc.).
        Box::new(fx_rewrite_matmul::FxRewriteMatmulRescalePass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        Box::new(
            fx_fuse_constant_args::FxFuseConstantArgsPass::new_zero_input_full_only(fuse_late),
        ),
    ]
}

/// Build the hybrid pipeline: IrPasses run on persistent IrGraph, FxPasses
/// cause a flush/reimport cycle. Same pass ordering as build_fx_pipeline.
pub fn build_hybrid_pipeline(exported_program: Option<pyo3::PyObject>) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Fx, Ir, IrEp};

    let ExportedProgramClones {
        fold_qdq,
        fuse: _fuse,
        fuse_grouped: _fuse_grouped,
        cast: _cast,
        fuse_placeholders: _fuse_placeholders,
        fuse_late: _fuse_late,
    } = ExportedProgramClones::new(exported_program);

    vec![
        // -- Early decompositions (before fold_qdq) --
        Ir(Box::new(crate::fx_annotate_output_dim_order_ir::IrAnnotateOutputDimOrderPass)),
        Ir(Box::new(crate::fx_decompose_ops_ir::IrDecomposeOpsPass)),
        // -- Pre-fold --
        Ir(Box::new(crate::fx_fuse_quant_activation_ir::IrFuseQuantActivationPass)),
        Ir(Box::new(fx_convert_clamp_ir::IrConvertClampPass)),
        // -- Fold Q/DQ + rescale (must be FxPass — needs exported_program) --
        Fx(Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(fold_qdq))),
        Ir(Box::new(crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass)),
        Ir(Box::new(crate::fx_decompose_linear_ir::IrDecomposeLinearPass)),
        Ir(Box::new(crate::fx_conv1d_unsqueeze_ir::IrConv1dUnsqueezePass)),
        Fx(Box::new(fx_insert_rescale::FxInsertRescalePass::new())),
        Ir(Box::new(crate::fx_decompose_quant_ir::IrDecomposeQuantPass)),
        // -- Post-fold: long run of IrPasses (no Python roundtrip) --
        Ir(Box::new(fx_convert_split_to_slice_ir::IrConvertSplitToSlicePass)),
        Ir(Box::new(fx_decompose_masked_fill_ir::IrDecomposeMaskedFillPass)),
        Ir(Box::new(crate::fx_canonicalize_gather_ir::IrCanonicalizeGatherPass)),
        Ir(Box::new(crate::fx_decompose_index_tensor_ir::IrDecomposeIndexTensorPass)),
        Ir(Box::new(crate::fx_decompose_index_select_ir::IrDecomposeIndexSelectPass)),
        Ir(Box::new(crate::fx_decompose_unfold_ir::IrDecomposeUnfoldPass)),
        Ir(Box::new(crate::fx_decompose_select_scatter_ir::IrDecomposeSelectScatterPass)),
        Ir(Box::new(crate::fx_decompose_strided_slice_ir::IrDecomposeStridedSlicePass)),
        Ir(Box::new(fx_expand_where_mask_ir::IrExpandWhereMaskPass)),
        Ir(Box::new(fx_decompose_tril_ir::IrDecomposeTrilPass)),
        Ir(Box::new(fx_decompose_logit_ir::IrDecomposeLogitPass)),
        Ir(Box::new(fx_decompose_addmm_ir::IrDecomposeAddmmPass)),
        Ir(Box::new(fx_decompose_cumsum_ir::IrDecomposeCumsumPass)),
        Ir(Box::new(fx_decompose_repeat_interleave_ir::IrDecomposeRepeatInterleavePass)),
        Ir(Box::new(fx_decompose_pixel_shuffle_ir::IrDecomposePixelShufflePass)),
        Ir(Box::new(fx_decompose_as_strided_copy_ir::IrDecomposeAsStridedCopyPass)),
        Ir(Box::new(crate::fx_quantize_clamp_args_ir::IrQuantizeClampArgsPass)),
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
        Ir(Box::new(crate::fx_decompose_adaptive_avgpool_ir::IrDecomposeAdaptiveAvgPoolPass)),
        Ir(Box::new(crate::fx_decompose_maxpool_dilation_ir::IrDecomposeMaxPoolDilationPass)),
        Ir(Box::new(crate::fx_decompose_mean_ir::IrDecomposeMeanPass)),
        Ir(Box::new(fx_convert_sum_ir::IrConvertSumPass)),
        IrEp(Box::new(crate::fx_insert_table_ops_ir::IrInsertTableOpsEpPass)),
        // -- ATen -> TOSA --
        Ir(Box::new(crate::fx_decompose_grouped_conv_ir::IrDecomposeGroupedConvPass)),
        IrEp(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEpPass::with_data_slices(),
        )),
        Fx(Box::new(fx_rewrite_conv::FxRewriteConvPass)),
        Ir(Box::new(crate::fx_size_adjust_input_ir::IrSizeAdjustInputPass)),
        Ir(Box::new(fx_rewrite_squeeze_ir::IrRewriteSqueezePass)),
        // -- Postprocessing --
        IrEp(Box::new(crate::fx_cast_int64_ir::IrCastInt64ToInt32EpPass)),
        Ir(Box::new(crate::fx_rewrite_index_put_ir::IrRewriteIndexPutPass)),
        IrEp(Box::new(
            crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEpPass,
        )),
        Fx(Box::new(fx_memory_format::FxToTosaMemoryFormatPass)),
        Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        Ir(Box::new(crate::fx_rewrite_matmul_ir::IrRewriteMatmulRescalePass)),
        Ir(Box::new(
            crate::fx_insert_conv_rescale_ir::IrInsertConvAccumulatorRescalePass,
        )),
        Fx(Box::new(fx_insert_rescale::FxInsertRescalePass::new())),
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
