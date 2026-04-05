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
use crate::fx_rewrite_squeeze;
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
        Box::new(fx_convert_clamp::FxConvertToClampPass),
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
        Box::new(fx_convert_split_to_slice::FxConvertSplitToSlicePass),
        // Decompose masked_fill -> where(mask, quantized_full, x).  Runs after
        // fold_qdq so the fill scalar can be quantized with output qparams.
        Box::new(fx_decompose_masked_fill::FxDecomposeMaskedFillPass),
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
        Box::new(fx_expand_where_mask::FxExpandWhereMaskPass),
        // Mirror the Python Arm backend's tril lowering:
        // arange + view + add(diagonal) + ge + where / bitwise_and.
        Box::new(fx_decompose_tril::FxDecomposeTrilPass),
        // Decompose logit -> clamp + mul + add + reciprocal + log
        // Python runs DecomposeLogitPass post-fold (line 285 in arm_pass_manager.py)
        Box::new(fx_decompose_logit::FxDecomposeLogitPass),
        // Decompose addmm -> mm + mul + add
        // Python runs DecomposeAddmmPass post-fold (line 297 in arm_pass_manager.py)
        Box::new(fx_decompose_addmm::FxDecomposeAddmmPass),
        // Decompose cumsum into chained slice_copy + add + cat
        Box::new(fx_decompose_cumsum::FxDecomposeCumsumPass),
        // Decompose repeat_interleave(self_int) into unsqueeze+repeat+view
        // so direct lowering reuses the same repeat grammar as the Python path.
        Box::new(fx_decompose_repeat_interleave::FxDecomposeRepeatInterleavePass),
        // Decompose pixel_shuffle/pixel_unshuffle into view+permute+view
        Box::new(fx_decompose_pixel_shuffle::FxDecomposePixelShufflePass),
        // Replace contiguous zero-offset as_strided_copy with view_copy,
        // matching the Python Arm pass and avoiding a raw unsupported ATen op.
        Box::new(fx_decompose_as_strided_copy::FxDecomposeAsStridedCopyPass),
        Box::new(fx_quantize_clamp_args::FxQuantizeClampArgsPass),
        // Convert full_like -> full with explicit shape
        // Python runs ConvertFullLikeToFullPass in scalars section (line 330)
        Box::new(fx_convert_full_like::FxConvertFullLikeToFullPass),
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(fuse)),
        // Decompose adaptive_avg_pool2d → avg_pool2d (or slice+pool+cat)
        Box::new(fx_decompose_adaptive_avgpool::FxDecomposeAdaptiveAvgPoolPass),
        // Decompose dilated max_pool2d into pad + space-to-batch + plain pool
        // before size adjustment and memory-format annotation.
        Box::new(fx_decompose_maxpool_dilation::FxDecomposeMaxPoolDilationPass),
        // Decompose mean.dim into sum*mul with keepdim=True + reshape
        Box::new(fx_decompose_mean::FxDecomposeMeanPass),
        Box::new(fx_convert_sum::FxConvertSumPass),
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
        Box::new(fx_rewrite_squeeze::FxRewriteSqueezePass),
        // -- Postprocessing/cleanup (Python ordering) --
        Box::new(fx_cast_int64::FxCastInt64ToInt32Pass::new(cast)),
        Box::new(fx_rewrite_index_put::FxRewriteIndexPutPass),
        Box::new(fx_fuse_equal_placeholders::FxFuseEqualPlaceholdersPass::new(fuse_placeholders)),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        Box::new(fx_remove_noop::FxRemoveNoopPass),
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

/// Build the FX pipeline for while_loop submodule graphs.
///
/// Uses Carried output qparam source on the final rescale: output[i] must
/// match placeholder[i] because while_loop body outputs feed back as inputs.
///
/// NOTE: This is more minimal than the cond pipeline because the full pipeline
/// (with fold_qdq) can double-transform while_loop bodies. The while_loop
/// repair Python functions still handle boundary rescaling until this pipeline
/// is expanded to handle the loop-carried constraint natively.
pub fn build_fx_submodule_pipeline() -> Vec<Box<dyn FxPass>> {
    let ep = ExportedProgramClones::new(None);
    vec![
        Box::new(fx_control_flow_metadata::FxControlFlowMetadataPass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(ep.fuse)),
        Box::new(fx_remove_noop::FxRemoveNoopPass),
        Box::new(fx_rewrite_matmul::FxRewriteMatmulRescalePass),
        Box::new(fx_insert_rescale::FxInsertRescalePass::new()),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
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
        Box::new(fx_remove_noop::FxRemoveNoopPass),
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
