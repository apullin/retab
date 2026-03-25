use crate::fx_cast_int64;
use crate::fx_convert_clamp;
use crate::fx_convert_full_like;
use crate::fx_conv1d_unsqueeze;
use crate::fx_convert_sum;
use crate::fx_decompose_addmm;
use crate::fx_decompose_linear;
use crate::fx_decompose_logit;
use crate::fx_decompose_masked_fill;
use crate::fx_decompose_mean;
use crate::fx_decompose_ops;
use crate::fx_decompose_quant;
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
use crate::fx_rewrite_matmul;
use crate::fx_rewrite_squeeze;

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
    // Clone the EP handle for passes that need it. Py<T>.clone() requires the GIL.
    let (ep_for_fuse, ep_for_cast, ep_for_fuse_placeholders, ep_for_fuse_late) =
        pyo3::Python::with_gil(|py| {
            (
                exported_program.as_ref().map(|ep| ep.clone_ref(py)),
                exported_program.as_ref().map(|ep| ep.clone_ref(py)),
                exported_program.as_ref().map(|ep| ep.clone_ref(py)),
                exported_program.as_ref().map(|ep| ep.clone_ref(py)),
            )
        });

    vec![
        // -- Early decompositions (before fold_qdq) --
        // Decompose atanh, hardsigmoid, stack, unbind into primitive ops
        // that TOSA can represent. (masked_fill handled post-fold below.)
        Box::new(fx_decompose_ops::FxDecomposeOpsPass),
        // -- Pre-fold --
        Box::new(fx_fuse_quant_activation::FxFuseQuantizedActivationPass),
        Box::new(fx_convert_clamp::FxConvertToClampPass),
        // -- Fold Q/DQ + rescale --
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(
            exported_program,
        )),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_decompose_linear::FxDecomposeLinearPass),
        Box::new(fx_conv1d_unsqueeze::FxConv1dUnsqueezePass),
        Box::new(fx_insert_rescale::FxInsertRescalePass),
        Box::new(fx_decompose_quant::FxDecomposeQuantNodesPass),
        // -- Post-fold decompositions (matches Python ordering) --
        // Decompose masked_fill -> where(mask, quantized_full, x).  Runs after
        // fold_qdq so the fill scalar can be quantized with output qparams.
        Box::new(fx_decompose_masked_fill::FxDecomposeMaskedFillPass),
        // Decompose logit -> clamp + mul + add + reciprocal + log
        // Python runs DecomposeLogitPass post-fold (line 285 in arm_pass_manager.py)
        Box::new(fx_decompose_logit::FxDecomposeLogitPass),
        // Decompose addmm -> mm + mul + add
        // Python runs DecomposeAddmmPass post-fold (line 297 in arm_pass_manager.py)
        Box::new(fx_decompose_addmm::FxDecomposeAddmmPass),
        Box::new(fx_quantize_clamp_args::FxQuantizeClampArgsPass),
        // Convert full_like -> full with explicit shape
        // Python runs ConvertFullLikeToFullPass in scalars section (line 330)
        Box::new(fx_convert_full_like::FxConvertFullLikeToFullPass),
        Box::new(fx_fuse_constant_args::FxFuseConstantArgsPass::new(
            ep_for_fuse,
        )),
        // Decompose mean.dim into avg_pool2d (spatial) + sum*mul (batch/channel)
        Box::new(fx_decompose_mean::FxDecomposeMeanPass),
        Box::new(fx_convert_sum::FxConvertSumPass),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        // -- ATen -> TOSA conversion --
        Box::new(fx_rewrite_conv::FxRewriteConvPass),
        // Rewrite squeeze/unsqueeze to view_copy with explicit shape,
        // and transpose_copy to permute_copy with full permutation.
        // Runs after conv rewriting so shapes are stable.
        Box::new(fx_rewrite_squeeze::FxRewriteSqueezePass),
        // -- Postprocessing/cleanup (Python ordering) --
        Box::new(fx_cast_int64::FxCastInt64ToInt32Pass::new(ep_for_cast)),
        Box::new(
            fx_fuse_equal_placeholders::FxFuseEqualPlaceholdersPass::new(ep_for_fuse_placeholders),
        ),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        Box::new(fx_remove_noop::FxRemoveNoopPass),
        Box::new(fx_insert_rescale::FxInsertRescalePass),
        // Fix matmul rescale: accumulator_scale = A_scale * B_scale,
        // not just input_scale. Must run after InsertRescalePass.
        Box::new(fx_rewrite_matmul::FxRewriteMatmulRescalePass),
        Box::new(
            fx_fuse_constant_args::FxFuseConstantArgsPass::new_zero_input_full_only(
                ep_for_fuse_late,
            ),
        ),
    ]
}
