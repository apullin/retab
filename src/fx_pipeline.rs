use crate::fx_convert_clamp;
use crate::fx_convert_sum;
use crate::fx_fold_qdq;
use crate::fx_fuse_duplicate_users;
use crate::fx_fuse_quant_activation;
use crate::fx_insert_rescale;
use crate::fx_insert_table_ops;
use crate::fx_memory_format;
use crate::fx_pass::FxPass;
use crate::fx_rewrite_conv;

/// Build the full FX-direct lowering pipeline in execution order.
///
/// The ordering mirrors the Python `ArmPassManager` pipeline:
///   1. Fuse quantized activations (relu between conv and Q)
///   2. Fold Q/DQ and annotate qparams
///   3. Insert TABLE ops (quantised lookup tables for rsqrt, exp, etc.)
///   4. Fuse duplicate users (deduplicate identical call_function nodes)
///   5. Convert sum to chained TOSA REDUCE_SUM ops
///   6. Convert relu/hardtanh to clamp
///   7. Insert TOSA rescale nodes (DQ->Q pairs + INT32 accumulators)
///   8. Rewrite conv2d targets and annotate conv metadata
///   9. Annotate TOSA memory format (dim_order, spatial_rank)
pub fn build_fx_pipeline() -> Vec<Box<dyn FxPass>> {
    vec![
        Box::new(fx_fuse_quant_activation::FxFuseQuantizedActivationPass),
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_convert_sum::FxConvertSumPass),
        Box::new(fx_convert_clamp::FxConvertToClampPass),
        Box::new(fx_insert_rescale::FxInsertRescalePass),
        Box::new(fx_rewrite_conv::FxRewriteConvPass),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
    ]
}
