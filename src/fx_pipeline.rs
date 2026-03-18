use crate::fx_convert_clamp;
use crate::fx_convert_sum;
use crate::fx_fold_qdq;
use crate::fx_fuse_duplicate_users;
use crate::fx_fuse_quant_activation;
use crate::fx_insert_table_ops;
use crate::fx_memory_format;
use crate::fx_pass::FxPass;
use crate::fx_python_bridge;
use crate::fx_rewrite_conv;

/// Build the full FX-direct lowering pipeline in execution order.
///
/// The ordering mirrors the Python `ArmPassManager` pipeline:
///   1. Fuse quantized activations (relu between conv and Q)
///   2. **Rust** fold Q/DQ and annotate QuantArgs namedtuples in node.meta
///   3. **Python** insert rescales (reads QuantArgs from our Rust fold)
///   4. Insert TABLE ops (quantised lookup tables for rsqrt, exp, etc.)
///   5. Fuse duplicate users (deduplicate identical call_function nodes)
///   6. Convert sum to chained TOSA REDUCE_SUM ops
///   7. Convert relu/hardtanh to clamp
///   8. Rewrite conv2d targets and annotate conv metadata
///   9. Annotate TOSA memory format (dim_order, spatial_rank)
///  10. Final dedup — steps 6-9 may create or expose duplicate nodes
///
/// Step 2 is our Rust fold pass that stores real QuantArgs namedtuples
/// (not plain dicts). Step 3 then runs only the Python rescale passes
/// which read those QuantArgs. This avoids the Python fold pass (which
/// requires ARM annotation metadata from edge lowering) while still
/// producing correct rescale ops.
pub fn build_fx_pipeline(_exported_program: Option<pyo3::PyObject>) -> Vec<Box<dyn FxPass>> {
    vec![
        Box::new(fx_fuse_quant_activation::FxFuseQuantizedActivationPass),
        // Rust fold: removes Q/DQ nodes, stores QuantArgs namedtuples in node.meta
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass),
        // Python rescale-only: reads QuantArgs from meta, inserts TOSA RESCALE ops
        Box::new(fx_python_bridge::FxPythonRescaleOnlyPass),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        Box::new(fx_convert_sum::FxConvertSumPass),
        Box::new(fx_convert_clamp::FxConvertToClampPass),
        Box::new(fx_rewrite_conv::FxRewriteConvPass),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        // Final dedup: clean up duplicates introduced or exposed by passes 6-9
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
    ]
}
