use crate::fx_convert_clamp;
use crate::fx_convert_sum;
use crate::fx_decompose_linear;
use crate::fx_decompose_quant;
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
///   2. **Rust** fold Q/DQ and annotate QuantArgs namedtuples in node.meta
///   3. **Rust** insert rescales (DQ->Q pairs + INT32 elementwise rescales)
///   4. Insert TABLE ops (quantised lookup tables for rsqrt, exp, etc.)
///   5. Fuse duplicate users (deduplicate identical call_function nodes)
///   6. Decompose linear into reshape+conv2d+reshape
///   7. Decompose remaining Q/DQ nodes into primitive arithmetic
///   8. Convert sum to chained TOSA REDUCE_SUM ops
///   9. Convert relu/hardtanh to clamp
///  10. Rewrite conv2d targets and annotate conv metadata
///  11. Annotate TOSA memory format (dim_order, spatial_rank)
///  12. Final dedup -- steps 8-11 may create or expose duplicate nodes
///
/// Steps 2 and 3 are both Rust passes. Step 2 stores real QuantArgs
/// namedtuples in node.meta. Step 3 reads those QuantArgs and inserts
/// TOSA RESCALE ops, matching the Python InsertRescalePass +
/// InsertRescaleInt32Pass behavior exactly.
pub fn build_fx_pipeline(exported_program: Option<pyo3::PyObject>) -> Vec<Box<dyn FxPass>> {
    vec![
        Box::new(fx_fuse_quant_activation::FxFuseQuantizedActivationPass),
        // Rust fold: removes Q/DQ nodes, stores QuantArgs namedtuples in node.meta
        Box::new(fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(exported_program)),
        // Rust rescale: replaces DQ->Q pairs and inserts INT32 rescales
        Box::new(fx_insert_rescale::FxInsertRescalePass),
        Box::new(fx_insert_table_ops::FxInsertTableOpsPass),
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
        // Post-fold decompositions
        Box::new(fx_decompose_linear::FxDecomposeLinearPass),
        Box::new(fx_decompose_quant::FxDecomposeQuantNodesPass),
        Box::new(fx_convert_sum::FxConvertSumPass),
        Box::new(fx_convert_clamp::FxConvertToClampPass),
        Box::new(fx_rewrite_conv::FxRewriteConvPass),
        Box::new(fx_memory_format::FxToTosaMemoryFormatPass),
        // Final dedup: clean up duplicates introduced or exposed by passes 8-12
        Box::new(fx_fuse_duplicate_users::FxFuseDuplicateUsersPass),
    ]
}
