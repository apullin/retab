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
use crate::pipeline_config::{LayoutMode, PipelineConfig, SoftmaxDecompositionConfig};

/// Build the lowering pipeline over the persistent IrGraph.
pub fn build_main_pipeline(config: &PipelineConfig) -> Vec<crate::fx_ir::PipelineStep> {
    lowering_steps(true, config)
}

/// The parent pipeline (`decompose_qdq`) or a control-flow region's.
///
/// `config.layout` picks the layout model (APU-135). Dim-order mode is RETAB's own
/// pipeline. Explicit mode is rc4's order (`backends/arm/_passes/arm_pass_manager.py`,
/// "APM" below; P2 design §1 and §4.1 with its critique): every pass that writes or
/// reads `tosa_dim_order` is off, rc4's layout passes take their APM slots, and
/// IrToTosaMemoryFormat's slot holds the phase-6 block (APM:715-749). In a region
/// (`!decompose_qdq`) explicit mode skips IrNormalizeDelegateIOLayout, which owns only
/// the delegate's IO (critique A2), and its constants are `get_attr` nodes; a region
/// conv's weight and bias captured from the parent are folded into region constants
/// at serialization, after the boundary contracts (`layout::rewrite_conv`,
/// `control_flow::captured_conv`; APU-135 S8).
fn lowering_steps(decompose_qdq: bool, config: &PipelineConfig) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Effectful, Ir};

    let explicit = config.layout == LayoutMode::Explicit;
    let in_region = !decompose_qdq;
    let mut steps = Vec::new();
    // -- Early decompositions (before fold_qdq) --
    match config.layout {
        LayoutMode::DimOrder => steps.push(Ir(Box::new(
            crate::fx_annotate_output_dim_order_ir::IrAnnotateOutputDimOrderPass,
        ))),
        // rc4's first pass (APM:544).
        LayoutMode::Explicit if !in_region => steps.push(Ir(Box::new(
            crate::layout::normalize_delegate_io::IrNormalizeDelegateIOLayoutPass,
        ))),
        LayoutMode::Explicit => {}
    }
    steps.extend(vec![
        Ir(Box::new(
            crate::fx_fuse_batch_norm_ir::IrFuseBatchNorm2dPass,
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
    ]);
    if explicit {
        // rc4 reads each cond and while output's RESCALE target at APM:595, from the
        // getitem's output qparams that FoldAndAnnotateQParams took from the quantize
        // after it (APM:574). RETAB inserts those RESCALEs at serialization, so it
        // records the targets here, from that quantize, before IrFoldQdq folds it and
        // before any layout pass can hide it (APU-135 S8).
        steps.push(Ir(Box::new(
            crate::control_flow::IrRecordControlFlowRescaleTargetsPass,
        )));
    }
    steps.extend(vec![
        // -- Fold Q/DQ + rescale --
        // Stock sets preserve_partial_binary_tensor_qdq when the TOSA spec
        // supports float or the output format is VGF; RETAB has no VGF path,
        // so the spec flag alone decides. U55 subsets never set it.
        Ir(Box::new(crate::fx_fold_qdq_ir::IrFoldQdqPass {
            preserve_partial_binary_tensor_qdq: config.support_float,
        })),
        Ir(Box::new(
            crate::fx_fuse_consecutive_clamps_ir::IrFuseConsecutiveClampsPass,
        )),
        Ir(Box::new(
            crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass,
        )),
    ]);
    if explicit {
        // FuseIdenticalInputTransformsPass(exported_program) (APM:591), right before
        // DecomposeLinear (APM:592); design §1a. rc4 runs the whole manager, with the
        // edge program, on each region too (tosa/backend.py), so regions get it as well.
        steps.push(Ir(Box::new(
            crate::layout::fuse_identical_input_transforms::IrFuseIdenticalInputTransformsPass,
        )));
    }
    steps.extend(vec![
        Ir(Box::new(crate::fx_decompose_flip_ir::IrDecomposeFlipPass)),
        Ir(Box::new(crate::fx_decompose_roll_ir::IrDecomposeRollPass)),
        Ir(Box::new(
            crate::fx_decompose_linear_ir::IrDecomposeLinearPass,
        )),
    ]);
    // rc4 rewrites conv1d inside RewriteConvPass (design §1a).
    if !explicit {
        steps.push(Ir(Box::new(
            crate::fx_conv1d_unsqueeze_ir::IrConv1dUnsqueezePass,
        )));
    }
    steps.extend(vec![
        Ir(Box::new(
            crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass,
        )),
        Ir(Box::new(
            crate::fx_insert_int32_rescale_ir::IrInsertInt32RescalePass,
        )),
        Ir(Box::new(
            crate::fx_insert_int32_rescale_ir::IrFuseConsecutiveRescalesPass,
        )),
    ]);
    if decompose_qdq {
        steps.push(Ir(Box::new(
            crate::fx_decompose_quant_ir::IrDecomposeQuantPass,
        )));
    }
    steps.extend(vec![
        // -- Post-fold: long run of IrPasses (no Python roundtrip) --
        Ir(Box::new(
            fx_convert_split_to_slice_ir::IrConvertSplitToSlicePass,
        )),
        Ir(Box::new(
            fx_replace_inf_values_ir::IrReplaceInfAndLimitValuesPass(config.quantize_inf),
        )),
    ]);
    if config.softmax == SoftmaxDecompositionConfig::Masked {
        steps.push(Ir(Box::new(
            fx_decompose_masked_fill_ir::IrDecomposeMaskedFillPass,
        )));
    }
    if explicit {
        // APM:611, :617 and :629: after the fold (round and sqrt read qparams) and
        // before IrReplaceScalarWithTensor (APM:656); design §4.4.
        steps.extend(vec![
            Ir(Box::new(
                crate::layout::decompositions::IrDecomposeRoundPass,
            )),
            Ir(Box::new(crate::layout::decompositions::IrDecomposeSqrtPass)),
            Ir(Box::new(crate::layout::decompositions::IrDecomposeSignPass)),
        ]);
    }
    steps.extend(vec![
        Ir(Box::new(
            crate::fx_decompose_embedding_ir::IrDecomposeEmbeddingPass,
        )),
        Ir(Box::new(
            crate::fx_canonicalize_gather_ir::IrCanonicalizeGatherPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_index_tensor_ir::IrDecomposeIndexTensorPass {
                decompose_constant_indices: config.is_u55_subset,
            },
        )),
        Ir(Box::new(
            crate::fx_decompose_index_select_ir::IrDecomposeIndexSelectPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_unfold_ir::IrDecomposeUnfoldPass {
                use_slice: config.is_u55_subset,
            },
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
        Ir(Box::new(
            crate::fx_decompose_div_tensor_mode_ir::IrDecomposeDivTensorModePass,
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
    ]);
    if explicit {
        // APM:664, right after ConvertFullLikeToFull (APM:661); design C3. RETAB runs
        // its avg-pool, var and mean decompositions after this slot, where rc4 runs
        // them before (critique B9); phase-6 slot 7 rank-matches what they create.
        steps.push(Ir(Box::new(
            crate::layout::match_arg_ranks::IrMatchArgRanksPass,
        )));
    }
    steps.extend(vec![
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
            crate::fx_normalize_maxpool_rank_ir::IrNormalizeMaxPool2dInputRankPass,
        )),
        Ir(Box::new(
            crate::fx_decompose_maxpool_dilation_ir::IrDecomposeMaxPoolDilationPass,
        )),
    ]);
    if config.is_u55_subset {
        steps.push(Ir(Box::new(
            crate::fx_decompose_large_stride_maxpool_ir::IrDecomposeLargeStrideMaxPool2dForU55Pass,
        )));
    }
    steps.extend(vec![
        Ir(Box::new(crate::fx_decompose_var_ir::IrDecomposeVarPass)),
        Ir(Box::new(crate::fx_decompose_mean_ir::IrDecomposeMeanPass)),
        Ir(Box::new(fx_convert_sum_ir::IrConvertBoolSumPass)),
        Ir(Box::new(
            crate::fx_legalize_integer_arithmetic_ir::IrLegalizeIntegerArithmeticPass,
        )),
    ]);
    if config.is_u55_subset {
        steps.extend(vec![
            Ir(Box::new(crate::fx_cast_to_int32_ir::IrCastToInt32Pass)),
            Ir(Box::new(crate::fx_broadcast_args_ir::IrBroadcastArgsPass)),
        ]);
    }
    steps.extend(vec![
        Ir(Box::new(fx_convert_sum_ir::IrConvertSumPass)),
        Effectful(Box::new(
            crate::fx_insert_table_ops_ir::IrInsertTableOpsEffectPass,
        )),
        // -- ATen -> TOSA --
        // Explicit mode also splits rank-5 depthwise convs (APU-135 P2, design §4.2-P).
        Ir(Box::new(
            crate::fx_decompose_grouped_conv_ir::IrDecomposeGroupedConvPass {
                layout: config.layout,
            },
        )),
    ]);
    if !explicit {
        // Explicit mode folds constants in the IR after the avg-pool rewrite instead
        // (IrMaterializeConstants), and IrRewriteConv types the conv bias itself.
        steps.extend(vec![
            Effectful(Box::new(
                crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::with_data_slices(
                ),
            )),
            Ir(Box::new(
                crate::fx_annotate_conv_bias_ir::IrAnnotateConvBiasDtypePass,
            )),
        ]);
    }
    // Explicit mode slices as rc4 does (APU-135 P2, design §4.2-P).
    steps.push(Ir(Box::new(
        crate::fx_size_adjust_input_ir::IrSizeAdjustInputPass {
            layout: config.layout,
        },
    )));
    if explicit {
        // APM:698, then ComputeConstantOpsAOT and FuseConstantArgs (APM:699-700), which
        // also materialize the avg-pool zero points.
        steps.extend(vec![
            Ir(Box::new(
                crate::layout::rewrite_avg_pool2d::IrRewriteAvgPool2dPass,
            )),
            Ir(Box::new(
                crate::layout::constants::IrMaterializeConstantsPass { in_region },
            )),
        ]);
    }
    steps.extend(vec![
        Ir(Box::new(fx_rewrite_squeeze_ir::IrRewriteSqueezePass)),
        Ir(Box::new(
            crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
        )),
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
    ]);
    if !explicit {
        // rc4 merges equal placeholders only after phase 6 (APM:755), where explicit
        // mode does. Merging here, on the data alone, sent the multihead attention
        // op tests to stock in explicit mode (`fx_fuse_equal_placeholders_ir`).
        steps.push(Effectful(Box::new(
            crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEffectPass {
                layout: config.layout,
            },
        )));
    }
    steps.extend(vec![
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass,
        )),
    ]);
    if !explicit {
        steps.extend(vec![
            // Before the layout pass, so the chunks get dim orders like any other node.
            Ir(Box::new(
                crate::fx_decompose_permute_for_u55_ir::IrDecomposePermuteForU55Pass {
                    split_permutes: config.u55_split_permutes.clone(),
                },
            )),
            // Linear's view pairs around an activation, before layout sees them.
            Ir(Box::new(
                crate::fx_cancel_views_around_unary_ir::IrCancelViewsAroundUnaryPass,
            )),
        ]);
    }
    // Copy lowering compares logical shapes and emits views/repeats for layout conversion.
    steps.push(Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)));
    match config.layout {
        LayoutMode::DimOrder => steps.extend(vec![
            Ir(Box::new(
                crate::fx_memory_format_ir::IrToTosaMemoryFormatPass,
            )),
            Ir(Box::new(crate::fx_rewrite_pad_ir::IrRewritePadPass)),
            Ir(Box::new(
                crate::fx_rewrite_cat_slice_ir::IrRewriteCatSlicePass,
            )),
            Ir(Box::new(
                crate::fx_fuse_consecutive_concats_ir::IrFuseConsecutiveConcatsPass,
            )),
            // Slice-view propagation matches aten.slice_copy, which RewriteSlice retargets.
            Ir(Box::new(
                crate::fx_propagate_slice_views_ir::IrPropagateSliceViewsPass,
            )),
            Ir(Box::new(
                crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
            )),
            Ir(Box::new(crate::fx_rewrite_slice_ir::IrRewriteSlicePass)),
            Ir(Box::new(
                crate::fx_fuse_consecutive_slices_ir::IrFuseConsecutiveSlicesPass,
            )),
            Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        ]),
        LayoutMode::Explicit => steps.extend(explicit_phase6_steps(config, in_region)),
    }
    steps.extend(vec![
        Effectful(Box::new(
            crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass,
        )),
        Effectful(Box::new(
            crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass::matmul_only(),
        )),
    ]);
    if explicit {
        // FuseEqualPlaceholders (APM:755): merges the per-conv weights of convs that
        // share one, and equal constants phase 6 created; design §1c. Its key holds
        // the val dtype, as rc4's holds the tensor's dtype.
        steps.push(Effectful(Box::new(
            crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEffectPass {
                layout: config.layout,
            },
        )));
    }
    steps.extend(vec![
        Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        Ir(Box::new(
            crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass,
        )),
    ]);
    if !explicit {
        steps.extend(vec![
            Ir(Box::new(
                crate::fx_rewrite_matmul_ir::IrRewriteMatmulRescalePass {
                    layout: config.layout,
                },
            )),
            // Explicit mode's IrRewriteConv puts the accumulator RESCALE in place; on
            // its graphs this pass would add a second one (design §1c).
            Ir(Box::new(
                crate::fx_insert_conv_rescale_ir::IrInsertConvAccumulatorRescalePass,
            )),
        ]);
    }
    steps.push(Ir(Box::new(
        crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
    )));
    steps.push(Ir(Box::new(
        crate::fx_fuse_cascaded_views_ir::IrFuseDuplicateViewsPass,
    )));
    if explicit {
        // After every explicit pass: a control-flow output whose boundary-contract
        // target their qparam-less copies hide, and that has no recorded target,
        // goes to stock.
        steps.push(Ir(Box::new(
            crate::layout::control_flow_guard::IrExplicitControlFlowOutputGuardPass,
        )));
    }
    // rc4 runs EnsureUniqueOutputNodes last; the while pipeline relies on that.
    steps.push(Ir(Box::new(
        crate::fx_ensure_unique_output_nodes_ir::IrEnsureUniqueOutputNodesPass,
    )));
    steps
}

/// Explicit mode's phase-6 block (APM:715-749, design §1b), in IrToTosaMemoryFormat's
/// slot. rc4's passes that never fire for TOSA-1.0 INT (DecomposeAdaptiveMaxPool2d,
/// FoldScalarMulIntoConv, the MXFP rewrites; design F1) are left out, and its tail
/// (ResolveViewCopyInferredDim, InsertConstShapes, InsertDataLayoutCasts) is the
/// serializer's.
fn explicit_phase6_steps(
    config: &PipelineConfig,
    in_region: bool,
) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    vec![
        Ir(Box::new(
            crate::layout::rewrite_upsample::IrRewriteUpsamplePass,
        )),
        Ir(Box::new(
            crate::layout::rewrite_max_pool2d::IrRewriteMaxPool2dPass,
        )),
        Ir(Box::new(crate::layout::rewrite_conv::IrRewriteConvPass::new(
            config, in_region,
        ))),
        // RewriteMatmulPass (APM:724); dim-order mode runs it after the late fusions.
        Ir(Box::new(
            crate::fx_rewrite_matmul_ir::IrRewriteMatmulRescalePass {
                layout: config.layout,
            },
        )),
        Ir(Box::new(
            crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass,
        )),
        // PropagateViewCopyPermuteDownPass(compile_spec, exported_program), then the
        // Up pass (APM:726-727): the TOSA overrides, with RemovePermutesAroundElementwise-
        // TosaOps as the driver's vertical fusion (APU-159).
        Ir(Box::new(
            crate::layout::propagate::IrPropagateViewCopyPermutePass::ARM_DOWN,
        )),
        Ir(Box::new(
            crate::layout::propagate::IrPropagateViewCopyPermutePass::ARM_UP,
        )),
        Ir(Box::new(crate::fx_rewrite_pad_ir::IrRewritePadPass)),
        Ir(Box::new(
            crate::layout::move_data_movement::IrMoveDataMovementOpsToSmallerDtypePass,
        )),
        Ir(Box::new(
            crate::layout::match_arg_ranks::IrMatchArgRanksPass,
        )),
        Ir(Box::new(
            crate::layout::high_rank_singleton_permute::IrRewriteHighRankSingletonPermutePass,
        )),
        Ir(Box::new(
            crate::fx_rewrite_cat_slice_ir::IrRewriteCatSlicePass,
        )),
        Ir(Box::new(
            crate::fx_fuse_consecutive_concats_ir::IrFuseConsecutiveConcatsPass,
        )),
        // Splits only the permutes Python's Vela probe named on the edge program
        // (design F8).
        Ir(Box::new(
            crate::fx_decompose_permute_for_u55_ir::IrDecomposePermuteForU55Pass {
                split_permutes: config.u55_split_permutes.clone(),
            },
        )),
        // RETAB-only; kept through P2 (design §1b slot 12).
        Ir(Box::new(
            crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
        )),
        Ir(Box::new(crate::fx_rewrite_slice_ir::IrRewriteSlicePass)),
        Ir(Box::new(
            crate::fx_fuse_consecutive_slices_ir::IrFuseConsecutiveSlicesPass,
        )),
        Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        Ir(Box::new(
            crate::layout::canonicalize::IrCanonicalizeViewCopyPermutePass,
        )),
        Ir(Box::new(
            crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass,
        )),
    ]
}

/// IR-native control-flow submodule pipelines: transforms run as IrPasses on a
/// persistent submodule IrGraph instead of Python FxPasses.
pub fn build_ir_while_submodule_pipeline(
    config: &PipelineConfig,
) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::Ir;
    let mut steps = vec![Ir(Box::new(crate::control_flow::IrStripWhileBoundaryQPass))];
    // Boundary Q/DQ must survive until control_flow::contracts aligns region IO.
    steps.extend(lowering_steps(false, config));
    let unique_outputs = steps.pop().unwrap();
    // Align loop-carried outputs once the producer scales are final.
    steps.push(Ir(Box::new(
        crate::control_flow::IrInsertCarriedOutputRescalePass,
    )));
    steps.push(unique_outputs);
    steps
}

pub fn build_ir_cond_submodule_pipeline(
    config: &PipelineConfig,
) -> Vec<crate::fx_ir::PipelineStep> {
    lowering_steps(false, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::PipelineStep;

    fn pass_names(steps: Vec<PipelineStep>) -> Vec<String> {
        steps
            .into_iter()
            .map(|step| match step {
                PipelineStep::Ir(pass) => pass.name().to_owned(),
                PipelineStep::Effectful(pass) => pass.name().to_owned(),
            })
            .collect()
    }

    #[test]
    fn rc4_fusions_and_cleanup_are_ordered_in_parent_and_region_pipelines() {
        for spec in ["TOSA-1.0+FP", "TOSA-1.0+INT+u55", "TOSA-1.0+INT+cf"] {
            let config = PipelineConfig::from_tosa_spec(spec);
            for names in [
                pass_names(lowering_steps(true, &config)),
                pass_names(build_ir_cond_submodule_pipeline(&config)),
                pass_names(build_ir_while_submodule_pipeline(&config)),
            ] {
                let position = |pass| names.iter().position(|name| name == pass).unwrap();
                let fold = position("IrFoldQdq");
                assert_eq!(
                    &names[fold + 1..fold + 3],
                    ["IrFuseConsecutiveClamps", "IrFuseDuplicateUsers"]
                );
                assert!(position("IrFuseConsecutiveClamps") < position("IrQuantizeClampArgs"));
                let layout = position("IrToTosaMemoryFormat");
                assert_eq!(names[layout - 1], "IrRemoveNoop");
                assert_eq!(
                    &names[layout + 1..layout + 9],
                    [
                        "IrRewritePad",
                        "IrRewriteCatSlice",
                        "IrFuseConsecutiveConcats",
                        "IrPropagateSliceViews",
                        "IrFuseCascadedViews",
                        "IrRewriteSlice",
                        "IrFuseConsecutiveSlices",
                        "IrRemoveNoop"
                    ]
                );
                let late_noop = names
                    .iter()
                    .rposition(|name| name == "IrRemoveNoop")
                    .unwrap();
                assert_eq!(
                    &names[late_noop + 1..late_noop + 4],
                    [
                        "IrFuseDuplicateUsers",
                        "IrRewriteMatmulRescale",
                        "IrInsertConvAccumulatorRescale"
                    ]
                );
                assert_eq!(names.last().unwrap(), "IrEnsureUniqueOutputNodes");
            }
        }
    }

    #[test]
    fn late_fusion_merges_equal_constant_users_then_makes_outputs_unique() {
        use crate::fx_ir::{IrArg, IrGraph};
        use crate::test_ir::{buffer, f32_call, f32_placeholder, f32_tensor, output};

        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3]);
        let first = buffer(&mut graph, "first", f32_tensor(vec![2, 3], &[1.0; 6]));
        let second = buffer(&mut graph, "second", f32_tensor(vec![2, 3], &[1.0; 6]));
        let left = f32_call(
            &mut graph,
            "left",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(first)],
            [2, 3],
        );
        let right = f32_call(
            &mut graph,
            "right",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(second)],
            [2, 3],
        );
        let out = output(&mut graph, left);
        graph.set_args(
            out,
            vec![IrArg::List(vec![IrArg::Node(left), IrArg::Node(right)])],
        );
        let config = PipelineConfig::from_tosa_spec("TOSA-1.0+FP");
        for step in lowering_steps(true, &config) {
            match step {
                PipelineStep::Ir(pass) => {
                    pass.run(&mut graph);
                }
                PipelineStep::Effectful(pass) => {
                    pass.run(&mut graph);
                }
            }
        }
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten.add.Tensor")
                .count(),
            1
        );
        let IrArg::List(outputs) = &graph.node(out).args[0] else {
            panic!("output list")
        };
        let identities: Vec<_> = outputs
            .iter()
            .map(|arg| graph.node(arg.as_node().unwrap()))
            .collect();
        assert_ne!(identities[0].id, identities[1].id);
        assert!(identities
            .iter()
            .all(|node| node.target == "backend.tosa.IDENTITY.default"));
        assert_eq!(identities[0].arg_node(0), identities[1].arg_node(0));
    }

    /// The passes that write or read `tosa_dim_order`, and the RETAB passes rc4 does
    /// inside RewriteConvPass, which explicit mode leaves out (design §1).
    const DIM_ORDER_ONLY: [&str; 7] = [
        "IrAnnotateOutputDimOrder",
        "IrConv1dUnsqueeze",
        "IrAnnotateConvBiasDtype",
        "IrCancelViewsAroundUnary",
        "IrToTosaMemoryFormat",
        "IrPropagateSliceViews",
        "IrInsertConvAccumulatorRescale",
    ];

    /// rc4's phase-6 block (APM:715-749) as explicit mode builds it, in
    /// IrToTosaMemoryFormat's slot right after the first late IrRemoveNoop.
    const EXPLICIT_PHASE6: [&str; 20] = [
        "IrRewriteUpsample",
        "IrRewriteMaxPool2d",
        "IrRewriteConv",
        "IrRewriteMatmulRescale",
        "IrFuseViewCopyTransform",
        "IrPropagateViewCopyPermuteDown",
        "IrPropagateViewCopyPermuteUp",
        "IrRewritePad",
        "IrMoveDataMovementOpsToSmallerDtype",
        "IrMatchArgRanks",
        "IrRewriteHighRankSingletonPermute",
        "IrRewriteCatSlice",
        "IrFuseConsecutiveConcats",
        "IrDecomposePermuteForU55",
        "IrFuseCascadedViews",
        "IrRewriteSlice",
        "IrFuseConsecutiveSlices",
        "IrRemoveNoop",
        "IrCanonicalizeViewCopyPermute",
        "IrFuseViewCopyTransform",
    ];

    fn explicit(spec: &str) -> PipelineConfig {
        PipelineConfig {
            layout: LayoutMode::Explicit,
            ..PipelineConfig::from_tosa_spec(spec)
        }
    }

    #[test]
    fn dim_order_mode_keeps_its_layout_passes_in_parent_and_region_pipelines() {
        let builders: [fn(&PipelineConfig) -> Vec<PipelineStep>; 3] = [
            build_main_pipeline,
            build_ir_cond_submodule_pipeline,
            build_ir_while_submodule_pipeline,
        ];
        for spec in [
            "TOSA-1.0+FP",
            "TOSA-1.0+INT+int16+int4+u55",
            "TOSA-1.0+INT+int16+int4+cf",
        ] {
            let dim_order = PipelineConfig::from_tosa_spec(spec);
            for build in builders {
                let names = pass_names(build(&dim_order));
                for pass in DIM_ORDER_ONLY {
                    assert!(names.iter().any(|name| name == pass), "{spec}: {pass}");
                }
                for pass in [
                    "IrNormalizeDelegateIOLayout",
                    "IrFuseIdenticalInputTransforms",
                    "IrRecordControlFlowRescaleTargets",
                    "IrDecomposeRound",
                    "IrDecomposeSqrt",
                    "IrDecomposeSign",
                    "IrRewriteAvgPool2d",
                    "IrMaterializeConstants",
                    "IrRewriteUpsample",
                    "IrRewriteMaxPool2d",
                    "IrRewriteConv",
                    "IrFuseViewCopyTransform",
                    "IrPropagateViewCopyPermuteDown",
                    "IrPropagateViewCopyPermuteUp",
                    "IrMoveDataMovementOpsToSmallerDtype",
                    "IrMatchArgRanks",
                    "IrRewriteHighRankSingletonPermute",
                    "IrCanonicalizeViewCopyPermute",
                ] {
                    assert!(!names.iter().any(|name| name == pass), "{spec}: {pass}");
                }
            }
        }
    }

    /// Explicit mode builds rc4's order (design §1, §4.1, critique A2, B9): the
    /// dim-order passes are gone, rc4's passes sit at their APM slots, and the
    /// phase-6 block replaces IrToTosaMemoryFormat. Regions get the same list without
    /// IrNormalizeDelegateIOLayout.
    #[test]
    fn explicit_mode_builds_rc4s_order_in_parent_and_region_pipelines() {
        for spec in [
            "TOSA-1.0+INT",
            "TOSA-1.0+INT+int16+int4+u55",
            "TOSA-1.0+INT+int16+int4+cf",
        ] {
            let config = explicit(spec);
            for (in_region, names) in [
                (false, pass_names(build_main_pipeline(&config))),
                (true, pass_names(build_ir_cond_submodule_pipeline(&config))),
                (true, pass_names(build_ir_while_submodule_pipeline(&config))),
            ] {
                let count = |pass: &str| names.iter().filter(|name| *name == pass).count();
                let position = |pass: &str| {
                    names
                        .iter()
                        .position(|name| name == pass)
                        .unwrap_or_else(|| panic!("{spec}: {pass} missing"))
                };
                for pass in DIM_ORDER_ONLY {
                    assert_eq!(count(pass), 0, "{spec}: {pass}");
                }

                // APM:544, the first pass, parent only.
                if in_region {
                    assert_eq!(count("IrNormalizeDelegateIOLayout"), 0, "{spec}");
                } else {
                    assert_eq!(names[0], "IrNormalizeDelegateIOLayout", "{spec}");
                    assert_eq!(count("IrNormalizeDelegateIOLayout"), 1, "{spec}");
                }

                // APM:586-592: FuseConsecutiveClamps, FuseDuplicateUsers, then
                // FuseIdenticalInputTransforms right before DecomposeLinear (RETAB
                // decomposes flip and roll there too).
                let fold = position("IrFoldQdq");
                assert_eq!(
                    &names[fold + 1..fold + 7],
                    [
                        "IrFuseConsecutiveClamps",
                        "IrFuseDuplicateUsers",
                        "IrFuseIdenticalInputTransforms",
                        "IrDecomposeFlip",
                        "IrDecomposeRoll",
                        "IrDecomposeLinear"
                    ],
                    "{spec}"
                );
                assert_eq!(count("IrFuseIdenticalInputTransforms"), 1, "{spec}");

                // The control-flow RESCALE targets, recorded right before the fold
                // (rc4 reads them at APM:595, from FoldAndAnnotate's annotation).
                let record = position("IrRecordControlFlowRescaleTargets");
                assert_eq!(names[record + 1], "IrFoldQdq", "{spec}");
                assert_eq!(count("IrRecordControlFlowRescaleTargets"), 1, "{spec}");

                // APM:611, :617 and :629: after the fold, before the scalar rewrite.
                let round = position("IrDecomposeRound");
                assert_eq!(
                    &names[round..round + 4],
                    [
                        "IrDecomposeRound",
                        "IrDecomposeSqrt",
                        "IrDecomposeSign",
                        "IrDecomposeEmbedding"
                    ],
                    "{spec}"
                );
                assert!(position("IrFoldQdq") < round);
                assert!(position("IrReplaceInfAndLimitValues") < round);
                assert!(round < position("IrReplaceScalarWithTensor"));
                if config.softmax == SoftmaxDecompositionConfig::Masked {
                    assert_eq!(names[round - 1], "IrDecomposeMaskedFill", "{spec}");
                }

                // APM:664, right after ConvertFullLikeToFull (design C3), and again
                // in phase 6.
                let full_like = position("IrConvertFullLike");
                assert_eq!(names[full_like + 1], "IrMatchArgRanks", "{spec}");
                assert_eq!(count("IrMatchArgRanks"), 2, "{spec}");

                // APM:695-700: SizeAdjustInput, RewriteAvgPool2d, then the constant
                // folds in the IR, in place of the with_data_slices effect pass and
                // the conv-bias annotation.
                let size_adjust = position("IrSizeAdjustInput");
                assert_eq!(
                    &names[size_adjust - 1..size_adjust + 3],
                    [
                        "IrDecomposeGroupedConv",
                        "IrSizeAdjustInput",
                        "IrRewriteAvgPool2d",
                        "IrMaterializeConstants"
                    ],
                    "{spec}"
                );
                assert!(position("IrInsertTableOpsEp") < size_adjust);
                assert_eq!(
                    count("IrFuseConstantInputsEp"),
                    2,
                    "{spec}: matmul_only twice"
                );

                // The phase-6 block, once, after the late IrRemoveNoop, and nothing
                // dim-order-specific before it.
                let phase6 = position("IrRewriteUpsample");
                assert_eq!(names[phase6 - 1], "IrRemoveNoop", "{spec}");
                assert_eq!(names[phase6 - 2], "IrFuseShapeOnlyConstantsEp", "{spec}");
                assert_eq!(
                    &names[phase6..phase6 + EXPLICIT_PHASE6.len()],
                    EXPLICIT_PHASE6
                );
                for pass in [
                    "IrRewriteConv",
                    "IrRewriteMatmulRescale",
                    "IrRewriteUpsample",
                    "IrRewriteMaxPool2d",
                    "IrPropagateViewCopyPermuteDown",
                    "IrPropagateViewCopyPermuteUp",
                    "IrMoveDataMovementOpsToSmallerDtype",
                    "IrDecomposePermuteForU55",
                    "IrRewriteAvgPool2d",
                ] {
                    assert_eq!(count(pass), 1, "{spec}: {pass}");
                }

                // After phase 6: the late constant effects, FuseEqualPlaceholders
                // (APM:755), the cleanup, and EnsureUniqueOutputNodes last.
                let tail = phase6 + EXPLICIT_PHASE6.len();
                assert_eq!(
                    &names[tail..tail + 6],
                    [
                        "IrFuseZeroInputConstantsEp",
                        "IrFuseShapeOnlyConstantsEp",
                        "IrFuseConstantInputsEp",
                        "IrFuseEqualPlaceholdersEp",
                        "IrRemoveNoop",
                        "IrFuseDuplicateUsers",
                    ],
                    "{spec}"
                );
                let guard = position("IrExplicitControlFlowOutputGuard");
                assert_eq!(count("IrExplicitControlFlowOutputGuard"), 1, "{spec}");
                assert_eq!(names[guard - 1], "IrFuseDuplicateViews", "{spec}");
                assert!(
                    guard >= names.len() - 3,
                    "{spec}: the guard runs after every explicit pass"
                );
                // Only after phase 6, as rc4 (APM:755).
                assert_eq!(count("IrFuseEqualPlaceholdersEp"), 1, "{spec}");
                assert_eq!(names.last().unwrap(), "IrEnsureUniqueOutputNodes");
            }
        }
    }

    /// Explicit mode changes only the layout passes: removing them from both modes'
    /// lists, and the explicit additions, leaves the same pipeline (minus the moved
    /// passes' positions).
    #[test]
    fn explicit_mode_keeps_every_other_pass_of_dim_order_mode() {
        let explicit_only = [
            "IrNormalizeDelegateIOLayout",
            "IrFuseIdenticalInputTransforms",
            "IrRecordControlFlowRescaleTargets",
            "IrDecomposeRound",
            "IrDecomposeSqrt",
            "IrDecomposeSign",
            "IrMatchArgRanks",
            "IrRewriteAvgPool2d",
            "IrMaterializeConstants",
            "IrRewriteUpsample",
            "IrRewriteMaxPool2d",
            "IrRewriteConv",
            "IrFuseViewCopyTransform",
            "IrPropagateViewCopyPermuteDown",
            "IrPropagateViewCopyPermuteUp",
            "IrMoveDataMovementOpsToSmallerDtype",
            "IrRewriteHighRankSingletonPermute",
            "IrCanonicalizeViewCopyPermute",
            "IrFuseEqualPlaceholdersEp",
            "IrExplicitControlFlowOutputGuard",
        ];
        for spec in ["TOSA-1.0+INT+int16+int4+u55", "TOSA-1.0+INT+int16+int4+cf"] {
            let dim_order = PipelineConfig::from_tosa_spec(spec);
            let config = explicit(spec);
            for (dim_order_names, explicit_names) in [
                (
                    pass_names(build_main_pipeline(&dim_order)),
                    pass_names(build_main_pipeline(&config)),
                ),
                (
                    pass_names(build_ir_while_submodule_pipeline(&dim_order)),
                    pass_names(build_ir_while_submodule_pipeline(&config)),
                ),
            ] {
                let mut shared: Vec<String> = dim_order_names
                    .into_iter()
                    .filter(|name| !DIM_ORDER_ONLY.contains(&name.as_str()))
                    .filter(|name| !explicit_only.contains(&name.as_str()))
                    .collect();
                let mut explicit_shared: Vec<String> = explicit_names
                    .into_iter()
                    .filter(|name| !explicit_only.contains(&name.as_str()))
                    .collect();
                shared.sort();
                explicit_shared.sort();
                // Dim-order mode's with_data_slices effect pass is the one
                // IrFuseConstantInputsEp explicit mode drops.
                let dropped = shared
                    .iter()
                    .position(|name| name == "IrFuseConstantInputsEp")
                    .unwrap();
                shared.remove(dropped);
                assert_eq!(shared, explicit_shared, "{spec}");
            }
        }
    }

    /// rc4's `linear_pc` graph at RewriteConvPass, a per-channel quantized Linear,
    /// with the all-zero bias of the multihead attention op tests' projections. Its
    /// weight's zero point `b__zero_point_0` comes first, unused: an all-zero int64
    /// buffer, whose data the importer narrows to int32. `b__zero_point_1` is the
    /// bias's zero point, all-zero int32.
    fn linear_with_a_zero_bias() -> crate::fx_ir::IrGraph {
        use crate::layout::replay;
        let fixture = include_str!("../tests/fixtures/layout_replay/rewrite_conv.txt");
        let case = replay::parse(fixture)
            .expect("a valid fixture")
            .into_iter()
            .find(|case| case.name == "linear_pc")
            .expect("the linear_pc case");
        let mut graph = replay::import(&case);
        let bias = crate::test_ir::live_node(&graph, "b__frozen_param1").id;
        let tensor = graph.node_mut(bias).tensor.as_mut().expect("bias data");
        tensor.data.data.fill(0);
        graph
    }

    fn run_steps(steps: &[PipelineStep], graph: &mut crate::fx_ir::IrGraph) {
        for step in steps {
            match step {
                PipelineStep::Ir(pass) => {
                    pass.run(graph);
                }
                PipelineStep::Effectful(pass) => {
                    pass.run(graph);
                }
            }
        }
    }

    /// APU-157: explicit mode used to merge as dim-order mode does, on the data
    /// alone and before phase 6. That hands the Linear's bias users the int64 zero
    /// point, and IrRewriteConv sends the graph to stock, as it did the multihead
    /// attention op tests.
    #[test]
    fn merging_on_the_data_before_rewrite_conv_hands_the_linear_an_int64_bias() {
        use crate::fx_fuse_equal_placeholders_ir::IrFuseEqualPlaceholdersEffectPass;
        use crate::fx_ir::{EffectfulIrPass, IrPass};
        use crate::layout::rewrite_conv::IrRewriteConvPass;

        let mut graph = linear_with_a_zero_bias();
        IrFuseEqualPlaceholdersEffectPass {
            layout: LayoutMode::DimOrder,
        }
        .run(&mut graph);
        crate::stock_fallback::take();
        IrRewriteConvPass {
            is_u55_subset: false,
            in_region: false,
            u55_split_permutes: Default::default(),
        }
        .run(&mut graph);
        let reason = crate::stock_fallback::take().expect("a stock fallback");
        assert!(
            reason.contains("a torch.int64 bias for a torch.int32 accumulator"),
            "{reason}"
        );
    }

    /// Explicit mode lowers that Linear from the postprocessing block on (where
    /// dim-order mode merges) without a fallback, and its TOSA convolution ends up
    /// reading the int32 zero constant, as rc4's reads `b__zero_point_1_common`.
    /// The int64 zero point stays apart, as rc4's `b__zero_point_0_common` does.
    #[test]
    fn explicit_mode_keeps_an_int64_zero_point_off_a_linears_zero_bias() {
        let config = explicit("TOSA-1.0+INT");
        let steps = lowering_steps(true, &config);
        let names = pass_names(lowering_steps(true, &config));
        let start = names
            .iter()
            .position(|name| name == "IrCastInt64ToInt32Ep")
            .unwrap();

        let mut graph = linear_with_a_zero_bias();
        crate::stock_fallback::take();
        run_steps(&steps[start..], &mut graph);
        assert_eq!(crate::stock_fallback::take(), None);

        let convs: Vec<_> = graph
            .live_nodes()
            .filter(|node| node.target == "backend.tosa.CONV2D.default")
            .collect();
        assert_eq!(convs.len(), 1);
        let bias = graph.node(convs[0].args[2].as_node().expect("a bias node"));
        assert_eq!(bias.name, "b__zero_point_1");
        assert_eq!(bias.meta.val_dtype(), Some("torch.int32"));
        let weight_zp = crate::test_ir::live_node(&graph, "b__zero_point_0");
        assert_eq!(weight_zp.meta.val_dtype(), Some("torch.int64"));
        assert!(!graph.has_live_users(weight_zp.id));
    }

    #[test]
    fn stable_softmax_skips_only_masked_fill_in_parent_and_region_pipelines() {
        let mut config = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+int4+u55");
        let stable = [
            pass_names(lowering_steps(true, &config)),
            pass_names(build_ir_cond_submodule_pipeline(&config)),
            pass_names(build_ir_while_submodule_pipeline(&config)),
        ];
        config.softmax = SoftmaxDecompositionConfig::Masked;
        let masked = [
            pass_names(lowering_steps(true, &config)),
            pass_names(build_ir_cond_submodule_pipeline(&config)),
            pass_names(build_ir_while_submodule_pipeline(&config)),
        ];

        for (stable, mut masked) in stable.into_iter().zip(masked) {
            assert_eq!(masked.len(), stable.len() + 1);
            assert!(masked.iter().any(|name| name == "IrDecomposeMaskedFill"));
            masked.retain(|name| name != "IrDecomposeMaskedFill");
            assert_eq!(stable, masked);
        }
    }

    #[test]
    fn u55_only_passes_are_gated_and_ordered_in_parent_and_region_pipelines() {
        for spec in [
            "TOSA-1.0+INT+int16+int4+u55",
            "TOSA-1.0+INT+int16+int4+cf",
            "TOSA-1.0+INT",
        ] {
            let config = PipelineConfig::from_tosa_spec(spec);
            for names in [
                pass_names(lowering_steps(true, &config)),
                pass_names(build_ir_cond_submodule_pipeline(&config)),
                pass_names(build_ir_while_submodule_pipeline(&config)),
            ] {
                for pass in [
                    "IrCastToInt32",
                    "IrBroadcastArgs",
                    "IrDecomposeLargeStrideMaxPool2dForU55",
                ] {
                    assert_eq!(names.iter().any(|name| name == pass), config.is_u55_subset);
                }
                if config.is_u55_subset {
                    let position = |pass| names.iter().position(|name| name == pass).unwrap();
                    assert!(
                        position("IrDecomposeMaxPoolDilation")
                            < position("IrDecomposeLargeStrideMaxPool2dForU55")
                    );
                    assert!(
                        position("IrDecomposeLargeStrideMaxPool2dForU55")
                            < position("IrSizeAdjustInput")
                    );
                    assert!(position("IrLegalizeIntegerArithmetic") < position("IrCastToInt32"));
                    assert!(position("IrCastToInt32") < position("IrBroadcastArgs"));
                    assert!(position("IrBroadcastArgs") < position("IrConvertSum"));
                }
            }
        }
    }

    #[test]
    fn index_and_unfold_modes_follow_the_spec_in_parent_and_region_pipelines() {
        use crate::fx_ir::{IrArg, IrGraph, NodeMeta};
        use crate::graph::ir::DType;
        use crate::store::TensorData;
        use crate::test_ir::{buffer, call, placeholder};

        for spec in ["TOSA-1.0+INT+u55", "TOSA-1.0+INT+cf", "TOSA-1.0+INT"] {
            let config = PipelineConfig::from_tosa_spec(spec);
            for steps in [
                lowering_steps(true, &config),
                build_ir_cond_submodule_pipeline(&config),
                build_ir_while_submodule_pipeline(&config),
            ] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [5], "torch.int8");
                let index = buffer(
                    &mut graph,
                    "index",
                    TensorData {
                        data: [1i32.to_le_bytes(), 2i32.to_le_bytes()].concat(),
                        shape: vec![2],
                        dtype: DType::Int32,
                    },
                );
                graph.nodes[index].meta = NodeMeta::tensor([2], "torch.int32");
                call(
                    &mut graph,
                    "indexed",
                    "aten.index.Tensor",
                    vec![IrArg::Node(x), IrArg::List(vec![IrArg::Node(index)])],
                    [2],
                    "torch.int8",
                );
                call(
                    &mut graph,
                    "unfold",
                    "aten.unfold_copy.default",
                    vec![IrArg::Node(x), IrArg::Int(0), IrArg::Int(3), IrArg::Int(2)],
                    [2, 3],
                    "torch.int8",
                );
                for step in steps {
                    if let PipelineStep::Ir(pass) = step {
                        if matches!(pass.name(), "IrDecomposeIndexTensor" | "IrDecomposeUnfold") {
                            assert!(pass.run(&mut graph));
                        }
                    }
                }
                let gathers = graph
                    .live_nodes()
                    .filter(|node| node.target == "backend.tosa.GATHER.default")
                    .count();
                assert_eq!(gathers, if config.is_u55_subset { 0 } else { 2 });
            }
        }
    }
}
