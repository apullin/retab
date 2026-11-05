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
use crate::pipeline_config::{PipelineConfig, SoftmaxDecompositionConfig};

/// Build the lowering pipeline over the persistent IrGraph.
pub fn build_hybrid_pipeline(
    _exported_program: Option<pyo3::Py<pyo3::PyAny>>,
    config: &PipelineConfig,
) -> Vec<crate::fx_ir::PipelineStep> {
    lowering_steps(true, config)
}

fn lowering_steps(decompose_qdq: bool, config: &PipelineConfig) -> Vec<crate::fx_ir::PipelineStep> {
    use crate::fx_ir::PipelineStep::{Effectful, Ir};

    let mut steps = vec![
        // -- Early decompositions (before fold_qdq) --
        Ir(Box::new(
            crate::fx_annotate_output_dim_order_ir::IrAnnotateOutputDimOrderPass,
        )),
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
        // -- Fold Q/DQ + rescale --
        // Stock sets preserve_partial_binary_tensor_qdq when the TOSA spec
        // supports float or the output format is VGF; RETAB has no VGF path,
        // so the spec flag alone decides. U55 subsets never set it.
        Ir(Box::new(crate::fx_fold_qdq_ir::IrFoldQdqPass {
            preserve_partial_binary_tensor_qdq: config.support_float,
        })),
        Ir(Box::new(crate::fx_decompose_flip_ir::IrDecomposeFlipPass)),
        Ir(Box::new(crate::fx_decompose_roll_ir::IrDecomposeRollPass)),
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
        Ir(Box::new(
            crate::fx_insert_int32_rescale_ir::IrFuseConsecutiveRescalesPass,
        )),
    ];
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
        Ir(Box::new(
            crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
        )),
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
        // Copy lowering compares logical shapes and emits views/repeats for layout conversion.
        Ir(Box::new(fx_remove_noop_ir::IrRemoveNoopPass)),
        Ir(Box::new(
            crate::fx_memory_format_ir::IrToTosaMemoryFormatPass,
        )),
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
    ]);
    steps.push(Ir(Box::new(
        crate::fx_propagate_slice_views_ir::IrPropagateSliceViewsPass,
    )));
    steps.push(Ir(Box::new(
        crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass,
    )));
    steps.push(Ir(Box::new(
        crate::fx_fuse_cascaded_views_ir::IrFuseDuplicateViewsPass,
    )));
    steps
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
    // Loop-carried output alignment runs last, once the producer scales are final.
    steps.push(Ir(Box::new(
        crate::control_flow::IrInsertCarriedOutputRescalePass,
    )));
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
