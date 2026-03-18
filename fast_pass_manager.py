"""Fast pass manager that skips no-op passes based on graph op analysis.

The ARM backend runs ~103 passes during TOSA lowering. For typical models like
cnn_layernorm4d, ~70 of these are no-ops (the ops they target don't exist in the
graph). Each no-op pass still costs ~500ms due to ExportPass graph retracing.

This module provides a monkey-patch for PassManager.__call__ that scans the
graph's op targets before each pass and skips it entirely if none of the pass's
trigger ops are present. This avoids the retrace cost, saving ~35-40s.

Usage:
    from fast_pass_manager import install_smart_skipping, uninstall_smart_skipping

    install_smart_skipping()
    # ... run ArmPassManager pipeline as usual ...
    uninstall_smart_skipping()
"""

import inspect
import logging
import time

from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pass trigger table: maps pass class names to op target substrings.
#
# If ALL of a pass's trigger substrings are absent from str(node.target) for
# every call_function node, the pass is a guaranteed no-op and can be skipped.
#
# Entries with empty lists [] mean "always run" (the pass either does
# non-trivial structural work, takes constructor args that affect behavior,
# or has complex matching logic that can't be summarized by op targets).
#
# Entries not in this table are also always run (unknown passes are never
# skipped).
# ---------------------------------------------------------------------------

PASS_TRIGGERS: dict[str, list[str]] = {
    # Trig / hyperbolic decompositions (call_operator pattern)
    "DecomposeAcoshPass": ["aten.acosh"],
    "DecomposeAsinhPass": ["aten.asinh"],
    "DecomposeCoshPass": ["aten.cosh"],
    "DecomposeSinhPass": ["aten.sinh"],
    "DecomposeTanPass": ["aten.tan"],
    "DecomposeAtanPass": ["aten.atan"],
    "DecomposeAtanhPass": ["aten.atanh"],
    "DecomposeAsinAndAcosPass": ["aten.asin", "aten.acos"],
    "DecomposeErfinvPass": ["aten.erfinv"],
    # Activation / elementwise decompositions
    "DecomposeEluPass": ["aten.elu"],
    "DecomposeExpm1Pass": ["aten.expm1"],
    "DecomposeLog1pPass": ["aten.log1p"],
    "DecomposeSignPass": ["aten.sign"],
    "DecomposeGeluPass": ["aten.gelu"],
    "DecomposeLeakyReLUPass": ["aten.leaky_relu"],
    "DecomposeLogitPass": ["aten.logit"],
    "DecomposeRoundPass": ["aten.round"],
    "DecomposeSqrtPass": ["aten.sqrt"],
    # Reduction / pooling decompositions
    "DecomposeAnyPass": ["aten.any"],
    "DecomposeAdaptiveAvgPool2dPass": ["aten.adaptive_avg_pool2d"],
    "DecomposeMaxPool2dPass": ["aten.max_pool2d"],
    "DecomposeCumsumPass": ["aten.cumsum"],
    # Tensor manipulation decompositions
    "DecomposeEmbeddingPass": ["aten.embedding"],
    "DecomposeIndexSelectToGatherPass": ["aten.index_select"],
    "DecomposeStridedSliceCopyPass": ["aten.as_strided_copy"],
    "DecomposeAsStridedCopyPass": ["aten.as_strided_copy"],
    "DecomposeSliceScatterPass": ["aten.slice_scatter"],
    "DecomposeSelectScatterPass": ["aten.select_scatter"],
    "DecomposeUnfoldToGatherPass": ["aten.unfold"],
    "DecomposeIndexCopyPass": ["aten.index_copy"],
    # Index ops
    "AccumulateIndexPutPass": ["aten.index_put"],
    "DecomposeIndexTensorToGatherPass": ["aten.index.Tensor"],
    "RewriteIndexPutPass": ["aten.index_put"],
    # Norm decompositions (call() graph-walk pattern)
    "DecomposeBatchNormNoStatsPass": [
        "aten._native_batch_norm_legit",
        "aten.batch_norm",
        "aten.native_batch_norm",
    ],
    "DecomposeGroupNormPass": ["aten.group_norm"],
    "FuseBatchNorm2dPass": ["aten.batch_norm", "aten._native_batch_norm"],
    # Math decompositions
    "DecomposeIntPowPass": ["aten.pow.Tensor_Scalar"],
    "DecomposeFloorDividePass": ["aten.floor_divide"],
    "DecomposeRemainderPass": ["aten.remainder"],
    "DecomposeDivTensorModePass": ["aten.div.Tensor_mode"],
    "DecomposeDivPass": ["aten.div"],
    "DecomposeGluPass": ["aten.glu"],
    "DecomposeAddmmPass": ["aten.addmm"],
    # Softmax
    "DecomposeSoftmaxPass": ["aten.softmax", "aten._softmax"],
    "DecomposeSoftmaxUnstablePass": ["aten.softmax", "aten._softmax"],
    "DecomposeMaskedFillPass": ["aten.masked_fill"],
    # Conv / linear
    "Conv1dUnsqueezePass": ["aten.conv1d"],
    "DecomposeGroupedConvPass": ["aten.convolution", "aten.conv2d"],
    "DecomposeConvWithInt16ActivationPass": ["aten.convolution"],
    # Reshape / view ops
    "ConvertSplitToSlicePass": ["aten.split"],
    "DecomposeSelectPass": ["aten.select_copy", "aten.select"],
    "ConvertSqueezesToViewPass": ["aten.squeeze"],
    "ConvertPermuteSingletonToViewPass": ["aten.permute"],
    "ConvertExpandCopyToRepeatPass": ["aten.expand_copy"],
    "UnsqueezeBeforeRepeatPass": ["aten.repeat"],
    # Rewrite passes
    "RewriteUpsamplePass": ["aten.upsample"],
    "RewritePadPass": ["aten.pad", "aten.constant_pad"],
    "RewriteSlicePass": ["aten.slice_copy"],
    "RewriteMatmulPass": ["aten.mm", "aten.bmm", "aten.matmul"],
    "RewriteLeLtToGeGtPass": ["aten.le", "aten.lt"],
    "RewriteBoolBitwiseToLogicalPass": ["aten.bitwise"],
    # Type conversion
    "ConvertMmToBmmPass": ["aten.mm"],
    "DecomposeNotEqualPass": ["aten.ne"],
    "CanonicalizeGatherPass": ["aten.gather", "aten.scatter"],
    "RemoveGetItemPass": ["getitem"],
    # Attention / similarity
    "DecomposeScaledDotProductAttentionPass": ["aten.scaled_dot_product_attention"],
    "DecomposeCosineSimilarityPass": ["aten.cosine_similarity"],
    "DecomposeLinalgVectorNormPass": ["aten.linalg_vector_norm"],
    "DecomposeMatmulPass": ["aten.matmul"],
    "DecomposeTrilPass": ["aten.tril"],
    # Misc
    "ConvertMinMaxPass": ["aten.min", "aten.max", "aten.amin", "aten.amax"],
    "DecomposeWhereScalarOtherPass": ["aten.where"],
    "ReplaceInfAndLimitValuesPass": [],  # always run (structural)
    "DecomposeTOSAUnsupportedClampPass": ["aten.clamp"],
    # Passes that always run (complex matching or structural transforms):
    "AnnotateOutputDimOrderPass": [],
    "FuseQuantizedActivationPass": [],
    "RewriteBoolToFp32CastViaInt8Pass": [],
    "ConvertToClampPass": [],
    "ConvertELUParamsPass": [],
    "ControlFlowConstInlinePass": [],
    "NormalizeWhileInitialArgsPass": [],
    "FoldAndAnnotateQParamsPass": [],
    "FuseDuplicateUsersPass": [],
    "DecomposeLinearPass": [],
    "InsertRescaleInt32Pass": [],
    "InsertControlFlowRescalesPass": [],
    "DecomposeQuantNodesPass": [],
    "QuantizeClampArgumentsPass": [],
    "DecomposeAddSubAlphaPass": [],
    "ReplaceScalarWithTensorByProfilePass": [],
    "ConvertFullLikeToFullPass": [],
    "MatchArgDtypePass": [],
    "UnsqueezeScalarPlaceholdersPass": [],
    "MatchArgRanksPass": [],
    "DecomposeLayerNormPass": [],
    "DecomposeVarPass": [],
    "DecomposeMeanDimPass": [],
    "DecomposeSumPass": [],
    "DecomposeAvgPool2dPass": [],
    "DecorateFp32toInt32CastingPass": [],
    "ComputeConstantOpsAOTPass": [],
    "FuseConstantArgsPass": [],
    "SizeAdjustInputPass": [],
    "CastToInt32Pass": [],
    "BroadcastArgsPass": [],
    "FuseViewCopyTransformPass": [],
    "InsertTableOpsPass": [],
    "RewriteConvPass": [],
    "CastInt64BuffersToInt32Pass": [],
    "FuseEqualPlaceholdersPass": [],
    "ToTosaMemoryFormatPass": [],
    "RemoveNoopPass": [],
    "InsertRescalePass": [],
    "PromoteBoolOperandsPass": [],
}


def _get_graph_targets(module) -> set[str]:
    """Collect all str(node.target) values for call_function nodes. O(N) scan."""
    if not isinstance(module, GraphModule):
        return set()
    targets = set()
    for node in module.graph.nodes:
        if node.op == "call_function":
            targets.add(str(node.target))
    return targets


def _should_skip_pass(pass_name: str, graph_targets: set[str]) -> bool:
    """Check if a pass can be skipped because its trigger ops aren't present."""
    triggers = PASS_TRIGGERS.get(pass_name)
    if triggers is None:
        return False  # unknown pass, never skip
    if not triggers:
        return False  # empty trigger list means always run
    # Check if any trigger substring appears in any graph target string
    for trigger in triggers:
        for target in graph_targets:
            if trigger in target:
                return False  # found a matching op, must run this pass
    return True  # no matching ops found, safe to skip


# ---------------------------------------------------------------------------
# Monkey-patch for PassManager.__call__
# ---------------------------------------------------------------------------

_original_pm_call = None


def _smart_pm_call(self, module):
    """Drop-in replacement for PassManager.__call__ with smart pass skipping.

    Same contract as the original: runs passes in order, returns PassResult.
    But before each pass, checks if the graph contains ops the pass targets.
    If not, skips the pass entirely (no ExportPass retracing).
    """
    # Order passes based on constraints (same as original)
    if not self._validated:
        self.solve_constraints()

    # Check graph invariants
    self.check(module)

    overall_modified = False
    skipped = 0
    total = len(self.passes)
    total_skip_time = 0.0

    # Pre-scan graph targets
    graph_targets = _get_graph_targets(module)

    for step in range(self.steps):
        modified = False

        for i, fn in enumerate(self.passes):
            fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__

            # Smart skip check
            if _should_skip_pass(fn_name, graph_targets):
                skipped += 1
                logger.debug("Skipping no-op pass '%s'", fn_name)
                continue

            logger.debug("Running pass '%s'", fn_name)
            t0 = time.perf_counter()

            try:
                res = fn(module)

                if not isinstance(res, PassResult) and not hasattr(
                    res, "graph_module"
                ):
                    raise TypeError(
                        f"The result of the pass {fn_name} should be type PassResult."
                        + "Please wrap it with pass_result_wrapper()"
                    )
                module = res.graph_module
                modified = modified or res.modified

                if isinstance(module, GraphModule):
                    module.recompile()

                    # Refresh graph targets after graph-modifying passes
                    if res.modified:
                        graph_targets = _get_graph_targets(module)

                if self.run_checks_after_each_pass:
                    self.check(module)

            except Exception as e:
                prev_pass_names = [
                    p.__name__ if inspect.isfunction(p) else type(p).__name__
                    for p in self.passes[:i]
                ]
                msg = f"An error occurred when running the '{fn_name}' pass after the following passes: {prev_pass_names}"
                raise Exception(msg) from e  # noqa: TRY002

            elapsed = time.perf_counter() - t0
            total_skip_time += elapsed

        overall_modified = overall_modified or modified
        if not modified:
            break

    ran = total - skipped
    logger.info(
        f"[SmartPassManager] Ran {ran}/{total} passes "
        f"(skipped {skipped} no-op passes)"
    )

    return PassResult(module, overall_modified)


def install_smart_skipping():
    """Monkey-patch PassManager.__call__ to add smart pass skipping."""
    global _original_pm_call

    from executorch.exir.pass_manager import PassManager

    if _original_pm_call is not None:
        logger.debug("[fast_pass_manager] Smart skipping already installed")
        return

    _original_pm_call = PassManager.__call__
    PassManager.__call__ = _smart_pm_call
    logger.info("[fast_pass_manager] Installed smart pass skipping")


def uninstall_smart_skipping():
    """Restore the original PassManager.__call__."""
    global _original_pm_call

    from executorch.exir.pass_manager import PassManager

    if _original_pm_call is None:
        return

    PassManager.__call__ = _original_pm_call
    _original_pm_call = None
    logger.info("[fast_pass_manager] Restored original PassManager.__call__")
