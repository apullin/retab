# RETAB's pipeline and rc4's ArmPassManager

RETAB's Rust pipelines stand in for `ArmPassManager._tosa_pipeline`, the pass
list that stock ExecuTorch v1.6.0-rc4 runs in `TOSABackend._preprocess`
(`backends/arm/_passes/arm_pass_manager.py`, cited as APM:<line>, as
[../src/fx_pipeline.rs](../src/fx_pipeline.rs) cites it). This page maps each
RETAB step to its rc4 step. It explains the steps only RETAB has, records where
RETAB's order differs before phase 6, and lists the rc4 steps RETAB has no step
for. Linear APU-125 tracks the follow-up.

## The two pipelines

rc4's `_tosa_pipeline` (APM:535-776) has 161 steps in seven phases. rc4 runs
the whole list again on each cond and while region.

| Phase | APM | Steps |
|---|---|---:|
| 1. Before the Q/DQ fold | 542-562 | 17 |
| 2. Q/DQ fold and INT32 rescales | 565-598 | 12 |
| 3. After the fold | 601-651 | 44 |
| 4. Scalars to tensors, dtypes and ranks | 654-666 | 9 |
| 5. After scalar removal | 669-712 | 40 |
| 6. ATen to TOSA | 715-749 | 26 |
| 7. Postprocessing | 752-770 | 13 |

`lowering_steps` in `src/fx_pipeline.rs` builds all of RETAB's pipelines:

- **Parent** (`build_main_pipeline`): 99 fixed steps, plus IrDecomposeMaskedFill
  when softmax is MASKED and three steps on Ethos-U55. That gives 100 steps for
  TOSA-1.0+INT and Ethos-U85, and 102 for Ethos-U55, whose default softmax is
  STABLE.
- **cond region** (`build_ir_cond_submodule_pipeline`): the parent list without
  two steps, so 98 steps (100 on U55).
- **while region** (`build_ir_while_submodule_pipeline`): the cond list plus two
  steps, so 100 steps (102 on U55).

These are v0.4.1's counts. The steps listed under
[Removed steps](#removed-steps) have since left all three lists.

The unit test `pipelines_build_rc4s_order_in_parent_and_regions` pins phase 6
and several of the other slots below. Of rc4's 161 steps, 107 have a RETAB
step: IrDecomposeOps alone covers 18 of
them, and IrCastInt64ToInt32Ep covers rc4's three CastInt64BuffersToInt32Pass
runs. The other 54 are listed under
[rc4 steps RETAB has no step for](#rc4-steps-retab-has-no-step-for).

"Fired" counts come from 325 sampled pipeline runs; see
[How the firing data was taken](#how-the-firing-data-was-taken).

## Parent pipeline

Numbers are the TOSA-1.0+INT positions at v0.4.1, which `RETAB_LOG` printed
for that spec; "U55" marks the steps only Ethos-U55 gets. The steps removed
since leave gaps ([Removed steps](#removed-steps)). **Moved** steps run
away from their rc4 slot ([Order differences](#order-differences-before-phase-6)).
**RETAB-only** steps are explained in [RETAB-only steps](#retab-only-steps).

| # | RETAB step | rc4 | Notes |
|---:|---|---|---|
| 1 | IrNormalizeDelegateIOLayout | NormalizeDelegateIOLayoutPass, APM:544 | Parent only. |
| 2 | IrFuseBatchNorm2d | FuseBatchNorm2dPass, :606 | Moved. |
| 3 | IrDecomposeOps | 18 decompositions, see [rules](#irdecomposeops-rules) | Moved. |
| 4 | IrDecorateFpToIntCast | DecorateFp32toInt32CastingPass, :685 | Moved. |
| 5 | IrFuseQuantActivation | FuseQuantizedActivationPass, :545 | |
| 6 | IrConvertClamp | ConvertToClampPass, :550 | |
| 7 | IrDecomposeLayerNorm | DecomposeLayerNormPass, :553 | |
| 8 | IrDecomposeVar | DecomposeVarPass, :554 | |
| 9 | IrRecordControlFlowRescaleTargets | (InsertControlFlowRescalesPass, :595) | RETAB-only. |
| 10 | IrFoldQdq | FoldAndAnnotateQParamsPass, :574 | |
| 11 | IrFuseConsecutiveClamps | FuseConsecutiveClampsPass, :586 | |
| 12 | IrFuseDuplicateUsers | FuseDuplicateUsersPass, :587 | |
| 13 | IrFuseIdenticalInputTransforms | FuseIdenticalInputTransformsPass, :591 | |
| 14-15 | IrDecomposeFlip, IrDecomposeRoll | DecomposeFlipPass :630, DecomposeRollPass :631 | Moved. |
| 16 | IrDecomposeLinear | DecomposeLinearPass, :592 | |
| 17 | IrInsertQdqRescale | InsertRescalePass, :766 | Moved. |
| 18 | IrInsertInt32Rescale | InsertRescaleInt32Pass, :593 | |
| 19 | IrFuseConsecutiveRescales | FuseConsecutiveRescalesPass, :594 | |
| 20 | IrDecomposeQuant | DecomposeQuantNodesPass, :596 | Parent only. |
| 21 | IrConvertSplitToSlice | ConvertSplitToSlicePass, :603 | |
| 22 | IrReplaceInfAndLimitValues | (ReplaceInfAndLimitValuesPass, quantizer side, :881) | RETAB-only. |
| 23 | IrDecomposeMaskedFill | DecomposeMaskedFillPass, :610 | MASKED softmax only, as rc4's skip list (:322-326). |
| 24-26 | IrDecomposeRound, IrDecomposeSqrt, IrDecomposeSign | :611, :617, :629 | |
| 27 | IrDecomposeEmbedding | DecomposeEmbeddingPass, :637 | |
| 28 | IrCanonicalizeGather | CanonicalizeGatherPass, :549 | Moved. |
| 29 | IrDecomposeIndexTensor | DecomposeIndexTensorToGatherPass, :643 | U55 mode as rc4. |
| 30 | IrDecomposeIndexSelect | DecomposeIndexSelectToGatherPass, :639 | |
| 31 | IrDecomposeUnfold | DecomposeUnfoldToGatherPass, :636 | U55 mode as rc4. |
| 33 | IrDecomposeSliceScatter | DecomposeSliceScatterPass, :641 | |
| 34 | IrDecomposeStridedSlice | DecomposeStridedSliceCopyPass, :640 | |
| 36 | IrDecomposeLogit | DecomposeLogitPass, :609 | |
| 37 | IrDecomposeNotEqual | DecomposeNotEqualPass, :671 | Moved. |
| 38 | IrReplaceScalarWithTensor | ReplaceScalarWithTensorByProfilePass, :656 | Scalar shifts only; the serializer takes the other scalar operands. |
| 39 | IrDecomposeDivTensorMode | DecomposeDivTensorModePass, :678 | Moved. |
| 40 | IrDecomposeAddmm | DecomposeAddmmPass, :621 | Moved. |
| 41 | IrDecomposeCumsum | DecomposeCumsumPass, :689 | Moved. |
| 44 | IrDecomposeAsStridedCopy | DecomposeAsStridedCopyPass, :690 | Moved. |
| 45 | IrQuantizeClampArgs | QuantizeClampArgumentsPass, :604 | Moved. |
| 46 | IrConvertFullLike | ConvertFullLikeToFullPass, :661 | |
| 47 | IrMatchArgRanks | MatchArgRanksPass, :664 | |
| 48-50 | IrFuseZeroInputConstantsEp, IrFuseShapeOnlyConstantsEp, IrFuseConstantInputsEp (`matmul_only`) | none | RETAB-only: constant sweep 1. |
| 51 | IrDecomposeAdaptiveAvgPool | DecomposeAdaptiveAvgPool2dPass, :647 | Moved. |
| 52 | IrDecomposeAvgPoolPad | DecomposeAvgPool2dPass, :649 | Moved. |
| 53 | IrNormalizeMaxPool2dInputRank | NormalizeMaxPool2dInputRankPass, :692 | |
| 54 | IrDecomposeMaxPoolDilation | DecomposeMaxPool2dPass, :693 | |
| U55 | IrDecomposeLargeStrideMaxPool2dForU55 | :694 | rc4 checks for U55 inside the pass. |
| 55 | IrDecomposeVar | none | RETAB-only: a second instance. |
| 56 | IrDecomposeMean | DecomposeMeanDimPass, :555 | Moved. |
| 57 | IrConvertBoolSum | ConvertBoolSumPass, :626 | Moved. |
| 58 | IrLegalizeIntegerArithmetic | none | RETAB-only. |
| U55 | IrCastToInt32, IrBroadcastArgs | :704, :705 | rc4 checks for U55 inside the passes. |
| 59 | IrConvertSum | DecomposeSumPass, :707 | |
| 60 | IrInsertTableOpsEp | InsertTableOpsPass, :708 | Records TABLE effects; the serializer emits the TABLEs. |
| 61 | IrDecomposeGroupedConv | DecomposeGroupedConvPass, :635 | Moved. |
| 62 | IrSizeAdjustInput | SizeAdjustInputPass, :695 | |
| 63 | IrRewriteAvgPool2d | RewriteAvgPool2dPass, :698 | |
| 64 | IrMaterializeConstants | ComputeConstantOpsAOTPass and FuseConstantArgsPass, :699-700 | |
| 65 | IrRewriteSqueeze | ConvertSqueezesToViewPass, :703 | Also turns transpose_copy into permute_copy. |
| 66 | IrFuseCascadedViews | none | RETAB-only. |
| 67 | IrCastInt64ToInt32Ep | CastInt64BuffersToInt32Pass, :638, :701, :754 | Once, where rc4 runs it three times. |
| 68-70 | IrNormalizeIndexPutNoneIndices, IrNormalizeIndexPutBoolMask, IrRewriteIndexPut | :673, :674, :675 | Moved. |
| 71-72 | IrFuseZeroInputConstantsEp, IrFuseShapeOnlyConstantsEp | none | RETAB-only: constant sweep 2. |
| 73 | IrRemoveNoop | RemoveNoopPass, :709 | |
| 74-92 | rc4's phase 6, 19 steps | :717-744 | [Phase 6](#phase-6). |
| 93-95 | IrFuseZeroInputConstantsEp, IrFuseShapeOnlyConstantsEp, IrFuseConstantInputsEp (`matmul_only`) | none | RETAB-only: constant sweep 3. |
| 96 | IrFuseEqualPlaceholdersEp | FuseEqualPlaceholdersPass, :755 | |
| 97 | IrRemoveNoop | RemoveNoopPass, :762 | |
| 98 | IrFuseDuplicateUsers | FuseDuplicateUsersPass, :765 | |
| 99 | IrControlFlowOutputGuard | none | RETAB-only. |
| 100 | IrEnsureUniqueOutputNodes | EnsureUniqueOutputNodesPass, :768 | Last, as in rc4. |

### Phase 6

Steps 74-92 are rc4's phase 6, in rc4's order:

| # | RETAB step | rc4 |
|---:|---|---|
| 74 | IrRewriteUpsample | RewriteUpsamplePass, :717 |
| 75 | IrRewriteMaxPool2d | RewriteMaxPool2dPass, :718, plus RemoveGetItemPass's share (:605) |
| 76 | IrRewriteConv | RewriteConvPass, :721 |
| 77 | IrRewriteMatmulRescale | RewriteMatmulPass, :724; the serializer turns aten mm and bmm into MATMUL (rc4's ConvertMmToBmmPass, :679) |
| 78 | IrFuseViewCopyTransform | FuseViewCopyTransformPass, :725 |
| 79-80 | IrPropagateViewCopyPermuteDown, IrPropagateViewCopyPermuteUp | :726-727, with RemovePermutesAroundElementwiseTosaOps inside, as in rc4 |
| 81 | IrRewritePad | RewritePadPass, :728 |
| 82 | IrMoveDataMovementOpsToSmallerDtype | :731 |
| 83 | IrMatchArgRanks | MatchArgRanksPass, :732 |
| 84 | IrRewriteHighRankSingletonPermute | :733 |
| 85 | IrRewriteCatSlice | :734 |
| 86 | IrFuseConsecutiveConcats | :735 |
| 87 | IrDecomposePermuteForU55 | :736; asks rc4's Vela probe through Python, and does nothing off U55 |
| 88 | IrRewriteSlice | :737 |
| 89 | IrFuseConsecutiveSlices | :738 |
| 90 | IrRemoveNoop | RemoveNoopPass, :741 |
| 91 | IrCanonicalizeViewCopyPermute | :742 |
| 92 | IrFuseViewCopyTransform | FuseViewCopyTransformPass, :744 |

rc4's other seven phase-6 steps have no RETAB step.
DecomposeAdaptiveMaxPool2dPass (:719), FoldScalarMulIntoConvPass (:720) and the
two MXFP rewrites (:722-723) never fire for TOSA-1.0 INT.
ResolveViewCopyInferredDimPass, InsertConstShapesPass and
InsertDataLayoutCastsPass (:745-747) are the serializer's work.

### IrDecomposeOps rules

IrDecomposeOps (step 3) is a rule table that stands in for 18 rc4 passes:

| Rule | rc4 | On a quantized op |
|---|---|---|
| atanh, tan, sinh, cosh, asinh, asin and acos, atan, acosh | :619, :620, :628, :614, :613, :615, :618, :612 | Left for a TABLE |
| elu, selu, celu | ConvertEluFamilyToEluPass :556, DecomposeEluPass :622 | Left for a TABLE |
| gelu | DecomposeGeluPass, :633 | Left for a TABLE |
| erfinv | DecomposeErfinvPass, :616 | Left for a TABLE |
| pow.Tensor_Scalar with an integer exponent | DecomposeIntPowPass, :624 | Left for a TABLE |
| group_norm | DecomposeGroupNormPass, :552 | Decomposed |
| batch_norm in inference | DecomposeBatchNormNoStatsPass, :607 | Decomposed |
| softmax, safe softmax, log_softmax | DecomposeSoftmaxPass, :682 | Decomposed |
| max.dim, min.dim | ConvertMinMaxPass, :683 | Decomposed |
| add and sub with alpha other than 1 | DecomposeAddSubAlphaPass, :634 | Decomposed |
| stack, unbind | none | Decomposed |

"Left for a TABLE" means the rule skips an op that sits between a dequantize and
a quantize, and the serializer emits it as a TABLE. That test needs the Q/DQ
nodes, so the table runs before IrFoldQdq. IrDecomposeOps fired on none of the
sampled graphs. The quantizer side or the edge decomposition table had already
decomposed these ops, and the edge table removes stack and unbind on every
graph ([RETAB's input](#retabs-input)).

## Regions

**cond.** `lowering_steps(false, …)` leaves out two parent-only steps:

- IrNormalizeDelegateIOLayout (1). rc4 runs it on each region, where it does
  nothing because region IO gets the parent's contiguous values. RETAB imports
  a region from the original edge submodule, whose IO keeps the export's dim
  order, so the step could permute region IO that rc4 leaves alone (P2
  critique A2).
- IrDecomposeQuant (20). A region's boundary Q/DQ nodes must survive until the
  boundary contracts align region IO at serialization. rc4 inserts those
  RESCALEs at :595, the step before it decomposes Q/DQ nodes (:596).

In a region, IrMaterializeConstants makes `get_attr` constants instead of
placeholders, and IrRewriteConv works in region mode.

**while.** The cond list, plus two steps:

- IrStripWhileBoundaryQ, first. It removes the quantize after each body input,
  stamping its qparams on the input, and records the output dequantizes'
  qparams on the output node. RETAB-only. It fired in all 14 sampled while
  bodies.
- IrInsertCarriedOutputRescale, just before IrEnsureUniqueOutputNodes. It
  rescales each loop-carried output to its input's qparams, the share of rc4's
  InsertControlFlowRescalesPass (:595) that falls on while bodies. It fired in
  12 of the 14.

## Order differences before phase 6

Phase 6 runs in rc4's order, and so does what RETAB keeps of phase 7 (:755,
:762, :765, :768). The steps below run away from their rc4 slots, two of them
phase-7 steps that RETAB runs early (:754, :766). "Historical" means no reason
is recorded: RETAB's order predates the rc4 port, and the gate passes with it
(op suite 2374/0, corpus parity). Each move to an rc4 slot needs its own gate.

| Step (#) | rc4 slot | RETAB slot | Why | Fired |
|---|---|---|---|---:|
| IrFuseBatchNorm2d (2) | :606, phase 3 | Before the fold | Moot: the port fuses only float32 weights, which INT graphs don't have; the quantizer folds batch norm into the conv before RETAB sees the graph. | 0 |
| IrDecomposeOps (3) | Phases 1, 3 and 5 | Before the fold | Deliberate: its TABLE test needs the Q/DQ nodes ([rules](#irdecomposeops-rules)). | 0 |
| IrDecorateFpToIntCast (4) | :685, phase 5 | Before the fold | Historical. | 0 |
| IrDecomposeFlip, IrDecomposeRoll (14-15) | :630-631, phase 3 | Before IrDecomposeLinear (:592) | Historical. Both are data movement that keeps its input's qparams, so the rescale steps around them are unaffected. | 0 |
| IrInsertQdqRescale (17) | InsertRescalePass, :766, phase 7 | Right after the fold | Historical. | 10 |
| IrCanonicalizeGather (28) | :549, phase 1 | After the fold | Historical. | 0 |
| IrDecomposeNotEqual (37) | :671, phase 5 | Before IrReplaceScalarWithTensor (:656) | RETAB kept v1.3.1's slot when upstream moved the pass. Either order ends in EQUAL and LOGICAL_NOT: rc4 makes ne.Scalar a tensor op first, and RETAB's serializer gives eq.Scalar its constant. | 0 |
| IrDecomposeDivTensorMode, IrDecomposeCumsum, IrDecomposeAsStridedCopy (39, 41, 44) | :678, :689, :690, phase 5 | Before IrConvertFullLike and IrMatchArgRanks (:661, :664) | Historical. The binary ops they create are rank-matched at step 47 rather than in phase 6 (:732). | 0 |
| IrDecomposeAddmm (40) | :621, phase 3 | After IrReplaceScalarWithTensor (:656) | Historical. It creates no `.Scalar` op, so the scalar rewrite's position doesn't matter. | 0 |
| IrQuantizeClampArgs (45) | :604, phase 3 | After the phase-4 and phase-5 decompositions | Historical. It keeps rc4's one constraint (:581-586): FuseConsecutiveClamps runs first, while clamp bounds are still float. | 15 |
| IrDecomposeAdaptiveAvgPool, IrDecomposeAvgPoolPad, IrDecomposeVar (second), IrDecomposeMean (51-52, 55-56) | :647, :649, :554-555 | After IrMatchArgRanks (:664) | Recorded in the code as P2 critique B9. The binary ops they create miss this rank match, so U55's IrBroadcastArgs sees them unmatched; phase 6's IrMatchArgRanks (:732) matches them before serialization. | 0 |
| IrConvertBoolSum (57) | :626, phase 3 | Phase 5 | Historical. | 0 |
| IrCastToInt32, IrBroadcastArgs (U55), IrConvertSum, IrInsertTableOpsEp (59-60) | :704-708 | Before :695-703, SizeAdjustInput through ConvertSqueezesToView | Historical. IrInsertTableOpsEp only records TABLE effects, and the layout passes treat a pending table op as a TABLE wherever it runs (`layout::pending`). | 85 and 48 |
| IrDecomposeGroupedConv (61) | :635, phase 3 | After :708, before :695 | Historical. IrMaterializeConstants (64) folds its weight slices, as rc4 folds them at :699-700. | 0 |
| IrCastInt64ToInt32Ep (67) | :638, :701, :754 | Once, after :703 | Deliberate: RETAB's importer and IrMaterializeConstants narrow int64 constant data to int32 when it fits, so the int64 buffers that rc4's later runs cast never exist as data here. The step retypes the exported program's int64 buffers. | 0 |
| IrNormalizeIndexPutNoneIndices, IrNormalizeIndexPutBoolMask, IrRewriteIndexPut (68-70) | :673-675, start of phase 5 | After the constant folds | Historical. | 0 |

IrInsertQdqRescale fired only on test_add's int32 cases and test_to_copy's
integer-to-float casts. IrCastToInt32 and IrBroadcastArgs never fired, while
IrConvertSum and IrInsertTableOpsEp fired 85 and 48 times. Within phase 3,
steps 21-36 also run in a slightly different order from :603-643, without
crossing a phase boundary.

## RETAB-only steps

| Step (#) | Why it exists | Fired | Verdict |
|---|---|---:|---|
| IrRecordControlFlowRescaleTargets (9) | RETAB inserts cond and while boundary RESCALEs at serialization, where rc4 inserts them at :595. This step records each output's target qparams from the quantize after its getitem, before IrFoldQdq folds that quantize. | n/a | Keep. It writes metadata only, so it never reports a change; any graph with a cond or while needs it. |
| IrReplaceInfAndLimitValues (22) | Replaces ±inf and float32 limits in literal arguments, which rc4 does only on the quantizer side (:881). | 0 | Duplicates RETAB's input contract. Remove with a gate. |
| Constant sweep 1 (48-50) | RETAB's older constant folding. It records MaterializeTensorNode effects, which ingest applies by node name. | 102 | Probably redundant. IrMaterializeConstants (64) re-plans with the same planners and gives each node it folds a new name, which voids the sweep's effect for that node. The sweep fired on 102 sampled graphs and step 64 on 106. An A/B settles it: skip the sweep, then byte-compare the corpus and the op suite. |
| IrDecomposeVar, second (55) | A second pass over var nodes, left from RETAB's older order. | 0 | Can't fire. IrDecomposeLayerNorm (7) is the only step that creates var nodes, and the first IrDecomposeVar (8) follows it. A var the first leaves has no input shape, and nothing in between gives it one. Remove with a gate. |
| IrLegalizeIntegerArithmetic (58) | TOSA's integer ADD, SUB and REDUCE_SUM take int32 operands. The step casts raw (unquantized) int8 or int16 operands of add, sub and sum to int32, and the result back. rc4's InsertRescaleInt32Pass handles only quantized operands. | 2 | Keep. It fired only on test_to_copy's two REDUNDANT_CAST cases, raw int8 and int16 adds that stock fails to lower; rc4 marks them as expected failures (APU-77). |
| IrFuseCascadedViews (66) | Fuses view chains and keeps the last view, where rc4's FuseViewCopyTransform (:725, :744) keeps the first. | 25 | Keep. APU-161 measured it: skipping it renamed tensors in 6 of 27 corpus delegates. It fired on 29 of 1,169 view-related op tests, and skipping it changed 28 of their dumps, 13 of them in op order, without getting closer to stock. |
| Constant sweeps 2 and 3 (71-72, 93-95) | The sweep-1 planners again, after IrMaterializeConstants and after phase 6. | 0 | Remove with a gate. |
| IrControlFlowOutputGuard (99) | Sends the graph to stock when layout copies hide a control-flow output's boundary target and step 9 recorded none. | n/a | Keep as a guard. It never changes the graph, and it requested no stock fallback in the sample. |
| IrStripWhileBoundaryQ (while regions) | See [Regions](#regions). | 14 | Keep. |

So five RETAB-only steps could never fire on the graphs stock's edge pipeline
produces. Four are removed ([Removed steps](#removed-steps)); the fifth is the
second IrDecomposeVar. Three more fired on no sampled graph:
IrReplaceInfAndLimitValues and constant sweeps 2 and 3.

### Removed steps

These RETAB-only steps fired on none of the 325 sampled runs, and can't fire,
so they were removed after v0.4.1. The ops they decomposed never reach RETAB:
ExecuTorch's edge decomposition table removes them, rc4's partitioner keeps
none of them (`ops_to_not_decompose`, `backends/arm/tosa/partitioner.py:890`),
and no RETAB step creates them. RETAB's edge fast paths, which `install()`
refuses, apply the same table. Should one of these ops reach RETAB anyway, as a
custom partition op for instance, the serializer stops with "RETAB has no TOSA
emitter"; rc4 has no backend pass or node visitor for these ops either.

| Step (v0.4.1 #) | Op | Why it can't fire |
|---|---|---|
| IrDecomposeSelectScatter (32) | select_scatter | The edge table removes `select_scatter.default`. On INT graphs rc4's quantizer-side DecomposeSelectScatterPass (:804) decomposes it first, so test_select_scatter's tosa_INT cases reach RETAB as arange, eq, expand_copy and where. |
| IrDecomposeTril (35) | tril | The edge table removes `tril.default`, and on INT graphs rc4's quantizer-side DecomposeTrilPass (:817) decomposes it first. |
| IrDecomposeRepeatInterleave (42) | repeat_interleave | The edge table removes `repeat_interleave.self_int`, so test_repeat's interleave case reaches RETAB as unsqueeze_copy, expand_copy and view_copy. rc4 has no such pass. |
| IrDecomposePixelShuffle (43) | pixel_shuffle, pixel_unshuffle | The edge table removes both, so they reach RETAB as view_copy and permute_copy. rc4 has no such pass. |

Removing them changed no output. The corpus delegates stayed byte-identical,
and so did the TOSA RETAB produced for each tosa_INT test of the op-test files
that build these ops (test_select_scatter, test_tril and test_repeat) and for
INT runs of test_pixel_shuffling's modules. That file has no tosa_INT case: its
TOSA tests use FP and mixed-precision specs, which RETAB leaves to stock.

## rc4 steps RETAB has no step for

The 54 rc4 steps without a RETAB step fall into six groups.

**The serializer does the work (15).** RETAB keeps ATen targets until
serialization, and its emitters lower them directly:

- ExirToTosaPass (:757): `src/ingest/target_map.rs` maps ATen targets to TOSA ops.
- ConvertMmToBmmPass (:679): mm and bmm become MATMUL.
- ConvertExpandCopyToRepeatPass and UnsqueezeBeforeRepeatPass (:687-688):
  expand_copy and repeat become TILE.
- RewriteLeLtToGeGtPass (:658), DecomposeAnyPass (:684) and DecomposeSelectPass
  (:702): le and lt, any, and select_copy have emitters.
- DecomposeTOSAUnsupportedClampPass (:551): the clamp emitter lowers int32
  clamps and tensor bounds through MAXIMUM and MINIMUM.
- RewriteBoolBitwiseToLogicalPass (:676): the bitwise emitters emit LOGICAL_*
  ops for bool outputs.
- InsertConstShapesPass (:746), DeduplicateConstShapesPass (:767),
  ResolveViewCopyInferredDimPass (:745, :759) and InsertDataLayoutCastsPass
  (:710, :747): the serializer writes CONST_SHAPE operands, view sizes and
  layout casts itself.

**The quantizer side already did it (11).** On INT graphs the annotation
pipeline has run these before quantization, so rc4's backend copies normally
find nothing to do: DecomposeLeakyReLUPass (:659; quantizer side :842 under
DECOMPOSE, the only leaky_relu setting RETAB accepts), DecomposePReLUPass
(:660; :834), DecomposeRemainderPass (:677; :819), DecomposeFloorDividePass
(:632; :820), DecomposeGluPass (:680; :862), DecomposeDivPass (:681; :863),
DecomposeMaxPool1dPass (:691; :867), AccumulateIndexPutPass (:642; :872),
PromoteBoolOperandsPass (:627; :815), ControlFlowConstInlinePass (:558; :849)
and NormalizeWhileInitialArgsPass (:559; :856).

**TABLE ops (3).** On INT graphs these ops are quantized, and RETAB emits their
TABLEs at serialization: ConvertELUParamsPass (:557), which stashes a quantized
ELU's float parameters for its TABLE, and DecomposeExpm1Pass (:623) and
DecomposeLog1pPass (:625), which skip quantized ops.

**Never fire for TOSA-1.0 INT, or the graph goes to stock (20):**

- Float or mixed profiles only: RewriteBoolToFp32CastViaInt8Pass (:546,
  INT+FP), DecomposeTopKPass (:547, TopK is FP only), DecomposeIsInfAndIsNanPass
  (:608, isinf and isnan stay on the CPU for INT), CastIntComparisonInputsPass
  (:657, FP-only specs), DecomposeProdPass (:706, prod is FP only),
  FoldScalarMulIntoConvPass (:720, float weights only).
- Dynamic W8A8, whose runtime qparams are computed in float (INT+FP specs):
  DecomposeChooseQParamsSymmetricPass, DetectDynamicW8A8LinearPass and
  LowerDynamicW8A8LinearPass (:569-573).
- Symbolic shapes, which RETAB sends to stock:
  DecomposeDynamicAdaptiveAvgPool2dPass (:648), DecomposeDynamicFullPass (:686),
  InsertDynamicPaddingPass (:758), SymbolicToTosaShapesPass (:760) and
  FuseConsecutiveConcatShapesPass (:761).
- The TOSA 1.1 adaptive-pool rewrites, RewriteAdaptiveAvgPool2dPass (:697) and
  DecomposeAdaptiveMaxPool2dPass (:719), and the MXFP rewrites,
  RewriteMXFPConv2dPass and RewriteMXFPLinearPass (:722-723).
- RemoveSafeSoftmaxGuardPass (:560): rc4 skips it unless the SDPA guard policy
  is REMOVE, which RETAB refuses, so stock lowers those graphs.
- DecomposeUnsupportedBilinearResizePass (:696): IrRewriteUpsample sends the
  exact 1/16 bilinear downscale it handles to stock.

**A gap (1).** DecomposeBoolEqualityPass (:672) has no RETAB step at v0.4.1,
so RETAB serializes a bool eq.Tensor as EQUAL, which TOSA accepts only for
numeric inputs. Generic INT and U85 specs delegate such an eq; U55 doesn't.
Branch fix/bool-equality (Linear APU-123) ports the pass at its rc4 slot.

**Not checked op by op (4):** PrepareGatherIndicesPass (:548), which removes
int32-to-int64 casts on gather indices; MatchArgDtypePass (:662);
UnsqueezeScalarPlaceholdersPass (:663); and
NormalizeTransformInputPlaceholdersPass (:756). The last one also keeps the
graph signature in step, which RETAB doesn't have, since it keys inputs by node
name; IrFuseIdenticalInputTransforms normalizes the placeholder targets itself.

## RETAB's input

Three stock stages shape the graph RETAB receives, and RETAB doesn't repeat
them:

1. rc4's pre-decomposition passes (`transform_for_pre_decomposition_pipeline`,
   APM:475), which exir runs before its edge decompositions:
   ConvertInt64OutputOpsToInt32Pass, plus DecomposeSDPAWithRegularSoftmaxPass
   when the SDPA guard policy is AUTO. RETAB refuses AUTO and leaves those
   graphs to stock. RETAB's edge-decomposition fast paths skip this stage,
   which is one reason they stay off.
2. ExecuTorch's edge decomposition table (`_default_decomposition_table` in
   `exir/tracer.py`), minus the ops rc4's partitioner keeps
   (`ops_to_not_decompose`, `backends/arm/tosa/partitioner.py:890`). Among
   others, it removes pixel_shuffle, pixel_unshuffle,
   repeat_interleave.self_int, select_scatter, tril, stack and unbind.
3. The quantizer's annotation pipeline (`transform_for_annotation_pipeline`,
   APM:793), which rewrites the model before quantization. On INT graphs it
   has already decomposed layer norm, var, mean, group norm, batch norm,
   softmax, min and max, select_scatter, tril, addmm and masked_fill, converted
   bool sums, and replaced ±inf literals.

The third stage is why IrDecomposeLayerNorm, both IrDecomposeVar steps,
IrDecomposeMean and IrDecomposeOps fired on no sampled graph, not even in
test_layer_norm, test_var, test_mean_dim, test_group_norm and test_batch_norm.
rc4's backend copies of those passes have nothing to do there either. RETAB's
backend pipeline repeats one quantizer-side step, IrReplaceInfAndLimitValues.
It also repeated IrDecomposeSelectScatter and IrDecomposeTril until they were
removed ([Removed steps](#removed-steps)).

## How the firing data was taken

A step fired when it reported a change (`RETAB_LOG` prints `modified=true`).
The counts come from a v0.4.1 release build on 2026-11-07, over 325 pipeline
runs:

- The private corpus: 26 models lowered for Ethos-U55, giving 28 parent
  pipelines. 27 delegates are numerically exact against stock, and stock fails
  to lower the 28th.
- 251 `tosa_INT` tests from 20 rc4 op-test files, chosen to reach the
  RETAB-only steps: test_add, test_batch_norm, test_cond, test_full,
  test_group_norm, test_isinf, test_layer_norm, test_masked_fill,
  test_mean_dim, test_pixel_shuffling (which has no `tosa_INT` case),
  test_repeat, test_select_scatter, test_stack, test_sub, test_sum,
  test_to_copy, test_tril, test_unbind, test_var and test_while. They gave 243
  parent, 40 cond and 14 while pipelines, with no stock fallback.

This is a sample, not the op suite. For a step that matches an rc4 step, a zero
only means these graphs didn't need it. The RETAB-only verdicts rest on the
reasons given for each step, not on the zeros alone.

To repeat a measurement, set `RETAB_LOG=1` and run pytest without xdist
workers (`-p no:xdist -s`), whose stderr is otherwise lost. Each step prints
`[IrPipeline] <step> — modified=<bool>`.
