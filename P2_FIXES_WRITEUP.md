# P2 Fixes: Inter-Op RESCALE, TRANSPOSE, SUB/RESHAPE

## Problem Summary

The cnn_layernorm4d model produces significantly different op counts from the Python reference pipeline:

| Op | Reference | Rust (before) | Issue |
|----|-----------|---------------|-------|
| RESCALE | 648 | 30 | 618 missing inter-op rescales |
| TRANSPOSE | 56 | 114 | 58 extra (2x too many) |
| SUB | 27 | 54 | 27 extra (2x) |
| RESHAPE | 27 | 54 | 27 extra (2x) |

## Root Causes and Fixes

### Fix 1: Missing Inter-Op RESCALE (618 missing)

**Root cause**: Two interacting issues:

1. **Missing qparams on decomposed ops.** The decomposition passes (LayerNorm, Var, MeanDim) run AFTER the model has been quantized. The quantizer wraps the original LayerNorm op with Q/DQ nodes, but the decomposed sub-ops (Sub, Mul, Mean, Sum, etc.) have no Q/DQ wrappers. After `FoldAndAnnotateQParamsPass`, only the boundary ops (those directly consuming a DQ output or feeding a Q node) receive `input_qparams`/`output_qparams`. The intermediate ops have no qparams, so `InsertRescalePass` skips them entirely.

2. **Rescale nodes missing output_qparams.** The accumulator rescale and mismatch rescale code creates RESCALE nodes but does not set `output_qparams` on them. On the second pass run (position 16 in phase2), these nodes are invisible to the mismatch detection.

**Fix**:
- **New `PropagateQParamsPass`** (`src/passes/propagate_qparams.rs`): Runs after `FoldAndAnnotateQParamsPass` and before `InsertRescalePass`. Iterates until convergence:
  - Forward: For CallFunction nodes with `input_qparams` but no `output_qparams`, sets `output_qparams[0] = input_qparams[0]` (element-wise/reduction ops preserve scale).
  - Backward: For CallFunction nodes missing `input_qparams` at a given arg position, copies the producer's `output_qparams[0]`.
- **Set `output_qparams` on created RESCALE nodes** in both `insert_accumulator_rescale` and the mismatch rescale path. This allows the second pass run to detect further mismatches downstream.
- **Unique rescale names** via atomic counter to avoid tensor store collisions when multiple rescales share the same producer.

### Fix 2: Extra TRANSPOSE (114 vs 56)

**Root cause**: The `ToTosaMemoryFormatPass` inserted one NHWC-to-NCHW transpose **per non-NHWC user** of each NHWC op. If a conv/pool op had N non-NHWC consumers, N separate transposes were created instead of one shared transpose.

**Fix**: Collect all non-NHWC users of each NHWC op, then create a single NHWC-to-NCHW transpose and rewire all those users to it. This halves the output transpose count.

### Fix 3: Extra SUB and RESHAPE (54 vs 27 each)

**Root cause**: The LayerNorm decomposition creates `Mean(x, dims)` and `Sub(x, mean)`. Then `Var(x, dims)` (also created by LayerNorm decomposition) is decomposed by `DecomposeVarPass`, which creates its own `Mean(x, dims)` and `Sub(x, mean)` with identical args. After `DecomposeMeanDimPass` decomposes all Mean nodes, the duplicate Sums, Fulls, and Muls also exist. The `FuseDuplicateUsersPass` was a no-op stub.

**Fix**: Implemented `FuseDuplicateUsersPass` to detect nodes with identical `(target, args, kwargs)` signatures and merge duplicates. Uses a loop-until-convergence approach so that after fusing Sum nodes, the downstream Mul nodes also become duplicates and get fused on the next iteration.

### Fix 4: Pipeline Selection

**Root cause**: `lower_to_tosa` always used `build_phase1_pipeline()` which lacks decomposition passes, qparams propagation, and the additional processing needed for complex models.

**Fix**: Auto-detect whether the graph contains LayerNorm, Var, or BatchNorm ops and select `build_phase2_pipeline()` accordingly. Simple models (tiny_cnn) continue to use phase1.

## Files Changed

- `src/passes/propagate_qparams.rs` — New pass for qparams propagation
- `src/passes/insert_rescale.rs` — Set output_qparams on rescale nodes; unique names via atomic counter
- `src/passes/memory_format.rs` — Share single NHWC-to-NCHW transpose among all non-NHWC users
- `src/passes/mod.rs` — Implemented FuseDuplicateUsersPass; added PropagateQParamsPass to phase2 pipeline
- `src/lib.rs` — Auto-detect phase1 vs phase2 pipeline based on graph contents
