# P0 Fixes Writeup

## Summary

Three critical issues were identified in the arm-tosa-engine that caused it to produce structurally valid but semantically incorrect TOSA flatbuffers. All three have been addressed.

---

## P0-1: Serializer doesn't embed tensor data

### Diagnosis

The serializer (`src/serialize/mod.rs`) already contained the correct logic to embed tensor data for Placeholder nodes (lines 41-57). It checks `store.get(name)`, and when data is found, creates a `TosaTensor` with the raw bytes and also emits a `CONST` operator.

The actual issue was that the store aliasing mechanism in `src/ingest/fx_bridge.rs` (lines 151-157) was the critical link: placeholder nodes have names like `b__frozen_param0`, while the state_dict uses FQN keys like `conv1.weight`. The `graph_signature.inputs_to_parameters` mapping connects these, and the alias is created at ingestion time.

### Fix

No code change was needed for P0-1 itself. The serializer already embedded data correctly. The real fix was ensuring passes (P0-3 fixes) preserve the correct node names through transformations so `store.get(name)` continues to work.

---

## P0-2: Serializer doesn't emit CONST ops for constant tensors

### Diagnosis

Like P0-1, the serializer already handled CONST operator emission (lines 51-57 of `serialize/mod.rs`). When a Placeholder node has associated data in the TensorStore, both the tensor entry (with data bytes) AND a CONST operator are created.

### Fix

No separate fix was needed. The existing serializer logic was correct.

---

## P0-3: Passes aren't transforming the graph

### Diagnosis

This was the primary issue. Multiple sub-problems were found:

1. **`ConvertToClampPass` was missing from `build_phase1_pipeline()`**. Relu nodes were never converted to TosaClamp. The cleanup pass had a comment saying "Relu is absorbed into clamp or removed" but didn't actually convert it.

2. **`ConvertToClampPass` produced `OpTarget::Clamp` (ATen) instead of `OpTarget::TosaClamp`**. The serializer's `emit_node` dispatch only recognizes TOSA-prefixed targets.

3. **`CleanupPass` didn't convert several ATen ops to TOSA equivalents**. Only MaxPool2d, AvgPool2d, and Reshape/View/Flatten were handled. Missing: Add, Sub, Mul, Permute, Relu, Clamp.

4. **`CleanupPass` didn't restructure pool args into kwargs**. When converting `MaxPool2d` -> `TosaMaxPool2d`, the original ATen positional args (`[input, kernel, stride, padding, ...]`) were preserved. But the TOSA emission visitors read pool parameters from kwargs (`kernel`, `stride`, `pad`), so they fell back to defaults.

5. **Node dtypes weren't updated after Q/DQ folding**. The `FoldAndAnnotateQParamsPass` bypassed Q/DQ nodes and annotated qparams, but the intermediate nodes (conv, relu, etc.) retained their original Float32 dtype from FX graph ingestion. This caused the serializer to emit FP32 tensors instead of INT8.

6. **Conv2d missing zero bias for quantized path**. When `aten.conv2d` had no bias (or `Arg::None` bias), the `RewriteConvPass` produced only 4 TOSA conv inputs instead of the required 5. TOSA CONV2D needs `[input, weight, bias, input_zp, weight_zp]`.

7. **TOSA RESHAPE missing CONST_SHAPE input**. The `emit_reshape` visitor only passed the input tensor, but TOSA RESHAPE requires `[input, shape]` where shape is a CONST_SHAPE tensor.

8. **TOSA RESCALE missing zero-point inputs**. The `emit_rescale` visitor only passed `[input, multiplier, shift]` but TOSA RESCALE requires `[input, multiplier, shift, input_zp, output_zp]`.

### Fixes Applied

**`src/passes/mod.rs`**:
- Added `ConvertToClampPass` to `build_phase1_pipeline()`, positioned after `FoldAndAnnotateQParamsPass` so dtype info is available.
- Added gated debug logging (`TOSA_ENGINE_DEBUG` env var) to PassRunner for diagnostics.

**`src/passes/convert_clamp.rs`**:
- Changed Relu conversion from `OpTarget::Clamp` to `OpTarget::TosaClamp`.
- Changed Hardtanh conversion from `OpTarget::Clamp` to `OpTarget::TosaClamp`.
- Added handling for `OpTarget::Clamp` -> `OpTarget::TosaClamp`.
- Uses kwargs `min_val`/`max_val` matching what the `emit_clamp` visitor expects.
- For quantized Relu, uses fixed bounds [0, 127] (int8 range).

**`src/passes/fold_qdq.rs`**:
- Added Pass 4: after bypassing Q/DQ nodes, iterate all CallFunction nodes and update `meta.val.dtype` based on annotated qparams. Nodes with `input_qparams` get the quantized dtype (typically Int8).

**`src/passes/cleanup.rs`**:
- Added ATen -> TOSA target conversions for: `Add` -> `TosaAdd`, `Sub` -> `TosaSub`, `Mul` -> `TosaMul`, `Permute` -> `TosaTranspose`, `Relu` -> `TosaClamp`, `Clamp` -> `TosaClamp`.
- MaxPool2d conversion now restructures positional args into kwargs: extracts kernel/stride/padding from positional args, converts padding from ATen format `[pH, pW]` to TOSA format `[top, bottom, left, right]`, and stores as kwargs.
- AvgPool2d conversion similarly restructures args into kwargs.
- Relu conversion in cleanup sets proper kwargs with `min_val=0`, `max_val=127`.

**`src/passes/rewrite_conv.rs`**:
- When the bias arg is `Arg::None` (no bias), creates a zero bias const tensor (int32 zeros with shape [out_channels]) and inserts it as a Placeholder node. This ensures TOSA CONV2D always has 5 inputs.
- Same fix applied to the `rewrite_linear` function.

**`src/serialize/visitors.rs`**:
- `emit_reshape`: Now creates a CONST_SHAPE tensor from the output shape metadata and a CONST_SHAPE operator. The RESHAPE operator inputs are `[input, shape]` matching the TOSA spec.
- `emit_rescale`: Now creates inline const tensors for `input_zp` and `output_zp` from kwargs. The RESCALE operator inputs are `[input, multiplier, shift, input_zp, output_zp]` matching the TOSA spec.

**`src/serialize/mod.rs`**:
- Added gated debug logging for serialization tracing.

---

## Verification

To verify fixes, rebuild and run comparison:

```bash
# Build
bash build.sh 2>&1 | tail -3

# Generate output with debug tracing
TOSA_ENGINE_DEBUG=1 bash -c 'unset CONDA_PREFIX && export VIRTUAL_ENV=/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv && export PATH="/Users/andrewpullin/.cargo/bin:$VIRTUAL_ENV/bin:/Users/andrewpullin/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" && cd /Users/andrewpullin/personal/executorch-work/u55-pipeline && python arm-tosa-engine/audit/generate_rust_output.py' 2>&1

# Compare
bash -c 'unset CONDA_PREFIX && export VIRTUAL_ENV=/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv && export PATH="/Users/andrewpullin/.cargo/bin:$VIRTUAL_ENV/bin:/Users/andrewpullin/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" && cd /Users/andrewpullin/personal/executorch-work/u55-pipeline && python arm-tosa-engine/audit/compare_tosa.py' 2>&1
```

### Expected improvements

| Metric | Before | After (expected) |
|--------|--------|-------------------|
| Op types | ATen names (relu, conv2d, flatten) | TOSA names (CLAMP, CONV2D, RESHAPE) |
| Tensor dtypes | All FP32 | INT8 for quantized tensors, INT32 for accumulators |
| Tensor data | 0 bytes embedded | 21KB+ (weights, biases, zp constants) |
| CONST ops | 0 | One per constant tensor |
| Total ops | Low (missing CONST, RESCALE, TRANSPOSE) | Closer to reference (38 ops) |

---

## Remaining Issues

1. **Flatten shape args**: When `aten.flatten` is converted to `TosaReshape`, the original flatten args `[input, start_dim, end_dim]` are not restructured into reshape args. The shape is taken from the output metadata, which is correct, but the extra int args remain (harmlessly, since `collect_input_names` ignores non-Node args).

2. **Per-channel quantization in rescale**: The `InsertRescalePass` uses a single scale/zp from the first per-channel entry. Full per-channel support would need per-channel multipliers.

3. **Weight transpose for NHWC**: When the memory format pass inserts NCHW->NHWC transposes, it only transposes the activation input. Weight tensors for conv may also need reordering depending on the TOSA backend expectations.

4. **Phase 2 pipeline**: The `ConvertToClampPass` in `build_phase2_pipeline()` runs before `FoldAndAnnotateQParamsPass`. This ordering means the dtype check for int8 clamp bounds won't work in phase2. This is acceptable since phase1 is the active pipeline, but should be fixed for phase2 usage.

---

## Files Modified

- `src/passes/mod.rs` - Added ConvertToClampPass to phase1 pipeline, debug logging
- `src/passes/convert_clamp.rs` - Relu/Hardtanh/Clamp -> TosaClamp with correct kwargs
- `src/passes/fold_qdq.rs` - Added Pass 4: dtype update after Q/DQ bypass
- `src/passes/cleanup.rs` - Added ATen->TOSA conversions for Add/Sub/Mul/Permute/Relu/Clamp, restructured pool args
- `src/passes/rewrite_conv.rs` - Create zero bias for bias-less conv, same for linear
- `src/serialize/visitors.rs` - Fixed emit_reshape (CONST_SHAPE), emit_rescale (zp inputs)
- `src/serialize/mod.rs` - Debug logging
