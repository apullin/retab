# P3: Close the RESCALE Gap (30 -> 648) -- Fixes Writeup

## Diagnostic Findings

### Root Cause of Missing Rescales

The gap between 30 RESCALE ops (ours) and 648 (reference) stems from two distinct issues:

1. **Accumulator rescales (30)**: Already handled correctly by `InsertRescalePass` for
   conv-like ops (Conv2d, DepthwiseConv2d, etc.). These account for our existing 30 rescales.

2. **Missing inter-op arithmetic rescales (~618)**: In the TOSA spec, integer arithmetic
   ops (ADD, SUB, MUL) require explicit RESCALE nodes:
   - **ADD/SUB**: Both inputs must be rescaled to a common scale (`2 * max(s_a, s_b)`)
     before the operation, producing 2 input rescales. The output then needs rescaling
     from the common scale to the consumer's expected scale (1 output rescale via
     mismatch detection). **Total: 3 rescales per ADD/SUB**.
   - **MUL**: The natural output scale is `s_a * s_b`, which differs from the
     propagated output scale. An output rescale bridges this gap.
     **Total: 1 rescale per MUL**.

### Why Mismatch Detection Alone Was Insufficient

`PropagateQParamsPass` copies `input_qparams[0]` to `output_qparams` (forward) and
copies producer `output_qparams` to consumer `input_qparams` (backward). This creates
a uniform scale through decomposed chains, eliminating all mismatches except at Q/DQ
boundaries. The result: `InsertRescalePass` only found mismatches at chain endpoints,
not within chains.

### Root Cause of Doubled Ops (54 vs 27)

LayerNorm decomposition creates `Mean(x, dims)` and `Sub(x, mean)`.
Var decomposition (which decomposes the Var node created by LayerNorm) also creates
`Mean(x, dims)` and `Sub(x, mean)` with identical args. `FuseDuplicateUsersPass`
should merge these, as they have the same `(target, args, kwargs)` signature.

The `is_fusable` list was already adequate for the core ops but was missing `View`,
`FullLike`, `Permute`, `TosaTranspose`, and `TosaTable`, which could prevent cascading
fusions in some cases.

## Fixes Applied

### 1. New Pass: `InsertArithmeticRescalesPass` (`src/passes/insert_arithmetic_rescales.rs`)

A dedicated pass that inserts TOSA-required rescales around quantized arithmetic ops:

- **ADD/SUB**: Inserts 2 input rescales (to `2 * max(s_a, s_b)`) and updates the
  op's `output_qparams` to the common scale. `InsertRescalePass` mismatch detection
  then adds the output rescale.
- **MUL**: Computes `natural_scale = s_a * s_b`. If this differs from the propagated
  `output_qparams`, inserts an output rescale from natural to desired scale.

Guards prevent double-processing when the pass runs twice (checks if inputs are
already TosaRescale nodes, or if users already include TosaRescale).

### 2. Pipeline Changes (`src/passes/mod.rs`)

- Added `InsertArithmeticRescalesPass` at step 6a (after PropagateQParams, before
  first InsertRescalePass)
- Added second `PropagateQParamsPass` at step 15a (re-propagates qparams through
  nodes inserted by BroadcastArgs, MatchArgDtype, MemoryFormat)
- Added second `InsertArithmeticRescalesPass` at step 15b (for newly created nodes)
- Added `DiagnosticPass` (gated behind `TOSA_ENGINE_DEBUG` env var) at two points
  for debugging qparam coverage

### 3. Expanded `is_fusable` in `FuseDuplicateUsersPass`

Added: `View`, `FullLike`, `Permute`, `TosaTranspose`, `TosaTable` to the fusable
ops list, enabling broader deduplication of equivalent subgraphs.

### 4. Diagnostic Pass (gated behind `TOSA_ENGINE_DEBUG`)

A `DiagnosticPass` struct with a label field reports:
- Total CallFunction nodes
- Nodes with input_qparams, output_qparams, both, or neither
- Per-node qparam details (scales at each position)

Set `TOSA_ENGINE_DEBUG=1` when running to see diagnostic output.

## Files Modified

| File | Change |
|------|--------|
| `src/passes/insert_arithmetic_rescales.rs` | **New file** -- TOSA arithmetic rescale insertion |
| `src/passes/mod.rs` | Pipeline ordering, DiagnosticPass, expanded is_fusable |
| `src/passes/propagate_qparams.rs` | Updated doc comments |

## Expected Behavior After Fix

With these changes:
- Each quantized ADD/SUB gets ~3 RESCALE ops (2 input + 1 output mismatch)
- Each quantized MUL with two quantized inputs gets ~1 RESCALE op (output)
- Accumulator ops (Conv2d, etc.) continue to get their accumulator rescales
- The total RESCALE count should approach the reference's 648

## How to Verify

```bash
# Build
bash build.sh

# Run with diagnostics
TOSA_ENGINE_DEBUG=1 bash -c 'unset CONDA_PREFIX && export VIRTUAL_ENV=... && python -c "
from model import get_model
import torch
# ... (standard quantized export flow for cnn_layernorm4d)
r = arm_tosa_engine.lower_to_tosa(...)
"' 2>&1 | grep -E "Diagnostic|InsertArithRescale|InsertRescale"

# Count RESCALE ops in output
# (use the TOSA flatbuffer inspection tool or grep debug output)
```
