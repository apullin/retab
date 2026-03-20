# Ticket 1D: Core Passes -- Writeup

## Overview

Six pass files implement a TOSA-lowering pipeline that transforms an ingested ATen-level quantized graph into TOSA IR suitable for Ethos-U55 compilation. All passes implement the `Pass` trait (`fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool`) and are orchestrated by `PassRunner`.

## Pass Descriptions

### 1. `fold_qdq.rs` -- FoldAndAnnotateQParamsPass

**Purpose:** Remove `QuantizePerTensor`, `DequantizePerTensor`, and `DequantizePerChannel` nodes by folding their quantization parameters into the metadata of adjacent compute nodes.

**How it works:**
- **Pass 1 (DQ annotation):** Iterates over all DQ nodes. For each, extracts `QuantArgs` (scale, zero_point, dtype, qmin, qmax) from the DQ node's literal args, then annotates every *user* of the DQ node by appending to `user.meta.input_qparams` at the corresponding arg position.
- **Pass 2 (Q annotation):** Iterates over all Q nodes. Extracts `QuantArgs` and annotates the Q node's *input producer* by appending to `producer.meta.output_qparams` at position 0.
- **Pass 3 (Bypass):** Calls `graph.bypass_node()` on every DQ and Q node, rewiring their first input directly to all their users and then erasing them.

### 2. `insert_rescale.rs` -- InsertRescalePass

**Purpose:** Insert `TosaRescale` nodes wherever a producer's output quantization parameters differ from a consumer's input quantization parameters (different scale or zero_point).

**How it works:**
- For each node with `output_qparams`, examines every user.
- If the user has an `input_qparams` entry for the arg position that references the producer, and the scale/zero_point differ, a rescale is needed.
- Computes a fixed-point `(multiplier, shift)` pair from the scale ratio using an `frexp`-based decomposition (matching the Python reference `_compute_multiplier_and_shift`).
- Creates `TosaConst` nodes for multiplier and shift, stores their data in the `TensorStore`, and inserts a `TosaRescale` node between producer and consumer.
- Rewires the consumer's arg to point at the rescale node.

### 3. `decompose_quant.rs` -- DecomposeQuantNodesPass

**Purpose:** Catch-all cleanup for any `QuantizePerTensor` / `DequantizePerTensor` / `DequantizePerChannel` nodes that survived the fold pass (e.g., at graph boundaries).

**How it works:** Bypasses each remaining Q/DQ node by wiring its input directly to its users.

### 4. `rewrite_conv.rs` -- RewriteConvPass

**Purpose:** Rewrite ATen `Conv2d`, `Linear`, and `Addmm` nodes into TOSA convolution operators.

**How it works:**
- **Conv2d:** Reads stride, padding, dilation, groups from args. If `groups == in_channels`, emits `TosaDepthwiseConv2d`; otherwise `TosaConv2d`. Creates `TosaConst` nodes for `input_zp` and `weight_zp` (from annotated `input_qparams`). Converts ATen-style padding `[pad_h, pad_w]` to TOSA-style `[top, bottom, left, right]`. Mutates the node in-place (target, args, kwargs) and re-propagates shapes.
- **Linear / Addmm:** Rewrites as `TosaConv2d` with 1x1 convolution semantics (stride=1, padding=0, dilation=1). Handles the different arg orderings of `Addmm` (bias, input, weight) vs `Linear` (input, weight, bias).

### 5. `memory_format.rs` -- ToTosaMemoryFormatPass

**Purpose:** Insert `TosaTranspose` nodes to convert between NCHW (ATen convention) and NHWC (TOSA convention) around NHWC-native ops (conv2d, depthwise conv2d, transpose conv2d, avg/max pool2d).

**How it works:**
- For each NHWC op, checks whether its input already comes from a transpose. If not, inserts an NCHW-to-NHWC transpose (perm `[0,2,3,1]`) before the op and rewires the input.
- For each user of an NHWC op that is *not* itself an NHWC op, inserts an NHWC-to-NCHW transpose (perm `[0,3,1,2]`) after the op and rewires.
- Sets `tosa_dim_order` metadata on affected nodes.

### 6. `cleanup.rs` -- CleanupPass

**Purpose:** Final cleanup: rewrite remaining ATen ops to their TOSA equivalents and perform dead code elimination.

**How it works:**
- **Op rewriting:** `MaxPool2d` -> `TosaMaxPool2d`, `AvgPool2d` -> `TosaAvgPool2d`, `Reshape`/`View`/`Flatten` -> `TosaReshape`.
- **DCE:** Iteratively removes nodes with no users that are not the graph output, until a fixed point is reached.

## Pass Pipeline Ordering

Defined in `build_phase1_pipeline()`:

```
1. FoldAndAnnotateQParamsPass   -- fold Q/DQ, annotate qparams on compute nodes
2. InsertRescalePass            -- insert TosaRescale where qparams mismatch
3. DecomposeQuantNodesPass      -- bypass any surviving Q/DQ nodes
4. RewriteConvPass              -- Conv2d/Linear/Addmm -> TosaConv2d/TosaDepthwiseConv2d
5. ToTosaMemoryFormatPass       -- insert NCHW<->NHWC transposes
6. CleanupPass                  -- rewrite remaining ATen ops + dead code elimination
```

The ordering is intentional:
- Q/DQ folding must happen first so that qparams are available for rescale insertion and conv rewriting.
- Rescale insertion must precede conv rewriting because conv rewriting reads `input_qparams` to create zero-point constants.
- Decompose runs after rescale/conv to clean up any Q/DQ nodes those passes did not consume.
- Memory format conversion runs after conv rewriting so it can find all TOSA conv/pool nodes.
- Cleanup is last to catch any dead nodes left by earlier passes.

## cargo check Results

```
cargo check: SUCCESS (0 errors)
```

90 warnings, all `dead_code` or `unused_*` -- expected since the passes module is not yet called from the PyO3 entry point. No compilation errors.

## Known Limitations

1. **Per-channel quantization is approximate.** `DequantizePerChannel` extracts only the first element of the scale/zero_point lists as a representative scalar. True per-channel rescale would need per-channel multiplier/shift arrays.

2. **No transposed convolution rewriting.** `ConvTranspose2d` is not handled by `RewriteConvPass`; it would need a `TosaTransposeConv2d` lowering path.

3. **Linear/Addmm as 1x1 conv assumes 4D input.** The rewrite does not reshape 2D `(batch, features)` inputs to `(batch, features, 1, 1)` before the 1x1 conv, which would be needed for correctness.

4. **Memory format pass does not handle weight transposes.** TOSA expects OHWI weights for conv2d, but the pass only transposes activations. Weight layout conversion would need to permute the weight tensor data in the `TensorStore`.

5. **Fixed-point scale conversion is basic.** The `frexp`-based multiplier/shift calculation handles common cases but does not clamp multiplier to `[-(1<<31), (1<<31)-1]` or handle subnormal/overflow edge cases.

6. **No element-wise op lowering.** Ops like `Add`, `Mul`, `Relu`, `Clamp` are not rewritten to their TOSA equivalents (`TosaAdd`, `TosaMul`, `TosaClamp`) by any pass. The cleanup pass explicitly skips `Relu`.

7. **Cleanup DCE relies on the `users` list being accurate.** If earlier passes have bugs in user-list bookkeeping, DCE could incorrectly remove live nodes or fail to remove dead ones.

8. **Single-output assumption.** `InsertRescalePass` only checks `output_qparams` at position 0. Multi-output nodes (if any existed) would not be handled.

9. **No pass iteration / fixed-point.** Each pass runs exactly once. Patterns that require multiple iterations (e.g., cascaded rescale simplification) are not handled.
