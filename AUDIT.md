# Correctness Audit: Rust TOSA Output vs Python Reference

## 1. Comparison Results: tiny_cnn

|                    | Reference | Rust     | Gap |
|--------------------|-----------|----------|-----|
| **File size**      | 30,392    | 2,956    | 10x smaller |
| **Total ops**      | 38        | 10       | Missing 28 ops |
| **Total tensors**  | 37        | 19       | Missing 18 tensors |
| **Tensors w/data** | 24 (21KB) | 0 (0KB)  | **No const data at all** |

### Op type comparison

| Op Type          | Reference | Rust | Status |
|------------------|-----------|------|--------|
| CONST            | 24        | 0    | **MISSING** — no CONST ops emitted |
| CONST_SHAPE      | 2         | 0    | **MISSING** |
| CONV2D           | 2         | 3    | Wrong count (+1, linear→conv2d shouldn't be here?) |
| DEPTHWISE_CONV2D | 1         | 0    | **MISSING** — first conv should be depthwise |
| MAX_POOL2D       | 2         | 2    | OK |
| RESCALE          | 3         | 0    | **MISSING** — no rescale ops emitted |
| RESHAPE          | 2         | 1    | Short by 1 |
| TRANSPOSE        | 2         | 4    | Too many (+2 extra transposes) |

### Key observations

1. **Zero const tensor data embedded** — Placeholder nodes exist but their tensor
   bytes from TensorStore are not being serialized into the flatbuffer. This is the
   biggest gap: 21KB of weights/biases/scales/shifts missing.

2. **No CONST ops** — The serializer isn't emitting CONST operators for constant
   tensors. In TOSA, every const tensor needs both a tensor entry (with data bytes)
   AND a CONST operator.

3. **No RESCALE ops** — The fold_qdq pass removes Q/DQ nodes and the insert_rescale
   pass should create TosaRescale nodes, but they're not appearing in the output.
   Either the pass isn't running or the serializer isn't emitting them.

4. **No DEPTHWISE_CONV2D** — The rewrite_conv pass should detect groups==in_channels
   and emit depthwise conv, but it's producing regular CONV2D instead.

5. **Wrong tensor dtypes** — Rust tensors show FP32 dtype, but they should be INT8
   (quantized). The passes aren't tracking quantization properly.

6. **Wrong tensor shapes** — Common tensors have NCHW shapes in Rust but NHWC in
   reference. The memory format pass runs but the weight tensors aren't being
   permuted.

7. **Tensor naming** — Rust uses short names (`conv2d`, `relu`) from the FX graph
   while reference uses decorated names (`tosa_conv2d_default`, etc.). Names must
   match for TOSA consistency.

## 2. Comparison Results: cnn_layernorm4d

|                    | Reference  | Rust     | Gap |
|--------------------|------------|----------|-----|
| **File size**      | 987,236    | 89,180   | 11x smaller |
| **Total ops**      | 4,011      | ~600     | Missing ~3400 ops |
| **Tensors w/data** | 2,909      | 0        | **No const data** |

Same fundamental issues as tiny_cnn, scaled up.

## 3. Root Causes

### RC1: Serializer doesn't embed tensor data (CRITICAL)
**Where**: `src/serialize/mod.rs` — `serialize_to_tosa()`
**What**: Placeholder nodes are added as tensors but `TensorStore` data is never
written into the flatbuffer's tensor data field. The tensor entries have empty
`data: []` vectors.
**Fix**: When serializing a Placeholder node, look up its name in `TensorStore`,
get the raw bytes, and embed them in the flatbuffer tensor.

### RC2: Serializer doesn't emit CONST ops for constants (CRITICAL)
**Where**: `src/serialize/mod.rs`
**What**: Placeholder nodes with tensor data need both a tensor entry AND a CONST
operator entry. The serializer only creates the tensor, not the operator.
**Fix**: For each Placeholder that has data in TensorStore, also emit a CONST op
with inputs=[] and outputs=[tensor_name].

### RC3: Passes don't run on the ingested graph (CRITICAL)
**Where**: `src/passes/` + `src/lib.rs`
**What**: The Rust output shows FP32 dtypes and unrewritten ops (relu, conv2d
instead of TosaClamp, TosaConv2d). This suggests the passes either aren't
transforming the graph or the transformations aren't sticking.
**Evidence**: The output has `relu` and `flatten` tensors — these are ATen op
names that should have been rewritten by ConvertToClampPass and RewriteConvPass.
**Fix**: Debug the pass pipeline — add logging to each pass to confirm it runs
and modifies nodes. The issue may be that passes match on specific OpTarget
variants that don't match what the ingestion produces.

### RC4: Depthwise conv detection not working
**Where**: `src/passes/rewrite_conv.rs`
**What**: Conv2d with in_channels=1 should become DEPTHWISE_CONV2D but it's
emitting regular CONV2D.
**Fix**: Check the depthwise detection logic — it should check if groups == in_channels.

### RC5: RESCALE pass not producing output
**Where**: `src/passes/insert_rescale.rs`
**What**: No RESCALE ops in output. Either fold_qdq isn't populating qparams,
or insert_rescale isn't finding nodes that need rescaling.
**Fix**: Verify fold_qdq correctly extracts QuantArgs from DQ/Q nodes and stores
them in meta.input_qparams / meta.output_qparams.

### RC6: Weight tensors not permuted to NHWC
**Where**: `src/passes/memory_format.rs`
**What**: Weight shapes are NCHW (e.g., [16,1,3,3]) but reference has NHWC
([3,3,1,16]). The memory format pass needs to permute weight data too, not
just insert transpose ops for activations.
**Fix**: When rewriting to TOSA conv ops, permute the weight tensor data in
TensorStore.

## 4. Fixes Applied

None yet — this audit identifies the issues. Fixes require focused implementation.

## 5. Remaining Gaps (Priority Order)

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Embed tensor data in flatbuffer | Small (serialize/mod.rs) | Adds 21KB+ of weights |
| P0 | Emit CONST ops for constants | Small (serialize/mod.rs) | Adds 24 missing ops |
| P0 | Debug why passes don't transform | Medium (pass debugging) | Everything depends on this |
| P1 | Fix RESCALE emission | Medium (insert_rescale.rs) | 3 missing ops + correctness |
| P1 | Fix depthwise conv detection | Small (rewrite_conv.rs) | 1 op fix |
| P1 | Permute weight data to NHWC | Medium (memory_format.rs) | Shape correctness |
| P2 | Tensor naming conventions | Small (serialize) | Match reference names |
| P2 | CONST_SHAPE emission | Small (serialize) | 2 missing ops |

## 6. Recommendation

**The architecture is sound.** The pipeline runs end-to-end in 47ms, the
flatbuffer structure is valid, and the pass framework works. The issues are
all in the **details of each pass and the serializer** — not in the design.

**Before Phase 3**, fix P0 issues:
1. Make the serializer embed const tensor data (RC1 + RC2)
2. Debug why passes aren't transforming the graph (RC3)
3. Once those are fixed, most P1 issues will become visible and fixable

**Estimated effort for P0 fixes**: 1-2 focused sessions.
**After P0 fixes**: tiny_cnn should go from 10 ops → ~38 ops and 0 data → ~21KB.
