# Ticket 1B Review: FX Graph Ingestion (PyO3 Bridge)

## Task A: Code Review

### cargo check
**PASS** -- compiles cleanly. 27 warnings, all `dead_code` for types/methods reserved for later pipeline stages (Tickets 2+). No errors.

### FX Node Walking Logic
**Correct.** `ingest_fx_graph()` in `fx_bridge.rs`:
- Iterates `graph_module.graph.nodes` via Python iterator protocol.
- Extracts `op`, `name`, `target`, `args`, `kwargs`, `meta` per node.
- Builds `name_map: HashMap<String, NodeId>` as it goes, so arg resolution works for the topological order FX guarantees.
- Correctly tracks `graph.inputs` (placeholders) and `graph.output`.
- `get_attr` and `call_method` ops silently fall through to `OpType::CallFunction` (line 45). This is fine for now since these rarely appear in quantized exported graphs, but should be revisited if custom modules or buffer accesses are needed.

### Target Mapping -- 15-op coverage
**All 15 required ops are covered:**

| Op | Mapped? | target_map.rs line |
|----|---------|-------------------|
| conv2d | Yes | L25 |
| relu | Yes | L28 |
| add | Yes | L31 |
| sub | Yes | L34 |
| mul | Yes | L37 |
| permute | Yes | L43 |
| reshape | Yes | L46 |
| full | Yes | L49 |
| avg_pool2d | Yes | L52 |
| max_pool2d | Yes | L55 |
| sum | Yes | L58 |
| rsqrt | Yes | L61 |
| quantize_per_tensor | Yes | L10 |
| dequantize_per_tensor | Yes | L13 |
| dequantize_per_channel | Yes | L16 |

**Issue (minor):** Line 22 has an operator-precedence bug:
```rust
if target_str.contains("aten.conv_transpose2d") || target_str.contains("aten.convolution") && target_str.contains("transpose") {
```
Due to `&&` binding tighter than `||`, this evaluates as:
```
A || (B && C)
```
which is actually the intended logic, so it works correctly in practice. However, adding explicit parentheses would make intent clear and prevent future maintenance errors.

### Arg Extraction
**Correct and thorough.** `extract_single_arg()` handles:
- `None` -- checked first
- FX Node refs -- duck-typed via `.name` + `.op` attributes, resolved through `name_map`
- `bool` -- checked before `int` (correct, since Python `bool` subclasses `int`)
- `i64` (int)
- `f64` (float)
- `String`
- Tuple/List -- dispatches to `NodeList`, `IntList`, or `FloatList` based on first element
- `torch.dtype` -- detected via type name, parsed through `parse_torch_dtype`
- Fallback -- stringified via `str()`

**Missing:** `OptionalNode` and `OptionalInt` variants exist in `ir.rs` (lines 176-177) but are never produced by the extractor. Currently, optional-node args (e.g., `bias=None` in conv2d) will produce `Arg::None`, which works but loses type information. Not a blocker, but the lowering pass should be aware.

### State Dict Ingestion
**Correct.** Uses `.contiguous().cpu().numpy().tobytes()` pipeline (lines 105-108), which is the standard numpy buffer protocol path. The resulting `Vec<u8>` is stored alongside parsed shape and dtype. This is clean and efficient.

**Note:** `parse_torch_dtype` (line 325) uses `contains()` substring matching, which means `"int8"` would also match `"uint8"` if that ever appears. Currently safe since symmetric quantization uses `torch.int8`, but worth noting.

### TensorInfo Fixed-Size Shape
The `TensorInfo` struct uses `shape: [i64; 4]` with `shape_len: u8`. This caps shape rank at 4 dimensions. This is fine for the CNN/vision use case (NCHW/NHWC) but will need adjustment for 5-D or higher-rank tensors (e.g., conv3d is already in the OpTarget enum).

## Task B: Reference TOSA Flatbuffer Generation

**BLOCKED:** The sandbox environment denied permission to run Python scripts in the u55-pipeline directory. The script at `/tmp/dump_tosa_ref2.py` exists and looks correct in structure -- it exports tiny_cnn, quantizes it, then uses `to_edge_transform_and_lower` with `EthosUPartitioner` and `dump_intermediate_artifacts_to()` to capture intermediate TOSA files.

The `test_data/` directory is currently empty (no `.tosa` files exist yet).

**To unblock**, run manually outside the sandbox:
```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline
VIRTUAL_ENV=.venv PATH="$VIRTUAL_ENV/bin:$PATH" python /tmp/dump_tosa_ref2.py
```

If that fails (e.g., Vela not installed or compile spec issues), the fallback approach would be to call `TOSABackend._preprocess()` directly, bypassing the Vela compilation step. A script for that approach would need to:
1. Export + quantize the model (same as `dump_tosa_ref2.py` stages 1-2)
2. Partition via `to_edge_transform_and_lower` but intercept the TOSA bytes before they reach Vela
3. Save raw TOSA flatbuffer bytes to `test_data/tiny_cnn_reference.tosa`

## Summary

| Check | Status |
|-------|--------|
| cargo check | PASS (27 dead-code warnings, expected) |
| FX node walking | PASS |
| 15-op target coverage | PASS (all 15 present) |
| Arg extraction types | PASS (all required types handled) |
| state_dict numpy buffer | PASS |
| Reference TOSA generation | BLOCKED (sandbox; run manually) |

### Recommended Follow-ups
1. Add explicit parentheses in `target_map.rs` line 22 for `conv_transpose2d` check.
2. Consider producing `Arg::OptionalNode(None)` instead of `Arg::None` for typed optional args, if downstream lowering needs to distinguish them.
3. Extend `TensorInfo` shape capacity beyond rank-4 before adding conv3d/transformer support.
4. Add a `uint8` case to `parse_torch_dtype` for future asymmetric quantization support.
