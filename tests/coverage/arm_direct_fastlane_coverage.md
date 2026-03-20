# Arm Direct Fast-Lane Coverage Matrix

Snapshot: 2026-03-19

Scope: the Arm direct-lowering seam in `arm-tosa-engine`, compared against the ET Arm support surface and the current engine ingest / serializer surface.

Sources:
- ET support list: `/Users/andrewpullin/personal/executorch-work/executorch/src/executorch/backends/arm/operator_support/tosa_profile_supported_op_lists.py`
- ET Arm tests: `/Users/andrewpullin/personal/executorch-work/executorch/src/executorch/backends/arm/test`
- Engine ingest map: `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src/ingest/target_map.rs`
- Engine serializer dispatch: `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src/serialize/visitors.rs`
- Local direct-fast-lane corpus: `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/model_corpus.py`
- Direct sweep harness: `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/sweep_direct_fastlane.py`

Legend:
- `yes`: explicitly covered by the source or the current direct-fast-lane corpus
- `partial`: some members of the family are covered, but a gap remains
- `no`: not currently covered in the direct fast lane

Current local result: `13 / 13` models pass in the direct fast lane.

| Family | Representative ops | ET-arm-supported | Direct-seam-reachable | Engine-ingest | Engine-serialize | Local-test-covered | Notes / gaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Conv / depthwise / linear / matmul | `conv2d`, `depthwise_conv2d`, `linear`, `matmul`, `bmm`, `addmm` | yes | yes | yes | yes | partial | Convolution and linear are exercised locally; raw `matmul` / `bmm` / `addmm` still need dedicated direct-fast-lane models. |
| Pool / residual / reduce | `avg_pool2d`, `max_pool2d`, `_adaptive_avg_pool2d`, `sum`, `mean`, `amax` | yes | yes | yes | yes | yes | This is now well covered by `tiny_cnn`, `residual_avgpool_cnn`, and `depthwise_separable_cnn`. |
| Layout / reshape / transpose | `permute_copy`, `view_copy`, `reshape`, `squeeze`, `unsqueeze`, dim-order copies | yes | yes | yes | yes | yes | NHWC layout handling is now validated, including the direct fast-lane reshape/broadcast fixes. |
| Norm / broadcast / table | `layer_norm`, `group_norm`, `native_layer_norm`, `native_group_norm`, `rsqrt`, `reciprocal`, `sqrt`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `TosaTable` | yes | yes | yes | yes | yes | This was the main historical failure class; it is now covered by the norm and reshape-norm corpus. |
| Recurrent / tuple plumbing | `unbind_copy`, `getitem`, `sigmoid`, `tanh`, `linear` | yes | partial | partial | yes | yes | ET lowers `unbind` through `unbind_copy`; the engine mostly sees tuple outputs / `getitem` aliases rather than a first-class unbind target. |
| Indexing / data movement | `select_copy`, `slice`, `gather`, `embedding`, `cat`, `pad`, `where` | yes | partial | partial | partial | no | This is the largest uncovered surface. `where` is especially thin in the engine path, and none of these families have direct-fast-lane model coverage yet. |
| Special unary / math | `silu`, `gelu`, `erf`, `sin`, `cos`, `tan`, `exp`, `log`, `tanh` | yes | partial | partial | partial | no | `gelu`/`tanh`/`exp`/`log`/`erf` have engine support through decomposition or table ops. `silu` is support-listed in ET, but is not yet a clean first-class direct-lane target. |
| Quantized I/O boundaries | `quantize_per_tensor`, `quantize_per_channel`, `dequantize_per_tensor`, `dequantize_per_channel` | yes | yes | yes | yes | yes | Direct-lowering packaging is working; keep per-channel and asymmetric boundary cases in regression tests. |

## Prioritized Gaps

1. Indexing / data movement
- Add direct-fast-lane models for `where`, `gather`, `embedding`, `slice`, `pad`, and `cat`.
- `where` is the clearest engine-side hole because the engine ingest table sees it, but the serializer dispatch does not yet show a first-class target.
- This is the best next coverage buildout target because it is supported-adjacent but still unexercised in the direct fast lane.

2. Special unary / math
- Add at least one direct-fast-lane `silu` model and one model that exercises a wider unary chain.
- The ET support list includes these ops, but the engine direct-lane path is still mixed: some ops use decomposition or table emission, while `silu` is not yet a clean engine target.

3. Raw matmul / batched matmul
- Add direct-fast-lane models for `matmul`, `bmm`, and `addmm`.
- Convolution and linear are covered today, but the raw matrix-math surface is not yet represented in the corpus.

4. Quantized edge cases
- Keep pressure on per-channel and asymmetric qparams, especially around non-conv graphs and broadcast-heavy paths.
- The packaging path is working, but this is where regressions tend to show up after new legalization rules.

## Notes

- The current corpus is a useful replacement-stage smoke test, not a proof of full ET parity.
- The matrix above should be treated as a living inventory for the direct-lowering seam, not as a generic ExecuTorch support matrix.
