# Arm Direct Fast-Lane Coverage Matrix

Snapshot: 2026-03-20

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

Current local result: `22 / 22` models pass in the direct fast lane.

| Family | Representative ops | ET-arm-supported | Direct-seam-reachable | Engine-ingest | Engine-serialize | Local-test-covered | Notes / gaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Conv / depthwise / linear / matmul | `conv2d`, `depthwise_conv2d`, `linear`, `matmul`, `bmm`, `addmm` | yes | yes | yes | yes | yes | Convolution, linear, `matmul`, `bmm`, and `addmm` now have direct-fast-lane coverage. |
| Pool / residual / reduce | `avg_pool2d`, `max_pool2d`, `_adaptive_avg_pool2d`, `sum`, `mean`, `amax` | yes | yes | yes | yes | yes | Covered by `tiny_cnn`, `residual_avgpool_cnn`, `depthwise_separable_cnn`, and norm-family models. |
| Layout / reshape / transpose | `permute_copy`, `view_copy`, `reshape`, `squeeze`, `unsqueeze`, dim-order copies | yes | yes | yes | yes | yes | NHWC layout handling is validated, including reshape/broadcast and runtime-input placeholder rules. |
| Norm / broadcast / table | `layer_norm`, `group_norm`, `native_layer_norm`, `native_group_norm`, `rsqrt`, `reciprocal`, `sqrt`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `TosaTable` | yes | yes | yes | yes | yes | This is now a strength, not the main risk area. |
| Recurrent / tuple plumbing | `unbind_copy`, `getitem`, `sigmoid`, `tanh`, `linear` | yes | partial | partial | yes | yes | Decomposed recurrent coverage is good. The engine still mostly sees tuple outputs and `getitem` aliases rather than a first-class unbind target. |
| Indexing / data movement | `select_copy`, `slice`, `gather`, `embedding`, `cat`, `pad`, `where` | yes | partial | partial | partial | partial | `slice`, `pad`, `cat`, `gather`, `index_select`, and `where` are covered. `embedding` remains the clearest untested member of this family. |
| Special unary / math | `silu`, `gelu`, `erf`, `sin`, `cos`, `tan`, `exp`, `log`, `tanh` | yes | partial | partial | partial | partial | `silu`, `gelu`, `erf`, `exp`, `log`, and `tanh` are covered. `sin`, `cos`, and `tan` remain untested in the direct fast lane. |
| Quantized I/O boundaries | `quantize_per_tensor`, `quantize_per_channel`, `dequantize_per_tensor`, `dequantize_per_channel` | yes | yes | yes | yes | yes | Direct-lowering packaging is working across the current corpus. Keep pressure on asymmetric/per-channel edge cases. |

## Current Gaps

1. Embedding / selected indexing paths
- The data-movement family is much healthier now, but `embedding` is still not represented in the local direct-fast-lane corpus.

2. Remaining special unary ops
- `sin`, `cos`, and `tan` are still support-adjacent but untested here.

3. Broader ET validation
- The local corpus is now green, but this is still not a proof of full ET parity.
- The main remaining risk has shifted from local model bring-up to the broader ET test surface.

## Notes

- The current corpus is a useful replacement-stage smoke/regression suite, not a proof of full ET parity.
- The matrix above should be treated as a living inventory for the direct-lowering seam, not as a generic ExecuTorch support matrix.
