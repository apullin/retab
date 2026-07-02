# Benchmark Report

This records the current local stock-vs-RETAB lowering checks. It is not a
formal benchmark suite; it is a practical guardrail for performance, artifact
shape, and runtime equivalence on the pinned companion stack.

Last updated: 2026-06-28.

## Command Shape

The benchmark compares stock ExecuTorch ARM lowering against the supported
RETAB integration on the same prepared quantized `ExportedProgram`.

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
PYTHONPATH=/Users/andrewpullin/personal/executorch-work/ethos-u-vela-5.0.0:/Users/andrewpullin/personal/executorch-work/executorch/src:$PWD \
  /Users/andrewpullin/personal/executorch-work/.venv/bin/python \
  scripts/benchmark_stock_vs_retab.py \
  --model tiny_cnn \
  --runs 3 \
  --warmup 1 \
  --calibration-runs 1
```

Repeat with `--model mobilenet_v3_small` for the real-model stress check.

## Timing Scopes

Do not compare timing numbers unless they measure the same scope:

- **Backend preprocess / TOSA generation**: the slice RETAB replaces. This is
  where Rust graph passes and TOSA serialization run.
- **End-to-end ARM lowering**: `to_edge_transform_and_lower` plus PTE packing.
  This includes ExecuTorch graph machinery, partitioning, backend preprocess,
  Vela compilation, and final program packaging.
- **Full export pipeline**: model export, quantization calibration/conversion,
  lowering, Vela, and PTE packaging.

The product-facing benchmark below reports end-to-end ARM lowering, not pure
RETAB preprocess time. Isolated backend-preprocess measurements can be tens of
milliseconds on smaller or already-fragmented delegate graphs, but that is not
the same as the full user-visible lowering job.

## Replaced-Phase Result

This helper captures the exact `TOSABackend._preprocess` interval for the stock
Python path and the direct RETAB path on the same public synthetic model:

```bash
PYTHONPATH=$PWD:/Users/andrewpullin/personal/executorch-work/executorch/src \
  python tests/compare_python_vs_direct_lowering.py --model tiny_cnn --seed 0
```

| Model | Stock Python `_preprocess` | RETAB `_preprocess` | Phase speedup |
| --- | ---: | ---: | ---: |
| `tiny_cnn` | `0.627s` | `0.066s` | `9.5x` |

The same run reported a stock reference lower interval of `1.142s` and a direct
RETAB lower interval of `0.066s`, but that direct helper is a TOSA artifact
comparison path, not the product-facing U55 end-to-end lowering benchmark below.

## End-to-End Lowering Results

| Model | Stock median | RETAB median | Speedup | PTE artifact |
| --- | ---: | ---: | ---: | --- |
| `tiny_cnn` | `1.401s` | `0.346s` | `4.05x` | identical |
| `mobilenet_v3_small` | `19.581s` | `7.948s` | `2.46x` | differs |

Log path: `target/perf-check-20260614-224136`.

These numbers are end-to-end ARM lowering checks. They include ExecuTorch
orchestration, backend partitioning, Vela compilation, and PTE packing. They are
not isolated measurements of the Rust serializer/preprocess core.

Additional public slow-path stress probe:

```bash
python scripts/benchmark_stock_vs_retab.py \
  --model densenet121 \
  --runs 1 \
  --warmup 0 \
  --calibration-runs 1
```

| Model | Stock lower | RETAB lower | Speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| `densenet121` | `68.319s` | `29.140s` | `2.34x` | many delegated fragments; public torchvision model |

The direct TOSA lowering sweep also reports an outer `lower_s`, but that is
still the full `to_edge_transform_and_lower` interval. It is not pure Rust
preprocess time:

| Phase | Time |
| --- | ---: |
| Quant/export prepare | `5.959s` |
| Outer lower | `5.605s` |
| Pack | `0.291s` |

Command:

```bash
python tests/sweep_direct_lowering.py --model mobilenet_v3_small --json-only
```

Before the direct-lower size guard, the CPU phase profiler gave this sharper
breakdown for the same model:

| Measured scope | Time |
| --- | ---: |
| Full outer lower | `5.580s` |
| `program._gen_edge_manager_for_partitioners` | `4.604s` |
| `program._generate_edge_program` | `1.867s` |
| `backend.TOSABackend.preprocess` total, 23 calls | `0.131s` |
| `retab_native.run_fx_passes` total, 23 calls | `0.066s` |
| `retab_native.ingest_and_serialize` total, 23 calls | `0.059s` |

That pre-guard MobileNet path was not spending seconds in Rust RETAB. The
remaining multi-second cost was mostly ExecuTorch/Torch graph machinery outside
the backend preprocess slice.

## N=34 Packed-Ensemble Stress Probe

The local `test-network` branch includes a full packed/semi-grouped ensemble
architecture in history (`dd8948:TEST_NETWORK.md`). This is a better proxy for
the deployed-style workload than public VGG: small parameter count, one clean
delegate, many graph nodes, grouped convolutions, rank-5 views, and channel-
direct normalization.

Temporary probe command shape:

```bash
/tmp/et.sh
```

where `/tmp/et.sh` reconstructed the N=34 model from `dd8948:TEST_NETWORK.md`
and wrapped RETAB native calls for timing. The first profile exposed accidental
quadratic graph-bookkeeping costs in user-edge lookup/rewrite. Subsequent runs
also removed wrapper overhead that no longer mattered after IR-native
serialization, such as copying the Python `state_dict` even though the Rust
serializer now uses the stashed IR tensor data.

### Current RETAB Preprocess Distribution

The most stable current measurement is the backend preprocess slice RETAB
actually owns. The harness prepares/lowers once to capture the delegated
`ExportedProgram`, then repeatedly calls RETAB `_all_rust_preprocess` on a fresh
deep-copy of that delegate. This avoids rerunning quantization, partitioning,
Vela, and PTE packing for every sample.

Post-cleanup 10-sample check:

| Scope | Mean | Stddev | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| RETAB `_preprocess` wrapper | `0.704s` | `0.027s` | `0.692s` | `0.676s` | `0.751s` |
| `retab_native.run_fx_passes` | `0.600s` | `0.023s` | `0.589s` | `0.576s` | `0.639s` |
| `retab_native.ingest_and_serialize` | `0.102s` | `0.004s` | `0.101s` | `0.097s` | `0.108s` |

Output TOSA bytes were stable across the samples: `7,108,436` B.

Previous 30-sample check before the last wrapper/rebuild cleanup:

| Scope | Mean | Stddev | Median | 10% trimmed mean |
| --- | ---: | ---: | ---: | ---: |
| RETAB `_preprocess` wrapper | `0.817s` | `0.239s` | `0.718s` | `0.750s` |
| `retab_native.run_fx_passes` | `0.647s` | `0.153s` | `0.583s` | `0.608s` |
| `retab_native.ingest_and_serialize` | `0.117s` | `0.036s` | `0.101s` | `0.109s` |

The wide 30-sample standard deviation came from several Python/process outliers,
not from unstable output: bytes were identical in every run.

### End-to-End Context

The full end-to-end interval around this N=34 model is intentionally not used
as the main RETAB speed claim. It includes substantial work outside RETAB:
Torch export, quantization calibration/conversion, ExecuTorch partitioning,
Vela/Regor, and PTE packing. In recent local probe runs the one-time
`to_edge_transform_and_lower` interval varied from roughly `41s` to `54s`,
while the repeated RETAB preprocess slice stayed around `0.7s`.

Fresh broad-process timing on 2026-06-28 with the current tree:

| Scope | Time |
| --- | ---: |
| Quant/export prepare | `75.067s` |
| `to_edge_transform_and_lower` total | `49.874s` |
| RETAB `_preprocess` inside lower | `0.851s` |
| `retab_native.run_fx_passes` | `0.713s` |
| `retab_native.ingest_and_serialize` | `0.136s` |
| Lowering outside RETAB `_preprocess` | `49.023s` |
| PTE pack | `3.136s` |
| Total prepare + lower + pack | `128.078s` |

Artifact/context from the same run:

| Metric | Value |
| --- | ---: |
| Quantized graph nodes | `8078` |
| Delegates | `1` |
| PTE size | `2,901,728` B |
| Vela SRAM | `86.81 KiB` |
| Vela off-chip flash | `1617.92 KiB` |
| NPU operators | `3513` |
| CPU operators | `0` |

In this current broad run, RETAB `_preprocess` is about `0.7%` of the full
prepare+lower+pack interval and about `1.7%` of the `to_edge_transform_and_lower`
interval. Further RETAB micro-optimization can improve the replacement-phase
multiplier, but it will not materially move this full-process wall time unless
more of the surrounding ExecuTorch/Torch/Vela process is also moved or bypassed.

Historical single-run local result after the first hot-path fixes:

| Metric | Value |
| --- | ---: |
| Parameters | `1,109,828` |
| Group sizes | `[7, 7, 7, 7, 6]` |
| Quant/export prepare | `62.530s` |
| Quantized graph nodes | `8078` |
| Delegates | `1` |
| RETAB lower interval | `28.041s` |
| RETAB backend preprocess override | `0.608s` |
| `retab_native.run_fx_passes` | `0.502s` |
| `retab_native.ingest_and_serialize` | `0.088s` |
| RETAB native timed subtotal | `0.591s` |
| PTE pack | `3.081s` |
| Direct Vela on captured TOSA | `0.851s` |
| TOSA size | `7,118,516` B |
| PTE size | `6,046,192` B |
| Total SRAM reported by Vela | `152.31 KiB` |
| NPU operators | `5592` |
| CPU operators | `0` |

The older `RETAB lower interval` row above is kept as historical context only.
It should not be quoted as a current repeated benchmark; the repeated preprocess
distribution above is the current RETAB-owned measurement.

Before/after on the RETAB-side hot path, showing the main algorithmic cleanup:

| Scope | Before | After |
| --- | ---: | ---: |
| RETAB backend preprocess override | `5.855s` | `0.608s` |
| `retab_native.run_fx_passes` | `5.743s` | `0.502s` |
| All profiled IR passes | `5.399s` | `0.331s` |
| `IrToTosaMemoryFormat` | `3.327s` | `0.076s` |
| `IrFoldQdq` | `0.829s` | `0.038s` |
| `IrInsertInt32Rescale` | `0.424s` | `0.017s` |
| `IrInsertConvAccumulatorRescale` | `0.339s` | `0.006s` |

Interpretation:

- This is not a Regor/Vela compile pathology locally. Direct Vela on the
  captured TOSA is about one second.
- The original `5.855s` preprocess interval was RETAB-side, not the full
  ExecuTorch outer lower and not direct Vela. Most of it came from repeated
  whole-graph user scans and whole-graph user-list rebuilds during IR rewrites.
- `run_fx_passes` is Rust/PyO3 native code, but it still crosses the Python
  object seam to read the FX graph and exported-program metadata. Treat this as
  RETAB native pass/ingest cost, not as pure isolated Rust compute.
- After the follow-on cleanup, the current center of the full RETAB
  `_preprocess` wrapper is about `0.7s` on this 8k-node graph. Reaching `0.05s`
  would require a different pipeline strategy, such as selective/dirty pass
  scheduling or avoiding the Python FX import seam, not just local hot-loop
  tuning.

## Direct-Lower Size Guard

Direct lowering is excellent for small graphs, but on larger graphs it can
speculatively preprocess many fragments before falling back to a one-delegate
standard result. The integration now skips direct lowering above
`RETAB_DIRECT_LOWER_NODE_LIMIT`, default `128` graph nodes.

Single-run local timings after the guard:

| Model | Nodes | Direct enabled | Direct disabled | Notes |
| --- | ---: | ---: | ---: | --- |
| `tiny_cnn` | `36` | `0.100s` | `0.394s` | direct path retained |
| `resnet18` | `231` | `2.121s` | `2.091s` | guarded path avoids 11 speculative preprocess calls |
| `mobilenet_v3_small` | `636` | `4.942s` | `5.009s` | guarded path avoids 23 speculative preprocess calls |

Focused model validation after the guard:

```bash
python -m pytest -q --numprocesses=4 \
  backends/arm/test/models/test_resnet18.py \
  backends/arm/test/models/test_mobilenet_v3_arm.py \
  -k tosa
```

Result: `7 passed`.

Full models validation after the guard: `152 passed, 3 skipped, 9 xfailed`.

The MobileNet artifact mismatch is not treated as a regression by itself here:
the full correctness matrix passes on the Rust serializer path. If a future
performance change also changes artifacts, run the artifact and semantic
comparisons from `DIRECT_LOWERING_WORKFLOW.md` before accepting it.

## Model-Family Runtime Sweep

This sweep compiled several distinct neural-network families with both RETAB
and stock ExecuTorch ARM lowering, then ran each PTE on Corstone-300/U55 FVP
with four deterministic input cases.

Result: `24 / 24` RETAB-vs-stock FVP output comparisons were byte-identical.

Lowering timings are single-run local probes, not repeated medians:

| Model | Stock lower | RETAB lower | Speedup |
| --- | ---: | ---: | ---: |
| `vgg16_tiny` | `5.279s` | `1.058s` | `4.99x` |
| `mobilenet_v2_96` | `17.452s` | `5.762s` | `3.03x` |
| `resnet18_64` | `10.662s` | `3.993s` | `2.67x` |
| `resnet8` | `4.151s` | `0.758s` | `5.48x` |
| `ds_cnn` | `4.090s` | `1.461s` | `2.80x` |
| `deep_autoencoder` | `3.822s` | `1.159s` | `3.30x` |

Runtime artifact shape:

| Model | RETAB PTE | Stock PTE | RETAB SRAM | Stock SRAM | RETAB NPU ops | Stock NPU ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vgg16_tiny` | `253,968` B | `253,968` B | `128.75 KiB` | `128.75 KiB` | `24` | `24` |
| `mobilenet_v2_96` | `2,305,376` B | `2,304,896` B | `274.41 KiB` | `274.41 KiB` | `94` | `90` |
| `resnet18_64` | `10,213,840` B | `10,213,840` B | `114.41 KiB` | `114.41 KiB` | `36` | `36` |
| `resnet8` | `88,736` B | `88,736` B | `50.78 KiB` | `50.78 KiB` | `18` | `18` |
| `ds_cnn` | `34,016` B | `34,016` B | `21.56 KiB` | `21.56 KiB` | `11` | `11` |
| `deep_autoencoder` | `284,880` B | `284,880` B | `0.75 KiB` | `0.75 KiB` | `10` | `10` |

`vgg16_tiny` is a reduced-channel VGG16-shaped model, not full torchvision
VGG16. Full VGG16 is not a practical routine U55/FVP sanity target because its
classifier weights dominate the artifact and runtime.
