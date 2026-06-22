# Performance Snapshot

This records the latest local stock-vs-RETAB lowering sanity check. It is not a
full benchmark suite; it is a guardrail that catches obvious performance loss
after correctness and refactor work.

Last updated: 2026-06-15.

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

## Current Results

| Model | Stock median | RETAB median | Speedup | PTE artifact |
| --- | ---: | ---: | ---: | --- |
| `tiny_cnn` | `1.401s` | `0.346s` | `4.05x` | identical |
| `mobilenet_v3_small` | `19.581s` | `7.948s` | `2.46x` | differs |

Log path: `target/perf-check-20260614-224136`.

These numbers are end-to-end ARM lowering checks. They include ExecuTorch
orchestration, backend partitioning, Vela compilation, and PTE packing. They are
not isolated measurements of the Rust serializer/preprocess core.

The direct TOSA fast-lane sweep also reports an outer `lower_s`, but that is
still the full `to_edge_transform_and_lower` interval. It is not pure Rust
preprocess time:

| Phase | Time |
| --- | ---: |
| Quant/export prepare | `5.959s` |
| Outer lower | `5.605s` |
| Pack | `0.291s` |

Command:

```bash
python tests/sweep_direct_fastlane.py --model mobilenet_v3_small --json-only
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
comparisons from `FASTLANE_WORKFLOW.md` before accepting it.
