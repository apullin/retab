# Performance Snapshot

This records the latest local stock-vs-RETAB lowering sanity check. It is not a
full benchmark suite; it is a guardrail that catches obvious performance loss
after correctness and refactor work.

Last run: 2026-06-14 on commit `6f254ec`.

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

The CPU phase profiler gives the sharper breakdown for the same model:

| Measured scope | Time |
| --- | ---: |
| Full outer lower | `5.580s` |
| `program._gen_edge_manager_for_partitioners` | `4.604s` |
| `program._generate_edge_program` | `1.867s` |
| `backend.TOSABackend.preprocess` total, 23 calls | `0.131s` |
| `retab_native.run_fx_passes` total, 23 calls | `0.066s` |
| `retab_native.ingest_and_serialize` total, 23 calls | `0.059s` |

So the current MobileNet direct path is not spending seconds in Rust RETAB.
The remaining multi-second cost is mostly ExecuTorch/Torch graph machinery
outside the backend preprocess slice.

The MobileNet artifact mismatch is not treated as a regression by itself here:
the full correctness matrix passes on the Rust serializer path. If a future
performance change also changes artifacts, run the artifact and semantic
comparisons from `FASTLANE_WORKFLOW.md` before accepting it.
