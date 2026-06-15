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

The MobileNet artifact mismatch is not treated as a regression by itself here:
the full correctness matrix passes on the Rust serializer path. If a future
performance change also changes artifacts, run the artifact and semantic
comparisons from `FASTLANE_WORKFLOW.md` before accepting it.

