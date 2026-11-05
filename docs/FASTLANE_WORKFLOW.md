# Fast-Lane Workflow

This note records the old direct Rust/TOSA fast lane, whose future is undecided.

It is intentionally operational, not aspirational. Use these commands and
expectations when changing the fast path.

For the current stock stack, see [COMPANION_REPOS.md](COMPANION_REPOS.md).
Maintained benchmarks, the differential harness, and the model corpus are
kept separately, as described in [the README](../README.md#validation).

## Acceptance Gate

The full direct ARM acceptance target is the Docker-backed ExecuTorch suite:

```bash
cd "$EXECUTORCH_SRC"
EXECUTORCH_ARM_DOCKER_IMAGE_TAG=arm-baremetal:latest \
SCCACHE_DISABLE=1 \
./.ci/scripts/run-arm-baremetal-docker-local.sh test_pytest_ops_tosa_direct_lower
```

Current expected shape of a good run:

- `2256 passed`
- `3 skipped`
- `23 xfailed`
- `0 failed`

Do not treat a perf patch as closed out until this suite is green again.

## Canonical Models

Use these models for direct fast-lane work:

- `resnet18`: conventional CNN sanity benchmark
- `mobilenet_v3_small`: harder real-model stress benchmark
- `cnn_layernorm4d`: layout-heavy oddball benchmark

`mobilenet_v3_small` is the best stress case when a change looks promising on a
normal CNN but may fragment or regress on a more awkward graph.

## Quick Lower-Only Probes

For coarse lower/pack timings:

```bash
cd "$RETAB_ROOT"
python tests/sweep_direct_fastlane.py --model resnet18 --json-only
python tests/sweep_direct_fastlane.py --model mobilenet_v3_small --json-only
python tests/sweep_direct_fastlane.py --model cnn_layernorm4d --json-only
```

This harness is good for fast iteration and corpus sweeps.

It is not the final word on microbenchmark stability because it uses one
subprocess per model.

## Artifact Checks

The minimum checked-in artifact check is exact direct-vs-reference TOSA
comparison:

```bash
cd "$RETAB_ROOT"
python tests/compare_python_vs_direct_fastlane.py \
  --model resnet18 \
  --model mobilenet_v3_small \
  --seed 0
```

This reports:

- TOSA SHA-256
- byte equality
- op sequence equality
- tensor metadata diffs

For deeper artifact inspection on one model:

```bash
python tests/compare_model_artifact_details.py --model mobilenet_v3_small --seed 0
```

For a stronger semantic check through the TOSA reference model:

```bash
python tests/compare_model_semantics.py --model mobilenet_v3_small --seed 0
```

## Patch Safety Rules

For a fast-lane optimization to be worth keeping:

- lower as a single delegate on the target model when that was the baseline
- keep artifact hashes identical unless the change is explicitly intended to
  alter the lowered form
- keep `while`/control-flow behavior on the generic path
- pass the full Docker acceptance gate before closeout

If a change wins time but fragments into multiple delegates, that is not a keep.

## Experimental Parallel Backend Preprocess

RETAB exposes an opt-in escape hatch for badly fragmented lowers that produce
many delegates:

```python
from retab.integration.executorch_arm import activated

with activated(parallel_backend_preprocess_workers=14):
    ...
```

Equivalent env-var form:

```bash
RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS=14
```

Use it only for fragmented multi-delegate cases. It is not enabled by default
because some cleaner single-delegate models regress or stay flat.

The current implementation also falls back to serial when debug artifact dumps
are enabled, to avoid path collisions across worker processes.

## Benchmark Discipline

Use subprocess sweeps for rough triage.

Use repeated medians in one persistent Python process for headline comparisons.
The subprocess harness is noisy enough that it should not be the only evidence
for a small claimed win.

For perf patches, the preferred sequence is:

1. quick lower-only probe on `resnet18` and `mobilenet_v3_small`
2. exact artifact compare on the same models
3. optional semantic compare when the artifact changed or the edit was risky
4. full Docker acceptance run
