# Validation Matrix

This is the local acceptance matrix for the Rust RETAB lowering engine. Use it
when deciding whether a change is green enough to commit or start a new refactor.

Last full local validation: 2026-06-14 on commit `6f254ec`.

## Environment

Run from the ExecuTorch checkout with the uv-managed environment:

```bash
cd /Users/andrewpullin/personal/executorch-work/executorch
source ../.venv/bin/activate
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
export ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING=1
export ARM_TEST_ENABLE_ARM_TOSA_ENGINE_PATCH=1
export ARM_TOSA_ENGINE_SERIALIZER=rust
export ARM_TOSA_ENGINE_ROOT=/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
export PYTHONPATH=$PWD/src
```

Before Python regression lanes, rebuild and reinstall the extension:

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
cargo check --lib
maturin develop --release
```

`maturin develop --release` must print `Installed retab-0.1.0`. If it does
not, the Python lanes may be using a stale binary.

## Current Green Matrix

| Lane | Command shape | Latest result |
| --- | --- | --- |
| Rust check | `cargo check --lib` | passed |
| Rust unit tests | `cargo test --lib` | `145 passed` |
| INT ops | `python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models -k "tosa and not FP and (INT or 16a8w or a16w8 or quantized or qat)"` | `2325 passed, 2 skipped, 59 xfailed` |
| FP ops | `python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models -k "tosa and FP"` | `1911 passed, 13 skipped, 66 xfailed` |
| Models | `python -m pytest -q --numprocesses=4 backends/arm/test/models -k tosa` | `152 passed, 3 skipped, 9 xfailed` |

Models must stay capped at `--numprocesses=4`; wider runs can trigger memory
pressure and swap death on the local machine. Ops lanes are safe to run wide.

## Fast Checks

Use these before a commit that only touches Rust lowering internals:

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
cargo fmt
cargo check --lib
cargo test --lib
```

For targeted Python failures, rebuild with `maturin develop --release` and run
the exact failing pytest node before launching broad lanes.

## Log Records

Recent full-regression logs:

| Date | Path | Notes |
| --- | --- | --- |
| 2026-06-14 | `target/full-regression-followup-20260614-221746` | FP ops and models follow-up after erfinv fix |
| 2026-06-14 | `target/full-regression-20260614-214921` | Earlier full run that exposed erfinv INT edge failures |

