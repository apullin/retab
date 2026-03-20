# Cleanup Inventory

Snapshot: 2026-03-20

Current state:
- The direct fast-lane path is the live path.
- The expanded local direct-fast-lane corpus is `22 / 22` green.
- The main remaining risk is broader ExecuTorch validation, not local model bring-up.

This inventory separates:
- cleanup that should happen now because it clarifies the product path
- cleanup that should wait until after broader ET validation
- artifacts and historical material that should be archived or ignored, not mixed into runtime code

## Keep As Primary Path

These files represent the dedicated path and should be treated as the product surface:

- [fast_preprocess.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/fast_preprocess.py)
- [src/fx_*](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src)
- [src/ingest/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src/ingest)
- [src/serialize/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src/serialize)
- [tests/sweep_direct_fastlane.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/sweep_direct_fastlane.py)
- [tests/compare_python_vs_direct_fastlane.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/compare_python_vs_direct_fastlane.py)
- [tests/families/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/families)
- [tests/coverage/arm_direct_fastlane_coverage.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/coverage/arm_direct_fastlane_coverage.md)

## Remove Or Simplify Soon

These items are now misleading or overly broad relative to the working direction.

### 1. Retire Hybrid / Python-pass Modes

The public patching API in [fast_preprocess.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/fast_preprocess.py) still exposes:
- `fast`
- `python_passes`
- `python_passes_fast`
- `all_rust`

Recommended cleanup:
- keep `all_rust` as the primary explicit mode
- rename or document `fast` if it is still intended to differ materially from `all_rust`
- remove `python_passes` and `python_passes_fast` once ET validation is complete
- remove stale hybrid wording in docstrings and logs

Why:
- these modes reflect intermediate bring-up states, not the intended product path
- they make tests and docs harder to interpret

### 2. Tighten Serializer Selection

[fast_preprocess.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/fast_preprocess.py) still defaults `ARM_TOSA_ENGINE_SERIALIZER` to `"python"` inside `_all_rust_preprocess`.

Recommended cleanup:
- make Rust serialization the default for the dedicated path
- keep Python serializer only as an explicit debug/compare path
- document the env var as a debug switch, not a normal mode selector

Why:
- today this is a footgun
- the working path is Rust FX + Rust serializer

### 3. Reframe Patch Tests Around The Dedicated Path

[tests/test_preprocess_patch.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/test_preprocess_patch.py) still centers old modes and the monkey-patched backend comparison.

Recommended cleanup:
- make `all_rust` the default path under test
- downgrade `python_passes` variants to optional debug coverage or remove them
- keep only the comparisons that still matter for validating the dedicated path

## Archive Or Quarantine

These are useful for archaeology, but they should stop cluttering the repo root.

### Historical Writeups

Historical writeups are now archived under [docs/history/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history):
- [ASSESSMENT.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/ASSESSMENT.md)
- [AUDIT.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/AUDIT.md)
- [PHASE1_STATUS.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/PHASE1_STATUS.md)
- [P0_FIXES_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/P0_FIXES_WRITEUP.md)
- [P2_FIXES_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/P2_FIXES_WRITEUP.md)
- [P3_FIXES_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/P3_FIXES_WRITEUP.md)
- [REVIEW_1B.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/REVIEW_1B.md)
- [TICKET_1A_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1A_WRITEUP.md)
- [TICKET_1B_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1B_WRITEUP.md)
- [TICKET_1C_IMPL_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1C_IMPL_WRITEUP.md)
- [TICKET_1C_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1C_WRITEUP.md)
- [TICKET_1D_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1D_WRITEUP.md)
- [TICKET_1E_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1E_WRITEUP.md)
- [TICKET_1F_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_1F_WRITEUP.md)
- [TICKET_2A_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_2A_WRITEUP.md)
- [TICKET_2B_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_2B_WRITEUP.md)
- [TICKET_2C_WRITEUP.md](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/docs/history/TICKET_2C_WRITEUP.md)

Recommended cleanup:
- keep new bring-up archaeology under `docs/history/`
- avoid adding fresh one-off status writeups in the repo root

### Diagnostic / Audit Scripts

These folders are still useful, but not as first-class product files:
- [audit/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/audit)
- [diagnostic/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/diagnostic)

Recommended cleanup:
- move them under `tools/diagnostic/` or `docs/diagnostic/`
- separate “kept for active debug” from “historical one-off probe”

## Ignore Or Delete From Worktree

These should not be treated as source:

- [arm_tosa_engine/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/arm_tosa_engine)
- [arm_tosa_engine-0.1.0.dist-info/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/arm_tosa_engine-0.1.0.dist-info)
- [dist/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/dist)
- [tmp_cnn_ln_dump/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tmp_cnn_ln_dump)
- [tmp_cnn_ln_dump2/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tmp_cnn_ln_dump2)
- [tmp_tiny_dump/](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tmp_tiny_dump)
- `tmp_*.log`
- Python `__pycache__/`

Recommended cleanup:
- ensure `.gitignore` covers all of them
- delete them from the working tree once no active investigation depends on them

## Defer Until After ET Validation

These may be good ideas, but doing them now would make regression attribution worse.

### 1. Broad FX/serializer refactors

Do not do broad code motion or style cleanup across:
- [src/fx_*](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src)
- [src/serialize/visitors.rs](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/src/serialize/visitors.rs)

Why:
- these areas still need broader ET battery validation
- large refactors would make failures harder to localize

### 2. Delete comparison infrastructure

Do not remove:
- [tests/compare_python_vs_direct_fastlane.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/compare_python_vs_direct_fastlane.py)
- [tests/lib/tosa_compare.py](/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/tests/lib/tosa_compare.py)

Why:
- these are still useful while validating against ET and the Python reference path

## Suggested Sequence

1. Run targeted `executorch` `exir` and `backends/arm` suites on the current green stack.
2. Clean up `fast_preprocess.py` modes and serializer defaults.
3. Archive root-level history and diagnostic documents into a docs subtree.
4. Tighten `.gitignore` and remove generated artifacts from the worktree.
5. Only after that, consider deeper internal refactors or commit-stack compression.

## Likely First Cleanup PR

A low-risk first cleanup PR in `arm-tosa-engine` should do only this:
- document the direct fast lane as the primary path
- mark hybrid/python-pass modes as deprecated
- update tests/docs to default to `all_rust`
- archive historical writeups under a docs subtree
- ignore generated artifacts

That would improve clarity substantially without touching the fragile lowering core.
