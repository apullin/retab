# RETAB

A fast TOSA generator for ExecuTorch ARM backends, written in Rust.

Takes a quantized FX graph from ExecuTorch's ARM partitioner and produces
a TOSA flatbuffer for the direct Rust/TOSA fast lane.

Current local benchmarking, artifact checks, and acceptance workflow are
documented in [docs/FASTLANE_WORKFLOW.md](docs/FASTLANE_WORKFLOW.md).

## What it replaces

```
ExecuTorch ARM pipeline:
  torch.export → quantize → partition → [TOSABackend._preprocess] → Vela → .pte
                                         ^^^^^^^^^^^^^^^^^^^^^^^^
                                         RETAB replaces this step
```

The `_preprocess` step runs ~30 graph transformation passes and serializes
to TOSA FlatBuffers. RETAB reimplements both in Rust (24k lines) with a
Python orchestration layer (7k lines).

## Validation

The maintained local validation contract is:

- artifact comparison against the Python reference path
- optional semantic comparison through the TOSA reference model
- full Docker-backed direct ARM acceptance in ExecuTorch

See [docs/FASTLANE_WORKFLOW.md](docs/FASTLANE_WORKFLOW.md) for the exact
commands and current expected acceptance result.

## Quick start

### Prerequisites

- Rust toolchain (stable)
- Python ≥ 3.10
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)
- ExecuTorch with the ARM backend (see below)

### Install RETAB

```bash
git clone https://github.com/apullin/retab.git
cd retab
pip install -e .   # or: maturin develop --release
```

### Set up ExecuTorch

Use the [`use-retab`](https://github.com/apullin/executorch/tree/use-retab)
branch which adds a one-commit hook to load RETAB:

```bash
git clone https://github.com/apullin/executorch.git -b use-retab
cd executorch
pip install -e .
```

### Run the test suite

```bash
# From the executorch directory:
EXECUTORCH_ARM_DOCKER_IMAGE_TAG=arm-baremetal:latest \
SCCACHE_DISABLE=1 \
  ./.ci/scripts/run-arm-baremetal-docker-local.sh test_pytest_ops_tosa_direct_lower
```

For the current direct fast-lane development workflow, including model probes
and artifact checks, see [docs/FASTLANE_WORKFLOW.md](docs/FASTLANE_WORKFLOW.md).

### Python API

```python
from retab import patch_tosa_backend, unpatch_tosa_backend

# Monkey-patch TOSABackend._preprocess with the fast Rust path
patch_tosa_backend()

# ... run ExecuTorch ARM lowering as usual ...
# RETAB is now active for all TOSA generation

# Restore the original Python path
unpatch_tosa_backend()
```

## Architecture

```
retab/
├── src/                          # Rust (24k lines)
│   ├── fx_*.rs                   # 47 FX graph passes (via PyO3)
│   ├── serialize/visitors/       # 107 TOSA op emitters
│   ├── serialize/mod.rs          # TOSA flatbuffer serialization
│   └── ingest/                   # FX graph → Rust IR conversion
├── python/retab/               # Python (7k lines)
│   ├── preprocess.py             # Main pipeline orchestration
│   ├── _control_flow.py          # Control-flow submodule handling
│   ├── _helpers.py               # FX graph manipulation helpers
│   └── _decomp_hooks.py          # Edge decomposition fast paths
└── tests/
    ├── fuzz_models.py            # Random CNN generator + validation
    └── compare_pipelines.py      # Python-vs-Rust exact match test
```

### Key abstractions

- **`FxDecomposer`** — typed DSL for FX graph decompositions (25+ ops)
- **`NodeInserter`** — fluent builder for FX node creation
- **`fx_pass!`** — macro generating FX pass boilerplate (19/35 passes)
- **`EmitContext`** — rich context for TOSA op emitters (14 accessors)
- **`register_unary!/binary!`** — data-driven emitter dispatch tables

## License

Apache-2.0
