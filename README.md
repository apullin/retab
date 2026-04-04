# RETAB

A fast TOSA generator for ExecuTorch ARM backends, written in Rust.

Takes a quantized FX graph from ExecuTorch's ARM partitioner and produces
a TOSA flatbuffer — the same output as the Python `TOSABackend._preprocess`,
but **500-1000x faster** (~56ms vs ~60s on a typical quantized CNN).

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

- **1,312 structured op tests**: 0 failures (ExecuTorch ARM test suite)
- **50 random CNN architectures**: exact numerical match vs Python pipeline
  through the TOSA reference model
- **Fuzz testing**: 50 random seeds, both pipelines produce identical outputs

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
ARM_TEST_ENABLE_ARM_TOSA_ENGINE_PATCH=1 \
ARM_TOSA_ENGINE_ROOT=/path/to/retab \
  pytest backends/arm/test/ops/ -k "tosa_INT[" -n 8
```

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
