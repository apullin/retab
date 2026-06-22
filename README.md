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
- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/)
- a seam-enabled ExecuTorch checkout with the required RETAB hook points
  (see [compat/executorch.toml](compat/executorch.toml))

### Bootstrap a dedicated UV environment

Use RETAB in its own virtual environment. Do not rely on the user's base
Python install.

```bash
git clone https://github.com/apullin/retab.git
cd retab
./scripts/bootstrap_uv.sh --source /path/to/executorch
```

This will:

- create `.venv/`
- install the Torch stack into that environment
- install ExecuTorch's Python dependencies and expose the source checkout via a
  `.pth` overlay
- build RETAB with `maturin develop --release`
- run a smoke lowering pass

### Build only

If the environment already exists and ExecuTorch is already installed:

```bash
./scripts/build_retab.sh
```

### Smoke lower

```bash
./scripts/smoke_checkout_build_and_lower.sh
```

### Example lowering script

```bash
.venv/bin/python examples/lower_with_retab.py --out tmp/tiny_cnn.pte
```

For highly fragmented multi-delegate lowers, RETAB also exposes an
experimental parallel backend-preprocess mode:

```bash
.venv/bin/python examples/lower_with_retab.py \
  --parallel-backend-preprocess-workers 14 \
  --out tmp/tiny_cnn.pte
```

### Benchmark stock ExecuTorch vs RETAB

```bash
.venv/bin/python scripts/benchmark_stock_vs_retab.py --model mobilenet_v3_small
```

To benchmark the experimental parallel backend-preprocess path on the RETAB
side only:

```bash
.venv/bin/python scripts/benchmark_stock_vs_retab.py \
  --model densenet121 \
  --parallel-backend-preprocess-workers 14
```

This runs the same prepared quantized model through:

- stock ExecuTorch Arm lowering
- RETAB-activated lowering

and reports median lower time plus final `.pte` SHA-256 hashes.

The stock-vs-RETAB benchmark currently expects an environment with the native
`tosa_serializer` package available for the stock ExecuTorch Arm path.

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
from retab.integration.executorch_arm import activated

with activated():
    # ... run ExecuTorch ARM lowering as usual ...
    # RETAB is now active for all TOSA generation
    ...
```

The same API exposes the experimental multi-process backend-preprocess path:

```python
with activated(parallel_backend_preprocess_workers=14):
    ...
```

Keep that opt-in. It is useful for heavily fragmented lowers with many
delegates, but it is not a universal win on already-clean single-delegate
graphs.

## Architecture

```
retab/
├── src/                          # Rust (24k lines)
│   ├── fx_*.rs                   # 47 FX graph passes (via PyO3)
│   ├── serialize/visitors/       # 107 TOSA op emitters
│   ├── serialize/mod.rs          # TOSA flatbuffer serialization
│   └── ingest/                   # FX graph → Rust IR conversion
├── python/retab/                 # Python integration seam
│   ├── preprocess.py             # Main pipeline orchestration
│   ├── _cf_ctx.py                # Control-flow delegate metadata reads
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
