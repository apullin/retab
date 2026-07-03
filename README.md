# RETAB

RETAB is a Rust implementation of ExecuTorch's ARM backend preprocess stage.
It takes the quantized FX graph produced by ExecuTorch's ARM partitioner and
emits the TOSA flatbuffer consumed by Vela for Ethos-U targets such as U55 and
U85.

A substantial Python layer is also shipped. It creates the pluggable seam that
lets RETAB replace the stock preprocess pipeline, and it supports integration,
orchestration, patching, test harnesses, and compatibility glue. The replacement
lowering engine itself lives in Rust: graph transformation, IR handling, and
TOSA serialization.

## Pinned companion stack

RETAB is only one piece of the local ARM lowering stack. The current working
configuration expects concordant commits in these companion repos:

| Repo | Branch | Commit | Why it is needed |
| --- | --- | --- | --- |
| [`apullin/executorch`](https://github.com/apullin/executorch) | `et-python-fastpaths` | `dac215665` | Provides the pluggable ExecuTorch ARM preprocess and Edge manager seams that let RETAB replace `TOSABackend._preprocess`. |
| [`apullin/tosa-rs`](https://github.com/apullin/tosa-rs) | `master` | `de6e6b0` | Provides the Rust TOSA FlatBuffers serialization/deserialization API used by RETAB's serializer. RETAB builds against the vendored copy at `../vendor/tosa-rs`. |
| [`apullin/ethos-u-vela`](https://github.com/apullin/ethos-u-vela) | `retab-regor-opt-5.1.0-curated` | `123a7e28` | Optional for RETAB unit builds, but this is the current downstream Vela/Regor branch used for end-to-end `.pte` generation and Regor speedups. |

Treat this as a pinned toolchain, not as independent rolling dependencies. If
one of these repos moves, rebuild RETAB and re-run the validation matrix before
advancing the pins. The longer compatibility map is in
[docs/COMPANION_REPOS.md](docs/COMPANION_REPOS.md).

## What it replaces

```
ExecuTorch ARM pipeline:
  torch.export → quantize → partition → [TOSABackend._preprocess] → Vela → .pte
                                         ^^^^^^^^^^^^^^^^^^^^^^^^
                                         RETAB replaces this step
```

The `_preprocess` step runs graph transformation passes and serializes to TOSA
FlatBuffers. RETAB reimplements that lowering engine in Rust while preserving
the ExecuTorch-facing Python API surface.

## Benchmarks

These are local U55 lowering measurements on the pinned companion stack above.
They are intended as order-of-magnitude evidence, not a public benchmark suite.
See [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) for command lines,
methodology notes, and runtime validation details.

Timing scope matters. RETAB replaces the backend preprocess/TOSA-generation
slice, but most user-visible scripts measure a wider interval that can also
include ExecuTorch graph machinery, partitioning, Vela compilation, and PTE
packing. Quote both scopes separately.

### Replaced Phase

This isolates the `_preprocess` phase RETAB replaces.

| Model | Stock Python `_preprocess` | RETAB `_preprocess` | Phase speedup |
| --- | ---: | ---: | ---: |
| `tiny_cnn` | `0.627s` | `0.066s` | `9.5x` |

For the larger local N=34 packed-ensemble stress probe, the current repeated
RETAB-owned `_preprocess` measurement is `0.704s ± 0.027s` over 10 samples
after one-time delegate capture. That is the meaningful current number for the
Rust replacement phase; the end-to-end interval around that model is dominated
by export, quantization, partitioning, Vela/Regor, and PTE packing.

### End-to-End Lowering

This wider table measures the full ExecuTorch ARM lowering interval around a
prepared quantized program: partitioning, backend preprocess, Vela, and PTE
packing. The stock column is the baseline total process time without RETAB.
Several rows are single-run public/synthetic model-family probes; see the
benchmark report for exact methodology.

| Model | Size / role | Stock Python ARM | RETAB | Speedup |
| --- | --- | ---: | ---: | ---: |
| `tiny_cnn` | small smoke model | `1.401s` | `0.346s` | `4.05x` |
| `resnet8` | small MLPerf Tiny CNN | `4.151s` | `0.758s` | `5.48x` |
| `deep_autoencoder` | small MLP-style model | `3.822s` | `1.159s` | `3.30x` |
| `vgg16_tiny` | VGG-shaped synthetic CNN | `5.279s` | `1.058s` | `4.99x` |
| `ds_cnn` | small depthwise speech CNN | `4.090s` | `1.461s` | `2.80x` |
| `resnet18_64` | medium residual CNN | `10.662s` | `3.993s` | `2.67x` |
| `mobilenet_v2_96` | medium depthwise CNN | `17.452s` | `5.762s` | `3.03x` |
| `mobilenet_v3_small` | medium real-model check | `19.581s` | `7.948s` | `2.46x` |
| `densenet121` | large public fragmentation stress CNN | `68.319s` | `29.140s` | `2.34x` |

Across the full six-model probe (`vgg16_tiny`, `mobilenet_v2_96`,
`resnet18_64`, `resnet8`, `ds_cnn`, `deep_autoencoder`), RETAB and stock were
byte-identical on FVP for 24/24 input cases. These common model families mostly
matched stock's Vela SRAM and PTE size, which is the expected result for a
semantics-preserving replacement backend.

## Validation

The maintained local validation contract is:

- artifact comparison against the Python reference path
- optional semantic comparison through the TOSA reference model
- full Docker-backed direct ARM acceptance in ExecuTorch

See [docs/DIRECT_LOWERING_WORKFLOW.md](docs/DIRECT_LOWERING_WORKFLOW.md) for
the exact commands and current expected acceptance result.

## Quick start

### Prerequisites

- Rust toolchain (stable)
- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/)
- a seam-enabled ExecuTorch checkout with the required RETAB hook points
  (see [compat/executorch.toml](compat/executorch.toml))
- the companion repository set described in
  [docs/COMPANION_REPOS.md](docs/COMPANION_REPOS.md)

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

For the current direct lowering development workflow, including model probes
and artifact checks, see
[docs/DIRECT_LOWERING_WORKFLOW.md](docs/DIRECT_LOWERING_WORKFLOW.md).

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
├── src/                          # Rust lowering engine
│   ├── fx_*.rs                   # FX graph and IR passes
│   ├── serialize/visitors/       # TOSA op emitters
│   ├── serialize/mod.rs          # TOSA flatbuffer serialization
│   └── ingest/                   # FX graph → Rust IR conversion
├── python/retab/                 # Python integration seam
│   ├── preprocess.py             # Main pipeline orchestration
│   ├── _cf_ctx.py                # Control-flow delegate metadata reads
│   └── _decomp_hooks.py          # Edge decomposition fast paths
└── tests/
    ├── sweep_direct_lowering.py  # Direct-lowering corpus probe
    ├── compare_python_vs_direct_lowering.py
    ├── fuzz_models.py            # Random CNN generator + validation
    └── compare_pipelines.py      # Python-vs-Rust exact match test
```

### Key abstractions

RETAB has converged on a few shared grammars for graph rewriting and
serialization:

- **`FxDecomposer`** — typed DSL for FX graph decompositions (25+ ops)
- **`NodeInserter`** — fluent builder for FX node creation
- **`fx_pass!`** — macro generating FX pass boilerplate (19/35 passes)
- **`EmitContext`** — rich context for TOSA op emitters (14 accessors)
- **`register_unary!/binary!`** — data-driven emitter dispatch tables

## License

Apache-2.0
