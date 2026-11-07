# RETAB

RETAB is a Rust/PyO3 implementation of ExecuTorch's Arm TOSA preprocessing
stage, focused on Ethos-U55 INT lowering. It transforms quantized FX graphs
and serializes TOSA flatbuffers for Vela. The goal is numerical agreement with
stock ExecuTorch v1.6.0-rc4, with matching structure where practical, while
reducing preprocessing time.

## Pinned companion stack

RETAB runs on stock ExecuTorch; no ExecuTorch fork is required. The tested
Python stack is:

| Component | Version |
| --- | --- |
| ExecuTorch | Wheel `1.6.0`, upstream tag `v1.6.0-rc4`, commit `4776f71b6` |
| torch | `2.15.0` |
| torchvision | `0.29.2` |
| torchao | `0.18.0` (the tested environment uses `0.18.0+cpu`) |
| ethos-u-vela | `5.2.0` from PyPI |
| tosa-tools | `2026.5.0` |
| flatbuffers (Python) | `25.2.10` |

Cargo fetches [`tosa-rs`](https://github.com/apullin/tosa-rs) v0.2.0 by git
revision. It ships its generated FlatBuffers code, so building RETAB needs no
`flatc`. The Rust FlatBuffers runtime is locked separately in `Cargo.lock`; the
Python package version above is not a Rust dependency pin. See
[docs/COMPANION_REPOS.md](docs/COMPANION_REPOS.md) and
[compat/executorch.toml](compat/executorch.toml) for compatibility details.

## Integration and scope

```text
torch.export -> quantize -> Arm partitioner -> TOSABackend._preprocess
                                                | RETAB: Rust passes + TOSA
                                                v
                                             Vela -> .pte
```

[python/retab/preprocess.py](python/retab/preprocess.py) replaces
`TOSABackend._preprocess` at runtime and preserves the stock implementation
for fallback and restoration. Activation also installs guarded decomposition
fast paths from `_decomp_hooks.py` against available ExecuTorch and torch
helpers.

RETAB handles TOSA 1.0 INT graphs with the target's default pass pipeline.
It hands these cases to stock preprocessing:

- float profiles;
- TOSA versions whose major/minor version is not 1.0;
- TOSA development mode;
- custom pass pipeline configurations;
- registered pass insertions;
- delegate-owned mutable buffers;
- symbolic tensor shapes.

`retab.preprocess.preprocess_stats()` reports RETAB calls and stock fallback
counts by reason in the current process.

## Quick start

### Prerequisites

- A stable Rust toolchain.
- Python >= 3.10; the tested environment uses Python 3.12.
- [uv](https://docs.astral.sh/uv/).
- Network access for the first `cargo build`, which fetches `tosa-rs` from
  GitHub (`cargo fetch` once to build offline afterwards).

### Create an environment explicitly

```bash
git clone https://github.com/apullin/retab.git
cd retab
export RETAB_ROOT="$PWD"
export VENV="$RETAB_ROOT/.venv"
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  "executorch==1.6.0" "torch==2.15.0" "torchvision==0.29.2" \
  "torchao==0.18.0" "ethos-u-vela==5.2.0" \
  "tosa-tools==2026.5.0" "flatbuffers==25.2.10" maturin pytest
```

This explicit install is the supported setup; RETAB ships no installer
script. See [docs/COMPANION_REPOS.md](docs/COMPANION_REPOS.md) for the tested
versions.

### Build and test

```bash
cd "$RETAB_ROOT"
export PYO3_PYTHON="$VENV/bin/python"
cargo test --lib --locked
./scripts/build_retab.sh --venv "$VENV"
PYTHONPATH="$RETAB_ROOT/python" "$VENV/bin/python" -P -m pytest tests
```

The build helper requires an existing environment supplied with `--venv` or
through an activated environment's `VIRTUAL_ENV`. It fails if neither is
provided and never creates an environment implicitly.

Use `cargo test --lib --locked` for the Rust check and maturin to build the
extension. `cargo build --lib` alone does not link this `cdylib` correctly.
`pyo3/extension-module` is enabled only by maturin, so Rust unit tests can link
to Python normally.

To produce a wheel:

```bash
"$VENV/bin/maturin" build --release --locked \
  --interpreter "$VENV/bin/python" --out "$RETAB_ROOT/target/wheels"
```

Maturin takes the package version from `Cargo.toml`.

`tests/` contains Python integration, placeholder binding, control-flow
metadata, and CNN lowering smoke tests. Tests requiring the native extension,
ExecuTorch's Arm backend, or the TOSA reference model skip when those
components are unavailable.

### Smoke lower

```bash
./scripts/smoke_checkout_build_and_lower.sh --venv "$VENV"
"$VENV/bin/python" -P examples/lower_with_retab.py --out tmp/tiny_cnn.pte
```

### Python API

```python
from retab.integration.executorch_arm import activated
from retab.preprocess import preprocess_stats

with activated():
    # Run ExecuTorch Arm lowering as usual.
    ...

print(preprocess_stats())
```

`install()` and `uninstall()` provide the same integration without a context
manager. Nested activation restores the stock backend when the outermost
activation exits.

By default only `TOSABackend._preprocess` is replaced. `activated(fast_paths=True)`
also installs RETAB's edge-manager and `run_decompositions` fast paths, which
speed up the rest of `to_edge_transform_and_lower` but have no test coverage on
rc4 yet.

For fragmented graphs, an experimental multi-process backend preprocessing
mode is available through `activated(parallel_backend_preprocess_workers=8)`
or `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS`. It is opt-in and returns to
serial execution for debug artifact dumps or when fork multiprocessing is
unavailable. Counters remain process-local.

[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) documents all environment reads.

## Validation

Benchmarks, the stock-versus-RETAB differential harness, and the model corpus
are kept separately in a private repository.

## Architecture

```text
retab/
├── src/
│   ├── lib.rs                   # PyO3 entry points and stashed IR hierarchy
│   ├── fx_ir.rs                 # Mutable rewrite graph and pass runner
│   ├── fx_meta.rs               # Tensor (value, dim order), quantization and passthrough metadata
│   ├── fx_pipeline.rs           # Ordered parent and submodule pipelines
│   ├── fx_*_ir.rs               # Rust graph transformation passes
│   ├── layout/                  # rc4's layout model: permute/view passes and rewrites
│   ├── control_flow/            # Region assembly and boundary contracts
│   ├── ingest/                  # Rewrite IR to serializer graph conversion
│   ├── graph/                   # Serializer graph (slotmap) and typed OpTarget
│   └── serialize/               # TOSA visitors and flatbuffer assembly
├── python/retab/
│   ├── integration/             # Stable ExecuTorch install/activation API
│   ├── preprocess.py            # Backend scope, orchestration, and counters
│   └── _decomp_hooks.py         # Guarded decomposition fast paths
├── compat/executorch.toml       # Tested stock stack and the seams RETAB uses
└── tests/                       # Python integration tests and rc4 inventories
```

`IrGraph`, `IrPass` / `EffectfulIrPass`, and `IrRewriteCursor` implement graph
rewrites. `NodeMeta` and `TensorMetaStamp` carry tensor facts. The serializer
uses a typed `OpTarget`, `EmitContext`, and an emitter dispatch table.

See [docs/rust_lowering_architecture.md](docs/rust_lowering_architecture.md)
for the current implementation map. Earlier design notes are under
[docs/history/](docs/history/). The fork-only direct-lowering lane is no
longer part of RETAB; it is parked as a speedup candidate and stays in git
history.

## License

Apache-2.0; see [LICENSE](LICENSE).
