# RETAB

RETAB is a drop-in Rust replacement for the TOSA preprocess step of
ExecuTorch's Arm backend (`TOSABackend._preprocess`), built for Ethos-U55 INT
lowering. It runs on stock ExecuTorch v1.6.0-rc4 (the `executorch==1.6.0`
wheel), with no fork: one call swaps it in at runtime, and the rest of
`to_edge_transform_and_lower`, Vela included, stays stock.

- **Same output as stock.** ExecuTorch's own Arm operator tests (the
  `tosa_INT` cases in `backends/arm/test/ops`, about 2,400) pass with RETAB
  installed, and on a 27-delegate model corpus Vela compiles RETAB's TOSA to
  command streams and weights byte-identical to stock's
  ([Validation](#validation)).
- **Stock fallback.** Graphs outside RETAB's scope, such as float profiles or
  symbolic shapes, go to stock preprocess automatically
  ([Scope](#integration-and-scope)).
- **Fast.** Preprocess takes about 20 ms instead of 3 s on resnet18, and about
  130 ms instead of 30-50 s on convnext_tiny ([Speedup](#speedup),
  preliminary).

## Install

RETAB builds from source: there is no PyPI release, and the `retab` package on
PyPI is an unrelated project. You need [uv](https://docs.astral.sh/uv/) and
[rustup](https://rustup.rs/); `rust-toolchain.toml` pins Rust 1.98.1, which
rustup installs on first use.

```bash
git clone https://github.com/apullin/retab.git
cd retab
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "executorch==1.6.0" "torch==2.15.0" "torchvision==0.29.2" "torchao==0.18.0" \
  "ethos-u-vela==5.2.0" "tosa-tools==2026.5.0" "flatbuffers==25.2.10" maturin pytest
./scripts/build_retab.sh --venv "$PWD/.venv"   # maturin develop --release --locked
```

The last line builds the extension and installs `retab` into the venv. To add
RETAB to an existing environment with this stack, run only that line, with
`--venv` pointing at it.

## Use

Wrap your usual Ethos-U lowering in `activated()`:

```python
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from retab.integration.executorch_arm import activated

with activated():  # TOSABackend._preprocess is RETAB's inside this block
    edge = to_edge_transform_and_lower(
        quantized_ep,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
pte = edge.to_executorch().buffer
```

Here `quantized_ep` is your PT2E-quantized `ExportedProgram` and `compile_spec`
your `EthosUCompileSpec`; [examples/lower_with_retab.py](examples/lower_with_retab.py)
builds both for a small CNN. To keep RETAB on for the whole process instead,
call `install()` once, and `uninstall()` to restore stock.
`retab.preprocess.preprocess_stats()` counts the delegates RETAB lowered and
the ones that went to stock, by reason.

## Speedup

RETAB speeds up one step, `TOSABackend._preprocess`, which turns each
delegate's graph into TOSA. Vela and the rest of ExecuTorch's lowering are
unchanged, so they dominate once preprocess takes milliseconds.

> **Preliminary.** These numbers come from the benchmark gauntlet in RETAB's
> private lab repository, measured on 2026-11-06 while other heavy jobs shared
> the machine. An idle-machine rerun of the full gauntlet will replace this
> table.

| Model | `_preprocess`, stock | `_preprocess`, RETAB | Lowering, stock | Lowering, RETAB | `.pte` vs stock |
| --- | ---: | ---: | ---: | ---: | --- |
| resnet18 | 3.0-3.1 s | 20 ms | 6.5-6.7 s | 3.5 s | identical |
| mobilenet_v3_small | 5.4-5.9 s | 27 ms | 10.4-14.5 s | 5.9 s | identical |
| convnext_tiny | 29.5-50.0 s | 129 ms | 48.0-67.8 s | 17.4 s | 16 bytes smaller |

That is roughly 150-390× faster preprocess and 1.7-3.9× faster lowering.
"Lowering" is the whole `to_edge_transform_and_lower` call: partitioning,
`_preprocess` and Vela. Each model is a torchvision network with random
weights and a 224×224 input (232×232 for mobilenet_v3_small), INT8-quantized
for Ethos-U55-128; each lowers to a single delegate. Times are medians of 3
runs after a warmup, on macOS on Apple silicon. Stock's ranges span two runs
that day. RETAB's times come from one run of commit f1f7323, which v0.4.0
contains; its corpus output is byte-identical to v0.4.1's.

## Pinned companion stack

The install line above is the tested Python stack:

| Component | Version |
| --- | --- |
| ExecuTorch | Wheel `1.6.0`, upstream tag `v1.6.0-rc4`, commit `4776f71b6` |
| torch | `2.15.0` |
| torchvision | `0.29.2` |
| torchao | `0.18.0` (the tested environment uses `0.18.0+cpu`) |
| ethos-u-vela | `5.2.0` from PyPI |
| tosa-tools | `2026.5.0` |
| flatbuffers (Python) | `25.2.10` |

It is tested on macOS on Apple silicon with Python 3.12; the package needs
Python 3.10 or later.

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
for fallback and restoration. RETAB's decomposition fast paths are disabled on
rc4 (see the [Python API](#python-api)).

RETAB handles TOSA 1.0 INT graphs with the target's default pass pipeline.
It hands these cases to stock preprocessing:

- float profiles;
- TOSA versions whose major/minor version is not 1.0;
- TOSA development mode;
- custom pass pipeline configurations;
- registered pass insertions;
- delegate-owned mutable buffers;
- symbolic tensor shapes;
- graphs a Rust pass hands back: inputs on which stock raises its own error,
  or that RETAB can't lower;
- graphs on which RETAB's Rust code panics, which is a RETAB bug. The
  fallback reason is `Rust panic`, and a warning gives the panic's message
  and location. Set `RETAB_DEBUG` to raise the panic instead.

## Development

The first `cargo build` needs network access to fetch `tosa-rs` from GitHub;
run `cargo fetch` once to build offline afterwards.

### Build and test

From the checkout, after [Install](#install):

```bash
export RETAB_ROOT="$PWD"
export VENV="$RETAB_ROOT/.venv"
export PYO3_PYTHON="$VENV/bin/python"
cargo test --lib --locked
./scripts/build_retab.sh --venv "$VENV"
PYTHONPATH="$RETAB_ROOT/python" "$VENV/bin/python" -P -m pytest tests
```

RETAB ships no installer script. The build helper requires an existing
environment supplied with `--venv` or through an activated environment's
`VIRTUAL_ENV`. It fails if neither is provided and never creates an
environment implicitly.

Use `cargo test --lib --locked` for the Rust check and maturin to build the
extension. `cargo build --lib` alone does not link this `cdylib` correctly.

To produce a wheel:

```bash
"$VENV/bin/maturin" build --release --locked \
  --interpreter "$VENV/bin/python" --out "$RETAB_ROOT/target/wheels"
```

Maturin takes the package version from `Cargo.toml`.

`tests/` contains Python integration, seam, placeholder binding,
control-flow, and CNN lowering smoke tests. Tests requiring the native
extension, ExecuTorch's Arm backend, or the TOSA reference model skip when
those components are unavailable.

### Smoke lower

```bash
./scripts/smoke_checkout_build_and_lower.sh --venv "$VENV"
"$VENV/bin/python" -P examples/lower_with_retab.py --out tmp/tiny_cnn.pte
```

Both lower the example's tiny CNN with an existing build; the script also
checks that it wrote a non-empty `tmp/tiny_cnn.pte`.

## Python API

[python/retab/integration/executorch_arm.py](python/retab/integration/executorch_arm.py)
is the stable surface: `activated()`, and `install()` and `uninstall()` for
the same integration without a context manager. Nested activation restores the
stock backend when the outermost activation exits.

By default, activation replaces only `TOSABackend._preprocess` and marks
Ethos-U compiles for RETAB's unused-input check. RETAB's edge-manager and
`run_decompositions` fast paths, which sped up the rest of
`to_edge_transform_and_lower` on earlier ExecuTorch versions, are disabled
until they are re-derived on v1.6.0-rc4: the edge-manager fast path skips
rc4's pre-decomposition transforms. Until then `activated(fast_paths=True)`
and `install(fast_paths=True)` raise `NotImplementedError`.

For fragmented graphs, an experimental multi-process backend preprocessing
mode is available through `activated(parallel_backend_preprocess_workers=8)`
or `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS`. It is opt-in and returns to
serial execution for debug artifact dumps or when fork multiprocessing is
unavailable. Counters remain process-local.

[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) documents all environment reads.

## Validation

Benchmarks, the stock-versus-RETAB differential harness, and the model corpus
are kept separately in a private repository. Results for v0.4.2:

- **Op suite.** ExecuTorch v1.6.0-rc4's `backends/arm/test/ops -k tosa_INT`,
  with RETAB swapped in for `TOSABackend._preprocess`: 2,374 passed and none
  failed (48 expected failures and 1 skip, as marked upstream). 14 of the
  tests go to stock by design: 8 use a float profile and 6 have symbolic
  shapes.
- **Corpus.** 27 delegates from 23 models (small CNNs and op-pattern models,
  decomposed Elman, GRU and LSTM RNNs, resnet18 and mobilenet_v3_small),
  quantized for Ethos-U55-128. For every delegate, stock's and RETAB's TOSA
  give bit-identical outputs on the TOSA reference model for random inputs,
  and Vela compiles them to byte-identical command streams and weights. The
  TOSA files themselves still differ: every op count matches stock's except
  CONST, but the op order often does not.

Byte identity is checked on the corpus only. Other models can differ, as
convnext_tiny's `.pte` does in the [Speedup](#speedup) table.

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
│   └── _decomp_hooks.py         # Decomposition fast paths (disabled on rc4)
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
