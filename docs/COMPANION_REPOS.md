# Companion Stack

RETAB works with stock ExecuTorch v1.6.0-rc4. It replaces
`TOSABackend._preprocess` at runtime; neither an ExecuTorch fork nor a custom
Vela branch is required. The compatibility record is
[../compat/executorch.toml](../compat/executorch.toml).

## Tested Python stack

| Component | Tested version | Source |
| --- | --- | --- |
| ExecuTorch | Wheel `1.6.0`; tag `v1.6.0-rc4`; commit `4776f71b6` | [pytorch/executorch](https://github.com/pytorch/executorch/tree/v1.6.0-rc4) |
| torch | `2.15.0` | PyPI |
| torchvision | `0.29.2` | PyPI |
| torchao | `0.18.0`; tested build `0.18.0+cpu` | Python package |
| ethos-u-vela | `5.2.0` | [PyPI](https://pypi.org/project/ethos-u-vela/5.2.0/) |
| tosa-tools | `2026.5.0` | [PyPI](https://pypi.org/project/tosa-tools/2026.5.0/) |
| flatbuffers (Python) | `25.2.10` | PyPI |

The tested interpreter is Python 3.12. The package's Python floor is 3.10,
which its union type syntax needs.
ExecuTorch's Arm backend imports `torchao.prototype.mx_formats`, supplied by
the tested torchao version.

Stock `TOSABackend.preprocess` and `EthosUBackend.preprocess` both call
`TOSABackend._preprocess`, so the runtime replacement reaches TOSA generation
without changing the downstream Vela compilation step. RETAB keeps stock
preprocessing available for unsupported profiles and graph contracts; see
[the README scope list](../README.md#integration-and-scope).

`_decomp_hooks.py` also installs guarded performance patches against available
ExecuTorch and torch helpers.

## Rust TOSA dependency

[../Cargo.toml](../Cargo.toml) pins [`tosa-rs`](https://github.com/apullin/tosa-rs)
by git revision (tag v0.2.0). It supplies the TOSA 1.0/1.1 schema types and
the generated FlatBuffers API, committed in that repository, so building RETAB
needs no `flatc`. Run `cargo fetch` once if later builds must work offline.

To edit RETAB and a local tosa-rs checkout together, override the dependency in
an untracked `.cargo/config.toml` (ignored by Git):

```toml
[patch."https://github.com/apullin/tosa-rs"]
tosa-rs = { path = "/path/to/tosa-rs" }
```

The Rust `flatbuffers` runtime is pinned to `25.2.10` in Cargo.toml, the same
version tosa-rs uses. This is separate from the Python `flatbuffers==25.2.10`
package; do not infer one from the other.

## Environment setup

The [README quick start](../README.md#quick-start) creates an environment with
uv and installs the tested stack from PyPI. That explicit install is the
supported setup: RETAB ships no installer script for the stack, and its build
helpers need an existing environment and never create one. RETAB only needs an
importable `executorch.backends.arm`, so an ExecuTorch built from a source
checkout also works when it is a tested version.

## Updating the stack

Verify proposed versions and source revisions, rebuild with
`cargo test --lib --locked` and maturin, and run the repository's Python tests.
The crate is a `cdylib`; use maturin for extension linkage rather than
`cargo build --lib` alone. Maturin reads RETAB's version from `Cargo.toml`.

Benchmarks, the differential harness against stock ExecuTorch, and the model
corpus are kept separately in a private repository.
