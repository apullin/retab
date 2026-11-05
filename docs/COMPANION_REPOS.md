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

The tested interpreter is Python 3.12. The package's Python floor is 3.10;
its runtime uses 3.10 union type syntax and does not import `tomllib`.
ExecuTorch's Arm backend imports `torchao.prototype.mx_formats`, supplied by
the tested torchao version.

Stock `TOSABackend.preprocess` and `EthosUBackend.preprocess` both call
`TOSABackend._preprocess`, so the runtime replacement reaches TOSA generation
without changing the downstream Vela compilation step. RETAB keeps stock
preprocessing available for unsupported profiles and graph contracts; see
[the README scope list](../README.md#integration-and-scope).

The override setters listed as optional in the compatibility file are used
when a checkout exposes them. Missing fork hooks do not prevent installation.
`_decomp_hooks.py` also installs guarded performance patches against available
ExecuTorch and torch helpers.

## Rust TOSA dependency

[../Cargo.toml](../Cargo.toml) still declares:

```toml
tosa-rs = { path = "../vendor/tosa-rs" }
```

Supply that tree at `$RETAB_ROOT/../vendor/tosa-rs` before building. It is
outside this Git repository, so a RETAB clone alone does not provide the Rust
TOSA dependency. The checked-in manifest does not identify an immutable
`tosa-rs` revision; this document therefore does not assert a source commit.

The dependency supplies the TOSA schema types and generated API used by
RETAB. Its `build.rs` runs `flatc`, which must be version 25 or newer. On
macOS, put Homebrew first:

```bash
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
flatc --version
```

Conda's `flatc` 24 generates an incompatible size-field API. RETAB's build
script checks the supplied serializer source for the matching size-field
patch; see [VENDOR_PATCHES.md](VENDOR_PATCHES.md).

The Rust `flatbuffers` runtime is `24.12.23` in `Cargo.lock`. This is separate
from the Python `flatbuffers==25.2.10` package and the `flatc` >= 25 build tool.
Do not infer a Cargo runtime version from the Python stack table.

## Existing source-checkout helpers

The wheel-based setup in [../README.md](../README.md#quick-start) is the
current starting point. `scripts/bootstrap_uv.sh` and
`scripts/install_executorch.sh` retain the source-overlay workflow. They need
an existing environment and an explicitly supplied ExecuTorch source path
(`--source` or `EXECUTORCH_SRC`). The installer also expects its source
serializer tree and uses standard-library `tomllib`, requiring Python 3.11
or newer for that script.

## Updating the stack

Verify proposed versions and source revisions, rebuild with
`cargo test --lib --locked` and maturin, and run the repository's Python tests.
The crate is a `cdylib`; use maturin for extension linkage rather than
`cargo build --lib` alone. Maturin reads RETAB's version from `Cargo.toml`.

Benchmarks, the differential harness against stock ExecuTorch, and the model
corpus are kept separately in a private repository.
