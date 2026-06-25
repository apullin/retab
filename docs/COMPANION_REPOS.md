# Companion Repositories

RETAB is the Rust implementation of ExecuTorch's ARM backend preprocess stage,
but it is not a standalone upstream-only checkout yet. A working tree needs a
small set of concordant commits in companion repositories.

## Required

### RETAB

- Fork: <https://github.com/apullin/retab>
- Branch: `main`
- Known-good baseline: `2aa0a7dff`
- Local path: `u55-pipeline/arm-tosa-engine`
- Role: PyO3 Rust extension plus the thin Python integration layer that
  replaces `TOSABackend._preprocess`.

This repository owns the FX-to-Rust-IR lowering passes, Rust IR rewrites,
control-flow hierarchy handling, and TOSA flatbuffer emission.

### ExecuTorch

- Fork: <https://github.com/apullin/executorch>
- Branch: `et-python-fastpaths`
- Known-good commit: `dac215665`
- Local path: `executorch`
- Upstream remote: <https://github.com/pytorch/executorch>

RETAB needs a seam-enabled ExecuTorch checkout. The required public hook points
are recorded in [../compat/executorch.toml](../compat/executorch.toml):

- `executorch.backends.arm.tosa.backend.TOSABackend.set_preprocess_override`
- `executorch.exir.program.set_edge_manager_for_partitioners_override`

The branch also carries the local direct-lowering ARM test lane and current
toolchain expectation fixes used by RETAB validation. Without these commits,
RETAB can still build as a Rust extension, but it cannot be installed as a
drop-in ExecuTorch ARM backend fast path.

### `tosa-rs`

- Fork: <https://github.com/apullin/tosa-rs>
- Branch: `master`
- Known-good commit: `de6e6b0`
- Source checkout: `../rust-tosa` in this workspace, symlinked to
  `/Users/andrewpullin/personal/rust-tosa`
- Vendored copy used by Cargo: `u55-pipeline/vendor/tosa-rs`

RETAB depends on `tosa-rs` through this local Cargo path:

```toml
tosa-rs = { path = "../vendor/tosa-rs" }
```

The crate provides the Rust TOSA FlatBuffers serialization/deserialization API
used by RETAB's serializer. The fork currently includes the TOSA 1.0 update,
builder API, serialization/deserialization layer, `union_type()` support,
`MulAttribute.shift`, and current WHILE_LOOP field names.

Because `u55-pipeline/vendor/tosa-rs` is outside the RETAB Git repository, keep
it synchronized deliberately when updating `../rust-tosa`. See
[VENDOR_PATCHES.md](VENDOR_PATCHES.md) for the local guard around generated
FlatBuffers `size` field compatibility.

## Optional For Full Local Toolchain

### Ethos-U Vela

- Fork: <https://github.com/apullin/ethos-u-vela>
- Branch: `retab-regor-opt-5.1.0-curated`
- Known-good commit: `d45b6236`
- Local path: `ethos-u-vela-5.1.0`
- Upstream remote:
  <https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela>

RETAB emits TOSA. Vela consumes that TOSA downstream when building an ARM PTE.
The curated branch is not required for `cargo check` or pure serializer tests,
but it is the local branch used for current end-to-end Vela/Regor optimization
experiments.

### TOSA Tools

- Upstream: <https://gitlab.arm.com/tosa/tosa-tools>
- Local path: `tosa-tools`

TOSA tools are optional for semantic reference-model comparison. They are not a
RETAB build dependency. The old local `macos-pybind-dynamic-lookup` branch was
an investigation against an outdated checkout; current TOSA tools already carry
the relevant macOS pybind fix upstream.

## Compatibility Rule

Treat these as one pinned toolchain, not independent rolling dependencies:

1. Update ExecuTorch, `tosa-rs`, Vela, or TOSA tools in isolation only on a
   temporary branch.
2. Rebuild RETAB with `cargo check --lib` and `maturin develop --release`.
3. Run the ARM ops and models regression lanes from
   [VALIDATION_MATRIX.md](VALIDATION_MATRIX.md).
4. Only then advance the documented branch/commit pointers here.

The key invariant is that the Python seam, Rust IR lowering, Rust TOSA
serializer, and downstream Vela consumer agree on the same TOSA schema and ARM
backend contracts.
