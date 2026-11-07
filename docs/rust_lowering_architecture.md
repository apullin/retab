# Rust Lowering Architecture

RETAB implements the Arm TOSA preprocessing stage for stock ExecuTorch
v1.6.0-rc4 (wheel `1.6.0`, commit `4776f71b6`), focused on Ethos-U55 INT
lowering. The tested stack uses torch `2.15.0`, torchao `0.18.0`, PyPI
ethos-u-vela `5.2.0`, tosa-tools `2026.5.0`, and Python flatbuffers `25.2.10`.
See [COMPANION_REPOS.md](COMPANION_REPOS.md) for build dependencies and
[../compat/executorch.toml](../compat/executorch.toml) for stock compatibility.

The central invariant is that Python imports the ExportedProgram boundary,
then Rust owns graph rewriting, control-flow boundary contracts, conversion
to the serializer graph, constant materialization, and TOSA serialization.
The lowered graph is retained as Rust IR rather than written back to Python
FX and re-ingested.

## Pipeline

```text
ExecuTorch ExportedProgram
  -> python/retab/integration/executorch_arm.py: activated() / install()
  -> python/retab/preprocess.py: scope check and backend orchestration
  -> src/lib.rs: run_fx_passes()
  -> IrGraph import + ordered Rust pass pipeline
  -> thread-local parent IR + deferred effects
  -> run_fx_cond_passes() / run_fx_submodule_passes() for nested regions
  -> thread-local submodule IRs + deferred effects
  -> ingest_and_serialize(): consume stashed hierarchy
  -> control-flow boundary contracts
  -> ingest/from_ir.rs: IrGraph -> Graph + TensorStore, apply effects
  -> ingest/finalize.rs: finalize converted graph
  -> control_flow/assembly.rs: assemble and prefix child regions
  -> serialize/collection.rs: collect region tensors and operators
  -> serialize/flatbuffer.rs: TOSA 1.0 flatbuffer bytes
  -> ExecuTorch's downstream Vela compilation
```

## Python integration and stock fallback

[../python/retab/integration/executorch_arm.py](../python/retab/integration/executorch_arm.py)
provides the stable, nestable `install()`, `uninstall()`, and `activated()`
API. The outermost activation installs the hooks; the final uninstall
restores the captured originals.

[../python/retab/preprocess.py](../python/retab/preprocess.py) captures stock
`TOSABackend` attributes before replacing `_preprocess`. Both stock TOSA and
Ethos-U backends call that entry point. Optional fork override setters are
supported when present, but `compat/executorch.toml` uses `mode = "stock"`.
Activation also installs guarded decomposition fast paths from
[../python/retab/_decomp_hooks.py](../python/retab/_decomp_hooks.py) against
available upstream helpers.

`stock_fallback_reason()` sends float profiles, TOSA major/minor versions
other than 1.0, development mode, custom pass pipelines, registered pass
insertions, delegate-owned mutable buffers, and symbolic shapes to the
captured stock implementation. `preprocess_stats()` returns process-local
RETAB counts and stock fallback counts by reason.

For an in-scope graph, Python copies the original state dictionary, calls
the Rust parent and submodule pass entry points, aligns graph-signature
input specs with placeholder order, and invokes `ingest_and_serialize()`.
It also handles requested artifact dumps and delegate metadata through
`_cf_ctx.py`. Serialization consumes the stashed IR; its compatibility
arguments do not cause a second Python graph import.

Experimental parallel backend preprocessing patches `preprocess_multimethod`
only when workers are explicitly enabled. It uses fork workers for multiple
delegates and falls back to serial execution for artifact dumps or an
unavailable fork start method. Counts remain local to each process.

## Rust entry points and hierarchy

[../src/lib.rs](../src/lib.rs) exposes the PyO3 functions used by Python:

| Entry point | Responsibility |
| --- | --- |
| `run_fx_passes` | Import the parent graph, snapshot control-flow qparams, run the parent pipeline, and stash IR plus effects. |
| `run_fx_cond_passes` | Import a cond branch, propagate parent boundary metadata, run its pipeline, and stash the named region. |
| `run_fx_submodule_passes` | Import a while submodule, propagate boundary metadata, run its pipeline, and stash the named region. |
| `ingest_and_serialize` | Consume the hierarchy, apply boundary contracts, convert graphs, apply effects, and return TOSA bytes. |

`StashedPipelineHierarchy` contains an optional parent pipeline and a
`BTreeMap` of named submodule pipelines. `STASHED_IR` is thread-local and is
consumed during serialization. Keep the pass calls and serialization on the
same thread. Running the parent entry point starts a new hierarchy and
clears previously stashed submodules.

Use deterministic ordering wherever map traversal can affect region names,
node creation, graph order, or serialized output.

## Rewrite IR and passes

[../src/fx_ir.rs](../src/fx_ir.rs) defines the Vec-backed `IrGraph`, index
`NodeId`, `IrNode`, and `IrArg`. Nodes hold their operation, target string,
arguments, kwargs, metadata, liveness, and imported tensor data. FX boundary
reads use helpers from `fx_py_attr.rs`.

`IrPass` mutates the graph directly. `EffectfulIrPass` returns graph changes
plus deferred lowering effects. `PipelineStep` combines these pass kinds,
and `run_hybrid_pipeline_ext_with_initial_ir` runs them over one persistent
IR. [../src/fx_pipeline.rs](../src/fx_pipeline.rs) defines the ordered parent,
cond, and while pipelines; implementations live in `src/fx_*_ir.rs` and
related domain modules.

`IrRewriteCursor` expresses anchored chains of inserted operations.
`IrGraph` also provides explicit node insertion, anchoring, use replacement,
and erasure methods. Rewrite passes retain raw target strings, with shared
classification in `fx_target.rs`; the later serializer graph has typed
`OpTarget` values.

Linear follows rc4's DecomposeLinearPass: the rows go in the width of a 1x1
convolution (`view [1, 1, rows, Ci]`, `permute(0, 3, 1, 2)`, conv,
`permute(0, 2, 3, 1)`, `view`), so Vela never sees a batch of rows.

## Layout

RETAB follows rc4's layout model (Linear APU-135). A node's value is its TOSA
shape, and every layout change is an explicit `permute_copy` node, which the
serializer lowers one-to-one to TRANSPOSE. The pipeline runs rc4's layout
passes at their ArmPassManager positions, which
[../src/fx_pipeline.rs](../src/fx_pipeline.rs) cites as `APM:<line>`:

- `IrNormalizeDelegateIOLayoutPass` makes the delegate's inputs and outputs
  contiguous.
- The rewrites in `src/layout/rewrite_*.rs` turn convolutions, pools and
  upsampling into channels-last TOSA ops between NHWC and NCHW permutes, with
  pre-permuted weights.
- The propagation and fusion passes in `src/layout/` move permutes and views
  through the graph and cancel or fuse them, as rc4's do.
- On Ethos-U55, `IrDecomposePermuteForU55Pass` splits permutes that Vela
  can't run. It checks each live permute that breaks U55's worst-case rule
  with rc4's one-TRANSPOSE Vela probe, called back into Python.

`src/layout/replay.rs` checks the ports against rc4 itself: it replays rc4's
Python passes on captured graphs in `tests/fixtures/layout_replay/` and
compares the results.

RETAB's earlier model, which recorded a channels-last `tosa_dim_order` in node
metadata for the serializer to apply, is deleted (APU-162): ingest and the
serializer take each node's layout from its val.

## Metadata and constants

[../src/fx_meta.rs](../src/fx_meta.rs) owns `TensorFacts`, `NodeMeta`,
`TensorMetaStamp`, and `QParamMap`. `fx_ir.rs` re-exports the metadata types
used by passes. Tensor shape, dtype and torch dim order (`val_dim_order`),
input/output quantization, and passthrough backend keys belong to this domain.

Passes must deliberately preserve, restamp, or clear tensor (including its dim
order) and quantization facts when replacing nodes. Shared metadata helpers and domain
modules such as `fx_quant.rs`, `fx_reduction.rs`, and `fx_index_domain.rs`
keep these semantics consistent.

Imported `NodeTensor` data carries constants into the rewrite IR.
[../src/lowering_effects.rs](../src/lowering_effects.rs) records deferred
constant deletion, tensor materialization, constant fusion, table insertion,
and input-buffer casts. `ingest/from_ir.rs` applies these effects to the
converted graph and [../src/store.rs](../src/store.rs) `TensorStore`, without
rewriting Python FX.

## Conversion to the serializer graph

[../src/ingest/from_ir.rs](../src/ingest/from_ir.rs) maps live IR nodes and
arguments to the slotmap-backed `Graph` in `src/graph/`. It copies constant
bytes from the IR into `TensorStore`, classifies inputs, and parses raw
operation targets through `ingest/target_map.rs`.

`graph_from_ir_preserve_placeholders()` keeps unused runtime placeholders
where a control-flow region's block signature requires them.
`apply_effects()` and its placeholder-preserving variant apply the deferred
effect batches. `ingest/finalize.rs` owns `finalize_converted_graph()`.
`suffix_graph()` gives a region's nodes and tensor-store keys stock's
`<name>_<region>` names, and qualifies the region names its COND_IF and
WHILE_LOOP ops reference as `<region>__<child>`.

Conversion must preserve graph topology, runtime-versus-constant input
classification, tensor-store keys, region names, and lifted constants from
both the ExportedProgram's constants and state dictionary.

## Control flow

[../src/control_flow.rs](../src/control_flow.rs) is the control-flow module
entry point. Its implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `control_flow/metadata.rs` | Pre-pass qparam snapshots, parent-to-child propagation, and submodule metadata passes. |
| `control_flow/contracts.rs` | Parent/child boundary contracts and carried-output compatibility. |
| `control_flow/assembly.rs` | Conversion, naming, and assembly of nested serializer regions. |
| `control_flow/captured_conv.rs` | Conv weights and biases a cond branch captures from an enclosing graph, rewritten as branch-local constants (rc4's `_rewrite_weight` and `_rewrite_control_flow_bias`). |
| `serialize/visitors/control_flow.rs` | Emit TOSA control-flow operations and region references. |

Cond outputs follow the parent-observed boundary contract. While
loop-carried outputs follow the corresponding loop-carried input contract.
These contracts are applied to Rust IR before region conversion. Regions are
emitted in stock's pre-order: cond branches as `[1, 2]`, while loops as
`[0, 1]`, each child before the next sibling.

## TOSA serialization

[../src/serialize/mod.rs](../src/serialize/mod.rs) coordinates serialization
and clears per-serialization shape/dtype caches. The implementation is split
across `collection.rs`, `constant_pool.rs`, `tensor_layout.rs`,
`attribute_flatbuffer.rs`, and `flatbuffer.rs`.

`serialize/visitors/mod.rs` builds a domain-registered emitter dispatch table
keyed by `graph::ir::OpTarget`. Each visitor receives `EmitContext` and
produces an `EmitResult` containing the required tensors and operations.
Flatbuffer assembly writes the TOSA 1.0 graph and its named regions using
schema types exposed through `serialize/tosa_types/mod.rs` and the
external `tosa-rs` dependency.

The serializer rejects INT64 tensors reaching TOSA 1.0 emission. Layout,
rescale rounding, tensor naming, constant data, and region signatures are
part of the stock parity contract.

## Build and repository checks

Use `$RETAB_ROOT` for this checkout and `$VENV` for an existing environment:

```bash
cd "$RETAB_ROOT"
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON="$VENV/bin/python"
cargo test --lib --locked
"$VENV/bin/maturin" develop --release --locked
PYTHONPATH="$RETAB_ROOT/python" "$VENV/bin/python" -P -m pytest tests
```

This crate is a `cdylib`; `cargo build --lib`
alone does not supply the required extension linkage. The
`pyo3/extension-module` feature belongs only to maturin's feature list.
Maturin derives the Python package version from `Cargo.toml`.

Benchmarks, the stock-versus-RETAB differential harness, and the model corpus
are kept separately in a private repository.

[ENVIRONMENT.md](ENVIRONMENT.md) records environment variables and legacy
aliases. Earlier proposals are preserved in
[history/RUST_GRAPH_IR.md](history/RUST_GRAPH_IR.md) and
[history/refactor_backlog.md](history/refactor_backlog.md); they are historical
rather than a description of the current module layout.
