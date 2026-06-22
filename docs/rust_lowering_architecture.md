# Rust Lowering Architecture

This document maps the Rust lowering engine in RETAB after the removal of
the Python FX round-trip. It is intended as a working guide for future agents
and maintainers who need to change the lowering path without rediscovering the
system from scratch.

The main invariant is:

```text
Python imports the ExportedProgram boundary.
Rust owns graph lowering, control-flow boundary handling, IR conversion,
constant/materialization effects, and TOSA serialization.
Python must not be used as an intermediate graph rewrite or re-ingest path.
```

## Repository Entry Points

The Rust crate lives at:

```text
/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
```

The ExecuTorch validation checkout lives at:

```text
/Users/andrewpullin/personal/executorch-work/executorch
```

The Python environment is:

```text
/Users/andrewpullin/personal/executorch-work/.venv
```

Use the uv-managed venv only. Do not use conda or pip directly.

```bash
cd /Users/andrewpullin/personal/executorch-work
source .venv/bin/activate
unset CONDA_PREFIX CONDA_DEFAULT_ENV
```

## High-Level Pipeline

The current path is:

```text
ExecuTorch ExportedProgram
  -> python/retab/preprocess.py
  -> PyO3 entrypoints in src/lib.rs
  -> Rust IrGraph rewrite pipeline
  -> stashed parent/submodule IR hierarchy
  -> control-flow boundary contracts on IR
  -> ingest/from_ir.rs converts IrGraph to serialize Graph
  -> lowering effects are applied to Graph and TensorStore
  -> serialize/* emits TOSA flatbuffer bytes
```

There is no intended "flush Rust IR back to Python FX, then re-ingest Python
FX into Rust" path. Any future code that tries to revive that pattern should
be treated as architecture debt unless there is a deliberate design decision
to reintroduce it.

## Python Seam

Primary file:

```text
python/retab/preprocess.py
```

This file owns the ExecuTorch backend integration seam. It calls Rust through
the `_native` PyO3 module. Python remains legitimate here because the input is
an ExecuTorch/PyTorch object graph.

What belongs here:

- Backend integration glue.
- Environment setup and selection of Rust serializer.
- Calls into Rust entrypoints.
- Lightweight extraction that must happen at the Python API boundary.

What does not belong here:

- Graph rewrite logic.
- Quantization propagation.
- Control-flow boundary mutation.
- TOSA serialization.
- Python re-ingest of graphs already represented in Rust.

## Rust Orchestration

Primary file:

```text
src/lib.rs
```

This file exposes the PyO3 functions used by the Python seam and coordinates
the top-level lowering flow.

Important responsibilities:

- `run_fx_passes`: build and run the parent `IrGraph` pipeline.
- `run_fx_submodule_passes`: build and run while-body submodule IR pipelines.
- `run_fx_cond_passes`: build and run cond-branch submodule IR pipelines.
- `ingest_and_serialize_tosa_pybytes`: consume the stashed IR hierarchy and
  serialize it.
- `STASHED_IR`: thread-local storage of the parent IR, submodule IRs, and
  effect batches.

The central data structure is conceptually:

```text
StashedPipelineHierarchy
  parent: IrGraph + effect batch
  submodules: BTreeMap<region_name, IrGraph + effect batch>
```

Use deterministic map iteration for anything that can affect node creation,
node naming, graph order, or serialized output. Prefer `BTreeMap` or sorted
keys over `HashMap` iteration in those paths.

## Rewrite IR

Primary file:

```text
src/fx_ir.rs
```

This module owns the mutable rewrite IR.

Important concepts:

- `IrGraph`: Vec-backed graph used by Rust passes.
- `NodeId = usize`: index into the IR node vector.
- `IrNode`: raw op, name, target string, args, kwargs, metadata, liveness.
- `IrArg`: Rust representation of FX arguments.
- `NodeMeta`: tensor facts imported from FX metadata and mutated by passes.
- `IrPass` / `EffectfulIrPass`: pass traits for graph-only and graph-plus-effect passes.
- `run_hybrid_pipeline_ext_with_initial_ir`: pass runner for the main rewrite
  pipeline.

The rewrite IR still uses raw target strings heavily. The serializer side has
a better typed operation grammar through `OpTarget`, but most rewrite passes
still use string predicates such as `target.contains(...)`. The initial shared
rewrite target classifier lives in:

```text
src/fx_target.rs
```

That classifier is intentionally partial. New passes should prefer adding
target categories there over adding another local string helper, but existing
passes should be migrated in small validated slices.

Graph mutation is currently direct:

```text
add_node
copy_meta_from
set_val
anchor_after / anchor_to
replace_all_uses
replace_and_erase
erase
```

For emitted chains, use `IrGraph::rewrite_cursor(anchor)` or
`IrGraph::rewrite_cursor_after(anchor, previous)`. `IrRewriteCursor` owns the
"first node anchors to the rewrite site, later nodes anchor after the previous
emitted node" grammar and provides append operations for call functions,
view/permute/slice copies, dtype casts, arange, and full-int constants.

Keep one-off anchors explicit when they describe a specific producer or
control-flow boundary placement rather than a freshly emitted rewrite chain.

## Pass Pipelines

Primary file:

```text
src/fx_pipeline.rs
```

This module defines the ordered lowering pipelines. The individual passes live
mostly in files named:

```text
src/fx_*_ir.rs
```

Current pass families include:

- Decomposition passes.
- Quantization and Q/DQ folding passes.
- Rescale insertion passes.
- Layout and memory-format passes.
- Constant materialization and folding passes.
- Table op insertion passes.
- Control-flow submodule passes.

The current pipeline structure is operationally successful, but the pass files
do not yet share a strong local grammar. The largest files are catch-all
modules that mix several concepts. This makes bugs harder to localize and
encourages copying patterns instead of reusing helpers.

## Metadata And Tensor Facts

Metadata is mostly stored in `NodeMeta` in `src/fx_ir.rs`.

Common facts include:

- Tensor shape.
- Tensor dtype.
- Input qparams.
- Output qparams.
- TOSA dim order.
- Spatial rank.
- Layout-neutral markers.
- Table qparams and other extra metadata.

This data is one of the most fragile parts of the lowering system. Passes must
state whether metadata is copied, restamped, cleared, inferred, or preserved
through a synthetic op.

`TensorFacts` is the named tensor value-fact domain. `NodeMeta` owns private
tensor, quant, and layout storage, and production passes should use `NodeMeta`
and `TensorMetaStamp` helpers rather than reaching into the storage layout or
rebuilding shape/dtype/qparam policy by hand:

```text
NodeMeta::tensor(shape, dtype)
NodeMeta::cloned_with_same_dtype_or*
NodeMeta::cloned_with_dtype_of_or*
NodeMeta::tensor_stamp_with_dtype_of_or*
TensorMetaStamp::{preserve_qparams, drop_qparams, empty_qparam_maps, plain_tensor, fresh_tensor}
NodeMeta::{input_qparams, output_qparams, copy_qparams_from}
NodeMeta::{set_val_dtype_for_existing_shape, clear_tosa_dim_order_if_rank_mismatch}
```

Explicit semantic tensors should remain explicit. Examples include index
tensors, bool masks, quant cast targets, int32 accumulator work tensors, float
row/column work tensors, and dtype-valued op arguments.

Avoid adding new ad hoc `meta.extra` keys unless the key is documented and has
a clear owner.

## Lowering Effects

Primary file:

```text
src/lowering_effects.rs
```

Effects are collected while running IR passes and applied after converting
`IrGraph` to the serialize graph.

Examples:

- Delete constant placeholders.
- Materialize constant nodes.
- Materialize zero-input constant nodes.
- Fuse constant nodes.
- Insert table ops.
- Cast input buffers.

These are lowering effects, not Python FX round-trip effects. They are applied
to the converted serialize graph and tensor store.

Effect application happens in:

```text
src/ingest/from_ir.rs
```

## IR To Serialize Graph

Primary file:

```text
src/ingest/from_ir.rs
```

This module converts rewrite IR into the typed serializer graph.

Important responsibilities:

- `graph_from_ir`: convert an `IrGraph` into a `Graph` plus `TensorStore`.
- `graph_from_ir_preserve_placeholders`: variant used where placeholder
  identity must be preserved.
- `apply_effects`: apply collected lowering effects.
- `finalize_converted_graph`: remove or normalize graph details before
  serialization.
- `prefix_graph`: prefix submodule node names and tensor store keys with a
  control-flow region name.

This module is the boundary where raw rewrite IR becomes the serializer's
typed graph. It is also where many latent assumptions about constants,
placeholder classification, and graph topology converge.

Any refactor here must preserve:

- Topological validity.
- Runtime input vs constant input classification.
- TensorStore key consistency.
- Control-flow region names.
- Lifted tensor constants from both `ep.constants` and `ep.state_dict`.

## Serialize Graph

Primary modules:

```text
src/graph/*
src/serialize/*
```

The serialize graph uses a slotmap-backed `Graph`, stable `NodeId`s, graph
order, explicit inputs, output node, and a node-name map.

The serializer has a better op grammar than the rewrite IR:

```text
graph::ir::OpTarget
```

Visitors are registered by domain in:

```text
src/serialize/visitors/*
```

The visitor dispatch table maps `OpTarget` to emitter functions. This is a
good pattern and should be preserved. The pressure point is that
`src/serialize/mod.rs` still owns too many unrelated utilities:

- Flatbuffer region assembly.
- Tensor and shape helpers.
- Weight permutation.
- Attribute construction.
- Output layout handling.
- Debug emission.

Future serializer refactors should split utilities by ownership while keeping
the visitor dispatch model intact.

## Control Flow

Main files:

```text
src/lib.rs
src/fx_control_flow_metadata.rs
src/ingest/from_ir.rs
src/serialize/visitors/control_flow.rs
```

Control-flow lowering has these phases:

```text
import parent FX graph into Rust IR and snapshot pre-pass CF qparams
run parent IR passes and stash the parent IR
import each submodule into Rust IR
propagate parent qparams/layout facts into submodule placeholders before submodule passes
run submodule IR passes
stash submodule IRs by region name
apply IR boundary contracts before serialization
convert parent and submodules from IR
prefix submodule graphs and tensor store keys
emit TOSA control-flow regions
```

The important semantic distinction is:

```text
cond branch outputs must match the parent-observed boundary contract
while body loop-carried outputs must match corresponding loop-carried inputs
```

The control-flow code now avoids a separate Python structural mutation seam.
The remaining ownership issue is module shape: `src/lib.rs` should stay an
orchestrator, while domain logic belongs in `src/control_flow.rs` or a future
`src/control_flow/*` module split. `src/fx_control_flow_metadata.rs` should
remain limited to IR metadata helpers and IR submodule passes.

## Current Pain Points

The audit found these repeated patterns and clashing abstractions:

- Rewrite passes use raw target strings while the serializer uses typed
  `OpTarget`.
- Metadata operations are repeated and implicit instead of expressed through a
  common tensor-facts API.
- Graph mutation patterns are repeated across many passes.
- Control-flow ownership is improved but still broad enough to justify a future
  Rust-only module split.
- Lowering effects now use current IR-native naming.
- Serializer utilities are concentrated in a large central module.
- Debug/config env parsing is scattered.
- Generated artifacts may still contain old linker symbol names, but active
  source no longer uses the old flush/write-back/re-ingest architecture.

## Where To Put New Code

Use these placement rules:

- New PyO3 boundary logic belongs in `src/lib.rs` only if it is orchestration.
- New graph rewrite logic belongs in an `fx_*_ir.rs` pass or a future pass
  submodule.
- New common rewrite operations belong in `src/fx_ir.rs` or a dedicated
  builder module.
- New target classification belongs near the IR/operation grammar, not inside
  one pass.
- New metadata semantics should be shared through `NodeMeta` helpers or a
  future tensor-facts abstraction.
- New post-conversion graph effects belong in the lowering-effect model and
  `src/ingest/from_ir.rs`.
- New TOSA emission belongs in `src/serialize/visitors/*` unless it is truly
  flatbuffer infrastructure.

## Validation Lanes

Fast Rust compile check:

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
cargo check --lib
```

Install Python extension:

```bash
cd /Users/andrewpullin/personal/executorch-work
source .venv/bin/activate
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd u55-pipeline/arm-tosa-engine
maturin develop --release
```

The `maturin` command must print that `retab-0.1.0` was installed. A silent
failure can leave a stale binary in the venv.

Control-flow validation:

```bash
cd /Users/andrewpullin/personal/executorch-work/executorch
source ../.venv/bin/activate
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING=1
export RETAB_ENABLE_PATCH=1
export RETAB_ROOT=/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
export PYTHONPATH=$PWD/src
python -m pytest -q backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py
```

Run control-flow tests twice when touching control-flow, metadata propagation,
submodule graph assembly, map iteration, or node creation order.

Full ops and models validation:

```bash
cd /Users/andrewpullin/personal/executorch-work/executorch
source ../.venv/bin/activate
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING=1
export RETAB_ENABLE_PATCH=1
export RETAB_ROOT=/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
export PYTHONPATH=$PWD/src

python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models \
  -k 'tosa and not FP and (INT or 16a8w or a16w8 or quantized or qat)'

python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models \
  -k 'tosa and FP'

python -m pytest -q --numprocesses=auto backends/arm/test/models -k tosa
```

Current known parity failures should be checked against the latest baseline
before being treated as acceptable. At the time this document was written, the
known non-regression failures were while-U85 INT cases, a8w4 conv/linear
coverage, debug feature cases, and erfinv at positive/negative infinity.
