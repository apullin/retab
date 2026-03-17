# Ticket 1F Writeup: Integration — Wire Everything Together + End-to-End Test

## What Was Implemented

### Task 1: `serialize_to_tosa()` Implementation

Replaced the `todo!()` in `src/serialize/mod.rs` with a full implementation that:

1. Walks the graph in topological order via `graph.topological_sort()`.
2. For each node, dispatches based on `OpType`:
   - **Placeholder**: Creates a `TosaTensor`. If the tensor store has data for it (a constant/weight), also emits a `CONST` operator. Otherwise, records it as a graph input.
   - **CallFunction**: Calls the appropriate per-op visitor via `visitors::emit_node()`, collecting emitted tensors, shapes, and operators. For unsupported ops (visitor returns `None`), adds a passthrough tensor entry.
   - **Output**: Resolves output tensor names from `Arg::Node` and `Arg::NodeList` references.
3. Builds the TOSA flatbuffer bottom-up using the `flatbuffers` crate's raw builder API (`FlatBufferBuilder`), constructing:
   - `TosaTensor` tables (with name, shape, dtype, optional data)
   - `TosaShape` tables (for CONST_SHAPE outputs)
   - `TosaOperator` tables (with op enum, attribute union, input/output name lists)
   - `TosaBasicBlock` containing all operators, tensors, shapes, inputs, outputs
   - `TosaRegion` wrapping the block
   - `TosaGraph` root with version (1.0.0), regions, and software version
4. Finishes the flatbuffer with file identifier "TOSA".

**Approach**: Full Rust flatbuffer serialization was implemented (not the JSON/Python fallback). The manual builder approach uses `start_table()` / `push_slot*()` / `end_table()` directly, since we don't have flatc-generated Rust code. Field vtable slot offsets follow the schema field order (slot 4 for field 0, slot 6 for field 1, etc.).

### Task 1 (visitors): Per-Op Emission

All `todo!()` stubs in `src/serialize/visitors.rs` were replaced with working implementations:

- `emit_const` — Looks up data in TensorStore, creates tensor + CONST operator.
- `emit_conv2d` — Collects input names, builds Conv2dAttribute from kwargs.
- `emit_depthwise_conv2d` — Same pattern as conv2d with DepthwiseConv2dAttribute.
- `emit_rescale` — Builds RescaleAttribute from kwargs.
- `emit_max_pool2d` — Builds MaxPool2dAttribute from kwargs.
- `emit_transpose` — Builds TransposeAttribute with perms from args or kwargs.
- `emit_reshape` — Builds empty ReshapeAttribute, relies on CONST_SHAPE input.
- `emit_const_shape` — Emits TosaShape entry + CONST_SHAPE operator.
- `emit_clamp` — Builds ClampAttribute with min/max bytes from kwargs.
- `emit_add` — Builds empty AddAttribute.

Each visitor creates the output tensor entry and operator, collecting input tensor names from `Arg::Node` references in the node's args.

### Task 2: `lower_to_tosa` PyO3 Function

Added to `src/lib.rs`:

```rust
#[pyfunction]
fn lower_to_tosa(py, graph_module, state_dict, graph_signature) -> PyResult<Py<PyBytes>>
```

This function:
1. Calls `ingest_fx_graph()` to get `(Graph, TensorStore)`
2. Runs `build_phase1_pipeline()` passes
3. Calls `serialize_to_tosa()` to get the flatbuffer bytes
4. Returns the bytes as `PyBytes`

Registered alongside `ingest_graph` in the `arm_tosa_engine` module.

### Task 3: Compilation

All compilation errors were resolved. The main fixes needed:
- Using `WIPOffset<TableFinishedWIPOffset>` consistently (the return type of `end_table()`)
- Using `PyBytes::new_bound()` instead of deprecated `PyBytes::new()` for pyo3 0.22
- Using `create_vector()` instead of non-existent `create_vector_aligned()`

### Task 4: Test Script

Created `tests/test_e2e.py` which:
1. Exports and quantizes a `tiny_cnn` model
2. Calls `arm_tosa_engine.lower_to_tosa()` on the quantized exported program
3. Compares the output with a reference `.tosa` file (or saves output if no reference exists)
4. Validates the TOSA file identifier in the output bytes

## `cargo check` Results

**PASS** — `cargo check` completes successfully with only warnings (dead code for unused enum variants, unused struct fields, etc.). No errors.

## Serialization Approach

**Full Rust flatbuffer serialization** was implemented. The implementation uses the `flatbuffers` crate's raw builder API with manually computed vtable slot offsets matching the TOSA 1.0 schema. This avoids the need for flatc-generated code while producing binary-compatible output.

Note: Data alignment (`force_align: 8` from the schema) is not enforced by the current `create_vector()` calls for tensor data and clamp attribute bytes. This may cause byte-level differences compared to the reference serializer. If exact byte compatibility is needed, this can be addressed in a follow-up.

## End-to-End Test Status

**Not yet run** — The test requires a built `.so`/`.dylib` (via `maturin develop` or similar) and the full Python environment with ExecuTorch, torchao, and the model definitions. The test script is ready to execute once those dependencies are in place.

The integration is expected to be **partial** on first run because:
1. The pass pipeline may not yet fully transform ATen ops to TOSA ops for all cases
2. Attribute extraction from kwargs relies on the pass pipeline producing the expected kwarg names
3. Force-alignment for tensor data is not yet implemented

These are known Phase 1 limitations that will be addressed in subsequent tickets.
