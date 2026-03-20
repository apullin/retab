# Ticket 1C-impl: TOSA Flatbuffer Codegen + Serialization Skeleton

## 1. flatc Codegen Status

**flatc was not used.** The `flatc` binary was available but could not be executed in the current environment (permission restrictions on arbitrary binary execution). Per the ticket's fallback plan, manual Rust struct definitions were created for all TOSA types needed by Phase 1.

The manual types faithfully mirror the TOSA 1.0 schema (`tosa.fbs`), including:
- All enum discriminant values matching the schema's `uint32` numbering
- All attribute table fields with correct types
- The full structural hierarchy: `TosaGraph > TosaRegion > TosaBasicBlock > TosaOperator / TosaTensor / TosaShape`

## 2. Files Created / Modified

### New files
| File | Purpose |
|------|---------|
| `src/serialize/mod.rs` | Module root; exposes `serialize_to_tosa()` public API |
| `src/serialize/tosa_generated/mod.rs` | Manual Rust type definitions for TOSA flatbuffer schema |
| `src/serialize/visitors.rs` | Per-op emission functions + dispatch |

### Modified files
| File | Change |
|------|--------|
| `Cargo.toml` | Added `flatbuffers = "24.3"` dependency |
| `src/lib.rs` | Added `mod serialize;` |

## 3. How Serialization Will Work

The `serialize_to_tosa()` function follows a **bottom-up builder pattern**:

1. **Topological walk** — `graph.topological_sort()` gives a deterministic node ordering that respects data dependencies.

2. **Per-node emission** — For each node in sorted order, `visitors::emit_node()` dispatches to the correct op-specific visitor (e.g., `emit_conv2d`, `emit_const`). Each visitor returns an `EmitResult` containing zero or more `TosaOperator`s, `TosaTensor`s, and `TosaShape`s.

3. **Block assembly** — All emitted operators and tensors are collected into a single `TosaBasicBlock` named `"main"`. Input/output tensor names are set from the graph's placeholder and output nodes.

4. **Graph wrapping** — The block goes into a `TosaRegion`, which goes into a `TosaGraph` along with `Version` (1.0.0) and `SoftwareVersion` metadata.

5. **Flatbuffer encoding** — The `flatbuffers` crate's `FlatBufferBuilder` will serialize the `TosaGraph` tree bottom-up (tensors first, then operators, then blocks, then regions, then root table) with the `"TOSA"` file identifier. This step is currently `todo!()`.

Key schema constraints respected in the type design:
- Operators reference tensors by **string name**, not index
- Every operator carries an **attribute** (even if empty, e.g., `AddAttribute`)
- CONST ops produce both a tensor entry (with inline data) and an operator entry
- Tensor data is little-endian bytes stored inline in the `data` field

## 4. cargo check Results

```
cargo check — PASS
```

69 warnings, all of which are expected for a skeleton module:
- Unused imports/variables in the `todo!()` visitor functions
- Dead code warnings for types that are defined but not yet constructed at call sites

No errors. The type structure is correct and the module integrates cleanly with the existing crate.
