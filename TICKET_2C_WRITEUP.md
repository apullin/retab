# Ticket 2C: Additional TOSA Visitors + Shape Rules

## Summary

Added TOSA serialization visitors, attribute structs, and shape propagation rules
for all remaining ops in the cnn_layernorm4d reference output that were not yet
implemented.

## New Visitors (`src/serialize/visitors.rs`)

| Visitor            | TOSA Op      | Notes |
|--------------------|-------------|-------|
| `emit_sub`         | SUB (30)     | Binary elementwise; empty SubAttribute |
| `emit_mul`         | MUL (28)     | Creates inline const shift tensor (int8 scalar = 0); MulAttribute { shift: 0 } |
| `emit_avg_pool2d`  | AVG_POOL2D (2) | Creates inline const tensors for input_zp and output_zp; AvgPool2dAttribute with kernel/stride/pad/acc_type/zp fields |
| `emit_reduce_sum`  | REDUCE_SUM (54) | Single-axis reduction; ReduceSumAttribute { axis } |
| `emit_table`       | TABLE (31)   | Lookup table op; empty TableAttribute; table data is an input tensor |

All new visitors follow the same pattern as existing ones: collect input names,
build the output tensor, construct the operator with the correct attribute, and
return an EmitResult.

### Helper Added

- `kwarg_i32(node, name, default)` -- extract a single i32 kwarg by name

## New Attribute Structs (`src/serialize/tosa_generated/mod.rs`)

| Struct                | Fields |
|-----------------------|--------|
| `SubAttribute`        | (empty) |
| `MulAttribute`        | `shift: i32` |
| `AvgPool2dAttribute`  | `kernel`, `stride`, `pad`, `acc_type`, `input_zp`, `output_zp` |
| `ReduceSumAttribute`  | `axis: i32` |
| `TableAttribute`      | (empty) |

Each struct was added to the `Attribute` enum with a corresponding `union_type()`
value matching the TOSA flatbuffer schema position.

## Flatbuffer Serialization (`src/serialize/mod.rs`)

`build_fb_attribute` was extended with match arms for:
- `Sub`, `Table` -- empty tables (grouped with existing empty-table arm)
- `Mul` -- single i32 shift field
- `AvgPool2d` -- kernel/stride/pad vectors + acc_type/input_zp/output_zp scalars
- `ReduceSum` -- single i32 axis field

## Dispatch (`src/serialize/visitors.rs`)

`emit_node()` now routes:
- `TosaSub` -> `emit_sub`
- `TosaMul` -> `emit_mul`
- `TosaAvgPool2d` -> `emit_avg_pool2d`
- `TosaReduceSum` -> `emit_reduce_sum`
- `TosaTable` -> `emit_table`

## Shape Propagation (`src/shape/mod.rs`)

Added `TosaTable` to the element-wise passthrough group (output shape = input
shape). All other needed ops were already covered:

| Op            | Shape Rule | Status |
|---------------|-----------|--------|
| TosaAdd       | broadcast | was already present |
| TosaSub       | broadcast | was already present |
| TosaMul       | broadcast | was already present |
| TosaAvgPool2d | pool2d    | was already present |
| TosaReduceSum | reduction | was already present |
| TosaClamp     | passthrough | was already present |
| TosaTable     | passthrough | **added in this ticket** |

## Verification

- `cargo check` passes (warnings only, no errors)
- `cargo test --lib` passes all 18 tests
