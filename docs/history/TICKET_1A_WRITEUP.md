# Ticket 1A: Crate Scaffolding + Graph IR

## Files Created

| File | Purpose |
|------|---------|
| `Cargo.toml` | Rust crate configuration with pyo3, smallvec, and slotmap dependencies |
| `pyproject.toml` | Maturin build config for building the Python extension module |
| `src/lib.rs` | PyO3 module entry point; declares `graph`, `store`, and `compile_spec` modules |
| `src/graph/mod.rs` | `Graph` struct definition with SlotMap arena, insertion-order tracking, input/output node tracking |
| `src/graph/ir.rs` | All IR type definitions: `OpType`, `OpTarget` (~50 variants), `DType`, `TensorInfo`, `QuantArgs`, `Arg`, `NodeMeta`, `Node` |
| `src/graph/builder.rs` | Graph mutation: `create_node`, `erase_node`, `replace_all_uses`, `bypass_node`, `insert_before`, `insert_after`, `nodes_by_target`, `all_node_ids`, `iter_nodes` |
| `src/graph/topo.rs` | Topological sort using Kahn's algorithm with fallback for disconnected nodes |
| `src/store.rs` | `TensorStore` and `TensorData` for named tensor buffer storage |
| `src/compile_spec.rs` | `CompileSpec` struct with target, system_config, memory_mode, extra_flags |

## Key Design Decisions

1. **TensorInfo uses a fixed-size `[i64; 4]` array instead of `SmallVec`** to enable `Copy`. A `shape_len: u8` field tracks the actual dimensionality. This keeps the struct on the stack and copyable as required by the spec. The `shape_slice()` method provides a clean view of the actual dimensions.

2. **SmallVec v1.13 instead of v2** -- SmallVec 2.0 is still in alpha (2.0.0-alpha.12) and not available as a stable release. SmallVec 1.x provides the same const-generic inline buffer support needed here.

3. **User tracking on nodes** -- Each `Node` has a `users: SmallVec<[NodeId; 4]>` field that is automatically maintained by all builder methods (`create_node`, `erase_node`, `replace_all_uses`, etc.). This makes rewiring operations efficient.

4. **Insertion order is explicit** -- The `Graph.order: Vec<NodeId>` vector tracks insertion order separately from the SlotMap, which does not guarantee iteration order. All query methods (`all_node_ids`, `iter_nodes`, `nodes_by_target`) respect this order.

5. **OpTarget includes additional ATen ops** beyond the spec minimum (Sigmoid, Tanh, Mean, Cat, Slice, etc.) to cover common model patterns without relying on `Other(String)`.

6. **`Arg::node_refs()`** helper method centralizes the logic for extracting NodeId references from any Arg variant, used by both builder and topo modules.

## Deviations from Spec

- **SmallVec version**: Used `smallvec = "1.13"` instead of `"2"` since v2 is alpha-only. No functional difference for our usage.
- **TensorInfo shape**: Used a fixed `[i64; 4]` array + length instead of `SmallVec<[i64; 4]>` to satisfy the `Copy` requirement. SmallVec is not Copy.
- **Additional OpTarget variants**: Added ~15 extra ATen variants (Sigmoid, Tanh, Cat, Select, View, etc.) beyond the explicitly listed ones for broader model coverage.
- **Builder methods take `name: String`** as a parameter rather than auto-generating names, giving callers control over naming.

## cargo check Output

```
warning: `arm_tosa_engine` (lib) generated 20 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s
```

All 20 warnings are "never used" diagnostics, which is expected since the types are not yet referenced from the PyO3 module entry point. No errors.
