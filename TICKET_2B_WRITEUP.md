# Ticket 2B: Scalar/Type Matching + Fusion Passes

## Summary

Implemented 7 new pass files and updated `mod.rs` with a full Phase 2 pipeline ordering (18 passes total).

## New Passes

### 1. `src/passes/broadcast_args.rs` — BroadcastArgsPass
Walks binary ops (Add, Sub, Mul, Div and their TOSA variants). When the two input tensors have different ranks, inserts a Reshape node before the lower-rank input to prepend leading 1-dims, making ranks match for TOSA compliance.

### 2. `src/passes/match_arg_dtype.rs` — MatchArgDtypePass
Walks binary ops and checks if both inputs share the same dtype. When they differ (e.g., Int8 vs Int32), inserts a TosaRescale node to promote the narrower type to the wider type. Uses a promotion hierarchy: Float32 > Int64 > Int32 > Int16.

### 3. `src/passes/fuse_constants.rs` — FuseConstantArgsPass
Finds CallFunction nodes where every Node-typed input is a constant Placeholder with data in TensorStore. Evaluates the operation in Rust and replaces the node with a pre-computed TosaConst placeholder.

Supported constant-foldable ops:
- **Add, Sub, Mul** — element-wise with broadcasting, supports Float32/Int32/Int64/Int8
- **Reshape** — just changes shape metadata, data unchanged
- **Permute** — reorders raw bytes based on permutation and strides
- **Full** — creates a filled tensor of the requested shape/dtype

Runs in a loop to handle cascading constant folds.

### 4. `src/passes/fuse_quantized_activation.rs` — FuseQuantizedActivationPass
When a Relu/Clamp/Hardtanh immediately follows a node with output_qparams, converts the floating-point clamp bounds into the quantized domain and tightens qmin/qmax on the producer's output_qparams. The activation node is then bypassed.

### 5. `src/passes/annotate_dim_order.rs` — AnnotateOutputDimOrderPass
Sets `tosa_dim_order` metadata on all nodes that lack it. Uses an identity permutation [0..rank). The memory_format pass later overrides this for NHWC ops.

### 6. `src/passes/cast_int.rs` — CastToInt32Pass + CastInt64BuffersToInt32Pass
- **CastToInt32Pass**: Walks Placeholder/TosaConst nodes with Int64 dtype, converts their tensor data in TensorStore from 8-byte i64 to 4-byte i32 (truncation), and updates the node's meta dtype.
- **CastInt64BuffersToInt32Pass**: Sweeps all TensorStore buffers with Int64 dtype and converts them, catching any stray buffers not referenced by graph nodes.

### 7. No-op Placeholder Passes (in `mod.rs`)
- **FuseDuplicateUsersPass** — placeholder, returns false
- **FuseEqualPlaceholdersPass** — placeholder, returns false
- **RemoveNoopPass** — placeholder, returns false

## Phase 2 Pipeline Ordering

`build_phase2_pipeline()` in `src/passes/mod.rs` orders all 18 passes:

1. AnnotateOutputDimOrderPass
2. ConvertToClampPass (from 2A)
3. FuseQuantizedActivationPass
4. Decomposition passes (LayerNorm, Var, MeanDim, FullLike — from 2A)
5. FoldAndAnnotateQParamsPass (existing)
6. FuseDuplicateUsersPass (no-op)
7. InsertRescalePass (existing)
8. DecomposeQuantNodesPass (existing)
9. BroadcastArgsPass
10. MatchArgDtypePass
11. FuseConstantArgsPass
12. CastToInt32Pass
13. RewriteConvPass (existing)
14. ToTosaMemoryFormatPass (existing)
15. RemoveNoopPass (no-op)
16. InsertRescalePass (existing, final)
17. FuseEqualPlaceholdersPass (no-op)
18. CleanupPass (existing)

## Cargo Check Results

```
cargo check: OK (0 errors)
cargo test:  18/18 passed
```

All warnings are "unused" warnings for code not yet called from the library entry point, which is expected since `build_phase2_pipeline` is not yet wired into `lower_to_tosa`.

## Files Changed

- `src/passes/mod.rs` — Added 7 new module declarations, 3 no-op pass structs, `build_phase2_pipeline()`
- `src/passes/broadcast_args.rs` — NEW
- `src/passes/match_arg_dtype.rs` — NEW
- `src/passes/fuse_constants.rs` — NEW
- `src/passes/fuse_quantized_activation.rs` — NEW
- `src/passes/annotate_dim_order.rs` — NEW
- `src/passes/cast_int.rs` — NEW

Phase 1 pipeline and all existing tests are unaffected.
