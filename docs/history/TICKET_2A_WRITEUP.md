# Ticket 2A: Decomposition Passes for cnn_layernorm4d

## Summary

Implemented decomposition and conversion passes that break complex ATen ops into simpler ops that the existing TOSA visitors can handle. These are needed for the `cnn_layernorm4d` model which uses LayerNorm, mean, var, sum, avg_pool2d.

## New Files

### `src/passes/decompose/mod.rs`
Module root declaring the three decomposition sub-passes.

### `src/passes/decompose/layernorm.rs` -- DecomposeLayerNormPass
Matches `OpTarget::LayerNorm` nodes and decomposes each into:
1. `mean = Mean(x, dims, keepdim=true)` -- E[x]
2. `diff = Sub(x, mean)` -- x - E[x]
3. `var = Var(x, dims, correction=0, keepdim=true)` -- Var[x]
4. `eps_tensor = Full(eps_shape, epsilon)` -- epsilon constant
5. `var_eps = Add(var, eps_tensor)` -- Var[x] + eps
6. `inv_std = Rsqrt(var_eps)` -- 1/sqrt(Var[x] + eps)
7. `normalized = Mul(diff, inv_std)` -- (x - E[x]) / sqrt(Var[x] + eps)
8. If weight present: `Reshape(weight) -> Mul(normalized, weight_reshaped)`
9. If bias present: `Reshape(bias) -> Add(result, bias_reshaped)`

Handles all LayerNorm arg patterns (2-6 args). Weight/bias reshape uses broadcastable shape (1 for non-reduced dims, original size for reduced dims).

### `src/passes/decompose/meandim.rs` -- DecomposeMeanDimPass
Matches `OpTarget::Mean` nodes and decomposes each into:
1. `sum = Sum(x, dims, keepdim)`
2. `count_tensor = Full(ones_shape, 1.0/count)` where count = product of reduced dim sizes
3. `result = Mul(sum, count_tensor)`

### `src/passes/decompose/var.rs` -- DecomposeVarPass
Matches `OpTarget::Var` nodes and decomposes each into:
1. `mean = Mean(x, dims, keepdim=true)` -- always keepdim for broadcast correctness
2. `diff = Sub(x, mean)`
3. `sq = Mul(diff, diff)`
4. `sum_sq = Sum(sq, dims, keepdim)`
5. `divisor = Full(ones_shape, 1.0 / max(0, N - correction))`
6. `result = Mul(sum_sq, divisor)`

Supports both Bessel-corrected (correction=1) and population (correction=0) variance.

### `src/passes/convert_clamp.rs` -- ConvertToClampPass
Converts activation ops to the unified `Clamp` op:
- `Relu(x)` -> `Clamp(x, min=0.0, max=f32::MAX)`
- `Hardtanh(x, min_val, max_val)` -> `Clamp(x, min=min_val, max=max_val)`

Rewrites nodes in-place (changes target and kwargs).

### `src/passes/convert_full.rs` -- ConvertFullLikeToFullPass
Converts `FullLike(input, fill_value)` to `Full(input_shape, fill_value)` by reading the input tensor's shape and rewriting the node in-place.

## Modified Files

### `src/passes/mod.rs`
- Added `mod` declarations for `decompose`, `convert_clamp`, `convert_full`
- Added `build_phase2_pipeline()` function that runs decompositions first, then Phase 1 passes
- Phase 1 pipeline unchanged

### `src/graph/ir.rs`
- Added `Var` and `FullLike` variants to `OpTarget`

### `src/shape/mod.rs`
- Added `Var` to the shape propagation dispatch (uses same reduction logic as `Mean`)

### `src/ingest/target_map.rs`
- Added parsing for `aten.var.*` -> `OpTarget::Var`
- Added parsing for `aten.full_like` -> `OpTarget::FullLike`

## Pipeline Ordering (Phase 2)

1. DecomposeLayerNormPass -- breaks LayerNorm into Mean, Var, Sub, Mul, etc.
2. DecomposeVarPass -- breaks Var into Mean, Sub, Mul, Sum, Full
3. DecomposeMeanDimPass -- breaks Mean into Sum, Full, Mul
4. ConvertToClampPass -- Relu/Hardtanh -> Clamp
5. ConvertFullLikeToFullPass -- FullLike -> Full
6. (Phase 1 passes: FoldQDQ, InsertRescale, DecomposeQuant, RewriteConv, MemoryFormat, Cleanup)

The ordering ensures that LayerNorm is decomposed first (producing Var and Mean nodes), then Var is decomposed (producing more Mean nodes), then Mean is decomposed last (into Sum+Mul which TOSA visitors handle directly).

## cargo check Results

```
cargo check passes with 0 errors, 0 new warnings.
All existing warnings are pre-existing dead-code warnings unrelated to this change.
```
