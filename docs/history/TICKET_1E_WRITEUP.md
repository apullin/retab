# Ticket 1E: Shape Propagation -- Writeup

## Files Created / Modified

### Created
- **`src/shape/mod.rs`** -- Shape propagation module with `propagate_shape()` public API, all compute functions, helpers, and unit tests.

### Modified
- **`src/lib.rs`** -- Added `mod shape;` declaration.

## Shape Rules Implemented

The `propagate_shape(graph, node_id)` function dispatches on `OpTarget` to compute output `TensorInfo`:

| Op(s) | Rule |
|---|---|
| **Conv2d / TosaConv2d / TosaDepthwiseConv2d** | `H_out = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1`, same for W. Output channels from weight dim 0. |
| **Relu, Clamp, Hardtanh, TosaClamp** | Element-wise: output shape = input shape. |
| **Add, Sub, Mul, Div, TosaAdd, TosaSub, TosaMul** | NumPy-style broadcasting: right-align shapes, take max of each dim (treating 1 as broadcastable). Dtype promotion (float > int). |
| **Permute / TosaTranspose** | `output[i] = input[perm[i]]`. |
| **Reshape / View / TosaReshape** | New shape from IntList arg; `-1` dimension inferred from total element count. |
| **Full** | Shape from IntList arg[0], dtype from kwargs or default Float32. |
| **AvgPool2d, MaxPool2d, TosaAvgPool2d, TosaMaxPool2d** | `H_out = (H + 2*pad - kH) / stride + 1`, channels preserved. |
| **Sum / TosaReduceSum** | Reduce along specified dims. If keepdim=true, reduced dims become 1; otherwise they are removed. |
| **Mean** | Same reduction logic as Sum. |
| **Rsqrt, Neg, Abs, Reciprocal, Exp, Log, Sigmoid, Tanh** | Element-wise unary: output = input. |
| **QuantizePerTensor** | Shape preserved, dtype becomes Int8. |
| **DequantizePerTensor / DequantizePerChannel** | Shape preserved, dtype becomes Float32. |
| **TosaRescale** | Shape preserved. |
| **TosaConst** | Uses existing `node.meta.val`. |
| **Fallback (all other ops)** | Propagates first input shape if args exist, else keeps existing meta. |

## Helper Functions

- `input_info(graph, node, arg_idx)` -- Extracts `TensorInfo` from the node referenced at `args[arg_idx]`.
- `int_list_arg(node, idx)` -- Extracts an IntList from the given arg index (handles both `IntList` and single `Int`).
- `kwarg_bool(node, name, default)` -- Looks up a boolean kwarg by name.
- `broadcast_dtype(a, b)` -- Dtype promotion for binary ops (Float32 > Float16 > BFloat16 > Int64 > Int32).

## Test Results

All 15 tests pass (`cargo test --lib`):

```
test shape::tests::test_broadcast ... ok
test shape::tests::test_broadcast_different_ranks ... ok
test shape::tests::test_conv2d_no_padding ... ok
test shape::tests::test_conv2d_shape ... ok
test shape::tests::test_dequantize_changes_dtype ... ok
test shape::tests::test_elementwise_preserves_shape ... ok
test shape::tests::test_full ... ok
test shape::tests::test_mean_keepdim ... ok
test shape::tests::test_permute ... ok
test shape::tests::test_pool2d ... ok
test shape::tests::test_quantize_changes_dtype ... ok
test shape::tests::test_reshape_flatten ... ok
test shape::tests::test_reshape_with_neg1 ... ok
test shape::tests::test_sum_keepdim ... ok
test shape::tests::test_sum_no_keepdim ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Test cases cover all ticket-required scenarios:
- Conv2d: (1,1,28,28) weight (16,1,3,3) pad=1 stride=1 -> (1,16,28,28)
- Broadcast: (1,16,1,1) + (1,16,28,28) -> (1,16,28,28)
- Reshape: (1,32,7,7) -> shape [1, 1568] -> (1,1568)
- Permute: (1,2,3,4) perm [0,3,1,2] -> (1,4,2,3)
- Pool2d: (1,16,28,28) kernel=2 stride=2 -> (1,16,14,14)
- Sum with keepdim: (1,16,28,28) dims=[2,3] keepdim=true -> (1,16,1,1)

## >4D Shape Limitation

`TensorInfo` uses a fixed `[i64; 4]` array, so shapes are capped at 4 dimensions. This is sufficient for the Phase 1 op set (conv2d, pool2d, standard vision models) but would need extension for:

- **Bmm / TosaMatmul** -- Batched matrix multiply can operate on 3D tensors (within limit) but higher-rank batch dimensions would need >4D.
- **Embedding** -- Output is typically 3D `(batch, seq_len, embed_dim)` which fits, but dynamic shapes could push beyond 4D.
- **Reshape/View** -- Users may reshape to >4D; currently truncated to 4 dims.
- **Unsqueeze** -- Repeated unsqueeze can push rank beyond 4.

The `TensorInfo::new` constructor silently truncates shapes longer than 4 dimensions via `.min(4)`. For Phase 1 (CNN-focused workloads), the 4D limit is sufficient.
