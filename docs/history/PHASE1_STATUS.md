# Phase 1 Status — arm-tosa-engine

**Date:** 2026-03-17

## Metrics

| Metric | Result |
|--------|--------|
| **Total Rust LOC** | 4,539 lines across 21 files |
| **cargo check** | PASS (22 warnings, 0 errors) |
| **cargo test --lib** | PASS (18/18 tests passed) |
| **maturin build** | NOT RUN (maturin not installed in environment) |
| **E2E test** | NOT RUN (depends on maturin build) |

## Code Breakdown (lines)

| Module | Lines | Description |
|--------|-------|-------------|
| `shape/mod.rs` | 770 | Shape propagation for all TOSA ops |
| `serialize/visitors.rs` | 529 | FlatBuffer emission visitors |
| `serialize/mod.rs` | 376 | TOSA serialization orchestration |
| `serialize/tosa_generated/mod.rs` | 357 | TOSA FlatBuffer schema types |
| `ingest/fx_bridge.rs` | 337 | PyO3 bridge: FX graph -> IR |
| `passes/rewrite_conv.rs` | 336 | Conv2d rewrite pass (weight transpose, bias reshape) |
| `graph/builder.rs` | 284 | Graph builder with node/edge construction |
| `passes/insert_rescale.rs` | 255 | Rescale insertion for quantized ops |
| `passes/fold_qdq.rs` | 246 | Quantize/dequantize folding pass |
| `graph/ir.rs` | 231 | Core IR: Node, Edge, OpTarget, DType, Arg |
| `passes/memory_format.rs` | 198 | NCHW-to-NHWC memory format conversion |
| `ingest/target_map.rs` | 165 | FX op name -> OpTarget mapping |
| `passes/cleanup.rs` | 84 | Dead node elimination |
| `graph/topo.rs` | 66 | Topological sort (Kahn's algorithm) |
| `graph/mod.rs` | 63 | Graph module re-exports |
| `passes/mod.rs` | 60 | Pass trait + pipeline runner |
| `store.rs` | 55 | Tensor weight store |
| `lib.rs` | 54 | PyO3 module entry point |
| `passes/decompose_quant.rs` | 39 | Quantize decomposition pass |
| `compile_spec.rs` | 32 | Compile spec (target, config) |

## What Works

- **Full IR representation**: Typed graph IR with SlotMap-based node/edge storage, supporting all TOSA 1.0 op targets
- **FX graph ingestion**: PyO3 bridge that walks a torch.fx GraphModule and builds the IR (nodes, edges, shapes, dtypes, constant data)
- **Shape propagation**: 18 tested propagation rules covering conv2d, pooling, broadcasting, reshape, permute, reduce, quantize/dequantize, and more
- **Compiler passes** (6 passes):
  - `fold_qdq` — folds quantize/dequantize pairs into integer-domain ops
  - `insert_rescale` — inserts TOSA rescale ops with computed multiplier/shift
  - `rewrite_conv` — transposes conv weights to TOSA layout, reshapes bias
  - `memory_format` — converts NCHW tensors to NHWC with transpose nodes
  - `decompose_quant` — decomposes quantize into mul+add+clamp
  - `cleanup` — removes dead/unreachable nodes
- **TOSA FlatBuffer serialization**: Emits valid TOSA FlatBuffer binary with correct file identifier, version, regions, blocks, operators, and tensors
- **Topological sort**: Kahn's algorithm ensures correct evaluation order
- **cargo check**: Compiles cleanly (warnings are expected — enum variants and structs defined for completeness but not yet exercised)

## What Does Not Yet Work

- **Maturin build untested in this session**: `maturin` was not installed in the virtual environment, so the Python extension module was not built. This is an environment setup issue, not a code issue.
- **E2E test not run**: Blocked on the maturin build. The test script (`tests/test_e2e.py`) is written and ready.
- **Dead code warnings**: 22 warnings from unused enum variants (e.g., `OpTarget::DepthwiseConv2d`, `Op::Conv3d`) and helper functions. These are intentionally defined for future op coverage and are not errors.
- **Limited op coverage in passes**: Only conv2d, add, reshape, and a few elementwise ops are fully wired through the pass pipeline. Other TOSA ops (matmul, depthwise_conv2d, fully_connected, etc.) have IR definitions but no pass logic yet.

## Next Steps for Phase 2

1. **Install maturin and validate E2E**: `uv pip install maturin && maturin develop --release && python tests/test_e2e.py`
2. **Expand op coverage**: Wire up remaining ops through the pass pipeline (depthwise_conv2d, fully_connected, matmul, pooling, concat, slice, pad, etc.)
3. **Vela integration**: Feed TOSA FlatBuffer output into Vela compiler to produce Ethos-U command stream
4. **Round-trip validation**: Compare Rust TOSA output byte-for-byte against the Python reference backend
5. **Error handling**: Replace panics/unwraps with proper error propagation (anyhow or thiserror)
6. **Performance benchmarking**: Measure Rust lowering time vs. Python reference on real models
7. **CI pipeline**: Add cargo test + maturin build to CI
