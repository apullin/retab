# arm-tosa-engine — Agent Orchestration

## Architecture

Work is dispatched as tickets to autonomous subagents. Each ticket has:
- **Goal**: What to build
- **Inputs**: Files to read, specs to follow
- **Outputs**: Files to create, tests to pass
- **Writeup**: Agent must document what it built and any deviations

The orchestrator (Claude) reviews writeups against actual code.

## Phase 1 Tickets

### Ticket 1A: Crate Scaffolding + Graph IR
Create the Rust project structure, Cargo.toml, PyO3/maturin config, and the
core Graph IR types (Node, Graph, OpTarget, Arg, TensorInfo, DType, QuantArgs).
No passes, no serialization — just the data structures and basic graph operations.

### Ticket 1B: FX Graph Ingestion (PyO3 Bridge)
Build the Python→Rust bridge that converts a torch.fx.GraphModule + state_dict
into the Rust Graph + TensorStore. Requires Ticket 1A complete.

### Ticket 1C: TOSA Flatbuffer Codegen + Serialization Skeleton
Run `flatc --rust` on tosa.fbs, set up the serialization module, implement
visitors for CONV2D, RESCALE, CLAMP, TRANSPOSE, CONST.

### Ticket 1D: Core Passes (conv2d path)
Implement the minimum passes for tiny_cnn: FoldAndAnnotateQParamsPass,
InsertRescaleInt32Pass, DecomposeQuantNodesPass, RewriteConvPass,
ToTosaMemoryFormatPass, InsertRescalePass.

### Ticket 1E: Shape Propagation
Implement shape inference for all Phase 1 ops: conv2d, relu/clamp, add,
reshape, permute, transpose, maxpool2d, Q/DQ, rescale, full.

### Ticket 1F: Integration Test
Wire everything together: PyO3 entry point, test harness comparing Rust vs
Python TOSA output on tiny_cnn.

## Status

| Ticket | Status | LOC | Notes |
|--------|--------|-----|-------|
| 1A | DONE ✓ | ~450 | Graph IR, slotmap arena, builder ops, topo sort |
| 1B | DONE ✓ | ~350 | FX→Rust ingestion, target mapping, state_dict numpy |
| 1C | DONE ✓ | research | Schema analysis, visitor patterns, reference TOSA dump |
| 1C-impl | DONE ✓ | ~500 | Manual TOSA types, serialization skeleton, visitors |
| 1D | DONE ✓ | ~1220 | 6 passes: fold_qdq, insert_rescale, rewrite_conv, memory_format, decompose_quant, cleanup |
| 1E | DONE ✓ | ~350 | Shape propagation for all Phase 1 ops, 15 tests pass |
| 1F | IN PROGRESS | — | Integration: wire lower_to_tosa, E2E test |

### Reference Data
- `test_data/tiny_cnn_reference.tosa` — 30,392 bytes, 38 ops, 7 unique op types
- `test_data/tiny_cnn_reference_summary.txt` — human-readable TOSA dump

### Phase 1 Results
- 4,539 Rust LOC, 18 unit tests pass, maturin build 6.5s
- E2E pipeline runs: 45ms (1,345x faster than Python's 60.6s)
- Output not yet byte-identical (partial op coverage)

## Phase 2 Tickets

### Ticket 2A: Decomposition Passes (LayerNorm path)
Implement the decomposition passes needed for cnn_layernorm4d:
DecomposeLayerNormPass, DecomposeVarPass, DecomposeMeanDimPass,
DecomposeSumPass, ConvertToClampPass, ConvertFullLikeToFullPass.
These turn complex ATen ops into simpler ones the TOSA visitors can handle.

### Ticket 2B: Scalar/Type Matching + Fusion Passes
BroadcastArgsPass, MatchArgDtypePass, MatchArgRanksPass,
UnsqueezeScalarPlaceholdersPass, FuseQuantizedActivationPass,
FuseDuplicateUsersPass, FuseEqualPlaceholdersPass, FuseConstantArgsPass,
AnnotateOutputDimOrderPass, CastToInt32Pass, CastInt64BuffersToInt32Pass.

### Ticket 2C: Additional TOSA Visitors + Shape Rules
Add TOSA visitors for: ADD, MUL, SUB, REDUCE_SUM, AVG_POOL2D, TABLE,
FULLY_CONNECTED, CLAMP. Extend shape propagation for any missing ops.

### Ticket 2D: Reference Data + Integration Test
Generate cnn_layernorm4d reference TOSA, wire Phase 2 passes into pipeline,
run comparison benchmark.

## Phase 2 Status

| Ticket | Status | Notes |
|--------|--------|-------|
| 2A | dispatched | — |
| 2B | dispatched | parallel with 2A |
| 2C | dispatched | parallel with 2A, 2B |
| 2D | waiting on 2A-2C | — |
