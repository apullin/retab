# Project Assessment — arm-tosa-engine

## What We Built

A Rust + Python hybrid tool to speed up ExecuTorch ARM/Ethos-U55 model lowering.

## Verified Results

| Approach | Time | Speedup | Vela ✓ | Status |
|----------|------|---------|--------|--------|
| **Baseline** (Python) | 59s | 1.0x | ✓ | Reference |
| **Python + skip no-ops** | 38s | 1.5x | ✓ | Working, on master |
| **Python passes + Rust serialize** | 59s | 1.0x | ✓ | Working, minimal gain |
| **Pure Rust passes + Rust serialize** | 0.05s | 1000x+ | ✗ | Fast but incorrect output |

## Branches

| Branch | What | State |
|--------|------|-------|
| `master` | Python skip-opt (38s), Vela-verified | Stable, working |
| `python-opt` | Same as master | Merged |
| `fx-graph-direct` | FX-direct Rust passes prototype | Working prototype, not Vela-verified |
| `wip/fuse-constants-rust` | Rust FuseConstantArgsPass | Compiles, not tested end-to-end |

## The Three Approaches

### 1. All-Python Optimization (master)
- Skip ~58 no-op passes: 59s → 38s
- Next: replace FuseConstantArgsPass (13s) → ~25s
- Floor: ~10s (ExportPass overhead for passes that DO work)
- **Pro**: correct by construction, no interop issues
- **Con**: can't get below ~10s due to ExportPass retracing

### 2. Hybrid (Python fold/rescale + Rust everything else)
- Python fold+rescale: ~1.5s (need annotations, can't avoid)
- Rust FX-direct passes: ~0.15s (table, dedup, clamp, conv, memory)
- FuseConstantArgs: ~13s (or ~0.1s if Rust'd)
- **Pro**: correct fold/rescale, fast everything else
- **Con**: still ~15s with Python FuseConstantArgs, complex integration
- **Target**: ~2s if FuseConstantArgs is Rust'd

### 3. All-Rust (FX-direct passes at preprocess point)
- All passes via Rust FX-direct: ~0.15s
- Rust TOSA serialization: ~0.05s
- FuseConstantArgs via Python delegation: ~13s (or ~0.1s if Rust'd)
- **Pro**: fastest possible, monolith under our control
- **Con**: QuantArgs format, qparams propagation, RESCALE count issues
- **Target**: <1s if everything works

## The Bottleneck Analysis

```
60s total breakdown:
├── 40s: ExportPass retracing on ~80 no-op passes (ELIMINATED by skip-opt)
├── 13s: FuseConstantArgsPass (actual computation)
├──  5s: ~20 passes that do real work × 500ms ExportPass overhead each
├──  1s: TOSA serialization (replaced by Rust)
└──  1s: Vela compilation (unchanged)
```

With skip-opt: 38s = 13s (FuseConstants) + 20s (active passes × ExportPass) + 5s (other)
With Rust FuseConstants: ~25s = 20s (active passes × ExportPass) + 5s (other)
With FX-direct for active passes: ~15s = 13s (FuseConstants) + 2s (other)
With everything optimized: ~2s = 1s (Rust FuseConstants) + 1s (other)

## Key Learnings

1. **ExportPass is the enemy, not pass logic.** Every pass costs 500ms for graph retracing regardless of whether it modifies anything. The pass logic itself is <10ms.

2. **The Python ARM backend is tightly coupled.** QuantArgs format, ARM annotations, ExportPass infrastructure — they all depend on each other. Partial reimplementation causes format mismatches.

3. **Integration point matters.** `TOSABackend.preprocess()` is the right hook — the graph has annotations, it's a single function, and everything upstream/downstream stays unchanged.

4. **Rust FX-direct works but needs QuantArgs compatibility.** Our FX-direct passes are 1000x faster and produce structurally correct output (11/14 op types match). The 3 remaining gaps are all about QuantArgs/rescale compatibility.

## Decision Point

**Option A: All-Python with optimizations**
- Ship the skip-opt (38s) now, add Rust FuseConstants later (~25s)
- Safe, incremental, limited upside (~25s floor)

**Option B: Hybrid at preprocess()**
- Python fold/rescale + Rust everything else
- Target ~2-15s depending on FuseConstants
- Medium complexity, good performance

**Option C: All-Rust at preprocess()**
- Fix QuantArgs compatibility, get all FX-direct passes working
- Target <1s
- Most complexity, best performance, full control
