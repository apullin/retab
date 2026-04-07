# Rust Graph IR for RETAB

## Goal

Replace PyO3 `Bound<'py, PyAny>` manipulation of Python FX graphs with a
Rust-native graph IR. Import once from Python at the boundary, run all passes
in pure Rust, export back to Python for serialization.

This eliminates `<'py>` lifetimes from all internal code, removes 40+ 
`.getattr("meta")` calls, 8+ `replace_all_uses_with` calls, and the
entire PyO3 ceremony from graph manipulation.

## Current state

Every pass operates on Python FX graph objects via PyO3:
```rust
fn some_pass<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<bool> {
    for node in graph_nodes(gm)? {           // Python iter
        let meta = node.getattr("meta")?;    // Python getattr
        let oqp = meta.get_item("output_qparams")?;  // Python dict access
        // ... 40+ lines of getattr/extract/set_item ...
    }
}
```

## Target state

```rust
fn some_pass(graph: &mut IrGraph) -> bool {
    for node in graph.nodes_mut() {
        if let Some(qp) = &node.meta.output_qparams {
            // pure Rust, no lifetimes, no PyResult
        }
    }
}
```

## Type design

### Node

```rust
pub type NodeId = usize;  // index into IrGraph.nodes

pub struct IrNode {
    pub id: NodeId,
    pub op: NodeOp,           // already exists
    pub name: String,
    pub target: String,       // str(node.target)
    pub args: Vec<IrArg>,
    pub meta: NodeMeta,
}

pub enum IrArg {
    Node(NodeId),             // reference to another node
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<IrArg>),
    Dtype(String),            // "torch.int8" etc
    None,
    // Opaque Python object (for things we can't represent yet)
    PyObj(usize),             // index into IrGraph.py_objects
}
```

### NodeMeta

Based on actual usage inventory (63x val, 50x output_qparams, 37x input_qparams,
19x tosa_dim_order, 13x tosa_spatial_rank, 9x arm_layout_neutral):

```rust
pub struct NodeMeta {
    // Core quantization (accessed 87x combined across all passes)
    pub input_qparams: Option<HashMap<usize, QParam>>,   // already have QParam!
    pub output_qparams: Option<HashMap<usize, QParam>>,

    // Tensor info (accessed 63x)
    pub val_dtype: Option<String>,
    pub val_shape: Option<Vec<i64>>,

    // Layout (accessed 19x + 13x + 9x)
    pub tosa_dim_order: Option<Vec<usize>>,
    pub tosa_spatial_rank: Option<usize>,
    pub arm_layout_neutral: bool,

    // Control flow markers
    pub arm_control_flow_placeholder: bool,
    pub pre_pass_arg_qparams: Option<HashMap<usize, QParam>>,

    // Catch-all for less common keys
    pub extra: HashMap<String, MetaValue>,
}

pub enum MetaValue {
    QParam(QParam),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    // For val (FakeTensor) — keep as opaque Python ref during migration
    PyObj(usize),
}
```

### Graph

```rust
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    // Opaque Python objects that we can't represent in Rust yet
    // (FakeTensors, torch.dtype objects, op targets)
    py_objects: Vec<pyo3::PyObject>,
}
```

## Import / Export

### Import: `IrGraph::from_fx(gm: &Bound<PyAny>) -> PyResult<IrGraph>`

Walk `gm.graph.nodes`, for each Python node:
1. Extract `op`, `name`, `target` (strings)
2. Extract `args` — map FX nodes to NodeIds, literals to IrArg variants
3. Extract `meta` — read known keys into typed fields, store FakeTensor 
   as opaque PyObj
4. Build `IrNode`, append to graph

For submodules (cond/while bodies): import recursively, store as separate
IrGraphs referenced by the cond/while node.

### Export: `IrGraph::write_back_to_fx(gm: &Bound<PyAny>) -> PyResult<()>`

Two strategies:

**A. Metadata-only writeback** (easiest, start here):
- Walk existing Python FX nodes
- For each, find matching IrNode by name
- Write back changed meta fields (qparams, dim_order, etc.)
- Don't modify graph structure — only metadata

**B. Full graph rebuild** (later):
- Clear the Python graph
- Re-create nodes from IrGraph
- Requires creating FX nodes with correct targets, args, fake tensors

## Migration path

### Phase 1: NodeMeta extraction (low risk) — DONE

Extract metadata into `NodeMeta` structs at pass entry, operate on them,
write back at pass exit. The FX graph structure stays Python-managed.

**Migrated passes:**
- `propagate_parent_qparams_to_placeholders` (qparam copy + while output stamp)
- `annotate_cf_parent_rank4_dim_order`
- `annotate_cf_rank4_dim_order`
- `force_cf_rank4_output_dim_order`
- `snapshot_pre_pass_cf_qparams`
- `extract_effective_qparam` (producer chain walk)
- `stamp_qparams`
- `FxWhileStripBoundaryQPass` DQ annotation
- `insert_cf_boundary_rescales` (input + output sections)

**Not migrated (need FakeTensor Python calls):**
- `restamp_cf_placeholder_dtypes` (calls `_branch_local_control_flow_output_qparam`, `fake_val.to(dtype=...)`)
- `refresh_tosa_transpose_fake_tensors` (creates FakeTensors via `new_empty`)
- `refresh_tosa_dim_order_fake_tensors` (calls `_get_matching_fake_tensor`)

```rust
fn some_pass(py: Python, gm: &Bound<PyAny>) -> PyResult<bool> {
    let mut metas: Vec<(Bound<PyAny>, NodeMeta)> = graph_nodes(gm)?
        .into_iter()
        .map(|n| {
            let meta = NodeMeta::from_fx(&n)?;
            Ok((n, meta))
        })
        .collect::<PyResult<_>>()?;
    
    // Pure Rust manipulation of metas — no <'py>
    for (_, meta) in &mut metas {
        if let Some(ref mut oqp) = meta.output_qparams { ... }
    }
    
    // Write back
    for (node, meta) in &metas {
        meta.write_back(&node)?;
    }
}
```

This is incremental — each pass can be migrated independently.

### Phase 2: Full IrGraph for metadata passes — DONE

IrGraph imports full node topology: IrNode with id, op, name, target,
args (Vec<IrArg> with NodeId references), meta (NodeMeta). Reverse
edges (users) built during import.

**Pure Rust `_ir` functions (no `<'py>`, no `PyResult`):**
- `extract_effective_qparam_ir(graph, start_id)` — producer chain walk
- `reverse_engineer_rescale_ir(graph, node)` — derive qparam from RESCALE args
- `annotate_rank4_dim_order_ir(graph)` — stamp NHWC on rank-4 nodes
- `stamp_while_body_output_qparams_ir(graph)` — loop-carried output qparams
- `force_rank4_output_dim_order_ir(graph)` — stamp NHWC on output args

**Wrapper pattern:** `IrGraph::from_fx(&submodule)?` → pure Rust → `ir.write_back(py, &submodule)?`

**Not migrated (need Python for FakeTensor or graph surgery):**
- Functions calling `QuantArgs.from_operator`, `_branch_local_control_flow_output_qparam`
- Functions doing `erase_node`, `replace_all_uses_with`, `call_function` (Phase 3)

### Phase 3: Graph surgery in IrGraph — PARTIAL

IrGraph now supports mutation:
- `replace_all_uses(old_id, new_id)` — update all args referencing old→new
- `erase(node_id)` — mark node dead
- `replace_and_erase(old_id, new_id)` — combined
- `add_node(...)` — append new node
- `write_back_with_surgery()` — apply replace/erase to Python graph

**Migrated to pure Rust surgery:**
- `remove_noop_passthroughs_ir` — detects noops, propagates metadata, replace_and_erase
- `collapse_qdq_pairs_ir` — detects matching Q→DQ pairs, bypasses to float source (ready but not yet wired into caller)

### Phase 3.1: RescaleInsert deferred node creation — DONE

`RescaleInsert` struct: decision made in pure Rust, applied via Python wrapper.
Fields: anchor_name, source_name, from_qp, to_qp, after, replace/erase names.
`apply()` method calls `insert_rescale_node` Python helper, then replace/erase.

**Additional passes migrated:**
- `rewrite_quantize_nodes_ir` — bypass same-qparam Q (surgery) + emit RescaleInsert for different
- `rewrite_dequantize_nodes_ir` — bypass DQ with RESCALE/output users (surgery)
- `collapse_qdq_pairs_ir` — wired into rewrite loop

**9 pure Rust `_ir` functions total.** The entire `rewrite_control_flow_quantization_boundaries`
loop now uses Rust for all decision-making. Python only for RESCALE FX node creation.

### Phase 4: Remove Python from hot path

With full IrGraph, the pipeline becomes:
```
Python entry → IrGraph::from_fx() → [all passes in pure Rust] → write_back_to_fx() → serialize
```

The `<'py>` lifetime exists only in `from_fx` and `write_back_to_fx`.

## What stays Python

- `FakeTensor` creation (requires torch runtime)
- `insert_rescale_node` in `_helpers.py` (creates FX nodes with correct
  meta["val"] FakeTensors) — until Phase 3 handles node creation
- `_decomp_hooks.py` (monkey-patching, not in hot path)
- `preprocess.py` entry point (thin orchestration layer)

## Risk

- **FakeTensor fidelity**: The `meta["val"]` FakeTensor is used by downstream
  serialization to determine shapes/dtypes. If we don't round-trip it
  perfectly, serialization breaks. Phase 1 (metadata-only) avoids this
  by keeping FakeTensors as opaque PyObjects.

- **Node ordering**: FX graph node order matters for serialization. IrGraph
  must preserve insertion order.

- **Submodule graphs**: cond/while bodies are separate GraphModules. The IR
  needs to handle nested graphs (IrGraph containing child IrGraphs).

## Success criteria

- Phase 1: At least one pass (e.g., FxControlFlowMetadataPass) runs with
  NodeMeta extraction, zero test regressions
- Full: All 2589 lines of fx_control_flow_metadata.rs have zero `<'py>`
  in pass logic (only in from_fx/write_back)
