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

### Phase 1: NodeMeta extraction (low risk)

Extract metadata into `NodeMeta` structs at pass entry, operate on them,
write back at pass exit. The FX graph structure stays Python-managed.

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

### Phase 2: Full IrGraph for metadata passes

Passes that only read/write metadata (FxControlFlowMetadataPass,
propagate_parent_qparams, etc.) can use IrGraph fully.

### Phase 3: Graph surgery in IrGraph

Passes that insert/remove nodes (insert_rescale, boundary rescales,
Q/DQ rewriting) need graph mutation support in IrGraph:
- `insert_after(anchor: NodeId, new_node: IrNode) -> NodeId`
- `insert_before(anchor: NodeId, new_node: IrNode) -> NodeId`
- `replace_all_uses(old: NodeId, new: NodeId)`
- `erase(node: NodeId)`

These map to the 8x `replace_all_uses_with` + 8x `erase_node` +
27x `call_function` (node creation) calls across the codebase.

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
