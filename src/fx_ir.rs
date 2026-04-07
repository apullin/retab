//! Rust-native graph IR for FX graph manipulation.
//!
//! Phase 1: NodeMeta — typed metadata extraction (done).
//! Phase 2: IrGraph — full node topology imported from Python FX graph.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_control_flow_metadata::{NodeOp, QParam};

// ---------------------------------------------------------------------------
// IrGraph — full graph topology
// ---------------------------------------------------------------------------

pub type NodeId = usize;

/// A node in the Rust IR graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    pub id: NodeId,
    pub op: NodeOp,
    pub name: String,
    pub target: String,
    pub args: Vec<IrArg>,
    pub meta: NodeMeta,
}

/// An argument to a node — either a reference to another node or a literal.
#[derive(Debug, Clone)]
pub enum IrArg {
    Node(NodeId),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<IrArg>),
    None,
}

/// The full graph: ordered list of nodes + lookup by name.
#[derive(Debug)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    name_to_id: HashMap<String, NodeId>,
    /// Reverse edges: users[node_id] = list of node IDs that reference this node in their args.
    users: Vec<Vec<NodeId>>,
}

impl IrGraph {
    /// Import from a Python FX GraphModule.
    pub fn from_fx(gm: &Bound<'_, PyAny>) -> PyResult<Self> {
        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();

        // First pass: assign IDs and build name→id map
        let mut name_to_id = HashMap::new();
        for (i, node) in py_nodes.iter().enumerate() {
            let name: String = node.getattr("name")
                .and_then(|v| v.extract()).unwrap_or_default();
            name_to_id.insert(name, i);
        }

        // Second pass: build IrNodes with resolved args
        let mut nodes = Vec::with_capacity(py_nodes.len());
        for (i, py_node) in py_nodes.iter().enumerate() {
            let name: String = py_node.getattr("name")
                .and_then(|v| v.extract()).unwrap_or_default();
            let target: String = py_node.getattr("target")
                .and_then(|t| t.str().map(|s| s.to_string()))
                .unwrap_or_default();
            let op = NodeOp::of(py_node);
            let meta = NodeMeta::from_fx(py_node)?;

            let py_args = crate::fx_graph::FxGraphModule::node_args(py_node)
                .unwrap_or_default();
            let args = py_args.iter()
                .map(|a| resolve_arg(a, &name_to_id))
                .collect();

            nodes.push(IrNode { id: i, op, name, target, args, meta });
        }

        // Build reverse edges
        let mut users = vec![Vec::new(); nodes.len()];
        for node in &nodes {
            for arg in &node.args {
                if let IrArg::Node(target_id) = arg {
                    users[*target_id].push(node.id);
                }
                if let IrArg::List(items) = arg {
                    for item in items {
                        if let IrArg::Node(target_id) = item {
                            users[*target_id].push(node.id);
                        }
                    }
                }
            }
        }

        Ok(Self { nodes, name_to_id, users })
    }

    /// Find a node by name.
    pub fn by_name(&self, name: &str) -> Option<&IrNode> {
        self.name_to_id.get(name).map(|&id| &self.nodes[id])
    }

    /// Iterate placeholders.
    pub fn placeholders(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|n| n.op == NodeOp::Placeholder)
    }

    /// Get the output node.
    pub fn output_node(&self) -> Option<&IrNode> {
        self.nodes.iter().find(|n| n.op == NodeOp::Output)
    }

    /// Iterate call_function nodes.
    pub fn call_functions(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|n| n.op == NodeOp::CallFunction)
    }

    /// Get node by ID.
    pub fn node(&self, id: NodeId) -> &IrNode {
        &self.nodes[id]
    }

    /// Get mutable node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut IrNode {
        &mut self.nodes[id]
    }

    /// Get users (nodes that reference this node in their args).
    pub fn users(&self, id: NodeId) -> &[NodeId] {
        &self.users[id]
    }

    /// Write back all dirty metadata to the Python FX graph.
    pub fn write_back(&self, py: Python<'_>, gm: &Bound<'_, PyAny>) -> PyResult<()> {
        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();

        for (ir_node, py_node) in self.nodes.iter().zip(py_nodes.iter()) {
            ir_node.meta.write_back(py, py_node)?;
        }
        Ok(())
    }
}

impl IrNode {
    /// Check if target contains a marker string.
    pub fn target_contains(&self, marker: &str) -> bool {
        self.op == NodeOp::CallFunction && self.target.contains(marker)
    }

    /// Get first arg if it's a node reference.
    pub fn first_node_arg(&self) -> Option<NodeId> {
        self.args.first().and_then(|a| match a {
            IrArg::Node(id) => Some(*id),
            _ => None,
        })
    }

    /// Is this a quantize_per_tensor call?
    pub fn is_quantize(&self) -> bool {
        self.target_contains("quantize_per_tensor") && !self.target_contains("dequantize")
    }

    /// Is this a dequantize_per_tensor call?
    pub fn is_dequantize(&self) -> bool {
        self.target_contains("dequantize_per_tensor")
    }

    /// Is this a TOSA RESCALE call?
    pub fn is_rescale(&self) -> bool {
        self.target_contains("RESCALE")
    }
}

/// Resolve a Python FX arg to an IrArg.
fn resolve_arg(py_arg: &Bound<'_, PyAny>, name_to_id: &HashMap<String, NodeId>) -> IrArg {
    // Check if it's a node (has .op attribute)
    if let Ok(name) = py_arg.getattr("name").and_then(|n| n.extract::<String>()) {
        if py_arg.getattr("op").is_ok() {
            if let Some(&id) = name_to_id.get(&name) {
                return IrArg::Node(id);
            }
        }
    }
    // Try literals
    if py_arg.is_none() { return IrArg::None; }
    if let Ok(v) = py_arg.extract::<bool>() { return IrArg::Bool(v); }
    if let Ok(v) = py_arg.extract::<i64>() { return IrArg::Int(v); }
    if let Ok(v) = py_arg.extract::<f64>() { return IrArg::Float(v); }
    if let Ok(v) = py_arg.extract::<String>() { return IrArg::Str(v); }
    // Try list/tuple
    if let Ok(iter) = py_arg.iter() {
        let items: Vec<IrArg> = iter
            .filter_map(|i| i.ok())
            .map(|i| resolve_arg(&i, name_to_id))
            .collect();
        if !items.is_empty() { return IrArg::List(items); }
    }
    IrArg::None
}

// ---------------------------------------------------------------------------
// NodeMeta — typed metadata extracted from a Python FX node
// ---------------------------------------------------------------------------

/// Metadata fields extracted from `node.meta`. The common fields are typed;
/// rare fields stay in the Python meta dict and are not touched.
#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    pub input_qparams: Option<HashMap<usize, QParam>>,
    pub output_qparams: Option<HashMap<usize, QParam>>,
    pub val_dtype: Option<String>,
    pub val_shape: Option<Vec<i64>>,
    pub tosa_dim_order: Option<Vec<usize>>,
    pub tosa_spatial_rank: Option<usize>,

    // Pre-pass snapshot of parent arg qparams (stored on cond/while nodes)
    pub pre_pass_arg_qparams: Option<HashMap<usize, QParam>>,

    // Track which fields were modified so write_back only touches what changed
    dirty: DirtyFlags,
}

#[derive(Debug, Clone, Default)]
struct DirtyFlags {
    input_qparams: bool,
    output_qparams: bool,
    tosa_dim_order: bool,
}

// ---------------------------------------------------------------------------
// QParam dict extraction helpers
// ---------------------------------------------------------------------------

fn extract_qparam_dict(py_dict: &Bound<'_, PyAny>) -> Option<HashMap<usize, QParam>> {
    if py_dict.is_none() || !py_dict.is_truthy().unwrap_or(false) {
        return None;
    }
    let mut map = HashMap::new();
    if let Ok(items) = py_dict.call_method0("items") {
        if let Ok(iter) = items.iter() {
            for item in iter.filter_map(|i| i.ok()) {
                if let (Ok(key), Ok(val)) = (
                    item.get_item(0).and_then(|k| k.extract::<usize>()),
                    item.get_item(1),
                ) {
                    map.insert(key, QParam::from_py(&val));
                }
            }
        }
    }
    if map.is_empty() { None } else { Some(map) }
}

pub(crate) fn qparam_dict_to_py<'py>(
    py: Python<'py>,
    map: &HashMap<usize, QParam>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (&idx, qp) in map {
        dict.set_item(idx, qp.to_py(py)?)?;
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// NodeMeta import / export
// ---------------------------------------------------------------------------

impl NodeMeta {
    /// Extract typed metadata from a Python FX node.
    pub fn from_fx(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        let meta = node.getattr("meta")?;

        let input_qparams = meta.get_item("input_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        let output_qparams = meta.get_item("output_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        let val = meta.get_item("val").ok();
        let val_dtype = val.as_ref()
            .and_then(|v| v.getattr("dtype").ok())
            .map(|d| format!("{d}"));
        let val_shape = val.as_ref()
            .and_then(|v| v.getattr("shape").ok())
            .and_then(|s| s.extract::<Vec<i64>>().ok());

        let tosa_dim_order = meta.get_item("tosa_dim_order")
            .ok()
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let tosa_spatial_rank = meta.get_item("tosa_spatial_rank")
            .ok()
            .and_then(|v| v.extract::<usize>().ok());

        let pre_pass_arg_qparams = meta.get_item("_pre_pass_arg_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        Ok(Self {
            input_qparams,
            output_qparams,
            val_dtype,
            val_shape,
            tosa_dim_order,
            tosa_spatial_rank,
            pre_pass_arg_qparams,
            dirty: DirtyFlags::default(),
        })
    }

    /// Write back only the changed metadata fields to a Python FX node.
    pub fn write_back(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
        let meta = node.getattr("meta")?;

        if self.dirty.output_qparams {
            match &self.output_qparams {
                Some(map) => meta.set_item("output_qparams", qparam_dict_to_py(py, map)?)?,
                None => { let _ = meta.del_item("output_qparams"); },
            }
        }

        if self.dirty.input_qparams {
            match &self.input_qparams {
                Some(map) => meta.set_item("input_qparams", qparam_dict_to_py(py, map)?)?,
                None => { let _ = meta.del_item("input_qparams"); },
            }
        }

        if self.dirty.tosa_dim_order {
            if let Some(ref order) = self.tosa_dim_order {
                let tuple = pyo3::types::PyTuple::new_bound(py, order);
                meta.set_item("tosa_dim_order", tuple)?;
            }
        }

        Ok(())
    }

    // --- Mutation helpers that set dirty flags ---

    pub fn set_output_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.output_qparams = Some(map);
        self.dirty.output_qparams = true;
    }

    pub fn set_input_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.input_qparams = Some(map);
        self.dirty.input_qparams = true;
    }

    pub fn set_single_output_qparam(&mut self, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(0, qp);
        self.set_output_qparams(map);
    }

    pub fn set_single_input_qparam(&mut self, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(0, qp);
        self.set_input_qparams(map);
    }

    pub fn set_both_qparams(&mut self, qp: QParam) {
        self.set_single_output_qparam(qp.clone());
        self.set_single_input_qparam(qp);
    }

    pub fn clear_output_qparams(&mut self) {
        self.output_qparams = None;
        self.dirty.output_qparams = true;
    }

    pub fn set_tosa_dim_order(&mut self, order: Vec<usize>) {
        self.tosa_dim_order = Some(order);
        self.dirty.tosa_dim_order = true;
    }

    /// Get the single output qparam (key 0), if present.
    pub fn output_qparam(&self) -> Option<&QParam> {
        self.output_qparams.as_ref()?.get(&0)
    }

    /// Get the single input qparam (key 0), if present.
    pub fn input_qparam(&self) -> Option<&QParam> {
        self.input_qparams.as_ref()?.get(&0)
    }
}
