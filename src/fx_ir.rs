//! Rust-native graph IR for FX graph manipulation.
//!
//! Phase 1: NodeMeta — typed metadata extraction (done).
//! Phase 2: IrGraph — full node topology imported from Python FX graph.
//! Phase 3: Graph surgery — replace_and_erase, write_back_with_surgery.
//! Phase 4: IrPass trait — pure Rust passes on IrGraph.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_control_flow_metadata::{NodeOp, QParam};

// ---------------------------------------------------------------------------
// IrGraph — full graph topology
// ---------------------------------------------------------------------------

pub type NodeId = usize;

/// Control-flow kind, if this node is a cond/while_loop call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfKind { Cond, While }

/// A node in the Rust IR graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    pub id: NodeId,
    pub op: NodeOp,
    pub name: String,
    pub target: String,
    pub args: Vec<IrArg>,
    pub meta: NodeMeta,
    pub alive: bool,
    /// Set for cond/while_loop call_function nodes.
    pub cf_kind: Option<CfKind>,
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

/// A hierarchical graph module: parent graph + named submodules.
#[derive(Debug)]
pub struct IrModule {
    pub parent: IrGraph,
    pub submodules: HashMap<String, IrGraph>,
}

impl IrModule {
    /// Import parent graph and all CF submodules.
    pub fn from_fx(py: Python<'_>, gm: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parent = IrGraph::from_fx(gm)?;
        let mut submodules = HashMap::new();
        // Find submodule names from cond/while node args
        for node in &parent.nodes {
            if node.cf_kind.is_none() || !node.alive { continue; }
            let sm_indices = match node.cf_kind {
                Some(CfKind::Cond) => vec![1, 2],   // if_fn, else_fn
                Some(CfKind::While) => vec![0, 1],   // cond_fn, body_fn
                None => vec![],
            };
            for &idx in &sm_indices {
                if let Some(IrArg::Node(sm_node_id)) = node.args.get(idx) {
                    let sm_name = &parent.nodes[*sm_node_id].target;
                    if !sm_name.is_empty() && !submodules.contains_key(sm_name) {
                        if let Ok(sm) = gm.call_method1("get_submodule", (sm_name,)) {
                            submodules.insert(sm_name.clone(), IrGraph::from_fx(&sm)?);
                        }
                    }
                }
            }
        }
        Ok(Self { parent, submodules })
    }

    /// Write back parent and all submodules.
    pub fn write_back(&self, py: Python<'_>, gm: &Bound<'_, PyAny>) -> PyResult<()> {
        self.parent.write_back(py, gm)?;
        for (name, ir) in &self.submodules {
            if let Ok(sm) = gm.call_method1("get_submodule", (name.as_str(),)) {
                ir.write_back(py, &sm)?;
            }
        }
        Ok(())
    }

    /// Write back with surgery for parent and all submodules.
    pub fn write_back_with_surgery(&self, py: Python<'_>, gm: &Bound<'_, PyAny>) -> PyResult<()> {
        self.parent.write_back_with_surgery(py, gm)?;
        for (name, ir) in &self.submodules {
            if let Ok(sm) = gm.call_method1("get_submodule", (name.as_str(),)) {
                ir.write_back_with_surgery(py, &sm)?;
            }
        }
        Ok(())
    }
}

/// The full graph: ordered list of nodes + lookup by name.
#[derive(Debug)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    name_to_id: HashMap<String, NodeId>,
    /// Reverse edges: users[node_id] = list of node IDs that reference this node in their args.
    users: Vec<Vec<NodeId>>,
    /// Replacement map: dead_id → replacement_id (recorded by replace_and_erase).
    replacements: HashMap<NodeId, NodeId>,
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

            let cf_kind = if op == NodeOp::CallFunction {
                if target.contains("cond") && target.contains("higher_order") { Some(CfKind::Cond) }
                else if target.contains("while_loop") { Some(CfKind::While) }
                else { None }
            } else { None };
            nodes.push(IrNode { id: i, op, name, target, args, meta, alive: true, cf_kind });
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

        Ok(Self { nodes, name_to_id, users, replacements: HashMap::new() })
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

    /// Iterate cond/while_loop nodes.
    pub fn cf_nodes(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|n| n.alive && n.cf_kind.is_some())
    }

    /// Get data args for a cond/while node (the carried/data arg list).
    /// cond: args[-1], while_loop: args[-2] + args[-1]
    pub fn cf_data_arg_ids(&self, cf_node: &IrNode) -> Vec<NodeId> {
        match cf_node.cf_kind {
            Some(CfKind::Cond) => {
                // Last arg is the data args list
                cf_node.args.last()
                    .and_then(|a| match a {
                        IrArg::List(items) => Some(items.iter()
                            .filter_map(|i| match i { IrArg::Node(id) => Some(*id), _ => None })
                            .collect()),
                        _ => None,
                    })
                    .unwrap_or_default()
            }
            Some(CfKind::While) => {
                // args[-2] + args[-1]
                let len = cf_node.args.len();
                if len < 2 { return Vec::new(); }
                let mut ids = Vec::new();
                for arg in &cf_node.args[len-2..] {
                    match arg {
                        IrArg::List(items) => {
                            for i in items {
                                if let IrArg::Node(id) = i { ids.push(*id); }
                            }
                        }
                        IrArg::Node(id) => ids.push(*id),
                        _ => {}
                    }
                }
                ids
            }
            None => Vec::new(),
        }
    }

    // --- Graph surgery ---

    /// Replace all references to `old_id` with `new_id` in args across the graph.
    pub fn replace_all_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        fn replace_in_arg(arg: &mut IrArg, old: NodeId, new: NodeId) {
            match arg {
                IrArg::Node(id) if *id == old => *id = new,
                IrArg::List(items) => {
                    for item in items { replace_in_arg(item, old, new); }
                }
                _ => {}
            }
        }
        for node in &mut self.nodes {
            if !node.alive { continue; }
            for arg in &mut node.args {
                replace_in_arg(arg, old_id, new_id);
            }
        }
        // Update users: move old's users to new
        let old_users = std::mem::take(&mut self.users[old_id]);
        for &user_id in &old_users {
            if !self.users[new_id].contains(&user_id) {
                self.users[new_id].push(user_id);
            }
        }
    }

    /// Mark a node as dead. It will be erased during write_back.
    pub fn erase(&mut self, id: NodeId) {
        self.nodes[id].alive = false;
    }

    /// Replace all uses of `old_id` with `new_id`, then erase `old_id`.
    pub fn replace_and_erase(&mut self, old_id: NodeId, new_id: NodeId) {
        self.replace_all_uses(old_id, new_id);
        self.erase(old_id);
        self.replacements.insert(old_id, new_id);
    }

    /// Insert a new node, returning its ID. The node is appended to the end;
    /// ordering relative to other nodes is handled during write_back.
    pub fn add_node(&mut self, op: NodeOp, name: String, target: String, args: Vec<IrArg>, meta: NodeMeta) -> NodeId {
        let id = self.nodes.len();
        // Update users for the new node's args
        for arg in &args {
            if let IrArg::Node(ref_id) = arg {
                if *ref_id < self.users.len() {
                    self.users[*ref_id].push(id);
                }
            }
        }
        self.users.push(Vec::new());
        self.name_to_id.insert(name.clone(), id);
        self.nodes.push(IrNode { id, op, name, target, args, meta, alive: true, cf_kind: None });
        id
    }

    /// Write back dirty metadata to the Python FX graph (metadata only, no surgery).
    pub fn write_back(&self, py: Python<'_>, gm: &Bound<'_, PyAny>) -> PyResult<()> {
        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();

        for (ir_node, py_node) in self.nodes.iter().zip(py_nodes.iter()) {
            if ir_node.alive {
                ir_node.meta.write_back(py, py_node)?;
            }
        }
        Ok(())
    }

    /// Write back metadata AND apply graph surgery to the Python FX graph.
    /// Surgery operations (replace_all_uses, erase) are applied using the
    /// recorded `replacements` map.
    pub fn write_back_with_surgery(
        &self,
        py: Python<'_>,
        gm: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();

        // Write metadata for alive nodes
        for (ir_node, py_node) in self.nodes.iter().zip(py_nodes.iter()) {
            if ir_node.alive {
                ir_node.meta.write_back(py, py_node)?;
            }
        }

        // Apply surgery: replace_all_uses + erase for dead nodes (reverse order
        // so dependents are erased before their sources)
        for ir_node in self.nodes.iter().rev() {
            if ir_node.alive || ir_node.id >= py_nodes.len() { continue; }
            let py_node = &py_nodes[ir_node.id];
            if let Some(&replacement_id) = self.replacements.get(&ir_node.id) {
                if replacement_id < py_nodes.len() {
                    py_node.call_method1("replace_all_uses_with", (&py_nodes[replacement_id],))?;
                }
            }
            graph.call_method1("erase_node", (py_node,))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RescaleInsert — deferred node creation instruction
// ---------------------------------------------------------------------------

/// Instruction to create a RESCALE node. Produced by pure Rust `_ir` functions,
/// applied by the Python wrapper via `insert_rescale_node`.
#[derive(Debug, Clone)]
pub struct RescaleInsert {
    /// Node to insert relative to (by name, for Python lookup).
    pub anchor_name: String,
    /// Input source node (by name).
    pub source_name: String,
    /// Source quantization domain.
    pub from_qp: QParam,
    /// Target quantization domain.
    pub to_qp: QParam,
    /// Insert after anchor (true) or before (false).
    pub after: bool,
    /// If set, replace all uses of this node with the new RESCALE.
    pub replace_node_name: Option<String>,
    /// If set, erase this node after replacement.
    pub erase_node_name: Option<String>,
}

impl RescaleInsert {
    /// Apply this instruction to a Python FX graph.
    pub fn apply<'py>(
        &self,
        py: Python<'py>,
        gm: &Bound<'py, PyAny>,
        py_nodes_by_name: &HashMap<String, Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let anchor = py_nodes_by_name.get(&self.anchor_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(self.anchor_name.clone()))?;
        let source = py_nodes_by_name.get(&self.source_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(self.source_name.clone()))?;

        let rescale = self.from_qp.insert_rescale_to(&self.to_qp, py, gm, anchor, source, self.after)?;

        if let Some(ref replace_name) = self.replace_node_name {
            if let Some(old_node) = py_nodes_by_name.get(replace_name) {
                old_node.call_method1("replace_all_uses_with", (&rescale,))?;
            }
        }
        if let Some(ref erase_name) = self.erase_node_name {
            if let Some(dead_node) = py_nodes_by_name.get(erase_name) {
                let graph = gm.getattr("graph")?;
                graph.call_method1("erase_node", (dead_node,))?;
            }
        }
        Ok(rescale)
    }
}

/// Build a name→Python node map for applying RescaleInserts.
pub fn build_py_node_map<'py>(gm: &Bound<'py, PyAny>) -> PyResult<HashMap<String, Bound<'py, PyAny>>> {
    let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
    let mut map = HashMap::new();
    for node in graph.getattr("nodes")?.iter()?.filter_map(|n| n.ok()) {
        let name: String = node.getattr("name")
            .and_then(|v| v.extract()).unwrap_or_default();
        map.insert(name, node);
    }
    Ok(map)
}

/// Instruction to create a DQ (dequantize_per_tensor) node.
#[derive(Debug, Clone)]
pub struct DqInsert {
    /// Compare node to insert before (by name).
    pub compare_node_name: String,
    /// Arg index in the compare node to replace.
    pub arg_idx: usize,
    /// Int source node (by name).
    pub source_name: String,
    /// Qparam for the DQ node.
    pub qp: QParam,
    /// Target float dtype string (e.g. "torch.float32").
    pub float_dtype: String,
}

impl DqInsert {
    /// Apply: create DQ node via FxNodeCreator, update compare arg.
    pub fn apply<'py>(
        &self,
        py: Python<'py>,
        gm: &Bound<'py, PyAny>,
        py_nodes_by_name: &HashMap<String, Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let compare = py_nodes_by_name.get(&self.compare_node_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(self.compare_node_name.clone()))?;
        let source = py_nodes_by_name.get(&self.source_name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(self.source_name.clone()))?;

        let creator = FxNodeCreator::new(py)?;
        let dq = creator.insert_dq(gm, compare, source, &self.qp, &self.float_dtype)?;
        compare.call_method1("update_arg", (self.arg_idx, &dq))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FxNodeCreator — create FX nodes from Rust without Python helpers
// ---------------------------------------------------------------------------

/// Creates FX nodes directly via PyO3, replacing _helpers.py functions.
pub struct FxNodeCreator<'py> {
    py: Python<'py>,
    rescale_op: Bound<'py, PyAny>,
    dq_op: Bound<'py, PyAny>,
}

impl<'py> FxNodeCreator<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        let exir_ops = py.import_bound("executorch.exir.dialects._ops")?.getattr("ops")?;
        let rescale_op = exir_ops.getattr("backend")?.getattr("tosa")?.getattr("RESCALE")?.getattr("default")?;
        let dq_op = exir_ops.getattr("edge")?.getattr("quantized_decomposed")?
            .getattr("dequantize_per_tensor")?.getattr("default")?;
        Ok(Self { py, rescale_op, dq_op })
    }

    /// Insert a node (call_function) before or after a reference node.
    fn insert_node(
        &self,
        gm: &Bound<'py, PyAny>,
        ref_node: &Bound<'py, PyAny>,
        target: &Bound<'py, PyAny>,
        args: &Bound<'py, PyAny>,
        after: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let graph = gm.getattr("graph")?;
        let ctx = if after {
            graph.call_method1("inserting_after", (ref_node,))?
        } else {
            graph.call_method1("inserting_before", (ref_node,))?
        };
        ctx.call_method0("__enter__")?;

        let kw = PyDict::new_bound(self.py);
        kw.set_item("args", args)?;
        let new_node = graph.call_method("call_function", (target,), Some(&kw))?;

        // Copy meta from ref_node, clear qparams
        let ref_meta = ref_node.getattr("meta")?;
        let copy_mod = self.py.import_bound("copy")?;
        let new_meta = copy_mod.call_method1("copy", (&ref_meta,))?;
        if new_meta.contains("input_qparams")? {
            new_meta.set_item("input_qparams", PyDict::new_bound(self.py))?;
        }
        if new_meta.contains("output_qparams")? {
            new_meta.set_item("output_qparams", PyDict::new_bound(self.py))?;
        }
        new_node.setattr("meta", &new_meta)?;

        let none = self.py.None().into_bound(self.py);
        ctx.call_method1("__exit__", (&none, &none, &none))?;
        Ok(new_node)
    }

    /// Insert a TOSA RESCALE node and set its FakeTensor metadata.
    pub fn insert_rescale(
        &self,
        gm: &Bound<'py, PyAny>,
        ref_node: &Bound<'py, PyAny>,
        input_node: &Bound<'py, PyAny>,
        from_qp: &QParam,
        to_qp: &QParam,
        after: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = self.py.import_bound("torch")?;
        let out_dtype = torch.getattr(&*to_qp.dtype_str.replace("torch.", ""))?;
        let scale_ratio = from_qp.scale / to_qp.scale;
        let scales = pyo3::types::PyList::new_bound(self.py, &[scale_ratio]);

        let args = pyo3::types::PyTuple::new_bound(self.py, &[
            input_node.clone(),
            out_dtype.clone().into_any(),
            scales.into_any(),
            from_qp.zp.into_py(self.py).into_bound(self.py),
            to_qp.zp.into_py(self.py).into_bound(self.py),
        ]);

        let rescale = self.insert_node(gm, ref_node, &self.rescale_op, &args.into_any(), after)?;

        // Set FakeTensor: use input node's val as base, restamp with output dtype
        let input_val = input_node.getattr("meta")?.get_item("val").ok();
        if let Some(ref base) = input_val {
            if base.hasattr("new_empty").unwrap_or(false) {
                let shape = base.getattr("shape")?;
                let kw = PyDict::new_bound(self.py);
                kw.set_item("dtype", &out_dtype)?;
                let new_val = base.call_method("new_empty", (&shape,), Some(&kw))?;
                rescale.getattr("meta")?.set_item("val", &new_val)?;
            }
        }
        // Clear special dtype marker
        let _ = rescale.getattr("meta")?.del_item("tosa_special_dtype");

        Ok(rescale)
    }

    /// Insert a dequantize_per_tensor node.
    pub fn insert_dq(
        &self,
        gm: &Bound<'py, PyAny>,
        ref_node: &Bound<'py, PyAny>,
        input_node: &Bound<'py, PyAny>,
        qp: &QParam,
        float_dtype_str: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = self.py.import_bound("torch")?;
        let qp_dtype = torch.getattr(&*qp.dtype_str.replace("torch.", ""))?;
        let float_dtype = torch.getattr(&*float_dtype_str.replace("torch.", ""))?;

        let args = pyo3::types::PyTuple::new_bound(self.py, &[
            input_node.clone(),
            qp.scale.into_py(self.py).into_bound(self.py),
            qp.zp.into_py(self.py).into_bound(self.py),
            qp.qmin.into_py(self.py).into_bound(self.py),
            qp.qmax.into_py(self.py).into_bound(self.py),
            qp_dtype.into_any(),
        ]);

        let dq = self.insert_node(gm, ref_node, &self.dq_op, &args.into_any(), false)?;

        // Set FakeTensor with float dtype
        let input_val = input_node.getattr("meta")?.get_item("val").ok();
        if let Some(ref base) = input_val {
            if base.hasattr("to").unwrap_or(false) {
                let kw = PyDict::new_bound(self.py);
                kw.set_item("dtype", &float_dtype)?;
                let new_val = base.call_method("to", (), Some(&kw))?;
                dq.getattr("meta")?.set_item("val", &new_val)?;
            }
        }
        // Set qparams
        let iqp = PyDict::new_bound(self.py);
        iqp.set_item(0i64, qp.to_py(self.py)?)?;
        dq.getattr("meta")?.set_item("input_qparams", &iqp)?;
        dq.getattr("meta")?.set_item("output_qparams", PyDict::new_bound(self.py))?;

        Ok(dq)
    }
}

impl<'py> FxNodeCreator<'py> {
    /// Resolve an op target by dotted name (e.g. "aten.sum.dim_IntList").
    pub fn resolve_op(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        // Try exir_ops first (executorch dialect)
        let parts: Vec<&str> = name.split('.').collect();
        let exir_ops = self.py.import_bound("executorch.exir.dialects._ops")?.getattr("ops")?;

        // Try edge dialect: ops.edge.aten.X.default
        if parts.len() >= 2 {
            let ns = parts[0]; // e.g. "aten"
            let op_name = parts[1]; // e.g. "sum"
            let overload = if parts.len() >= 3 { parts[2] } else { "default" };

            if let Ok(edge) = exir_ops.getattr("edge") {
                if let Ok(ns_obj) = edge.getattr(ns) {
                    if let Ok(op) = ns_obj.getattr(op_name) {
                        if let Ok(overload_obj) = op.getattr(overload) {
                            return Ok(overload_obj);
                        }
                    }
                }
            }
        }

        // Fallback: try torch.ops directly
        let torch_ops = self.py.import_bound("torch")?.getattr("ops")?;
        if parts.len() >= 2 {
            let ns_obj = torch_ops.getattr(parts[0])?;
            let op = ns_obj.getattr(parts[1])?;
            if parts.len() >= 3 {
                return op.getattr(parts[2]);
            }
            return op.getattr("default");
        }

        Err(pyo3::exceptions::PyValueError::new_err(format!("Cannot resolve op: {name}")))
    }

    /// Insert a generic call_function node with resolved op target.
    pub fn insert_call_function(
        &self,
        gm: &Bound<'py, PyAny>,
        ref_node: &Bound<'py, PyAny>,
        op_name: &str,
        args: &Bound<'py, PyAny>,
        after: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let target = self.resolve_op(op_name)?;
        self.insert_node(gm, ref_node, &target, args, after)
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

    /// Check if all args are non-node (literals/None) — a zero-input const source.
    pub fn is_zero_input_const(&self) -> bool {
        self.args.iter().all(|a| !matches!(a, IrArg::Node(_)))
    }

    /// Parse Q/DQ literal args into a QParam.
    /// Q/DQ nodes have args: (input, scale, zp, qmin, qmax, dtype).
    pub fn parse_qdq_qparam(&self) -> Option<QParam> {
        if self.args.len() < 6 { return None; }
        let scale = match &self.args[1] { IrArg::Float(v) => *v, IrArg::Int(v) => *v as f64, _ => return None };
        let zp = match &self.args[2] { IrArg::Int(v) => *v, _ => return None };
        let qmin = match &self.args[3] { IrArg::Int(v) => *v, _ => return None };
        let qmax = match &self.args[4] { IrArg::Int(v) => *v, _ => return None };
        // dtype arg may not resolve cleanly — check if it's a string or use target info
        let dtype_str = match &self.args[5] {
            IrArg::Str(s) => s.clone(),
            _ => {
                if qmin == -128 && qmax == 127 { "torch.int8".into() }
                else if qmin == -32768 { "torch.int16".into() }
                else { return None; }
            }
        };
        Some(QParam { scale, zp, qmin, qmax, dtype_str })
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

/// Rust-side representation of a FakeTensor's core metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeTensorMeta {
    pub shape: Vec<i64>,
    pub dtype: String,
}

impl FakeTensorMeta {
    pub fn from_py(val: &Bound<'_, PyAny>) -> Option<Self> {
        let shape = val.getattr("shape").ok()?.extract::<Vec<i64>>().ok()?;
        let dtype = val.getattr("dtype").ok().map(|d| format!("{d}"))?;
        Some(Self { shape, dtype })
    }

    /// Reconstruct as a Python FakeTensor using an existing FakeTensor as the base.
    /// This preserves the FakeTensorMode so all tensors in the graph share the same mode.
    pub fn to_py_from<'py>(&self, py: Python<'py>, base: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch")?;
        let dtype = torch.getattr(&*self.dtype.replace("torch.", ""))?;
        let shape = pyo3::types::PyList::new_bound(py, &self.shape);
        let kw = PyDict::new_bound(py);
        kw.set_item("dtype", &dtype)?;
        base.call_method("new_empty", (&shape,), Some(&kw))
    }

    /// Reconstruct as a Python FakeTensor with a fresh FakeTensorMode.
    /// Only use when no existing FakeTensor is available.
    pub fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch")?;
        let dtype = torch.getattr(&*self.dtype.replace("torch.", ""))?;
        let shape = pyo3::types::PyTuple::new_bound(py, &self.shape);
        let kw = PyDict::new_bound(py);
        kw.set_item("dtype", &dtype)?;
        let mode = py.import_bound("torch._subclasses.fake_tensor")?
            .getattr("FakeTensorMode")?.call0()?;
        let none = py.None().into_bound(py);
        mode.call_method0("__enter__")?;
        let fake = torch.call_method("empty", (&shape,), Some(&kw));
        let _ = mode.call_method1("__exit__", (&none, &none, &none));
        fake
    }

    pub fn rank(&self) -> usize { self.shape.len() }
}

/// Metadata fields extracted from `node.meta`. The common fields are typed;
/// rare fields stay in the Python meta dict and are not touched.
#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    pub input_qparams: Option<HashMap<usize, QParam>>,
    pub output_qparams: Option<HashMap<usize, QParam>>,
    pub val: Option<FakeTensorMeta>,
    pub tosa_dim_order: Option<Vec<usize>>,
    pub tosa_spatial_rank: Option<usize>,
    pub pre_pass_arg_qparams: Option<HashMap<usize, QParam>>,
    pub table_input_qparam: Option<QParam>,
    pub table_output_qparam: Option<QParam>,
    pub(crate) dirty: DirtyFlags,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirtyFlags {
    pub(crate) input_qparams: bool,
    pub(crate) output_qparams: bool,
    pub(crate) tosa_dim_order: bool,
    pub(crate) pre_pass_arg_qparams: bool,
    pub(crate) val: bool,
    pub(crate) table_qparams: bool,
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

        let val = meta.get_item("val").ok()
            .and_then(|v| FakeTensorMeta::from_py(&v));

        let tosa_dim_order = meta.get_item("tosa_dim_order")
            .ok()
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let tosa_spatial_rank = meta.get_item("tosa_spatial_rank")
            .ok()
            .and_then(|v| v.extract::<usize>().ok());

        let pre_pass_arg_qparams = meta.get_item("_pre_pass_arg_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        let table_input_qparam = meta.get_item("arm_saved_table_input_qparam")
            .ok()
            .filter(|v| !v.is_none())
            .map(|v| QParam::from_py(&v));
        let table_output_qparam = meta.get_item("arm_saved_table_output_qparam")
            .ok()
            .filter(|v| !v.is_none())
            .map(|v| QParam::from_py(&v));

        Ok(Self {
            input_qparams,
            output_qparams,
            val,
            tosa_dim_order,
            tosa_spatial_rank,
            pre_pass_arg_qparams,
            table_input_qparam,
            table_output_qparam,
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

        if self.dirty.pre_pass_arg_qparams {
            if let Some(ref map) = self.pre_pass_arg_qparams {
                meta.set_item("_pre_pass_arg_qparams", qparam_dict_to_py(py, map)?)?;
            }
        }

        if self.dirty.table_qparams {
            if let Some(ref qp) = self.table_input_qparam {
                meta.set_item("arm_saved_table_input_qparam", qp.to_py(py)?)?;
            }
            if let Some(ref qp) = self.table_output_qparam {
                meta.set_item("arm_saved_table_output_qparam", qp.to_py(py)?)?;
            }
        }

        if self.dirty.val {
            if let Some(ref ftm) = self.val {
                // Use existing val as base to preserve FakeTensorMode
                let existing = meta.get_item("val").ok();
                let new_val = if let Some(ref base) = existing {
                    ftm.to_py_from(py, base)?
                } else {
                    ftm.to_py(py)?
                };
                meta.set_item("val", new_val)?;
            }
        }

        Ok(())
    }

    // --- Accessors ---

    pub fn val_shape(&self) -> Option<&[i64]> {
        self.val.as_ref().map(|v| v.shape.as_slice())
    }

    pub fn val_dtype(&self) -> Option<&str> {
        self.val.as_ref().map(|v| v.dtype.as_str())
    }

    pub fn rank(&self) -> Option<usize> {
        self.val.as_ref().map(|v| v.rank())
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

    pub fn set_val(&mut self, ftm: FakeTensorMeta) {
        self.val = Some(ftm);
        self.dirty.val = true;
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

// ===========================================================================
// IrPass — pure Rust pass trait for IrGraph
// ===========================================================================

/// A pure Rust compiler pass that operates on IrGraph.
/// No <'py>, no PyResult, no PyO3 in the pass body.
pub trait IrPass {
    fn name(&self) -> &str;

    /// Run the pass on the graph. Returns true if modified.
    fn run(&self, graph: &mut IrGraph) -> bool;
}

/// A node creation instruction emitted by an IrPass.
/// Applied to the Python FX graph during write_back.
#[derive(Debug, Clone)]
pub enum NodeInsert {
    /// Insert a TOSA RESCALE node.
    Rescale(RescaleInsert),
    /// Insert a dequantize_per_tensor node.
    Dq(DqInsert),
    /// Insert a generic call_function node.
    CallFunction {
        /// Target op string (e.g. "tosa.TRANSPOSE.default")
        target: String,
        /// Insert before this node (by name)
        before_node: String,
        /// Args: node references by name, or literal values
        args: Vec<IrArg>,
        /// Copy metadata from this node (by name)
        copy_meta_from: String,
        /// If set, replace all uses of this node with the new one
        replace_node: Option<String>,
    },
}

/// Result of running an IrPass: metadata changes are in the IrGraph,
/// structural changes (surgery + node creation) are returned here.
#[derive(Default)]
pub struct IrPassResult {
    pub modified: bool,
    pub inserts: Vec<NodeInsert>,
}

/// Run a sequence of IrPasses on an IrGraph imported from a Python GraphModule.
/// Handles the full lifecycle: import → passes → write_back + apply inserts.
pub fn run_ir_passes<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    passes: &[&dyn IrPass],
) -> PyResult<bool> {
    let mut ir = IrGraph::from_fx(gm)?;
    let mut any_modified = false;
    let mut all_inserts: Vec<NodeInsert> = Vec::new();

    for pass in passes {
        let before = ir.nodes.iter().filter(|n| n.alive).count();
        let modified = pass.run(&mut ir);
        let after = ir.nodes.iter().filter(|n| n.alive).count();
        if modified {
            any_modified = true;
            crate::runner_logging::log_line(format_args!(
                "[IrPassRunner] {} — modified, nodes: {} -> {}", pass.name(), before, after
            ));
        }
    }

    if any_modified {
        ir.write_back_with_surgery(py, gm)?;
        // Apply node inserts
        if !all_inserts.is_empty() {
            let py_map = build_py_node_map(gm)?;
            for insert in &all_inserts {
                match insert {
                    NodeInsert::Rescale(ri) => { ri.apply(py, gm, &py_map)?; },
                    NodeInsert::Dq(di) => { di.apply(py, gm, &py_map)?; },
                    NodeInsert::CallFunction { .. } => {
                        // TODO: implement generic call_function creation
                    },
                }
            }
        }
        let graph = gm.getattr("graph")?;
        graph.call_method0("eliminate_dead_code")?;
        graph.call_method0("lint")?;
        gm.call_method0("recompile")?;
    }

    Ok(any_modified)
}

// ===========================================================================
// IrArg convenience methods
// ===========================================================================

impl IrArg {
    pub fn as_int(&self) -> Option<i64> {
        match self { IrArg::Int(v) => Some(*v), _ => None }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self { IrArg::Float(v) => Some(*v), _ => None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { IrArg::Bool(v) => Some(*v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { IrArg::Str(s) => Some(s), _ => None }
    }
    pub fn as_node(&self) -> Option<NodeId> {
        match self { IrArg::Node(id) => Some(*id), _ => None }
    }
    pub fn as_int_list(&self) -> Option<Vec<i64>> {
        match self {
            IrArg::List(items) => {
                items.iter().map(|a| a.as_int()).collect()
            }
            _ => None,
        }
    }
}
