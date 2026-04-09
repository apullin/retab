//! Rust-native graph IR for FX graph manipulation.
//!
//! Phase 1: NodeMeta — typed metadata extraction (done).
//! Phase 2: IrGraph — full node topology imported from Python FX graph.
//! Phase 3: Graph surgery — replace_and_erase, write_back_with_surgery.
//! Phase 4: IrPass trait — pure Rust passes on IrGraph.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use crate::fx_ep_effects::{FxEpApplyContext, FxEpEffectBatch, FxEpInputKind};
pub(crate) use crate::fx_control_flow_metadata::{NodeOp, QParam};
use crate::store::TensorData;
use crate::tensor_data::extract_tensor_data;

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
    /// Keyword arguments (e.g. dtype for _to_copy). Stored separately from args.
    pub kwargs: HashMap<String, IrArg>,
    pub meta: NodeMeta,
    pub tensor: Option<NodeTensor>,
    pub alive: bool,
    /// Set for cond/while_loop call_function nodes.
    pub cf_kind: Option<CfKind>,
    /// True if args were modified and need write-back to Python.
    pub(crate) args_dirty: bool,
    /// True when write-back must rebuild the whole args tuple because literal
    /// values changed. Pure node-reference rewrites can use replace_input_with.
    pub(crate) args_rebuild_required: bool,
    /// Optional node whose Python meta dict should seed a newly-created node.
    /// Defaults to the insertion anchor when unset.
    pub(crate) meta_copy_source: Option<NodeId>,
}

#[derive(Debug, Clone)]
pub enum TensorSource {
    ExportedProgramInput(PlaceholderTensorBinding),
    StateDict(String),
    GraphModuleAttr(String),
}

#[derive(Debug, Clone)]
pub struct NodeTensor {
    pub source: TensorSource,
    pub data: TensorData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceholderTensorBinding {
    pub input_kind: FxEpInputKind,
    pub target_name: String,
}

/// An argument to a node — either a reference to another node or a literal.
#[derive(Debug, Clone)]
pub enum IrArg {
    Node(NodeId),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// A torch.dtype value (e.g. "torch.int8"). Resolves to the Python dtype
    /// object during write-back, unlike Str which stays as a string.
    Dtype(String),
    List(Vec<IrArg>),
    None,
}

/// A hierarchical graph module: parent graph + named submodules.
#[derive(Debug)]
pub struct IrModule {
    pub parent: IrGraph,
    pub submodules: HashMap<String, IrGraph>,
}

fn build_placeholder_tensor_bindings(
    exported_program: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, PlaceholderTensorBinding>> {
    let mut placeholder_bindings = HashMap::new();
    let graph_signature = exported_program.getattr("graph_signature")?;
    for (attr_name, input_kind) in [
        ("inputs_to_parameters", FxEpInputKind::Parameter),
        ("inputs_to_buffers", FxEpInputKind::Buffer),
        (
            "inputs_to_lifted_tensor_constants",
            FxEpInputKind::LiftedTensorConstant,
        ),
    ] {
        if let Ok(mapping) = graph_signature.getattr(attr_name) {
            if let Ok(dict) = mapping.downcast::<PyDict>() {
                for (k, v) in dict.iter() {
                    if let (Ok(placeholder_name), Ok(target_name)) =
                        (k.extract::<String>(), v.extract::<String>())
                    {
                        placeholder_bindings.insert(
                            placeholder_name,
                            PlaceholderTensorBinding {
                                input_kind,
                                target_name,
                            },
                        );
                    }
                }
            } else if let Ok(items) = mapping.call_method0("items") {
                for item in items.iter()? {
                    let item = item?;
                    if let Ok((placeholder_name, target_name)) =
                        item.extract::<(String, String)>()
                    {
                        placeholder_bindings.insert(
                            placeholder_name,
                            PlaceholderTensorBinding {
                                input_kind,
                                target_name,
                            },
                        );
                    }
                }
            }
        }
    }
    Ok(placeholder_bindings)
}

fn extract_node_tensor(
    gm: &Bound<'_, PyAny>,
    py_node: &Bound<'_, PyAny>,
    name: &str,
    target: &str,
    state_dict: Option<&Bound<'_, PyAny>>,
    placeholder_bindings: &HashMap<String, PlaceholderTensorBinding>,
) -> Option<NodeTensor> {
    let op: String = py_node.getattr("op").ok()?.extract().ok()?;
    match op.as_str() {
        "placeholder" => {
            let binding = placeholder_bindings.get(name)?;
            if let Some(state_dict) = state_dict {
                if let Ok(value) = state_dict.get_item(binding.target_name.as_str()) {
                    if let Ok(data) = extract_tensor_data(&value) {
                        return Some(NodeTensor {
                            source: TensorSource::ExportedProgramInput(binding.clone()),
                            data,
                        });
                    }
                }
            }
            if let Ok(value) = gm.getattr(binding.target_name.as_str()) {
                if let Ok(data) = extract_tensor_data(&value) {
                    return Some(NodeTensor {
                        source: TensorSource::ExportedProgramInput(binding.clone()),
                        data,
                    });
                }
            }
            None
        }
        "get_attr" => {
            let value = gm.getattr(target).ok()?;
            if value.getattr("graph").is_ok() {
                return None;
            }
            let data = extract_tensor_data(&value).ok()?;
            Some(NodeTensor {
                source: TensorSource::GraphModuleAttr(target.to_string()),
                data,
            })
        }
        _ => None,
    }
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
    /// Explicit position anchors: new_node_id → original_node_id.
    /// During write_back_with_surgery, nodes with anchors are inserted before
    /// the original node's position (like replacements, but without implying
    /// a use-replacement relationship).  Passes call `anchor_to` for
    /// intermediate nodes that must appear near a specific original node.
    anchors: HashMap<NodeId, NodeId>,
    /// Explicit after-anchors for nodes that must be defined after a live
    /// producer while still preceding rewired consumers.
    anchors_after: HashMap<NodeId, NodeId>,
}

impl IrGraph {
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            name_to_id: HashMap::new(),
            users: Vec::new(),
            replacements: HashMap::new(),
            anchors: HashMap::new(),
            anchors_after: HashMap::new(),
        }
    }

    pub fn from_fx_with_tensors(
        gm: &Bound<'_, PyAny>,
        exported_program: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();

        let (placeholder_bindings, state_dict) = if let Some(ep) = exported_program {
            (
                build_placeholder_tensor_bindings(ep)?,
                ep.getattr("state_dict").ok(),
            )
        } else {
            (HashMap::new(), None)
        };

        let mut name_to_id = HashMap::new();
        for (i, node) in py_nodes.iter().enumerate() {
            let name: String = node.getattr("name")
                .and_then(|v| v.extract()).unwrap_or_default();
            name_to_id.insert(name, i);
        }

        let mut nodes = Vec::with_capacity(py_nodes.len());
        for (i, py_node) in py_nodes.iter().enumerate() {
            let name: String = py_node.getattr("name")
                .and_then(|v| v.extract()).unwrap_or_default();
            let target: String = py_node.getattr("target")
                .and_then(|t| t.str().map(|s| s.to_string()))
                .unwrap_or_default();
            let op = NodeOp::of(py_node);
            let meta = NodeMeta::from_fx(py_node)?;
            let tensor = extract_node_tensor(
                gm,
                py_node,
                &name,
                &target,
                state_dict.as_ref(),
                &placeholder_bindings,
            );

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
            let mut kwargs = HashMap::new();
            if let Ok(py_kwargs) = crate::fx_graph::FxGraphModule::node_kwargs(py_node) {
                if !py_kwargs.is_none() {
                    if let Ok(items) = py_kwargs.call_method0("items") {
                        if let Ok(iter) = items.iter() {
                            for item in iter.filter_map(|i| i.ok()) {
                                if let (Ok(key), Ok(val)) = (
                                    item.get_item(0).and_then(|k| k.extract::<String>()),
                                    item.get_item(1),
                                ) {
                                    kwargs.insert(key, resolve_arg(&val, &name_to_id));
                                }
                            }
                        }
                    }
                }
            }
            nodes.push(IrNode {
                id: i,
                op,
                name,
                target,
                args,
                kwargs,
                meta,
                tensor,
                alive: true,
                cf_kind,
                args_dirty: false,
                args_rebuild_required: false,
                meta_copy_source: None,
            });
        }

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

        Ok(Self {
            nodes,
            name_to_id,
            users,
            replacements: HashMap::new(),
            anchors: HashMap::new(),
            anchors_after: HashMap::new(),
        })
    }

    /// Import from a Python FX GraphModule.
    pub fn from_fx(gm: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_fx_with_tensors(gm, None)
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

    /// Update a node's args and mark dirty for write-back.
    pub fn set_args(&mut self, id: NodeId, args: Vec<IrArg>) {
        self.nodes[id].args = args;
        self.nodes[id].args_dirty = true;
        self.nodes[id].args_rebuild_required = true;
        self.rebuild_users();
    }

    /// Add a node with kwargs (e.g. dtype for _to_copy).
    pub fn add_node_with_kwargs(
        &mut self,
        op: NodeOp,
        name: String,
        target: String,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_node(op, name, target, args, meta);
        self.nodes[id].kwargs = kwargs;
        id
    }

    /// Update a single arg by index and mark dirty.
    pub fn set_arg(&mut self, id: NodeId, index: usize, value: IrArg) {
        if index < self.nodes[id].args.len() {
            let rebuild_required = !matches!(value, IrArg::Node(_));
            self.nodes[id].args[index] = value;
            self.nodes[id].args_dirty = true;
            self.nodes[id].args_rebuild_required |= rebuild_required;
            self.rebuild_users();
        }
    }

    /// Update a kwarg and mark dirty for write-back.
    pub fn set_kwarg(&mut self, id: NodeId, key: impl Into<String>, value: IrArg) {
        self.nodes[id].kwargs.insert(key.into(), value);
        self.nodes[id].args_dirty = true;
        self.nodes[id].args_rebuild_required = true;
    }

    /// Get users (nodes that reference this node in their args).
    pub fn users(&self, id: NodeId) -> &[NodeId] {
        &self.users[id]
    }

    pub fn live_user_ids(&self, id: NodeId) -> Vec<NodeId> {
        fn arg_contains_node(arg: &IrArg, target: NodeId) -> bool {
            match arg {
                IrArg::Node(id) => *id == target,
                IrArg::List(items) => items.iter().any(|item| arg_contains_node(item, target)),
                _ => false,
            }
        }

        self.nodes
            .iter()
            .filter(|node| node.alive)
            .filter(|node| {
                node.args.iter().any(|arg| arg_contains_node(arg, id))
                    || node.kwargs.values().any(|arg| arg_contains_node(arg, id))
            })
            .map(|node| node.id)
            .collect()
    }

    pub fn has_live_users(&self, id: NodeId) -> bool {
        !self.live_user_ids(id).is_empty()
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
        fn replace_in_arg(arg: &mut IrArg, old: NodeId, new: NodeId) -> bool {
            match arg {
                IrArg::Node(id) if *id == old => {
                    *id = new;
                    true
                }
                IrArg::List(items) => {
                    let mut changed = false;
                    for item in items {
                        changed |= replace_in_arg(item, old, new);
                    }
                    changed
                }
                _ => false,
            }
        }
        for node in &mut self.nodes {
            if !node.alive { continue; }
            let mut changed = false;
            for arg in &mut node.args {
                changed |= replace_in_arg(arg, old_id, new_id);
            }
            if changed {
                node.args_dirty = true;
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

    /// Replace uses of `old_id` with `new_id` in a single user node.
    pub fn replace_input_use(&mut self, user_id: NodeId, old_id: NodeId, new_id: NodeId) -> bool {
        fn replace_in_arg(arg: &mut IrArg, old: NodeId, new: NodeId) -> bool {
            match arg {
                IrArg::Node(id) if *id == old => {
                    *id = new;
                    true
                }
                IrArg::List(items) => {
                    let mut changed = false;
                    for item in items {
                        changed |= replace_in_arg(item, old, new);
                    }
                    changed
                }
                _ => false,
            }
        }

        let mut changed = false;
        if let Some(node) = self.nodes.get_mut(user_id) {
            if !node.alive {
                return false;
            }
            for arg in &mut node.args {
                changed |= replace_in_arg(arg, old_id, new_id);
            }
            for arg in node.kwargs.values_mut() {
                changed |= replace_in_arg(arg, old_id, new_id);
            }
            if changed {
                node.args_dirty = true;
            }
        }
        if changed {
            self.rebuild_users();
        }
        changed
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

    /// Record a position anchor: during write_back_with_surgery, `new_id`
    /// will be inserted before `original_id`'s position in the Python graph.
    /// Use this for intermediate nodes that must appear near a specific
    /// original node (e.g., view_copy/expand_copy that feed a replacement
    /// where node).
    pub fn anchor_to(&mut self, new_id: NodeId, original_id: NodeId) {
        self.anchors.insert(new_id, original_id);
    }

    /// Record an after-anchor for a new node that consumes a live producer.
    pub fn anchor_after(&mut self, new_id: NodeId, original_id: NodeId) {
        self.anchors_after.insert(new_id, original_id);
    }

    /// Override the Python metadata source used when creating a new node.
    pub fn copy_meta_from(&mut self, new_id: NodeId, source_id: NodeId) {
        if new_id < self.nodes.len() && source_id < self.nodes.len() {
            self.nodes[new_id].meta_copy_source = Some(source_id);
        }
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
        self.nodes.push(IrNode { id, op, name, target, args, kwargs: HashMap::new(), meta, tensor: None, alive: true, cf_kind: None, args_dirty: false, args_rebuild_required: false, meta_copy_source: None });
        id
    }

    fn rebuild_users(&mut self) {
        fn collect_users_from_arg(arg: &IrArg, user_id: NodeId, users: &mut [Vec<NodeId>]) {
            match arg {
                IrArg::Node(id) => {
                    if *id < users.len() && !users[*id].contains(&user_id) {
                        users[*id].push(user_id);
                    }
                }
                IrArg::List(items) => {
                    for item in items {
                        collect_users_from_arg(item, user_id, users);
                    }
                }
                _ => {}
            }
        }

        self.users = vec![Vec::new(); self.nodes.len()];
        for node in &self.nodes {
            if !node.alive {
                continue;
            }
            for arg in &node.args {
                collect_users_from_arg(arg, node.id, &mut self.users);
            }
            for arg in node.kwargs.values() {
                collect_users_from_arg(arg, node.id, &mut self.users);
            }
        }
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
    /// Handles three cases:
    /// 1. Metadata updates on alive original nodes
    /// 2. Creation of new Python FX nodes for IrNodes added by passes
    /// 3. replace_all_uses + erase for dead original nodes
    pub fn write_back_with_surgery(
        &self,
        py: Python<'_>,
        gm: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        fn resolve_effective_anchor(graph: &IrGraph, node_id: NodeId) -> (Option<NodeId>, bool) {
            let mut seen = std::collections::HashSet::new();
            let mut insert_after_first = false;
            let mut current = if let Some((&old_id, _)) =
                graph.replacements.iter().find(|(_, &new_id)| new_id == node_id)
            {
                Some(old_id)
            } else if let Some(&anchor_id) = graph.anchors_after.get(&node_id) {
                insert_after_first = true;
                Some(anchor_id)
            } else {
                graph.anchors.get(&node_id).copied()
            };

            while let Some(anchor_id) = current {
                if !seen.insert(anchor_id) {
                    return (Some(anchor_id), insert_after_first);
                }
                current = graph.replacements.iter()
                    .find(|(_, &new_id)| new_id == anchor_id)
                    .map(|(&old_id, _)| old_id)
                    .or_else(|| graph.anchors_after.get(&anchor_id).copied())
                    .or_else(|| graph.anchors.get(&anchor_id).copied());
                if current.is_none() {
                    return (Some(anchor_id), insert_after_first);
                }
            }

            (None, false)
        }

        let graph = if gm.hasattr("graph")? { gm.getattr("graph")? } else { gm.clone() };
        let py_nodes: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?
            .iter()?.filter_map(|n| n.ok()).collect();
        let original_count = py_nodes.len();

        // Write metadata for alive original nodes (args deferred until after new node creation)
        for (ir_node, py_node) in self.nodes.iter().zip(py_nodes.iter()) {
            if ir_node.alive {
                ir_node.meta.write_back(py, py_node)?;
            }
        }

        // Build extended lookup: original nodes + slots for new nodes
        let mut all_py: Vec<Option<Bound<'_, PyAny>>> = py_nodes.iter()
            .map(|n| Some(n.clone()))
            .collect();

        // Create Python FX nodes for IrNodes added by passes (id >= original_count).
        // Multi-pass: cross-pass replace_all_uses can make earlier new nodes
        // reference later new nodes. We iterate until all alive nodes are created.
        if self.nodes.len() > original_count {
            let creator = FxNodeCreator::new(py)?;
            let copy_mod = py.import_bound("copy")?;
            let mut last_for_anchor: HashMap<(NodeId, bool), Bound<'_, PyAny>> = HashMap::new();

            // Pre-fill slots for all new nodes
            for ir_node in &self.nodes[original_count..] {
                all_py.push(None);
            }

            // Iterate until no more nodes can be created
            let mut progress = true;
            while progress {
                progress = false;
                for idx in original_count..self.nodes.len() {
                    let ir_node = &self.nodes[idx];
                    if !ir_node.alive || all_py[idx].is_some() { continue; }

                    // Check if all node-arg dependencies are resolved
                    let deps_ready = ir_node.args.iter().all(|a| match a {
                        IrArg::Node(dep_id) => all_py.get(*dep_id).and_then(|o| o.as_ref()).is_some(),
                        IrArg::List(items) => items.iter().all(|item| match item {
                            IrArg::Node(dep_id) => all_py.get(*dep_id).and_then(|o| o.as_ref()).is_some(),
                            _ => true,
                        }),
                        _ => true,
                    });
                    if !deps_ready { continue; }

                    let ir_node = &self.nodes[idx]; // reborrow
                let (anchor_id, insert_after_first) = resolve_effective_anchor(self, ir_node.id);

                // Insert in anchor-chain order: use last_for_anchor to chain nodes
                // sharing the same anchor and placement side. For the first node
                // in a chain, insert before or after the anchor as requested.
                let (ref_node, insert_after) = if let Some(aid) = anchor_id {
                    let anchor_key = (aid, insert_after_first);
                    if let Some(prev) = last_for_anchor.get(&anchor_key) {
                        (prev.clone(), true)
                    } else if let Some(Some(a)) = all_py.get(aid) {
                        (a.clone(), insert_after_first)
                    } else {
                        (find_output_node(&all_py), false)
                    }
                } else {
                    (find_output_node(&all_py), false)
                };

                let py_args: Vec<Bound<'_, PyAny>> = ir_node.args.iter()
                    .map(|a| ir_arg_to_py(py, a, &all_py))
                    .collect::<PyResult<_>>()?;
                let py_tuple = pyo3::types::PyTuple::new_bound(py, &py_args);

                // Resolve the Python node used for the initial meta copy. This
                // defaults to the position anchor, but passes can override it
                // when insertion position and metadata source differ.
                let meta_src_node = ir_node.meta_copy_source
                    .or(anchor_id)
                    .and_then(|aid| all_py.get(aid))
                    .and_then(|opt| opt.as_ref())
                    .unwrap_or(&ref_node);

                let target_op = creator.resolve_op(&ir_node.target)?;
                let ctx = if insert_after {
                    graph.call_method1("inserting_after", (&ref_node,))?
                } else {
                    graph.call_method1("inserting_before", (&ref_node,))?
                };
                ctx.call_method0("__enter__")?;
                let call_kw = PyDict::new_bound(py);
                call_kw.set_item("args", py_tuple)?;
                // Pass node kwargs if any
                if !ir_node.kwargs.is_empty() {
                    let py_kwargs = PyDict::new_bound(py);
                    for (key, val) in &ir_node.kwargs {
                        py_kwargs.set_item(key.as_str(), ir_arg_to_py(py, val, &all_py)?)?;
                    }
                    call_kw.set_item("kwargs", py_kwargs)?;
                }
                let new_py = graph.call_method("call_function", (&target_op,), Some(&call_kw))?;
                // FX auto-names new call_function nodes from the operator target.
                // Preserve the IR name so later effect batches can address
                // freshly-created nodes by the same stable key.
                new_py.setattr("name", ir_node.name.as_str())?;
                let meta = copy_mod.call_method1("copy", (meta_src_node.getattr("meta")?,))?;
                new_py.setattr("meta", &meta)?;
                let none = py.None().into_bound(py);
                ctx.call_method1("__exit__", (&none, &none, &none))?;

                ir_node.meta.write_back(py, &new_py)?;
                if let Some(aid) = anchor_id {
                    last_for_anchor.insert((aid, insert_after_first), new_py.clone());
                }
                all_py[idx] = Some(new_py);
                progress = true;
                }
            }
        }

        // Write back modified args for alive original nodes
        // (after new node creation, so new node references resolve)
        fn py_node_id(graph: &IrGraph, py_arg: &Bound<'_, PyAny>) -> Option<NodeId> {
            let name = py_arg.getattr("name").ok()?.extract::<String>().ok()?;
            if py_arg.getattr("op").is_err() {
                return None;
            }
            graph.by_name(&name).map(|node| node.id)
        }

        fn collect_node_replacements<'py>(
            graph: &IrGraph,
            orig_py: &Bound<'py, PyAny>,
            ir_arg: &IrArg,
            replacements: &mut Vec<(Bound<'py, PyAny>, NodeId)>,
        ) {
            match ir_arg {
                IrArg::Node(new_id) => {
                    if py_node_id(graph, orig_py).is_some_and(|old_id| old_id != *new_id) {
                        replacements.push((orig_py.clone(), *new_id));
                    }
                }
                IrArg::List(items) => {
                    if let Ok(iter) = orig_py.iter() {
                        for (orig_item, ir_item) in iter.filter_map(|item| item.ok()).zip(items) {
                            collect_node_replacements(graph, &orig_item, ir_item, replacements);
                        }
                    }
                }
                _ => {}
            }
        }

        for ir_node in &self.nodes[..original_count] {
            if !ir_node.alive || !ir_node.args_dirty { continue; }
            let py_node = match all_py.get(ir_node.id).and_then(|o| o.as_ref()) {
                Some(n) => n,
                None => continue,
            };
            let orig_args = crate::fx_graph::FxGraphModule::node_args(py_node)
                .unwrap_or_default();

            if ir_node.args_rebuild_required {
                // Full args rebuild — needed for passes like quantize_clamp_args
                // that change literal values (Int/Float)
                let py_args: Vec<Bound<'_, PyAny>> = ir_node.args.iter()
                    .map(|a| ir_arg_to_py(py, a, &all_py))
                    .collect::<PyResult<_>>()?;
                let py_tuple = pyo3::types::PyTuple::new_bound(py, &py_args);
                py_node.setattr("args", py_tuple)?;
            } else {
                // Only Node refs changed — use replace_input_with to preserve
                // Python container types (tuples, dtype objects) on unchanged
                // args, including nested output/list args.
                let mut replacements = Vec::new();
                for (idx, ir_arg) in ir_node.args.iter().enumerate() {
                    if let Some(old_py) = orig_args.get(idx) {
                        collect_node_replacements(self, &old_py, ir_arg, &mut replacements);
                    }
                }
                for (old_py, new_id) in replacements {
                    if let Some(Some(new_py)) = all_py.get(new_id) {
                        let _ = py_node.call_method1("replace_input_with", (&old_py, new_py));
                    }
                }
            }

            if !ir_node.kwargs.is_empty() {
                let py_kwargs = PyDict::new_bound(py);
                for (key, val) in &ir_node.kwargs {
                    py_kwargs.set_item(key.as_str(), ir_arg_to_py(py, val, &all_py)?)?;
                }
                py_node.setattr("kwargs", py_kwargs)?;
            }
        }

        // Apply surgery: replace_all_uses + erase for dead original nodes
        // (reverse order so dependents are erased before their sources)
        for ir_node in self.nodes[..original_count].iter().rev() {
            if ir_node.alive { continue; }
            let py_node = match all_py.get(ir_node.id).and_then(|o| o.as_ref()) {
                Some(n) => n,
                None => continue,
            };
            if let Some(&replacement_id) = self.replacements.get(&ir_node.id) {
                if let Some(Some(replacement_py)) = all_py.get(replacement_id) {
                    py_node.call_method1("replace_all_uses_with", (replacement_py,))?;
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

        // Try backend dialect: ops.backend.tosa.GATHER.default
        if parts.len() >= 3 && parts[0] == "backend" {
            if let Ok(backend) = exir_ops.getattr("backend") {
                let mut obj = backend;
                for part in &parts[1..] {
                    obj = match obj.getattr(*part) {
                        Ok(o) => o,
                        Err(_) => break,
                    };
                }
                if obj.is_callable() {
                    return Ok(obj);
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
        Some(QParam::per_tensor(scale, zp, qmin, qmax, dtype_str))
    }
}

/// Find the output node in the Python node list (for fallback insertion point).
fn find_output_node<'py>(all_py: &[Option<Bound<'py, PyAny>>]) -> Bound<'py, PyAny> {
    all_py.iter().rev()
        .find_map(|opt| opt.as_ref().filter(|n| {
            n.getattr("op")
                .and_then(|o| o.extract::<String>())
                .map(|s| s == "output")
                .unwrap_or(false)
        }))
        .expect("graph must have an output node")
        .clone()
}

/// Convert an IrArg back to a Python object for node creation.
fn ir_arg_to_py<'py>(
    py: Python<'py>,
    arg: &IrArg,
    py_nodes: &[Option<Bound<'py, PyAny>>],
) -> PyResult<Bound<'py, PyAny>> {
    match arg {
        IrArg::Node(id) => py_nodes.get(*id)
            .and_then(|opt| opt.clone())
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("No Python node for IR node {id}")
            )),
        IrArg::Int(v) => Ok(v.into_py(py).into_bound(py)),
        IrArg::Float(v) => Ok(v.into_py(py).into_bound(py)),
        IrArg::Bool(v) => Ok(v.into_py(py).into_bound(py)),
        IrArg::Str(s) => Ok(s.into_py(py).into_bound(py)),
        IrArg::Dtype(s) => {
            let torch = py.import_bound("torch")?;
            Ok(torch.getattr(&*s.replace("torch.", ""))?)
        }
        IrArg::List(items) => {
            let py_items: Vec<Bound<'py, PyAny>> = items.iter()
                .map(|item| ir_arg_to_py(py, item, py_nodes))
                .collect::<PyResult<_>>()?;
            Ok(pyo3::types::PyList::new_bound(py, &py_items).into_any())
        }
        IrArg::None => Ok(py.None().into_bound(py)),
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
    // Check for torch.dtype objects (have is_floating_point attribute)
    if py_arg.hasattr("is_floating_point").unwrap_or(false) {
        if let Ok(s) = py_arg.str().map(|s| s.to_string()) {
            return IrArg::Dtype(s);
        }
    }
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
/// Untyped meta value for passthrough of custom keys (arm_layout_neutral, etc).
#[derive(Debug, Clone)]
pub enum MetaValue {
    Bool(bool),
    Int(i64),
    IntList(Vec<usize>),
    Str(String),
}

#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    pub input_qparams: Option<HashMap<usize, QParam>>,
    pub output_qparams: Option<HashMap<usize, QParam>>,
    pub val: Option<FakeTensorMeta>,
    pub val_dim_order: Option<Vec<usize>>,
    pub tosa_dim_order: Option<Vec<usize>>,
    pub original_dim_orders: Option<Vec<Vec<usize>>>,
    pub tosa_spatial_rank: Option<usize>,
    pub pre_pass_arg_qparams: Option<HashMap<usize, QParam>>,
    pub table_input_qparam: Option<QParam>,
    pub table_output_qparam: Option<QParam>,
    /// Arbitrary extra meta keys for passthrough (arm_layout_neutral, skip_size_adjust, etc).
    pub extras: HashMap<String, MetaValue>,
    pub(crate) dirty: DirtyFlags,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirtyFlags {
    pub(crate) input_qparams: bool,
    pub(crate) output_qparams: bool,
    pub(crate) tosa_dim_order: bool,
    pub(crate) original_dim_orders: bool,
    pub(crate) pre_pass_arg_qparams: bool,
    pub(crate) val: bool,
    pub(crate) table_qparams: bool,
    pub(crate) extras: bool,
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

        let val_obj = meta.get_item("val").ok();
        let val = val_obj.as_ref()
            .and_then(|v| FakeTensorMeta::from_py(v));
        let val_dim_order = val_obj.as_ref()
            .and_then(|v| v.call_method0("dim_order").ok())
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let tosa_dim_order = meta.get_item("tosa_dim_order")
            .ok()
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let original_dim_orders = meta.get_item("original_dim_orders")
            .ok()
            .and_then(|v| v.extract::<Vec<Vec<usize>>>().ok());

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

        // Import known extra meta keys
        let mut extras = HashMap::new();
        if let Ok(v) = meta.get_item("arm_layout_neutral") {
            if let Ok(b) = v.extract::<bool>() {
                extras.insert("arm_layout_neutral".into(), MetaValue::Bool(b));
            }
        }
        if let Ok(v) = meta.get_item("skip_size_adjust") {
            if let Ok(b) = v.extract::<bool>() {
                extras.insert("skip_size_adjust".into(), MetaValue::Bool(b));
            }
        }
        if let Ok(custom) = meta.get_item("custom") {
            if let Ok(ann_info) = custom.get_item("_arm_annotation_info") {
                if let Ok(b) = ann_info.get_item("quantized").and_then(|v| v.extract::<bool>()) {
                    extras.insert("arm_quantized_annotation".into(), MetaValue::Bool(b));
                }
            }
        }
        if let Ok(v) = meta.get_item("tosa_special_dtype") {
            if let Ok(s) = v.str().and_then(|s| s.extract::<String>()) {
                extras.insert("tosa_special_dtype".into(), MetaValue::Str(s));
            }
        }

        Ok(Self {
            input_qparams,
            output_qparams,
            val,
            val_dim_order,
            tosa_dim_order,
            original_dim_orders,
            tosa_spatial_rank,
            pre_pass_arg_qparams,
            table_input_qparam,
            table_output_qparam,
            extras,
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

        if self.dirty.original_dim_orders {
            if let Some(ref orders) = self.original_dim_orders {
                let py_orders = pyo3::types::PyList::empty_bound(py);
                for order in orders {
                    py_orders.append(order.clone())?;
                }
                meta.set_item("original_dim_orders", py_orders)?;
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

        if self.dirty.extras {
            for (key, value) in &self.extras {
                match value {
                    MetaValue::Bool(v) => meta.set_item(key.as_str(), *v)?,
                    MetaValue::Int(v) => meta.set_item(key.as_str(), *v)?,
                    MetaValue::IntList(v) => {
                        let tuple = pyo3::types::PyTuple::new_bound(py, v);
                        meta.set_item(key.as_str(), tuple)?;
                    }
                    MetaValue::Str(v) => meta.set_item(key.as_str(), v.as_str())?,
                }
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

    pub fn val_dim_order(&self) -> Option<&[usize]> {
        self.val_dim_order.as_deref()
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

    pub fn set_original_dim_orders(&mut self, orders: Vec<Vec<usize>>) {
        self.original_dim_orders = Some(orders);
        self.dirty.original_dim_orders = true;
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

    /// Set a custom extra meta key.
    pub fn set_extra(&mut self, key: impl Into<String>, value: MetaValue) {
        self.extras.insert(key.into(), value);
        self.dirty.extras = true;
    }

    /// Get a custom extra meta key.
    pub fn extra_bool(&self, key: &str) -> Option<bool> {
        match self.extras.get(key) {
            Some(MetaValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn extra_str(&self, key: &str) -> Option<&str> {
        match self.extras.get(key) {
            Some(MetaValue::Str(v)) => Some(v.as_str()),
            _ => None,
        }
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

#[derive(Debug, Default)]
pub struct IrEpPassResult {
    pub modified: bool,
    pub effects: FxEpEffectBatch,
}

impl IrEpPassResult {
    pub fn changed(&self) -> bool {
        self.modified || !self.effects.effects.is_empty()
    }
}

/// A Rust compiler pass that mutates `IrGraph` and may also need to apply
/// ExportedProgram bookkeeping effects when the graph is flushed back to Python.
pub trait IrEpPass {
    fn name(&self) -> &str;

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult;
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
    run_ir_passes_with_tensors(py, gm, None, passes)
}

pub fn run_ir_passes_with_tensors<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    exported_program: Option<&Bound<'py, PyAny>>,
    passes: &[&dyn IrPass],
) -> PyResult<bool> {
    let mut ir = IrGraph::from_fx_with_tensors(gm, exported_program)?;
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

// ===========================================================================
// IrPassAdapter — bridge IrPass into the FxPass pipeline
// ===========================================================================

/// Wraps a pure Rust `IrPass` as an `FxPass`, allowing it to be used
/// in the existing FxPass pipeline. Handles the full IrGraph lifecycle:
/// import from Python → run pass → write back with surgery.
pub struct IrPassAdapter<P: IrPass> {
    pass: P,
}

impl<P: IrPass> IrPassAdapter<P> {
    pub fn new(pass: P) -> Self {
        Self { pass }
    }
}

impl<P: IrPass> crate::fx_pass::FxPass for IrPassAdapter<P> {
    fn name(&self) -> &str {
        self.pass.name()
    }

    fn run(&self, gm: &crate::fx_graph::FxGraphModule) -> PyResult<bool> {
        run_ir_passes(gm.py(), gm.gm_ref(), &[&self.pass])
    }
}

// ===========================================================================
// Hybrid pipeline: persistent IrGraph with FxPass escape hatches
// ===========================================================================

/// A step in the hybrid pipeline — either a pure Rust IrPass or a Python FxPass.
pub enum PipelineStep {
    Ir(Box<dyn IrPass>),
    IrEp(Box<dyn IrEpPass>),
    Fx(Box<dyn crate::fx_pass::FxPass>),
}

/// Flush the IrGraph back to Python: write_back_with_surgery + cleanup.
fn flush_ir_to_python(
    ir: &IrGraph,
    py: Python<'_>,
    gm: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    pending_effects: &mut FxEpEffectBatch,
) -> PyResult<()> {
    ir.write_back_with_surgery(py, gm)?;
    if !pending_effects.effects.is_empty() {
        let exported_program = exported_program.ok_or_else(|| {
            PyRuntimeError::new_err(
                "effectful Ir pipeline step requires an ExportedProgram at flush time",
            )
        })?;
        let ctx = FxEpApplyContext::new(gm, exported_program)?;
        let _ = pending_effects.apply(&ctx)?;
        pending_effects.effects.clear();
    }
    let graph = gm.getattr("graph")?;
    graph.call_method0("eliminate_dead_code")?;
    graph.call_method0("lint")?;
    gm.call_method0("recompile")?;
    Ok(())
}

/// Run a hybrid pipeline: IrPasses mutate a persistent IrGraph in pure Rust.
/// FxPasses cause a flush→run→reimport cycle. Minimizes Python roundtrips.
pub fn run_hybrid_pipeline<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    exported_program: Option<&Bound<'py, PyAny>>,
    steps: &[PipelineStep],
) -> PyResult<bool> {
    use crate::runner_logging::{log_line, log_pass_start, log_pass_result};
    use std::time::Instant;

    let fxgm = crate::fx_graph::FxGraphModule::new(gm);
    let mut ir: Option<IrGraph> = None;
    let mut any_modified = false;
    let mut ir_dirty = false;  // tracks if the IrGraph has unflushed changes
    let mut pending_effects = FxEpEffectBatch::default();

    for (i, step) in steps.iter().enumerate() {
        match step {
            PipelineStep::Ir(pass) => {
                // Ensure we have an IrGraph
                if ir.is_none() {
                    ir = Some(IrGraph::from_fx_with_tensors(gm, exported_program)?);
                    ir_dirty = false;
                }
                let graph = ir.as_mut().unwrap();
                let before = graph.nodes.iter().filter(|n| n.alive).count();
                log_pass_start("HybridPipeline", i, steps.len(), pass.name());
                let modified = pass.run(graph);
                let after = graph.nodes.iter().filter(|n| n.alive).count();
                log_pass_result("HybridPipeline", pass.name(), modified, before, after);
                if modified {
                    any_modified = true;
                    ir_dirty = true;
                }
            }
            PipelineStep::IrEp(pass) => {
                if ir.is_none() {
                    ir = Some(IrGraph::from_fx_with_tensors(gm, exported_program)?);
                    ir_dirty = false;
                }
                let graph = ir.as_mut().unwrap();
                let before = graph.nodes.iter().filter(|n| n.alive).count();
                log_pass_start("HybridPipeline", i, steps.len(), pass.name());
                let result = pass.run(graph);
                let after = graph.nodes.iter().filter(|n| n.alive).count();
                log_pass_result("HybridPipeline", pass.name(), result.changed(), before, after);
                if result.changed() {
                    any_modified = true;
                    ir_dirty = true;
                    pending_effects.effects.extend(result.effects.effects);
                }
            }
            PipelineStep::Fx(pass) => {
                // Flush IrGraph to Python if dirty
                if ir_dirty {
                    if let Some(ref graph) = ir {
                        flush_ir_to_python(
                            graph,
                            py,
                            gm,
                            exported_program,
                            &mut pending_effects,
                        )?;
                    }
                    ir = None;
                    ir_dirty = false;
                } else {
                    // Drop the clean IrGraph — FxPass will read Python directly
                    ir = None;
                }
                let before = fxgm.node_count()?;
                log_pass_start("HybridPipeline", i, steps.len(), pass.name());
                let modified = pass.run(&fxgm)?;
                let after = fxgm.node_count()?;
                log_pass_result("HybridPipeline", pass.name(), modified, before, after);
                if modified {
                    any_modified = true;
                }
                // Next IrPass will re-import from Python
            }
        }
    }

    // Final flush if IrGraph has unflushed changes
    if ir_dirty {
        if let Some(ref graph) = ir {
            flush_ir_to_python(graph, py, gm, exported_program, &mut pending_effects)?;
        }
    }

    // Post-pipeline cleanup (same as run_fx_passes)
    if any_modified {
        let helpers = py.import_bound("fx_helpers")?;
        helpers.call_method1("refresh_safe_quantized_fake_tensors", (gm,))?;
    }

    Ok(any_modified)
}
