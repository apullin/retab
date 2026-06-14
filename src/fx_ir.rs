//! Rust-native graph IR for FX graph manipulation.
//!
//! Defines typed FX metadata, imported graph topology, graph-surgery helpers,
//! and the pass traits used by the Rust lowering pipeline.

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::fx_py_attr::{attrs, items, meta as meta_view};
pub(crate) use crate::fx_quant::QParam;
use crate::fx_target::{classify_ir_target, IrTargetKind};
use crate::graph::ir::DType;
use crate::lowering_effects::{LoweringEffectBatch, LoweringInputKind};
use crate::store::TensorData;
use crate::tensor_data::extract_tensor_data;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeOp {
    Placeholder,
    CallFunction,
    GetAttr,
    Output,
    Other,
}

impl NodeOp {
    pub(crate) fn of(node: &Bound<'_, PyAny>) -> Self {
        match attrs(node)
            .first_or::<String>(&["op"], String::new())
            .as_str()
        {
            "placeholder" => Self::Placeholder,
            "call_function" => Self::CallFunction,
            "get_attr" => Self::GetAttr,
            "output" => Self::Output,
            _ => Self::Other,
        }
    }
}

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
    /// Keyword arguments (e.g. dtype for _to_copy). Stored separately from args.
    pub kwargs: HashMap<String, IrArg>,
    pub meta: NodeMeta,
    pub tensor: Option<NodeTensor>,
    pub alive: bool,
}

#[derive(Debug, Clone)]
pub enum TensorSource {
    ExportedProgramInput(PlaceholderTensorBinding),
    GraphModuleAttr,
}

#[derive(Debug, Clone)]
pub struct NodeTensor {
    pub source: TensorSource,
    pub data: TensorData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceholderTensorBinding {
    pub input_kind: LoweringInputKind,
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
    /// A torch.dtype value (e.g. "torch.int8"), distinct from an ordinary string.
    Dtype(String),
    List(Vec<IrArg>),
    None,
}

pub(crate) fn i64_list_arg(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

pub(crate) fn dtype_kwarg(dtype: impl Into<String>) -> HashMap<String, IrArg> {
    let mut kwargs = HashMap::new();
    kwargs.insert("dtype".into(), IrArg::Dtype(dtype.into()));
    kwargs
}

pub(crate) fn shape_product(shape: &[i64]) -> i64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

pub(crate) fn broadcast_shapes(shapes: &[Vec<i64>]) -> Option<Vec<i64>> {
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(max_rank);
    for axis in 0..max_rank {
        let mut chosen = 1i64;
        for shape in shapes {
            let dim = *shape.iter().rev().nth(axis).unwrap_or(&1);
            if dim != 1 {
                if chosen == 1 {
                    chosen = dim;
                } else if chosen != dim {
                    return None;
                }
            }
        }
        out.push(chosen);
    }
    out.reverse();
    Some(out)
}

pub(crate) fn normalized_reduction_dims(
    dim_arg: Option<&IrArg>,
    input_shape: &[i64],
    skip_unit_dims: bool,
) -> Vec<i64> {
    let rank_i64 = input_shape.len() as i64;
    let dims = dim_arg
        .and_then(IrArg::as_int_list_or_scalar)
        .unwrap_or_else(|| (0..rank_i64).collect());
    let mut normalized = Vec::new();

    for dim in dims {
        let dim = if dim < 0 { dim + rank_i64 } else { dim };
        if dim < 0 || dim >= rank_i64 {
            continue;
        }
        if skip_unit_dims && input_shape[dim as usize] == 1 {
            continue;
        }
        if !normalized.contains(&dim) {
            normalized.push(dim);
        }
    }

    normalized.sort_unstable();
    normalized
}

pub(crate) fn reduced_shape(input_shape: &[i64], dims: &[i64], keepdim: bool) -> Vec<i64> {
    if keepdim {
        let mut shape = input_shape.to_vec();
        for &dim in dims {
            shape[dim as usize] = 1;
        }
        return shape;
    }

    let mut shape: Vec<i64> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &extent)| {
            if dims.contains(&(idx as i64)) {
                None
            } else {
                Some(extent)
            }
        })
        .collect();
    if shape.is_empty() {
        shape.push(1);
    }
    shape
}

fn build_placeholder_tensor_bindings(
    exported_program: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, PlaceholderTensorBinding>> {
    let mut placeholder_bindings = HashMap::new();
    let graph_signature = attrs(exported_program).object("graph_signature")?;
    for (attr_name, input_kind) in [
        ("inputs_to_parameters", LoweringInputKind::Parameter),
        ("inputs_to_buffers", LoweringInputKind::Buffer),
        (
            "inputs_to_lifted_tensor_constants",
            LoweringInputKind::LiftedTensorConstant,
        ),
    ] {
        if let Ok(mapping) = attrs(&graph_signature).object(attr_name) {
            for (placeholder_name, target_name) in items(&mapping).typed_pairs::<String, String>() {
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
    Ok(placeholder_bindings)
}

struct FxTensorImportContext<'a, 'py> {
    gm: &'a Bound<'py, PyAny>,
    state_dict: Option<&'a Bound<'py, PyAny>>,
    constants: Option<&'a Bound<'py, PyAny>>,
    placeholder_bindings: &'a HashMap<String, PlaceholderTensorBinding>,
}

impl<'a, 'py> FxTensorImportContext<'a, 'py> {
    fn tensor_for_node(
        &self,
        py_node: &Bound<'py, PyAny>,
        name: &str,
        target: &str,
    ) -> Option<NodeTensor> {
        let op: String = attrs(py_node).optional("op")?;
        match op.as_str() {
            "placeholder" => self.placeholder_tensor(name),
            "get_attr" => self.graph_attr_tensor(target),
            _ => None,
        }
    }

    fn placeholder_tensor(&self, name: &str) -> Option<NodeTensor> {
        let binding = self.placeholder_bindings.get(name)?;
        self.placeholder_tensor_from_mapping(self.state_dict, binding)
            // Lifted tensor constants are not in state_dict; they live in
            // ep.constants keyed by target_name.
            .or_else(|| self.placeholder_tensor_from_mapping(self.constants, binding))
            .or_else(|| self.placeholder_tensor_from_gm_attr(binding))
    }

    fn placeholder_tensor_from_mapping(
        &self,
        mapping: Option<&Bound<'py, PyAny>>,
        binding: &PlaceholderTensorBinding,
    ) -> Option<NodeTensor> {
        let value = items(mapping?).key(binding.target_name.as_str())?;
        tensor_from_py_value(&value, TensorSource::ExportedProgramInput(binding.clone()))
    }

    fn placeholder_tensor_from_gm_attr(
        &self,
        binding: &PlaceholderTensorBinding,
    ) -> Option<NodeTensor> {
        let value = attrs(self.gm).object(binding.target_name.as_str()).ok()?;
        tensor_from_py_value(&value, TensorSource::ExportedProgramInput(binding.clone()))
    }

    fn graph_attr_tensor(&self, target: &str) -> Option<NodeTensor> {
        let value = attrs(self.gm).object(target).ok()?;
        if attrs(&value).has("graph") {
            return None;
        }
        tensor_from_py_value(&value, TensorSource::GraphModuleAttr)
    }
}

fn tensor_from_py_value(value: &Bound<'_, PyAny>, source: TensorSource) -> Option<NodeTensor> {
    Some(NodeTensor {
        source,
        data: extract_tensor_data(value).ok()?,
    })
}

fn collect_fx_nodes<'py>(graph: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let nodes = attrs(graph)
        .object("nodes")?
        .iter()?
        .filter_map(|node| node.ok())
        .collect();
    Ok(nodes)
}

fn build_name_to_id(py_nodes: &[Bound<'_, PyAny>]) -> HashMap<String, NodeId> {
    py_nodes
        .iter()
        .enumerate()
        .map(|(id, node)| (attrs(node).or_value("name", String::new()), id))
        .collect()
}

fn import_fx_node<'py>(
    id: NodeId,
    py_node: &Bound<'py, PyAny>,
    name_to_id: &HashMap<String, NodeId>,
    tensor_import: &FxTensorImportContext<'_, 'py>,
) -> PyResult<IrNode> {
    let py_node_attrs = attrs(py_node);
    let name = py_node_attrs.or_value("name", String::new());
    let target = py_node_attrs.required_str("target").unwrap_or_default();
    let op = NodeOp::of(py_node);
    let meta = NodeMeta::from_fx(py_node)?;
    let tensor = tensor_import.tensor_for_node(py_node, &name, &target);
    let args = resolve_fx_args(py_node, name_to_id);
    let kwargs = resolve_fx_kwargs(py_node, name_to_id);

    Ok(IrNode {
        id,
        op,
        name,
        target,
        args,
        kwargs,
        meta,
        tensor,
        alive: true,
    })
}

fn resolve_fx_args(py_node: &Bound<'_, PyAny>, name_to_id: &HashMap<String, NodeId>) -> Vec<IrArg> {
    fx_node_args(py_node)
        .unwrap_or_default()
        .iter()
        .map(|arg| resolve_arg(arg, name_to_id))
        .collect()
}

fn resolve_fx_kwargs(
    py_node: &Bound<'_, PyAny>,
    name_to_id: &HashMap<String, NodeId>,
) -> HashMap<String, IrArg> {
    let Ok(py_kwargs) = fx_node_kwargs(py_node) else {
        return HashMap::new();
    };
    if py_kwargs.is_none() {
        return HashMap::new();
    }
    items(&py_kwargs)
        .typed_key_pairs::<String>()
        .into_iter()
        .map(|(key, val)| (key, resolve_arg(&val, name_to_id)))
        .collect()
}

fn fx_node_args<'py>(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let args = attrs(node).object("args")?;
    let mut result = Vec::new();
    for arg in args.iter()? {
        result.push(arg?);
    }
    Ok(result)
}

fn fx_node_kwargs<'py>(node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    attrs(node).object("kwargs")
}

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

fn build_user_edges(nodes: &[IrNode]) -> Vec<Vec<NodeId>> {
    let mut users = vec![Vec::new(); nodes.len()];
    for node in nodes {
        if !node.alive {
            continue;
        }
        for arg in &node.args {
            collect_users_from_arg(arg, node.id, &mut users);
        }
        for arg in node.kwargs.values() {
            collect_users_from_arg(arg, node.id, &mut users);
        }
    }
    users
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
    /// Explicit position anchors retained for IR ordering and serialization hints.
    anchors: HashMap<NodeId, NodeId>,
    /// Explicit after-anchors for nodes that must be defined after a live
    /// producer while still preceding rewired consumers.
    anchors_after: HashMap<NodeId, NodeId>,
}

pub(crate) struct RescaleRewrite {
    name: String,
    source: NodeId,
    output_dtype: String,
    scales: Vec<f64>,
    input_zp: i64,
    output_zp: i64,
    meta: NodeMeta,
}

impl RescaleRewrite {
    pub(crate) fn new(
        name: impl Into<String>,
        source: NodeId,
        output_dtype: impl Into<String>,
        scales: Vec<f64>,
        input_zp: i64,
        output_zp: i64,
        meta: NodeMeta,
    ) -> Self {
        Self {
            name: name.into(),
            source,
            output_dtype: output_dtype.into(),
            scales,
            input_zp,
            output_zp,
            meta,
        }
    }
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

    #[cfg(test)]
    pub(crate) fn replacement_of(&self, id: NodeId) -> Option<NodeId> {
        self.replacements.get(&id).copied()
    }

    #[cfg(test)]
    pub(crate) fn by_name(&self, name: &str) -> Option<&IrNode> {
        self.name_to_id.get(name).and_then(|id| self.nodes.get(*id))
    }

    pub fn from_fx_with_tensors(
        gm: &Bound<'_, PyAny>,
        exported_program: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let gm_attrs = attrs(gm);
        let graph = if gm_attrs.try_has("graph")? {
            gm_attrs.object("graph")?
        } else {
            gm.clone()
        };
        let py_nodes = collect_fx_nodes(&graph)?;

        let (placeholder_bindings, state_dict, constants) = if let Some(ep) = exported_program {
            (
                build_placeholder_tensor_bindings(ep)?,
                attrs(ep).object("state_dict").ok(),
                // Lifted tensor constants live in `ep.constants`, NOT `state_dict`.
                attrs(ep).object("constants").ok(),
            )
        } else {
            (HashMap::new(), None, None)
        };
        let tensor_import = FxTensorImportContext {
            gm,
            state_dict: state_dict.as_ref(),
            constants: constants.as_ref(),
            placeholder_bindings: &placeholder_bindings,
        };

        let name_to_id = build_name_to_id(&py_nodes);
        let nodes = py_nodes
            .iter()
            .enumerate()
            .map(|(id, py_node)| import_fx_node(id, py_node, &name_to_id, &tensor_import))
            .collect::<PyResult<Vec<_>>>()?;

        let users = build_user_edges(&nodes);

        Ok(Self {
            nodes,
            name_to_id,
            users,
            replacements: HashMap::new(),
            anchors: HashMap::new(),
            anchors_after: HashMap::new(),
        })
    }

    /// Iterate placeholders.
    pub fn placeholders(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|n| n.op == NodeOp::Placeholder)
    }

    /// Get the output node.
    pub fn output_node(&self) -> Option<&IrNode> {
        self.nodes.iter().find(|n| n.op == NodeOp::Output)
    }

    pub fn node_opt(&self, id: NodeId) -> Option<&IrNode> {
        self.nodes.get(id)
    }

    /// Get node by ID.
    pub fn node(&self, id: NodeId) -> &IrNode {
        &self.nodes[id]
    }

    /// Get mutable node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut IrNode {
        &mut self.nodes[id]
    }

    /// Update a node's args and rebuild reverse-use lists.
    pub fn set_args(&mut self, id: NodeId, args: Vec<IrArg>) {
        self.nodes[id].args = args;
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

    pub fn add_call_function(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        self.add_node(NodeOp::CallFunction, name.into(), target.into(), args, meta)
    }

    pub fn add_call_function_with_val(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        base_meta: &NodeMeta,
        shape: Vec<i64>,
        dtype: impl Into<String>,
    ) -> NodeId {
        self.add_call_function(name, target, args, base_meta.cloned_with_val(shape, dtype))
    }

    pub fn add_call_function_with_kwargs(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        self.add_node_with_kwargs(
            NodeOp::CallFunction,
            name.into(),
            target.into(),
            args,
            kwargs,
            meta,
        )
    }

    pub fn add_anchored_call_function(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_call_function(name, target, args, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub fn add_after_call_function(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_call_function(name, target, args, meta);
        self.anchor_after(id, anchor);
        id
    }

    pub fn replace_with_call_function(
        &mut self,
        old_id: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_anchored_call_function(old_id, name, target, args, meta);
        self.replace_and_erase(old_id, id);
        id
    }

    pub fn replace_with_call_function_with_val(
        &mut self,
        old_id: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        base_meta: &NodeMeta,
        shape: Vec<i64>,
        dtype: impl Into<String>,
    ) -> NodeId {
        let id = self.add_anchored_call_function_with_val(
            old_id, name, target, args, base_meta, shape, dtype,
        );
        self.replace_and_erase(old_id, id);
        id
    }

    pub fn add_anchored_call_function_with_val(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        base_meta: &NodeMeta,
        shape: Vec<i64>,
        dtype: impl Into<String>,
    ) -> NodeId {
        let id = self.add_call_function_with_val(name, target, args, base_meta, shape, dtype);
        self.anchor_to(id, anchor);
        id
    }

    pub fn add_anchored_call_function_with_kwargs(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_call_function_with_kwargs(name, target, args, kwargs, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub(crate) fn add_anchored_dtype_cast(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        source: NodeId,
        dtype: impl Into<String>,
        meta: NodeMeta,
    ) -> NodeId {
        self.add_anchored_call_function_with_kwargs(
            anchor,
            name,
            "aten._to_copy.default",
            vec![IrArg::Node(source)],
            dtype_kwarg(dtype),
            meta,
        )
    }

    pub(crate) fn replace_with_dtype_cast(
        &mut self,
        old_id: NodeId,
        name: impl Into<String>,
        source: NodeId,
        dtype: impl Into<String>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_anchored_dtype_cast(old_id, name, source, dtype, meta);
        self.replace_and_erase(old_id, id);
        id
    }

    pub(crate) fn add_view_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        shape: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        self.add_call_function(
            name,
            "aten.view_copy.default",
            vec![IrArg::Node(source), i64_list_arg(shape)],
            meta,
        )
    }

    pub(crate) fn add_anchored_view_copy(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        source: NodeId,
        shape: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_view_copy(name, source, shape, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub(crate) fn add_after_view_copy(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        source: NodeId,
        shape: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_view_copy(name, source, shape, meta);
        self.anchor_after(id, anchor);
        id
    }

    pub(crate) fn add_permute_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        perm: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        self.add_call_function(
            name,
            "aten.permute_copy.default",
            vec![IrArg::Node(source), i64_list_arg(perm)],
            meta,
        )
    }

    pub(crate) fn add_anchored_permute_copy(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        source: NodeId,
        perm: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_permute_copy(name, source, perm, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub(crate) fn replace_with_permute_copy(
        &mut self,
        old_id: NodeId,
        name: impl Into<String>,
        source: NodeId,
        perm: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_anchored_permute_copy(old_id, name, source, perm, meta);
        self.replace_and_erase(old_id, id);
        id
    }

    pub(crate) fn replace_with_view_copy(
        &mut self,
        old_id: NodeId,
        name: impl Into<String>,
        source: NodeId,
        shape: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_anchored_view_copy(old_id, name, source, shape, meta);
        self.replace_and_erase(old_id, id);
        id
    }

    pub(crate) fn add_rescale(&mut self, spec: RescaleRewrite) -> NodeId {
        let RescaleRewrite {
            name,
            source,
            output_dtype,
            scales,
            input_zp,
            output_zp,
            meta,
        } = spec;
        self.add_node(
            NodeOp::CallFunction,
            name,
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                IrArg::Node(source),
                IrArg::Dtype(output_dtype),
                IrArg::List(scales.into_iter().map(IrArg::Float).collect()),
                IrArg::Int(input_zp),
                IrArg::Int(output_zp),
            ],
            meta,
        )
    }

    pub(crate) fn add_anchored_rescale(&mut self, anchor: NodeId, spec: RescaleRewrite) -> NodeId {
        let id = self.add_rescale(spec);
        self.anchor_to(id, anchor);
        id
    }

    pub(crate) fn add_after_rescale(&mut self, anchor: NodeId, spec: RescaleRewrite) -> NodeId {
        let id = self.add_rescale(spec);
        self.anchor_after(id, anchor);
        id
    }

    pub fn add_full_int(
        &mut self,
        name: impl Into<String>,
        shape: &[i64],
        value: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        self.add_call_function_with_kwargs(
            name,
            "aten.full.default",
            vec![i64_list_arg(shape), IrArg::Int(value)],
            dtype_kwarg(dtype),
            meta,
        )
    }

    pub fn add_full_float(
        &mut self,
        name: impl Into<String>,
        shape: &[i64],
        value: f64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        self.add_call_function_with_kwargs(
            name,
            "aten.full.default",
            vec![i64_list_arg(shape), IrArg::Float(value)],
            dtype_kwarg(dtype),
            meta,
        )
    }

    pub fn add_anchored_full_int(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        shape: &[i64],
        value: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_full_int(name, shape, value, dtype, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub fn add_anchored_full_float(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        shape: &[i64],
        value: f64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_full_float(name, shape, value, dtype, meta);
        self.anchor_to(id, anchor);
        id
    }

    pub fn add_arange(
        &mut self,
        name: impl Into<String>,
        start: i64,
        end: i64,
        step: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let mut args = vec![IrArg::Int(start), IrArg::Int(end)];
        if step != 1 {
            args.push(IrArg::Int(step));
        }
        self.add_call_function_with_kwargs(
            name,
            "aten.arange.start_step",
            args,
            dtype_kwarg(dtype),
            meta,
        )
    }

    pub fn add_anchored_arange(
        &mut self,
        anchor: NodeId,
        name: impl Into<String>,
        start: i64,
        end: i64,
        step: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.add_arange(name, start, end, step, dtype, meta);
        self.anchor_to(id, anchor);
        id
    }

    /// Update a single arg by index and mark dirty.
    pub fn set_arg(&mut self, id: NodeId, index: usize, value: IrArg) {
        if index < self.nodes[id].args.len() {
            self.nodes[id].args[index] = value;
            self.rebuild_users();
        }
    }

    /// Update a kwarg.
    pub fn set_kwarg(&mut self, id: NodeId, key: impl Into<String>, value: IrArg) {
        self.nodes[id].kwargs.insert(key.into(), value);
    }

    /// Get users (nodes that reference this node in their args).
    pub fn users(&self, id: NodeId) -> &[NodeId] {
        &self.users[id]
    }

    pub fn live_nodes(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|node| node.alive)
    }

    pub fn live_node_count(&self) -> usize {
        self.live_nodes().count()
    }

    pub fn live_user_ids(&self, id: NodeId) -> Vec<NodeId> {
        self.live_nodes()
            .filter(|node| {
                node.args.iter().any(|arg| arg.contains_node(id))
                    || node.kwargs.values().any(|arg| arg.contains_node(id))
            })
            .map(|node| node.id)
            .collect()
    }

    pub fn live_node_ids(&self) -> Vec<NodeId> {
        self.live_nodes().map(|node| node.id).collect()
    }

    pub fn live_call_function_ids(&self) -> Vec<NodeId> {
        self.live_nodes()
            .filter(|node| node.op == NodeOp::CallFunction)
            .map(|node| node.id)
            .collect()
    }

    pub fn rewrite_live_call_functions(
        &mut self,
        mut rewrite: impl FnMut(&mut Self, NodeId) -> bool,
    ) -> bool {
        let mut modified = false;
        for id in self.live_call_function_ids() {
            modified |= rewrite(self, id);
        }
        modified
    }

    pub fn live_placeholder_ids(&self) -> Vec<NodeId> {
        self.live_nodes()
            .filter(|node| node.op == NodeOp::Placeholder)
            .map(|node| node.id)
            .collect()
    }

    pub fn live_output_ids(&self) -> Vec<NodeId> {
        self.live_nodes()
            .filter(|node| node.op == NodeOp::Output)
            .map(|node| node.id)
            .collect()
    }

    pub fn has_live_users(&self, id: NodeId) -> bool {
        !self.live_user_ids(id).is_empty()
    }

    pub fn single_live_user(&self, id: NodeId) -> Option<NodeId> {
        let users = self.live_user_ids(id);
        (users.len() == 1).then(|| users[0])
    }

    pub fn live_call_function_users(&self, id: NodeId) -> Vec<NodeId> {
        self.live_user_ids(id)
            .into_iter()
            .filter(|&user_id| self.nodes[user_id].op == NodeOp::CallFunction)
            .collect()
    }

    pub fn single_live_call_function_user(&self, id: NodeId) -> Option<NodeId> {
        let users = self.live_call_function_users(id);
        (users.len() == 1).then(|| users[0])
    }

    pub fn live_getitem_users(&self, id: NodeId) -> Vec<(NodeId, i64)> {
        self.live_nodes()
            .filter(|node| node.is_getitem())
            .filter(|node| node.first_node_arg() == Some(id))
            .map(|node| (node.id, node.arg_int_or(1, 0)))
            .collect()
    }

    pub(crate) fn node_shape(&self, id: NodeId) -> Option<&[i64]> {
        self.nodes[id].meta.val_shape()
    }

    pub(crate) fn node_shape_vec(&self, id: NodeId) -> Option<Vec<i64>> {
        self.node_shape(id).map(|shape| shape.to_vec())
    }

    pub(crate) fn node_dtype(&self, id: NodeId) -> Option<&str> {
        self.nodes[id].meta.val_dtype()
    }

    pub(crate) fn node_dtype_or_float32(&self, id: NodeId) -> &str {
        self.nodes[id].meta.val_dtype_or_float32()
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
            if !node.alive {
                continue;
            }
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
        }
        if changed {
            self.rebuild_users();
        }
        changed
    }

    pub(crate) fn replace_live_uses_except(
        &mut self,
        old_id: NodeId,
        new_id: NodeId,
        excluded_user: NodeId,
    ) -> bool {
        let users = self.live_user_ids(old_id);
        let mut rewired = false;
        for user_id in users {
            if user_id == excluded_user {
                continue;
            }
            rewired |= self.replace_input_use(user_id, old_id, new_id);
        }
        rewired
    }

    /// Mark a node as dead. Conversion and structural rebuilds skip dead nodes.
    pub fn erase(&mut self, id: NodeId) {
        self.nodes[id].alive = false;
    }

    /// Replace all uses of `old_id` with `new_id`, then erase `old_id`.
    pub fn replace_and_erase(&mut self, old_id: NodeId, new_id: NodeId) {
        self.replace_all_uses(old_id, new_id);
        self.erase(old_id);
        self.replacements.insert(old_id, new_id);
    }

    /// Record a position anchor for IR-inserted nodes.
    pub fn anchor_to(&mut self, new_id: NodeId, original_id: NodeId) {
        self.anchors.insert(new_id, original_id);
    }

    /// Record an after-anchor for a new node that consumes a live producer.
    pub fn anchor_after(&mut self, new_id: NodeId, original_id: NodeId) {
        self.anchors_after.insert(new_id, original_id);
    }

    /// Insert a new node, returning its ID. The node is appended to the end;
    /// ordering relative to other nodes is handled by anchors and conversion.
    pub fn add_node(
        &mut self,
        op: NodeOp,
        name: String,
        target: String,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.nodes.len();
        for arg in &args {
            collect_users_from_arg(arg, id, &mut self.users);
        }
        self.users.push(Vec::new());
        self.name_to_id.insert(name.clone(), id);
        self.nodes.push(IrNode {
            id,
            op,
            name,
            target,
            args,
            kwargs: HashMap::new(),
            meta,
            tensor: None,
            alive: true,
        });
        id
    }

    fn rebuild_users(&mut self) {
        self.users = build_user_edges(&self.nodes);
    }
}

impl IrNode {
    pub(crate) fn target_kind(&self) -> IrTargetKind {
        classify_ir_target(&self.target)
    }

    pub(crate) fn is_call_target_kind(&self, kind: IrTargetKind) -> bool {
        self.op == NodeOp::CallFunction && self.target_kind() == kind
    }

    pub(crate) fn arg(&self, index: usize) -> Option<&IrArg> {
        self.args.get(index)
    }

    pub(crate) fn kwarg(&self, key: &str) -> Option<&IrArg> {
        self.kwargs.get(key)
    }

    pub(crate) fn arg_clone_or_none(&self, index: usize) -> IrArg {
        self.arg(index).cloned().unwrap_or(IrArg::None)
    }

    pub(crate) fn kwarg_clone_or_none(&self, key: &str) -> IrArg {
        self.kwarg(key).cloned().unwrap_or(IrArg::None)
    }

    pub(crate) fn arg_node(&self, index: usize) -> Option<NodeId> {
        self.arg(index).and_then(IrArg::as_node)
    }

    pub(crate) fn arg_int(&self, index: usize) -> Option<i64> {
        self.arg(index).and_then(IrArg::as_int)
    }

    pub(crate) fn kwarg_int(&self, key: &str) -> Option<i64> {
        self.kwarg(key).and_then(IrArg::as_int)
    }

    pub(crate) fn arg_or_kwarg_int(&self, index: usize, key: &str) -> Option<i64> {
        self.arg_int(index).or_else(|| self.kwarg_int(key))
    }

    pub(crate) fn arg_int_or(&self, index: usize, default: i64) -> i64 {
        self.arg_int(index).unwrap_or(default)
    }

    pub(crate) fn arg_numeric_f64(&self, index: usize) -> Option<f64> {
        self.arg(index).and_then(IrArg::as_numeric_f64)
    }

    pub(crate) fn kwarg_numeric_f64(&self, key: &str) -> Option<f64> {
        self.kwarg(key).and_then(IrArg::as_numeric_f64)
    }

    pub(crate) fn arg_or_kwarg_numeric_f64(&self, index: usize, key: &str) -> Option<f64> {
        self.arg_numeric_f64(index)
            .or_else(|| self.kwarg_numeric_f64(key))
    }

    pub(crate) fn arg_or_kwarg_scalar_literal_f64(&self, index: usize, key: &str) -> Option<f64> {
        self.arg(index)
            .and_then(IrArg::as_scalar_literal_f64)
            .or_else(|| self.kwarg(key).and_then(IrArg::as_scalar_literal_f64))
    }

    pub(crate) fn arg_bool(&self, index: usize) -> Option<bool> {
        self.arg(index).and_then(IrArg::as_bool)
    }

    pub(crate) fn kwarg_bool(&self, key: &str) -> Option<bool> {
        self.kwarg(key).and_then(IrArg::as_bool)
    }

    pub(crate) fn arg_or_kwarg_bool(&self, index: usize, key: &str) -> Option<bool> {
        self.arg_bool(index).or_else(|| self.kwarg_bool(key))
    }

    pub(crate) fn arg_bool_or(&self, index: usize, default: bool) -> bool {
        self.arg_bool(index).unwrap_or(default)
    }

    pub(crate) fn kwarg_str(&self, key: &str) -> Option<&str> {
        self.kwarg(key).and_then(IrArg::as_str)
    }

    pub(crate) fn arg_or_kwarg_str(&self, index: usize, key: &str) -> Option<&str> {
        self.arg(index)
            .and_then(IrArg::as_str)
            .or_else(|| self.kwarg_str(key))
    }

    pub(crate) fn arg_int_list(&self, index: usize) -> Option<Vec<i64>> {
        self.arg(index).and_then(IrArg::as_int_list)
    }

    pub(crate) fn arg_int_pair_or_scalar(&self, index: usize) -> Option<(i64, i64)> {
        self.arg(index).and_then(IrArg::as_int_pair_or_scalar)
    }

    /// Get first arg if it's a node reference.
    pub fn first_node_arg(&self) -> Option<NodeId> {
        self.arg_node(0)
    }

    /// Is this a quantize_per_tensor call?
    pub fn is_quantize(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::QuantizePerTensor)
    }

    /// Is this a dequantize_per_tensor call?
    pub fn is_dequantize(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::DequantizePerTensor)
    }

    pub(crate) fn is_quantize_any(&self) -> bool {
        self.op == NodeOp::CallFunction
            && matches!(
                self.target_kind(),
                IrTargetKind::QuantizePerTensor | IrTargetKind::QuantizePerChannel
            )
    }

    pub(crate) fn is_dequantize_any(&self) -> bool {
        self.op == NodeOp::CallFunction
            && matches!(
                self.target_kind(),
                IrTargetKind::DequantizePerTensor | IrTargetKind::DequantizePerChannel
            )
    }

    pub(crate) fn is_q_or_dq(&self) -> bool {
        self.is_quantize_any() || self.is_dequantize_any()
    }

    pub(crate) fn is_relu(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenRelu)
    }

    pub(crate) fn is_hardtanh(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenHardtanh)
    }

    pub(crate) fn is_conv_or_linear(&self) -> bool {
        self.op == NodeOp::CallFunction
            && matches!(
                self.target_kind(),
                IrTargetKind::AtenConv | IrTargetKind::AtenLinear
            )
    }

    pub(crate) fn is_linear(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenLinear)
    }

    pub(crate) fn is_convolution(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenConv)
    }

    pub(crate) fn is_cond(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::Cond)
    }

    pub(crate) fn is_while_loop(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::WhileLoop)
    }

    pub(crate) fn is_control_flow(&self) -> bool {
        self.is_cond() || self.is_while_loop()
    }

    pub(crate) fn is_getitem(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::GetItem)
    }

    /// Is this a TOSA RESCALE call?
    pub fn is_rescale(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::TosaRescale)
    }

    /// Is this an ATen mm/bmm call that produces an accumulator needing rescale?
    pub(crate) fn is_mm_or_bmm(&self) -> bool {
        self.is_mm() || self.is_bmm()
    }

    pub(crate) fn is_mm(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenMm)
    }

    pub(crate) fn is_bmm(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenBmm)
    }

    pub(crate) fn is_addmm(&self) -> bool {
        self.is_call_target_kind(IrTargetKind::AtenAddmm)
    }

    /// Parse Q/DQ literal args into a QParam.
    /// Q/DQ nodes have args: (input, scale, zp, qmin, qmax, dtype).
    pub fn parse_qdq_qparam(&self) -> Option<QParam> {
        if self.args.len() < 6 {
            return None;
        }
        let scale = match &self.args[1] {
            IrArg::Float(v) => *v,
            IrArg::Int(v) => *v as f64,
            _ => return None,
        };
        let zp = match &self.args[2] {
            IrArg::Int(v) => *v,
            _ => return None,
        };
        let qmin = match &self.args[3] {
            IrArg::Int(v) => *v,
            _ => return None,
        };
        let qmax = match &self.args[4] {
            IrArg::Int(v) => *v,
            _ => return None,
        };
        // dtype arg may not resolve cleanly — check if it's a string or use target info
        let dtype_str = match &self.args[5] {
            IrArg::Str(s) => s.clone(),
            _ => {
                if qmin == -128 && qmax == 127 {
                    "torch.int8".into()
                } else if qmin == -32768 {
                    "torch.int16".into()
                } else {
                    return None;
                }
            }
        };
        Some(QParam::per_tensor(scale, zp, qmin, qmax, dtype_str))
    }
}

/// Resolve a Python FX arg to an IrArg.
fn resolve_arg(py_arg: &Bound<'_, PyAny>, name_to_id: &HashMap<String, NodeId>) -> IrArg {
    let py_arg_attrs = attrs(py_arg);
    // Check if it's a node (has .op attribute)
    if let Some(name) = py_arg_attrs.optional::<String>("name") {
        if py_arg_attrs.has("op") {
            if let Some(&id) = name_to_id.get(&name) {
                return IrArg::Node(id);
            }
        }
    }
    // Try literals
    if py_arg.is_none() {
        return IrArg::None;
    }
    if let Ok(v) = py_arg.extract::<bool>() {
        return IrArg::Bool(v);
    }
    if let Ok(v) = py_arg.extract::<i64>() {
        return IrArg::Int(v);
    }
    if let Ok(v) = py_arg.extract::<f64>() {
        return IrArg::Float(v);
    }
    if let Ok(v) = py_arg.extract::<String>() {
        return IrArg::Str(v);
    }
    // Check for torch.dtype objects (have is_floating_point attribute)
    if py_arg_attrs.has("is_floating_point") {
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
        if !items.is_empty() {
            return IrArg::List(items);
        }
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
        let val_attrs = attrs(val);
        let shape = val_attrs.optional("shape")?;
        let dtype = val_attrs.display_optional("dtype")?;
        Some(Self { shape, dtype })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// Metadata fields extracted from `node.meta`. The common fields are typed;
/// rare fields stay in the Python meta dict and are not touched.
/// Untyped meta value for passthrough of custom keys (arm_constant_placeholder, etc).
#[derive(Debug, Clone)]
pub enum MetaValue {
    Bool(bool),
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
    /// Arbitrary extra meta keys for passthrough (skip_size_adjust, arm_constant_placeholder, etc).
    pub extras: HashMap<String, MetaValue>,
}

// ---------------------------------------------------------------------------
// QParam dict extraction helpers
// ---------------------------------------------------------------------------

fn extract_qparam_dict(py_dict: &Bound<'_, PyAny>) -> Option<HashMap<usize, QParam>> {
    if py_dict.is_none() || !py_dict.is_truthy().unwrap_or(false) {
        return None;
    }
    let mut map = HashMap::new();
    for (key, val) in items(py_dict).typed_key_pairs::<usize>() {
        map.insert(key, QParam::from_py(&val));
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

// ---------------------------------------------------------------------------
// NodeMeta import / export
// ---------------------------------------------------------------------------

impl NodeMeta {
    pub fn tensor(shape: impl Into<Vec<i64>>, dtype: impl Into<String>) -> Self {
        let mut meta = Self::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.into(),
            dtype: dtype.into(),
        });
        meta
    }

    /// Extract typed metadata from a Python FX node.
    pub fn from_fx(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        let meta = attrs(node).object("meta")?;
        let meta_attrs = meta_view(&meta);

        let input_qparams = meta_attrs
            .value("input_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        let output_qparams = meta_attrs
            .value("output_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        let val_obj = meta_attrs.value("val");
        let val = val_obj.as_ref().and_then(|v| FakeTensorMeta::from_py(v));
        let val_dim_order = val_obj
            .as_ref()
            .and_then(|v| attrs(v).call_optional("dim_order"));

        let tosa_dim_order = meta_attrs.optional("tosa_dim_order");

        let original_dim_orders = meta_attrs.optional("original_dim_orders");

        let tosa_spatial_rank = meta_attrs.optional("tosa_spatial_rank");

        let pre_pass_arg_qparams = meta_attrs
            .value("_pre_pass_arg_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        // Import known extra meta keys
        let mut extras = HashMap::new();
        for key in [
            "arm_constant_placeholder",
            "arm_control_flow_placeholder",
            "arm_control_flow_parent_output_user",
        ] {
            if let Some(b) = meta_attrs.optional::<bool>(key) {
                extras.insert(key.into(), MetaValue::Bool(b));
            }
        }
        if let Some(b) = meta_attrs.optional::<bool>("skip_size_adjust") {
            extras.insert("skip_size_adjust".into(), MetaValue::Bool(b));
        }
        if let Some(custom) = meta_attrs.value("custom") {
            if let Some(ann_info) = items(&custom).key("_arm_annotation_info") {
                if let Some(b) = items(&ann_info).optional_key::<bool>("quantized") {
                    extras.insert("arm_quantized_annotation".into(), MetaValue::Bool(b));
                }
            }
        }
        if let Some(s) = meta_attrs.display_optional("tosa_special_dtype") {
            extras.insert("tosa_special_dtype".into(), MetaValue::Str(s));
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
            extras,
        })
    }

    // --- Accessors ---

    pub fn val_shape(&self) -> Option<&[i64]> {
        self.val.as_ref().map(|v| v.shape.as_slice())
    }

    pub fn val_dtype(&self) -> Option<&str> {
        self.val.as_ref().map(|v| v.dtype.as_str())
    }

    pub fn val_dtype_kind(&self) -> Option<DType> {
        self.val_dtype().map(DType::from_torch_str)
    }

    pub fn val_dtype_is(&self, dtype: DType) -> bool {
        self.val_dtype_kind() == Some(dtype)
    }

    pub fn val_dtype_or<'a>(&'a self, default: &'a str) -> &'a str {
        self.val_dtype().unwrap_or(default)
    }

    pub fn val_dtype_or_float32(&self) -> &str {
        self.val_dtype_or("torch.float32")
    }

    pub fn rank(&self) -> Option<usize> {
        self.val.as_ref().map(|v| v.rank())
    }

    pub fn val_dim_order(&self) -> Option<&[usize]> {
        self.val_dim_order.as_deref()
    }

    // --- Mutation helpers ---

    pub fn set_output_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.output_qparams = Some(map);
    }

    pub fn set_input_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.input_qparams = Some(map);
    }

    pub fn set_output_qparams_from(&mut self, pairs: impl IntoIterator<Item = (usize, QParam)>) {
        self.set_output_qparams(pairs.into_iter().collect());
    }

    pub fn set_input_qparams_from(&mut self, pairs: impl IntoIterator<Item = (usize, QParam)>) {
        self.set_input_qparams(pairs.into_iter().collect());
    }

    pub fn set_empty_input_qparams(&mut self) {
        self.set_input_qparams(HashMap::new());
    }

    pub fn set_single_output_qparam(&mut self, qp: QParam) {
        self.set_output_qparam_at(0, qp);
    }

    pub fn set_output_qparam_at(&mut self, index: usize, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(index, qp);
        self.set_output_qparams(map);
    }

    pub fn set_single_input_qparam(&mut self, qp: QParam) {
        self.set_input_qparam_at(0, qp);
    }

    pub fn set_input_qparam_at(&mut self, index: usize, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(index, qp);
        self.set_input_qparams(map);
    }

    pub fn set_both_qparams(&mut self, qp: QParam) {
        self.set_single_output_qparam(qp.clone());
        self.set_single_input_qparam(qp);
    }

    pub fn set_pre_pass_arg_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.pre_pass_arg_qparams = Some(map);
    }

    pub fn clear_output_qparams(&mut self) {
        self.output_qparams = None;
    }

    pub fn clear_input_qparams(&mut self) {
        self.input_qparams = None;
    }

    pub fn set_tosa_dim_order(&mut self, order: Vec<usize>) {
        self.tosa_dim_order = Some(order);
    }

    pub fn set_val_dim_order(&mut self, order: Vec<usize>) {
        self.val_dim_order = Some(order);
    }

    pub fn set_original_dim_orders(&mut self, orders: Vec<Vec<usize>>) {
        self.original_dim_orders = Some(orders);
    }

    pub fn set_tosa_spatial_rank(&mut self, rank: usize) -> bool {
        if self.tosa_spatial_rank == Some(rank) {
            return false;
        }
        self.tosa_spatial_rank = Some(rank);
        true
    }

    pub fn tosa_spatial_rank_or(&self, default: usize) -> usize {
        self.tosa_spatial_rank.unwrap_or(default)
    }

    pub fn set_val(&mut self, ftm: FakeTensorMeta) {
        self.val = Some(ftm);
    }

    pub fn set_val_dtype_for_existing_shape(&mut self, dtype: impl Into<String>) -> bool {
        let Some(shape) = self.val_shape().map(|shape| shape.to_vec()) else {
            return false;
        };
        self.set_val(FakeTensorMeta {
            shape,
            dtype: dtype.into(),
        });
        true
    }

    pub fn cloned_with_val(&self, shape: impl Into<Vec<i64>>, dtype: impl Into<String>) -> Self {
        let mut meta = self.clone();
        meta.set_val(FakeTensorMeta {
            shape: shape.into(),
            dtype: dtype.into(),
        });
        meta
    }

    pub fn cloned_without_qparams(
        &self,
        shape: impl Into<Vec<i64>>,
        dtype: impl Into<String>,
    ) -> Self {
        self.cloned_with_val(shape, dtype).without_qparams()
    }

    pub fn cloned_plain_tensor(
        &self,
        shape: impl Into<Vec<i64>>,
        dtype: impl Into<String>,
    ) -> Self {
        let mut meta = self.cloned_with_val(shape, dtype);
        meta.set_empty_input_qparams();
        meta.clear_output_qparams();
        meta.clear_tosa_special_dtype();
        meta
    }

    pub fn cloned_with_empty_qparam_maps(
        &self,
        shape: impl Into<Vec<i64>>,
        dtype: impl Into<String>,
    ) -> Self {
        let mut meta = self.cloned_with_val(shape, dtype);
        meta.set_empty_qparam_maps();
        meta
    }

    pub fn without_qparams(mut self) -> Self {
        self.clear_all_qparams();
        self
    }

    /// Get the single output qparam (key 0), if present.
    pub fn output_qparam(&self) -> Option<&QParam> {
        self.output_qparams.as_ref()?.get(&0)
    }

    /// Get the single input qparam (key 0), if present.
    pub fn input_qparam(&self) -> Option<&QParam> {
        self.input_qparam_at(0)
    }

    /// Get an input qparam by positional argument index, if present.
    pub fn input_qparam_at(&self, index: usize) -> Option<&QParam> {
        self.input_qparams.as_ref()?.get(&index)
    }

    pub fn input_qparams(&self) -> Option<&HashMap<usize, QParam>> {
        self.input_qparams.as_ref()
    }

    pub fn cloned_input_qparams(&self) -> Option<HashMap<usize, QParam>> {
        self.input_qparams.clone()
    }

    pub fn cloned_output_qparams(&self) -> Option<HashMap<usize, QParam>> {
        self.output_qparams.clone()
    }

    pub fn has_qparam_metadata(&self) -> bool {
        self.input_qparams.is_some() || self.output_qparams.is_some()
    }

    pub fn has_nonempty_qparam_metadata(&self) -> bool {
        self.input_qparams
            .as_ref()
            .is_some_and(|qparams| !qparams.is_empty())
            || self
                .output_qparams
                .as_ref()
                .is_some_and(|qparams| !qparams.is_empty())
    }

    pub fn has_nonempty_input_and_output_qparams(&self) -> bool {
        self.input_qparams
            .as_ref()
            .is_some_and(|qparams| !qparams.is_empty())
            && self
                .output_qparams
                .as_ref()
                .is_some_and(|qparams| !qparams.is_empty())
    }

    pub fn sorted_input_qparams(&self) -> Vec<(usize, QParam)> {
        let mut qparams: Vec<_> = self
            .input_qparams
            .as_ref()
            .map(|map| {
                map.iter()
                    .map(|(&index, qparam)| (index, qparam.clone()))
                    .collect()
            })
            .unwrap_or_default();
        qparams.sort_by_key(|(index, _)| *index);
        qparams
    }

    pub fn copy_input_qparams_from(&mut self, other: &Self) -> bool {
        let Some(iqp) = other.input_qparams.clone() else {
            return false;
        };
        self.set_input_qparams(iqp);
        true
    }

    pub fn copy_output_qparams_from(&mut self, other: &Self) -> bool {
        let Some(oqp) = other.output_qparams.clone() else {
            return false;
        };
        self.set_output_qparams(oqp);
        true
    }

    pub fn copy_qparams_from(&mut self, other: &Self) {
        self.copy_input_qparams_from(other);
        self.copy_output_qparams_from(other);
    }

    pub fn clear_all_qparams(&mut self) {
        self.clear_input_qparams();
        self.clear_output_qparams();
    }

    pub fn clear_tosa_special_dtype(&mut self) -> bool {
        self.extras.remove("tosa_special_dtype").is_some()
    }

    pub fn clear_tosa_dim_order_if_rank_mismatch(&mut self, rank: usize) -> bool {
        if self
            .tosa_dim_order
            .as_ref()
            .is_some_and(|order| order.len() != rank)
        {
            self.tosa_dim_order = None;
            true
        } else {
            false
        }
    }

    pub fn set_empty_qparam_maps(&mut self) {
        self.set_input_qparams(HashMap::new());
        self.set_output_qparams(HashMap::new());
    }

    pub fn rescale_output_meta(&self, output_qparam: QParam) -> Self {
        self.rescale_output_meta_with_shape(self, output_qparam)
    }

    pub fn rescale_output_meta_with_shape(
        &self,
        shape_source: &Self,
        output_qparam: QParam,
    ) -> Self {
        let mut meta = self.clone();
        meta.set_empty_input_qparams();
        meta.set_single_output_qparam(output_qparam.clone());
        if let Some(order) = meta.tosa_dim_order.clone() {
            meta.set_tosa_dim_order(order);
        }
        if let Some(orders) = meta.original_dim_orders.clone() {
            meta.set_original_dim_orders(orders);
        }
        meta.clear_tosa_special_dtype();
        if let Some(shape) = shape_source.val_shape() {
            meta.set_val(FakeTensorMeta {
                shape: shape.to_vec(),
                dtype: output_qparam.dtype_str,
            });
        }
        meta
    }

    pub fn rescale_meta_with_input_qparam(
        &self,
        shape_source: &Self,
        input_qparam: QParam,
        output_qparam: QParam,
    ) -> Self {
        let mut meta = self.rescale_output_meta_with_shape(shape_source, output_qparam);
        meta.set_single_input_qparam(input_qparam);
        meta
    }

    /// Set a custom extra meta key.
    pub fn set_extra(&mut self, key: impl Into<String>, value: MetaValue) {
        self.extras.insert(key.into(), value);
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
pub struct EffectfulIrPassResult {
    pub modified: bool,
    pub effects: LoweringEffectBatch,
}

impl EffectfulIrPassResult {
    pub fn changed(&self) -> bool {
        self.modified || !self.effects.items.is_empty()
    }
}

/// A Rust compiler pass that mutates `IrGraph` and may also record deferred
/// lowering effects applied during IR-to-serialize conversion.
pub trait EffectfulIrPass {
    fn name(&self) -> &str;

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult;
}

// ===========================================================================
// IrArg convenience methods
// ===========================================================================

impl IrArg {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            IrArg::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self {
            IrArg::Float(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_numeric_f64(&self) -> Option<f64> {
        match self {
            IrArg::Float(v) => Some(*v),
            IrArg::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_scalar_literal_f64(&self) -> Option<f64> {
        self.as_numeric_f64()
            .or_else(|| self.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            IrArg::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            IrArg::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_node(&self) -> Option<NodeId> {
        match self {
            IrArg::Node(id) => Some(*id),
            _ => None,
        }
    }
    pub fn contains_node(&self, target: NodeId) -> bool {
        match self {
            IrArg::Node(id) => *id == target,
            IrArg::List(items) => items.iter().any(|item| item.contains_node(target)),
            _ => false,
        }
    }
    pub fn as_int_list(&self) -> Option<Vec<i64>> {
        match self {
            IrArg::List(items) => items.iter().map(|a| a.as_int()).collect(),
            _ => None,
        }
    }

    pub fn as_int_list_or_scalar(&self) -> Option<Vec<i64>> {
        match self {
            IrArg::Int(v) => Some(vec![*v]),
            IrArg::List(_) => self.as_int_list(),
            _ => None,
        }
    }

    pub fn as_int_pair_or_scalar(&self) -> Option<(i64, i64)> {
        let values = self.as_int_list_or_scalar()?;
        match values.as_slice() {
            [v] => Some((*v, *v)),
            [a, b, ..] => Some((*a, *b)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node(id: NodeId, args: Vec<IrArg>, kwargs: HashMap<String, IrArg>) -> IrNode {
        IrNode {
            id,
            op: NodeOp::Other,
            name: format!("n{id}"),
            target: String::new(),
            args,
            kwargs,
            meta: NodeMeta::default(),
            tensor: None,
            alive: true,
        }
    }

    #[test]
    fn build_user_edges_includes_nested_args_and_kwargs() {
        let mut kwargs = HashMap::new();
        kwargs.insert("other".into(), IrArg::Node(0));
        let nodes = vec![
            test_node(0, vec![], HashMap::new()),
            test_node(1, vec![], HashMap::new()),
            test_node(
                2,
                vec![IrArg::List(vec![
                    IrArg::List(vec![IrArg::Node(0)]),
                    IrArg::Node(1),
                ])],
                kwargs,
            ),
        ];

        let users = build_user_edges(&nodes);

        assert_eq!(users[0], vec![2]);
        assert_eq!(users[1], vec![2]);
        assert!(users[2].is_empty());
    }
}

// ===========================================================================
// Hybrid pipeline: persistent IrGraph with deferred lowering effects
// ===========================================================================

/// A step in the hybrid pipeline: either a pure Rust `IrPass` or an `EffectfulIrPass`
/// with deferred lowering effects.
pub enum PipelineStep {
    Ir(Box<dyn IrPass>),
    Effectful(Box<dyn EffectfulIrPass>),
}

pub fn run_hybrid_pipeline_ext_with_initial_ir<'py>(
    _py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    exported_program: Option<&Bound<'py, PyAny>>,
    steps: &[PipelineStep],
    initial_ir: Option<IrGraph>,
) -> PyResult<(bool, Option<IrGraph>, LoweringEffectBatch)> {
    use crate::runner_logging::{log_pass_result, log_pass_start};

    let mut ir = initial_ir;
    let mut any_modified = false;
    let mut effects = LoweringEffectBatch::default();

    for (i, step) in steps.iter().enumerate() {
        match step {
            PipelineStep::Ir(pass) => {
                // Ensure we have an IrGraph
                if ir.is_none() {
                    ir = Some(IrGraph::from_fx_with_tensors(gm, exported_program)?);
                }
                let graph = ir.as_mut().unwrap();
                let before = graph.live_node_count();
                log_pass_start("HybridPipeline", i, steps.len(), pass.name());
                let modified = pass.run(graph);
                let after = graph.live_node_count();
                log_pass_result("HybridPipeline", pass.name(), modified, before, after);
                if modified {
                    any_modified = true;
                }
            }
            PipelineStep::Effectful(pass) => {
                if ir.is_none() {
                    ir = Some(IrGraph::from_fx_with_tensors(gm, exported_program)?);
                }
                let graph = ir.as_mut().unwrap();
                let before = graph.live_node_count();
                log_pass_start("HybridPipeline", i, steps.len(), pass.name());
                let result = pass.run(graph);
                let after = graph.live_node_count();
                log_pass_result(
                    "HybridPipeline",
                    pass.name(),
                    result.changed(),
                    before,
                    after,
                );
                if result.changed() {
                    any_modified = true;
                    effects.items.extend(result.effects.items);
                }
            }
        }
    }

    Ok((any_modified, ir, effects))
}
