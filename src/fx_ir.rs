//! Rust-native graph IR for FX graph manipulation.
//!
//! Defines imported graph topology, graph-surgery helpers, and the pass traits
//! used by the Rust lowering pipeline.

use std::collections::HashMap;

use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;

pub(crate) use crate::fx_meta::{MetaValue, NodeMeta, QParamMap, TensorMetaStamp};
use crate::fx_py_attr::{attrs, items};
pub(crate) use crate::fx_quant::QParam;
use crate::fx_target::{classify_ir_target, IrTargetKind};
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

/// Reduce an FX target string to the bare op name, e.g. "aten.relu.default".
///
/// `EdgeOpOverload` stringifies as
///   "<EdgeOpOverload: aten.foo.default>: schema = aten::foo(... bool transposed ...) -> ..."
/// Classification must see only the op name. Trailing schema text makes substring
/// matchers fire on operand names (`is_layout_passthrough_target` matched "transpose"
/// inside convolution's "bool transposed" and skipped the Int32->Int8 accumulator
/// rescale), and the wrapper defeats exact matches (a relu test must not also claim
/// "aten.leaky_relu.default").
pub(crate) fn normalize_fx_target(raw: &str) -> String {
    let without_schema = raw.split(": schema").next().unwrap_or(raw).trim();
    without_schema
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
        .and_then(|s| s.split_once(": ").map(|(_, op)| op))
        .unwrap_or(without_schema)
        .trim()
        .to_string()
}

/// A node in the Rust IR graph.
///
/// `IrGraph` indexes the node references in `args` and `kwargs`. Change them through
/// the graph's methods, or call `IrGraph::sync_node_users` after editing them in place.
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

pub(crate) fn normalize_dim(dim: i64, rank: usize) -> Option<usize> {
    let rank = rank as i64;
    if rank == 0 {
        return None;
    }
    let dim = if dim < 0 { dim + rank } else { dim };
    (0..rank).contains(&dim).then_some(dim as usize)
}

pub(crate) fn wrap_dim(dim: i64, rank: usize) -> Option<usize> {
    let rank = rank as i64;
    (rank > 0).then_some(dim.rem_euclid(rank) as usize)
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
    ) -> PyResult<Option<NodeTensor>> {
        let Some(op) = attrs(py_node).optional::<String>("op") else {
            return Ok(None);
        };
        match op.as_str() {
            "placeholder" => self.placeholder_tensor(name),
            "get_attr" => self.graph_attr_tensor(target),
            _ => Ok(None),
        }
    }

    fn placeholder_tensor(&self, name: &str) -> PyResult<Option<NodeTensor>> {
        let Some(binding) = self.placeholder_bindings.get(name) else {
            return Ok(None);
        };
        for mapping in [self.state_dict, self.constants] {
            if let Some(tensor) = self.placeholder_tensor_from_mapping(mapping, binding)? {
                return Ok(Some(tensor));
            }
        }
        self.placeholder_tensor_from_gm_attr(binding)
    }

    fn placeholder_tensor_from_mapping(
        &self,
        mapping: Option<&Bound<'py, PyAny>>,
        binding: &PlaceholderTensorBinding,
    ) -> PyResult<Option<NodeTensor>> {
        let Some(value) =
            mapping.and_then(|mapping| items(mapping).key(binding.target_name.as_str()))
        else {
            return Ok(None);
        };
        tensor_from_py_value(&value, TensorSource::ExportedProgramInput(binding.clone()))
    }

    fn placeholder_tensor_from_gm_attr(
        &self,
        binding: &PlaceholderTensorBinding,
    ) -> PyResult<Option<NodeTensor>> {
        let Ok(value) = attrs(self.gm).object(binding.target_name.as_str()) else {
            return Ok(None);
        };
        tensor_from_py_value(&value, TensorSource::ExportedProgramInput(binding.clone()))
    }

    fn graph_attr_tensor(&self, target: &str) -> PyResult<Option<NodeTensor>> {
        let Ok(value) = attrs(self.gm).object(target) else {
            return Ok(None);
        };
        if attrs(&value).has("graph") {
            return Ok(None);
        }
        tensor_from_py_value(&value, TensorSource::GraphModuleAttr)
    }
}

fn tensor_from_py_value(
    value: &Bound<'_, PyAny>,
    source: TensorSource,
) -> PyResult<Option<NodeTensor>> {
    match extract_tensor_data(value) {
        Ok(data) => Ok(Some(NodeTensor { source, data })),
        Err(error) if error.is_instance_of::<PyOverflowError>(value.py()) => Err(error),
        Err(_) => Ok(None),
    }
}

fn collect_fx_nodes<'py>(graph: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let nodes = attrs(graph)
        .object("nodes")?
        .try_iter()?
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
    let raw_target = py_node_attrs.required_str("target").unwrap_or_default();
    let target = normalize_fx_target(&raw_target);
    let op = NodeOp::of(py_node);
    let meta = NodeMeta::from_fx(py_node)?;
    let tensor = tensor_import.tensor_for_node(py_node, &name, &target)?;
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
    for arg in args.try_iter()? {
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

/// From-scratch reverse edges: `users[x]` lists each live node whose args or kwargs
/// reference `x` at any list depth, once, in ascending id order. References to ids
/// at or past `nodes.len()` are skipped.
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

/// The node ids `node` references in its args and kwargs, ascending and deduplicated.
/// Dead nodes reference nothing, matching `build_user_edges`.
fn node_operands(node: &IrNode) -> Vec<NodeId> {
    let mut operands = Vec::new();
    if node.alive {
        for arg in node.args.iter().chain(node.kwargs.values()) {
            arg.collect_node_ids(&mut operands);
        }
        operands.sort_unstable();
        operands.dedup();
    }
    operands
}

fn insert_sorted(list: &mut Vec<NodeId>, id: NodeId) {
    if let Err(position) = list.binary_search(&id) {
        list.insert(position, id);
    }
}

fn remove_sorted(list: &mut Vec<NodeId>, id: NodeId) {
    if let Ok(position) = list.binary_search(&id) {
        list.remove(position);
    }
}

/// How an edit affected the user lists that `users(id)` reported before the index
/// became incremental. See `IrGraph::legacy_users`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LegacyUsers {
    /// The old code rebuilt or updated those lists for this edit; the caller handles
    /// the overlay.
    Updated,
    /// The old code left those lists alone, so the overlay keeps each touched list as
    /// it was.
    Untouched,
}

#[cfg(test)]
thread_local! {
    /// Test builds check the incremental user index after every edit. Benchmarks on
    /// large graphs switch this off.
    static CHECK_USERS_AFTER_EDIT: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// The full graph: ordered list of nodes + lookup by name.
#[derive(Debug)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    name_to_id: HashMap<String, NodeId>,
    /// Exact reverse edges, updated by every edit: `users[x]` lists each live node whose
    /// args or kwargs reference `x` at any list depth, once, in ascending id order. It
    /// always equals `build_user_edges(&nodes)`.
    users: Vec<Vec<NodeId>>,
    /// `operands[u]`: the ids `u` references, ascending and deduplicated, as recorded in
    /// `users` (empty while `u` is dead). An edit to `u` diffs against it and touches
    /// only the lists that change.
    operands: Vec<Vec<NodeId>>,
    /// Users of ids at or past `nodes.len()` (forward references). `add_node` adopts
    /// them, so `users` keeps matching a from-scratch rebuild.
    pending_users: HashMap<NodeId, Vec<NodeId>>,
    /// Compatibility overlay for `users(id)`. Before the index became incremental,
    /// `users(id)` returned a cache that only `set_args`, `set_arg`,
    /// `replace_input_use` and `eliminate_dead_code` rebuilt. Between rebuilds it kept
    /// erased users, ignored kwargs edits and in-place edits, and `replace_all_uses`
    /// appended the moved users after the existing ones. Some `users(id)` callers can
    /// see the difference, so each entry here holds that cache's list for an id where
    /// it differs from `users`. The old rebuild points clear the overlay. Once those
    /// callers move to `live_user_ids`, the overlay can go.
    legacy_users: HashMap<NodeId, Vec<NodeId>>,
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
            operands: Vec::new(),
            pending_users: HashMap::new(),
            legacy_users: HashMap::new(),
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

        let mut graph = Self {
            nodes,
            name_to_id,
            users: Vec::new(),
            operands: Vec::new(),
            pending_users: HashMap::new(),
            legacy_users: HashMap::new(),
            replacements: HashMap::new(),
            anchors: HashMap::new(),
            anchors_after: HashMap::new(),
        };
        graph.rebuild_users();
        Ok(graph)
    }

    /// Iterate placeholders.
    /// Whether a node, live or erased, already has `name`.
    pub(crate) fn contains_name(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }

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

    /// Get mutable node by ID. After changing the node references in its `args` or
    /// `kwargs` through this, call `sync_node_users(id)`.
    pub fn node_mut(&mut self, id: NodeId) -> &mut IrNode {
        &mut self.nodes[id]
    }

    /// Update a node's args and their reverse-use entries.
    pub fn set_args(&mut self, id: NodeId, args: Vec<IrArg>) {
        self.nodes[id].args = args;
        self.sync_operands(id, LegacyUsers::Updated);
        self.reset_legacy_users();
        self.check_users_after_edit();
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
        // The old code never recorded kwargs references made here.
        self.sync_operands(id, LegacyUsers::Untouched);
        self.check_users_after_edit();
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_slice_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        dim: i64,
        start: i64,
        end: i64,
        step: Option<i64>,
        meta: NodeMeta,
    ) -> NodeId {
        let mut args = vec![
            IrArg::Node(source),
            IrArg::Int(dim),
            IrArg::Int(start),
            IrArg::Int(end),
        ];
        if let Some(step) = step {
            args.push(IrArg::Int(step));
        }
        self.add_call_function(name, "aten.slice_copy.Tensor", args, meta)
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

    /// Update a single arg by index. An out-of-range index is ignored.
    pub fn set_arg(&mut self, id: NodeId, index: usize, value: IrArg) {
        if index < self.nodes[id].args.len() {
            self.nodes[id].args[index] = value;
            self.sync_operands(id, LegacyUsers::Updated);
            self.reset_legacy_users();
            self.check_users_after_edit();
        }
    }

    /// Update a kwarg.
    pub fn set_kwarg(&mut self, id: NodeId, key: impl Into<String>, value: IrArg) {
        self.nodes[id].kwargs.insert(key.into(), value);
        self.sync_operands(id, LegacyUsers::Untouched);
        self.check_users_after_edit();
    }

    /// Remove a kwarg, returning its value.
    pub(crate) fn remove_kwarg(&mut self, id: NodeId, key: &str) -> Option<IrArg> {
        let removed = self.nodes[id].kwargs.remove(key);
        self.sync_operands(id, LegacyUsers::Untouched);
        self.check_users_after_edit();
        removed
    }

    /// Re-record `id`'s node references after its `args` or `kwargs` were edited in
    /// place through `node_mut`. O(references of `id`).
    pub(crate) fn sync_node_users(&mut self, id: NodeId) {
        self.sync_operands(id, LegacyUsers::Untouched);
        self.check_users_after_edit();
    }

    /// The user list this node had under the pre-incremental cache (see
    /// `legacy_users`): ascending ids right after a rebuild, but it can still hold
    /// erased users, miss kwargs and in-place edits, or list `replace_all_uses` moves
    /// out of order. Use `live_user_ids` for the exact list.
    pub fn users(&self, id: NodeId) -> &[NodeId] {
        self.legacy_users.get(&id).unwrap_or(&self.users[id])
    }

    /// Every live node that references `id` in its args or kwargs, once each, in
    /// ascending id order.
    fn exact_users(&self, id: NodeId) -> &[NodeId] {
        match self.users.get(id) {
            Some(users) => users,
            None => self.pending_users.get(&id).map_or(&[], Vec::as_slice),
        }
    }

    pub fn live_nodes(&self) -> impl Iterator<Item = &IrNode> {
        self.nodes.iter().filter(|node| node.alive)
    }

    pub fn live_node_count(&self) -> usize {
        self.live_nodes().count()
    }

    /// Every live node that references `id` in its args or kwargs, once each, in
    /// ascending id order. O(users of `id`).
    pub fn live_user_ids(&self, id: NodeId) -> Vec<NodeId> {
        self.exact_users(id).to_vec()
    }

    pub fn live_node_ids(&self) -> Vec<NodeId> {
        self.live_nodes().map(|node| node.id).collect()
    }

    pub(crate) fn topological_live_node_ids(&self) -> Vec<NodeId> {
        let mut predecessors = vec![Vec::new(); self.nodes.len()];
        for (&id, &anchor) in &self.anchors {
            predecessors[anchor].push(id);
        }
        for (&id, &anchor) in &self.anchors_after {
            predecessors[id].push(anchor);
        }
        for node in self.live_nodes() {
            for arg in node.args.iter().chain(node.kwargs.values()) {
                arg.collect_node_ids(&mut predecessors[node.id]);
            }
        }
        for deps in &mut predecessors {
            deps.sort_unstable();
            deps.dedup();
        }

        let mut visited = vec![false; self.nodes.len()];
        let mut result = Vec::new();
        // Dead anchors retain the position of their replacement chains.
        for start in 0..self.nodes.len() {
            let mut stack = vec![(start, false)];
            while let Some((id, post)) = stack.pop() {
                if post {
                    if self.nodes[id].alive {
                        result.push(id);
                    }
                } else if !visited[id] {
                    visited[id] = true;
                    stack.push((id, true));
                    stack.extend(predecessors[id].iter().rev().map(|&dep| (dep, false)));
                }
            }
        }
        result
    }

    /// Renumber the nodes so that `order`, which lists every live node once,
    /// becomes the node-vector order; dead nodes follow in their old relative
    /// order. A fresh `from_fx_with_tensors` graph has this shape (ids in FX
    /// node order), and when `order` is topological both
    /// `topological_live_node_ids` and `ingest::from_ir`, which seeds its
    /// def-use sort with id order, then emit the live nodes in `order`. Node
    /// references, user lists, the name index and the replacement map are
    /// rewritten; the position anchors are dropped because `order` already
    /// places every node. Returns the old-to-new id map.
    pub(crate) fn reorder_nodes(&mut self, order: &[NodeId]) -> Vec<NodeId> {
        fn remap(arg: &mut IrArg, new_ids: &[NodeId]) {
            match arg {
                IrArg::Node(id) => {
                    if let Some(&new_id) = new_ids.get(*id) {
                        *id = new_id;
                    }
                }
                IrArg::List(items) => {
                    for item in items {
                        remap(item, new_ids);
                    }
                }
                _ => {}
            }
        }

        const UNASSIGNED: NodeId = NodeId::MAX;
        let mut new_ids = vec![UNASSIGNED; self.nodes.len()];
        let mut next_id = 0;
        for &id in order {
            assert!(
                self.nodes[id].alive && new_ids[id] == UNASSIGNED,
                "reorder_nodes: node {id} is dead or listed twice"
            );
            new_ids[id] = next_id;
            next_id += 1;
        }
        for (id, new_id) in new_ids.iter_mut().enumerate() {
            if *new_id == UNASSIGNED {
                assert!(
                    !self.nodes[id].alive,
                    "reorder_nodes: live node {id} is missing from the order"
                );
                *new_id = next_id;
                next_id += 1;
            }
        }

        let mut nodes = std::mem::take(&mut self.nodes);
        nodes.sort_unstable_by_key(|node| new_ids[node.id]);
        for node in &mut nodes {
            node.id = new_ids[node.id];
            for arg in node.args.iter_mut().chain(node.kwargs.values_mut()) {
                remap(arg, &new_ids);
            }
        }
        self.nodes = nodes;
        for id in self.name_to_id.values_mut() {
            *id = new_ids[*id];
        }
        self.replacements = self
            .replacements
            .drain()
            .map(|(dead, replacement)| (new_ids[dead], new_ids[replacement]))
            .collect();
        self.anchors.clear();
        self.anchors_after.clear();
        self.rebuild_users();
        new_ids
    }

    pub(crate) fn constant_shape(&self, arg: &IrArg) -> Option<Vec<i64>> {
        if let IrArg::Node(id) = arg {
            let node = self.node(*id);
            if !node.alive
                || node.op != NodeOp::CallFunction
                || !matches!(
                    node.target.as_str(),
                    "backend.tosa.CONST_SHAPE.default" | "tosa.CONST_SHAPE.default"
                )
            {
                return None;
            }
            node.args.first()?.as_int_list()
        } else {
            arg.as_int_list()
        }
    }

    pub(crate) fn shape_in_node_order(
        &self,
        node: NodeId,
        input: NodeId,
        shape: &[i64],
    ) -> Option<Vec<i64>> {
        let input_order = self.node(input).meta.tosa_dim_order();
        let output_order = self.node(node).meta.tosa_dim_order().or(input_order);
        let physical = if let Some(order) = input_order {
            if order.len() != shape.len() {
                return None;
            }
            order
                .iter()
                .map(|&axis| shape.get(axis).copied())
                .collect::<Option<Vec<_>>>()?
        } else {
            shape.to_vec()
        };
        let mut logical = physical.clone();
        if let Some(order) = output_order {
            if order.len() != shape.len() {
                return None;
            }
            for (index, &axis) in order.iter().enumerate() {
                *logical.get_mut(axis)? = physical[index];
            }
        }
        Some(logical)
    }

    pub(crate) fn logical_input_shape(&self, node: NodeId, input: NodeId) -> Option<Vec<i64>> {
        let mut current = input;
        let mut permutations = Vec::new();
        while crate::fx_target::is_tosa_transpose_target(&self.node(current).target)
            || crate::fx_target::is_permute_target(&self.node(current).target)
        {
            permutations.push(self.node(current).arg_int_list(1)?);
            current = self.node(current).arg_node(0)?;
        }
        let mut physical = self.node_shape_vec(current)?;
        if let Some(order) = self.node(current).meta.tosa_dim_order() {
            if order.len() != physical.len() {
                return None;
            }
            physical = order
                .iter()
                .map(|&axis| physical.get(axis).copied())
                .collect::<Option<Vec<_>>>()?;
        }
        // A generated transpose's val and dim-order metadata both describe
        // its layout; derive its physical shape from the operand and perms.
        for permutation in permutations.into_iter().rev() {
            if permutation.len() != physical.len() {
                return None;
            }
            physical = permutation
                .into_iter()
                .map(|axis| normalize_dim(axis, physical.len()).map(|axis| physical[axis]))
                .collect::<Option<Vec<_>>>()?;
        }
        let mut logical = physical.clone();
        if let Some(order) = self
            .node(node)
            .meta
            .tosa_dim_order()
            .or(self.node(input).meta.tosa_dim_order())
        {
            if order.len() != physical.len() {
                return None;
            }
            for (index, &axis) in order.iter().enumerate() {
                *logical.get_mut(axis)? = physical[index];
            }
        }
        Some(logical)
    }

    pub(crate) fn eliminate_dead_code(&mut self) -> bool {
        self.eliminate_dead_code_keeping(|_| false)
    }

    /// `eliminate_dead_code`, with the live nodes `keep` accepts as extra roots.
    pub(crate) fn eliminate_dead_code_keeping(&mut self, keep: impl Fn(&IrNode) -> bool) -> bool {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack: Vec<_> = self
            .live_nodes()
            .filter(|node| {
                matches!(
                    node.op,
                    NodeOp::Placeholder | NodeOp::Output | NodeOp::Other
                ) || keep(node)
            })
            .map(|node| node.id)
            .collect();
        while let Some(id) = stack.pop() {
            if reachable[id] || !self.nodes[id].alive {
                continue;
            }
            reachable[id] = true;
            for arg in self.nodes[id]
                .args
                .iter()
                .chain(self.nodes[id].kwargs.values())
            {
                arg.collect_node_ids(&mut stack);
            }
        }
        let mut erased = Vec::new();
        for node in &mut self.nodes {
            if node.alive && !reachable[node.id] {
                node.alive = false;
                erased.push(node.id);
            }
        }
        for &id in &erased {
            self.sync_operands(id, LegacyUsers::Updated);
        }
        let modified = !erased.is_empty();
        if modified {
            self.reset_legacy_users();
            self.check_users_after_edit();
        }
        modified
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
        !self.exact_users(id).is_empty()
    }

    pub fn single_live_user(&self, id: NodeId) -> Option<NodeId> {
        match self.exact_users(id) {
            [user] => Some(*user),
            _ => None,
        }
    }

    pub fn live_call_function_users(&self, id: NodeId) -> Vec<NodeId> {
        self.exact_users(id)
            .iter()
            .copied()
            .filter(|&user_id| self.nodes[user_id].op == NodeOp::CallFunction)
            .collect()
    }

    /// Live getitem nodes whose first arg is `id`, in ascending id order, with their
    /// index arg (default 0).
    pub fn live_getitem_users(&self, id: NodeId) -> Vec<(NodeId, i64)> {
        self.exact_users(id)
            .iter()
            .map(|&user_id| &self.nodes[user_id])
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
        // Every live node that references `old_id` is in its exact user list. The old
        // code rewrote every live node, which changes the same set.
        let rewired = self.users[old_id].clone();
        if old_id == new_id {
            return;
        }
        for &user_id in &rewired {
            let node = &mut self.nodes[user_id];
            for arg in node.args.iter_mut().chain(node.kwargs.values_mut()) {
                replace_node_in_arg(arg, old_id, new_id);
            }
        }
        // The old cache moved its list for `old_id` onto the end of the list for
        // `new_id`, skipping ids already there. Read both before the exact lists change.
        // The list for `new_id` needs an overlay entry if the move appends to it or the
        // exact list changes under it.
        let moved = self
            .legacy_users
            .remove(&old_id)
            .unwrap_or_else(|| self.users[old_id].clone());
        let pin_new = !moved.is_empty() || (!rewired.is_empty() && new_id < self.users.len());
        let legacy_new = pin_new.then(|| {
            self.legacy_users
                .get(&new_id)
                .unwrap_or(&self.users[new_id])
                .clone()
        });
        for &user_id in &rewired {
            self.sync_operands(user_id, LegacyUsers::Updated);
        }
        if let Some(mut list) = legacy_new {
            for user_id in moved {
                if !list.contains(&user_id) {
                    list.push(user_id);
                }
            }
            self.legacy_users.insert(new_id, list);
        }
        self.check_users_after_edit();
    }

    /// Replace uses of `old_id` with `new_id` in a single user node.
    pub fn replace_input_use(&mut self, user_id: NodeId, old_id: NodeId, new_id: NodeId) -> bool {
        let mut changed = false;
        if let Some(node) = self.nodes.get_mut(user_id) {
            if !node.alive {
                return false;
            }
            for arg in node.args.iter_mut().chain(node.kwargs.values_mut()) {
                changed |= replace_node_in_arg(arg, old_id, new_id);
            }
        }
        if changed {
            self.sync_operands(user_id, LegacyUsers::Updated);
            self.reset_legacy_users();
            self.check_users_after_edit();
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
        // The old cache kept the erased node in its operands' user lists.
        self.sync_operands(id, LegacyUsers::Untouched);
        self.check_users_after_edit();
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
        self.users
            .push(self.pending_users.remove(&id).unwrap_or_default());
        self.operands.push(Vec::new());
        self.sync_operands(id, LegacyUsers::Updated);
        // The old cache appended `id` to the list of each operand below `id`, as the
        // exact lists just did (`id` is the largest id), and started an empty list for
        // `id`. Only overlay entries and a non-empty list for `id` need fixing.
        for &operand in &self.operands[id] {
            if operand < id {
                if let Some(list) = self.legacy_users.get_mut(&operand) {
                    if !list.contains(&id) {
                        list.push(id);
                    }
                }
            }
        }
        if !self.users[id].is_empty() {
            self.legacy_users.insert(id, Vec::new());
        }
        self.check_users_after_edit();
        id
    }

    /// Recompute every user list from scratch in O(nodes + references). Edits through
    /// the graph methods keep the lists current; this is for bulk in-place edits. Like
    /// the old full rebuild, it resets `users(id)` to the exact lists.
    pub(crate) fn rebuild_users(&mut self) {
        self.users = build_user_edges(&self.nodes);
        self.operands = self.nodes.iter().map(node_operands).collect();
        self.pending_users.clear();
        for (user_id, operands) in self.operands.iter().enumerate() {
            for &operand in operands
                .iter()
                .filter(|&&operand| operand >= self.nodes.len())
            {
                self.pending_users.entry(operand).or_default().push(user_id);
            }
        }
        self.reset_legacy_users();
        self.check_users_after_edit();
    }

    /// Mark an edit that rebuilt every user list in the old code, which made
    /// `users(id)` exact again.
    fn reset_legacy_users(&mut self) {
        if !self.legacy_users.is_empty() {
            // A fresh map, not `clear()`: clearing keeps a large table to sweep again.
            self.legacy_users = HashMap::new();
        }
    }

    /// Bring the user lists in line with `id`'s current args and kwargs (none if `id` is
    /// dead), touching only the lists of operands that were added or dropped.
    fn sync_operands(&mut self, id: NodeId, legacy: LegacyUsers) {
        let current = node_operands(&self.nodes[id]);
        let recorded = std::mem::take(&mut self.operands[id]);
        for &operand in &recorded {
            if current.binary_search(&operand).is_err() {
                self.record_use(operand, id, false, legacy);
            }
        }
        for &operand in &current {
            if recorded.binary_search(&operand).is_err() {
                self.record_use(operand, id, true, legacy);
            }
        }
        self.operands[id] = current;
    }

    /// Add `user` to (or drop it from) the exact user list of `operand`.
    fn record_use(&mut self, operand: NodeId, user: NodeId, used: bool, legacy: LegacyUsers) {
        let Some(list) = self.users.get_mut(operand) else {
            let pending = self.pending_users.entry(operand).or_default();
            if used {
                insert_sorted(pending, user);
            } else {
                remove_sorted(pending, user);
                if pending.is_empty() {
                    self.pending_users.remove(&operand);
                }
            }
            return;
        };
        if legacy == LegacyUsers::Untouched {
            self.legacy_users
                .entry(operand)
                .or_insert_with(|| list.clone());
        }
        if used {
            insert_sorted(list, user);
        } else {
            remove_sorted(list, user);
        }
    }

    /// Test builds check the user index against a from-scratch rebuild after each edit.
    fn check_users_after_edit(&self) {
        #[cfg(test)]
        if CHECK_USERS_AFTER_EDIT.with(std::cell::Cell::get) {
            self.assert_users_consistent();
        }
    }

    /// Assert that the incremental user index matches a from-scratch rebuild.
    #[cfg(test)]
    pub(crate) fn assert_users_consistent(&self) {
        assert_eq!(
            self.users,
            build_user_edges(&self.nodes),
            "incremental user lists differ from a rebuild"
        );
        assert_eq!(self.operands.len(), self.nodes.len());
        let mut pending: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in &self.nodes {
            let operands = node_operands(node);
            for &operand in operands
                .iter()
                .filter(|&&operand| operand >= self.nodes.len())
            {
                pending.entry(operand).or_default().push(node.id);
            }
            assert_eq!(
                self.operands[node.id], operands,
                "recorded operands of node {} are stale",
                node.id
            );
        }
        assert_eq!(
            self.pending_users, pending,
            "forward-reference users are stale"
        );
        assert!(self.legacy_users.keys().all(|&id| id < self.nodes.len()));
    }
}

/// Rewrite every reference to `old` in `arg` (at any list depth) to `new`.
fn replace_node_in_arg(arg: &mut IrArg, old: NodeId, new: NodeId) -> bool {
    match arg {
        IrArg::Node(id) if *id == old => {
            *id = new;
            true
        }
        IrArg::List(items) => {
            let mut changed = false;
            for item in items {
                changed |= replace_node_in_arg(item, old, new);
            }
            changed
        }
        _ => false,
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
        let dtype_str = match &self.args[5] {
            IrArg::Dtype(s) | IrArg::Str(s) => s.clone(),
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

/// Cursor for emitting a topo-sensitive rewrite chain at a single anchor.
///
/// The first inserted node is anchored to the original node. Each later node is
/// anchored after the previous inserted node, so passes do not have to carry
/// local `prev` state or repeat `anchor_to`/`anchor_after` sequencing.
pub(crate) struct IrRewriteCursor<'a> {
    graph: &'a mut IrGraph,
    anchor: NodeId,
    previous: Option<NodeId>,
}

impl IrGraph {
    pub(crate) fn rewrite_cursor(&mut self, anchor: NodeId) -> IrRewriteCursor<'_> {
        IrRewriteCursor {
            graph: self,
            anchor,
            previous: None,
        }
    }

    pub(crate) fn rewrite_cursor_after(
        &mut self,
        anchor: NodeId,
        previous: NodeId,
    ) -> IrRewriteCursor<'_> {
        IrRewriteCursor {
            graph: self,
            anchor,
            previous: Some(previous),
        }
    }
}

impl IrRewriteCursor<'_> {
    pub(crate) fn graph(&self) -> &IrGraph {
        self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut IrGraph {
        self.graph
    }

    pub(crate) fn append_existing(&mut self, id: NodeId) -> NodeId {
        if let Some(previous) = self.previous {
            self.graph.anchor_after(id, previous);
        } else {
            self.graph.anchor_to(id, self.anchor);
        }
        self.previous = Some(id);
        id
    }

    pub(crate) fn call_function(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_call_function(name, target, args, meta);
        self.append_existing(id)
    }

    pub(crate) fn call_function_with_kwargs(
        &mut self,
        name: impl Into<String>,
        target: impl Into<String>,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self
            .graph
            .add_call_function_with_kwargs(name, target, args, kwargs, meta);
        self.append_existing(id)
    }

    pub(crate) fn dtype_cast(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        dtype: impl Into<String>,
        meta: NodeMeta,
    ) -> NodeId {
        self.call_function_with_kwargs(
            name,
            "aten._to_copy.default",
            vec![IrArg::Node(source)],
            dtype_kwarg(dtype),
            meta,
        )
    }

    pub(crate) fn dtype_cast_if_needed(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        if self.graph.node(source).meta.val_dtype() == Some(dtype) {
            source
        } else {
            self.dtype_cast(name, source, dtype, meta)
        }
    }

    pub(crate) fn view_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        shape: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_view_copy(name, source, shape, meta);
        self.append_existing(id)
    }

    pub(crate) fn permute_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        perm: &[i64],
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_permute_copy(name, source, perm, meta);
        self.append_existing(id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn slice_copy(
        &mut self,
        name: impl Into<String>,
        source: NodeId,
        dim: i64,
        start: i64,
        end: i64,
        step: Option<i64>,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self
            .graph
            .add_slice_copy(name, source, dim, start, end, step, meta);
        self.append_existing(id)
    }

    pub(crate) fn full_int(
        &mut self,
        name: impl Into<String>,
        shape: &[i64],
        value: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_full_int(name, shape, value, dtype, meta);
        self.append_existing(id)
    }

    pub(crate) fn full_float(
        &mut self,
        name: impl Into<String>,
        shape: &[i64],
        value: f64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_full_float(name, shape, value, dtype, meta);
        self.append_existing(id)
    }

    pub(crate) fn arange(
        &mut self,
        name: impl Into<String>,
        start: i64,
        end: i64,
        step: i64,
        dtype: &str,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.add_arange(name, start, end, step, dtype, meta);
        self.append_existing(id)
    }

    pub(crate) fn rescale(&mut self, spec: RescaleRewrite) -> NodeId {
        let id = self.graph.add_rescale(spec);
        self.append_existing(id)
    }

    pub(crate) fn replace_anchor_with(&mut self, replacement: NodeId) -> NodeId {
        self.graph.replace_and_erase(self.anchor, replacement);
        replacement
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
    if let Ok(iter) = py_arg.try_iter() {
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
    pub(crate) fn collect_node_ids(&self, ids: &mut Vec<NodeId>) {
        match self {
            IrArg::Node(id) => ids.push(*id),
            IrArg::List(items) => {
                for item in items {
                    item.collect_node_ids(ids);
                }
            }
            _ => {}
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

// ===========================================================================
// IR pipeline: one persistent IrGraph, with lowering effects deferred to serialization
// ===========================================================================

/// A pipeline step: a pure `IrPass`, or an `EffectfulIrPass` that also records
/// deferred lowering effects.
pub enum PipelineStep {
    Ir(Box<dyn IrPass>),
    Effectful(Box<dyn EffectfulIrPass>),
}

/// Run `steps` over `graph` and collect the effectful passes' lowering effects.
pub fn run_ir_pipeline(
    steps: &[PipelineStep],
    mut graph: IrGraph,
) -> (bool, IrGraph, LoweringEffectBatch) {
    use crate::runner_logging::{log_pass_result, log_pass_start};

    let mut any_modified = false;
    let mut effects = LoweringEffectBatch::default();
    let probe = crate::config::debug_node_probe();
    let probe_after = |graph: &IrGraph, pass: &str| {
        let Some(needle) = probe.as_deref() else {
            return;
        };
        for node in graph.live_nodes().filter(|node| node.name.contains(needle)) {
            eprintln!(
                "[NodeProbe] after {pass}: {} target={} shape={:?} dtype={:?} out_qparam={:?}",
                node.name,
                node.target,
                graph.node_shape(node.id),
                node.meta.val_dtype(),
                node.meta
                    .output_qparam()
                    .map(|q| (q.scale, q.zp, q.dtype_str.clone())),
            );
        }
    };

    for (i, step) in steps.iter().enumerate() {
        let before = graph.live_node_count();
        let (name, modified) = match step {
            PipelineStep::Ir(pass) => {
                log_pass_start("IrPipeline", i, steps.len(), pass.name());
                (pass.name(), pass.run(&mut graph))
            }
            PipelineStep::Effectful(pass) => {
                log_pass_start("IrPipeline", i, steps.len(), pass.name());
                let result = pass.run(&mut graph);
                let changed = result.changed();
                if changed {
                    effects.items.extend(result.effects.items);
                }
                (pass.name(), changed)
            }
        };
        log_pass_result("IrPipeline", name, modified, before, graph.live_node_count());
        probe_after(&graph, name);
        any_modified |= modified;
    }

    (any_modified, graph, effects)
}

#[cfg(test)]
mod users_tests;

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
    fn normalize_fx_target_strips_edge_wrapper_and_schema() {
        assert_eq!(
            normalize_fx_target(
                "<EdgeOpOverload: aten.leaky_relu.default>: schema = aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"
            ),
            "aten.leaky_relu.default"
        );
        assert_eq!(
            normalize_fx_target("<EdgeOpOverload: aten.relu.default>"),
            "aten.relu.default"
        );
        assert_eq!(
            normalize_fx_target("aten.relu.default"),
            "aten.relu.default"
        );
        assert_eq!(
            normalize_fx_target("<built-in function getitem>"),
            "<built-in function getitem>"
        );
    }

    #[test]
    fn parse_qdq_preserves_typed_narrow_range_int16() {
        let node = test_node(
            0,
            vec![
                IrArg::Node(1),
                IrArg::Float(0.25),
                IrArg::Int(0),
                IrArg::Int(-32767),
                IrArg::Int(32767),
                IrArg::Dtype("torch.int16".into()),
            ],
            HashMap::new(),
        );
        let qparam = node.parse_qdq_qparam().unwrap();
        assert_eq!(qparam.dtype_str, "torch.int16");
        assert_eq!(qparam.qmin, -32767);
        assert_eq!(qparam.qmax, 32767);
        assert_eq!(qparam.scale, 0.25);
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

    #[test]
    fn rewrite_cursor_anchors_first_node_to_anchor_and_later_nodes_after_previous() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            vec![],
            NodeMeta::tensor([1], "torch.float32"),
        );
        let anchor = graph.add_node(
            NodeOp::CallFunction,
            "anchor".into(),
            "aten.neg.default".into(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor([1], "torch.float32"),
        );

        let mut cursor = graph.rewrite_cursor(anchor);
        let view = cursor.view_copy(
            "view",
            x,
            &[1, 1],
            NodeMeta::tensor([1, 1], "torch.float32"),
        );
        let neg = cursor.call_function(
            "neg",
            "aten.neg.default",
            vec![IrArg::Node(view)],
            NodeMeta::tensor([1, 1], "torch.float32"),
        );

        assert_eq!(graph.anchors.get(&view), Some(&anchor));
        assert_eq!(graph.anchors_after.get(&neg), Some(&view));
    }

    #[test]
    fn rewrite_cursor_replace_anchor_rewires_users() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            vec![],
            NodeMeta::tensor([1], "torch.float32"),
        );
        let anchor = graph.add_node(
            NodeOp::CallFunction,
            "anchor".into(),
            "aten.neg.default".into(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor([1], "torch.float32"),
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".into(),
            "aten.relu.default".into(),
            vec![IrArg::Node(anchor)],
            NodeMeta::tensor([1], "torch.float32"),
        );

        let mut cursor = graph.rewrite_cursor(anchor);
        let replacement = cursor.call_function(
            "replacement",
            "aten.abs.default",
            vec![IrArg::Node(x)],
            NodeMeta::tensor([1], "torch.float32"),
        );
        cursor.replace_anchor_with(replacement);

        assert!(!graph.nodes[anchor].alive);
        assert_eq!(graph.nodes[user].args[0].as_node(), Some(replacement));
    }

    #[test]
    fn rewrite_cursor_after_anchors_first_node_after_existing_predecessor() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            vec![],
            NodeMeta::tensor([2], "torch.float32"),
        );
        let anchor = graph.add_node(
            NodeOp::CallFunction,
            "anchor".into(),
            "aten.relu.default".into(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor([2], "torch.float32"),
        );

        let mut cursor = graph.rewrite_cursor_after(anchor, x);
        let slice = cursor.slice_copy(
            "slice",
            x,
            0,
            0,
            1,
            None,
            NodeMeta::tensor([1], "torch.float32"),
        );

        assert_eq!(graph.anchors_after.get(&slice), Some(&x));
        assert!(!graph.anchors.contains_key(&slice));
    }
}
