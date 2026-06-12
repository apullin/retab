//! Control-flow metadata stamping, boundary RESCALE insertion, and Q/DQ
//! rewriting for cond and while_loop submodules.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::fx_graph::{meta_val, py_attr, FxGraphModule};
use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrNode, IrPass, NodeId, NodeMeta};
use crate::fx_pass::FxPass;
use crate::runner_logging::log_line;

/// IrPass wrapper for the while-loop boundary Q/DQ strip (submodule IR pipeline).
pub struct IrStripWhileBoundaryQPass;
impl IrPass for IrStripWhileBoundaryQPass {
    fn name(&self) -> &str {
        "IrStripWhileBoundaryQ"
    }
    fn run(&self, graph: &mut IrGraph) -> bool {
        strip_while_boundary_qdq_ir(graph)
    }
}

/// IrPass: align each while_loop body output to its corresponding placeholder's
/// quantization domain by inserting a RESCALE (output[i] feeds back as input[i],
/// so their scale/zp must match). The IR analogue of FxInsertRescalePass::carried.
pub struct IrInsertCarriedOutputRescalePass;
impl IrPass for IrInsertCarriedOutputRescalePass {
    fn name(&self) -> &str {
        "IrInsertCarriedOutputRescale"
    }
    fn run(&self, graph: &mut IrGraph) -> bool {
        let ph_qps: Vec<Option<QParam>> = graph
            .placeholders()
            .map(|ph| ph.meta.output_qparam().or(ph.meta.input_qparam()).cloned())
            .collect();
        if ph_qps.iter().all(Option::is_none) {
            return false;
        }
        let Some(out_id) = graph.output_node().map(|n| n.id) else {
            return false;
        };
        let out_args = graph.node(out_id).args.clone();
        let val_ids: Vec<Option<NodeId>> = match out_args.first() {
            Some(IrArg::List(items)) => items
                .iter()
                .map(|a| match a {
                    IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect(),
            _ => out_args
                .iter()
                .map(|a| match a {
                    IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect(),
        };

        let mut modified = false;
        for (idx, ph_qp) in ph_qps.iter().enumerate() {
            let Some(ph_qp) = ph_qp else { continue };
            let Some(Some(src_id)) = val_ids.get(idx).copied() else {
                continue;
            };
            let Some(cur) = extract_effective_qparam_ir(graph, src_id) else {
                continue;
            };
            if cur.same_scale_zp(ph_qp) || cur.scale == 0.0 || ph_qp.scale == 0.0 {
                continue;
            }

            let mut meta = graph.node(src_id).meta.clone();
            meta.set_input_qparams(HashMap::new());
            meta.set_single_output_qparam(ph_qp.clone());
            meta.extras.remove("tosa_special_dtype");
            if let Some(shape) = graph.node(src_id).meta.val_shape() {
                meta.set_val(FakeTensorMeta {
                    shape: shape.to_vec(),
                    dtype: ph_qp.dtype_str.clone(),
                });
            }
            let src_name = graph.node(src_id).name.clone();
            let rescale = graph.add_node(
                NodeOp::CallFunction,
                format!("{src_name}_carried_rescale"),
                "backend.tosa.RESCALE.default".to_string(),
                vec![
                    IrArg::Node(src_id),
                    IrArg::Dtype(ph_qp.dtype_str.clone()),
                    IrArg::List(vec![IrArg::Float(cur.scale / ph_qp.scale)]),
                    IrArg::Int(cur.zp),
                    IrArg::Int(ph_qp.zp),
                ],
                meta,
            );
            graph.anchor_after(rescale, src_id);
            graph.copy_meta_from(rescale, src_id);
            replace_output_arg_use(graph, out_id, src_id, rescale);
            modified = true;
        }
        modified
    }
}

/// Replace a use of `old` with `new` in the output node's args (handles the
/// `output((v0, v1, ...),)` tuple-wrapped form).
fn replace_output_arg_use(graph: &mut IrGraph, out_id: NodeId, old: NodeId, new: NodeId) {
    let node = graph.node_mut(out_id);
    for arg in node.args.iter_mut() {
        match arg {
            IrArg::Node(id) if *id == old => *id = new,
            IrArg::List(items) => {
                for it in items.iter_mut() {
                    if let IrArg::Node(id) = it {
                        if *id == old {
                            *id = new;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// CfCtx — cached Python imports and shared context
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfKind {
    Cond,
    While,
}

/// Cached Python module imports for control-flow operations.
struct CfCtx<'py> {
    cond_op: Bound<'py, PyAny>,
    while_op: Bound<'py, PyAny>,
}

impl<'py> CfCtx<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let torch = py.import_bound("torch")?.into_any();
        let ho = torch.getattr("ops")?.getattr("higher_order")?;
        let cond_op = ho.getattr("cond")?;
        let while_op = ho.getattr("while_loop")?;
        Ok(Self { cond_op, while_op })
    }

    fn cf_kind(&self, node: &Bound<'py, PyAny>) -> PyResult<Option<CfKind>> {
        let target = node.getattr("target")?;
        if target.eq(&self.cond_op).unwrap_or(false) {
            Ok(Some(CfKind::Cond))
        } else if target.eq(&self.while_op).unwrap_or(false) {
            Ok(Some(CfKind::While))
        } else {
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// QParam — Rust-native quantization parameter struct
// ---------------------------------------------------------------------------

/// Quantization parameters extracted once from Python QuantArgs objects.
/// Eliminates repeated getattr chains throughout the control-flow code.
#[derive(Debug, Clone)]
pub(crate) struct QParam {
    pub scale: f64,
    pub zp: i64,
    pub qmin: i64,
    pub qmax: i64,
    pub dtype_str: String,
    /// Per-channel scales (empty for per-tensor quantization).
    pub per_channel_scales: Vec<f64>,
    /// Per-channel zero points (empty for per-tensor quantization).
    pub per_channel_zps: Vec<i64>,
    /// Quantization axis for per-channel (default 0).
    pub axis: i64,
}

impl QParam {
    /// Extract from a Python QuantArgs object (or anything with scale/zp/dtype attrs).
    pub(crate) fn from_py(qp: &Bound<'_, PyAny>) -> Self {
        let per_channel = qp
            .getattr("per_channel")
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(false);
        let (per_channel_scales, per_channel_zps) = if per_channel {
            let scales = qp
                .getattr("scale")
                .and_then(|v| v.extract::<Vec<f64>>())
                .unwrap_or_default();
            let zps = qp
                .getattr("zp")
                .and_then(|v| v.extract::<Vec<i64>>())
                .or_else(|_| {
                    qp.getattr("zero_point")
                        .and_then(|v| v.extract::<Vec<i64>>())
                })
                .unwrap_or_default();
            (scales, zps)
        } else {
            (Vec::new(), Vec::new())
        };
        let axis = qp.getattr("axis").and_then(|v| v.extract()).unwrap_or(0);
        Self {
            scale: qp.getattr("scale").and_then(|v| v.extract()).unwrap_or(1.0),
            zp: qp
                .getattr("zp")
                .and_then(|v| v.extract())
                .or_else(|_| qp.getattr("zero_point").and_then(|v| v.extract()))
                .unwrap_or(0),
            qmin: qp.getattr("qmin").and_then(|v| v.extract()).unwrap_or(-128),
            qmax: qp.getattr("qmax").and_then(|v| v.extract()).unwrap_or(127),
            dtype_str: qp
                .getattr("dtype")
                .ok()
                .map(|d| format!("{d}"))
                .unwrap_or_default(),
            per_channel_scales,
            per_channel_zps,
            axis,
        }
    }

    /// Create a per-tensor QParam (most common case).
    pub fn per_tensor(scale: f64, zp: i64, qmin: i64, qmax: i64, dtype_str: String) -> Self {
        Self {
            scale,
            zp,
            qmin,
            qmax,
            dtype_str,
            per_channel_scales: Vec::new(),
            per_channel_zps: Vec::new(),
            axis: 0,
        }
    }

    /// Is this per-channel quantization?
    pub fn is_per_channel(&self) -> bool {
        !self.per_channel_scales.is_empty()
    }

    /// Slice per-channel qparams to a sub-range (for grouped conv decomposition).
    pub fn slice_channels(&self, start: usize, end: usize) -> Self {
        let mut sliced = self.clone();
        if self.is_per_channel() {
            sliced.per_channel_scales = self.per_channel_scales[start..end].to_vec();
            sliced.per_channel_zps =
                self.per_channel_zps[start..end.min(self.per_channel_zps.len())].to_vec();
        }
        sliced
    }

    /// Parse from Q/DQ node literal args: (input, scale, zp, qmin, qmax, dtype).
    fn from_q_node_args(args: &[Bound<'_, PyAny>]) -> Option<Self> {
        if args.len() < 6 {
            return None;
        }
        let dtype_str = format!("{}", args[5]);
        if !dtype_str.contains("int") {
            return None;
        }
        Some(Self {
            scale: args[1].extract().unwrap_or(1.0),
            zp: args[2].extract().unwrap_or(0),
            qmin: args[3].extract().unwrap_or(-128),
            qmax: args[4].extract().unwrap_or(127),
            dtype_str,
            per_channel_scales: Vec::new(),
            per_channel_zps: Vec::new(),
            axis: 0,
        })
    }

    /// Convert back to a Python QuantArgs object.
    pub(crate) fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cls = py
            .import_bound("executorch.backends.arm._passes.quant_args")?
            .getattr("QuantArgs")?;
        let torch = py.import_bound("torch")?;
        let dtype = torch.getattr(&*self.dtype_str.replace("torch.", ""))?;
        let kw = PyDict::new_bound(py);
        kw.set_item("scale", self.scale)?;
        kw.set_item("zp", self.zp)?;
        kw.set_item("qmin", self.qmin)?;
        kw.set_item("qmax", self.qmax)?;
        kw.set_item("dtype", dtype)?;
        kw.set_item("axis", self.axis)?;
        if self.is_per_channel() {
            kw.set_item("per_channel", true)?;
            let py_scales = pyo3::types::PyList::new_bound(py, &self.per_channel_scales);
            kw.set_item("scale", py_scales)?;
            let py_zps = pyo3::types::PyList::new_bound(py, &self.per_channel_zps);
            kw.set_item("zp", py_zps)?;
        } else {
            kw.set_item("per_channel", false)?;
        }
        cls.call((), Some(&kw))
    }

    fn same_scale_zp(&self, other: &Self) -> bool {
        (self.scale - other.scale).abs() < 1e-10 && self.zp == other.zp
    }
}

impl From<&Bound<'_, PyAny>> for QParam {
    fn from(qp: &Bound<'_, PyAny>) -> Self {
        Self::from_py(qp)
    }
}

// ---------------------------------------------------------------------------
// NodeOp — typed FX node operations
// ---------------------------------------------------------------------------

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
        match py_attr::<String>(node, &["op"], String::new()).as_str() {
            "placeholder" => Self::Placeholder,
            "call_function" => Self::CallFunction,
            "get_attr" => Self::GetAttr,
            "output" => Self::Output,
            _ => Self::Other,
        }
    }
}

/// Collect nodes from a graph, optionally filtered by op type.
fn graph_nodes<'py>(graph_or_module: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let graph = if graph_or_module.hasattr("graph")? {
        graph_or_module.getattr("graph")?
    } else {
        graph_or_module.clone()
    };
    graph.getattr("nodes")?.iter()?.collect()
}

fn nodes_by_op<'py>(
    graph_or_module: &Bound<'py, PyAny>,
    op: NodeOp,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    Ok(graph_nodes(graph_or_module)?
        .into_iter()
        .filter(|n| NodeOp::of(n) == op)
        .collect())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get all (name, submodule, cf_node) triples from the Python helper.
fn get_cf_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let arm_utils = py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
    let result = arm_utils.call_method1("get_cond_while_submodules_nested", (gm,))?;
    let mut out = Vec::new();
    for item in result.iter()? {
        let item = item?;
        let tup = item.downcast::<PyTuple>()?;
        let name: String = tup.get_item(0)?.extract()?;
        let submodule = tup.get_item(1)?;
        let cf_node = tup.get_item(2)?;
        out.push((name, submodule.clone(), cf_node.clone()));
    }
    Ok(out)
}

// submodule_nodes, submodule_placeholders, submodule_output_node
// replaced by graph_nodes(), nodes_by_op(), output_node() above.

/// Recompile a submodule.
fn recompile(submodule: &Bound<'_, PyAny>) -> PyResult<()> {
    submodule.call_method0("recompile")?;
    Ok(())
}

/// Getitem sort key: (is_getitem=0, getitem_idx, name) for getitems,
/// (1, 0, name) for everything else. Getitems sort first, by index.
fn user_sort_key(user: &Bound<'_, PyAny>) -> (u8, i64, String) {
    let target_str = FxGraphModule::node_target_str(user).unwrap_or_default();
    let args = FxGraphModule::node_args(user).unwrap_or_default();
    let name = FxGraphModule::node_name(user).unwrap_or_default();
    if target_str.contains("getitem") && args.len() > 1 {
        if let Ok(idx) = args[1].extract::<i64>() {
            return (0, idx, name);
        }
    }
    (1, 0, name)
}

/// `node.users` sorted by getitem index (getitems first, then others by name).
fn ordered_cf_users<'py>(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let users = node.getattr("users")?;
    let mut result: Vec<Bound<'py, PyAny>> = users
        .call_method0("keys")?
        .iter()?
        .filter_map(|k| k.ok())
        .collect();
    result.sort_by(|a, b| user_sort_key(a).cmp(&user_sort_key(b)));
    Ok(result)
}

/// Is node a quantize_per_tensor call_function?
/// Check if a call_function node's target string contains a marker.
fn target_contains(node: &Bound<'_, PyAny>, marker: &str) -> bool {
    NodeOp::of(node) == NodeOp::CallFunction
        && FxGraphModule::node_target_str(node)
            .map(|t| t.contains(marker))
            .unwrap_or(false)
}

fn is_quantize_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "quantize_per_tensor") && !target_contains(node, "dequantize")
}

// ---------------------------------------------------------------------------
// 1. _regularize_control_flow_submodules
//    Syncs placeholder fake tensors with parent args, stamps metadata.
// ---------------------------------------------------------------------------

/// Collect items from a Python iterable into a Vec.
fn py_iter_collect<'py>(obj: &Bound<'py, PyAny>) -> Vec<Bound<'py, PyAny>> {
    obj.iter()
        .ok()
        .map(|it| it.filter_map(|v| v.ok()).collect())
        .unwrap_or_default()
}

/// Get the first arg of a node if it's also a node (common guard pattern).
fn first_node_arg<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let args = FxGraphModule::node_args(node)?;
    Ok(args.first().filter(|a| FxGraphModule::is_node(a)).cloned())
}

/// Resolve data args for a control-flow node.
/// cond(pred, if_fn, else_fn, [data_args]) → returns data_args
/// while_loop(cond_fn, body_fn, carried_args, additional_args) → returns carried + additional
fn resolve_cf_parent_args<'py>(cf_node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let target_str = FxGraphModule::node_target_str(cf_node).unwrap_or_default();
    let all_args = FxGraphModule::node_args(cf_node)?;

    if target_str.contains("cond") {
        Ok(all_args
            .last()
            .map(|a| py_iter_collect(a))
            .unwrap_or_default())
    } else if target_str.contains("while_loop") && all_args.len() >= 2 {
        let mut v = py_iter_collect(&all_args[all_args.len() - 2]);
        v.extend(py_iter_collect(&all_args[all_args.len() - 1]));
        Ok(v)
    } else {
        Ok(Vec::new())
    }
}

/// Structural regularization: stamps placeholder markers, syncs fake tensors
/// and dim_order, but does NOT propagate qparams. Run before parent passes.
fn regularize_control_flow_submodules_structural<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut regularized: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let placeholders = nodes_by_op(submodule, NodeOp::Placeholder)?;
        let parent_args = resolve_cf_parent_args(cf_node)?;

        for ph in &placeholders {
            let meta = ph.getattr("meta")?;
            meta.set_item("is_input", true)?;
            meta.set_item("arm_control_flow_placeholder", true)?;
        }

        for (ph, parent_arg) in placeholders.iter().zip(parent_args.iter()) {
            let parent_m = NodeMeta::from_fx(parent_arg)?;
            let mut ph_m = NodeMeta::from_fx(ph)?;

            // Copy FakeTensor shape/dtype from parent to placeholder
            if let Some(ref parent_val) = parent_m.val {
                ph_m.set_val(parent_val.clone());
            }
            // Copy dim_order
            if let Some(ref tdo) = parent_m.tosa_dim_order {
                ph_m.set_tosa_dim_order(tdo.clone());
            }
            ph_m.write_back(py, ph)?;
        }

        regularized += 1;
    }

    Ok(regularized)
}

// ---------------------------------------------------------------------------
// 2. _annotate_control_flow_parent_rank4_dim_order
//    Stamps NHWC dim_order on parent CF node users (getitem nodes).
// ---------------------------------------------------------------------------

const NHWC_DIM_ORDER: [usize; 4] = [0, 2, 3, 1];

fn annotate_cf_parent_rank4_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut annotated: u32 = 0;

    for (_name, _submodule, cf_node) in &submodules {
        for user in &ordered_cf_users(cf_node)? {
            let mut m = NodeMeta::from_fx(user)?;
            if m.rank() != Some(4) {
                continue;
            }
            user.getattr("meta")?
                .set_item("arm_control_flow_parent_output_user", true)?;
            if m.tosa_dim_order.as_deref() == Some(&NHWC_DIM_ORDER) {
                continue;
            }
            m.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
            m.write_back(py, user)?;
            annotated += 1;
        }
    }
    Ok(annotated)
}

/// Pure Rust: walk producer chain to find effective qparam.
pub(crate) fn extract_effective_qparam_ir(graph: &IrGraph, start: NodeId) -> Option<QParam> {
    let mut current = start;
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(current) {
            break;
        }
        let node = graph.node(current);
        if let Some(qp) = node.meta.output_qparam() {
            return Some(qp.clone());
        }
        if let Some(qp) = node.meta.input_qparam() {
            return Some(qp.clone());
        }
        // RESCALE reverse-engineering
        if let Some(qp) = reverse_engineer_rescale_ir(graph, node) {
            return Some(qp);
        }
        current = match node.first_node_arg() {
            Some(id) => id,
            None => break,
        };
    }
    None
}

/// Pure Rust: derive qparam from RESCALE args.
pub(crate) fn reverse_engineer_rescale_ir(graph: &IrGraph, node: &IrNode) -> Option<QParam> {
    if !node.is_rescale() || node.args.len() < 5 {
        return None;
    }
    let source_id = match node.args[0] {
        IrArg::Node(id) => id,
        _ => return None,
    };
    let source_qp = extract_effective_qparam_ir(graph, source_id)?;
    // args[2] = scale list, args[4] = output_zp
    let mult_val = match &node.args[2] {
        IrArg::List(items) if items.len() == 1 => match &items[0] {
            IrArg::Float(v) => *v,
            IrArg::Int(v) => *v as f64,
            _ => return None,
        },
        IrArg::Float(v) => *v,
        _ => return None,
    };
    if mult_val == 0.0 {
        return None;
    }
    let output_zp = match &node.args[4] {
        IrArg::Int(v) => *v,
        _ => return None,
    };
    // args[1] = dtype string (stored in target or as a string arg)
    let dtype_str = match &node.args[1] {
        IrArg::Str(s) => s.clone(),
        _ => {
            // dtype might be stored as a non-string Python object — fall back to target
            let s = format!("{:?}", node.args[1]);
            if s.contains("int") {
                s
            } else {
                return None;
            }
        }
    };
    if !dtype_str.contains("int") {
        return None;
    }
    let (qmin, qmax) = if dtype_str.contains("int8") {
        (-128i64, 127i64)
    } else if dtype_str.contains("int16") {
        (-32768, 32767)
    } else {
        (i32::MIN as i64, i32::MAX as i64)
    };
    Some(QParam::per_tensor(
        source_qp.scale / mult_val,
        output_zp,
        qmin,
        qmax,
        dtype_str,
    ))
}

/// Pure Rust: annotate rank-4 call_function nodes with NHWC dim order.
fn annotate_rank4_dim_order_ir(graph: &mut IrGraph) -> u32 {
    let mut count = 0;
    for node in &mut graph.nodes {
        if node.op != NodeOp::CallFunction {
            continue;
        }
        if node.meta.tosa_dim_order.is_some() {
            continue;
        }
        if node.meta.rank() != Some(4) {
            continue;
        }
        node.meta.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
        count += 1;
    }
    count
}

fn annotate_cf_rank4_dim_order<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut annotated: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let mut ir = IrGraph::from_fx(&submodule)?;
        let count = annotate_rank4_dim_order_ir(&mut ir);
        if count > 0 {
            ir.write_back(py, &submodule)?;
            recompile(&submodule)?;
            annotated += count;
        }
    }
    Ok(annotated)
}

// ---------------------------------------------------------------------------
// 4. _force_control_flow_rank4_output_dim_order
//    Stamps dim_order on output-feeding nodes with rank-4 tensors.
// ---------------------------------------------------------------------------

/// Pure Rust: stamp output node's input_qparams from placeholder qparams (loop-carried).
pub(crate) fn stamp_while_body_output_qparams_ir(graph: &mut IrGraph) -> usize {
    let ph_qparams: Vec<(usize, QParam)> = graph
        .placeholders()
        .enumerate()
        .filter_map(|(idx, ph)| {
            ph.meta
                .output_qparam()
                .or(ph.meta.input_qparam())
                .cloned()
                .map(|qp| (idx, qp))
        })
        .collect();

    if ph_qparams.is_empty() {
        return 0;
    }
    let count = ph_qparams.len();

    let out_id = match graph.output_node() {
        Some(n) => n.id,
        None => return 0,
    };
    let mut map = HashMap::new();
    for (idx, qp) in ph_qparams {
        map.insert(idx, qp);
    }
    graph.node_mut(out_id).meta.set_input_qparams(map);
    count
}

/// Pure Rust: stamp NHWC dim_order on rank-4 output args.
fn force_rank4_output_dim_order_ir(graph: &mut IrGraph) -> u32 {
    let out = match graph.output_node() {
        Some(n) => n.id,
        None => return 0,
    };
    let out_arg_ids: Vec<NodeId> = graph
        .node(out)
        .args
        .iter()
        .filter_map(|a| match a {
            IrArg::Node(id) => Some(*id),
            IrArg::List(_items) => None, // handled below
            _ => None,
        })
        .collect();
    // Also handle output((val1, val2, ...),) — first arg is a list
    let out_arg_ids = if out_arg_ids.is_empty() {
        if let Some(IrArg::List(items)) = graph.node(out).args.first() {
            items
                .iter()
                .filter_map(|a| match a {
                    IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        }
    } else {
        out_arg_ids
    };

    let mut count = 0;
    for id in out_arg_ids {
        let node = &mut graph.nodes[id];
        if node.meta.rank() != Some(4) {
            continue;
        }
        if node.meta.tosa_dim_order.as_deref() == Some(&NHWC_DIM_ORDER) {
            continue;
        }
        node.meta.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
        count += 1;
    }
    count
}

fn force_cf_rank4_output_dim_order<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut forced: u32 = 0;
    for (_name, submodule, _cf_node) in &submodules {
        let mut ir = IrGraph::from_fx(&submodule)?;
        let count = force_rank4_output_dim_order_ir(&mut ir);
        if count > 0 {
            ir.write_back(py, &submodule)?;
            recompile(&submodule)?;
            forced += count;
        }
    }
    Ok(forced)
}

// ---------------------------------------------------------------------------
// 5. _restamp_control_flow_placeholder_dtypes
//    Updates placeholder val dtype to match qparam dtype.
// ---------------------------------------------------------------------------

/// Pure Rust: find the effective qparam dtype for a placeholder by walking users.
/// Mirrors _branch_local_control_flow_output_qparam from Python.
fn branch_local_qparam_dtype_ir(graph: &IrGraph, ph_id: NodeId) -> Option<String> {
    let ph = graph.node(ph_id);
    // Check saved table input qparam (stored by snapshot_cf_table_qparams)
    // Not available in IrGraph metadata — fall through to output_qparams

    // BFS: walk users looking for table qparam or output_qparams
    let mut pending: Vec<NodeId> = graph.users(ph_id).to_vec();
    let mut seen = std::collections::HashSet::new();
    seen.insert(ph_id);
    while let Some(uid) = pending.pop() {
        if !seen.insert(uid) {
            continue;
        }
        let user = graph.node(uid);
        if !user.alive {
            continue;
        }
        // If user has output_qparams with int dtype, use that
        if let Some(qp) = user.meta.output_qparam() {
            if qp.dtype_str.contains("int8") || qp.dtype_str.contains("int16") {
                return Some(qp.dtype_str.clone());
            }
        }
        pending.extend_from_slice(graph.users(uid));
    }
    // Fallback: placeholder's own output_qparams
    if let Some(qp) = ph.meta.output_qparam() {
        if qp.dtype_str.contains("int8") || qp.dtype_str.contains("int16") {
            return Some(qp.dtype_str.clone());
        }
    }
    // Fallback: input_qparams
    if let Some(qp) = ph.meta.input_qparam() {
        if qp.dtype_str.contains("int8") || qp.dtype_str.contains("int16") {
            return Some(qp.dtype_str.clone());
        }
    }
    None
}

/// Pure Rust: restamp placeholder FakeTensor dtype to match qparam dtype.
fn restamp_placeholder_dtypes_ir(graph: &mut IrGraph) -> u32 {
    let ph_ids: Vec<NodeId> = graph.placeholders().map(|n| n.id).collect();
    let mut count = 0u32;
    for ph_id in ph_ids {
        // Skip if any user is a quantize node
        if graph
            .users(ph_id)
            .iter()
            .any(|&uid| graph.nodes[uid].alive && graph.nodes[uid].is_quantize())
        {
            continue;
        }
        let target_dtype = match branch_local_qparam_dtype_ir(graph, ph_id) {
            Some(d) => d,
            None => continue,
        };
        let current_dtype = match graph.nodes[ph_id].meta.val_dtype() {
            Some(d) => d.to_string(),
            None => continue,
        };
        if current_dtype == target_dtype {
            continue;
        }
        // Restamp: change the FakeTensorMeta dtype
        if let Some(ref mut ftm) = graph.nodes[ph_id].meta.val {
            ftm.dtype = target_dtype;
            graph.nodes[ph_id].meta.dirty.val = true;
            count += 1;
        }
    }
    count
}

fn restamp_cf_placeholder_dtypes<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut restamped: u32 = 0;
    for (_name, submodule, _cf_node) in &submodules {
        let mut ir = IrGraph::from_fx(&submodule)?;
        let count = restamp_placeholder_dtypes_ir(&mut ir);
        if count > 0 {
            ir.write_back(py, &submodule)?;
            let graph = submodule.getattr("graph")?;
            graph.call_method0("lint")?;
            recompile(&submodule)?;
            restamped += count;
        }
    }
    Ok(restamped)
}

// ---------------------------------------------------------------------------
// 6. _snapshot_control_flow_table_qparams
//    Saves qparam metadata on TABLE-eligible nodes.
// ---------------------------------------------------------------------------

/// Table-eligible op target markers (math functions that get TABLE lowering).
const TABLE_OP_MARKERS: &[&str] = &[
    "aten.sin.",
    "aten.cos.",
    "aten.exp.",
    "aten.log.",
    "aten.rsqrt.",
    "aten.sigmoid.",
    "aten.tanh.",
    "aten.erf.",
    "aten.ceil.",
    "aten.floor.",
    "aten.hardswish.",
    "aten.hardsigmoid.",
    "aten.leaky_relu.",
    "aten.gelu.",
    "aten.silu.",
    "aten.reciprocal.",
];

/// Pure Rust: snapshot table input/output qparams.
fn snapshot_table_qparams_ir(graph: &mut IrGraph) -> u32 {
    let mut count = 0u32;
    let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();
    for id in ids {
        if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
            continue;
        }
        if !TABLE_OP_MARKERS
            .iter()
            .any(|m| graph.nodes[id].target.contains(m))
        {
            continue;
        }

        if let Some(input_id) = graph.nodes[id].first_node_arg() {
            if let Some(qp) = graph.node(input_id).parse_qdq_qparam() {
                graph.nodes[id].meta.table_input_qparam = Some(qp);
                graph.nodes[id].meta.dirty.table_qparams = true;
                count += 1;
            }
        }
        let user_ids = graph.users(id).to_vec();
        for uid in user_ids {
            if !graph.nodes[uid].alive {
                continue;
            }
            if let Some(qp) = graph.node(uid).parse_qdq_qparam() {
                graph.nodes[id].meta.table_output_qparam = Some(qp);
                graph.nodes[id].meta.dirty.table_qparams = true;
                count += 1;
                break;
            }
        }
    }
    count
}

fn snapshot_cf_table_qparams<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut snapshotted: u32 = 0;
    for (_name, submodule, _cf_node) in &submodules {
        let mut ir = IrGraph::from_fx(&submodule)?;
        let count = snapshot_table_qparams_ir(&mut ir);
        if count > 0 {
            ir.write_back(py, &submodule)?;
            snapshotted += count;
        }
    }
    Ok(snapshotted)
}

// ---------------------------------------------------------------------------
// 7. _refresh_tosa_dim_order_fake_tensors
//    Re-sync fake tensors with tosa_dim_order metadata.
// ---------------------------------------------------------------------------

pub(crate) fn refresh_tosa_dim_order_fake_tensors<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let cf_ctx = py.import_bound("retab._cf_ctx")?;
    let get_matching_fake_tensor = cf_ctx.getattr("_get_matching_fake_tensor")?;

    // Collect all modules: parent + submodules
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }

    for module in &modules {
        let mut updated = false;
        for node in graph_nodes(module)? {
            let meta = node.getattr("meta")?;
            if meta_val(&meta, "val").is_none() {
                continue;
            }
            if meta_val(&meta, "tosa_dim_order").is_none() {
                continue;
            }
            let new_val = get_matching_fake_tensor.call1((&node,))?;
            meta.set_item("val", &new_val)?;
            updated = true;
        }
        if updated {
            recompile(module)?;
        }
    }

    Ok(())
}

// ===========================================================================
// Pre-pass qparam snapshot
// ===========================================================================

fn snapshot_pre_pass_cf_qparams<'py>(py: Python<'py>, gm: &Bound<'py, PyAny>) -> PyResult<u32> {
    let ctx = CfCtx::new(py)?;
    let nodes = graph_nodes(gm)?;

    let mut saved = 0u32;
    for node in &nodes {
        if NodeOp::of(node) != NodeOp::CallFunction {
            continue;
        }
        let _cf = match ctx.cf_kind(node)? {
            Some(k) => k,
            None => continue,
        };

        let parent_args = resolve_cf_parent_args(node)?;
        let mut snapshot: HashMap<usize, QParam> = HashMap::new();
        for (i, arg) in parent_args.iter().enumerate() {
            let arg_meta = NodeMeta::from_fx(arg)?;
            if let Some(qp) = arg_meta.output_qparam() {
                snapshot.insert(i, qp.clone());
                saved += 1;
                continue;
            }
            if let Some(qp) = extract_effective_qparam(arg)? {
                snapshot.insert(i, qp);
                saved += 1;
                continue;
            }
            if is_dequantize_node(arg) || is_quantize_node(arg) {
                if let Ok(Some(qp)) = parse_qparam(arg) {
                    snapshot.insert(i, qp);
                    saved += 1;
                }
            }
        }
        if !snapshot.is_empty() {
            let cf_meta = node.getattr("meta")?;
            cf_meta.set_item(
                "_pre_pass_arg_qparams",
                crate::fx_ir::qparam_dict_to_py(py, &snapshot)?,
            )?;
        }
    }
    if saved > 0 {
        log_line(format_args!(
            "[snapshot-pre-pass-qparams] saved {} arg qparam(s)",
            saved
        ));
    }
    Ok(saved)
}

// ===========================================================================
// Structural-only pre-pass (no qparam propagation)
// ===========================================================================

/// Pre-pass that stamps structural metadata on control-flow submodules WITHOUT
/// propagating qparams. Run before parent passes.
pub struct FxControlFlowStructuralPass;

impl FxPass for FxControlFlowStructuralPass {
    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        let mut any_modified = false;

        // 0. Snapshot qparams from pre-pass parent args (before fold_qdq destroys them)
        snapshot_pre_pass_cf_qparams(py, gm_ref)?;

        // 1. Structural regularize (no qparams)
        let count = regularize_control_flow_submodules_structural(py, gm_ref)?;
        if count > 0 {
            any_modified = true;
        }

        // 2-4. Dim order annotations
        annotate_cf_parent_rank4_dim_order(py, gm_ref)?;
        annotate_cf_rank4_dim_order(py, gm_ref)?;
        force_cf_rank4_output_dim_order(py, gm_ref)?;

        // 5. Restamp placeholder dtypes
        restamp_cf_placeholder_dtypes(py, gm_ref)?;

        // 6. Snapshot table qparams
        snapshot_cf_table_qparams(py, gm_ref)?;

        // 7. Refresh fake tensors with dim_order
        refresh_tosa_dim_order_fake_tensors(py, gm_ref)?;

        if any_modified {
            gm.recompile()?;
        }

        Ok(any_modified)
    }
}

fn is_tosa_rescale_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "RESCALE")
}

/// Extract the effective quantization parameters for a node by walking
/// producer edges until a node with qparam metadata or a RESCALE is found.
fn extract_effective_qparam(node: &Bound<'_, PyAny>) -> PyResult<Option<QParam>> {
    let mut current = node.clone();
    let mut seen = std::collections::HashSet::new();

    loop {
        if current.getattr("op").is_err() {
            break;
        }
        let node_id: usize = current
            .call_method0("__hash__")
            .and_then(|h| h.extract())
            .unwrap_or(0);
        if !seen.insert(node_id) {
            break;
        }

        let m = NodeMeta::from_fx(&current)?;
        if let Some(qp) = m.output_qparam() {
            return Ok(Some(qp.clone()));
        }
        if let Some(qp) = m.input_qparam() {
            return Ok(Some(qp.clone()));
        }

        let args = FxGraphModule::node_args(&current)?;
        if let Some(qp) = try_reverse_engineer_rescale_qparam(&current, &args)? {
            return Ok(Some(qp));
        }
        current = match first_node_arg(&current)? {
            Some(a) => a,
            None => break,
        };
    }
    Ok(None)
}

/// Derive a QParam from a TOSA RESCALE node: output_scale = source_scale / multiplier.
fn try_reverse_engineer_rescale_qparam(
    node: &Bound<'_, PyAny>,
    args: &[Bound<'_, PyAny>],
) -> PyResult<Option<QParam>> {
    if !is_tosa_rescale_node(node) || args.len() < 5 {
        return Ok(None);
    }
    let source = &args[0];
    if source.getattr("op").is_err() {
        return Ok(None);
    }
    let source_qp = match extract_effective_qparam(source)? {
        Some(qp) => qp,
        None => return Ok(None),
    };
    let mult_val: f64 = args[2]
        .get_item(0_usize)
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(0.0);
    if mult_val == 0.0 || args[2].len().unwrap_or(0) != 1 {
        return Ok(None);
    }
    let output_zp: i64 = match args[4].extract() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let dtype_str = format!("{}", args[1]);
    if !dtype_str.contains("int") {
        return Ok(None);
    }
    let (qmin, qmax) = if dtype_str.contains("int8") {
        (-128i64, 127i64)
    } else if dtype_str.contains("int16") {
        (-32768, 32767)
    } else {
        (i32::MIN as i64, i32::MAX as i64)
    };
    Ok(Some(QParam {
        scale: source_qp.scale / mult_val,
        zp: output_zp,
        qmin,
        qmax,
        dtype_str,
        per_channel_scales: Vec::new(),
        per_channel_zps: Vec::new(),
        axis: 0,
    }))
}

fn is_dequantize_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "dequantize_per_tensor")
}

// ---------------------------------------------------------------------------
// Q/DQ node helpers
// ---------------------------------------------------------------------------

/// Parse Q/DQ node literal args into a Rust QParam.
fn parse_qparam(node: &Bound<'_, PyAny>) -> PyResult<Option<QParam>> {
    let args = FxGraphModule::node_args(node)?;
    Ok(QParam::from_q_node_args(&args))
}

/// Pure Rust: strip boundary Q from placeholders, annotate DQ qparams on output.
pub(crate) fn strip_while_boundary_qdq_ir(graph: &mut IrGraph) -> bool {
    let mut modified = false;

    // Strip Q nodes from placeholders
    let ph_ids: Vec<NodeId> = graph.placeholders().map(|n| n.id).collect();
    for ph_id in ph_ids {
        let user_ids = graph.users(ph_id).to_vec();
        for q_id in user_ids {
            if !graph.nodes[q_id].alive || !graph.nodes[q_id].is_quantize() {
                continue;
            }
            // Stamp Q's qparams on placeholder
            if let Some(qp) = graph.nodes[q_id].parse_qdq_qparam() {
                graph.nodes[ph_id].meta.set_single_output_qparam(qp);
            }
            graph.replace_and_erase(q_id, ph_id);
            modified = true;
        }
    }

    // Annotate DQ qparams on output node (without stripping DQ)
    if let Some(out_id) = graph.output_node().map(|n| n.id) {
        // Get output arg node IDs
        let out_arg_ids: Vec<NodeId> = graph
            .node(out_id)
            .args
            .iter()
            .flat_map(|a| match a {
                IrArg::Node(id) => vec![*id],
                IrArg::List(items) => items
                    .iter()
                    .filter_map(|i| match i {
                        IrArg::Node(id) => Some(*id),
                        _ => None,
                    })
                    .collect(),
                _ => vec![],
            })
            .collect();

        let mut iqp_map: HashMap<usize, QParam> = graph
            .node(out_id)
            .meta
            .input_qparams
            .clone()
            .unwrap_or_default();
        for (i, &arg_id) in out_arg_ids.iter().enumerate() {
            if !graph.nodes[arg_id].alive || !graph.nodes[arg_id].is_dequantize() {
                continue;
            }
            if let Some(qp) = graph.nodes[arg_id].parse_qdq_qparam() {
                iqp_map.insert(i, qp);
            }
        }
        if !iqp_map.is_empty() {
            graph.node_mut(out_id).meta.set_input_qparams(iqp_map);
            modified = true;
        }
    }

    modified
}
