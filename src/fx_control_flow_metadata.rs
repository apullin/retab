//! Control-flow metadata stamping, boundary RESCALE insertion, and Q/DQ
//! rewriting for cond and while_loop submodules.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_graph::{meta_get, meta_val, py_attr, FxGraphModule};
use crate::fx_ir::NodeMeta;
use crate::fx_pass::FxPass;
use crate::runner_logging::log_line;

// ---------------------------------------------------------------------------
// CfCtx — cached Python imports and shared context
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CfKind { Cond, While }

/// Cached Python module imports for control-flow operations.
struct CfCtx<'py> {
    torch: Bound<'py, PyAny>,
    cond_op: Bound<'py, PyAny>,
    while_op: Bound<'py, PyAny>,
}

impl<'py> CfCtx<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let torch = py.import_bound("torch")?.into_any();
        let ho = torch.getattr("ops")?.getattr("higher_order")?;
        let cond_op = ho.getattr("cond")?;
        let while_op = ho.getattr("while_loop")?;
        Ok(Self { torch, cond_op, while_op })
    }

    fn int8(&self) -> PyResult<Bound<'py, PyAny>> { self.torch.getattr("int8") }
    fn int16(&self) -> PyResult<Bound<'py, PyAny>> { self.torch.getattr("int16") }

    fn cf_kind(&self, node: &Bound<'py, PyAny>) -> PyResult<Option<CfKind>> {
        let target = node.getattr("target")?;
        if target.eq(&self.cond_op).unwrap_or(false) { Ok(Some(CfKind::Cond)) }
        else if target.eq(&self.while_op).unwrap_or(false) { Ok(Some(CfKind::While)) }
        else { Ok(None) }
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
}

impl QParam {
    /// Extract from a Python QuantArgs object (or anything with scale/zp/dtype attrs).
    pub(crate) fn from_py(qp: &Bound<'_, PyAny>) -> Self {
        Self {
            scale: qp.getattr("scale").and_then(|v| v.extract()).unwrap_or(1.0),
            zp: qp.getattr("zp").and_then(|v| v.extract()).unwrap_or(0),
            qmin: qp.getattr("qmin").and_then(|v| v.extract()).unwrap_or(-128),
            qmax: qp.getattr("qmax").and_then(|v| v.extract()).unwrap_or(127),
            dtype_str: qp.getattr("dtype").ok().map(|d| format!("{d}")).unwrap_or_default(),
        }
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
        kw.set_item("axis", 0i64)?;
        kw.set_item("per_channel", false)?;
        cls.call((), Some(&kw))
    }

    fn same_as(&self, other: &Self) -> bool {
        (self.scale - other.scale).abs() < 1e-10
            && self.zp == other.zp
            && self.dtype_str == other.dtype_str
            && self.qmin == other.qmin
            && self.qmax == other.qmax
    }

    fn same_scale_zp(&self, other: &Self) -> bool {
        (self.scale - other.scale).abs() < 1e-10 && self.zp == other.zp
    }

    /// Store as a single-entry qparam dict {0: self} on node metadata.
    /// Store as a single-entry qparam dict {0: self} on node metadata.
    fn stamp_on_meta<'py>(&self, py: Python<'py>, meta: &Bound<'py, PyAny>, key: &str) -> PyResult<()> {
        let dict = PyDict::new_bound(py);
        dict.set_item(0i64, self.to_py(py)?)?;
        meta.set_item(key, dict)?;
        Ok(())
    }

    /// Stamp both input_qparams and output_qparams on a node's metadata.
    fn stamp_both<'py>(&self, py: Python<'py>, meta: &Bound<'py, PyAny>) -> PyResult<()> {
        self.stamp_on_meta(py, meta, "output_qparams")?;
        self.stamp_on_meta(py, meta, "input_qparams")?;
        Ok(())
    }

    /// Insert a RESCALE node converting from `self` domain to `target` domain.
    /// Returns the new rescale node. `after=true` inserts after `anchor`,
    /// `after=false` inserts before `anchor`.
    fn insert_rescale_to<'py>(
        &self,
        target: &QParam,
        py: Python<'py>,
        submodule: &Bound<'py, PyAny>,
        anchor: &Bound<'py, PyAny>,
        source_val: &Bound<'py, PyAny>,
        after: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let helpers = py.import_bound("retab._helpers")?;
        let insert_fn = helpers.getattr("insert_rescale_node")?;
        let torch = py.import_bound("torch")?;
        let dtype = torch.getattr(&*target.dtype_str.replace("torch.", ""))?;
        let scales = PyList::new_bound(py, &[self.scale / target.scale]);
        let kw = PyDict::new_bound(py);
        kw.set_item("after", after)?;
        insert_fn.call(
            (submodule, anchor, source_val, &dtype, &scales, self.zp, target.zp),
            Some(&kw),
        )
    }
}

impl From<&Bound<'_, PyAny>> for QParam {
    fn from(qp: &Bound<'_, PyAny>) -> Self {
        Self::from_py(qp)
    }
}

/// Extract a QParam from a Python qparams dict (tries key 0, then single-value, then raw).
fn extract_qparam(qparams: &Bound<'_, PyAny>) -> Option<QParam> {
    if qparams.is_none() || !qparams.is_truthy().unwrap_or(false) {
        return None;
    }
    if let Ok(val) = qparams.get_item(0i64) {
        if !val.is_none() {
            return Some(QParam::from_py(&val));
        }
    }
    if let Ok(len) = qparams.len() {
        if len == 1 {
            if let Ok(values) = qparams.call_method0("values") {
                if let Ok(mut iter) = values.iter() {
                    if let Some(Ok(val)) = iter.next() {
                        return Some(QParam::from_py(&val));
                    }
                }
            }
        }
    }
    Some(QParam::from_py(qparams))
}

// ---------------------------------------------------------------------------
// NodeOp — typed FX node operations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeOp {
    Placeholder,
    CallFunction,
    GetAttr,
    Output,
    Other,
}

impl NodeOp {
    fn of(node: &Bound<'_, PyAny>) -> Self {
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

fn nodes_by_op<'py>(graph_or_module: &Bound<'py, PyAny>, op: NodeOp) -> PyResult<Vec<Bound<'py, PyAny>>> {
    Ok(graph_nodes(graph_or_module)?.into_iter().filter(|n| NodeOp::of(n) == op).collect())
}

fn output_node<'py>(graph_or_module: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(graph_nodes(graph_or_module)?.into_iter().find(|n| NodeOp::of(n) == NodeOp::Output))
}

/// Unpack the output node's args tuple: output((val1, val2, ...),) → [val1, val2, ...]
fn unpack_output_vals<'py>(output: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let oargs = FxGraphModule::node_args(output)?;
    Ok(if oargs.len() == 1 {
        if let Ok(it) = oargs[0].iter() { it.filter_map(|v| v.ok()).collect() }
        else { vec![oargs[0].clone()] }
    } else { oargs })
}

/// Rebuild the output node's args as a tuple after modifying values.
fn repack_output_vals(py: Python<'_>, output: &Bound<'_, PyAny>, vals: &[Bound<'_, PyAny>]) -> PyResult<()> {
    let new_tuple = PyTuple::new_bound(py, vals);
    output.call_method("update_arg", (0i64, &new_tuple), None)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get all (name, submodule, cf_node) triples from the Python helper.
fn get_cf_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let arm_utils =
        py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
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
    let mut result: Vec<Bound<'py, PyAny>> = users.call_method0("keys")?
        .iter()?.filter_map(|k| k.ok()).collect();
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
    obj.iter().ok()
        .map(|it| it.filter_map(|v| v.ok()).collect())
        .unwrap_or_default()
}

/// Get the first arg of a node if it's also a node (common guard pattern).
fn first_node_arg<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let args = FxGraphModule::node_args(node)?;
    Ok(args.first()
        .filter(|a| FxGraphModule::is_node(a))
        .cloned())
}

/// Collect a node's users into a Vec.
fn node_users<'py>(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    Ok(py_iter_collect(&node.getattr("users")?.call_method0("keys")?))
}

/// Resolve data args for a control-flow node.
/// cond(pred, if_fn, else_fn, [data_args]) → returns data_args
/// while_loop(cond_fn, body_fn, carried_args, additional_args) → returns carried + additional
fn resolve_cf_parent_args<'py>(
    cf_node: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let target_str = FxGraphModule::node_target_str(cf_node).unwrap_or_default();
    let all_args = FxGraphModule::node_args(cf_node)?;

    if target_str.contains("cond") {
        Ok(all_args.last().map(|a| py_iter_collect(a)).unwrap_or_default())
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
    let cf_ctx = py.import_bound("retab._cf_ctx")?;
    let get_matching_fake_tensor = cf_ctx.getattr("_get_matching_fake_tensor")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut regularized: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let placeholders = nodes_by_op(submodule, NodeOp::Placeholder)?;
        for ph in &placeholders {
            let meta = ph.getattr("meta")?;
            meta.set_item("is_input", true)?;
            meta.set_item("arm_control_flow_placeholder", true)?;
        }

        let parent_args = resolve_cf_parent_args(cf_node)?;

        for (ph, parent_arg) in placeholders.iter().zip(parent_args.iter()) {
            let fake = get_matching_fake_tensor.call1((parent_arg,))?;
            let meta = ph.getattr("meta")?;
            meta.set_item("val", &fake)?;

            let parent_meta = parent_arg.getattr("meta")?;
            if meta_val(&parent_meta, "tosa_dim_order").is_some() {
                let dim_order = parent_meta.get_item("tosa_dim_order")?;
                let py_tuple = PyTuple::new_bound(py, &py_iter_collect(&dim_order));
                meta.set_item("tosa_dim_order", &py_tuple)?;
            }
            // No qparam propagation here — that happens post-parent-passes
        }

        regularized += 1;
    }

    Ok(regularized)
}

/// Full regularization including qparam propagation. Used by the legacy path.
fn regularize_control_flow_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let count = regularize_control_flow_submodules_structural(py, gm)?;
    propagate_parent_qparams_to_placeholders(py, gm)?;
    Ok(count)
}

/// Copy input_qparams and output_qparams from parent args to submodule
/// placeholders. Run AFTER parent passes so placeholders see post-fold qparams.
pub(crate) fn propagate_parent_qparams_to_placeholders<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut propagated: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let placeholders = nodes_by_op(submodule, NodeOp::Placeholder)?;
        let parent_args = resolve_cf_parent_args(cf_node)?;
        let cf_meta = NodeMeta::from_fx(cf_node)?;

        for (i, (ph, parent_arg)) in placeholders.iter().zip(parent_args.iter()).enumerate() {
            let parent = NodeMeta::from_fx(parent_arg)?;
            let mut ph_meta = NodeMeta::from_fx(ph)?;

            // First try direct metadata copy from parent
            let mut found = false;
            if let Some(ref oqp) = parent.output_qparams {
                ph_meta.set_output_qparams(oqp.clone());
                propagated += 1;
                found = true;
            }
            if let Some(ref iqp) = parent.input_qparams {
                ph_meta.set_input_qparams(iqp.clone());
                propagated += 1;
                found = true;
            }

            // If no metadata qparams, try the pre-pass snapshot
            if !found {
                if let Some(ref saved) = cf_meta.pre_pass_arg_qparams {
                    if let Some(qp) = saved.get(&i) {
                        ph_meta.set_both_qparams(qp.clone());
                        propagated += 1;
                        found = true;
                    }
                }
            }

            // Last fallback: walk the (post-pass) producer chain
            if !found {
                if let Some(qp) = extract_effective_qparam(parent_arg)? {
                    ph_meta.set_both_qparams(qp);
                    propagated += 1;
                }
            }

            ph_meta.write_back(py, ph)?;
        }
    }

    // For while_loop bodies: stamp output node's input_qparams to match
    // placeholder qparams (loop-carried constraint: body outputs feed back
    // as next iteration's inputs, so output scale/zp must equal input scale/zp).
    for (_name, submodule, cf_node) in &submodules {
        let target_str = cf_node
            .getattr("target")
            .and_then(|t| t.str().map(|s| s.to_string()))
            .unwrap_or_default();
        if !target_str.contains("while_loop") {
            continue;
        }
        // Check if this submodule is the body (args[1]) not the cond (args[0])
        let cf_args = FxGraphModule::node_args(cf_node)?;
        if cf_args.len() < 2 {
            continue;
        }
        let body_target = cf_args[1]
            .getattr("target")
            .and_then(|t| t.extract::<String>())
            .unwrap_or_default();
        let sub_name = _name.to_string();
        if !sub_name.is_empty() && !body_target.is_empty() && !sub_name.contains(&body_target) {
            continue; // This is the cond submodule, skip
        }

        let placeholders = nodes_by_op(submodule, NodeOp::Placeholder)?;
        if let Some(output) = output_node(submodule)? {
            let mut out_meta = NodeMeta::from_fx(&output)?;
            let mut loop_carried = HashMap::new();
            for (idx, ph) in placeholders.iter().enumerate() {
                let ph_m = NodeMeta::from_fx(ph)?;
                if let Some(qp) = ph_m.output_qparam().or(ph_m.input_qparam()) {
                    loop_carried.insert(idx, qp.clone());
                    propagated += 1;
                }
            }
            if !loop_carried.is_empty() {
                log_line(format_args!(
                    "[propagate-boundary-qparams] stamped while_loop body output with {} loop-carried qparam(s)",
                    loop_carried.len()
                ));
                out_meta.set_input_qparams(loop_carried);
                out_meta.write_back(py, &output)?;
            }
        }
    }

    if propagated > 0 {
        log_line(format_args!(
            "[propagate-boundary-qparams] propagated {} qparam(s) total",
            propagated
        ));
    }

    Ok(propagated)
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
            if m.val_shape.as_ref().map(|s| s.len()) != Some(4) { continue; }
            user.getattr("meta")?.set_item("arm_control_flow_parent_output_user", true)?;
            if m.tosa_dim_order.as_deref() == Some(&NHWC_DIM_ORDER) { continue; }
            m.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
            m.write_back(py, user)?;
            annotated += 1;
        }
    }
    Ok(annotated)
}

// ---------------------------------------------------------------------------
// 3. _annotate_control_flow_rank4_dim_order
//    Stamps dim_order on rank-4 call_function nodes in submodules.
// ---------------------------------------------------------------------------

fn annotate_cf_rank4_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut annotated: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let mut modified = false;
        for node in graph_nodes(&submodule)? {
            if NodeOp::of(&node) != NodeOp::CallFunction { continue; }
            let mut m = NodeMeta::from_fx(&node)?;
            if m.tosa_dim_order.is_some() { continue; }
            if m.val_shape.as_ref().map(|s| s.len()) != Some(4) { continue; }
            m.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
            m.write_back(py, &node)?;
            annotated += 1;
            modified = true;
        }
        if modified { recompile(&submodule)?; }
    }
    Ok(annotated)
}

// ---------------------------------------------------------------------------
// 4. _force_control_flow_rank4_output_dim_order
//    Stamps dim_order on output-feeding nodes with rank-4 tensors.
// ---------------------------------------------------------------------------

fn force_cf_rank4_output_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut forced: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let out = match output_node(&submodule)? { Some(n) => n, None => continue };
        let output_args = unpack_output_vals(&out)?;
        let mut modified = false;
        for arg in &output_args {
            if !FxGraphModule::is_node(arg) { continue; }
            let mut m = NodeMeta::from_fx(arg)?;
            if m.val_shape.as_ref().map(|s| s.len()) != Some(4) { continue; }
            if m.tosa_dim_order.as_deref() == Some(&NHWC_DIM_ORDER) { continue; }
            m.set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
            m.write_back(py, arg)?;
            forced += 1;
            modified = true;
        }
        if modified { recompile(&submodule)?; }
    }
    Ok(forced)
}

// ---------------------------------------------------------------------------
// 5. _restamp_control_flow_placeholder_dtypes
//    Updates placeholder val dtype to match qparam dtype.
// ---------------------------------------------------------------------------

fn restamp_cf_placeholder_dtypes<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_mod = py.import_bound("retab._cf_ctx")?;
    let branch_local_fn =
        cf_mod.getattr("_branch_local_control_flow_output_qparam")?;
    let extract_fn = cf_mod.getattr("_extract_single_qparam_entry")?;

    let torch = py.import_bound("torch")?;
    let int8 = torch.getattr("int8")?;
    let int16 = torch.getattr("int16")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut restamped: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let mut updated = false;
        let placeholders = nodes_by_op(&submodule, NodeOp::Placeholder)?;

        for ph in &placeholders {
            // Skip if any user is a quantize node
            let users = FxGraphModule::node_users(ph)?;
            let has_quantize_user = users.iter().any(|u| is_quantize_node(u));
            if has_quantize_user {
                continue;
            }

            // Get qparam
            let qparam = branch_local_fn.call1((ph,))?;

            // Check if qparam.dtype is int8 or int16; if not, try input_qparams
            let qparam = if qparam.is_none() {
                py.None().into_bound(py)
            } else {
                let dtype = qparam.getattr("dtype");
                let is_int8_or_int16 = match &dtype {
                    Ok(dt) => dt.eq(&int8).unwrap_or(false) || dt.eq(&int16).unwrap_or(false),
                    Err(_) => false,
                };
                if !is_int8_or_int16 {
                    let meta = ph.getattr("meta")?;
                    let input_qparams = meta.call_method1("get", ("input_qparams",))?;
                    let extracted = extract_fn.call1((&input_qparams,))?;
                    if extracted.is_none() {
                        qparam
                    } else {
                        extracted
                    }
                } else {
                    qparam
                }
            };

            if qparam.is_none() {
                continue;
            }

            let meta = ph.getattr("meta")?;
            let fake_val = match meta_val(&meta, "val") {
                Some(v) => v,
                None => continue,
            };
            if !fake_val.hasattr("to")? {
                continue;
            }

            // Check if dtype already matches
            let fake_dtype = fake_val.getattr("dtype").ok();
            let qparam_dtype = qparam.getattr("dtype").ok();
            match (&fake_dtype, &qparam_dtype) {
                (Some(fd), Some(qd)) if fd.eq(qd).unwrap_or(false) => continue,
                _ => {}
            }

            // Restamp
            if let Some(qd) = qparam_dtype {
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("dtype", &qd)?;
                let new_val = fake_val.call_method("to", (), Some(&kwargs))?;
                meta.set_item("val", &new_val)?;
                updated = true;
                restamped += 1;
            }
        }

        if updated {
            let graph = submodule.getattr("graph")?;
            graph.call_method0("lint")?;
            recompile(&submodule)?;
        }
    }

    Ok(restamped)
}

// ---------------------------------------------------------------------------
// 6. _snapshot_control_flow_table_qparams
//    Saves qparam metadata on TABLE-eligible nodes.
// ---------------------------------------------------------------------------

fn snapshot_cf_table_qparams<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let table_ops_mod =
        py.import_bound("executorch.backends.arm._passes.insert_table_ops")?;
    let table_ops_cls = table_ops_mod.getattr("TableOps")?;
    let included_ops = table_ops_cls.call_method0("included_ops")?;
    let table_targets = py_iter_collect(&included_ops);

    let quant_args_mod =
        py.import_bound("executorch.backends.arm._passes.quant_args")?;
    let quant_args_cls = quant_args_mod.getattr("QuantArgs")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut snapshotted: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        for node in graph_nodes(&submodule)? {
            if NodeOp::of(&node) != NodeOp::CallFunction { continue; }
            let target = node.getattr("target")?;
            let is_table = table_targets.iter().any(|t| target.eq(t).unwrap_or(false));
            if !is_table {
                continue;
            }

            let meta = node.getattr("meta")?;
            let args = FxGraphModule::node_args(&node)?;

            // Input qparam: from first arg
            if !args.is_empty() && FxGraphModule::is_node(&args[0]) {
                let input_node = &args[0];
                let input_target = input_node.getattr("target")?;
                let input_args = FxGraphModule::node_args(input_node)?;
                let input_args_tuple = PyTuple::new_bound(py, &input_args);
                match quant_args_cls.call_method1("from_operator", (&input_target, &input_args_tuple)) {
                    Ok(qparam) => {
                        meta.set_item("arm_saved_table_input_qparam", &qparam)?;
                        snapshotted += 1;
                    }
                    Err(_) => {}
                }
            }

            // Output qparam: from first user
            let users = FxGraphModule::node_users(&node)?;
            for user in &users {
                let user_target = user.getattr("target")?;
                let user_args = FxGraphModule::node_args(user)?;
                let user_args_tuple = PyTuple::new_bound(py, &user_args);
                match quant_args_cls.call_method1("from_operator", (&user_target, &user_args_tuple)) {
                    Ok(qparam) => {
                        meta.set_item("arm_saved_table_output_qparam", &qparam)?;
                        snapshotted += 1;
                        break;
                    }
                    Err(_) => continue,
                }
            }
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

// ---------------------------------------------------------------------------
// Combined FxPass
// ---------------------------------------------------------------------------

/// Strip boundary Q nodes from while_loop placeholders before fold_qdq.
/// Loop-carried values are already quantized in the placeholder's domain.
/// The Q node's literal args are from calibration, not runtime truth.
/// Removing them lets fold_qdq see the placeholder directly and use
/// its propagated qparams for downstream annotation.
pub struct FxWhileStripBoundaryQPass;

impl FxPass for FxWhileStripBoundaryQPass {
    fn name(&self) -> &str {
        "FxWhileStripBoundaryQ"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let graph = gm.gm_ref().getattr("graph")?;
        let mut modified = false;

        // Strip boundary Q nodes: placeholder → Q → ops  ⟹  placeholder → ops
        // Store Q's qparams on the placeholder's output_qparams metadata
        // so fold_qdq sees the placeholder with correct annotations.
        for node in gm.nodes()? {
            let op: String = py_attr(&node, &["op"], String::new());
            if op != "placeholder" { continue; }
            let users = node_users(&node)?;
            for user in &users {
                if !is_quantize_node(user) { continue; }
                // Always stamp Q literal qparams — these represent the
                // body-internal quantization domain. The propagated qparams
                // (external domain) live on the while_loop parent node and
                // are used for boundary RESCALE scale computation.
                if let Some(qp) = parse_per_tensor_qparam(py, user)? {
                    let meta = node.getattr("meta")?;
                    let dict = pyo3::types::PyDict::new_bound(py);
                    dict.set_item(0i64, &qp)?;
                    meta.set_item("output_qparams", &dict)?;
                }
                user.call_method1("replace_all_uses_with", (&node,))?;
                graph.call_method1("erase_node", (user,))?;
                modified = true;
            }
        }

        // Annotate boundary DQ qparams on the output node WITHOUT stripping.
        // The DQ nodes must stay (they convert INT8→float for the parent graph).
        // We just stamp their literal qparams on output_node.meta["input_qparams"]
        // so insert_cf_boundary_rescales can compute the correct output RESCALE.
        let output_node = output_node(gm.gm_ref())?;
        if let Some(output) = output_node {
            let out_meta = output.getattr("meta")?;
            if out_meta.get_item("input_qparams").is_err() {
                out_meta.set_item("input_qparams", pyo3::types::PyDict::new_bound(py))?;
            }

            let oargs = FxGraphModule::node_args(&output)?;
            let ovals: Vec<Bound<'_, PyAny>> = if oargs.len() == 1 {
                if let Ok(it) = oargs[0].iter() { it.filter_map(|v| v.ok()).collect() }
                else { vec![oargs[0].clone()] }
            } else { oargs };

            for (i, val) in ovals.iter().enumerate() {
                if !FxGraphModule::is_node(val) || !is_dequantize_node(val) { continue; }
                if let Some(qp) = parse_per_tensor_qparam(py, val)? {
                    let iqp_dict = out_meta.get_item("input_qparams")?;
                    iqp_dict.set_item(i, &qp)?;
                }
            }
        }

        if modified {
            cleanup_graph(&graph)?;
            gm.recompile()?;
        }
        Ok(modified)
    }
}

pub struct FxControlFlowMetadataPass;

impl FxPass for FxControlFlowMetadataPass {
    fn name(&self) -> &str {
        "FxControlFlowMetadata"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        // Check if there are any cond/while_loop nodes at all.
        // If not, skip everything.
        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        let mut any_modified = false;

        // 1. Regularize (sync placeholders with parent args)
        let count = regularize_control_flow_submodules(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] regularized {} control-flow submodule(s)",
                count
            ));
            any_modified = true;
        }

        // 2. Annotate parent rank-4 dim_order
        let count = annotate_cf_parent_rank4_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] annotated NHWC dim-order on {} parent CF output user(s)",
                count
            ));
            any_modified = true;
        }

        // 3. Annotate rank-4 dim_order in submodules
        let count = annotate_cf_rank4_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] annotated NHWC dim-order on {} interior rank-4 node(s)",
                count
            ));
            any_modified = true;
        }

        // 4. Force output dim_order
        let count = force_cf_rank4_output_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] forced NHWC dim-order on {} rank-4 output(s)",
                count
            ));
            any_modified = true;
        }

        // 5. Restamp placeholder dtypes
        let count = restamp_cf_placeholder_dtypes(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] restamped {} placeholder dtype(s)",
                count
            ));
            any_modified = true;
        }

        // 6. Snapshot table qparams
        let count = snapshot_cf_table_qparams(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] snapshotted {} table qparam attachment(s)",
                count
            ));
            any_modified = true;
        }

        // 7. Refresh fake tensors with dim_order
        refresh_tosa_dim_order_fake_tensors(py, gm_ref)?;

        // Recompile the parent graph module if we made changes
        if any_modified {
            gm.recompile()?;
        }

        Ok(any_modified)
    }
}

// ===========================================================================
// Pre-pass qparam snapshot
// ===========================================================================

/// Before parent passes run, walk DQ nodes feeding into cond/while data args
/// and snapshot their qparams onto the cond node's metadata.  Parent passes
/// (fold_qdq, memory format) may destroy these nodes, so we save them now.
fn snapshot_pre_pass_cf_qparams<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let ctx = CfCtx::new(py)?;
    let nodes = graph_nodes(gm)?;

    let mut saved = 0u32;
    for node in &nodes {
        if NodeOp::of(node) != NodeOp::CallFunction { continue; }
        let cf = match ctx.cf_kind(node)? { Some(k) => k, None => continue };

        let parent_args = resolve_cf_parent_args(node)?;
        let mut snapshot: HashMap<usize, QParam> = HashMap::new();
        for (i, arg) in parent_args.iter().enumerate() {
            let arg_meta = NodeMeta::from_fx(arg)?;
            // Try output_qparams from metadata
            if let Some(qp) = arg_meta.output_qparam() {
                snapshot.insert(i, qp.clone());
                saved += 1;
                continue;
            }
            // Walk producer chain
            if let Some(qp) = extract_effective_qparam(arg)? {
                snapshot.insert(i, qp);
                saved += 1;
                continue;
            }
            // Last resort: Q/DQ literal args
            if is_dequantize_node(arg) || is_quantize_node(arg) {
                if let Ok(Some(qp)) = parse_qparam(arg) {
                    snapshot.insert(i, qp);
                    saved += 1;
                }
            }
        }
        if !snapshot.is_empty() {
            let cf_meta = node.getattr("meta")?;
            cf_meta.set_item("_pre_pass_arg_qparams",
                crate::fx_ir::qparam_dict_to_py(py, &snapshot)?)?;
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
    fn name(&self) -> &str {
        "FxControlFlowStructural"
    }

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

// ===========================================================================
// QParam propagation post-pass
// ===========================================================================

/// Post-pass that copies parent post-pass qparams onto submodule placeholders.
/// Run AFTER parent passes so placeholders see correct post-fold qparams.
pub struct FxControlFlowQParamPropagationPass;

impl FxPass for FxControlFlowQParamPropagationPass {
    fn name(&self) -> &str {
        "FxControlFlowQParamPropagation"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        // Propagate qparams from parent args to submodule placeholders
        let count = propagate_parent_qparams_to_placeholders(py, gm_ref)?;

        // Restamp placeholder dtypes now that qparams are present
        restamp_cf_placeholder_dtypes(py, gm_ref)?;

        // Re-snapshot table qparams with correct qparams
        snapshot_cf_table_qparams(py, gm_ref)?;

        Ok(count > 0)
    }
}

// ===========================================================================
// Cleanup pass: graph surgery functions ported from _cf_rewrites.py / _cf_ctx.py
// ===========================================================================

// ---------------------------------------------------------------------------
// Helpers for the cleanup pass
// ---------------------------------------------------------------------------

fn is_tosa_rescale_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "RESCALE")
}

fn is_tosa_transpose_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "TRANSPOSE")
}

/// Cleanup graph: eliminate dead code + lint.
fn cleanup_graph(graph: &Bound<'_, PyAny>) -> PyResult<()> {
    graph.call_method0("eliminate_dead_code")?;
    graph.call_method0("lint")?;
    Ok(())
}

/// Get the rescale signature: canonicalized args[1:5].
fn rescale_signature<'py>(node: &Bound<'py, PyAny>) -> Option<Vec<String>> {
    if !is_tosa_rescale_node(node) {
        return None;
    }
    let args = FxGraphModule::node_args(node).ok()?;
    if args.len() < 5 {
        return None;
    }
    let mut sig = Vec::new();
    for arg in &args[1..5] {
        sig.push(format!("{}", arg));
    }
    Some(sig)
}

/// Extract a single Python qparam entry from a qparams dict.
/// Returns Bound<PyAny> for callers that need to pass it back to Python.
/// For Rust-native QParam, use extract_qparam() instead.
fn extract_single_qparam_entry<'py>(
    qparams: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    if qparams.is_none() {
        return None;
    }
    let is_truthy = qparams.is_truthy().unwrap_or(false);
    if !is_truthy {
        return None;
    }
    // Try dict with key 0
    if let Ok(val) = qparams.get_item(0i64) {
        if !val.is_none() {
            return Some(val);
        }
    }
    // Try dict with len==1 -> return first value
    if let Ok(len) = qparams.len() {
        if len == 1 {
            if let Ok(values) = qparams.call_method0("values") {
                if let Ok(mut iter) = values.iter() {
                    if let Some(Ok(val)) = iter.next() {
                        return Some(val);
                    }
                }
            }
        }
    }
    Some(qparams.clone())
}

/// Extract the effective quantization parameters for a node by walking
/// producer edges until a node with qparam metadata or a RESCALE is found.
fn extract_effective_qparam(
    node: &Bound<'_, PyAny>,
) -> PyResult<Option<QParam>> {
    let mut current = node.clone();
    let mut seen = std::collections::HashSet::new();

    loop {
        if current.getattr("op").is_err() { break; }
        let node_id: usize = current.call_method0("__hash__")
            .and_then(|h| h.extract()).unwrap_or(0);
        if !seen.insert(node_id) { break; }

        let meta = current.getattr("meta")?;
        if let Ok(oqp) = meta.get_item("output_qparams") {
            if let Some(qp) = extract_qparam(&oqp) { return Ok(Some(qp)); }
        }
        if let Ok(iqp) = meta.get_item("input_qparams") {
            if let Some(qp) = extract_qparam(&iqp) { return Ok(Some(qp)); }
        }

        let args = FxGraphModule::node_args(&current)?;
        if let Some(qp) = try_reverse_engineer_rescale_qparam(&current, &args)? {
            return Ok(Some(qp));
        }
        current = match first_node_arg(&current)? { Some(a) => a, None => break };
    }
    Ok(None)
}

/// Derive a QParam from a TOSA RESCALE node: output_scale = source_scale / multiplier.
fn try_reverse_engineer_rescale_qparam(
    node: &Bound<'_, PyAny>,
    args: &[Bound<'_, PyAny>],
) -> PyResult<Option<QParam>> {
    if !is_tosa_rescale_node(node) || args.len() < 5 { return Ok(None); }
    let source = &args[0];
    if source.getattr("op").is_err() { return Ok(None); }
    let source_qp = match extract_effective_qparam(source)? {
        Some(qp) => qp,
        None => return Ok(None),
    };
    let mult_val: f64 = args[2].get_item(0_usize)
        .ok().and_then(|v| v.extract().ok()).unwrap_or(0.0);
    if mult_val == 0.0 || args[2].len().unwrap_or(0) != 1 { return Ok(None); }
    let output_zp: i64 = match args[4].extract() { Ok(v) => v, Err(_) => return Ok(None) };
    let dtype_str = format!("{}", args[1]);
    if !dtype_str.contains("int") { return Ok(None); }
    let (qmin, qmax) = if dtype_str.contains("int8") { (-128i64, 127i64) }
        else if dtype_str.contains("int16") { (-32768, 32767) }
        else { (i32::MIN as i64, i32::MAX as i64) };
    Ok(Some(QParam {
        scale: source_qp.scale / mult_val,
        zp: output_zp,
        qmin, qmax,
        dtype_str,
    }))
}

/// Stamp single qparams on a node (input_qparam and/or output_qparam).
/// Stamp input/output qparams on a node's metadata from QParam values.
fn stamp_qparams(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    input: Option<&QParam>,
    output: Option<&QParam>,
) -> PyResult<()> {
    let meta = node.getattr("meta")?;
    if let Some(qp) = input {
        qp.stamp_on_meta(py, &meta, "input_qparams")?;
    }
    if let Some(qp) = output {
        qp.stamp_on_meta(py, &meta, "output_qparams")?;
    }
    Ok(())
}

/// Copy output contract meta from src to dst.
fn copy_output_contract_meta<'py>(
    dst: &Bound<'py, PyAny>,
    src: &Bound<'py, PyAny>,
    copy_output_qparams: bool,
) -> PyResult<()> {
    let dst_meta = dst.getattr("meta")?;
    let src_meta = src.getattr("meta")?;
    // val
    if let Some(val) = meta_val(&src_meta, "val") {
        dst_meta.set_item("val", &val)?;
    }
    // tosa_dim_order
    if let Some(tdo) = meta_val(&src_meta, "tosa_dim_order") {
        let py = dst.py();
        let t = PyTuple::new_bound(py, &py_iter_collect(&tdo));
        dst_meta.set_item("tosa_dim_order", &t)?;
    }
    if copy_output_qparams {
        if let Some(oqp) = meta_val(&src_meta, "output_qparams") {
            let d = oqp.call_method0("copy")?;
            dst_meta.set_item("output_qparams", &d)?;
        }
    }
    Ok(())
}

/// Get all modules (parent + submodules) for iteration.
fn get_all_modules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }
    Ok(modules)
}

/// Get `_actual_shape_from_meta` for a node. Returns None if not available.
fn actual_shape_from_meta<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Vec<i64>>> {
    let meta = node.getattr("meta")?;
    let fake_tensor = match meta_val(&meta, "val") {
        Some(v) => v,
        None => return Ok(None),
    };
    let shape_obj = match fake_tensor.getattr("shape") {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let shape: Vec<i64> = match shape_obj.extract() {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let dim_order = meta_val(&meta, "tosa_dim_order");
    if let Some(tdo) = dim_order {
        let tdo_vec: Vec<usize> = match tdo.extract() {
            Ok(v) => v,
            Err(_) => return Ok(Some(shape)),
        };
        if tdo_vec.len() == shape.len() {
            let identity: Vec<usize> = (0..shape.len()).collect();
            if tdo_vec != identity {
                let permuted: Vec<i64> = tdo_vec.iter().map(|&idx| shape[idx]).collect();
                return Ok(Some(permuted));
            }
        }
    }
    Ok(Some(shape))
}

// ---------------------------------------------------------------------------
// 1. _remove_control_flow_noop_passthroughs
// ---------------------------------------------------------------------------

/// Check if two Python values have the same extractable value for a given attr.
fn meta_attr_matches<T: PartialEq + for<'a> pyo3::FromPyObject<'a>>(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    attr: &str,
) -> bool {
    let av = a.getattr(attr).ok().and_then(|v| v.extract::<T>().ok());
    let bv = b.getattr(attr).ok().and_then(|v| v.extract::<T>().ok());
    match (av, bv) {
        (Some(a), Some(b)) => a == b,
        _ => true, // if either is missing, don't block
    }
}

/// Copy a metadata key from src to dst if dst doesn't have it.
fn propagate_meta_key(
    py: Python<'_>,
    dst_meta: &Bound<'_, PyAny>,
    src_meta: &Bound<'_, PyAny>,
    key: &str,
) -> PyResult<()> {
    if meta_val(dst_meta, key).is_some() { return Ok(()); }
    if let Some(val) = meta_val(src_meta, key) {
        if !val.is_truthy().unwrap_or(false) { return Ok(()); }
        let copy = val.call_method0("copy")?;
        dst_meta.set_item(key, &copy)?;
    }
    Ok(())
}

pub(crate) fn remove_control_flow_noop_passthroughs<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut rewritten: u32 = 0;

    let noop_markers = &[
        "dim_order_ops._clone_dim_order",
        "aten.clone",
        "aten.alias_copy",
        "aten.detach_copy",
    ];

    for (_name, submodule, _cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;
        let mut updated = false;
        let nodes = graph_nodes(&submodule)?;

        for node in &nodes {
            if !noop_markers.iter().any(|m| target_contains(node, m)) { continue; }
            let source = match first_node_arg(node)? { Some(a) => a, None => continue };

            let node_meta = node.getattr("meta")?;
            let source_meta = source.getattr("meta")?;
            let nv = meta_val(&node_meta, "val");
            let sv = meta_val(&source_meta, "val");

            // Skip if dtype, shape, or dim_order mismatch
            if let (Some(ref nv), Some(ref sv)) = (&nv, &sv) {
                if !meta_attr_matches::<String>(nv, sv, "dtype") { continue; }
                if !meta_attr_matches::<Vec<i64>>(nv, sv, "shape") { continue; }
            }
            if !meta_attr_matches::<Vec<i64>>(&node_meta, &source_meta, "tosa_dim_order") {
                continue;
            }

            // Propagate metadata from noop node to source
            propagate_meta_key(py, &source_meta, &node_meta, "tosa_dim_order")?;
            propagate_meta_key(py, &source_meta, &node_meta, "input_qparams")?;
            propagate_meta_key(py, &source_meta, &node_meta, "output_qparams")?;

            node.call_method1("replace_all_uses_with", (source,))?;
            graph.call_method1("erase_node", (node,))?;
            rewritten += 1;
            updated = true;
        }

        if updated {
            cleanup_graph(&graph)?;
            recompile(&submodule)?;
        }
    }

    Ok(rewritten)
}

// ---------------------------------------------------------------------------
// 2. _prune_duplicate_terminal_rescales
// ---------------------------------------------------------------------------

pub(crate) fn prune_duplicate_terminal_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let mut pruned: u32 = 0;

    // Include parent + submodules (matches Python include_parent=True)
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }

    for module in &modules {
        let graph = module.getattr("graph")?;
        let mut modified = false;

        let nodes = graph_nodes(module)?;

        for node in &nodes {
            if !is_tosa_rescale_node(node) {
                continue;
            }
            // Check single user
            let users = FxGraphModule::node_users(node)?;
            if users.len() != 1 {
                continue;
            }
            let user = &users[0];

            if !is_tosa_rescale_node(user) {
                continue;
            }

            // user.args[0] is node
            let user_args = FxGraphModule::node_args(user)?;
            if user_args.is_empty() {
                continue;
            }
            let user_arg0_is_node = user_args[0].is(node);
            if !user_arg0_is_node {
                continue;
            }

            // Compare rescale signatures
            let node_sig = rescale_signature(node);
            let user_sig = rescale_signature(user);
            match (&node_sig, &user_sig) {
                (Some(ns), Some(us)) if ns == us => {}
                _ => continue,
            }

            // Check qparams: first has no output_qparams, user has output_qparams
            let node_meta = node.getattr("meta")?;
            let user_meta = user.getattr("meta")?;

            let first_output_qparam = {
                let oqp = meta_val(&node_meta, "output_qparams");
                match oqp {
                    Some(ref v) => extract_single_qparam_entry(v),
                    None => None,
                }
            };
            let final_output_qparam = {
                let oqp = meta_val(&user_meta, "output_qparams");
                match oqp {
                    Some(ref v) => extract_single_qparam_entry(v),
                    None => None,
                }
            };

            if first_output_qparam.is_some() {
                continue;
            }
            if final_output_qparam.is_none() {
                continue;
            }
            let final_output_qparam = final_output_qparam.unwrap();

            // Rewrite: node.args = (source, user.args[1..5])
            let node_args = FxGraphModule::node_args(node)?;
            let source = &node_args[0];

            let new_args = PyTuple::new_bound(
                py,
                &[
                    source.clone(),
                    user_args[1].clone(),
                    user_args[2].clone(),
                    user_args[3].clone(),
                    user_args[4].clone(),
                ],
            );
            node.setattr("args", &new_args)?;

            // Copy output contract meta (without output_qparams)
            copy_output_contract_meta(node, user, false)?;

            // If node has no input_qparams, try to get from source
            if meta_val(&node_meta, "input_qparams").is_none()
                || !node_meta
                    .get_item("input_qparams")
                    .map(|v| v.is_truthy().unwrap_or(false))
                    .unwrap_or(false)
            {
                let source_meta = source.getattr("meta")?;
                if let Some(soqp) = meta_val(&source_meta, "output_qparams") {
                    if let Some(qp) = extract_qparam(&soqp) {
                        stamp_qparams(py, node, Some(&qp), None)?;
                    }
                }
            }

            let final_qp = QParam::from_py(&final_output_qparam);
            stamp_qparams(py, node, None, Some(&final_qp))?;

            // Replace user with node and erase user
            user.call_method1("replace_all_uses_with", (node,))?;
            graph.call_method1("erase_node", (user,))?;
            pruned += 1;
            modified = true;
        }

        if modified {
            cleanup_graph(&graph)?;
            recompile(module)?;
        }
    }

    Ok(pruned)
}

// ---------------------------------------------------------------------------
// 6. _refresh_tosa_transpose_fake_tensors
// ---------------------------------------------------------------------------

pub(crate) fn refresh_tosa_transpose_fake_tensors<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let modules = get_all_modules(py, gm)?;
    let mut refreshed: u32 = 0;

    for module in &modules {
        let mut updated = false;

        for node in graph_nodes(module)? {
            if !is_tosa_transpose_node(&node) {
                continue;
            }

            let args = FxGraphModule::node_args(&node)?;
            if args.len() < 2 {
                continue;
            }

            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            let input_meta = input_node.getattr("meta")?;
            let input_fake = match meta_val(&input_meta, "val") {
                Some(v) => v,
                None => continue,
            };

            if !input_fake.hasattr("new_empty")? {
                continue;
            }

            let input_shape = match actual_shape_from_meta(input_node)? {
                Some(s) => s,
                None => continue,
            };

            // Parse perm from args[1]
            let perm_obj = &args[1];
            let perm: Vec<usize> = match perm_obj.extract() {
                Ok(v) => v,
                Err(_) => continue,
            };

            if perm.len() != input_shape.len() {
                continue;
            }

            // Compute new shape
            let new_shape: Vec<i64> = perm.iter().map(|&idx| input_shape[idx]).collect();

            let node_meta = node.getattr("meta")?;
            let node_fake = meta_val(&node_meta, "val").unwrap_or(input_fake.clone());

            let base_tensor = if node_fake.hasattr("new_empty")? {
                node_fake.clone()
            } else {
                input_fake.clone()
            };

            // Get dtype
            let dtype = node_fake
                .getattr("dtype")
                .or_else(|_| input_fake.getattr("dtype"))?;

            // Create new tensor: base_tensor.new_empty(new_shape, dtype=dtype)
            let shape_list = PyList::new_bound(py, &new_shape);
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", &dtype)?;
            let new_val = base_tensor.call_method("new_empty", (&shape_list,), Some(&kwargs))?;

            node_meta.set_item("val", &new_val)?;

            // Set tosa_dim_order to identity
            let identity: Vec<i64> = (0..new_shape.len() as i64).collect();
            let identity_tuple = PyTuple::new_bound(py, &identity);
            node_meta.set_item("tosa_dim_order", &identity_tuple)?;

            updated = true;
            refreshed += 1;
        }

        if updated {
            recompile(module)?;
        }
    }

    Ok(refreshed)
}

// NOTE: _refresh_tosa_dim_order_fake_tensors (item 7) is already implemented
// above as `refresh_tosa_dim_order_fake_tensors`.

fn is_dequantize_node(node: &Bound<'_, PyAny>) -> bool {
    target_contains(node, "dequantize_per_tensor")
}

// ---------------------------------------------------------------------------
// Q/DQ node helpers
// ---------------------------------------------------------------------------

/// Parse scale/zp/qmin/qmax/dtype from a Q or DQ node's literal args.
/// Returns a Python QuantArgs object or None.
/// Parse Q/DQ node literal args into a Python QuantArgs object.
/// Returns Python QuantArgs object for callers that need to pass it to Python functions.
fn parse_per_tensor_qparam<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let args = FxGraphModule::node_args(node)?;
    match QParam::from_q_node_args(&args) {
        Some(qp) => Ok(Some(qp.to_py(py)?)),
        None => Ok(None),
    }
}

/// Parse Q/DQ node literal args into a Rust QParam.
fn parse_qparam(node: &Bound<'_, PyAny>) -> PyResult<Option<QParam>> {
    let args = FxGraphModule::node_args(node)?;
    Ok(QParam::from_q_node_args(&args))
}

/// Check if a node is a zero-input constant source (get_attr or no input args).
fn is_zero_input_const_source(node: &Bound<'_, PyAny>) -> bool {
    let op: String = py_attr(node, &["op"], String::new());
    if op == "get_attr" {
        return true;
    }
    if op != "call_function" {
        return false;
    }
    let args = FxGraphModule::node_args(node).unwrap_or_default();
    args.iter().all(|a| !FxGraphModule::is_node(a))
}

/// Collapse matching Q->DQ pairs in a submodule graph.
/// Rust port of Python `_collapse_control_flow_qdq_pairs`.
fn collapse_control_flow_qdq_pairs<'py>(
    _py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    quant_count: &mut u32,
) -> PyResult<bool> {
    let mut modified = false;
    let nodes: Vec<Bound<'py, PyAny>> = graph
        .getattr("nodes")?
        .iter()?
        .filter_map(|n| n.ok())
        .collect();

    for node in &nodes {
        if !is_dequantize_node(node) { continue; }
        let quant_node = match first_node_arg(node)? { Some(a) => a, None => continue };
        if !is_quantize_node(&quant_node) {
            continue;
        }
        let quant_qp = match parse_qparam(&quant_node)? {
            Some(qp) => qp,
            None => continue,
        };
        let dequant_qp = match parse_qparam(node)? {
            Some(qp) => qp,
            None => continue,
        };
        if !quant_qp.same_as(&dequant_qp) {
            continue;
        }
        let float_source = match first_node_arg(&quant_node)? { Some(a) => a, None => continue };

        // Skip if zero-input const feeding a scalar binary
        if is_zero_input_const_source(&float_source) {
            // Check if any user of the DQ node is a binary op (simplified check)
            let users = node.getattr("users")?;
            let user_count: usize = users.len().unwrap_or(0);
            if user_count > 0 {
                let mut feeds_binary = false;
                for user in users.iter()? {
                    let user = user?;
                    let target = FxGraphModule::node_target_str(&user)?;
                    if target.contains("add.") || target.contains("sub.")
                        || target.contains("mul.") {
                        feeds_binary = true;
                        break;
                    }
                }
                if feeds_binary {
                    continue;
                }
            }
        }

        // Bypass: replace DQ with float source
        node.call_method1("replace_all_uses_with", (float_source,))?;
        graph.call_method1("erase_node", (node,))?;
        // Erase Q if no remaining users
        let q_users = quant_node.getattr("users")?;
        if q_users.len().unwrap_or(1) == 0 {
            graph.call_method1("erase_node", (quant_node,))?;
        }
        *quant_count += 1;
        modified = true;
    }
    Ok(modified)
}

/// Rewrite standalone dequantize nodes in a submodule graph.
/// Rust port with Python fallback for the int32 cluster case.
fn rewrite_control_flow_dequantize_nodes<'py>(
    _py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    _submodule: &Bound<'py, PyAny>,
    _cf_node: &Bound<'py, PyAny>,
    dequant_count: &mut u32,
) -> PyResult<bool> {
    let mut modified = false;
    let nodes: Vec<Bound<'py, PyAny>> = graph
        .getattr("nodes")?
        .iter()?
        .filter_map(|n| n.ok())
        .collect();

    for node in &nodes {
        if !is_dequantize_node(node) { continue; }
        let source_node = match first_node_arg(node)? { Some(a) => a, None => continue };
        let users: Vec<Bound<'py, PyAny>> = node
            .getattr("users")?
            .iter()?
            .filter_map(|u| u.ok())
            .collect();
        if users.is_empty() {
            continue;
        }

        // Int32 cluster rewrite — dead under correct qparam propagation.
        if let Ok(Some(_source_qp)) = extract_effective_qparam(&source_node) {
        }

        // Simple bypass: if all users are RESCALE or output, remove the DQ
        let all_rescale_or_output = users.iter().all(|user| {
            is_tosa_rescale_node(user)
                || NodeOp::of(user) == NodeOp::Output
        });
        if !all_rescale_or_output {
            continue;
        }

        node.call_method1("replace_all_uses_with", (source_node,))?;
        graph.call_method1("erase_node", (node,))?;
        *dequant_count += 1;
        modified = true;
    }
    Ok(modified)
}

/// Rewrite standalone quantize nodes in a submodule graph.
/// Cases 1 (same qparam bypass) and 3 (rescale insertion) in Rust,
/// case 2 (const materialization) delegates to Python.
fn rewrite_control_flow_quantize_nodes<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    submodule: &Bound<'py, PyAny>,
    _cf_node: &Bound<'py, PyAny>,
    quant_count: &mut u32,
) -> PyResult<bool> {
    let mut modified = false;
    let nodes: Vec<Bound<'py, PyAny>> = graph
        .getattr("nodes")?
        .iter()?
        .filter_map(|n| n.ok())
        .collect();

    for node in &nodes {
        if !is_quantize_node(node) { continue; }
        let source_node = match first_node_arg(node)? { Some(a) => a, None => continue };
        let target = match parse_qparam(node)? {
            Some(qp) => qp,
            None => continue,
        };
        let source = extract_effective_qparam(&source_node)?;

        // Skip float sources with no qparams (unless zero-input const)
        let source_dtype_str = source_node.getattr("meta")?
            .get_item("val").ok()
            .and_then(|v| v.getattr("dtype").ok())
            .map(|d| format!("{d}")).unwrap_or_default();
        let source_is_int = source_dtype_str.contains("int");
        if !source_is_int && source.is_none() && !is_zero_input_const_source(&source_node) {
            continue;
        }

        if source.is_none() {
            continue;
        }
        let source = source.unwrap();

        // Case 1: same qparam → bypass (erase Q node)
        if source.same_as(&target) {
            node.call_method1("replace_all_uses_with", (source_node,))?;
            graph.call_method1("erase_node", (node,))?;
            *quant_count += 1;
            modified = true;
            continue;
        }

        // Case 3: different qparam → insert RESCALE
        let rescale = source.insert_rescale_to(&target, py, submodule, node, &source_node, false)?;
        stamp_qparams(py, &rescale, Some(&source), Some(&target))?;
        node.call_method1("replace_all_uses_with", (&rescale,))?;
        graph.call_method1("erase_node", (node,))?;
        *quant_count += 1;
        modified = true;
    }
    Ok(modified)
}

// ---------------------------------------------------------------------------
// _rewrite_control_flow_quantization_boundaries
//
// Iterates control-flow submodules and rewrites quantize/dequantize nodes:
//   1. Collapse matching Q->DQ pairs (bypass to float source)
//   2. Rewrite standalone quantize nodes into rescale or const materializations
//   3. Rewrite standalone dequantize nodes into int32 lifts or direct edges
//
// Iterates control-flow submodules and rewrites Q/DQ boundary nodes
// into RESCALE ops or bypasses.
// ---------------------------------------------------------------------------

pub(crate) fn rewrite_control_flow_quantization_boundaries<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    if submodules.is_empty() {
        return Ok(0);
    }

    let mut total_rewrites: u32 = 0;
    let mut quant_count: u32 = 0;
    let mut dequant_count: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;

        let m1 = collapse_control_flow_qdq_pairs(py, &graph, &mut quant_count)?;
        let m2 = rewrite_control_flow_quantize_nodes(
            py, &graph, submodule, cf_node, &mut quant_count,
        )?;
        let m3 = rewrite_control_flow_dequantize_nodes(
            py, &graph, submodule, cf_node, &mut dequant_count,
        )?;

        if m1 || m2 || m3 {
            cleanup_graph(&graph)?;
            recompile(submodule)?;
            total_rewrites += 1;
        }
    }

    if quant_count > 0 {
        log_line(format_args!(
            "[cf-rewrite-quant-boundaries] rewrote {} control-flow quantize node(s)",
            quant_count
        ));
    }
    if dequant_count > 0 {
        log_line(format_args!(
            "[cf-rewrite-quant-boundaries] rewrote {} control-flow dequantize node(s)",
            dequant_count
        ));
    }

    Ok(total_rewrites + quant_count + dequant_count)
}

// ---------------------------------------------------------------------------
// _restore_control_flow_float_compare_inputs
//
// Finds compare ops (eq, ge, gt, lt, le) in submodules where one input is
// quantized (int8/int16/int32) and the other is float. Inserts a
// dequantize_per_tensor node to restore the float input for the compare.
// ---------------------------------------------------------------------------

pub(crate) fn restore_control_flow_float_compare_inputs<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    if submodules.is_empty() {
        return Ok(0);
    }

    let ctx = CfCtx::new(py)?;
    let int8 = ctx.int8()?;
    let int16 = ctx.int16()?;
    let int32 = ctx.torch.getattr("int32")?;
    let float16 = ctx.torch.getattr("float16")?;
    let float32 = ctx.torch.getattr("float32")?;
    let float64 = ctx.torch.getattr("float64")?;

    let arm_utils =
        py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
    let create_node_fn = arm_utils.getattr("create_node")?;
    let exir_ops = py
        .import_bound("executorch.exir.dialects._ops")?
        .getattr("ops")?;
    let dq_op = exir_ops
        .getattr("edge")?
        .getattr("quantized_decomposed")?
        .getattr("dequantize_per_tensor")?
        .getattr("default")?;

    let compare_markers = &[
        "aten.gt.",
        "aten.ge.",
        "aten.lt.",
        "aten.le.",
        "aten.eq.",
    ];

    let mut restored: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;
        let mut updated = false;

        let nodes = graph_nodes(&submodule)?;

        for node in &nodes {
            if !compare_markers.iter().any(|m| target_contains(node, m)) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.len() != 2 {
                continue;
            }

            // Try each arg index
            let mut did_restore = false;
            for idx in 0..2usize {
                let arg = &args[idx];
                let other = &args[1 - idx];

                if !FxGraphModule::is_node(arg) || !FxGraphModule::is_node(other) {
                    continue;
                }

                let arg_meta = arg.getattr("meta")?;
                let other_meta = other.getattr("meta")?;
                let arg_val = meta_val(&arg_meta, "val");
                let other_val = meta_val(&other_meta, "val");

                let arg_dtype = arg_val.as_ref().and_then(|v| v.getattr("dtype").ok());
                let other_dtype = other_val.as_ref().and_then(|v| v.getattr("dtype").ok());

                // arg must be int8/int16/int32
                let arg_is_int = match &arg_dtype {
                    Some(dt) => {
                        dt.eq(&int8).unwrap_or(false)
                            || dt.eq(&int16).unwrap_or(false)
                            || dt.eq(&int32).unwrap_or(false)
                    }
                    None => false,
                };

                // other must be float
                let other_is_float = match &other_dtype {
                    Some(dt) => {
                        dt.eq(&float16).unwrap_or(false)
                            || dt.eq(&float32).unwrap_or(false)
                            || dt.eq(&float64).unwrap_or(false)
                    }
                    None => false,
                };

                if !arg_is_int || !other_is_float {
                    continue;
                }

                let qp = match extract_effective_qparam(&arg)? {
                    Some(qp) => qp,
                    None => continue,
                };
                let torch = py.import_bound("torch")?;
                let qparam_dtype = torch.getattr(&*qp.dtype_str.replace("torch.", ""))?;

                // graph.inserting_before(node) -- use context manager
                let inserting_ctx = graph.call_method1("inserting_before", (node,))?;
                inserting_ctx.call_method0("__enter__")?;

                // create_node(graph=ctx.graph, op_target=dq_op, args=(...), from_node=node)
                let dq_args = PyTuple::new_bound(
                    py,
                    &[
                        arg.clone(),
                        qp.scale.into_py(py).into_bound(py),
                        qp.zp.into_py(py).into_bound(py),
                        qp.qmin.into_py(py).into_bound(py),
                        qp.qmax.into_py(py).into_bound(py),
                        qparam_dtype.clone(),
                    ],
                );
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("graph", &graph)?;
                kwargs.set_item("op_target", &dq_op)?;
                kwargs.set_item("args", &dq_args)?;
                kwargs.set_item("from_node", node)?;
                let dq = create_node_fn.call((), Some(&kwargs))?;

                // Exit context manager
                let none = py.None().into_bound(py);
                inserting_ctx.call_method1("__exit__", (&none, &none, &none))?;

                // Set meta on the dq node
                let dq_meta = dq.getattr("meta")?;
                if let Some(ref av) = arg_val {
                    if av.hasattr("to").unwrap_or(false) {
                        if let Some(ref odt) = other_dtype {
                            let to_kwargs = PyDict::new_bound(py);
                            to_kwargs.set_item("dtype", odt)?;
                            let new_val = av.call_method("to", (), Some(&to_kwargs))?;
                            dq_meta.set_item("val", &new_val)?;
                        }
                    }
                }
                // input_qparams = {0: qparam}
                let iqp = PyDict::new_bound(py);
                iqp.set_item(0i64, qp.to_py(py)?)?;
                dq_meta.set_item("input_qparams", &iqp)?;
                // output_qparams = {}
                let oqp = PyDict::new_bound(py);
                dq_meta.set_item("output_qparams", &oqp)?;

                // node.update_arg(idx, dq)
                node.call_method1("update_arg", (idx, &dq))?;
                restored += 1;
                updated = true;
                did_restore = true;
                break;
            }

            if did_restore {
                // Only break the inner arg loop, continue to next node
            }
        }

        if updated {
            cleanup_graph(&graph)?;
            recompile(&submodule)?;
        }
    }

    if restored > 0 {
        log_line(format_args!(
            "[cf-restore-float-compare] restored {} control-flow float-compare dequantize input(s)",
            restored
        ));
    }

    Ok(restored)
}

// ---------------------------------------------------------------------------
// _normalize_control_flow_output_layouts
//
// Wrapper that delegates to Python _normalize_control_flow_output_contracts
// with normalize_outputs=True.
// ---------------------------------------------------------------------------

/// No-op — layout normalization not needed with correct qparam propagation.
pub(crate) fn normalize_control_flow_output_layouts<'py>(
    _py: Python<'py>,
    _gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    Ok(0)
}

// ---------------------------------------------------------------------------
// Combined Cleanup Pass
// ---------------------------------------------------------------------------

// ===========================================================================
// Control-flow boundary RESCALE insertion
// Boundary RESCALE insertion (matches ET ARM InsertControlFlowRescalesPass):
//   - Input RESCALE: external(parent) → internal(body) after placeholder
//   - Output RESCALE: internal(body) → external(consumer) before output
// ===========================================================================

/// Insert boundary RESCALE pairs for while_loop submodules.
/// Must run AFTER submodule fold_qdq (so placeholders have output_qparams
/// from Q literals, and output node has input_qparams from DQ literals).
pub(crate) fn insert_cf_boundary_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let ctx = CfCtx::new(py)?;
    let nodes = graph_nodes(gm)?;

    let mut count = 0u32;

    for node in &nodes {
        if NodeOp::of(node) != NodeOp::CallFunction { continue; }
        let cf = match ctx.cf_kind(node)? { Some(k) => k, None => continue };

        // Get external qparams from the cf node's input_qparams
        // External qparam: while_loop carried_args at idx 2, cond data_args at idx 3
        let cf_meta = node.getattr("meta")?;
        let ext_qp_dict = cf_meta.get_item("input_qparams").ok();
        // The external qparam is at key matching the carried_args index in the node
        let carried_idx: usize = if cf == CfKind::While { 2 } else { 3 };
        let ext_qp = ext_qp_dict.as_ref()
            .and_then(|d| d.get_item(carried_idx).ok())
            .map(|v| QParam::from_py(&v))
            .or_else(|| {
                cf_meta.get_item("_pre_pass_arg_qparams").ok()
                    .and_then(|d| extract_qparam(&d))
            });

        // Get submodules
        let args = FxGraphModule::node_args(node)?;
        let submodule_indices: Vec<usize> = if cf == CfKind::While {
            vec![0, 1] // cond_fn, body_fn
        } else {
            vec![1, 2] // if_branch, else_branch
        };

        for &sm_idx in &submodule_indices {
            if sm_idx >= args.len() { continue; }
            let sm_name: String = args[sm_idx].getattr("target")
                .and_then(|t| t.extract()).unwrap_or_default();
            let submodule = match gm.call_method1("get_submodule", (&sm_name,)) {
                Ok(sm) => sm,
                Err(_) => continue,
            };
            let sg = submodule.getattr("graph")?;
            let mut modified = false;

            // --- Input RESCALE: after placeholder ---
            if let Some(ref ext) = ext_qp {
                let placeholders: Vec<Bound<'py, PyAny>> = sg.getattr("nodes")?
                    .iter()?.filter_map(|n| n.ok())
                    .filter(|n| NodeOp::of(n) == NodeOp::Placeholder)
                    .collect();

                for ph in &placeholders {
                    let ph_meta = ph.getattr("meta")?;
                    let internal = match ph_meta.get_item("output_qparams").ok().and_then(|oqp| extract_qparam(&oqp)) {
                        Some(qp) => qp,
                        None => continue,
                    };
                    if ext.same_scale_zp(&internal) || ext.scale == 0.0 || internal.scale == 0.0 { continue; }

                    let rescale = ext.insert_rescale_to(&internal, py, &submodule, ph, ph, true)?;
                    ph.call_method1("replace_all_uses_with", (&rescale,))?;
                    rescale.call_method("update_arg", (0i64, ph), None)?;
                    let _ = ph_meta.del_item("output_qparams");
                    modified = true;
                    count += 1;
                }
            }

            // --- Output RESCALE: before output (while body only, or cond branches) ---
            if (cf == CfKind::While && sm_idx == 1) || cf == CfKind::Cond {
                let output_qparams = infer_cf_output_qparams_map(node)?;
                if !output_qparams.is_empty() {
                    if let Some(output) = output_node(&submodule)? {
                        let ovals = unpack_output_vals(&output)?;

                        let out_meta = output.getattr("meta")?;
                        let int_out_qp = out_meta.get_item("input_qparams").ok();

                        let mut new_vals = ovals.clone();
                        let mut out_modified = false;
                        for (idx, ext_qp) in &output_qparams {
                            let out_val = match ovals.get(*idx) {
                                Some(v) if FxGraphModule::is_node(v) => v,
                                _ => continue,
                            };
                            let internal = match int_out_qp.as_ref()
                                .and_then(|d| d.get_item(*idx).ok())
                                .map(|v| QParam::from_py(&v)) {
                                Some(qp) => qp,
                                None => continue,
                            };
                            if internal.same_scale_zp(ext_qp) || internal.scale == 0.0 || ext_qp.scale == 0.0 { continue; }

                            let rescale = internal.insert_rescale_to(ext_qp, py, &submodule, &output, out_val, false)?;
                            new_vals[*idx] = rescale;
                            out_modified = true;
                            count += 1;
                        }
                        if out_modified {
                            repack_output_vals(py, &output, &new_vals)?;
                        }
                    }
                }
            }

            if modified {
                cleanup_graph(&sg)?;
                recompile(&submodule)?;
            }
        }
    }

    if count > 0 {
        log_line(format_args!(
            "[insert-cf-boundary-rescales] inserted {} boundary RESCALE(s)", count
        ));
    }
    Ok(count)
}

// ===========================================================================
// Additional control-flow functions (delegate to Python via PyO3)
// ===========================================================================

/// Infer output qparams from cond/while_loop getitem consumer nodes.
fn infer_cf_output_qparams_map(
    cf_node: &Bound<'_, PyAny>,
) -> PyResult<Vec<(usize, QParam)>> {
    let users = ordered_cf_users(cf_node)?;
    let mut result = Vec::new();
    for user in &users {
        let u_args = FxGraphModule::node_args(user)?;
        if u_args.len() < 2 { continue; }
        let idx: usize = match u_args[1].extract() { Ok(i) => i, Err(_) => continue };
        let meta = user.getattr("meta")?;
        // Try output_qparams
        if let Ok(oqp) = meta.get_item("output_qparams") {
            if let Some(qp) = extract_qparam(&oqp) {
                result.push((idx, qp));
                continue;
            }
        }
        // Fallback: downstream user's input_qparams
        let user_users = node_users(user)?;
        let mut found = false;
        for uu in &user_users {
            if let Ok(iqp) = uu.getattr("meta")?.get_item("input_qparams") {
                if let Some(qp) = extract_qparam(&iqp) {
                    result.push((idx, qp));
                    found = true;
                    break;
                }
            }
        }
        if found { continue; }
        // Fallback: DQ literal args
        for uu in &user_users {
            if is_dequantize_node(uu) {
                if let Ok(Some(qp)) = parse_qparam(uu) {
                    result.push((idx, qp));
                    break;
                }
            }
        }
    }
    Ok(result)
}

/// Insert boundary RESCALE nodes at submodule output edges.
/// Input rescales are dead with correct qparam propagation.
/// Output rescales use infer_cf_output_qparams_map for target qparams.
pub(crate) fn reapply_control_flow_boundary_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    _include_inputs: bool,
    include_outputs: bool,
) -> PyResult<u32> {
    if !include_outputs { return Ok(0); }

    let ctx = CfCtx::new(py)?;
    let nodes = graph_nodes(gm)?;

    let mut any_modified = false;

    for node in &nodes {
        if NodeOp::of(node) != NodeOp::CallFunction { continue; }
        let cf = match ctx.cf_kind(node)? { Some(k) => k, None => continue };

        let args = FxGraphModule::node_args(node)?;

        // Get submodule(s) to process for outputs
        let submodules_to_process: Vec<Bound<'py, PyAny>> = if cf == CfKind::Cond {
            if args.len() >= 3 {
                let if_name: String = args[1].getattr("target").and_then(|t| t.extract()).unwrap_or_default();
                let else_name: String = args[2].getattr("target").and_then(|t| t.extract()).unwrap_or_default();
                let mut v = Vec::new();
                if let Ok(sm) = gm.call_method1("get_submodule", (&if_name,)) { v.push(sm); }
                if let Ok(sm) = gm.call_method1("get_submodule", (&else_name,)) { v.push(sm); }
                v
            } else { Vec::new() }
        } else {
            // while_loop: only body gets output rescales
            if args.len() >= 2 {
                let body_name: String = args[1].getattr("target").and_then(|t| t.extract()).unwrap_or_default();
                if let Ok(sm) = gm.call_method1("get_submodule", (&body_name,)) { vec![sm] }
                else { Vec::new() }
            } else { Vec::new() }
        };

        // Target qparams
        let targets: Vec<(usize, QParam)> = if cf == CfKind::While {
            let parent_args = resolve_cf_parent_args(node)?;
            parent_args.iter().enumerate()
                .filter_map(|(i, pa)| extract_effective_qparam(pa).ok().flatten().map(|qp| (i, qp)))
                .collect()
        } else {
            infer_cf_output_qparams_map(node)?
        };

        if targets.is_empty() { continue; }

        for submodule in &submodules_to_process {
            let mut modified = false;
            let output = match output_node(submodule)? { Some(o) => o, None => continue };
            let mut ovals = unpack_output_vals(&output)?;
            let was_tuple_wrapped = {
                let oargs = FxGraphModule::node_args(&output)?;
                oargs.len() == 1 && oargs[0].iter().is_ok()
            };

            for (idx, tgt) in &targets {
                let out_val = match ovals.get(*idx) {
                    Some(v) if FxGraphModule::is_node(v) => v,
                    _ => continue,
                };
                let cur = extract_effective_qparam(out_val)?
                    .unwrap_or(QParam { scale: 1.0, zp: 0, qmin: -128, qmax: 127, dtype_str: "torch.int8".into() });
                if cur.same_scale_zp(tgt) || cur.scale == 0.0 || tgt.scale == 0.0 { continue; }

                let rescale = cur.insert_rescale_to(tgt, py, submodule, &output, out_val, false)?;
                ovals[*idx] = rescale;
                modified = true;
            }

            if modified {
                if was_tuple_wrapped {
                    repack_output_vals(py, &output, &ovals)?;
                } else {
                    for (idx, _) in &targets {
                        output.call_method("update_arg", (*idx as i64, &ovals[*idx]), None)?;
                    }
                }
                let sg = submodule.getattr("graph")?;
                cleanup_graph(&sg)?;
                recompile(submodule)?;
                any_modified = true;
            }
        }
    }

    if any_modified {
        cleanup_graph(&gm.getattr("graph")?)?;
        gm.call_method0("recompile")?;
    }
    Ok(if any_modified { 1 } else { 0 })
}

// Combined Output Normalize Phase
// ===========================================================================

/// Runs the full output normalize phase in a single Rust call.
/// Replaces 11 individual Python->Rust round-trips with one call.
pub(crate) fn run_output_normalize_phase<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<()> {
    normalize_control_flow_output_layouts(py, gm)?;
    reapply_control_flow_boundary_rescales(py, gm, false, true)?;
    rewrite_control_flow_quantization_boundaries(py, gm)?;
    remove_control_flow_noop_passthroughs(py, gm)?;
    {
        let gm_wrapper = crate::fx_graph::FxGraphModule::new(gm);
        crate::fx_pass::FxPass::run(&FxControlFlowMetadataPass, &gm_wrapper)?;
    }
    refresh_tosa_transpose_fake_tensors(py, gm)?;
    refresh_tosa_dim_order_fake_tensors(py, gm)?;
    prune_duplicate_terminal_rescales(py, gm)?;
    restore_control_flow_float_compare_inputs(py, gm)?;
    Ok(())
}

// ===========================================================================
// Combined Boundary Phase
// ===========================================================================

/// Runs the post-Rust control-flow boundary phase in a single Rust call.
/// Includes boundary rescale repair and the 813-line binary input rescale refresh.
pub(crate) fn run_boundary_phase<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    _tosa_spec: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // 1. Rewrite quantization boundaries
    rewrite_control_flow_quantization_boundaries(py, gm)?;
    // 2. Reapply boundary rescales (inputs only)
    reapply_control_flow_boundary_rescales(py, gm, true, false)?;
    // 3. Stamp control-flow metadata
    {
        let gm_wrapper = crate::fx_graph::FxGraphModule::new(gm);
        crate::fx_pass::FxPass::run(&FxControlFlowMetadataPass, &gm_wrapper)?;
    }
    Ok(())
}

// ===========================================================================
