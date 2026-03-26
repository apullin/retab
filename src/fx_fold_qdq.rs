use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FoldAndAnnotateQParamsPass`.
///
/// Mirrors the Python `FoldAndAnnotateQParamsPass.call()` logic exactly:
///   1. For each foldable node, iterate its args to find DQ inputs
///   2. Extract qparams from DQ nodes using `QuantArgs.from_operator`
///   3. Store in `node.meta["input_qparams"]` / `node.meta["output_qparams"]`
///   4. Bypass and erase the folded DQ/Q nodes
///
/// The `is_foldable()` check matches Python: nodes must have ARM annotation
/// info with `quantized=True`, or be constant-generating ops.
pub struct FxFoldAndAnnotateQParamsPass {
    /// ExportedProgram for resolving per-channel parameter nodes.
    /// `None` when running without an EP (parameters resolved via gm attrs).
    exported_program: Option<PyObject>,
}

impl FxFoldAndAnnotateQParamsPass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self { exported_program }
    }
}

// ---------------------------------------------------------------------------
// Target-name matching helpers
// ---------------------------------------------------------------------------

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor") || target.contains("quantize_per_channel")
}

fn is_q_or_dq(target: &str) -> bool {
    is_q(target) || is_dq(target)
}

/// Matches Python's `ComputeConstantOpsAOTPass.targeted_ops` +
/// `exir_ops.edge.aten.full_like.default`.
fn is_constant_op(target: &str) -> bool {
    target.contains("full_like.default")
        || target.contains("full.default")
        || target.contains("arange.start_step")
        || target.contains("eye.default")
        || target.contains("linspace.default")
        || target.contains("scalar_tensor.default")
}

// ---------------------------------------------------------------------------
// is_foldable — matches Python FoldAndAnnotateQParamsPass.is_foldable()
// ---------------------------------------------------------------------------

/// Check if a node should have its Q/DQ neighbors folded.
///
/// Mirrors Python logic:
///   1. Must be call_function
///   2. Skip Q/DQ ops themselves
///   3. Always fold constant-generating ops
///   4. Otherwise, require ARM annotation info with quantized=True
fn is_foldable(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let op = FxGraphModule::node_op(node)?;
    if op != "call_function" {
        return Ok(false);
    }

    let target = FxGraphModule::node_target_str(node)?;

    // Don't fold chains of quant-ops into each other.
    if is_q_or_dq(&target) {
        return Ok(false);
    }

    // Always fold q-dq into constant ops.
    if is_constant_op(&target) {
        return Ok(true);
    }

    // Check ARM annotation info: node.meta.get("custom", {}).get("_arm_annotation_info")
    let meta = node.getattr("meta")?;
    let custom = meta.call_method1("get", ("custom",))?;
    if custom.is_none() {
        return Ok(false);
    }

    // Check for the annotation key
    let ann_key = "_arm_annotation_info";
    let has_key: bool = custom
        .call_method1("__contains__", (ann_key,))
        .and_then(|v| v.extract())
        .unwrap_or(false);
    if !has_key {
        return Ok(false);
    }

    // Check quantized flag
    let ann_dict = custom.get_item(ann_key)?;
    let quantized: bool = ann_dict
        .call_method1("get", ("quantized", false))
        .and_then(|v| v.extract())
        .unwrap_or(false);

    Ok(quantized)
}

// ---------------------------------------------------------------------------
// QuantArgs construction via Python's QuantArgs.from_operator
// ---------------------------------------------------------------------------

/// Import the QuantArgs class.
fn get_quant_args_cls<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")
}

/// Call `QuantArgs.from_operator(op_target, args)` -- delegates entirely to
/// Python so we get identical field values (scale, zp, qmin, qmax, dtype,
/// axis, per_channel) as the Python pass.
///
/// `resolved_args` should have parameter nodes already resolved to tensors
/// for the scale/zp positions.
fn quant_args_from_operator<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    resolved_args: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let cls = get_quant_args_cls(py)?;
    cls.call_method1("from_operator", (target, resolved_args))
}

fn int4_special_dtype<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("executorch.backends.arm.tosa.mapping")?
        .getattr("TosaSpecialDtype")?
        .getattr("INT4")
}

fn quant_args_is_int4(quant_args: &Bound<'_, PyAny>) -> bool {
    let qmin: Option<i64> = quant_args.getattr("qmin").ok().and_then(|v| v.extract().ok());
    let qmax: Option<i64> = quant_args.getattr("qmax").ok().and_then(|v| v.extract().ok());
    let dtype = quant_args
        .getattr("dtype")
        .ok()
        .and_then(|v| v.str().ok().map(|s| s.to_string_lossy().into_owned()));

    matches!(qmin, Some(-7))
        && matches!(qmax, Some(7))
        && dtype.as_deref().is_some_and(|s| s.contains("int8"))
}

/// Resolve DQ node args: if scale (args[1]) or zp (args[2]) are FX Nodes
/// referencing parameters, replace them with the actual tensor values.
///
/// Uses `arm_pass_utils.is_param_node` and `arm_pass_utils.get_param_tensor`
/// exactly as Python does.
fn resolve_dq_args<'py>(
    py: Python<'py>,
    dq_node: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    exported_program: &Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let args = dq_node.getattr("args")?;
    let args_list: Vec<Bound<'py, PyAny>> = args.iter()?.collect::<PyResult<_>>()?;

    if args_list.len() < 3 {
        return Ok(args);
    }

    let mut new_args: Vec<Bound<'py, PyAny>> = args_list.clone();

    // Resolve scale (args[1]) and zp (args[2]) if they are param nodes
    if let Some(ep) = exported_program {
        let utils = py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
        let is_param_node_fn = utils.getattr("is_param_node")?;
        let get_param_tensor_fn = utils.getattr("get_param_tensor")?;

        for idx in [1, 2] {
            if idx < new_args.len() && FxGraphModule::is_node(&new_args[idx]) {
                let is_param: bool = is_param_node_fn
                    .call1((ep, &new_args[idx]))
                    .and_then(|v| v.extract())
                    .unwrap_or(false);
                if is_param {
                    let tensor = get_param_tensor_fn.call1((ep, &new_args[idx]))?;
                    new_args[idx] = tensor;
                }
            }
        }
    } else {
        // No exported_program -- try getattr on GraphModule for Node-type args
        for idx in [1, 2] {
            if idx < new_args.len() && FxGraphModule::is_node(&new_args[idx]) {
                let target: String = new_args[idx].getattr("target")?.extract()?;
                if let Ok(tensor) = gm.gm_ref().getattr(target.as_str()) {
                    new_args[idx] = tensor;
                }
            }
        }
    }

    // Build a Python tuple from resolved args
    let tuple = pyo3::types::PyTuple::new_bound(py, &new_args);
    Ok(tuple.into_any())
}

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------

impl FxPass for FxFoldAndAnnotateQParamsPass {
    fn name(&self) -> &str {
        "FxFoldAndAnnotateQParams"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut modified = false;

        // Bind the exported_program PyObject to this Python session
        let exported_program: Option<Bound<'_, PyAny>> =
            self.exported_program.as_ref().map(|ep| ep.bind(py).clone());

        // Snapshot all nodes up-front (Python list is live; we need a stable vec).
        let all_nodes = gm.nodes()?;

        // Nodes to erase after annotation (collected to avoid mutating during iteration)
        let mut nodes_to_erase: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            if !is_foldable(node)? {
                continue;
            }

            // Initialize input_qparams and output_qparams dicts
            let meta = node.getattr("meta")?;
            let input_qparams = PyDict::new_bound(py);
            let output_qparams = PyDict::new_bound(py);
            meta.set_item("input_qparams", &input_qparams)?;
            meta.set_item("output_qparams", &output_qparams)?;

            // ----------------------------------------------------------
            // Input side: iterate args, find DQ nodes, extract qparams
            // ----------------------------------------------------------
            let args = FxGraphModule::node_args(node)?;
            for (i, arg) in args.iter().enumerate() {
                // arg can be a single Node or a list/tuple of Nodes
                let arg_nodes: Vec<Bound<'_, PyAny>> = if FxGraphModule::is_node(arg) {
                    vec![arg.clone()]
                } else if let Ok(iter) = arg.iter() {
                    // list or tuple of args
                    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
                    if items.iter().all(|item| FxGraphModule::is_node(item)) {
                        items
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

                let mut input_qp_for_pos: Option<Bound<'_, PyAny>> = None;
                let mut dq_nodes_to_bypass: Vec<Bound<'_, PyAny>> = Vec::new();

                for arg_node in &arg_nodes {
                    if !FxGraphModule::is_node(arg_node) {
                        input_qp_for_pos = None;
                        dq_nodes_to_bypass.clear();
                        break;
                    }

                    let arg_op = FxGraphModule::node_op(arg_node)?;
                    if arg_op != "call_function" {
                        continue;
                    }

                    let arg_target = FxGraphModule::node_target_str(arg_node)?;
                    if !is_dq(&arg_target) {
                        continue;
                    }

                    // DQ node -- extract qparams
                    let dq_target = arg_node.getattr("target")?;
                    let resolved_args =
                        resolve_dq_args(py, arg_node, gm, &exported_program)?;
                    let quant_args =
                        quant_args_from_operator(py, &dq_target, &resolved_args)?;

                    dq_nodes_to_bypass.push(arg_node.clone());
                    input_qp_for_pos = Some(quant_args);
                }

                if let Some(qa) = input_qp_for_pos {
                    input_qparams.set_item(i, &qa)?;
                    let special_dtype = if quant_args_is_int4(&qa) {
                        Some(int4_special_dtype(py)?)
                    } else {
                        None
                    };

                    // Bypass each DQ node: replace the foldable node's input
                    // from the DQ to the DQ's input (dq.args[0])
                    for dq_node in &dq_nodes_to_bypass {
                        let dq_input = dq_node.getattr("args")?.get_item(0)?;
                        if let Some(ref special_dtype) = special_dtype {
                            dq_input
                                .getattr("meta")?
                                .set_item("tosa_special_dtype", special_dtype)?;
                        }
                        // node.replace_input_with(dq_node, dq_input)
                        node.call_method1("replace_input_with", (dq_node, &dq_input))?;
                        // Erase if no remaining users
                        let users = FxGraphModule::node_users(dq_node)?;
                        if users.is_empty() {
                            nodes_to_erase.push(dq_node.clone());
                        }
                    }
                }
            }

            // ----------------------------------------------------------
            // Output side: iterate users, find Q nodes, extract qparams
            // ----------------------------------------------------------
            // Snapshot users first (Python does copy.copy(n.users)) since
            // we mutate during iteration.
            let users = FxGraphModule::node_users(node)?;
            for (i, user) in users.iter().enumerate() {
                let user_op = FxGraphModule::node_op(user)?;
                if user_op != "call_function" {
                    continue;
                }
                let user_target = FxGraphModule::node_target_str(user)?;
                if !is_q(&user_target) {
                    continue;
                }

                // Q node found -- extract qparams via from_operator
                let q_target = user.getattr("target")?;
                let q_args = user.getattr("args")?;
                let quant_args = quant_args_from_operator(py, &q_target, &q_args)?;

                output_qparams.set_item(i, &quant_args)?;

                // Bypass: replace all uses of Q with the foldable node
                FxGraphModule::replace_all_uses_with(user, node)?;
                nodes_to_erase.push(user.clone());
            }

            // Handle sum.dim_IntList: set dtype kwarg to output qparams dtype.
            // Mirrors Python: set_node_arg(n, "dtype", output_dtype)
            let target = FxGraphModule::node_target_str(node)?;
            if target.contains("sum.dim_IntList") && output_qparams.len() > 0 {
                if let Some(qa) = output_qparams.get_item(0)? {
                    let output_dtype = qa.getattr("dtype")?;
                    // node.kwargs is immutable; must replace the whole dict
                    let old_kwargs = node.getattr("kwargs")?;
                    let new_kwargs = PyDict::new_bound(py);
                    for (k, v) in old_kwargs.downcast::<PyDict>()?.iter() {
                        new_kwargs.set_item(k, v)?;
                    }
                    new_kwargs.set_item("dtype", &output_dtype)?;
                    node.setattr("kwargs", new_kwargs)?;
                }
            }

            modified = true;
        }

        // Erase collected nodes
        for dead_node in &nodes_to_erase {
            let users = FxGraphModule::node_users(dead_node)?;
            if users.is_empty() {
                gm.erase_node(dead_node)?;
            }
        }

        if modified {
            gm.recompile()?;
        }

        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests are in Python (test_fold_compare.py).
}
