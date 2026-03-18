use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// Quantization parameters extracted from a DQ or Q node.
struct QParams<'py> {
    scale: ScaleValue<'py>,
    zero_point: ZpValue<'py>,
    qmin: i64,
    qmax: i64,
    dtype: Bound<'py, PyAny>, // Python torch.dtype object
    axis: i64,
    per_channel: bool,
}

/// Scale can be a scalar float (per-tensor) or a list of floats (per-channel).
enum ScaleValue<'py> {
    Scalar(f64),
    Tensor(Bound<'py, PyAny>), // Python list or tensor — kept as-is for QuantArgs
}

/// Zero-point can be a scalar int (per-tensor) or a list of ints (per-channel).
enum ZpValue<'py> {
    Scalar(i64),
    Tensor(Bound<'py, PyAny>), // Python list or tensor — kept as-is for QuantArgs
}

/// FX-graph-direct implementation of `FoldAndAnnotateQParamsPass`.
///
/// This pass operates on the *live* Python FX graph via PyO3.  It:
///   1. Iterates nodes looking for `quantize_per_tensor` / `dequantize_per_tensor`
///      / `dequantize_per_channel` targets.
///   2. Extracts scale / zero-point / dtype from the Q/DQ node args.
///   3. Stores qparams in `node.meta["input_qparams"]` and `node.meta["output_qparams"]`
///      of the adjacent non-Q/DQ nodes.
///   4. Calls `node.replace_all_uses_with()` and `graph.erase_node()` to remove
///      the Q/DQ nodes.
///
/// The heavy iteration and pattern matching happen in Rust; only the actual
/// graph mutations cross the PyO3 boundary.
pub struct FxFoldAndAnnotateQParamsPass;

// ---------------------------------------------------------------------------
// Target-name matching helpers
// ---------------------------------------------------------------------------

fn is_dequantize_per_tensor(target: &str) -> bool {
    target.contains("dequantize_per_tensor")
}

fn is_dequantize_per_channel(target: &str) -> bool {
    target.contains("dequantize_per_channel")
}

fn is_dq(target: &str) -> bool {
    is_dequantize_per_tensor(target) || is_dequantize_per_channel(target)
}

fn is_quantize_per_tensor(target: &str) -> bool {
    target.contains("quantize_per_tensor")
}

// ---------------------------------------------------------------------------
// QParam extraction
// ---------------------------------------------------------------------------

/// Extract QParams from a dequantize_per_tensor node.
/// Args layout: [input, scale, zero_point, quant_min, quant_max, dtype]
fn extract_dq_per_tensor_qparams<'py>(
    py: Python<'py>,
    args: &[Bound<'py, PyAny>],
) -> PyResult<Option<QParams<'py>>> {
    if args.len() < 2 {
        return Ok(None);
    }
    let scale: f64 = match args[1].extract() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let zero_point: i64 = args.get(2).and_then(|a| a.extract().ok()).unwrap_or(0);
    let qmin: i64 = args.get(3).and_then(|a| a.extract().ok()).unwrap_or(-128);
    let qmax: i64 = args.get(4).and_then(|a| a.extract().ok()).unwrap_or(127);
    let dtype = match args.get(5) {
        Some(d) => d.clone(),
        None => py.eval_bound("__import__('torch').int8", None, None)?,
    };
    Ok(Some(QParams {
        scale: ScaleValue::Scalar(scale),
        zero_point: ZpValue::Scalar(zero_point),
        qmin,
        qmax,
        dtype,
        axis: 0,
        per_channel: false,
    }))
}

/// Extract QParams from a dequantize_per_channel node.
/// Args layout: [input, scale_node, zp_node, axis, quant_min, quant_max, dtype]
///
/// scale and zp may be FX Node references (pointing to parameter tensors) or
/// direct tensor/list values. For per-channel QuantArgs we need Python lists,
/// so we call `.tolist()` on tensors.
fn extract_dq_per_channel_qparams<'py>(
    py: Python<'py>,
    args: &[Bound<'py, PyAny>],
    gm: &FxGraphModule<'py>,
) -> PyResult<Option<QParams<'py>>> {
    if args.len() < 2 {
        return Ok(None);
    }

    // Resolve a value that may be an FX Node referencing a parameter tensor.
    // If it's a Node, look up the tensor via graph_module.get_attr(node.target).
    let resolve_param = |arg: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if FxGraphModule::is_node(arg) {
            // It's an FX Node — get the underlying tensor from graph_module attrs
            let target: String = arg.getattr("target")?.extract()?;
            match gm.gm_ref().getattr(target.as_str()) {
                Ok(tensor) => Ok(tensor),
                Err(_) => Ok(arg.clone()), // fall back to the node itself
            }
        } else {
            Ok(arg.clone())
        }
    };

    // Convert a tensor/scalar to a Python list for QuantArgs.
    let to_list = |val: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        // Already a list
        if val.is_instance_of::<pyo3::types::PyList>() {
            return Ok(val.clone());
        }
        // Tensor — call .tolist()
        if let Ok(list_val) = val.call_method0("tolist") {
            return Ok(list_val);
        }
        // Scalar — wrap in a list
        Ok(val.clone())
    };

    let scale_raw = resolve_param(&args[1])?;
    let scale_list = to_list(&scale_raw)?;

    let zp_raw = args
        .get(2)
        .map(|a| resolve_param(a))
        .transpose()?
        .unwrap_or_else(|| py.eval_bound("[0]", None, None).unwrap());
    let zp_list = to_list(&zp_raw)?;

    let axis: i64 = args.get(3).and_then(|a| a.extract().ok()).unwrap_or(0);
    let qmin: i64 = args.get(4).and_then(|a| a.extract().ok()).unwrap_or(-128);
    let qmax: i64 = args.get(5).and_then(|a| a.extract().ok()).unwrap_or(127);
    let dtype = match args.get(6) {
        Some(d) => d.clone(),
        None => py.eval_bound("__import__('torch').int8", None, None)?,
    };

    Ok(Some(QParams {
        scale: ScaleValue::Tensor(scale_list),
        zero_point: ZpValue::Tensor(zp_list),
        qmin,
        qmax,
        dtype,
        axis,
        per_channel: true,
    }))
}

/// Extract QParams from a quantize_per_tensor node.
/// Same arg layout as dequantize_per_tensor.
fn extract_q_per_tensor_qparams<'py>(
    py: Python<'py>,
    args: &[Bound<'py, PyAny>],
) -> PyResult<Option<QParams<'py>>> {
    extract_dq_per_tensor_qparams(py, args)
}

// ---------------------------------------------------------------------------
// Helper: build a Python QuantArgs namedtuple from our Rust QParams
// ---------------------------------------------------------------------------

/// Import and cache the QuantArgs class from the ARM backend.
fn get_quant_args_cls<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")
}

/// Create a Python `QuantArgs(scale, zp, qmin, qmax, dtype, axis, per_channel)` namedtuple.
fn qparams_to_quant_args<'py>(
    py: Python<'py>,
    qp: &QParams<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let cls = get_quant_args_cls(py)?;

    let scale_arg: Bound<'py, PyAny> = match &qp.scale {
        ScaleValue::Scalar(v) => v.into_py(py).into_bound(py),
        ScaleValue::Tensor(list) => list.clone(),
    };
    let zp_arg: Bound<'py, PyAny> = match &qp.zero_point {
        ZpValue::Scalar(v) => v.into_py(py).into_bound(py),
        ZpValue::Tensor(list) => list.clone(),
    };

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("scale", &scale_arg)?;
    kwargs.set_item("zp", &zp_arg)?;
    kwargs.set_item("qmin", qp.qmin)?;
    kwargs.set_item("qmax", qp.qmax)?;
    kwargs.set_item("dtype", &qp.dtype)?;
    kwargs.set_item("axis", qp.axis)?;
    kwargs.set_item("per_channel", qp.per_channel)?;

    // QuantArgs(**kwargs) — NamedTuple supports keyword construction
    cls.call((), Some(&kwargs))
}

/// Find which arg positions in `user_node.args` reference `target_node`
/// (by Python identity, i.e. `is`).
fn find_arg_positions(
    user_node: &Bound<'_, PyAny>,
    target_node: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let args = FxGraphModule::node_args(user_node)?;
    let mut positions = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if arg.is(target_node) {
            positions.push(i);
        }
    }
    Ok(positions)
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

        // Snapshot all nodes up-front (Python list is live; we need a stable vec).
        let all_nodes = gm.nodes()?;

        // ------------------------------------------------------------------
        // Pass 1: Annotate input_qparams on DQ users
        // ------------------------------------------------------------------
        let mut dq_nodes: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_dq(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let qp = if is_dequantize_per_channel(&target) {
                extract_dq_per_channel_qparams(py, &args, gm)?
            } else {
                extract_dq_per_tensor_qparams(py, &args)?
            };

            let qp = match qp {
                Some(qp) => qp,
                None => continue,
            };

            // Build the QuantArgs namedtuple once for this DQ node
            let quant_args = qparams_to_quant_args(py, &qp)?;

            // For each user of this DQ node, annotate input_qparams
            let users = FxGraphModule::node_users(node)?;

            for user in &users {
                let positions = find_arg_positions(user, node)?;
                // Get or create node.meta["input_qparams"] (a Python dict: pos -> QuantArgs)
                let meta = user.getattr("meta")?;
                let input_qp = match meta.call_method1("get", ("input_qparams",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => {
                        let d = PyDict::new_bound(py);
                        meta.set_item("input_qparams", &d)?;
                        d.into_any()
                    }
                };
                for pos in positions {
                    let existing = input_qp.call_method1("get", (pos,))?;
                    if existing.is_none() {
                        input_qp.set_item(pos, &quant_args)?;
                    }
                }
            }

            dq_nodes.push(node.clone());
        }

        // ------------------------------------------------------------------
        // Pass 2: Annotate output_qparams on Q inputs
        // ------------------------------------------------------------------
        let mut q_nodes: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_quantize_per_tensor(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let qp = match extract_q_per_tensor_qparams(py, &args)? {
                Some(qp) => qp,
                None => continue,
            };

            // The input to the Q node is args[0]
            if let Some(input_node) = args.first() {
                if FxGraphModule::is_node(input_node) {
                    let meta = input_node.getattr("meta")?;
                    let output_qp = match meta.call_method1("get", ("output_qparams",)) {
                        Ok(v) if !v.is_none() => v,
                        _ => {
                            let d = PyDict::new_bound(py);
                            meta.set_item("output_qparams", &d)?;
                            d.into_any()
                        }
                    };
                    let quant_args = qparams_to_quant_args(py, &qp)?;
                    let existing = output_qp.call_method1("get", (0,))?;
                    if existing.is_none() {
                        output_qp.set_item(0, &quant_args)?;
                    }
                }
            }

            q_nodes.push(node.clone());
        }

        // ------------------------------------------------------------------
        // Pass 3: Bypass DQ and Q nodes
        // ------------------------------------------------------------------
        for dq_node in &dq_nodes {
            let args = FxGraphModule::node_args(dq_node)?;
            if let Some(input_node) = args.first() {
                if FxGraphModule::is_node(input_node) {
                    FxGraphModule::replace_all_uses_with(dq_node, input_node)?;
                    gm.erase_node(dq_node)?;
                    modified = true;
                }
            }
        }

        for q_node in &q_nodes {
            let args = FxGraphModule::node_args(q_node)?;
            if let Some(input_node) = args.first() {
                if FxGraphModule::is_node(input_node) {
                    FxGraphModule::replace_all_uses_with(q_node, input_node)?;
                    gm.erase_node(q_node)?;
                    modified = true;
                }
            }
        }

        // ------------------------------------------------------------------
        // Recompile the graph module so the forward() is updated.
        // ------------------------------------------------------------------
        if modified {
            gm.recompile()?;
        }

        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests will be in Python; Rust unit tests are tricky
    // since we need a live Python FX graph.
}
