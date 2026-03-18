use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// Quantization parameters extracted from a DQ or Q node.
struct QParams {
    scale: f64,
    zero_point: i64,
    qmin: i64,
    qmax: i64,
    dtype_str: String,
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
fn extract_dq_per_tensor_qparams(args: &[Bound<'_, PyAny>]) -> PyResult<Option<QParams>> {
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
    let dtype_str = args
        .get(5)
        .and_then(|a| a.str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "torch.int8".to_string());
    Ok(Some(QParams {
        scale,
        zero_point,
        qmin,
        qmax,
        dtype_str,
    }))
}

/// Extract QParams from a dequantize_per_channel node.
/// Args layout: [input, scale_tensor, zp_tensor, axis, quant_min, quant_max, dtype]
/// We take the first element of the scale/zp tensors as a representative.
fn extract_dq_per_channel_qparams(
    _py: Python<'_>,
    args: &[Bound<'_, PyAny>],
) -> PyResult<Option<QParams>> {
    if args.len() < 2 {
        return Ok(None);
    }
    // scale might be a tensor or a float
    let scale: f64 = if let Ok(v) = args[1].extract::<f64>() {
        v
    } else {
        // Try indexing [0] on a tensor / list
        match args[1].call_method1("__getitem__", (0_i64,)) {
            Ok(v) => {
                // Could be a tensor scalar — call .item()
                v.call_method0("item")
                    .and_then(|i| i.extract())
                    .or_else(|_| v.extract())
                    .unwrap_or(1.0)
            }
            Err(_) => return Ok(None),
        }
    };
    let zero_point: i64 = if let Ok(v) = args.get(2).map(|a| a.extract::<i64>()).unwrap_or(Ok(0))
    {
        v
    } else {
        // tensor / list — take [0]
        args.get(2)
            .and_then(|a| {
                a.call_method1("__getitem__", (0_i64,))
                    .ok()
                    .and_then(|v| {
                        v.call_method0("item")
                            .and_then(|i| i.extract().map_err(Into::into))
                            .or_else(|_: PyErr| v.extract())
                            .ok()
                    })
            })
            .unwrap_or(0)
    };
    // axis at index 3 (ignored)
    let qmin: i64 = args.get(4).and_then(|a| a.extract().ok()).unwrap_or(-128);
    let qmax: i64 = args.get(5).and_then(|a| a.extract().ok()).unwrap_or(127);
    let dtype_str = args
        .get(6)
        .and_then(|a| a.str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "torch.int8".to_string());
    Ok(Some(QParams {
        scale,
        zero_point,
        qmin,
        qmax,
        dtype_str,
    }))
}

/// Extract QParams from a quantize_per_tensor node.
/// Same arg layout as dequantize_per_tensor.
fn extract_q_per_tensor_qparams(args: &[Bound<'_, PyAny>]) -> PyResult<Option<QParams>> {
    extract_dq_per_tensor_qparams(args)
}

// ---------------------------------------------------------------------------
// Helper: build a Python dict representing qparams
// ---------------------------------------------------------------------------

fn qparams_to_py_dict<'py>(py: Python<'py>, qp: &QParams) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("scale", qp.scale)?;
    d.set_item("zero_point", qp.zero_point)?;
    d.set_item("qmin", qp.qmin)?;
    d.set_item("qmax", qp.qmax)?;
    d.set_item("dtype", &qp.dtype_str)?;
    Ok(d)
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
        struct DqInfo<'py> {
            node: Bound<'py, PyAny>,
            #[allow(dead_code)]
            qparams: QParams,
        }
        let mut dq_nodes: Vec<DqInfo> = Vec::new();

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
                extract_dq_per_channel_qparams(py, &args)?
            } else {
                extract_dq_per_tensor_qparams(&args)?
            };

            let qp = match qp {
                Some(qp) => qp,
                None => continue,
            };

            // For each user of this DQ node, annotate input_qparams
            let users = FxGraphModule::node_users(node)?;
            let qp_dict = qparams_to_py_dict(py, &qp)?;

            for user in &users {
                let positions = find_arg_positions(user, node)?;
                // Get or create node.meta["input_qparams"] (a Python dict: pos -> qparams)
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
                    // Only set if not already present (use dict.get to avoid KeyError)
                    let existing = input_qp.call_method1("get", (pos,))?;
                    if existing.is_none() {
                        input_qp.set_item(pos, &qp_dict)?;
                    }
                }
            }

            dq_nodes.push(DqInfo {
                node: node.clone(),
                qparams: qp,
            });
        }

        // ------------------------------------------------------------------
        // Pass 2: Annotate output_qparams on Q inputs
        // ------------------------------------------------------------------
        struct QInfo<'py> {
            node: Bound<'py, PyAny>,
            #[allow(dead_code)]
            qparams: QParams,
        }
        let mut q_nodes: Vec<QInfo> = Vec::new();

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
            let qp = match extract_q_per_tensor_qparams(&args)? {
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
                    let qp_dict = qparams_to_py_dict(py, &qp)?;
                    // Position 0 for the single output
                    let existing = output_qp.call_method1("get", (0,))?;
                    if existing.is_none() {
                        output_qp.set_item(0, &qp_dict)?;
                    }
                }
            }

            q_nodes.push(QInfo {
                node: node.clone(),
                qparams: qp,
            });
        }

        // ------------------------------------------------------------------
        // Pass 3: Bypass DQ and Q nodes
        // ------------------------------------------------------------------
        for dq in &dq_nodes {
            let args = FxGraphModule::node_args(&dq.node)?;
            if let Some(input_node) = args.first() {
                if FxGraphModule::is_node(input_node) {
                    FxGraphModule::replace_all_uses_with(&dq.node, input_node)?;
                    gm.erase_node(&dq.node)?;
                    modified = true;
                }
            }
        }

        for q in &q_nodes {
            // Q node's output feeds into its users.  Replace uses of Q with
            // the Q node's own output-side user?  No — Q is a passthrough
            // from the quantized domain.  We replace uses of Q with Q's input
            // (args[0]).  But wait, at this point the DQ nodes are already
            // erased, so Q's input (which was a DQ output) has been rewired.
            // Actually, the Q node's users should now point to Q's first arg.
            let args = FxGraphModule::node_args(&q.node)?;
            if let Some(input_node) = args.first() {
                if FxGraphModule::is_node(input_node) {
                    FxGraphModule::replace_all_uses_with(&q.node, input_node)?;
                    gm.erase_node(&q.node)?;
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
