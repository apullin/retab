use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;
use crate::fx_utils;

/// FX-graph-direct decomposition of `aten.logit` into primitive ops.
///
/// Mirrors Python `DecomposeLogitPass`:
///   logit(x, eps=None) ->
///     if eps is not None:
///       x = clamp(x, eps, 1 - eps)
///     neg_x = mul(x, -1.0)
///     denom = add(neg_x, 1.0)       # (1 - x)
///     frac = reciprocal(denom)       # 1 / (1 - x)
///     log_input = mul(x, frac)       # x / (1 - x)
///     result = log(log_input)
///
/// Must run before fold_qdq so the decomposed ops get quantization handling.
pub struct FxDecomposeLogitPass;

fn is_logit(target: &str) -> bool {
    target.contains("logit")
}

impl FxPass for FxDecomposeLogitPass {
    fn name(&self) -> &str {
        "FxDecomposeLogit"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = FxGraphModule::debug();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_logit(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_logit(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_logit<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }

    let mut x = args[0].clone();

    // Extract eps from args[1] or kwargs
    let eps: Option<f64> = if args.len() > 1 && !args[1].is_none() {
        args[1].extract().ok()
    } else {
        let kwargs = node.getattr("kwargs")?;
        match kwargs.call_method1("get", ("eps",)) {
            Ok(v) if !v.is_none() => v.extract().ok(),
            _ => None,
        }
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeLogit] decomposing: {} (eps={:?})", name, eps);
    }

    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let reciprocal_op = resolve_aten_op(py, "reciprocal.default")?;
    let log_op = resolve_aten_op(py, "log.default")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    let input_shape = FxGraphModule::node_shape(&x).unwrap_or_else(|_| vec![1]);
    let builtins = py.import_bound("builtins")?;

    // Helper: create a scalar full node with given value
    let make_scalar = |val: f64| -> PyResult<Bound<'py, PyAny>> {
        let ones_shape = PyTuple::new_bound(py, vec![1i64; input_shape.len()].iter().copied());
        let float_val = builtins.call_method1("float", (val,))?;
        let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), float_val.as_ref()]);
        insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)
    };

    // If eps is not None: x = clamp(x, eps, 1 - eps)
    if let Some(eps_val) = eps {
        let clamp_op = resolve_aten_op(py, "clamp.default")?;
        let eps_py = builtins.call_method1("float", (eps_val,))?;
        let one_minus_eps = builtins.call_method1("float", (1.0 - eps_val,))?;
        let clamp_args =
            PyTuple::new_bound(py, &[x.as_ref(), eps_py.as_ref(), one_minus_eps.as_ref()]);
        x = insert_call_before(py, graph, node, &clamp_op, clamp_args.as_ref(), node)?;
    }

    // neg_x = mul(x, -1.0)
    let neg_one = make_scalar(-1.0)?;
    let neg_x_args = PyTuple::new_bound(py, &[x.as_ref(), neg_one.as_ref()]);
    let neg_x = insert_call_before(py, graph, node, &mul_op, neg_x_args.as_ref(), node)?;

    // denom = add(neg_x, 1.0)  => (1 - x)
    let one = make_scalar(1.0)?;
    let denom_args = PyTuple::new_bound(py, &[neg_x.as_ref(), one.as_ref()]);
    let denom = insert_call_before(py, graph, node, &add_op, denom_args.as_ref(), node)?;

    // frac = reciprocal(denom)  => 1 / (1 - x)
    let frac_args = PyTuple::new_bound(py, &[denom.as_ref()]);
    let frac = insert_call_before(py, graph, node, &reciprocal_op, frac_args.as_ref(), node)?;

    // log_input = mul(x, frac)  => x / (1 - x)
    let log_input_args = PyTuple::new_bound(py, &[x.as_ref(), frac.as_ref()]);
    let log_input = insert_call_before(py, graph, node, &mul_op, log_input_args.as_ref(), node)?;

    // result = log(log_input)
    let log_args = PyTuple::new_bound(py, &[log_input.as_ref()]);
    let result = insert_call_before(py, graph, node, &log_op, log_args.as_ref(), node)?;

    // Copy original node's meta onto the final result
    fx_utils::copy_meta(py, node, &result)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
