use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.addmm` into mm + mul + add.
///
/// Mirrors Python `DecomposeAddmmPass`:
///   addmm(input, mat1, mat2, beta=1.0, alpha=1.0) ->
///     mm_result = mm(mat1, mat2)
///     if alpha != 1.0: mm_result = mul(mm_result, full(alpha))
///     if beta != 1.0: input = mul(input, full(beta))
///     result = add(mm_result, input)
///
/// Must run early in the pipeline so downstream passes (rescale, memory
/// format) process the decomposed primitive ops.
pub struct FxDecomposeAddmmPass;

fn is_addmm(target: &str) -> bool {
    target.contains("addmm")
}

impl FxPass for FxDecomposeAddmmPass {
    fn name(&self) -> &str {
        "FxDecomposeAddmm"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_addmm(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_addmm(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

/// Decompose a single addmm node.
///
/// addmm(input, mat1, mat2, beta=1.0, alpha=1.0) ->
///   mm_result = mm(mat1, mat2)
///   scaled_mm = mul(mm_result, alpha) if alpha != 1.0 else mm_result
///   scaled_input = mul(input, beta) if beta != 1.0 else input
///   result = add(scaled_mm, scaled_input)
fn decompose_addmm<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 3 {
        return Ok(());
    }

    let input = &args[0];
    let mat1 = &args[1];
    let mat2 = &args[2];

    // Extract beta and alpha from kwargs or positional args
    let kwargs = node.getattr("kwargs")?;

    let beta: f64 = if args.len() > 3 {
        args[3].extract().unwrap_or(1.0)
    } else {
        match kwargs.call_method1("get", ("beta", 1.0f64)) {
            Ok(v) => v.extract().unwrap_or(1.0),
            Err(_) => 1.0,
        }
    };

    let alpha: f64 = if args.len() > 4 {
        args[4].extract().unwrap_or(1.0)
    } else {
        match kwargs.call_method1("get", ("alpha", 1.0f64)) {
            Ok(v) => v.extract().unwrap_or(1.0),
            Err(_) => 1.0,
        }
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeAddmm] decomposing: {} (beta={}, alpha={})",
            name, beta, alpha
        );
    }

    let mm_op = resolve_aten_op(py, "mm.default")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // mm_result = mm(mat1, mat2)
    let mm_args = PyTuple::new_bound(py, &[mat1.as_ref(), mat2.as_ref()]);
    let mm_result = insert_call_before(py, graph, node, &mm_op, mm_args.as_ref(), node)?;

    // Scale mm_result by alpha if alpha != 1.0
    let scaled_mm = if (alpha - 1.0).abs() > 1e-10 {
        let input_shape = FxGraphModule::node_shape(&mm_result).unwrap_or_else(|_| vec![1]);
        let ones_shape = PyTuple::new_bound(py, vec![1i64; input_shape.len()].iter().copied());
        let builtins = py.import_bound("builtins")?;
        let alpha_val = builtins.call_method1("float", (alpha,))?;
        let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), alpha_val.as_ref()]);
        let alpha_node =
            insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)?;
        let mul_args = PyTuple::new_bound(py, &[mm_result.as_ref(), alpha_node.as_ref()]);
        insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?
    } else {
        mm_result
    };

    // Scale input by beta if beta != 1.0
    let scaled_input = if (beta - 1.0).abs() > 1e-10 {
        let input_shape = FxGraphModule::node_shape(input).unwrap_or_else(|_| vec![1]);
        let ones_shape = PyTuple::new_bound(py, vec![1i64; input_shape.len()].iter().copied());
        let builtins = py.import_bound("builtins")?;
        let beta_val = builtins.call_method1("float", (beta,))?;
        let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), beta_val.as_ref()]);
        let beta_node =
            insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)?;
        let mul_args = PyTuple::new_bound(py, &[input.as_ref(), beta_node.as_ref()]);
        insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?
    } else {
        input.clone()
    };

    // result = add(scaled_mm, scaled_input)
    let add_args = PyTuple::new_bound(py, &[scaled_mm.as_ref(), scaled_input.as_ref()]);
    let result = insert_call_before(py, graph, node, &add_op, add_args.as_ref(), node)?;

    // Copy original node's meta onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
