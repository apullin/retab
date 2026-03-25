use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.mean.dim` into
/// `sum(keepdim=True) + mul(1/count) + view_copy(output_shape)`.
///
/// Always uses keepdim=True for the sum to keep tensors at input rank,
/// then reshapes to the target output shape.  For 4D inputs with spatial
/// dim reductions, uses avg_pool2d (matching Python DecomposeMeanDimPass).
pub struct FxDecomposeMeanPass;

fn is_mean_dim(target: &str) -> bool {
    target.contains("mean.dim") || target.contains("mean.default")
}

impl FxPass for FxDecomposeMeanPass {
    fn name(&self) -> &str {
        "FxDecomposeMean"
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
            if is_mean_dim(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_mean(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_mean<'py>(
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
    let x = &args[0];

    let input_shape = FxGraphModule::node_shape(x).unwrap_or_default();
    let input_rank = input_shape.len() as i64;
    let output_shape = FxGraphModule::node_shape(node).unwrap_or_else(|_| vec![1]);

    // Extract dims
    let dims: Vec<i64> = if args.len() > 1 {
        let dims_obj = &args[1];
        if dims_obj.is_none() {
            (0..input_rank).collect()
        } else {
            match dims_obj.extract::<Vec<i64>>() {
                Ok(d) => d,
                Err(_) => match dims_obj.extract::<i64>() {
                    Ok(d) => vec![d],
                    Err(_) => (0..input_rank).collect(),
                },
            }
        }
    } else {
        (0..input_rank).collect()
    };

    // Normalize negative dims
    let dims: Vec<i64> = dims
        .into_iter()
        .map(|d| if d < 0 { d + input_rank } else { d })
        .collect();

    // Filter out dims where input_shape[dim] == 1 (matches Python)
    let dims: Vec<i64> = dims
        .into_iter()
        .filter(|&d| input_shape.get(d as usize).copied().unwrap_or(1) != 1)
        .collect();

    if dims.is_empty() {
        FxGraphModule::replace_all_uses_with(node, x)?;
        gm.erase_node(node)?;
        return Ok(());
    }

    let keepdim: bool = if args.len() > 2 {
        args[2].extract().unwrap_or(false)
    } else {
        false
    };

    // Compute count = product of dim sizes
    let count: f64 = dims
        .iter()
        .map(|&d| input_shape.get(d as usize).copied().unwrap_or(1) as f64)
        .product();
    if count == 0.0 {
        return Ok(());
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeMean] {} dims={:?} keepdim={} count={} shape={:?}->{:?}",
            name, dims, keepdim, count, input_shape, output_shape
        );
    }

    let sum_op = resolve_aten_op(py, "sum.dim_IntList")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // sum(x, dims, keepdim=True) — always keepdim=True to preserve rank
    let dims_py = pyo3::types::PyList::new_bound(py, dims.iter().copied());
    let keepdim_true = true.into_py(py).into_bound(py);
    let sum_args = PyTuple::new_bound(
        py,
        &[x.as_ref(), dims_py.as_ref(), keepdim_true.as_ref()],
    );
    let sum_node = insert_call_before(py, graph, node, &sum_op, sum_args.as_ref(), node)?;

    // full([1]*input_rank, 1/count)
    let ones_shape = PyTuple::new_bound(py, vec![1i64; input_shape.len()].iter().copied());
    let builtins = py.import_bound("builtins")?;
    let inv_count = builtins.call_method1("float", (1.0 / count,))?;
    let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), inv_count.as_ref()]);
    let count_node = insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)?;

    // mul(sum, 1/count)
    let mul_args = PyTuple::new_bound(py, &[sum_node.as_ref(), count_node.as_ref()]);
    let mut result = insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?;

    // If keepdim=False, reshape to target output shape
    if !keepdim {
        let view_op = resolve_aten_op(py, "view_copy.default")?;
        let shape_list = pyo3::types::PyList::new_bound(py, output_shape.iter().copied());
        let view_args = PyTuple::new_bound(py, &[result.as_ref(), shape_list.as_ref()]);
        result = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), node)?;
    }

    // Copy original node's meta onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
