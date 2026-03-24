use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.mean.dim` into `sum + mul(1/count)`.
///
/// This must run before `FxConvertSumPass` (which rewrites sum to chained
/// REDUCE_SUM ops) and before `FxToTosaMemoryFormatPass` (which inserts
/// NCHW/NHWC transposes).  By decomposing early, the downstream passes
/// handle the primitive ops correctly — in particular, rank changes from
/// non-keepdim mean reductions no longer confuse the memory format pass.
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

/// mean(x, dims, keepdim) -> mul(sum(x, dims, keepdim), full(1/count))
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

    // Extract dims: args[1] may be a list/tuple of ints, a single int, or None.
    let input_shape = FxGraphModule::node_shape(x).unwrap_or_default();
    let input_rank = input_shape.len() as i64;

    let dims: Vec<i64> = if args.len() > 1 {
        let dims_obj = &args[1];
        if dims_obj.is_none() {
            // None means reduce all dims
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
        // mean.default: reduce all dims
        (0..input_rank).collect()
    };

    // Normalize negative dims
    let dims: Vec<i64> = dims
        .into_iter()
        .map(|d| if d < 0 { d + input_rank } else { d })
        .collect();

    // Extract keepdim (default False)
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
            "[DecomposeMean] decomposing mean: {} (dims={:?}, keepdim={}, count={})",
            name, dims, keepdim, count
        );
    }

    let sum_op = resolve_aten_op(py, "sum.dim_IntList")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // Create sum(x, dims, keepdim)
    let dims_py = pyo3::types::PyList::new_bound(py, dims.iter().copied());
    let keepdim_py = keepdim.into_py(py).into_bound(py);
    let sum_args = PyTuple::new_bound(
        py,
        &[x.as_ref(), dims_py.as_ref(), keepdim_py.as_ref()],
    );
    let sum_node = insert_call_before(py, graph, node, &sum_op, sum_args.as_ref(), node)?;

    // Create full([1]*output_rank, 1.0/count)
    let output_shape = FxGraphModule::node_shape(node).unwrap_or_else(|_| vec![1]);
    let out_rank = output_shape.len();
    let ones_shape = PyTuple::new_bound(py, vec![1i64; out_rank].iter().copied());
    let inv_count_val = {
        let builtins = py.import_bound("builtins")?;
        builtins.call_method1("float", (1.0 / count,))?
    };
    let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), inv_count_val.as_ref()]);
    let count_node = insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)?;

    // Create mul(sum, count)
    let mul_args = PyTuple::new_bound(py, &[sum_node.as_ref(), count_node.as_ref()]);
    let result = insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?;

    // Copy original node's meta onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
