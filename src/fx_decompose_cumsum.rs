use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, propagate_qparams, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.cumsum` into chained
/// slice_copy + add ops.
///
/// cumsum(x, dim) is decomposed as:
///   result = slice(x, dim, 0, 1)
///   for i in 1..size[dim]:
///       prev = slice(result, dim, i-1, i)
///       curr = slice(x, dim, i, i+1)
///       elem = add(prev, curr)
///       result = cat([result, elem], dim)
pub struct FxDecomposeCumsumPass;

fn is_cumsum(target: &str) -> bool {
    target.contains("cumsum")
}

impl FxPass for FxDecomposeCumsumPass {
    fn name(&self) -> &str {
        "FxDecomposeCumsum"
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
            if is_cumsum(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_cumsum(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_cumsum<'py>(
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
    let rank = input_shape.len() as i64;
    if rank == 0 {
        return Ok(());
    }

    // Extract dim from args[1] (default 0)
    let dim_raw: i64 = if args.len() > 1 {
        args[1].extract().unwrap_or(0)
    } else {
        0
    };
    // Normalize negative dims
    let dim = ((dim_raw % rank) + rank) % rank;
    let dim_size = input_shape.get(dim as usize).copied().unwrap_or(0);

    if dim_size <= 1 {
        // Nothing to accumulate -- just replace with the input
        FxGraphModule::replace_all_uses_with(node, x)?;
        gm.erase_node(node)?;
        return Ok(());
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeCumsum] {} dim={} dim_size={} shape={:?}",
            name, dim, dim_size, input_shape
        );
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    // result = slice(x, dim, 0, 1)
    let dim_py = dim.into_py(py).into_bound(py);
    let zero_py = 0i64.into_py(py).into_bound(py);
    let one_py = 1i64.into_py(py).into_bound(py);
    let init_args = PyTuple::new_bound(
        py,
        &[x.as_ref(), dim_py.as_ref(), zero_py.as_ref(), one_py.as_ref()],
    );
    let mut result = insert_call_before(py, graph, node, &slice_op, init_args.as_ref(), node)?;

    for i in 1..dim_size {
        let i_py = i.into_py(py).into_bound(py);
        let i_plus_1_py = (i + 1).into_py(py).into_bound(py);
        let i_minus_1_py = (i - 1).into_py(py).into_bound(py);

        // prev = slice(result, dim, i-1, i)
        let prev_args = PyTuple::new_bound(
            py,
            &[result.as_ref(), dim_py.as_ref(), i_minus_1_py.as_ref(), i_py.as_ref()],
        );
        let prev = insert_call_before(py, graph, node, &slice_op, prev_args.as_ref(), node)?;

        // curr = slice(x, dim, i, i+1)
        let curr_args = PyTuple::new_bound(
            py,
            &[x.as_ref(), dim_py.as_ref(), i_py.as_ref(), i_plus_1_py.as_ref()],
        );
        let curr = insert_call_before(py, graph, node, &slice_op, curr_args.as_ref(), node)?;

        // elem = add(prev, curr)
        let add_args = PyTuple::new_bound(py, &[prev.as_ref(), curr.as_ref()]);
        let elem = insert_call_before(py, graph, node, &add_op, add_args.as_ref(), node)?;
        // ADD needs input_qparams for BOTH inputs: {0: qp, 1: qp}
        {
            let src_meta = node.getattr("meta")?;
            let dst_meta = elem.getattr("meta")?;
            if let Ok(iqp) = src_meta.call_method1("get", ("input_qparams",)) {
                if !iqp.is_none() {
                    // Get qp[0] and set it for both inputs
                    if let Ok(qp0) = iqp.call_method1("get", (0,)) {
                        if !qp0.is_none() {
                            let new_iqp = pyo3::types::PyDict::new_bound(py);
                            new_iqp.set_item(0, &qp0)?;
                            new_iqp.set_item(1, &qp0)?;
                            dst_meta.set_item("input_qparams", new_iqp)?;
                        }
                    }
                }
            }
            if let Ok(oqp) = src_meta.call_method1("get", ("output_qparams",)) {
                if !oqp.is_none() {
                    dst_meta.set_item("output_qparams", &oqp)?;
                }
            }
        }

        // result = cat([result, elem], dim)
        let cat_list = pyo3::types::PyList::new_bound(py, &[result.as_ref(), elem.as_ref()]);
        let cat_args = PyTuple::new_bound(py, &[cat_list.as_ref(), dim_py.as_ref()]);
        result = insert_call_before(py, graph, node, &cat_op, cat_args.as_ref(), node)?;
        propagate_qparams(node, &result)?;
    }

    // Copy original node's meta onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
