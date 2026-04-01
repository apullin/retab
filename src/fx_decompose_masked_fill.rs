use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// Late-stage decomposition of `masked_fill` into `where(mask, full(value), x)`.
    ///
    /// Runs AFTER `FxFoldAndAnnotateQParamsPass` so that:
    ///   - The `masked_fill` node already has `input_qparams` / `output_qparams`.
    ///   - The fill value scalar can be quantized to match the tensor data type.
    ///   - The created `full` constant is quantized at the same scale/zp as `x`.
    ///
    /// For non-quantized (FP) graphs, the pass simply produces float `full` + `where`.
    FxDecomposeMaskedFillPass, "FxDecomposeMaskedFill",
    matches: |target, _node| target.contains("masked_fill"),
    transform: |py, graph, gm, node, debug| decompose_masked_fill(py, &graph, gm, node, debug)
);

fn decompose_masked_fill<'py>(
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
    let x = &args[0];
    let mask = &args[1];
    let value = &args[2]; // scalar

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeMaskedFill] decomposing masked_fill: {}", name);
    }

    let full_like_op = resolve_aten_op(py, "full_like.default")?;
    let where_op = resolve_aten_op(py, "where.self")?;

    // Create full(shape, fill_value, dtype=...)
    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;

    let full_like_args = PyTuple::new_bound(py, &[x.as_ref(), value.as_ref()]);
    let fill_node = graph.call_method1("call_function", (&full_like_op, &full_like_args))?;
    fx_utils::copy_meta_and_refresh(py, node, &fill_node)?;

    // Create where(mask, fill_value, x) -- True positions get fill_value
    let where_args = PyTuple::new_bound(py, &[mask.as_ref(), fill_node.as_ref(), x.as_ref()]);
    let result_node = graph.call_method1("call_function", (&where_op, &where_args))?;

    // Copy the original node's full meta (including qparams) to the where node
    fx_utils::copy_meta_and_refresh(py, node, &result_node)?;

    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;

    FxGraphModule::replace_all_uses_with(node, &result_node)?;
    gm.erase_node(node)?;
    Ok(())
}
