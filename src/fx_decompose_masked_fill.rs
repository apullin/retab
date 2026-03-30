use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

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
    matches: |target| target.contains("masked_fill"),
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

    let full_op = resolve_aten_op(py, "full.default")?;
    let where_op = resolve_aten_op(py, "where.self")?;

    // Get the input shape for the full constant
    let input_shape = FxGraphModule::node_shape(x).unwrap_or_else(|_| vec![1]);
    let shape_tuple = PyTuple::new_bound(py, input_shape.iter().copied());

    // Check whether the graph is quantized: look for output_qparams on the node.
    let is_quantized = fx_utils::is_quantized(node);

    // Determine the fill value and dtype for the full node.
    let qparams_0 = if is_quantized {
        fx_utils::get_output_qparam(node)
    } else {
        None
    };

    let (fill_value, full_kwargs): (Bound<'py, PyAny>, Option<Bound<'py, PyDict>>) =
        if let Some(qparams) = &qparams_0 {
            let scale: f64 = qparams.getattr("scale")?.extract().unwrap_or(1.0);
            let zp: i64 = qparams
                .getattr("zp")
                .or_else(|_| qparams.getattr("zero_point"))
                .and_then(|v| v.extract())
                .unwrap_or(0);
            let qmin: i64 = qparams
                .getattr("qmin")
                .and_then(|v| v.extract())
                .unwrap_or(-128);
            let qmax: i64 = qparams
                .getattr("qmax")
                .and_then(|v| v.extract())
                .unwrap_or(127);
            let output_dtype = qparams.getattr("dtype")?;

            let float_val: f64 = value.extract().unwrap_or(0.0);
            let quantized_val = ((float_val / scale) + zp as f64).round() as i64;
            let clamped = quantized_val.max(qmin).min(qmax);

            if debug {
                eprintln!(
                    "[DecomposeMaskedFill] quantizing fill value: {} -> {} (scale={}, zp={})",
                    float_val, clamped, scale, zp
                );
            }

            let builtins = py.import_bound("builtins")?;
            let fill_py = builtins.call_method1("int", (clamped,))?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", output_dtype)?;
            (fill_py, Some(kwargs))
        } else {
            // FP path: use the scalar value directly.
            (value.clone(), None)
        };

    // Create full(shape, fill_value, dtype=...)
    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;

    let full_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), fill_value.as_ref()]);
    let full_kwargs_dict = full_kwargs.unwrap_or_else(|| PyDict::new_bound(py));
    let fill_node =
        graph.call_method1("call_function", (&full_op, &full_args, &full_kwargs_dict))?;

    // Copy meta from original node for the full node
    fx_utils::copy_meta(py, node, &fill_node)?;
    // Clear qparams for the intermediate full node
    let full_meta = fill_node.getattr("meta")?;
    let _ = full_meta.call_method1("__setitem__", ("input_qparams", PyDict::new_bound(py)));
    let _ = full_meta.call_method1("__setitem__", ("output_qparams", PyDict::new_bound(py)));
    fx_utils::refresh_fake_tensor(py, &fill_node)?;

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
