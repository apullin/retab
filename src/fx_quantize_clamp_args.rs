use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct implementation of `QuantizeClampArgumentsPass`.
    ///
    /// After fold_qdq annotates output_qparams on clamp nodes, this pass quantizes
    /// the min/max scalar arguments to match the output quantization scale. Without
    /// this, clamp would operate on float min/max values while the actual tensor
    /// data is quantized integers, causing incorrect clipping.
    ///
    /// For each `aten.clamp.default` node that has output_qparams:
    ///   - Reads output_qparams[first] to get the QuantArgs
    ///   - Calls qargs.quantize_value(min_val) and qargs.quantize_value(max_val)
    ///   - Updates args[1] and args[2] in-place with the quantized values
    FxQuantizeClampArgsPass, "FxQuantizeClampArgs",
    matches: |target, _node| target.contains("clamp.default"),
    transform: |py, graph, gm, node, debug| quantize_clamp_args(py, node)
);

fn quantize_clamp_args<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // Check for output_qparams in meta.
    let meta = node.getattr("meta")?;
    let output_qparams = match meta.call_method1("get", ("output_qparams",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(()),
    };

    // output_qparams is a dict[int, QuantArgs]; grab the first value.
    let values = match output_qparams.call_method0("values") {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let values_iter = match values.iter() {
        Ok(it) => it,
        Err(_) => return Ok(()),
    };
    let qargs = match values_iter.into_iter().next() {
        Some(Ok(v)) => v,
        _ => return Ok(()),
    };

    let args = FxGraphModule::node_args(node)?;
    // args: [input_node, min_val, max_val]
    // min_val is args[1], max_val is args[2] (both optional/may be None)

    let mut new_args: Vec<Bound<'_, PyAny>> = args.clone();
    let mut did_update = false;

    // Quantize min_val (args[1]) if present and not None
    if let Some(min_arg) = args.get(1) {
        if !min_arg.is_none() {
            let quantized = qargs.call_method1("quantize_value", (min_arg,))?;
            let quantized_scalar = quantized.call_method0("item")?;
            new_args[1] = quantized_scalar;
            did_update = true;
        }
    }

    // Quantize max_val (args[2]) if present and not None
    if let Some(max_arg) = args.get(2) {
        if !max_arg.is_none() {
            let quantized = qargs.call_method1("quantize_value", (max_arg,))?;
            let quantized_scalar = quantized.call_method0("item")?;
            new_args[2] = quantized_scalar;
            did_update = true;
        }
    }

    if did_update {
        // Update node.args with the new quantized values.
        let py_args = pyo3::types::PyTuple::new_bound(py, &new_args);
        node.setattr("args", py_args)?;
    }
    Ok(())
}
