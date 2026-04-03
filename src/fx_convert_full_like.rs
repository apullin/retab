use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::{meta_val, FxGraphModule};
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct conversion of `aten.full_like.default` to `aten.full.default`.
    ///
    /// Mirrors Python `ConvertFullLikeToFullPass`:
    ///   full_like(x, fill_value) -> full(x.shape, fill_value, dtype=x.dtype)
    ///
    /// TOSA has no full_like equivalent; this converts to the explicit-shape form
    /// so the serializer can emit a CONST tensor.
    FxConvertFullLikeToFullPass, "FxConvertFullLikeToFull",
    matches: |target, _node| target.contains("full_like"),
    transform: |py, graph, gm, node, debug| {
        convert_full_like(py, &graph, gm, node, debug)
    }
);

fn convert_full_like<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 2 {
        return Ok(());
    }

    let input_node = &args[0];
    let fill_value = &args[1];

    // Get input shape and dtype
    let input_shape = FxGraphModule::node_shape(input_node)?;
    let input_meta = input_node.getattr("meta")?;
    let input_val = meta_val(&input_meta, "val");
    let dtype = input_val.as_ref().and_then(|v| v.getattr("dtype").ok());

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[ConvertFullLikeToFull] converting: {} (shape={:?})",
            name, input_shape
        );
    }

    let full_op = resolve_aten_op(py, "full.default")?;

    // Build shape as a Python tuple
    let shape_py = PyTuple::new_bound(py, input_shape.iter().copied());

    // Create kwargs with dtype if available
    let kwargs = pyo3::types::PyDict::new_bound(py);
    if let Some(dt) = dtype {
        kwargs.set_item("dtype", dt)?;
    }
    // Flat extract: carry over non-None kwargs from the original full_like node.
    let orig_kwargs = node.getattr("kwargs")?;
    if let Ok(orig_dict) = orig_kwargs.downcast::<pyo3::types::PyDict>() {
        for (key, val) in orig_dict.iter() {
            if !val.is_none() {
                kwargs.set_item(key, val)?;
            }
        }
    }

    // Insert full(shape, fill_value, dtype=dtype) before the original node
    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;
    let full_args = PyTuple::new_bound(py, &[shape_py.as_ref(), fill_value.as_ref()]);
    // Graph.call_function(target, args, kwargs) — kwargs is a positional param
    let new_node = graph.call_method1("call_function", (&full_op, &full_args, &kwargs))?;
    inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

    // Copy meta and refresh FakeTensor
    fx_utils::copy_meta_and_refresh(py, node, &new_node)?;

    FxGraphModule::replace_all_uses_with(node, &new_node)?;
    gm.erase_node(node)?;
    Ok(())
}
