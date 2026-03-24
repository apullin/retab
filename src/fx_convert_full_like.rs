use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct conversion of `aten.full_like.default` to `aten.full.default`.
///
/// Mirrors Python `ConvertFullLikeToFullPass`:
///   full_like(x, fill_value) -> full(x.shape, fill_value, dtype=x.dtype)
///
/// TOSA has no full_like equivalent; this converts to the explicit-shape form
/// so the serializer can emit a CONST tensor.
pub struct FxConvertFullLikeToFullPass;

fn is_full_like(target: &str) -> bool {
    target.contains("full_like")
}

impl FxPass for FxConvertFullLikeToFullPass {
    fn name(&self) -> &str {
        "FxConvertFullLikeToFull"
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
            if is_full_like(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            convert_full_like(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

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
    let input_val = input_meta.call_method1("get", ("val",))?;
    let dtype = if !input_val.is_none() {
        input_val.getattr("dtype").ok()
    } else {
        None
    };

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
    // Carry over any kwargs from the original full_like node
    let orig_kwargs = node.getattr("kwargs")?;
    if let Ok(iter) = orig_kwargs.call_method0("items") {
        if let Ok(items) = iter.iter() {
            for item in items {
                if let Ok(item) = item {
                    let key: String = item.get_item(0)?.extract()?;
                    let val = item.get_item(1)?;
                    if !val.is_none() {
                        kwargs.set_item(key, val)?;
                    }
                }
            }
        }
    }

    // Insert full(shape, fill_value, dtype=dtype) before the original node
    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;
    let full_args = PyTuple::new_bound(py, &[shape_py.as_ref(), fill_value.as_ref()]);
    let new_node = graph.call_method("call_function", (&full_op, &full_args), Some(&kwargs))?;
    inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

    // Copy meta from original node
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    new_node.setattr("meta", &final_meta)?;

    // Try to refresh FakeTensor
    let _ = {
        let fxh = py.import_bound("fx_helpers");
        if let Ok(fxh) = fxh {
            fxh.call_method1("refresh_node_fake_tensor", (&new_node,))
        } else {
            Ok(py.None().into_bound(py))
        }
    };

    FxGraphModule::replace_all_uses_with(node, &new_node)?;
    gm.erase_node(node)?;
    Ok(())
}
