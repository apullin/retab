use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{resolve_aten_op, NodeInserter};
use crate::fx_graph::{arg_extract, meta_val, FxGraphModule};

crate::fx_pass!(
    /// FX-graph-direct implementation of sum-to-REDUCE_SUM conversion.
    ///
    /// Rewrites `aten.sum.dim_IntList` nodes into one or more chained
    /// `aten.sum.dim_IntList` operations (one per reduction axis), since TOSA
    /// REDUCE_SUM only reduces a single axis at a time.
    ///
    /// For multi-axis reductions the dims are processed in descending order
    /// so that earlier reductions don't shift the indices of later ones.
    ///
    /// All graph surgery is performed in Rust via `NodeInserter`.
    FxConvertSumPass, "FxConvertSum",
    matches: |target, _node| target.contains("sum.dim_IntList"),
    transform: |py, graph, gm, node, debug| convert_sum(py, &graph, gm, node, debug)
);

fn convert_sum<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;

    // args[0] = input tensor, args[1] = dims list
    if args.len() < 2 {
        return Ok(());
    }

    let input_node = &args[0];
    if !FxGraphModule::is_node(input_node) {
        return Ok(());
    }

    // Extract the dims list. It may be a Python list/tuple of ints.
    let dims: Vec<i64> = match args[1].extract::<Vec<i64>>() {
        Ok(d) => d,
        Err(_) => match args[1].extract::<i64>() {
            Ok(d) => vec![d],
            Err(_) => return Ok(()),
        },
    };

    if dims.is_empty() {
        return Ok(());
    }

    // Normalize negative dims: TOSA REDUCE_SUM requires non-negative axes.
    let input_shape = FxGraphModule::node_shape(input_node).unwrap_or_default();
    let input_rank = input_shape.len() as i64;
    let mut dims: Vec<i64> = dims
        .into_iter()
        .map(|d| if d < 0 { d + input_rank } else { d })
        .collect();

    // Check for keepdim (args[2] if present, default false).
    let keepdim: bool = arg_extract(&args, 2, false);

    // Sort dims descending to keep indices stable during chained reductions.
    dims.sort_unstable_by(|a, b| b.cmp(a));

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[ConvertSum] {} dims={:?} keepdim={} input_shape={:?}",
            name, dims, keepdim, input_shape
        );
    }

    let target_op = resolve_aten_op(py, "sum.dim_IntList")?;
    let copy_mod = py.import_bound("copy")?;

    let mut current_input = input_node.clone();
    let mut last_reduce: Option<Bound<'py, PyAny>> = None;

    // Track qparams for chaining: first reduce inherits from the original node,
    // subsequent reduces take the previous output_qparams as their input_qparams.
    let orig_meta = node.getattr("meta")?;
    let mut current_input_qparams = copy_mod.call_method1("copy", (
        meta_val(&orig_meta, "input_qparams")
            .unwrap_or_else(|| py.None().into_bound(py)),
    ))?;
    let current_output_qparams = copy_mod.call_method1("copy", (
        meta_val(&orig_meta, "output_qparams")
            .unwrap_or_else(|| py.None().into_bound(py)),
    ))?;

    for dim in &dims {
        let ins = NodeInserter::before(py, graph, node).keep_qparams();

        // Build args: (current_input, [dim], keepdim)
        let dim_list = pyo3::types::PyList::new_bound(py, [*dim]);
        let keepdim_py = keepdim.into_py(py).into_bound(py);
        let call_args = PyTuple::new_bound(
            py,
            &[current_input.as_ref(), dim_list.as_ref(), keepdim_py.as_ref()],
        );
        let reduce_node = ins.call(&target_op, call_args.as_ref())?;

        // Set name: <original_name>_reduce_dim<N>
        let orig_name: String = FxGraphModule::node_name(node)?;
        reduce_node.setattr("name", format!("{}_reduce_dim{}", orig_name, dim))?;

        // Copy meta from original node, then override qparams
        let meta_copy = copy_mod.call_method1("copy", (orig_meta.clone(),))?;
        meta_copy.set_item("input_qparams", &current_input_qparams)?;
        meta_copy.set_item("output_qparams", &current_output_qparams)?;
        reduce_node.setattr("meta", &meta_copy)?;

        // Compute the output shape for this reduce step and set meta["val"]
        let cur_input_meta = current_input.getattr("meta")?;
        let cur_input_val = meta_val(&cur_input_meta, "val")
            .unwrap_or_else(|| py.None().into_bound(py));
        // Handle list/tuple vals
        let input_val = if cur_input_val.is_instance_of::<pyo3::types::PyList>()
            || cur_input_val.is_instance_of::<pyo3::types::PyTuple>()
        {
            let len: usize = cur_input_val.len()?;
            if len > 0 {
                cur_input_val.get_item(0)?
            } else {
                cur_input_val
            }
        } else {
            cur_input_val
        };

        if !input_val.is_none() && input_val.hasattr("new_empty")? {
            let cur_shape: Vec<i64> = input_val.getattr("shape")?.extract()?;
            let rank = cur_shape.len();
            let normalized_dim = (*dim as usize) % rank;
            let mut output_shape = cur_shape;
            if keepdim {
                output_shape[normalized_dim] = 1;
            } else {
                output_shape.remove(normalized_dim);
            }
            let new_val = input_val.call_method1("new_empty", (output_shape,))?;
            reduce_node.getattr("meta")?.set_item("val", new_val)?;
        }

        current_input = reduce_node.clone();
        current_input_qparams = copy_mod.call_method1("copy", (&current_output_qparams,))?;
        last_reduce = Some(reduce_node);
    }

    // Copy qparams from original node onto the final reduce, then replace.
    if let Some(ref final_node) = last_reduce {
        let final_meta = final_node.getattr("meta")?;
        if let Some(iqp) = meta_val(&orig_meta, "input_qparams") {
            final_meta.set_item("input_qparams", iqp)?;
        }
        if let Some(oqp) = meta_val(&orig_meta, "output_qparams") {
            final_meta.set_item("output_qparams", oqp)?;
        }

        FxGraphModule::replace_all_uses_with(node, final_node)?;
        gm.erase_node(node)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
