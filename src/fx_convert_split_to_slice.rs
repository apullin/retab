use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct conversion of split_with_sizes_copy/split_copy to
    /// multiple slice_copy ops.  Mirrors Python `ConvertSplitToSlicePass`.
    ///
    /// split(tensor, sizes, dim) -> [slice(tensor, dim, start, end) for each chunk]
    /// Each getitem user of the split is replaced with its corresponding slice.
    FxConvertSplitToSlicePass, "FxConvertSplitToSlice",
    matches: |target, _node| target.contains("split_with_sizes_copy") || target.contains("split_copy"),
    transform: |py, graph, gm, node, debug| convert_split_wrapper(py, &graph, gm, node, debug)
);

fn convert_split_wrapper<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    convert_split(py, graph, gm, node, &slice_op, debug)?;
    graph.call_method0("eliminate_dead_code")?;
    Ok(())
}

fn convert_split<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    _gm: &FxGraphModule<'py>,
    split_node: &Bound<'py, PyAny>,
    slice_op: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(split_node)?;
    if args.is_empty() {
        return Ok(());
    }

    let input_node = &args[0];
    let input_shape = FxGraphModule::node_shape(input_node)?;
    let rank = input_shape.len() as i64;

    // args[1] = split_lengths (list of ints or single int)
    // args[2] = dim (default 0)
    let dim_raw: i64 = if args.len() > 2 {
        args[2].extract().unwrap_or(0)
    } else {
        0
    };
    let dim = ((dim_raw % rank) + rank) % rank;
    let dim_size = input_shape.get(dim as usize).copied().unwrap_or(0);

    // Parse split_lengths
    let split_lengths: Vec<i64> = if args.len() > 1 {
        match args[1].extract::<Vec<i64>>() {
            Ok(lengths) => lengths,
            Err(_) => {
                // Single int: split into equal chunks
                let chunk_size: i64 = args[1].extract().unwrap_or(dim_size);
                if chunk_size <= 0 {
                    return Ok(());
                }
                let mut lengths = vec![chunk_size; (dim_size / chunk_size) as usize];
                let remainder = dim_size % chunk_size;
                if remainder > 0 {
                    lengths.push(remainder);
                }
                lengths
            }
        }
    } else {
        return Ok(());
    };

    // Compute start/end for each slice
    let mut starts = Vec::with_capacity(split_lengths.len());
    let mut ends = Vec::with_capacity(split_lengths.len());
    let mut start = 0i64;
    for &length in &split_lengths {
        starts.push(start);
        ends.push(start + length);
        start += length;
    }

    if debug {
        let name = FxGraphModule::node_name(split_node)?;
        eprintln!(
            "[ConvertSplitToSlice] {} dim={} sizes={:?} starts={:?} ends={:?}",
            name, dim, split_lengths, starts, ends
        );
    }

    // Get the getitem users of the split node
    let users = FxGraphModule::node_users(split_node)?;

    // Create inserting_before context
    let inserting = graph.call_method1("inserting_before", (split_node,))?;
    inserting.call_method0("__enter__")?;

    for user in &users {
        let user_target = FxGraphModule::node_target_str(user)?;
        if !user_target.contains("getitem") {
            continue;
        }

        // getitem args: (split_node, index)
        let user_args = FxGraphModule::node_args(user)?;
        let index: usize = if user_args.len() > 1 {
            user_args[1].extract().unwrap_or(0)
        } else {
            0
        };

        if index >= starts.len() {
            continue;
        }

        // Create slice_copy(input, dim, start, end)
        let dim_py = dim.into_py(py).into_bound(py);
        let start_py = starts[index].into_py(py).into_bound(py);
        let end_py = ends[index].into_py(py).into_bound(py);
        let slice_args = PyTuple::new_bound(
            py,
            &[
                input_node.as_ref(),
                dim_py.as_ref(),
                start_py.as_ref(),
                end_py.as_ref(),
            ],
        );
        let slice_node = graph.call_method1("call_function", (slice_op, &slice_args))?;

        // Copy metadata from the split node, selecting the right index
        fx_utils::copy_meta(py, split_node, &slice_node)?;
        let meta = slice_node.getattr("meta")?;

        // Select the right val/tensor_meta for this index
        let val = meta.call_method1("get", ("val",))?;
        if !val.is_none() {
            if let Ok(val_list) = val.extract::<Vec<Bound<'_, PyAny>>>() {
                if index < val_list.len() {
                    meta.set_item("val", &val_list[index])?;
                }
            } else if let Ok(val_tuple) = val.call_method1("__getitem__", (index,)) {
                meta.set_item("val", val_tuple)?;
            }
        }

        // Copy output_qparams from the getitem user (not the split)
        let user_meta = user.getattr("meta")?;
        if let Ok(oqp) = user_meta.call_method1("get", ("output_qparams",)) {
            if !oqp.is_none() {
                meta.set_item("output_qparams", oqp)?;
            }
        }

        fx_utils::refresh_fake_tensor(py, &slice_node)?;

        // Replace getitem with slice
        FxGraphModule::replace_all_uses_with(user, &slice_node)?;
    }

    inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    Ok(())
}
