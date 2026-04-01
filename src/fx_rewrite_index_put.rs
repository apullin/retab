use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

/// Compute (N, K, W, C) dimensions for the TOSA scatter op.
///
/// - N = 1 (batch is always 1 for this rewrite)
/// - K = product of source dims covered by the indices
/// - W = product of index shape (number of scatter writes)
/// - C = total source elements / K (trailing dims not indexed)
fn calculate_tosa_scatter_dims(
    index_shape: &[i64],
    source_shape: &[i64],
    num_index_dims: usize,
) -> (i64, i64, i64, i64) {
    let n: i64 = 1;
    let w: i64 = if index_shape.is_empty() {
        1
    } else {
        index_shape.iter().copied().product()
    };
    let mut k: i64 = 1;
    for (i, &dim) in source_shape.iter().enumerate() {
        if i < num_index_dims {
            k *= dim;
        }
    }
    let total_vals: i64 = source_shape.iter().copied().product();
    let c: i64 = if k != 0 { total_vals / k } else { 1 };
    (n, k, w, c)
}

/// Compute per-dimension strides for the source tensor.
fn calculate_values_stride(source_shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; source_shape.len()];
    let mut stride: i64 = 1;
    for i in (0..source_shape.len()).rev() {
        strides[i] = stride;
        stride *= source_shape[i];
    }
    strides
}

/// Rewrite `aten.index_put` into view+scatter+view for TOSA.
///
/// 1. Expand None indices to explicit arange tensors
/// 2. Flatten indices into [N, W]
/// 3. Reshape source to [N, K, C], values to [N, W, C]
/// 4. Emit tosa.SCATTER
/// 5. Reshape back to the original source shape
fn rewrite_index_put_to_tosa<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &'py FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    _debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 3 {
        return Ok(());
    }

    let source = &args[0];
    let indices_obj = &args[1];
    let values = &args[2];

    // Check accumulate flag
    let accumulate = if args.len() > 3 {
        args[3].extract::<bool>().unwrap_or(false)
    } else {
        false
    };
    if accumulate {
        return Ok(());
    }

    // indices must be a list/tuple
    let builtins = py.import_bound("builtins")?;
    let is_list_or_tuple = builtins
        .call_method1(
            "isinstance",
            (indices_obj, PyTuple::new_bound(py, &[
                builtins.getattr("list")?,
                builtins.getattr("tuple")?,
            ])),
        )?
        .extract::<bool>()?;
    if !is_list_or_tuple {
        return Ok(());
    }

    let source_shape = FxGraphModule::node_shape(source)?;
    if source_shape.is_empty() {
        return Ok(());
    }

    // Collect indices, expanding None to arange
    let indices_len: usize = builtins
        .call_method1("len", (indices_obj,))?
        .extract()?;

    // Find index_dtype from the first non-None index
    let mut index_dtype: Option<Bound<'py, PyAny>> = None;
    for i in 0..indices_len {
        let idx = indices_obj.get_item(i)?;
        if !idx.is_none() && FxGraphModule::is_node(&idx) {
            let meta = idx.getattr("meta")?;
            let val = meta.get_item("val")?;
            index_dtype = Some(val.getattr("dtype")?);
            break;
        }
    }
    let index_dtype = match index_dtype {
        Some(dt) => dt,
        None => return Ok(()),
    };

    let d = FxDecomposer::new(py, graph, gm, node);

    // Process indices: expand None to arange, collect Nodes
    let mut processed_indices: Vec<Bound<'py, PyAny>> = Vec::new();
    for dim_idx in 0..indices_len {
        let idx = indices_obj.get_item(dim_idx)?;
        if idx.is_none() {
            let expanded = d.arange(0, source_shape[dim_idx], &index_dtype)?;
            processed_indices.push(expanded);
        } else if FxGraphModule::is_node(&idx) {
            processed_indices.push(idx);
        } else {
            return Ok(());
        }
    }

    if processed_indices.is_empty() {
        return Ok(());
    }

    // Compute broadcast shape of all indices
    let torch = py.import_bound("torch")?;
    let mut index_shapes: Vec<Vec<i64>> = Vec::new();
    for idx in &processed_indices {
        let shape = FxGraphModule::node_shape(idx)?;
        index_shapes.push(shape);
    }

    // Use torch.broadcast_shapes to compute the broadcast shape
    let shape_tuples: Vec<Bound<'py, PyAny>> = index_shapes
        .iter()
        .map(|s| {
            PyTuple::new_bound(py, s).into_any()
        })
        .collect();
    let broadcast_shape: Vec<i64> = match torch
        .call_method1("broadcast_shapes", PyTuple::new_bound(py, &shape_tuples))
    {
        Ok(result) => result.extract()?,
        Err(_) => return Ok(()),
    };

    let num_index_dims = processed_indices.len();
    let (n, k, w, c) =
        calculate_tosa_scatter_dims(&broadcast_shape, &source_shape, num_index_dims);

    if !(n > 0 && k > 0 && w > 0 && c > 0) {
        return Ok(());
    }

    let values_strides = calculate_values_stride(&source_shape);

    // Build flat index: sum of (index[dim] * stride_scale) for each dim
    let mut flat_indices: Option<Bound<'py, PyAny>> = None;
    for (dim_idx, index_node) in processed_indices.iter().enumerate() {
        let scale_val = values_strides[dim_idx] / c;
        let mul_const = d.full_int(&[1], scale_val, &index_dtype)?;
        let scaled_index = d.mul(index_node, &mul_const)?;

        flat_indices = Some(match flat_indices {
            None => scaled_index,
            Some(prev) => d.add(&scaled_index, &prev)?,
        });
    }

    let flat_indices = match flat_indices {
        Some(fi) => fi,
        None => return Ok(()),
    };

    // Reshape flat indices, source, and values for scatter
    let reshape_indices = d.view(&flat_indices, &[n, w])?;
    let reshape_source = d.view(source, &[n, k, c])?;
    let reshape_values = d.view(values, &[n, w, c])?;

    // Emit tosa.SCATTER(reshape_source, reshape_indices, reshape_values)
    let scatter_node = d.scatter(&reshape_source, &reshape_indices, &reshape_values)?;

    // Reshape back to original source shape
    let output = d.view(&scatter_node, &source_shape)?;

    // Replace original node and clean up
    FxGraphModule::replace_all_uses_with(node, &output)?;
    gm.erase_node(node)?;
    graph.call_method0("eliminate_dead_code")?;

    Ok(())
}

crate::fx_pass!(FxRewriteIndexPutPass, "FxRewriteIndexPut",
    matches: |target, _node| target.contains("index_put.default"),
    transform: |py, graph, gm, node, debug| {
        rewrite_index_put_to_tosa(py, &graph, gm, node, debug)
    }
);
