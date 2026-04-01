use pyo3::prelude::*;

use crate::fx_decompose_ops::{propagate_qparams, FxDecomposer};
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.index.Tensor` into
    /// `backend.tosa.GATHER` plus basic arith.
    FxDecomposeIndexTensorPass, "FxDecomposeIndexTensor",
    matches: |target, _node| target.contains("aten.index.Tensor"),
    transform: |py, graph, gm, node, _debug| decompose_single_index_tensor(py, &graph, gm, node).map(|_| ())
);

fn copy_qparams_only(src_meta: &Bound<'_, PyAny>, dst_meta: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(input_qparams) = src_meta.call_method1("get", ("input_qparams",)) {
        if !input_qparams.is_none() {
            dst_meta.set_item("input_qparams", input_qparams)?;
        }
    }
    if let Ok(output_qparams) = src_meta.call_method1("get", ("output_qparams",)) {
        if !output_qparams.is_none() {
            dst_meta.set_item("output_qparams", output_qparams)?;
        }
    }
    Ok(())
}

fn broadcast_shape(shapes: &[Vec<i64>]) -> PyResult<Vec<i64>> {
    let max_rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(max_rank);

    for axis in 0..max_rank {
        let mut chosen = 1i64;
        for shape in shapes {
            let dim = *shape.iter().rev().nth(axis).unwrap_or(&1);
            if dim != 1 {
                if chosen == 1 {
                    chosen = dim;
                } else if chosen != dim {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "non-broadcastable index dims at axis -{}",
                        axis + 1
                    )));
                }
            }
        }
        out.push(chosen);
    }

    out.reverse();
    Ok(out)
}

fn shape_to_stride(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= *dim;
    }
    strides
}

fn decompose_single_index_tensor<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &'py FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 2 {
        return Ok(false);
    }
    let x = &args[0];
    let indices = match args[1].extract::<Vec<Bound<'py, PyAny>>>() {
        Ok(v) => v,
        Err(_) => return Ok(false),
    };
    if indices.is_empty() {
        return Ok(false);
    }

    let x_shape = FxGraphModule::node_shape(x)?;
    if x_shape.is_empty() {
        return Ok(false);
    }
    let m = indices.len();
    if m > x_shape.len() {
        return Ok(false);
    }

    let mut index_nodes = Vec::with_capacity(indices.len());
    let mut index_shapes = Vec::with_capacity(indices.len());
    for index in &indices {
        if !FxGraphModule::is_node(index) {
            return Ok(false);
        }
        index_nodes.push(index.clone());
        index_shapes.push(FxGraphModule::node_shape(index)?);
    }

    let broadcast = broadcast_shape(&index_shapes)?;
    let w = broadcast.iter().copied().product::<i64>().max(1);
    let leading: Vec<i64> = x_shape.iter().take(m).copied().collect();
    let trailing: Vec<i64> = x_shape.iter().skip(m).copied().collect();
    let k = leading.iter().copied().product::<i64>().max(1);
    let c = trailing.iter().copied().product::<i64>().max(1);
    let strides = shape_to_stride(&x_shape);
    let lin_scales: Vec<i64> = strides.iter().take(m).map(|stride| stride / c).collect();

    let x_meta = x.getattr("meta")?;
    let index_dtype = index_nodes[0]
        .getattr("meta")?
        .get_item("val")?
        .getattr("dtype")?;

    let d = FxDecomposer::new(py, graph, gm, node);

    // Flatten x to (1, K, C)
    let x_1kc = d.view(x, &[1, k, c])?;
    propagate_qparams(x, &x_1kc)?;

    // Build linearised flat index by accumulating scaled per-dimension indices
    let mut lin_w: Option<Bound<'py, PyAny>> = None;
    for (index, (index_shape, lin_scale)) in index_nodes
        .iter()
        .zip(index_shapes.iter().zip(lin_scales.iter()))
    {
        // Align rank if needed
        let mut idx_current = index.clone();
        if index_shape.len() != broadcast.len() {
            let aligned_shape: Vec<i64> = std::iter::repeat(1i64)
                .take(broadcast.len() - index_shape.len())
                .chain(index_shape.iter().copied())
                .collect();
            idx_current = d.view(&idx_current, &aligned_shape)?;
        }

        // Broadcast to common shape, then flatten to (W,)
        let idx_broadcast = d.expand(&idx_current, &broadcast)?;
        let idx_flat = d.view(&idx_broadcast, &[w])?;

        // Scale by linearised stride
        let scale_const = d.full_int(&[1], *lin_scale, &index_dtype)?;
        let idx_scaled = d.mul(&idx_flat, &scale_const)?;

        lin_w = Some(match lin_w {
            None => idx_scaled,
            Some(prev) => d.add(&prev, &idx_scaled)?,
        });
    }

    let Some(lin_w) = lin_w else {
        return Ok(false);
    };

    // Unsqueeze to (1, W) for GATHER
    let idx_1w = d.unsqueeze(&lin_w, 0)?;

    // GATHER(x_1kc, idx_1w) -> (1, W, C)
    let gathered = d.gather(&x_1kc, &idx_1w)?;
    {
        d.set_meta_val_shape(x, &gathered, &[1, w, c])?;
        copy_qparams_only(&x_meta, &gathered.getattr("meta")?)?;
    }

    // Reshape to (...broadcast, ...trailing)
    let out_shape: Vec<i64> = broadcast
        .iter()
        .copied()
        .chain(trailing.iter().copied())
        .collect();
    let out = d.view(&gathered, &out_shape)?;
    {
        d.set_meta_val_shape(x, &out, &out_shape)?;
        copy_qparams_only(&x_meta, &out.getattr("meta")?)?;
    }

    d.replace(&out)
}
