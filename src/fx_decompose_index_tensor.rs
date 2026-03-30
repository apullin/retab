use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, propagate_qparams, resolve_aten_op};
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.index.Tensor` into
    /// `backend.tosa.GATHER` plus basic arith.
    FxDecomposeIndexTensorPass, "FxDecomposeIndexTensor",
    matches: |target| target.contains("aten.index.Tensor"),
    transform: |py, graph, gm, node, _debug| decompose_single_index_tensor(py, &graph, gm, node).map(|_| ())
);

fn resolve_tosa_gather_op<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.eval_bound(
        "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'backend'), 'tosa'), 'GATHER').default",
        None,
        None,
    )
}

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

fn set_tensor_meta_val_shape(
    template_meta: &Bound<'_, PyAny>,
    dst_meta: &Bound<'_, PyAny>,
    shape: &[i64],
) -> PyResult<()> {
    let template_val = template_meta.get_item("val")?;
    if !template_val.hasattr("new_empty")? {
        return Ok(());
    }
    let dtype = template_val.getattr("dtype")?;
    let kwargs = pyo3::types::PyDict::new_bound(template_val.py());
    kwargs.set_item("dtype", dtype)?;
    let new_val = template_val.call_method("new_empty", (shape.to_vec(),), Some(&kwargs))?;
    dst_meta.set_item("val", new_val)?;
    Ok(())
}

fn insert_full_before<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    meta_src: &Bound<'py, PyAny>,
    full_op: &Bound<'py, PyAny>,
    shape: &[i64],
    fill_value: i64,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;

    let shape_py = PyTuple::new_bound(py, shape.iter().copied());
    let fill_py = fill_value.into_py(py).into_bound(py);
    let args = PyTuple::new_bound(py, &[shape_py.as_ref(), fill_py.as_ref()]);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("dtype", dtype)?;
    let new_node = graph.call_method1("call_function", (full_op, args, kwargs))?;

    let copy_mod = py.import_bound("copy")?;
    let meta_copy = copy_mod.call_method1("copy", (meta_src.getattr("meta")?,))?;
    meta_copy.set_item("input_qparams", pyo3::types::PyDict::new_bound(py))?;
    meta_copy.set_item("output_qparams", pyo3::types::PyDict::new_bound(py))?;
    new_node.setattr("meta", &meta_copy)?;

    let fx_helpers = py.import_bound("fx_helpers")?;
    let _ = fx_helpers.call_method1("refresh_node_fake_tensor", (&new_node,));
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn decompose_single_index_tensor<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
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
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let mul_tensor_op = resolve_aten_op(py, "mul.Tensor")?;
    let add_tensor_op = resolve_aten_op(py, "add.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;
    let gather_op = resolve_tosa_gather_op(py)?;
    let index_dtype = index_nodes[0]
        .getattr("meta")?
        .get_item("val")?
        .getattr("dtype")?;

    let x_shape_1kc = pyo3::types::PyList::new_bound(py, [1i64, k, c]);
    let x_view_args = PyTuple::new_bound(py, &[x.as_ref(), x_shape_1kc.as_ref()]);
    let x_1kc = insert_call_before(py, graph, node, &view_op, x_view_args.as_ref(), node)?;
    propagate_qparams(x, &x_1kc)?;

    let mut lin_w: Option<Bound<'py, PyAny>> = None;
    for (index, (index_shape, lin_scale)) in index_nodes
        .iter()
        .zip(index_shapes.iter().zip(lin_scales.iter()))
    {
        let mut idx_current = index.clone();
        if index_shape.len() != broadcast.len() {
            let aligned_shape: Vec<i64> = std::iter::repeat(1i64)
                .take(broadcast.len() - index_shape.len())
                .chain(index_shape.iter().copied())
                .collect();
            let aligned_shape_py = pyo3::types::PyList::new_bound(py, aligned_shape);
            let idx_aligned_args =
                PyTuple::new_bound(py, &[idx_current.as_ref(), aligned_shape_py.as_ref()]);
            idx_current =
                insert_call_before(py, graph, node, &view_op, idx_aligned_args.as_ref(), node)?;
        }

        let broadcast_py = pyo3::types::PyList::new_bound(py, broadcast.iter().copied());
        let idx_broadcast_args =
            PyTuple::new_bound(py, &[idx_current.as_ref(), broadcast_py.as_ref()]);
        let idx_broadcast = insert_call_before(
            py,
            graph,
            node,
            &expand_op,
            idx_broadcast_args.as_ref(),
            node,
        )?;

        let idx_shape_w = pyo3::types::PyList::new_bound(py, [w]);
        let idx_view_args = PyTuple::new_bound(py, &[idx_broadcast.as_ref(), idx_shape_w.as_ref()]);
        let idx_flat = insert_call_before(py, graph, node, &view_op, idx_view_args.as_ref(), node)?;

        let scale_const = insert_full_before(
            py,
            graph,
            node,
            index,
            &full_op,
            &[1],
            *lin_scale,
            &index_dtype,
        )?;
        let idx_scaled_args = PyTuple::new_bound(py, &[idx_flat.as_ref(), scale_const.as_ref()]);
        let idx_scaled = insert_call_before(
            py,
            graph,
            node,
            &mul_tensor_op,
            idx_scaled_args.as_ref(),
            node,
        )?;

        lin_w = Some(match lin_w {
            None => idx_scaled,
            Some(prev) => {
                let add_args = PyTuple::new_bound(py, &[prev.as_ref(), idx_scaled.as_ref()]);
                insert_call_before(py, graph, node, &add_tensor_op, add_args.as_ref(), node)?
            }
        });
    }

    let Some(lin_w) = lin_w else {
        return Ok(false);
    };

    let unsqueeze_args = PyTuple::new_bound(
        py,
        &[lin_w.as_ref(), 0i64.into_py(py).into_bound(py).as_ref()],
    );
    let idx_1w = insert_call_before(
        py,
        graph,
        node,
        &unsqueeze_op,
        unsqueeze_args.as_ref(),
        node,
    )?;

    let gather_args = PyTuple::new_bound(py, &[x_1kc.as_ref(), idx_1w.as_ref()]);
    let gathered = insert_call_before(py, graph, node, &gather_op, gather_args.as_ref(), node)?;
    {
        let gathered_meta = gathered.getattr("meta")?;
        set_tensor_meta_val_shape(&x_meta, &gathered_meta, &[1, w, c])?;
        copy_qparams_only(&x_meta, &gathered_meta)?;
    }

    let out_shape: Vec<i64> = broadcast
        .iter()
        .copied()
        .chain(trailing.iter().copied())
        .collect();
    let out_shape_py = pyo3::types::PyList::new_bound(py, out_shape.iter().copied());
    let out_view_args = PyTuple::new_bound(py, &[gathered.as_ref(), out_shape_py.as_ref()]);
    let out = insert_call_before(py, graph, node, &view_op, out_view_args.as_ref(), node)?;
    {
        let out_meta = out.getattr("meta")?;
        set_tensor_meta_val_shape(&x_meta, &out_meta, &out_shape)?;
        copy_qparams_only(&x_meta, &out_meta)?;
    }

    FxGraphModule::replace_all_uses_with(node, &out)?;
    gm.erase_node(node)?;
    Ok(true)
}
