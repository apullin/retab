use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.select_scatter` into
    /// arange + view + eq + unsqueeze + expand + where, mirroring the Python pass.
    FxDecomposeSelectScatterPass, "FxDecomposeSelectScatter",
    matches: |target| target.contains("select_scatter.default"),
    transform: |py, graph, gm, node, debug| decompose_select_scatter(py, &graph, gm, node, debug).map(|_| ())
);

fn copied_meta<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    clear_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let copy_mod = py.import_bound("copy")?;
    let meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    if clear_qparams {
        meta.set_item("input_qparams", PyDict::new_bound(py))?;
        meta.set_item("output_qparams", PyDict::new_bound(py))?;
    }
    Ok(meta)
}

fn insert_call_before_with_kwargs<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
    meta_src: &Bound<'py, PyAny>,
    clear_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = graph.call_method1("call_function", (target, args, kwargs))?;
    let meta_copy = copied_meta(py, meta_src, clear_qparams)?;
    new_node.setattr("meta", &meta_copy)?;
    let _ = fx_utils::refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
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

fn decompose_select_scatter<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 4 {
        return Ok(false);
    }

    let input_tensor = &args[0];
    let src_tensor = &args[1];
    let dim: i64 = args[2].extract()?;
    let mut index: i64 = args[3].extract()?;
    if !FxGraphModule::is_node(input_tensor) || !FxGraphModule::is_node(src_tensor) {
        return Ok(false);
    }

    let input_shape = FxGraphModule::node_shape(input_tensor)?;
    if input_shape.is_empty() {
        return Ok(false);
    }
    let rank = input_shape.len() as i64;
    let dim_norm = dim.rem_euclid(rank) as usize;
    let dim_size = input_shape[dim_norm];
    if index < 0 {
        index += dim_size;
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeSelectScatter] {} dim={} index={} input_shape={:?}",
            name, dim, index, input_shape
        );
    }

    let torch = py.import_bound("torch")?;
    let int32_dtype = torch.getattr("int32")?;

    let arange_op = resolve_aten_op(py, "arange.start_step")?;
    let eq_op = resolve_aten_op(py, "eq.Tensor")?;
    let where_op = resolve_aten_op(py, "where.self")?;
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    let arange_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            dim_size.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let arange_kwargs = PyDict::new_bound(py);
    arange_kwargs.set_item("dtype", int32_dtype.clone())?;
    let arange_node = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &arange_op,
        arange_args.as_ref(),
        &arange_kwargs,
        node,
        true,
    )?;

    let mut mask_shape = vec![1i64; rank as usize];
    mask_shape[dim_norm] = dim_size;
    let mask_shape_py = PyList::new_bound(py, mask_shape.iter().copied());
    let view_args = PyTuple::new_bound(py, &[arange_node.as_ref(), mask_shape_py.as_ref()]);
    let viewed_arange =
        insert_call_before(py, graph, node, &view_op, view_args.as_ref(), &arange_node)?;

    let scalar_shape = PyTuple::new_bound(py, [1i64]);
    let scalar_value = index.into_py(py).into_bound(py);
    let scalar_kwargs = PyDict::new_bound(py);
    scalar_kwargs.set_item("dtype", int32_dtype)?;
    let scalar_args = PyTuple::new_bound(py, &[scalar_shape.as_ref(), scalar_value.as_ref()]);
    let index_tensor = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &full_op,
        scalar_args.as_ref(),
        &scalar_kwargs,
        node,
        true,
    )?;

    let mask_args = PyTuple::new_bound(py, &[viewed_arange.as_ref(), index_tensor.as_ref()]);
    let mask_node = insert_call_before(py, graph, node, &eq_op, mask_args.as_ref(), node)?;

    let unsqueeze_args = PyTuple::new_bound(
        py,
        &[
            src_tensor.as_ref(),
            (dim_norm as i64).into_py(py).into_bound(py).as_ref(),
        ],
    );
    let unsqueeze_node = insert_call_before(
        py,
        graph,
        node,
        &unsqueeze_op,
        unsqueeze_args.as_ref(),
        src_tensor,
    )?;
    copy_qparams_only(
        &src_tensor.getattr("meta")?,
        &unsqueeze_node.getattr("meta")?,
    )?;

    let expand_shape_py = PyList::new_bound(py, input_shape.iter().copied());
    let expand_args = PyTuple::new_bound(py, &[unsqueeze_node.as_ref(), expand_shape_py.as_ref()]);
    let expand_node = insert_call_before(
        py,
        graph,
        node,
        &expand_op,
        expand_args.as_ref(),
        &unsqueeze_node,
    )?;
    copy_qparams_only(
        &unsqueeze_node.getattr("meta")?,
        &expand_node.getattr("meta")?,
    )?;

    let where_args = PyTuple::new_bound(
        py,
        &[
            mask_node.as_ref(),
            expand_node.as_ref(),
            input_tensor.as_ref(),
        ],
    );
    let where_node = insert_call_before(py, graph, node, &where_op, where_args.as_ref(), node)?;

    fx_utils::copy_meta(py, node, &where_node)?;

    FxGraphModule::replace_all_uses_with(node, &where_node)?;
    gm.erase_node(node)?;
    Ok(true)
}
