use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{resolve_aten_op, NodeInserter};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.select_scatter` into
    /// arange + view + eq + unsqueeze + expand + where, mirroring the Python pass.
    FxDecomposeSelectScatterPass, "FxDecomposeSelectScatter",
    matches: |target, _node| target.contains("select_scatter.default"),
    transform: |py, graph, gm, node, debug| decompose_select_scatter(py, &graph, gm, node, debug).map(|_| ())
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

    let ins = NodeInserter::before(py, graph, node);

    let arange_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            dim_size.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let arange_kwargs = PyDict::new_bound(py);
    arange_kwargs.set_item("dtype", int32_dtype.clone())?;
    let arange_node = ins.call_with_kwargs(&arange_op, arange_args.as_ref(), &arange_kwargs)?;

    let mut mask_shape = vec![1i64; rank as usize];
    mask_shape[dim_norm] = dim_size;
    let mask_shape_py = PyList::new_bound(py, mask_shape.iter().copied());
    let view_args = PyTuple::new_bound(py, &[arange_node.as_ref(), mask_shape_py.as_ref()]);
    let viewed_arange = ins.clone().meta_from(&arange_node).call(&view_op, view_args.as_ref())?;

    let scalar_shape = PyTuple::new_bound(py, [1i64]);
    let scalar_value = index.into_py(py).into_bound(py);
    let scalar_kwargs = PyDict::new_bound(py);
    scalar_kwargs.set_item("dtype", int32_dtype)?;
    let scalar_args = PyTuple::new_bound(py, &[scalar_shape.as_ref(), scalar_value.as_ref()]);
    let index_tensor = ins.call_with_kwargs(&full_op, scalar_args.as_ref(), &scalar_kwargs)?;

    let mask_args = PyTuple::new_bound(py, &[viewed_arange.as_ref(), index_tensor.as_ref()]);
    let mask_node = ins.call(&eq_op, mask_args.as_ref())?;

    let unsqueeze_args = PyTuple::new_bound(
        py,
        &[
            src_tensor.as_ref(),
            (dim_norm as i64).into_py(py).into_bound(py).as_ref(),
        ],
    );
    let unsqueeze_node = ins.clone().meta_from(src_tensor).call(&unsqueeze_op, unsqueeze_args.as_ref())?;
    copy_qparams_only(
        &src_tensor.getattr("meta")?,
        &unsqueeze_node.getattr("meta")?,
    )?;

    let expand_shape_py = PyList::new_bound(py, input_shape.iter().copied());
    let expand_args = PyTuple::new_bound(py, &[unsqueeze_node.as_ref(), expand_shape_py.as_ref()]);
    let expand_node = ins.clone().meta_from(&unsqueeze_node).call(&expand_op, expand_args.as_ref())?;
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
    let where_node = ins.call(&where_op, where_args.as_ref())?;

    fx_utils::copy_meta(py, node, &where_node)?;

    FxGraphModule::replace_all_uses_with(node, &where_node)?;
    gm.erase_node(node)?;
    Ok(true)
}
