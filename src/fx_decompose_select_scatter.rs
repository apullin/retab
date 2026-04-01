use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{py_int, resolve_aten_op, FxDecomposer};
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.select_scatter` into
    /// arange + view + eq + unsqueeze + expand + where, mirroring the Python pass.
    FxDecomposeSelectScatterPass, "FxDecomposeSelectScatter",
    matches: |target, _node| target.contains("select_scatter.default"),
    transform: |py, graph, gm, node, debug| decompose_select_scatter(py, &graph, gm, node, debug).map(|_| ())
);

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

    let d = FxDecomposer::new(py, graph, gm, node);

    // arange(0, dim_size, dtype=int32)
    let arange_node = d.arange(0, dim_size, &int32_dtype)?;

    // view(arange, mask_shape) where mask_shape = [1, ..., dim_size, ..., 1]
    let mut mask_shape = vec![1i64; rank as usize];
    mask_shape[dim_norm] = dim_size;
    let viewed_arange = d.view(&arange_node, &mask_shape)?;

    // full([1], index, dtype=int32)
    let index_tensor = d.full_int(&[1], index, &int32_dtype)?;

    // eq(viewed_arange, index_tensor) -> boolean mask
    let mask_node = d.eq(&viewed_arange, &index_tensor)?;

    // unsqueeze(src, dim) — needs meta from src_tensor and qparams preserved
    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let dim_py = py_int(py, dim_norm as i64)?;
    let unsqueeze_args = PyTuple::new_bound(py, &[src_tensor.as_ref(), dim_py.as_ref()]);
    let unsqueeze_node = d
        .inserter()
        .clone()
        .meta_from(src_tensor)
        .keep_qparams()
        .call(&unsqueeze_op, unsqueeze_args.as_ref())?;

    // expand(unsqueezed, input_shape) — needs meta from unsqueeze_node and qparams preserved
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let expand_shape_py = pyo3::types::PyList::new_bound(py, input_shape.iter().copied());
    let expand_args =
        PyTuple::new_bound(py, &[unsqueeze_node.as_ref(), expand_shape_py.as_ref()]);
    let expand_node = d
        .inserter()
        .clone()
        .meta_from(&unsqueeze_node)
        .keep_qparams()
        .call(&expand_op, expand_args.as_ref())?;

    // where(mask, expanded_src, input) -> result
    let where_node = d.where_(&mask_node, &expand_node, input_tensor)?;

    d.replace(&where_node)?;
    Ok(true)
}
