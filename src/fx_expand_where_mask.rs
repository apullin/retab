use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// Expand lower-rank boolean masks feeding `where` to the full output shape.
    ///
    /// Python decomposition passes often rely on PyTorch broadcasting for masks
    /// such as `[H, W]` against data tensors `[N, H, W]`. The direct TOSA SELECT
    /// emission is stricter, so we materialize the broadcast explicitly as
    /// `view_copy + expand_copy` in the FX graph.
    FxExpandWhereMaskPass, "FxExpandWhereMask",
    matches: |target| target.contains("where.self") || target.contains("aten_where"),
    transform: |py, graph, gm, node, debug| expand_where_mask(py, &graph, gm, node)
);

fn expand_where_mask<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 3 {
        return Ok(());
    }

    let mask = &args[0];
    if !FxGraphModule::is_node(mask) {
        return Ok(());
    }

    let output_shape = FxGraphModule::node_shape(node)?;
    let mask_shape = FxGraphModule::node_shape(mask)?;
    if output_shape.is_empty() || mask_shape.is_empty() {
        return Ok(());
    }

    if mask_shape.len() > output_shape.len() {
        return Ok(());
    }

    let mut padded_mask_shape = vec![1i64; output_shape.len() - mask_shape.len()];
    padded_mask_shape.extend(mask_shape.iter().copied());

    let broadcastable = padded_mask_shape
        .iter()
        .zip(output_shape.iter())
        .all(|(mask_dim, out_dim)| *mask_dim == 1 || *mask_dim == *out_dim);
    if !broadcastable || padded_mask_shape == output_shape {
        return Ok(());
    }

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let where_op = resolve_aten_op(py, "where.self")?;

    let view_shape = PyList::new_bound(py, padded_mask_shape.iter().copied());
    let view_args = PyTuple::new_bound(py, &[mask.as_ref(), view_shape.as_ref()]);
    let viewed_mask = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), mask)?;

    let expand_shape = PyList::new_bound(py, output_shape.iter().copied());
    let expand_args = PyTuple::new_bound(py, &[viewed_mask.as_ref(), expand_shape.as_ref()]);
    let expanded_mask = insert_call_before(
        py,
        graph,
        node,
        &expand_op,
        expand_args.as_ref(),
        &viewed_mask,
    )?;

    let where_args = PyTuple::new_bound(
        py,
        &[expanded_mask.as_ref(), args[1].as_ref(), args[2].as_ref()],
    );
    let result_node = insert_call_before(py, graph, node, &where_op, where_args.as_ref(), node)?;

    fx_utils::copy_meta_and_refresh(py, node, &result_node)?;

    FxGraphModule::replace_all_uses_with(node, &result_node)?;
    gm.erase_node(node)?;
    Ok(())
}
