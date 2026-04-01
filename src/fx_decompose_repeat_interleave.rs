use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, propagate_qparams, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// Decompose `repeat_interleave.self_int` into unsqueeze + repeat + view.
    FxDecomposeRepeatInterleavePass, "FxDecomposeRepeatInterleave",
    matches: |target, _node| target.contains("repeat_interleave.self_int"),
    transform: |py, graph, gm, node, debug| decompose_repeat_interleave_self_int(py, &graph, gm, node, debug).map(|_| ())
);

fn decompose_repeat_interleave_self_int<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 3 {
        return Ok(false);
    }

    let x = &args[0];
    let repeats: i64 = args[1].extract()?;
    if repeats < 0 {
        return Ok(false);
    }
    let Some(dim_obj) = args.get(2) else {
        return Ok(false);
    };
    if dim_obj.is_none() {
        return Ok(false);
    }
    let mut dim: i64 = dim_obj.extract()?;

    let input_shape = FxGraphModule::node_shape(x)?;
    let rank = input_shape.len() as i64;
    if rank == 0 {
        return Ok(false);
    }

    if dim < 0 {
        dim += rank;
    }
    if !(0..rank).contains(&dim) {
        return Ok(false);
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeRepeatInterleave] {} repeats={} dim={} shape={:?}",
            name, repeats, dim, input_shape
        );
    }

    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let repeat_op = resolve_aten_op(py, "repeat.default")?;
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let builtins = py.import_bound("builtins")?;

    let unsqueeze_dim = dim + 1;
    let unsqueeze_dim_obj = builtins.call_method1("int", (unsqueeze_dim,))?;
    let unsqueeze_args = PyTuple::new_bound(py, &[x.as_ref(), unsqueeze_dim_obj.as_ref()]);
    let unsqueezed = insert_call_before(
        py,
        graph,
        node,
        &unsqueeze_op,
        unsqueeze_args.as_ref(),
        node,
    )?;
    propagate_qparams(node, &unsqueezed)?;

    let repeat_rank = rank + 1;
    let mut multiples = vec![1i64; repeat_rank as usize];
    multiples[unsqueeze_dim as usize] = repeats;
    let multiples_list = PyList::new_bound(py, multiples);
    let repeat_args = PyTuple::new_bound(py, &[unsqueezed.as_ref(), multiples_list.as_ref()]);
    let repeated = insert_call_before(py, graph, node, &repeat_op, repeat_args.as_ref(), node)?;
    propagate_qparams(node, &repeated)?;

    let mut final_shape = input_shape;
    final_shape[dim as usize] *= repeats;
    let final_shape_list = PyList::new_bound(py, final_shape);
    let view_args = PyTuple::new_bound(py, &[repeated.as_ref(), final_shape_list.as_ref()]);
    let result = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), node)?;
    propagate_qparams(node, &result)?;

    fx_utils::copy_meta(py, node, &result)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
