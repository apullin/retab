use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

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

    let d = FxDecomposer::new(py, graph, gm, node);

    // unsqueeze(x, dim + 1)
    let unsqueeze_dim = dim + 1;
    let unsqueezed = d.unsqueeze(x, unsqueeze_dim)?;
    d.propagate_qparams(&unsqueezed)?;

    // repeat(unsqueezed, multiples) where multiples[unsqueeze_dim] = repeats
    let repeat_rank = rank + 1;
    let mut multiples = vec![1i64; repeat_rank as usize];
    multiples[unsqueeze_dim as usize] = repeats;
    let repeated = d.repeat(&unsqueezed, &multiples)?;
    d.propagate_qparams(&repeated)?;

    // view(repeated, final_shape) where final_shape[dim] *= repeats
    let mut final_shape = input_shape;
    final_shape[dim as usize] *= repeats;
    let result = d.view(&repeated, &final_shape)?;
    d.propagate_qparams(&result)?;

    d.replace(&result)?;
    Ok(true)
}
