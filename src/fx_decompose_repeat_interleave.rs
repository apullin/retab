use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, propagate_qparams, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

pub struct FxDecomposeRepeatInterleavePass;

fn is_repeat_interleave_self_int(target: &str) -> bool {
    target.contains("repeat_interleave.self_int")
}

impl FxPass for FxDecomposeRepeatInterleavePass {
    fn name(&self) -> &str {
        "FxDecomposeRepeatInterleave"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_repeat_interleave_self_int(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;
        for node in &candidates {
            if decompose_repeat_interleave_self_int(py, &graph, gm, node, debug)? {
                modified = true;
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

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

    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
