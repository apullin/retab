use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.index_select` into
/// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
pub struct FxDecomposeIndexSelectToGatherPass;

fn is_index_select(target: &str) -> bool {
    target.contains("index_select.default")
}

fn resolve_tosa_gather_op<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.eval_bound(
        "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'backend'), 'tosa'), 'GATHER').default",
        None,
        None,
    )
}

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

fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    helpers.call_method1("refresh_node_fake_tensor", (node,))?;
    Ok(())
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
    let _ = refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
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
    let kwargs = PyDict::new_bound(template_val.py());
    kwargs.set_item("dtype", dtype)?;
    let new_val = template_val.call_method("new_empty", (shape.to_vec(),), Some(&kwargs))?;
    dst_meta.set_item("val", new_val)?;
    Ok(())
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

fn product(shape: &[i64]) -> i64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

impl FxPass for FxDecomposeIndexSelectToGatherPass {
    fn name(&self) -> &str {
        "FxDecomposeIndexSelectToGather"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            if FxGraphModule::node_op(node)? != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_index_select(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;
        for node in &candidates {
            if decompose_index_select(py, &graph, gm, node, debug)? {
                modified = true;
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_index_select<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 3 {
        return Ok(false);
    }

    let x = &args[0];
    let dim: i64 = args[1].extract()?;
    let index = &args[2];
    if !FxGraphModule::is_node(x) || !FxGraphModule::is_node(index) {
        return Ok(false);
    }

    let x_shape = FxGraphModule::node_shape(x)?;
    let index_shape = FxGraphModule::node_shape(index)?;
    let x_rank = x_shape.len() as i64;
    if x_rank < 1 || index_shape.len() != 1 {
        return Ok(false);
    }

    let dim_norm = dim.rem_euclid(x_rank) as usize;
    let before = x_shape[..dim_norm].to_vec();
    let after = x_shape[dim_norm + 1..].to_vec();
    let n = product(&before);
    let k = x_shape[dim_norm];
    let c = product(&after);
    let w = index_shape[0];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeIndexSelectToGather] {} dim={} input={:?} index={:?} -> before={:?} after={:?} N={} K={} C={} W={}",
            name, dim, x_shape, index_shape, before, after, n, k, c, w
        );
    }

    let x_meta = x.getattr("meta")?;
    let x_dtype_str: String = x_meta.get_item("val")?.getattr("dtype")?.str()?.extract()?;
    let needs_bool_cast = x_dtype_str == "torch.bool";

    let torch = py.import_bound("torch")?;
    let int8_dtype = torch.getattr("int8")?;
    let bool_dtype = torch.getattr("bool")?;

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let to_copy_op = resolve_aten_op(py, "_to_copy.default")?;
    let gather_op = resolve_tosa_gather_op(py)?;

    let mut x_for_gather = x.clone();
    if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[x.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", int8_dtype)?;
        x_for_gather = insert_call_before_with_kwargs(
            py,
            graph,
            node,
            &to_copy_op,
            cast_args.as_ref(),
            &cast_kwargs,
            x,
            true,
        )?;
    }

    let values_shape = PyList::new_bound(py, [n, k, c]);
    let values_args = PyTuple::new_bound(py, &[x_for_gather.as_ref(), values_shape.as_ref()]);
    let values_nkc = insert_call_before(
        py,
        graph,
        node,
        &view_op,
        values_args.as_ref(),
        &x_for_gather,
    )?;
    copy_qparams_only(&x_for_gather.getattr("meta")?, &values_nkc.getattr("meta")?)?;

    let idx_1w_args = PyTuple::new_bound(
        py,
        &[index.as_ref(), 0i64.into_py(py).into_bound(py).as_ref()],
    );
    let idx_1w = insert_call_before(py, graph, node, &unsqueeze_op, idx_1w_args.as_ref(), index)?;

    let indices_shape = PyList::new_bound(py, [n, w]);
    let indices_args = PyTuple::new_bound(py, &[idx_1w.as_ref(), indices_shape.as_ref()]);
    let indices_nw =
        insert_call_before(py, graph, node, &expand_op, indices_args.as_ref(), &idx_1w)?;

    let gather_args = PyTuple::new_bound(py, &[values_nkc.as_ref(), indices_nw.as_ref()]);
    let gathered_nwc = insert_call_before(py, graph, node, &gather_op, gather_args.as_ref(), node)?;
    {
        let gather_meta = gathered_nwc.getattr("meta")?;
        let gather_template_meta = x_for_gather.getattr("meta")?;
        set_tensor_meta_val_shape(&gather_template_meta, &gather_meta, &[n, w, c])?;
        copy_qparams_only(&gather_template_meta, &gather_meta)?;
    }

    let out_shape: Vec<i64> = before
        .iter()
        .copied()
        .chain(std::iter::once(w))
        .chain(after.iter().copied())
        .collect();
    let out_shape_py = PyList::new_bound(py, out_shape.iter().copied());
    let out_args = PyTuple::new_bound(py, &[gathered_nwc.as_ref(), out_shape_py.as_ref()]);
    let out = insert_call_before(py, graph, node, &view_op, out_args.as_ref(), &gathered_nwc)?;
    copy_qparams_only(&gathered_nwc.getattr("meta")?, &out.getattr("meta")?)?;

    let result = if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[out.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", bool_dtype)?;
        insert_call_before_with_kwargs(
            py,
            graph,
            node,
            &to_copy_op,
            cast_args.as_ref(),
            &cast_kwargs,
            &out,
            true,
        )?
    } else {
        out
    };

    let final_meta = copied_meta(py, node, false)?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
