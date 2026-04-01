use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{resolve_aten_op, NodeInserter};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.index_select` into
    /// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
    FxDecomposeIndexSelectToGatherPass, "FxDecomposeIndexSelectToGather",
    matches: |target, _node| target.contains("index_select.default"),
    transform: |py, graph, gm, node, debug| decompose_index_select(py, &graph, gm, node, debug).map(|_| ())
);

fn resolve_tosa_gather_op<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.eval_bound(
        "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'backend'), 'tosa'), 'GATHER').default",
        None,
        None,
    )
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

    let ins = NodeInserter::before(py, graph, node);

    let mut x_for_gather = x.clone();
    if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[x.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", int8_dtype)?;
        x_for_gather = ins.clone().meta_from(x).call_with_kwargs(&to_copy_op, cast_args.as_ref(), &cast_kwargs)?;
    }

    let values_shape = PyList::new_bound(py, [n, k, c]);
    let values_args = PyTuple::new_bound(py, &[x_for_gather.as_ref(), values_shape.as_ref()]);
    let values_nkc = ins.clone().meta_from(&x_for_gather).call(&view_op, values_args.as_ref())?;
    copy_qparams_only(&x_for_gather.getattr("meta")?, &values_nkc.getattr("meta")?)?;

    let idx_1w_args = PyTuple::new_bound(
        py,
        &[index.as_ref(), 0i64.into_py(py).into_bound(py).as_ref()],
    );
    let idx_1w = ins.clone().meta_from(index).call(&unsqueeze_op, idx_1w_args.as_ref())?;

    let indices_shape = PyList::new_bound(py, [n, w]);
    let indices_args = PyTuple::new_bound(py, &[idx_1w.as_ref(), indices_shape.as_ref()]);
    let indices_nw = ins.clone().meta_from(&idx_1w).call(&expand_op, indices_args.as_ref())?;

    let gather_args = PyTuple::new_bound(py, &[values_nkc.as_ref(), indices_nw.as_ref()]);
    let gathered_nwc = ins.call(&gather_op, gather_args.as_ref())?;
    {
        let gather_meta = gathered_nwc.getattr("meta")?;
        let gather_template_meta = x_for_gather.getattr("meta")?;
        set_tensor_meta_val_shape(&gather_template_meta, &gather_meta, &[n, w, c])?;
        copy_qparams_only(&gather_template_meta, &gather_meta)?;
    }

    let ins = NodeInserter::before(py, graph, node);
    let out_shape: Vec<i64> = before
        .iter()
        .copied()
        .chain(std::iter::once(w))
        .chain(after.iter().copied())
        .collect();
    let out_shape_py = PyList::new_bound(py, out_shape.iter().copied());
    let out_args = PyTuple::new_bound(py, &[gathered_nwc.as_ref(), out_shape_py.as_ref()]);
    let out = ins.clone().meta_from(&gathered_nwc).call(&view_op, out_args.as_ref())?;
    copy_qparams_only(&gathered_nwc.getattr("meta")?, &out.getattr("meta")?)?;

    let result = if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[out.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", bool_dtype)?;
        ins.clone().meta_from(&out).call_with_kwargs(&to_copy_op, cast_args.as_ref(), &cast_kwargs)?
    } else {
        out
    };

    fx_utils::copy_meta(py, node, &result)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
