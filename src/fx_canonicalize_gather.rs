use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct canonicalization of `aten.gather` into
    /// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
    FxCanonicalizeGatherPass, "FxCanonicalizeGather",
    matches: |target| target.contains("aten.gather.default"),
    transform: |py, graph, gm, node, debug| canonicalize_gather(py, &graph, gm, node, debug).map(|_| ())
);

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

fn canonicalize_gather<'py>(
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
    let dim: i64 = args[1].extract()?;
    let index = &args[2];
    if !FxGraphModule::is_node(x) || !FxGraphModule::is_node(index) {
        return Ok(false);
    }

    let x_shape = FxGraphModule::node_shape(x)?;
    let index_shape = FxGraphModule::node_shape(index)?;
    let x_rank = x_shape.len() as i64;
    if x_rank < 1 || !matches!(x_shape.len(), 2 | 3) || !matches!(index_shape.len(), 2 | 3) {
        return Ok(false);
    }

    let dim_norm = dim.rem_euclid(x_rank) as usize;
    let matching_batch = index_shape.first() == x_shape.first();
    if dim_norm != 1 || !matching_batch {
        return Ok(false);
    }

    if x_shape.len() == 3 && (index_shape.len() != 3 || index_shape[2] != x_shape[2]) {
        return Ok(false);
    }

    let n = x_shape[0];
    let k = x_shape[1];
    let w = index_shape[1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[CanonicalizeGather] {} dim={} input={:?} index={:?}",
            name, dim, x_shape, index_shape
        );
    }

    let x_meta = x.getattr("meta")?;
    let x_dtype_str: String = x_meta.get_item("val")?.getattr("dtype")?.str()?.extract()?;
    let needs_bool_cast = x_dtype_str == "torch.bool";

    let torch = py.import_bound("torch")?;
    let int8_dtype = torch.getattr("int8")?;
    let bool_dtype = torch.getattr("bool")?;

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let to_copy_op = resolve_aten_op(py, "_to_copy.default")?;
    let permute_copy_op = resolve_aten_op(py, "permute_copy.default")?;
    let gather_op = resolve_tosa_gather_op(py)?;

    let mut values_in = x.clone();
    if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[x.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", int8_dtype)?;
        values_in = insert_call_before_with_kwargs(
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

    let gathered = if x_shape.len() == 2 {
        let values_shape = PyList::new_bound(py, [n, k, 1]);
        let values_args = PyTuple::new_bound(py, &[values_in.as_ref(), values_shape.as_ref()]);
        let values_3d =
            insert_call_before(py, graph, node, &view_op, values_args.as_ref(), &values_in)?;
        copy_qparams_only(&values_in.getattr("meta")?, &values_3d.getattr("meta")?)?;

        let gather_args = PyTuple::new_bound(py, &[values_3d.as_ref(), index.as_ref()]);
        let gathered_3d =
            insert_call_before(py, graph, node, &gather_op, gather_args.as_ref(), node)?;
        {
            let gather_meta = gathered_3d.getattr("meta")?;
            let gather_template_meta = values_in.getattr("meta")?;
            set_tensor_meta_val_shape(&gather_template_meta, &gather_meta, &[n, w, 1])?;
            copy_qparams_only(&gather_template_meta, &gather_meta)?;
        }

        let out_shape = PyList::new_bound(py, [n, w]);
        let out_args = PyTuple::new_bound(py, &[gathered_3d.as_ref(), out_shape.as_ref()]);
        let out = insert_call_before(py, graph, node, &view_op, out_args.as_ref(), &gathered_3d)?;
        copy_qparams_only(&gathered_3d.getattr("meta")?, &out.getattr("meta")?)?;
        out
    } else {
        let c = x_shape[2];

        let values_perm = PyList::new_bound(py, [0, 2, 1]);
        let values_perm_args = PyTuple::new_bound(py, &[values_in.as_ref(), values_perm.as_ref()]);
        let values_permuted = insert_call_before(
            py,
            graph,
            node,
            &permute_copy_op,
            values_perm_args.as_ref(),
            &values_in,
        )?;
        copy_qparams_only(
            &values_in.getattr("meta")?,
            &values_permuted.getattr("meta")?,
        )?;

        let values_shape = PyList::new_bound(py, [n * c, k, 1]);
        let values_shape_args =
            PyTuple::new_bound(py, &[values_permuted.as_ref(), values_shape.as_ref()]);
        let values_reshaped = insert_call_before(
            py,
            graph,
            node,
            &view_op,
            values_shape_args.as_ref(),
            &values_permuted,
        )?;
        copy_qparams_only(
            &values_permuted.getattr("meta")?,
            &values_reshaped.getattr("meta")?,
        )?;

        let indices_perm = PyList::new_bound(py, [0, 2, 1]);
        let indices_perm_args = PyTuple::new_bound(py, &[index.as_ref(), indices_perm.as_ref()]);
        let indices_permuted = insert_call_before(
            py,
            graph,
            node,
            &permute_copy_op,
            indices_perm_args.as_ref(),
            index,
        )?;

        let indices_shape = PyList::new_bound(py, [n * c, w]);
        let indices_shape_args =
            PyTuple::new_bound(py, &[indices_permuted.as_ref(), indices_shape.as_ref()]);
        let indices_reshaped = insert_call_before(
            py,
            graph,
            node,
            &view_op,
            indices_shape_args.as_ref(),
            &indices_permuted,
        )?;

        let gather_args =
            PyTuple::new_bound(py, &[values_reshaped.as_ref(), indices_reshaped.as_ref()]);
        let gathered_raw =
            insert_call_before(py, graph, node, &gather_op, gather_args.as_ref(), node)?;
        {
            let gather_meta = gathered_raw.getattr("meta")?;
            let gather_template_meta = values_in.getattr("meta")?;
            set_tensor_meta_val_shape(&gather_template_meta, &gather_meta, &[n * c, w, 1])?;
            copy_qparams_only(&gather_template_meta, &gather_meta)?;
        }

        let gathered_shape = PyList::new_bound(py, [n, c, w]);
        let gathered_shape_args =
            PyTuple::new_bound(py, &[gathered_raw.as_ref(), gathered_shape.as_ref()]);
        let gathered_reshaped = insert_call_before(
            py,
            graph,
            node,
            &view_op,
            gathered_shape_args.as_ref(),
            &gathered_raw,
        )?;
        copy_qparams_only(
            &gathered_raw.getattr("meta")?,
            &gathered_reshaped.getattr("meta")?,
        )?;

        let out_perm = PyList::new_bound(py, [0, 2, 1]);
        let out_perm_args =
            PyTuple::new_bound(py, &[gathered_reshaped.as_ref(), out_perm.as_ref()]);
        let out = insert_call_before(
            py,
            graph,
            node,
            &permute_copy_op,
            out_perm_args.as_ref(),
            &gathered_reshaped,
        )?;
        copy_qparams_only(&gathered_reshaped.getattr("meta")?, &out.getattr("meta")?)?;
        out
    };

    let result = if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[gathered.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", bool_dtype)?;
        insert_call_before_with_kwargs(
            py,
            graph,
            node,
            &to_copy_op,
            cast_args.as_ref(),
            &cast_kwargs,
            &gathered,
            true,
        )?
    } else {
        gathered
    };

    fx_utils::copy_meta(py, node, &result)?;
    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
