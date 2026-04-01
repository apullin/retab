use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.unfold_copy` into
    /// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
    FxDecomposeUnfoldToGatherPass, "FxDecomposeUnfoldToGather",
    matches: |target, _node| target.contains("unfold_copy"),
    transform: |py, graph, gm, node, debug| decompose_unfold_copy(py, &graph, gm, node, debug).map(|_| ())
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

fn mark_layout_neutral<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<()> {
    let meta = node.getattr("meta")?;
    meta.set_item("arm_layout_neutral", true)?;
    meta.set_item("tosa_spatial_rank", 0)?;

    let val = match meta.call_method1("get", ("val",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(()),
    };
    let shape = match val.getattr("shape") {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let rank: usize = py
        .import_bound("builtins")?
        .call_method1("len", (&shape,))?
        .extract()
        .unwrap_or(0);
    let dim_order = PyTuple::new_bound(py, (0..rank).map(|dim| dim));
    meta.set_item("tosa_dim_order", dim_order)?;
    Ok(())
}

fn product(shape: &[i64]) -> i64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn decompose_unfold_copy<'py>(
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
    let x = &args[0];
    if !FxGraphModule::is_node(x) {
        return Ok(false);
    }

    let dim: i64 = args[1].extract()?;
    let size: i64 = args[2].extract()?;
    let step: i64 = args[3].extract()?;
    if size <= 0 || step <= 0 {
        return Ok(false);
    }

    let input_shape = FxGraphModule::node_shape(x)?;
    let rank = input_shape.len() as i64;
    if rank < 1 {
        return Ok(false);
    }

    let dim_norm = dim.rem_euclid(rank) as usize;
    let k = input_shape[dim_norm];
    if k < size {
        return Ok(false);
    }

    let u = (k - size) / step + 1;
    let uc = u * size;
    let pre = input_shape[..dim_norm].to_vec();
    let post = input_shape[dim_norm + 1..].to_vec();
    let p = product(&pre);
    let q = product(&post);
    let original_users = FxGraphModule::node_users(node)?;

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeUnfoldToGather] {} dim={} size={} step={} input={:?} -> pre={:?} post={:?} P={} K={} Q={} U={} C={}",
            name, dim, size, step, input_shape, pre, post, p, k, q, u, size
        );
    }

    mark_layout_neutral(py, x)?;

    let x_meta = x.getattr("meta")?;
    let x_dtype_str: String = x_meta.get_item("val")?.getattr("dtype")?.str()?.extract()?;
    let needs_bool_cast = x_dtype_str == "torch.bool";

    let torch = py.import_bound("torch")?;
    let int8_dtype = torch.getattr("int8")?;
    let int32_dtype = torch.getattr("int32")?;
    let bool_dtype = torch.getattr("bool")?;

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")?;
    let expand_op = resolve_aten_op(py, "expand_copy.default")?;
    let add_tensor_op = resolve_aten_op(py, "add.Tensor")?;
    let mul_tensor_op = resolve_aten_op(py, "mul.Tensor")?;
    let arange_op = resolve_aten_op(py, "arange.start_step")?;
    let permute_op = resolve_aten_op(py, "permute_copy.default")?;
    let to_copy_op = resolve_aten_op(py, "_to_copy.default")?;
    let full_op = resolve_aten_op(py, "full.default")?;
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
        mark_layout_neutral(py, &x_for_gather)?;
    }

    let values_shape = PyList::new_bound(py, [p, k, q]);
    let values_args = PyTuple::new_bound(py, &[x_for_gather.as_ref(), values_shape.as_ref()]);
    let values_pkq = insert_call_before(
        py,
        graph,
        node,
        &view_op,
        values_args.as_ref(),
        &x_for_gather,
    )?;
    mark_layout_neutral(py, &values_pkq)?;
    copy_qparams_only(&x_for_gather.getattr("meta")?, &values_pkq.getattr("meta")?)?;

    let arange_kwargs = PyDict::new_bound(py);
    arange_kwargs.set_item("dtype", int32_dtype.clone())?;

    let idx_u_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            u.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let idx_u = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &arange_op,
        idx_u_args.as_ref(),
        &arange_kwargs,
        node,
        true,
    )?;
    mark_layout_neutral(py, &idx_u)?;

    let step_shape = PyTuple::new_bound(py, [1i64]);
    let step_value = step.into_py(py).into_bound(py);
    let step_kwargs = PyDict::new_bound(py);
    step_kwargs.set_item("dtype", int32_dtype.clone())?;
    let step_args = PyTuple::new_bound(py, &[step_shape.as_ref(), step_value.as_ref()]);
    let step_tensor = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &full_op,
        step_args.as_ref(),
        &step_kwargs,
        node,
        true,
    )?;
    mark_layout_neutral(py, &step_tensor)?;

    let idx_u_scaled_args = PyTuple::new_bound(py, &[idx_u.as_ref(), step_tensor.as_ref()]);
    let idx_u_scaled = insert_call_before(
        py,
        graph,
        node,
        &mul_tensor_op,
        idx_u_scaled_args.as_ref(),
        node,
    )?;
    mark_layout_neutral(py, &idx_u_scaled)?;

    let idx_u2d_args = PyTuple::new_bound(
        py,
        &[
            idx_u_scaled.as_ref(),
            1i64.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let idx_u2d = insert_call_before(py, graph, node, &unsqueeze_op, idx_u2d_args.as_ref(), node)?;
    mark_layout_neutral(py, &idx_u2d)?;

    let idx_c_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            size.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let idx_c = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &arange_op,
        idx_c_args.as_ref(),
        &arange_kwargs,
        node,
        true,
    )?;
    mark_layout_neutral(py, &idx_c)?;

    let idx_c2d_args = PyTuple::new_bound(
        py,
        &[idx_c.as_ref(), 0i64.into_py(py).into_bound(py).as_ref()],
    );
    let idx_c2d = insert_call_before(py, graph, node, &unsqueeze_op, idx_c2d_args.as_ref(), node)?;
    mark_layout_neutral(py, &idx_c2d)?;

    let idx2d_args = PyTuple::new_bound(py, &[idx_u2d.as_ref(), idx_c2d.as_ref()]);
    let idx2d = insert_call_before(py, graph, node, &add_tensor_op, idx2d_args.as_ref(), node)?;
    mark_layout_neutral(py, &idx2d)?;

    let idx1d_shape = PyList::new_bound(py, [uc]);
    let idx1d_args = PyTuple::new_bound(py, &[idx2d.as_ref(), idx1d_shape.as_ref()]);
    let idx1d = insert_call_before(py, graph, node, &view_op, idx1d_args.as_ref(), node)?;
    mark_layout_neutral(py, &idx1d)?;

    let idx_1xuc_args = PyTuple::new_bound(
        py,
        &[idx1d.as_ref(), 0i64.into_py(py).into_bound(py).as_ref()],
    );
    let idx_1xuc =
        insert_call_before(py, graph, node, &unsqueeze_op, idx_1xuc_args.as_ref(), node)?;
    mark_layout_neutral(py, &idx_1xuc)?;

    let indices_shape = PyList::new_bound(py, [p, uc]);
    let indices_args = PyTuple::new_bound(py, &[idx_1xuc.as_ref(), indices_shape.as_ref()]);
    let indices_puc = insert_call_before(py, graph, node, &expand_op, indices_args.as_ref(), node)?;
    mark_layout_neutral(py, &indices_puc)?;

    let gather_args = PyTuple::new_bound(py, &[values_pkq.as_ref(), indices_puc.as_ref()]);
    let gathered = insert_call_before(py, graph, node, &gather_op, gather_args.as_ref(), node)?;
    mark_layout_neutral(py, &gathered)?;
    {
        let gather_meta = gathered.getattr("meta")?;
        let gather_template_meta = x_for_gather.getattr("meta")?;
        set_tensor_meta_val_shape(&gather_template_meta, &gather_meta, &[p, uc, q])?;
        copy_qparams_only(&gather_template_meta, &gather_meta)?;
    }

    let pucq_shape = PyList::new_bound(py, [p, u, size, q]);
    let pucq_args = PyTuple::new_bound(py, &[gathered.as_ref(), pucq_shape.as_ref()]);
    let pucq = insert_call_before(py, graph, node, &view_op, pucq_args.as_ref(), &gathered)?;
    mark_layout_neutral(py, &pucq)?;
    copy_qparams_only(&gathered.getattr("meta")?, &pucq.getattr("meta")?)?;

    let mut shaped = if post.is_empty() {
        let shape = PyList::new_bound(py, [p, u, size]);
        let args = PyTuple::new_bound(py, &[pucq.as_ref(), shape.as_ref()]);
        let out = insert_call_before(py, graph, node, &view_op, args.as_ref(), &pucq)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&pucq.getattr("meta")?, &out.getattr("meta")?)?;
        out
    } else {
        let shape: Vec<i64> = std::iter::once(p)
            .chain(std::iter::once(u))
            .chain(std::iter::once(size))
            .chain(post.iter().copied())
            .collect();
        let shape_py = PyList::new_bound(py, shape.iter().copied());
        let args = PyTuple::new_bound(py, &[pucq.as_ref(), shape_py.as_ref()]);
        let out = insert_call_before(py, graph, node, &view_op, args.as_ref(), &pucq)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&pucq.getattr("meta")?, &out.getattr("meta")?)?;
        out
    };

    if pre.is_empty() {
        let shape: Vec<i64> = std::iter::once(u)
            .chain(std::iter::once(size))
            .chain(post.iter().copied())
            .collect();
        let shape_py = PyList::new_bound(py, shape.iter().copied());
        let args = PyTuple::new_bound(py, &[shaped.as_ref(), shape_py.as_ref()]);
        let out = insert_call_before(py, graph, node, &view_op, args.as_ref(), &shaped)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&shaped.getattr("meta")?, &out.getattr("meta")?)?;
        shaped = out;
    } else {
        let shape: Vec<i64> = pre
            .iter()
            .copied()
            .chain(std::iter::once(u))
            .chain(std::iter::once(size))
            .chain(post.iter().copied())
            .collect();
        let shape_py = PyList::new_bound(py, shape.iter().copied());
        let args = PyTuple::new_bound(py, &[shaped.as_ref(), shape_py.as_ref()]);
        let out = insert_call_before(py, graph, node, &view_op, args.as_ref(), &shaped)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&shaped.getattr("meta")?, &out.getattr("meta")?)?;
        shaped = out;
    }

    let pre_len = pre.len() as i64;
    let post_len = post.len() as i64;
    let rank_cur = pre_len + 1 + 1 + post_len;
    let c_idx = pre_len + 1;
    let perm: Vec<i64> = (0..pre_len + 1)
        .chain(c_idx + 1..rank_cur)
        .chain(std::iter::once(c_idx))
        .collect();
    let perm_py = PyList::new_bound(py, perm.iter().copied());
    let perm_args = PyTuple::new_bound(py, &[shaped.as_ref(), perm_py.as_ref()]);
    let permuted = insert_call_before(py, graph, node, &permute_op, perm_args.as_ref(), &shaped)?;
    mark_layout_neutral(py, &permuted)?;
    copy_qparams_only(&shaped.getattr("meta")?, &permuted.getattr("meta")?)?;

    let result = if needs_bool_cast {
        let cast_args = PyTuple::new_bound(py, &[permuted.as_ref()]);
        let cast_kwargs = PyDict::new_bound(py);
        cast_kwargs.set_item("dtype", bool_dtype)?;
        insert_call_before_with_kwargs(
            py,
            graph,
            node,
            &to_copy_op,
            cast_args.as_ref(),
            &cast_kwargs,
            &permuted,
            true,
        )?
    } else {
        permuted
    };
    mark_layout_neutral(py, &result)?;

    fx_utils::copy_meta(py, node, &result)?;
    mark_layout_neutral(py, &result)?;
    for user in &original_users {
        mark_layout_neutral(py, user)?;
    }

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
