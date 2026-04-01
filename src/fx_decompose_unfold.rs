use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{propagate_qparams, FxDecomposer};
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.unfold_copy` into
    /// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
    FxDecomposeUnfoldToGatherPass, "FxDecomposeUnfoldToGather",
    matches: |target, _node| target.contains("unfold_copy"),
    transform: |py, graph, gm, node, debug| decompose_unfold_copy(py, &graph, gm, node, debug).map(|_| ())
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
    gm: &'py FxGraphModule<'py>,
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

    let d = FxDecomposer::new(py, graph, gm, node);

    // Optional bool -> int8 cast
    let mut x_for_gather = x.clone();
    if needs_bool_cast {
        x_for_gather = d.cast(x, &int8_dtype)?;
        mark_layout_neutral(py, &x_for_gather)?;
    }

    // Flatten input to (P, K, Q) for GATHER
    let values_pkq = d.view(&x_for_gather, &[p, k, q])?;
    mark_layout_neutral(py, &values_pkq)?;
    propagate_qparams(&x_for_gather, &values_pkq)?;

    // Build sliding-window index: arange(0, U) * step  +  arange(0, size)
    //   idx_u_scaled: shape (U,) with values [0, step, 2*step, ...]
    let idx_u = d.arange(0, u, &int32_dtype)?;
    mark_layout_neutral(py, &idx_u)?;

    let step_tensor = d.full_int(&[1], step, &int32_dtype)?;
    mark_layout_neutral(py, &step_tensor)?;

    let idx_u_scaled = d.mul(&idx_u, &step_tensor)?;
    mark_layout_neutral(py, &idx_u_scaled)?;

    //   Reshape to (U, 1) for broadcast add
    let idx_u2d = d.unsqueeze(&idx_u_scaled, 1)?;
    mark_layout_neutral(py, &idx_u2d)?;

    //   idx_c: shape (size,) with values [0, 1, ..., size-1]
    let idx_c = d.arange(0, size, &int32_dtype)?;
    mark_layout_neutral(py, &idx_c)?;

    //   Reshape to (1, size) for broadcast add
    let idx_c2d = d.unsqueeze(&idx_c, 0)?;
    mark_layout_neutral(py, &idx_c2d)?;

    //   2D index (U, size) via broadcast add, then flatten to (U*size,)
    let idx2d = d.add(&idx_u2d, &idx_c2d)?;
    mark_layout_neutral(py, &idx2d)?;

    let idx1d = d.view(&idx2d, &[uc])?;
    mark_layout_neutral(py, &idx1d)?;

    //   Unsqueeze to (1, U*size), then expand to (P, U*size)
    let idx_1xuc = d.unsqueeze(&idx1d, 0)?;
    mark_layout_neutral(py, &idx_1xuc)?;

    let indices_puc = d.expand(&idx_1xuc, &[p, uc])?;
    mark_layout_neutral(py, &indices_puc)?;

    // GATHER(values_pkq, indices_puc) -> (P, U*size, Q)
    let gathered = d.gather(&values_pkq, &indices_puc)?;
    mark_layout_neutral(py, &gathered)?;
    {
        d.set_meta_val_shape(&x_for_gather, &gathered, &[p, uc, q])?;
        copy_qparams_only(&x_for_gather.getattr("meta")?, &gathered.getattr("meta")?)?;
    }

    // Reshape (P, U*size, Q) -> (P, U, size, Q)
    let pucq = d.view(&gathered, &[p, u, size, q])?;
    mark_layout_neutral(py, &pucq)?;
    copy_qparams_only(&gathered.getattr("meta")?, &pucq.getattr("meta")?)?;

    // Reshape to (...pre, U, size, ...post) — collapse P and Q back
    let mut shaped = if post.is_empty() {
        let out = d.view(&pucq, &[p, u, size])?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&pucq.getattr("meta")?, &out.getattr("meta")?)?;
        out
    } else {
        let shape: Vec<i64> = std::iter::once(p)
            .chain(std::iter::once(u))
            .chain(std::iter::once(size))
            .chain(post.iter().copied())
            .collect();
        let out = d.view(&pucq, &shape)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&pucq.getattr("meta")?, &out.getattr("meta")?)?;
        out
    };

    // Expand P back to pre dims
    {
        let shape: Vec<i64> = pre
            .iter()
            .copied()
            .chain(std::iter::once(u))
            .chain(std::iter::once(size))
            .chain(post.iter().copied())
            .collect();
        let out = d.view(&shaped, &shape)?;
        mark_layout_neutral(py, &out)?;
        copy_qparams_only(&shaped.getattr("meta")?, &out.getattr("meta")?)?;
        shaped = out;
    }

    // Permute to move the `size` dimension to the end:
    //   (...pre, U, size, ...post) -> (...pre, U, ...post, size)
    let pre_len = pre.len() as i64;
    let post_len = post.len() as i64;
    let rank_cur = pre_len + 1 + 1 + post_len;
    let c_idx = pre_len + 1;
    let perm: Vec<i64> = (0..pre_len + 1)
        .chain(c_idx + 1..rank_cur)
        .chain(std::iter::once(c_idx))
        .collect();
    let permuted = d.permute(&shaped, &perm)?;
    mark_layout_neutral(py, &permuted)?;
    copy_qparams_only(&shaped.getattr("meta")?, &permuted.getattr("meta")?)?;

    // Optional int8 -> bool cast back
    let result = if needs_bool_cast {
        d.cast(&permuted, &bool_dtype)?
    } else {
        permuted
    };
    mark_layout_neutral(py, &result)?;

    crate::fx_utils::copy_meta(py, node, &result)?;
    mark_layout_neutral(py, &result)?;
    for user in &original_users {
        mark_layout_neutral(py, user)?;
    }

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
