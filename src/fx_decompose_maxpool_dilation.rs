use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of dilated max_pool2d into a space-to-batch
    /// envelope around a non-dilated max_pool2d. This mirrors the Python Arm pass
    /// closely enough that the serializer only needs to handle the ordinary pool.
    FxDecomposeMaxPoolDilationPass, "FxDecomposeMaxPoolDilation",
    matches: |target, _node| target.contains("max_pool2d") && !target.contains("with_indices"),
    transform: |py, graph, gm, node, debug| decompose_max_pool_dilation(py, &graph, gm, node)
);

fn copied_meta<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    clear_input_qparams: bool,
    clear_output_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let copy_mod = py.import_bound("copy")?;
    let meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    if clear_input_qparams {
        meta.set_item("input_qparams", PyDict::new_bound(py))?;
    }
    if clear_output_qparams {
        meta.set_item("output_qparams", PyDict::new_bound(py))?;
    }
    Ok(meta)
}

fn set_meta_and_refresh(
    py: Python<'_>,
    inserted: &Bound<'_, PyAny>,
    meta: &Bound<'_, PyAny>,
) -> PyResult<()> {
    inserted.setattr("meta", meta)?;
    fx_utils::refresh_fake_tensor(py, inserted)
}

fn normalize_pair_arg(
    value: Option<&Bound<'_, PyAny>>,
    default: (i64, i64),
) -> PyResult<(i64, i64)> {
    let Some(value) = value else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok((v, v));
    }
    if let Ok((a, b)) = value.extract::<(i64, i64)>() {
        return Ok((a, b));
    }
    if let Ok(v) = value.extract::<Vec<i64>>() {
        return Ok(match v.as_slice() {
            [] => default,
            [a] => (*a, *a),
            [a, b, ..] => (*a, *b),
        });
    }
    Ok(default)
}

fn bool_arg(value: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    let Some(value) = value else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    value.extract::<bool>().or(Ok(default))
}

fn pack_dimension(
    dilation: i64,
    padding: i64,
    original_size: i64,
    kernel_size: i64,
    stride: i64,
) -> (i64, i64, i64) {
    let padded_size = original_size + padding * 2;
    let extra_padding = (dilation - (padded_size % dilation)) % dilation;
    let packed_dim_size = (padded_size + extra_padding) / dilation;
    let effective_kernel_size = dilation * (kernel_size - 1) + 1;
    let output_size = if padded_size < effective_kernel_size {
        0
    } else {
        1 + (padded_size - effective_kernel_size) / stride
    };
    (packed_dim_size, padding + extra_padding, output_size)
}

fn pad_value_for_input<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let meta = input.getattr("meta")?;
    let fake = meta.get_item("val")?;
    let dtype = fake.getattr("dtype")?;
    let is_fp = dtype.getattr("is_floating_point")?.extract::<bool>()?;
    if is_fp {
        Ok(f64::NEG_INFINITY.into_py(py).into_bound(py))
    } else {
        let torch = py.import_bound("torch")?;
        let iinfo = torch.call_method1("iinfo", (dtype,))?;
        Ok(iinfo.getattr("min")?)
    }
}

fn decompose_max_pool_dilation<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    let Some(x) = args.first() else {
        return Ok(());
    };

    let input_shape = FxGraphModule::node_shape(x)?;
    if input_shape.len() < 4 {
        return Ok(());
    }
    let [n, c, h, w]: [i64; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    let (k_h, k_w) = normalize_pair_arg(args.get(1), (1, 1))?;
    let (s_h, s_w) = normalize_pair_arg(args.get(2), (k_h, k_w))?;
    let (pad_h, pad_w) = normalize_pair_arg(args.get(3), (0, 0))?;
    let (d_h, d_w) = normalize_pair_arg(args.get(4), (1, 1))?;
    let ceil_mode = bool_arg(args.get(5), false)?;

    if d_h == 1 && d_w == 1 {
        return Ok(());
    }

    let (h_pack, bottom_padding, h_final) = pack_dimension(d_h, pad_h, h, k_h, s_h);
    let (w_pack, right_padding, w_final) = pack_dimension(d_w, pad_w, w, k_w, s_w);

    let pad_op = resolve_aten_op(py, "constant_pad_nd.default")?;
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let permute_op = resolve_aten_op(py, "permute_copy.default")?;
    let pool_op = resolve_aten_op(py, "max_pool2d.default")?;
    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;

    let meta_with_qparams = copied_meta(py, node, false, false)?;
    let meta_without_qparams = copied_meta(py, node, true, true)?;
    let meta_without_output_qparams = copied_meta(py, node, false, true)?;

    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;

    let pad_value = pad_value_for_input(py, x)?;
    let pads = PyList::new_bound(
        py,
        [pad_w, right_padding, pad_h, bottom_padding, 0, 0, 0, 0],
    );
    let pad_args = PyTuple::new_bound(py, &[x.as_ref(), pads.as_ref(), pad_value.as_ref()]);
    let x_pad = graph.call_method1("call_function", (&pad_op, &pad_args))?;
    set_meta_and_refresh(py, &x_pad, &meta_without_output_qparams)?;

    let pack_shape = PyList::new_bound(py, [n, c, h_pack, d_h, w_pack, d_w]);
    let pack_args = PyTuple::new_bound(py, &[x_pad.as_ref(), pack_shape.as_ref()]);
    let x_pack = graph.call_method1("call_function", (&view_op, &pack_args))?;
    set_meta_and_refresh(py, &x_pack, &meta_without_qparams)?;

    let pack_perm = PyList::new_bound(py, [3, 5, 0, 1, 2, 4]);
    let perm_args = PyTuple::new_bound(py, &[x_pack.as_ref(), pack_perm.as_ref()]);
    let x_perm = graph.call_method1("call_function", (&permute_op, &perm_args))?;
    set_meta_and_refresh(py, &x_perm, &meta_without_qparams)?;

    let batch_shape = PyList::new_bound(py, [n * d_h * d_w, c, h_pack, w_pack]);
    let batch_args = PyTuple::new_bound(py, &[x_perm.as_ref(), batch_shape.as_ref()]);
    let x_batch = graph.call_method1("call_function", (&view_op, &batch_args))?;
    set_meta_and_refresh(py, &x_batch, &meta_without_qparams)?;

    let kernel = PyList::new_bound(py, [k_h, k_w]);
    let stride = PyList::new_bound(py, [s_h, s_w]);
    let zero_pad = PyList::new_bound(py, [0, 0]);
    let pool_args = PyTuple::new_bound(
        py,
        &[
            x_batch.as_ref(),
            kernel.as_ref(),
            stride.as_ref(),
            zero_pad.as_ref(),
            1i64.into_py(py).into_bound(py).as_ref(),
            ceil_mode.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let pooled = graph.call_method1("call_function", (&pool_op, &pool_args))?;
    set_meta_and_refresh(py, &pooled, &meta_with_qparams)?;
    pooled.getattr("meta")?.set_item("skip_size_adjust", true)?;

    let pooled_shape = FxGraphModule::node_shape(&pooled)?;
    if pooled_shape.len() < 4 {
        inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
        return Ok(());
    }
    let c_out = pooled_shape[1];
    let h_out = pooled_shape[2];
    let w_out = pooled_shape[3];

    let unpack_shape = PyList::new_bound(py, [d_h, d_w, n, c_out, h_out, w_out]);
    let unpack_args = PyTuple::new_bound(py, &[pooled.as_ref(), unpack_shape.as_ref()]);
    let unpack = graph.call_method1("call_function", (&view_op, &unpack_args))?;
    set_meta_and_refresh(py, &unpack, &meta_without_qparams)?;

    let unpack_perm = PyList::new_bound(py, [2, 3, 4, 0, 5, 1]);
    let unpack_perm_args = PyTuple::new_bound(py, &[unpack.as_ref(), unpack_perm.as_ref()]);
    let unpack_perm_node = graph.call_method1("call_function", (&permute_op, &unpack_perm_args))?;
    set_meta_and_refresh(py, &unpack_perm_node, &meta_without_qparams)?;

    let final_shape = PyList::new_bound(py, [n, c_out, h_out * d_h, w_out * d_w]);
    let final_view_args =
        PyTuple::new_bound(py, &[unpack_perm_node.as_ref(), final_shape.as_ref()]);
    let out = graph.call_method1("call_function", (&view_op, &final_view_args))?;
    set_meta_and_refresh(py, &out, &meta_without_qparams)?;

    let h_slice_args = PyTuple::new_bound(
        py,
        &[
            out.as_ref(),
            2i64.into_py(py).into_bound(py).as_ref(),
            0i64.into_py(py).into_bound(py).as_ref(),
            h_final.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let out_h = graph.call_method1("call_function", (&slice_op, &h_slice_args))?;
    set_meta_and_refresh(py, &out_h, &meta_with_qparams)?;

    let w_slice_args = PyTuple::new_bound(
        py,
        &[
            out_h.as_ref(),
            3i64.into_py(py).into_bound(py).as_ref(),
            0i64.into_py(py).into_bound(py).as_ref(),
            w_final.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let out_hw = graph.call_method1("call_function", (&slice_op, &w_slice_args))?;
    set_meta_and_refresh(py, &out_hw, &meta_with_qparams)?;

    inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

    FxGraphModule::replace_all_uses_with(node, &out_hw)?;
    gm.erase_node(node)?;
    Ok(())
}
