use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// Mirror the Python Arm backend's cumsum lowering:
    /// reshape -> convolution(with ones kernel) -> slice -> reshape.
    FxDecomposeCumsumPass, "FxDecomposeCumsum",
    matches: |target| target.contains("cumsum"),
    transform: |py, graph, gm, node, debug| {
        decompose_cumsum(py, &graph, gm, node, debug)?;
        Ok(())
    }
);

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

fn insert_call_before<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    meta_src: &Bound<'py, PyAny>,
    clear_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = match kwargs {
        Some(kwargs) => graph.call_method1("call_function", (target, args, kwargs))?,
        None => graph.call_method1("call_function", (target, args))?,
    };
    let meta_copy = copied_meta(py, meta_src, clear_qparams)?;
    new_node.setattr("meta", &meta_copy)?;
    let _ = fx_utils::refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn make_qparam<'py>(
    py: Python<'py>,
    scale: f64,
    zero_point: i64,
    qmin: i64,
    qmax: i64,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")?
        .call1((scale, zero_point, qmin, qmax, dtype))
}

fn set_qparam_map<'py>(
    py: Python<'py>,
    meta: &Bound<'py, PyAny>,
    key: &str,
    entries: &[(usize, Bound<'py, PyAny>)],
) -> PyResult<()> {
    let qparams = PyDict::new_bound(py);
    for (index, qparam) in entries {
        qparams.set_item(*index, qparam)?;
    }
    meta.set_item(key, qparams)?;
    Ok(())
}

fn decompose_cumsum<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 2 || !FxGraphModule::is_node(&args[0]) {
        return Ok(false);
    }

    let input_node = &args[0];
    let original_shape = FxGraphModule::node_shape(input_node)?;
    if original_shape.is_empty() {
        return Ok(false);
    }

    let rank = original_shape.len() as i64;
    let dim_raw: i64 = args[1].extract()?;
    let dim = dim_raw.rem_euclid(rank) as usize;

    let pre_cumsum_dim: i64 = if dim > 0 {
        original_shape[..dim].iter().product()
    } else {
        1
    };
    let cumsum_dim = original_shape[dim];
    let post_cumsum_dim: i64 = if dim < original_shape.len() - 1 {
        original_shape[dim + 1..].iter().product()
    } else {
        1
    };

    if cumsum_dim <= 1 {
        FxGraphModule::replace_all_uses_with(node, input_node)?;
        gm.erase_node(node)?;
        return Ok(true);
    }

    let conv_shape = vec![pre_cumsum_dim, 1, cumsum_dim, post_cumsum_dim];
    let pad_shape = vec![cumsum_dim - 1, 0];
    let weight_shape = vec![1, 1, cumsum_dim, 1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeCumsum] {} dim={} shape={:?} conv_shape={:?}",
            name, dim, original_shape, conv_shape
        );
    }

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let conv_op = resolve_aten_op(py, "convolution.default")?;
    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    let view_shape_py = PyList::new_bound(py, conv_shape.iter().copied());
    let view_args = PyTuple::new_bound(py, &[input_node.as_ref(), view_shape_py.as_ref()]);
    let view_node = insert_call_before(py, graph, node, &view_op, &view_args, None, node, true)?;

    let input_meta = input_node.getattr("meta")?;
    let input_val = input_meta.call_method1("get", ("val",))?;
    let dtype = if !input_val.is_none() {
        input_val.getattr("dtype").ok()
    } else {
        None
    };
    let one_py = 1i64.into_py(py).into_bound(py);

    let weight_shape_py = PyTuple::new_bound(py, weight_shape.iter().copied());
    let weight_args = PyTuple::new_bound(py, &[weight_shape_py.as_ref(), one_py.as_ref()]);
    let weight_kwargs = PyDict::new_bound(py);
    if let Some(dtype) = dtype.as_ref() {
        weight_kwargs.set_item("dtype", dtype)?;
    }
    if let Ok(device) = input_val.getattr("device") {
        if !device.is_none() {
            weight_kwargs.set_item("device", device)?;
        }
    }
    let weight_node = insert_call_before(
        py,
        graph,
        node,
        &full_op,
        &weight_args,
        Some(&weight_kwargs),
        node,
        true,
    )?;

    let stride_py = PyList::new_bound(py, [1i64, 1i64]);
    let pad_py = PyList::new_bound(py, pad_shape.iter().copied());
    let dilation_py = PyList::new_bound(py, [1i64, 1i64]);
    let transposed_false = false.into_py(py).into_bound(py);
    let output_padding_py = PyList::new_bound(py, [0i64]);
    let groups_one = 1i64.into_py(py).into_bound(py);
    let conv_args = PyTuple::new_bound(
        py,
        &[
            view_node.as_ref(),
            weight_node.as_ref(),
            py.None().bind(py).as_ref(),
            stride_py.as_ref(),
            pad_py.as_ref(),
            dilation_py.as_ref(),
            transposed_false.as_ref(),
            output_padding_py.as_ref(),
            groups_one.as_ref(),
        ],
    );
    let conv_node = insert_call_before(py, graph, node, &conv_op, &conv_args, None, node, false)?;

    let node_meta = node.getattr("meta")?;
    let original_input_qparams = node_meta.call_method1("get", ("input_qparams",))?;
    if !original_input_qparams.is_none() && original_input_qparams.len().unwrap_or(0) > 0 {
        let torch_int8 = py.import_bound("torch")?.getattr("int8")?;
        let weight_qp = make_qparam(py, 1.0, 0, -128, 127, &torch_int8)?;

        let conv_meta = conv_node.getattr("meta")?;
        let new_input_qparams = original_input_qparams.call_method0("copy")?;
        new_input_qparams.set_item(1, &weight_qp)?;
        conv_meta.set_item("input_qparams", new_input_qparams)?;

        let weight_meta = weight_node.getattr("meta")?;
        set_qparam_map(py, &weight_meta, "output_qparams", &[(0, weight_qp)])?;
    }

    let slice_dim_py = 2i64.into_py(py).into_bound(py);
    let zero_py = 0i64.into_py(py).into_bound(py);
    let end_py = cumsum_dim.into_py(py).into_bound(py);
    let slice_args = PyTuple::new_bound(
        py,
        &[
            conv_node.as_ref(),
            slice_dim_py.as_ref(),
            zero_py.as_ref(),
            end_py.as_ref(),
        ],
    );
    let slice_node = insert_call_before(py, graph, node, &slice_op, &slice_args, None, node, true)?;

    let original_shape_py = PyList::new_bound(py, original_shape.iter().copied());
    let view_original_args =
        PyTuple::new_bound(py, &[slice_node.as_ref(), original_shape_py.as_ref()]);
    let result = insert_call_before(
        py,
        graph,
        node,
        &view_op,
        &view_original_args,
        None,
        node,
        true,
    )?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
