use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// Mirror the Python Arm backend's cumsum lowering:
    /// reshape -> convolution(with ones kernel) -> slice -> reshape.
    FxDecomposeCumsumPass, "FxDecomposeCumsum",
    matches: |target, _node| target.contains("cumsum"),
    transform: |py, graph, gm, node, debug| {
        decompose_cumsum(py, &graph, gm, node, debug)?;
        Ok(())
    }
);


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

    let d = FxDecomposer::new(py, graph, gm, node);

    // view(input, conv_shape)
    let view_node = d.view(input_node, &conv_shape)?;

    // Extract dtype and device from the input tensor's meta val
    let input_meta = input_node.getattr("meta")?;
    let input_val = input_meta.call_method1("get", ("val",))?;
    let dtype = if !input_val.is_none() {
        input_val.getattr("dtype").ok()
    } else {
        None
    };
    let device = if !input_val.is_none() {
        input_val.getattr("device").ok().filter(|d| !d.is_none())
    } else {
        None
    };

    // full(weight_shape, 1.0, dtype=dtype, device=device)
    let weight_node = d.full_with_device(
        &weight_shape,
        1.0,
        dtype.as_ref(),
        device.as_ref(),
    )?;

    // convolution(view, weight, None, [1,1], [cumsum_dim-1, 0], [1,1], false, [0], 1)
    let conv_node = d.convolution(
        &view_node,
        &weight_node,
        None,
        &[1, 1],
        &pad_shape,
        &[1, 1],
        false,
        &[0],
        1,
    )?;

    // Quantization parameter fixup for the convolution weight
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

    // slice(conv, dim=2, start=0, end=cumsum_dim)
    let slice_node = d.slice(&conv_node, 2, 0, cumsum_dim)?;

    // view(slice, original_shape)
    let result = d.view(&slice_node, &original_shape)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
