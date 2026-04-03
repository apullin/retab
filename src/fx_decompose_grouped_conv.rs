use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{resolve_aten_op, NodeInserter};
use crate::fx_graph::{py_attr, FxGraphModule};

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("conv2d.default")
}

/// Check whether a convolution node is a grouped (non-depthwise) conv with groups > 1.
fn is_grouped_conv(target: &str, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_convolution(target) {
        return Ok(false);
    }

    let args = FxGraphModule::node_args(node)?;
    let is_raw_conv2d = target.contains("conv2d.default");

    // Need enough args to extract groups
    if !is_raw_conv2d && args.len() < 9 {
        return Ok(false);
    }
    if is_raw_conv2d && args.len() < 2 {
        return Ok(false);
    }

    let groups: i64 = if is_raw_conv2d {
        args.get(6).and_then(|a| a.extract().ok()).unwrap_or(1)
    } else {
        args[8].extract().unwrap_or(1)
    };

    if groups <= 1 {
        return Ok(false);
    }

    // Check if depthwise (in_channels == groups) -- skip those
    let input_node = &args[0];
    let in_channels = FxGraphModule::node_shape(input_node)
        .ok()
        .and_then(|s| s.get(1).copied())
        .unwrap_or(0);

    let transposed: bool = if is_raw_conv2d {
        false
    } else {
        args[6].extract().unwrap_or(false)
    };

    if !transposed && in_channels == groups {
        // Depthwise -- handled by FxRewriteConvPass
        return Ok(false);
    }

    Ok(true)
}

fn is_per_channel_qparams(qarg: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(py_attr(qarg, &["per_channel"], false))
}

fn split_per_channel_qparams<'py>(
    py: Python<'py>,
    qarg: &Bound<'py, PyAny>,
    start_index: i64,
    stop_index: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let scales: Vec<f64> = qarg.getattr("scale")?.extract()?;
    let zps: Vec<i64> = qarg.getattr("zp")?.extract()?;
    let qmin: i64 = qarg.getattr("qmin")?.extract()?;
    let qmax: i64 = qarg.getattr("qmax")?.extract()?;
    let dtype = qarg.getattr("dtype")?;
    let axis: i64 = py_attr(qarg, &["axis"], 0);
    let quant_args = py
        .import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")?;

    let start = start_index.max(0) as usize;
    let stop = stop_index.max(start_index).max(0) as usize;
    let scale_slice = scales[start..stop].to_vec();
    let zp_slice = zps[start..stop].to_vec();

    quant_args.call1((scale_slice, zp_slice, qmin, qmax, dtype, axis, true))
}

fn configure_grouped_conv_qparams<'py>(
    py: Python<'py>,
    src_node: &Bound<'py, PyAny>,
    conv_node: &Bound<'py, PyAny>,
    group_index: i64,
    output_slice_size: i64,
    transposed: bool,
) -> PyResult<()> {
    let src_meta = src_node.getattr("meta")?;
    let conv_meta = conv_node.getattr("meta")?;

    if let Ok(output_qparams) = src_meta.call_method1("get", ("output_qparams",)) {
        if !output_qparams.is_none() {
            // Match Python DecomposeGroupedConvPass: per-group convs keep the
            // original output-domain qparams, while only weight/bias input
            // qparams are sliced to the per-group channel range.
            conv_meta.set_item("output_qparams", output_qparams.call_method0("copy")?)?;
        }
    }

    let Ok(input_qparams) = src_meta.call_method1("get", ("input_qparams",)) else {
        return Ok(());
    };
    if input_qparams.is_none() || input_qparams.len().unwrap_or(0) == 0 {
        return Ok(());
    }

    let new_qparams = input_qparams.call_method0("copy")?;

    if let Ok(weight_qarg) = new_qparams.get_item(1) {
        if is_per_channel_qparams(&weight_qarg)? {
            if transposed {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Grouped transpose conv with per-channel quantization is unsupported",
                ));
            }
            let start_index = group_index * output_slice_size;
            let stop_index = (group_index + 1) * output_slice_size;
            let split = split_per_channel_qparams(py, &weight_qarg, start_index, stop_index)?;
            new_qparams.set_item(1, split)?;
        }
    }

    if let Ok(bias_qarg) = new_qparams.get_item(2) {
        if is_per_channel_qparams(&bias_qarg)? {
            let start_index = group_index * output_slice_size;
            let stop_index = (group_index + 1) * output_slice_size;
            let split = split_per_channel_qparams(py, &bias_qarg, start_index, stop_index)?;
            new_qparams.set_item(2, split)?;
        }
    }

    conv_meta.set_item("input_qparams", &new_qparams)?;
    restamp_grouped_conv_output_meta(&src_meta, &conv_meta, output_slice_size)?;
    Ok(())
}

fn restamp_grouped_conv_output_meta(
    src_meta: &Bound<'_, PyAny>,
    conv_meta: &Bound<'_, PyAny>,
    output_slice_size: i64,
) -> PyResult<()> {
    let mut output_val = match src_meta.call_method1("get", ("val",)) {
        Ok(val) if !val.is_none() => val,
        _ => return Ok(()),
    };
    if let Ok(seq) = output_val.downcast::<pyo3::types::PyTuple>() {
        if let Ok(first) = seq.get_item(0) {
            output_val = first;
        }
    } else if let Ok(seq) = output_val.downcast::<pyo3::types::PyList>() {
        if let Ok(first) = seq.get_item(0) {
            output_val = first;
        }
    }
    if !output_val.hasattr("shape")? || !output_val.hasattr("new_empty")? {
        return Ok(());
    }

    let mut shape: Vec<i64> = py_attr(&output_val, &["shape"], Vec::new());
    if shape.len() >= 2 {
        shape[1] = output_slice_size;
        let dtype = output_val.getattr("dtype")?;
        let kwargs = pyo3::types::PyDict::new_bound(output_val.py());
        kwargs.set_item("dtype", dtype)?;
        let new_val = output_val.call_method("new_empty", (shape,), Some(&kwargs))?;
        conv_meta.set_item("val", new_val)?;
    }
    Ok(())
}

fn restamp_grouped_cat_qparams<'py>(
    py: Python<'py>,
    final_meta: &Bound<'py, PyAny>,
    output_slices: &[Bound<'py, PyAny>],
) -> PyResult<()> {
    let copy_mod = py.import_bound("copy")?;
    final_meta.set_item("input_qparams", pyo3::types::PyDict::new_bound(py))?;

    if output_slices.is_empty() {
        return Ok(());
    }

    let cat_output_qparams = pyo3::types::PyDict::new_bound(py);

    for slice in output_slices.iter() {
        let slice_meta = slice.getattr("meta")?;
        let output_qparams = match slice_meta.call_method1("get", ("output_qparams",)) {
            Ok(v) if !v.is_none() => v,
            _ => continue,
        };
        if output_qparams.len().unwrap_or(0) == 0 {
            continue;
        }
        let qp0 = output_qparams.get_item(0)?;
        let qp_copy = copy_mod.call_method1("copy", (&qp0,))?;
        cat_output_qparams.set_item(0, qp_copy)?;
        break;
    }
    if cat_output_qparams.len() > 0 {
        final_meta.set_item("output_qparams", cat_output_qparams)?;
    }
    Ok(())
}

fn get_single_output_qparam<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let meta = node.getattr("meta")?;
    let output_qparams = match meta.call_method1("get", ("output_qparams",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(None),
    };
    if output_qparams.len().unwrap_or(0) == 0 {
        return Ok(None);
    }
    Ok(Some(output_qparams.get_item(0)?))
}

fn same_per_tensor_qparam(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<bool> {
    let lhs_pc: bool = py_attr(lhs, &["per_channel"], false);
    let rhs_pc: bool = py_attr(rhs, &["per_channel"], false);
    if lhs_pc || rhs_pc {
        return Ok(false);
    }

    let lhs_scale: f64 = py_attr(lhs, &["scale"], 0.0);
    let rhs_scale: f64 = py_attr(rhs, &["scale"], 0.0);
    let lhs_zp: i64 = py_attr(lhs, &["zp"], 0);
    let rhs_zp: i64 = py_attr(rhs, &["zp"], 0);
    let lhs_dtype = lhs.getattr("dtype")?.str()?.to_string();
    let rhs_dtype = rhs.getattr("dtype")?.str()?.to_string();

    Ok(lhs_scale == rhs_scale && lhs_zp == rhs_zp && lhs_dtype == rhs_dtype)
}

fn bypass_redundant_grouped_output_boundaries(
    gm: &FxGraphModule<'_>,
    original_node: &Bound<'_, PyAny>,
    cat_node: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let Some(cat_output_qparam) = get_single_output_qparam(cat_node)? else {
        return Ok(());
    };

    for user in FxGraphModule::node_users(original_node)? {
        if FxGraphModule::node_op(&user)? != "call_function" {
            continue;
        }

        let target = FxGraphModule::node_target_str(&user)?;
        if !target.contains("tosa.RESCALE.default")
            && !target.contains("quantize_per_tensor.default")
        {
            continue;
        }

        let args = FxGraphModule::node_args(&user)?;
        if args.is_empty() || !args[0].is(original_node) {
            continue;
        }

        let Some(user_output_qparam) = get_single_output_qparam(&user)? else {
            continue;
        };
        if !same_per_tensor_qparam(&cat_output_qparam, &user_output_qparam)? {
            continue;
        }

        FxGraphModule::replace_all_uses_with(&user, cat_node)?;
        gm.erase_node(&user)?;
    }

    Ok(())
}

fn grouped_conv_debug_enabled() -> bool {
    std::env::var("ARM_TOSA_ENGINE_DEBUG_GROUPED_CONV").is_ok()
}

fn log_grouped_cat_meta(
    cat_node: &Bound<'_, PyAny>,
    output_slices: &[Bound<'_, PyAny>],
) -> PyResult<()> {
    if !grouped_conv_debug_enabled() {
        return Ok(());
    }

    let cat_meta = cat_node.getattr("meta")?;
    eprintln!(
        "[GroupedConvDebug] cat={} qin={} qout={}",
        FxGraphModule::node_name(cat_node).unwrap_or_else(|_| "<unknown>".into()),
        cat_meta.call_method1("get", ("input_qparams",))?.str()?,
        cat_meta.call_method1("get", ("output_qparams",))?.str()?,
    );
    for slice in output_slices {
        let meta = slice.getattr("meta")?;
        eprintln!(
            "[GroupedConvDebug] slice={} qout={}",
            FxGraphModule::node_name(slice).unwrap_or_else(|_| "<unknown>".into()),
            meta.call_method1("get", ("output_qparams",))?.str()?,
        );
    }
    Ok(())
}

crate::fx_pass!(
    /// FX-graph-direct decomposition of grouped convolutions into
    /// `slice_copy -> conv2d (groups=1) -> cat`.
    ///
    /// TOSA does not support grouped convolutions (groups > 1) that are not
    /// depthwise. This pass splits them into per-group convolutions using
    /// slice_copy on input/weight/bias, then concatenates the results.
    ///
    /// Matches Python `DecomposeGroupedConvPass`.
    FxDecomposeGroupedConvPass, "FxDecomposeGroupedConv",
    matches: |target, node| is_grouped_conv(&target, &node).unwrap_or(false),
    transform: |py, graph, gm, node, debug| {
        decompose_grouped_conv(py, &graph, gm, node, debug)
    }
);

fn decompose_grouped_conv<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    let target = FxGraphModule::node_target_str(node)?;
    let is_raw_conv2d = target.contains("conv2d.default");

    let input_node = &args[0];
    let weight_node = &args[1];
    let bias_node = &args[2];

    let groups: i64 = if is_raw_conv2d {
        args.get(6).and_then(|a| a.extract().ok()).unwrap_or(1)
    } else {
        args[8].extract().unwrap_or(1)
    };

    let transposed: bool = if is_raw_conv2d {
        false
    } else {
        args[6].extract().unwrap_or(false)
    };

    let weight_shape = FxGraphModule::node_shape(weight_node).unwrap_or_default();

    // Compute slice sizes from weight shape
    // weight shape: [out_channels, in_channels/groups, kH, kW]
    // For transposed conv: [in_channels, out_channels/groups, kH, kW]
    let (input_slice_size, output_slice_size) = if transposed {
        let input_slice = weight_shape.first().copied().unwrap_or(0) / groups;
        let output_slice = weight_shape.get(1).copied().unwrap_or(0);
        (input_slice, output_slice)
    } else {
        let input_slice = weight_shape.get(1).copied().unwrap_or(0);
        let output_slice = weight_shape.first().copied().unwrap_or(0) / groups;
        (input_slice, output_slice)
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeGroupedConv] {}: groups={}, in_slice={}, out_slice={}, transposed={}",
            name, groups, input_slice_size, output_slice_size, transposed
        );
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    // Resolve the conv op to use for the per-group convolutions
    let conv_op = if is_raw_conv2d {
        resolve_aten_op(py, "conv2d.default")?
    } else {
        resolve_aten_op(py, "convolution.default")?
    };

    let ins = NodeInserter::before(py, graph, node);
    let mut output_slices: Vec<Bound<'py, PyAny>> = Vec::new();

    for i in 0..groups {
        // Slice input along channel dim (dim=1)
        let in_start = i * input_slice_size;
        let in_stop = (i + 1) * input_slice_size;
        let input_slice_args = PyTuple::new_bound(
            py,
            &[
                input_node.as_ref(),
                1i64.into_py(py).into_bound(py).as_ref(),
                in_start.into_py(py).into_bound(py).as_ref(),
                in_stop.into_py(py).into_bound(py).as_ref(),
            ],
        );
        let input_slice = ins.clone().meta_from(input_node).call(&slice_op, input_slice_args.as_ref())?;

        // Slice weight along dim 0 (output channels for normal, input channels for transposed)
        let (w_start, w_stop) = if transposed {
            (i * input_slice_size, (i + 1) * input_slice_size)
        } else {
            (i * output_slice_size, (i + 1) * output_slice_size)
        };
        let weight_slice_args = PyTuple::new_bound(
            py,
            &[
                weight_node.as_ref(),
                0i64.into_py(py).into_bound(py).as_ref(),
                w_start.into_py(py).into_bound(py).as_ref(),
                w_stop.into_py(py).into_bound(py).as_ref(),
            ],
        );
        let weight_slice = ins.clone().meta_from(weight_node).call(&slice_op, weight_slice_args.as_ref())?;

        // Slice bias along dim 0 (output channels)
        let bias_slice = if bias_node.is_none() {
            py.None().into_bound(py)
        } else {
            let b_start = i * output_slice_size;
            let b_stop = (i + 1) * output_slice_size;
            let bias_slice_args = PyTuple::new_bound(
                py,
                &[
                    bias_node.as_ref(),
                    0i64.into_py(py).into_bound(py).as_ref(),
                    b_start.into_py(py).into_bound(py).as_ref(),
                    b_stop.into_py(py).into_bound(py).as_ref(),
                ],
            );
            ins.clone().meta_from(bias_node).call(&slice_op, bias_slice_args.as_ref())?
        };

        // Build conv args with groups=1
        let conv_args = if is_raw_conv2d {
            // conv2d args: (input, weight, bias, stride, padding, dilation, groups)
            let mut conv_arg_vec: Vec<Bound<'py, PyAny>> = vec![
                input_slice.clone(),
                weight_slice.clone(),
                bias_slice.clone(),
            ];
            // Copy stride, padding, dilation from original args (indices 3..6)
            for idx in 3..6 {
                if let Some(a) = args.get(idx) {
                    conv_arg_vec.push(a.clone());
                }
            }
            // groups=1
            conv_arg_vec.push(1i64.into_py(py).into_bound(py));
            PyTuple::new_bound(py, &conv_arg_vec)
        } else {
            // convolution args: (input, weight, bias, stride, padding, dilation,
            //                    transposed, output_padding, groups)
            let mut conv_arg_vec: Vec<Bound<'py, PyAny>> = vec![
                input_slice.clone(),
                weight_slice.clone(),
                bias_slice.clone(),
            ];
            // Copy stride, padding, dilation, transposed, output_padding (indices 3..8)
            for idx in 3..8 {
                conv_arg_vec.push(args[idx].clone());
            }
            // groups=1
            conv_arg_vec.push(1i64.into_py(py).into_bound(py));
            PyTuple::new_bound(py, &conv_arg_vec)
        };

        let conv_node = ins.call(&conv_op, conv_args.as_ref())?;
        configure_grouped_conv_qparams(py, node, &conv_node, i, output_slice_size, transposed)?;
        output_slices.push(conv_node);
    }

    // Cat all outputs along dim 1
    let slices_list = pyo3::types::PyList::new_bound(py, &output_slices);
    let cat_args = PyTuple::new_bound(
        py,
        &[
            slices_list.as_ref(),
            1i64.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let cat_node = ins.call(&cat_op, cat_args.as_ref())?;
    let cat_meta = cat_node.getattr("meta")?;
    restamp_grouped_cat_qparams(py, &cat_meta, &output_slices)?;
    log_grouped_cat_meta(&cat_node, &output_slices)?;
    bypass_redundant_grouped_output_boundaries(gm, node, &cat_node)?;

    FxGraphModule::replace_all_uses_with(node, &cat_node)?;
    gm.erase_node(node)?;
    Ok(())
}
