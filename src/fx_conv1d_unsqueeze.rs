use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

fn is_conv1d(target: &str, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !(target.contains("convolution.default") || target.contains("conv1d.default")) {
        return Ok(false);
    }

    // For conv1d.default, the op is always 1D regardless of arg count
    // (optional args stride/padding/dilation/groups may be omitted).
    if target.contains("conv1d.default") {
        return Ok(true);
    }

    // For convolution.default, check stride length to identify 1D.
    let args = FxGraphModule::node_args(node)?;
    if let Some(stride) = args.get(3) {
        let py = node.py();
        let builtins = py.import_bound("builtins")?;
        let stride_len: usize = builtins
            .call_method1("len", (stride,))?
            .extract()
            .unwrap_or(0);
        Ok(stride_len == 1)
    } else {
        Ok(false)
    }
}

/// Extract a tuple of i64 from a Python object (list or tuple).
fn extract_i64_tuple(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let py = obj.py();
    let builtins = py.import_bound("builtins")?;
    let len: usize = builtins.call_method1("len", (obj,))?.extract()?;
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let val: i64 = obj.get_item(i)?.extract()?;
        result.push(val);
    }
    Ok(result)
}

/// Unsqueeze a conv1d node into conv2d form.
///
/// Rewrites:
///   view_copy(input[..., W] -> [..., W, 1])
///   view_copy(weight[..., KW] -> [..., KW, 1])
///   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
///   view_copy(output[..., W_out, 1] -> [..., W_out])
fn unsqueeze_conv1d_node<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &'py FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    _debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    let target_str = FxGraphModule::node_target_str(node)?;

    let (x, weight, bias, stride, pad, dilation, transposed, output_padding, groups);

    if target_str.contains("conv1d.default") {
        if args.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Expected aten.conv1d.default to have at least 2 args, got {}",
                args.len()
            )));
        }
        x = args[0].clone();
        weight = args[1].clone();
        bias = args.get(2).cloned();
        stride = if args.len() > 3 {
            extract_i64_tuple(&args[3])?
        } else {
            vec![1]
        };
        pad = if args.len() > 4 {
            extract_i64_tuple(&args[4])?
        } else {
            vec![0]
        };
        dilation = if args.len() > 5 {
            extract_i64_tuple(&args[5])?
        } else {
            vec![1]
        };
        groups = if args.len() > 6 {
            args[6].extract::<i64>()?
        } else {
            1
        };
        transposed = false;
        output_padding = vec![0];
    } else {
        // convolution.default: 9 args
        if args.len() < 9 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Expected aten.convolution.default to have 9 args, got {}",
                args.len()
            )));
        }
        x = args[0].clone();
        weight = args[1].clone();
        bias = if args[2].is_none() {
            None
        } else {
            Some(args[2].clone())
        };
        stride = extract_i64_tuple(&args[3])?;
        pad = extract_i64_tuple(&args[4])?;
        dilation = extract_i64_tuple(&args[5])?;
        transposed = args[6].extract::<bool>()?;
        output_padding = extract_i64_tuple(&args[7])?;
        groups = args[8].extract::<i64>()?;
    }

    // Only handle 1D (stride length == 1)
    if stride.len() != 1 {
        return Ok(());
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    // Get input/weight shapes and append [1] for unsqueeze
    let x_shape = FxGraphModule::node_shape(&x)?;
    let mut x_unsqueeze_shape: Vec<i64> = x_shape.clone();
    x_unsqueeze_shape.push(1);

    let w_shape = FxGraphModule::node_shape(&weight)?;
    let mut w_unsqueeze_shape: Vec<i64> = w_shape.clone();
    w_unsqueeze_shape.push(1);

    // Unsqueeze input and weight: view_copy(x, [..., 1])
    let x_unsqueezed = d.view(&x, &x_unsqueeze_shape)?;
    let w_unsqueezed = d.view(&weight, &w_unsqueeze_shape)?;

    // Build 2D conv args: stride+[1], pad+[0], dilation+[1], output_padding+[0]
    let mut stride_2d = stride.clone();
    stride_2d.push(1);
    let mut pad_2d = pad.clone();
    pad_2d.push(0);
    let mut dilation_2d = dilation.clone();
    dilation_2d.push(1);
    let mut output_padding_2d = output_padding.clone();
    output_padding_2d.push(0);

    // Create the conv2d node
    let conv2d = d.convolution(
        &x_unsqueezed,
        &w_unsqueezed,
        bias.as_ref(),
        &stride_2d,
        &pad_2d,
        &dilation_2d,
        transposed,
        &output_padding_2d,
        groups,
    )?;

    // Squeeze output: view_copy(conv2d, shape[:-1])
    let conv2d_shape = FxGraphModule::node_shape(&conv2d)?;
    let squeezed_shape: Vec<i64> = conv2d_shape[..conv2d_shape.len() - 1].to_vec();
    let squeezed = d.view(&conv2d, &squeezed_shape)?;

    // Replace original node and erase
    FxGraphModule::replace_all_uses_with(node, &squeezed)?;
    gm.erase_node(node)?;
    graph.call_method0("eliminate_dead_code")?;

    Ok(())
}

crate::fx_pass!(
    /// FX-graph-direct implementation of Arm's Conv1dUnsqueezePass.
    ///
    /// TOSA has no native 1D convolution. The reference Arm pass rewrites conv1d
    /// into:
    ///   view_copy(input[..., W] -> [..., W, 1])
    ///   view_copy(weight[..., KW] -> [..., KW, 1])
    ///   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
    ///   view_copy(output[..., W_out, 1] -> [..., W_out])
    ///
    /// This pass mirrors that shape legalization before `FxRewriteConvPass`.
    FxConv1dUnsqueezePass, "FxConv1dUnsqueeze",
    matches: |target, node| is_conv1d(&target, &node).unwrap_or(false),
    transform: |py, graph, gm, node, debug| {
        unsqueeze_conv1d_node(py, &graph, gm, node, debug)
    }
);
