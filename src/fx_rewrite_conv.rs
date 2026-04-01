use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{
    resolve_tosa_conv2d_op, resolve_tosa_conv3d_op, resolve_tosa_depthwise_conv2d_op,
    resolve_tosa_rescale_op, resolve_tosa_transpose_conv2d_op, NodeInserter,
};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of conv rewriting.
///
/// Detects `aten.convolution.default` and raw `aten.conv2d.default` nodes and
/// rewrites them to the appropriate TOSA conv op entirely in Rust.  This
/// matches the Python `RewriteConvPass` behavior:
///   - Changes target from aten.convolution to tosa.CONV2D / CONV3D /
///     DEPTHWISE_CONV2D / TRANSPOSE_CONV2D
///   - Rearranges args (pad format, bias handling)
///   - Creates a zero-bias buffer when bias is None
///   - Attaches conv_info metadata with quantization zero-points
///   - For 3D quantized convolutions, inserts an output RESCALE node
pub struct FxRewriteConvPass;

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("conv2d.default")
}

fn infer_in_channels(
    input_node: &Bound<'_, PyAny>,
    input_shape: &[i64],
    weight_shape: &[i64],
    groups: i64,
) -> PyResult<i64> {
    // The direct path may already have converted activations into channels-last
    // layout before conv rewriting. Respect `tosa_dim_order` when present so
    // we read the logical C axis, not just shape[1].
    let meta = input_node.getattr("meta")?;
    let dim_order = meta.call_method1("get", ("tosa_dim_order",))?;
    if !dim_order.is_none() {
        let order: Vec<usize> = dim_order.extract().unwrap_or_default();
        if let Some(channel_axis) = order.iter().position(|&axis| axis == 1) {
            if let Some(&channels) = input_shape.get(channel_axis) {
                if channels > 0 {
                    return Ok(channels);
                }
            }
        }
    }

    if let Some(&channels) = input_shape.get(1) {
        if channels > 0 {
            return Ok(channels);
        }
    }

    if weight_shape.len() >= 2 {
        let channels_per_group = weight_shape[1];
        if channels_per_group > 0 && groups > 0 {
            return Ok(channels_per_group * groups);
        }
    }

    Ok(0)
}

// ---------------------------------------------------------------------------
// Pure-Rust equivalents of the deleted Python helpers
// ---------------------------------------------------------------------------

/// Expand scalar/1-element spatial params to the requested rank.
fn expand_spatial_param(param: &Bound<'_, PyAny>, spatial_rank: usize) -> Vec<i64> {
    // Could be a single int
    if let Ok(val) = param.extract::<i64>() {
        return vec![val; spatial_rank];
    }
    // Could be a list/tuple
    if let Ok(vals) = param.extract::<Vec<i64>>() {
        if vals.len() == 1 && spatial_rank > 1 {
            return vec![vals[0]; spatial_rank];
        }
        return vals;
    }
    vec![0; spatial_rank]
}

/// Mirror Arm RewriteConvPass trailing-pad legality adjustment.
fn adjust_pad_if_needed(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad: i64,
    dilation: i64,
) -> i64 {
    let effective_kernel = (kernel_size - 1) * dilation + 1;
    let remainder = (input_size + 2 * pad - effective_kernel) % stride;
    if remainder > pad {
        pad
    } else {
        pad - remainder
    }
}

/// Get `meta["val"]` from a node, unwrapping list/tuple to get the first element.
fn get_first_fake_tensor<'py>(node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let meta = node.getattr("meta")?;
    let val = meta.get_item("val")?;
    let builtins = node.py().import_bound("builtins")?;
    let is_seq = builtins
        .call_method1(
            "isinstance",
            (
                &val,
                PyTuple::new_bound(
                    node.py(),
                    &[builtins.getattr("list")?, builtins.getattr("tuple")?],
                ),
            ),
        )?
        .extract::<bool>()?;
    if is_seq {
        val.get_item(0)
    } else {
        Ok(val)
    }
}

/// Get a qparam field by attribute or dict key.
fn qparam_field<'py>(
    qparam: &Bound<'py, PyAny>,
    attr: &str,
    fallback_key: Option<&str>,
) -> Option<Bound<'py, PyAny>> {
    if let Ok(val) = qparam.getattr(attr) {
        if !val.is_none() {
            return Some(val);
        }
    }
    if let Some(key) = fallback_key {
        if let Ok(val) = qparam.call_method1("get", (key,)) {
            if !val.is_none() {
                return Some(val);
            }
        }
    }
    if let Ok(val) = qparam.call_method1("get", (attr,)) {
        if !val.is_none() {
            return Some(val);
        }
    }
    None
}

fn qparam_f64(qparam: &Bound<'_, PyAny>, attr: &str, default: f64) -> f64 {
    qparam_field(qparam, attr, None)
        .and_then(|v| {
            v.extract::<f64>().ok().or_else(|| {
                // Could be a list — take first element
                v.get_item(0_usize)
                    .ok()
                    .and_then(|item| item.extract::<f64>().ok())
            })
        })
        .unwrap_or(default)
}

fn qparam_i64(
    qparam: &Bound<'_, PyAny>,
    attr: &str,
    fallback_key: Option<&str>,
    default: i64,
) -> i64 {
    qparam_field(qparam, attr, fallback_key)
        .and_then(|v| v.extract::<i64>().ok())
        .unwrap_or(default)
}

fn qparam_get<'py>(
    qparams: &Bound<'py, PyAny>,
    idx: usize,
) -> Option<Bound<'py, PyAny>> {
    // Could be a dict or a list/tuple
    if let Ok(val) = qparams.call_method1("get", (idx,)) {
        if !val.is_none() {
            return Some(val);
        }
    }
    if let Ok(val) = qparams.get_item(idx) {
        if !val.is_none() {
            return Some(val);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Core rewrite logic — replaces the Python `rewrite_conv_to_tosa`
// ---------------------------------------------------------------------------

/// Classification of what kind of TOSA conv op to emit.
enum ConvKind {
    TransposeConv2d,
    DepthwiseConv2d,
    Conv3d,
    Conv2d,
}

#[allow(clippy::too_many_arguments)]
fn rewrite_conv_to_tosa<'py>(
    py: Python<'py>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    is_depthwise: bool,
    input_zp: i64,
    weight_zp: i64,
    _in_channels: i64,
) -> PyResult<()> {
    let graph = gm.gm_ref().getattr("graph")?;
    let args = FxGraphModule::node_args(node)?;
    let target_str = FxGraphModule::node_target_str(node)?;
    let is_raw_conv2d = target_str.contains("conv2d.default");

    // -- Extract args --------------------------------------------------------
    let (x, weight, mut bias_opt, stride_py, pad_py, dilation_py, transposed, output_padding_py, groups);

    if is_raw_conv2d {
        if args.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Expected aten.conv2d.default to have at least 2 args, got {}",
                args.len()
            )));
        }
        x = args[0].clone();
        weight = args[1].clone();
        bias_opt = if args.len() > 2 && !args[2].is_none() {
            Some(args[2].clone())
        } else {
            None
        };
        stride_py = if args.len() > 3 {
            args[3].clone()
        } else {
            PyTuple::new_bound(py, [1i64, 1]).into_any()
        };
        pad_py = if args.len() > 4 {
            args[4].clone()
        } else {
            PyTuple::new_bound(py, [0i64, 0]).into_any()
        };
        dilation_py = if args.len() > 5 {
            args[5].clone()
        } else {
            PyTuple::new_bound(py, [1i64, 1]).into_any()
        };
        groups = if args.len() > 6 {
            args[6].extract().unwrap_or(1)
        } else {
            1i64
        };
        transposed = false;
        output_padding_py = PyTuple::new_bound(py, [0i64, 0]).into_any();
    } else {
        if args.len() < 9 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Expected aten.convolution.default to have 9 args, got {}",
                args.len()
            )));
        }
        x = args[0].clone();
        weight = args[1].clone();
        bias_opt = if !args[2].is_none() {
            Some(args[2].clone())
        } else {
            None
        };
        stride_py = args[3].clone();
        pad_py = args[4].clone();
        dilation_py = args[5].clone();
        transposed = args[6].extract::<bool>().unwrap_or(false);
        output_padding_py = args[7].clone();
        groups = args[8].extract().unwrap_or(1);
    }

    // -- Spatial rank and parameter expansion ---------------------------------
    let input_fake = get_first_fake_tensor(&x)?;
    let input_ndim: usize = input_fake
        .getattr("shape")?
        .extract::<Vec<i64>>()?
        .len();
    let spatial_rank = if input_ndim >= 2 { input_ndim - 2 } else { 0 };

    let stride_list = expand_spatial_param(&stride_py, spatial_rank);
    let dilation_list = expand_spatial_param(&dilation_py, spatial_rank);
    let pad_list = expand_spatial_param(&pad_py, spatial_rank);
    let output_padding_list = expand_spatial_param(&output_padding_py, spatial_rank);

    let stride_tuple: Vec<i64> = stride_list.clone();
    let dilation_tuple: Vec<i64> = dilation_list.clone();

    let weight_fake = get_first_fake_tensor(&weight)?;
    let input_shape: Vec<i64> = input_fake.getattr("shape")?.extract()?;
    let weight_shape: Vec<i64> = weight_fake.getattr("shape")?.extract()?;

    // -- Determine input contract dtype from qparams -------------------------
    let meta = node.getattr("meta")?;
    let input_qparams_raw = meta.call_method1("get", ("input_qparams",))?;
    let input_qparam = if !input_qparams_raw.is_none() {
        // Could be a dict
        let qp = input_qparams_raw.call_method1("get", (0,));
        match qp {
            Ok(v) if !v.is_none() => Some(v),
            _ => {
                // single-entry dict: get the only value
                if let Ok(vals) = input_qparams_raw.call_method0("values") {
                    if let Ok(mut it) = vals.iter() {
                        it.next().and_then(|v| v.ok())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    } else {
        None
    };

    let input_dtype = input_fake.getattr("dtype")?;
    let input_contract_dtype = if let Some(ref qp) = input_qparam {
        qp.getattr("dtype").unwrap_or_else(|_| input_dtype.clone())
    } else {
        input_dtype.clone()
    };

    // -- Handle missing bias: create a zero bias buffer ----------------------
    if bias_opt.is_none() {
        let output_val = meta.call_method1("get", ("val",))?;
        let output_channels: i64 = if !output_val.is_none() {
            let shape: Vec<i64> = output_val.getattr("shape")?.extract()?;
            shape.get(1).copied().unwrap_or(0)
        } else {
            weight.getattr("meta")?.get_item("val")?.getattr("shape")?.get_item(0)?.extract()?
        };

        let has_qparams = {
            let oqp = meta.call_method1("get", ("output_qparams",))?;
            if oqp.is_none() {
                false
            } else {
                let builtins = py.import_bound("builtins")?;
                let length: usize = builtins
                    .call_method1("len", (&oqp,))
                    .and_then(|v| v.extract())
                    .unwrap_or(0);
                length > 0
            }
        };

        let torch = py.import_bound("torch")?;
        let bias_dtype = if has_qparams {
            torch.getattr("int32")?
        } else {
            if !output_val.is_none() {
                output_val.getattr("dtype")?
            } else {
                torch.getattr("float32")?
            }
        };

        let bias_data = torch.call_method1(
            "zeros",
            (PyTuple::new_bound(py, [output_channels]),),
        )?;
        // Set dtype via .to()
        let bias_data = bias_data.call_method1("to", (&bias_dtype,))?;

        let node_name = FxGraphModule::node_name(node)?;
        let mut attr_name = format!("{}_bias", node_name);
        let mut counter = 0u32;
        while gm.gm_ref().hasattr(attr_name.as_str())? {
            counter += 1;
            attr_name = format!("{}_bias_{}", node_name, counter);
        }

        gm.gm_ref()
            .call_method1("register_buffer", (attr_name.as_str(), &bias_data))?;

        let insert_ctx = graph.call_method1("inserting_before", (node,))?;
        insert_ctx.call_method0("__enter__")?;
        let bias_node = graph.call_method1("get_attr", (attr_name.as_str(),))?;
        let bias_meta = PyDict::new_bound(py);
        bias_meta.set_item("val", &bias_data)?;
        bias_node.setattr("meta", &bias_meta)?;
        insert_ctx.call_method(
            "__exit__",
            (py.None(), py.None(), py.None()),
            None,
        )?;
        bias_opt = Some(bias_node);
    }

    // Handle INT16 input contract dtype: set TosaSpecialDtype on bias
    let torch = py.import_bound("torch")?;
    let int16_dtype = torch.getattr("int16")?;
    if input_contract_dtype.eq(&int16_dtype)? {
        if let Some(ref bias_node) = bias_opt {
            let result: PyResult<()> = (|| {
                let mapping = py.import_bound("executorch.backends.arm.tosa.mapping")?;
                let tosa_special = mapping.getattr("TosaSpecialDtype")?;
                let meta_key: String = tosa_special.call_method0("meta_key")?.extract()?;
                let int48 = tosa_special.getattr("INT48")?;
                let bias_meta = bias_node.getattr("meta")?;
                bias_meta.set_item(meta_key.as_str(), &int48)?;
                Ok(())
            })();
            // Ignore errors — TosaSpecialDtype may not be available
            let _ = result;
        }
    }

    let bias = bias_opt.as_ref().expect("bias should be set by this point");

    // -- Classify conv kind --------------------------------------------------
    let kind = if transposed {
        ConvKind::TransposeConv2d
    } else if is_depthwise {
        ConvKind::DepthwiseConv2d
    } else if spatial_rank == 3 {
        ConvKind::Conv3d
    } else {
        ConvKind::Conv2d
    };

    // -- Build TOSA padding (symmetric → [top, bottom, left, right, ...]) ----
    let build_tosa_pad = |pad_list: &[i64],
                          input_shape: &[i64],
                          weight_shape: &[i64],
                          stride_list: &[i64],
                          dilation_list: &[i64]|
     -> Vec<i64> {
        let mut tosa_pad = Vec::new();
        for (axis_index, &p) in pad_list.iter().enumerate() {
            tosa_pad.push(p); // leading pad
            tosa_pad.push(p); // trailing pad (adjusted below)
            let trailing_index = axis_index * 2 + 1;
            let in_size = input_shape.get(axis_index + 2).copied().unwrap_or(0);
            let k_size = weight_shape.get(axis_index + 2).copied().unwrap_or(0);
            tosa_pad[trailing_index] = adjust_pad_if_needed(
                in_size,
                k_size,
                stride_list[axis_index],
                tosa_pad[trailing_index],
                dilation_list[axis_index],
            );
        }
        tosa_pad
    };

    // -- Resolve target op and build conv_args / conv_kwargs -----------------
    let (target_op, conv_args, conv_kwargs): (
        Bound<'py, PyAny>,
        Bound<'py, PyTuple>,
        Option<Bound<'py, PyDict>>,
    ) = match kind {
        ConvKind::TransposeConv2d => {
            if spatial_rank != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Only 2D transpose convolutions are supported in the Arm backend.",
                ));
            }
            if dilation_list.iter().any(|&d| d != 1) {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Transpose convolutions with dilation are not supported in the Arm backend.",
                ));
            }
            let op = resolve_tosa_transpose_conv2d_op(py)?;
            let tosa_pad = PyList::new_bound(
                py,
                [
                    -pad_list[0],
                    -pad_list[0] + output_padding_list[0],
                    -pad_list[1],
                    -pad_list[1] + output_padding_list[1],
                ],
            );
            let stride_py = PyTuple::new_bound(py, &stride_tuple);
            let args = PyTuple::new_bound(
                py,
                &[
                    x.as_ref(),
                    weight.as_ref(),
                    bias.as_ref(),
                    tosa_pad.as_ref(),
                    stride_py.as_ref(),
                ],
            );
            let kw = PyDict::new_bound(py);
            kw.set_item("input_zp", input_zp)?;
            kw.set_item("weight_zp", weight_zp)?;
            (op, args, Some(kw))
        }
        ConvKind::DepthwiseConv2d => {
            let op = resolve_tosa_depthwise_conv2d_op(py)?;
            let tosa_pad_vec =
                build_tosa_pad(&pad_list, &input_shape, &weight_shape, &stride_list, &dilation_list);
            let tosa_pad = PyList::new_bound(py, &tosa_pad_vec);
            let stride_py = PyTuple::new_bound(py, &stride_tuple);
            let dilation_py = PyTuple::new_bound(py, &dilation_tuple);
            let args = PyTuple::new_bound(
                py,
                &[
                    x.as_ref(),
                    weight.as_ref(),
                    bias.as_ref(),
                    stride_py.as_ref(),
                    tosa_pad.as_ref(),
                    dilation_py.as_ref(),
                ],
            );
            (op, args, None)
        }
        ConvKind::Conv3d => {
            if groups != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "CONV3D with groups != 1 is not supported in the Arm backend.",
                ));
            }
            let op = resolve_tosa_conv3d_op(py)?;
            let tosa_pad_vec =
                build_tosa_pad(&pad_list, &input_shape, &weight_shape, &stride_list, &dilation_list);
            let tosa_pad = PyList::new_bound(py, &tosa_pad_vec);
            let stride_py = PyTuple::new_bound(py, &stride_tuple);
            let dilation_py = PyTuple::new_bound(py, &dilation_tuple);
            let args = PyTuple::new_bound(
                py,
                &[
                    x.as_ref(),
                    weight.as_ref(),
                    bias.as_ref(),
                    stride_py.as_ref(),
                    tosa_pad.as_ref(),
                    dilation_py.as_ref(),
                ],
            );
            (op, args, None)
        }
        ConvKind::Conv2d => {
            let op = resolve_tosa_conv2d_op(py)?;
            let tosa_pad_vec =
                build_tosa_pad(&pad_list, &input_shape, &weight_shape, &stride_list, &dilation_list);
            let tosa_pad = PyList::new_bound(py, &tosa_pad_vec);
            let stride_py = PyTuple::new_bound(py, &stride_tuple);
            let dilation_py = PyTuple::new_bound(py, &dilation_tuple);
            let args = PyTuple::new_bound(
                py,
                &[
                    x.as_ref(),
                    weight.as_ref(),
                    bias.as_ref(),
                    stride_py.as_ref(),
                    tosa_pad.as_ref(),
                    dilation_py.as_ref(),
                ],
            );
            (op, args, None)
        }
    };

    // -- Insert the new TOSA node after the original -------------------------
    let ins = NodeInserter::after(py, &graph, node).keep_qparams();
    let tosa_op = if let Some(ref kw) = conv_kwargs {
        ins.call_with_kwargs(&target_op, conv_args.as_ref(), kw)?
    } else {
        ins.call(&target_op, conv_args.as_ref())?
    };

    // -- Compute meta["val"] for the new node --------------------------------
    let input_fake = get_first_fake_tensor(&x)?;
    let output_fake_result = get_first_fake_tensor(node);
    let weight_fake = get_first_fake_tensor(&weight)?;
    let bias_fake = get_first_fake_tensor(bias).ok();

    // Try to call the target op to get exact output metadata
    let set_meta_result: PyResult<()> = (|| {
        // Build args for the op call: (input_fake, weight_fake, bias_fake, *conv_args[3:])
        let extra_args_start = 3usize;
        let mut call_args_vec: Vec<Bound<'py, PyAny>> = Vec::new();
        call_args_vec.push(input_fake.clone());
        call_args_vec.push(weight_fake.clone());
        call_args_vec.push(
            bias_fake
                .clone()
                .unwrap_or_else(|| py.None().into_bound(py)),
        );
        let conv_args_len = conv_args.len();
        for i in extra_args_start..conv_args_len {
            call_args_vec.push(conv_args.get_item(i)?);
        }
        let call_args = PyTuple::new_bound(py, &call_args_vec);
        let result = if let Some(ref kw) = conv_kwargs {
            target_op.call(&call_args, Some(kw))?
        } else {
            target_op.call1(&call_args)?
        };
        tosa_op.getattr("meta")?.set_item("val", result)?;
        Ok(())
    })();

    if set_meta_result.is_err() {
        // Fallback for quantized ops
        let int8_dtype = torch.getattr("int8")?;
        let int16_dtype = torch.getattr("int16")?;
        let int32_dtype = torch.getattr("int32")?;
        let input_dt = input_fake.getattr("dtype")?;
        if let Ok(ref output_fake) = output_fake_result {
            if input_dt.eq(&int8_dtype)? || input_dt.eq(&int16_dtype)? {
                let new_val = output_fake.call_method(
                    "new_empty",
                    (output_fake.getattr("shape")?,),
                    Some(&{
                        let kw = PyDict::new_bound(py);
                        kw.set_item("dtype", &int32_dtype)?;
                        kw
                    }),
                )?;
                tosa_op.getattr("meta")?.set_item("val", &new_val)?;

                if input_dt.eq(&int16_dtype)? {
                    let result: PyResult<()> = (|| {
                        let mapping =
                            py.import_bound("executorch.backends.arm.tosa.mapping")?;
                        let tosa_special = mapping.getattr("TosaSpecialDtype")?;
                        let meta_key: String =
                            tosa_special.call_method0("meta_key")?.extract()?;
                        let int48 = tosa_special.getattr("INT48")?;
                        tosa_op.getattr("meta")?.set_item(meta_key.as_str(), &int48)?;
                        Ok(())
                    })();
                    let _ = result;
                }
            }
        }
    }

    // -- For 3D quantized conv, insert output RESCALE ------------------------
    let mut replacement = tosa_op.clone();

    if spatial_rank == 3 {
        if output_fake_result.is_ok() {
            let input_qp = {
                let iqp = meta.call_method1("get", ("input_qparams",))?;
                if iqp.is_none() {
                    None
                } else {
                    qparam_get(&iqp, 0)
                }
            };
            let weight_qp = {
                let iqp = meta.call_method1("get", ("input_qparams",))?;
                if iqp.is_none() {
                    None
                } else {
                    qparam_get(&iqp, 1)
                }
            };
            let output_qp = {
                let oqp = meta.call_method1("get", ("output_qparams",))?;
                if oqp.is_none() {
                    None
                } else {
                    qparam_get(&oqp, 0)
                }
            };

            let int8_dtype = torch.getattr("int8")?;
            let int32_dtype = torch.getattr("int32")?;
            let input_dt = input_fake.getattr("dtype")?;
            let tosa_val_dt = tosa_op
                .getattr("meta")?
                .call_method1("get", ("val",))?;
            let tosa_op_dtype = if !tosa_val_dt.is_none() {
                tosa_val_dt.getattr("dtype").ok()
            } else {
                None
            };

            if input_dt.eq(&int8_dtype)?
                && tosa_op_dtype
                    .as_ref()
                    .map(|d| d.eq(&int32_dtype).unwrap_or(false))
                    .unwrap_or(false)
                && input_qp.is_some()
                && weight_qp.is_some()
                && output_qp.is_some()
            {
                let input_qp = input_qp.unwrap();
                let weight_qp = weight_qp.unwrap();
                let output_qp = output_qp.unwrap();

                let input_scale = qparam_f64(&input_qp, "scale", 1.0);
                let output_scale = qparam_f64(&output_qp, "scale", 1.0);
                let output_zp_val =
                    qparam_i64(&output_qp, "zp", Some("zero_point"), 0);
                let output_dtype = output_qp.getattr("dtype")?;
                let per_channel = qparam_field(&weight_qp, "per_channel", None)
                    .and_then(|v| v.extract::<bool>().ok())
                    .unwrap_or(false);

                let weight_scales: Vec<f64> = if per_channel {
                    qparam_field(&weight_qp, "scale", None)
                        .and_then(|v| {
                            v.extract::<Vec<f64>>().ok().or_else(|| {
                                v.extract::<f64>().ok().map(|s| vec![s])
                            })
                        })
                        .unwrap_or_else(|| vec![1.0])
                } else {
                    vec![qparam_f64(&weight_qp, "scale", 1.0)]
                };

                if output_scale != 0.0 {
                    let post_conv_scale: Vec<f64> = weight_scales
                        .iter()
                        .map(|ws| (input_scale * ws) / output_scale)
                        .collect();

                    // Insert RESCALE node after tosa_op
                    let rescale_op = resolve_tosa_rescale_op(py)?;
                    let scale_list = PyList::new_bound(py, &post_conv_scale);
                    let rescale_args = PyTuple::new_bound(
                        py,
                        &[
                            tosa_op.as_ref(),
                            output_dtype.as_ref(),
                            scale_list.as_ref(),
                            0i64.into_py(py).bind(py).as_ref(),
                            output_zp_val.into_py(py).bind(py).as_ref(),
                        ],
                    );
                    let rescale_ins = NodeInserter::after(py, &graph, &tosa_op);
                    let rescale = rescale_ins.call(&rescale_op, rescale_args.as_ref())?;

                    // Restamp rescale output metadata
                    let tosa_val = get_first_fake_tensor(&tosa_op)?;
                    if tosa_val.hasattr("new_empty")? {
                        let new_val = tosa_val.call_method(
                            "new_empty",
                            (tosa_val.getattr("shape")?,),
                            Some(&{
                                let kw = PyDict::new_bound(py);
                                kw.set_item("dtype", &output_dtype)?;
                                kw
                            }),
                        )?;
                        rescale.getattr("meta")?.set_item("val", &new_val)?;
                    }

                    // Propagate qparams
                    let rescale_meta = rescale.getattr("meta")?;
                    let iqp_map = PyDict::new_bound(py);
                    iqp_map.set_item(0, &input_qp)?;
                    rescale_meta.set_item("input_qparams", &iqp_map)?;
                    let oqp_map = PyDict::new_bound(py);
                    oqp_map.set_item(0, &output_qp)?;
                    rescale_meta.set_item("output_qparams", &oqp_map)?;
                    // Remove any inherited tosa_special_dtype
                    let _ = rescale_meta.call_method1("pop", ("tosa_special_dtype", py.None()));

                    replacement = rescale;
                }
            }
        }
    }

    // -- Replace the original node and clean up ------------------------------
    FxGraphModule::replace_all_uses_with(node, &replacement)?;
    gm.erase_node(node)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// FxPass implementation
// ---------------------------------------------------------------------------

impl FxPass for FxRewriteConvPass {
    fn name(&self) -> &str {
        "FxRewriteConv"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut modified = false;
        let all_nodes = gm.nodes()?;

        // Collect candidates to avoid mutating during iteration (the rewrite
        // erases nodes and inserts new ones).
        let mut candidates: Vec<(Bound<'_, PyAny>, bool, i64, i64, i64, i64, i64)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_convolution(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let is_raw_conv2d = target.contains("conv2d.default");

            if !is_raw_conv2d && args.len() < 9 {
                continue;
            }
            if is_raw_conv2d && args.len() < 2 {
                continue;
            }

            let input_node = &args[0];
            let weight_node = &args[1];
            let groups: i64 = if is_raw_conv2d {
                args.get(6).and_then(|arg| arg.extract().ok()).unwrap_or(1)
            } else {
                args[8].extract().unwrap_or(1)
            };

            let weight_shape = FxGraphModule::node_shape(weight_node).unwrap_or_default();
            let out_channels = weight_shape.first().copied().unwrap_or(0);
            let input_shape = FxGraphModule::node_shape(input_node).unwrap_or_default();
            let in_channels =
                infer_in_channels(input_node, &input_shape, &weight_shape, groups)?;

            let is_depthwise = weight_shape.len() >= 4
                && groups > 1
                && in_channels > 0
                && groups == in_channels
                && out_channels % in_channels == 0;

            // Get quantization info from meta if available
            let meta = node.getattr("meta")?;
            let input_qparams = meta.call_method1("get", ("input_qparams",))?;

            let mut input_zp: i64 = 0;
            let mut weight_zp: i64 = 0;

            if !input_qparams.is_none() {
                if let Ok(qp0) = input_qparams.get_item(0_usize) {
                    input_zp = qp0
                        .getattr("zp")
                        .and_then(|v| v.extract())
                        .or_else(|_| {
                            qp0.call_method1("get", ("zero_point",))
                                .and_then(|v| v.extract())
                        })
                        .unwrap_or(0);
                }
                if let Ok(qp1) = input_qparams.get_item(1_usize) {
                    weight_zp = qp1
                        .getattr("zp")
                        .and_then(|v| v.extract())
                        .or_else(|_| {
                            qp1.call_method1("get", ("zero_point",))
                                .and_then(|v| v.extract())
                        })
                        .unwrap_or(0);
                }
            }

            // Store conv metadata for downstream passes (rescale, serialization)
            let conv_meta = PyDict::new_bound(py);
            conv_meta.set_item("is_depthwise", is_depthwise)?;
            conv_meta.set_item("input_zp", input_zp)?;
            conv_meta.set_item("weight_zp", weight_zp)?;
            conv_meta.set_item("groups", groups)?;
            conv_meta.set_item("in_channels", in_channels)?;
            conv_meta.set_item("out_channels", out_channels)?;
            FxGraphModule::set_meta(node, "conv_info", conv_meta.as_ref())?;

            candidates.push((
                node.clone(),
                is_depthwise,
                input_zp,
                weight_zp,
                in_channels,
                out_channels,
                groups,
            ));
        }

        let debug = FxGraphModule::debug();

        for (node, is_depthwise, input_zp, weight_zp, in_channels, out_channels, groups) in
            &candidates
        {
            if debug {
                let node_name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[RewriteConv] {}: in_ch={}, out_ch={}, groups={}, depthwise={}",
                    node_name, in_channels, out_channels, groups, is_depthwise
                );
            }

            rewrite_conv_to_tosa(
                py,
                gm,
                node,
                *is_depthwise,
                *input_zp,
                *weight_zp,
                *in_channels,
            )?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
