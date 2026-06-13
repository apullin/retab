use super::*;

/// Compute a single spatial output dimension for convolution:
///   output = (input + pad_before + pad_after - (kernel - 1) * dilation - 1) / stride + 1
fn conv_spatial_output_dim(
    input_len: i32,
    pad_before: i32,
    pad_after: i32,
    kernel: i32,
    dilation: i32,
    stride: i32,
) -> i32 {
    (input_len + pad_before + pad_after - (kernel - 1) * dilation - 1) / stride + 1
}

/// Unified convolution output shape computation for 2D, 3D, and depthwise
/// convolutions in NHWC / NDHWC layout. The spatial rank is inferred from
/// `stride.len()`.
///
/// For depthwise convolutions (`is_depthwise = true`), the weight shape is read
/// from the node metadata (logical NCHW layout) and the output-channel count is
/// taken from `ws[0]` with kernels at `ws[2..]`.  For regular convolutions the
/// weight shape comes from `actual_tensor_shape` and the kernel/OC split is
/// inferred from whether the last weight dim matches the input channel count.
fn compute_conv_output_impl(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
    is_depthwise: bool,
) -> TosaTensor {
    let semantic_dtype = node_output_dtype(node);
    let dtype = if node.meta.input_qparams.is_empty()
        && matches!(semantic_dtype, DType::Fp32 | DType::Fp16 | DType::Bf16)
    {
        semantic_dtype
    } else {
        conv_accumulator_dtype(node)
    };
    let spatial_rank = stride.len(); // 2 for conv2d / depthwise, 3 for conv3d
    let tensor_rank = spatial_rank + 2; // N + spatial dims + C

    let computed = (|| -> Option<Vec<i32>> {
        let input_id = match node.args.first()? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let weight_id = match node.args.get(1)? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let input_node = graph.node(input_id)?;
        let weight_node = graph.node(weight_id)?;

        let inp = actual_tensor_shape(graph, store, input_node);
        if inp.len() != tensor_rank {
            return None;
        }
        let n = inp[0];
        // Spatial input dims are inp[1..1+spatial_rank].
        let spatial_inputs: Vec<i32> = inp[1..1 + spatial_rank].to_vec();

        // Determine output channels and kernel spatial dims.
        let (oc, kernels): (i32, Vec<i32>) = if is_depthwise {
            let ic = *inp.get(1 + spatial_rank)?;
            let logical_ws = weight_node.meta.val.shape_slice();
            let actual_ws = actual_tensor_shape(graph, store, weight_node);
            if logical_ws.len() == tensor_rank && logical_ws.get(1).copied() == Some(1) {
                // Logical depthwise metadata is still OIHW [OC, 1, KH, KW] even if
                // `actual_tensor_shape()` has already applied dim-order reordering on a
                // placeholder. Using the reordered placeholder shape here swaps KH/KW for
                // cases like cumsum's synthetic kernel [1, 1, K, 1] and yields invalid
                // depthwise output shapes.
                let oc = logical_ws[0] as i32;
                let kernels: Vec<i32> = logical_ws[2..2 + spatial_rank]
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                (oc, kernels)
            } else if spatial_rank == 2
                && actual_ws.len() == tensor_rank
                && actual_ws.get(2).copied() == Some(ic)
            {
                // Memory-format lowering emits depthwise weights as HWCM.
                let multiplier = actual_ws.get(3).copied().unwrap_or(1).max(1);
                (ic * multiplier, actual_ws[0..2].to_vec())
            } else {
                if logical_ws.len() != tensor_rank {
                    return None;
                }
                // Logical depthwise metadata is [OC, 1, kH, kW].
                let oc = logical_ws[0] as i32;
                let kernels: Vec<i32> = logical_ws[2..2 + spatial_rank]
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                (oc, kernels)
            }
        } else {
            let ws = actual_tensor_shape(graph, store, weight_node);
            if ws.len() != tensor_rank {
                return None;
            }
            let ic = *inp.get(1 + spatial_rank)?; // last dim of NHWC input
            let oc = ws[0];
            // If last dim of weight matches input channels, kernels are at
            // ws[1..1+spatial_rank]; otherwise they are at ws[2..2+spatial_rank]
            // (weight stored in a different layout).
            let kernels: Vec<i32> = if ws[tensor_rank - 1] == ic {
                ws[1..1 + spatial_rank].to_vec()
            } else {
                ws[2..2 + spatial_rank].to_vec()
            };
            (oc, kernels)
        };

        // Compute each spatial output dimension.
        let mut shape = Vec::with_capacity(tensor_rank);
        shape.push(n);
        for i in 0..spatial_rank {
            let pad_before = *pad.get(i * 2).unwrap_or(&0);
            let pad_after = *pad.get(i * 2 + 1).unwrap_or(&0);
            let s = *stride.get(i).unwrap_or(&1);
            let d = *dilation.get(i).unwrap_or(&1);
            shape.push(conv_spatial_output_dim(
                spatial_inputs[i],
                pad_before,
                pad_after,
                kernels[i],
                d,
                s,
            ));
        }
        shape.push(oc);

        Some(shape)
    })();

    let fallback_shape = {
        let shape = raw_i32_shape(node);
        if shape.len() == tensor_rank && tensor_rank >= 3 {
            // Permute from NCHW/NCDHW to NHWC/NDHWC: move dim[1] to the end.
            let mut nhwc = Vec::with_capacity(tensor_rank);
            nhwc.push(shape[0]);
            nhwc.extend_from_slice(&shape[2..]);
            nhwc.push(shape[1]);
            nhwc
        } else {
            shape
        }
    };

    if crate::config::debug_slice_enabled() {
        eprintln!(
            "[ConvShapeDebug] node={} computed={:?} fallback={:?}",
            node.name, computed, fallback_shape
        );
    }

    let final_shape = match computed {
        Some(shape) if shape.iter().all(|&dim| dim > 0) => shape,
        _ => fallback_shape,
    };

    TosaTensor::new(&node.name, final_shape, dtype)
}

pub(super) fn compute_conv2d_nhwc_output(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> TosaTensor {
    compute_conv_output_impl(graph, store, node, pad, stride, dilation, false)
}

pub(super) fn compute_conv3d_ndhwc_output(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> TosaTensor {
    compute_conv_output_impl(graph, store, node, pad, stride, dilation, false)
}

pub(super) fn compute_depthwise_conv2d_nhwc_output(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> TosaTensor {
    compute_conv_output_impl(graph, store, node, pad, stride, dilation, true)
}

pub(super) fn compute_pool_nhwc_output(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
) -> TosaTensor {
    let dtype = node_output_dtype(node);

    let computed = (|| -> Option<Vec<i32>> {
        let input_id = match node.args.first()? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let input_node = graph.node(input_id)?;

        let inp = actual_tensor_shape(graph, store, input_node);
        if inp.len() != 4 {
            return None;
        }
        let (n, ih, iw, c) = (inp[0], inp[1], inp[2], inp[3]);

        let kh = *kernel.first().unwrap_or(&2);
        let kw = *kernel.get(1).unwrap_or(&2);
        let pad_top = *pad.first().unwrap_or(&0);
        let pad_bottom = *pad.get(1).unwrap_or(&0);
        let pad_left = *pad.get(2).unwrap_or(&0);
        let pad_right = *pad.get(3).unwrap_or(&0);
        let stride_h = *stride.first().unwrap_or(&2);
        let stride_w = *stride.get(1).unwrap_or(&2);

        let oh = (ih + pad_top + pad_bottom - kh) / stride_h + 1;
        let ow = (iw + pad_left + pad_right - kw) / stride_w + 1;

        Some(vec![n, oh, ow, c])
    })();

    let fallback_shape = {
        let shape = raw_i32_shape(node);
        if shape.len() == 4 {
            vec![shape[0], shape[2], shape[3], shape[1]]
        } else {
            shape
        }
    };

    let final_shape = match computed {
        Some(shape) if shape.iter().all(|&dim| dim > 0) => shape,
        _ => fallback_shape,
    };

    TosaTensor::new(&node.name, final_shape, dtype)
}

fn conv_pad_remainder(input_len: i32, pad: i32, dilation: i32, kernel: i32, stride: i32) -> i32 {
    (input_len + 2 * pad - dilation * (kernel - 1) - 1) % stride
}

fn conv_tosa_pad_remainder(
    input_len: i32,
    pad_before: i32,
    pad_after: i32,
    dilation: i32,
    kernel: i32,
    stride: i32,
) -> i32 {
    (input_len + pad_before + pad_after - dilation * (kernel - 1) - 1) % stride
}

fn same_padding_pair(input_len: i32, kernel: i32, stride: i32, dilation: i32) -> (i32, i32) {
    let out_len = (input_len + stride - 1) / stride;
    let needed = ((out_len - 1) * stride + dilation * (kernel - 1) + 1 - input_len).max(0);
    let before = needed / 2;
    let after = needed - before;
    (before, after)
}

pub(super) fn adjusted_tosa_pad_for_raw_aten_conv(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    padding: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> Vec<i32> {
    let symmetric_pad = if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    };

    let Some(input_id) = node.args.first().and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return symmetric_pad;
    };
    let Some(weight_id) = node.args.get(1).and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return symmetric_pad;
    };
    let Some(input_node) = graph.node(input_id) else {
        return symmetric_pad;
    };
    let Some(weight_node) = graph.node(weight_id) else {
        return symmetric_pad;
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let weight_shape = actual_tensor_shape(graph, store, weight_node);
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return symmetric_pad;
    }

    let input_channels = *input_shape.get(3).unwrap_or(&0);
    let (kernel_h, kernel_w) = if weight_shape.get(3).copied().unwrap_or(0) == input_channels {
        (weight_shape[1], weight_shape[2])
    } else {
        (weight_shape[2], weight_shape[3])
    };

    let input_h = input_shape[1];
    let input_w = input_shape[2];
    let stride_h = *stride.first().unwrap_or(&1);
    let stride_w = *stride.get(1).unwrap_or(&stride_h);
    let dilation_h = *dilation.first().unwrap_or(&1);
    let dilation_w = *dilation.get(1).unwrap_or(&dilation_h);

    let padding_mode = arg_string(node, 4).or_else(|| kwarg_string(node, "padding"));
    if matches!(padding_mode.as_deref(), Some("same")) {
        let (pad_top, pad_bottom) = same_padding_pair(input_h, kernel_h, stride_h, dilation_h);
        let (pad_left, pad_right) = same_padding_pair(input_w, kernel_w, stride_w, dilation_w);
        return vec![pad_top, pad_bottom, pad_left, pad_right];
    }

    let pad_h = *padding.first().unwrap_or(&0);
    let pad_w = *padding.get(1).unwrap_or(&pad_h);

    let pad_bottom = {
        let remainder = conv_pad_remainder(input_h, pad_h, dilation_h, kernel_h, stride_h);
        if remainder > pad_h {
            pad_h
        } else {
            pad_h - remainder
        }
    };
    let pad_right = {
        let remainder = conv_pad_remainder(input_w, pad_w, dilation_w, kernel_w, stride_w);
        if remainder > pad_w {
            pad_w
        } else {
            pad_w - remainder
        }
    };

    vec![pad_h, pad_bottom, pad_w, pad_right]
}

pub(super) fn adjusted_tosa_pad_for_emitted_conv2d(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> Vec<i32> {
    if pad.len() != 4 {
        return pad.to_vec();
    }

    let Some(input_id) = node.args.first().and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return pad.to_vec();
    };
    let Some(weight_id) = node.args.get(1).and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return pad.to_vec();
    };
    let Some(input_node) = graph.node(input_id) else {
        return pad.to_vec();
    };
    let Some(weight_node) = graph.node(weight_id) else {
        return pad.to_vec();
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let weight_shape = actual_tensor_shape(graph, store, weight_node);
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return pad.to_vec();
    }

    let input_channels = *input_shape.get(3).unwrap_or(&0);
    let (kernel_h, kernel_w) = if weight_shape.get(3).copied().unwrap_or(0) == input_channels {
        (weight_shape[1], weight_shape[2])
    } else {
        (weight_shape[2], weight_shape[3])
    };

    let input_dims = [input_shape[1], input_shape[2]];
    let kernel_dims = [kernel_h, kernel_w];
    let stride_dims = [*stride.first().unwrap_or(&1), *stride.get(1).unwrap_or(&1)];
    let dilation_dims = [
        *dilation.first().unwrap_or(&1),
        *dilation.get(1).unwrap_or(&1),
    ];

    let mut adjusted = pad.to_vec();
    for axis in 0..2 {
        let pad_before = adjusted[axis * 2];
        let pad_after = adjusted[axis * 2 + 1];
        let remainder = conv_tosa_pad_remainder(
            input_dims[axis],
            pad_before,
            pad_after,
            dilation_dims[axis],
            kernel_dims[axis],
            stride_dims[axis],
        );
        if remainder <= pad_after {
            adjusted[axis * 2 + 1] = pad_after - remainder;
        }
    }

    adjusted
}

pub(super) fn adjusted_tosa_pad_for_emitted_conv3d(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> Vec<i32> {
    if pad.len() != 6 {
        return pad.to_vec();
    }

    let Some(input_id) = node.args.first().and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return pad.to_vec();
    };
    let Some(weight_id) = node.args.get(1).and_then(|arg| match arg {
        Arg::Node(id) => Some(*id),
        _ => None,
    }) else {
        return pad.to_vec();
    };
    let Some(input_node) = graph.node(input_id) else {
        return pad.to_vec();
    };
    let Some(weight_node) = graph.node(weight_id) else {
        return pad.to_vec();
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let weight_shape = actual_tensor_shape(graph, store, weight_node);
    if input_shape.len() != 5 || weight_shape.len() != 5 {
        return pad.to_vec();
    }

    let input_channels = *input_shape.get(4).unwrap_or(&0);
    let (kernel_d, kernel_h, kernel_w) =
        if weight_shape.get(4).copied().unwrap_or(0) == input_channels {
            (weight_shape[1], weight_shape[2], weight_shape[3])
        } else {
            (weight_shape[2], weight_shape[3], weight_shape[4])
        };

    let input_dims = [input_shape[1], input_shape[2], input_shape[3]];
    let kernel_dims = [kernel_d, kernel_h, kernel_w];
    let stride_dims = [
        *stride.first().unwrap_or(&1),
        *stride.get(1).unwrap_or(&1),
        *stride.get(2).unwrap_or(&1),
    ];
    let dilation_dims = [
        *dilation.first().unwrap_or(&1),
        *dilation.get(1).unwrap_or(&1),
        *dilation.get(2).unwrap_or(&1),
    ];

    let mut adjusted = pad.to_vec();
    for axis in 0..3 {
        let pad_before = adjusted[axis * 2];
        let pad_after = adjusted[axis * 2 + 1];
        let remainder = conv_tosa_pad_remainder(
            input_dims[axis],
            pad_before,
            pad_after,
            dilation_dims[axis],
            kernel_dims[axis],
            stride_dims[axis],
        );
        if remainder <= pad_after {
            adjusted[axis * 2 + 1] = pad_after - remainder;
        }
    }

    adjusted
}
