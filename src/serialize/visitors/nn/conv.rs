use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaConv2d, emit_conv2d as EmitFn),
        (OpTarget::TosaConv3d, emit_conv3d),
        (OpTarget::TosaDepthwiseConv2d, emit_depthwise_conv2d),
        (OpTarget::Conv2d, emit_aten_conv2d),
        (OpTarget::DepthwiseConv2d, emit_aten_conv2d),
    ]);
}

pub(in crate::serialize::visitors) fn emit_aten_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let stride = arg_i32_list(node, 3);
    let padding = arg_i32_list(node, 4);
    let dilation = arg_i32_list(node, 5);
    let groups = arg_i64(node, 8, 1);
    let pad = adjusted_tosa_pad_for_raw_aten_conv(graph, store, node, &padding, &stride, &dilation);
    let acc_type = conv_accumulator_dtype(node);

    let input_zp_val = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 0)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or(0);
    let weight_zp_val = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 1)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or(0);
    let (input_zp_name, weight_zp_name) =
        push_input_weight_zp_tensors(&mut result, graph, node, input_zp_val, weight_zp_val);
    let conv_inputs = extend_conv_inputs(input_names, input_zp_name, weight_zp_name);

    let is_depthwise = if let Some(conv_info_entry) = node
        .kwargs
        .iter()
        .find(|(k, _)| k == "conv_info_is_depthwise")
    {
        matches!(&conv_info_entry.1, Arg::Int(v) if *v != 0)
    } else if groups > 1 {
        let in_channels = if let Some(Arg::Node(input_id)) = node.args.first() {
            graph
                .node(*input_id)
                .map(|n| {
                    let s = n.meta.val.shape_slice();
                    if s.len() >= 2 {
                        s[1]
                    } else {
                        0
                    }
                })
                .unwrap_or(0)
        } else {
            0
        };
        groups == in_channels
    } else {
        false
    };

    let stride = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride
    };
    let dilation = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation
    };

    if is_depthwise {
        result.push_output_op(
            compute_depthwise_conv2d_nhwc_output(
                graph, store, node, &pad, &stride, &dilation,
            ),
            Op::DepthwiseConv2d,
            Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type,
            }),
            conv_inputs,
        );
    } else {
        result.push_output_op(
            compute_conv2d_nhwc_output(
                graph, store, node, &pad, &stride, &dilation,
            ),
            Op::Conv2d,
            Attribute::Conv2d(Conv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type,
            }),
            conv_inputs,
        );
    }

    result
}

pub(in crate::serialize::visitors) fn emit_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if kw.is_empty() {
            arg_i32_list(node, 4)
        } else {
            kw
        }
    };
    let stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 3)
        } else {
            kw
        }
    };
    let dilation = {
        let kw = kwarg_i32_list(node, "dilation");
        if kw.is_empty() {
            arg_i32_list(node, 5)
        } else {
            kw
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);

    let raw_pad = if pad.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride.clone()
    };
    let dilation_for_shape = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation.clone()
    };
    let pad_for_shape = adjusted_tosa_pad_for_emitted_conv2d(
        graph,
        store,
        node,
        &raw_pad,
        &stride_for_shape,
        &dilation_for_shape,
    );
    result.tensors.push(compute_conv2d_nhwc_output(
        graph,
        store,
        node,
        &pad_for_shape,
        &stride_for_shape,
        &dilation_for_shape,
    ));

    let acc_type = conv_accumulator_dtype(node);
    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let input_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
        let (input_zp_name, weight_zp_name) =
            push_input_weight_zp_tensors(&mut result, graph, node, input_zp_val, weight_zp_val);
        extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
    };

    result.operators.push(TosaOperator::new(
        Op::Conv2d,
        Attribute::Conv2d(Conv2dAttribute {
            pad: pad_for_shape,
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            dilation: if dilation.is_empty() {
                vec![1, 1]
            } else {
                dilation
            },
            local_bound,
            acc_type,
        }),
        conv_inputs,
        vec![node.name.clone()],
    ));
    result
}

pub(in crate::serialize::visitors) fn emit_conv3d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if kw.is_empty() {
            arg_i32_list(node, 4)
        } else {
            kw
        }
    };
    let stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 3)
        } else {
            kw
        }
    };
    let dilation = {
        let kw = kwarg_i32_list(node, "dilation");
        if kw.is_empty() {
            arg_i32_list(node, 5)
        } else {
            kw
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);

    let raw_pad = if pad.is_empty() {
        vec![0, 0, 0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![1, 1, 1]
    } else {
        stride.clone()
    };
    let dilation_for_shape = if dilation.is_empty() {
        vec![1, 1, 1]
    } else {
        dilation.clone()
    };
    let pad_for_shape = adjusted_tosa_pad_for_emitted_conv3d(
        graph,
        store,
        node,
        &raw_pad,
        &stride_for_shape,
        &dilation_for_shape,
    );
    result.tensors.push(compute_conv3d_ndhwc_output(
        graph,
        store,
        node,
        &pad_for_shape,
        &stride_for_shape,
        &dilation_for_shape,
    ));

    let acc_type = conv_accumulator_dtype(node);
    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let input_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
        let (input_zp_name, weight_zp_name) =
            push_input_weight_zp_tensors(&mut result, graph, node, input_zp_val, weight_zp_val);
        extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
    };

    result.operators.push(TosaOperator::new(
        Op::Conv3d,
        Attribute::Conv3d(Conv3dAttribute {
            pad: pad_for_shape,
            stride: if stride.is_empty() {
                vec![1, 1, 1]
            } else {
                stride
            },
            dilation: if dilation.is_empty() {
                vec![1, 1, 1]
            } else {
                dilation
            },
            local_bound,
            acc_type,
        }),
        conv_inputs,
        vec![node.name.clone()],
    ));
    result
}

pub(in crate::serialize::visitors) fn emit_depthwise_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if kw.is_empty() {
            arg_i32_list(node, 4)
        } else {
            kw
        }
    };
    let stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 3)
        } else {
            kw
        }
    };
    let dilation = {
        let kw = kwarg_i32_list(node, "dilation");
        if kw.is_empty() {
            arg_i32_list(node, 5)
        } else {
            kw
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);
    let acc_type = conv_accumulator_dtype(node);

    let raw_pad = if pad.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride.clone()
    };
    let dilation_for_shape = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation.clone()
    };
    let pad_for_shape = adjusted_tosa_pad_for_emitted_conv2d(
        graph,
        store,
        node,
        &raw_pad,
        &stride_for_shape,
        &dilation_for_shape,
    );
    result.tensors.push(compute_depthwise_conv2d_nhwc_output(
        graph,
        store,
        node,
        &pad_for_shape,
        &stride_for_shape,
        &dilation_for_shape,
    ));

    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let input_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
        let (input_zp_name, weight_zp_name) =
            push_input_weight_zp_tensors(&mut result, graph, node, input_zp_val, weight_zp_val);
        extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
    };

    result.operators.push(TosaOperator::new(
        Op::DepthwiseConv2d,
        Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
            pad: pad_for_shape,
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            dilation: if dilation.is_empty() {
                vec![1, 1]
            } else {
                dilation
            },
            local_bound,
            acc_type,
        }),
        conv_inputs,
        vec![node.name.clone()],
    ));
    result
}
