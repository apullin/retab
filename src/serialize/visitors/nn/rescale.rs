use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([(OpTarget::TosaRescale, emit_rescale as EmitFn)]);
}

struct PostRescaleReshape {
    input_name: String,
    output_shape: Vec<i32>,
    output_dtype: DType,
}

struct RescaleOutputPlan {
    input_tosa_dtype: DType,
    output_name: String,
    output_shape: Vec<i32>,
    post_reshape: Option<PostRescaleReshape>,
}

struct RescaleParams {
    input_zp: i32,
    output_zp: i32,
    multiplier_name: String,
    shift_name: String,
    scale32: bool,
    per_channel: bool,
    input_unsigned: bool,
    output_unsigned: bool,
}

fn is_depthwise_aten_conv2d(graph: &Graph, store: &TensorStore, node: &Node) -> bool {
    let groups = aten_conv_groups(node);
    if groups <= 1 {
        return false;
    }
    let Some(weight_node) = node
        .args
        .get(1)
        .and_then(Arg::as_node)
        .and_then(|id| graph.node(id))
    else {
        return false;
    };

    let logical_weight = weight_node.meta.val.shape_slice();
    if logical_weight.len() >= 2 && logical_weight[1] == 1 {
        return true;
    }

    let actual_weight = actual_tensor_shape(graph, store, weight_node);
    actual_weight.len() == 4
        && actual_weight.get(2).copied() == Some(groups as i32)
        && actual_weight.get(3).copied().unwrap_or(0) > 0
}

fn rescale_input_shape(graph: &Graph, store: &TensorStore, input_node: &Node) -> Vec<i32> {
    match input_node.target {
        OpTarget::Conv2d if input_rank(graph, input_node) == 5 => {
            let stride = conv_stride_arg(input_node, 3);
            let dilation = conv_dilation_arg(input_node, 3);
            let raw_pad = symmetric_spatial_pad(&conv_pad_arg(input_node, 3), 3);
            let pad = adjusted_tosa_pad_for_emitted_conv3d(
                graph, store, input_node, &raw_pad, &stride, &dilation,
            );
            compute_conv3d_ndhwc_output(graph, store, input_node, &pad, &stride, &dilation).shape
        }
        OpTarget::TosaConv3d => {
            let stride = conv_stride_arg(input_node, 3);
            let dilation = conv_dilation_arg(input_node, 3);
            let raw_pad = conv_pad_arg(input_node, 6);
            let pad = adjusted_tosa_pad_for_emitted_conv3d(
                graph, store, input_node, &raw_pad, &stride, &dilation,
            );
            compute_conv3d_ndhwc_output(graph, store, input_node, &pad, &stride, &dilation).shape
        }
        OpTarget::Conv2d | OpTarget::TosaConv2d => {
            let stride_for_shape = conv_stride_arg(input_node, 2);
            let dilation_for_shape = conv_dilation_arg(input_node, 2);
            let raw_pad = conv_pad_arg(input_node, 4);
            let pad_for_shape = match input_node.target {
                OpTarget::Conv2d => adjusted_tosa_pad_for_raw_aten_conv(
                    graph,
                    store,
                    input_node,
                    &raw_pad,
                    &stride_for_shape,
                    &dilation_for_shape,
                ),
                _ => adjusted_tosa_pad_for_emitted_conv2d(
                    graph,
                    store,
                    input_node,
                    &raw_pad,
                    &stride_for_shape,
                    &dilation_for_shape,
                ),
            };
            if matches!(input_node.target, OpTarget::Conv2d)
                && is_depthwise_aten_conv2d(graph, store, input_node)
            {
                compute_depthwise_conv2d_nhwc_output(
                    graph,
                    store,
                    input_node,
                    &pad_for_shape,
                    &stride_for_shape,
                    &dilation_for_shape,
                )
                .shape
            } else {
                compute_conv2d_nhwc_output(
                    graph,
                    store,
                    input_node,
                    &pad_for_shape,
                    &stride_for_shape,
                    &dilation_for_shape,
                )
                .shape
            }
        }
        OpTarget::DepthwiseConv2d | OpTarget::TosaDepthwiseConv2d => {
            let stride_for_shape = conv_stride_arg(input_node, 2);
            let dilation_for_shape = conv_dilation_arg(input_node, 2);
            let raw_pad = conv_pad_arg(input_node, 4);
            let pad_for_shape = match input_node.target {
                OpTarget::DepthwiseConv2d => adjusted_tosa_pad_for_raw_aten_conv(
                    graph,
                    store,
                    input_node,
                    &raw_pad,
                    &stride_for_shape,
                    &dilation_for_shape,
                ),
                _ => adjusted_tosa_pad_for_emitted_conv2d(
                    graph,
                    store,
                    input_node,
                    &raw_pad,
                    &stride_for_shape,
                    &dilation_for_shape,
                ),
            };
            compute_depthwise_conv2d_nhwc_output(
                graph,
                store,
                input_node,
                &pad_for_shape,
                &stride_for_shape,
                &dilation_for_shape,
            )
            .shape
        }
        _ => actual_tensor_shape(graph, store, input_node),
    }
}

fn rescale_output_dtype(node: &Node) -> DType {
    match node.args.get(1) {
        Some(Arg::Dtype(dtype)) => ir_dtype_to_tosa(*dtype),
        _ => ir_dtype_to_tosa(node.meta.val.dtype),
    }
}

fn plan_rescale_output(graph: &Graph, node: &Node, store: &TensorStore) -> RescaleOutputPlan {
    let logical_output_shape = raw_i32_shape(node);
    let output_dtype = rescale_output_dtype(node);
    let input_tosa_dtype = first_input_node(graph, node)
        .map(|input_node| match input_node.op {
            OpType::Placeholder => store
                .get(&input_node.name)
                .map(|tensor_data| ir_dtype_to_tosa(tensor_data.dtype))
                .unwrap_or_else(|| actual_tensor_dtype(graph, input_node)),
            _ => actual_tensor_dtype(graph, input_node),
        })
        .unwrap_or(DType::Int8);

    let Some(input_node) = first_input_node(graph, node) else {
        return RescaleOutputPlan {
            input_tosa_dtype,
            output_name: node.name.clone(),
            output_shape: actual_tensor_shape(graph, store, node),
            post_reshape: None,
        };
    };
    let input_shape = rescale_input_shape(graph, store, input_node);

    let Some(batched_shape) = promoted_2d_matmul_shape(graph, store, input_node) else {
        return RescaleOutputPlan {
            input_tosa_dtype,
            output_name: node.name.clone(),
            output_shape: input_shape,
            post_reshape: None,
        };
    };

    if logical_output_shape.len() != 2 {
        return RescaleOutputPlan {
            input_tosa_dtype,
            output_name: node.name.clone(),
            output_shape: logical_output_shape,
            post_reshape: None,
        };
    }

    let output_name = format!("{}_batched", node.name);
    RescaleOutputPlan {
        input_tosa_dtype,
        output_name: output_name.clone(),
        output_shape: batched_shape,
        post_reshape: Some(PostRescaleReshape {
            input_name: output_name,
            output_shape: logical_output_shape,
            output_dtype,
        }),
    }
}

fn direct_rescale_scales(node: &Node) -> Vec<f64> {
    match node.args.get(2) {
        Some(Arg::FloatList(list)) => list.iter().copied().collect(),
        Some(Arg::Float(scale)) => vec![*scale],
        _ => vec![1.0],
    }
}

fn emit_direct_rescale_param_tensors(
    result: &mut EmitResult,
    node: &Node,
    input_tosa_dtype: DType,
) -> (String, String, bool, bool) {
    let scales = direct_rescale_scales(node);
    let scale32 = input_tosa_dtype != DType::Int48;
    let scale_width = if scale32 { 32 } else { 16 };
    // TOSA apply_scale_32 requires |value| < 2^(shift-1).  When scale > 1
    // the computed shift is small, and large INT32 accumulator values can
    // overflow.  Clamp shift to >= 20 only for large-scale INT32 rescales
    // (accumulator→output).  Small-scale INT32 rescales (INT8→INT32 upscale,
    // scale ≤ 1) already produce shift >= 31 and don't need clamping.
    let (multipliers, shifts): (Vec<i32>, Vec<i8>) = scales
        .iter()
        .map(|&scale| {
            let (mut m, mut s) = compute_multiplier_and_shift_with_width(scale, scale_width);
            if input_tosa_dtype == DType::Int32 && scale > 1.0 && s < 20 {
                let deficit = (20 - s) as i32;
                m >>= deficit.min(31);
                s = 20;
            }
            (m, s)
        })
        .unzip();
    let multiplier_name = format!("{}_multipliers", node.name);
    let shift_name = format!("{}_shifts", node.name);

    let (mult_dtype, mult_data) = if scale32 {
        (
            DType::Int32,
            multipliers.iter().flat_map(|m| m.to_le_bytes()).collect(),
        )
    } else {
        (
            DType::Int16,
            multipliers
                .iter()
                .flat_map(|m| (*m as i16).to_le_bytes())
                .collect(),
        )
    };

    push_rank1_const(
        result,
        multiplier_name.clone(),
        mult_dtype,
        mult_data,
        multipliers.len(),
    );
    push_rank1_const(
        result,
        shift_name.clone(),
        DType::Int8,
        shifts.iter().map(|&shift| shift as u8).collect(),
        shifts.len(),
    );

    (multiplier_name, shift_name, scale32, scales.len() > 1)
}

fn parse_rescale_params(
    input_names: &[String],
    node: &Node,
    result: &mut EmitResult,
    input_tosa_dtype: DType,
) -> RescaleParams {
    let input_unsigned = kwarg_bool(node, "input_unsigned", false);
    let output_unsigned = kwarg_bool(node, "output_unsigned", false);

    if input_names.len() < 3 {
        let (multiplier_name, shift_name, scale32, per_channel) =
            emit_direct_rescale_param_tensors(result, node, input_tosa_dtype);
        return RescaleParams {
            input_zp: arg_i64(node, 3, 0) as i32,
            output_zp: arg_i64(node, 4, 0) as i32,
            multiplier_name,
            shift_name,
            scale32,
            per_channel,
            input_unsigned,
            output_unsigned,
        };
    }

    RescaleParams {
        input_zp: kwarg_i32(node, "input_zp", 0),
        output_zp: kwarg_i32(node, "output_zp", 0),
        multiplier_name: input_names.get(1).cloned().unwrap_or_default(),
        shift_name: input_names.get(2).cloned().unwrap_or_default(),
        scale32: kwarg_bool(node, "scale32", true),
        per_channel: kwarg_bool(node, "per_channel", false),
        input_unsigned,
        output_unsigned,
    }
}

fn push_rescale_zp_tensor(
    result: &mut EmitResult,
    name: String,
    dtype: DType,
    value: i32,
    is_unsigned: bool,
) -> String {
    let normalized = normalize_rescale_zp(dtype, value, is_unsigned);
    push_rank1_const(
        result,
        name,
        dtype,
        encode_zp_for_dtype(normalized, dtype),
        1,
    )
}

pub(in crate::serialize::visitors) fn emit_rescale(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let output_plan = plan_rescale_output(graph, node, store);
    let rescale_dtype = rescale_output_dtype(node);

    result.tensors.push(make_empty_tensor(
        output_plan.output_name.clone(),
        output_plan.output_shape,
        rescale_dtype,
    ));

    let params = parse_rescale_params(
        &input_names,
        node,
        &mut result,
        output_plan.input_tosa_dtype,
    );
    let input_zp_name = push_rescale_zp_tensor(
        &mut result,
        format!("{}_input_zp", node.name),
        output_plan.input_tosa_dtype,
        params.input_zp,
        params.input_unsigned,
    );
    let output_zp_name = push_rescale_zp_tensor(
        &mut result,
        format!("{}_output_zp", node.name),
        rescale_dtype,
        params.output_zp,
        params.output_unsigned,
    );

    result.push_op(
        Op::Rescale,
        Attribute::Rescale(RescaleAttribute {
            scale32: params.scale32,
            rounding_mode: RoundingMode::SingleRound,
            per_channel: params.per_channel,
            input_unsigned: params.input_unsigned,
            output_unsigned: params.output_unsigned,
        }),
        vec![
            ctx.input_name(0).unwrap_or_default(),
            params.multiplier_name,
            params.shift_name,
            input_zp_name,
            output_zp_name,
        ],
        vec![output_plan.output_name.clone()],
    );

    if let Some(post_reshape) = output_plan.post_reshape {
        push_reshape_with_shape(
            &mut result,
            post_reshape.input_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            post_reshape.output_shape,
            post_reshape.output_dtype,
        );
    }

    result
}
