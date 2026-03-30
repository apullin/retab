use super::*;

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

fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

fn push_rank1_const(
    result: &mut EmitResult,
    name: String,
    dtype: DType,
    values: Vec<u8>,
    len: usize,
) -> String {
    push_inline_const(
        result,
        make_rank1_const_tensor(name.clone(), dtype, values, len),
    );
    name
}

fn plan_rescale_output(graph: &Graph, node: &Node, store: &TensorStore) -> RescaleOutputPlan {
    let logical_output_shape = actual_tensor_shape(graph, store, node);
    let output_dtype = ir_dtype_to_tosa(node.meta.val.dtype);
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
            output_shape: logical_output_shape,
            post_reshape: None,
        };
    };

    let Some(batched_shape) = promoted_2d_matmul_shape(graph, store, input_node) else {
        return RescaleOutputPlan {
            input_tosa_dtype,
            output_name: node.name.clone(),
            output_shape: logical_output_shape,
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
    let (multipliers, shifts): (Vec<i32>, Vec<i8>) = scales
        .iter()
        .map(|&scale| compute_multiplier_and_shift_with_width(scale, scale_width))
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

pub(in crate::serialize::visitors) fn emit_rescale(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let output_plan = plan_rescale_output(graph, node, store);
    let rescale_dtype = ir_dtype_to_tosa(node.meta.val.dtype);

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

    result.operators.push(TosaOperator::new(
        Op::Rescale,
        Attribute::Rescale(RescaleAttribute {
            scale32: params.scale32,
            rounding_mode: RoundingMode::SingleRound,
            per_channel: params.per_channel,
            input_unsigned: params.input_unsigned,
            output_unsigned: params.output_unsigned,
        }),
        vec![
            input_names.first().cloned().unwrap_or_default(),
            params.multiplier_name,
            params.shift_name,
            input_zp_name,
            output_zp_name,
        ],
        vec![output_plan.output_name.clone()],
    ));

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
