use super::emitter::EmitContext;
use super::*;

mod conv;
mod pad;
mod pool;
mod rescale;
mod transpose_conv;

pub fn register(table: &mut EmitterTable) {
    for register in [
        conv::register as fn(&mut EmitterTable),
        pad::register,
        pool::register,
        rescale::register,
        transpose_conv::register,
    ] {
        register(table);
    }
}

fn push_input_weight_zp_tensors(
    result: &mut EmitResult,
    graph: &Graph,
    node: &Node,
    input_zp_val: i32,
    weight_zp_val: i32,
) -> (String, String) {
    let input_dtype = node_tensor_dtype(graph, node, 0);
    let weight_dtype = node_tensor_dtype(graph, node, 1);
    let input_zp_name = format!("{}_input_zp", node.name);
    push_zero_point_const(result, input_zp_name.clone(), input_dtype, input_zp_val);

    let weight_zp_name = format!("{}_weight_zp", node.name);
    push_zero_point_const(result, weight_zp_name.clone(), weight_dtype, weight_zp_val);

    (input_zp_name, weight_zp_name)
}

fn input_weight_zero_points(node: &Node, kwarg_fallback: bool) -> (i32, i32) {
    let input_zp = input_qparam_zero_point(node, 0)
        .or_else(|| kwarg_fallback.then(|| kwarg_i32(node, "input_zp", 0)))
        .unwrap_or(0);
    let weight_zp = input_qparam_zero_point(node, 1)
        .or_else(|| kwarg_fallback.then(|| kwarg_i32(node, "weight_zp", 0)))
        .unwrap_or(0);
    (input_zp, weight_zp)
}

fn aten_conv_groups(node: &Node) -> i64 {
    let groups_arg = if node.args.len() > 8 { 8 } else { 6 };
    arg_i64(node, groups_arg, 1)
}

fn input_rank(graph: &Graph, node: &Node) -> usize {
    match node.args.first() {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(|n| n.meta.val.shape_len as usize)
            .unwrap_or(0),
        _ => 0,
    }
}

fn conv_stride_arg(node: &Node, spatial_rank: usize) -> Vec<i32> {
    kwarg_or_arg_i32_list_or_repeat(node, "stride", 3, spatial_rank, 1)
}

fn conv_dilation_arg(node: &Node, spatial_rank: usize) -> Vec<i32> {
    kwarg_or_arg_i32_list_or_repeat(node, "dilation", 5, spatial_rank, 1)
}

fn conv_pad_arg(node: &Node, pad_len: usize) -> Vec<i32> {
    kwarg_or_arg_i32_list_or_repeat(node, "pad", 4, pad_len, 0)
}

fn extend_conv_inputs(
    mut input_names: Vec<String>,
    input_zp_name: String,
    weight_zp_name: String,
) -> Vec<String> {
    input_names.truncate(3);
    input_names.push(input_zp_name);
    input_names.push(weight_zp_name);
    input_names
}

fn prepare_conv_inputs(
    result: &mut EmitResult,
    graph: &Graph,
    node: &Node,
    mut input_names: Vec<String>,
    acc_type: DType,
    input_zp: i32,
    weight_zp: i32,
) -> Vec<String> {
    if matches!(input_names.len(), 2 | 4) {
        input_names.insert(
            2,
            push_zero_bias_tensor(result, node, synthetic_bias_dtype(node, acc_type)),
        );
    }
    if input_names.len() >= 5 {
        return input_names;
    }
    let (input_zp_name, weight_zp_name) =
        push_input_weight_zp_tensors(result, graph, node, input_zp, weight_zp);
    extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
}

fn symmetric_spatial_pad(padding: &[i32], spatial_rank: usize) -> Vec<i32> {
    if padding.len() == spatial_rank * 2 {
        return padding.to_vec();
    }
    let mut pad = Vec::with_capacity(spatial_rank * 2);
    for axis in 0..spatial_rank {
        let value = padding
            .get(axis)
            .copied()
            .or_else(|| padding.first().copied())
            .unwrap_or(0);
        pad.push(value);
        pad.push(value);
    }
    pad
}

fn synthetic_bias_dtype(node: &Node, acc_type: DType) -> DType {
    let semantic_dtype = node_output_dtype(node);
    if node.meta.input_qparams.is_empty() && is_float_dtype(semantic_dtype) {
        semantic_dtype
    } else {
        acc_type
    }
}

fn output_channels(node: &Node) -> i32 {
    node.meta
        .val
        .shape_slice()
        .get(1)
        .copied()
        .or_else(|| node.meta.val.shape_slice().first().copied())
        .unwrap_or(0) as i32
}

fn zero_bytes(dtype: DType, len: usize) -> Vec<u8> {
    vec![0; tosa_dtype_storage_size(dtype) * len]
}

fn push_zero_bias_tensor(result: &mut EmitResult, node: &Node, dtype: DType) -> String {
    let channels = output_channels(node).max(0) as usize;
    result.push_const(TosaTensor::with_data(
        &format!("{}_bias", node.name),
        vec![channels as i32],
        dtype,
        zero_bytes(dtype, channels),
    ))
}
