use super::*;

mod conv;
mod pad;
mod pool;
mod rescale;
mod transpose_conv;

pub(super) use conv::{emit_aten_conv2d, emit_conv2d, emit_conv3d, emit_depthwise_conv2d};
pub(super) use pad::emit_pad;
pub(super) use pool::{emit_aten_max_pool2d, emit_avg_pool2d, emit_max_pool2d};
pub(super) use rescale::emit_rescale;
pub(super) use transpose_conv::emit_transpose_conv2d;

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
    push_inline_const(
        result,
        make_rank1_const_tensor(
            input_zp_name.clone(),
            input_dtype,
            encode_zp_for_dtype(input_zp_val, input_dtype),
            1,
        ),
    );

    let weight_zp_name = format!("{}_weight_zp", node.name);
    push_inline_const(
        result,
        make_rank1_const_tensor(
            weight_zp_name.clone(),
            weight_dtype,
            encode_zp_for_dtype(weight_zp_val, weight_dtype),
            1,
        ),
    );

    (input_zp_name, weight_zp_name)
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
