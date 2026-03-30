use super::*;
use super::emitter::EmitContext;

mod conv;
mod pad;
mod pool;
mod rescale;
mod transpose_conv;

pub fn register(table: &mut std::collections::HashMap<crate::graph::ir::OpTarget, super::emitter::EmitFn>) {
    conv::register(table);
    pad::register(table);
    pool::register(table);
    rescale::register(table);
    transpose_conv::register(table);
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
