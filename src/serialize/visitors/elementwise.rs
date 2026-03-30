use super::*;

mod comparison;
mod logical;
mod unary;

pub(super) use comparison::{
    emit_aten_eq, emit_aten_ge, emit_aten_gt, emit_aten_le, emit_aten_lt, emit_equal, emit_greater,
    emit_greater_equal, emit_maximum, emit_minimum, emit_pow,
};
pub(super) use logical::{
    emit_arithmetic_right_shift, emit_bitwise_and, emit_bitwise_not, emit_bitwise_or,
    emit_bitwise_xor, emit_logical_and, emit_logical_left_shift, emit_logical_not, emit_logical_or,
    emit_logical_xor,
};
pub(super) use unary::{
    emit_abs, emit_cast, emit_cos, emit_erf, emit_exp, emit_log, emit_reciprocal, emit_rsqrt,
    emit_sigmoid, emit_sin, emit_table, emit_tanh,
};

fn emit_direct_unary(graph: &Graph, node: &Node, op: Op, attribute: Attribute) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator::new(op, attribute, input_names, vec![node.name.clone()]));
    result
}

fn emit_direct_binary(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    let mut result = EmitResult::default();
    let mut inputs = collect_input_names(graph, node);
    maybe_materialize_binary_broadcast(graph, store, node, &mut result, &mut inputs);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator::new(op, attribute, inputs, vec![node.name.clone()]));
    result
}

fn emit_bool_binary(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    op: Op,
    attribute: Attribute,
    swap_inputs: bool,
) -> EmitResult {
    let mut result = EmitResult::default();
    let mut inputs = collect_input_names(graph, node);
    maybe_materialize_binary_broadcast(graph, store, node, &mut result, &mut inputs);
    if swap_inputs && inputs.len() >= 2 {
        inputs.swap(0, 1);
    }
    result
        .tensors
        .push(make_bool_output_tensor(graph, store, node));
    result.operators.push(TosaOperator::new(op, attribute, inputs, vec![node.name.clone()]));
    result
}
