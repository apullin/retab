use super::*;
use super::emitter::EmitContext;

mod comparison;
mod logical;
mod unary;

pub fn register(table: &mut std::collections::HashMap<crate::graph::ir::OpTarget, super::emitter::EmitFn>) {
    unary::register(table);
    logical::register(table);
    comparison::register(table);
}


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
