use super::*;

pub(super) fn emit_direct_unary(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    let mut result = EmitResult::default();
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        op,
        attribute,
        collect_input_names(graph, node),
    );
    result
}

pub(super) fn emit_direct_binary(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    let mut result = EmitResult::default();
    let mut inputs = collect_input_names(graph, node);
    maybe_materialize_binary_broadcast(graph, store, node, &mut result, &mut inputs);
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        op,
        attribute,
        inputs,
    );
    result
}

pub(super) fn emit_bool_binary(
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
    let broadcast_shape = actual_tensor_shape(graph, store, node);
    for (idx, arg) in node.args.iter().enumerate().take(inputs.len()) {
        let Some(input_node) = (match arg {
            Arg::Node(id) => graph.node(*id),
            _ => None,
        }) else {
            continue;
        };
        let semantic_dtype = node_output_dtype(input_node);
        let structural_dtype = actual_tensor_dtype(graph, input_node);
        if !matches!(semantic_dtype, DType::Fp32 | DType::Fp16 | DType::Bf16)
            || semantic_dtype == structural_dtype
        {
            continue;
        }
        let cast_name = format!("{}_input{}_cast", node.name, idx);
        result.tensors.push(temp_tensor(
            cast_name.clone(),
            broadcast_shape.clone(),
            semantic_dtype,
        ));
        result.operators.push(TosaOperator::new(
            Op::Cast,
            Attribute::Cast(CastAttribute {}),
            vec![inputs[idx].clone()],
            vec![cast_name.clone()],
        ));
        inputs[idx] = cast_name;
    }
    if swap_inputs && inputs.len() >= 2 {
        inputs.swap(0, 1);
    }
    result.push_output_op(
        make_bool_output_tensor(graph, store, node),
        op,
        attribute,
        inputs,
    );
    result
}

pub(super) fn emit_identity(graph: &Graph, store: &TensorStore, node: &Node) -> EmitResult {
    emit_direct_unary(graph, store, node, Op::Identity, Attribute::None)
}

pub(super) fn is_integral_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Int8 | DType::Int16 | DType::Int32)
}

pub(super) fn push_shift_const(result: &mut EmitResult, node_name: &str) -> String {
    let shift_name = format!("{}_shift", node_name);
    result.push_const(make_rank1_const_tensor(
        shift_name.clone(),
        DType::Int8,
        vec![0u8],
        1,
    ));
    shift_name
}

pub(super) fn push_rank1_const(
    result: &mut EmitResult,
    name: String,
    dtype: DType,
    data: Vec<u8>,
    len: usize,
) -> String {
    push_inline_const(
        result,
        make_rank1_const_tensor(name.clone(), dtype, data, len),
    );
    name
}

pub(super) fn temp_tensor(name: String, shape: Vec<i32>, dtype: DType) -> TosaTensor {
    make_empty_tensor(name, shape, dtype)
}
