use super::*;

pub(in crate::serialize::visitors) fn emit_transpose_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let out_pad = kwarg_i32_list(node, "out_pad");
    let stride = kwarg_i32_list(node, "stride");
    let local_bound = kwarg_bool(node, "local_bound", false);
    let acc_type = DType::Int32;

    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let (input_zp_name, weight_zp_name) = push_input_weight_zp_tensors(
            &mut result,
            graph,
            node,
            kwarg_i32(node, "input_zp", 0),
            kwarg_i32(node, "weight_zp", 0),
        );
        extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
    };

    result.operators.push(TosaOperator {
        op: Op::TransposeConv2d,
        attribute: Attribute::TransposeConv2d(TransposeConv2dAttribute {
            out_pad: if out_pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                out_pad
            },
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            local_bound,
            acc_type,
        }),
        inputs: conv_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}
