use super::*;

pub(in crate::serialize::visitors) fn emit_aten_max_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(graph, node);
    let input_names = collect_input_names(graph, node);
    let kernel = arg_i32_list(node, 1);
    let stride = arg_i32_list(node, 2);
    let padding = arg_i32_list(node, 3);
    let pad = if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    };

    let kernel_for_shape = if kernel.is_empty() { vec![2, 2] } else { kernel.clone() };
    let stride_for_shape = if stride.is_empty() { vec![2, 2] } else { stride.clone() };
    let mut output_tensor =
        compute_pool_nhwc_output(graph, store, node, &kernel_for_shape, &stride_for_shape, &pad);
    output_tensor.name = output_name.clone();
    result.tensors.push(output_tensor);
    result.operators.push(TosaOperator {
        op: Op::MaxPool2d,
        attribute: Attribute::MaxPool2d(MaxPool2dAttribute {
            kernel: if kernel.is_empty() { vec![2, 2] } else { kernel },
            stride: if stride.is_empty() { vec![2, 2] } else { stride },
            pad,
            nan_mode: NanPropagationMode::Propagate,
        }),
        inputs: input_names,
        outputs: vec![output_name],
        location: None,
    });
    result
}

pub(in crate::serialize::visitors) fn emit_max_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(graph, node);
    let input_names = collect_input_names(graph, node);
    let kernel = kwarg_i32_list(node, "kernel");
    let stride = kwarg_i32_list(node, "stride");
    let pad = kwarg_i32_list(node, "pad");

    let kernel_for_shape = if kernel.is_empty() { vec![2, 2] } else { kernel.clone() };
    let stride_for_shape = if stride.is_empty() { vec![2, 2] } else { stride.clone() };
    let pad_for_shape = if pad.is_empty() { vec![0, 0, 0, 0] } else { pad.clone() };
    let mut output_tensor = compute_pool_nhwc_output(
        graph,
        store,
        node,
        &kernel_for_shape,
        &stride_for_shape,
        &pad_for_shape,
    );
    output_tensor.name = output_name.clone();
    result.tensors.push(output_tensor);
    result.operators.push(TosaOperator {
        op: Op::MaxPool2d,
        attribute: Attribute::MaxPool2d(MaxPool2dAttribute {
            kernel: if kernel.is_empty() { vec![2, 2] } else { kernel },
            stride: if stride.is_empty() { vec![2, 2] } else { stride },
            pad: if pad.is_empty() { vec![0, 0, 0, 0] } else { pad },
            nan_mode: NanPropagationMode::Propagate,
        }),
        inputs: input_names,
        outputs: vec![output_name],
        location: None,
    });
    result
}

pub(in crate::serialize::visitors) fn emit_avg_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let kernel = {
        let kw = kwarg_i32_list(node, "kernel");
        if kw.is_empty() { arg_i32_list(node, 1) } else { kw }
    };
    let mut stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() { arg_i32_list(node, 2) } else { kw }
    };
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if !kw.is_empty() {
            kw
        } else {
            let padding = arg_i32_list(node, 3);
            if padding.len() >= 2 {
                vec![padding[0], padding[0], padding[1], padding[1]]
            } else if padding.len() == 1 {
                vec![padding[0], padding[0], padding[0], padding[0]]
            } else {
                vec![0, 0, 0, 0]
            }
        }
    };
    if stride.is_empty() {
        stride = kernel.clone();
    }

    result.tensors.push(compute_pool_nhwc_output(
        graph, store, node, &kernel, &stride, &pad,
    ));
    let acc_type = DType::Int32;
    let input_zp = node_input_zero_point(graph, node, 0, "input_zp", 0);
    let output_zp = node_output_zero_point(node, "output_zp", 0);
    let input_dtype = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|n| ir_dtype_to_tosa(n.meta.val.dtype))
            .unwrap_or(DType::Int8),
        _ => DType::Int8,
    };
    let output_dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    let input_zp_name = format!("{}_input_zp", node.name);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: input_dtype,
        data: encode_zp_for_dtype(input_zp, input_dtype),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![input_zp_name.clone()],
        location: None,
    });

    let output_zp_name = format!("{}_output_zp", node.name);
    result.tensors.push(TosaTensor {
        name: output_zp_name.clone(),
        shape: vec![1],
        dtype: output_dtype,
        data: encode_zp_for_dtype(output_zp, output_dtype),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![output_zp_name.clone()],
        location: None,
    });

    let mut pool_inputs = input_names;
    pool_inputs.push(input_zp_name);
    pool_inputs.push(output_zp_name);
    result.operators.push(TosaOperator {
        op: Op::AvgPool2d,
        attribute: Attribute::AvgPool2d(AvgPool2dAttribute {
            kernel,
            stride,
            pad,
            acc_type,
        }),
        inputs: pool_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}
