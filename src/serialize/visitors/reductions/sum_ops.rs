use super::*;

pub(in crate::serialize::visitors) fn emit_reduce_sum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let spec = ParsedSingleAxisReduceSpec::parse(graph, node, store);
    let mut result = EmitResult::default();

    let mut reduce_shape = spec.input_shape.clone();
    if let Some(slot) = reduce_shape.get_mut(spec.tosa_axis as usize) {
        *slot = 1;
    }

    let reduce_output = if spec.keepdim {
        node.name.clone()
    } else {
        format!("{}_reduce_axis{}", node.name, spec.raw_axis)
    };

    result.tensors.push(TosaTensor {
        name: reduce_output.clone(),
        shape: reduce_shape.clone(),
        dtype: spec.reduce_dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    result.operators.push(TosaOperator {
        op: Op::ReduceSum,
        attribute: Attribute::ReduceSum(ReduceSumAttribute {
            axis: spec.tosa_axis,
        }),
        inputs: spec.input_names,
        outputs: vec![reduce_output.clone()],
        location: None,
    });

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            reduce_output,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            node_output_dtype(node),
        );
    }

    result
}

pub(in crate::serialize::visitors) fn emit_aten_sum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();

    if spec.input_name.is_empty() {
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator {
            op: Op::ReduceSum,
            attribute: Attribute::ReduceSum(ReduceSumAttribute { axis: 0 }),
            inputs: vec![spec.input_name],
            outputs: vec![node.name.clone()],
            location: None,
        });
        return result;
    }

    let input_node = match node.args.first() {
        Some(Arg::Node(id)) => graph.node(*id),
        _ => None,
    };
    let input_dtype = input_node
        .map(|n| passthrough_input_dtype(graph, n))
        .unwrap_or_else(|| node_output_dtype(node));
    let base_dtype = match input_dtype {
        DType::Int8 | DType::Int16 => DType::Int32,
        _ => node_output_dtype(node),
    };
    let mut current_input = spec.input_name;
    let mut current_shape = spec.input_shape;

    for (idx, axis) in spec.axes.iter().enumerate() {
        let mut out_shape = current_shape.clone();
        if let Some(slot) = out_shape.get_mut(axis.tosa_axis as usize) {
            *slot = 1;
        }

        let is_last_reduce = idx == spec.axes.len() - 1 && spec.keepdim;
        let out_name = if is_last_reduce {
            node.name.clone()
        } else {
            format!("{}_reduce_axis{}", node.name, axis.raw_dim)
        };

        result.tensors.push(TosaTensor {
            name: out_name.clone(),
            shape: out_shape.clone(),
            dtype: base_dtype,
            data: Vec::new(),
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });

        result.operators.push(TosaOperator {
            op: Op::ReduceSum,
            attribute: Attribute::ReduceSum(ReduceSumAttribute {
                axis: axis.tosa_axis,
            }),
            inputs: vec![current_input.clone()],
            outputs: vec![out_name.clone()],
            location: None,
        });

        current_input = out_name;
        current_shape = out_shape;
    }

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            current_input,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            base_dtype,
        );
    }

    result
}
