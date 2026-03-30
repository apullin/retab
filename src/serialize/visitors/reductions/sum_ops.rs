use super::*;

pub(in crate::serialize::visitors) fn emit_reduce_sum(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
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

    result.tensors.push(TosaTensor::new(
        reduce_output.clone(),
        reduce_shape.clone(),
        spec.reduce_dtype,
    ));

    result.operators.push(TosaOperator::new(
        Op::ReduceSum,
        Attribute::ReduceSum(ReduceSumAttribute {
            axis: spec.tosa_axis,
        }),
        spec.input_names,
        vec![reduce_output.clone()],
    ));

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

pub(in crate::serialize::visitors) fn emit_aten_sum(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();

    if spec.input_name.is_empty() {
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator::new(
            Op::ReduceSum,
            Attribute::ReduceSum(ReduceSumAttribute { axis: 0 }),
            vec![spec.input_name],
            vec![node.name.clone()],
        ));
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

        result.tensors.push(TosaTensor::new(
            out_name.clone(),
            out_shape.clone(),
            base_dtype,
        ));

        result.operators.push(TosaOperator::new(
            Op::ReduceSum,
            Attribute::ReduceSum(ReduceSumAttribute {
                axis: axis.tosa_axis,
            }),
            vec![current_input.clone()],
            vec![out_name.clone()],
        ));

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
