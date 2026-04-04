use super::*;

pub(super) fn maybe_materialize_binary_broadcast(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    result: &mut EmitResult,
    input_names: &mut [String],
) {
    if input_names.len() < 2 {
        return;
    }

    let input_nodes: Vec<&Node> = node
        .args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Node(id) => graph.node(*id),
            _ => None,
        })
        .collect();
    if input_nodes.len() < 2 {
        return;
    }

    let input_shapes = [
        actual_tensor_shape(graph, store, input_nodes[0]),
        actual_tensor_shape(graph, store, input_nodes[1]),
    ];

    let (smaller_idx, target_shape) = if input_shapes[0].len() < input_shapes[1].len() {
        let target_shape = broadcast_shape_for_binary_inputs(
            &raw_i32_shape(input_nodes[0]),
            &raw_i32_shape(input_nodes[1]),
            input_nodes[1],
        );
        (0usize, target_shape)
    } else if input_shapes[1].len() < input_shapes[0].len() {
        let target_shape = broadcast_shape_for_binary_inputs(
            &raw_i32_shape(input_nodes[1]),
            &raw_i32_shape(input_nodes[0]),
            input_nodes[0],
        );
        (1usize, target_shape)
    } else {
        return;
    };

    let broadcast_name = format!("{}_input{}_broadcast", node.name, smaller_idx);
    let shape_name = format!("{}_shape", broadcast_name);
    let rank = target_shape.len() as u32;
    let shape_data: Vec<u8> = target_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();

    let broadcast_dtype = actual_tensor_dtype(graph, input_nodes[smaller_idx]);

    result.tensors.push(TosaTensor::new(
        &broadcast_name,
        target_shape,
        broadcast_dtype,
    ));
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank,
        data: shape_data,
    });
    result.operators.push(TosaOperator::new(
        Op::ConstShape,
        Attribute::ConstShape(ConstShapeAttribute {}),
        vec![],
        vec![shape_name.clone()],
    ));
    result.operators.push(TosaOperator::new(
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute {}),
        vec![input_names[smaller_idx].clone(), shape_name],
        vec![broadcast_name.clone()],
    ));
    input_names[smaller_idx] = broadcast_name;
}

pub(super) fn maybe_materialize_clamp_bound_broadcast(
    graph: &Graph,
    store: &TensorStore,
    clamp_node: &Node,
    bound_node: &Node,
    bound_name: &mut String,
    result: &mut EmitResult,
    suffix: &str,
) {
    let clamp_shape = actual_tensor_shape(graph, store, clamp_node);
    let bound_shape = actual_tensor_shape(graph, store, bound_node);
    if bound_shape.len() >= clamp_shape.len() {
        return;
    }

    let target_shape = broadcast_shape_for_binary_inputs(
        &raw_i32_shape(bound_node),
        &raw_i32_shape(clamp_node),
        clamp_node,
    );

    let broadcast_name = format!("{}_{}_broadcast", clamp_node.name, suffix);
    let shape_name = format!("{}_shape", broadcast_name);
    push_reshape_with_shape(
        result,
        bound_name.clone(),
        broadcast_name.clone(),
        shape_name,
        target_shape,
        node_output_dtype(bound_node),
    );
    *bound_name = broadcast_name;
}

pub(super) fn resolve_clamp_bound_input(
    graph: &Graph,
    node: &Node,
    arg: Option<&Arg>,
    suffix: &str,
    output_rank: usize,
    dtype: crate::graph::ir::DType,
    result: &mut EmitResult,
) -> Option<(String, Option<NodeId>)> {
    let shape = vec![1; output_rank];
    match arg {
        Some(Arg::Node(id)) => {
            resolve_input_name_from_node_id(graph, *id).map(|name| (name, Some(*id)))
        }
        Some(arg @ (Arg::Int(_) | Arg::Float(_) | Arg::Bool(_))) => {
            let name = format!("{}_{}_const", node.name, suffix);
            let tensor = make_filled_const_tensor(name.clone(), shape, dtype, arg.as_f64())?;
            push_inline_const(result, tensor);
            Some((name, None))
        }
        _ => None,
    }
}

/// Best-effort rank-4 NHWC broadcast shape for lower-rank binary operands.
///
/// The direct fast lane commonly produces per-channel constants with shapes
/// like `[C, 1, 1]` feeding rank-4 NHWC binary ops. TOSA requires equal ranks
/// for these binary ops, so we materialize a broadcast reshape that preserves
/// the channel value in the last dimension.
pub(super) fn broadcast_shape_for_binary_inputs(
    smaller_raw: &[i32],
    larger_raw: &[i32],
    larger_node: &Node,
) -> Vec<i32> {
    let mut padded_raw = vec![1; larger_raw.len().saturating_sub(smaller_raw.len())];
    padded_raw.extend_from_slice(smaller_raw);

    if let Some(order) = larger_node.meta.tosa_dim_order.as_ref() {
        if order.len() == padded_raw.len() {
            let reordered: Vec<i32> = order
                .iter()
                .filter_map(|&idx| padded_raw.get(idx as usize).copied())
                .collect();
            if reordered.len() == padded_raw.len() {
                return reordered;
            }
        }
    }

    if larger_raw.len() == 4 && smaller_raw.len() < 4 {
        let non_ones: Vec<i32> = smaller_raw.iter().copied().filter(|&d| d != 1).collect();
        if non_ones.len() <= 1 {
            return vec![1, 1, 1, non_ones.first().copied().unwrap_or(1)];
        }
    }

    padded_raw
}
