use super::*;

fn first_tensor_ir_dtype(graph: &Graph, node: &Node) -> crate::graph::ir::DType {
    node.args
        .iter()
        .find_map(|arg| graph.node(arg.as_node()?).map(|input| input.meta.val.dtype))
        .unwrap_or(node.meta.val.dtype)
}

/// Collect the two logical binary operands, materializing scalar literals as
/// TOSA CONST tensors. FX often represents predicates like `x.sum() > 0` as a
/// tensor node plus a numeric literal; plain `collect_input_names` would drop
/// that literal and emit an invalid unary comparison.
pub(super) fn collect_binary_input_names(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    result: &mut EmitResult,
) -> Vec<String> {
    let const_shape = binary_input_broadcast_shape(graph, store, node)
        .unwrap_or_else(|| actual_tensor_shape(graph, store, node));
    let const_dtype = first_tensor_ir_dtype(graph, node);
    let mut names = Vec::with_capacity(2);
    let mut has_literal = false;

    for (idx, arg) in node.args.iter().enumerate() {
        if names.len() >= 2 {
            break;
        }

        match arg {
            Arg::Node(id) => {
                if let Some(name) = resolve_input_name_from_node_id(graph, *id) {
                    names.push(name);
                }
            }
            arg @ (Arg::Int(_) | Arg::Float(_) | Arg::Bool(_)) => {
                let name = format!("{}_input{}_const", node.name, idx);
                if push_filled_const_if_supported(
                    result,
                    name.clone(),
                    const_shape.clone(),
                    const_dtype,
                    arg.as_f64(),
                ) {
                    names.push(name);
                    has_literal = true;
                }
            }
            _ => {}
        }
    }

    if !has_literal {
        maybe_materialize_binary_broadcast(graph, store, node, result, &mut names);
    }

    names
}

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
        .filter_map(|arg| graph.node(arg.as_node()?))
        .collect();
    if input_nodes.len() < 2 {
        return;
    }

    let input_shapes = [
        actual_tensor_shape(graph, store, input_nodes[0]),
        actual_tensor_shape(graph, store, input_nodes[1]),
    ];

    let (smaller_idx, larger_idx) = if input_shapes[0].len() < input_shapes[1].len() {
        (0usize, 1usize)
    } else if input_shapes[1].len() < input_shapes[0].len() {
        (1usize, 0usize)
    } else {
        return;
    };
    let target_shape = broadcast_shape_for_binary_inputs(
        &raw_i32_shape(input_nodes[smaller_idx]),
        val_rank(input_nodes[larger_idx]),
    );

    let broadcast_name = format!("{}_input{}_broadcast", node.name, smaller_idx);
    let broadcast_dtype = actual_tensor_dtype(graph, input_nodes[smaller_idx]);

    let shape_name = format!("{}_shape", broadcast_name);
    push_const_shape(result, shape_name.clone(), target_shape.clone());
    result.push_output_op(
        TosaTensor::new(&broadcast_name, target_shape.clone(), broadcast_dtype),
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute {}),
        vec![input_names[smaller_idx].clone(), shape_name],
    );
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

    let target_shape =
        broadcast_shape_for_binary_inputs(&raw_i32_shape(bound_node), val_rank(clamp_node));

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
            push_filled_const_if_supported(result, name.clone(), shape, dtype, arg.as_f64())
                .then_some((name, None))
        }
        _ => None,
    }
}

/// The reshape target that brings a lower-rank binary or clamp operand up to `rank`, the
/// other operand's: its shape with leading 1s, as for numpy broadcasting.
pub(super) fn broadcast_shape_for_binary_inputs(smaller: &[i32], rank: usize) -> Vec<i32> {
    let mut padded = vec![1; rank.saturating_sub(smaller.len())];
    padded.extend_from_slice(smaller);
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_rank_operands_gain_leading_unit_axes() {
        assert_eq!(broadcast_shape_for_binary_inputs(&[4], 4), vec![1, 1, 1, 4]);
        assert_eq!(
            broadcast_shape_for_binary_inputs(&[4, 1], 4),
            vec![1, 1, 4, 1]
        );
        assert_eq!(
            broadcast_shape_for_binary_inputs(&[1, 25, 20], 4),
            vec![1, 1, 25, 20]
        );
    }
}
