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

    let (smaller_idx, target_shape) = if input_shapes[0].len() < input_shapes[1].len() {
        let target_shape = broadcast_shape_for_binary_inputs(
            &raw_i32_shape(input_nodes[0]),
            &raw_i32_shape(input_nodes[1]),
            node,
        );
        (0usize, target_shape)
    } else if input_shapes[1].len() < input_shapes[0].len() {
        let target_shape = broadcast_shape_for_binary_inputs(
            &raw_i32_shape(input_nodes[1]),
            &raw_i32_shape(input_nodes[0]),
            node,
        );
        (1usize, target_shape)
    } else {
        return;
    };

    let broadcast_name = format!("{}_input{}_broadcast", node.name, smaller_idx);
    let broadcast_dtype = actual_tensor_dtype(graph, input_nodes[smaller_idx]);

    let larger_idx = 1 - smaller_idx;
    if let Some((padded_shape, perms)) = broadcast_transpose_for_binary_inputs(
        &raw_i32_shape(input_nodes[smaller_idx]),
        &raw_i32_shape(input_nodes[larger_idx]),
        node,
    ) {
        let pre_name = format!("{}_pre", broadcast_name);
        push_reshape_with_shape(
            result,
            input_names[smaller_idx].clone(),
            pre_name.clone(),
            format!("{}_shape", pre_name),
            padded_shape,
            broadcast_dtype,
        );
        result.push_output_op(
            TosaTensor::new(&broadcast_name, target_shape, broadcast_dtype),
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute { perms }),
            vec![pre_name],
        );
        input_names[smaller_idx] = broadcast_name;
        return;
    }

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
            push_filled_const_if_supported(result, name.clone(), shape, dtype, arg.as_f64())
                .then_some((name, None))
        }
        _ => None,
    }
}

/// Best-effort broadcast reshape target for lower-rank binary operands.
///
/// The layout contract belongs to the current binary/clamp op, not
/// necessarily to either operand. Placeholder-backed operands often still
/// carry raw NCHW shapes while the current op emits NHWC, so the reshape
/// target must be reordered using the current op's `tosa_dim_order`.
pub(super) fn broadcast_shape_for_binary_inputs(
    smaller_raw: &[i32],
    larger_raw: &[i32],
    layout_node: &Node,
) -> Vec<i32> {
    let mut padded_raw = vec![1; larger_raw.len().saturating_sub(smaller_raw.len())];
    padded_raw.extend_from_slice(smaller_raw);

    if let Some(order) = layout_node.meta.tosa_dim_order.as_ref() {
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

    padded_raw
}

/// A flat reshape into the reordered target only preserves data when the
/// layout keeps the operand's non-unit axes in their original relative order
/// (true for the common per-channel [1,C,1,1] -> [1,1,1,C] case). When the
/// layout permutes non-unit axes, the operand must instead be reshaped to the
/// rank-padded logical shape and physically transposed; returns that
/// (padded_shape, perms) plan, or None when the plain reshape is safe.
pub(super) fn broadcast_transpose_for_binary_inputs(
    smaller_raw: &[i32],
    larger_raw: &[i32],
    layout_node: &Node,
) -> Option<(Vec<i32>, Vec<i32>)> {
    let mut padded_raw = vec![1; larger_raw.len().saturating_sub(smaller_raw.len())];
    padded_raw.extend_from_slice(smaller_raw);

    let order = layout_node.meta.tosa_dim_order.as_ref()?;
    if order.len() != padded_raw.len() {
        return None;
    }
    let mut last_nonunit: i64 = -1;
    let mut permutes_nonunit = false;
    for &axis in order.iter() {
        if padded_raw.get(axis as usize).copied().unwrap_or(1) != 1 {
            if (axis as i64) < last_nonunit {
                permutes_nonunit = true;
                break;
            }
            last_nonunit = axis as i64;
        }
    }
    if !permutes_nonunit {
        return None;
    }
    let perms: Vec<i32> = order.iter().map(|&axis| axis as i32).collect();
    Some((padded_raw, perms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{
        DType as IrDType, NodeMeta as GraphNodeMeta, OpTarget, OpType, TensorInfo,
    };
    use crate::graph::Graph;
    use smallvec::{smallvec, SmallVec};

    fn meta(shape: &[i64], dtype: IrDType) -> GraphNodeMeta {
        GraphNodeMeta::new(TensorInfo::new(shape, dtype))
    }

    #[test]
    fn one_d_broadcast_preserves_trailing_axis_semantics_in_nhwc() {
        let mut graph = Graph::new();
        let mut larger_meta = meta(&[1, 2, 4, 4], IrDType::Int32);
        larger_meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));
        let larger = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("larger".to_string()),
            smallvec![],
            Vec::new(),
            larger_meta,
            "larger".to_string(),
        );
        let larger_node = graph.node(larger).expect("larger node");

        assert_eq!(
            broadcast_shape_for_binary_inputs(&[4], &[1, 2, 4, 4], larger_node),
            vec![1, 1, 4, 1]
        );
    }

    #[test]
    fn two_d_broadcast_preserves_trailing_axes_semantics_in_nhwc() {
        let mut graph = Graph::new();
        let mut larger_meta = meta(&[1, 1, 4, 4], IrDType::Int32);
        larger_meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));
        let larger = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("larger".to_string()),
            smallvec![],
            Vec::new(),
            larger_meta,
            "larger".to_string(),
        );
        let larger_node = graph.node(larger).expect("larger node");

        assert_eq!(
            broadcast_shape_for_binary_inputs(&[4, 1], &[1, 1, 4, 4], larger_node),
            vec![1, 4, 1, 1]
        );
    }

    #[test]
    fn broadcast_uses_current_binary_layout_when_operands_do_not() {
        let mut graph = Graph::new();
        let larger = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("larger".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 10, 25, 20], IrDType::Int32),
            "larger".to_string(),
        );
        let mut binary_meta = meta(&[1, 10, 25, 20], IrDType::Int32);
        binary_meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));
        let binary = graph.create_node(
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![],
            Vec::new(),
            binary_meta,
            "binary".to_string(),
        );
        let binary_node = graph.node(binary).expect("binary node");
        let larger_node = graph.node(larger).expect("larger node");

        assert_eq!(larger_node.meta.tosa_dim_order, None);
        assert_eq!(
            broadcast_shape_for_binary_inputs(&[1, 25, 20], &[1, 10, 25, 20], binary_node),
            vec![1, 25, 20, 1]
        );
    }
}
