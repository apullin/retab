use super::*;

pub(super) fn binary_output_shapes(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> (Vec<i32>, Vec<i32>) {
    let logical_shape = actual_tensor_shape(graph, store, node);
    if let Some(input_shape) = primary_binary_input_layout_shape(graph, store, node, &logical_shape)
    {
        return (logical_shape, input_shape);
    }
    let emitted_input_shapes = emitted_binary_input_shapes(graph, store, node);
    let equal_rank_layout_shape = match emitted_input_shapes.as_slice() {
        [lhs, rhs] => equal_rank_binary_layout_target_shape(&[lhs.clone(), rhs.clone()], node),
        _ => None,
    };
    let emitted_shape = equal_rank_layout_shape.unwrap_or_else(|| {
        emitted_input_shapes
            .into_iter()
            .reduce(|lhs, rhs| broadcast_two_shapes(&lhs, &rhs).unwrap_or(rhs))
            .unwrap_or_else(|| {
                binary_input_broadcast_shape(graph, store, node)
                    .unwrap_or_else(|| logical_shape.clone())
            })
    });
    (logical_shape, emitted_shape)
}

fn primary_binary_input_layout_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    logical_shape: &[i32],
) -> Option<Vec<i32>> {
    let input_shapes: Vec<Vec<i32>> = node
        .args
        .iter()
        .filter_map(|arg| graph.node(arg.as_node()?))
        .take(2)
        .map(|input| actual_tensor_shape(graph, store, input))
        .collect();
    let primary = input_shapes.first()?;
    if primary == logical_shape
        || primary.len() != logical_shape.len()
        || primary.iter().product::<i32>() != logical_shape.iter().product::<i32>()
    {
        return None;
    }
    if input_shapes
        .iter()
        .skip(1)
        .any(|shape| shape.iter().product::<i32>() != 1)
    {
        return None;
    }
    if input_shapes
        .iter()
        .skip(1)
        .any(|shape| !same_rank_or_scalar_broadcast(shape, primary))
    {
        return None;
    }
    Some(primary.clone())
}

fn same_rank_or_scalar_broadcast(shape: &[i32], target: &[i32]) -> bool {
    shape.len() < target.len()
        || (shape.len() == target.len()
            && shape
                .iter()
                .zip(target.iter())
                .all(|(&dim, &target_dim)| dim == target_dim || dim == 1))
}

fn output_layout_transition_perms(
    node: &Node,
    emitted_shape: &[i32],
    logical_shape: &[i32],
) -> Option<Vec<i32>> {
    if emitted_shape.len() != logical_shape.len() {
        return None;
    }

    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if order.len() != emitted_shape.len() {
            return None;
        }
        if apply_tosa_dim_order(node, emitted_shape.to_vec()) == logical_shape {
            return Some(order.iter().map(|&axis| axis as i32).collect());
        }
        if apply_tosa_dim_order(node, logical_shape.to_vec()) == emitted_shape {
            return inverse_tosa_dim_order(order);
        }
    }

    if let Some(order) = channel_last_order(logical_shape.len()) {
        if permute_shape(logical_shape, &order).as_deref() == Some(emitted_shape) {
            return inverse_permutation(&order);
        }
        if permute_shape(emitted_shape, &order).as_deref() == Some(logical_shape) {
            return Some(order);
        }
    }

    infer_shape_permutation(emitted_shape, logical_shape)
}

pub(super) fn push_binary_output_with_contract(
    result: &mut EmitResult,
    node: &Node,
    op: Op,
    attribute: Attribute,
    inputs: Vec<String>,
    dtype: DType,
    logical_shape: Vec<i32>,
    emitted_shape: Vec<i32>,
) {
    // The temp+reshape contract is only valid for a rank/layout change (same element
    // count). When emitted_shape is a pure broadcast of logical_shape (different element
    // count), TOSA's native size-1 broadcasting already yields logical_shape, so emit
    // directly rather than via an (invalid) element-count-changing reshape.
    let emitted_elems = emitted_shape.iter().map(|&d| d as i64).product::<i64>();
    let logical_elems = logical_shape.iter().map(|&d| d as i64).product::<i64>();
    if emitted_shape != logical_shape && emitted_elems == logical_elems {
        let temp_name = format!("{}_broadcast_output", node.name);
        result.push_output_op(
            make_empty_tensor_for_node(node, temp_name.clone(), emitted_shape.clone(), dtype),
            op,
            attribute,
            inputs,
        );
        if let Some(perms) = output_layout_transition_perms(node, &emitted_shape, &logical_shape) {
            result.push_output_op(
                make_empty_tensor_for_node(node, node.name.clone(), logical_shape, dtype),
                Op::Transpose,
                Attribute::Transpose(TransposeAttribute { perms }),
                vec![temp_name],
            );
        } else {
            push_reshape_with_shape_for_node(
                result,
                Some(node),
                temp_name,
                node.name.clone(),
                format!("{}_shape", node.name),
                logical_shape,
                dtype,
            );
        }
    } else {
        result.push_output_op(
            make_empty_tensor_for_node(node, node.name.clone(), logical_shape, dtype),
            op,
            attribute,
            inputs,
        );
    }
}

pub(super) fn emit_direct_unary(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    let mut result = EmitResult::default();
    let output_shape = node
        .args
        .first()
        .and_then(|arg| arg.as_node())
        .and_then(|id| graph.node(id))
        .map(|input| actual_tensor_shape(graph, store, input))
        .unwrap_or_else(|| actual_tensor_shape(graph, store, node));
    result.push_output_op(
        make_empty_tensor_for_node(
            node,
            node.name.clone(),
            output_shape,
            node_output_dtype(node),
        ),
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
    let inputs = collect_binary_input_names(graph, store, node, &mut result);
    let (logical_shape, emitted_shape) = binary_output_shapes(graph, store, node);
    let output_dtype = node_output_dtype(node);
    push_binary_output_with_contract(
        &mut result,
        node,
        op,
        attribute,
        inputs,
        output_dtype,
        logical_shape,
        emitted_shape,
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
    let mut inputs = collect_binary_input_names(graph, store, node, &mut result);
    let (logical_shape, emitted_shape) = binary_output_shapes(graph, store, node);
    for (idx, arg) in node.args.iter().enumerate().take(inputs.len()) {
        let Some(input_node) = arg.as_node().and_then(|id| graph.node(id)) else {
            continue;
        };
        let semantic_dtype = node_output_dtype(input_node);
        let structural_dtype = actual_tensor_dtype(graph, input_node);
        if !is_float_dtype(semantic_dtype) || semantic_dtype == structural_dtype {
            continue;
        }
        let cast_name = format!("{}_input{}_cast", node.name, idx);
        result.push_output_op(
            temp_tensor(cast_name.clone(), emitted_shape.clone(), semantic_dtype),
            Op::Cast,
            Attribute::Cast(CastAttribute {}),
            vec![inputs[idx].clone()],
        );
        inputs[idx] = cast_name;
    }
    if swap_inputs && inputs.len() >= 2 {
        inputs.swap(0, 1);
    }
    push_binary_output_with_contract(
        &mut result,
        node,
        op,
        attribute,
        inputs,
        DType::Bool,
        logical_shape,
        emitted_shape,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{
        DType as IrDType, NodeMeta as GraphNodeMeta, OpTarget, OpType, TensorInfo,
    };
    use crate::graph::Graph;
    use smallvec::smallvec;

    fn meta(shape: &[i64], dtype: IrDType) -> GraphNodeMeta {
        GraphNodeMeta::new(TensorInfo::new(shape, dtype))
    }

    #[test]
    fn binary_emit_uses_native_same_rank_broadcast() {
        let mut graph = Graph::new();
        let x = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 4, 5, 1], IrDType::Float32),
            "x".to_string(),
        );
        let y = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("y".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 4, 1, 1], IrDType::Float32),
            "y".to_string(),
        );
        let add = graph.create_node(
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(x), Arg::Node(y)],
            Vec::new(),
            meta(&[1, 4, 5], IrDType::Float32),
            "add".to_string(),
        );
        let store = TensorStore::default();

        let result = emit_direct_binary(
            &graph,
            graph.node(add).expect("add node"),
            &store,
            Op::Add,
            Attribute::Add(AddAttribute {}),
        );

        assert_eq!(result.operators.len(), 1);
        assert_eq!(result.operators[0].op, Op::Add);
        assert_eq!(result.operators[0].outputs, vec!["add".to_string()]);
        assert_eq!(result.tensors[0].shape, vec![1, 4, 5, 1]);
    }

    #[test]
    fn binary_emit_uses_post_reshape_broadcast_shape_for_lower_rank_rhs() {
        let mut graph = Graph::new();

        let lhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("lhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 1, 1024, 8], IrDType::Int32),
            "lhs".to_string(),
        );

        let rhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("rhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[8, 1, 1], IrDType::Int32),
            "rhs".to_string(),
        );

        let mut add_meta = meta(&[1, 1, 1024, 8], IrDType::Int32);
        add_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let add = graph.create_node(
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(lhs), Arg::Node(rhs)],
            Vec::new(),
            add_meta,
            "add".to_string(),
        );

        let store = TensorStore::default();
        let result = emit_direct_binary(
            &graph,
            graph.node(add).expect("add node"),
            &store,
            Op::Add,
            Attribute::Add(AddAttribute {}),
        );

        // The lower-rank rhs is reshaped up to the broadcast rank before the add;
        // TOSA then broadcasts size-1 axes inside the binary op.
        assert_eq!(result.operators[0].op, Op::ConstShape);
        assert_eq!(result.operators[1].op, Op::Reshape);
        assert_eq!(
            result.operators[1].outputs,
            vec!["add_input1_broadcast".to_string()]
        );
        assert_eq!(result.operators[2].op, Op::Add);
        assert_eq!(result.operators[2].outputs, vec!["add".to_string()]);
        assert_eq!(result.tensors[0].shape, vec![1, 1, 1, 8]);
        let broadcast_tensor = result
            .tensors
            .iter()
            .find(|tensor| tensor.name == "add_input1_broadcast")
            .expect("broadcast tensor");
        assert_eq!(broadcast_tensor.shape, vec![1, 1, 1, 8]);
    }

    #[test]
    fn binary_emit_transposes_equal_rank_operand_to_current_layout() {
        let mut graph = Graph::new();
        let lhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("lhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 4, 3, 8, 8], IrDType::Int32),
            "lhs".to_string(),
        );
        let rhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("rhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[1, 3, 8, 8, 4], IrDType::Int32),
            "rhs".to_string(),
        );
        let mut add_meta = meta(&[1, 4, 3, 8, 8], IrDType::Int32);
        add_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 4, 1]);
        let add = graph.create_node(
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(lhs), Arg::Node(rhs)],
            Vec::new(),
            add_meta,
            "add".to_string(),
        );

        let result = emit_direct_binary(
            &graph,
            graph.node(add).expect("add node"),
            &TensorStore::default(),
            Op::Add,
            Attribute::Add(AddAttribute {}),
        );

        assert_eq!(result.operators[0].op, Op::Transpose);
        assert_eq!(result.operators[0].outputs, vec!["add_input1_layout"]);
        assert_eq!(result.tensors[0].shape, vec![1, 4, 3, 8, 8]);
        assert_eq!(result.operators[1].op, Op::Add);
        assert_eq!(
            result.operators[1].inputs,
            vec!["lhs".to_string(), "add_input1_layout".to_string()]
        );
        assert_eq!(
            result.operators[1].outputs,
            vec!["add_broadcast_output".to_string()]
        );
        assert_eq!(result.operators[2].op, Op::Transpose);
        assert_eq!(result.operators[2].inputs, vec!["add_broadcast_output"]);
        assert_eq!(result.operators[2].outputs, vec!["add"]);
    }

    #[test]
    fn binary_emit_infers_equal_rank_logical_layout_bridge() {
        let mut graph = Graph::new();
        let lhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("lhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[4, 4, 32, 32], IrDType::Float32),
            "lhs".to_string(),
        );
        let rhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("rhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[4, 32, 32, 4], IrDType::Float32),
            "rhs".to_string(),
        );
        let add = graph.create_node(
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(lhs), Arg::Node(rhs)],
            Vec::new(),
            meta(&[4, 4, 32, 32], IrDType::Float32),
            "add".to_string(),
        );

        let result = emit_direct_binary(
            &graph,
            graph.node(add).expect("add node"),
            &TensorStore::default(),
            Op::Add,
            Attribute::Add(AddAttribute {}),
        );

        assert_eq!(result.operators[0].op, Op::Transpose);
        assert_eq!(result.operators[0].outputs, vec!["add_input1_layout"]);
        assert_eq!(result.tensors[0].shape, vec![4, 4, 32, 32]);
        assert_eq!(
            result.operators[1].inputs,
            vec!["lhs".to_string(), "add_input1_layout".to_string()]
        );
        assert_eq!(result.operators[1].op, Op::Add);
        assert_eq!(result.operators[1].outputs, vec!["add".to_string()]);
    }

    #[test]
    fn binary_emit_transposes_equal_rank_fp_broadcast_operand_to_physical_layout() {
        let mut graph = Graph::new();
        let mut lhs_meta = meta(&[2, 16, 8, 128], IrDType::Float32);
        lhs_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let lhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("lhs".to_string()),
            smallvec![],
            Vec::new(),
            lhs_meta,
            "lhs".to_string(),
        );
        let rhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("rhs".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[2, 1, 8, 128], IrDType::Float32),
            "rhs".to_string(),
        );
        let mut mul_meta = meta(&[2, 16, 8, 128], IrDType::Float32);
        mul_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let mul = graph.create_node(
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![Arg::Node(lhs), Arg::Node(rhs)],
            Vec::new(),
            mul_meta,
            "mul".to_string(),
        );

        let result = emit_direct_binary(
            &graph,
            graph.node(mul).expect("mul node"),
            &TensorStore::default(),
            Op::Mul,
            Attribute::Mul(MulAttribute { shift: 0 }),
        );

        assert_eq!(result.operators[0].op, Op::Transpose);
        assert_eq!(result.operators[0].outputs, vec!["mul_input1_layout"]);
        assert_eq!(result.tensors[0].shape, vec![2, 8, 128, 1]);
        assert_eq!(result.operators[1].op, Op::Mul);
        assert_eq!(result.operators[1].outputs, vec!["mul".to_string()]);
        assert_eq!(result.tensors[1].shape, vec![2, 8, 128, 16]);
    }
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

pub(super) fn push_zero_point_const(
    result: &mut EmitResult,
    name: String,
    dtype: DType,
    value: i32,
) -> String {
    push_rank1_const(result, name, dtype, encode_zp_for_dtype(value, dtype), 1)
}

pub(super) fn temp_tensor(name: String, shape: Vec<i32>, dtype: DType) -> TosaTensor {
    make_empty_tensor(name, shape, dtype)
}
