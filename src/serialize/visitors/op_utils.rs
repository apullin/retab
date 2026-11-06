use super::*;

fn emitted_binary_input_shapes(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<Vec<i32>> {
    let input_nodes: Vec<&Node> = node
        .args
        .iter()
        .filter_map(|arg| graph.node(arg.as_node()?))
        .take(2)
        .collect();
    if input_nodes.len() < 2 {
        return input_nodes
            .into_iter()
            .map(|input| actual_tensor_shape(graph, store, input))
            .collect();
    }

    let mut shapes = vec![
        actual_tensor_shape(graph, store, input_nodes[0]),
        actual_tensor_shape(graph, store, input_nodes[1]),
    ];

    if shapes[0].len() != shapes[1].len() {
        let (smaller_idx, larger_idx) = if shapes[0].len() < shapes[1].len() {
            (0usize, 1usize)
        } else {
            (1usize, 0usize)
        };

        let target = broadcast_shape_for_binary_inputs(
            &raw_i32_shape(input_nodes[smaller_idx]),
            &raw_i32_shape(input_nodes[larger_idx]),
        );
        if target.len() == shapes[larger_idx].len() {
            shapes[smaller_idx] = target;
        }
    }

    shapes
}

fn binary_output_shapes(graph: &Graph, store: &TensorStore, node: &Node) -> (Vec<i32>, Vec<i32>) {
    let logical_shape = raw_i32_shape(node);
    let emitted_shape = emitted_binary_input_shapes(graph, store, node)
        .into_iter()
        .reduce(|lhs, rhs| broadcast_two_shapes(&lhs, &rhs).unwrap_or(rhs))
        .unwrap_or_else(|| {
            binary_input_broadcast_shape(graph, store, node)
                .unwrap_or_else(|| logical_shape.clone())
        });
    (logical_shape, emitted_shape)
}

#[allow(clippy::too_many_arguments)]
fn push_binary_output_with_contract(
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
            make_empty_tensor(temp_name.clone(), emitted_shape, dtype),
            op,
            attribute,
            inputs,
        );
        push_reshape_with_shape(
            result,
            temp_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            logical_shape,
            dtype,
        );
    } else {
        result.push_output_op(
            make_empty_tensor(node.name.clone(), logical_shape, dtype),
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
    let mut inputs = collect_binary_input_names(graph, store, node, &mut result);
    if op == Op::Mul {
        inputs.push(push_shift_const(&mut result, &node.name));
    }
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
    fn binary_emit_reshapes_broadcast_rank_back_to_logical_output() {
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

        assert_eq!(result.operators.len(), 3);
        assert_eq!(result.operators[0].op, Op::Add);
        assert_eq!(
            result.operators[0].outputs,
            vec!["add_broadcast_output".to_string()]
        );
        assert_eq!(result.operators[2].op, Op::Reshape);
        assert_eq!(result.operators[2].outputs, vec!["add".to_string()]);
        assert_eq!(result.tensors[0].shape, vec![1, 4, 5, 1]);
        assert_eq!(result.tensors[1].shape, vec![1, 4, 5]);
    }
}
