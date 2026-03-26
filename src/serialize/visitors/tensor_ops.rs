use super::*;

struct ParsedShapeTransformSpec {
    input_names: Vec<String>,
    output_shape: Vec<i32>,
    output_name: String,
    output_dtype: DType,
}

impl ParsedShapeTransformSpec {
    // Parse once at the shape-op boundary so emitters can work from a typed
    // representation of the target tensor instead of re-deriving it ad hoc.
    fn parse(graph: &Graph, store: &TensorStore, node: &Node) -> Self {
        Self {
            input_names: collect_input_names(graph, node),
            output_shape: actual_tensor_shape(graph, store, node),
            output_name: primary_output_name(graph, node),
            output_dtype: passthrough_input_dtype(graph, node),
        }
    }
}

pub(super) fn emit_full(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let output_shape = actual_tensor_shape(graph, store, node);
    let fill_value = match node.args.get(1) {
        Some(Arg::Float(value)) => *value,
        Some(Arg::Int(value)) => *value as f64,
        Some(Arg::Bool(value)) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    let dtype = node.meta.val.dtype;
    let tensor = make_filled_const_tensor(node.name.clone(), output_shape, dtype, fill_value)
        .unwrap_or_else(|| {
            panic!(
                "emit_full: unsupported dtype {:?} for node '{}'",
                dtype, node.name
            )
        });
    let mut result = EmitResult::default();
    push_inline_const(&mut result, tensor);
    result
}

pub(super) fn emit_aten_reshape(
    graph: &Graph,
    node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_reshape(graph, node_id, node, store)
}

pub(super) fn emit_aten_permute(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let spec = ParsedShapeTransformSpec::parse(graph, store, node);
    let perms = arg_i32_list(node, 1);
    result.tensors.push(make_empty_tensor(
        spec.output_name.clone(),
        spec.output_shape,
        spec.output_dtype,
    ));
    result.operators.push(TosaOperator {
        op: Op::Transpose,
        attribute: Attribute::Transpose(TransposeAttribute { perms }),
        inputs: spec.input_names,
        outputs: vec![spec.output_name],
        location: None,
    });
    result
}

pub(super) fn emit_expand(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let Some(input_name) = input_names.first().cloned() else {
        return result;
    };
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return result;
    };
    let Some(input_node) = graph.node(*input_id) else {
        return result;
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() {
        return result;
    }

    let padding = output_shape.len().saturating_sub(input_shape.len());
    let mut padded_input_shape = vec![1; padding];
    padded_input_shape.extend(input_shape.iter().copied());
    let multiples: Vec<i32> = padded_input_shape
        .iter()
        .zip(output_shape.iter())
        .map(|(&input_dim, &output_dim)| {
            if input_dim == output_dim {
                1
            } else if input_dim == 1 && output_dim >= 1 {
                output_dim
            } else {
                1
            }
        })
        .collect();

    let multiples_name = format!("{}_multiples", node.name);
    push_inline_const(
        &mut result,
        make_shape_const_tensor(multiples_name.clone(), multiples),
    );
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Tile,
        attribute: Attribute::None,
        inputs: vec![input_name, multiples_name],
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

pub(super) fn emit_const(
    _graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let name = &node.name;
    let (shape, dtype, data) = if let Some(tensor_data) = store.get(name) {
        (
            tensor_data.shape.iter().map(|&d| d as i32).collect(),
            ir_dtype_to_tosa(tensor_data.dtype),
            tensor_data.data.clone(),
        )
    } else {
        (raw_i32_shape(node), ir_dtype_to_tosa(node.meta.val.dtype), Vec::new())
    };

    result.tensors.push(make_tensor(name.clone(), shape, dtype, data));
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![name.clone()],
        location: None,
    });
    result
}

pub(super) fn emit_transpose(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let perms = {
        let perms = arg_i32_list(node, 1);
        if perms.is_empty() {
            kwarg_i32_list(node, "perms")
        } else {
            perms
        }
    };
    let mut output_shape = raw_i32_shape(node);
    if let Some(Arg::Node(input_id)) = node.args.first() {
        if let Some(input_node) = graph.node(*input_id) {
            let input_shape = actual_tensor_shape(graph, store, input_node);
            if input_shape.len() == perms.len() {
                let candidate: Vec<i32> = perms
                    .iter()
                    .filter_map(|&idx| input_shape.get(idx as usize).copied())
                    .collect();
                if candidate.len() == input_shape.len() {
                    output_shape = candidate;
                }
            }
        }
    }
    result.tensors.push(make_empty_tensor(
        primary_output_name(graph, node),
        output_shape,
        passthrough_input_dtype(graph, node),
    ));
    result.operators.push(TosaOperator {
        op: Op::Transpose,
        attribute: Attribute::Transpose(TransposeAttribute { perms }),
        inputs: input_names,
        outputs: vec![primary_output_name(graph, node)],
        location: None,
    });
    result
}

pub(super) fn emit_reshape(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let out_shape = actual_tensor_shape(graph, store, node);
    result.tensors.push(make_empty_tensor(
        primary_output_name(graph, node),
        out_shape.clone(),
        passthrough_input_dtype(graph, node),
    ));

    let shape_name = format!("{}_shape", node.name);
    let shape_data: Vec<u8> = out_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank: out_shape.len() as u32,
        data: shape_data,
    });
    result.operators.push(TosaOperator {
        op: Op::ConstShape,
        attribute: Attribute::ConstShape(ConstShapeAttribute),
        inputs: vec![],
        outputs: vec![shape_name.clone()],
        location: None,
    });

    let mut reshape_inputs = input_names;
    reshape_inputs.push(shape_name);
    result.operators.push(TosaOperator {
        op: Op::Reshape,
        attribute: Attribute::Reshape(ReshapeAttribute),
        inputs: reshape_inputs,
        outputs: vec![primary_output_name(graph, node)],
        location: None,
    });
    result
}

pub(super) fn emit_concat(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let raw_axis = match node.args.get(1) {
        Some(Arg::Int(axis)) => *axis as i32,
        _ => kwarg_i32(node, "axis", 0),
    };
    let axis = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|input_node| map_axis_to_tosa_dim_order(input_node, raw_axis))
            .unwrap_or(raw_axis),
        Some(Arg::NodeList(ids)) => ids
            .first()
            .and_then(|input_id| graph.node(*input_id))
            .map(|input_node| map_axis_to_tosa_dim_order(input_node, raw_axis))
            .unwrap_or(raw_axis),
        _ => raw_axis,
    };
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Concat,
        attribute: Attribute::Concat(ConcatAttribute { axis }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}
