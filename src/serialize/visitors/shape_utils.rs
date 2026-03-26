use super::*;

pub(super) fn literal_output_shape(node: &Node) -> Option<Vec<i32>> {
    match node.args.get(1) {
        Some(Arg::IntList(shape)) => Some(shape.iter().map(|&d| d as i32).collect()),
        _ => None,
    }
}

pub(super) fn raw_i32_shape(node: &Node) -> Vec<i32> {
    node.meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect()
}

pub(super) fn apply_tosa_dim_order(node: &Node, shape: Vec<i32>) -> Vec<i32> {
    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if order.len() == shape.len() {
            let reordered: Vec<i32> = order
                .iter()
                .filter_map(|&idx| shape.get(idx as usize).copied())
                .collect();
            if reordered.len() == shape.len() {
                return reordered;
            }
        }
    }
    shape
}

pub(super) fn map_axis_to_tosa_dim_order(node: &Node, axis: i32) -> i32 {
    let rank = node.meta.val.shape_len as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return axis;
    }

    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if let Some(pos) = order.iter().position(|&dim| dim as i32 == normalized) {
            return pos as i32;
        }
    }

    normalized
}

fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

fn transpose_perms(node: &Node) -> Vec<i32> {
    let perms = arg_i32_list(node, 1);
    if perms.is_empty() {
        kwarg_i32_list(node, "perms")
    } else {
        perms
    }
}

fn is_layout_passthrough_qdq(node: &Node) -> bool {
    match &node.target {
        OpTarget::Other(name) => {
            name.contains("quantize_per_tensor")
                || name.contains("dequantize_per_tensor")
                || name.contains("dequantize_per_channel")
        }
        _ => matches!(
            node.target,
            OpTarget::TosaRescale
                | OpTarget::QuantizePerTensor
                | OpTarget::DequantizePerTensor
                | OpTarget::DequantizePerChannel
        ),
    }
}

fn passthrough_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !is_layout_passthrough_qdq(node) {
        return None;
    }
    Some(actual_tensor_shape(graph, store, first_input_node(graph, node)?))
}

fn transposed_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !matches!(node.target, OpTarget::TosaTranspose | OpTarget::Permute) {
        return None;
    }

    let input_shape = actual_tensor_shape(graph, store, first_input_node(graph, node)?);
    let perms = transpose_perms(node);
    if input_shape.len() != perms.len() {
        return None;
    }

    let output_shape: Vec<i32> = perms
        .iter()
        .filter_map(|&idx| input_shape.get(idx as usize).copied())
        .collect();
    (output_shape.len() == input_shape.len()).then_some(output_shape)
}

fn sliced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !matches!(node.target, OpTarget::Slice | OpTarget::TosaSlice) {
        return None;
    }

    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.is_empty() {
        return None;
    }

    strided_slice_shape(node, input_node, &input_shape)
        .or_else(|| explicit_slice_size_shape(node, &input_shape))
}

fn reduced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !matches!(node.target, OpTarget::Sum | OpTarget::TosaReduceSum) {
        return None;
    }

    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.is_empty() {
        return None;
    }

    let axes = arg_i32_list(node, 1);
    if axes.is_empty() {
        return None;
    }
    let keepdim = arg_bool(node, 2, false);
    let mut output_shape = input_shape.clone();
    let mut mapped_axes: Vec<usize> = axes
        .into_iter()
        .map(|axis| map_axis_to_tosa_dim_order(input_node, axis) as usize)
        .collect();

    if keepdim {
        for axis in mapped_axes {
            if let Some(slot) = output_shape.get_mut(axis) {
                *slot = 1;
            }
        }
        return Some(output_shape);
    }

    mapped_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in mapped_axes {
        if axis < output_shape.len() {
            output_shape.remove(axis);
        }
    }
    Some(output_shape)
}

fn strided_slice_shape(node: &Node, input_node: &Node, input_shape: &[i32]) -> Option<Vec<i32>> {
    if node.args.len() < 4 {
        return None;
    }

    let rank = input_shape.len() as i64;
    let raw_dim = arg_i64(node, 1, 0);
    let normalized_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
    let clamped_dim = normalized_dim.clamp(0, rank.saturating_sub(1));
    let dim = map_axis_to_tosa_dim_order(input_node, clamped_dim as i32) as usize;
    let input_dim = input_shape.get(dim).copied()? as i64;
    let start = arg_i64(node, 2, 0);
    let end = arg_i64(node, 3, input_dim);
    let step = arg_i64(node, 4, 1);
    let start_idx = if start < 0 {
        (input_dim + start).max(0)
    } else {
        start.min(input_dim)
    };
    let mut end_idx = if end < 0 { input_dim + end } else { end };
    end_idx = end_idx.clamp(start_idx, input_dim);

    let slice_len = if step == 1 {
        (end_idx - start_idx).max(0)
    } else {
        ((end_idx - start_idx).max(0) + step - 1) / step
    };

    let mut output_shape = input_shape.to_vec();
    *output_shape.get_mut(dim)? = slice_len as i32;
    Some(output_shape)
}

fn explicit_slice_size_shape(node: &Node, input_shape: &[i32]) -> Option<Vec<i32>> {
    let size = kwarg_i32_list(node, "size");
    if size.len() != input_shape.len() {
        return None;
    }

    let reordered = apply_tosa_dim_order(node, size.clone());
    for candidate in [size.clone(), reordered] {
        if candidate
            .iter()
            .zip(input_shape.iter())
            .all(|(out_dim, in_dim)| *out_dim >= 0 && *out_dim <= *in_dim)
        {
            return Some(candidate);
        }
    }

    Some(size)
}

fn nhwc_layout_shape(node: &Node, raw_shape: &[i32], reordered_shape: &[i32]) -> Option<Vec<i32>> {
    let logical_nhwc = raw_shape.len() == 4
        && (is_getitem_node(node)
            || matches!(
                node.target,
                OpTarget::Conv2d
                    | OpTarget::DepthwiseConv2d
                    | OpTarget::TosaConv2d
                    | OpTarget::TosaDepthwiseConv2d
                    | OpTarget::MaxPool2d
                    | OpTarget::AvgPool2d
                    | OpTarget::TosaMaxPool2d
                    | OpTarget::TosaAvgPool2d
            ));
    if !logical_nhwc || reordered_shape != raw_shape {
        return None;
    }
    Some(vec![raw_shape[0], raw_shape[2], raw_shape[3], raw_shape[1]])
}

/// Reconstruct the concrete TOSA tensor shape once at the serializer boundary
/// instead of re-validating dim-order and layout facts ad hoc at each emitter.
pub(super) fn actual_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    if matches!(node.target, OpTarget::TosaReshape | OpTarget::Reshape | OpTarget::View) {
        if let Some(shape) = literal_output_shape(node) {
            return apply_tosa_dim_order(node, shape);
        }
    }

    if let Some(shape) = passthrough_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = transposed_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = sliced_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = reduced_tensor_shape(graph, store, node) {
        return shape;
    }

    let raw_shape = raw_i32_shape(node);
    let reordered_shape = apply_tosa_dim_order(node, raw_shape.clone());

    if node.op == OpType::Placeholder {
        return reordered_shape;
    }

    nhwc_layout_shape(node, &raw_shape, &reordered_shape).unwrap_or(reordered_shape)
}

pub(super) fn promoted_2d_matmul_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !matches!(
        node.target,
        OpTarget::Matmul | OpTarget::Bmm | OpTarget::TosaMatmul
    ) {
        return None;
    }

    match (node.args.first(), node.args.get(1)) {
        (Some(Arg::Node(a_id)), Some(Arg::Node(b_id))) => {
            let a_shape = graph
                .node(*a_id)
                .map(|n| actual_tensor_shape(graph, store, n))
                .unwrap_or_default();
            let b_shape = graph
                .node(*b_id)
                .map(|n| actual_tensor_shape(graph, store, n))
                .unwrap_or_default();
            if a_shape.len() == 2 && b_shape.len() == 2 {
                Some(vec![1, a_shape[0], b_shape[1]])
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn make_output_tensor(node: &Node) -> TosaTensor {
    let final_shape = apply_tosa_dim_order(node, raw_i32_shape(node));
    let dtype = node_output_dtype(node);
    make_empty_tensor(node.name.clone(), final_shape, dtype)
}

pub(super) fn make_bool_output_tensor(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> TosaTensor {
    make_empty_tensor(node.name.clone(), actual_tensor_shape(graph, store, node), DType::Bool)
}

pub(super) fn push_reshape_with_shape(
    result: &mut EmitResult,
    input_name: String,
    output_name: String,
    shape_name: String,
    out_shape: Vec<i32>,
    dtype: DType,
) {
    result
        .tensors
        .push(make_empty_tensor(output_name.clone(), out_shape.clone(), dtype));

    let rank = out_shape.len() as u32;
    let shape_data: Vec<u8> = out_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank,
        data: shape_data,
    });
    result.operators.push(TosaOperator {
        op: Op::ConstShape,
        attribute: Attribute::ConstShape(ConstShapeAttribute),
        inputs: vec![],
        outputs: vec![shape_name.clone()],
        location: None,
    });
    result.operators.push(TosaOperator {
        op: Op::Reshape,
        attribute: Attribute::Reshape(ReshapeAttribute),
        inputs: vec![input_name, shape_name],
        outputs: vec![output_name],
        location: None,
    });
}
