use super::*;
use std::cell::RefCell;
use std::collections::HashMap as StdHashMap;

thread_local! {
    static SHAPE_CACHE: RefCell<StdHashMap<String, Vec<i32>>> = RefCell::new(StdHashMap::new());
}

/// Clear the shape cache (call before each graph serialization).
pub(crate) fn clear_shape_cache() {
    SHAPE_CACHE.with(|c| c.borrow_mut().clear());
}

pub(super) fn literal_output_shape(node: &Node) -> Option<Vec<i32>> {
    match node.args.get(1) {
        Some(Arg::IntList(shape)) => Some(shape.iter().map(|&d| d as i32).collect()),
        _ => None,
    }
}

pub(super) fn raw_i32_shape(node: &Node) -> Vec<i32> {
    let shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();

    if shape.is_empty() && matches!(node.target, OpTarget::ScalarTensor) {
        // TOSA does not model rank-0 tensors directly; represent scalar_tensor
        // as a rank-1 [1] tensor so downstream broadcast/rescale shapes stay
        // self-consistent.
        return vec![1];
    }

    shape
}

fn resolve_inferred_shape(shape: &[i32], input_shape: &[i32]) -> Option<Vec<i32>> {
    let negative_positions: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim < 0).then_some(idx))
        .collect();
    if negative_positions.is_empty() {
        return Some(shape.to_vec());
    }
    if negative_positions.len() > 1 {
        return None;
    }

    let input_numel = input_shape
        .iter()
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim as i64))?;
    let known_product = shape
        .iter()
        .filter(|&&dim| dim >= 0)
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim as i64))?;
    if known_product == 0 || input_numel % known_product != 0 {
        return None;
    }

    let mut resolved = shape.to_vec();
    resolved[negative_positions[0]] = (input_numel / known_product) as i32;
    Some(resolved)
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

pub(super) fn map_axis_to_tosa_dim_order(graph: &Graph, node: &Node, axis: i32) -> i32 {
    let rank = node.meta.val.shape_len as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return axis;
    }

    if matches!(node.target, OpTarget::TosaTranspose) {
        if let Some(input_node) = first_input_node(graph, node) {
            let input_axis = map_axis_to_tosa_dim_order(graph, input_node, normalized);
            let perms = transpose_perms(node);
            if let Some(pos) = perms.iter().position(|&dim| dim == input_axis) {
                return pos as i32;
            }
        }
    }

    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if let Some(pos) = order.iter().position(|&dim| dim as i32 == normalized) {
            return pos as i32;
        }
    }

    normalized
}

pub(super) fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

pub(super) fn transpose_perms(node: &Node) -> Vec<i32> {
    arg_or_kwarg_i32_list(node, 1, "perms")
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
    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if matches!(node.target, OpTarget::TosaRescale) && input_node.target == OpTarget::TosaTable {
        return Some(input_shape);
    }

    Some(input_shape)
}

fn is_shape_preserving_unary(node: &Node) -> bool {
    matches!(
        node.target,
        OpTarget::Relu
            | OpTarget::Hardtanh
            | OpTarget::Clamp
            | OpTarget::LeakyRelu
            | OpTarget::Sigmoid
            | OpTarget::Tanh
            | OpTarget::Exp
            | OpTarget::Log
            | OpTarget::Neg
            | OpTarget::Reciprocal
            | OpTarget::Rsqrt
            | OpTarget::Abs
            | OpTarget::Floor
            | OpTarget::Ceil
            | OpTarget::To
            | OpTarget::Sqrt
            | OpTarget::Log1p
            | OpTarget::Expm1
            | OpTarget::Sign
            | OpTarget::Round
            | OpTarget::Sin
            | OpTarget::Cos
            | OpTarget::TosaSigmoid
            | OpTarget::TosaTanh
            | OpTarget::TosaExp
            | OpTarget::TosaLog
            | OpTarget::TosaAbs
            | OpTarget::TosaReciprocal
            | OpTarget::TosaErf
            | OpTarget::TosaCos
            | OpTarget::TosaSin
            | OpTarget::TosaRsqrt
    )
}

fn shape_preserving_unary_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !is_shape_preserving_unary(node) {
        return None;
    }
    let input_node = first_input_node(graph, node)?;
    Some(actual_tensor_shape(graph, store, input_node))
}

pub(super) fn transposed_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
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

    strided_slice_shape(graph, node, input_node, &input_shape)
        .or_else(|| explicit_slice_size_shape(node, &input_shape))
}

fn reduced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !matches!(node.target, OpTarget::Sum | OpTarget::TosaReduceSum) {
        return None;
    }

    if matches!(node.target, OpTarget::Sum) {
        return Some(apply_tosa_dim_order(node, raw_i32_shape(node)));
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
        .map(|axis| map_axis_to_tosa_dim_order(graph, input_node, axis) as usize)
        .collect();

    if keepdim {
        for axis in mapped_axes {
            if let Some(slot) = output_shape.get_mut(axis) {
                *slot = 1;
            }
        }
        return Some(output_shape);
    }

    let input_raw_shape = raw_i32_shape(input_node);
    if input_raw_shape.len() == input_shape.len() {
        let rank = input_raw_shape.len() as i32;
        let mut raw_axes: Vec<usize> = arg_i32_list(node, 1)
            .into_iter()
            .map(|axis| if axis < 0 { axis + rank } else { axis })
            .filter(|&axis| axis >= 0 && axis < rank)
            .map(|axis| axis as usize)
            .collect();
        raw_axes.sort_unstable_by(|a, b| b.cmp(a));

        let mut reduced_raw_shape = input_raw_shape;
        for axis in raw_axes {
            if axis < reduced_raw_shape.len() {
                reduced_raw_shape.remove(axis);
            }
        }
        return Some(apply_tosa_dim_order(node, reduced_raw_shape));
    }

    mapped_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in mapped_axes {
        if axis < output_shape.len() {
            output_shape.remove(axis);
        }
    }
    Some(output_shape)
}

fn checked_numel(shape: &[i32]) -> Option<i64> {
    shape.iter().try_fold(1i64, |acc, &dim| {
        (dim >= 0)
            .then_some(dim as i64)
            .and_then(|dim| acc.checked_mul(dim))
    })
}

pub(super) fn broadcast_two_shapes(lhs: &[i32], rhs: &[i32]) -> Option<Vec<i32>> {
    let rank = lhs.len().max(rhs.len());
    let lhs_pad = rank.saturating_sub(lhs.len());
    let rhs_pad = rank.saturating_sub(rhs.len());
    let mut out = Vec::with_capacity(rank);

    for idx in 0..rank {
        let lhs_dim = if idx < lhs_pad { 1 } else { lhs[idx - lhs_pad] };
        let rhs_dim = if idx < rhs_pad { 1 } else { rhs[idx - rhs_pad] };
        let out_dim = match (lhs_dim, rhs_dim) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => return None,
        };
        out.push(out_dim);
    }

    Some(out)
}

pub(super) fn binary_input_broadcast_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    let mut input_shapes = node.args.iter().filter_map(|arg| {
        graph
            .node(arg.as_node()?)
            .map(|input_node| actual_tensor_shape(graph, store, input_node))
    });

    let mut candidate = input_shapes.next()?;
    for input_shape in input_shapes {
        candidate = broadcast_two_shapes(&candidate, &input_shape)?;
    }

    Some(candidate)
}

fn broadcasted_input_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    let candidate = binary_input_broadcast_shape(graph, store, node)?;

    let raw_shape = raw_i32_shape(node);
    if raw_shape.len() != candidate.len() {
        return None;
    }

    (checked_numel(&raw_shape)? == checked_numel(&candidate)?).then_some(candidate)
}

fn table_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if node.target != OpTarget::TosaTable {
        return None;
    }
    let input_node = first_input_node(graph, node)?;
    Some(actual_tensor_shape(graph, store, input_node))
}

pub(super) fn pool_post_pad(
    input_size: i32,
    kernel_size: i32,
    stride: i32,
    pad: i32,
    ceil_mode: bool,
) -> i32 {
    let numer = input_size - kernel_size + 2 * pad;
    let output_size = if ceil_mode {
        ((numer + stride - 1) / stride) + 1
    } else {
        (numer / stride) + 1
    };
    ((output_size - 1) * stride - input_size + kernel_size - pad).max(0)
}

pub(super) fn symmetric_pool_pad(padding: &[i32]) -> Vec<i32> {
    if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    }
}

pub(super) fn adjusted_max_pool_pad(
    input_shape: &[i32],
    kernel: &[i32],
    stride: &[i32],
    padding: &[i32],
    ceil_mode: bool,
) -> Vec<i32> {
    if padding.len() == 4 {
        return padding.to_vec();
    }

    let symmetric = symmetric_pool_pad(padding);
    if input_shape.len() != 4 {
        return symmetric;
    }

    vec![
        symmetric[0],
        pool_post_pad(
            input_shape[1],
            *kernel.first().unwrap_or(&2),
            *stride.first().unwrap_or(&2),
            symmetric[0],
            ceil_mode,
        ),
        symmetric[2],
        pool_post_pad(
            input_shape[2],
            *kernel.get(1).unwrap_or(&2),
            *stride.get(1).unwrap_or(&2),
            symmetric[2],
            ceil_mode,
        ),
    ]
}

fn pool_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    let is_max_pool = matches!(node.target, OpTarget::MaxPool2d | OpTarget::TosaMaxPool2d);
    let is_avg_pool = matches!(node.target, OpTarget::AvgPool2d | OpTarget::TosaAvgPool2d);
    if !is_max_pool && !is_avg_pool {
        return None;
    }

    let input_shape = actual_tensor_shape(graph, store, first_input_node(graph, node)?);
    if input_shape.len() != 4 {
        return None;
    }

    let kernel = i32_list_or(kwarg_or_arg_i32_list(node, "kernel", 1), vec![2, 2]);
    let stride = i32_list_or(kwarg_or_arg_i32_list(node, "stride", 2), kernel.clone());
    let raw_padding = kwarg_or_arg_i32_list(node, "pad", 3);
    let pad = if is_max_pool {
        let ceil_mode = kwarg_bool(node, "ceil_mode", arg_bool(node, 5, false));
        adjusted_max_pool_pad(&input_shape, &kernel, &stride, &raw_padding, ceil_mode)
    } else if raw_padding.len() == 4 {
        raw_padding
    } else {
        symmetric_pool_pad(&raw_padding)
    };

    let n = input_shape[0];
    let ih = input_shape[1];
    let iw = input_shape[2];
    let c = input_shape[3];
    let kh = *kernel.first().unwrap_or(&2);
    let kw = *kernel.get(1).unwrap_or(&2);
    let sh = *stride.first().unwrap_or(&2);
    let sw = *stride.get(1).unwrap_or(&2);
    let oh = (ih + pad[0] + pad[1] - kh) / sh + 1;
    let ow = (iw + pad[2] + pad[3] - kw) / sw + 1;

    (oh > 0 && ow > 0).then_some(vec![n, oh, ow, c])
}

fn strided_slice_shape(
    graph: &Graph,
    node: &Node,
    input_node: &Node,
    input_shape: &[i32],
) -> Option<Vec<i32>> {
    if node.args.len() < 4 {
        return None;
    }

    let rank = input_shape.len() as i64;
    let raw_dim = arg_i64(node, 1, 0);
    let normalized_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
    let clamped_dim = normalized_dim.clamp(0, rank.saturating_sub(1));
    let dim = node
        .meta
        .tosa_dim_order
        .as_ref()
        .map(|_| map_axis_to_tosa_dim_order(graph, node, clamped_dim as i32) as usize)
        .unwrap_or_else(|| {
            map_axis_to_tosa_dim_order(graph, input_node, clamped_dim as i32) as usize
        });
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
pub(crate) fn actual_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    // Memoize: check cache first
    if let Some(cached) = SHAPE_CACHE.with(|c| c.borrow().get(&node.name).cloned()) {
        return cached;
    }
    let raw_result = actual_tensor_shape_impl(graph, store, node);
    let result = if node.meta.preserve_scalar_shape && raw_result.is_empty() {
        raw_result
    } else {
        normalize_tensor_shape(raw_result)
    };
    SHAPE_CACHE.with(|c| c.borrow_mut().insert(node.name.clone(), result.clone()));
    result
}

fn actual_tensor_shape_impl(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    if matches!(
        node.target,
        OpTarget::TosaReshape | OpTarget::Reshape | OpTarget::View
    ) {
        if let Some(shape) = literal_output_shape(node) {
            let input_shape = first_input_node(graph, node)
                .map(|input| actual_tensor_shape(graph, store, input))
                .unwrap_or_default();
            if let Some(resolved_shape) = resolve_inferred_shape(&shape, &input_shape) {
                return apply_tosa_dim_order(node, resolved_shape);
            }
        }
    }

    if let Some(shape) = passthrough_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = shape_preserving_unary_tensor_shape(graph, store, node) {
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

    if let Some(shape) = table_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = broadcasted_input_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = pool_tensor_shape(graph, store, node) {
        return shape;
    }

    let mut raw_shape = raw_i32_shape(node);
    if raw_shape.iter().any(|&dim| dim < 0) {
        if let Some(input_node) = first_input_node(graph, node) {
            let input_shape = actual_tensor_shape(graph, store, input_node);
            if let Some(resolved_shape) = resolve_inferred_shape(&raw_shape, &input_shape) {
                raw_shape = resolved_shape;
            }
        }
    }
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
    let final_shape = logical_output_shape(node);
    let dtype = node_output_dtype(node);
    make_empty_tensor_for_node(node, node.name.clone(), final_shape, dtype)
}

pub(super) fn logical_output_shape(node: &Node) -> Vec<i32> {
    let raw_result = apply_tosa_dim_order(node, raw_i32_shape(node));
    if node.meta.preserve_scalar_shape && raw_result.is_empty() {
        raw_result
    } else {
        normalize_tensor_shape(raw_result)
    }
}

pub(super) fn make_actual_output_tensor(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> TosaTensor {
    make_empty_tensor_for_node(
        node,
        node.name.clone(),
        actual_tensor_shape(graph, store, node),
        node_output_dtype(node),
    )
}

pub(super) fn push_reshape_with_shape(
    result: &mut EmitResult,
    input_name: String,
    output_name: String,
    shape_name: String,
    out_shape: Vec<i32>,
    dtype: DType,
) {
    push_reshape_with_shape_for_node(
        result,
        None,
        input_name,
        output_name,
        shape_name,
        out_shape,
        dtype,
    );
}

pub(super) fn push_reshape_with_shape_for_node(
    result: &mut EmitResult,
    node: Option<&Node>,
    input_name: String,
    output_name: String,
    shape_name: String,
    out_shape: Vec<i32>,
    dtype: DType,
) {
    let out_shape = if node
        .map(|node| node.meta.preserve_scalar_shape)
        .unwrap_or(false)
        && out_shape.is_empty()
    {
        out_shape
    } else {
        normalize_tensor_shape(out_shape)
    };
    let output_tensor = node
        .map(|node| make_empty_tensor_for_node(node, output_name.clone(), out_shape.clone(), dtype))
        .unwrap_or_else(|| make_empty_tensor(output_name.clone(), out_shape.clone(), dtype));
    push_const_shape(result, shape_name.clone(), out_shape);
    result.push_output_op(
        output_tensor,
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute {}),
        vec![input_name, shape_name],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{Arg, DType, NodeMeta, OpTarget, OpType, TensorInfo};
    use crate::graph::Graph;
    use crate::store::TensorStore;
    use smallvec::smallvec;

    fn meta(shape: &[i64], dtype: DType) -> NodeMeta {
        NodeMeta::new(TensorInfo::new(shape, dtype))
    }

    #[test]
    fn reciprocal_uses_input_shape_when_meta_is_stale_scalar() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[10, 10, 10], DType::Float32),
            "x".to_string(),
        );
        let reciprocal = graph.create_node(
            OpType::CallFunction,
            OpTarget::Reciprocal,
            smallvec![Arg::Node(input)],
            Vec::new(),
            meta(&[], DType::Float32),
            "x_recip".to_string(),
        );

        let store = TensorStore::default();
        let reciprocal_node = graph.node(reciprocal).expect("reciprocal node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, reciprocal_node),
            vec![10, 10, 10]
        );
    }
}
