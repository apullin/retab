use super::*;
use crate::serialize::op_semantics;
use crate::serialize::tensor_layout::apply_tosa_dim_order;
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

pub(super) fn normalize_axis(axis: i32, rank: usize) -> Option<usize> {
    let rank = rank as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    (0..rank)
        .contains(&normalized)
        .then_some(normalized as usize)
}

pub(super) fn axis_order_for_node(graph: &Graph, node: &Node, rank: usize) -> Vec<usize> {
    let mut order = vec![0usize; rank];
    let mut seen = vec![false; rank];
    for raw_axis in 0..rank {
        let mapped = map_axis_to_tosa_dim_order(graph, node, raw_axis as i32);
        let Some(pos) = normalize_axis(mapped, rank) else {
            return node_dim_order_or_identity(node, rank);
        };
        if seen[pos] {
            return node_dim_order_or_identity(node, rank);
        }
        order[pos] = raw_axis;
        seen[pos] = true;
    }
    order
}

pub(super) fn node_dim_order_or_identity(node: &Node, rank: usize) -> Vec<usize> {
    node.meta
        .tosa_dim_order
        .as_ref()
        .and_then(|order| {
            let mut seen = vec![false; rank];
            let mut out = Vec::with_capacity(rank);
            if order.len() != rank {
                return None;
            }
            for &axis in order {
                let axis = axis as usize;
                if axis >= rank || seen[axis] {
                    return None;
                }
                seen[axis] = true;
                out.push(axis);
            }
            Some(out)
        })
        .unwrap_or_else(|| (0..rank).collect())
}

pub(super) fn mapped_axis_for_current_shape(
    raw_axis: i32,
    mapped_axis: i32,
    current_shape: &[i32],
) -> i32 {
    let Some(mapped_axis_idx) = normalize_axis(mapped_axis, current_shape.len()) else {
        return mapped_axis;
    };
    let Some(raw_axis_idx) = normalize_axis(raw_axis, current_shape.len()) else {
        return mapped_axis;
    };
    if mapped_axis_idx != raw_axis_idx
        && current_shape.get(mapped_axis_idx).copied() == Some(1)
        && current_shape.get(raw_axis_idx).is_some_and(|&dim| dim != 1)
    {
        return raw_axis_idx as i32;
    }
    mapped_axis
}

pub(super) fn map_axis_to_actual_tensor_axis(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    axis: i32,
) -> i32 {
    let rank = node.meta.val.shape_len as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return axis;
    }

    if op_semantics::is_qdq_shape_passthrough(&node.target) {
        if let Some(input_node) = first_input_node(graph, node) {
            let node_shape = actual_tensor_shape(graph, store, node);
            let input_shape = actual_tensor_shape(graph, store, input_node);
            if node_shape == input_shape {
                return map_axis_to_actual_tensor_axis(graph, store, input_node, axis);
            }
        }
    }

    if !matches!(node.target, OpTarget::TosaTranspose) {
        if let Some(physical_shape) = physical_ordered_raw_shape(node) {
            if actual_tensor_shape(graph, store, node) != physical_shape {
                return normalized;
            }
        }
    }

    let has_non_identity_dim_order = node
        .meta
        .tosa_dim_order
        .as_ref()
        .map(|order| {
            order
                .iter()
                .enumerate()
                .any(|(idx, &axis)| axis as usize != idx)
        })
        .unwrap_or(false);
    if raw_i32_shape(node) == actual_tensor_shape(graph, store, node)
        && !matches!(node.target, OpTarget::TosaTranspose)
        && !has_non_identity_dim_order
    {
        return normalized;
    }

    map_axis_to_tosa_dim_order(graph, node, axis)
}

pub(super) fn map_slice_axis_to_input_layout(
    graph: &Graph,
    node: &Node,
    input_node: &Node,
    input_shape: &[i32],
    axis: i32,
) -> usize {
    let rank = input_shape.len() as i32;
    if rank == 0 {
        return 0;
    }
    let normalized = if axis < 0 { axis + rank } else { axis };
    let clamped = normalized.clamp(0, rank.saturating_sub(1));

    let node_order_matches_input = node.meta.tosa_dim_order.is_some()
        && apply_tosa_dim_order(node, raw_i32_shape(input_node)) == input_shape;
    if node_order_matches_input {
        map_axis_to_tosa_dim_order(graph, node, clamped) as usize
    } else {
        map_axis_to_tosa_dim_order(graph, input_node, clamped) as usize
    }
}

pub(super) fn adjust_slice_axis_for_bounds(
    input_shape: &[i32],
    mapped_dim: usize,
    raw_dim: usize,
    start: i64,
    end: i64,
) -> usize {
    let Some(&mapped_size) = input_shape.get(mapped_dim) else {
        return mapped_dim;
    };
    let Some(&raw_size) = input_shape.get(raw_dim) else {
        return mapped_dim;
    };

    let mapped_size = i64::from(mapped_size);
    let raw_size = i64::from(raw_size);
    let start_exceeds_mapped = start >= 0 && start > mapped_size;
    let end_exceeds_mapped = end >= 0 && end != i64::MAX && end > mapped_size;
    let start_fits_raw = start < 0 || start <= raw_size;
    let end_fits_raw = end < 0 || end == i64::MAX || end <= raw_size;
    if mapped_dim != raw_dim
        && (start_exceeds_mapped || end_exceeds_mapped)
        && start_fits_raw
        && end_fits_raw
    {
        raw_dim
    } else {
        mapped_dim
    }
}

pub(super) fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

fn first_input_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    Some(actual_tensor_shape(
        graph,
        store,
        first_input_node(graph, node)?,
    ))
}

fn normalize_shape_for_node(node: Option<&Node>, shape: Vec<i32>) -> Vec<i32> {
    if node
        .map(|node| node.meta.preserve_scalar_shape)
        .unwrap_or(false)
        && shape.is_empty()
    {
        shape
    } else {
        normalize_tensor_shape(shape)
    }
}

pub(super) fn transpose_perms(node: &Node) -> Vec<i32> {
    let mut perms = arg_or_kwarg_i32_list(node, 1, "perms");
    let rank = perms.len() as i32;
    for dim in &mut perms {
        if *dim < 0 {
            *dim += rank;
        }
    }
    perms
}

fn passthrough_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !op_semantics::is_qdq_shape_passthrough(&node.target) {
        return None;
    }
    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if matches!(node.target, OpTarget::TosaRescale) && input_node.target == OpTarget::TosaTable {
        return Some(input_shape);
    }

    Some(input_shape)
}

fn physical_ordered_raw_shape(node: &Node) -> Option<Vec<i32>> {
    let order = node.meta.tosa_dim_order.as_ref()?;
    if order
        .iter()
        .enumerate()
        .all(|(idx, &axis)| axis as usize == idx)
    {
        return None;
    }
    let raw_shape = raw_i32_shape(node);
    (raw_shape.len() == order.len()).then(|| apply_tosa_dim_order(node, raw_shape))
}

fn shape_preserving_unary_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !op_semantics::is_shape_preserving_unary(&node.target) {
        return None;
    }
    physical_ordered_raw_shape(node).or_else(|| first_input_shape(graph, store, node))
}

pub(super) fn transposed_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !op_semantics::is_transpose_shape_op(&node.target) {
        return None;
    }

    let input_shape = first_input_shape(graph, store, node)?;
    let perms = transpose_perms(node);
    if input_shape.len() != perms.len() {
        return None;
    }

    permute_shape(&input_shape, &perms)
}

fn sliced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !op_semantics::is_slice_shape_op(&node.target) {
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

fn reduction_axes(node: &Node) -> Vec<i32> {
    let mut axes = arg_i32_list(node, 1);
    if axes.is_empty() {
        if let Some(Arg::Int(dim)) = node.args.get(1) {
            axes.push(*dim as i32);
        }
    }
    axes
}

fn reduced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !op_semantics::is_axis_reduction(&node.target) {
        return None;
    }

    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.is_empty() {
        return None;
    }

    let axes = reduction_axes(node);
    if axes.is_empty() {
        return None;
    }
    let keepdim = arg_bool(node, 2, false);
    let mut output_shape = input_shape.clone();
    let mut mapped_axes: Vec<usize> = axes
        .iter()
        .copied()
        .map(|axis| {
            (if node.meta.retab_reduce_axes_are_physical {
                let rank = input_shape.len() as i32;
                if axis < 0 {
                    axis + rank
                } else {
                    axis
                }
            } else {
                map_axis_to_actual_tensor_axis(graph, store, input_node, axis)
            }) as usize
        })
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
        let mut raw_axes: Vec<usize> = axes
            .iter()
            .copied()
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

pub(super) fn emitted_binary_input_shapes(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Vec<Vec<i32>> {
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
            &shapes[larger_idx],
            node,
        );
        if target.len() == shapes[larger_idx].len() {
            shapes[smaller_idx] = target;
        }
    }

    shapes
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
    if !op_semantics::is_binary_broadcast(&node.target) {
        return None;
    }

    if let Some(shape) = physical_ordered_raw_shape(node) {
        return Some(shape);
    }

    let candidate = binary_input_broadcast_shape(graph, store, node)?;
    Some(candidate)
}

fn table_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if node.target != OpTarget::TosaTable {
        return None;
    }
    first_input_shape(graph, store, node)
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

    let (kernel_h, kernel_w) = i32_pair_or_default(kernel, 2);
    let (stride_h, stride_w) = i32_pair_or_default(stride, 2);
    vec![
        symmetric[0],
        pool_post_pad(input_shape[1], kernel_h, stride_h, symmetric[0], ceil_mode),
        symmetric[2],
        pool_post_pad(input_shape[2], kernel_w, stride_w, symmetric[2], ceil_mode),
    ]
}

pub(super) fn pool_nhwc_shape(
    input_shape: &[i32],
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
) -> Option<Vec<i32>> {
    if input_shape.len() != 4 {
        return None;
    }

    let (kh, kw) = i32_pair_or_default(kernel, 2);
    let (sh, sw) = i32_pair_or_default(stride, 2);
    let (pad_top, pad_bottom, pad_left, pad_right) = i32_quad_or_default(pad, 0);
    let oh = (input_shape[1] + pad_top + pad_bottom - kh) / sh + 1;
    let ow = (input_shape[2] + pad_left + pad_right - kw) / sw + 1;

    (oh > 0 && ow > 0).then_some(vec![input_shape[0], oh, ow, input_shape[3]])
}

pub(super) fn pool_input_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    let input_node = first_input_node(graph, node)?;
    if matches!(input_node.target, OpTarget::Pad | OpTarget::TosaPad) {
        return pad_tensor_shape(graph, store, input_node);
    }
    Some(actual_tensor_shape(graph, store, input_node))
}

pub(super) fn pad_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !matches!(node.target, OpTarget::Pad | OpTarget::TosaPad) {
        return None;
    }
    let input_shape = first_input_shape(graph, store, node)?;
    let rank = input_shape.len();
    let pad = arg_i32_list(node, 1);
    let pad_pairs: Vec<(i32, i32)> = pad
        .chunks(2)
        .filter_map(|pair| match pair {
            [before, after] => Some((*before, *after)),
            _ => None,
        })
        .collect();
    let mut per_dim_pad = vec![(0i32, 0i32); rank];
    for (i, &pair) in pad_pairs.iter().enumerate() {
        let dim = rank.saturating_sub(1 + i);
        if dim < rank {
            per_dim_pad[dim] = pair;
        }
    }

    let reordered_pad = if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if order.len() == rank {
            order
                .iter()
                .map(|&idx| per_dim_pad.get(idx as usize).copied().unwrap_or((0, 0)))
                .collect()
        } else {
            per_dim_pad
        }
    } else {
        per_dim_pad
    };

    let mut output_shape = input_shape;
    for (i, (before, after)) in reordered_pad.iter().enumerate() {
        if let Some(slot) = output_shape.get_mut(i) {
            *slot += before + after;
        }
    }
    Some(output_shape)
}

fn pool_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    let is_max_pool = op_semantics::is_max_pool(&node.target);
    let is_avg_pool = op_semantics::is_avg_pool(&node.target);
    if !is_max_pool && !is_avg_pool {
        return None;
    }

    let input_shape = pool_input_shape(graph, store, node)?;
    if input_shape.len() != 4 {
        return None;
    }

    let kernel = kwarg_or_arg_i32_list_or(node, "kernel", 1, vec![2, 2]);
    let stride = kwarg_or_arg_i32_list_or(node, "stride", 2, kernel.clone());
    let raw_padding = kwarg_or_arg_i32_list(node, "pad", 3);
    let pad = if is_max_pool {
        let ceil_mode = kwarg_bool(node, "ceil_mode", arg_bool(node, 5, false));
        adjusted_max_pool_pad(&input_shape, &kernel, &stride, &raw_padding, ceil_mode)
    } else if raw_padding.len() == 4 {
        raw_padding
    } else {
        let ceil_mode = kwarg_bool(node, "ceil_mode", arg_bool(node, 4, false));
        adjusted_max_pool_pad(&input_shape, &kernel, &stride, &raw_padding, ceil_mode)
    };

    pool_nhwc_shape(&input_shape, &kernel, &stride, &pad)
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
    let mut dim =
        map_slice_axis_to_input_layout(graph, node, input_node, input_shape, clamped_dim as i32);
    let start = arg_i64(node, 2, 0);
    let end = arg_i64(node, 3, input_shape.get(dim).copied().unwrap_or(0) as i64);
    dim = adjust_slice_axis_for_bounds(input_shape, dim, clamped_dim as usize, start, end);
    let input_dim = input_shape.get(dim).copied()? as i64;
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
    let layout_getitem = is_getitem_node(node) && !node.meta.retab_logical_layout_boundary;
    let logical_nhwc = raw_shape.len() == 4
        && (layout_getitem || op_semantics::uses_nhwc_layout_shape(&node.target));
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
    let result = normalize_shape_for_node(Some(node), raw_result);
    SHAPE_CACHE.with(|c| c.borrow_mut().insert(node.name.clone(), result.clone()));
    result
}

fn actual_tensor_shape_impl(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    if op_semantics::is_shape_literal_reshape(&node.target) {
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

    if let Some(shape) = pad_tensor_shape(graph, store, node) {
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
    if !op_semantics::is_matmul_shape_op(&node.target) {
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
    normalize_shape_for_node(Some(node), raw_result)
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
    let out_shape = normalize_shape_for_node(node, out_shape);
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

    #[test]
    fn transpose_perms_normalizes_negative_axes() {
        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[1, 1, 49, 64, 256], DType::Int8),
            "x".to_string(),
        );
        let permute = graph.create_node(
            OpType::CallFunction,
            OpTarget::Permute,
            smallvec![Arg::Node(input), Arg::IntList(smallvec![0, 1, -1, -2, -3])],
            Vec::new(),
            meta(&[1, 1, 256, 64, 49], DType::Int8),
            "permute".to_string(),
        );

        let node = graph.node(permute).expect("permute node");
        assert_eq!(transpose_perms(node), vec![0, 1, 4, 3, 2]);
    }

    #[test]
    fn physical_cast_shape_uses_its_own_dim_order() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[1, 3, 224, 224], DType::Float32),
            "x".to_string(),
        );
        let mut cast_meta = meta(&[1, 3, 224, 224], DType::Int8);
        cast_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let cast = graph.create_node(
            OpType::CallFunction,
            OpTarget::To,
            smallvec![Arg::Node(input)],
            Vec::new(),
            cast_meta,
            "x_cast".to_string(),
        );

        let store = TensorStore::default();
        let cast_node = graph.node(cast).expect("cast node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, cast_node),
            vec![1, 224, 224, 3]
        );
    }

    #[test]
    fn physical_binary_shape_uses_its_own_dim_order() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let lhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[1, 3, 224, 224], DType::Float32),
            "lhs".to_string(),
        );
        let rhs = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[1], DType::Float32),
            "rhs".to_string(),
        );
        let mut add_meta = meta(&[1, 3, 224, 224], DType::Float32);
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
        let add_node = graph.node(add).expect("add node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, add_node),
            vec![1, 224, 224, 3]
        );
    }

    #[test]
    fn logical_control_flow_getitem_keeps_contract_shape() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let cond = graph.create_node(
            OpType::CallFunction,
            OpTarget::CondIf,
            smallvec![],
            Vec::new(),
            meta(&[1, 2, 2, 1], DType::Float32),
            "cond".to_string(),
        );
        let mut getitem_meta = meta(&[1, 1, 2, 2], DType::Float32);
        getitem_meta.tosa_dim_order = Some(smallvec![0, 1, 2, 3]);
        getitem_meta.retab_logical_layout_boundary = true;
        let getitem = graph.create_node(
            OpType::CallFunction,
            OpTarget::Other("<built-in function getitem>".to_string()),
            smallvec![Arg::Node(cond), Arg::Int(0)],
            Vec::new(),
            getitem_meta,
            "getitem".to_string(),
        );

        let store = TensorStore::default();
        let getitem_node = graph.node(getitem).expect("getitem node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, getitem_node),
            vec![1, 1, 2, 2]
        );
    }

    #[test]
    fn scalar_dim_reduction_shape_matches_emitted_axis() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let mut input_meta = meta(&[1, 16, 16, 16], DType::Bool);
        input_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::LogicalNot,
            smallvec![],
            Vec::new(),
            input_meta,
            "mask".to_string(),
        );

        let mut any_meta = meta(&[1, 16, 16, 1], DType::Bool);
        any_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let any = graph.create_node(
            OpType::CallFunction,
            OpTarget::Any,
            smallvec![Arg::Node(input), Arg::Int(3), Arg::Bool(true)],
            Vec::new(),
            any_meta,
            "mask_any".to_string(),
        );

        let store = TensorStore::default();
        let any_node = graph.node(any).expect("any node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, any_node),
            vec![1, 16, 1, 16]
        );
    }

    #[test]
    fn qdq_passthrough_does_not_remap_reduction_axes_from_stale_dim_order() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[4, 1, 4, 1024], DType::Int32),
            "grouped".to_string(),
        );
        let mut rescale_meta = meta(&[4, 1, 4, 1024], DType::Int32);
        rescale_meta.tosa_dim_order = Some(smallvec![0, 2, 3, 1]);
        let rescale = graph.create_node(
            OpType::CallFunction,
            OpTarget::TosaRescale,
            smallvec![Arg::Node(input)],
            Vec::new(),
            rescale_meta,
            "grouped_rescale".to_string(),
        );
        let sum = graph.create_node(
            OpType::CallFunction,
            OpTarget::Sum,
            smallvec![
                Arg::Node(rescale),
                Arg::IntList(smallvec![3, 2]),
                Arg::Bool(true)
            ],
            Vec::new(),
            meta(&[4, 1, 1, 1], DType::Int32),
            "grouped_sum".to_string(),
        );

        let store = TensorStore::default();
        let sum_node = graph.node(sum).expect("sum node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, sum_node),
            vec![4, 1, 1, 1]
        );
    }

    #[test]
    fn mapped_axis_falls_back_to_raw_axis_with_remaining_extent() {
        assert_eq!(mapped_axis_for_current_shape(2, 1, &[4, 1, 4, 1]), 2);
        assert_eq!(mapped_axis_for_current_shape(2, 1, &[4, 4, 1, 1]), 1);
    }
}
