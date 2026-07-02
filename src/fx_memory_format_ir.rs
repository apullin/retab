use std::collections::{HashMap, HashSet, VecDeque};

use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp,
};
use crate::fx_target::{
    classify_ir_target, is_axis_reduction_target, is_binary_arithmetic_target,
    is_boundary_rescale_target, is_cat_target, is_clamp_target, is_comparison_target,
    is_constant_only_shape_transform_target, is_conv1d_default_target, is_conv2d_default_target,
    is_conv3d_default_target, is_expand_target, is_layout_passthrough_target, is_max_pool2d_target,
    is_pad_target, is_permute_target, is_plain_avg_pool2d_target, is_raw_aten_convolution_target,
    is_resize_2d_target, is_select_copy_target, is_slice_tensor_target, is_tosa_transpose_target,
    is_transpose_conv2d_target, is_view_like_target, is_where_target,
    is_zero_input_constant_target, noop_target, table_op_target_kind, IrTargetKind,
};
use crate::runner_logging::log_line;

pub struct IrToTosaMemoryFormatPass;

const RETAB_PHYSICAL_PERMUTE_ARGS: &str = "retab_physical_permute_args";
const RETAB_KEEP_IDENTITY_PERMUTE: &str = "retab_keep_identity_permute";

fn channels_last_order(rank: usize, spatial_rank: usize) -> Vec<usize> {
    if rank <= 2 {
        return (0..rank).collect();
    }
    let spatial_rank = spatial_rank.min(rank.saturating_sub(2));
    let channel_axis = rank - (spatial_rank + 1);
    let mut order: Vec<usize> = (0..channel_axis).collect();
    order.extend(channel_axis + 1..rank);
    order.push(channel_axis);
    order
}

fn channels_last_inverse_order(rank: usize, spatial_rank: usize) -> Vec<usize> {
    let order = channels_last_order(rank, spatial_rank);
    inverse_order(&order)
}

fn inverse_order(order: &[usize]) -> Vec<usize> {
    let rank = order.len();
    let mut inverse = vec![0usize; rank];
    for (idx, &axis) in order.iter().enumerate() {
        inverse[axis] = idx;
    }
    inverse
}

fn inverse_i64_permutation(perm: &[i64]) -> Option<Vec<usize>> {
    let mut inverse = vec![usize::MAX; perm.len()];
    for (idx, &axis) in perm.iter().enumerate() {
        let axis = usize::try_from(axis).ok()?;
        if axis >= perm.len() || inverse[axis] != usize::MAX {
            return None;
        }
        inverse[axis] = idx;
    }
    Some(inverse)
}

fn rank4_spatial_only_permute(perm: &[i64]) -> bool {
    perm.len() == 4 && perm[0] == 0 && perm[1] == 1 && matches!(&perm[2..], [2, 3] | [3, 2])
}

fn output_order_preserving_physical_input(
    input_order: &[usize],
    user_perm: &[i64],
) -> Option<Vec<usize>> {
    let user_inv = inverse_i64_permutation(user_perm)?;
    input_order
        .iter()
        .map(|&axis| user_inv.get(axis).copied())
        .collect()
}

fn physical_storage_preserving_output_order(
    graph: &IrGraph,
    permute_id: NodeId,
    input_id: NodeId,
    user_perm: &[i64],
    input_order: &[usize],
) -> Option<Vec<usize>> {
    if !rank4_spatial_only_permute(user_perm) {
        return None;
    }
    let output_order = output_order_preserving_physical_input(input_order, user_perm)?;
    let Some(input_shape) = physical_shape_for_node(graph, input_id) else {
        return None;
    };
    let Some(output_shape) = graph.node_shape(permute_id) else {
        return None;
    };
    (permuted_shape(output_shape, &output_order) == input_shape).then_some(output_order)
}

fn should_materialize_logical_spatial_permute(
    graph: &IrGraph,
    permute_id: NodeId,
    input_id: NodeId,
    user_perm: &[i64],
    input_order: &[usize],
) -> bool {
    rank4_spatial_only_permute(user_perm)
        && input_order == channels_last_order(4, spatial_rank(graph, input_id, 2))
        && output_is_physical(graph, permute_id)
}

fn initial_spatial_rank(rank: usize) -> usize {
    if rank >= 4 {
        rank - 2
    } else {
        0
    }
}

fn memory_format_differs(shape: &[i64], spatial_rank: usize) -> bool {
    let rank = shape.len();
    if rank <= 2 {
        return false;
    }
    let spatial_rank = spatial_rank.min(rank.saturating_sub(2));
    if spatial_rank == 0 {
        return false;
    }
    if rank != spatial_rank + 2 {
        return false;
    }
    let channel_idx = rank.saturating_sub(spatial_rank + 1);
    let channel_dim = shape.get(channel_idx).copied().unwrap_or(1);
    let spatial_dims = &shape[channel_idx + 1..];
    channel_dim > 1 && spatial_dims.iter().any(|&d| d > 1)
}

fn is_channel_reshape(
    input_shape: &[i64],
    output_shape: &[i64],
    input_spatial_rank: usize,
    output_spatial_rank: usize,
) -> bool {
    let valid = [4, 5, 6];
    if !valid.contains(&input_shape.len()) || !valid.contains(&output_shape.len()) {
        return false;
    }
    if input_shape.len() < output_shape.len()
        && input_shape.first() == output_shape.first()
        && input_shape.get(1) == output_shape.get(1)
    {
        return false;
    }
    let in_sr = input_spatial_rank.min(input_shape.len().saturating_sub(2));
    let out_sr = output_spatial_rank.min(output_shape.len().saturating_sub(2));
    let in_ch_idx = input_shape.len().saturating_sub(in_sr + 1);
    let out_ch_idx = output_shape.len().saturating_sub(out_sr + 1);

    let c_old = input_shape.get(in_ch_idx).copied().unwrap_or(1);
    let c_new = output_shape.get(out_ch_idx).copied().unwrap_or(1);
    let n_old: i64 = input_shape[..in_ch_idx].iter().product();
    let n_new: i64 = output_shape[..out_ch_idx].iter().product();

    n_old != n_new || c_old != c_new
}

fn same_shape_and_dtype(graph: &IrGraph, lhs: NodeId, rhs: NodeId) -> bool {
    graph.node_shape(lhs) == graph.node_shape(rhs) && graph.node_dtype(lhs) == graph.node_dtype(rhs)
}

fn is_input_placeholder(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::Placeholder || node.meta.tensor_facts().is_none() {
        return false;
    }
    if node.meta.extra_bool("arm_constant_placeholder") == Some(true) {
        return false;
    }
    if node.name.starts_with("p_") || node.name.starts_with("b_") || node.name.starts_with("c_") {
        return false;
    }
    node.tensor.is_none()
}

fn is_constant_placeholder(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    node.op == NodeOp::Placeholder && node.tensor.is_some() && node.meta.tensor_facts().is_some()
}

fn is_spatial_conv_feature_value(graph: &IrGraph, id: NodeId) -> bool {
    let mut stack = vec![id];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !graph.nodes[id].alive {
            continue;
        }
        let node = &graph.nodes[id];
        if graph
            .node_shape(id)
            .is_some_and(|shape| shape.len() == 4 && shape[2] > 1 && shape[3] > 1)
            && (is_raw_aten_convolution_target(&node.target)
                || is_transpose_conv2d_target(&node.target))
        {
            return true;
        }
        if node.is_q_or_dq()
            || is_boundary_rescale_target(&node.target)
            || noop_target(&node.target).is_some()
        {
            if let Some(input_id) = node.first_node_arg() {
                stack.push(input_id);
            }
        }
    }
    false
}

fn is_channel_reshape_edge(graph: &IrGraph, input_id: NodeId, view_id: NodeId) -> bool {
    if !is_view_like_target(&graph.nodes[view_id].target)
        && !is_constant_only_shape_transform_target(&graph.nodes[view_id].target)
    {
        return false;
    }
    let Some(input_shape) = graph.node_shape(input_id) else {
        return false;
    };
    let Some(output_shape) = graph.node_shape(view_id) else {
        return false;
    };
    is_channel_reshape(
        input_shape,
        output_shape,
        spatial_rank(graph, input_id, 2),
        spatial_rank(graph, view_id, 2),
    )
}

fn view_will_materialize_input_layout_boundary(graph: &IrGraph, view_id: NodeId) -> bool {
    let Some(input_id) = graph.nodes[view_id].first_node_arg() else {
        return false;
    };
    let Some(input_shape) = graph.node_shape(input_id) else {
        return false;
    };
    let input_rank = input_shape.len();
    let input_spatial_rank = spatial_rank(graph, input_id, 2);
    value_emits_physical_layout(graph, input_id, input_rank, input_spatial_rank)
        && !is_permute_target(&graph.nodes[input_id].target)
        && is_channel_reshape_edge(graph, input_id, view_id)
}

fn is_rank4_to_rank5_depth_before_channel_view(input_shape: &[i64], output_shape: &[i64]) -> bool {
    input_shape.len() == 4
        && output_shape.len() == 5
        && input_shape.iter().product::<i64>() == output_shape.iter().product::<i64>()
        && output_shape[0] * output_shape[1] == input_shape[0]
        && output_shape[2] == input_shape[1]
        && output_shape[3] == input_shape[2]
        && output_shape[4] == input_shape[3]
}

fn is_layout_sensitive_sink_target(target: &str) -> bool {
    is_raw_aten_convolution_target(target)
        || is_conv1d_default_target(target)
        || is_conv2d_default_target(target)
        || is_conv3d_default_target(target)
        || is_transpose_conv2d_target(target)
        || is_plain_avg_pool2d_target(target)
        || is_max_pool2d_target(target)
        || is_resize_2d_target(target)
        || matches!(classify_ir_target(target), IrTargetKind::WhileLoop)
        || target.contains("backend.tosa.SLICE")
}

fn is_bool_layout_transparent_target(target: &str) -> bool {
    is_comparison_target(target)
        || [
            "logical_not",
            "logical_and",
            "logical_or",
            "logical_xor",
            "bitwise_not",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
        ]
        .iter()
        .any(|marker| target.contains(marker))
}

/// Whether a node's output is consumed in physical (NHWC) layout. False (logical) when the
/// output only reaches the graph Output through quantize/dequantize and layout-neutral ops;
/// true once any non-transparent consumer (conv, pool, another layout op, ...) is found.
fn output_is_physical(graph: &IrGraph, id: NodeId) -> bool {
    let mut stack = live_user_ids_by_arg_scan(graph, id);
    let mut seen: HashSet<NodeId> = HashSet::new();
    while let Some(uid) = stack.pop() {
        if !seen.insert(uid) {
            continue;
        }
        let node = &graph.nodes[uid];
        if node.op == NodeOp::Output {
            continue;
        }
        if node.meta.extra_bool("retab_logical_layout_boundary") == Some(true) {
            continue;
        }
        if is_channel_reshape_edge(graph, id, uid) {
            continue;
        }
        if (is_view_like_target(&node.target)
            || is_permute_target(&node.target)
            || is_constant_only_shape_transform_target(&node.target))
            && node.meta.rank().unwrap_or(0) < 4
        {
            continue;
        }
        if output_layout_transparent_node(graph, uid) {
            stack.extend(live_user_ids_by_arg_scan(graph, uid));
        } else if is_layout_sensitive_sink_target(&node.target) {
            return true;
        }
    }
    false
}

fn value_has_physical_origin(graph: &IrGraph, id: NodeId) -> bool {
    let mut stack = vec![id];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !graph.nodes[id].alive {
            continue;
        }
        let node = &graph.nodes[id];
        if node.op == NodeOp::Placeholder {
            if let Some(order) = node.meta.val_dim_order() {
                if node.meta.rank().unwrap_or(0) >= 4
                    && order.iter().enumerate().any(|(axis, &dim)| axis != dim)
                {
                    return true;
                }
            }
            continue;
        }
        if node.op != NodeOp::CallFunction {
            continue;
        }
        if node.meta.extra_bool("retab_logical_layout_boundary") == Some(true) {
            continue;
        }
        if is_pool_data_getitem(graph, id) {
            if let Some(input_id) = node.first_node_arg() {
                stack.push(input_id);
            }
            continue;
        }
        if is_view_like_target(&node.target)
            || is_constant_only_shape_transform_target(&node.target)
        {
            if let Some(input_id) = node.first_node_arg() {
                let input_rank = graph.nodes[input_id].meta.rank().unwrap_or(0);
                let output_rank = node.meta.rank().unwrap_or(0);
                if input_rank != output_rank && (input_rank < 4 || output_rank < 4) {
                    continue;
                }
            }
        }
        if is_permute_target(&node.target)
            && permute_axes(graph, id).is_some_and(|perm| !identity_permutation(&perm))
        {
            continue;
        }
        if !output_layout_transparent_node(graph, id) {
            if is_layout_sensitive_sink_target(&node.target) {
                return true;
            }
            continue;
        }
        for arg in &node.args {
            collect_arg_node_ids(arg, &mut stack);
        }
        for arg in node.kwargs.values() {
            collect_arg_node_ids(arg, &mut stack);
        }
    }
    false
}

fn value_uses_physical_layout(graph: &IrGraph, id: NodeId) -> bool {
    output_is_physical(graph, id) || value_has_physical_origin(graph, id)
}

fn value_emits_physical_layout(
    graph: &IrGraph,
    id: NodeId,
    rank: usize,
    spatial_rank: usize,
) -> bool {
    let physical_order = channels_last_order(rank, spatial_rank);
    if graph.nodes[id].meta.tosa_dim_order() == Some(physical_order.as_slice()) {
        return true;
    }
    rank >= 4
        && !is_logical_layout_boundary(graph, id)
        && output_layout_transparent_node(graph, id)
        && value_uses_physical_layout(graph, id)
}

fn is_logical_layout_boundary(graph: &IrGraph, id: NodeId) -> bool {
    graph.nodes[id]
        .meta
        .extra_bool("retab_logical_layout_boundary")
        == Some(true)
}

fn is_cond_output_getitem(graph: &IrGraph, id: NodeId) -> bool {
    graph.nodes[id].is_getitem()
        && graph.nodes[id].first_node_arg().is_some_and(|source_id| {
            matches!(
                classify_ir_target(&graph.nodes[source_id].target),
                IrTargetKind::Cond
            )
        })
}

fn is_pool_data_getitem(graph: &IrGraph, id: NodeId) -> bool {
    graph.nodes[id].is_getitem()
        && graph.nodes[id].arg_int_or(1, 0) == 0
        && graph.nodes[id]
            .first_node_arg()
            .is_some_and(|source_id| is_max_pool2d_target(&graph.nodes[source_id].target))
}

fn preserves_natural_spatial_rank(graph: &IrGraph, id: NodeId) -> bool {
    graph.nodes[id]
        .meta
        .extra_bool("retab_preserve_spatial_rank")
        == Some(true)
}

fn dim_order_or_identity(order: Option<&[usize]>, rank: usize) -> Vec<usize> {
    order
        .map(|order| order.to_vec())
        .unwrap_or_else(|| (0..rank).collect())
}

fn public_input_dim_order(
    graph: &IrGraph,
    id: NodeId,
    rank: usize,
    spatial_rank: usize,
) -> Vec<usize> {
    if rank >= 4 {
        let channels_last = channels_last_order(rank, spatial_rank);
        if graph.nodes[id].meta.val_dim_order() == Some(channels_last.as_slice()) {
            return channels_last;
        }
    }
    (0..rank).collect()
}

fn physical_order_perm(input_order: &[usize], output_order: &[usize]) -> Option<Vec<usize>> {
    if input_order.len() != output_order.len() {
        return None;
    }
    output_order
        .iter()
        .map(|&logical_axis| {
            input_order
                .iter()
                .position(|&input_axis| input_axis == logical_axis)
        })
        .collect()
}

fn dim_order_copy_perm(graph: &IrGraph, id: NodeId) -> Option<Vec<usize>> {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction || noop_target(&node.target).is_none() {
        return None;
    }
    let input_id = node.first_node_arg()?;
    let rank = node.meta.rank()?;
    if rank < 4 || graph.nodes[input_id].meta.rank()? != rank {
        return None;
    }
    let input_order = dim_order_or_identity(graph.nodes[input_id].meta.val_dim_order(), rank);
    let output_order = node.meta.val_dim_order()?;
    if input_order == output_order {
        return None;
    }
    physical_order_perm(&input_order, output_order)
}

fn dim_order_copy_is_boundary_like(graph: &IrGraph, id: NodeId) -> bool {
    let mut stack = live_user_ids_by_arg_scan(graph, id);
    let mut seen = HashSet::new();
    while let Some(user_id) = stack.pop() {
        if !seen.insert(user_id) {
            continue;
        }
        let user = &graph.nodes[user_id];
        if user.op == NodeOp::Output {
            continue;
        }
        if user.is_q_or_dq() || is_boundary_rescale_target(&user.target) {
            stack.extend(live_user_ids_by_arg_scan(graph, user_id));
            continue;
        }
        return false;
    }
    true
}

fn has_layout_output_user(graph: &IrGraph, id: NodeId) -> bool {
    graph.live_user_ids(id).into_iter().any(|user_id| {
        let user = &graph.nodes[user_id];
        user.alive && is_tosa_transpose_target(&user.target) && user.name.contains("_output_layout")
    })
}

fn is_permute_or_noop_wrapped_permute(graph: &IrGraph, id: NodeId) -> bool {
    if is_permute_target(&graph.nodes[id].target) {
        return true;
    }
    noop_target(&graph.nodes[id].target).is_some()
        && graph.nodes[id]
            .first_node_arg()
            .is_some_and(|input_id| is_permute_target(&graph.nodes[input_id].target))
}

fn is_control_flow_submodule(graph: &IrGraph) -> bool {
    graph.nodes.iter().any(|node| {
        node.alive
            && (node.meta.extra_bool("arm_control_flow_placeholder") == Some(true)
                || node
                    .meta
                    .extras
                    .contains_key("arm_control_flow_output_fake_tensors")
                || node
                    .meta
                    .extras
                    .contains_key("arm_control_flow_output_dequant_qparams"))
    })
}

fn arg_constant_subgraph(graph: &IrGraph, arg: &IrArg, memo: &mut HashMap<NodeId, bool>) -> bool {
    match arg {
        IrArg::Node(node_id) => is_constant_subgraph(graph, *node_id, memo),
        IrArg::List(values) => values
            .iter()
            .all(|value| arg_constant_subgraph(graph, value, memo)),
        _ => true,
    }
}

fn is_constant_subgraph(graph: &IrGraph, id: NodeId, memo: &mut HashMap<NodeId, bool>) -> bool {
    if let Some(&cached) = memo.get(&id) {
        return cached;
    }

    let node = &graph.nodes[id];
    let result = if !node.alive {
        false
    } else if node.op == NodeOp::Placeholder {
        node.tensor.is_some() || node.meta.extra_bool("arm_constant_placeholder") == Some(true)
    } else if node.op == NodeOp::CallFunction {
        if is_zero_input_constant_target(&node.target) {
            true
        } else if is_constant_only_shape_transform_target(&node.target) {
            node.args
                .iter()
                .all(|arg| arg_constant_subgraph(graph, arg, memo))
                && node
                    .kwargs
                    .values()
                    .all(|arg| arg_constant_subgraph(graph, arg, memo))
        } else {
            false
        }
    } else {
        false
    };

    memo.insert(id, result);
    result
}

fn is_transposed_aten_convolution(node: &crate::fx_ir::IrNode) -> bool {
    if !is_raw_aten_convolution_target(&node.target) {
        return false;
    }
    node.arg_or_kwarg_bool(6, "transposed") == Some(true)
}

fn is_transpose_conv2d_weight(graph: &IrGraph, id: NodeId) -> bool {
    graph.live_user_ids(id).into_iter().any(|user_id| {
        let user = &graph.nodes[user_id];
        if user.op != NodeOp::CallFunction {
            return false;
        }
        if !(is_transpose_conv2d_target(&user.target) || is_transposed_aten_convolution(user)) {
            return false;
        }
        matches!(user.args.get(1), Some(IrArg::Node(weight_id)) if *weight_id == id)
    })
}

fn collect_output_node_ids(arg: &IrArg, output_nodes: &mut Vec<NodeId>) {
    match arg {
        IrArg::Node(id) => output_nodes.push(*id),
        IrArg::List(items) => {
            for item in items {
                collect_output_node_ids(item, output_nodes);
            }
        }
        _ => {}
    }
}

fn collect_arg_node_ids(arg: &IrArg, nodes: &mut Vec<NodeId>) {
    match arg {
        IrArg::Node(id) => nodes.push(*id),
        IrArg::List(items) => {
            for item in items {
                collect_arg_node_ids(item, nodes);
            }
        }
        _ => {}
    }
}

fn live_user_ids_by_arg_scan(graph: &IrGraph, id: NodeId) -> Vec<NodeId> {
    graph
        .live_node_ids()
        .into_iter()
        .filter(|&candidate| {
            candidate != id
                && graph.nodes[candidate]
                    .args
                    .iter()
                    .chain(graph.nodes[candidate].kwargs.values())
                    .any(|arg| arg.contains_node(id))
        })
        .collect()
}

fn output_node_ids(graph: &IrGraph, output_id: NodeId) -> Vec<NodeId> {
    let mut output_nodes = Vec::new();
    for arg in &graph.nodes[output_id].args {
        collect_output_node_ids(arg, &mut output_nodes);
    }
    output_nodes
}

fn permuted_shape(shape: &[i64], perm: &[usize]) -> Vec<i64> {
    perm.iter()
        .filter_map(|&idx| shape.get(idx).copied())
        .collect()
}

fn physical_shape_for_node(graph: &IrGraph, id: NodeId) -> Option<Vec<i64>> {
    let shape = graph.node_shape(id)?;
    graph.nodes[id]
        .meta
        .tosa_dim_order()
        .filter(|order| order.len() == shape.len())
        .map(|order| permuted_shape(shape, order))
        .or_else(|| Some(shape.to_vec()))
}

fn physical_binary_output_shape(graph: &IrGraph, id: NodeId) -> Option<Vec<i64>> {
    let node = &graph.nodes[id];
    if !is_binary_arithmetic_target(&node.target) {
        return None;
    }
    let shape = graph.node_shape(id)?;
    let order = node.meta.tosa_dim_order()?;
    if order.len() != shape.len() || identity_permutation(order) {
        return None;
    }
    Some(permuted_shape(shape, order))
}

fn int_perm_matches_order(perm: &[i64], order: &[usize]) -> bool {
    perm.len() == order.len()
        && perm
            .iter()
            .map(|&axis| usize::try_from(axis).ok())
            .eq(order.iter().map(|&axis| Some(axis)))
}

fn is_physical_identity_layout_permute(
    graph: &IrGraph,
    permute_id: NodeId,
    input_id: NodeId,
    user_perm: &[i64],
    input_order: &[usize],
) -> bool {
    if !int_perm_matches_order(user_perm, input_order) {
        return false;
    }
    graph
        .node_shape(input_id)
        .zip(graph.node_shape(permute_id))
        .is_some_and(|(input_shape, output_shape)| {
            permuted_shape(input_shape, input_order) == output_shape
        })
}

#[derive(Clone, Copy)]
enum PhysicalPermuteConsumer {
    Binary(NodeId),
    Rescale(NodeId),
    Noop(NodeId),
    Other,
}

#[derive(Clone, Copy)]
enum PhysicalPermuteBinaryRewrite {
    DirectBinary {
        binary_id: NodeId,
        old_input_id: NodeId,
    },
    ThroughRescale {
        rescale_id: NodeId,
        old_input_id: NodeId,
        binary_id: NodeId,
    },
}

fn classify_physical_permute_consumer(graph: &IrGraph, id: NodeId) -> PhysicalPermuteConsumer {
    if physical_binary_output_shape(graph, id).is_some() {
        PhysicalPermuteConsumer::Binary(id)
    } else if graph.nodes[id].alive && is_boundary_rescale_target(&graph.nodes[id].target) {
        PhysicalPermuteConsumer::Rescale(id)
    } else if graph.nodes[id].alive && noop_target(&graph.nodes[id].target).is_some() {
        PhysicalPermuteConsumer::Noop(id)
    } else {
        PhysicalPermuteConsumer::Other
    }
}

fn collect_rescale_binary_rewrites(
    graph: &IrGraph,
    rescale_id: NodeId,
    old_input_id: NodeId,
    rewrites: &mut Vec<PhysicalPermuteBinaryRewrite>,
) {
    for binary_id in graph.live_user_ids(rescale_id) {
        if physical_binary_output_shape(graph, binary_id).is_some() {
            rewrites.push(PhysicalPermuteBinaryRewrite::ThroughRescale {
                rescale_id,
                old_input_id,
                binary_id,
            });
        }
    }
}

fn collect_noop_wrapped_physical_permute_rewrites(
    graph: &IrGraph,
    noop_id: NodeId,
    rewrites: &mut Vec<PhysicalPermuteBinaryRewrite>,
) {
    for next_id in graph.live_user_ids(noop_id) {
        match classify_physical_permute_consumer(graph, next_id) {
            PhysicalPermuteConsumer::Binary(binary_id) => {
                rewrites.push(PhysicalPermuteBinaryRewrite::DirectBinary {
                    binary_id,
                    old_input_id: noop_id,
                });
            }
            PhysicalPermuteConsumer::Rescale(rescale_id) => {
                collect_rescale_binary_rewrites(graph, rescale_id, noop_id, rewrites);
            }
            PhysicalPermuteConsumer::Noop(_) | PhysicalPermuteConsumer::Other => {}
        }
    }
}

fn collect_physical_permute_binary_rewrites(
    graph: &IrGraph,
    permute_id: NodeId,
) -> Vec<PhysicalPermuteBinaryRewrite> {
    let mut rewrites = Vec::new();
    for user_id in graph.live_user_ids(permute_id) {
        match classify_physical_permute_consumer(graph, user_id) {
            PhysicalPermuteConsumer::Binary(binary_id) => {
                rewrites.push(PhysicalPermuteBinaryRewrite::DirectBinary {
                    binary_id,
                    old_input_id: permute_id,
                });
            }
            PhysicalPermuteConsumer::Rescale(rescale_id) => {
                collect_rescale_binary_rewrites(graph, rescale_id, permute_id, &mut rewrites);
            }
            PhysicalPermuteConsumer::Noop(noop_id) => {
                collect_noop_wrapped_physical_permute_rewrites(graph, noop_id, &mut rewrites);
            }
            PhysicalPermuteConsumer::Other => {}
        }
    }
    rewrites
}

fn binary_id_for_physical_permute_rewrite(rewrite: PhysicalPermuteBinaryRewrite) -> NodeId {
    match rewrite {
        PhysicalPermuteBinaryRewrite::DirectBinary { binary_id, .. }
        | PhysicalPermuteBinaryRewrite::ThroughRescale { binary_id, .. } => binary_id,
    }
}

fn erase_if_dead(graph: &mut IrGraph, id: NodeId) {
    if graph.live_user_ids(id).is_empty() {
        graph.erase(id);
    }
}

fn apply_physical_permute_binary_rewrite(
    graph: &mut IrGraph,
    rewrite: PhysicalPermuteBinaryRewrite,
    source_id: NodeId,
    source_shape: &[i64],
) -> bool {
    let binary_id = binary_id_for_physical_permute_rewrite(rewrite);
    let Some(user_shape) = physical_binary_output_shape(graph, binary_id) else {
        return false;
    };
    if source_shape != user_shape {
        return false;
    }

    match rewrite {
        PhysicalPermuteBinaryRewrite::DirectBinary {
            binary_id,
            old_input_id,
        } => {
            let modified = graph.replace_input_use(binary_id, old_input_id, source_id);
            if modified {
                erase_if_dead(graph, old_input_id);
            }
            modified
        }
        PhysicalPermuteBinaryRewrite::ThroughRescale {
            rescale_id,
            old_input_id,
            binary_id,
        } => {
            if !graph.replace_input_use(rescale_id, old_input_id, source_id) {
                return false;
            }
            if let Some(order) = graph.nodes[binary_id].meta.cloned_tosa_dim_order() {
                graph.nodes[rescale_id].meta.set_tosa_dim_order(order);
            }
            if let Some(spatial_rank) = graph.nodes[binary_id].meta.tosa_spatial_rank() {
                graph.nodes[rescale_id]
                    .meta
                    .set_tosa_spatial_rank(spatial_rank);
            }
            erase_if_dead(graph, old_input_id);
            true
        }
    }
}

fn unique_name(graph: &IrGraph, base: &str) -> String {
    if graph.nodes.iter().all(|node| node.name != base) {
        return base.to_string();
    }
    let mut counter = 0u32;
    loop {
        let candidate = format!("{}_{}", base, counter);
        if graph.nodes.iter().all(|node| node.name != candidate) {
            return candidate;
        }
        counter += 1;
    }
}

fn transpose_meta(input_meta: &NodeMeta, perm: &[usize]) -> NodeMeta {
    let mut meta = input_meta
        .tensor_facts()
        .map(|val| input_meta.cloned_with_same_dtype_or_float32(permuted_shape(&val.shape, perm)))
        .unwrap_or_else(|| input_meta.clone());
    meta.set_tosa_dim_order((0..perm.len()).collect());
    meta
}

fn add_transpose(
    graph: &mut IrGraph,
    input_id: NodeId,
    base_name: String,
    perm: &[usize],
) -> NodeId {
    let perm_i64: Vec<i64> = perm.iter().map(|&value| value as i64).collect();
    let name = unique_name(graph, &base_name);
    graph.add_node(
        NodeOp::CallFunction,
        name,
        "backend.tosa.TRANSPOSE.default".to_string(),
        vec![IrArg::Node(input_id), i64_list(&perm_i64)],
        transpose_meta(&graph.nodes[input_id].meta, perm),
    )
}

fn transpose_perm(graph: &IrGraph, id: NodeId) -> Option<Vec<usize>> {
    if !is_tosa_transpose_target(&graph.nodes[id].target) {
        return None;
    }
    graph.nodes[id]
        .args
        .get(1)?
        .as_int_list()?
        .into_iter()
        .map(|axis| usize::try_from(axis).ok())
        .collect()
}

fn inverse_permutations(lhs: &[usize], rhs: &[usize]) -> bool {
    lhs.len() == rhs.len()
        && rhs
            .iter()
            .enumerate()
            .all(|(out_axis, &mid_axis)| mid_axis < lhs.len() && lhs[mid_axis] == out_axis)
}

fn identity_permutation(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(axis, &value)| axis == value)
}

fn compose_permutations(first: &[usize], second: &[usize]) -> Option<Vec<usize>> {
    if first.len() != second.len() {
        return None;
    }
    second
        .iter()
        .map(|&axis| first.get(axis).copied())
        .collect()
}

fn permute_axes(graph: &IrGraph, id: NodeId) -> Option<Vec<usize>> {
    if !is_permute_target(&graph.nodes[id].target) {
        return None;
    }
    graph.nodes[id]
        .args
        .get(1)?
        .as_int_list()?
        .into_iter()
        .map(|axis| usize::try_from(axis).ok())
        .collect()
}

fn insert_transpose_before(
    graph: &mut IrGraph,
    user_id: NodeId,
    input_id: NodeId,
    name: String,
    perm: &[usize],
) -> NodeId {
    let transpose_id = add_transpose(graph, input_id, name, perm);
    graph.anchor_after(transpose_id, input_id);
    graph.replace_input_use(user_id, input_id, transpose_id);
    transpose_id
}

fn insert_transpose_after(
    graph: &mut IrGraph,
    producer_id: NodeId,
    name: String,
    perm: &[usize],
) -> NodeId {
    let users = live_user_ids_by_arg_scan(graph, producer_id);
    let transpose_id = add_transpose(graph, producer_id, name, perm);
    graph.nodes[transpose_id]
        .meta
        .set_tosa_dim_order(perm.to_vec());
    if let Some(rank) = graph.nodes[producer_id].meta.rank() {
        graph.nodes[producer_id]
            .meta
            .set_tosa_dim_order((0..rank).collect());
    }
    graph.anchor_after(transpose_id, producer_id);
    for user_id in users {
        graph.replace_input_use(user_id, producer_id, transpose_id);
    }
    transpose_id
}

fn set_spatial_rank(graph: &mut IrGraph, id: NodeId, rank: usize) -> bool {
    graph.nodes[id].meta.set_tosa_spatial_rank(rank)
}

fn spatial_rank(graph: &IrGraph, id: NodeId, default: usize) -> usize {
    graph.nodes[id].meta.tosa_spatial_rank_or(default)
}

fn layout_transparent_node(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction {
        return false;
    }
    node.is_q_or_dq()
        || is_layout_passthrough_target(&node.target)
        || is_slice_tensor_target(&node.target)
        || is_expand_target(&node.target)
        || is_boundary_rescale_target(&node.target)
        || is_binary_arithmetic_target(&node.target)
        || is_cat_target(&node.target)
        || is_clamp_target(&node.target)
        || is_where_target(&node.target)
        || is_bool_layout_transparent_target(&node.target)
        || is_axis_reduction_target(&node.target)
        || table_op_target_kind(&node.target).is_some()
        || is_tosa_layout_neutral_target(&node.target)
}

fn output_layout_transparent_node(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction {
        return false;
    }
    node.is_q_or_dq()
        || noop_target(&node.target).is_some()
        || is_boundary_rescale_target(&node.target)
        || is_view_like_target(&node.target)
        || is_permute_target(&node.target)
        || is_pool_data_getitem(graph, id)
        || is_select_copy_target(&node.target)
        || is_slice_tensor_target(&node.target)
        || is_expand_target(&node.target)
        || is_constant_only_shape_transform_target(&node.target)
        || is_zero_input_constant_target(&node.target)
        || is_binary_arithmetic_target(&node.target)
        || is_cat_target(&node.target)
        || is_clamp_target(&node.target)
        || is_where_target(&node.target)
        || is_bool_layout_transparent_target(&node.target)
        || is_axis_reduction_target(&node.target)
        || table_op_target_kind(&node.target).is_some()
}

fn inverse_permute_passthrough_node(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction {
        return false;
    }
    node.is_q_or_dq()
        || is_boundary_rescale_target(&node.target)
        || is_binary_arithmetic_target(&node.target)
        || table_op_target_kind(&node.target).is_some()
}

fn spatial_rank_follows_inputs(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction {
        return false;
    }
    layout_transparent_node(graph, id)
}

fn is_tosa_layout_neutral_target(target: &str) -> bool {
    let lower = target.to_ascii_lowercase();
    lower.contains("tosa")
        && [
            "add",
            "sub",
            "mul",
            "reduce_sum",
            "table",
            "slice",
            "tile",
            "concat",
            "reshape",
        ]
        .iter()
        .any(|op| lower.contains(op))
}

fn direct_input_spatial_ranks(graph: &IrGraph, id: NodeId, limit: usize) -> Vec<usize> {
    let mut input_ids = Vec::new();
    for arg in &graph.nodes[id].args {
        collect_arg_node_ids(arg, &mut input_ids);
    }
    let output_rank = graph.nodes[id].meta.rank().unwrap_or(0);
    let ranks: Vec<(bool, usize)> = input_ids
        .into_iter()
        .filter(|&input_id| graph.nodes[input_id].alive)
        .filter_map(|input_id| {
            let facts = graph.nodes[input_id].meta.tensor_facts()?;
            let input_rank = facts.rank();
            let rank = spatial_rank(graph, input_id, initial_spatial_rank(input_rank)).min(limit);
            if input_rank < output_rank && rank == 0 {
                return None;
            }
            let carries_layout = facts.shape.iter().any(|&dim| dim > 1);
            Some((carries_layout, rank))
        })
        .collect();
    let has_layout_input = ranks.iter().any(|(carries_layout, _)| *carries_layout);
    ranks
        .into_iter()
        .filter(|(carries_layout, _)| *carries_layout || !has_layout_input)
        .map(|(_, rank)| rank)
        .collect()
}

fn hard_user_spatial_rank_floor(graph: &IrGraph, id: NodeId, limit: usize) -> usize {
    graph
        .users(id)
        .iter()
        .copied()
        .filter(|&user_id| graph.nodes[user_id].alive)
        .filter_map(|user_id| {
            if graph.nodes[user_id].op == NodeOp::Output {
                let rank = graph.nodes[id].meta.tensor_facts()?.rank();
                return Some(initial_spatial_rank(rank).min(limit));
            }
            if spatial_rank_follows_inputs(graph, user_id) {
                return None;
            }
            let rank = graph.nodes[user_id].meta.tensor_facts()?.rank();
            Some(spatial_rank(graph, user_id, initial_spatial_rank(rank)).min(limit))
        })
        .max()
        .unwrap_or(0)
}

fn layout_neutral_spatial_rank_candidate(graph: &IrGraph, id: NodeId) -> Option<usize> {
    if !spatial_rank_follows_inputs(graph, id) {
        return None;
    }
    let val = graph.nodes[id].meta.tensor_facts()?;
    let limit = val.rank().saturating_sub(2);
    if limit == 0 {
        return None;
    }
    let input_floor = direct_input_spatial_ranks(graph, id, limit)
        .into_iter()
        .max()?;
    Some(input_floor.max(hard_user_spatial_rank_floor(graph, id, limit)))
}

fn allow_layout_rank_lowering(
    graph: &IrGraph,
    id: NodeId,
    current: usize,
    candidate: usize,
) -> bool {
    if candidate >= current {
        return true;
    }
    graph.node_shape(id).is_some_and(|shape| {
        shape.len() == 5
            && current == 3
            && candidate == 2
            && shape.get(2..).is_some_and(|spatial| spatial.contains(&1))
    })
}

fn enqueue_layout_rank_user(
    graph: &IrGraph,
    queue: &mut VecDeque<NodeId>,
    queued: &mut HashSet<NodeId>,
    id: NodeId,
) {
    if graph.nodes[id].alive && spatial_rank_follows_inputs(graph, id) && queued.insert(id) {
        queue.push_back(id);
    }
}

impl IrToTosaMemoryFormatPass {
    fn compose_adjacent_permutes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut changed = true;
        while changed {
            changed = false;
            for first_id in graph.live_call_function_ids() {
                if !graph.nodes[first_id].alive || !is_permute_target(&graph.nodes[first_id].target)
                {
                    continue;
                }
                let Some(second_id) = graph.single_live_user(first_id) else {
                    continue;
                };
                if !graph.nodes[second_id].alive
                    || !is_permute_target(&graph.nodes[second_id].target)
                    || graph.nodes[second_id].first_node_arg() != Some(first_id)
                {
                    continue;
                }
                let Some(source_id) = graph.nodes[first_id].first_node_arg() else {
                    continue;
                };
                let Some(first_perm) = permute_axes(graph, first_id) else {
                    continue;
                };
                let Some(second_perm) = permute_axes(graph, second_id) else {
                    continue;
                };
                let Some(composed) = compose_permutations(&first_perm, &second_perm) else {
                    continue;
                };

                let composed_arg: Vec<i64> = composed.into_iter().map(|axis| axis as i64).collect();
                graph.set_arg(second_id, 0, IrArg::Node(source_id));
                graph.set_arg(second_id, 1, i64_list(&composed_arg));
                graph.erase(first_id);
                modified = true;
                changed = true;
                break;
            }
        }
        modified
    }

    fn materialize_dim_order_copies(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            let Some(perm) = dim_order_copy_perm(graph, id) else {
                continue;
            };
            if !dim_order_copy_is_boundary_like(graph, id) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let transpose_id = insert_transpose_before(
                graph,
                id,
                input_id,
                format!("{}_layout", graph.nodes[id].name),
                &perm,
            );
            graph.nodes[transpose_id]
                .meta
                .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
            modified = true;
        }
        modified
    }

    fn collapse_layout_neutral_inverse_permutes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for first_id in graph.live_call_function_ids() {
            if !graph.nodes[first_id].alive || !is_permute_target(&graph.nodes[first_id].target) {
                continue;
            }
            let Some(source_id) = graph.nodes[first_id].first_node_arg() else {
                continue;
            };
            let Some(first_perm) = permute_axes(graph, first_id) else {
                continue;
            };
            let Some(source_shape) = graph.node_shape_vec(source_id) else {
                continue;
            };

            let mut middle_nodes = Vec::new();
            let mut previous_id = first_id;
            let Some(mut cursor_id) = graph.single_live_user(first_id) else {
                continue;
            };
            let second_id = loop {
                if !graph.nodes[cursor_id].alive {
                    break None;
                }
                if is_permute_target(&graph.nodes[cursor_id].target) {
                    let Some(second_perm) = permute_axes(graph, cursor_id) else {
                        break None;
                    };
                    if inverse_permutations(&first_perm, &second_perm)
                        && graph.node_shape(cursor_id) == Some(source_shape.as_slice())
                    {
                        break Some(cursor_id);
                    }
                    break None;
                }
                if !inverse_permute_passthrough_node(graph, cursor_id)
                    || graph.nodes[cursor_id].first_node_arg() != Some(previous_id)
                    || graph.node_shape(cursor_id) != graph.node_shape(previous_id)
                {
                    break None;
                }
                middle_nodes.push(cursor_id);
                previous_id = cursor_id;
                let Some(next_id) = graph.single_live_user(cursor_id) else {
                    break None;
                };
                cursor_id = next_id;
            };
            let Some(second_id) = second_id else {
                continue;
            };

            let replacement_id = middle_nodes.last().copied().unwrap_or(source_id);
            if let Some(&first_middle) = middle_nodes.first() {
                graph.replace_input_use(first_middle, first_id, source_id);
            }
            for &middle_id in &middle_nodes {
                if let Some(mut facts) = graph.nodes[middle_id].meta.tensor_facts().cloned() {
                    facts.shape = source_shape.clone();
                    graph.nodes[middle_id].meta.set_val(facts);
                }
            }
            for user_id in graph.live_user_ids(second_id) {
                graph.replace_input_use(user_id, second_id, replacement_id);
            }
            graph.erase(first_id);
            graph.erase(second_id);
            modified = true;
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn collapse_inverse_view_pairs(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for second_id in graph.live_call_function_ids() {
            if !graph.nodes[second_id].alive || !is_view_like_target(&graph.nodes[second_id].target)
            {
                continue;
            }
            let Some(first_id) = graph.nodes[second_id].first_node_arg() else {
                continue;
            };
            if !graph.nodes[first_id].alive || !is_view_like_target(&graph.nodes[first_id].target) {
                continue;
            }
            let Some(source_id) = graph.nodes[first_id].first_node_arg() else {
                continue;
            };
            if !same_shape_and_dtype(graph, source_id, second_id) {
                continue;
            }

            let first_users = graph.live_user_ids(first_id);
            graph.replace_and_erase(second_id, source_id);
            if first_users.len() == 1 {
                graph.erase(first_id);
            }
            modified = true;
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn collapse_view_inverse_permute_view_chains(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for first_view in graph.live_call_function_ids() {
            if !graph.nodes[first_view].alive
                || !is_view_like_target(&graph.nodes[first_view].target)
            {
                continue;
            }
            let Some(source_id) = graph.nodes[first_view].first_node_arg() else {
                continue;
            };
            let Some(first_perm) = graph.single_live_user(first_view) else {
                continue;
            };
            let Some(first_axes) = permute_axes(graph, first_perm) else {
                continue;
            };
            let mut middle_nodes = Vec::new();
            let mut previous_id = first_perm;
            let Some(mut cursor_id) = graph.single_live_user(first_perm) else {
                continue;
            };
            let second_perm = loop {
                if permute_axes(graph, cursor_id).is_some() {
                    break Some(cursor_id);
                }
                let same_shape = graph.node_shape(cursor_id) == graph.node_shape(previous_id);
                let passthrough = inverse_permute_passthrough_node(graph, cursor_id)
                    || noop_target(&graph.nodes[cursor_id].target).is_some();
                if !same_shape
                    || !passthrough
                    || graph.nodes[cursor_id].first_node_arg() != Some(previous_id)
                {
                    break None;
                }
                middle_nodes.push(cursor_id);
                previous_id = cursor_id;
                let Some(next_id) = graph.single_live_user(cursor_id) else {
                    break None;
                };
                cursor_id = next_id;
            };
            let Some(second_perm) = second_perm else {
                continue;
            };
            let Some(second_axes) = permute_axes(graph, second_perm) else {
                continue;
            };
            if !inverse_permutations(&first_axes, &second_axes) {
                continue;
            }
            let Some(second_view) = graph.single_live_user(second_perm) else {
                continue;
            };
            if !graph.nodes[second_view].alive
                || !is_view_like_target(&graph.nodes[second_view].target)
                || !same_shape_and_dtype(graph, source_id, second_view)
            {
                continue;
            }

            graph.replace_and_erase(second_view, source_id);
            graph.erase(second_perm);
            for middle_id in middle_nodes {
                graph.erase(middle_id);
            }
            graph.erase(first_perm);
            graph.erase(first_view);
            modified = true;
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn annotate_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_node_ids() {
            let Some(val) = graph.nodes[id].meta.tensor_facts() else {
                continue;
            };
            let rank = initial_spatial_rank(val.rank());
            modified |= set_spatial_rank(graph, id, rank);
        }
        for id in graph.live_call_function_ids() {
            if !is_view_like_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(input_shape) = graph.node_shape(input_id) else {
                continue;
            };
            let Some(output_shape) = graph.node_shape(id) else {
                continue;
            };
            if is_rank4_to_rank5_depth_before_channel_view(input_shape, output_shape)
                && is_spatial_conv_feature_value(graph, input_id)
            {
                let rank = spatial_rank(graph, input_id, initial_spatial_rank(input_shape.len()));
                modified |= set_spatial_rank(graph, id, rank);
                graph.nodes[id]
                    .meta
                    .set_extra("retab_preserve_spatial_rank", MetaValue::Bool(true));
            }
        }
        modified
    }

    fn propagate_layout_neutral_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut queue = VecDeque::new();
        let mut queued = HashSet::new();
        for id in graph.live_node_ids() {
            enqueue_layout_rank_user(graph, &mut queue, &mut queued, id);
        }

        while let Some(id) = queue.pop_front() {
            queued.remove(&id);
            if preserves_natural_spatial_rank(graph, id) {
                continue;
            }
            let Some(propagated) = layout_neutral_spatial_rank_candidate(graph, id) else {
                continue;
            };
            let current = spatial_rank(graph, id, 0);
            if propagated != current && allow_layout_rank_lowering(graph, id, current, propagated) {
                modified |= set_spatial_rank(graph, id, propagated);
                for &user_id in graph.users(id) {
                    enqueue_layout_rank_user(graph, &mut queue, &mut queued, user_id);
                }
            }
        }
        modified
    }

    fn propagate_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut changed = true;
        while changed {
            changed = false;
            for id in graph.live_node_ids().into_iter().rev() {
                if graph.nodes[id].meta.tensor_facts().is_none() {
                    continue;
                }
                if preserves_natural_spatial_rank(graph, id) {
                    continue;
                }
                let rank = graph.nodes[id]
                    .meta
                    .tensor_facts()
                    .map(|val| val.rank())
                    .unwrap_or(0);
                let limit = rank.saturating_sub(2);
                let current = spatial_rank(graph, id, 0);
                let mut propagated = graph
                    .users(id)
                    .iter()
                    .filter(|&&user_id| graph.nodes[user_id].alive)
                    .map(|&user_id| spatial_rank(graph, user_id, 0).min(limit))
                    .fold(current, usize::max);
                if is_view_like_target(&graph.nodes[id].target) {
                    if let Some(input_id) = graph.nodes[id].first_node_arg() {
                        propagated = propagated.max(spatial_rank(graph, input_id, 0).min(limit));
                    }
                }
                if propagated != current {
                    modified |= set_spatial_rank(graph, id, propagated);
                    changed = true;
                }
            }
        }
        modified
    }

    fn annotate_dim_orders(&self, graph: &mut IrGraph, is_control_flow_gm: bool) -> bool {
        let mut modified = false;
        for id in graph.live_node_ids() {
            let Some(val) = graph.nodes[id].meta.tensor_facts() else {
                continue;
            };
            let rank = val.rank();
            let sr = spatial_rank(graph, id, 0);

            if graph.nodes[id].op == NodeOp::Output
                && !is_control_flow_gm
                && graph.nodes[id].meta.original_dim_orders().is_none()
            {
                let orders: Vec<Vec<usize>> = output_node_ids(graph, id)
                    .into_iter()
                    .filter_map(|output_id| {
                        graph.nodes[output_id]
                            .meta
                            .val_dim_order()
                            .map(|order| order.to_vec())
                    })
                    .collect();
                if !orders.is_empty() {
                    graph.nodes[id].meta.set_original_dim_orders(orders);
                    modified = true;
                }
            }

            let dim_order = if is_input_placeholder(graph, id) {
                public_input_dim_order(graph, id, rank, sr)
            } else if is_constant_placeholder(graph, id) && !output_is_physical(graph, id) {
                (0..rank).collect::<Vec<_>>()
            } else if graph.nodes[id].meta.extra_bool(RETAB_KEEP_IDENTITY_PERMUTE) == Some(true) {
                dim_order_or_identity(graph.nodes[id].meta.tosa_dim_order(), rank)
            } else if graph.nodes[id]
                .meta
                .extra_bool("retab_force_identity_dim_order")
                == Some(true)
            {
                (0..rank).collect::<Vec<_>>()
            } else if is_logical_layout_boundary(graph, id) {
                (0..rank).collect::<Vec<_>>()
            } else if graph.nodes[id].op == NodeOp::Output {
                if is_control_flow_gm && rank >= 4 {
                    channels_last_order(rank, sr)
                } else {
                    (0..rank).collect::<Vec<_>>()
                }
            } else if rank == 4 && is_transpose_conv2d_weight(graph, id) {
                vec![1, 2, 3, 0]
            } else if rank >= 4 && value_uses_physical_layout(graph, id) {
                channels_last_order(rank, sr)
            } else {
                (0..rank).collect::<Vec<_>>()
            };

            if graph.nodes[id].meta.tosa_dim_order() != Some(dim_order.as_slice()) {
                graph.nodes[id].meta.set_tosa_dim_order(dim_order);
                modified = true;
            }
        }
        modified
    }

    fn mark_cond_output_getitems_logical(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_cond_output_getitem(graph, id) {
                continue;
            }
            let Some(rank) = graph.nodes[id].meta.rank() else {
                continue;
            };
            if rank < 4 {
                continue;
            }
            let identity: Vec<usize> = (0..rank).collect();
            if graph.nodes[id].meta.tosa_dim_order() != Some(identity.as_slice()) {
                graph.nodes[id].meta.set_tosa_dim_order(identity);
                modified = true;
            }
            if graph.nodes[id]
                .meta
                .extra_bool("retab_logical_layout_boundary")
                != Some(true)
            {
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                modified = true;
            }
        }
        modified
    }

    fn mark_constant_pads_logical(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_pad_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(rank) = graph.nodes[id].meta.rank() else {
                continue;
            };
            if rank < 4 {
                continue;
            }
            if graph.nodes[id]
                .meta
                .extra_bool("retab_logical_layout_boundary")
                != Some(true)
            {
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                modified = true;
            }
        }
        modified
    }

    fn insert_view_boundary_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut constant_memo = HashMap::new();
        for id in graph.live_call_function_ids() {
            if !is_view_like_target(&graph.nodes[id].target) {
                continue;
            }
            if is_constant_subgraph(graph, id, &mut constant_memo) {
                continue;
            }

            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(input_shape) = graph.node_shape_vec(input_id) else {
                continue;
            };
            let Some(output_shape) = graph.node_shape_vec(id) else {
                continue;
            };
            let input_rank = input_shape.len();
            let output_rank = output_shape.len();
            let input_spatial_rank = spatial_rank(graph, input_id, 2);
            let output_spatial_rank = spatial_rank(graph, id, 2);
            if input_rank > 4
                && output_rank == 4
                && is_permute_or_noop_wrapped_permute(graph, input_id)
            {
                graph.nodes[id]
                    .meta
                    .set_tosa_dim_order((0..output_rank).collect());
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                modified = true;
            }
            let channel_reshape = is_channel_reshape(
                &input_shape,
                &output_shape,
                input_spatial_rank,
                output_spatial_rank,
            );
            let materialized_channel_reshape = channel_reshape;

            let input_memory_format_differs =
                memory_format_differs(&input_shape, input_spatial_rank);
            // Only undo channels-last layout before a view if the producer is
            // already physical; logical q/dq chains must not be permuted here.
            let input_emits_physical_layout =
                value_emits_physical_layout(graph, input_id, input_rank, input_spatial_rank);
            let needs_input_layout_boundary = input_emits_physical_layout
                && !is_permute_target(&graph.nodes[input_id].target)
                && (materialized_channel_reshape
                    || ((input_rank >= 4 && output_rank < 4) && input_memory_format_differs));
            if needs_input_layout_boundary {
                let perm = channels_last_inverse_order(input_rank, input_spatial_rank);
                insert_transpose_before(
                    graph,
                    id,
                    input_id,
                    format!("{}_input_layout", graph.nodes[id].name),
                    &perm,
                );
                graph.nodes[id]
                    .meta
                    .set_tosa_dim_order((0..output_rank).collect());
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                modified = true;
            }

            let output_memory_format_differs =
                memory_format_differs(&output_shape, output_spatial_rank);
            if output_is_physical(graph, id)
                && (materialized_channel_reshape
                    || ((input_rank < 4 && output_rank >= 4) && output_memory_format_differs))
            {
                if materialized_channel_reshape
                    && output_rank > 5
                    && graph.single_live_user(id).is_some_and(|user_id| {
                        is_view_like_target(&graph.nodes[user_id].target)
                            || is_permute_target(&graph.nodes[user_id].target)
                    })
                {
                    continue;
                }
                let perm = channels_last_order(output_rank, output_spatial_rank);
                insert_transpose_after(
                    graph,
                    id,
                    format!("{}_output_layout", graph.nodes[id].name),
                    &perm,
                );
                modified = true;
            }
        }
        modified
    }

    fn insert_logical_boundary_output_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_logical_layout_boundary(graph, id) || has_layout_output_user(graph, id) {
                continue;
            }
            let Some(shape) = graph.node_shape(id) else {
                continue;
            };
            let rank = shape.len();
            if rank < 4 || !output_is_physical(graph, id) {
                continue;
            }
            let identity: Vec<usize> = (0..rank).collect();
            if graph.nodes[id].meta.tosa_dim_order() != Some(identity.as_slice()) {
                continue;
            }
            let perm = channels_last_order(rank, spatial_rank(graph, id, 2));
            insert_transpose_after(
                graph,
                id,
                format!("{}_output_layout", graph.nodes[id].name),
                &perm,
            );
            modified = true;
        }
        modified
    }

    fn insert_logical_boundary_input_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_logical_layout_boundary(graph, id) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            if is_tosa_transpose_target(&graph.nodes[input_id].target) {
                continue;
            }
            let Some(input_shape) = graph.node_shape(input_id) else {
                continue;
            };
            let rank = input_shape.len();
            if rank < 4 {
                continue;
            }
            if rank > 4 && is_permute_or_noop_wrapped_permute(graph, input_id) {
                continue;
            }
            let input_order =
                dim_order_or_identity(graph.nodes[input_id].meta.tosa_dim_order(), rank);
            if identity_permutation(&input_order) {
                continue;
            }
            let perm = inverse_order(&input_order);
            insert_transpose_before(
                graph,
                id,
                input_id,
                format!("{}_input_layout", graph.nodes[id].name),
                &perm,
            );
            modified = true;
        }
        modified
    }

    fn absorb_nchw_permute_before_pool(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].alive || !is_permute_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(perm) = permute_axes(graph, id) else {
                continue;
            };
            if perm != [0, 3, 1, 2] {
                continue;
            }
            let Some(source_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(user_id) = graph.single_live_user(id) else {
                continue;
            };
            let user_target = &graph.nodes[user_id].target;
            if !is_plain_avg_pool2d_target(user_target) && !is_max_pool2d_target(user_target) {
                continue;
            }
            let Some(source_shape) = graph.node_shape(source_id) else {
                continue;
            };
            let Some(output_shape) = graph.node_shape(id) else {
                continue;
            };
            if source_shape.len() != 4
                || output_shape.len() != 4
                || source_shape[0] != output_shape[0]
                || source_shape[1] != output_shape[2]
                || source_shape[2] != output_shape[3]
                || source_shape[3] != output_shape[1]
            {
                continue;
            }

            graph.replace_input_use(user_id, id, source_id);
            if graph.live_user_ids(id).is_empty() {
                graph.erase(id);
            }
            modified = true;
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn absorb_physical_permute_before_binary(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].alive || !is_permute_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(perm) = permute_axes(graph, id) else {
                continue;
            };
            if identity_permutation(&perm) {
                continue;
            }
            let Some(source_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(source_shape) = physical_shape_for_node(graph, source_id) else {
                continue;
            };

            for rewrite in collect_physical_permute_binary_rewrites(graph, id) {
                modified |=
                    apply_physical_permute_binary_rewrite(graph, rewrite, source_id, &source_shape);
            }

            if graph.live_user_ids(id).is_empty() {
                graph.erase(id);
            }
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn insert_physical_transparent_input_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut boundaries: HashMap<(NodeId, Vec<usize>), NodeId> = HashMap::new();
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].alive || !is_slice_tensor_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(required_order) = graph.nodes[id].meta.cloned_tosa_dim_order() else {
                continue;
            };
            let rank = required_order.len();
            if rank < 4 || identity_permutation(&required_order) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            if is_tosa_transpose_target(&graph.nodes[input_id].target)
                || is_constant_placeholder(graph, input_id)
            {
                continue;
            }
            let input_order =
                dim_order_or_identity(graph.nodes[input_id].meta.tosa_dim_order(), rank);
            if input_order == required_order {
                continue;
            }
            let Some(perm) = physical_order_perm(&input_order, &required_order) else {
                continue;
            };
            let boundary_id = *boundaries
                .entry((input_id, perm.clone()))
                .or_insert_with(|| {
                    let name = format!("{}_input_layout", graph.nodes[input_id].name);
                    let transpose_id = add_transpose(graph, input_id, name, &perm);
                    graph.anchor_after(transpose_id, input_id);
                    transpose_id
                });
            graph.replace_input_use(id, input_id, boundary_id);
            modified = true;
        }
        modified
    }

    fn cancel_inverse_layout_transpose_pairs(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for first_id in graph.live_call_function_ids() {
            if !graph.nodes[first_id].alive
                || !graph.nodes[first_id].name.contains("_output_layout")
            {
                continue;
            }
            let Some(input_id) = graph.nodes[first_id].first_node_arg() else {
                continue;
            };
            let Some(first_perm) = transpose_perm(graph, first_id) else {
                continue;
            };

            let mut cancelled_for_first = false;
            let candidate_users = graph.live_user_ids(first_id);
            for second_id in candidate_users {
                if !graph.nodes[second_id].alive
                    || !graph.nodes[second_id].name.contains("_layout")
                    || graph.nodes[second_id].first_node_arg() != Some(first_id)
                {
                    continue;
                }
                let Some(second_perm) = transpose_perm(graph, second_id) else {
                    continue;
                };
                if !inverse_permutations(&first_perm, &second_perm) {
                    continue;
                }
                if graph
                    .node_shape(input_id)
                    .zip(graph.node_shape(second_id))
                    .is_some_and(|(input_shape, second_shape)| input_shape != second_shape)
                {
                    continue;
                }

                let second_users = graph.live_user_ids(second_id);
                if second_users.is_empty() {
                    continue;
                }
                for user_id in second_users {
                    graph.replace_input_use(user_id, second_id, input_id);
                }
                graph.erase(second_id);
                modified = true;
                cancelled_for_first = true;
            }

            if cancelled_for_first && graph.live_user_ids(first_id).is_empty() {
                graph.erase(first_id);
            }
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }

    fn insert_input_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_node_ids() {
            if !is_input_placeholder(graph, id) {
                continue;
            }
            if graph.nodes[id]
                .meta
                .extra_bool("arm_control_flow_placeholder")
                == Some(true)
            {
                continue;
            }
            let Some(shape) = graph.node_shape(id) else {
                continue;
            };
            let rank = shape.len();
            if rank < 4 {
                continue;
            }
            let perm = channels_last_order(rank, spatial_rank(graph, id, 2));
            if !output_is_physical(graph, id) {
                continue;
            }
            if graph.nodes[id].meta.tosa_dim_order() == Some(perm.as_slice()) {
                continue;
            }
            insert_transpose_after(
                graph,
                id,
                format!("{}_input_layout", graph.nodes[id].name),
                &perm,
            );
            modified = true;
        }
        modified
    }

    fn insert_output_transposes(&self, graph: &mut IrGraph, is_control_flow_gm: bool) -> bool {
        if is_control_flow_gm {
            return false;
        }
        let mut modified = false;
        for output_id in graph.live_output_ids() {
            let outputs = output_node_ids(graph, output_id);
            let original_dim_orders = graph.nodes[output_id]
                .meta
                .original_dim_orders()
                .map(|orders| orders.to_vec());
            for (index, output_arg) in outputs.into_iter().enumerate() {
                if !graph.nodes[output_arg].alive
                    || graph.nodes[output_arg]
                        .meta
                        .extra_bool("arm_control_flow_parent_output_user")
                        == Some(true)
                    || is_cond_output_getitem(graph, output_arg)
                {
                    continue;
                }
                let Some(shape) = graph.node_shape(output_arg) else {
                    continue;
                };
                let rank = shape.len();
                if rank < 4 {
                    continue;
                }
                let force_logical_output = graph.nodes[output_arg]
                    .meta
                    .extra_bool("retab_force_logical_graph_output")
                    == Some(true);
                if force_logical_output {
                    let channels_last =
                        channels_last_order(rank, spatial_rank(graph, output_arg, 2));
                    let is_originally_channels_last = original_dim_orders
                        .as_ref()
                        .and_then(|orders| orders.get(index))
                        .map(|order| *order == channels_last)
                        .unwrap_or(false);
                    if !is_originally_channels_last {
                        continue;
                    }
                }
                if !value_has_physical_origin(graph, output_arg) && !force_logical_output {
                    continue;
                }
                if is_permute_target(&graph.nodes[output_arg].target)
                    && !output_is_physical(graph, output_arg)
                {
                    continue;
                }
                let channels_last = channels_last_order(rank, spatial_rank(graph, output_arg, 2));
                let is_originally_channels_last = original_dim_orders
                    .as_ref()
                    .and_then(|orders| orders.get(index))
                    .map(|order| *order == channels_last)
                    .unwrap_or(false);
                if is_originally_channels_last && !force_logical_output {
                    continue;
                }
                let perm = if force_logical_output && is_originally_channels_last {
                    channels_last_order(rank, spatial_rank(graph, output_arg, 2))
                } else {
                    channels_last_inverse_order(rank, spatial_rank(graph, output_arg, 2))
                };
                let logical_val =
                    graph.nodes[output_arg]
                        .meta
                        .tensor_facts()
                        .cloned()
                        .map(|mut val| {
                            let identity = (0..rank).collect::<Vec<_>>();
                            if graph.nodes[output_arg]
                                .meta
                                .tosa_dim_order()
                                .is_some_and(|order| order != identity.as_slice())
                            {
                                val.shape = permuted_shape(&val.shape, &perm);
                            }
                            val
                        });
                let transpose_id = insert_transpose_before(
                    graph,
                    output_id,
                    output_arg,
                    format!("{}_output_layout", graph.nodes[output_arg].name),
                    &perm,
                );
                if let Some(val) = logical_val {
                    graph.nodes[transpose_id].meta.set_val(val);
                }
                modified = true;
            }
        }
        modified
    }

    fn remap_permute_args(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_permute_target(&graph.nodes[id].target) {
                continue;
            }
            if is_logical_layout_boundary(graph, id) {
                continue;
            }
            if graph.nodes[id].meta.extra_bool(RETAB_PHYSICAL_PERMUTE_ARGS) == Some(true) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(input_shape) = graph.node_shape(input_id) else {
                continue;
            };
            let input_rank = input_shape.len();
            if input_rank < 4 {
                continue;
            }
            if view_will_materialize_input_layout_boundary(graph, input_id) {
                continue;
            }

            let expected_cl = channels_last_order(input_rank, spatial_rank(graph, input_id, 2));
            let input_is_nhwc = graph.nodes[input_id]
                .meta
                .tosa_dim_order()
                .map(|order| order == expected_cl.as_slice())
                .unwrap_or(false)
                || is_tosa_transpose_target(&graph.nodes[input_id].target);
            if !input_is_nhwc {
                continue;
            }

            let Some(user_perm) = graph.nodes[id]
                .args
                .get(1)
                .and_then(|arg| arg.as_int_list())
            else {
                continue;
            };
            if user_perm.len() != input_rank {
                continue;
            }

            let allow_physical_identity = graph.nodes[id]
                .meta
                .extra_bool("retab_allow_physical_identity_permute")
                == Some(true);
            let input_order = graph.nodes[input_id]
                .meta
                .cloned_tosa_dim_order()
                .or_else(|| transpose_perm(graph, input_id))
                .unwrap_or_else(|| {
                    channels_last_order(input_rank, spatial_rank(graph, input_id, 2))
                });
            if should_materialize_logical_spatial_permute(
                graph,
                id,
                input_id,
                &user_perm,
                &input_order,
            ) {
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                graph.nodes[id]
                    .meta
                    .set_tosa_dim_order((0..input_rank).collect());
                modified = true;
                continue;
            }
            let input_order_inv = inverse_order(&input_order);
            // Input is physically NHWC (transposed by f). Compose f on the output side only when
            // the permute's output is itself consumed in physical layout
            // (input_order^-1 o user_perm o output_order);
            // when the output is logical (reaches the graph boundary through only q/dq and
            // layout-neutral ops) use identity as output_order so an explicit rank-4 layout
            // permute like [0,2,3,1] is not double-transposed.
            let physical_output = output_is_physical(graph, id);
            let output_order = if physical_output {
                dim_order_or_identity(graph.nodes[id].meta.tosa_dim_order(), input_rank)
            } else {
                (0..input_rank).collect()
            };
            let physical_input_order =
                channels_last_order(input_rank, spatial_rank(graph, input_id, 2));
            let physical_identity_layout_permute =
                is_physical_identity_layout_permute(graph, id, input_id, &user_perm, &input_order);
            let storage_preserving_output_order = physical_storage_preserving_output_order(
                graph,
                id,
                input_id,
                &user_perm,
                &input_order,
            );
            let remapped = if storage_preserving_output_order.is_some()
                || physical_identity_layout_permute
                || (allow_physical_identity
                    && user_perm
                        .iter()
                        .map(|&axis| axis as usize)
                        .eq(physical_input_order.iter().copied()))
            {
                (0..input_rank).map(|axis| axis as i64).collect()
            } else {
                let mut remapped = vec![0i64; input_rank];
                for j in 0..input_rank {
                    let logical_output_axis = output_order[j];
                    let user_mapped = user_perm[logical_output_axis] as usize;
                    if user_mapped >= input_order_inv.len() {
                        continue;
                    }
                    remapped[j] = input_order_inv[user_mapped] as i64;
                }
                remapped
            };

            if remapped == user_perm {
                continue;
            }
            let remapped_identity = remapped
                .iter()
                .enumerate()
                .all(|(idx, &axis)| axis == idx as i64);
            let original_identity = user_perm
                .iter()
                .enumerate()
                .all(|(idx, &axis)| axis == idx as i64);
            graph.set_arg(id, 1, i64_list(&remapped));
            if let Some(order) = storage_preserving_output_order {
                graph.nodes[id].meta.set_tosa_dim_order(order);
                graph.nodes[id]
                    .meta
                    .set_extra(RETAB_KEEP_IDENTITY_PERMUTE, MetaValue::Bool(true));
            }
            graph.nodes[id]
                .meta
                .set_extra(RETAB_PHYSICAL_PERMUTE_ARGS, MetaValue::Bool(true));
            if remapped_identity
                && !original_identity
                && !physical_output
                && !allow_physical_identity
                && !physical_identity_layout_permute
            {
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_non_singleton_permute", MetaValue::Bool(true));
                graph.nodes[id]
                    .meta
                    .set_extra("retab_logical_layout_boundary", MetaValue::Bool(true));
                graph.nodes[id]
                    .meta
                    .set_tosa_dim_order((0..input_rank).collect());
            }
            modified = true;
        }
        modified
    }

    fn remove_identity_permutes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].alive || !is_permute_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id]
                .meta
                .extra_bool("retab_logical_non_singleton_permute")
                == Some(true)
                || graph.nodes[id].meta.extra_bool(RETAB_KEEP_IDENTITY_PERMUTE) == Some(true)
            {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };
            let Some(perm) = graph.nodes[id].args.get(1).and_then(IrArg::as_int_list) else {
                continue;
            };
            if perm
                .iter()
                .enumerate()
                .all(|(idx, &axis)| axis == idx as i64)
            {
                graph.replace_and_erase(id, input_id);
                modified = true;
            }
        }
        if modified {
            graph.rebuild_users();
        }
        modified
    }
}

impl IrPass for IrToTosaMemoryFormatPass {
    fn name(&self) -> &str {
        "IrToTosaMemoryFormat"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        use std::time::Instant;

        fn run_phase(
            graph: &mut IrGraph,
            profile: bool,
            name: &str,
            mut phase: impl FnMut(&mut IrGraph) -> bool,
        ) -> bool {
            let started = Instant::now();
            let modified = phase(graph);
            if profile {
                eprintln!(
                    "[RETAB][layout-profile] phase={} modified={} nodes={} time_s={:.6}",
                    name,
                    modified,
                    graph.live_node_count(),
                    started.elapsed().as_secs_f64()
                );
            }
            modified
        }

        let is_control_flow_gm = is_control_flow_submodule(graph);
        let mut modified = false;
        let profile = crate::config::profile_passes_enabled();
        modified |= run_phase(graph, profile, "collapse_inverse_view_pairs", |graph| {
            self.collapse_inverse_view_pairs(graph)
        });
        modified |= run_phase(graph, profile, "compose_adjacent_permutes", |graph| {
            self.compose_adjacent_permutes(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "collapse_view_inverse_permute_view_chains",
            |graph| self.collapse_view_inverse_permute_view_chains(graph),
        );
        modified |= run_phase(graph, profile, "materialize_dim_order_copies", |graph| {
            self.materialize_dim_order_copies(graph)
        });
        modified |= run_phase(graph, profile, "annotate_spatial_ranks", |graph| {
            self.annotate_spatial_ranks(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "propagate_layout_neutral_spatial_ranks",
            |graph| self.propagate_layout_neutral_spatial_ranks(graph),
        );
        modified |= run_phase(graph, profile, "propagate_spatial_ranks", |graph| {
            self.propagate_spatial_ranks(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "mark_cond_output_getitems_logical",
            |graph| self.mark_cond_output_getitems_logical(graph),
        );
        modified |= run_phase(graph, profile, "mark_constant_pads_logical", |graph| {
            self.mark_constant_pads_logical(graph)
        });
        modified |= run_phase(graph, profile, "annotate_dim_orders", |graph| {
            self.annotate_dim_orders(graph, is_control_flow_gm)
        });
        modified |= run_phase(graph, profile, "remap_permute_args", |graph| {
            self.remap_permute_args(graph)
        });
        modified |= run_phase(graph, profile, "annotate_dim_orders", |graph| {
            self.annotate_dim_orders(graph, is_control_flow_gm)
        });
        modified |= run_phase(
            graph,
            profile,
            "collapse_layout_neutral_inverse_permutes",
            |graph| self.collapse_layout_neutral_inverse_permutes(graph),
        );
        modified |= run_phase(graph, profile, "insert_view_boundary_transposes", |graph| {
            self.insert_view_boundary_transposes(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "insert_logical_boundary_input_transposes",
            |graph| self.insert_logical_boundary_input_transposes(graph),
        );
        modified |= run_phase(
            graph,
            profile,
            "insert_logical_boundary_output_transposes",
            |graph| self.insert_logical_boundary_output_transposes(graph),
        );
        modified |= run_phase(graph, profile, "insert_input_transposes", |graph| {
            self.insert_input_transposes(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "insert_physical_transparent_input_transposes",
            |graph| self.insert_physical_transparent_input_transposes(graph),
        );
        modified |= run_phase(graph, profile, "insert_output_transposes", |graph| {
            self.insert_output_transposes(graph, is_control_flow_gm)
        });
        modified |= run_phase(
            graph,
            profile,
            "cancel_inverse_layout_transpose_pairs",
            |graph| self.cancel_inverse_layout_transpose_pairs(graph),
        );
        modified |= run_phase(graph, profile, "remap_permute_args", |graph| {
            self.remap_permute_args(graph)
        });
        modified |= run_phase(
            graph,
            profile,
            "absorb_physical_permute_before_binary",
            |graph| self.absorb_physical_permute_before_binary(graph),
        );
        modified |= run_phase(graph, profile, "remove_identity_permutes", |graph| {
            self.remove_identity_permutes(graph)
        });
        modified |= run_phase(graph, profile, "absorb_nchw_permute_before_pool", |graph| {
            self.absorb_nchw_permute_before_pool(graph)
        });

        if crate::config::debug_layout_enabled() {
            for id in graph.live_node_ids() {
                let node = &graph.nodes[id];
                let Some(val) = node.meta.tensor_facts() else {
                    continue;
                };
                eprintln!(
                    "[LayoutDebug] {} target={} shape={:?} sr={:?} dim_order={:?}",
                    node.name,
                    node.target,
                    val.shape,
                    node.meta.tosa_spatial_rank(),
                    node.meta.cloned_tosa_dim_order(),
                );
            }
        }

        if modified {
            log_line(format_args!(
                "[IrToTosaMemoryFormat] annotated layout and inserted boundary transposes"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_node(
        graph: &mut IrGraph,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
    ) -> NodeId {
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            target.to_string(),
            args,
            NodeMeta::tensor(shape, "torch.int8"),
        )
    }

    fn add_transpose_node(
        graph: &mut IrGraph,
        name: &str,
        input: NodeId,
        perm: &[i64],
        shape: &[i64],
    ) -> NodeId {
        add_node(
            graph,
            name,
            "backend.tosa.TRANSPOSE.default",
            vec![IrArg::Node(input), i64_list(perm)],
            shape,
        )
    }

    #[test]
    fn packed_rank5_tensors_do_not_require_physical_3d_layout() {
        assert!(!memory_format_differs(&[1, 2, 32, 1, 48], 2));
        assert!(memory_format_differs(&[1, 32, 2, 1, 48], 3));
        assert!(!is_channel_reshape(
            &[1, 32, 1, 48],
            &[1, 1, 32, 1, 48],
            2,
            2
        ));
        assert!(is_channel_reshape(
            &[1, 32, 1, 48],
            &[1, 1, 32, 1, 48],
            2,
            3
        ));
    }

    #[test]
    fn runtime_rank4_input_feeding_conv_gets_explicit_input_layout_transpose() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 224, 224], "torch.int8"),
        );
        let conv = add_node(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![IrArg::Node(input)],
            &[1, 32, 112, 112],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));
        assert!(pass.insert_input_transposes(&mut graph));

        assert_eq!(
            graph.nodes[input].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
        let transpose = graph.nodes[conv].first_node_arg().unwrap();
        assert!(graph.nodes[transpose].name.contains("x_input_layout"));
        assert_eq!(
            graph.nodes[transpose].meta.tosa_dim_order(),
            Some(&[0, 2, 3, 1][..])
        );
    }

    #[test]
    fn channels_last_runtime_input_feeding_pool_keeps_physical_abi() {
        let mut graph = IrGraph::empty();
        let mut input_meta = NodeMeta::tensor([1, 1280, 7, 7], "torch.int8");
        input_meta.set_val_dim_order(vec![0, 2, 3, 1]);
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            input_meta,
        );
        let pool = add_node(
            &mut graph,
            "avg_pool2d",
            "aten.avg_pool2d.default",
            vec![IrArg::Node(input)],
            &[1, 1280, 1, 1],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));
        assert!(!pass.insert_input_transposes(&mut graph));

        assert_eq!(
            graph.nodes[input].meta.tosa_dim_order(),
            Some(&[0, 2, 3, 1][..])
        );
        assert_eq!(graph.nodes[pool].first_node_arg(), Some(input));
    }

    #[test]
    fn quantize_prelude_chain_propagates_physical_layout_to_binary() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 224, 224], "torch.float32"),
        );
        let round = add_node(
            &mut graph,
            "round",
            "aten.round.default",
            vec![IrArg::Node(input)],
            &[1, 3, 224, 224],
        );
        let zp = graph.add_node(
            NodeOp::Placeholder,
            "zp".to_string(),
            "zp".to_string(),
            Vec::new(),
            NodeMeta::tensor([1], "torch.float32"),
        );
        let add = add_node(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(round), IrArg::Node(zp)],
            &[1, 3, 224, 224],
        );
        let clamp = add_node(
            &mut graph,
            "clamp",
            "aten.clamp.default",
            vec![IrArg::Node(add)],
            &[1, 3, 224, 224],
        );
        let cast = add_node(
            &mut graph,
            "cast",
            "aten._to_copy.default",
            vec![IrArg::Node(clamp)],
            &[1, 3, 224, 224],
        );
        add_node(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![IrArg::Node(cast)],
            &[1, 32, 112, 112],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));

        assert_eq!(
            graph.nodes[add].meta.tosa_dim_order(),
            Some(&[0, 2, 3, 1][..])
        );
        assert_eq!(
            graph.nodes[clamp].meta.tosa_dim_order(),
            Some(&[0, 2, 3, 1][..])
        );
    }

    #[test]
    fn residual_skip_branch_does_not_hide_physical_origin() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 4, 4], "torch.int8"),
        );
        let input_layout = add_transpose_node(
            &mut graph,
            "x_input_layout",
            input,
            &[0, 2, 3, 1],
            &[1, 4, 4, 3],
        );
        let conv = add_node(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![IrArg::Node(input_layout)],
            &[1, 3, 4, 4],
        );
        let add = add_node(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(conv), IrArg::Node(input_layout)],
            &[1, 3, 4, 4],
        );

        assert!(value_has_physical_origin(&graph, add));
    }

    #[test]
    fn scalar_binary_physical_origin_rank4_to_rank3_view_gets_logical_input_boundary() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([4, 4, 32, 32], "torch.float32"),
        );
        let input_layout = add_transpose_node(
            &mut graph,
            "x_input_layout",
            input,
            &[0, 2, 3, 1],
            &[4, 32, 32, 4],
        );
        let conv = add_node(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![IrArg::Node(input_layout)],
            &[4, 4, 32, 32],
        );
        graph.nodes[conv].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
        let scalar = graph.add_node(
            NodeOp::Placeholder,
            "scalar".to_string(),
            "scalar".to_string(),
            Vec::new(),
            NodeMeta::tensor([1], "torch.float32"),
        );
        let div = add_node(
            &mut graph,
            "div",
            "aten.div.Tensor",
            vec![IrArg::Node(conv), IrArg::Node(scalar)],
            &[4, 4, 32, 32],
        );
        graph.nodes[div].meta.set_tosa_dim_order(vec![0, 1, 2, 3]);
        let view = add_node(
            &mut graph,
            "flatten_channels",
            "aten.view_copy.default",
            vec![IrArg::Node(div), i64_list(&[4, 4, 1024])],
            &[4, 4, 1024],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.insert_view_boundary_transposes(&mut graph));

        let boundary = graph.nodes[view].first_node_arg().unwrap();
        assert_ne!(boundary, div);
        assert!(graph.nodes[boundary]
            .name
            .contains("flatten_channels_input_layout"));
        assert_eq!(graph.nodes[boundary].first_node_arg(), Some(div));
        assert_eq!(
            graph.nodes[boundary]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 3, 1, 2])
        );
        assert_eq!(
            graph.nodes[view]
                .meta
                .extra_bool("retab_logical_layout_boundary"),
            Some(true)
        );
    }

    #[test]
    fn semantic_hw_permute_between_convs_stays_physical() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 8, 8], "torch.float32"),
        );
        let conv1 = add_node(
            &mut graph,
            "conv1",
            "aten.convolution.default",
            vec![IrArg::Node(input)],
            &[1, 3, 8, 8],
        );
        let add = add_node(
            &mut graph,
            "residual_add",
            "aten.add.Tensor",
            vec![IrArg::Node(conv1), IrArg::Node(input)],
            &[1, 3, 8, 8],
        );
        let permute = add_node(
            &mut graph,
            "swap_hw",
            "aten.permute_copy.default",
            vec![IrArg::Node(add), i64_list(&[0, 1, 3, 2])],
            &[1, 3, 8, 8],
        );
        let conv2 = add_node(
            &mut graph,
            "conv2",
            "aten.convolution.default",
            vec![IrArg::Node(permute)],
            &[1, 3, 8, 8],
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(conv2)],
            NodeMeta::tensor([1, 3, 8, 8], "torch.float32"),
        );

        assert!(IrToTosaMemoryFormatPass.run(&mut graph));
        assert_eq!(
            graph.nodes[permute]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 3, 2])
        );
        assert_eq!(
            graph.nodes[permute].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
        assert_eq!(
            graph.nodes[permute]
                .meta
                .extra_bool("retab_logical_layout_boundary"),
            Some(true)
        );
    }

    #[test]
    fn pool_data_getitem_output_restores_public_layout() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 8, 16, 16], "torch.int8"),
        );
        let input_layout = add_transpose_node(
            &mut graph,
            "x_input_layout",
            input,
            &[0, 2, 3, 1],
            &[1, 16, 16, 8],
        );
        let pool = add_node(
            &mut graph,
            "max_pool",
            "aten.max_pool2d_with_indices.default",
            vec![IrArg::Node(input_layout)],
            &[1, 8, 8, 8],
        );
        let lhs = add_node(
            &mut graph,
            "getitem",
            "<built-in function getitem>",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            &[1, 8, 8, 8],
        );
        let rhs = add_node(
            &mut graph,
            "getitem_1",
            "<built-in function getitem>",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            &[1, 8, 8, 8],
        );
        let add = add_node(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            &[1, 8, 8, 8],
        );
        let output = graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(add)],
            NodeMeta::tensor([1, 8, 8, 8], "torch.int8"),
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.run(&mut graph));

        let output_arg = graph.nodes[output].first_node_arg().unwrap();
        assert!(graph.nodes[output_arg].name.contains("add_output_layout"));
        assert_eq!(graph.nodes[output_arg].first_node_arg(), Some(add));
    }

    #[test]
    fn packed_rank5_view_does_not_insert_input_layout_transpose() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "tree_output".to_string(),
            "tree_output".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 32, 1, 48], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "packed_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 1, 32, 1, 48])],
            &[1, 1, 32, 1, 48],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        assert!(!pass.propagate_spatial_ranks(&mut graph));
        assert!(pass.annotate_dim_orders(&mut graph, false));
        assert!(!pass.insert_view_boundary_transposes(&mut graph));

        assert_eq!(spatial_rank(&graph, view, 0), 2);
        assert_eq!(graph.nodes[view].first_node_arg(), Some(input));
        assert_eq!(
            graph.nodes[view].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3, 4][..])
        );
        assert!(graph
            .nodes
            .iter()
            .filter(|node| node.alive)
            .all(|node| !node.name.contains("packed_view_input_layout")));
    }

    #[test]
    fn packed_rank5_channel_split_view_gets_logical_reshape_boundary() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "tree_output".to_string(),
            "tree_output".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 64, 1, 48], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "packed_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 2, 32, 1, 48])],
            &[1, 2, 32, 1, 48],
        );
        let user = add_node(
            &mut graph,
            "packed_user",
            "backend.tosa.SLICE.default",
            vec![IrArg::Node(view)],
            &[1, 1, 32, 1, 48],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        assert!(pass.propagate_layout_neutral_spatial_ranks(&mut graph));
        assert!(!pass.propagate_spatial_ranks(&mut graph));
        assert!(pass.annotate_dim_orders(&mut graph, false));
        assert!(pass.insert_view_boundary_transposes(&mut graph));

        let input_layout = graph.nodes[view].first_node_arg().unwrap();
        assert_eq!(input_layout, input);
        assert!(graph
            .nodes
            .iter()
            .filter(|node| node.alive)
            .all(|node| !node.name.contains("packed_view_input_layout")));

        let output_layout = graph.nodes[user].first_node_arg().unwrap();
        assert_ne!(output_layout, view);
        assert!(graph.nodes[output_layout]
            .name
            .contains("packed_view_output_layout"));
        assert_eq!(
            graph.nodes[output_layout]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 3, 4, 2])
        );
        assert_eq!(
            graph.nodes[view].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3, 4][..])
        );
    }

    #[test]
    fn layout_neutral_rank5_chain_inherits_2d_spatial_rank() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 64, 1, 48], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "packed_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 2, 32, 1, 48])],
            &[1, 2, 32, 1, 48],
        );
        let sum = add_node(
            &mut graph,
            "packed_sum",
            "backend.tosa.REDUCE_SUM.default",
            vec![IrArg::Node(view), i64_list(&[2]), IrArg::Bool(true)],
            &[1, 2, 1, 1, 48],
        );
        let scalar = graph.add_node(
            NodeOp::Placeholder,
            "broadcast_scalar".to_string(),
            "broadcast_scalar".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 1, 1, 1, 1], "torch.int8"),
        );
        let mul = add_node(
            &mut graph,
            "packed_mul",
            "backend.tosa.MUL.default",
            vec![IrArg::Node(sum), IrArg::Node(scalar)],
            &[1, 2, 1, 1, 48],
        );

        assert!(IrToTosaMemoryFormatPass.annotate_spatial_ranks(&mut graph));
        assert!(IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph));
        assert!(!IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph));
        assert_eq!(spatial_rank(&graph, view, 0), 2);
        assert_eq!(spatial_rank(&graph, sum, 0), 2);
        assert_eq!(spatial_rank(&graph, mul, 0), 2);
    }

    #[test]
    fn collapses_inverse_view_pair_before_layout_rewrite() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "grouped".to_string(),
            "grouped".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 2, 32, 1, 48], "torch.int8"),
        );
        let flattened = add_node(
            &mut graph,
            "flatten_members",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[2, 32, 1, 48])],
            &[2, 32, 1, 48],
        );
        let restored = add_node(
            &mut graph,
            "restore_members",
            "aten.view_copy.default",
            vec![IrArg::Node(flattened), i64_list(&[1, 2, 32, 1, 48])],
            &[1, 2, 32, 1, 48],
        );
        let sum = add_node(
            &mut graph,
            "member_sum",
            "aten.sum.dim_IntList",
            vec![
                IrArg::Node(restored),
                IrArg::List(vec![IrArg::Int(2)]),
                IrArg::Bool(true),
            ],
            &[1, 2, 1, 1, 48],
        );

        assert!(IrToTosaMemoryFormatPass.collapse_inverse_view_pairs(&mut graph));
        assert!(!graph.nodes[flattened].alive);
        assert!(!graph.nodes[restored].alive);
        assert_eq!(graph.nodes[sum].first_node_arg(), Some(input));
    }

    #[test]
    fn collapses_view_inverse_permute_view_chain_before_layout_rewrite() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "conv2d_out".to_string(),
            "conv2d_out".to_string(),
            Vec::new(),
            NodeMeta::tensor([3, 4, 8, 8], "torch.int8"),
        );
        let expand = add_node(
            &mut graph,
            "expand_to_5d",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 3, 4, 8, 8])],
            &[1, 3, 4, 8, 8],
        );
        let first_permute = add_node(
            &mut graph,
            "channels_before_depth",
            "aten.permute_copy.default",
            vec![IrArg::Node(expand), i64_list(&[0, 2, 1, 3, 4])],
            &[1, 4, 3, 8, 8],
        );
        let clone = add_node(
            &mut graph,
            "clone_dim_order",
            "dim_order_ops._clone_dim_order.default",
            vec![IrArg::Node(first_permute)],
            &[1, 4, 3, 8, 8],
        );
        let second_permute = add_node(
            &mut graph,
            "depth_before_channels",
            "aten.permute_copy.default",
            vec![IrArg::Node(clone), i64_list(&[0, 2, 1, 3, 4])],
            &[1, 3, 4, 8, 8],
        );
        let flatten = add_node(
            &mut graph,
            "flatten_to_4d",
            "aten.view_copy.default",
            vec![IrArg::Node(second_permute), i64_list(&[3, 4, 8, 8])],
            &[3, 4, 8, 8],
        );
        let user = add_node(
            &mut graph,
            "pointwise",
            "aten.convolution.default",
            vec![IrArg::Node(flatten)],
            &[3, 4, 8, 8],
        );

        assert!(IrToTosaMemoryFormatPass.collapse_view_inverse_permute_view_chains(&mut graph));
        assert_eq!(graph.nodes[user].first_node_arg(), Some(input));
        for id in [expand, first_permute, clone, second_permute, flatten] {
            assert!(!graph.nodes[id].alive);
        }
    }

    #[test]
    fn composes_adjacent_permutes_to_one_logical_permute() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([2, 3, 4], "torch.int8"),
        );
        let first = add_node(
            &mut graph,
            "permute_1",
            "aten.permute_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 2, 0])],
            &[3, 4, 2],
        );
        let second = add_node(
            &mut graph,
            "permute_2",
            "aten.permute_copy.default",
            vec![IrArg::Node(first), i64_list(&[2, 0, 1])],
            &[2, 3, 4],
        );
        let third = add_node(
            &mut graph,
            "permute_3",
            "aten.permute_copy.default",
            vec![IrArg::Node(second), i64_list(&[0, 2, 1])],
            &[2, 4, 3],
        );

        assert!(IrToTosaMemoryFormatPass.compose_adjacent_permutes(&mut graph));

        assert!(!graph.nodes[first].alive);
        assert!(!graph.nodes[second].alive);
        assert_eq!(graph.nodes[third].first_node_arg(), Some(input));
        assert_eq!(
            graph.nodes[third].args.get(1).and_then(IrArg::as_int_list),
            Some(vec![0, 2, 1])
        );
        assert_eq!(
            graph
                .live_call_function_ids()
                .into_iter()
                .filter(|&id| is_permute_target(&graph.nodes[id].target))
                .count(),
            1
        );
    }

    #[test]
    fn rank4_reduction_operand_keeps_default_spatial_rank_after_low_rank_view() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "stacked".to_string(),
            "stacked".to_string(),
            Vec::new(),
            NodeMeta::tensor([2, 1, 2], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "mean_input_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 2, 1, 2])],
            &[1, 2, 1, 2],
        );
        let sum = add_node(
            &mut graph,
            "mean_sum",
            "aten.sum.dim_IntList",
            vec![
                IrArg::Node(view),
                IrArg::List(vec![IrArg::Int(1)]),
                IrArg::Bool(true),
            ],
            &[1, 1, 1, 2],
        );

        assert!(IrToTosaMemoryFormatPass.annotate_spatial_ranks(&mut graph));
        assert!(!IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph));
        assert!(IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph));
        assert_eq!(spatial_rank(&graph, view, 0), 2);
        assert_eq!(spatial_rank(&graph, sum, 0), 2);
    }

    #[test]
    fn attention_mask_softmax_chain_stays_logical_layout() {
        let mut graph = IrGraph::empty();
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([16, 16, 64], "torch.float32"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([16, 64, 16], "torch.float32"),
        );
        let logits_3d = graph.add_node(
            NodeOp::CallFunction,
            "logits_bmm".to_string(),
            "aten.bmm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            NodeMeta::tensor([16, 16, 16], "torch.float32"),
        );
        let logits = graph.add_node(
            NodeOp::CallFunction,
            "logits_view".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(logits_3d), i64_list(&[1, 16, 16, 16])],
            NodeMeta::tensor([1, 16, 16, 16], "torch.float32"),
        );
        let eq = graph.add_node(
            NodeOp::CallFunction,
            "logits_eq".to_string(),
            "aten.eq.Scalar".to_string(),
            vec![IrArg::Node(logits), IrArg::Float(f64::NEG_INFINITY)],
            NodeMeta::tensor([1, 16, 16, 16], "torch.bool"),
        );
        let not = graph.add_node(
            NodeOp::CallFunction,
            "logits_not".to_string(),
            "aten.logical_not.default".to_string(),
            vec![IrArg::Node(eq)],
            NodeMeta::tensor([1, 16, 16, 16], "torch.bool"),
        );
        let any = graph.add_node(
            NodeOp::CallFunction,
            "logits_any".to_string(),
            "aten.any.dim".to_string(),
            vec![IrArg::Node(not), IrArg::Int(-1), IrArg::Bool(true)],
            NodeMeta::tensor([1, 16, 16, 1], "torch.bool"),
        );
        let mask = graph.add_node(
            NodeOp::CallFunction,
            "logits_mask".to_string(),
            "aten.logical_not.default".to_string(),
            vec![IrArg::Node(any)],
            NodeMeta::tensor([1, 16, 16, 1], "torch.bool"),
        );
        let softmax = add_node(
            &mut graph,
            "softmax",
            "aten.mul.Tensor",
            vec![IrArg::Node(logits), IrArg::Node(logits)],
            &[1, 16, 16, 16],
        );
        let zero = graph.add_node(
            NodeOp::Placeholder,
            "zero".to_string(),
            "zero".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 16, 16, 16], "torch.float32"),
        );
        let selected = graph.add_node(
            NodeOp::CallFunction,
            "safe_softmax_where".to_string(),
            "aten.where.self".to_string(),
            vec![IrArg::Node(mask), IrArg::Node(zero), IrArg::Node(softmax)],
            NodeMeta::tensor([1, 16, 16, 16], "torch.float32"),
        );
        let selected_3d = graph.add_node(
            NodeOp::CallFunction,
            "selected_view".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(selected), i64_list(&[16, 16, 16])],
            NodeMeta::tensor([16, 16, 16], "torch.float32"),
        );
        add_node(
            &mut graph,
            "attention_out",
            "aten.bmm.default",
            vec![IrArg::Node(selected_3d), IrArg::Node(lhs)],
            &[16, 16, 64],
        );

        let pass = IrToTosaMemoryFormatPass;
        pass.run(&mut graph);

        assert_eq!(
            graph.nodes[logits].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
        assert!(graph
            .nodes
            .iter()
            .filter(|node| node.alive)
            .all(|node| !node.name.contains("logits_view_output_layout")));
    }

    #[test]
    fn conv3d_consumer_raises_layout_neutral_chain_to_3d_spatial_rank() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 64, 1, 48], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "volume_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 32, 2, 1, 48])],
            &[1, 32, 2, 1, 48],
        );
        let mul = add_node(
            &mut graph,
            "volume_mul",
            "aten.mul.Tensor",
            vec![IrArg::Node(view), IrArg::Node(view)],
            &[1, 32, 2, 1, 48],
        );
        let _conv = add_node(
            &mut graph,
            "conv3d",
            "aten.conv3d.default",
            vec![IrArg::Node(mul)],
            &[1, 16, 2, 1, 48],
        );

        assert!(IrToTosaMemoryFormatPass.annotate_spatial_ranks(&mut graph));
        IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph);
        IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph);
        assert_eq!(spatial_rank(&graph, view, 0), 3);
        assert_eq!(spatial_rank(&graph, mul, 0), 3);
    }

    #[test]
    fn ncdhw_rank5_to_rank4_view_materializes_logical_boundary() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 2, 3, 8, 8], "torch.int8"),
        );
        let conv3d = add_node(
            &mut graph,
            "conv3d",
            "aten.convolution.default",
            vec![IrArg::Node(input)],
            &[1, 4, 3, 8, 8],
        );
        let view = add_node(
            &mut graph,
            "conv3d_flatten",
            "aten.view_copy.default",
            vec![IrArg::Node(conv3d), i64_list(&[3, 4, 8, 8])],
            &[3, 4, 8, 8],
        );
        let _conv2d = add_node(
            &mut graph,
            "conv2d",
            "aten.convolution.default",
            vec![IrArg::Node(view)],
            &[3, 4, 8, 8],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));
        assert!(pass.insert_view_boundary_transposes(&mut graph));

        assert_ne!(graph.nodes[view].first_node_arg(), Some(conv3d));
        assert_eq!(
            graph.nodes[view].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
        assert!(graph
            .nodes
            .iter()
            .filter(|node| node.alive)
            .any(|node| node.name.contains("conv3d_flatten_input_layout")));
    }

    #[test]
    fn depth_before_channel_rank5_view_preserves_2d_spatial_rank() {
        let mut graph = IrGraph::empty();
        let source = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([3, 4, 8, 8], "torch.int8"),
        );
        let input = add_node(
            &mut graph,
            "conv2d_out",
            "aten.convolution.default",
            vec![IrArg::Node(source)],
            &[3, 4, 8, 8],
        );
        let view = add_node(
            &mut graph,
            "restore_depth",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 3, 4, 8, 8])],
            &[1, 3, 4, 8, 8],
        );
        let permute = add_node(
            &mut graph,
            "channels_before_depth",
            "aten.permute_copy.default",
            vec![IrArg::Node(view), i64_list(&[0, 2, 1, 3, 4])],
            &[1, 4, 3, 8, 8],
        );
        let _user = add_node(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(permute), IrArg::Node(permute)],
            &[1, 4, 3, 8, 8],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);

        assert_eq!(spatial_rank(&graph, view, 0), 2);
        assert_eq!(
            graph.nodes[view]
                .meta
                .extra_bool("retab_preserve_spatial_rank"),
            Some(true)
        );
    }

    #[test]
    fn qkv_rank4_to_rank5_view_keeps_natural_3d_spatial_rank() {
        let mut graph = IrGraph::empty();
        let input = add_node(
            &mut graph,
            "qkv_pack",
            "aten.view_copy.default",
            Vec::new(),
            &[3, 2, 3, 4],
        );
        let view = add_node(
            &mut graph,
            "qkv_unsqueeze",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 3, 2, 3, 4])],
            &[1, 3, 2, 3, 4],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);

        assert_eq!(spatial_rank(&graph, view, 0), 3);
        assert_eq!(
            graph.nodes[view]
                .meta
                .extra_bool("retab_preserve_spatial_rank"),
            None
        );
    }

    #[test]
    fn rank4_pack_after_low_rank_view_does_not_inherit_conv_layout() {
        let mut graph = IrGraph::empty();
        let source = graph.add_node(
            NodeOp::Placeholder,
            "linear_input".to_string(),
            "linear_input".to_string(),
            Vec::new(),
            NodeMeta::tensor([6, 1, 1, 4], "torch.int8"),
        );
        let conv = add_node(
            &mut graph,
            "linear_conv",
            "aten.convolution.default",
            vec![IrArg::Node(source)],
            &[6, 1, 1, 12],
        );
        let flat = add_node(
            &mut graph,
            "linear_flat",
            "aten.view_copy.default",
            vec![IrArg::Node(conv), i64_list(&[3, 2, 12])],
            &[3, 2, 12],
        );
        let qkv_pack = add_node(
            &mut graph,
            "qkv_pack",
            "aten.view_copy.default",
            vec![IrArg::Node(flat), i64_list(&[3, 3, 4, 2])],
            &[3, 3, 4, 2],
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));

        assert_eq!(
            graph.nodes[qkv_pack].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
    }

    #[test]
    fn select_copy_does_not_force_rank4_pack_physical() {
        let mut graph = IrGraph::empty();
        let qkv_pack = add_node(
            &mut graph,
            "qkv_pack",
            "aten.view_copy.default",
            Vec::new(),
            &[3, 3, 2, 4],
        );
        let select = add_node(
            &mut graph,
            "select_q",
            "aten.select_copy.int",
            vec![IrArg::Node(qkv_pack), IrArg::Int(0), IrArg::Int(0)],
            &[3, 2, 4],
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(select)],
            NodeMeta::default(),
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));

        assert_eq!(
            graph.nodes[qkv_pack].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
    }

    #[test]
    fn depth_before_channel_rank5_to_rank4_view_stays_physical() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "depth_before_channel".to_string(),
            "depth_before_channel".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 4, 8, 8], "torch.int8"),
        );
        graph.nodes[input].meta.set_tosa_spatial_rank(2);
        graph.nodes[input]
            .meta
            .set_tosa_dim_order(vec![0, 1, 3, 4, 2]);
        let view = add_node(
            &mut graph,
            "flatten_depth",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[3, 4, 8, 8])],
            &[3, 4, 8, 8],
        );
        graph.nodes[view].meta.set_tosa_spatial_rank(2);
        graph.nodes[view].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
        let _conv_user = add_node(
            &mut graph,
            "conv2d_user",
            "aten.convolution.default",
            vec![IrArg::Node(view)],
            &[3, 4, 8, 8],
        );

        assert!(!IrToTosaMemoryFormatPass.insert_view_boundary_transposes(&mut graph));
        assert_eq!(graph.nodes[view].first_node_arg(), Some(input));
    }

    #[test]
    fn remaps_rank5_depth_channel_permute_with_distinct_input_output_layouts() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "conv2d_out".to_string(),
            "conv2d_out".to_string(),
            Vec::new(),
            NodeMeta::tensor([3, 4, 8, 8], "torch.int8"),
        );
        let view = add_node(
            &mut graph,
            "restore_depth",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 3, 4, 8, 8])],
            &[1, 3, 4, 8, 8],
        );
        graph.nodes[view].meta.set_tosa_spatial_rank(2);
        graph.nodes[view]
            .meta
            .set_tosa_dim_order(vec![0, 1, 3, 4, 2]);
        let permute = add_node(
            &mut graph,
            "channels_before_depth",
            "aten.permute_copy.default",
            vec![IrArg::Node(view), i64_list(&[0, 2, 1, 3, 4])],
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[permute].meta.set_tosa_spatial_rank(3);
        graph.nodes[permute]
            .meta
            .set_tosa_dim_order(vec![0, 2, 3, 4, 1]);
        let _conv_user = add_node(
            &mut graph,
            "physical_user",
            "aten.convolution.default",
            vec![IrArg::Node(permute)],
            &[1, 4, 3, 8, 8],
        );

        assert!(IrToTosaMemoryFormatPass.remap_permute_args(&mut graph));
        assert_eq!(
            graph.nodes[permute]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 2, 3, 4])
        );
    }

    #[test]
    fn removes_channels_last_restore_permute_before_high_rank_shape_chain() {
        let mut graph = IrGraph::empty();
        let conv = add_node(
            &mut graph,
            "conv2d",
            "aten.conv2d.default",
            Vec::new(),
            &[1, 64, 16, 16],
        );
        graph.nodes[conv].meta.set_tosa_spatial_rank(2);
        graph.nodes[conv].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
        let permute = add_node(
            &mut graph,
            "channels_last",
            "aten.permute.default",
            vec![IrArg::Node(conv), i64_list(&[0, 2, 3, 1])],
            &[1, 16, 16, 64],
        );
        graph.nodes[permute].meta.set_tosa_spatial_rank(2);
        graph.nodes[permute]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3]);
        let view = add_node(
            &mut graph,
            "split_height",
            "aten.reshape.default",
            vec![IrArg::Node(permute), i64_list(&[1, 16, 8, 2, 64])],
            &[1, 16, 8, 2, 64],
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(view)],
            NodeMeta::default(),
        );

        assert!(IrToTosaMemoryFormatPass.remap_permute_args(&mut graph));
        assert_eq!(
            graph.nodes[permute]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 2, 3])
        );
        assert_ne!(
            graph.nodes[permute]
                .meta
                .extra_bool("retab_logical_layout_boundary"),
            Some(true)
        );
        assert!(IrToTosaMemoryFormatPass.remove_identity_permutes(&mut graph));
        assert_eq!(graph.nodes[view].first_node_arg(), Some(conv));
    }

    #[test]
    fn absorbs_rank5_restore_permute_before_physical_binary() {
        let mut graph = IrGraph::empty();
        let conv2d = add_node(
            &mut graph,
            "conv2d_out",
            "aten.convolution.default",
            Vec::new(),
            &[3, 4, 8, 8],
        );
        graph.nodes[conv2d].meta.set_tosa_spatial_rank(2);
        graph.nodes[conv2d]
            .meta
            .set_tosa_dim_order(vec![0, 2, 3, 1]);
        let restore_depth = add_node(
            &mut graph,
            "restore_depth",
            "aten.view_copy.default",
            vec![IrArg::Node(conv2d), i64_list(&[1, 3, 4, 8, 8])],
            &[1, 3, 4, 8, 8],
        );
        graph.nodes[restore_depth].meta.set_tosa_spatial_rank(2);
        graph.nodes[restore_depth]
            .meta
            .set_tosa_dim_order(vec![0, 1, 3, 4, 2]);
        let restore_channels = add_node(
            &mut graph,
            "restore_channels",
            "aten.permute_copy.default",
            vec![IrArg::Node(restore_depth), i64_list(&[0, 2, 1, 3, 4])],
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[restore_channels].meta.set_tosa_spatial_rank(3);
        graph.nodes[restore_channels]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3, 4]);
        let clone = add_node(
            &mut graph,
            "clone_dim_order",
            "dim_order_ops._clone_dim_order.default",
            vec![IrArg::Node(restore_channels)],
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[clone].meta.set_tosa_spatial_rank(3);
        graph.nodes[clone]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3, 4]);
        let rescale = add_node(
            &mut graph,
            "residual_arg0_int32_rescale",
            "backend.tosa.RESCALE.default",
            vec![IrArg::Node(clone)],
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[rescale].meta.set_tosa_spatial_rank(3);
        graph.nodes[rescale]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3, 4]);
        let residual = add_node(
            &mut graph,
            "residual",
            "aten.convolution.default",
            Vec::new(),
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[residual].meta.set_tosa_spatial_rank(3);
        graph.nodes[residual]
            .meta
            .set_tosa_dim_order(vec![0, 2, 3, 4, 1]);
        let add = add_node(
            &mut graph,
            "residual_add",
            "aten.add.Tensor",
            vec![IrArg::Node(rescale), IrArg::Node(residual)],
            &[1, 4, 3, 8, 8],
        );
        graph.nodes[add].meta.set_tosa_spatial_rank(3);
        graph.nodes[add]
            .meta
            .set_tosa_dim_order(vec![0, 2, 3, 4, 1]);

        assert!(IrToTosaMemoryFormatPass.absorb_physical_permute_before_binary(&mut graph));
        assert_eq!(graph.nodes[add].first_node_arg(), Some(rescale));
        assert_eq!(graph.nodes[rescale].first_node_arg(), Some(restore_depth));
        assert_eq!(
            graph.nodes[rescale].meta.tosa_dim_order(),
            Some(&[0, 2, 3, 4, 1][..])
        );
        assert!(!graph.nodes[clone].alive);
        assert!(!graph.nodes[restore_channels].alive);
    }

    #[test]
    fn physical_output_identity_permute_does_not_become_logical_boundary() {
        let mut graph = IrGraph::empty();
        let input = add_node(
            &mut graph,
            "gelu_out",
            "aten.gelu.default",
            Vec::new(),
            &[1, 8, 16, 16],
        );
        graph.nodes[input].meta.set_tosa_spatial_rank(2);
        graph.nodes[input].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
        let permute = add_node(
            &mut graph,
            "semantic_nhwc",
            "aten.permute_copy.default",
            vec![IrArg::Node(input), i64_list(&[0, 2, 3, 1])],
            &[1, 16, 16, 8],
        );
        graph.nodes[permute].meta.set_tosa_spatial_rank(2);
        graph.nodes[permute]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3]);
        let pool = add_node(
            &mut graph,
            "physical_pool",
            "aten.avg_pool2d.default",
            vec![IrArg::Node(permute)],
            &[1, 8, 8, 8],
        );

        assert!(IrToTosaMemoryFormatPass.remap_permute_args(&mut graph));
        assert_eq!(
            graph.nodes[permute]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 2, 3])
        );
        assert_ne!(
            graph.nodes[permute]
                .meta
                .extra_bool("retab_logical_layout_boundary"),
            Some(true)
        );
        assert!(IrToTosaMemoryFormatPass.remove_identity_permutes(&mut graph));
        assert_eq!(graph.nodes[pool].first_node_arg(), Some(input));
    }

    #[test]
    fn layernorm_identity_permute_hint_keeps_reduction_input_physical() {
        let mut graph = IrGraph::empty();
        let input = add_node(
            &mut graph,
            "gelu_out",
            "aten.gelu.default",
            Vec::new(),
            &[1, 8, 16, 16],
        );
        graph.nodes[input].meta.set_tosa_spatial_rank(2);
        graph.nodes[input].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
        let permute = add_node(
            &mut graph,
            "layernorm_input",
            "aten.permute_copy.default",
            vec![IrArg::Node(input), i64_list(&[0, 2, 3, 1])],
            &[1, 16, 16, 8],
        );
        graph.nodes[permute].meta.set_tosa_spatial_rank(2);
        graph.nodes[permute]
            .meta
            .set_tosa_dim_order(vec![0, 1, 2, 3]);
        graph.nodes[permute].meta.set_extra(
            "retab_allow_physical_identity_permute",
            MetaValue::Bool(true),
        );
        let sum = add_node(
            &mut graph,
            "layernorm_mean_sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(permute), i64_list(&[3]), IrArg::Bool(true)],
            &[1, 16, 16, 1],
        );

        assert!(IrToTosaMemoryFormatPass.remap_permute_args(&mut graph));
        assert_eq!(
            graph.nodes[permute]
                .args
                .get(1)
                .and_then(IrArg::as_int_list),
            Some(vec![0, 1, 2, 3])
        );
        assert_ne!(
            graph.nodes[permute]
                .meta
                .extra_bool("retab_logical_layout_boundary"),
            Some(true)
        );
        assert!(IrToTosaMemoryFormatPass.remove_identity_permutes(&mut graph));
        assert_eq!(graph.nodes[sum].first_node_arg(), Some(input));
    }

    #[test]
    fn semantic_output_permute_does_not_inherit_physical_origin() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 2, 3, 4], "torch.float32"),
        );
        graph.nodes[input].meta.set_val_dim_order(vec![0, 2, 3, 1]);
        let permute = add_node(
            &mut graph,
            "semantic_permute",
            "aten.permute_copy.default",
            vec![IrArg::Node(input), i64_list(&[0, 2, 1, 3])],
            &[1, 3, 2, 4],
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(permute)],
            NodeMeta::default(),
        );

        let pass = IrToTosaMemoryFormatPass;
        assert!(pass.annotate_spatial_ranks(&mut graph));
        pass.propagate_layout_neutral_spatial_ranks(&mut graph);
        pass.propagate_spatial_ranks(&mut graph);
        assert!(pass.annotate_dim_orders(&mut graph, false));

        assert_eq!(
            graph.nodes[permute].meta.tosa_dim_order(),
            Some(&[0, 1, 2, 3][..])
        );
    }

    #[test]
    fn rank5_graph_output_keeps_default_3d_spatial_rank() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 4, 5], "torch.float32"),
        );
        let view = add_node(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list(&[1, 1, 4, 5, 3])],
            &[1, 1, 4, 5, 3],
        );
        let mul = add_node(
            &mut graph,
            "mul",
            "aten.mul.Tensor",
            vec![IrArg::Node(view), IrArg::Node(view)],
            &[1, 1, 4, 5, 3],
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(mul)],
            NodeMeta::default(),
        );

        assert!(IrToTosaMemoryFormatPass.annotate_spatial_ranks(&mut graph));
        IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph);
        IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph);
        assert_eq!(spatial_rank(&graph, view, 0), 3);
        assert_eq!(spatial_rank(&graph, mul, 0), 3);
    }

    #[test]
    fn cancels_single_consumer_inverse_layout_transposes() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 2, 32, 1, 48], "torch.int8"),
        );
        let output_layout = add_transpose_node(
            &mut graph,
            "view_output_layout",
            input,
            &[0, 2, 3, 4, 1],
            &[1, 32, 1, 48, 2],
        );
        let input_layout = add_transpose_node(
            &mut graph,
            "next_view_input_layout",
            output_layout,
            &[0, 4, 1, 2, 3],
            &[1, 2, 32, 1, 48],
        );
        let view = add_node(
            &mut graph,
            "next_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input_layout), i64_list(&[1, 2, 32, 48])],
            &[1, 2, 32, 48],
        );

        assert!(IrToTosaMemoryFormatPass.cancel_inverse_layout_transpose_pairs(&mut graph));
        assert!(!graph.nodes[output_layout].alive);
        assert!(!graph.nodes[input_layout].alive);
        assert_eq!(graph.nodes[view].first_node_arg(), Some(input));
    }

    #[test]
    fn keeps_stamped_identity_permute() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 1, 16, 64, 16], "torch.int8"),
        );
        let mut permute_meta = NodeMeta::tensor([1, 1, 16, 64, 16], "torch.int8");
        permute_meta.set_extra("retab_logical_non_singleton_permute", MetaValue::Bool(true));
        let permute = graph.add_permute_copy("permute_1", input, &[0, 1, 2, 3, 4], permute_meta);

        assert!(!IrToTosaMemoryFormatPass.remove_identity_permutes(&mut graph));
        assert!(graph.nodes[permute].alive);
    }

    #[test]
    fn removes_inverse_branch_from_shared_layout_transpose() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 2, 32, 1, 48], "torch.int8"),
        );
        let output_layout = add_transpose_node(
            &mut graph,
            "view_output_layout",
            input,
            &[0, 2, 3, 4, 1],
            &[1, 32, 1, 48, 2],
        );
        let input_layout = add_transpose_node(
            &mut graph,
            "next_view_input_layout",
            output_layout,
            &[0, 4, 1, 2, 3],
            &[1, 2, 32, 1, 48],
        );
        let _view = add_node(
            &mut graph,
            "next_view",
            "aten.view_copy.default",
            vec![IrArg::Node(input_layout), i64_list(&[1, 2, 32, 48])],
            &[1, 2, 32, 48],
        );
        let _shared_user = add_node(
            &mut graph,
            "shared_rescale",
            "backend.tosa.RESCALE.default",
            vec![IrArg::Node(output_layout)],
            &[1, 32, 1, 48, 2],
        );

        assert!(IrToTosaMemoryFormatPass.cancel_inverse_layout_transpose_pairs(&mut graph));
        assert!(graph.nodes[output_layout].alive);
        assert!(!graph.nodes[input_layout].alive);
        assert_eq!(graph.nodes[_view].first_node_arg(), Some(input));
        assert_eq!(
            graph.nodes[_shared_user].first_node_arg(),
            Some(output_layout)
        );
    }
}
