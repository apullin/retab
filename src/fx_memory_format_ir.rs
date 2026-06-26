use std::collections::{HashMap, HashSet};

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::{
    is_add_sub_target, is_boundary_rescale_target, is_constant_only_shape_transform_target,
    is_layout_passthrough_target, is_mean_target, is_mul_target, is_permute_target,
    is_raw_aten_convolution_target, is_slice_tensor_target, is_sum_dim_int_list_target,
    is_tosa_transpose_target, is_transpose_conv2d_target, is_view_like_target,
    is_zero_input_constant_target, table_op_target_kind,
};
use crate::runner_logging::log_line;

pub struct IrToTosaMemoryFormatPass;

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
    let mut inverse = vec![0usize; rank];
    for (idx, &axis) in order.iter().enumerate() {
        inverse[axis] = idx;
    }
    inverse
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

fn is_channel_reshape(input_shape: &[i64], output_shape: &[i64]) -> bool {
    let valid = [4, 5, 6];
    if !valid.contains(&input_shape.len()) || !valid.contains(&output_shape.len()) {
        return false;
    }
    let in_sr = initial_spatial_rank(input_shape.len());
    let out_sr = initial_spatial_rank(output_shape.len());
    let in_ch_idx = input_shape.len().saturating_sub(in_sr + 1);
    let out_ch_idx = output_shape.len().saturating_sub(out_sr + 1);

    let c_old = input_shape.get(in_ch_idx).copied().unwrap_or(1);
    let c_new = output_shape.get(out_ch_idx).copied().unwrap_or(1);
    let n_old: i64 = input_shape[..in_ch_idx].iter().product();
    let n_new: i64 = output_shape[..out_ch_idx].iter().product();

    n_old != n_new || c_old != c_new
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

/// Whether a node's output is consumed in physical (NHWC) layout. False (logical) when the
/// output only reaches the graph Output through quantize/dequantize and layout-neutral ops;
/// true once any non-transparent consumer (conv, pool, another layout op, ...) is found.
fn output_is_physical(graph: &IrGraph, id: NodeId) -> bool {
    let mut stack = graph.live_user_ids(id);
    let mut seen: HashSet<NodeId> = HashSet::new();
    while let Some(uid) = stack.pop() {
        if !seen.insert(uid) {
            continue;
        }
        let node = &graph.nodes[uid];
        if node.op == NodeOp::Output {
            continue;
        }
        if node.is_q_or_dq() {
            stack.extend(graph.live_user_ids(uid));
        } else {
            return true;
        }
    }
    false
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
    let users = graph.live_user_ids(producer_id);
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

fn spatial_rank_follows_inputs(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.op != NodeOp::CallFunction {
        return false;
    }
    node.is_q_or_dq()
        || is_layout_passthrough_target(&node.target)
        || is_slice_tensor_target(&node.target)
        || is_boundary_rescale_target(&node.target)
        || is_add_sub_target(&node.target)
        || is_mul_target(&node.target)
        || is_sum_dim_int_list_target(&node.target)
        || is_mean_target(&node.target)
        || table_op_target_kind(&node.target).is_some()
        || is_tosa_layout_neutral_target(&node.target)
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
    let ranks: Vec<(bool, usize)> = input_ids
        .into_iter()
        .filter(|&input_id| graph.nodes[input_id].alive)
        .filter_map(|input_id| {
            let facts = graph.nodes[input_id].meta.tensor_facts()?;
            let carries_layout = facts.shape.iter().any(|&dim| dim > 1);
            Some((
                carries_layout,
                spatial_rank(graph, input_id, initial_spatial_rank(facts.rank())).min(limit),
            ))
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
        .live_user_ids(id)
        .into_iter()
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

impl IrToTosaMemoryFormatPass {
    fn annotate_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_node_ids() {
            let Some(val) = graph.nodes[id].meta.tensor_facts() else {
                continue;
            };
            let rank = initial_spatial_rank(val.rank());
            modified |= set_spatial_rank(graph, id, rank);
        }
        modified
    }

    fn propagate_layout_neutral_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut changed = true;
        while changed {
            changed = false;
            for id in graph.live_node_ids() {
                if !spatial_rank_follows_inputs(graph, id) {
                    continue;
                }
                let Some(val) = graph.nodes[id].meta.tensor_facts() else {
                    continue;
                };
                let limit = val.rank().saturating_sub(2);
                if limit == 0 {
                    continue;
                }
                let input_floor = direct_input_spatial_ranks(graph, id, limit)
                    .into_iter()
                    .max();
                let Some(input_floor) = input_floor else {
                    continue;
                };
                let propagated = input_floor.max(hard_user_spatial_rank_floor(graph, id, limit));
                if propagated != spatial_rank(graph, id, 0) {
                    modified |= set_spatial_rank(graph, id, propagated);
                    changed = true;
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
                if rank >= 4 {
                    graph.nodes[id]
                        .meta
                        .val_dim_order()
                        .map(|order| order.to_vec())
                        .unwrap_or_else(|| (0..rank).collect::<Vec<_>>())
                } else {
                    // The TOSA runner only permutes inputs for channels-last (NHWC/NNHWC) dim
                    // orders; rank<4 inputs (e.g. a torch.t input with dim_order [1,0]) are fed in
                    // logical layout, so serialize them logically rather than physically permuted.
                    (0..rank).collect::<Vec<_>>()
                }
            } else if graph.nodes[id].op == NodeOp::Output {
                if is_control_flow_gm && rank >= 4 {
                    channels_last_order(rank, sr)
                } else {
                    (0..rank).collect::<Vec<_>>()
                }
            } else if rank == 4 && is_transpose_conv2d_weight(graph, id) {
                vec![1, 2, 3, 0]
            } else if rank >= 4 {
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
            let channel_reshape = is_channel_reshape(&input_shape, &output_shape);

            let input_memory_format_differs =
                memory_format_differs(&input_shape, spatial_rank(graph, input_id, 2));
            if (channel_reshape || (input_rank >= 4 && output_rank < 4))
                && input_memory_format_differs
            {
                let perm =
                    channels_last_inverse_order(input_rank, spatial_rank(graph, input_id, 2));
                insert_transpose_before(
                    graph,
                    id,
                    input_id,
                    format!("{}_input_layout", graph.nodes[id].name),
                    &perm,
                );
                modified = true;
            }

            let output_memory_format_differs =
                memory_format_differs(&output_shape, spatial_rank(graph, id, 2));
            if (channel_reshape || (input_rank < 4 && output_rank >= 4))
                && output_memory_format_differs
            {
                let perm = channels_last_order(output_rank, spatial_rank(graph, id, 2));
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
                    || !graph.nodes[second_id].name.contains("_input_layout")
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
                // Insert the NHWC->logical output transpose unless the model's
                // original output was genuinely channels-last. The previous test
                // (orig == identity) skipped the transpose for ANY non-contiguous
                // original layout — e.g. a permuted output whose meta carries the
                // permute's dim_order in the quantized lane — leaving the output
                // physically NHWC and scrambled.
                let channels_last = channels_last_order(rank, spatial_rank(graph, output_arg, 2));
                let is_originally_channels_last = original_dim_orders
                    .as_ref()
                    .and_then(|orders| orders.get(index))
                    .map(|order| *order == channels_last)
                    .unwrap_or(false);
                if is_originally_channels_last {
                    continue;
                }
                let perm = channels_last_inverse_order(rank, spatial_rank(graph, output_arg, 2));
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

            let f = channels_last_order(input_rank, spatial_rank(graph, input_id, 2));
            let f_inv = channels_last_inverse_order(input_rank, spatial_rank(graph, input_id, 2));
            // Input is physically NHWC (transposed by f). Compose f on the output side only when
            // the permute's output is itself consumed in physical layout (f_inv o user_perm o f);
            // when the output is logical (reaches the graph boundary through only q/dq and
            // layout-neutral ops) use f_inv o user_perm so an explicit rank-4 layout permute like
            // [0,2,3,1] is not double-transposed.
            let physical_output = output_is_physical(graph, id);
            let mut remapped = vec![0i64; input_rank];
            for j in 0..input_rank {
                let nchw_axis = if physical_output { f[j] } else { j };
                let user_mapped = user_perm[nchw_axis] as usize;
                if user_mapped >= f_inv.len() {
                    continue;
                }
                remapped[j] = f_inv[user_mapped] as i64;
            }

            if remapped == user_perm {
                continue;
            }
            graph.set_arg(id, 1, i64_list(&remapped));
            modified = true;
        }
        modified
    }
}

impl IrPass for IrToTosaMemoryFormatPass {
    fn name(&self) -> &str {
        "IrToTosaMemoryFormat"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let is_control_flow_gm = is_control_flow_submodule(graph);
        let mut modified = false;
        modified |= self.annotate_spatial_ranks(graph);
        modified |= self.propagate_layout_neutral_spatial_ranks(graph);
        modified |= self.propagate_spatial_ranks(graph);
        modified |= self.annotate_dim_orders(graph, is_control_flow_gm);
        modified |= self.insert_view_boundary_transposes(graph);
        modified |= self.insert_input_transposes(graph);
        modified |= self.insert_output_transposes(graph, is_control_flow_gm);
        modified |= self.cancel_inverse_layout_transpose_pairs(graph);
        modified |= self.remap_permute_args(graph);

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
        assert!(IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph));
        assert!(IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph));
        assert_eq!(spatial_rank(&graph, view, 0), 3);
        assert_eq!(spatial_rank(&graph, mul, 0), 3);
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
        assert!(IrToTosaMemoryFormatPass.propagate_layout_neutral_spatial_ranks(&mut graph));
        assert!(IrToTosaMemoryFormatPass.propagate_spatial_ranks(&mut graph));
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
