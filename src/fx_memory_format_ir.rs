use std::collections::{HashMap, HashSet};

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta,
    NodeOp,
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

fn memory_format_differs(shape: &[i64], rank: usize) -> bool {
    if rank <= 2 {
        return false;
    }
    let spatial_rank = initial_spatial_rank(rank);
    if spatial_rank == 0 {
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
    if node.op != NodeOp::Placeholder || node.meta.val.is_none() {
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
        if node.target.contains("quantize") {
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

fn is_view_like(target: &str) -> bool {
    target.contains("view_copy")
        || target.contains("flatten")
        || target.contains("reshape")
        || target.contains("aten.view.")
        || target.contains("aten._unsafe_view")
}

fn is_zero_input_constant_target(target: &str) -> bool {
    target.contains("aten.full.default")
        || target.contains("aten.arange")
        || target.contains("aten.linspace")
        || target.contains("aten.eye")
}

fn is_constant_only_shape_target(target: &str) -> bool {
    is_view_like(target)
        || target.contains("aten.slice_copy.Tensor")
        || target.contains("aten.unsqueeze_copy.default")
        || target.contains("aten.permute_copy.default")
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
        } else if is_constant_only_shape_target(&node.target) {
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

fn is_permute(target: &str) -> bool {
    target.contains("permute_copy") || target.contains("aten.permute.")
}

fn is_transposed_aten_convolution(node: &crate::fx_ir::IrNode) -> bool {
    let target = node.target.to_ascii_lowercase();
    if !(target.contains("aten.convolution") || target.contains("edge__ops_aten_convolution")) {
        return false;
    }
    matches!(node.args.get(6), Some(IrArg::Bool(true)))
        || matches!(node.kwargs.get("transposed"), Some(IrArg::Bool(true)))
}

fn is_transpose_conv2d_weight(graph: &IrGraph, id: NodeId) -> bool {
    graph.live_user_ids(id).into_iter().any(|user_id| {
        let user = &graph.nodes[user_id];
        if user.op != NodeOp::CallFunction {
            return false;
        }
        let target = user.target.to_ascii_lowercase();
        if !(target.contains("transpose_conv2d")
            || target.contains("transposeconv2d")
            || is_transposed_aten_convolution(user))
        {
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
    let mut meta = input_meta.clone();
    if let Some(val) = input_meta.val.as_ref() {
        meta.set_val(FakeTensorMeta {
            shape: permuted_shape(&val.shape, perm),
            dtype: val.dtype.clone(),
        });
    }
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
    if graph.nodes[id].meta.tosa_spatial_rank == Some(rank) {
        return false;
    }
    graph.nodes[id].meta.tosa_spatial_rank = Some(rank);
    graph.nodes[id]
        .meta
        .set_extra("tosa_spatial_rank", MetaValue::Int(rank as i64));
    true
}

fn spatial_rank(graph: &IrGraph, id: NodeId, default: usize) -> usize {
    graph.nodes[id].meta.tosa_spatial_rank.unwrap_or(default)
}

impl IrToTosaMemoryFormatPass {
    fn annotate_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in 0..graph.nodes.len() {
            if !graph.nodes[id].alive {
                continue;
            }
            let Some(val) = graph.nodes[id].meta.val.as_ref() else {
                continue;
            };
            let rank = initial_spatial_rank(val.rank());
            modified |= set_spatial_rank(graph, id, rank);
        }
        modified
    }

    fn propagate_spatial_ranks(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut changed = true;
        while changed {
            changed = false;
            for id in (0..graph.nodes.len()).rev() {
                if !graph.nodes[id].alive || graph.nodes[id].meta.val.is_none() {
                    continue;
                }
                let rank = graph.nodes[id]
                    .meta
                    .val
                    .as_ref()
                    .map(|val| val.rank())
                    .unwrap_or(0);
                let limit = rank.saturating_sub(2);
                let current = spatial_rank(graph, id, 0);
                let propagated = graph
                    .users(id)
                    .iter()
                    .filter(|&&user_id| graph.nodes[user_id].alive)
                    .map(|&user_id| spatial_rank(graph, user_id, 0).min(limit))
                    .fold(current, usize::max);
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
        let ids: Vec<_> = (0..graph.nodes.len()).collect();
        for id in ids {
            if !graph.nodes[id].alive {
                continue;
            }
            let Some(val) = graph.nodes[id].meta.val.as_ref() else {
                continue;
            };
            let rank = val.rank();
            let sr = spatial_rank(graph, id, 0);

            if graph.nodes[id].op == NodeOp::Output
                && !is_control_flow_gm
                && graph.nodes[id].meta.original_dim_orders.is_none()
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

            if graph.nodes[id].meta.tosa_dim_order.as_deref() != Some(dim_order.as_slice()) {
                graph.nodes[id].meta.set_tosa_dim_order(dim_order);
                modified = true;
            }
        }
        modified
    }

    fn insert_view_boundary_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<_> = (0..graph.nodes.len()).collect();
        let mut constant_memo = HashMap::new();
        for id in ids {
            if !graph.nodes[id].alive
                || graph.nodes[id].op != NodeOp::CallFunction
                || !is_view_like(&graph.nodes[id].target)
            {
                continue;
            }
            if is_constant_subgraph(graph, id, &mut constant_memo) {
                continue;
            }

            let Some(input_id) = graph.nodes[id].args.first().and_then(|arg| arg.as_node()) else {
                continue;
            };
            let Some(input_shape) = graph.nodes[input_id]
                .meta
                .val_shape()
                .map(|shape| shape.to_vec())
            else {
                continue;
            };
            let Some(output_shape) = graph.nodes[id].meta.val_shape().map(|shape| shape.to_vec())
            else {
                continue;
            };
            let input_rank = input_shape.len();
            let output_rank = output_shape.len();
            let channel_reshape = is_channel_reshape(&input_shape, &output_shape);

            let input_memory_format_differs = memory_format_differs(&input_shape, input_rank);
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

            let output_memory_format_differs = memory_format_differs(&output_shape, output_rank);
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

    fn insert_input_transposes(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<_> = (0..graph.nodes.len()).collect();
        for id in ids {
            if !graph.nodes[id].alive || !is_input_placeholder(graph, id) {
                continue;
            }
            if graph.nodes[id]
                .meta
                .extra_bool("arm_control_flow_placeholder")
                == Some(true)
            {
                continue;
            }
            let Some(shape) = graph.nodes[id].meta.val_shape() else {
                continue;
            };
            let rank = shape.len();
            if rank < 4 {
                continue;
            }
            let perm = channels_last_order(rank, spatial_rank(graph, id, 2));
            if graph.nodes[id].meta.tosa_dim_order.as_deref() == Some(perm.as_slice()) {
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
        let output_ids: Vec<_> = graph
            .nodes
            .iter()
            .filter(|node| node.alive && node.op == NodeOp::Output)
            .map(|node| node.id)
            .collect();

        for output_id in output_ids {
            let outputs = output_node_ids(graph, output_id);
            let original_dim_orders = graph.nodes[output_id].meta.original_dim_orders.clone();
            for (index, output_arg) in outputs.into_iter().enumerate() {
                if !graph.nodes[output_arg].alive
                    || graph.nodes[output_arg]
                        .meta
                        .extra_bool("arm_control_flow_parent_output_user")
                        == Some(true)
                {
                    continue;
                }
                let Some(shape) = graph.nodes[output_arg].meta.val_shape() else {
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
                let logical_val = graph.nodes[output_arg].meta.val.clone().map(|mut val| {
                    let identity = (0..rank).collect::<Vec<_>>();
                    if graph.nodes[output_arg]
                        .meta
                        .tosa_dim_order
                        .as_ref()
                        .is_some_and(|order| order != &identity)
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
        let ids: Vec<_> = (0..graph.nodes.len()).collect();
        for id in ids {
            if !graph.nodes[id].alive
                || graph.nodes[id].op != NodeOp::CallFunction
                || !is_permute(&graph.nodes[id].target)
            {
                continue;
            }
            let Some(input_id) = graph.nodes[id].args.first().and_then(|arg| arg.as_node()) else {
                continue;
            };
            let Some(input_shape) = graph.nodes[input_id].meta.val_shape() else {
                continue;
            };
            let input_rank = input_shape.len();
            if input_rank < 4 {
                continue;
            }

            let expected_cl = channels_last_order(input_rank, spatial_rank(graph, input_id, 2));
            let input_is_nhwc = graph.nodes[input_id]
                .meta
                .tosa_dim_order
                .as_deref()
                .map(|order| order == expected_cl.as_slice())
                .unwrap_or(false)
                || (graph.nodes[input_id].target.contains("tosa")
                    && graph.nodes[input_id]
                        .target
                        .to_ascii_lowercase()
                        .contains("transpose"));
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
        modified |= self.propagate_spatial_ranks(graph);
        modified |= self.annotate_dim_orders(graph, is_control_flow_gm);
        modified |= self.insert_view_boundary_transposes(graph);
        modified |= self.insert_input_transposes(graph);
        modified |= self.insert_output_transposes(graph, is_control_flow_gm);
        modified |= self.remap_permute_args(graph);

        if std::env::var("ARM_TOSA_ENGINE_DEBUG_LAYOUT")
            .ok()
            .as_deref()
            == Some("1")
        {
            for node in graph.nodes.iter().filter(|node| node.alive) {
                let Some(val) = node.meta.val.as_ref() else {
                    continue;
                };
                eprintln!(
                    "[LayoutDebug] {} target={} shape={:?} sr={:?} dim_order={:?}",
                    node.name,
                    node.target,
                    val.shape,
                    node.meta.tosa_spatial_rank,
                    node.meta.tosa_dim_order,
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
