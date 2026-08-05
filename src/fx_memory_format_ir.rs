use std::collections::{HashMap, HashSet};

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::{
    is_constant_only_shape_transform_target, is_layout_anchor_target, is_permute_target,
    is_raw_aten_convolution_target, is_tosa_transpose_target, is_transpose_conv2d_target,
    is_view_like_target, is_zero_input_constant_target,
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

/// One transpose shared by every consumer that needs it. Stock likewise inserts a single
/// permute after an anchor and lets all of its users share it, so sharing here is what
/// keeps the emitted TRANSPOSE count equal to stock's.
///
/// `physical_output` marks the direction. A transpose that *produces* channels-last
/// advertises the permuted order and hands the producer back to logical order -- every
/// consumer that wanted the producer channels-last is in `users` and has just been
/// rewired, so nothing is left behind expecting the old annotation.
fn insert_transpose_for_users(
    graph: &mut IrGraph,
    producer_id: NodeId,
    users: &[NodeId],
    name: String,
    perm: &[usize],
    physical_output: bool,
) -> NodeId {
    let transpose_id = add_transpose(graph, producer_id, name, perm);
    if physical_output {
        graph.nodes[transpose_id]
            .meta
            .set_tosa_dim_order(perm.to_vec());
        if let Some(rank) = graph.nodes[producer_id].meta.rank() {
            graph.nodes[producer_id]
                .meta
                .set_tosa_dim_order((0..rank).collect());
        }
    }
    graph.anchor_after(transpose_id, producer_id);
    for &user_id in users {
        graph.replace_input_use(user_id, producer_id, transpose_id);
    }
    transpose_id
}

/// The layout a node currently advertises, as opposed to the one the plan asked for.
/// `insert_view_boundary_transposes` rewrites annotations as it goes, so the boundary pass
/// that runs after it has to read the graph rather than the plan.
fn annotated_physical(graph: &IrGraph, id: NodeId) -> bool {
    let Some(rank) = graph.nodes[id].meta.rank() else {
        return false;
    };
    if rank < 4 {
        return false;
    }
    // Read the stamped spatial rank rather than defaulting: this has to agree with
    // `annotate_dim_orders` exactly or a node would look like it disagreed with itself.
    // `annotate_spatial_ranks` stamps everything that has tensor facts.
    let Some(sr) = graph.nodes[id].meta.tosa_spatial_rank() else {
        return false;
    };
    let expected = channels_last_order(rank, sr);
    graph.nodes[id].meta.tosa_dim_order() == Some(expected.as_slice())
}

fn set_spatial_rank(graph: &mut IrGraph, id: NodeId, rank: usize) -> bool {
    graph.nodes[id].meta.set_tosa_spatial_rank(rank)
}

fn spatial_rank(graph: &IrGraph, id: NodeId, default: usize) -> usize {
    graph.nodes[id].meta.tosa_spatial_rank_or(default)
}

fn input_node_ids(graph: &IrGraph, id: NodeId) -> Vec<NodeId> {
    let mut ids = Vec::new();
    let node = &graph.nodes[id];
    for arg in node.args.iter().chain(node.kwargs.values()) {
        collect_output_node_ids(arg, &mut ids);
    }
    ids
}

/// The sides of a reshape on which the channels-last permutation is layout-significant.
///
/// A reshape can only be applied to a tensor in logical order, so a channels-last operand
/// has to be permuted back first and the result permuted forward again. That is only worth
/// anything when the permutation actually reorders data: `[1, 6, 256] -> [1, 1, 6, 256]`
/// merely adds a unit dim, and `[1, 6] -> [1, 6, 1, 1]` only moves unit dims, so
/// channels-last is a free relabel there and the reshape is layout-transparent. Stock draws
/// the same line, admitting single-unit-dim views into a permutable region and rejecting
/// every other reshape (`RemovePermutesAroundElementwiseOps::_is_squeeze_unsqueeze_view`).
///
/// Single source of truth for both `LayoutPlan` and `insert_view_boundary_transposes`.
struct ViewLayoutSides {
    input_shape: Vec<i64>,
    output_shape: Vec<i64>,
    input_side: bool,
    output_side: bool,
}

fn view_layout_sides(graph: &IrGraph, id: NodeId) -> Option<ViewLayoutSides> {
    if !is_view_like_target(&graph.nodes[id].target) {
        return None;
    }
    let input_id = graph.nodes[id].first_node_arg()?;
    let input_shape = graph.node_shape_vec(input_id)?;
    let output_shape = graph.node_shape_vec(id)?;
    let input_rank = input_shape.len();
    let output_rank = output_shape.len();
    let channel_reshape = is_channel_reshape(&input_shape, &output_shape);
    let input_side = (channel_reshape || (input_rank >= 4 && output_rank < 4))
        && memory_format_differs(&input_shape, input_rank);
    let output_side = (channel_reshape || (input_rank < 4 && output_rank >= 4))
        && memory_format_differs(&output_shape, output_rank);
    Some(ViewLayoutSides {
        input_shape,
        output_shape,
        input_side,
        output_side,
    })
}

/// A reshape that has to run in logical order, so it breaks a channels-last span.
fn view_is_layout_barrier(graph: &IrGraph, id: NodeId) -> bool {
    view_layout_sides(graph, id)
        .is_some_and(|sides| sides.input_side || sides.output_side)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LayoutRole {
    /// TOSA form is defined channels-last, so this op is always physical.
    Anchor,
    /// Delegate input or graph output. Physical only when torch itself declared the
    /// boundary tensor channels-last.
    Boundary,
    /// A rank- or channel-changing reshape. Always logical; see `view_is_layout_barrier`.
    Barrier,
    /// Compile-time value. Follows its consumers, because the payload is materialized in
    /// whatever layout the annotation asks for.
    Constant,
    /// Rank < 4, so logical and physical order coincide. Layout-agnostic: the emitters
    /// already reconcile a lower-rank operand against a channels-last one.
    Logical,
    /// Everything else. Layout is decided by the span it belongs to.
    Span,
}

/// Which rank>=4 tensors are physically channels-last.
///
/// Stock ExecuTorch keeps the ARM delegate's interior in logical (NCHW) order and wraps
/// each layout anchor -- an op whose TOSA form is defined channels-last: convolution,
/// pooling, resize -- in a local `permute -> op -> permute` island. Islands merge into a
/// wider channels-last span only when `RemovePermutesAroundElementwiseOps` finds the
/// elementwise run between two of them bounded by a matching permute pair on *every*
/// edge; a run that reaches a delegate input or the delegate output is rejected outright
/// and stays logical (`backends/transforms/remove_permutes_around_elementwise_ops.py`,
/// `visit`: `elif user.op == "output": return False`).
///
/// So a span is channels-last exactly when every tensor crossing its border comes from or
/// goes to an anchor. This pass used to annotate *every* rank>=4 tensor channels-last,
/// which agrees with stock wherever anchors bound both sides but diverges wherever a span
/// touches the delegate boundary. The divergence is observable on rank>=4 constants,
/// because RETAB bakes the layout permutation into the payload rather than emitting a
/// TRANSPOSE: a LayerNorm's affine pair came out `[1, 16, 16, 8]` where stock emits
/// `[1, 8, 16, 16]`.
struct LayoutPlan {
    role: HashMap<NodeId, LayoutRole>,
    physical: HashSet<NodeId>,
}

impl LayoutPlan {
    /// Control-flow submodules keep the pre-existing blanket annotation: their boundary
    /// tensors are channels-last by construction (see `annotate_dim_orders`), so there is
    /// no logical delegate edge for a span to terminate against.
    fn all_physical(graph: &IrGraph) -> Self {
        let mut plan = Self {
            role: HashMap::new(),
            physical: HashSet::new(),
        };
        for id in graph.live_node_ids() {
            plan.role.insert(id, LayoutRole::Span);
            plan.physical.insert(id);
        }
        plan
    }

    fn build(graph: &IrGraph) -> Self {
        let mut constant_memo = HashMap::new();
        let mut role: HashMap<NodeId, LayoutRole> = HashMap::new();
        let mut physical: HashSet<NodeId> = HashSet::new();

        // Boundary tensors torch itself declared channels-last: the delegate hands them
        // over already permuted, so they anchor a span physically instead of logically.
        let mut channels_last_boundary: HashSet<NodeId> = HashSet::new();
        for output_id in graph.live_output_ids() {
            let Some(orders) = graph.nodes[output_id]
                .meta
                .original_dim_orders()
                .map(|orders| orders.to_vec())
            else {
                continue;
            };
            for (index, arg_id) in output_node_ids(graph, output_id).into_iter().enumerate() {
                let Some(shape) = graph.node_shape(arg_id) else {
                    continue;
                };
                let rank = shape.len();
                let expected = channels_last_order(rank, spatial_rank(graph, arg_id, 2));
                if orders.get(index) == Some(&expected) {
                    channels_last_boundary.insert(arg_id);
                }
            }
        }

        for id in graph.live_node_ids() {
            let node = &graph.nodes[id];
            let rank = node.meta.tensor_facts().map(|val| val.rank()).unwrap_or(0);
            let assigned = if node.op == NodeOp::Output || is_input_placeholder(graph, id) {
                LayoutRole::Boundary
            } else if is_constant_subgraph(graph, id, &mut constant_memo) {
                LayoutRole::Constant
            } else if node.op == NodeOp::CallFunction && is_layout_anchor_target(&node.target) {
                // Before the rank test: a `max_pool2d_with_indices` returns a tuple and so
                // carries no rank of its own, but it still anchors the span around it.
                LayoutRole::Anchor
            } else if rank < 4 {
                LayoutRole::Logical
            } else if view_is_layout_barrier(graph, id) {
                LayoutRole::Barrier
            } else {
                LayoutRole::Span
            };

            // Anchors are physical by definition. Barriers are logical by definition: a
            // reshape only makes sense on a tensor in logical order, which is why
            // `insert_view_boundary_transposes` brackets them.
            if assigned == LayoutRole::Anchor {
                physical.insert(id);
            }
            if assigned == LayoutRole::Boundary
                && rank >= 4
                && node.op == NodeOp::Placeholder
                && node.meta.val_dim_order()
                    == Some(channels_last_order(rank, spatial_rank(graph, id, 2)).as_slice())
            {
                physical.insert(id);
                channels_last_boundary.insert(id);
            }
            role.insert(id, assigned);
        }

        // Maximal connected runs of spans. Data edges are followed both ways, so a run is
        // cut exactly at anchors, barriers, constants and the delegate boundary.
        let mut region_of: HashMap<NodeId, usize> = HashMap::new();
        let mut regions: Vec<Vec<NodeId>> = Vec::new();
        for start in graph.live_node_ids() {
            if role.get(&start) != Some(&LayoutRole::Span) || region_of.contains_key(&start) {
                continue;
            }
            let index = regions.len();
            let mut members = Vec::new();
            let mut stack = vec![start];
            region_of.insert(start, index);
            while let Some(id) = stack.pop() {
                members.push(id);
                let neighbours = graph
                    .live_user_ids(id)
                    .into_iter()
                    .chain(input_node_ids(graph, id));
                for next in neighbours {
                    if role.get(&next) != Some(&LayoutRole::Span) || region_of.contains_key(&next) {
                        continue;
                    }
                    region_of.insert(next, index);
                    stack.push(next);
                }
            }
            regions.push(members);
        }

        let region_members: Vec<HashSet<NodeId>> = regions
            .iter()
            .map(|members| members.iter().copied().collect())
            .collect();
        let mut region_physical: Vec<bool> = region_members
            .iter()
            .map(|members| Self::span_is_physical(graph, &role, members, &channels_last_boundary))
            .collect();
        for (index, members) in regions.iter().enumerate() {
            if region_physical[index] {
                physical.extend(members.iter().copied());
            }
        }

        // Constants follow their consumers. A constant shared between a channels-last
        // consumer and a logical span promotes that span, because RETAB has no site that
        // can emit a TRANSPOSE on a constant and so cannot serve both layouts at once.
        let constants: Vec<NodeId> = graph
            .live_node_ids()
            .into_iter()
            .filter(|id| role.get(id) == Some(&LayoutRole::Constant))
            .filter(|&id| graph.node_shape(id).is_some_and(|shape| shape.len() >= 4))
            .collect();
        loop {
            let mut changed = false;
            for &id in &constants {
                if physical.contains(&id) {
                    continue;
                }
                let users: Vec<NodeId> = graph
                    .live_user_ids(id)
                    .into_iter()
                    .filter(|user| role.get(user) != Some(&LayoutRole::Logical))
                    .collect();
                if users.is_empty() || users.iter().any(|user| physical.contains(user)) {
                    physical.insert(id);
                    changed = true;
                }
            }
            for (index, members) in regions.iter().enumerate() {
                if region_physical[index] {
                    continue;
                }
                let consumes_physical_constant = members.iter().any(|&member| {
                    input_node_ids(graph, member).into_iter().any(|producer| {
                        role.get(&producer) == Some(&LayoutRole::Constant)
                            && physical.contains(&producer)
                    })
                });
                if consumes_physical_constant {
                    region_physical[index] = true;
                    physical.extend(members.iter().copied());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        Self { role, physical }
    }

    fn span_is_physical(
        graph: &IrGraph,
        role: &HashMap<NodeId, LayoutRole>,
        members: &HashSet<NodeId>,
        channels_last_boundary: &HashSet<NodeId>,
    ) -> bool {
        for &id in members {
            let borders = graph
                .live_user_ids(id)
                .into_iter()
                .chain(input_node_ids(graph, id));
            for border in borders {
                if members.contains(&border) {
                    continue;
                }
                match role.get(&border) {
                    // Bounded by an anchor on this edge, which is what merges an
                    // elementwise run into the surrounding channels-last island.
                    Some(LayoutRole::Anchor) => {}
                    // Materialized in whichever layout this span settles on.
                    Some(LayoutRole::Constant) => {}
                    // Rank < 4: logical and physical order coincide, so this edge says
                    // nothing about the span's layout.
                    Some(LayoutRole::Logical) => {}
                    // The delegate boundary is logical unless torch declared the tensor
                    // channels-last, in which case it anchors physically. A delegate input
                    // narrower than rank 4 carries no layout, so it says nothing either way;
                    // the graph output always does, because the crossing tensor is this
                    // rank>=4 member itself.
                    Some(LayoutRole::Boundary) => {
                        let is_graph_output = graph.nodes[border].op == NodeOp::Output;
                        if !is_graph_output && graph.nodes[border].meta.rank().unwrap_or(0) < 4 {
                            continue;
                        }
                        if !channels_last_boundary.contains(&border)
                            && !channels_last_boundary.contains(&id)
                        {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    fn is_physical(&self, id: NodeId) -> bool {
        self.physical.contains(&id)
    }

    fn is_barrier(&self, id: NodeId) -> bool {
        self.role.get(&id) == Some(&LayoutRole::Barrier)
    }

    /// False for nodes a transpose insertion created after the plan was computed.
    fn knows(&self, id: NodeId) -> bool {
        self.role.contains_key(&id)
    }
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

    /// Record each graph output's torch dim order before anything is annotated, so both
    /// `LayoutPlan` and `insert_output_transposes` can tell a genuinely channels-last
    /// model output from one that merely passed through a channels-last span.
    fn record_original_dim_orders(&self, graph: &mut IrGraph, is_control_flow_gm: bool) -> bool {
        if is_control_flow_gm {
            return false;
        }
        let mut modified = false;
        for id in graph.live_output_ids() {
            if graph.nodes[id].meta.original_dim_orders().is_some() {
                continue;
            }
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
        modified
    }

    fn annotate_dim_orders(
        &self,
        graph: &mut IrGraph,
        plan: &LayoutPlan,
        is_control_flow_gm: bool,
    ) -> bool {
        let mut modified = false;
        for id in graph.live_node_ids() {
            let Some(val) = graph.nodes[id].meta.tensor_facts() else {
                continue;
            };
            let rank = val.rank();
            let sr = spatial_rank(graph, id, 0);

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
            } else if rank >= 4 && plan.is_physical(id) {
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

    /// Bracket each reshape that has to run in logical order with the permutes that take it
    /// there and back. Only the sides `view_layout_sides` reports as significant need one,
    /// and only against a neighbour that is actually channels-last.
    fn insert_view_boundary_transposes(&self, graph: &mut IrGraph, plan: &LayoutPlan) -> bool {
        let mut modified = false;
        let mut constant_memo = HashMap::new();
        for id in graph.live_call_function_ids() {
            let Some(sides) = view_layout_sides(graph, id) else {
                continue;
            };
            if is_constant_subgraph(graph, id, &mut constant_memo) {
                continue;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                continue;
            };

            if sides.input_side && plan.is_physical(input_id) {
                let input_rank = sides.input_shape.len();
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

            if sides.output_side {
                let physical_users: Vec<NodeId> = graph
                    .live_user_ids(id)
                    .into_iter()
                    .filter(|&user| plan.is_physical(user))
                    .collect();
                if !physical_users.is_empty() {
                    let output_rank = sides.output_shape.len();
                    let perm = channels_last_order(output_rank, spatial_rank(graph, id, 2));
                    insert_transpose_for_users(
                        graph,
                        id,
                        &physical_users,
                        format!("{}_output_layout", graph.nodes[id].name),
                        &perm,
                        true,
                    );
                    modified = true;
                }
            }
        }
        modified
    }

    /// Reconcile every data edge whose two ends disagree on layout. This does the job the
    /// permutes bounding stock's channels-last islands do: a delegate input reaching a
    /// convolution gets one on the way in, and a convolution feeding a span that runs
    /// logically gets one on the way out.
    fn insert_layout_boundary_transposes(&self, graph: &mut IrGraph, plan: &LayoutPlan) -> bool {
        let mut modified = false;
        for producer_id in graph.live_node_ids() {
            // A node an earlier insertion created is already on the correct side of the
            // boundary it was made for.
            if !plan.knows(producer_id) {
                continue;
            }
            if graph.nodes[producer_id]
                .meta
                .extra_bool("arm_control_flow_placeholder")
                == Some(true)
            {
                continue;
            }
            let Some(shape) = graph.node_shape(producer_id) else {
                continue;
            };
            let rank = shape.len();
            if rank < 4 {
                continue;
            }
            let producer_physical = annotated_physical(graph, producer_id);

            let needy: Vec<NodeId> = graph
                .live_user_ids(producer_id)
                .into_iter()
                .filter(|&user| {
                    // `insert_output_transposes` owns the graph output edge.
                    if graph.nodes[user].op == NodeOp::Output {
                        return false;
                    }
                    // A node the plan has never seen was inserted for a boundary that is
                    // already resolved. A consumer with no rank of its own (a tuple-valued
                    // pool) or a rank<4 one reconciles the operand itself.
                    if !plan.knows(user) {
                        return false;
                    }
                    if graph.nodes[user].meta.rank().unwrap_or(0) < 4 {
                        return false;
                    }
                    // The view pass owns the permute that takes a channels-last tensor into
                    // the logical order a reshape needs.
                    if plan.is_barrier(user) && producer_physical {
                        return false;
                    }
                    annotated_physical(graph, user) != producer_physical
                })
                .collect();
            if needy.is_empty() {
                continue;
            }

            let sr = spatial_rank(graph, producer_id, 2);
            let (perm, suffix) = if producer_physical {
                (channels_last_inverse_order(rank, sr), "output_layout")
            } else {
                (channels_last_order(rank, sr), "input_layout")
            };
            insert_transpose_for_users(
                graph,
                producer_id,
                &needy,
                format!("{}_{}", graph.nodes[producer_id].name, suffix),
                &perm,
                !producer_physical,
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
                // A span that terminates at the delegate output already runs logically, so
                // there is nothing to undo here -- the boundary sits further upstream, at
                // the anchor that span left behind.
                if !annotated_physical(graph, output_arg) {
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
        modified |= self.propagate_spatial_ranks(graph);
        // The plan reads the recorded output dim orders, so record them first.
        modified |= self.record_original_dim_orders(graph, is_control_flow_gm);
        let plan = if is_control_flow_gm {
            LayoutPlan::all_physical(graph)
        } else {
            LayoutPlan::build(graph)
        };
        modified |= self.annotate_dim_orders(graph, &plan, is_control_flow_gm);
        modified |= self.insert_view_boundary_transposes(graph, &plan);
        modified |= self.insert_layout_boundary_transposes(graph, &plan);
        modified |= self.insert_output_transposes(graph, is_control_flow_gm);
        modified |= self.remap_permute_args(graph);

        if crate::config::debug_layout_enabled() {
            for id in graph.live_node_ids() {
                let node = &graph.nodes[id];
                let Some(val) = node.meta.tensor_facts() else {
                    continue;
                };
                eprintln!(
                    "[LayoutDebug] {} target={} shape={:?} sr={:?} dim_order={:?} role={:?} physical={}",
                    node.name,
                    node.target,
                    val.shape,
                    node.meta.tosa_spatial_rank(),
                    node.meta.cloned_tosa_dim_order(),
                    plan.role.get(&id),
                    plan.is_physical(id),
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
