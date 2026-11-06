//! Port of rc4's `FuseIdenticalInputTransformsPass`
//! (`backends/transforms/fuse_identical_input_transforms_pass.py`), as Arm
//! configures it (`backends/arm/_passes/fuse_identical_input_transforms_pass.py`):
//! an `ArmOpTargetedPass` that skips a graph without a target op.
//!
//! Identical views or permutes feeding a multi-input elementwise op or a
//! concat sink below it, with the op's arguments remapped:
//!
//! ```text
//! add(permute(x, P), permute(y, P))   ->   permute(add(x, y), P)
//! cat([view(x, S), view(y, S)], d)    ->   view(cat([x, y], d'), S')
//! ```
//!
//! An elementwise op may also have inputs whose value is all size-1 dims at
//! the transform's rank, since they read the same in either layout
//! ([`is_layout_invariant`]).
//!
//! [`fuse_identical_input_transforms`] is the pass body inside a
//! [`LayoutCtx`], for the propagation driver's down-pass `fuse_horizontal`
//! (`backends/transforms/propagate_view_copy_permute_pass.py:921-928`, with
//! [`FuseIdenticalInputTransforms::DRIVER`]); [`IrFuseIdenticalInputTransformsPass`]
//! is the standalone pass at `arm_pass_manager.py:591`
//! ([`FuseIdenticalInputTransforms::ARM`]).
//!
//! Differences from rc4, none visible in the graph:
//!
//! - rc4's closing `ExportPass` retrace (`:199-204`) renames nodes and
//!   re-derives every value. The port sets the values rc4 sets explicitly: the
//!   sunk op keeps its dtype at its new shape (`:251-252`), and the new
//!   transform takes its value from its input (`refresh_permute_view_meta`,
//!   `:262`). Nothing downstream changes, because the new transform restores
//!   the op's old value.
//! - `NormalizeTransformInputPlaceholdersPass` (`:26-113`) also renames the
//!   graph signature's input specs; RETAB keys inputs by node name and has no
//!   signature, so only the placeholder targets are normalized.
//! - New transforms are named after their target (`aten_view_copy_default`,
//!   with a suffix when taken), as FX would name them, though rc4's retrace
//!   renames them again.
//!
//! Complexity: rc4 sweeps every node until a sweep changes nothing. The pass
//! never moves a node or creates or erases a target op, so the sweeps visit
//! the target ops collected once, in graph order. A sink only changes its op,
//! the op's inputs and the users it redirects, so the second sweep finds
//! nothing; with the default permute targets the attempt memo makes it, and
//! every later call on the same ctx, skip ops whose inputs haven't changed.
//! A call costs O(N) to collect the ops plus O(inputs) per op and per sink.

use std::collections::HashMap;

use crate::fx_ir::{broadcast_shapes, i64_list_arg, shape_product, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::{LayoutCtx, RuleId};
use crate::layout::dim_maps::{PermuteMap, ViewMap};
use crate::layout::fx_node::{all_input_nodes, py_eq};
use crate::layout::shape::{int_list, mark_contiguous, refreshed_meta};

/// `_VIEW_TARGET` (`:130`).
pub(crate) const VIEW_TARGET: &str = "aten.view_copy.default";
/// `_PERMUTE_TARGET` (`:131`), the only permute Arm uses.
pub(crate) const PERMUTE_TARGET: &str = "aten.permute_copy.default";

/// `_CONCAT_OPS` (`:132-135`).
const CONCAT_OPS: &[&str] = &["aten.cat.default", "aten.concatenate.default"];

/// `_BINARY_ELEMENTWISE_OPS` (`:137-154`).
const BINARY_ELEMENTWISE_OPS: &[&str] = &[
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_left_shift.Tensor",
    "aten.eq.Tensor",
    "aten.gt.Tensor",
    "aten.ge.Tensor",
    "aten.lt.Tensor",
    "aten.le.Tensor",
    "aten.ne.Tensor",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Tensor",
    "aten.remainder.Tensor",
];

/// `_NARY_ELEMENTWISE_OPS` (`:155-157`).
const NARY_ELEMENTWISE_OPS: &[&str] = &["aten.where.self"];

/// rc4's `delegation_tag` meta key, which a new transform takes from the op
/// it follows (`_new_transform_meta`, `:270-276`). RETAB's importer doesn't
/// keep it, so it is only ever an `extras` entry a RETAB pass wrote.
pub(crate) const DELEGATION_TAG_META_KEY: &str = "delegation_tag";

/// Memo rule for a target op that had nothing to sink; only used with the
/// default permute targets.
const NOTHING_TO_SINK: RuleId = RuleId("fuse_identical_input_transforms");

/// How rc4 constructs the pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FuseIdenticalInputTransforms<'a> {
    /// `permute_targets` (`:163-173`): the targets that count as a permute.
    pub(crate) permute_targets: &'a [&'a str],
    /// Whether an `exported_program` was passed, which runs
    /// `NormalizeTransformInputPlaceholdersPass` first (`:179-185`).
    pub(crate) normalize_placeholders: bool,
}

impl FuseIdenticalInputTransforms<'static> {
    /// `FuseIdenticalInputTransformsPass(permute_targets=self._permute_targets)`
    /// in the propagation driver (propagate_view_copy_permute_pass.py:923-925);
    /// Arm's propagation passes keep the default permute targets.
    pub(crate) const DRIVER: Self = Self {
        permute_targets: &[PERMUTE_TARGET],
        normalize_placeholders: false,
    };

    /// `FuseIdenticalInputTransformsPass(exported_program)` (arm_pass_manager.py:591).
    pub(crate) const ARM: Self = Self {
        permute_targets: &[PERMUTE_TARGET],
        normalize_placeholders: true,
    };
}

impl FuseIdenticalInputTransforms<'_> {
    /// `n.target in self._targets` (`:173`), for a call.
    fn is_transform(&self, graph: &IrGraph, id: NodeId) -> bool {
        let node = graph.node(id);
        node.op == NodeOp::CallFunction && (node.target == VIEW_TARGET || self.is_permute_target(&node.target))
    }

    /// `_is_permute` (`:175-176`).
    fn is_permute_target(&self, target: &str) -> bool {
        self.permute_targets.contains(&target)
    }
}

/// Which of `target_ops` (`:160`) a target is. `_updated_node_args`
/// (`:278-292`) treats binary and n-ary elementwise ops alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    /// `_ELEMENTWISE_OPS` (`:158`).
    Elementwise,
    /// `_CONCAT_OPS`.
    Concat,
}

fn target_kind(target: &str) -> Option<TargetKind> {
    if BINARY_ELEMENTWISE_OPS.contains(&target) || NARY_ELEMENTWISE_OPS.contains(&target) {
        Some(TargetKind::Elementwise)
    } else if CONCAT_OPS.contains(&target) {
        Some(TargetKind::Concat)
    } else {
        None
    }
}

fn is_target_op(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    node.op == NodeOp::CallFunction && target_kind(&node.target).is_some()
}

/// The Arm `FuseIdenticalInputTransformsPass(exported_program)` at
/// `arm_pass_manager.py:591`.
pub(crate) struct IrFuseIdenticalInputTransformsPass;

impl IrPass for IrFuseIdenticalInputTransformsPass {
    fn name(&self) -> &str {
        "IrFuseIdenticalInputTransforms"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        // `ArmOpTargetedPass.should_run_pass` (backends/arm/_passes/arm_pass.py:295-305).
        if !graph
            .live_call_function_ids()
            .into_iter()
            .any(|id| is_target_op(graph, id))
        {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let modified = fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::ARM);
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// `FuseIdenticalInputTransformsPass.is_layout_invariant(node, rank)`
/// (`:316-323`): `id`'s value has rank `rank` and only size-1 dims.
pub(crate) fn is_layout_invariant(graph: &IrGraph, id: NodeId, rank: usize) -> bool {
    graph
        .node_shape(id)
        .is_some_and(|shape| shape.len() == rank && shape.iter().all(|&dim| dim == 1))
}

/// `FuseIdenticalInputTransformsPass(...).call(graph_module)` (`:178-206`)
/// inside `ctx`; returns rc4's `modified`. Dead code is eliminated from the
/// whole graph when anything changed, as rc4 does.
pub(crate) fn fuse_identical_input_transforms(
    ctx: &mut LayoutCtx<'_>,
    config: &FuseIdenticalInputTransforms<'_>,
) -> bool {
    let mut modified =
        config.normalize_placeholders && normalize_transform_input_placeholders(ctx, config.permute_targets);
    let memo = config.permute_targets == [PERMUTE_TARGET];
    let ops: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| is_target_op(ctx.graph(), id))
        .collect();
    loop {
        let mut iteration_modified = false;
        for &op in &ops {
            if memo && !ctx.should_retry(NOTHING_TO_SINK, op) {
                continue;
            }
            while sink_identical_input_transforms(ctx, op, config) {
                iteration_modified = true;
            }
            if memo {
                let reads = sink_reads(ctx, op, config);
                ctx.record_attempt(NOTHING_TO_SINK, op, reads);
            }
        }
        if !iteration_modified {
            break;
        }
        modified = true;
    }
    if modified {
        ctx.eliminate_dead_code();
    }
    modified
}

/// Every node a failed sink at `op` read: the op, its inputs and the inputs'
/// producers.
fn sink_reads(ctx: &LayoutCtx<'_>, op: NodeId, config: &FuseIdenticalInputTransforms<'_>) -> Vec<NodeId> {
    let graph = ctx.graph();
    let inputs = all_input_nodes(graph.node(op));
    let mut reads = inputs.clone();
    for input in inputs {
        if config.is_transform(graph, input) {
            reads.extend(all_input_nodes(graph.node(input)).first().copied());
        }
    }
    reads
}

/// A sink, planned in full before the graph changes.
struct SinkPlan {
    input_nodes: Vec<NodeId>,
    node_args: Vec<IrArg>,
    node_kwargs: HashMap<String, IrArg>,
    node_output_shape: Vec<i64>,
    transform_target: String,
    transform_args: Vec<IrArg>,
    transform_kwargs: HashMap<String, IrArg>,
    transform_meta: NodeMeta,
}

/// `_sink_identical_input_transforms` (`:208-268`).
fn sink_identical_input_transforms(
    ctx: &mut LayoutCtx<'_>,
    node: NodeId,
    config: &FuseIdenticalInputTransforms<'_>,
) -> bool {
    let Some(plan) = plan_sink(ctx.graph(), node, config) else {
        return false;
    };
    let graph = ctx.graph();
    let producers: Vec<NodeId> = plan
        .input_nodes
        .iter()
        .map(|&input| {
            if config.is_transform(graph, input) {
                transform_input(graph, input)
            } else {
                input
            }
        })
        .collect();
    ctx.set_args(node, plan.node_args);
    ctx.set_kwargs(node, plan.node_kwargs);
    for (&input, &producer) in plan.input_nodes.iter().zip(&producers) {
        ctx.replace_input_with(node, input, producer);
    }
    for &input in &plan.input_nodes {
        if config.is_transform(ctx.graph(), input) && !ctx.graph().has_live_users(input) {
            ctx.erase_node(input);
        }
    }
    // `node.meta["val"] = node_val.new_empty(node_output_shape)` keeps the dtype.
    let meta = ctx.meta_mut(node);
    let dtype = meta.val_dtype().expect("the op has a value").to_owned();
    meta.set_tensor_facts(TensorFacts::new(plan.node_output_shape, dtype));
    mark_contiguous(meta);

    let name = plan.transform_target.replace('.', "_");
    let new_node = ctx.inserting_after(node, |ctx| {
        ctx.create_node(
            NodeOp::CallFunction,
            name,
            plan.transform_target,
            plan.transform_args,
            plan.transform_kwargs,
            plan.transform_meta,
        )
    });
    // `refresh_permute_view_meta(new_node)` (backends/transforms/permute_view_meta.py:22-56).
    let meta = refreshed_meta(ctx.graph(), new_node)
        .unwrap_or_else(|error| panic!("cannot refresh the sunk transform: {error}"));
    *ctx.meta_mut(new_node) = meta;
    for user in ctx.graph().live_user_ids(node) {
        if user != new_node {
            ctx.replace_input_with(user, node, new_node);
        }
    }
    true
}

/// `input_node.all_input_nodes[0]` for a transform input.
fn transform_input(graph: &IrGraph, transform: NodeId) -> NodeId {
    *all_input_nodes(graph.node(transform))
        .first()
        .unwrap_or_else(|| panic!("IndexError: transform {} reads no node", graph.node(transform).name))
}

/// `input_node.meta["val"].shape`, which raises `KeyError` without a value.
fn val_shape(graph: &IrGraph, id: NodeId) -> Vec<i64> {
    graph
        .node_shape_vec(id)
        .unwrap_or_else(|| panic!("KeyError: 'val' on {}", graph.node(id).name))
}

/// `transform.args[1]`, which raises `IndexError` when it is absent.
fn transform_arg(graph: &IrGraph, transform: NodeId) -> &IrArg {
    graph
        .node(transform)
        .args
        .get(1)
        .unwrap_or_else(|| panic!("IndexError: {} has no args[1]", graph.node(transform).name))
}

/// The checks and argument updates of `_sink_identical_input_transforms` up
/// to the first graph edit (`:209-232`).
fn plan_sink(graph: &IrGraph, node: NodeId, config: &FuseIdenticalInputTransforms<'_>) -> Option<SinkPlan> {
    let op = graph.node(node);
    let kind = target_kind(&op.target)?;
    let input_nodes = all_input_nodes(op);
    if input_nodes.len() < 2 {
        return None;
    }
    let node_val_shape = graph.node_shape_vec(node)?;
    let transforms: Vec<NodeId> = input_nodes
        .iter()
        .copied()
        .filter(|&input| config.is_transform(graph, input))
        .collect();
    let &transform = transforms.first()?;
    if !inputs_share_transform_or_are_layout_invariant(graph, node, kind, transform, &transforms, &input_nodes) {
        return None;
    }
    let (node_args, node_kwargs, transform_args, node_output_shape) = match kind {
        TargetKind::Elementwise => {
            update_node_args_binary(graph, node, transform, &node_val_shape, &input_nodes, config)?
        }
        TargetKind::Concat => update_node_args_concat(graph, node, transform, &node_val_shape, &input_nodes, config)?,
    };
    let transform_node = graph.node(transform);
    Some(SinkPlan {
        input_nodes,
        node_args,
        node_kwargs,
        node_output_shape,
        transform_target: transform_node.target.clone(),
        transform_args,
        transform_kwargs: transform_node.kwargs.clone(),
        transform_meta: new_transform_meta(graph, node, transform),
    })
}

/// `_new_transform_meta` (`:270-276`): the transform's meta, with the op's
/// delegation tag or none.
fn new_transform_meta(graph: &IrGraph, node: NodeId, transform: NodeId) -> NodeMeta {
    let mut meta = graph.node(transform).meta.clone();
    match graph.node(node).meta.extras.get(DELEGATION_TAG_META_KEY) {
        Some(tag) => {
            meta.extras.insert(DELEGATION_TAG_META_KEY.to_string(), tag.clone());
        }
        None => {
            meta.extras.remove(DELEGATION_TAG_META_KEY);
        }
    }
    meta
}

/// `_inputs_share_transform_or_are_layout_invariant` (`:294-314`), with
/// `_transforms_are_identical` (`:325-338`) and `_transforms_only_used_by_node`
/// (`:340-346`).
fn inputs_share_transform_or_are_layout_invariant(
    graph: &IrGraph,
    node: NodeId,
    kind: TargetKind,
    transform: NodeId,
    transforms: &[NodeId],
    input_nodes: &[NodeId],
) -> bool {
    let target = &graph.node(transforms[0]).target;
    if !transforms.iter().all(|&input| &graph.node(input).target == target) {
        return false;
    }
    let transform_arg_0 = transform_arg(graph, transforms[0]);
    if !transforms
        .iter()
        .all(|&input| py_eq(transform_arg(graph, input), transform_arg_0))
    {
        return false;
    }
    if !transforms
        .iter()
        .all(|&input| graph.live_user_ids(input).iter().all(|&user| user == node))
    {
        return false;
    }
    if transforms.len() == input_nodes.len() {
        return true;
    }
    if kind == TargetKind::Concat {
        return false;
    }
    let Some(rank) = graph.node_shape(transform).map(<[i64]>::len) else {
        return false;
    };
    input_nodes
        .iter()
        .all(|input| transforms.contains(input) || is_layout_invariant(graph, *input, rank))
}

/// The transform's arguments with `node` as input: `(node, *transform.args[1:])`.
fn transform_args_after(graph: &IrGraph, node: NodeId, transform: NodeId) -> Vec<IrArg> {
    std::iter::once(IrArg::Node(node))
        .chain(graph.node(transform).args.iter().skip(1).cloned())
        .collect()
}

/// `node_output_shape[dim]` with Python indexing, after rc4's one-step
/// normalization of a negative permute dim.
fn permuted_shape(shape: &[i64], dims: &[i64]) -> Vec<i64> {
    let rank = shape.len() as i64;
    dims.iter()
        .map(|&dim| {
            let dim = if dim >= 0 { dim } else { dim + rank };
            let index = if dim < 0 { dim + rank } else { dim };
            assert!(
                (0..rank).contains(&index),
                "IndexError: permute dim {dim} out of range for rank {rank}"
            );
            shape[index as usize]
        })
        .collect()
}

type NodeArgsUpdate = (Vec<IrArg>, HashMap<String, IrArg>, Vec<IrArg>, Vec<i64>);

/// `_update_node_args_binary` (`:348-385`), also used for `where`.
fn update_node_args_binary(
    graph: &IrGraph,
    node: NodeId,
    transform: NodeId,
    node_val_shape: &[i64],
    input_nodes: &[NodeId],
    config: &FuseIdenticalInputTransforms<'_>,
) -> Option<NodeArgsUpdate> {
    let producer_shapes: Vec<Vec<i64>> = input_nodes
        .iter()
        .map(|&input| {
            let source = if config.is_transform(graph, input) {
                transform_input(graph, input)
            } else {
                input
            };
            val_shape(graph, source)
        })
        .collect();
    let node_output_shape = broadcast_shapes(&producer_shapes)?;

    let target = &graph.node(transform).target;
    let mut transform_args = transform_args_after(graph, node, transform);
    if target == VIEW_TARGET {
        transform_args = vec![IrArg::Node(node), i64_list_arg(node_val_shape)];
        // `:370-374`: the broadcast must already be one of the producer shapes.
        if !producer_shapes.contains(&node_output_shape) {
            return None;
        }
    }
    if config.is_permute_target(target) {
        let dims = int_list(transform_arg(graph, transform)).expect("permute dims are ints");
        if node_val_shape != permuted_shape(&node_output_shape, &dims) {
            return None;
        }
    }
    let op = graph.node(node);
    Some((op.args.clone(), op.kwargs.clone(), transform_args, node_output_shape))
}

/// `_update_node_args_concat` (`:387-427`) with `_concat_dim` (`:447-449`).
fn update_node_args_concat(
    graph: &IrGraph,
    node: NodeId,
    transform: NodeId,
    node_val_shape: &[i64],
    input_nodes: &[NodeId],
    config: &FuseIdenticalInputTransforms<'_>,
) -> Option<NodeArgsUpdate> {
    let op = graph.node(node);
    let dim_arg = match op.args.get(1) {
        Some(arg) => Some(arg),
        None => op.kwargs.get("dim"),
    };
    // `isinstance(dim, int)`, which a bool also passes.
    let concat_dim = match dim_arg {
        None => 0,
        Some(IrArg::Int(dim)) => *dim,
        Some(IrArg::Bool(dim)) => i64::from(*dim),
        Some(_) => return None,
    };
    let new_concat_dim = mapped_concat_dim(graph, transform, concat_dim, config)?;

    let tensors = op
        .args
        .first()
        .unwrap_or_else(|| panic!("IndexError: {} has no tensors argument", op.name));
    let mut node_args = vec![tensors.clone(), IrArg::Int(new_concat_dim as i64)];
    node_args.extend(op.args.iter().skip(2).cloned());
    let mut node_kwargs = op.kwargs.clone();
    node_kwargs.remove("dim");

    let producer_shapes: Vec<Vec<i64>> = input_nodes
        .iter()
        .map(|&input| val_shape(graph, transform_input(graph, input)))
        .collect();
    if !concat_producer_shapes_match(&producer_shapes, new_concat_dim) {
        return None;
    }
    let mut node_output_shape = producer_shapes[0].clone();
    node_output_shape[new_concat_dim] = producer_shapes.iter().map(|shape| shape[new_concat_dim]).sum();

    let target = &graph.node(transform).target;
    let mut transform_args = transform_args_after(graph, node, transform);
    if target == VIEW_TARGET {
        if shape_product(&node_output_shape) != shape_product(node_val_shape) {
            return None;
        }
        transform_args = vec![IrArg::Node(node), i64_list_arg(node_val_shape)];
    }
    if config.is_permute_target(target) {
        let dims = int_list(transform_arg(graph, transform)).expect("permute dims are ints");
        if node_val_shape != permuted_shape(&node_output_shape, &dims) {
            return None;
        }
    }
    Some((node_args, node_kwargs, transform_args, node_output_shape))
}

/// `_concat_producer_shapes_match` (`:429-445`); `concat_dim` is already
/// normalized.
fn concat_producer_shapes_match(producer_shapes: &[Vec<i64>], concat_dim: usize) -> bool {
    let reference = &producer_shapes[0];
    let rank = reference.len();
    concat_dim < rank
        && producer_shapes.iter().all(|shape| {
            shape.len() == rank && (0..rank).all(|dim| dim == concat_dim || shape[dim] == reference[dim])
        })
}

/// `_mapped_concat_dim` (`:451-461`): the concat dim before the transform.
/// `PermuteMap` raises on invalid dims, and `map_dims_inverse` on a dim out
/// of range, and so does the port.
fn mapped_concat_dim(
    graph: &IrGraph,
    transform: NodeId,
    concat_dim: i64,
    config: &FuseIdenticalInputTransforms<'_>,
) -> Option<usize> {
    let arg = transform_arg(graph, transform);
    if config.is_permute_target(&graph.node(transform).target) {
        let dims = int_list(arg).expect("permute dims are ints");
        let map = PermuteMap::new(&dims).unwrap_or_else(|error| panic!("ValueError: {error:?}"));
        let mapped = map
            .map_dims_inverse(&[concat_dim])
            .unwrap_or_else(|error| panic!("AssertionError: {error:?}"));
        return Some(mapped[0]);
    }
    let source = val_shape(graph, transform_input(graph, transform));
    let size = int_list(arg).expect("a view size is ints");
    let view_map = ViewMap::from_shapes(&source, &size).unwrap_or_else(|error| panic!("AssertionError: {error:?}"));
    if !view_map.is_valid_map() {
        return None;
    }
    let mapped = view_map
        .map_dim_inverse(&[concat_dim])
        .expect("the map is valid")?;
    (mapped.len() == 1).then(|| mapped[0])
}

/// `NormalizeTransformInputPlaceholdersPass._normalize_transform_user_input_placeholders`
/// (`:46-81`): a placeholder whose name or target looks like a transform's
/// gets its name as target. rc4 also renames the input spec of the old target
/// (`:68-78`); RETAB has no graph signature (inputs are keyed by node name).
fn normalize_transform_input_placeholders(ctx: &mut LayoutCtx<'_>, permute_targets: &[&str]) -> bool {
    // `_target_placeholder_name` (`:100-101`).
    let base_names: Vec<String> = std::iter::once(VIEW_TARGET)
        .chain(permute_targets.iter().copied())
        .map(|target| target.replace('.', "_"))
        .collect();
    let placeholders: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| ctx.node(id).op == NodeOp::Placeholder)
        .collect();
    let mut modified = false;
    for id in placeholders {
        let node = ctx.node(id);
        let represents_transform = base_names.iter().any(|base| {
            matches_transform_placeholder_name(&node.name, base)
                || matches_transform_placeholder_name(&node.target, base)
        });
        if represents_transform && node.target != node.name {
            let name = node.name.clone();
            ctx.set_target(id, name);
            modified = true;
        }
    }
    modified
}

/// `_matches_transform_placeholder_name` (`:90-98`): `base` or `base_<digits>`.
/// Python's `isdecimal` also takes non-ASCII decimal digits, which FX names
/// don't hold.
fn matches_transform_placeholder_name(name: &str, base: &str) -> bool {
    name == base
        || name
            .strip_prefix(base)
            .and_then(|rest| rest.strip_prefix('_'))
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph) -> bool {
        IrFuseIdenticalInputTransformsPass.run(g.graph_mut())
    }

    #[test]
    fn placeholder_names_match_transform_bases() {
        let base = "aten_view_copy_default";
        assert!(matches_transform_placeholder_name("aten_view_copy_default", base));
        assert!(matches_transform_placeholder_name("aten_view_copy_default_12", base));
        assert!(!matches_transform_placeholder_name("aten_view_copy_default_", base));
        assert!(!matches_transform_placeholder_name("aten_view_copy_default_x1", base));
        assert!(!matches_transform_placeholder_name("aten_view_copy_default1", base));
        assert!(!matches_transform_placeholder_name("custom_view_copy_default", base));
    }

    #[test]
    fn layout_invariant_values_are_all_ones_at_the_rank() {
        let mut g = LayoutGraph::new();
        let ones = g.input("ones", &[1, 1, 1]);
        let scalar = g.input("scalar", &[]);
        let wide = g.input("wide", &[1, 2, 1]);
        assert!(is_layout_invariant(g.graph(), ones, 3));
        assert!(!is_layout_invariant(g.graph(), ones, 2));
        assert!(is_layout_invariant(g.graph(), scalar, 0));
        assert!(!is_layout_invariant(g.graph(), wide, 3));
    }

    /// A permute dim normalizes once, then indexes like Python.
    #[test]
    fn permuted_shapes_index_like_python() {
        assert_eq!(permuted_shape(&[2, 3, 4], &[0, 2, 1]), vec![2, 4, 3]);
        assert_eq!(permuted_shape(&[2, 3, 4], &[-3, -1, -2]), vec![2, 4, 3]);
        // -4 + 3 = -1, which Python reads as the last dim.
        assert_eq!(permuted_shape(&[2, 3, 4], &[-4]), vec![4]);
    }

    /// The docstring example (`:121-125`): `add(permute(x), permute(y))`
    /// becomes `permute(add(x, y))`, and the add's users read the permute.
    #[test]
    fn identical_permutes_sink_below_an_add() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let y = g.input("y", &[2, 3, 4]);
        let px = g.permute(x, &[0, 2, 1]);
        let py = g.permute(y, &[0, 2, 1]);
        let add = g.call(ADD, vec![node(px), node(py)]);
        let neg = g.call(NEG, vec![node(add)]);
        g.output(&[neg]);
        let (add, neg) = (g.name(add), g.name(neg));

        assert!(run(&mut g));

        g.assert_targets(&[ADD, PERMUTE, NEG]);
        let (add, neg) = (g.id(&add), g.id(&neg));
        assert_eq!(g.shape(add), &[2, 3, 4]);
        let permute = g.graph().node(neg).arg_node(0).unwrap();
        assert_eq!(g.graph().node(permute).arg_node(0), Some(add));
        assert_eq!(g.shape(permute), &[2, 4, 3]);
        g.assert_permute_dims(&[&[0, 2, 1]]);
    }

    #[test]
    fn a_layout_invariant_operand_lets_a_single_transform_sink() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let ones = g.input("ones", &[1, 1, 1]);
        let px = g.permute(x, &[0, 2, 1]);
        let mul = g.call(MUL, vec![node(px), node(ones)]);
        g.output(&[mul]);

        assert!(run(&mut g));

        g.assert_targets(&[MUL, PERMUTE]);
    }

    #[test]
    fn a_concat_needs_every_input_transformed() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let y = g.input("y", &[2, 4, 3]);
        let px = g.permute(x, &[0, 2, 1]);
        let cat = g.cat(&[px, y], 1);
        g.output(&[cat]);

        assert!(!run(&mut g));
    }

    /// The concat dim maps back through the permute, and a kwarg `dim` moves
    /// into the positional args.
    #[test]
    fn concat_dims_map_through_the_permute() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let y = g.input("y", &[2, 5, 4]);
        let px = g.permute(x, &[0, 2, 1]);
        let py = g.permute(y, &[0, 2, 1]);
        let tensors = IrArg::List(vec![node(px), node(py)]);
        let cat = g.call_with_kwargs(CAT, vec![tensors], &[("dim", IrArg::Int(-1))]);
        g.output(&[cat]);
        let cat = g.name(cat);

        assert!(run(&mut g));

        let cat = g.id(&cat);
        let op = g.graph().node(cat);
        assert_eq!(op.args[1].as_int(), Some(1));
        assert!(op.kwargs.is_empty());
        assert_eq!(g.shape(cat), &[2, 8, 4]);
        g.assert_targets(&[CAT, PERMUTE]);
    }

    /// A target op with a non-transform input that isn't all ones keeps its
    /// transforms, and so does one whose transforms have other users.
    #[test]
    fn mixed_and_shared_inputs_are_left_alone() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let y = g.input("y", &[2, 4, 3]);
        let px = g.permute(x, &[0, 2, 1]);
        let add = g.call(ADD, vec![node(px), node(y)]);
        let z = g.input("z", &[2, 3, 4]);
        let pz = g.permute(z, &[0, 2, 1]);
        let pw = g.permute(z, &[0, 2, 1]);
        let sub = g.call("aten.sub.Tensor", vec![node(pz), node(pw)]);
        g.output(&[add, sub, pz]);

        assert!(!run(&mut g));
    }

    /// The ctx entry point the propagation driver uses: no placeholder
    /// normalization, and a second call finds nothing new.
    #[test]
    fn the_driver_configuration_runs_inside_a_ctx() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[6]);
        let y = g.input("y", &[6]);
        let vx = g.view(x, &[2, 3]);
        let vy = g.view(y, &[2, 3]);
        let add = g.call(ADD, vec![node(vx), node(vy)]);
        g.output(&[add]);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER));
        assert!(!fuse_identical_input_transforms(&mut ctx, &FuseIdenticalInputTransforms::DRIVER));
        ctx.finish();
        g.assert_targets(&[ADD, VIEW]);
    }
}
