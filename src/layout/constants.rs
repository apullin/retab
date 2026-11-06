//! Explicit mode's stand-in for rc4's ComputeConstantOpsAOTPass and
//! FuseConstantArgsPass (`backends/arm/_passes/fuse_constant_ops_pass.py`,
//! line numbers below are that file's), which rc4 runs back to back at
//! `arm_pass_manager.py:699-700` (APU-135 P2, critique B8).
//!
//! From that slot on, rc4 has no constant subgraphs left: ComputeConstantOpsAOT
//! evaluates factories (`full`, `arange`, ...) into `<node>_pre_computed`
//! buffers at the head of the graph, and FuseConstantArgs evaluates every op
//! whose inputs are all constants into a `<node>_fused_const` placeholder.
//! Phase 6 (RewriteConv's weights, the permute and view passes) only ever sees
//! those placeholders.
//!
//! RETAB folds the same subgraphs as deferred lowering effects: a planner
//! records a MaterializeTensorNode effect keyed by the node's name and ingest
//! applies it (`ingest::from_ir::apply_effects`), so in dim-order mode the call
//! nodes stay in the IR until conversion. [`IrMaterializeConstantsPass`] runs
//! the same planners at rc4's slot and applies their plan to the IR instead:
//!
//! - **What** becomes a constant, and which bytes it holds, is the planners'
//!   decision. The pass runs IrFuseZeroInputConstantsEffectPass (rc4's
//!   ComputeConstantOpsAOT share), IrFuseShapeOnlyConstantsEffectPass and
//!   IrFuseConstantInputsEffectPass::with_data_slices (FuseConstantArgs's
//!   share; the variant the pipeline runs at this slot, `fx_pipeline.rs:208`),
//!   and gives each planned node the tensor ingest would store under its name
//!   (`ingest::from_ir::materialized_tensor`, which narrows int64 data to
//!   int32 when it fits). The planners' rc4 quirks, such as quantizing a
//!   factory with its output qparams, come along unchanged.
//! - **How** the IR changes follows rc4: the names, the constant kinds and
//!   insert points, the special-dtype propagation, the graph-wide dead-code
//!   elimination and the deletion of the constant placeholders the folds
//!   leave unused.
//!
//! Each new constant gets a new name, so an effect an earlier pass planned
//! under the folded node's name finds no live node at ingest and is skipped
//! (P2 design F9); the constant already holds the same data. Nodes the
//! planners leave alone (targets they do not fold, transposed convs, ...)
//! stay call nodes and keep the dim-order treatment.
//!
//! In a region the constants are `get_attr` nodes (see
//! [`GraphScope::Region`]) and no placeholder is added or deleted, since the
//! boundary contracts pair region placeholders with the parent's operands by
//! position (critique B2).

use std::collections::HashMap;

use crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass;
use crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass;
use crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass;
use crate::fx_ir::{
    EffectfulIrPass, EffectfulIrPassResult, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeOp,
    TensorSource,
};
use crate::ingest::from_ir::materialized_tensor;
use crate::layout::ctx::{GraphScope, InsertPoint, LayoutCtx};
use crate::lowering_effects::{LoweringEffect, LoweringInputKind};
use crate::store::TensorData;

const SPECIAL_DTYPE: &str = "tosa_special_dtype";

/// Whether `node` is a constant: a live placeholder or `get_attr` node that
/// carries its data. In the parent these are the exported program's
/// parameters, buffers and lifted constants plus the constants the passes
/// create; rc4's FuseConstantArgs folds an op only when every input is one
/// (`:248-266`).
pub(crate) fn is_constant(node: &IrNode) -> bool {
    node.alive && matches!(node.op, NodeOp::Placeholder | NodeOp::GetAttr) && node.tensor.is_some()
}

/// rc4's `get_constant_placeholder_kind` for a constant: the exported-program
/// input kind it is bound to. A `GraphModuleAttr` constant (a region constant,
/// or a fused batch norm's weight) counts as a buffer, the kind of RETAB's INT
/// frozen parameters.
pub(crate) fn constant_kind(node: &IrNode) -> LoweringInputKind {
    match node.tensor.as_ref().map(|tensor| &tensor.source) {
        Some(TensorSource::ExportedProgramInput(binding)) => binding.input_kind,
        _ => LoweringInputKind::Buffer,
    }
}

/// rc4's ComputeConstantOpsAOTPass and FuseConstantArgsPass, applied in the IR
/// for explicit mode (see the module docs). `in_region` is the pipeline's
/// region flag: region constants are `get_attr` nodes.
pub(crate) struct IrMaterializeConstantsPass {
    pub(crate) in_region: bool,
}

impl IrPass for IrMaterializeConstantsPass {
    fn name(&self) -> &str {
        "IrMaterializeConstants"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut plan = Plan::of(graph);
        if plan.factories.is_empty() && plan.folds.is_empty() {
            return false;
        }
        let scope = GraphScope::of_region(self.in_region);
        let placeholders_before = (scope == GraphScope::Region).then(|| placeholder_names(graph));

        let mut ctx = LayoutCtx::with_scope(graph, scope);
        let precomputed = compute_constant_ops_aot(&mut ctx, &mut plan.factories);
        let fused_inputs = fuse_constant_args(&mut ctx, &mut plan.folds);
        if !precomputed && fused_inputs.is_empty() {
            // Every planned fold was skipped; nothing changed.
            return false;
        }
        // Both rc4 passes end with a graph-wide `eliminate_dead_code()`
        // (`:278-280`, `:369-372`), which drops the folded nodes.
        ctx.eliminate_dead_code();
        if scope == GraphScope::Parent {
            delete_unused_constant_inputs(&mut ctx, fused_inputs);
        }
        ctx.finish();

        if let Some(before) = placeholders_before {
            debug_assert_eq!(
                placeholder_names(graph),
                before,
                "a region's placeholders must stay as they are"
            );
        }
        true
    }
}

/// The planners' constants by node: rc4's two passes' shares.
#[derive(Default)]
struct Plan {
    /// ComputeConstantOpsAOT's share: zero-input factories.
    factories: HashMap<NodeId, TensorData>,
    /// FuseConstantArgs's share: ops whose inputs are all constant.
    folds: HashMap<NodeId, TensorData>,
}

impl Plan {
    /// Run the effect planners on `graph`, which they only read, and key
    /// their MaterializeTensorNode effects by node. A node two planners plan
    /// keeps the later plan, as ingest applies effects in order. Planner
    /// order: IrFuseConstantInputsEffectPass plans every node the shape-only
    /// planner does, with the same bytes; the shape-only planner runs too so
    /// that the pass covers each planner whose effects ingest applies.
    fn of(graph: &mut IrGraph) -> Self {
        let factories = IrFuseZeroInputConstantsEffectPass.run(graph);
        let shape_only = IrFuseShapeOnlyConstantsEffectPass.run(graph);
        let constant_inputs = IrFuseConstantInputsEffectPass::with_data_slices().run(graph);

        let ids = live_ids_by_name(graph);
        let mut plan = Self::default();
        record(&mut plan.factories, factories, &ids);
        record(&mut plan.folds, shape_only, &ids);
        record(&mut plan.folds, constant_inputs, &ids);
        plan
    }
}

/// Live node ids by name; `None` for a name two live nodes share, which no
/// name-keyed effect can address reliably (ingest would pick one).
fn live_ids_by_name(graph: &IrGraph) -> HashMap<String, Option<NodeId>> {
    let mut ids: HashMap<String, Option<NodeId>> = HashMap::new();
    for node in graph.live_nodes() {
        ids.entry(node.name.clone())
            .and_modify(|id| *id = None)
            .or_insert(Some(node.id));
    }
    ids
}

fn record(
    plan: &mut HashMap<NodeId, TensorData>,
    result: EffectfulIrPassResult,
    ids: &HashMap<String, Option<NodeId>>,
) {
    for effect in result.effects.items {
        let LoweringEffect::MaterializeTensorNode(effect) = effect else {
            continue;
        };
        if let Some(&Some(id)) = ids.get(&effect.node_name) {
            plan.insert(
                id,
                materialized_tensor(&effect.data, &effect.shape, effect.dtype),
            );
        }
    }
}

/// ComputeConstantOpsAOTPass.call (`:355-374`): each planned factory, in graph
/// order, becomes `<node>_pre_computed`, a persistent buffer inserted before
/// the graph's first node (`:337-350`). The constants therefore end up at the
/// head in reverse graph order: an avg-pool's output zero point before its
/// input zero point (critique A7).
fn compute_constant_ops_aot(
    ctx: &mut LayoutCtx<'_>,
    planned: &mut HashMap<NodeId, TensorData>,
) -> bool {
    let mut changed = false;
    for id in ctx.order().to_vec() {
        let Some(data) = planned.remove(&id) else {
            continue;
        };
        let name = format!("{}_pre_computed", ctx.node(id).name);
        let constant = ctx.with_insert_point(InsertPoint::Start, |ctx| {
            ctx.create_constant(name, LoweringInputKind::Buffer, data)
        });
        ctx.replace_all_uses_with(id, constant);
        changed = true;
    }
    changed
}

/// FuseConstantArgsPass.call (`:228-285`) over the planned folds, in graph
/// order: each becomes `<node>_fused_const`, of its first input's kind and
/// inserted directly before that input (`:192-207`), with the inputs' special
/// dtype (`:209`). A fold in a chain reads the constant that replaced its
/// producer, so a chain ends as one constant where the chain's source was.
///
/// rc4 folds a node only when every input is a constant by the time the sweep
/// reaches it, and never one marked SHAPE (`:155-157`). The planners' plan
/// meets the first rule except below a SHAPE-marked node, which the shape-only
/// planner folds; such a node and everything planned on top of it stay call
/// nodes here, and their deferred effects still fold them at ingest.
///
/// Returns the folded nodes' inputs, rc4's `input_nodes_to_maybe_delete`.
fn fuse_constant_args(
    ctx: &mut LayoutCtx<'_>,
    planned: &mut HashMap<NodeId, TensorData>,
) -> Vec<NodeId> {
    let mut fused_inputs = Vec::new();
    for id in ctx.order().to_vec() {
        let Some(data) = planned.remove(&id) else {
            continue;
        };
        if is_shape_marked(ctx.node(id)) {
            continue;
        }
        let inputs = ctx.all_input_nodes(id);
        let Some(&first) = inputs.first() else {
            continue;
        };
        if !inputs.iter().all(|&input| is_constant(ctx.node(input))) {
            continue;
        }
        let name = format!("{}_fused_const", ctx.node(id).name);
        let kind = constant_kind(ctx.node(first));
        let special_dtype = propagated_special_dtype(ctx.graph(), &inputs, &data);
        let constant = ctx.inserting_before(first, |ctx| ctx.create_constant(name, kind, data));
        if let Some(special_dtype) = special_dtype {
            ctx.meta_mut(constant)
                .set_extra(SPECIAL_DTYPE, MetaValue::Str(special_dtype));
        }
        ctx.replace_all_uses_with(id, constant);
        fused_inputs.extend(inputs);
    }
    fused_inputs
}

/// FuseConstantArgsPass.maybe_delete (`:215-221`), after the dead-code
/// elimination: the folds' constant placeholders that nothing reads any more
/// are deleted, as rc4 deletes them from the graph and the exported program.
fn delete_unused_constant_inputs(ctx: &mut LayoutCtx<'_>, inputs: Vec<NodeId>) {
    for input in inputs {
        let node = ctx.node(input);
        if node.op == NodeOp::Placeholder && is_constant(node) && !ctx.graph().has_live_users(input)
        {
            ctx.erase_node(input);
        }
    }
}

/// A special dtype as rc4 names it: RETAB's passes write `INT4`, `INT48` or
/// `SHAPE`; an imported one reads `TosaSpecialDtype.INT4`.
fn special_dtype_name(raw: &str) -> &str {
    raw.strip_prefix("TosaSpecialDtype.").unwrap_or(raw)
}

fn is_shape_marked(node: &IrNode) -> bool {
    node.meta
        .extra_str(SPECIAL_DTYPE)
        .is_some_and(|raw| special_dtype_name(raw) == "SHAPE")
}

/// FuseConstantArgsPass._propagate_special_dtype (`:132-147`): the inputs'
/// special dtype other than SHAPE, when there is exactly one and the folded
/// data fits it. This is how an INT4 weight's fold stays INT4, which
/// RewriteConvPass then copies onto the conv-local weight.
fn propagated_special_dtype(
    graph: &IrGraph,
    inputs: &[NodeId],
    data: &TensorData,
) -> Option<String> {
    let mut found: Vec<&str> = Vec::new();
    for &input in inputs {
        let Some(raw) = graph.node(input).meta.extra_str(SPECIAL_DTYPE) else {
            continue;
        };
        let name = special_dtype_name(raw);
        if name != "SHAPE" && !found.iter().any(|seen| special_dtype_name(seen) == name) {
            found.push(raw);
        }
    }
    let [raw] = found.as_slice() else {
        if found.len() > 1 {
            crate::runner_logging::log_line(format_args!(
                "[IrMaterializeConstants] propagating mixed special dtypes {found:?} is not \
                 implemented, skipping"
            ));
        }
        return None;
    };
    // `TosaSpecialDtype.max()` covers only INT4 and INT48; rc4 fails the fold
    // for any other special dtype, which an INT graph never carries.
    let max = match special_dtype_name(raw) {
        "INT4" => 7.0,
        "INT48" => ((1i64 << 47) - 1) as f64,
        _ => return None,
    };
    // `data.abs().max() <= special_dtype.max()`, which a NaN fails.
    let values = data.vec_f64()?;
    let fits = !values.is_empty() && values.iter().all(|value| value.abs() <= max);
    fits.then(|| (*raw).to_string())
}

fn placeholder_names(graph: &IrGraph) -> Vec<String> {
    graph
        .live_placeholder_ids()
        .into_iter()
        .map(|id| graph.node(id).name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::fx_ir::{IrArg, NodeMeta, QParam};
    use crate::graph::ir::DType;
    use crate::lowering_effects::LoweringEffectBatch;
    use crate::store::TensorStore;
    use crate::test_ir::{call, call_with_meta, constant_placeholder, placeholder};

    const VIEW: &str = "aten.view_copy.default";
    const PERMUTE: &str = "aten.permute_copy.default";
    const SLICE: &str = "aten.slice_copy.Tensor";
    const ADD: &str = "aten.add.Tensor";
    const FULL: &str = "aten.full.default";

    fn ints(values: &[i64]) -> IrArg {
        crate::fx_ir::i64_list_arg(values)
    }

    fn node(id: NodeId) -> IrArg {
        IrArg::Node(id)
    }

    fn int8_tensor(shape: &[i64], values: &[i8]) -> TensorData {
        TensorData {
            data: values.iter().map(|&v| v as u8).collect(),
            shape: shape.to_vec(),
            dtype: DType::Int8,
        }
    }

    fn int32_tensor(shape: &[i64], values: &[i32]) -> TensorData {
        TensorData {
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            dtype: DType::Int32,
        }
    }

    fn f32_tensor(shape: &[i64], values: &[f32]) -> TensorData {
        TensorData {
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            dtype: DType::Float32,
        }
    }

    fn int8_qparam(scale: f64, zp: i64) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, "torch.int8".to_string())
    }

    /// A constant placeholder of `kind` whose meta records the data's value,
    /// as RETAB's importer gives parameters.
    fn constant(
        graph: &mut IrGraph,
        name: &str,
        data: TensorData,
        kind: LoweringInputKind,
    ) -> NodeId {
        let meta = NodeMeta::tensor(data.shape.clone(), data.dtype.torch_str());
        let id = constant_placeholder(graph, name, data, kind);
        graph.node_mut(id).meta = meta;
        id
    }

    fn weight(graph: &mut IrGraph, name: &str, data: TensorData) -> NodeId {
        constant(graph, name, data, LoweringInputKind::Buffer)
    }

    fn output(graph: &mut IrGraph, results: &[NodeId]) -> NodeId {
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::List(results.iter().map(|&id| node(id)).collect())],
            NodeMeta::default(),
        )
    }

    fn full(graph: &mut IrGraph, name: &str, shape: &[i64], value: f64, meta: NodeMeta) -> NodeId {
        let id = call_with_meta(
            graph,
            name,
            FULL,
            vec![ints(shape), IrArg::Float(value)],
            meta,
        );
        graph.set_kwarg(id, "dtype", IrArg::Dtype("torch.float32".to_string()));
        id
    }

    fn run(graph: &mut IrGraph) -> bool {
        IrMaterializeConstantsPass { in_region: false }.run(graph)
    }

    fn run_in_region(graph: &mut IrGraph) -> bool {
        IrMaterializeConstantsPass { in_region: true }.run(graph)
    }

    /// Live node names in graph order (node-id order after `finish`).
    fn names(graph: &IrGraph) -> Vec<String> {
        graph
            .topological_live_node_ids()
            .into_iter()
            .map(|id| graph.node(id).name.clone())
            .collect()
    }

    fn live(graph: &IrGraph, name: &str) -> NodeId {
        graph
            .live_nodes()
            .find(|node| node.name == name)
            .unwrap_or_else(|| panic!("no live node {name} in {:?}", names(graph)))
            .id
    }

    fn tensor<'g>(graph: &'g IrGraph, name: &str) -> &'g TensorData {
        &graph
            .node(live(graph, name))
            .tensor
            .as_ref()
            .expect("a constant")
            .data
    }

    fn assert_tensor_eq(actual: &TensorData, expected: &TensorData, what: &str) {
        assert_eq!(
            (&actual.data, &actual.shape, actual.dtype),
            (&expected.data, &expected.shape, expected.dtype),
            "{what}"
        );
    }

    fn kind(graph: &IrGraph, name: &str) -> String {
        let node = graph.node(live(graph, name));
        match &node.tensor.as_ref().expect("a constant").source {
            TensorSource::ExportedProgramInput(binding) => {
                assert_eq!(
                    binding.target_name, node.name,
                    "{name}'s target is its name"
                );
                format!("{:?}", binding.input_kind)
            }
            TensorSource::GraphModuleAttr => "attr".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // rc4's structure
    // -----------------------------------------------------------------------

    /// FuseConstantArgs on `view_copy(W) -> permute_copy`: each fold goes
    /// directly before its first input, so the chain's last constant takes
    /// the source weight's place; the intermediate constant and the weight
    /// lose their last user and are deleted (`maybe_delete`); the folded
    /// calls go with the dead-code elimination. The constant holds the
    /// permuted data with rc4's meta: the data's value, contiguous, no qparams.
    #[test]
    fn a_constant_chain_becomes_one_constant_where_its_source_was() {
        let mut graph = IrGraph::empty();
        let w = weight(
            &mut graph,
            "b__frozen_param0",
            int8_tensor(&[2, 3], &[1, 2, 3, 4, 5, 6]),
        );
        let x = placeholder(&mut graph, "x", [3, 2], "torch.int8");
        let view = call(
            &mut graph,
            "aten_view_copy_default",
            VIEW,
            vec![node(w), ints(&[3, 2])],
            [3, 2],
            "torch.int8",
        );
        let permute = call(
            &mut graph,
            "aten_permute_copy_default",
            PERMUTE,
            vec![node(view), ints(&[1, 0])],
            [2, 3],
            "torch.int8",
        );
        let x_t = call(
            &mut graph,
            "x_t",
            PERMUTE,
            vec![node(x), ints(&[1, 0])],
            [2, 3],
            "torch.int8",
        );
        let add = call(
            &mut graph,
            "aten_add_tensor",
            ADD,
            vec![node(x_t), node(permute)],
            [2, 3],
            "torch.float32",
        );
        output(&mut graph, &[add]);

        assert!(run(&mut graph));
        assert_eq!(
            names(&graph),
            [
                "aten_permute_copy_default_fused_const",
                "x",
                "x_t",
                "aten_add_tensor",
                "output"
            ]
        );
        let constant = live(&graph, "aten_permute_copy_default_fused_const");
        assert_eq!(graph.node(constant).op, NodeOp::Placeholder);
        assert_eq!(
            kind(&graph, "aten_permute_copy_default_fused_const"),
            "Buffer"
        );
        // [[1, 2, 3], [4, 5, 6]] viewed as [3, 2] and transposed.
        assert_tensor_eq(
            tensor(&graph, "aten_permute_copy_default_fused_const"),
            &int8_tensor(&[2, 3], &[1, 3, 5, 2, 4, 6]),
            "permuted weight",
        );
        let meta = &graph.node(constant).meta;
        assert_eq!(meta.val_shape(), Some(&[2, 3][..]));
        assert_eq!(meta.val_dtype(), Some("torch.int8"));
        assert_eq!(meta.val_dim_order(), Some(&[0, 1][..]));
        assert!(meta.input_qparams().is_none() && meta.output_qparams().is_none());
        assert_eq!(
            graph.node(live(&graph, "aten_add_tensor")).args[1].as_node(),
            Some(constant)
        );
        for gone in [
            "b__frozen_param0",
            "aten_view_copy_default",
            "aten_view_copy_default_fused_const",
        ] {
            assert!(
                graph.live_nodes().all(|node| node.name != gone),
                "{gone} survived"
            );
        }
    }

    /// A fold whose input another node still reads leaves that input alone,
    /// and an intermediate constant with another user survives too.
    #[test]
    fn constants_that_are_still_read_survive() {
        let mut graph = IrGraph::empty();
        let w = weight(&mut graph, "w", int8_tensor(&[4], &[1, 2, 3, 4]));
        let x = placeholder(&mut graph, "x", [2, 2], "torch.int8");
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(w), ints(&[2, 2])],
            [2, 2],
            "torch.int8",
        );
        let permute = call(
            &mut graph,
            "permute",
            PERMUTE,
            vec![node(view), ints(&[1, 0])],
            [2, 2],
            "torch.int8",
        );
        let add_view = call(
            &mut graph,
            "add_view",
            ADD,
            vec![node(x), node(view)],
            [2, 2],
            "torch.int8",
        );
        let add_permute = call(
            &mut graph,
            "add_permute",
            ADD,
            vec![node(x), node(permute)],
            [2, 2],
            "torch.int8",
        );
        let neg_w = call(
            &mut graph,
            "neg_w",
            "aten.neg.default",
            vec![node(w)],
            [4],
            "torch.int8",
        );
        output(&mut graph, &[add_view, add_permute, neg_w]);

        assert!(run(&mut graph));
        assert_eq!(
            names(&graph),
            [
                "permute_fused_const",
                "view_fused_const",
                "w",
                "x",
                "add_view",
                "add_permute",
                "neg_w",
                "output"
            ]
        );
        assert_eq!(
            graph.node(live(&graph, "add_view")).args[1].as_node(),
            Some(live(&graph, "view_fused_const"))
        );
    }

    /// Grouped-conv weight slices share their first input: each constant goes
    /// directly before it, in fold order.
    #[test]
    fn folds_of_one_input_line_up_before_it() {
        let mut graph = IrGraph::empty();
        let w = weight(&mut graph, "w", int8_tensor(&[4, 1], &[1, 2, 3, 4]));
        let x = placeholder(&mut graph, "x", [2, 1], "torch.int8");
        let s0 = call(
            &mut graph,
            "slice0",
            SLICE,
            vec![node(w), IrArg::Int(0), IrArg::Int(0), IrArg::Int(2)],
            [2, 1],
            "torch.int8",
        );
        let s1 = call(
            &mut graph,
            "slice1",
            SLICE,
            vec![node(w), IrArg::Int(0), IrArg::Int(2), IrArg::Int(4)],
            [2, 1],
            "torch.int8",
        );
        let a0 = call(
            &mut graph,
            "a0",
            ADD,
            vec![node(x), node(s0)],
            [2, 1],
            "torch.int8",
        );
        let a1 = call(
            &mut graph,
            "a1",
            ADD,
            vec![node(x), node(s1)],
            [2, 1],
            "torch.int8",
        );
        output(&mut graph, &[a0, a1]);

        assert!(run(&mut graph));
        assert_eq!(
            names(&graph),
            [
                "slice0_fused_const",
                "slice1_fused_const",
                "x",
                "a0",
                "a1",
                "output"
            ]
        );
        assert_tensor_eq(
            tensor(&graph, "slice0_fused_const"),
            &int8_tensor(&[2, 1], &[1, 2]),
            "slice0",
        );
        assert_tensor_eq(
            tensor(&graph, "slice1_fused_const"),
            &int8_tensor(&[2, 1], &[3, 4]),
            "slice1",
        );
    }

    /// ComputeConstantOpsAOT puts each `_pre_computed` buffer before the first
    /// node, so the factories end up at the head in reverse order: an
    /// avg-pool's output zero point first, then its input zero point
    /// (critique A7, `aten_full_default_1_pre_computed` before
    /// `aten_full_default_pre_computed`). Both are quantized with the pool's
    /// output qparams, rc4's ComputeConstantOpsAOT quirk (critique B2).
    #[test]
    fn factories_become_buffers_at_the_head_in_reverse_order() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 4, 4, 4], "torch.int8");
        let mut pool_meta = NodeMeta::tensor([1], "torch.float32");
        pool_meta.set_single_input_qparam(int8_qparam(0.25, -3));
        pool_meta.set_single_output_qparam(int8_qparam(0.5, 1));
        let input_zp = full(
            &mut graph,
            "aten_full_default",
            &[1],
            -3.0,
            pool_meta.clone(),
        );
        let output_zp = full(&mut graph, "aten_full_default_1", &[1], 1.0, pool_meta);
        let pool = call(
            &mut graph,
            "pool",
            "backend.tosa.AVG_POOL2D.default",
            vec![
                node(x),
                node(input_zp),
                node(output_zp),
                ints(&[2, 2]),
                ints(&[2, 2]),
                ints(&[0, 0, 0, 0]),
            ],
            [1, 2, 2, 4],
            "torch.float32",
        );
        output(&mut graph, &[pool]);

        assert!(run(&mut graph));
        assert_eq!(
            names(&graph),
            [
                "aten_full_default_1_pre_computed",
                "aten_full_default_pre_computed",
                "x",
                "pool",
                "output"
            ]
        );
        for name in [
            "aten_full_default_1_pre_computed",
            "aten_full_default_pre_computed",
        ] {
            assert_eq!(kind(&graph, name), "Buffer");
        }
        // quantize(-3.0, 0.5, 1) = -5 and quantize(1.0, 0.5, 1) = 3.
        assert_tensor_eq(
            tensor(&graph, "aten_full_default_pre_computed"),
            &int8_tensor(&[1], &[-5]),
            "input zp",
        );
        assert_tensor_eq(
            tensor(&graph, "aten_full_default_1_pre_computed"),
            &int8_tensor(&[1], &[3]),
            "output zp",
        );
        let pool = graph.node(live(&graph, "pool"));
        assert_eq!(
            pool.args[1].as_node(),
            Some(live(&graph, "aten_full_default_pre_computed"))
        );
        assert_eq!(
            pool.args[2].as_node(),
            Some(live(&graph, "aten_full_default_1_pre_computed"))
        );
    }

    /// A fold takes its first input's kind; a fold of a `_pre_computed`
    /// buffer is a buffer.
    #[test]
    fn folds_take_their_first_inputs_kind() {
        let mut graph = IrGraph::empty();
        let p = constant(
            &mut graph,
            "p",
            f32_tensor(&[2], &[1.0, 2.0]),
            LoweringInputKind::Parameter,
        );
        let c = constant(
            &mut graph,
            "c",
            int32_tensor(&[2], &[3, 4]),
            LoweringInputKind::LiftedTensorConstant,
        );
        let view_p = call(
            &mut graph,
            "view_p",
            VIEW,
            vec![node(p), ints(&[2, 1])],
            [2, 1],
            "torch.float32",
        );
        let view_c = call(
            &mut graph,
            "view_c",
            VIEW,
            vec![node(c), ints(&[1, 2])],
            [1, 2],
            "torch.int32",
        );
        let mut meta = NodeMeta::tensor([2], "torch.float32");
        meta.set_single_output_qparam(int8_qparam(0.5, 0));
        let f = full(&mut graph, "f", &[2], 1.0, meta);
        let view_f = call(
            &mut graph,
            "view_f",
            VIEW,
            vec![node(f), ints(&[2, 1])],
            [2, 1],
            "torch.int8",
        );
        output(&mut graph, &[view_p, view_c, view_f]);

        assert!(run(&mut graph));
        assert_eq!(kind(&graph, "view_p_fused_const"), "Parameter");
        assert_eq!(kind(&graph, "view_c_fused_const"), "LiftedTensorConstant");
        assert_eq!(kind(&graph, "view_f_fused_const"), "Buffer");
        assert!(
            graph.live_nodes().all(|node| node.name != "f_pre_computed"),
            "the factory's buffer was folded on"
        );
        assert_eq!(
            names(&graph)[0],
            "view_f_fused_const",
            "it took the factory buffer's place"
        );
    }

    /// An int64 plan becomes int32 when its values fit, as ingest's
    /// materialization stores it; values that do not fit stay int64.
    #[test]
    fn int64_plans_narrow_like_ingest() {
        let mut graph = IrGraph::empty();
        let small = call(
            &mut graph,
            "small",
            "aten.arange.start_step",
            vec![IrArg::Int(0), IrArg::Int(3), IrArg::Int(1)],
            [3],
            "torch.int64",
        );
        graph.set_kwarg(small, "dtype", IrArg::Dtype("torch.int64".to_string()));
        let big = call_with_meta(
            &mut graph,
            "big",
            FULL,
            vec![ints(&[1]), IrArg::Int(1 << 40)],
            NodeMeta::tensor([1], "torch.int64"),
        );
        graph.set_kwarg(big, "dtype", IrArg::Dtype("torch.int64".to_string()));
        output(&mut graph, &[small, big]);

        assert!(run(&mut graph));
        assert_tensor_eq(
            tensor(&graph, "small_pre_computed"),
            &int32_tensor(&[3], &[0, 1, 2]),
            "narrowed",
        );
        assert_eq!(
            graph
                .node(live(&graph, "small_pre_computed"))
                .meta
                .val_dtype(),
            Some("torch.int32")
        );
        let big = tensor(&graph, "big_pre_computed");
        assert_eq!(
            (big.dtype, big.data.clone()),
            (DType::Int64, (1i64 << 40).to_le_bytes().to_vec())
        );
    }

    /// A folded output is returned as the constant itself.
    #[test]
    fn a_folded_output_is_returned_as_its_constant() {
        let mut graph = IrGraph::empty();
        let w = weight(&mut graph, "w", int8_tensor(&[2], &[1, 2]));
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(w), ints(&[1, 2])],
            [1, 2],
            "torch.int8",
        );
        output(&mut graph, &[view]);
        assert!(run(&mut graph));
        let out = graph.output_node().expect("an output node");
        let IrArg::List(results) = &out.args[0] else {
            panic!("an output list")
        };
        assert_eq!(results[0].as_node(), Some(live(&graph, "view_fused_const")));
    }

    /// The constant's name is new even when an erased node had it, so no
    /// effect planned under an old name can reach it (P2 design F9).
    #[test]
    fn constants_never_take_an_erased_nodes_name() {
        let mut graph = IrGraph::empty();
        let stale = placeholder(&mut graph, "view_fused_const", [1], "torch.int8");
        graph.erase(stale);
        let w = weight(&mut graph, "w", int8_tensor(&[2], &[1, 2]));
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(w), ints(&[1, 2])],
            [1, 2],
            "torch.int8",
        );
        output(&mut graph, &[view]);
        assert!(run(&mut graph));
        assert_eq!(names(&graph), ["view_fused_const_1", "output"]);
    }

    /// rc4 creates a `_pre_computed` buffer for a factory nobody reads too;
    /// the dead-code elimination drops the factory and the buffer stays.
    #[test]
    fn an_unused_factory_leaves_an_unused_buffer_like_rc4() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1], "torch.float32");
        let mut meta = NodeMeta::tensor([1], "torch.float32");
        meta.set_single_output_qparam(int8_qparam(1.0, 0));
        full(&mut graph, "unused", &[1], 2.0, meta);
        output(&mut graph, &[x]);
        assert!(run(&mut graph));
        assert_eq!(names(&graph), ["unused_pre_computed", "x", "output"]);
    }

    /// Like rc4's `eliminate_dead_code()`, a change drops every dead pure node,
    /// not only the folded ones; impure nodes stay. Without a change the
    /// graph is untouched.
    #[test]
    fn a_change_eliminates_dead_code_graph_wide() {
        let build = |with_fold: bool| {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2], "torch.int8");
            call(
                &mut graph,
                "dead_neg",
                "aten.neg.default",
                vec![node(x)],
                [2],
                "torch.int8",
            );
            call(
                &mut graph,
                "dead_inplace",
                "aten.add_.Tensor",
                vec![node(x), node(x)],
                [2],
                "torch.int8",
            );
            let mut results = vec![x];
            if with_fold {
                let w = weight(&mut graph, "w", int8_tensor(&[2], &[1, 2]));
                results.push(call(
                    &mut graph,
                    "view",
                    VIEW,
                    vec![node(w), ints(&[2, 1])],
                    [2, 1],
                    "torch.int8",
                ));
            }
            output(&mut graph, &results);
            graph
        };

        let mut graph = build(true);
        assert!(run(&mut graph));
        assert_eq!(
            names(&graph),
            ["x", "dead_inplace", "view_fused_const", "output"]
        );

        let mut graph = build(false);
        let before = names(&graph);
        assert!(!run(&mut graph));
        assert_eq!(names(&graph), before);
        assert!(graph.live_nodes().any(|node| node.name == "dead_neg"));
    }

    // -----------------------------------------------------------------------
    // Special dtypes
    // -----------------------------------------------------------------------

    fn marked(graph: &mut IrGraph, id: NodeId, special_dtype: &str) {
        graph
            .node_mut(id)
            .meta
            .set_extra(SPECIAL_DTYPE, MetaValue::Str(special_dtype.to_string()));
    }

    fn special_dtype(graph: &IrGraph, name: &str) -> Option<String> {
        graph
            .node(live(graph, name))
            .meta
            .extra_str(SPECIAL_DTYPE)
            .map(str::to_string)
    }

    /// `_propagate_special_dtype`: an INT4 weight's fold is INT4 while its
    /// values fit [-7, 7]; data beyond INT4's range, mixed special dtypes and
    /// SHAPE (ignored, `:137`) behave as in rc4.
    #[test]
    fn the_inputs_special_dtype_carries_over_when_the_data_fits() {
        let mut graph = IrGraph::empty();
        let w4 = weight(&mut graph, "w4", int8_tensor(&[2, 2], &[-7, 0, 3, 7]));
        marked(&mut graph, w4, "INT4");
        let wide = weight(&mut graph, "wide", int8_tensor(&[2], &[8, 1]));
        marked(&mut graph, wide, "INT4");
        let w48 = weight(&mut graph, "w48", int32_tensor(&[2], &[1, 2]));
        marked(&mut graph, w48, "TosaSpecialDtype.INT48");
        let shape = weight(&mut graph, "shape", int32_tensor(&[2], &[5, 6]));
        marked(&mut graph, shape, "SHAPE");
        let v4 = call(
            &mut graph,
            "v4",
            VIEW,
            vec![node(w4), ints(&[4])],
            [4],
            "torch.int8",
        );
        let v_wide = call(
            &mut graph,
            "v_wide",
            VIEW,
            vec![node(wide), ints(&[2, 1])],
            [2, 1],
            "torch.int8",
        );
        let v48 = call(
            &mut graph,
            "v48",
            VIEW,
            vec![node(w48), ints(&[1, 2])],
            [1, 2],
            "torch.int32",
        );
        let mixed = call(
            &mut graph,
            "mixed",
            "aten.gt.Tensor",
            vec![node(w48), node(wide)],
            [2],
            "torch.bool",
        );
        let with_shape = call(
            &mut graph,
            "with_shape",
            "aten.gt.Tensor",
            vec![node(shape), node(w4)],
            [2, 2],
            "torch.bool",
        );
        output(&mut graph, &[v4, v_wide, v48, mixed, with_shape]);

        assert!(run(&mut graph));
        assert_eq!(
            special_dtype(&graph, "v4_fused_const").as_deref(),
            Some("INT4")
        );
        assert_eq!(
            special_dtype(&graph, "v_wide_fused_const"),
            None,
            "8 is outside INT4"
        );
        assert_eq!(
            special_dtype(&graph, "v48_fused_const").as_deref(),
            Some("TosaSpecialDtype.INT48")
        );
        assert_eq!(
            special_dtype(&graph, "mixed_fused_const"),
            None,
            "INT48 and INT4"
        );
        assert_eq!(
            special_dtype(&graph, "with_shape_fused_const").as_deref(),
            Some("INT4"),
            "a bool fold of an INT4 input fits INT4, as in rc4"
        );
    }

    /// rc4 never fuses a SHAPE-marked node (`:155-157`), but the shape-only
    /// planner plans one. The pass leaves it, and the fold planned on top of
    /// it, as call nodes for ingest's effects.
    #[test]
    fn shape_marked_nodes_and_their_dependents_stay_call_nodes() {
        let mut graph = IrGraph::empty();
        let w = weight(&mut graph, "w", int32_tensor(&[2], &[1, 2]));
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(w), ints(&[1, 2])],
            [1, 2],
            "torch.int32",
        );
        marked(&mut graph, view, "SHAPE");
        let permute = call(
            &mut graph,
            "permute",
            PERMUTE,
            vec![node(view), ints(&[1, 0])],
            [2, 1],
            "torch.int32",
        );
        output(&mut graph, &[permute]);

        let planned = IrFuseShapeOnlyConstantsEffectPass.run(&mut graph);
        assert_eq!(planned.effects.items.len(), 2, "both are planned");
        let before = names(&graph);
        assert!(!run(&mut graph));
        assert_eq!(names(&graph), before);
    }

    // -----------------------------------------------------------------------
    // The same data as the ingest-time effects
    // -----------------------------------------------------------------------

    /// A parent graph with one of each planner's folds, every result an output:
    /// factories (quantized by their own and by a binary user's qparams, an
    /// int64 arange, linspace, eye, scalar_tensor), a shape-only chain, a data
    /// slice, a cast of a lifted constant, a comparison, a quantized
    /// reciprocal and quantized mm, linear and conv folds, plus a chain with a
    /// runtime user.
    fn every_planner_graph() -> IrGraph {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.int8");
        let q = placeholder(&mut graph, "q", [2], "torch.int8");
        graph
            .node_mut(q)
            .meta
            .set_single_output_qparam(int8_qparam(0.25, 0));
        let w = weight(&mut graph, "w", int8_tensor(&[2, 3], &[1, 2, 3, 4, 5, 6]));
        let w2 = weight(&mut graph, "w2", int32_tensor(&[4], &[10, 20, 30, 40]));
        let lifted = constant(
            &mut graph,
            "lifted",
            int32_tensor(&[2], &[-3, 9]),
            LoweringInputKind::LiftedTensorConstant,
        );
        let fa = weight(&mut graph, "fa", f32_tensor(&[2], &[1.0, -2.0]));
        let fb = weight(&mut graph, "fb", f32_tensor(&[2], &[0.5, -1.0]));
        let divisor = weight(&mut graph, "divisor", f32_tensor(&[1, 1], &[2.0]));
        let lhs = weight(&mut graph, "qlhs", int8_tensor(&[1, 2], &[1, 2]));
        let rhs = weight(&mut graph, "qrhs", int8_tensor(&[2, 1], &[3, 4]));
        let lin_w = weight(&mut graph, "qlin_w", int8_tensor(&[1, 2], &[3, 4]));
        let conv_x = weight(&mut graph, "qconv_x", int8_tensor(&[1, 1, 1, 2], &[2, 4]));
        let conv_w = weight(&mut graph, "qconv_w", int8_tensor(&[1, 1, 1, 2], &[1, 2]));

        let mut results = Vec::new();
        // Shape-only chain with a runtime user.
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(w), ints(&[3, 2])],
            [3, 2],
            "torch.int8",
        );
        let permute = call(
            &mut graph,
            "permute",
            PERMUTE,
            vec![node(view), ints(&[1, 0])],
            [2, 3],
            "torch.int8",
        );
        results.push(call(
            &mut graph,
            "add_x",
            ADD,
            vec![node(x), node(permute)],
            [2, 3],
            "torch.int8",
        ));
        // Factories.
        let mut own = NodeMeta::tensor([2], "torch.float32");
        own.set_single_output_qparam(int8_qparam(0.5, 1));
        results.push(full(&mut graph, "full_own", &[2], 3.0, own));
        let by_user = full(
            &mut graph,
            "full_by_user",
            &[2],
            3.0,
            NodeMeta::tensor([2], "torch.float32"),
        );
        results.push(call_with_meta(
            &mut graph,
            "add_q",
            ADD,
            vec![node(by_user), node(q)],
            NodeMeta::tensor([2], "torch.float32"),
        ));
        let arange = call(
            &mut graph,
            "arange",
            "aten.arange.start_step",
            vec![IrArg::Int(1), IrArg::Int(7), IrArg::Int(2)],
            [3],
            "torch.int64",
        );
        graph.set_kwarg(arange, "dtype", IrArg::Dtype("torch.int64".to_string()));
        results.push(arange);
        let linspace = call(
            &mut graph,
            "linspace",
            "aten.linspace.default",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)],
            [5],
            "torch.float32",
        );
        graph.set_kwarg(linspace, "dtype", IrArg::Dtype("torch.float32".to_string()));
        let linspace_view = call(
            &mut graph,
            "linspace_view",
            VIEW,
            vec![node(linspace), ints(&[5, 1])],
            [5, 1],
            "torch.float32",
        );
        results.push(linspace_view);
        let eye = call(
            &mut graph,
            "eye",
            "aten.eye.default",
            vec![IrArg::Int(2)],
            [2, 2],
            "torch.float32",
        );
        graph.set_kwarg(eye, "dtype", IrArg::Dtype("torch.float32".to_string()));
        results.push(eye);
        results.push(call(
            &mut graph,
            "scalar",
            "aten.scalar_tensor.default",
            vec![IrArg::Int(9)],
            Vec::<i64>::new(),
            "torch.int32",
        ));
        // Data slice, cast, comparison, reciprocal.
        results.push(call(
            &mut graph,
            "slice",
            SLICE,
            vec![node(w2), IrArg::Int(0), IrArg::Int(1), IrArg::Int(3)],
            [2],
            "torch.int32",
        ));
        let cast = call(
            &mut graph,
            "cast",
            "aten._to_copy.default",
            vec![node(lifted)],
            [2],
            "torch.int16",
        );
        graph.set_kwarg(cast, "dtype", IrArg::Dtype("torch.int16".to_string()));
        results.push(cast);
        results.push(call(
            &mut graph,
            "gt",
            "aten.gt.Tensor",
            vec![node(fa), node(fb)],
            [2],
            "torch.bool",
        ));
        let mut recip = NodeMeta::tensor([1, 1], "torch.float32");
        recip.set_single_output_qparam(int8_qparam(0.5, 0));
        results.push(call_with_meta(
            &mut graph,
            "recip",
            "aten.reciprocal.default",
            vec![node(divisor)],
            recip,
        ));
        // Quantized mm, linear and conv.
        let quantized = |input_scale: f64, weight_scale: f64, output_scale: f64| {
            let mut meta = NodeMeta::default();
            meta.set_binary_input_output_qparams(
                int8_qparam(input_scale, 0),
                int8_qparam(weight_scale, 0),
                int8_qparam(output_scale, 0),
            );
            meta
        };
        results.push(call_with_meta(
            &mut graph,
            "qmm",
            "aten.mm.default",
            vec![node(lhs), node(rhs)],
            quantized(0.5, 0.5, 0.25),
        ));
        results.push(call_with_meta(
            &mut graph,
            "qlinear",
            "aten.linear.default",
            vec![node(lhs), node(lin_w), IrArg::None],
            quantized(0.5, 0.5, 0.25),
        ));
        results.push(call_with_meta(
            &mut graph,
            "qconv",
            "aten.convolution.default",
            vec![
                node(conv_x),
                node(conv_w),
                IrArg::None,
                ints(&[1, 1]),
                ints(&[0, 0]),
                ints(&[1, 1]),
                IrArg::Bool(false),
                ints(&[0, 0]),
                IrArg::Int(1),
            ],
            quantized(0.5, 0.25, 0.25),
        ));
        output(&mut graph, &results);
        graph
    }

    fn materialized_effects(graph: &mut IrGraph) -> LoweringEffectBatch {
        let mut effects = LoweringEffectBatch::default();
        for result in [
            IrFuseZeroInputConstantsEffectPass.run(graph),
            IrFuseShapeOnlyConstantsEffectPass.run(graph),
            IrFuseConstantInputsEffectPass::with_data_slices().run(graph),
        ] {
            effects.items.extend(result.effects.items);
        }
        effects
    }

    /// The node names `run` folded, keyed by the constant that replaced them.
    fn folded_names(graph: &IrGraph) -> HashMap<String, String> {
        graph
            .live_nodes()
            .filter(|node| is_constant(node))
            .filter_map(|node| {
                let source = node
                    .name
                    .strip_suffix("_fused_const")
                    .or_else(|| node.name.strip_suffix("_pre_computed"))?;
                Some((node.name.clone(), source.to_string()))
            })
            .collect()
    }

    /// The pass's constants hold exactly what ingest stores for the effects
    /// the planners plan on the same graph, after `apply_effects` and the
    /// int64 narrowing, with the same converted value, and every folded node
    /// the outputs reach in the dim-order route is a constant here.
    #[test]
    fn constants_hold_what_ingest_stores_for_the_planners_effects() {
        let mut deferred = every_planner_graph();
        let effects = materialized_effects(&mut deferred);
        let mut ingest_store = TensorStore::new();
        let mut ingested = crate::ingest::from_ir::graph_from_ir(&deferred, &mut ingest_store);
        crate::ingest::from_ir::apply_effects(&mut ingested, &mut ingest_store, &effects);

        let mut materialized = every_planner_graph();
        assert!(run(&mut materialized));
        let mut store = TensorStore::new();
        let converted = crate::ingest::from_ir::graph_from_ir(&materialized, &mut store);

        let folded = folded_names(&materialized);
        let mut expected: Vec<&str> = vec![
            "permute",
            "full_own",
            "full_by_user",
            "arange",
            "linspace_view",
            "eye",
            "scalar",
            "slice",
            "cast",
            "gt",
            "recip",
            "qmm",
            "qlinear",
            "qconv",
        ];
        let mut actual: Vec<&str> = folded.values().map(String::as_str).collect();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected);

        let converted_val = |graph: &crate::graph::Graph, name: &str| {
            let (id, _) = graph
                .node_names
                .iter()
                .find(|(id, node_name)| node_name == name && graph.order.contains(id))
                .unwrap_or_else(|| panic!("{name} is not in the converted graph"));
            let val = &graph.nodes[*id].meta.val;
            (val.shape_slice().to_vec(), val.dtype)
        };
        for (constant, source) in &folded {
            let stored = store.get(constant).expect("the constant's data is stored");
            let ingest = ingest_store
                .get(source)
                .expect("ingest materialized the source");
            assert_tensor_eq(stored, ingest, source);
            assert_tensor_eq(tensor(&materialized, constant), ingest, source);
            assert_eq!(
                converted_val(&converted, constant),
                converted_val(&ingested, source),
                "{source}'s converted value"
            );
        }
        // Spot checks of the planners' arithmetic.
        let data = |name: &str| {
            store
                .get(&format!("{name}_fused_const"))
                .or_else(|| store.get(&format!("{name}_pre_computed")))
                .unwrap()
                .clone()
        };
        assert_tensor_eq(
            &data("full_own"),
            &int8_tensor(&[2], &[7, 7]),
            "3.0 at scale 0.5, zp 1",
        );
        assert_tensor_eq(
            &data("full_by_user"),
            &int8_tensor(&[2], &[12, 12]),
            "3.0 at q's scale 0.25",
        );
        assert_tensor_eq(
            &data("arange"),
            &int32_tensor(&[3], &[1, 3, 5]),
            "int64 narrowed to int32",
        );
        assert_tensor_eq(
            &data("qconv"),
            &int8_tensor(&[1, 1, 1, 1], &[5]),
            "1.25 at scale 0.25",
        );
    }

    /// Ingest applies every effect an earlier pass planned under the folded
    /// nodes' names too. Those nodes are gone, so the effects find nothing
    /// to act on and the constants keep their data.
    #[test]
    fn effects_planned_under_folded_names_are_skipped_at_ingest() {
        let mut graph = every_planner_graph();
        let early_effects = materialized_effects(&mut graph);
        assert!(run(&mut graph));
        let mut store = TensorStore::new();
        let mut converted = crate::ingest::from_ir::graph_from_ir(&graph, &mut store);
        let before: Vec<(String, Vec<u8>)> = folded_names(&graph)
            .keys()
            .map(|name| (name.clone(), store.get(name).unwrap().data.clone()))
            .collect();
        let order_before = converted.order.len();
        crate::ingest::from_ir::apply_effects(&mut converted, &mut store, &early_effects);
        assert_eq!(converted.order.len(), order_before);
        for (name, data) in before {
            assert_eq!(store.get(&name).unwrap().data, data, "{name}");
        }
    }

    // -----------------------------------------------------------------------
    // Regions
    // -----------------------------------------------------------------------

    /// In a region the constants are `get_attr` nodes, the placeholders stay
    /// exactly as they were, and ingest keeps the constants out of the
    /// region's runtime inputs while storing their data.
    #[test]
    fn region_constants_are_get_attrs_and_leave_the_placeholders_alone() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [4], "torch.int8");
        let y = placeholder(&mut graph, "y", [4], "torch.int8");
        let mut meta = NodeMeta::tensor([2, 2], "torch.float32");
        meta.set_single_output_qparam(int8_qparam(0.5, 0));
        let f = full(&mut graph, "f", &[2, 2], 1.0, meta);
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(f), ints(&[4])],
            [4],
            "torch.int8",
        );
        let add = call(
            &mut graph,
            "add",
            ADD,
            vec![node(x), node(view)],
            [4],
            "torch.int8",
        );
        output(&mut graph, &[add, y]);

        assert!(run_in_region(&mut graph));
        assert_eq!(
            names(&graph),
            ["view_fused_const", "x", "y", "add", "output"]
        );
        let constant = graph.node(live(&graph, "view_fused_const"));
        assert_eq!(
            (constant.op, constant.target.as_str()),
            (NodeOp::GetAttr, "view_fused_const")
        );
        assert_eq!(kind(&graph, "view_fused_const"), "attr");
        assert_tensor_eq(
            tensor(&graph, "view_fused_const"),
            &int8_tensor(&[4], &[2, 2, 2, 2]),
            "region fold",
        );
        assert_eq!(placeholder_names(&graph), ["x", "y"]);

        let mut store = TensorStore::new();
        let converted =
            crate::ingest::from_ir::graph_from_ir_preserve_placeholders(&graph, &mut store);
        let inputs: Vec<&str> = converted
            .inputs
            .iter()
            .map(|&id| converted.nodes[id].name.as_str())
            .collect();
        assert_eq!(inputs, ["x", "y"]);
        assert!(store.get("view_fused_const").is_some());
    }

    /// Nothing planned, nothing changed: no renumbering either.
    #[test]
    fn a_graph_without_constants_is_left_alone() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2], "torch.int8");
        let neg = call(
            &mut graph,
            "neg",
            "aten.neg.default",
            vec![node(x)],
            [2],
            "torch.int8",
        );
        let relu = call(
            &mut graph,
            "relu",
            "aten.relu.default",
            vec![node(x)],
            [2],
            "torch.int8",
        );
        graph.anchor_to(relu, neg);
        output(&mut graph, &[neg, relu]);
        let ids = graph.live_node_ids();
        assert!(!run(&mut graph));
        assert_eq!(graph.live_node_ids(), ids);
        assert_eq!(
            names(&graph),
            ["x", "relu", "neg", "output"],
            "the anchor still applies"
        );
    }

    #[test]
    fn helpers_classify_constants() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1], "torch.int8");
        let p = constant(
            &mut graph,
            "p",
            int8_tensor(&[1], &[1]),
            LoweringInputKind::Parameter,
        );
        let attr = graph.add_constant_node(
            NodeOp::GetAttr,
            "attr".to_string(),
            "attr".to_string(),
            crate::fx_ir::NodeTensor {
                source: TensorSource::GraphModuleAttr,
                data: int8_tensor(&[1], &[2]),
            },
            NodeMeta::tensor([1], "torch.int8"),
        );
        let view = call(
            &mut graph,
            "view",
            VIEW,
            vec![node(p), ints(&[1])],
            [1],
            "torch.int8",
        );
        assert!(!is_constant(graph.node(x)) && !is_constant(graph.node(view)));
        assert!(is_constant(graph.node(p)) && is_constant(graph.node(attr)));
        assert_eq!(constant_kind(graph.node(p)), LoweringInputKind::Parameter);
        assert_eq!(constant_kind(graph.node(attr)), LoweringInputKind::Buffer);
        graph.erase(view);
        graph.erase(p);
        assert!(!is_constant(graph.node(p)), "an erased constant is not one");
    }
}
