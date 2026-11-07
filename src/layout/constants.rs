//! The pipeline's stand-in for rc4's ComputeConstantOpsAOTPass and
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
//! applies it (`ingest::from_ir::apply_effects`), so the effect passes leave the
//! call nodes in the IR until conversion. [`IrMaterializeConstantsPass`] runs
//! the same planners at rc4's slot and applies their plan to the IR instead:
//!
//! - **What** becomes a constant, and which bytes it holds, is the planners'
//!   decision. The pass runs IrFuseZeroInputConstantsEffectPass (rc4's
//!   ComputeConstantOpsAOT share) and
//!   IrFuseConstantInputsEffectPass::with_data_slices (FuseConstantArgs's
//!   share; the variant that also plans everything
//!   IrFuseShapeOnlyConstantsEffectPass plans),
//!   and gives each planned node the tensor ingest would store under its name
//!   (`ingest::from_ir::materialized_tensor`, which narrows int64 data to
//!   int32 when it fits). The planners' rc4 quirks, such as quantizing a
//!   factory with its output qparams, come along. One rule is rc4's rather
//!   than the planners': a `full` that carries qparam maps is quantized only
//!   with its output qparams, as ComputeConstantOpsAOT does (`:327-335`),
//!   where the planners, as the effect passes run them, fall back to its
//!   input qparams ([`FullQParams::Rc4`]). Both planners apply it, so a fold
//!   over such a `full` reads the bytes the `full` is materialized with.
//! - **How** the IR changes follows rc4: the names, the constant kinds and
//!   insert points, the special-dtype propagation, the folds rc4 skips or
//!   abandons, the graph-wide dead-code elimination and the deletion of the
//!   constant placeholders the folds leave unused.
//!
//! Each new constant gets a new name, so an effect an earlier pass planned
//! under the folded node's name finds no live node at ingest and is skipped
//! (P2 design F9); the constant already holds the same data. Nodes the
//! planners leave alone (targets they do not fold, transposed convs, ...)
//! stay call nodes, which the effect passes that run later can still plan for
//! ingest.
//!
//! In a region the constants are `get_attr` nodes (see
//! [`GraphScope::Region`]) and no placeholder is added or deleted, since the
//! boundary contracts pair region placeholders with the parent's operands by
//! position (critique B2). Ingest tells such a constant from a branch
//! reference by its data, so a folded cond predicate stays a tensor.

use std::collections::HashMap;

use crate::fx_fuse_constant_inputs_ir::IrFuseConstantInputsEffectPass;
use crate::fx_fuse_zero_input_constants_ir::{zero_input_constant_effects, FullQParams};
use crate::fx_ir::{
    EffectfulIrPass, EffectfulIrPassResult, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeOp,
    TensorSource,
};
use crate::graph::ir::DType;
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
/// (see the module docs). `in_region` is the pipeline's
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
        let placeholders_before =
            (scope == GraphScope::Region).then(|| graph.live_placeholder_names());

        let mut ctx = LayoutCtx::with_scope(graph, scope);
        // Each rc4 pass that changes the graph ends with a graph-wide
        // `eliminate_dead_code()` (`:369-372`, `:278-280`), which drops the
        // folded nodes. ComputeConstantOpsAOT's runs before FuseConstantArgs
        // sweeps, so that sweep never folds a dead op.
        let precomputed = compute_constant_ops_aot(&mut ctx, &mut plan.factories);
        if precomputed {
            ctx.eliminate_dead_code();
        }
        let fused = fuse_constant_args(&mut ctx, &mut plan.folds);
        if !fused.inputs.is_empty() {
            ctx.eliminate_dead_code();
            if scope == GraphScope::Parent {
                delete_unused_constant_inputs(&mut ctx, fused.inputs);
            }
        } else if !precomputed && !fused.left_unused_constants {
            // Every planned fold was skipped; nothing changed.
            return false;
        }
        // An abandoned fold alone still changed the graph, although rc4
        // reports `modified=False` and so skips the dead-code elimination.
        ctx.finish();

        if let Some(before) = placeholders_before {
            debug_assert_eq!(
                graph.live_placeholder_names(),
                before,
                "a region's placeholders must stay as they are"
            );
        }
        true
    }
}

/// The planners' constants by node: rc4's two passes' shares.
struct Plan {
    /// ComputeConstantOpsAOT's share: zero-input factories.
    factories: HashMap<NodeId, TensorData>,
    /// FuseConstantArgs's share: ops whose inputs are all constant.
    folds: HashMap<NodeId, TensorData>,
}

impl Plan {
    /// Run the effect planners on `graph`, which they only read, with rc4's
    /// rule for `full` ([`FullQParams::Rc4`]), and key their
    /// MaterializeTensorNode effects by node.
    ///
    /// IrFuseShapeOnlyConstantsEffectPass needs no run of its own:
    /// IrFuseConstantInputsEffectPass plans every node it plans, with the same
    /// bytes. Both seed the same constants and factories, the constant-inputs
    /// planner counts at least as many nodes constant, and a shape-only target
    /// reads only its first argument (a test pins this). matmul_only is
    /// likewise the subset of with_data_slices without slices.
    fn of(graph: &mut IrGraph) -> Self {
        let factories = zero_input_constant_effects(graph, FullQParams::Rc4);
        let folds = IrFuseConstantInputsEffectPass::with_data_slices()
            .with_full_qparams(FullQParams::Rc4)
            .run(graph);

        let ids = live_ids_by_name(graph);
        Self {
            factories: planned_constants(factories, &ids),
            folds: planned_constants(folds, &ids),
        }
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

/// Each MaterializeTensorNode effect's node and the tensor ingest stores for
/// it. A node planned twice keeps the later plan, as ingest applies effects in
/// order.
fn planned_constants(
    result: EffectfulIrPassResult,
    ids: &HashMap<String, Option<NodeId>>,
) -> HashMap<NodeId, TensorData> {
    let mut planned = HashMap::new();
    for effect in result.effects.items {
        let LoweringEffect::MaterializeTensorNode(effect) = effect else {
            continue;
        };
        if let Some(&Some(id)) = ids.get(&effect.node_name) {
            planned.insert(
                id,
                materialized_tensor(effect.data, effect.shape, effect.dtype),
            );
        }
    }
    planned
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

/// What FuseConstantArgs's sweep did.
struct Fused {
    /// The folded nodes' inputs, rc4's `input_nodes_to_maybe_delete`.
    inputs: Vec<NodeId>,
    /// Whether an abandoned fold left its unused constant behind.
    left_unused_constants: bool,
}

/// FuseConstantArgsPass.call (`:228-285`) over the planned folds, in graph
/// order: each becomes `<node>_fused_const`, of its first input's kind and
/// inserted directly before that input (`:192-207`), with the inputs' special
/// dtype (`:209`). A fold in a chain reads the constant that replaced its
/// producer, so a chain ends as one constant where the chain's source was.
///
/// rc4 folds a node only when every input is a constant by the time the sweep
/// reaches it. It never folds a node marked SHAPE (`:155-157`), nor one whose
/// args or kwargs hold a SHAPE-marked node (`_arg_contains_symbolic_shape`,
/// `:243-246`; its symbolic-value cases have no counterpart in RETAB's static
/// IR). The planners' plan meets the first rule except below a skipped node:
/// the shape-only planner folds SHAPE-marked nodes, and the constant-inputs
/// planner folds their users. Such nodes, and everything planned on top of
/// them, stay call nodes here, and their deferred effects still fold them at
/// ingest.
///
/// When the special-dtype check raises (see [`propagated_special_dtype`]),
/// rc4 logs the exception and abandons the fold (`:267-276`): the op stays,
/// and so does the unused constant `_fuse_nodes` had already created.
fn fuse_constant_args(ctx: &mut LayoutCtx<'_>, planned: &mut HashMap<NodeId, TensorData>) -> Fused {
    let mut fused = Fused {
        inputs: Vec::new(),
        left_unused_constants: false,
    };
    for id in ctx.order().to_vec() {
        let Some(data) = planned.remove(&id) else {
            continue;
        };
        let inputs = ctx.all_input_nodes(id);
        let Some(&first) = inputs.first() else {
            continue;
        };
        if is_shape_marked(ctx.node(id))
            || inputs.iter().any(|&input| is_shape_marked(ctx.node(input)))
            || !inputs.iter().all(|&input| is_constant(ctx.node(input)))
        {
            continue;
        }
        let name = format!("{}_fused_const", ctx.node(id).name);
        let kind = constant_kind(ctx.node(first));
        let special_dtype = propagated_special_dtype(ctx.graph(), &inputs, &data);
        let constant = ctx.inserting_before(first, |ctx| ctx.create_constant(name, kind, data));
        match special_dtype {
            Ok(Some(special_dtype)) => {
                ctx.meta_mut(constant)
                    .set_extra(SPECIAL_DTYPE, MetaValue::Str(special_dtype));
            }
            Ok(None) => {}
            Err(error) => {
                crate::runner_logging::log_line(format_args!(
                    "[IrMaterializeConstants] failed to fuse constant op {} due to \
                     exception: {error}",
                    ctx.node(id).name
                ));
                fused.left_unused_constants = true;
                continue;
            }
        }
        ctx.replace_all_uses_with(id, constant);
        fused.inputs.extend(inputs);
    }
    fused
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
/// data passes rc4's range check ([`passes_range_check`]). This is how an
/// INT4 weight's fold stays INT4, which RewriteConvPass then copies onto the
/// conv-local weight.
///
/// `Err` is the exception the check raises, which makes rc4 abandon the fold:
/// torch's error for bool, empty or float8 data, or `TosaSpecialDtype.max()`'s
/// for a special dtype other than INT4 and INT48, which an INT graph never
/// carries.
fn propagated_special_dtype(
    graph: &IrGraph,
    inputs: &[NodeId],
    data: &TensorData,
) -> Result<Option<String>, String> {
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
        return Ok(None);
    };
    let bound = match special_dtype_name(raw) {
        "INT4" => 7,
        "INT48" => (1 << 47) - 1,
        _ => return Err(format!("Unrecognized TosaSpecialDtype {raw}.")),
    };
    Ok(passes_range_check(data, bound)?.then(|| (*raw).to_string()))
}

/// rc4's range check `data.abs().max() <= special_dtype.max()` (`:146`), with
/// `bound` the special dtype's max and torch's semantics, which compute in
/// the data's dtype:
///
/// - `abs` wraps a signed integer dtype's minimum to itself, which passes
///   any bound;
/// - the bound converts to the data's dtype, wrapping too: for int8, int16
///   and int32 data INT48's bound becomes -1, which only the minimum passes
///   (rc4 keeps INT48 on int64 data, not on int32 data);
/// - a float bound rounds (INT48's to 2**47 in float32 and bfloat16) or
///   overflows (float16), and a NaN fails the check.
///
/// `Err` is torch's error for the data that `abs` or `max` rejects.
fn passes_range_check(data: &TensorData, bound: i64) -> Result<bool, String> {
    if data.dtype == DType::Bool {
        return Err("\"abs_cpu\" not implemented for 'Bool'".to_string());
    }
    if data.element_count() == 0 {
        return Err(
            "max(): Expected reduction dim to be specified for input.numel() == 0".to_string(),
        );
    }
    let bits = match data.dtype {
        DType::Int8 => 8,
        DType::Int16 => 16,
        DType::Int32 => 32,
        DType::Int64 => 64,
        DType::Float32 | DType::BFloat16 | DType::Float16 => {
            let bound = match data.dtype {
                DType::Float16 if bound > 65504 => f64::INFINITY,
                _ => f64::from(bound as f32),
            };
            let values = data.vec_f64().expect("float data decodes");
            return Ok(values.iter().all(|value| value.abs() <= bound));
        }
        dtype => return Err(format!("\"max_all\" not implemented for {dtype:?}")),
    };
    let wrap = |value: i64| (value << (64 - bits)) >> (64 - bits);
    let values = data.vec_i64().expect("integer data decodes");
    Ok(values
        .iter()
        .all(|&value| wrap(value.wrapping_abs()) <= wrap(bound)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::fx_fuse_shape_only_constants_ir::IrFuseShapeOnlyConstantsEffectPass;
    use crate::fx_fuse_zero_input_constants_ir::IrFuseZeroInputConstantsEffectPass;
    use crate::fx_ir::{IrArg, NodeMeta, QParam};
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

    fn int64_tensor(shape: &[i64], values: &[i64]) -> TensorData {
        TensorData {
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            dtype: DType::Int64,
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

    /// rc4's ComputeConstantOpsAOT quantizes a factory only with its own
    /// output qparams and otherwise evaluates it with its kwargs (`:327-335`).
    /// rc4's passes copy their node's meta, qparams included, onto the `full`
    /// nodes they create (`ArmPass.call_scalar`), so the pass gives that rule
    /// to every `full` with qparam maps, empty ones included
    /// ([`FullQParams::Rc4`]). The planners, as the effect passes run them,
    /// quantize such a `full` with its input qparams or, with empty maps, with
    /// its users' (APU-156 review). A bare `full`, as RETAB's decompositions
    /// create them, keeps that inference here too, and a fold over a `full`
    /// holds the bytes the `full` gets.
    #[test]
    fn a_full_with_qparam_maps_is_quantized_only_with_its_output_qparams() {
        /// `full((2,), 3.0, dtype=torch.int8)`, as `call_scalar` spells it.
        fn int8_full(graph: &mut IrGraph, name: &str, meta: NodeMeta) -> NodeId {
            let id = full(graph, name, &[2], 3.0, meta);
            graph.set_kwarg(id, "dtype", IrArg::Dtype("torch.int8".to_string()));
            id
        }
        let build = || {
            let mut graph = IrGraph::empty();
            let q = placeholder(&mut graph, "q", [2], "torch.int8");
            graph
                .node_mut(q)
                .meta
                .set_single_output_qparam(int8_qparam(0.25, 0));
            let mut with_output = NodeMeta::tensor([2], "torch.int8");
            with_output.set_single_output_qparam(int8_qparam(0.5, 1));
            let mut with_input = NodeMeta::tensor([2], "torch.int8");
            with_input.set_single_input_qparam(int8_qparam(0.5, 1));
            let mut empty = NodeMeta::tensor([2], "torch.int8");
            empty.set_empty_qparam_maps();
            let bare = NodeMeta::tensor([2], "torch.int8");

            let mut results = vec![
                int8_full(&mut graph, "with_output", with_output),
                int8_full(&mut graph, "with_input", with_input.clone()),
            ];
            let folded = int8_full(&mut graph, "folded", with_input);
            results.push(call(
                &mut graph,
                "view",
                VIEW,
                vec![node(folded), ints(&[2, 1])],
                [2, 1],
                "torch.int8",
            ));
            // Each with a binary user whose other operand has qparams.
            for (name, meta) in [("empty", empty), ("bare", bare)] {
                let factory = int8_full(&mut graph, name, meta);
                results.push(call_with_meta(
                    &mut graph,
                    &format!("add_{name}"),
                    ADD,
                    vec![node(factory), node(q)],
                    NodeMeta::tensor([2], "torch.float32"),
                ));
            }
            output(&mut graph, &results);
            graph
        };

        // quantize(3.0, 0.5, 1) = 7 and quantize(3.0, 0.25, 0) = 12.
        let mut graph = build();
        let factories = effects_by_name(IrFuseZeroInputConstantsEffectPass.run(&mut graph));
        let folds =
            effects_by_name(IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph));
        for (name, values) in [
            ("with_output", [7, 7]),
            ("with_input", [7, 7]),
            ("empty", [12, 12]),
            ("bare", [12, 12]),
        ] {
            assert_eq!(factories[name].0, values, "the effect pass: {name}");
        }
        assert_eq!(folds["view"].0, [7, 7], "the effect pass: view");

        let mut graph = build();
        assert!(run(&mut graph));
        for (name, values) in [
            ("with_output", [7, 7]),
            ("with_input", [3, 3]),
            ("empty", [3, 3]),
            ("bare", [12, 12]),
        ] {
            assert_tensor_eq(
                tensor(&graph, &format!("{name}_pre_computed")),
                &int8_tensor(&[2], &values),
                name,
            );
        }
        assert_tensor_eq(
            tensor(&graph, "view_fused_const"),
            &int8_tensor(&[2, 1], &[3, 3]),
            "the fold over `folded`",
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

    /// ComputeConstantOpsAOT's dead-code elimination runs before
    /// FuseConstantArgs sweeps. With a factory in the graph a dead fold is
    /// dropped, not folded, and its input placeholder stays; without one, rc4
    /// folds the dead op too, deletes its input and keeps the unused constant.
    #[test]
    fn a_dead_fold_is_dropped_once_a_factory_was_materialized() {
        let build = |with_factory: bool| {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2], "torch.int8");
            let w = weight(&mut graph, "w", int8_tensor(&[2], &[1, 2]));
            call(
                &mut graph,
                "dead_view",
                VIEW,
                vec![node(w), ints(&[2, 1])],
                [2, 1],
                "torch.int8",
            );
            let mut results = vec![x];
            if with_factory {
                let mut meta = NodeMeta::tensor([1], "torch.float32");
                meta.set_single_output_qparam(int8_qparam(1.0, 0));
                results.push(full(&mut graph, "f", &[1], 3.0, meta));
            }
            output(&mut graph, &results);
            graph
        };

        let mut graph = build(true);
        assert!(run(&mut graph));
        assert_eq!(names(&graph), ["f_pre_computed", "x", "w", "output"]);

        let mut graph = build(false);
        assert!(run(&mut graph));
        assert_eq!(names(&graph), ["x", "dead_view_fused_const", "output"]);
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

    /// `_propagate_special_dtype`, as in the rc4 venv: an INT4 weight's fold
    /// is INT4 while its values fit [-7, 7], and INT48 carries over onto int64
    /// data. Data beyond INT4's range, INT48 on int32 data (rc4 compares in
    /// int32, where INT48's bound wraps to -1) and mixed special dtypes get
    /// none. A bool fold with mixed special dtypes never reaches the range
    /// check, so it is fused.
    #[test]
    fn the_inputs_special_dtype_carries_over_when_the_data_fits() {
        let mut graph = IrGraph::empty();
        let w4 = weight(&mut graph, "w4", int8_tensor(&[2, 2], &[-7, 0, 3, 7]));
        marked(&mut graph, w4, "INT4");
        let wide = weight(&mut graph, "wide", int8_tensor(&[2], &[8, 1]));
        marked(&mut graph, wide, "INT4");
        let w48 = weight(&mut graph, "w48", int32_tensor(&[2], &[1, 2]));
        marked(&mut graph, w48, "TosaSpecialDtype.INT48");
        let w48_64 = weight(&mut graph, "w48_64", int64_tensor(&[2], &[1 << 40, 2]));
        marked(&mut graph, w48_64, "INT48");
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
        let v48_64 = call(
            &mut graph,
            "v48_64",
            VIEW,
            vec![node(w48_64), ints(&[1, 2])],
            [1, 2],
            "torch.int64",
        );
        let mixed = call(
            &mut graph,
            "mixed",
            "aten.gt.Tensor",
            vec![node(w48), node(wide)],
            [2],
            "torch.bool",
        );
        output(&mut graph, &[v4, v_wide, v48, v48_64, mixed]);

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
            special_dtype(&graph, "v48_fused_const"),
            None,
            "INT48's bound wraps to -1 in int32"
        );
        assert_eq!(
            special_dtype(&graph, "v48_64_fused_const").as_deref(),
            Some("INT48")
        );
        assert_eq!(
            special_dtype(&graph, "mixed_fused_const"),
            None,
            "INT48 and INT4"
        );
    }

    /// The range check against the rc4 venv (`data.abs().max() <=
    /// special_dtype.max()` on each tensor): integer data compares in its own
    /// dtype, where `abs` of the minimum and INT48's bound wrap; a float bound
    /// rounds or overflows; bool, empty and float8 data raise.
    #[test]
    fn the_range_check_follows_torch() {
        use crate::tensor_data::{encode_bf16, encode_f16};

        const INT4: i64 = 7;
        const INT48: i64 = (1 << 47) - 1;
        let raw = |dtype: DType, data: Vec<u8>, len: usize| TensorData {
            data,
            shape: vec![len as i64],
            dtype,
        };
        let i8s = |values: &[i8]| int8_tensor(&[values.len() as i64], values);
        let i16s = |values: &[i16]| {
            let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            raw(DType::Int16, data, values.len())
        };
        let i32s = |values: &[i32]| int32_tensor(&[values.len() as i64], values);
        let i64s = |values: &[i64]| int64_tensor(&[values.len() as i64], values);
        let f32s = |values: &[f32]| f32_tensor(&[values.len() as i64], values);
        let half = |dtype: DType, bits: u16| raw(dtype, bits.to_le_bytes().to_vec(), 1);
        let f16 = |value: f32| half(DType::Float16, encode_f16(value));
        let bf16 = |value: f32| half(DType::BFloat16, encode_bf16(value));
        let cases = [
            ("int8 [-7, 7]", i8s(&[-7, 7]), INT4, true),
            ("int8 [-8]", i8s(&[-8]), INT4, false),
            ("int8 [-128, 3]", i8s(&[-128, 3]), INT4, true),
            ("int8 [1, 2]", i8s(&[1, 2]), INT48, false),
            ("int8 [-128]", i8s(&[-128]), INT48, true),
            ("int16 [1, -32768]", i16s(&[1, -32768]), INT48, false),
            ("int32 [1, 2]", i32s(&[1, 2]), INT48, false),
            ("int64 [1, 2]", i64s(&[1, 2]), INT48, true),
            ("int64 [2**50]", i64s(&[1 << 50]), INT48, false),
            ("int64 [min]", i64s(&[i64::MIN]), INT4, true),
            ("f32 [8]", f32s(&[8.0]), INT4, false),
            ("f32 [8]", f32s(&[8.0]), INT48, true),
            ("f32 [nan, 1]", f32s(&[f32::NAN, 1.0]), INT48, false),
            ("f32 [inf]", f32s(&[f32::INFINITY]), INT48, false),
            ("f32 [2**47]", f32s(&[2f32.powi(47)]), INT48, true),
            ("bf16 [inf]", bf16(f32::INFINITY), INT48, false),
            ("f16 [inf]", f16(f32::INFINITY), INT48, true),
            ("f16 [8]", f16(8.0), INT4, false),
        ];
        for (what, data, bound, expected) in cases {
            assert_eq!(
                passes_range_check(&data, bound),
                Ok(expected),
                "{what} against {bound}"
            );
        }
        let raising = [
            ("bool", raw(DType::Bool, vec![1], 1)),
            ("empty", i8s(&[])),
            ("float8", raw(DType::Float8E4M3, vec![0x38], 1)),
        ];
        for (what, data) in raising {
            assert!(passes_range_check(&data, INT4).is_err(), "{what}");
        }
    }

    /// When the range check raises (bool data, empty data), rc4 abandons the
    /// fold after creating its constant, as in the rc4 venv (`p_w4,
    /// aten_gt_tensor_1_fused_const, p_w4b`): the op stays a call node, and
    /// the unused `_fused_const` stays directly before the op's first input,
    /// without a special dtype. With no fold done rc4 reports `modified=False`,
    /// so no dead code goes either.
    #[test]
    fn an_abandoned_fold_leaves_the_op_and_its_unused_constant_like_rc4() {
        let mut graph = IrGraph::empty();
        let w4 = weight(&mut graph, "w4", int8_tensor(&[2], &[-7, 3]));
        marked(&mut graph, w4, "INT4");
        let w48 = weight(&mut graph, "w48", int32_tensor(&[2], &[1, 2]));
        marked(&mut graph, w48, "TosaSpecialDtype.INT48");
        let other = weight(&mut graph, "other", int8_tensor(&[2], &[1, 2]));
        let x = placeholder(&mut graph, "x", [2], "torch.int8");
        call(
            &mut graph,
            "dead_neg",
            "aten.neg.default",
            vec![node(x)],
            [2],
            "torch.int8",
        );
        let gt4 = call(
            &mut graph,
            "gt4",
            "aten.gt.Tensor",
            vec![node(other), node(w4)],
            [2],
            "torch.bool",
        );
        let gt48 = call(
            &mut graph,
            "gt48",
            "aten.gt.Tensor",
            vec![node(w48), node(other)],
            [2],
            "torch.bool",
        );
        let empty = call(
            &mut graph,
            "empty",
            SLICE,
            vec![node(w4), IrArg::Int(0), IrArg::Int(1), IrArg::Int(1)],
            [0],
            "torch.int8",
        );
        output(&mut graph, &[x, gt4, gt48, empty]);
        let planned =
            effects_by_name(IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph));
        for name in ["gt4", "gt48", "empty"] {
            assert!(planned.contains_key(name), "{name} is planned");
        }

        assert!(run(&mut graph), "the unused constants are a change");
        assert_eq!(
            names(&graph),
            [
                "empty_fused_const",
                "w4",
                "gt48_fused_const",
                "w48",
                "gt4_fused_const",
                "other",
                "x",
                "dead_neg",
                "gt4",
                "gt48",
                "empty",
                "output"
            ]
        );
        for name in ["gt4_fused_const", "gt48_fused_const", "empty_fused_const"] {
            let constant = live(&graph, name);
            assert!(!graph.has_live_users(constant), "{name} is unused");
            assert_eq!(special_dtype(&graph, name), None, "{name}");
            assert_eq!(kind(&graph, name), "Buffer", "{name}");
        }
        assert_tensor_eq(
            tensor(&graph, "gt4_fused_const"),
            &TensorData {
                data: vec![1, 0],
                shape: vec![2],
                dtype: DType::Bool,
            },
            "1 > -7 and 2 > 3",
        );
        for name in ["gt4", "gt48", "empty"] {
            assert_eq!(graph.node(live(&graph, name)).op, NodeOp::CallFunction);
        }
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

    /// The materializing planners' effects on `graph`, with `full_qparams`
    /// deciding how a `full` is quantized: [`FullQParams::Planner`] for the
    /// effect passes, [`FullQParams::Rc4`] for the pass's plan.
    fn materialized_effects(graph: &mut IrGraph, full_qparams: FullQParams) -> LoweringEffectBatch {
        let mut effects = LoweringEffectBatch::default();
        for result in [
            zero_input_constant_effects(graph, full_qparams),
            IrFuseShapeOnlyConstantsEffectPass.run(graph),
            IrFuseConstantInputsEffectPass::with_data_slices()
                .with_full_qparams(full_qparams)
                .run(graph),
        ] {
            effects.items.extend(result.effects.items);
        }
        effects
    }

    fn effects_by_name(
        effects: EffectfulIrPassResult,
    ) -> HashMap<String, (Vec<u8>, Vec<i64>, DType)> {
        effects
            .effects
            .items
            .into_iter()
            .filter_map(|effect| match effect {
                LoweringEffect::MaterializeTensorNode(effect) => {
                    Some((effect.node_name, (effect.data, effect.shape, effect.dtype)))
                }
                _ => None,
            })
            .collect()
    }

    /// `Plan::of` skips IrFuseShapeOnlyConstantsEffectPass because
    /// IrFuseConstantInputsEffectPass::with_data_slices plans every node it
    /// plans, with the same bytes. Should either planner change, this fails.
    #[test]
    fn the_constant_inputs_planner_covers_the_shape_only_planner() {
        let mut graph = every_planner_graph();
        let shape_only = effects_by_name(IrFuseShapeOnlyConstantsEffectPass.run(&mut graph));
        let constant_inputs =
            effects_by_name(IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph));
        let mut names: Vec<&String> = shape_only.keys().collect();
        names.sort_unstable();
        assert_eq!(names, ["linspace_view", "permute", "view"]);
        for (name, planned) in &shape_only {
            assert_eq!(constant_inputs.get(name), Some(planned), "{name}");
        }
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
    /// the planners plan on the same graph (with the pass's rule for `full`),
    /// after `apply_effects` and the int64 narrowing, with the same converted
    /// value, and every folded node the outputs reach in the effect route is
    /// a constant here.
    #[test]
    fn constants_hold_what_ingest_stores_for_the_planners_effects() {
        let mut deferred = every_planner_graph();
        let effects = materialized_effects(&mut deferred, FullQParams::Rc4);
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
        let early_effects = materialized_effects(&mut graph, FullQParams::Planner);
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
        assert_eq!(graph.live_placeholder_names(), ["x", "y"]);

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

    // -----------------------------------------------------------------------
    // Review (S0a): folds rc4 skips or abandons; region cond predicates
    // -----------------------------------------------------------------------

    /// rc4's FuseConstantArgs skips any node whose args include a SHAPE-marked
    /// node (`_arg_contains_symbolic_shape`, `:243-246`), whether or not the
    /// node itself is marked. Checked in the rc4 venv
    /// (/tmp/s0a_review_probe/shape_input_probe.py): `gt(p_shape[SHAPE], p_w4)`
    /// stays a call node. The planner plans it; the pass leaves it, creating
    /// nothing.
    #[test]
    fn a_node_reading_a_shape_marked_constant_is_not_folded() {
        let mut graph = IrGraph::empty();
        let shape = weight(&mut graph, "shape", int32_tensor(&[2], &[5, 6]));
        marked(&mut graph, shape, "SHAPE");
        let w = weight(&mut graph, "w", int32_tensor(&[2], &[1, 9]));
        let gt = call(
            &mut graph,
            "gt",
            "aten.gt.Tensor",
            vec![node(shape), node(w)],
            [2],
            "torch.bool",
        );
        output(&mut graph, &[gt]);
        let planned =
            effects_by_name(IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph));
        assert!(
            planned.contains_key("gt"),
            "the planner plans the comparison"
        );

        assert!(!run(&mut graph), "nothing changes");
        assert!(
            graph
                .live_nodes()
                .any(|node| node.name == "gt" && node.op == NodeOp::CallFunction),
            "rc4 leaves gt a call node; got {:?}",
            names(&graph)
        );
        assert!(graph
            .live_nodes()
            .all(|node| !node.name.starts_with("gt_fused_const")));
    }

    /// rc4's `_propagate_special_dtype` evaluates `data.abs().max()` on the
    /// folded data. For a bool result torch raises ("abs_cpu" not implemented
    /// for 'Bool'); `call` catches it and the op is not fused (rc4 only leaves
    /// an unused `_fused_const` placeholder behind). Checked in the rc4 venv
    /// (/tmp/s0a_review_probe/shape_input_probe.py): `gt(p_w4b, p_w4[INT4])`
    /// stays a call node. So does the pass's, and no bool constant is marked
    /// INT4, which the serializer would have packed as INT4.
    #[test]
    fn a_bool_fold_with_one_int4_input_is_not_fused_like_rc4() {
        let mut graph = IrGraph::empty();
        let w4 = weight(&mut graph, "w4", int8_tensor(&[2], &[-7, 3]));
        marked(&mut graph, w4, "INT4");
        let other = weight(&mut graph, "other", int8_tensor(&[2], &[1, 2]));
        let gt = call(
            &mut graph,
            "gt",
            "aten.gt.Tensor",
            vec![node(other), node(w4)],
            [2],
            "torch.bool",
        );
        output(&mut graph, &[gt]);

        run(&mut graph);
        assert!(
            graph.live_nodes().all(|node| {
                node.meta.val_dtype() != Some("torch.bool")
                    || node.meta.extra_str(SPECIAL_DTYPE).is_none()
            }),
            "no bool tensor may carry INT4"
        );
        assert!(
            graph
                .live_nodes()
                .any(|node| node.name == "gt" && node.op == NodeOp::CallFunction),
            "rc4 leaves gt a call node; got {:?}",
            names(&graph)
        );
    }

    /// Region constants are `get_attr` nodes, and `graph_from_ir` used to emit
    /// every `get_attr` that is a direct arg of a cond as a region name
    /// (`ingest/from_ir.rs`), so a cond predicate the pass folded inside a
    /// region became the string "pred_pre_computed" and lost its constant. It
    /// now treats only a `get_attr` without data as a region reference, so the
    /// in-IR route keeps the tensor predicate that the effect route gets by
    /// folding the predicate at ingest. Exporting a nested
    /// `torch.cond(torch.full((), True), ...)` inside a branch gives this shape.
    #[test]
    fn a_folded_cond_predicate_in_a_region_stays_a_tensor() {
        use crate::graph::ir::{Arg, OpTarget};

        let build = || {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2], "torch.int8");
            let pred = call(
                &mut graph,
                "pred",
                FULL,
                vec![ints(&[]), IrArg::Bool(true)],
                Vec::<i64>::new(),
                "torch.bool",
            );
            graph.set_kwarg(pred, "dtype", IrArg::Dtype("torch.bool".to_string()));
            let branches: Vec<NodeId> = ["true_graph_0", "false_graph_0"]
                .into_iter()
                .map(|name| {
                    graph.add_node(
                        NodeOp::GetAttr,
                        name.to_string(),
                        name.to_string(),
                        Vec::new(),
                        NodeMeta::default(),
                    )
                })
                .collect();
            let cond = call(
                &mut graph,
                "cond",
                "torch.ops.higher_order.cond",
                vec![
                    node(pred),
                    node(branches[0]),
                    node(branches[1]),
                    IrArg::List(vec![node(x)]),
                ],
                [2],
                "torch.int8",
            );
            output(&mut graph, &[cond]);
            graph
        };
        let predicate = |graph: &crate::graph::Graph, store: &TensorStore| {
            let cond = graph
                .order
                .iter()
                .map(|&id| &graph.nodes[id])
                .find(|node| node.target == OpTarget::CondIf)
                .expect("a cond");
            match &cond.args[0] {
                Arg::Node(id) => {
                    let name = graph.nodes[*id].name.clone();
                    Ok(store.get(&name).map(|tensor| tensor.data.clone()))
                }
                other => Err(format!("{other:?}")),
            }
        };

        // Effect route: the factory's effect folds the predicate at ingest.
        let mut deferred = build();
        let effects = IrFuseZeroInputConstantsEffectPass
            .run(&mut deferred)
            .effects;
        let mut store = TensorStore::new();
        let mut converted =
            crate::ingest::from_ir::graph_from_ir_preserve_placeholders(&deferred, &mut store);
        crate::ingest::from_ir::apply_effects_preserve_placeholders(
            &mut converted,
            &mut store,
            &effects,
        );
        assert_eq!(predicate(&converted, &store), Ok(Some(vec![1])));

        // In-IR route: the pass folds the predicate into a region get_attr.
        let mut materialized = build();
        assert!(run_in_region(&mut materialized));
        let mut store = TensorStore::new();
        let converted =
            crate::ingest::from_ir::graph_from_ir_preserve_placeholders(&materialized, &mut store);
        assert_eq!(predicate(&converted, &store), Ok(Some(vec![1])));
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
