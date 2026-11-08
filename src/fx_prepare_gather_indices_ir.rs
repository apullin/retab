//! Port of rc4's `PrepareGatherIndicesPass`
//! (`backends/arm/_passes/prepare_gather_indices_pass.py`, ExecuTorch #22998).
//!
//! Portable ExecuTorch's gather takes int64 indices, and TOSA's GATHER takes
//! int32 ones. Before partitioning, rc4's ConvertInt64OutputOpsToInt32Pass
//! narrows index producers such as topk to int32 and casts their indices back
//! to int64 in front of a gather. rc4's partitioner delegates that cast along
//! with the gather when the cast is a safe boundary, and this pass then points
//! the gather at the cast's int32 input. It never narrows any other int64
//! tensor.
//!
//! rc4's TopKGather case needs it on TOSA-1.0+INT and Ethos-U85: topk indices,
//! then unsqueeze, expand, the int64 cast and gather. TopK is FP-only, so it
//! stays on the CPU, and the delegate gets the int32 indices and the cast.
//! Without this pass the cast's INT64 tensor reached RETAB's serializer, which
//! refuses it, so RETAB failed where stock lowers the graph (APU-167).
//!
//! A boundary is safe (`is_safe_int32_to_int64_gather_boundary`) when it is a
//! `_to_dim_order_copy` with `dtype=torch.int64` of an int32 node, keeps that
//! node's dim order, and has users that are all compatible gathers
//! (`_is_compatible_gather_user`). A compatible gather has three args with the
//! boundary as the index. Its values and indices are rank 2 or 3, its
//! `dim % rank` is 1, and that values axis is at most INT32_MAX long. The batch
//! sizes match, and rank-3 indices match the values' last axis. The pass
//! rewires a gather (`_is_supported_gather`) whose index is a safe boundary and
//! whose values have a dtype the spec can gather (`_values_dtype_supported`):
//!
//! - bool, int8, int16 and int32 need the INT profile;
//! - float16 and float32 need either profile, since an INT spec gathers them
//!   quantized;
//! - bfloat16, float8_e4m3fn and float8_e5m2 also need their extension, which
//!   TOSA defines for the FP profile only.
//!
//! As in rc4:
//!
//! - The pass visits the gathers in graph order and judges each on the graph
//!   as it is by then. A boundary shared by several gathers stays safe while
//!   gathers move off it, and goes once the last one has.
//! - A rewired gather reads the boundary's input (`gather.replace_input_with`).
//!   When the pass changed something, the whole graph loses its dead code
//!   (`graph.eliminate_dead_code()`), which takes the boundary with it.
//! - It looks only at the graph it is given. rc4 runs its whole pipeline on
//!   each cond and while region, and RETAB runs this pass in each region's
//!   pipeline too.
//!
//! The pipeline runs the port at rc4's slot (`arm_pass_manager.py:548`), after
//! FuseQuantizedActivation (`:545`) and before ConvertToClamp (`:550`). Between
//! those rc4 also runs RewriteBoolToFp32CastViaInt8 and DecomposeTopK
//! (`:546-547`), which never fire for INT specs, and CanonicalizeGather
//! (`:549`), which RETAB runs after the Q/DQ fold. Before the fold a quantized
//! gather's values are a dequantize, so the dtype gate sees float32, as rc4's
//! does.
//!
//! RETAB's targets don't tell an edge op from its ATen op, while rc4 matches
//! only the edge gather and cast. A delegated graph holds edge ops only, so the
//! two agree on every graph RETAB lowers.
//!
//! RETAB differs from rc4 in one way. rc4 leaves a gather it can't prepare
//! alone, and stock's serializer then raises for the cast's int64 tensor.
//! RETAB can't lower that tensor either, and it also can't judge a boundary
//! whose dtype, shape or dim order its metadata lacks. So when a gather still
//! reads an int64 cast after the pass, the pass hands the graph to stock
//! (`stock_fallback::request`), which lowers it or raises its own error.

use crate::fx_ir::{wrap_dim, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::layout::ctx::LayoutCtx;
use crate::layout::fx_node::rc4_target;
use crate::pipeline_config::PipelineConfig;
use crate::runner_logging::log_line;

const GATHER: &str = "aten.gather.default";
const DIM_ORDER_COPY: &str = "dim_order_ops._to_dim_order_copy.default";
const INT32: &str = "torch.int32";
const INT64: &str = "torch.int64";
const INT32_MAX: i64 = i32::MAX as i64;

/// rc4's PrepareGatherIndicesPass (`arm_pass_manager.py:548`); see the module
/// docs. The flags are the spec capabilities its dtype gate reads.
pub(crate) struct IrPrepareGatherIndicesPass {
    support_integer: bool,
    support_float: bool,
    bf16: bool,
    fp8e4m3: bool,
    fp8e5m2: bool,
}

impl IrPrepareGatherIndicesPass {
    pub(crate) fn new(config: &PipelineConfig) -> Self {
        // `support_extension` holds for an extension that the spec lists and one
        // of its profiles defines; TOSA defines these three for FP only.
        let fp_extension = |extension| config.support_float && config.lists_extension(extension);
        Self {
            support_integer: config.support_integer,
            support_float: config.support_float,
            bf16: fp_extension("bf16"),
            fp8e4m3: fp_extension("fp8e4m3"),
            fp8e5m2: fp_extension("fp8e5m2"),
        }
    }

    /// `_values_dtype_supported`.
    fn values_dtype_supported(&self, dtype: &str) -> bool {
        let floating_profile = self.support_float || self.support_integer;
        match dtype {
            "torch.bool" | "torch.int8" | "torch.int16" | "torch.int32" => self.support_integer,
            "torch.bfloat16" => floating_profile && self.bf16,
            "torch.float8_e4m3fn" => floating_profile && self.fp8e4m3,
            "torch.float8_e5m2" => floating_profile && self.fp8e5m2,
            "torch.float16" | "torch.float32" => floating_profile,
            _ => false,
        }
    }

    /// `_is_supported_gather`: the boundary a gather can drop, or `None`. rc4
    /// then repeats the shape checks that the boundary check has just made for
    /// this gather, one of the boundary's users.
    fn supported_boundary(&self, graph: &IrGraph, id: NodeId) -> Option<NodeId> {
        let node = graph.node(id);
        if node.op != NodeOp::CallFunction || node.target != GATHER || node.args.len() != 3 {
            return None;
        }
        let values = node.args[0].as_node()?;
        let indices = node.args[2].as_node()?;
        let supported = self.values_dtype_supported(graph.node_dtype(values)?)
            && graph.node_dtype(indices) == Some(INT64)
            && is_safe_boundary(graph, indices);
        supported.then_some(indices)
    }
}

impl IrPass for IrPrepareGatherIndicesPass {
    fn name(&self) -> &str {
        "IrPrepareGatherIndices"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = self.prepare(graph);
        if let Some(reason) = surviving_index_cast(graph) {
            crate::stock_fallback::request(format!("IrPrepareGatherIndices: {reason}"));
        }
        modified
    }
}

impl IrPrepareGatherIndicesPass {
    /// rc4's `call`, behind its `should_run_pass`.
    fn prepare(&self, graph: &mut IrGraph) -> bool {
        // `should_run_pass`: without a supported gather the graph stays as it is,
        // dead code included.
        if !graph
            .live_call_function_ids()
            .into_iter()
            .any(|id| self.supported_boundary(graph, id).is_some())
        {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let mut rewired = 0;
        for gather in ctx.order().to_vec() {
            let Some(boundary) = self.supported_boundary(ctx.graph(), gather) else {
                continue;
            };
            // `_int32_input_before_boundary`.
            let Some(source) = ctx.node(boundary).args[0].as_node() else {
                continue;
            };
            ctx.replace_input_with(gather, boundary, source);
            rewired += 1;
        }
        // Rewiring a gather only takes a user off its boundary, so every gather
        // `should_run_pass` found is still supported when the loop reaches it.
        if rewired == 0 {
            return false;
        }
        ctx.eliminate_dead_code();
        ctx.finish();
        log_line(format_args!(
            "[IrPrepareGatherIndices] {rewired} gather(s) now read int32 indices"
        ));
        true
    }
}

/// `is_safe_int32_to_int64_gather_boundary`.
fn is_safe_boundary(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    if node.op != NodeOp::CallFunction || rc4_target(&node.target) != DIM_ORDER_COPY {
        return false;
    }
    if !matches!(node.kwargs.get("dtype"), Some(IrArg::Dtype(dtype)) if dtype == INT64) {
        return false;
    }
    let Some(source) = node.args.first().and_then(IrArg::as_node) else {
        return false;
    };
    if graph.node_dtype(source) != Some(INT32) {
        return false;
    }
    // RETAB's metadata can lack a dim order, which rc4's fake tensors always
    // have; such a boundary isn't safe here.
    match (
        graph.node(source).meta.val_dim_order(),
        node.meta.val_dim_order(),
    ) {
        (Some(source_order), Some(order)) if source_order == order => {}
        _ => return false,
    }
    let users = graph.users(id);
    !users.is_empty()
        && users
            .iter()
            .all(|&user| is_compatible_gather_user(graph, id, user))
}

/// `_is_compatible_gather_user`. Shapes are static here: RETAB sends graphs
/// with symbolic shapes to stock, so `statically_known_true` is a comparison.
fn is_compatible_gather_user(graph: &IrGraph, boundary: NodeId, gather: NodeId) -> bool {
    let node = graph.node(gather);
    if node.op != NodeOp::CallFunction
        || node.target != GATHER
        || node.args.len() != 3
        || node.args[2].as_node() != Some(boundary)
    {
        return false;
    }
    let (Some(values), Some(indices)) = (
        node.args[0]
            .as_node()
            .and_then(|values| graph.node_shape(values)),
        graph.node_shape(boundary),
    ) else {
        return false;
    };
    if !matches!(values.len(), 2 | 3) || !matches!(indices.len(), 2 | 3) {
        return false;
    }
    // Python's `%`, as `wrap_dim` computes it.
    let Some(dim) = node.args[1]
        .as_int()
        .and_then(|dim| wrap_dim(dim, values.len()))
    else {
        return false;
    };
    if dim != 1 || values[dim] > INT32_MAX || values[0] != indices[0] {
        return false;
    }
    indices.len() == 2 || values.last() == indices.last()
}

/// Why a live gather still reads an int64 `_to_dim_order_copy`, whose tensor
/// TOSA 1.0 can't hold, or `None`.
fn surviving_index_cast(graph: &IrGraph) -> Option<String> {
    graph.live_call_function_ids().into_iter().find_map(|id| {
        let node = graph.node(id);
        if node.target != GATHER {
            return None;
        }
        let cast = graph.node(node.args.get(2)?.as_node()?);
        let int64_cast = rc4_target(&cast.target) == DIM_ORDER_COPY
            && (cast.meta.val_dtype() == Some(INT64)
                || matches!(cast.kwargs.get("dtype"), Some(IrArg::Dtype(dtype)) if dtype == INT64));
        int64_cast.then(|| {
            format!(
                "gather '{}' reads the int64 index cast '{}', which is not a safe int32 \
                 boundary to remove",
                node.name, cast.name
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::fx_ir::NodeMeta;
    use crate::layout::shape::mark_contiguous;
    use crate::test_ir::{call_with_meta, live_node, meta, placeholder};

    const F32: &str = "torch.float32";
    const UNSQUEEZE: &str = "aten.unsqueeze_copy.default";
    const EXPAND: &str = "aten.expand_copy.default";

    fn pass(spec: &str) -> IrPrepareGatherIndicesPass {
        IrPrepareGatherIndicesPass::new(&PipelineConfig::from_tosa_spec(spec))
    }

    /// A value meta with a contiguous dim order, as FX import gives a node.
    fn value(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = meta(shape.to_vec(), dtype);
        mark_contiguous(&mut meta);
        meta
    }

    fn input(graph: &mut IrGraph, name: &str, shape: &[i64], dtype: &str) -> NodeId {
        let id = placeholder(graph, name, shape.to_vec(), dtype);
        mark_contiguous(&mut graph.node_mut(id).meta);
        id
    }

    fn op(
        graph: &mut IrGraph,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> NodeId {
        call_with_meta(graph, name, target, args, meta)
    }

    /// `x.to(torch.int64)` as the edge dialect spells it.
    fn int64_cast(graph: &mut IrGraph, name: &str, source: NodeId, meta: NodeMeta) -> NodeId {
        graph.add_node_with_kwargs(
            NodeOp::CallFunction,
            name.to_owned(),
            DIM_ORDER_COPY.to_owned(),
            vec![IrArg::Node(source)],
            HashMap::from([("dtype".to_owned(), IrArg::Dtype(INT64.to_owned()))]),
            meta,
        )
    }

    fn gather(
        graph: &mut IrGraph,
        name: &str,
        values: NodeId,
        dim: i64,
        index: NodeId,
        meta: NodeMeta,
    ) -> NodeId {
        op(
            graph,
            name,
            GATHER,
            vec![IrArg::Node(values), IrArg::Int(dim), IrArg::Node(index)],
            meta,
        )
    }

    fn output(graph: &mut IrGraph, outputs: &[NodeId]) {
        let args = outputs.iter().map(|&id| IrArg::Node(id)).collect();
        graph.add_node(
            NodeOp::Output,
            "output".to_owned(),
            "output".to_owned(),
            vec![IrArg::List(args)],
            NodeMeta::default(),
        );
    }

    /// The name of the node a live gather reads as its index.
    fn index_of(graph: &IrGraph, gather: &str) -> String {
        let id = live_node(graph, gather).args[2]
            .as_node()
            .expect("a node index");
        graph.node(id).name.clone()
    }

    fn is_live(graph: &IrGraph, name: &str) -> bool {
        graph.live_nodes().any(|node| node.name == name)
    }

    /// Run the pass and return whether it changed the graph and the stock
    /// fallback reason it requested, if any.
    fn run(pass: &IrPrepareGatherIndicesPass, graph: &mut IrGraph) -> (bool, Option<String>) {
        crate::stock_fallback::take();
        let modified = pass.run(graph);
        (modified, crate::stock_fallback::take())
    }

    /// rc4's `TopKGather` delegate (test_convert_int64_output_ops_to_int32.py) as
    /// rc4's partitioner cuts it: the topk and its int32 cast stay on the CPU, so
    /// the int32 indices [2, 3] are an input, unsqueezed, expanded to the
    /// features' last axis and cast back to int64 for the gather.
    fn topk_gather(values_dtype: &str, channels: i64) -> IrGraph {
        let mut graph = IrGraph::empty();
        let features = input(&mut graph, "features", &[2, 8, channels], values_dtype);
        let indices = input(&mut graph, "indices", &[2, 3], INT32);
        let unsqueeze = op(
            &mut graph,
            "unsqueeze",
            UNSQUEEZE,
            vec![IrArg::Node(indices), IrArg::Int(-1)],
            value(&[2, 3, 1], INT32),
        );
        let expand = op(
            &mut graph,
            "expand",
            EXPAND,
            vec![
                IrArg::Node(unsqueeze),
                IrArg::List(vec![IrArg::Int(-1), IrArg::Int(-1), IrArg::Int(2)]),
            ],
            value(&[2, 3, 2], INT32),
        );
        let cast = int64_cast(&mut graph, "cast", expand, value(&[2, 3, 2], INT64));
        let out = gather(
            &mut graph,
            "gather",
            features,
            1,
            cast,
            value(&[2, 3, 2], values_dtype),
        );
        output(&mut graph, &[out]);
        graph
    }

    /// rc4's `test_prepare_gather_index_dtype`, edge ops: with matching last axes
    /// the gather reads the expand's int32 indices and the boundary goes.
    #[test]
    fn topk_gather_reads_its_int32_indices() {
        let mut graph = topk_gather(F32, 2);
        let (modified, fallback) = run(&pass("TOSA-1.0+INT"), &mut graph);
        assert!(modified);
        assert_eq!(fallback, None);
        assert_eq!(index_of(&graph, "gather"), "expand");
        assert!(!is_live(&graph, "cast"));
        assert_eq!(live_node(&graph, "expand").meta.val_dtype(), Some(INT32));
        assert!(!graph
            .live_nodes()
            .any(|node| node.meta.val_dtype() == Some(INT64)));
    }

    /// The same with three feature channels: rank-3 indices must match the
    /// values' last axis, so rc4 keeps the boundary, and RETAB, which can't
    /// lower its int64 tensor, hands the graph to stock.
    #[test]
    fn a_trailing_axis_mismatch_keeps_the_boundary_and_goes_to_stock() {
        let mut graph = topk_gather(F32, 3);
        let (modified, fallback) = run(&pass("TOSA-1.0+INT"), &mut graph);
        assert!(!modified);
        assert_eq!(index_of(&graph, "gather"), "cast");
        let reason = fallback.expect("a stock fallback");
        assert!(
            reason.starts_with("IrPrepareGatherIndices: gather 'gather'"),
            "{reason}"
        );
        assert!(reason.contains("'cast'"), "{reason}");
    }

    /// rc4's `test_prepare_gather_index_dtype_for_tosa_capabilities`: the values
    /// dtype the spec can gather decides, cast by cast.
    #[test]
    fn the_values_dtype_must_be_one_the_spec_gathers() {
        for (dtype, spec, supported) in [
            (INT32, "TOSA-1.0+INT", true),
            (INT32, "TOSA-1.0+FP", false),
            (F32, "TOSA-1.0+FP", true),
            (F32, "TOSA-1.0+INT", true),
            ("torch.bfloat16", "TOSA-1.0+FP+bf16", true),
            ("torch.bfloat16", "TOSA-1.0+FP", false),
            ("torch.float8_e4m3fn", "TOSA-1.0+FP+fp8e4m3", true),
            ("torch.float8_e4m3fn", "TOSA-1.0+FP", false),
            ("torch.float8_e5m2", "TOSA-1.0+FP+fp8e5m2", true),
            ("torch.float8_e5m2", "TOSA-1.0+FP", false),
        ] {
            let mut graph = topk_gather(dtype, 2);
            let (modified, fallback) = run(&pass(spec), &mut graph);
            assert_eq!(modified, supported, "{dtype} on {spec}");
            assert_eq!(fallback.is_some(), !supported, "{dtype} on {spec}");
            assert_eq!(is_live(&graph, "cast"), !supported, "{dtype} on {spec}");
            let index = if supported { "expand" } else { "cast" };
            assert_eq!(index_of(&graph, "gather"), index, "{dtype} on {spec}");
        }
    }

    /// The other dtypes RETAB's specs meet: bool, int8 and int16 values need the
    /// INT profile, float16 takes either, and int64 values never qualify.
    #[test]
    fn the_dtype_gate_follows_rc4_for_every_profile() {
        let int = pass("TOSA-1.0+INT+int16+int4+cf");
        let fp = pass("TOSA-1.0+FP");
        let both = pass("TOSA-1.0+INT+FP");
        for dtype in ["torch.bool", "torch.int8", "torch.int16", INT32] {
            assert!(int.values_dtype_supported(dtype), "{dtype}");
            assert!(!fp.values_dtype_supported(dtype), "{dtype}");
            assert!(both.values_dtype_supported(dtype), "{dtype}");
        }
        for gate in [&int, &fp, &both] {
            assert!(gate.values_dtype_supported("torch.float16"));
            assert!(gate.values_dtype_supported(F32));
            for dtype in [INT64, "torch.float64", "torch.uint8", "torch.bfloat16"] {
                assert!(!gate.values_dtype_supported(dtype), "{dtype}");
            }
        }
        // An extension without the FP profile is no capability: TOSA 1.0 defines
        // bf16 for FP only, and rc4 refuses such a spec.
        assert!(!pass("TOSA-1.0+INT+bf16").values_dtype_supported("torch.bfloat16"));
        assert!(pass("TOSA-1.0+INT+FP+bf16").values_dtype_supported("torch.bfloat16"));
    }

    /// rc4's `test_symbolic_gather_shape_mismatch_rejects_boundary`, with static
    /// shapes: RETAB sends symbolic ones to stock. Batch sizes that differ make
    /// the boundary unsafe.
    #[test]
    fn a_batch_mismatch_keeps_the_boundary() {
        let mut graph = IrGraph::empty();
        let features = input(&mut graph, "features", &[3, 8, 2], F32);
        let indices = input(&mut graph, "indices", &[2, 3, 2], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[2, 3, 2], INT64));
        let out = gather(
            &mut graph,
            "gather",
            features,
            1,
            cast,
            value(&[2, 3, 2], F32),
        );
        output(&mut graph, &[out]);
        assert!(!is_safe_boundary(&graph, cast));
        let (modified, fallback) = run(&pass("TOSA-1.0+INT"), &mut graph);
        assert!(!modified);
        assert!(fallback.is_some());
        assert_eq!(index_of(&graph, "gather"), "cast");
    }

    /// rc4's `test_prepare_scalar_gather_is_unchanged`: a rank-0 gather of an
    /// int64 input is no target and no int64 cast, so nothing happens.
    #[test]
    fn a_scalar_gather_is_unchanged() {
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[], F32);
        let indices = input(&mut graph, "indices", &[], INT64);
        let out = gather(&mut graph, "gather", values, 0, indices, value(&[], F32));
        output(&mut graph, &[out]);
        assert_eq!(run(&pass("TOSA-1.0+FP"), &mut graph), (false, None));
        assert_eq!(index_of(&graph, "gather"), "indices");
    }

    /// rc4's `test_prepare_gather_does_not_narrow_arbitrary_int64_indices`: an
    /// int64 index that no cast produced stays int64, and it is no index cast.
    #[test]
    fn arbitrary_int64_indices_are_not_narrowed() {
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[2, 8, 2], F32);
        let indices = input(&mut graph, "indices", &[2, 3, 2], INT64);
        let out = gather(
            &mut graph,
            "gather",
            values,
            1,
            indices,
            value(&[2, 3, 2], F32),
        );
        output(&mut graph, &[out]);
        assert_eq!(run(&pass("TOSA-1.0+FP"), &mut graph), (false, None));
        assert_eq!(index_of(&graph, "gather"), "indices");
        assert_eq!(live_node(&graph, "indices").meta.val_dtype(), Some(INT64));
    }

    /// rc4's `test_prepare_gather_preserves_layout_changing_boundary`: a cast
    /// that makes transposed int32 indices contiguous is no boundary to drop.
    #[test]
    fn a_layout_changing_boundary_is_preserved() {
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[2, 8, 2], F32);
        let indices = input(&mut graph, "indices", &[2, 3, 2], INT32);
        graph
            .node_mut(indices)
            .meta
            .set_val_dim_order(vec![0, 2, 1]);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[2, 3, 2], INT64));
        let out = gather(
            &mut graph,
            "gather",
            values,
            1,
            cast,
            value(&[2, 3, 2], F32),
        );
        output(&mut graph, &[out]);
        let (modified, fallback) = run(&pass("TOSA-1.0+FP"), &mut graph);
        assert!(!modified);
        assert!(fallback.is_some());
        assert_eq!(index_of(&graph, "gather"), "cast");
    }

    /// A dim order RETAB's metadata lacks can't be compared, so the port can't
    /// call the boundary safe and leaves the graph to stock.
    #[test]
    fn a_missing_dim_order_goes_to_stock() {
        let mut graph = topk_gather(F32, 2);
        let cast = live_node(&graph, "cast").id;
        graph.node_mut(cast).meta = meta(vec![2, 3, 2], INT64);
        let (modified, fallback) = run(&pass("TOSA-1.0+INT"), &mut graph);
        assert!(!modified);
        assert!(fallback.is_some());
        assert_eq!(index_of(&graph, "gather"), "cast");
    }

    /// rc4's `test_prepare_gather_skips_large_indexable_dimension`: indices into
    /// an axis longer than INT32_MAX keep their int64 boundary.
    #[test]
    fn a_large_indexable_dimension_keeps_the_boundary() {
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[1, INT32_MAX + 1], F32);
        let indices = input(&mut graph, "indices", &[1, 1], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[1, 1], INT64));
        let out = gather(&mut graph, "gather", values, 1, cast, value(&[1, 1], F32));
        output(&mut graph, &[out]);
        let (modified, fallback) = run(&pass("TOSA-1.0+FP"), &mut graph);
        assert!(!modified);
        assert!(fallback.is_some());
        assert_eq!(index_of(&graph, "gather"), "cast");

        // One element shorter, the axis fits int32 and the boundary goes.
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[1, INT32_MAX], F32);
        let indices = input(&mut graph, "indices", &[1, 1], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[1, 1], INT64));
        let out = gather(&mut graph, "gather", values, -1, cast, value(&[1, 1], F32));
        output(&mut graph, &[out]);
        assert_eq!(run(&pass("TOSA-1.0+FP"), &mut graph), (true, None));
        assert_eq!(index_of(&graph, "gather"), "indices");
    }

    /// Two gathers on one boundary, as in rc4's shared-boundary partitioner
    /// test: each moves off in graph order, the boundary staying safe for the
    /// one left, and the boundary goes with the last.
    #[test]
    fn a_shared_boundary_goes_once_every_gather_moves_off() {
        let mut graph = IrGraph::empty();
        let first = input(&mut graph, "first", &[2, 8, 2], F32);
        let second = input(&mut graph, "second", &[2, 8, 2], F32);
        let indices = input(&mut graph, "indices", &[2, 3, 2], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[2, 3, 2], INT64));
        let a = gather(&mut graph, "a", first, 1, cast, value(&[2, 3, 2], F32));
        let b = gather(&mut graph, "b", second, -2, cast, value(&[2, 3, 2], F32));
        output(&mut graph, &[a, b]);
        assert_eq!(run(&pass("TOSA-1.0+INT"), &mut graph), (true, None));
        assert_eq!(index_of(&graph, "a"), "indices");
        assert_eq!(index_of(&graph, "b"), "indices");
        assert!(!is_live(&graph, "cast"));
    }

    /// A shared boundary whose second gather has values the spec can't gather:
    /// it is still a safe boundary, so the first gather moves off, but the
    /// boundary stays for the second, and the graph goes to stock.
    #[test]
    fn a_gather_with_unsupported_values_keeps_a_shared_boundary() {
        let mut graph = IrGraph::empty();
        let first = input(&mut graph, "first", &[2, 8, 2], F32);
        let second = input(&mut graph, "second", &[2, 8, 2], INT32);
        let indices = input(&mut graph, "indices", &[2, 3, 2], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[2, 3, 2], INT64));
        let a = gather(&mut graph, "a", first, 1, cast, value(&[2, 3, 2], F32));
        let b = gather(&mut graph, "b", second, 1, cast, value(&[2, 3, 2], INT32));
        output(&mut graph, &[a, b]);
        let (modified, fallback) = run(&pass("TOSA-1.0+FP"), &mut graph);
        assert!(modified);
        assert_eq!(index_of(&graph, "a"), "indices");
        assert_eq!(index_of(&graph, "b"), "cast");
        assert!(fallback.expect("a stock fallback").contains("gather 'b'"));
    }

    /// A boundary that also feeds a non-gather user, here the output, is not
    /// safe for any of its gathers.
    #[test]
    fn a_boundary_with_another_user_is_kept() {
        let mut graph = IrGraph::empty();
        let values = input(&mut graph, "values", &[2, 8], F32);
        let indices = input(&mut graph, "indices", &[2, 3], INT32);
        let cast = int64_cast(&mut graph, "cast", indices, value(&[2, 3], INT64));
        let out = gather(&mut graph, "gather", values, 1, cast, value(&[2, 3], F32));
        output(&mut graph, &[out, cast]);
        let (modified, fallback) = run(&pass("TOSA-1.0+INT"), &mut graph);
        assert!(!modified);
        assert!(fallback.is_some());
        assert_eq!(index_of(&graph, "gather"), "cast");
    }

    /// rc4 ends a change with `graph.eliminate_dead_code()` on the whole graph,
    /// and skips the pass, dead code included, when no gather qualifies.
    #[test]
    fn dead_code_goes_only_when_the_pass_changes_something() {
        for channels in [2, 3] {
            let mut graph = topk_gather(F32, channels);
            let features = live_node(&graph, "features").id;
            op(
                &mut graph,
                "unused",
                "aten.abs.default",
                vec![IrArg::Node(features)],
                value(&[2, 8, channels], F32),
            );
            let (modified, _) = run(&pass("TOSA-1.0+INT"), &mut graph);
            assert_eq!(modified, channels == 2);
            assert_eq!(is_live(&graph, "unused"), !modified, "{channels} channels");
        }
    }

    /// RETAB spells the cast without its `dim_order_ops.` namespace in places;
    /// it is still rc4's edge cast.
    #[test]
    fn the_cast_matches_in_either_spelling() {
        let mut graph = topk_gather(F32, 2);
        let cast = live_node(&graph, "cast").id;
        graph.node_mut(cast).target = "_to_dim_order_copy.default".to_owned();
        assert_eq!(run(&pass("TOSA-1.0+INT"), &mut graph), (true, None));
        assert_eq!(index_of(&graph, "gather"), "expand");
    }
}
