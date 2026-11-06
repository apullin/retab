//! Adversarial fidelity tests for the four P1 layout ports (APU-153 review):
//! `normalize_delegate_io`, `match_arg_ranks`, `high_rank_singleton_permute`
//! and `fuse_view_copy`, checked against ExecuTorch v1.6.0-rc4.
//!
//! Each test names the rc4 behaviour it pins. "venv" marks graphs whose rc4
//! output was taken by running the rc4 pass in `.venv-et16`; the test rebuilds
//! that graph. A test marked `#[ignore = "BUG: ..."]` fails on the current port
//! and documents a divergence for the pass files.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::QParamMap;
use crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass;
use crate::layout::high_rank_singleton_permute::IrRewriteHighRankSingletonPermutePass;
use crate::layout::match_arg_ranks::IrMatchArgRanksPass;
use crate::layout::normalize_delegate_io::IrNormalizeDelegateIOLayoutPass;
use crate::layout::shape::{has_local_rule, refreshed_meta};
use crate::layout::testing::*;
use crate::test_ir::{f32_tensor, live_signature, parameter, qparam};

const NHWC: [usize; 4] = [0, 2, 3, 1];

fn normalize(g: &mut LayoutGraph) -> bool {
    IrNormalizeDelegateIOLayoutPass.run(g.graph_mut())
}

fn match_ranks(g: &mut LayoutGraph) -> bool {
    IrMatchArgRanksPass.run(g.graph_mut())
}

fn high_rank(g: &mut LayoutGraph) -> bool {
    IrRewriteHighRankSingletonPermutePass.run(g.graph_mut())
}

fn fuse(g: &mut LayoutGraph) -> bool {
    IrFuseViewCopyTransformPass.run(g.graph_mut())
}

/// A call with the meta given here, for targets the local refresh has no rule
/// for (TOSA ops, `aten.permute.default`, ops outside the pointwise table).
fn raw_call(
    g: &mut LayoutGraph,
    name: &str,
    target: &str,
    args: Vec<IrArg>,
    shape: &[i64],
    dtype: &str,
) -> NodeId {
    g.graph_mut().add_call_function_with_kwargs(
        name,
        target,
        args,
        HashMap::new(),
        NodeMeta::tensor(shape.to_vec(), dtype),
    )
}

fn set_dim_order(g: &mut LayoutGraph, id: NodeId, order: &[usize]) {
    g.graph_mut()
        .node_mut(id)
        .meta
        .set_val_dim_order(order.to_vec());
}

/// Live node names in graph order.
fn names(g: &LayoutGraph) -> Vec<String> {
    g.graph()
        .topological_live_node_ids()
        .into_iter()
        .map(|id| g.name(id))
        .collect()
}

/// The entries of the output node's result list.
fn outputs(g: &LayoutGraph) -> Vec<IrArg> {
    let output = g.graph().live_output_ids()[0];
    match &g.graph().node(output).args[0] {
        IrArg::List(items) => items.clone(),
        other => vec![other.clone()],
    }
}

fn output_names(g: &LayoutGraph) -> Vec<String> {
    outputs(g)
        .iter()
        .map(|arg| {
            arg.as_node()
                .map_or_else(|| "None".to_string(), |id| g.name(id))
        })
        .collect()
}

/// Every live call whose target has a local shape rule holds the value facts
/// its operands give it, which is what rc4's retrace (`ExportPass.call`)
/// recomputes for every node after a pass.
fn assert_vals_match_retrace(g: &LayoutGraph) {
    for id in g.graph().topological_live_node_ids() {
        let node = g.graph().node(id);
        if node.op != NodeOp::CallFunction || !has_local_rule(&node.target) {
            continue;
        }
        let expected =
            refreshed_meta(g.graph(), id).unwrap_or_else(|error| panic!("{}: {error}", node.name));
        assert_eq!(
            node.meta.val_shape(),
            expected.val_shape(),
            "shape of {} ({})",
            node.name,
            node.target
        );
        assert_eq!(
            node.meta.val_dtype(),
            expected.val_dtype(),
            "dtype of {} ({})",
            node.name,
            node.target
        );
    }
}

/// `id`'s dim order, when it has one, is the identity of its own rank: every
/// node these passes create or re-trace is contiguous in rc4.
fn assert_dim_order_fits(g: &LayoutGraph, id: NodeId) {
    let node = g.graph().node(id);
    let Some(order) = node.meta.val_dim_order() else {
        return;
    };
    let rank = g.shape(id).len();
    assert_eq!(
        order,
        (0..rank).collect::<Vec<_>>().as_slice(),
        "dim order of {} (rank {rank})",
        node.name
    );
}

// ---------------------------------------------------------------------------
// FuseViewCopyTransform (backends/transforms/fuse_view_copy.py)
// ---------------------------------------------------------------------------

/// venv: `x[1,1] -> view([1]) -> relu -> view([])` becomes `view(x, []) ->
/// relu`, and the retrace gives both nodes shape `()`. FX import turns the
/// empty size list into `IrArg::None`, which `as_int_list` does not read, so
/// the port leaves the retained nodes at shape `[1]` while the start view now
/// reshapes to rank 0. The serializer reads a View's size only as an int list
/// (`literal_output_shape`) and otherwise falls back to that stale meta.
#[test]
#[ignore = "BUG: fuse_view_copy skips the meta refresh when the final view is rank 0 (size IrArg::None)"]
fn fuse_view_copy_refreshes_meta_for_a_rank0_final_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1]);
    let v0 = g.view(x, &[1]);
    let relu = g.call(RELU, vec![node(v0)]);
    let v1 = g.call(VIEW, vec![node(relu), IrArg::None]);
    g.output(&[v1]);
    let (v0, relu) = (g.name(v0), g.name(relu));

    assert!(fuse(&mut g));

    let (v0, relu) = (g.id(&v0), g.id(&relu));
    g.assert_targets(&[VIEW, RELU]);
    assert!(matches!(g.graph().node(v0).args[1], IrArg::None));
    assert!(g.shape(v0).is_empty(), "start view: {:?}", g.shape(v0));
    assert!(g.shape(relu).is_empty(), "relu: {:?}", g.shape(relu));
    assert_vals_match_retrace(&g);
}

/// venv: `x[2,3,4] -> view([-1]) -> relu -> view([2,-1])` becomes
/// `view(x, [2,-1]) -> relu`, both with shape `(2, 12)`. Export keeps the `-1`
/// (rc4 resolves it only in ResolveViewCopyInferredDimPass,
/// `arm_pass_manager.py:745`, after FuseViewCopyTransformPass), but the port
/// copies the size argument into the retained nodes' shapes verbatim. Today's
/// serializer re-resolves a negative dim, but the passes in between read val
/// shapes, and so will the explicit-mode serializer (port plan, P2).
#[test]
#[ignore = "BUG: fuse_view_copy writes an unresolved -1 from the final view's size into val shapes"]
fn fuse_view_copy_resolves_an_inferred_dim_in_the_final_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v0 = g.view(x, &[-1]);
    let relu = g.call(RELU, vec![node(v0)]);
    let v1 = g.view(relu, &[2, -1]);
    g.output(&[v1]);
    let (v0, relu) = (g.name(v0), g.name(relu));

    assert!(fuse(&mut g));

    let (v0, relu) = (g.id(&v0), g.id(&relu));
    assert_eq!(g.graph().node(v0).args[1].as_int_list(), Some(vec![2, -1]));
    assert_eq!(g.shape(v0), &[2, 12]);
    assert_eq!(g.shape(relu), &[2, 12]);
    assert_vals_match_retrace(&g);
}

/// rc4 inserts TABLE ops (InsertTableOpsPass, `arm_pass_manager.py:708`)
/// before FuseViewCopyTransformPass (`:725`), so a quantized sigmoid is a
/// `backend.tosa.TABLE` there; TABLE is not in `UNARY_ELEMENTWISE_OPS`
/// (`fuse_view_copy.py:43-68`) and the chain stops. RETAB defers TABLE
/// (`fx_insert_table_ops_ir`), so the node is still `aten.sigmoid.default` with
/// int8 qparams and the port fuses through it. venv, INT: `x[2,3] -> view(6)
/// -> sigmoid -> view(2,1,1,3)` ends as TABLE on `(2,3)` then RESHAPE
/// `(2,1,1,3)`. The port instead makes the TABLE read `[2,1,1,3]`, the
/// `[N,1,1,C]` operand rc4's propagation guard keeps away from TABLE for an
/// Ethos-U85 miscompile (`propagate_view_copy_permute_pass.py:61-81`).
#[test]
#[ignore = "BUG: fuse_view_copy fuses through a pending TABLE (quantized aten.sigmoid), which rc4 sees as backend.tosa.TABLE"]
fn fuse_view_copy_stops_at_a_pending_table() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3], "torch.int8");
    let v0 = g.view(x, &[6]);
    let sigmoid = g.call(SIGMOID, vec![node(v0)]);
    g.graph_mut()
        .node_mut(sigmoid)
        .meta
        .set_unary_input_output_qparams(
            qparam(0.05, 0, "torch.int8"),
            qparam(1.0 / 256.0, -128, "torch.int8"),
        );
    let v1 = g.view(sigmoid, &[2, 1, 1, 3]);
    g.output(&[v1]);

    assert!(!fuse(&mut g));
    g.assert_targets(&[VIEW, SIGMOID, VIEW]);
}

/// The float twin of the test above: venv, FP: the views fuse, giving RESHAPE
/// `(2,1,1,3)` then SIGMOID.
#[test]
fn fuse_view_copy_fuses_through_a_float_sigmoid() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v0 = g.view(x, &[6]);
    let sigmoid = g.call(SIGMOID, vec![node(v0)]);
    let v1 = g.view(sigmoid, &[2, 1, 1, 3]);
    g.output(&[v1]);
    let sigmoid = g.name(sigmoid);

    assert!(fuse(&mut g));

    g.assert_targets(&[VIEW, SIGMOID]);
    assert_eq!(g.shape(g.id(&sigmoid)), &[2, 1, 1, 3]);
    assert_vals_match_retrace(&g);
}

/// rc4's fusible set is `UNARY_ELEMENTWISE_OPS` plus `view_copy`
/// (`fuse_view_copy.py:43-68,94`), as dumped from the venv; RETAB adds its
/// short spellings of the two dim-order ops. Anything else stops the chain,
/// including `aten.view.default`, TABLE in both spellings and table-able ops
/// outside the list.
#[test]
fn fuse_view_copy_fusible_set_is_rc4s() {
    let fusible = [
        "aten.alias_copy.default",
        "aten.clone.default",
        "dim_order_ops._clone_dim_order.default",
        "_clone_dim_order.default",
        "aten._to_copy.default",
        "dim_order_ops._to_dim_order_copy.default",
        "_to_dim_order_copy.default",
        "quantized_decomposed.quantize_per_tensor.default",
        "quantized_decomposed.dequantize_per_tensor.default",
        "aten.abs.default",
        "aten.clamp.default",
        "aten.ceil.default",
        "aten.floor.default",
        "aten.neg.default",
        "aten.relu.default",
        "aten.round.default",
        "aten.sigmoid.default",
        "aten.silu.default",
        "aten.sqrt.default",
        "aten.tanh.default",
        "aten.sign.default",
        "aten.reciprocal.default",
        "aten.gelu.default",
        "aten.rsqrt.default",
        "aten.exp.default",
        "aten.log.default",
    ];
    let not_fusible = [
        "aten.view.default",
        "aten.permute_copy.default",
        "aten.detach_copy.default",
        "aten.hardtanh.default",
        "aten.leaky_relu.default",
        "aten.hardswish.default",
        "aten.erf.default",
        "aten.sin.default",
        "aten.clamp.Tensor",
        "aten.relu_.default",
        "backend.tosa.TABLE.default",
        "tosa.TABLE.default",
        "backend.tosa.RESCALE.default",
        "dim_order_ops._empty_dim_order.default",
    ];
    for (target, expect_fused) in fusible
        .iter()
        .map(|target| (*target, true))
        .chain(not_fusible.iter().map(|target| (*target, false)))
    {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let v0 = g.view(x, &[6]);
        let op = raw_call(&mut g, "op", target, vec![node(v0)], &[6], "torch.float32");
        let v1 = g.view(op, &[3, 2]);
        g.output(&[v1]);

        assert_eq!(fuse(&mut g), expect_fused, "{target}");
        if expect_fused {
            g.assert_targets(&[VIEW, target]);
            assert_eq!(g.shape(g.id("op")), &[3, 2], "{target}");
        } else {
            g.assert_targets(&[VIEW, target, VIEW]);
        }
    }
}

/// FX counts the output node as a user (`len(end_node.users) == 1`,
/// `fuse_view_copy.py:100`), so a view that is also a graph output never
/// starts or continues a chain.
#[test]
fn fuse_view_copy_a_view_that_is_also_an_output_stops_the_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v0 = g.view(x, &[6]);
    let v1 = g.view(v0, &[3, 2]);
    g.output(&[v0, v1]);

    assert!(!fuse(&mut g));
    g.assert_targets(&[VIEW, VIEW]);
}

/// rc4 calls `graph.eliminate_dead_code()` on the whole graph once a chain
/// fuses (`fuse_view_copy.py:198-199`): unrelated dead nodes go too.
#[test]
fn fuse_view_copy_eliminates_unrelated_dead_code_when_it_fuses() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let dead = g.call(NEG, vec![node(x)]);
    let v0 = g.view(x, &[6]);
    let v1 = g.view(v0, &[3, 2]);
    g.output(&[v1]);
    let dead = g.name(dead);

    assert!(fuse(&mut g));

    g.assert_targets(&[VIEW]);
    assert!(g.graph().by_name(&dead).is_none_or(|node| !node.alive));
}

/// The retrace keeps every meta key but `val` (`exir/pass_base.py:686-690`), so the
/// start view and the unary ops keep their qparams.
#[test]
fn fuse_view_copy_keeps_qparams_on_retained_nodes() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3], "torch.int8");
    let v0 = g.view(x, &[6]);
    let neg = g.call(NEG, vec![node(v0)]);
    let v1 = g.view(neg, &[3, 2]);
    g.output(&[v1]);
    let qp = qparam(0.25, 1, "torch.int8");
    for id in [v0, neg] {
        g.graph_mut().node_mut(id).meta.set_both_qparams(qp.clone());
    }
    let (v0, neg) = (g.name(v0), g.name(neg));

    assert!(fuse(&mut g));

    for name in [v0, neg] {
        let meta = &g.graph().node(g.id(&name)).meta;
        assert_eq!(meta.output_qparam(), Some(&qp), "{name}");
        assert_eq!(meta.input_qparam(), Some(&qp), "{name}");
    }
}

/// Two independent chains in one sweep, visited in graph order, each fused
/// into its own first view (`graph.find_nodes` sorts by position).
#[test]
fn fuse_view_copy_fuses_independent_chains_in_one_sweep() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let a0 = g.view(x, &[6]);
    let a1 = g.view(a0, &[3, 2]);
    let b0 = g.view(x, &[1, 6]);
    let abs = g.call(ABS, vec![node(b0)]);
    let b1 = g.view(abs, &[6, 1]);
    g.output(&[a1, b1]);
    let (a0, b0, abs) = (g.name(a0), g.name(b0), g.name(abs));

    assert!(fuse(&mut g));

    g.assert_targets(&[VIEW, VIEW, ABS]);
    assert_eq!(output_names(&g), [a0.clone(), abs.clone()]);
    assert_eq!(g.shape(g.id(&a0)), &[3, 2]);
    assert_eq!(g.shape(g.id(&b0)), &[6, 1]);
    assert_eq!(g.shape(g.id(&abs)), &[6, 1]);
    assert_vals_match_retrace(&g);
}

/// The retrace gives the retained nodes contiguous values of the new rank;
/// the port keeps the dim orders they had (rank 1 here, on rank-2 shapes).
#[test]
#[ignore = "BUG (minor): fuse_view_copy leaves retained nodes' val_dim_order at the old rank"]
fn fuse_view_copy_retained_nodes_get_a_dim_order_of_their_new_rank() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v0 = g.view(x, &[6]);
    let relu = g.call(RELU, vec![node(v0)]);
    let v1 = g.view(relu, &[3, 2]);
    g.output(&[v1]);
    // As imported: every value carries its `dim_order()`.
    set_dim_order(&mut g, x, &[0, 1]);
    set_dim_order(&mut g, v0, &[0]);
    set_dim_order(&mut g, relu, &[0]);
    set_dim_order(&mut g, v1, &[0, 1]);
    let (v0, relu) = (g.name(v0), g.name(relu));

    assert!(fuse(&mut g));

    assert_dim_order_fits(&g, g.id(&v0));
    assert_dim_order_fits(&g, g.id(&relu));
}

// ---------------------------------------------------------------------------
// RewriteHighRankSingletonPermutePass
// ---------------------------------------------------------------------------

/// venv: a permute of an all-singleton `[1,1,1,1,1]` reduces to rank 0:
/// `view(x, []) -> view([1,1,1,1,1])`, with no permute in between.
#[test]
fn high_rank_all_singleton_permute_becomes_two_views_through_rank0() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 1, 1]);
    let p = g.permute(x, &[4, 3, 2, 1, 0]);
    g.output(&[p]);

    assert!(high_rank(&mut g));

    g.assert_targets(&[VIEW, VIEW]);
    let views = g.graph().topological_live_node_ids();
    let views: Vec<NodeId> = views
        .into_iter()
        .filter(|&id| g.graph().node(id).target == VIEW)
        .collect();
    assert!(g.shape(views[0]).is_empty());
    assert_eq!(g.shape(views[1]), &[1, 1, 1, 1, 1]);
    assert_vals_match_retrace(&g);
}

/// The rank-0 view above gets its empty size as `IrArg::List(vec![])`; FX
/// import, which every other rank-0 view comes through, spells it
/// `IrArg::None` (`fx_ir::resolve_arg`). Readers that handle one spelling
/// miss the other (see `fuse_view_copy_refreshes_meta_for_a_rank0_final_view`).
#[test]
#[ignore = "BUG (minor): high_rank_singleton_permute spells an empty view size List([]), import spells it None"]
fn high_rank_rank0_view_size_uses_the_import_spelling() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 1, 1]);
    let p = g.permute(x, &[4, 3, 2, 1, 0]);
    g.output(&[p]);

    assert!(high_rank(&mut g));

    let first = g
        .graph()
        .topological_live_node_ids()
        .into_iter()
        .find(|&id| g.graph().node(id).target == VIEW)
        .unwrap();
    assert!(
        matches!(g.graph().node(first).args[1], IrArg::None),
        "{:?}",
        g.graph().node(first).args[1]
    );
}

/// rc4's retrace interprets every node, dead ones included, so a dead permute
/// is rewritten into a dead chain, and nothing eliminates it.
#[test]
fn high_rank_rewrites_a_dead_permute_too() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let _dead = g.permute(x, &[0, 4, 1, 2, 3]);
    let relu = g.call(RELU, vec![node(x)]);
    g.output(&[relu]);

    assert!(high_rank(&mut g));

    g.assert_targets(&[VIEW, PERMUTE, VIEW, RELU]);
}

/// Back-to-back high-rank permutes: rc4 rewrites the second with the first's
/// rewritten output as input (`args[0].data`); the shapes agree either way.
#[test]
fn high_rank_rewrites_consecutive_permutes() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let p1 = g.permute(x, &[0, 4, 1, 2, 3]);
    let p2 = g.permute(p1, &[0, 2, 3, 4, 1]);
    g.output(&[p2]);

    assert!(high_rank(&mut g));

    g.assert_targets(&[VIEW, PERMUTE, VIEW, VIEW, PERMUTE, VIEW]);
    g.assert_permute_dims(&[&[3, 0, 1, 2], &[1, 2, 3, 0]]);
    assert_vals_match_retrace(&g);
}

/// `aten.permute.default` is a target too (`:29-33`); the reduced permute keeps
/// the original op (`super().call_operator(op, ...)`, `:103-108`).
#[test]
fn high_rank_keeps_the_aten_permute_target() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let p = raw_call(
        &mut g,
        "p",
        "aten.permute.default",
        vec![node(x), ints(&[0, 4, 1, 2, 3])],
        &[1, 5, 2, 3, 4],
        "torch.float32",
    );
    g.output(&[p]);

    assert!(high_rank(&mut g));

    g.assert_targets(&[VIEW, "aten.permute.default", VIEW]);
}

/// Every new node gets the permute's meta (`super().call_operator(..., meta)`,
/// `:94-114`), so per-channel qparams are copied verbatim, axis included, as
/// rc4 does.
#[test]
fn high_rank_copies_per_channel_qparams_verbatim() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4, 5], "torch.int8");
    let p = g.permute(x, &[0, 4, 1, 2, 3]);
    g.output(&[p]);
    let mut per_channel = qparam(0.5, 0, "torch.int8");
    per_channel.per_channel_scales = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    per_channel.per_channel_zps = vec![0; 5];
    per_channel.axis = 1;
    g.graph_mut()
        .node_mut(p)
        .meta
        .set_single_output_qparam(per_channel.clone());

    assert!(high_rank(&mut g));

    for id in g.graph().live_call_function_ids() {
        let meta = &g.graph().node(id).meta;
        assert_eq!(meta.output_qparam(), Some(&per_channel));
        assert_eq!(meta.val_dtype(), Some("torch.int8"));
    }
}

/// rc4 hands the permute's kwargs to the reduced permute only; both views get
/// `{}` (`:94-114`).
#[test]
fn high_rank_passes_kwargs_to_the_reduced_permute_only() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let p = g.call_with_kwargs(
        PERMUTE,
        vec![node(x), ints(&[0, 4, 1, 2, 3])],
        &[("marker", IrArg::Int(7))],
    );
    g.output(&[p]);

    assert!(high_rank(&mut g));

    for id in g.graph().live_call_function_ids() {
        let node = g.graph().node(id);
        let has_marker = node.kwargs.get("marker").and_then(IrArg::as_int) == Some(7);
        assert_eq!(has_marker, node.target == PERMUTE, "{}", node.name);
        assert_eq!(
            node.kwargs.len(),
            usize::from(node.target == PERMUTE),
            "{}",
            node.name
        );
    }
}

/// A permutation that is not one (duplicate dims) fails rc4's sorted check
/// (`:89-91`) and the permute stays.
#[test]
fn high_rank_leaves_an_invalid_permutation_alone() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let p = raw_call(
        &mut g,
        "p",
        PERMUTE,
        vec![node(x), ints(&[0, 4, 4, 2, 3])],
        &[1, 5, 5, 3, 4],
        "torch.float32",
    );
    g.output(&[p]);

    assert!(!high_rank(&mut g));
}

/// venv: the reduced view and permute are contiguous rank-4 values. The port
/// gives them the permute's whole meta, including its rank-5 dim order.
#[test]
#[ignore = "BUG (minor): high_rank_singleton_permute copies the rank-5 val_dim_order onto the rank-4 nodes"]
fn high_rank_new_nodes_get_dim_orders_of_their_own_rank() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3, 4, 5]);
    let p = g.permute(x, &[0, 4, 1, 2, 3]);
    g.output(&[p]);
    set_dim_order(&mut g, x, &[0, 1, 2, 3, 4]);
    set_dim_order(&mut g, p, &[0, 1, 2, 3, 4]);

    assert!(high_rank(&mut g));

    for id in g.graph().live_call_function_ids() {
        assert_dim_order_fits(&g, id);
    }
}

// ---------------------------------------------------------------------------
// MatchArgRanksPass
// ---------------------------------------------------------------------------

/// rc4's `target_ops` (`match_arg_ranks_pass.py:47-86`) as dumped from the
/// venv; FX import spells the TOSA ones `tosa.X.default`, RETAB's own passes
/// `backend.tosa.X.default`.
const RC4_RANK_MATCH_TARGETS: &[&str] = &[
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.div.Tensor_mode",
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_left_shift.Tensor",
    "aten.eq.Tensor",
    "aten.gt.Tensor",
    "aten.ge.Tensor",
    "aten.lt.Tensor",
    "aten.le.Tensor",
    "aten.ne.Tensor",
    "aten.pow.Tensor_Tensor",
    "aten.remainder.Tensor",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_xor.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.maximum.default",
    "aten.minimum.default",
    "tosa.ADD.default",
    "tosa.ARITHMETIC_RIGHT_SHIFT.default",
    "tosa.BITWISE_AND.default",
    "tosa.BITWISE_OR.default",
    "tosa.BITWISE_XOR.default",
    "tosa.EQUAL.default",
    "tosa.GREATER.default",
    "tosa.GREATER_EQUAL.default",
    "tosa.LOGICAL_AND.default",
    "tosa.LOGICAL_LEFT_SHIFT.default",
    "tosa.LOGICAL_OR.default",
    "tosa.LOGICAL_XOR.default",
    "tosa.MAXIMUM.default",
    "tosa.MINIMUM.default",
    "tosa.MUL.default",
    "tosa.POW.default",
    "tosa.SUB.default",
];

fn binary_rank_mismatch(target: &str) -> (LayoutGraph, String) {
    let mut g = LayoutGraph::new();
    let a = g.input("a", &[2, 3]);
    let b = g.input("b", &[3]);
    let op = raw_call(
        &mut g,
        "op",
        target,
        vec![node(a), node(b)],
        &[2, 3],
        "torch.float32",
    );
    g.output(&[op]);
    let name = g.name(op);
    (g, name)
}

/// Every rc4 target, in both TOSA spellings, rank-matches; binary ops rc4 does
/// not list are left alone.
#[test]
fn match_arg_ranks_target_set_is_rc4s() {
    let backend_spellings: Vec<String> = RC4_RANK_MATCH_TARGETS
        .iter()
        .filter_map(|target| target.strip_prefix("tosa."))
        .map(|rest| format!("backend.tosa.{rest}"))
        .collect();
    let targets = RC4_RANK_MATCH_TARGETS
        .iter()
        .copied()
        .chain(backend_spellings.iter().map(String::as_str));
    for target in targets {
        let (mut g, op) = binary_rank_mismatch(target);
        assert!(match_ranks(&mut g), "{target}");
        let op = g.id(&op);
        let view = g.graph().node(op).args[1].as_node().unwrap();
        assert_eq!(g.graph().node(view).target, VIEW, "{target}");
        assert_eq!(g.shape(view), &[1, 3], "{target}");
    }
    for target in [
        "aten.atan2.default",
        "aten.logical_and.default",
        "aten.fmod.Tensor",
        "aten.add.Scalar",
        "aten.where.ScalarOther",
        "tosa.INTDIV.default",
        "backend.tosa.INTDIV.default",
        "tosa.LOGICAL_RIGHT_SHIFT.default",
        "backend.tosa.SELECT.default",
        "tosa.TABLE.default",
    ] {
        let (mut g, _) = binary_rank_mismatch(target);
        assert!(!match_ranks(&mut g), "{target}");
    }
}

/// A rank-0 operand becomes all ones (`[1] * (max_rank - 0)`, `:94`).
#[test]
fn match_arg_ranks_lifts_a_rank0_operand() {
    let mut g = LayoutGraph::new();
    let a = g.input("a", &[2, 3]);
    let s = g.input("s", &[]);
    let add = g.call(ADD, vec![node(a), node(s)]);
    g.output(&[add]);
    let add = g.name(add);

    assert!(match_ranks(&mut g));

    let add = g.id(&add);
    let view = g.graph().node(add).args[1].as_node().unwrap();
    assert_eq!(g.graph().node(view).args[1].as_int_list(), Some(vec![1, 1]));
    assert_eq!(g.shape(view), &[1, 1]);
    assert_vals_match_retrace(&g);
}

/// rc4 reads only `node.args` (`:113-117`): a tensor passed by keyword neither
/// sets the maximum rank nor gets a view.
#[test]
fn match_arg_ranks_ignores_keyword_tensors() {
    let mut g = LayoutGraph::new();
    let a = g.input("a", &[3]);
    let b = g.input("b", &[2, 3]);
    let add = g.call_with_kwargs(ADD, vec![node(a)], &[("other", node(b))]);
    g.output(&[add]);

    assert!(!match_ranks(&mut g));
    g.assert_targets(&[ADD]);
}

/// Each target gets its own view of a shared lower-rank operand, inserted
/// directly before it (`inserting_before(node)`, `:95`), and only that
/// target's use moves (`node.replace_input_with`, `:102`).
#[test]
fn match_arg_ranks_gives_each_target_its_own_view() {
    let mut g = LayoutGraph::new();
    let a = g.input("a", &[2, 3]);
    let b = g.input("b", &[3]);
    let add = g.call(ADD, vec![node(a), node(b)]);
    let relu = g.call(RELU, vec![node(add)]);
    let mul = g.call(MUL, vec![node(relu), node(b)]);
    let neg = g.call(NEG, vec![node(b)]);
    g.output(&[mul, neg]);
    let (b, add, mul, neg) = (g.name(b), g.name(add), g.name(mul), g.name(neg));

    assert!(match_ranks(&mut g));

    g.assert_targets(&[VIEW, ADD, RELU, VIEW, MUL, NEG]);
    let order = names(&g);
    let position = |name: &str| order.iter().position(|n| n == name).unwrap();
    let view_of = |target: &str| g.graph().node(g.id(target)).args[1].as_node().unwrap();
    let (add_view, mul_view) = (view_of(&add), view_of(&mul));
    assert_ne!(add_view, mul_view);
    assert_eq!(position(&g.name(add_view)) + 1, position(&add));
    assert_eq!(position(&g.name(mul_view)) + 1, position(&mul));
    assert_eq!(g.graph().node(g.id(&neg)).args[0].as_node(), Some(g.id(&b)));
    assert_vals_match_retrace(&g);
}

/// The view is created without `from_node` (`:96-101`), so the retrace leaves
/// it only `val` (the operand's dtype at the new rank) and no qparams; the
/// target keeps its index-keyed input qparams.
#[test]
fn match_arg_ranks_view_has_the_operand_dtype_and_no_qparams() {
    let mut g = LayoutGraph::new();
    let a = g.typed_input("a", &[2, 3], "torch.int32");
    let b = g.typed_input("b", &[3], "torch.int32");
    let add = g.call(ADD, vec![node(a), node(b)]);
    g.output(&[add]);
    let qp = qparam(0.5, 0, "torch.int32");
    g.graph_mut()
        .node_mut(b)
        .meta
        .set_single_output_qparam(qp.clone());
    g.graph_mut()
        .node_mut(add)
        .meta
        .set_binary_input_output_qparams(qp.clone(), qp.clone(), qp.clone());
    let add = g.name(add);

    assert!(match_ranks(&mut g));

    let add = g.id(&add);
    let view = g.graph().node(add).args[1].as_node().unwrap();
    assert_eq!(g.dtype(view), "torch.int32");
    assert!(!g.graph().node(view).meta.has_qparam_metadata());
    assert_eq!(g.graph().node(add).meta.input_qparam_at(1), Some(&qp));
}

// ---------------------------------------------------------------------------
// NormalizeDelegateIOLayoutPass
// ---------------------------------------------------------------------------

/// venv: `(x, x + y)` with channels-last `x`, `y`. The passthrough output is
/// rewired to `x`'s logical permute like any other user and gets no output
/// permute of its own, because the permute's value is contiguous; the add gets
/// one. Order: `x, x_permute, y, y_permute, add, add_permute`.
#[test]
fn normalize_io_passthrough_output_keeps_the_logical_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let y = g.input("y", &[1, 3, 4, 5]);
    let add = g.call(ADD, vec![node(x), node(y)]);
    g.output(&[x, add]);
    for id in [x, y, add] {
        set_dim_order(&mut g, id, &NHWC);
    }
    let add = g.name(add);

    assert!(normalize(&mut g));

    assert_eq!(
        names(&g),
        [
            "x",
            "x_logical",
            "y",
            "y_logical",
            add.as_str(),
            &format!("{add}_dim_order"),
            "output"
        ]
    );
    assert_eq!(
        output_names(&g),
        ["x_logical".to_string(), format!("{add}_dim_order")]
    );
    assert_eq!(g.shape(g.id("x")), &[1, 4, 5, 3]);
    assert_eq!(g.shape(g.id(&format!("{add}_dim_order"))), &[1, 4, 5, 3]);
    assert_vals_match_retrace(&g);
}

/// `(x, x)` with channels-last `x`: both outputs read the one logical permute.
#[test]
fn normalize_io_input_returned_twice_shares_its_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    g.output(&[x, x]);
    set_dim_order(&mut g, x, &NHWC);

    assert!(normalize(&mut g));

    g.assert_targets(&[PERMUTE]);
    assert_eq!(output_names(&g), ["x_logical", "x_logical"]);
}

/// `is_param_node` (`:48`): parameters, buffers and lifted constants keep
/// their exported layout even when it is not the identity, as a
/// channels-last conv weight is (venv: `p_conv_weight` has dim order
/// `(0, 2, 3, 1)`).
#[test]
fn normalize_io_leaves_channels_last_parameters_alone() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let w = parameter(
        g.graph_mut(),
        "p_conv_weight",
        f32_tensor(vec![3, 1, 1, 1], &[1.0, 2.0, 3.0]),
    );
    g.graph_mut().node_mut(w).meta = NodeMeta::tensor(vec![3, 1, 1, 1], "torch.float32");
    set_dim_order(&mut g, w, &NHWC);
    let relu = g.call(RELU, vec![node(x)]);
    let out = raw_call(
        &mut g,
        "conv",
        "aten.convolution.default",
        vec![node(relu), node(w)],
        &[1, 3, 4, 5],
        "torch.float32",
    );
    g.output(&[out]);

    assert!(!normalize(&mut g));
    assert_eq!(g.shape(w), &[3, 1, 1, 1]);
}

/// An unused channels-last input still takes its physical shape, and its
/// permute is created with no users; rc4 does not eliminate it either.
#[test]
fn normalize_io_rewrites_an_unused_input_too() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let y = g.input("y", &[2]);
    let neg = g.call(NEG, vec![node(y)]);
    g.output(&[neg]);
    set_dim_order(&mut g, x, &NHWC);

    assert!(normalize(&mut g));

    assert_eq!(g.shape(g.id("x")), &[1, 4, 5, 3]);
    assert_eq!(names(&g)[..3], ["x", "x_logical", "y"]);
    assert!(!g.graph().has_live_users(g.id("x_logical")));
}

/// `replace_input_with` rewrites keyword uses too (torch/fx/node.py:868-884).
#[test]
fn normalize_io_rewires_keyword_uses() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let y = g.input("y", &[1, 3, 4, 5]);
    let add = g.call_with_kwargs(ADD, vec![node(y)], &[("other", node(x))]);
    g.output(&[add]);
    set_dim_order(&mut g, x, &NHWC);
    let add = g.name(add);

    assert!(normalize(&mut g));

    let add = g.graph().node(g.id(&add));
    assert_eq!(add.kwargs["other"].as_node(), Some(g.id("x_logical")));
}

/// `_rewrite_output_arg` returns anything that is not a node, tuple or list
/// unchanged (`:119`).
#[test]
fn normalize_io_keeps_none_outputs() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let relu = g.call(RELU, vec![node(x)]);
    g.graph_mut().add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        vec![IrArg::List(vec![node(relu), IrArg::None])],
        NodeMeta::default(),
    );
    set_dim_order(&mut g, relu, &NHWC);
    let relu = g.name(relu);

    assert!(normalize(&mut g));

    assert_eq!(
        output_names(&g),
        [format!("{relu}_dim_order"), "None".to_string()]
    );
}

/// An empty output tuple, which FX import turns into `IrArg::None`, is left as
/// it is (`:119`).
#[test]
fn normalize_io_keeps_an_empty_output_tuple() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    g.graph_mut().add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        vec![IrArg::None],
        NodeMeta::default(),
    );
    set_dim_order(&mut g, x, &NHWC);

    assert!(normalize(&mut g));

    assert!(matches!(outputs(&g)[..], [IrArg::None]));
    g.assert_targets(&[PERMUTE]);
}

/// A rank-1 input has dim order `(0,)` and a rank-0 input `()`: both are the
/// identity (`:42-43`).
#[test]
fn normalize_io_low_rank_inputs_are_identities() {
    let mut g = LayoutGraph::new();
    let a = g.input("a", &[3]);
    let s = g.input("s", &[]);
    let add = g.call(ADD, vec![node(a), node(s)]);
    g.output(&[add]);
    set_dim_order(&mut g, a, &[0]);
    set_dim_order(&mut g, s, &[]);

    assert!(!normalize(&mut g));
}

/// The input permute copies the placeholder's meta (`from_node=node`) and
/// empties every qparam map present (`arm_pass_utils.py:173-182`).
#[test]
fn normalize_io_input_permute_empties_both_qparam_maps() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 3, 4, 5], "torch.int8");
    let relu = g.call(RELU, vec![node(x)]);
    g.output(&[relu]);
    set_dim_order(&mut g, x, &NHWC);
    g.graph_mut()
        .node_mut(x)
        .meta
        .set_both_qparams(qparam(0.5, 0, "torch.int8"));

    assert!(normalize(&mut g));

    let meta = &g.graph().node(g.id("x_logical")).meta;
    assert!(meta.input_qparams().is_some_and(QParamMap::is_empty));
    assert!(meta.output_qparams().is_some_and(QParamMap::is_empty));
    assert_eq!(meta.val_dtype(), Some("torch.int8"));
    // The placeholder keeps its own qparams.
    assert!(g.graph().node(g.id("x")).meta.output_qparam().is_some());
}

/// venv: after rc4's retrace, `relu` downstream of the logical permute is
/// contiguous (`dim_order (0, 1, 2, 3)`). The port leaves the channels-last
/// dim order it was imported with.
#[test]
#[ignore = "BUG (minor): normalize_delegate_io leaves interior nodes' val_dim_order as imported; rc4's retrace makes them contiguous"]
fn normalize_io_interior_nodes_become_contiguous() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 4, 5]);
    let relu = g.call(RELU, vec![node(x)]);
    g.output(&[relu]);
    set_dim_order(&mut g, x, &NHWC);
    set_dim_order(&mut g, relu, &NHWC);
    let relu = g.name(relu);

    assert!(normalize(&mut g));

    assert_dim_order_fits(&g, g.id(&relu));
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

/// The same input graph gives the same output graph, names included, for each
/// pass (no hash-order dependence).
#[test]
fn every_pass_is_deterministic() {
    fn build() -> LayoutGraph {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let y = g.input("y", &[5]);
        let add = g.call(ADD, vec![node(x), node(y)]);
        let v0 = g.view(add, &[1, 3, 20]);
        let relu = g.call(RELU, vec![node(v0)]);
        let v1 = g.view(relu, &[1, 1, 3, 4, 5]);
        let p = g.permute(v1, &[0, 1, 3, 4, 2]);
        let neg = g.call(NEG, vec![node(p)]);
        g.output(&[neg, add]);
        set_dim_order(&mut g, x, &NHWC);
        set_dim_order(&mut g, add, &NHWC);
        g
    }
    for pass in [normalize, match_ranks, high_rank, fuse] {
        let (mut first, mut second) = (build(), build());
        assert!(pass(&mut first));
        assert!(pass(&mut second));
        assert_eq!(
            live_signature(first.graph()),
            live_signature(second.graph())
        );
    }
}
