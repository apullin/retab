//! Ports of rc4's own tests for `RemovePermutesAroundElementwiseOps` and
//! `RemovePermutesAroundElementwiseTosaOps` (APU-158):
//!
//! - `backends/arm/test/passes/test_remove_permutes_around_elementwise_tosa_ops.py`
//!   (10 tests), run with the TOSA configuration unless the test builds the
//!   base pass;
//! - the remove-permutes tests of
//!   `backends/transforms/test/test_permute_optimization_passes.py`
//!   (`RemovePermutesAcrossViewTest`, `RemovePermutesAroundRepeatInterleaveTest`,
//!   `RemovePermutesAroundElementwiseOpsTest` and `LayoutPermuteVisibilityTest`,
//!   47 tests), run with the base pass as rc4 constructs it there.
//!
//! The sources are rc4 with the release/1.6 backports of #24315 and of
//! #24469's mixed-rank broadcast guard, and the fix stacked on them that keeps
//! the guard from counting a TABLE's table and keeps only that table out of
//! the constant edges (fork branch `retab/pick-1.6/guard-table-exemption`,
//! c3172c0028), whose fixes the port follows. They add six of those tests:
//! #24315's `test_lower_rank_constant_is_transposed_into_region_layout`,
//! #24469's `test_numel_one_operand_does_not_block_region` and
//! `test_cat_with_different_concat_sizes_does_not_block_region`, the guard
//! backport's `test_squeezed_lower_rank_operand_blocks_region`, and the fix's
//! `test_remove_permutes_around_rms_norm_with_table_tosa_INT` and
//! `test_remove_permutes_around_table_with_constant_input_tosa_INT`. Each test
//! replays the calls its ExecuTorch test made, recorded in
//! tests/fixtures/remove_permutes_upstream/rc4_test_calls.txt by running the
//! test there with the pass instrumented (see the recorder's docstring): the
//! port must return that pass's graph exactly (see `replay` for what is
//! compared). Then the test's own assertions run on the port's result. The
//! numeric checks (`validate_numerics`, `assert_close`) need execution, which
//! RETAB does not have; matching the recorded graph node for node stands in
//! for them.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::layout::remove_permutes::{
    is_constant_in, IrRemovePermutesAroundElementwiseOpsPass, RemovePermutesConfig,
};
use crate::layout::replay::{self, Case, Options};

const CALLS: &str = include_str!("../../tests/fixtures/remove_permutes_upstream/rc4_test_calls.txt");
const PERMUTE: &str = "aten.permute_copy.default";
const VIEW: &str = "aten.view_copy.default";

fn case(name: &str) -> Case {
    replay::parse(CALLS)
        .expect("the recorded calls parse")
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no recorded call {name}"))
}

/// Replay the recorded call `name` with `config`; the port must match rc4.
/// Returns the port's graph and modified flag for the test's own assertions.
fn replay_call(config: RemovePermutesConfig, name: &str) -> (IrGraph, bool) {
    replay_case(config, case(name))
}

fn replay_case(config: RemovePermutesConfig, case: Case) -> (IrGraph, bool) {
    let name = case.name.clone();
    let pass = IrRemovePermutesAroundElementwiseOpsPass { config };
    let result = replay::replay_case(&pass, &case, Options::default());
    assert!(result.mismatch.is_none(), "{}", result.report);
    let mut graph = replay::build(&case.before);
    let modified = pass.run(&mut graph);
    assert_eq!(modified, case.modified, "{name}: modified flag");
    (graph, modified)
}

fn base(name: &str) -> (IrGraph, bool) {
    replay_call(RemovePermutesConfig::base(), name)
}

fn tosa(name: &str) -> (IrGraph, bool) {
    replay_call(RemovePermutesConfig::tosa(), name)
}

fn live_calls(graph: &IrGraph, target: &str) -> Vec<NodeId> {
    graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| {
            let node = graph.node(id);
            node.op == NodeOp::CallFunction && node.target == target
        })
        .collect()
}

/// rc4's `count_node`.
fn count(graph: &IrGraph, target: &str) -> usize {
    live_calls(graph, target).len()
}

/// The node `replay::build` named after its position in the recorded graph.
fn named(graph: &IrGraph, name: &str) -> NodeId {
    graph
        .by_name(name)
        .unwrap_or_else(|| panic!("no node named {name}"))
        .id
}

fn int_list(graph: &IrGraph, id: NodeId, index: usize) -> Vec<i64> {
    crate::layout::shape::int_list(&graph.node(id).args[index])
        .unwrap_or_else(|| panic!("arg {index} of {} is not an int list", graph.node(id).name))
}

fn node_args(graph: &IrGraph, id: NodeId) -> Vec<NodeId> {
    graph.node(id).args.iter().filter_map(IrArg::as_node).collect()
}

// ---------------------------------------------------------------------------
// test_remove_permutes_around_elementwise_tosa_ops.py
// ---------------------------------------------------------------------------

/// `test_constant_input_cache_refreshes_for_reused_pass`: the subclass reads
/// its constants from the graph signature on every call. RETAB marks a
/// signature constant with tensor data on its own node, so a constant of one
/// graph cannot leak into another; a placeholder without that data and
/// without a constant prefix is not a constant.
#[test]
fn constant_input_cache_refreshes_for_reused_pass() {
    let config = RemovePermutesConfig::tosa();
    for (call, name) in [(0, "first_constant"), (1, "second_constant")] {
        let call = format!("test_constant_input_cache_refreshes_for_reused_pass#{call}");
        let (mut graph, _) = tosa(&call);
        let constant = named(&graph, name);
        assert!(is_constant_in(&mut graph, &config, constant));
        graph.node_mut(constant).tensor = None;
        assert!(!is_constant_in(&mut graph, &config, constant));
    }
}

/// `test_extra_permutable_ops_makes_op_permutable`.
#[test]
fn extra_permutable_ops_makes_op_permutable() {
    let (baseline, modified) = base("test_extra_permutable_ops_makes_op_permutable#0");
    assert!(!modified);
    assert_eq!(count(&baseline, PERMUTE), 2);

    let config = RemovePermutesConfig {
        extra_permutable_ops: vec!["backend.tosa.RESCALE.default"],
        ..RemovePermutesConfig::base()
    };
    let (result, modified) = replay_call(config, "test_extra_permutable_ops_makes_op_permutable#1");
    assert!(modified);
    assert_eq!(count(&result, PERMUTE), 0);
    assert_eq!(count(&result, "tosa.RESCALE.default"), 1);
}

/// `test_remove_permutes_around_rescale_tosa_INT`.
#[test]
fn remove_permutes_around_rescale_tosa_int() {
    let (graph, modified) = tosa("test_remove_permutes_around_rescale_tosa_INT#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    assert_eq!(count(&graph, "tosa.RESCALE.default"), 1);
}

/// `test_terminal_sink_view_broadcast_is_optimized_tosa_INT`. The test gives
/// its RESCALE a float val (`torch.randn`), which rc4's closing retrace
/// replaces with the int8 RESCALE computes. The RESCALE sits below the sink,
/// outside the region, and the port re-derives only the values a rewrite can
/// change, so the replay starts from the consistent val.
#[test]
fn terminal_sink_view_broadcast_is_optimized_tosa_int() {
    let mut case = case("test_terminal_sink_view_broadcast_is_optimized_tosa_INT#0");
    let rescale = &mut case.before.nodes[5];
    assert!(matches!(&rescale.op, replay::Op::Call { target, .. } if target == "tosa.RESCALE.default"));
    rescale.val = replay::Val::Tensor {
        dtype: "torch.int8".to_string(),
        shape: vec![1, 1, 4],
    };
    let (graph, modified) = replay_case(RemovePermutesConfig::tosa(), case);
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    let sub = live_calls(&graph, "aten.sub.Tensor")[0];
    let rescale = live_calls(&graph, "tosa.RESCALE.default")[0];
    assert_eq!(node_args(&graph, sub), [named(&graph, "direct"), rescale]);
}

/// `test_remove_permutes_around_singleton_view_and_gate_region_tosa_INT`.
#[test]
fn remove_permutes_around_singleton_view_and_gate_region_tosa_int() {
    let (graph, modified) =
        tosa("test_remove_permutes_around_singleton_view_and_gate_region_tosa_INT#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    let views = live_calls(&graph, VIEW);
    let pool_view = views
        .iter()
        .copied()
        .find(|&id| node_args(&graph, id) != [named(&graph, "gate_source")])
        .expect("the pool view");
    let gate = views
        .iter()
        .copied()
        .find(|&id| node_args(&graph, id) == [named(&graph, "gate_source")])
        .expect("the gate view");
    assert_eq!(int_list(&graph, pool_view, 1), [1, 8, 1, 4]);
    assert_eq!(int_list(&graph, gate, 1), [1, 1, 4]);
}

/// `test_remove_permutes_around_amax_reduction_region_tosa_INT`.
#[test]
fn remove_permutes_around_amax_reduction_region_tosa_int() {
    let (graph, modified) = tosa("test_remove_permutes_around_amax_reduction_region_tosa_INT#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    let maximum = live_calls(&graph, "aten.amax.default")[0];
    assert_eq!(graph.node(maximum).args[1].as_int(), Some(1));
}

/// `test_remove_permutes_around_rms_norm_with_table_tosa_INT`, the test of the
/// fix that keeps the mixed-rank guard from counting a TABLE's table: an int8
/// RMS norm between a permute and its inverse, whose rsqrt is a TABLE. Its
/// rank-1 table is no broadcast operand, so both permutes go, the sum's dim
/// follows the layout, and the TABLE keeps its input and its table as they
/// are. Without the fix the guard kept both permutes.
#[test]
fn remove_permutes_around_rms_norm_with_table_tosa_int() {
    let (graph, modified) = tosa("test_remove_permutes_around_rms_norm_with_table_tosa_INT#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    assert_eq!(count(&graph, VIEW), 0);
    // `sum_of_squares`, `table_input` and `rsqrt` are %5, %6 and %7 in the
    // recorded graph.
    assert_eq!(int_list(&graph, named(&graph, "n5"), 1), [3]);
    assert_eq!(
        node_args(&graph, named(&graph, "n7")),
        [named(&graph, "n6"), named(&graph, "b_rsqrt_table_constant")]
    );
}

/// `test_remove_permutes_around_table_with_constant_input_tosa_INT`, the
/// fix's test that only a TABLE's table keeps its layout: a mul between a
/// permute and its inverse reads a TABLE whose data input is a constant. The
/// data input gets the region's compensating permute, the one permute left,
/// and the TABLE keeps its table. rc4 drops every constant input of a TABLE,
/// so once the guard skips TABLE, the mul read the data input in the old
/// layout and rc4's retrace raised.
#[test]
fn remove_permutes_around_table_with_constant_input_tosa_int() {
    let (graph, modified) =
        tosa("test_remove_permutes_around_table_with_constant_input_tosa_INT#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 1);
    // `gate`, the TABLE, is %4 in the recorded graph.
    let gate = node_args(&graph, named(&graph, "n4"));
    assert_eq!(gate.len(), 2);
    let state_permute = gate[0];
    assert_eq!(graph.node(state_permute).target, PERMUTE);
    assert_eq!(node_args(&graph, state_permute), [named(&graph, "b_state")]);
    assert_eq!(int_list(&graph, state_permute, 1), [0, 2, 3, 1]);
    assert_eq!(gate[1], named(&graph, "b_sigmoid_table_constant"));
}

/// `test_remove_permutes_around_gelu_with_folded_scalar_constants_tosa_FP`:
/// numel-1 constants stay wired, so every permute in the region cancels.
#[test]
fn remove_permutes_around_gelu_with_folded_scalar_constants_tosa_fp() {
    let (graph, modified) =
        tosa("test_remove_permutes_around_gelu_with_folded_scalar_constants_tosa_FP#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    assert_eq!(count(&graph, "aten.erf.default"), 1);
}

/// `test_remove_permutes_skips_stale_shared_boundary_subgraph_tosa_FP`.
#[test]
fn remove_permutes_skips_stale_shared_boundary_subgraph_tosa_fp() {
    let (graph, modified) =
        tosa("test_remove_permutes_skips_stale_shared_boundary_subgraph_tosa_FP#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 1);
    // `second_mul` is %5 in the recorded graph.
    let second_mul = named(&graph, "n5");
    assert_eq!(
        graph.node(second_mul).args[1].as_node(),
        Some(named(&graph, "p_layer_norm_weight"))
    );
}

// ---------------------------------------------------------------------------
// test_permute_optimization_passes.py
// ---------------------------------------------------------------------------

/// The cases that expect a rewrite removing every permute.
#[test]
fn regions_whose_permutes_all_cancel() {
    for test in [
        "test_permute_view_squeeze_elementwise_view_unsqueeze_permute",
        "test_4d_permute_squeeze_clamp_3d_permute",
        "test_permute_view_add_sub_mul_view_permute",
        "test_permute_squeeze_clamp_add_permute",
        "test_permute_unsqueeze_copy_mul_squeeze_copy_permute",
        "test_4d_permute_squeeze_copy_clamp_3d_permute",
        "test_permute_unsqueeze_copy_neg_dim_mul_squeeze_copy_permute",
        "test_unsqueeze_at_moved_position",
        "test_squeeze_dims_multiple_unit_dims",
        "test_view_copy_multi_unit_dim_rank_change",
        "test_permutation_sink_view_terminal_broadcast_is_optimized",
        "test_repeat_interleave_with_keyword_arguments",
        "test_repeat_interleave_after_nop_stretch",
        "test_repeat_interleave_scale_four_on_middle_dim",
        "test_repeat_interleave_composes_with_elementwise",
        "test_resnet_stretch_region",
        "test_deep_elementwise_region_does_not_recurse",
    ] {
        let (graph, modified) = base(&format!("{test}#0"));
        assert!(modified, "{test}");
        assert_eq!(count(&graph, PERMUTE), 0, "{test}");
    }
}

/// The cases that expect the pass to leave the graph alone, with rc4's
/// permute counts.
#[test]
fn regions_the_pass_must_not_rewrite() {
    for (test, permutes) in [
        ("test_permute_unsqueeze_cat_mul_squeeze_permute#0", 3),
        ("test_no_fire_non_squeeze_view#0", 2),
        ("test_squeeze_on_non_unit_dim_is_not_optimized#0", 2),
        ("test_squeeze_on_non_unit_dim_is_not_optimized#1", 2),
        ("test_upstream_view_rank_mismatch_no_crash#0", 2),
        ("test_split_sink_view_is_not_remapped_for_broadcast#0", 2),
        ("test_upstream_squeeze_view_rank_mismatch_no_crash#0", 2),
        ("test_broadcast_rank_increase_no_crash#0", 1),
        ("test_non_merging_view_after_expand_is_not_optimized#0", 1),
        ("test_expand_with_extra_user_is_not_optimized#0", 2),
        ("test_no_permutes_is_noop#0", 0),
    ] {
        let (graph, modified) = base(test);
        assert!(!modified, "{test}");
        assert_eq!(count(&graph, PERMUTE), permutes, "{test}");
    }
}

/// `test_target_filter_blocks_permutation`.
#[test]
fn target_filter_blocks_permutation() {
    let (unfiltered, _) = base("test_target_filter_blocks_permutation#0");
    assert_eq!(count(&unfiltered, PERMUTE), 0);

    let config = RemovePermutesConfig {
        target_filters: vec![("aten.mul.Tensor", |_, _| false)],
        ..RemovePermutesConfig::base()
    };
    let (graph, modified) = replay_call(config, "test_target_filter_blocks_permutation#1");
    assert!(!modified);
    assert_eq!(count(&graph, PERMUTE), 2);
}

/// `test_permute_constant_pad_nd_permute` and `test_permute_aten_pad_permute`.
#[test]
fn pads_are_remapped() {
    for (test, target) in [
        ("test_permute_constant_pad_nd_permute#0", "aten.constant_pad_nd.default"),
        ("test_permute_aten_pad_permute#0", "aten.pad.default"),
    ] {
        let (graph, modified) = base(test);
        assert!(modified, "{test}");
        assert_eq!(count(&graph, PERMUTE), 0, "{test}");
        let pads = live_calls(&graph, target);
        assert_eq!(pads.len(), 1, "{test}");
        assert_eq!(int_list(&graph, pads[0], 1), [0, 1], "{test}");
    }
}

/// `test_4d_permute_squeeze_view_slice_mul_3d_permute`: the end permute does
/// not match the adapted identity, so it absorbs the start permute and stays.
#[test]
fn squeeze_of_a_moved_unit_dim_keeps_the_end_permute() {
    let (graph, modified) = base("test_4d_permute_squeeze_view_slice_mul_3d_permute#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 1);
}

/// `test_permutation_sink_view_splitting_the_non_unit_dim`,
/// `test_permutation_sink_view_preserves_cat_layout` and
/// `test_permutation_sink_view_preserves_keyword_broadcast_layout`: the sink
/// keeps its shape.
#[test]
fn sink_views_keep_their_shape() {
    for (test, shape) in [
        ("test_permutation_sink_view_splitting_the_non_unit_dim#0", vec![1, 2, 2]),
        ("test_permutation_sink_view_preserves_cat_layout#0", vec![1, 1, 4]),
        (
            "test_permutation_sink_view_preserves_keyword_broadcast_layout#0",
            vec![1, 1, 4],
        ),
    ] {
        let (graph, modified) = base(test);
        assert!(modified, "{test}");
        assert_eq!(count(&graph, PERMUTE), 0, "{test}");
        let views = live_calls(&graph, VIEW);
        assert_eq!(views.len(), 1, "{test}");
        assert_eq!(int_list(&graph, views[0], 1), shape, "{test}");
    }
}

/// `test_shared_singleton_view_boundaries_are_not_remapped`: both outgoing
/// permutes stay; only the incoming one folds into them.
#[test]
fn shared_singleton_view_boundaries_are_not_remapped() {
    let (graph, _) = base("test_shared_singleton_view_boundaries_are_not_remapped#0");
    assert_eq!(count(&graph, PERMUTE), 2);
}

/// `test_removes_permutes_around_repeat_interleave`.
#[test]
fn removes_permutes_around_repeat_interleave() {
    let (graph, modified) = base("test_removes_permutes_around_repeat_interleave#0");
    assert!(modified);
    assert_eq!(count(&graph, PERMUTE), 0);
    assert_eq!(count(&graph, "aten.expand_copy.default"), 1);
}

/// `test_repeat_interleave_rank_mismatch_is_not_rewritten`: rc4's
/// `RankMismatchPass` reports a rank-3 permutation for the start permute.
#[test]
fn repeat_interleave_rank_mismatch_is_not_rewritten() {
    let case = case("test_repeat_interleave_rank_mismatch_is_not_rewritten#0");
    let graph = replay::build(&case.before);
    let start = live_calls(&graph, PERMUTE)[0];
    let config = RemovePermutesConfig {
        permutation_override: Some((graph.node(start).name.clone(), vec![0, 2, 1])),
        ..RemovePermutesConfig::base()
    };
    let (graph, modified) =
        replay_call(config, "test_repeat_interleave_rank_mismatch_is_not_rewritten#0");
    assert!(!modified);
    assert_eq!(count(&graph, PERMUTE), 2);
}

/// `test_region_ending_in_permute_then_view` and
/// `test_non_matching_end_permute_is_composed`: the end permute absorbs the
/// start permute.
#[test]
fn non_matching_end_permutes_are_composed() {
    for (test, dims) in [
        ("test_region_ending_in_permute_then_view#0", [1, 0, 2]),
        ("test_non_matching_end_permute_is_composed#0", [2, 1, 0]),
    ] {
        let (graph, modified) = base(test);
        assert!(modified, "{test}");
        let permutes = live_calls(&graph, PERMUTE);
        assert_eq!(permutes.len(), 1, "{test}");
        assert_eq!(int_list(&graph, permutes[0], 1), dims, "{test}");
    }
}

/// `test_view_seeded_region_with_shared_view`: rc4 checks numerics only;
/// replaying the call checks the port leaves the graph rc4 leaves.
#[test]
fn view_seeded_region_with_shared_view() {
    base("test_view_seeded_region_with_shared_view#0");
}

/// `test_chained_regions_absorb_into_last_permute`: one region folds per
/// call, so the chain needs three calls (each replayed), and running the port
/// to a fixed point from the first graph reaches rc4's last graph, one
/// permute carrying [2, 0, 1].
#[test]
fn chained_regions_absorb_into_last_permute() {
    for call in 0..3 {
        base(&format!("test_chained_regions_absorb_into_last_permute#{call}"));
    }
    let first = case("test_chained_regions_absorb_into_last_permute#0");
    let last = case("test_chained_regions_absorb_into_last_permute#2");
    let pass = IrRemovePermutesAroundElementwiseOpsPass {
        config: RemovePermutesConfig::base(),
    };
    let mut graph = replay::build(&first.before);
    let mut rounds = 0;
    while pass.run(&mut graph) {
        rounds += 1;
        assert!(rounds < 5, "the pass did not reach a fixed point");
    }
    let snapshot = replay::snapshot(&graph).expect("a valid graph");
    replay::compare(&last.after, &snapshot, Options::default())
        .unwrap_or_else(|mismatch| panic!("fixed point differs from rc4: {mismatch:?}"));
    let permutes = live_calls(&graph, PERMUTE);
    assert_eq!(permutes.len(), 1);
    assert_eq!(int_list(&graph, permutes[0], 1), [2, 0, 1]);
}

/// `test_permute_bypassed_by_one_region_is_kept_by_another`.
#[test]
fn permute_bypassed_by_one_region_is_kept_by_another() {
    let (_, modified) = base("test_permute_bypassed_by_one_region_is_kept_by_another#0");
    assert!(modified);
}

/// `test_lower_rank_constant_is_transposed_into_region_layout`, #24315's
/// regression test: a lower-rank constant whose non-unit dims the end
/// permutation reorders is viewed to the region's rank and permuted; one whose
/// non-unit dims keep their order gets a single view that drops only leading
/// 1s. rc4 views every such constant to the permuted shape minus its leading
/// dims: call #0's `[6, 4]` becomes `[4, 6]`, silently wrong output, and call
/// #1's `[4]` becomes `[1]`, which raises.
#[test]
fn lower_rank_constant_is_transposed_into_region_layout() {
    // Per recorded call: the view of the constant, and the permute after it.
    let calls: [(&[i64], Option<&[i64]>); 7] = [
        (&[1, 6, 4], Some(&[0, 2, 1])),
        (&[4, 1], None),
        (&[1, 6], None),
        (&[1, 4, 5, 3], Some(&[0, 3, 1, 2])),
        (&[1, 1, 5, 3], Some(&[0, 3, 1, 2])),
        (&[3, 1, 1], None),
        (&[1, 4, 1], None),
    ];
    for (call, (shape, dims)) in calls.into_iter().enumerate() {
        let test = format!("test_lower_rank_constant_is_transposed_into_region_layout#{call}");
        let (graph, modified) = base(&test);
        assert!(modified, "{test}");
        assert_eq!(
            count(&graph, PERMUTE),
            usize::from(dims.is_some()),
            "{test}"
        );
        let add = live_calls(&graph, "aten.add.Tensor")[0];
        let [input, mut operand] = node_args(&graph, add)[..] else {
            panic!("{test}: the add reads two nodes");
        };
        assert_eq!(input, named(&graph, "x"), "{test}");
        if let Some(dims) = dims {
            assert_eq!(graph.node(operand).target, PERMUTE, "{test}");
            assert_eq!(int_list(&graph, operand, 1), dims, "{test}");
            operand = node_args(&graph, operand)[0];
        }
        assert_eq!(graph.node(operand).target, VIEW, "{test}");
        assert_eq!(
            node_args(&graph, operand),
            [named(&graph, "p_const")],
            "{test}"
        );
        assert_eq!(int_list(&graph, operand, 1), shape, "{test}");
    }
}

/// `test_numel_one_operand_does_not_block_region` and
/// `test_cat_with_different_concat_sizes_does_not_block_region`, #24469's
/// tests of its mixed-rank broadcast guard: a numel-1 operand broadcasts the
/// same under any permutation, and cat never broadcasts, so neither keeps the
/// region's permutes.
#[test]
fn numel_one_operands_and_cat_pass_the_mixed_rank_guard() {
    for test in [
        "test_numel_one_operand_does_not_block_region#0",
        "test_cat_with_different_concat_sizes_does_not_block_region#0",
    ] {
        let (graph, modified) = base(test);
        assert!(modified, "{test}");
        assert_eq!(count(&graph, PERMUTE), 0, "{test}");
    }
}

/// `test_squeezed_lower_rank_operand_blocks_region`, the guard's regression
/// test in the release/1.6 backport: the add broadcasts the permuted
/// `[1, 1, 4, 4]` against its squeeze to `[4, 4]`. Without the permutes the
/// `[4, 4]` would line up against the last two dims of `[1, 4, 4, 1]`, and the
/// region would return `[1, 4, 4, 4]` instead of `[1, 4, 4, 1]`, so the guard
/// keeps them.
#[test]
fn squeezed_lower_rank_operand_blocks_region() {
    let (graph, modified) = base("test_squeezed_lower_rank_operand_blocks_region#0");
    assert!(!modified);
    assert_eq!(count(&graph, PERMUTE), 2);
    let output = graph.output_node().expect("an output");
    let result = match &output.args[0] {
        IrArg::List(results) => results[0].as_node(),
        result => result.as_node(),
    }
    .expect("the output reads a node");
    assert_eq!(graph.node_shape(result), Some(&[1, 4, 4, 1][..]));
}

/// `LayoutPermuteVisibilityTest.test_inverse_layout_copy_pair_around_elementwise_is_removed`:
/// `channels_last.permute_copy` counts as a permute.
#[test]
fn inverse_layout_copy_pair_around_elementwise_is_removed() {
    let (graph, modified) =
        base("test_inverse_layout_copy_pair_around_elementwise_is_removed#0");
    assert!(modified);
    assert_eq!(count(&graph, "channels_last.permute_copy.default"), 0);
}

/// Every recorded call is claimed by a test above.
#[test]
fn every_recorded_call_is_ported() {
    let ported = [
        "test_constant_input_cache_refreshes_for_reused_pass",
        "test_extra_permutable_ops_makes_op_permutable",
        "test_remove_permutes_around_rescale_tosa_INT",
        "test_terminal_sink_view_broadcast_is_optimized_tosa_INT",
        "test_remove_permutes_around_singleton_view_and_gate_region_tosa_INT",
        "test_remove_permutes_around_amax_reduction_region_tosa_INT",
        "test_remove_permutes_around_rms_norm_with_table_tosa_INT",
        "test_remove_permutes_around_table_with_constant_input_tosa_INT",
        "test_remove_permutes_around_gelu_with_folded_scalar_constants_tosa_FP",
        "test_remove_permutes_skips_stale_shared_boundary_subgraph_tosa_FP",
        "test_permute_view_squeeze_elementwise_view_unsqueeze_permute",
        "test_4d_permute_squeeze_clamp_3d_permute",
        "test_permute_view_add_sub_mul_view_permute",
        "test_permute_squeeze_clamp_add_permute",
        "test_permute_unsqueeze_copy_mul_squeeze_copy_permute",
        "test_4d_permute_squeeze_copy_clamp_3d_permute",
        "test_permute_unsqueeze_copy_neg_dim_mul_squeeze_copy_permute",
        "test_unsqueeze_at_moved_position",
        "test_squeeze_dims_multiple_unit_dims",
        "test_view_copy_multi_unit_dim_rank_change",
        "test_permutation_sink_view_terminal_broadcast_is_optimized",
        "test_repeat_interleave_with_keyword_arguments",
        "test_repeat_interleave_after_nop_stretch",
        "test_repeat_interleave_scale_four_on_middle_dim",
        "test_repeat_interleave_composes_with_elementwise",
        "test_resnet_stretch_region",
        "test_deep_elementwise_region_does_not_recurse",
        "test_permute_unsqueeze_cat_mul_squeeze_permute",
        "test_no_fire_non_squeeze_view",
        "test_squeeze_on_non_unit_dim_is_not_optimized",
        "test_upstream_view_rank_mismatch_no_crash",
        "test_split_sink_view_is_not_remapped_for_broadcast",
        "test_upstream_squeeze_view_rank_mismatch_no_crash",
        "test_broadcast_rank_increase_no_crash",
        "test_non_merging_view_after_expand_is_not_optimized",
        "test_expand_with_extra_user_is_not_optimized",
        "test_no_permutes_is_noop",
        "test_target_filter_blocks_permutation",
        "test_permute_constant_pad_nd_permute",
        "test_permute_aten_pad_permute",
        "test_4d_permute_squeeze_view_slice_mul_3d_permute",
        "test_permutation_sink_view_splitting_the_non_unit_dim",
        "test_permutation_sink_view_preserves_cat_layout",
        "test_permutation_sink_view_preserves_keyword_broadcast_layout",
        "test_shared_singleton_view_boundaries_are_not_remapped",
        "test_removes_permutes_around_repeat_interleave",
        "test_repeat_interleave_rank_mismatch_is_not_rewritten",
        "test_region_ending_in_permute_then_view",
        "test_non_matching_end_permute_is_composed",
        "test_view_seeded_region_with_shared_view",
        "test_chained_regions_absorb_into_last_permute",
        "test_permute_bypassed_by_one_region_is_kept_by_another",
        "test_lower_rank_constant_is_transposed_into_region_layout",
        "test_numel_one_operand_does_not_block_region",
        "test_cat_with_different_concat_sizes_does_not_block_region",
        "test_squeezed_lower_rank_operand_blocks_region",
        "test_inverse_layout_copy_pair_around_elementwise_is_removed",
    ];
    assert_eq!(ported.len(), 57);
    for case in replay::parse(CALLS).expect("the recorded calls parse") {
        let test = case.name.split('#').next().unwrap_or_default();
        assert!(ported.contains(&test), "recorded call {} is not ported", case.name);
    }
}
