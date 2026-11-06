//! Adversarial fidelity review of the APU-160 ports (FuseDuplicateUsers,
//! FuseIdenticalInputTransforms and MoveDataMovementOpsToSmallerDtype) against
//! ExecuTorch v1.6.0-rc4.
//!
//! Each test pins what rc4 does on a graph the port gets wrong. "venv" marks
//! results taken by running the rc4 passes in `.venv-et16`; the embedded
//! fixtures were printed by `gen_layout_replay.py`'s `run_case` for these
//! graphs, so `replay` compares the port with rc4's exact result. The tests
//! are ignored with the bug they show, so the suite stays green until the port
//! is fixed; run them with `cargo test --lib fuse_helpers_review -- --ignored`.

use crate::fx_ir::{IrPass, NodeId, QParam};
use crate::layout::ctx::LayoutCtx;
use crate::layout::fuse_duplicate_users::{fuse_duplicate_users, IrFuseDuplicateUsersPass, TOSA_EXCLUDED_TARGETS};
use crate::layout::move_data_movement::IrMoveDataMovementOpsToSmallerDtypePass;
use crate::layout::replay::{self, Options};
use crate::layout::testing::*;
use crate::test_ir::qparam;

/// The comparison the APU-160 replays use: the default plus the `modified` flag.
const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: false,
    modified: true,
};

// ---------------------------------------------------------------------------
// FuseDuplicateUsers: pending TABLE ops
// ---------------------------------------------------------------------------

/// A quantized sigmoid as RETAB holds it where rc4 already has a TABLE
/// (`layout::pending::is_pending_table`): int8 value, input and output qparams.
fn pending_sigmoid(g: &mut LayoutGraph, x: NodeId, output: QParam) -> NodeId {
    let sigmoid = g.call(SIGMOID, vec![node(x)]);
    g.graph_mut()
        .node_mut(sigmoid)
        .meta
        .set_unary_input_output_qparams(qparam(0.5, 0, "torch.int8"), output);
    sigmoid
}

/// `fuse_duplicate_users` is documented as the body of the propagation
/// driver's `fuse_horizontal` (propagate_view_copy_permute_pass.py:569-576),
/// which runs at phase 6, after InsertTableOpsPass (`arm_pass_manager.py:707`).
/// There each TABLE reads its own `b_<name>_table_constant` placeholder
/// (insert_table_ops.py), so two TABLEs of one input are never duplicates.
///
/// venv, `x:int8[2,3]` read by two sigmoids with equal qparams:
/// - FuseDuplicateUsersPass alone (the `:587` position): modified=True,
///   `['aten.sigmoid.default']`;
/// - InsertTableOpsPass, then FuseDuplicateUsersPass (phase 6):
///   `['tosa.TABLE.default', 'tosa.TABLE.default']`, placeholders
///   `b_aten_sigmoid_default_table_constant` and `..._1_...`; modified=False.
///
/// RETAB still holds both as `aten.sigmoid` with the same args, and the port
/// fuses them; it never consults `is_pending_table`, unlike the P1 port of
/// FuseViewCopyTransform (`p1_review_tests::fuse_view_copy_stops_at_a_pending_table`).
#[test]
#[ignore = "BUG: fuse_duplicate_users (the driver's fuse_horizontal) fuses pending TABLE ops; rc4's phase-6 TABLEs each read their own table constant and never fuse"]
fn fuse_duplicate_users_in_the_driver_keeps_pending_tables_apart() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3], "torch.int8");
    let first = pending_sigmoid(&mut g, x, qparam(0.5, 0, "torch.int8"));
    let second = pending_sigmoid(&mut g, x, qparam(0.5, 0, "torch.int8"));
    g.output(&[first, second]);

    let fused = {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        let fused = fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS);
        if fused {
            ctx.finish();
        }
        fused
    };

    assert!(!fused, "rc4's phase-6 FuseDuplicateUsers reports modified=False");
    assert_eq!(g.count(SIGMOID), 2, "rc4 keeps both TABLEs");
}

/// `IrFuseDuplicateUsersPass` is documented as the standalone pass at both
/// `arm_pass_manager.py:587` and `:765`. At `:765` rc4 has TABLE nodes, and
/// FuseEqualPlaceholdersPass (`:755`) has merged only table constants with
/// equal data, so two TABLEs of one input fuse only when their tables match.
///
/// venv, `x:int8[2,3]` read by two sigmoids, output scales 0.5 and 0.25:
/// - `:587` (FuseDuplicateUsersPass alone): modified=True, one
///   `aten.sigmoid.default` (rc4 ignores qparams there, which the port matches);
/// - `:765` (InsertTableOpsPass, FuseEqualPlaceholdersPass, then
///   FuseDuplicateUsersPass): modified=False, two `tosa.TABLE.default`.
///   With equal output scales the same sequence ends with one TABLE.
///
/// The port fuses the two pending sigmoids at `:765` whatever their qparams,
/// so one consumer reads a value quantized for the other: a miscompile, not
/// just an op-count difference. One `IrFuseDuplicateUsersPass` cannot match
/// rc4 at both positions.
#[test]
#[ignore = "BUG: IrFuseDuplicateUsersPass at :765 fuses pending TABLE ops whose output qparams differ; rc4's TABLE constants differ there, so it keeps both"]
fn fuse_duplicate_users_at_765_keeps_tables_with_different_qparams() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3], "torch.int8");
    let first = pending_sigmoid(&mut g, x, qparam(0.5, 0, "torch.int8"));
    let second = pending_sigmoid(&mut g, x, qparam(0.25, 0, "torch.int8"));
    g.output(&[first, second]);

    let fused = IrFuseDuplicateUsersPass.run(g.graph_mut());

    assert!(!fused, "rc4 at :765 reports modified=False");
    assert_eq!(g.count(SIGMOID), 2, "rc4 keeps both TABLEs");
}

// ---------------------------------------------------------------------------
// MoveDataMovementOpsToSmallerDtype: the closing retrace
// ---------------------------------------------------------------------------

/// venv: rc4 moves the permute above the widening RESCALE, which rewires the
/// RESCALE's other user, the `add`, to a `[1,3,4,2]` value. rc4's closing
/// retrace then raises: "RuntimeError: Attempting to broadcast a dimension of
/// length 4 at -1! Mismatching argument at index 1 had torch.Size([1, 2, 3,
/// 4]); but expected shape should be broadcastable to [1, 3, 4, 2]".
///
/// The port's `retrace` gets `Incompatible` for the `add`, and because the
/// `add` is not a touched node it keeps the stale `[1,2,3,4]` value and
/// returns modified=true: a graph whose `add` no longer matches its operands,
/// where rc4 (and the P1 rule "fail like rc4", commit 46b9368) fails.
const ADD_READS_A_REWIRED_RESCALE: &str = "\
case add_reads_a_rewired_rescale
before
%0 = placeholder x : torch.int8[1, 2, 3, 4] q=none
%1 = placeholder y : torch.int32[1, 2, 3, 4] q=none
%2 = tosa.RESCALE.default(%0, torch.int32, [0.5], 0, 0) : torch.int32[1, 2, 3, 4] q=none
%3 = aten.permute_copy.default(%2, [0, 2, 3, 1]) : torch.int32[1, 3, 4, 2] q=none
%4 = aten.add.Tensor(%2, %1) : torch.int32[1, 2, 3, 4] q=none
output([%3, %4])
after raises=RuntimeError
end
";

#[test]
#[ignore = "BUG: MoveDataMovementOps' retrace keeps a stale value where a rewired user's operands no longer broadcast; rc4's retrace raises RuntimeError"]
fn move_data_movement_fails_where_rc4s_retrace_raises() {
    replay::check_known_with(
        &IrMoveDataMovementOpsToSmallerDtypePass,
        ADD_READS_A_REWIRED_RESCALE,
        &[],
        WITH_FLAG,
    );
}

/// Two sound moves (no other users, no casts). venv: rc4's retrace keeps
/// `exp` and `sqrt` of the int8/int32 value at `torch.float32`, as torch's
/// unary float ops promote integer inputs.
///
/// The port's `retraced_facts` re-derives the dtype of every pointwise user
/// of a moved node by operand promotion, excepting only true division
/// (`TRUE_DIVISION`), so it rewrites both values to the integer input dtype,
/// although a sound move cannot change any value downstream (the rule
/// `layout::shape`'s module docs give for keeping recorded dtypes).
const FLOAT_UNARY_AFTER_A_MOVE: &str = "\
case exp_after_narrowing_rescale
before
%0 = placeholder x : torch.int32[2, 3] q=none
%1 = aten.permute_copy.default(%0, [1, 0]) : torch.int32[3, 2] q=none
%2 = tosa.RESCALE.default(%1, torch.int8, [0.5], 0, 0) : torch.int8[3, 2] q=none
%3 = aten.exp.default(%2) : torch.float32[3, 2] q=none
output([%3])
after modified=true
%0 = placeholder x : torch.int32[2, 3] q=none
%1 = tosa.RESCALE.default(%0, torch.int8, [0.5], 0, 0) : torch.int8[2, 3] q=none
%2 = aten.permute_copy.default(%1, [1, 0]) : torch.int8[3, 2] q=none
%3 = aten.exp.default(%2) : torch.float32[3, 2] q=none
output([%3])
end
case sqrt_after_widening_rescale
before
%0 = placeholder x : torch.int8[2, 3] q=none
%1 = tosa.RESCALE.default(%0, torch.int32, [0.5], 0, 0) : torch.int32[2, 3] q=none
%2 = aten.permute_copy.default(%1, [1, 0]) : torch.int32[3, 2] q=none
%3 = aten.sqrt.default(%2) : torch.float32[3, 2] q=none
output([%3])
after modified=true
%0 = placeholder x : torch.int8[2, 3] q=none
%1 = aten.permute_copy.default(%0, [1, 0]) : torch.int8[3, 2] q=none
%2 = tosa.RESCALE.default(%1, torch.int32, [0.5], 0, 0) : torch.int32[3, 2] q=none
%3 = aten.sqrt.default(%2) : torch.float32[3, 2] q=none
output([%3])
end
";

#[test]
#[ignore = "BUG: MoveDataMovementOps' retrace gives exp/sqrt of an integer value an integer dtype after a sound move; rc4 keeps torch.float32"]
fn move_data_movement_keeps_float_results_of_unary_ops() {
    replay::check_known_with(
        &IrMoveDataMovementOpsToSmallerDtypePass,
        FLOAT_UNARY_AFTER_A_MOVE,
        &[],
        WITH_FLAG,
    );
}
