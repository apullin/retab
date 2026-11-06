//! Ports of rc4's own tests for the four P1 layout passes (APU-135 P1).
//!
//! Each test names the rc4 test it ports. The graphs were confirmed by
//! running those tests under ExecuTorch v1.6.0-rc4 with every call of the
//! four passes dumping the graph it received and the graph it returned (see
//! `rc4_dump` for the format).
//!
//! - rc4's pass unit tests are ported whole.
//! - rc4's pipeline tests (whole-model lowering, TRANSPOSE counts, TOSA
//!   validity) are ported at the pass level: the fixture is the graph the pass
//!   received inside rc4's pipeline and the expectation is the graph it
//!   returned there, so each test checks this pass's share of the upstream
//!   result. What the rest of the pipeline adds (a TRANSPOSE count, numerics)
//!   needs passes RETAB does not have yet and is listed in the doc comment.
//! - Where a pipeline graph is too large to transcribe (MobileNetV2), the
//!   fixture is an excerpt and says so.
//!
//! rc4's `ExportPass` retrace re-derives every node's value after a pass, so
//! interior `dim_order()`s are not part of what these tests compare; the
//! boundary dim orders `NormalizeDelegateIOLayoutPass` owns are checked
//! separately.

use crate::fx_ir::{IrArg, IrPass, NodeId};
use crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass;
use crate::layout::high_rank_singleton_permute::IrRewriteHighRankSingletonPermutePass;
use crate::layout::match_arg_ranks::IrMatchArgRanksPass;
use crate::layout::normalize_delegate_io::IrNormalizeDelegateIOLayoutPass;
use crate::layout::rc4_dump::{assert_graph, check_pass, parse};
use crate::layout::testing::{LayoutGraph, VIEW};

/// Run `pass` again on its own output, which must give it nothing to do.
/// rc4 runs `MatchArgRanksPass` and `FuseViewCopyTransformPass` twice per
/// pipeline (`arm_pass_manager.py:664, 732` and `:725, 744`), and
/// `RewriteHighRankSingletonPermutePass`'s permutes come out at rank 4 or
/// below.
fn assert_rerun_is_a_no_op(pass: &dyn IrPass, g: &mut LayoutGraph) {
    assert!(
        !pass.run(g.graph_mut()),
        "a second {} changed the graph",
        pass.name()
    );
}

fn inputs_of(g: &LayoutGraph, id: NodeId) -> Vec<NodeId> {
    g.graph()
        .node(id)
        .args
        .iter()
        .filter_map(IrArg::as_node)
        .collect()
}

fn only_call(g: &LayoutGraph, target: &str) -> NodeId {
    let ids: Vec<NodeId> = g
        .graph()
        .live_call_function_ids()
        .into_iter()
        .filter(|&id| g.graph().node(id).target == target)
        .collect();
    assert_eq!(ids.len(), 1, "live {target} nodes");
    ids[0]
}

// ---------------------------------------------------------------------------
// MatchArgRanksPass (`backends/arm/_passes/match_arg_ranks_pass.py`)
// ---------------------------------------------------------------------------

fn match_arg_ranks(before: &[&str], after: Option<&[&str]>) -> LayoutGraph {
    let mut g = check_pass(&IrMatchArgRanksPass, before, after);
    assert_rerun_is_a_no_op(&IrMatchArgRanksPass, &mut g);
    g
}

const TOSA_ADD_BEFORE: &[&str] = &[
    "%x = placeholder [1, 270, 480, 3, 2, 2] float32",
    "%y = placeholder [1, 1, 1, 1] float32",
    "%0 = tosa.ADD.default(%x, %y) [1, 270, 480, 3, 2, 2] float32",
    "output [%0]",
];

const TOSA_ADD_AFTER: &[&str] = &[
    "%x = placeholder [1, 270, 480, 3, 2, 2] float32",
    "%y = placeholder [1, 1, 1, 1] float32",
    "%0 = aten.view_copy.default(%y, [1, 1, 1, 1, 1, 1]) [1, 1, 1, 1, 1, 1] float32",
    "%1 = tosa.ADD.default(%x, %0) [1, 270, 480, 3, 2, 2] float32",
    "output [%1]",
];

/// `test_match_arg_ranks_pass.py::test_match_arg_ranks_handles_tosa_binary_ops`:
/// a rank-4 operand of a TOSA `ADD` gets a view to rank 6. Run in both of
/// RETAB's spellings of the TOSA dialect.
#[test]
fn match_arg_ranks_handles_tosa_binary_ops() {
    for spelling in ["tosa.ADD.default", "backend.tosa.ADD.default"] {
        let before: Vec<String> = TOSA_ADD_BEFORE
            .iter()
            .map(|line| line.replace("tosa.ADD.default", spelling))
            .collect();
        let after: Vec<String> = TOSA_ADD_AFTER
            .iter()
            .map(|line| line.replace("tosa.ADD.default", spelling))
            .collect();
        let before: Vec<&str> = before.iter().map(String::as_str).collect();
        let after: Vec<&str> = after.iter().map(String::as_str).collect();
        let g = match_arg_ranks(&before, Some(&after));

        // The upstream assertions, literally.
        let view = only_call(&g, VIEW);
        let add = only_call(&g, spelling);
        assert_eq!(inputs_of(&g, view), [g.id("y")]);
        assert_eq!(g.graph().node(view).args[1].as_int_list(), Some(vec![1; 6]));
        assert_eq!(inputs_of(&g, add), [g.id("x"), view]);
        // `create_node` gets no `from_node`: the view has a value and nothing else.
        assert!(!g.graph().node(view).meta.has_qparam_metadata());
    }
}

const RANK_MATCH_BEFORE_PROPAGATION: &[&str] = &[
    "%b__frozen_param0 = placeholder [240] int16 param",
    "%0 = tosa.RESCALE.default(%b__frozen_param0) [240] int32 iq={} oq={}",
    "%b__frozen_param1 = placeholder [240] int16 param",
    "%1 = tosa.RESCALE.default(%b__frozen_param1) [240] int32 iq={} oq={}",
    "%b__frozen_param2 = placeholder [240] int16 param",
    "%2 = tosa.RESCALE.default(%b__frozen_param2) [240] int32 iq={} oq={}",
    "%b__frozen_param3 = placeholder [240] int16 param",
    "%3 = tosa.RESCALE.default(%b__frozen_param3) [240] int32 iq={} oq={}",
    "%x = placeholder [1, 240] int16",
    "%4 = aten.view_copy.default(%x, [1, 1, 1, 240]) [1, 1, 1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
    "%5 = tosa.RESCALE.default(%4) [1, 1, 1, 240] int32 iq={} oq={}",
    "%6 = aten.mul.Tensor(%5, %0) [1, 1, 1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
    "%7 = tosa.RESCALE.default(%6) [1, 1, 1, 240] int16 iq={} oq={}",
    "%8 = tosa.RESCALE.default(%7) [1, 1, 1, 240] int32 iq={} oq={}",
    "%9 = aten.add.Tensor(%8, %1) [1, 1, 1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
    "%10 = tosa.RESCALE.default(%9) [1, 1, 1, 240] int16 iq={} oq={}",
    "%11 = aten.permute_copy.default(%10, [0, 3, 1, 2]) [1, 240, 1, 1] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
    "%12 = aten.view_copy.default(%11, [1, 240]) [1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
    "%13 = tosa.RESCALE.default(%12) [1, 240] int32 iq={} oq={}",
    "%14 = aten.mul.Tensor(%13, %2) [1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
    "%15 = tosa.RESCALE.default(%14) [1, 240] int16 iq={} oq={}",
    "%16 = tosa.RESCALE.default(%15) [1, 240] int32 iq={} oq={}",
    "%17 = aten.add.Tensor(%16, %3) [1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
    "%18 = tosa.RESCALE.default(%17) [1, 240] int16 iq={} oq={}",
    "output [%18]",
];

/// `test_rank_match_after_propagation.py::test_per_channel_affine_rank4_u55_INT`,
/// at `MatchArgRanksPass`'s first run (`arm_pass_manager.py:664`): the four
/// rank-1 per-channel constants of `y * scale + bias`, around a rank-4 and a
/// rank-2 activation, each get a view directly before their user.
///
/// Not ported: the U55 lowering itself (Vela), which needs the rest of the
/// pipeline.
#[test]
fn rank_match_after_propagation_first_run() {
    match_arg_ranks(
        RANK_MATCH_BEFORE_PROPAGATION,
        Some(&[
            "%b__frozen_param0 = placeholder [240] int16",
            "%0 = tosa.RESCALE.default(%b__frozen_param0) [240] int32 iq={} oq={}",
            "%b__frozen_param1 = placeholder [240] int16",
            "%1 = tosa.RESCALE.default(%b__frozen_param1) [240] int32 iq={} oq={}",
            "%b__frozen_param2 = placeholder [240] int16",
            "%2 = tosa.RESCALE.default(%b__frozen_param2) [240] int32 iq={} oq={}",
            "%b__frozen_param3 = placeholder [240] int16",
            "%3 = tosa.RESCALE.default(%b__frozen_param3) [240] int32 iq={} oq={}",
            "%x = placeholder [1, 240] int16",
            "%4 = aten.view_copy.default(%x, [1, 1, 1, 240]) [1, 1, 1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
            "%5 = tosa.RESCALE.default(%4) [1, 1, 1, 240] int32 iq={} oq={}",
            "%6 = aten.view_copy.default(%0, [1, 1, 1, 240]) [1, 1, 1, 240] int32",
            "%7 = aten.mul.Tensor(%5, %6) [1, 1, 1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
            "%8 = tosa.RESCALE.default(%7) [1, 1, 1, 240] int16 iq={} oq={}",
            "%9 = tosa.RESCALE.default(%8) [1, 1, 1, 240] int32 iq={} oq={}",
            "%10 = aten.view_copy.default(%1, [1, 1, 1, 240]) [1, 1, 1, 240] int32",
            "%11 = aten.add.Tensor(%9, %10) [1, 1, 1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
            "%12 = tosa.RESCALE.default(%11) [1, 1, 1, 240] int16 iq={} oq={}",
            "%13 = aten.permute_copy.default(%12, [0, 3, 1, 2]) [1, 240, 1, 1] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
            "%14 = aten.view_copy.default(%13, [1, 240]) [1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
            "%15 = tosa.RESCALE.default(%14) [1, 240] int32 iq={} oq={}",
            "%16 = aten.view_copy.default(%2, [1, 240]) [1, 240] int32",
            "%17 = aten.mul.Tensor(%15, %16) [1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
            "%18 = tosa.RESCALE.default(%17) [1, 240] int16 iq={} oq={}",
            "%19 = tosa.RESCALE.default(%18) [1, 240] int32 iq={} oq={}",
            "%20 = aten.view_copy.default(%3, [1, 240]) [1, 240] int32",
            "%21 = aten.add.Tensor(%19, %20) [1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
            "%22 = tosa.RESCALE.default(%21) [1, 240] int16 iq={} oq={}",
            "output [%22]",
        ]),
    );
}

/// `test_rank_match_after_propagation.py::test_per_channel_affine_rank4_u55_INT`,
/// at `MatchArgRanksPass`'s second run, after propagation
/// (`arm_pass_manager.py:732`). In rc4 propagation folds the permute away and
/// leaves every binary op's operands rank-matched, so the run the test guards
/// changes nothing.
#[test]
fn rank_match_after_propagation_second_run() {
    match_arg_ranks(
        &[
            "%b__frozen_param0 = placeholder [240] int16 param",
            "%0 = tosa.RESCALE.default(%b__frozen_param0) [240] int32 iq={} oq={}",
            "%b__frozen_param1 = placeholder [240] int16 param",
            "%1 = tosa.RESCALE.default(%b__frozen_param1) [240] int32 iq={} oq={}",
            "%b__frozen_param2 = placeholder [240] int16 param",
            "%2 = tosa.RESCALE.default(%b__frozen_param2) [240] int32 iq={} oq={}",
            "%b__frozen_param3 = placeholder [240] int16 param",
            "%3 = tosa.RESCALE.default(%b__frozen_param3) [240] int32 iq={} oq={}",
            "%x = placeholder [1, 240] int16",
            "%4 = aten.view_copy.default(%x, [1, 1, 1, 240]) [1, 1, 1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
            "%5 = tosa.RESCALE.default(%4) [1, 1, 1, 240] int32 iq={} oq={}",
            "%6 = aten.view_copy.default(%0, [1, 1, 1, 240]) [1, 1, 1, 240] int32",
            "%7 = aten.mul.Tensor(%5, %6) [1, 1, 1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
            "%8 = tosa.RESCALE.default(%7) [1, 1, 1, 240] int16 iq={} oq={}",
            "%9 = tosa.RESCALE.default(%8) [1, 1, 1, 240] int32 iq={} oq={}",
            "%10 = aten.view_copy.default(%1, [1, 1, 1, 240]) [1, 1, 1, 240] int32",
            "%11 = aten.add.Tensor(%9, %10) [1, 1, 1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
            "%12 = tosa.RESCALE.default(%11) [1, 1, 1, 240] int16 iq={} oq={}",
            "%13 = aten.view_copy.default(%12, [1, 240]) [1, 240] int16 iq={0:int16@0.000244140625:0} oq={0:int16@0.000244140625:0}",
            "%14 = tosa.RESCALE.default(%13) [1, 240] int32 iq={} oq={}",
            "%15 = aten.view_copy.default(%2, [1, 240]) [1, 240] int32",
            "%16 = aten.mul.Tensor(%14, %15) [1, 240] int32 iq={0:int32@0.000244140625:0, 1:int32@0.000244140625:0} oq={0:int32@0.00000005960464477539063:0}",
            "%17 = tosa.RESCALE.default(%16) [1, 240] int16 iq={} oq={}",
            "%18 = tosa.RESCALE.default(%17) [1, 240] int32 iq={} oq={}",
            "%19 = aten.view_copy.default(%3, [1, 240]) [1, 240] int32",
            "%20 = aten.add.Tensor(%18, %19) [1, 240] int32 iq={0:int32@0.00000011920928955078125:0, 1:int32@0.00000011920928955078125:0} oq={0:int32@0.00000011920928955078125:0}",
            "%21 = tosa.RESCALE.default(%20) [1, 240] int16 iq={} oq={}",
            "output [%21]",
        ],
        None,
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_groupnorm_channels_last`,
/// at `MatchArgRanksPass` (`:664`): the decomposed group norm's rank-1 `full`
/// constants get views to rank 4 before the `mul`s and the `add` that read
/// them; the `sub`s and the other `mul`s already match. The `full`s' empty
/// qparam maps do not reach the views.
#[test]
fn transpose_counts_groupnorm_channels_last_match_arg_ranks() {
    match_arg_ranks(
        &[
            "%weight = placeholder [4] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 4, 4, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 4, 4, 4] float32",
            "%1 = aten.view_copy.default(%0, [1, 2, 2, 16]) [1, 2, 2, 16] float32",
            "%2 = aten.sum.dim_IntList(%1) [1, 2, 1, 1] float32",
            "%3 = aten.full.default() [1] float32 iq={} oq={}",
            "%4 = aten.mul.Tensor(%2, %3) [1, 2, 1, 1] float32",
            "%5 = aten.sub.Tensor(%1, %4) [1, 2, 2, 16] float32",
            "%6 = aten.full.default() [1] float32 iq={} oq={}",
            "%7 = aten.mul.Tensor(%2, %6) [1, 2, 1, 1] float32",
            "%8 = aten.sub.Tensor(%1, %7) [1, 2, 2, 16] float32",
            "%9 = aten.mul.Tensor(%8, %8) [1, 2, 2, 16] float32",
            "%10 = aten.sum.dim_IntList(%9) [1, 2, 1, 1] float32",
            "%11 = aten.full.default() [1] float32 iq={} oq={}",
            "%12 = aten.mul.Tensor(%10, %11) [1, 2, 1, 1] float32",
            "%13 = aten.full.default() [1] float32 iq={} oq={}",
            "%14 = aten.add.Tensor(%12, %13) [1, 2, 1, 1] float32",
            "%15 = aten.rsqrt.default(%14) [1, 2, 1, 1] float32",
            "%16 = aten.mul.Tensor(%5, %15) [1, 2, 2, 16] float32",
            "%17 = aten.view_copy.default(%weight, [1, 2, 2, 1]) [1, 2, 2, 1] float32",
            "%18 = aten.mul.Tensor(%16, %17) [1, 2, 2, 16] float32",
            "%19 = aten.view_copy.default(%bias, [1, 2, 2, 1]) [1, 2, 2, 1] float32",
            "%20 = aten.add.Tensor(%18, %19) [1, 2, 2, 16] float32",
            "%21 = aten.view_copy.default(%20, [1, 4, 4, 4]) [1, 4, 4, 4] float32",
            "%22 = aten.permute_copy.default(%21, [0, 2, 3, 1]) [1, 4, 4, 4] float32",
            "output [%22]",
        ],
        Some(&[
            "%weight = placeholder [4] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 4, 4, 4] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 4, 4, 4] float32",
            "%1 = aten.view_copy.default(%0, [1, 2, 2, 16]) [1, 2, 2, 16] float32",
            "%2 = aten.sum.dim_IntList(%1) [1, 2, 1, 1] float32",
            "%3 = aten.full.default() [1] float32 iq={} oq={}",
            "%4 = aten.view_copy.default(%3, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%5 = aten.mul.Tensor(%2, %4) [1, 2, 1, 1] float32",
            "%6 = aten.sub.Tensor(%1, %5) [1, 2, 2, 16] float32",
            "%7 = aten.full.default() [1] float32 iq={} oq={}",
            "%8 = aten.view_copy.default(%7, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%9 = aten.mul.Tensor(%2, %8) [1, 2, 1, 1] float32",
            "%10 = aten.sub.Tensor(%1, %9) [1, 2, 2, 16] float32",
            "%11 = aten.mul.Tensor(%10, %10) [1, 2, 2, 16] float32",
            "%12 = aten.sum.dim_IntList(%11) [1, 2, 1, 1] float32",
            "%13 = aten.full.default() [1] float32 iq={} oq={}",
            "%14 = aten.view_copy.default(%13, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%15 = aten.mul.Tensor(%12, %14) [1, 2, 1, 1] float32",
            "%16 = aten.full.default() [1] float32 iq={} oq={}",
            "%17 = aten.view_copy.default(%16, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%18 = aten.add.Tensor(%15, %17) [1, 2, 1, 1] float32",
            "%19 = aten.rsqrt.default(%18) [1, 2, 1, 1] float32",
            "%20 = aten.mul.Tensor(%6, %19) [1, 2, 2, 16] float32",
            "%21 = aten.view_copy.default(%weight, [1, 2, 2, 1]) [1, 2, 2, 1] float32",
            "%22 = aten.mul.Tensor(%20, %21) [1, 2, 2, 16] float32",
            "%23 = aten.view_copy.default(%bias, [1, 2, 2, 1]) [1, 2, 2, 1] float32",
            "%24 = aten.add.Tensor(%22, %23) [1, 2, 2, 16] float32",
            "%25 = aten.view_copy.default(%24, [1, 4, 4, 4]) [1, 4, 4, 4] float32",
            "%26 = aten.permute_copy.default(%25, [0, 2, 3, 1]) [1, 4, 4, 4] float32",
            "output [%26]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_FP[channels_last_inside_partition]`,
/// at `MatchArgRanksPass` (`:664`): the lifted scalar constant, cast and
/// unsqueezed to rank 1, gets a view to rank 4 for the `mul`.
#[test]
fn dim_order_channels_last_inside_partition_match_arg_ranks() {
    match_arg_ranks(
        &[
            "%weight = placeholder [2, 2, 3, 3] float32 param",
            "%bias = placeholder [2] float32 param",
            "%c = placeholder [1] int32 param",
            "%x = placeholder [1, 2, 3, 3] float32",
            "%0 = dim_order_ops._clone_dim_order.default(%x) [1, 2, 3, 3] float32 dim_order=[0, 2, 3, 1]",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 2, 1, 1] float32 dim_order=[0, 2, 3, 1]",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 2, 1, 1] float32",
            "%3 = dim_order_ops._to_dim_order_copy.default(%c) [1] float32",
            "%4 = aten.mul.Tensor(%2, %3) [1, 2, 1, 1] float32",
            "output [%4]",
        ],
        Some(&[
            "%weight = placeholder [2, 2, 3, 3] float32",
            "%bias = placeholder [2] float32",
            "%c = placeholder [1] int32",
            "%x = placeholder [1, 2, 3, 3] float32",
            "%0 = dim_order_ops._clone_dim_order.default(%x) [1, 2, 3, 3] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 2, 1, 1] float32",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 2, 1, 1] float32",
            "%3 = dim_order_ops._to_dim_order_copy.default(%c) [1] float32",
            "%4 = aten.view_copy.default(%3, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%5 = aten.mul.Tensor(%2, %4) [1, 2, 1, 1] float32",
            "output [%5]",
        ]),
    );
}

/// `test_avg_pool2d.py::test_avg_pool2d_tosa_FP[channels_last_adaptive_avg_pool]`,
/// at `MatchArgRanksPass` (`:664`): the mean's decomposition multiplies a
/// rank-4 sum by a rank-1 `full`, which gets a view to rank 4.
#[test]
fn avg_pool2d_channels_last_adaptive_avg_pool_match_arg_ranks() {
    match_arg_ranks(
        &[
            "%x = placeholder [1, 7, 7, 1280] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 1280, 7, 7] float32",
            "%1 = aten.sum.dim_IntList(%0) [1, 1280, 1, 1] float32",
            "%2 = aten.full.default() [1] float32 iq={} oq={}",
            "%3 = aten.mul.Tensor(%1, %2) [1, 1280, 1, 1] float32",
            "%4 = aten.as_strided_copy.default(%3) [1, 1280, 1, 1] float32",
            "output [%4]",
        ],
        Some(&[
            "%x = placeholder [1, 7, 7, 1280] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 1280, 7, 7] float32",
            "%1 = aten.sum.dim_IntList(%0) [1, 1280, 1, 1] float32",
            "%2 = aten.full.default() [1] float32 iq={} oq={}",
            "%3 = aten.view_copy.default(%2, [1, 1, 1, 1]) [1, 1, 1, 1] float32",
            "%4 = aten.mul.Tensor(%1, %3) [1, 1280, 1, 1] float32",
            "%5 = aten.as_strided_copy.default(%4) [1, 1280, 1, 1] float32",
            "output [%5]",
        ]),
    );
}

/// Not an upstream test: rc4's `call` loops over a snapshot of `node.args`,
/// so a lower-rank operand passed twice gets a second view after the first
/// has already replaced both uses. rc4 (run directly on this graph) leaves
/// that view in the graph, dead:
///
/// ```text
/// %0 = aten.view_copy.default(%x, [1, 3]) [1, 3] float32
/// %1 = aten.view_copy.default(%x, [1, 3]) [1, 3] float32
/// %2 = aten.where.self(%c, %0, %0) [2, 3] float32
/// ```
///
/// RETAB never creates it, so the live graphs agree.
#[test]
fn match_arg_ranks_repeated_operand_matches_rc4_live_graph() {
    match_arg_ranks(
        &[
            "%c = placeholder [2, 3] bool",
            "%x = placeholder [3] float32",
            "%0 = aten.where.self(%c, %x, %x) [2, 3] float32",
            "output [%0]",
        ],
        Some(&[
            "%c = placeholder [2, 3] bool",
            "%x = placeholder [3] float32",
            "%0 = aten.view_copy.default(%x, [1, 3]) [1, 3] float32",
            "%1 = aten.where.self(%c, %0, %0) [2, 3] float32",
            "output [%1]",
        ]),
    );
}

/// Not an upstream test: a lower-rank operand shared by two binary ops gets
/// one view per user, each directly before it, and in either operand
/// position (confirmed by running rc4's pass on this graph).
#[test]
fn match_arg_ranks_shared_operand_gets_a_view_per_user() {
    match_arg_ranks(
        &[
            "%a = placeholder [2, 3, 4] float32",
            "%b = placeholder [2, 3, 4] float32",
            "%s = placeholder [4] float32",
            "%0 = aten.mul.Tensor(%a, %s) [2, 3, 4] float32",
            "%1 = aten.add.Tensor(%s, %b) [2, 3, 4] float32",
            "output [%0, %1]",
        ],
        Some(&[
            "%a = placeholder [2, 3, 4] float32",
            "%b = placeholder [2, 3, 4] float32",
            "%s = placeholder [4] float32",
            "%0 = aten.view_copy.default(%s, [1, 1, 4]) [1, 1, 4] float32",
            "%1 = aten.mul.Tensor(%a, %0) [2, 3, 4] float32",
            "%2 = aten.view_copy.default(%s, [1, 1, 4]) [1, 1, 4] float32",
            "%3 = aten.add.Tensor(%2, %b) [2, 3, 4] float32",
            "output [%1, %3]",
        ]),
    );
}

// ---------------------------------------------------------------------------
// FuseViewCopyTransform (`backends/transforms/fuse_view_copy.py`), run by Arm
// as `FuseViewCopyTransformPass` with no extra ops or filters.
// ---------------------------------------------------------------------------

fn fuse_view_copy(before: &[&str], after: Option<&[&str]>) -> LayoutGraph {
    let mut g = check_pass(&IrFuseViewCopyTransformPass, before, after);
    assert_rerun_is_a_no_op(&IrFuseViewCopyTransformPass, &mut g);
    g
}

/// `backends/transforms/test/test_fuse_view_copy.py::TestFuseViewCopyTransform::test_static_view_chain_is_fused`:
/// `x.view(4, 8).relu().view(32).sqrt().view(2, 16)` keeps one view.
#[test]
fn transforms_static_view_chain_is_fused() {
    let g = fuse_view_copy(
        &[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [4, 8]) [4, 8] float32",
            "%1 = aten.relu.default(%0) [4, 8] float32",
            "%2 = aten.view_copy.default(%1, [32]) [32] float32",
            "%3 = aten.sqrt.default(%2) [32] float32",
            "%4 = aten.view_copy.default(%3, [2, 16]) [2, 16] float32",
            "output [%4]",
        ],
        Some(&[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [2, 16]) [2, 16] float32",
            "%1 = aten.relu.default(%0) [2, 16] float32",
            "%2 = aten.sqrt.default(%1) [2, 16] float32",
            "output [%2]",
        ]),
    );
    assert_eq!(g.count(VIEW), 1);
}

const SOFTMAX_BETWEEN_VIEWS: &[&str] = &[
    "%x = placeholder [2, 16] float32",
    "%0 = aten.view_copy.default(%x, [1, 2, 16]) [1, 2, 16] float32",
    "%1 = aten._softmax.default(%0) [1, 2, 16] float32",
    "%2 = aten.view_copy.default(%1, [2, 16]) [2, 16] float32",
    "output [%2]",
];

/// `backends/transforms/test/test_fuse_view_copy.py::TestFuseViewCopyTransform::test_extra_op_is_not_fused_when_filter_rejects`
/// and the Arm configuration of `test_extra_op_is_fused_when_filter_accepts`.
/// Both upstream tests opt `_softmax` in with `extra_fusible_ops`; Arm passes
/// none (`fuse_view_copy_transform_pass.py`), and the RETAB port has no such
/// parameter. Without it, rc4's `FuseViewCopyTransform()` keeps both views
/// (confirmed by running it on this graph), which is what the rejecting
/// filter expects. The accepting filter's single view is not reachable in
/// the Arm configuration.
#[test]
fn transforms_softmax_is_not_fusible_in_the_arm_configuration() {
    let g = fuse_view_copy(SOFTMAX_BETWEEN_VIEWS, None);
    assert_eq!(g.count(VIEW), 2);
}

/// Static-shape counterpart of `backends/transforms/test/test_fuse_view_copy.py::TestFuseViewCopyTransform::test_chain_ending_in_a_later_computed_shape_stays_ordered`.
/// The upstream test needs a view size computed after the first view
/// (`n * 2` from a symbolic `n`); RETAB's view sizes are integer lists, so
/// `_shape_is_available_at` always holds. With `x` exported at a static
/// `[8, 32]`, rc4 fuses the chain into one view, and so does RETAB.
#[test]
fn transforms_later_computed_shape_static_counterpart() {
    fuse_view_copy(
        &[
            "%c = placeholder [] float32",
            "%x = placeholder [8, 32] float32",
            "%0 = aten.view_copy.default(%x, [32, 8]) [32, 8] float32",
            "%1 = aten.relu.default(%0) [32, 8] float32",
            "%2 = aten.view_copy.default(%1, [16, 16]) [16, 16] float32",
            "%3 = aten.add.Tensor(%2, %c) [16, 16] float32",
            "output [%3]",
        ],
        Some(&[
            "%c = placeholder [] float32",
            "%x = placeholder [8, 32] float32",
            "%0 = aten.view_copy.default(%x, [16, 16]) [16, 16] float32",
            "%1 = aten.relu.default(%0) [16, 16] float32",
            "%2 = aten.add.Tensor(%1, %c) [16, 16] float32",
            "output [%2]",
        ]),
    );
}

/// Static-shape counterpart of `backends/transforms/test/test_fuse_view_copy.py::TestFuseViewCopyTransform::test_chain_fuses_up_to_the_first_unavailable_shape`.
/// Upstream, the last view's symbolic size keeps it out of the fusion; with
/// static shapes all three views fuse in rc4 and in RETAB.
#[test]
fn transforms_first_unavailable_shape_static_counterpart() {
    fuse_view_copy(
        &[
            "%c = placeholder [] float32",
            "%x = placeholder [8, 32] float32",
            "%0 = aten.view_copy.default(%x, [32, 8]) [32, 8] float32",
            "%1 = aten.relu.default(%0) [32, 8] float32",
            "%2 = aten.view_copy.default(%1, [32, 4, 2]) [32, 4, 2] float32",
            "%3 = aten.sqrt.default(%2) [32, 4, 2] float32",
            "%4 = aten.view_copy.default(%3, [16, 16]) [16, 16] float32",
            "%5 = aten.add.Tensor(%4, %c) [16, 16] float32",
            "output [%5]",
        ],
        Some(&[
            "%c = placeholder [] float32",
            "%x = placeholder [8, 32] float32",
            "%0 = aten.view_copy.default(%x, [16, 16]) [16, 16] float32",
            "%1 = aten.relu.default(%0) [16, 16] float32",
            "%2 = aten.sqrt.default(%1) [16, 16] float32",
            "%3 = aten.add.Tensor(%2, %c) [16, 16] float32",
            "output [%3]",
        ]),
    );
}

/// `backends/arm/test/passes/test_fuse_view_copy.py::test_fuse_view_copy_transform_tosa_FP[fuse_sequential_views]`:
/// three views in a row become one (`ops_before_pass` 3, `ops_after_pass` 1).
#[test]
fn arm_fuse_sequential_views() {
    let g = fuse_view_copy(
        &[
            "%x = placeholder [2, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [1, 2, 3, 4]) [1, 2, 3, 4] float32",
            "%1 = aten.view_copy.default(%0, [2, 3, 4, 1]) [2, 3, 4, 1] float32",
            "%2 = aten.view_copy.default(%1, [2, 3, 4]) [2, 3, 4] float32",
            "output [%2]",
        ],
        Some(&[
            "%x = placeholder [2, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [2, 3, 4]) [2, 3, 4] float32",
            "output [%0]",
        ]),
    );
    assert_eq!(g.count(VIEW), 1);
}

/// `backends/arm/test/passes/test_fuse_view_copy.py::test_fuse_view_copy_transform_tosa_FP[fuse_sequential_with_noops_views]`:
/// four views separated by a clone, an int32 cast and `abs`, `reciprocal`
/// and `sqrt` become one (4 -> 1). Every op before the last view takes its
/// shape and keeps its dtype.
#[test]
fn arm_fuse_sequential_with_noops_views() {
    let g = fuse_view_copy(
        &[
            "%x = placeholder [2, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [1, 2, 3, 4]) [1, 2, 3, 4] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [1, 2, 3, 4] float32",
            "%2 = aten.view_copy.default(%1, [2, 3, 4, 1]) [2, 3, 4, 1] float32",
            "%3 = dim_order_ops._to_dim_order_copy.default(%2) [2, 3, 4, 1] int32",
            "%4 = aten.view_copy.default(%3, [2, 3, 4]) [2, 3, 4] int32",
            "%5 = aten.abs.default(%4) [2, 3, 4] int32",
            "%6 = aten.reciprocal.default(%5) [2, 3, 4] float32",
            "%7 = aten.sqrt.default(%6) [2, 3, 4] float32",
            "%8 = aten.view_copy.default(%7, [12, 2]) [12, 2] float32",
            "output [%8]",
        ],
        Some(&[
            "%x = placeholder [2, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [12, 2]) [12, 2] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [12, 2] float32",
            "%2 = dim_order_ops._to_dim_order_copy.default(%1) [12, 2] int32",
            "%3 = aten.abs.default(%2) [12, 2] int32",
            "%4 = aten.reciprocal.default(%3) [12, 2] float32",
            "%5 = aten.sqrt.default(%4) [12, 2] float32",
            "output [%5]",
        ]),
    );
    assert_eq!(g.count(VIEW), 1);
}

/// `backends/arm/test/passes/test_fuse_view_copy.py::test_fuse_view_copy_transform_tosa_FP[dont_fuse_branching_views]`:
/// a view with two users starts no chain, and the branch views end at an
/// `add`, so all three views stay (3 -> 3).
#[test]
fn arm_dont_fuse_branching_views() {
    let g = fuse_view_copy(
        &[
            "%x = placeholder [2, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [1, 2, 3, 4]) [1, 2, 3, 4] float32",
            "%1 = aten.abs.default(%0) [1, 2, 3, 4] float32",
            "%2 = aten.view_copy.default(%1, [2, 3, 4, 1]) [2, 3, 4, 1] float32",
            "%3 = aten.ceil.default(%0) [1, 2, 3, 4] float32",
            "%4 = aten.view_copy.default(%3, [2, 3, 4, 1]) [2, 3, 4, 1] float32",
            "%5 = aten.add.Tensor(%2, %4) [2, 3, 4, 1] float32",
            "output [%5]",
        ],
        None,
    );
    assert_eq!(g.count(VIEW), 3);
}

/// `backends/arm/test/passes/test_fuse_view_copy.py::test_canonicalize_fuses_view_introduced_from_permute_tosa_FP`,
/// up to `_AssertAfterInitialFusePass`: the two views fuse and the permute
/// stays (one view, one permute).
///
/// Not ported: the `CanonicalizeViewCopyPermutePass` half (`ops_after_pass`
/// one view, no permute), as that pass is not in RETAB yet.
#[test]
fn arm_fuse_views_before_a_permute() {
    let g = fuse_view_copy(
        &[
            "%x = placeholder [2, 1, 3, 4] float32",
            "%0 = aten.view_copy.default(%x, [1, 2, 1, 3, 4]) [1, 2, 1, 3, 4] float32",
            "%1 = aten.view_copy.default(%0, [2, 1, 3, 4]) [2, 1, 3, 4] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [2, 3, 4, 1] float32",
            "output [%2]",
        ],
        Some(&[
            "%x = placeholder [2, 1, 3, 4] float32",
            "%0 = aten.view_copy.default(%x, [2, 1, 3, 4]) [2, 1, 3, 4] float32",
            "%1 = aten.permute_copy.default(%0, [0, 2, 3, 1]) [2, 3, 4, 1] float32",
            "output [%1]",
        ]),
    );
    g.assert_counts(1, 1);
}

/// Not an upstream test: torch.export keeps a view's `-1`
/// (`x.view(4, 8).relu().view(-1, 16)` exports `view_copy(relu, [-1, 16])`),
/// and rc4 gives the first view that size list unchanged
/// (`fuse_view_copy.py:190-192`). Its retrace still gives the view and the
/// `relu` the value `[2, 16]` (confirmed by running rc4 on this graph).
#[test]
#[ignore = "BUG: fuse_view_copy writes the last view's raw size list, -1 included, into the fused nodes' values"]
fn fused_view_with_an_inferred_dim_keeps_resolved_values() {
    fuse_view_copy(
        &[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [4, 8]) [4, 8] float32",
            "%1 = aten.relu.default(%0) [4, 8] float32",
            "%2 = aten.view_copy.default(%1, [-1, 16]) [2, 16] float32",
            "output [%2]",
        ],
        Some(&[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [-1, 16]) [2, 16] float32",
            "%1 = aten.relu.default(%0) [2, 16] float32",
            "output [%1]",
        ]),
    );
}

/// The companion case, an inferred dim in the first view
/// (`x.view(-1, 8).relu().view(2, 16)`): the first view takes the last
/// view's size, which has none (confirmed against rc4).
#[test]
fn fused_view_replaces_an_inferred_first_size() {
    fuse_view_copy(
        &[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [-1, 8]) [4, 8] float32",
            "%1 = aten.relu.default(%0) [4, 8] float32",
            "%2 = aten.view_copy.default(%1, [2, 16]) [2, 16] float32",
            "output [%2]",
        ],
        Some(&[
            "%x = placeholder [2, 16] float32",
            "%0 = aten.view_copy.default(%x, [2, 16]) [2, 16] float32",
            "%1 = aten.relu.default(%0) [2, 16] float32",
            "output [%1]",
        ]),
    );
}

/// `test_mobilenet_v2_arm.py::test_mv2_tosa_FP_channels_last`, at
/// `FuseViewCopyTransformPass`'s first run (`arm_pass_manager.py:725`), on an
/// excerpt: the classifier head, from the pooled features' `mul` to the
/// output. The three views before the classifier's input permute fuse into
/// one; the view after the classifier stays. The fused constant and the
/// classifier weights stand in as inputs.
#[test]
fn mobilenet_v2_channels_last_classifier_views_fuse() {
    fuse_view_copy(
        &[
            "%sum = placeholder [1, 1280, 1, 1] float32",
            "%scale = placeholder [1, 1, 1, 1] float32",
            "%weight = placeholder [1000, 1, 1, 1280] float32",
            "%bias = placeholder [1000] float32",
            "%0 = aten.mul.Tensor(%sum, %scale) [1, 1280, 1, 1] float32",
            "%1 = aten.view_copy.default(%0, [1, 1280, 1, 1]) [1, 1280, 1, 1] float32",
            "%2 = aten.view_copy.default(%1, [1, 1280]) [1, 1280] float32",
            "%3 = aten.view_copy.default(%2, [1, 1, 1, 1280]) [1, 1, 1, 1280] float32",
            "%4 = aten.permute_copy.default(%3, [0, 3, 1, 2]) [1, 1280, 1, 1] float32",
            "%5 = aten.permute_copy.default(%4, [0, 2, 3, 1]) [1, 1, 1, 1280] float32",
            "%6 = tosa.CONV2D.default(%5, %weight, %bias) [1, 1, 1, 1000] float32",
            "%7 = aten.permute_copy.default(%6, [0, 3, 1, 2]) [1, 1000, 1, 1] float32",
            "%8 = aten.permute_copy.default(%7, [0, 2, 3, 1]) [1, 1, 1, 1000] float32",
            "%9 = aten.view_copy.default(%8, [1, 1000]) [1, 1000] float32",
            "output [%9]",
        ],
        Some(&[
            "%sum = placeholder [1, 1280, 1, 1] float32",
            "%scale = placeholder [1, 1, 1, 1] float32",
            "%weight = placeholder [1000, 1, 1, 1280] float32",
            "%bias = placeholder [1000] float32",
            "%0 = aten.mul.Tensor(%sum, %scale) [1, 1280, 1, 1] float32",
            "%1 = aten.view_copy.default(%0, [1, 1, 1, 1280]) [1, 1, 1, 1280] float32",
            "%2 = aten.permute_copy.default(%1, [0, 3, 1, 2]) [1, 1280, 1, 1] float32",
            "%3 = aten.permute_copy.default(%2, [0, 2, 3, 1]) [1, 1, 1, 1280] float32",
            "%4 = tosa.CONV2D.default(%3, %weight, %bias) [1, 1, 1, 1000] float32",
            "%5 = aten.permute_copy.default(%4, [0, 3, 1, 2]) [1, 1000, 1, 1] float32",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [1, 1, 1, 1000] float32",
            "%7 = aten.view_copy.default(%6, [1, 1000]) [1, 1000] float32",
            "output [%7]",
        ]),
    );
}

// ---------------------------------------------------------------------------
// NormalizeDelegateIOLayoutPass
// (`backends/arm/_passes/normalize_delegate_io_layout_pass.py`). rc4 has no
// unit test for it; these are the rc4 tests whose models have channels-last
// inputs or outputs, at the pass (rc4's first, `arm_pass_manager.py:544`).
// ---------------------------------------------------------------------------

fn output_args(g: &LayoutGraph) -> Vec<NodeId> {
    let output = g.graph().live_output_ids()[0];
    let mut outputs = Vec::new();
    for arg in &g.graph().node(output).args {
        arg.collect_node_ids(&mut outputs);
    }
    outputs
}

fn is_identity(order: Option<&[usize]>) -> bool {
    order.is_none_or(|order| order.iter().enumerate().all(|(index, &axis)| index == axis))
}

fn normalize_delegate_io(before: &[&str], after: Option<&[&str]>) -> LayoutGraph {
    let g = check_pass(&IrNormalizeDelegateIOLayoutPass, before, after);
    // Every output leaves in its exported dim order: either it already was
    // contiguous, or it is now a permute, whose value is a fresh contiguous
    // tensor (`permute_fake_tensor_metadata`, `arm_pass_utils.py:820-830`).
    for output in output_args(&g) {
        let node = g.graph().node(output);
        assert!(
            is_identity(node.meta.val_dim_order()),
            "output {} has dim order {:?}",
            node.name,
            node.meta.val_dim_order()
        );
    }
    g
}

/// `test_dim_order.py::test_dim_order_tosa_FP[channels_last_input]`
/// (`ChannelsLastInput`; `y` passes through outside the delegate): the input
/// takes its NHWC shape behind a `[0, 3, 1, 2]` permute, which both `mul`
/// operands read, and the channels-last result gets a `[0, 2, 3, 1]` permute.
#[test]
fn dim_order_tosa_fp_channels_last_input() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.mul.Tensor(%x, %x) [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "output [%0]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] float32",
            "%1 = aten.mul.Tensor(%0, %0) [1, 2, 3, 4] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 3, 4, 2] float32",
            "output [%2]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_FP[channels_first_output]`
/// (`ChannelsFirstOutput`): both users of the channels-last input read the
/// permute; the contiguous result needs none.
#[test]
fn dim_order_tosa_fp_channels_first_output() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = dim_order_ops._clone_dim_order.default(%x) [1, 2, 3, 4] float32",
            "%1 = aten.mul.Tensor(%0, %x) [1, 2, 3, 4] float32",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [1, 2, 3, 4] float32",
            "%2 = aten.mul.Tensor(%1, %0) [1, 2, 3, 4] float32",
            "output [%2]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_FP[channels_last_output]`
/// (`ChannelsLastOutput`): a contiguous input, and an output cloned to
/// channels-last, which gets the permute into that order.
#[test]
fn dim_order_tosa_fp_channels_last_output() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 2, 2] float32",
            "%0 = aten.mul.Tensor(%x, %x) [1, 2, 2, 2] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [1, 2, 2, 2] float32 dim_order=[0, 2, 3, 1]",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 2, 2, 2] float32",
            "%0 = aten.mul.Tensor(%x, %x) [1, 2, 2, 2] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [1, 2, 2, 2] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 2, 2, 2] float32",
            "output [%2]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_FP[channels_last_inside_partition]`
/// (`ChannelsLastInsidePartition`): dim orders change only inside the
/// partition, so the boundaries are left alone.
#[test]
fn dim_order_tosa_fp_channels_last_inside_partition() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [2, 2, 3, 3] float32 param",
            "%bias = placeholder [2] float32 param",
            "%c = placeholder [] int64 param",
            "%x = placeholder [1, 2, 3, 3] float32",
            "%0 = dim_order_ops._clone_dim_order.default(%x) [1, 2, 3, 3] float32 dim_order=[0, 2, 3, 1]",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 2, 1, 1] float32 dim_order=[0, 2, 3, 1]",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 2, 1, 1] float32",
            "%3 = dim_order_ops._to_dim_order_copy.default(%c) [] float32",
            "%4 = aten.mul.Tensor(%2, %3) [1, 2, 1, 1] float32",
            "output [%4]",
        ],
        None,
    );
}

/// `test_dim_order.py::test_dim_order_tosa_INT[channels_last_input]`: the
/// quantized form; the int8 input and output permutes keep the int8 dtype.
#[test]
fn dim_order_tosa_int_channels_last_input() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] int8 dim_order=[0, 2, 3, 1]",
            "%0 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%1 = aten.mul.Tensor(%0, %0) [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%2 = quantized_decomposed.quantize_per_tensor.default(%1) [1, 2, 3, 4] int8 dim_order=[0, 2, 3, 1]",
            "output [%2]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] int8",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] int8",
            "%1 = quantized_decomposed.dequantize_per_tensor.default(%0) [1, 2, 3, 4] float32",
            "%2 = aten.mul.Tensor(%1, %1) [1, 2, 3, 4] float32",
            "%3 = quantized_decomposed.quantize_per_tensor.default(%2) [1, 2, 3, 4] int8",
            "%4 = aten.permute_copy.default(%3, [0, 2, 3, 1]) [1, 3, 4, 2] int8",
            "output [%4]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_INT[channels_first_output]`: both
/// dequantizes of the input read the one permute.
#[test]
fn dim_order_tosa_int_channels_first_output() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] int8 dim_order=[0, 2, 3, 1]",
            "%0 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%1 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 2, 3, 4] float32",
            "%3 = quantized_decomposed.quantize_per_tensor.default(%2) [1, 2, 3, 4] int8",
            "%4 = quantized_decomposed.dequantize_per_tensor.default(%3) [1, 2, 3, 4] float32",
            "%5 = aten.mul.Tensor(%4, %0) [1, 2, 3, 4] float32",
            "%6 = quantized_decomposed.quantize_per_tensor.default(%5) [1, 2, 3, 4] int8",
            "output [%6]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] int8",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] int8",
            "%1 = quantized_decomposed.dequantize_per_tensor.default(%0) [1, 2, 3, 4] float32",
            "%2 = quantized_decomposed.dequantize_per_tensor.default(%0) [1, 2, 3, 4] float32",
            "%3 = dim_order_ops._clone_dim_order.default(%2) [1, 2, 3, 4] float32",
            "%4 = quantized_decomposed.quantize_per_tensor.default(%3) [1, 2, 3, 4] int8",
            "%5 = quantized_decomposed.dequantize_per_tensor.default(%4) [1, 2, 3, 4] float32",
            "%6 = aten.mul.Tensor(%5, %1) [1, 2, 3, 4] float32",
            "%7 = quantized_decomposed.quantize_per_tensor.default(%6) [1, 2, 3, 4] int8",
            "output [%7]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_INT[channels_last_output]`.
#[test]
fn dim_order_tosa_int_channels_last_output() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 2, 2] int8",
            "%0 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 2, 2] float32",
            "%1 = aten.mul.Tensor(%0, %0) [1, 2, 2, 2] float32",
            "%2 = quantized_decomposed.quantize_per_tensor.default(%1) [1, 2, 2, 2] int8",
            "%3 = quantized_decomposed.dequantize_per_tensor.default(%2) [1, 2, 2, 2] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [1, 2, 2, 2] float32 dim_order=[0, 2, 3, 1]",
            "%5 = quantized_decomposed.quantize_per_tensor.default(%4) [1, 2, 2, 2] int8 dim_order=[0, 2, 3, 1]",
            "output [%5]",
        ],
        Some(&[
            "%x = placeholder [1, 2, 2, 2] int8",
            "%0 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 2, 2] float32",
            "%1 = aten.mul.Tensor(%0, %0) [1, 2, 2, 2] float32",
            "%2 = quantized_decomposed.quantize_per_tensor.default(%1) [1, 2, 2, 2] int8",
            "%3 = quantized_decomposed.dequantize_per_tensor.default(%2) [1, 2, 2, 2] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [1, 2, 2, 2] float32",
            "%5 = quantized_decomposed.quantize_per_tensor.default(%4) [1, 2, 2, 2] int8",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [1, 2, 2, 2] int8",
            "output [%6]",
        ]),
    );
}

/// `test_dim_order.py::test_dim_order_tosa_INT[channels_last_inside_partition]`:
/// the quantized form, with its parameters, is left alone too.
#[test]
fn dim_order_tosa_int_channels_last_inside_partition() {
    normalize_delegate_io(
        &[
            "%b__scale_0 = placeholder [2] float32 param",
            "%b__zero_point_0 = placeholder [2] int64 param",
            "%b__frozen_param0 = placeholder [2, 2, 3, 3] int8 param",
            "%b__scale_1 = placeholder [2] float32 param",
            "%b__zero_point_1 = placeholder [2] int32 param",
            "%b__frozen_param1 = placeholder [2] int32 param",
            "%b__frozen_param2 = placeholder [1, 1, 1, 1] int8 param",
            "%x = placeholder [1, 2, 3, 3] int8",
            "%0 = quantized_decomposed.dequantize_per_tensor.default(%x) [1, 2, 3, 3] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [1, 2, 3, 3] float32 dim_order=[0, 2, 3, 1]",
            "%2 = quantized_decomposed.quantize_per_tensor.default(%1) [1, 2, 3, 3] int8 dim_order=[0, 2, 3, 1]",
            "%3 = quantized_decomposed.dequantize_per_tensor.default(%2) [1, 2, 3, 3] float32 dim_order=[0, 2, 3, 1]",
            "%4 = quantized_decomposed.dequantize_per_channel.default(%b__frozen_param0, %b__scale_0, %b__zero_point_0) [2, 2, 3, 3] float32",
            "%5 = quantized_decomposed.dequantize_per_channel.default(%b__frozen_param1, %b__scale_1, %b__zero_point_1) [2] float32",
            "%6 = aten.convolution.default(%3, %4, %5) [1, 2, 1, 1] float32 dim_order=[0, 2, 3, 1]",
            "%7 = quantized_decomposed.quantize_per_tensor.default(%6) [1, 2, 1, 1] int8 dim_order=[0, 2, 3, 1]",
            "%8 = quantized_decomposed.dequantize_per_tensor.default(%7) [1, 2, 1, 1] float32 dim_order=[0, 2, 3, 1]",
            "%9 = dim_order_ops._clone_dim_order.default(%8) [1, 2, 1, 1] float32",
            "%10 = quantized_decomposed.quantize_per_tensor.default(%9) [1, 2, 1, 1] int8",
            "%11 = quantized_decomposed.dequantize_per_tensor.default(%10) [1, 2, 1, 1] float32",
            "%12 = quantized_decomposed.dequantize_per_tensor.default(%b__frozen_param2) [1, 1, 1, 1] float32",
            "%13 = aten.mul.Tensor(%11, %12) [1, 2, 1, 1] float32",
            "%14 = quantized_decomposed.quantize_per_tensor.default(%13) [1, 2, 1, 1] int8",
            "output [%14]",
        ],
        None,
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[conv2d_rank4_channels_last]`:
/// a permute on each side of the convolution. Not checked: rc4's TRANSPOSE
/// count of 0, which needs `RewriteConvPass` and the permute propagation.
#[test]
fn transpose_counts_conv2d_rank4_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4, 2, 3, 3] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 2, 8, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.convolution.default(%x, %weight, %bias) [1, 4, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [4, 2, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 8, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 8, 8] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 6, 6] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 6, 6, 4] float32",
            "output [%2]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[conv3d_rank4_channels_last]`:
/// the input's NHWC shape equals its logical shape, but it still gets the
/// permute. The conv3d output is contiguous. Not checked: rc4's TRANSPOSE
/// count of 2.
#[test]
fn transpose_counts_conv3d_rank4_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4, 2, 3, 3, 3] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [2, 6, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.unsqueeze_copy.default(%x) [1, 2, 6, 6, 6] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%2 = aten.squeeze_copy.dims(%1) [4, 4, 4, 4] float32",
            "output [%2]",
        ],
        Some(&[
            "%weight = placeholder [4, 2, 3, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [2, 6, 6, 6] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [2, 6, 6, 6] float32",
            "%1 = aten.unsqueeze_copy.default(%0) [1, 2, 6, 6, 6] float32",
            "%2 = aten.convolution.default(%1, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%3 = aten.squeeze_copy.dims(%2) [4, 4, 4, 4] float32",
            "output [%3]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[conv3d_rank5_channels_last]`
/// (xfailed upstream for numerics, not layout): a `channels_last_3d` input
/// takes its NDHWC shape behind a `[0, 4, 1, 2, 3]` permute. Not checked:
/// rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_conv3d_rank5_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4, 2, 3, 3, 3] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 2, 6, 6, 6] float32 dim_order=[0, 2, 3, 4, 1]",
            "%0 = aten.convolution.default(%x, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [4, 2, 3, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 6, 6, 6, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 4, 1, 2, 3]) [1, 2, 6, 6, 6] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "output [%1]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[linear_rank4_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_linear_rank4_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4, 8] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 2, 2, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.linear.default(%x, %weight, %bias) [1, 2, 2, 4] float32",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [4, 8] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 2, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 2, 8] float32",
            "%1 = aten.linear.default(%0, %weight, %bias) [1, 2, 2, 4] float32",
            "output [%1]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[matmul_rank4_channels_last]`:
/// each input's permute goes directly after that input, so it sits between
/// the two placeholders. Not checked: rc4's TRANSPOSE count of 2.
#[test]
fn transpose_counts_matmul_rank4_channels_last() {
    normalize_delegate_io(
        &[
            "%a = placeholder [2, 2, 2, 3] float32 dim_order=[0, 2, 3, 1]",
            "%b = placeholder [2, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.expand_copy.default(%a) [2, 2, 2, 3] float32",
            "%1 = dim_order_ops._clone_dim_order.default(%0) [2, 2, 2, 3] float32",
            "%2 = aten.view_copy.default(%1, [4, 2, 3]) [4, 2, 3] float32",
            "%3 = aten.expand_copy.default(%b) [2, 2, 3, 4] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [2, 2, 3, 4] float32",
            "%5 = aten.view_copy.default(%4, [4, 3, 4]) [4, 3, 4] float32",
            "%6 = aten.bmm.default(%2, %5) [4, 2, 4] float32",
            "%7 = aten.view_copy.default(%6, [2, 2, 2, 4]) [2, 2, 2, 4] float32",
            "output [%7]",
        ],
        Some(&[
            "%a = placeholder [2, 2, 3, 2] float32",
            "%0 = aten.permute_copy.default(%a, [0, 3, 1, 2]) [2, 2, 2, 3] float32",
            "%b = placeholder [2, 3, 4, 2] float32",
            "%1 = aten.permute_copy.default(%b, [0, 3, 1, 2]) [2, 2, 3, 4] float32",
            "%2 = aten.expand_copy.default(%0) [2, 2, 2, 3] float32",
            "%3 = dim_order_ops._clone_dim_order.default(%2) [2, 2, 2, 3] float32",
            "%4 = aten.view_copy.default(%3, [4, 2, 3]) [4, 2, 3] float32",
            "%5 = aten.expand_copy.default(%1) [2, 2, 3, 4] float32",
            "%6 = dim_order_ops._clone_dim_order.default(%5) [2, 2, 3, 4] float32",
            "%7 = aten.view_copy.default(%6, [4, 3, 4]) [4, 3, 4] float32",
            "%8 = aten.bmm.default(%4, %7) [4, 2, 4] float32",
            "%9 = aten.view_copy.default(%8, [2, 2, 2, 4]) [2, 2, 2, 4] float32",
            "output [%9]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[pixel_shuffle_channels_last]`:
/// the rank-6 permute of the decomposition is interior and stays; the
/// boundaries get one permute each. Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_pixel_shuffle_channels_last() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 8, 2, 2] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.view_copy.default(%x, [1, 2, 2, 2, 2, 2]) [1, 2, 2, 2, 2, 2] float32",
            "%1 = aten.permute_copy.default(%0, [0, 1, 4, 2, 5, 3]) [1, 2, 2, 2, 2, 2] float32",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 2, 2, 2, 2, 2] float32",
            "%3 = aten.view_copy.default(%2, [1, 2, 4, 4]) [1, 2, 4, 4] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [1, 2, 4, 4] float32 dim_order=[0, 2, 3, 1]",
            "output [%4]",
        ],
        Some(&[
            "%x = placeholder [1, 2, 2, 8] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 8, 2, 2] float32",
            "%1 = aten.view_copy.default(%0, [1, 2, 2, 2, 2, 2]) [1, 2, 2, 2, 2, 2] float32",
            "%2 = aten.permute_copy.default(%1, [0, 1, 4, 2, 5, 3]) [1, 2, 2, 2, 2, 2] float32",
            "%3 = dim_order_ops._clone_dim_order.default(%2) [1, 2, 2, 2, 2, 2] float32",
            "%4 = aten.view_copy.default(%3, [1, 2, 4, 4]) [1, 2, 4, 4] float32",
            "%5 = dim_order_ops._clone_dim_order.default(%4) [1, 2, 4, 4] float32",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [1, 4, 4, 2] float32",
            "output [%6]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[grouped_conv_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 0.
#[test]
fn transpose_counts_grouped_conv_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4, 2, 3, 3] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 4, 8, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.convolution.default(%x, %weight, %bias) [1, 4, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [4, 2, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 8, 8, 4] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 4, 8, 8] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 6, 6] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 6, 6, 4] float32",
            "output [%2]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[transpose_conv_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 0.
#[test]
fn transpose_counts_transpose_conv_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [2, 4, 3, 3] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 2, 8, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.convolution.default(%x, %weight, %bias) [1, 4, 10, 10] float32 dim_order=[0, 2, 3, 1]",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [2, 4, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 8, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 8, 8] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 10, 10] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 10, 10, 4] float32",
            "output [%2]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[transposes_channels_last]`:
/// the input's only user is itself a permute, which now reads the boundary
/// permute. Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_transposes_channels_last() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.permute_copy.default(%x, [0, 1, 3, 2]) [1, 2, 4, 3] float32",
            "%1 = aten.permute_copy.default(%0, [0, 3, 1, 2]) [1, 3, 2, 4] float32",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] float32",
            "%1 = aten.permute_copy.default(%0, [0, 1, 3, 2]) [1, 2, 4, 3] float32",
            "%2 = aten.permute_copy.default(%1, [0, 3, 1, 2]) [1, 3, 2, 4] float32",
            "output [%2]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[maxpool2d_dilation_channels_last]`:
/// the output is a `getitem` of the pool's tuple, and its permute goes
/// directly after it. Not checked: rc4's TRANSPOSE count of 3.
#[test]
fn transpose_counts_maxpool2d_dilation_channels_last() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 8, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.max_pool2d_with_indices.default(%x) -",
            "%1 = getitem(%0) [1, 2, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 8, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 8, 8] float32",
            "%1 = aten.max_pool2d_with_indices.default(%0) -",
            "%2 = getitem(%1) [1, 2, 6, 6] float32",
            "%3 = aten.permute_copy.default(%2, [0, 2, 3, 1]) [1, 6, 6, 2] float32",
            "output [%3]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[cumsum_rank4_dim3_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_cumsum_rank4_dim3_channels_last() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.cumsum.default(%x) [1, 2, 3, 4] float32",
            "output [%0]",
        ],
        Some(&[
            "%x = placeholder [1, 3, 4, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] float32",
            "%1 = aten.cumsum.default(%0) [1, 2, 3, 4] float32",
            "output [%1]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_groupnorm_channels_last`:
/// torch exports this group norm's output channels-last, the case the
/// upstream test expects 2 TRANSPOSEs for (not checked); here, a permute on
/// each side.
#[test]
fn transpose_counts_groupnorm_channels_last() {
    normalize_delegate_io(
        &[
            "%weight = placeholder [4] float32 param",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 4, 4, 4] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.native_group_norm.default(%x, %weight, %bias) -",
            "%1 = getitem(%0) [1, 4, 4, 4] float32 dim_order=[0, 2, 3, 1]",
            "output [%1]",
        ],
        Some(&[
            "%weight = placeholder [4] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 4, 4, 4] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 4, 4, 4] float32",
            "%1 = aten.native_group_norm.default(%0, %weight, %bias) -",
            "%2 = getitem(%1) [1, 4, 4, 4] float32",
            "%3 = aten.permute_copy.default(%2, [0, 2, 3, 1]) [1, 4, 4, 4] float32",
            "output [%3]",
        ]),
    );
}

/// `test_pixel_shuffling.py::test_pixel_unshuffle_tosa_FP[rand_4d_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn pixel_unshuffle_rand_4d_channels_last() {
    normalize_delegate_io(
        &[
            "%inputs = placeholder [1, 12, 270, 480] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.view_copy.default(%inputs, [1, 12, 135, 2, 240, 2]) [1, 12, 135, 2, 240, 2] float32",
            "%1 = aten.permute_copy.default(%0, [0, 1, 3, 5, 2, 4]) [1, 12, 2, 2, 135, 240] float32",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 12, 2, 2, 135, 240] float32",
            "%3 = aten.view_copy.default(%2, [1, 48, 135, 240]) [1, 48, 135, 240] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [1, 48, 135, 240] float32 dim_order=[0, 2, 3, 1]",
            "output [%4]",
        ],
        Some(&[
            "%inputs = placeholder [1, 270, 480, 12] float32",
            "%0 = aten.permute_copy.default(%inputs, [0, 3, 1, 2]) [1, 12, 270, 480] float32",
            "%1 = aten.view_copy.default(%0, [1, 12, 135, 2, 240, 2]) [1, 12, 135, 2, 240, 2] float32",
            "%2 = aten.permute_copy.default(%1, [0, 1, 3, 5, 2, 4]) [1, 12, 2, 2, 135, 240] float32",
            "%3 = dim_order_ops._clone_dim_order.default(%2) [1, 12, 2, 2, 135, 240] float32",
            "%4 = aten.view_copy.default(%3, [1, 48, 135, 240]) [1, 48, 135, 240] float32",
            "%5 = dim_order_ops._clone_dim_order.default(%4) [1, 48, 135, 240] float32",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [1, 135, 240, 48] float32",
            "output [%6]",
        ]),
    );
}

/// `test_pixel_shuffling.py::test_pixel_shuffle_tosa_FP[rand_4d_channels_last]`.
/// Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn pixel_shuffle_rand_4d_channels_last() {
    normalize_delegate_io(
        &[
            "%inputs = placeholder [1, 12, 270, 480] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.view_copy.default(%inputs, [1, 3, 2, 2, 270, 480]) [1, 3, 2, 2, 270, 480] float32",
            "%1 = aten.permute_copy.default(%0, [0, 1, 4, 2, 5, 3]) [1, 3, 270, 2, 480, 2] float32",
            "%2 = dim_order_ops._clone_dim_order.default(%1) [1, 3, 270, 2, 480, 2] float32",
            "%3 = aten.view_copy.default(%2, [1, 3, 540, 960]) [1, 3, 540, 960] float32",
            "%4 = dim_order_ops._clone_dim_order.default(%3) [1, 3, 540, 960] float32 dim_order=[0, 2, 3, 1]",
            "output [%4]",
        ],
        Some(&[
            "%inputs = placeholder [1, 270, 480, 12] float32",
            "%0 = aten.permute_copy.default(%inputs, [0, 3, 1, 2]) [1, 12, 270, 480] float32",
            "%1 = aten.view_copy.default(%0, [1, 3, 2, 2, 270, 480]) [1, 3, 2, 2, 270, 480] float32",
            "%2 = aten.permute_copy.default(%1, [0, 1, 4, 2, 5, 3]) [1, 3, 270, 2, 480, 2] float32",
            "%3 = dim_order_ops._clone_dim_order.default(%2) [1, 3, 270, 2, 480, 2] float32",
            "%4 = aten.view_copy.default(%3, [1, 3, 540, 960]) [1, 3, 540, 960] float32",
            "%5 = dim_order_ops._clone_dim_order.default(%4) [1, 3, 540, 960] float32",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [1, 540, 960, 3] float32",
            "output [%6]",
        ]),
    );
}

/// `test_avg_pool2d.py::test_avg_pool2d_tosa_FP[channels_last_adaptive_avg_pool]`:
/// the input permute only; torch reports the `[1, 1280, 1, 1]` result's dim
/// order as the identity, so the output stays.
#[test]
fn avg_pool2d_channels_last_adaptive_avg_pool() {
    normalize_delegate_io(
        &[
            "%x = placeholder [1, 1280, 7, 7] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.mean.dim(%x) [1, 1280, 1, 1] float32",
            "%1 = aten.as_strided_copy.default(%0) [1, 1280, 1, 1] float32",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 7, 7, 1280] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 1280, 7, 7] float32",
            "%1 = aten.mean.dim(%0) [1, 1280, 1, 1] float32",
            "%2 = aten.as_strided_copy.default(%1) [1, 1280, 1, 1] float32",
            "output [%2]",
        ]),
    );
}

/// `test_mobilenet_v2_arm.py::test_mv2_tosa_FP_channels_last`, on an excerpt:
/// the 262 parameters (three kept) and the input, the first convolution, and
/// the classifier head. The `hardtanh` stands for the network between the
/// first convolution and the last one's activation. The input gets its
/// permute; the rank-2 output needs none.
#[test]
fn mobilenet_v2_channels_last_boundaries() {
    normalize_delegate_io(
        &[
            "%p_features_0_0_weight = placeholder [32, 3, 3, 3] float32 param",
            "%p_classifier_1_weight = placeholder [1000, 1280] float32 param",
            "%p_classifier_1_bias = placeholder [1000] float32 param",
            "%x = placeholder [1, 3, 224, 224] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.convolution.default(%x, %p_features_0_0_weight) [1, 32, 112, 112] float32 dim_order=[0, 2, 3, 1]",
            "%1 = aten.hardtanh.default(%0) [1, 1280, 7, 7] float32 dim_order=[0, 2, 3, 1]",
            "%2 = aten.mean.dim(%1) [1, 1280, 1, 1] float32",
            "%3 = aten.as_strided_copy.default(%2) [1, 1280, 1, 1] float32",
            "%4 = aten.view_copy.default(%3, [1, 1280]) [1, 1280] float32",
            "%5 = aten.linear.default(%4, %p_classifier_1_weight, %p_classifier_1_bias) [1, 1000] float32",
            "output [%5]",
        ],
        Some(&[
            "%p_features_0_0_weight = placeholder [32, 3, 3, 3] float32",
            "%p_classifier_1_weight = placeholder [1000, 1280] float32",
            "%p_classifier_1_bias = placeholder [1000] float32",
            "%x = placeholder [1, 224, 224, 3] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 3, 224, 224] float32",
            "%1 = aten.convolution.default(%0, %p_features_0_0_weight) [1, 32, 112, 112] float32",
            "%2 = aten.hardtanh.default(%1) [1, 1280, 7, 7] float32",
            "%3 = aten.mean.dim(%2) [1, 1280, 1, 1] float32",
            "%4 = aten.as_strided_copy.default(%3) [1, 1280, 1, 1] float32",
            "%5 = aten.view_copy.default(%4, [1, 1280]) [1, 1280] float32",
            "%6 = aten.linear.default(%5, %p_classifier_1_weight, %p_classifier_1_bias) [1, 1000] float32",
            "output [%6]",
        ]),
    );
}

/// Not an upstream test: the `conv2d_rank4_channels_last` model converted to
/// channels-last whole (`module.to(memory_format=torch.channels_last)`), run
/// through rc4's TOSA FP pipeline. The weight is channels-last too, and rc4
/// leaves it alone (`is_param_node`): only runtime inputs are rewritten.
#[test]
fn channels_last_parameters_are_left_alone() {
    let g = normalize_delegate_io(
        &[
            "%weight = placeholder [4, 2, 3, 3] float32 param dim_order=[0, 2, 3, 1]",
            "%bias = placeholder [4] float32 param",
            "%x = placeholder [1, 2, 8, 8] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.convolution.default(%x, %weight, %bias) [1, 4, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "output [%0]",
        ],
        Some(&[
            "%weight = placeholder [4, 2, 3, 3] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 8, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 8, 8] float32",
            "%1 = aten.convolution.default(%0, %weight, %bias) [1, 4, 6, 6] float32",
            "%2 = aten.permute_copy.default(%1, [0, 2, 3, 1]) [1, 6, 6, 4] float32",
            "output [%2]",
        ]),
    );
    let weight = g.graph().node(g.id("weight"));
    assert_eq!(weight.meta.val_dim_order(), Some(&[0, 2, 3, 1][..]));
}

/// Where RETAB and rc4 knowingly differ. rc4 gives a rewritten input
/// `input_fake.reshape(boundary_shape)` (`:57`), which is a view, keeping a
/// non-identity `dim_order()`, whenever the strides allow one. In
/// `transpose_counts_linear_rank4_channels_last` rc4's `x` leaves the pass as
/// `[1, 2, 8, 2]` with dim order `[0, 2, 3, 1]`; in
/// `transpose_counts_conv3d_rank4_channels_last`,
/// `transpose_counts_groupnorm_channels_last` and the matmul's `a` too. rc4
/// reads no input's dim order after this pass (only
/// `PrepareGatherIndicesPass` compares dim orders, of an index cast), so the
/// leftover order does nothing there. RETAB writes the identity, which also
/// makes a second run a no-op; rc4's would rewrite those inputs again.
#[test]
fn normalized_inputs_take_the_identity_dim_order() {
    let mut g = parse(&[
        "%weight = placeholder [4, 8] float32 param",
        "%bias = placeholder [4] float32 param",
        "%x = placeholder [1, 2, 2, 8] float32 dim_order=[0, 2, 3, 1]",
        "%0 = aten.linear.default(%x, %weight, %bias) [1, 2, 2, 4] float32",
        "output [%0]",
    ]);
    assert!(IrNormalizeDelegateIOLayoutPass.run(g.graph_mut()));
    let x = g.graph().node(g.id("x"));
    assert_eq!(x.meta.val_shape(), Some(&[1, 2, 8, 2][..]));
    assert_eq!(x.meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
    assert!(!IrNormalizeDelegateIOLayoutPass.run(g.graph_mut()));
    assert_graph(
        &g,
        &[
            "%weight = placeholder [4, 8] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 2, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 2, 8] float32",
            "%1 = aten.linear.default(%0, %weight, %bias) [1, 2, 2, 4] float32",
            "output [%1]",
        ],
        "the graph after a second run",
    );
}

// ---------------------------------------------------------------------------
// RewriteHighRankSingletonPermutePass
// (`backends/arm/_passes/rewrite_high_rank_singleton_permute_pass.py`). rc4
// has no unit test for it; these are the rc4 tests whose graphs reach it with
// a permute of rank above 4 (`arm_pass_manager.py:733`).
// ---------------------------------------------------------------------------

fn high_rank(before: &[&str], after: Option<&[&str]>) -> LayoutGraph {
    let mut g = check_pass(&IrRewriteHighRankSingletonPermutePass, before, after);
    assert_rerun_is_a_no_op(&IrRewriteHighRankSingletonPermutePass, &mut g);
    g
}

/// What `test_reshape_permute_5d_regression.py` reaches the pass with:
/// `ReshapePermuteVariantModel`'s conv and ReLU are a CONV2D and its RESCALE,
/// and propagation has folded each variant's reshapes and permutes into one
/// view (`%3`) and one permute (`%4`).
const RESHAPE_PERMUTE_PREFIX: &[&str] = &[
    "%weight = placeholder [64, 3, 3, 3] int8",
    "%bias = placeholder [64] int32",
    "%x = placeholder [1, 3, 224, 224] int8",
    "%0 = aten.permute_copy.default(%x, [0, 2, 3, 1]) [1, 224, 224, 3] int8 iq={} oq={}",
    "%1 = tosa.CONV2D.default(%0, %weight, %bias) [1, 112, 112, 64] int32 oq={q}",
    "%2 = tosa.RESCALE.default(%1) [1, 112, 112, 64] int8 iq={} oq={}",
];

/// One `test_reshape_permute_5d_regression.py` variant: `{q}` in a line is
/// the variant's output qparam map. The three new nodes carry the permute's
/// qparams, as `super().call_operator(..., meta)` gives each of them the
/// permute's meta.
///
/// Not checked: the upstream test's own assertion, that every TOSA TRANSPOSE
/// in the serialized graph has an output shape matching its permutation,
/// which needs serialization.
fn reshape_permute_variant(scale: &str, view: &str, permute: &str, rewritten: [&str; 3]) {
    let q = format!("{{0:int8@{scale}:-128}}");
    let fill = |line: &str| line.replace("{q}", &q);
    let mut before: Vec<String> = RESHAPE_PERMUTE_PREFIX.iter().map(|l| fill(l)).collect();
    before.push(fill(view));
    let mut after = before.clone();
    before.extend([fill(permute), "output [%4]".to_string()]);
    after.extend(rewritten.iter().map(|l| fill(l)));
    after.push("output [%6]".to_string());
    let before: Vec<&str> = before.iter().map(String::as_str).collect();
    let after: Vec<&str> = after.iter().map(String::as_str).collect();
    high_rank(&before, Some(&after));
}

/// `test_reshape_permute_5d_regression.py::test_reshape_permute_chains_tosa_INT_transpose_shapes_valid[reshape_permute_5d]`:
/// two singletons leave a rank-3 permute.
#[test]
fn reshape_permute_5d() {
    reshape_permute_variant(
        "0.009707257151603699",
        "%3 = aten.view_copy.default(%2, [1, 1, 49, 64, 256]) [1, 1, 49, 64, 256] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 1, 3, 2, 4]) [1, 1, 64, 49, 256] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [49, 64, 256]) [49, 64, 256] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [1, 0, 2]) [64, 49, 256] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 64, 49, 256]) [1, 1, 64, 49, 256] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_unsqueeze_4d]`: the model's rank-4 permute and
/// unsqueeze reach the pass as the same rank-5 view and permute as
/// `reshape_permute_5d`, and get the same rewrite.
#[test]
fn reshape_permute_unsqueeze_4d() {
    reshape_permute_variant(
        "0.009053396992385387",
        "%3 = aten.view_copy.default(%2, [1, 1, 49, 64, 256]) [1, 1, 49, 64, 256] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 1, 3, 2, 4]) [1, 1, 64, 49, 256] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [49, 64, 256]) [49, 64, 256] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [1, 0, 2]) [64, 49, 256] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 64, 49, 256]) [1, 1, 64, 49, 256] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_singleton_mid_5d]`: the singletons are not leading.
#[test]
fn reshape_permute_singleton_mid_5d() {
    reshape_permute_variant(
        "0.009788578376173973",
        "%3 = aten.view_copy.default(%2, [1, 49, 1, 64, 256]) [1, 49, 1, 64, 256] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 2, 4, 1, 3]) [1, 1, 256, 49, 64] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [49, 64, 256]) [49, 64, 256] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [2, 0, 1]) [256, 49, 64] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 256, 49, 64]) [1, 1, 256, 49, 64] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_negative_dims_5d]`: the model's
/// `permute(0, 1, -1, -2, -3)` reaches the pass normalized; see
/// `high_rank_permute_with_negative_dims` for the negative form.
#[test]
fn reshape_permute_negative_dims_5d() {
    reshape_permute_variant(
        "0.00944787822663784",
        "%3 = aten.view_copy.default(%2, [1, 1, 49, 64, 256]) [1, 1, 49, 64, 256] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 1, 4, 3, 2]) [1, 1, 256, 64, 49] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [49, 64, 256]) [49, 64, 256] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [2, 1, 0]) [256, 64, 49] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 256, 64, 49]) [1, 1, 256, 64, 49] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_nonsingleton_5d]`: one singleton, the batch, leaves
/// exactly rank 4.
#[test]
fn reshape_permute_nonsingleton_5d() {
    reshape_permute_variant(
        "0.009605309925973415",
        "%3 = aten.view_copy.default(%2, [1, 7, 8, 112, 128]) [1, 7, 8, 112, 128] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 2, 1, 4, 3]) [1, 8, 7, 128, 112] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [7, 8, 112, 128]) [7, 8, 112, 128] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [1, 0, 3, 2]) [8, 7, 128, 112] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 8, 7, 128, 112]) [1, 8, 7, 128, 112] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_rank6_singleton]`.
#[test]
fn reshape_permute_rank6_singleton() {
    reshape_permute_variant(
        "0.009512959979474545",
        "%3 = aten.view_copy.default(%2, [1, 1, 7, 8, 112, 128]) [1, 1, 7, 8, 112, 128] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 1, 3, 2, 5, 4]) [1, 1, 8, 7, 128, 112] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [7, 8, 112, 128]) [7, 8, 112, 128] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [1, 0, 3, 2]) [8, 7, 128, 112] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 8, 7, 128, 112]) [1, 1, 8, 7, 128, 112] int8 iq={q} oq={q}",
        ],
    );
}

/// `...[reshape_permute_chained]`: the model's rank-5 and rank-6 permutes
/// reach the pass fused into one rank-6 permute.
#[test]
fn reshape_permute_chained() {
    reshape_permute_variant(
        "0.009637427516281605",
        "%3 = aten.view_copy.default(%2, [1, 1, 49, 64, 16, 16]) [1, 1, 49, 64, 16, 16] int8 iq={} oq={q}",
        "%4 = aten.permute_copy.default(%3, [0, 1, 3, 4, 5, 2]) [1, 1, 64, 16, 16, 49] int8 iq={q} oq={q}",
        [
            "%4 = aten.view_copy.default(%3, [49, 64, 16, 16]) [49, 64, 16, 16] int8 iq={q} oq={q}",
            "%5 = aten.permute_copy.default(%4, [1, 2, 3, 0]) [64, 16, 16, 49] int8 iq={q} oq={q}",
            "%6 = aten.view_copy.default(%5, [1, 1, 64, 16, 16, 49]) [1, 1, 64, 16, 16, 49] int8 iq={q} oq={q}",
        ],
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[conv3d_rank4_channels_last]`:
/// the conv3d's NDHWC output goes back to NCDHW through a rank-5 permute
/// that the batch singleton reduces to rank 4; the squeeze's view now reads
/// the restored view. Not checked: rc4's TRANSPOSE count of 2.
#[test]
fn transpose_counts_conv3d_rank4_channels_last_high_rank() {
    high_rank(
        &[
            "%weight = placeholder [4, 3, 3, 3, 2] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [2, 6, 6, 6] float32 dim_order=[0, 2, 3, 1]",
            "%0 = aten.permute_copy.default(%x, [3, 1, 2, 0]) [6, 6, 6, 2] float32",
            "%1 = aten.view_copy.default(%0, [1, 6, 6, 6, 2]) [1, 6, 6, 6, 2] float32",
            "%2 = tosa.CONV3D.default(%1, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%3 = aten.permute_copy.default(%2, [0, 4, 1, 2, 3]) [1, 4, 4, 4, 4] float32",
            "%4 = aten.view_copy.default(%3, [4, 4, 4, 4]) [4, 4, 4, 4] float32",
            "output [%4]",
        ],
        Some(&[
            "%weight = placeholder [4, 3, 3, 3, 2] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [2, 6, 6, 6] float32",
            "%0 = aten.permute_copy.default(%x, [3, 1, 2, 0]) [6, 6, 6, 2] float32",
            "%1 = aten.view_copy.default(%0, [1, 6, 6, 6, 2]) [1, 6, 6, 6, 2] float32",
            "%2 = tosa.CONV3D.default(%1, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%3 = aten.view_copy.default(%2, [4, 4, 4, 4]) [4, 4, 4, 4] float32",
            "%4 = aten.permute_copy.default(%3, [3, 0, 1, 2]) [4, 4, 4, 4] float32",
            "%5 = aten.view_copy.default(%4, [1, 4, 4, 4, 4]) [1, 4, 4, 4, 4] float32",
            "%6 = aten.view_copy.default(%5, [4, 4, 4, 4]) [4, 4, 4, 4] float32",
            "output [%6]",
        ]),
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[conv3d_rank5_channels_last]`
/// (xfailed upstream for numerics): the output permute of a conv3d.
/// Not checked: rc4's TRANSPOSE count of 1.
#[test]
fn transpose_counts_conv3d_rank5_channels_last_high_rank() {
    high_rank(
        &[
            "%weight = placeholder [4, 3, 3, 3, 2] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 6, 6, 6, 2] float32",
            "%0 = tosa.CONV3D.default(%x, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%1 = aten.permute_copy.default(%0, [0, 4, 1, 2, 3]) [1, 4, 4, 4, 4] float32",
            "output [%1]",
        ],
        Some(&[
            "%weight = placeholder [4, 3, 3, 3, 2] float32",
            "%bias = placeholder [4] float32",
            "%x = placeholder [1, 6, 6, 6, 2] float32",
            "%0 = tosa.CONV3D.default(%x, %weight, %bias) [1, 4, 4, 4, 4] float32",
            "%1 = aten.view_copy.default(%0, [4, 4, 4, 4]) [4, 4, 4, 4] float32",
            "%2 = aten.permute_copy.default(%1, [3, 0, 1, 2]) [4, 4, 4, 4] float32",
            "%3 = aten.view_copy.default(%2, [1, 4, 4, 4, 4]) [1, 4, 4, 4, 4] float32",
            "output [%3]",
        ]),
    );
}

/// `test_conv3d.py::test_convolution_3d_tosa_FP_multi_op` (`Conv3dMultiOp`):
/// the conv3d's input and output permutes are both rewritten. The output
/// permute has two users, the reshape into the conv2ds and the residual
/// `add`, and both read the restored view; the rank-4 permutes stay.
#[test]
fn conv3d_multi_op_tosa_fp() {
    high_rank(
        &[
            "%w3d = placeholder [4, 3, 3, 3, 2] float32",
            "%b3d = placeholder [4] float32",
            "%w_main = placeholder [4, 3, 3, 4] float32",
            "%b_main = placeholder [4] float32",
            "%w_pw = placeholder [4, 1, 1, 4] float32 dim_order=[0, 3, 1, 2]",
            "%b_pw = placeholder [4] float32",
            "%x = placeholder [1, 2, 3, 8, 8] float32",
            "%0 = aten.permute_copy.default(%x, [0, 2, 3, 4, 1]) [1, 3, 8, 8, 2] float32",
            "%1 = tosa.CONV3D.default(%0, %w3d, %b3d) [1, 3, 8, 8, 4] float32",
            "%2 = aten.permute_copy.default(%1, [0, 4, 1, 2, 3]) [1, 4, 3, 8, 8] float32",
            "%3 = aten.view_copy.default(%2, [3, 4, 8, 8]) [3, 4, 8, 8] float32",
            "%4 = aten.permute_copy.default(%3, [0, 2, 3, 1]) [3, 8, 8, 4] float32",
            "%5 = tosa.CONV2D.default(%4, %w_main, %b_main) [3, 8, 8, 4] float32",
            "%6 = aten.clamp.default(%5) [3, 8, 8, 4] float32",
            "%7 = tosa.CONV2D.default(%6, %w_pw, %b_pw) [3, 8, 8, 4] float32",
            "%8 = aten.permute_copy.default(%7, [3, 0, 1, 2]) [4, 3, 8, 8] float32",
            "%9 = aten.view_copy.default(%8, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] float32",
            "%10 = aten.add.Tensor(%9, %2) [1, 4, 3, 8, 8] float32",
            "output [%10]",
        ],
        Some(&[
            "%w3d = placeholder [4, 3, 3, 3, 2] float32",
            "%b3d = placeholder [4] float32",
            "%w_main = placeholder [4, 3, 3, 4] float32",
            "%b_main = placeholder [4] float32",
            "%w_pw = placeholder [4, 1, 1, 4] float32",
            "%b_pw = placeholder [4] float32",
            "%x = placeholder [1, 2, 3, 8, 8] float32",
            "%0 = aten.view_copy.default(%x, [2, 3, 8, 8]) [2, 3, 8, 8] float32",
            "%1 = aten.permute_copy.default(%0, [1, 2, 3, 0]) [3, 8, 8, 2] float32",
            "%2 = aten.view_copy.default(%1, [1, 3, 8, 8, 2]) [1, 3, 8, 8, 2] float32",
            "%3 = tosa.CONV3D.default(%2, %w3d, %b3d) [1, 3, 8, 8, 4] float32",
            "%4 = aten.view_copy.default(%3, [3, 8, 8, 4]) [3, 8, 8, 4] float32",
            "%5 = aten.permute_copy.default(%4, [3, 0, 1, 2]) [4, 3, 8, 8] float32",
            "%6 = aten.view_copy.default(%5, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] float32",
            "%7 = aten.view_copy.default(%6, [3, 4, 8, 8]) [3, 4, 8, 8] float32",
            "%8 = aten.permute_copy.default(%7, [0, 2, 3, 1]) [3, 8, 8, 4] float32",
            "%9 = tosa.CONV2D.default(%8, %w_main, %b_main) [3, 8, 8, 4] float32",
            "%10 = aten.clamp.default(%9) [3, 8, 8, 4] float32",
            "%11 = tosa.CONV2D.default(%10, %w_pw, %b_pw) [3, 8, 8, 4] float32",
            "%12 = aten.permute_copy.default(%11, [3, 0, 1, 2]) [4, 3, 8, 8] float32",
            "%13 = aten.view_copy.default(%12, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] float32",
            "%14 = aten.add.Tensor(%13, %6) [1, 4, 3, 8, 8] float32",
            "output [%14]",
        ]),
    );
}

/// `test_conv3d.py::test_convolution_3d_tosa_INT_multi_op`: the quantized
/// form. The permutes' empty qparam maps stay empty maps on all three new
/// nodes.
#[test]
fn conv3d_multi_op_tosa_int() {
    high_rank(
        &[
            "%w3d = placeholder [4, 3, 3, 3, 2] int8",
            "%b3d = placeholder [4] int32",
            "%w_main = placeholder [4, 3, 3, 4] int8",
            "%b_main = placeholder [4] int32",
            "%w_pw = placeholder [4, 1, 1, 4] int8 dim_order=[0, 3, 1, 2]",
            "%b_pw = placeholder [4] int32",
            "%x = placeholder [1, 2, 3, 8, 8] int8",
            "%0 = aten.permute_copy.default(%x, [0, 2, 3, 4, 1]) [1, 3, 8, 8, 2] int8 iq={} oq={}",
            "%1 = tosa.CONV3D.default(%0, %w3d, %b3d) [1, 3, 8, 8, 4] int32 oq={0:int8@0.012977935373783112:-14}",
            "%2 = tosa.RESCALE.default(%1) [1, 3, 8, 8, 4] int8 iq={} oq={}",
            "%3 = aten.permute_copy.default(%2, [0, 4, 1, 2, 3]) [1, 4, 3, 8, 8] int8 iq={} oq={}",
            "%4 = tosa.RESCALE.default(%3) [1, 4, 3, 8, 8] int32 iq={} oq={}",
            "%5 = aten.view_copy.default(%3, [3, 4, 8, 8]) [3, 4, 8, 8] int8 iq={0:int8@0.012977935373783112:-14} oq={0:int8@0.012977935373783112:-14}",
            "%6 = aten.permute_copy.default(%5, [0, 2, 3, 1]) [3, 8, 8, 4] int8 iq={} oq={}",
            "%7 = tosa.CONV2D.default(%6, %w_main, %b_main) [3, 8, 8, 4] int32 oq={0:int8@0.004354184493422508:-128}",
            "%8 = tosa.RESCALE.default(%7) [3, 8, 8, 4] int8 iq={} oq={}",
            "%9 = tosa.CONV2D.default(%8, %w_pw, %b_pw) [3, 8, 8, 4] int32 oq={0:int8@0.004443853162229061:27}",
            "%10 = tosa.RESCALE.default(%9) [3, 8, 8, 4] int8 iq={} oq={}",
            "%11 = aten.permute_copy.default(%10, [3, 0, 1, 2]) [4, 3, 8, 8] int8 iq={0:int8@0.004443853162229061:27} oq={0:int8@0.004443853162229061:27}",
            "%12 = aten.view_copy.default(%11, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] int8 iq={0:int8@0.004443853162229061:27} oq={0:int8@0.004443853162229061:27}",
            "%13 = tosa.RESCALE.default(%12) [1, 4, 3, 8, 8] int32 iq={} oq={}",
            "%14 = aten.add.Tensor(%13, %4) [1, 4, 3, 8, 8] int32 iq={0:int32@0.0000000247534472919142:0, 1:int32@0.0000000247534472919142:0} oq={0:int32@0.0000000247534472919142:0}",
            "%15 = tosa.RESCALE.default(%14) [1, 4, 3, 8, 8] int8 iq={} oq={}",
            "output [%15]",
        ],
        Some(&[
            "%w3d = placeholder [4, 3, 3, 3, 2] int8",
            "%b3d = placeholder [4] int32",
            "%w_main = placeholder [4, 3, 3, 4] int8",
            "%b_main = placeholder [4] int32",
            "%w_pw = placeholder [4, 1, 1, 4] int8",
            "%b_pw = placeholder [4] int32",
            "%x = placeholder [1, 2, 3, 8, 8] int8",
            "%0 = aten.view_copy.default(%x, [2, 3, 8, 8]) [2, 3, 8, 8] int8 iq={} oq={}",
            "%1 = aten.permute_copy.default(%0, [1, 2, 3, 0]) [3, 8, 8, 2] int8 iq={} oq={}",
            "%2 = aten.view_copy.default(%1, [1, 3, 8, 8, 2]) [1, 3, 8, 8, 2] int8 iq={} oq={}",
            "%3 = tosa.CONV3D.default(%2, %w3d, %b3d) [1, 3, 8, 8, 4] int32 oq={0:int8@0.012977935373783112:-14}",
            "%4 = tosa.RESCALE.default(%3) [1, 3, 8, 8, 4] int8 iq={} oq={}",
            "%5 = aten.view_copy.default(%4, [3, 8, 8, 4]) [3, 8, 8, 4] int8 iq={} oq={}",
            "%6 = aten.permute_copy.default(%5, [3, 0, 1, 2]) [4, 3, 8, 8] int8 iq={} oq={}",
            "%7 = aten.view_copy.default(%6, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] int8 iq={} oq={}",
            "%8 = tosa.RESCALE.default(%7) [1, 4, 3, 8, 8] int32 iq={} oq={}",
            "%9 = aten.view_copy.default(%7, [3, 4, 8, 8]) [3, 4, 8, 8] int8 iq={0:int8@0.012977935373783112:-14} oq={0:int8@0.012977935373783112:-14}",
            "%10 = aten.permute_copy.default(%9, [0, 2, 3, 1]) [3, 8, 8, 4] int8 iq={} oq={}",
            "%11 = tosa.CONV2D.default(%10, %w_main, %b_main) [3, 8, 8, 4] int32 oq={0:int8@0.004354184493422508:-128}",
            "%12 = tosa.RESCALE.default(%11) [3, 8, 8, 4] int8 iq={} oq={}",
            "%13 = tosa.CONV2D.default(%12, %w_pw, %b_pw) [3, 8, 8, 4] int32 oq={0:int8@0.004443853162229061:27}",
            "%14 = tosa.RESCALE.default(%13) [3, 8, 8, 4] int8 iq={} oq={}",
            "%15 = aten.permute_copy.default(%14, [3, 0, 1, 2]) [4, 3, 8, 8] int8 iq={0:int8@0.004443853162229061:27} oq={0:int8@0.004443853162229061:27}",
            "%16 = aten.view_copy.default(%15, [1, 4, 3, 8, 8]) [1, 4, 3, 8, 8] int8 iq={0:int8@0.004443853162229061:27} oq={0:int8@0.004443853162229061:27}",
            "%17 = tosa.RESCALE.default(%16) [1, 4, 3, 8, 8] int32 iq={} oq={}",
            "%18 = aten.add.Tensor(%17, %8) [1, 4, 3, 8, 8] int32 iq={0:int32@0.0000000247534472919142:0, 1:int32@0.0000000247534472919142:0} oq={0:int32@0.0000000247534472919142:0}",
            "%19 = tosa.RESCALE.default(%18) [1, 4, 3, 8, 8] int8 iq={} oq={}",
            "output [%19]",
        ]),
    );
}

/// The negative form of `reshape_permute_negative_dims_5d`'s permute, run
/// through rc4's pass directly: `_normalize_permutation` takes `dim % rank`.
#[test]
fn high_rank_permute_with_negative_dims() {
    high_rank(
        &[
            "%x = placeholder [1, 1, 49, 64, 256] float32",
            "%0 = aten.permute_copy.default(%x, [0, 1, -1, -2, -3]) [1, 1, 256, 64, 49] float32",
            "output [%0]",
        ],
        Some(&[
            "%x = placeholder [1, 1, 49, 64, 256] float32",
            "%0 = aten.view_copy.default(%x, [49, 64, 256]) [49, 64, 256] float32",
            "%1 = aten.permute_copy.default(%0, [2, 1, 0]) [256, 64, 49] float32",
            "%2 = aten.view_copy.default(%1, [1, 1, 256, 64, 49]) [1, 1, 256, 64, 49] float32",
            "output [%2]",
        ]),
    );
}

/// A permute that moves only singletons, run through rc4's pass directly: the
/// reduced permutation is the identity, so only the two views remain
/// (`:100-101`).
#[test]
fn high_rank_identity_reduced_permutation_leaves_two_views() {
    high_rank(
        &[
            "%x = placeholder [2, 1, 3, 1, 4] float32",
            "%0 = aten.permute_copy.default(%x, [1, 0, 3, 2, 4]) [1, 2, 1, 3, 4] float32",
            "output [%0]",
        ],
        Some(&[
            "%x = placeholder [2, 1, 3, 1, 4] float32",
            "%0 = aten.view_copy.default(%x, [2, 3, 4]) [2, 3, 4] float32",
            "%1 = aten.view_copy.default(%0, [1, 2, 1, 3, 4]) [1, 2, 1, 3, 4] float32",
            "output [%1]",
        ]),
    );
}

/// Two chained rank-5 permutes, run through rc4's pass directly: the second
/// is planned from the first's value, which the first's restoring view keeps.
#[test]
fn high_rank_chained_permutes_are_both_rewritten() {
    high_rank(
        &[
            "%x = placeholder [1, 2, 3, 4, 5] float32",
            "%0 = aten.permute_copy.default(%x, [0, 4, 1, 2, 3]) [1, 5, 2, 3, 4] float32",
            "%1 = aten.permute_copy.default(%0, [0, 2, 3, 4, 1]) [1, 2, 3, 4, 5] float32",
            "output [%1]",
        ],
        Some(&[
            "%x = placeholder [1, 2, 3, 4, 5] float32",
            "%0 = aten.view_copy.default(%x, [2, 3, 4, 5]) [2, 3, 4, 5] float32",
            "%1 = aten.permute_copy.default(%0, [3, 0, 1, 2]) [5, 2, 3, 4] float32",
            "%2 = aten.view_copy.default(%1, [1, 5, 2, 3, 4]) [1, 5, 2, 3, 4] float32",
            "%3 = aten.view_copy.default(%2, [5, 2, 3, 4]) [5, 2, 3, 4] float32",
            "%4 = aten.permute_copy.default(%3, [1, 2, 3, 0]) [2, 3, 4, 5] float32",
            "%5 = aten.view_copy.default(%4, [1, 2, 3, 4, 5]) [1, 2, 3, 4, 5] float32",
            "output [%5]",
        ]),
    );
}

/// `test_pixel_shuffling.py::test_pixel_shuffle_tosa_FP[rand_4d]`: the
/// decomposition's rank-6 permute has one singleton, which leaves rank 5, so
/// it stays (`:78-79`). The same holds for every pixel shuffle and unshuffle
/// case in that file.
#[test]
fn pixel_shuffle_rank6_permute_is_left_alone() {
    high_rank(
        &[
            "%inputs = placeholder [1, 12, 64, 64] float32",
            "%0 = aten.view_copy.default(%inputs, [1, 3, 2, 2, 64, 64]) [1, 3, 2, 2, 64, 64] float32",
            "%1 = aten.permute_copy.default(%0, [0, 1, 4, 2, 5, 3]) [1, 3, 64, 2, 64, 2] float32",
            "%2 = aten.view_copy.default(%1, [1, 3, 128, 128]) [1, 3, 128, 128] float32",
            "output [%2]",
        ],
        None,
    );
}

/// `test_transpose_counts.py::test_transpose_counts_tosa_FP_channels_last[maxpool2d_dilation_channels_last]`:
/// the dilated max pool's space-to-batch permutes are rank 6 with one
/// singleton each, so both stay.
#[test]
fn maxpool2d_dilation_rank6_permutes_are_left_alone() {
    high_rank(
        &[
            "%x = placeholder [1, 8, 8, 2] float32",
            "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 8, 8] float32",
            "%1 = tosa.CONST_SHAPE.default() - oq={}",
            "%2 = tosa.PAD.default(%0, %1) [1, 2, 10, 10] float32 oq={}",
            "%3 = aten.view_copy.default(%2, [1, 2, 5, 2, 5, 2]) [1, 2, 5, 2, 5, 2] float32 iq={} oq={}",
            "%4 = aten.permute_copy.default(%3, [3, 5, 0, 2, 4, 1]) [2, 2, 1, 5, 5, 2] float32 iq={} oq={}",
            "%5 = aten.view_copy.default(%4, [4, 5, 5, 2]) [4, 5, 5, 2] float32",
            "%6 = tosa.MAX_POOL2D.default(%5) [4, 3, 3, 2] float32",
            "%7 = aten.view_copy.default(%6, [2, 2, 1, 3, 3, 2]) [2, 2, 1, 3, 3, 2] float32",
            "%8 = aten.permute_copy.default(%7, [2, 3, 0, 4, 1, 5]) [1, 3, 2, 3, 2, 2] float32 iq={} oq={}",
            "%9 = aten.view_copy.default(%8, [1, 6, 6, 2]) [1, 6, 6, 2] float32",
            "%10 = aten.slice_copy.Tensor(%9) [1, 6, 6, 2] float32",
            "%11 = aten.slice_copy.Tensor(%10) [1, 6, 6, 2] float32",
            "output [%11]",
        ],
        None,
    );
}

/// `test_permute.py::test_permute_u55_INT[to_ndhwc]`, through the U55
/// pipeline without the FVP run: a rank-5 permute with no singleton stays.
#[test]
fn permute_to_ndhwc_without_singletons_is_left_alone() {
    high_rank(
        &[
            "%x = placeholder [2, 2, 41, 41, 41] int8",
            "%0 = aten.permute_copy.default(%x, [0, 2, 3, 4, 1]) [2, 41, 41, 41, 2] int8 iq={0:int8@0.003919638693332672:-128} oq={0:int8@0.003919638693332672:-128}",
            "output [%0]",
        ],
        None,
    );
}
