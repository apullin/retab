//! Adversarial review of the propagation port (APU-159): rc4 behaviours the
//! port missed, each now fixed and kept as a regression test, plus
//! diagnostics that replay the review's probe fixtures
//! (`cargo test --lib propagate_review -- --ignored`).
//!
//! The rc4 evidence for each comes from probe scripts run in the rc4 venv
//! (tests/fixtures/propagate_review, named in each test), which build the
//! graph with `tests/fixtures/layout_replay/gen_layout_replay.py`'s builder
//! and run the real passes as `arm_pass_manager.py:726-727` builds them.

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::propagate::IrPropagateViewCopyPermutePass;
use crate::layout::propagate_replay_tests::DownThenUp;
use crate::layout::replay::{self, Options};

const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: true,
    float32_activations: false,
    modified: true,
};

/// The graph a fixture case starts from.
fn before(fixture: &str, name: &str) -> IrGraph {
    let cases = replay::parse(fixture).expect("a valid fixture");
    let case = cases
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    replay::build(&case.before)
}

/// Replays `$RETAB_REVIEW_DIR/{down,up,down_up}.txt`, written by the review's
/// probe scripts, through the Down pass, the Up pass and both in sequence.
#[test]
#[ignore = "diagnostic: replays review probe fixtures in $RETAB_REVIEW_DIR"]
fn propagate_review_external() {
    let Ok(dir) = std::env::var("RETAB_REVIEW_DIR") else {
        println!("RETAB_REVIEW_DIR is not set; nothing to replay");
        return;
    };
    let passes: [(&dyn IrPass, &str); 3] = [
        (&IrPropagateViewCopyPermutePass::ARM_DOWN, "down.txt"),
        (&IrPropagateViewCopyPermutePass::ARM_UP, "up.txt"),
        (&DownThenUp, "down_up.txt"),
    ];
    for (pass, file) in passes {
        let path = std::path::Path::new(&dir).join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            println!("{}: skipped", path.display());
            continue;
        };
        let results = replay::replay(pass, &text, WITH_FLAG);
        let failures: Vec<_> = results.iter().filter(|result| result.mismatch.is_some()).collect();
        println!(
            "{}: {} of {} differ ({})",
            path.display(),
            failures.len(),
            results.len(),
            replay::class_counts(failures.iter().copied())
        );
        let limit: usize = std::env::var("RETAB_REVIEW_SHOW").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let skip = std::env::var("RETAB_REVIEW_SKIP_CLASS").unwrap_or_default();
        for failure in failures
            .iter()
            .filter(|failure| failure.mismatch.as_ref().is_some_and(|m| m.class != skip))
            .take(limit)
        {
            println!("{}", failure.report);
        }
    }
}

/// Finds the live call with `target`; panics unless there is exactly one.
fn only_call(graph: &IrGraph, target: &str) -> crate::fx_ir::NodeId {
    let ids: Vec<_> = graph
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| graph.node(id).target == target)
        .collect();
    assert_eq!(ids.len(), 1, "live {target} nodes: {ids:?}");
    ids[0]
}

/// A rank-changing view feeding a 16-bit table, whose operand is the
/// `[N, 1, 1, C]` shape (N > 1) the TOSA TABLE guard exists for
/// (backends/arm/_passes/propagate_view_copy_permute_pass.py:61-80,
/// MLBEDSW-11805: Ethos-U85 miscompiles such a TABLE).
const UNSAFE_INT16_TABLE: &str = "\
case view_into_int16_table_unsafe_operand
before
%0 = placeholder x : torch.int16[2, 8] q=none
%1 = aten.view_copy.default(%0, [2, 1, 1, 8]) : torch.int16[2, 1, 1, 8] q=none
%2 = aten.sigmoid.default(%1) : torch.int16[2, 1, 1, 8] q=in{0}/out{0} qp=in0:0.0009765625:0:-32768:32767:torch.int16 qp=out0:3.0517578125e-05:0:-32768:32767:torch.int16
output([%2])
after modified=false same
end
";

/// rc4 (probe_table16.py in the review's scripts, run in the venv):
/// InsertTableOps turns the sigmoid into `TABLE(view, table) -> RESCALE`
/// (a 16-bit table with a non-zero shift). The Down pass sinks the view past
/// the TABLE, because the TABLE then reads `x`'s safe `[2, 8]`
/// (`_shape_seen_by_next_node` is the view's input moving down), and stops it
/// at the RESCALE (transparent, so a rank-changing view can't cross). The Up
/// pass can't hoist it back: the TABLE would read the view's `[2, 1, 1, 8]`,
/// which the guard rejects. rc4's final TABLE operand is `[2, 8]`:
///
/// ```text
/// after Down: TABLE(x, c) : int32[2, 8] -> view [2, 1, 1, 8] -> RESCALE int16
/// after Up:   (unchanged)  TABLE operand shape after Down+Up: [2, 8]
/// ```
///
/// The port used to treat the pending 16-bit table as transparent (for its
/// RESCALE), so the view could never cross it moving down and the table kept
/// reading `[2, 1, 1, 8]`, the operand rc4 steers away from. The driver now
/// gives the table rc4's two nodes (`layout::pending::split_table_rescales`),
/// so the view parks between them as in rc4, and the table stays split.
#[test]
fn int16_table_does_not_keep_the_unsafe_operand_rc4_removes() {
    use crate::layout::pending::{has_split_rescale, TABLE_RESCALE_OF_KEY};
    let mut graph = before(UNSAFE_INT16_TABLE, "view_into_int16_table_unsafe_operand");
    DownThenUp.run(&mut graph);
    let table = only_call(&graph, "aten.sigmoid.default");
    let operand = graph.node(table).args[0].as_node().expect("a node operand");
    assert_eq!(
        graph.node_shape(operand),
        Some(&[2, 8][..]),
        "rc4's TABLE reads x's [2, 8] after Down+Up; the port's reads {:?}",
        graph.node_shape(operand)
    );
    assert!(has_split_rescale(graph.node(table)));
    assert_eq!(graph.node(table).meta.val_dtype(), Some("torch.int32"));
    let view = only_call(&graph, "aten.view_copy.default");
    let rescale = only_call(&graph, "tosa.RESCALE.default");
    assert_eq!(graph.live_user_ids(table), [view]);
    assert_eq!(graph.node(rescale).args[0].as_node(), Some(view));
    assert_eq!(graph.node(rescale).meta.extra_str(TABLE_RESCALE_OF_KEY), Some(graph.node(table).name.as_str()));
}

/// A structural signature of `graph` that ignores node order: each node is
/// described by its op, value and qparam keys with node references replaced
/// by the referenced node's description, recursively; the signature is the
/// sorted descriptions of the live nodes plus the output's.
fn order_free_signature(graph: &replay::Graph) -> (Vec<String>, String) {
    fn describe(graph: &replay::Graph, index: usize, memo: &mut Vec<Option<String>>) -> String {
        if let Some(text) = &memo[index] {
            return text.clone();
        }
        let node = &graph.nodes[index];
        let op = match &node.op {
            replay::Op::Placeholder { name, .. } => format!("placeholder {name}"),
            replay::Op::GetAttr { name } => format!("get_attr {name}"),
            replay::Op::Call { target, args, kwargs } => {
                let args: Vec<String> = args.iter().map(|arg| arg_text(graph, arg, memo)).collect();
                let kwargs: Vec<String> = kwargs
                    .iter()
                    .map(|(key, arg)| format!("{key}={}", arg_text(graph, arg, memo)))
                    .collect();
                format!("{target}({}; {})", args.join(", "), kwargs.join(", "))
            }
        };
        let text = format!("{op} : {:?} q={:?}/{:?} s={:?}", node.val, node.q_in, node.q_out, node.special);
        memo[index] = Some(text.clone());
        text
    }
    fn arg_text(graph: &replay::Graph, arg: &replay::Arg, memo: &mut Vec<Option<String>>) -> String {
        match arg {
            replay::Arg::Node(index) => format!("<{}>", describe(graph, *index, memo)),
            // The importer reads `[]` as `None`, and the replay compares them equal.
            replay::Arg::List(items) if items.is_empty() => "None".to_string(),
            replay::Arg::List(items) => {
                let items: Vec<String> = items.iter().map(|item| arg_text(graph, item, memo)).collect();
                format!("[{}]", items.join(", "))
            }
            other => format!("{other:?}"),
        }
    }
    let mut memo = vec![None; graph.nodes.len()];
    let mut nodes: Vec<String> = (0..graph.nodes.len())
        .filter(|&index| !graph.nodes[index].dead)
        .map(|index| describe(graph, index, &mut memo))
        .collect();
    nodes.sort();
    let output = arg_text(graph, &graph.output, &mut memo);
    (nodes, output)
}

/// For each case of `$RETAB_REVIEW_DIR/down_up.txt` where the port's
/// Down+Up differs from rc4's, says whether the two graphs are the same up to
/// node order.
#[test]
#[ignore = "diagnostic: classifies Down+Up mismatches in $RETAB_REVIEW_DIR"]
fn propagate_review_order_only() {
    let Ok(dir) = std::env::var("RETAB_REVIEW_DIR") else {
        println!("RETAB_REVIEW_DIR is not set; nothing to replay");
        return;
    };
    let text = std::fs::read_to_string(std::path::Path::new(&dir).join("down_up.txt")).expect("down_up.txt");
    let cases = replay::parse(&text).expect("a valid fixture");
    let (mut order_only, mut structural, mut other) = (Vec::new(), Vec::new(), Vec::new());
    for case in &cases {
        let result = replay::replay_case(&DownThenUp, case, WITH_FLAG);
        let Some(mismatch) = &result.mismatch else { continue };
        if !matches!(mismatch.class, "target" | "args" | "shape" | "dtype" | "nodes" | "output") {
            other.push(format!("{} ({})", case.name, mismatch.class));
            continue;
        }
        let mut graph = replay::build(&case.before);
        DownThenUp.run(&mut graph);
        let actual = replay::snapshot(&graph).expect("a valid graph");
        if order_free_signature(&case.after) == order_free_signature(&actual) {
            order_only.push(case.name.clone());
        } else {
            structural.push(case.name.clone());
        }
    }
    println!("order only ({}): {}", order_only.len(), order_only.join(" "));
    println!("structural ({}): {}", structural.len(), structural.join(" "));
    println!("other ({}): {}", other.len(), other.join(" "));
}

/// Two 16-bit tables behind a shared permute; one branch reaches its table
/// through a rank-changing view. Recorded in the venv by
/// probe_t16_extra_transpose.py (the review's scripts): InsertTableOps, the
/// Arm Down pass, the Arm Up pass, then the TABLEs restored to RETAB's form.
const SHARED_PERMUTE_INT16_TABLES: &str = "\
case shared_permute_into_int16_tables
before
%0 = placeholder x1 : torch.int16[2, 1, 3] q=none
%1 = aten.permute_copy.default(%0, [1, 2, 0]) : torch.int16[1, 3, 2] q=none
%2 = aten.permute_copy.default(%1, [0, 2, 1]) : torch.int16[1, 2, 3] q=none
%3 = aten.tanh.default(%2) : torch.int16[1, 2, 3] q=in{0}/out{0} qp=in0:0.00390625:0:-32768:32767:torch.int16 qp=out0:0.000244140625:0:-32768:32767:torch.int16
%4 = aten.permute_copy.default(%3, [1, 2, 0]) : torch.int16[2, 3, 1] q=none
%5 = aten.sigmoid.default(%4) : torch.int16[2, 3, 1] q=in{0}/out{0} qp=in0:0.0009765625:0:-32768:32767:torch.int16 qp=out0:3.0517578125e-05:0:-32768:32767:torch.int16
%6 = aten.view_copy.default(%1, [1, 6]) : torch.int16[1, 6] q=none
%7 = aten.sigmoid.default(%6) : torch.int16[1, 6] q=in{0}/out{0} qp=in0:0.00048828125:0:-32768:32767:torch.int16 qp=out0:0.00048828125:0:-32768:32767:torch.int16
output([%5, %7])
after modified=true
%0 = placeholder x1 : torch.int16[2, 1, 3] q=none
%1 = aten.view_copy.default(%0, [2, 3, 1]) : torch.int16[2, 3, 1] q=in{0}/out{0} qp=in0:0.00390625:0:-32768:32767:torch.int16 qp=out0:0.000244140625:0:-32768:32767:torch.int16
%2 = aten.tanh.default(%1) : torch.int16[2, 3, 1] q=in{0}/out{0} qp=in0:0.00390625:0:-32768:32767:torch.int16 qp=out0:0.000244140625:0:-32768:32767:torch.int16
%3 = aten.sigmoid.default(%2) : torch.int16[2, 3, 1] q=in{0}/out{0} qp=in0:0.0009765625:0:-32768:32767:torch.int16 qp=out0:3.0517578125e-05:0:-32768:32767:torch.int16
%4 = aten.permute_copy.default(%0, [1, 2, 0]) : torch.int16[1, 3, 2] q=in{0}/out{0} qp=in0:0.00048828125:0:-32768:32767:torch.int16 qp=out0:0.00048828125:0:-32768:32767:torch.int16
%5 = aten.view_copy.default(%4, [1, 6]) : torch.int16[1, 6] q=none
%6 = aten.sigmoid.default(%5) : torch.int16[1, 6] q=in{0}/out{0} qp=in0:0.00048828125:0:-32768:32767:torch.int16 qp=out0:0.00048828125:0:-32768:32767:torch.int16
output([%3, %6])
end
";

/// rc4's Down pass sinks the view `%6` past its TABLE and parks it before
/// the TABLE's RESCALE (trace_t16_minimal.py prints each stage); with that
/// user gone the shared permute's remaining chain cancels and canonicalizes,
/// and after Up rc4 has one permute and two views. The port could not sink
/// the rank-changing view into the pending 16-bit table (it counted as
/// transparent for its RESCALE), so the shared permute kept two users and the
/// other branch ended as a second permute: two TRANSPOSEs where rc4 has one.
/// With the table split for the pass the two agree; the committed
/// `propagate_view_copy_permute_down_up` fixture holds this case and random
/// graphs with 16-bit tables.
#[test]
fn int16_tables_behind_a_shared_permute_match_rc4_after_down_and_up() {
    replay::check_cases_with(
        &DownThenUp,
        SHARED_PERMUTE_INT16_TABLES,
        &["shared_permute_into_int16_tables"],
        WITH_FLAG,
    );
}

/// Recorded in the venv by probe_user_order_min.py (the review's scripts):
/// the minimized form of a case user_order_motif.py found.
const DRIVER_USER_ORDER: &str = "\
case driver_user_order
before
%0 = placeholder x0 : torch.float32[3, 3] q=none
%1 = aten.permute_copy.default(%0, [1, 0]) : torch.float32[3, 3] q=none
%2 = aten.permute_copy.default(%1, [1, 0]) : torch.float32[3, 3] q=none
%3 = aten.view_copy.default(%2, [3, 3]) : torch.float32[3, 3] q=none
%4 = aten.neg.default(%3) : torch.float32[3, 3] q=none
%5 = aten.permute_copy.default(%4, [1, 0]) : torch.float32[3, 3] q=none
%6 = aten.sub.Tensor(%5, %2) : torch.float32[3, 3] q=none
%7 = aten.permute_copy.default(%6, [1, 0]) : torch.float32[3, 3] q=none
%8 = aten.permute_copy.default(%7, [1, 0]) : torch.float32[3, 3] q=none
%9 = aten.permute_copy.default(%8, [1, 0]) : torch.float32[3, 3] q=none
%10 = aten.view_copy.default(%9, [3, 3]) : torch.float32[3, 3] q=none
%11 = aten.mul.Tensor(%9, %10) : torch.float32[3, 3] q=none
%12 = aten.sub.Tensor(%11, %1) : torch.float32[3, 3] q=none
%13 = aten.permute_copy.default(%12, [1, 0]) : torch.float32[3, 3] q=none
output([%1, %13])
after modified=true
%0 = placeholder x0 : torch.float32[3, 3] q=none
%1 = aten.permute_copy.default(%0, [1, 0]) : torch.float32[3, 3] q=none
%2 = aten.neg.default(%1) : torch.float32[3, 3] q=none
%3 = aten.sub.Tensor(%2, %0) : torch.float32[3, 3] q=none
%4 = aten.mul.Tensor(%3, %3) : torch.float32[3, 3] q=none
%5 = aten.sub.Tensor(%4, %0) : torch.float32[3, 3] q=none
output([%1, %5])
end
";

/// The driver runs RemovePermutesAroundElementwiseTosaOps after sweeps that
/// rewired nodes, with no retrace in between, so rc4 hands it users in FX
/// insertion order, which the sweeps' rewiring has taken out of graph order,
/// and that order decides which region it rewrites. rc4 ends with one permute
/// (the returned `x.permute`); re-sorting every node's users into graph order
/// before each fusion sub-pass makes rc4 end with two
/// (probe_user_order_min.py prints both; pinpoint_order.py shows sorting
/// before RemovePermutes alone is enough). The port used to end with two; it
/// now keeps torch.fx's user order through the sweeps
/// (`LayoutCtx::users_in_fx_order`) and resets it where rc4 re-traces.
#[test]
fn down_pass_matches_rc4_when_sweeps_reorder_users() {
    replay::check_cases_with(
        &IrPropagateViewCopyPermutePass::ARM_DOWN,
        DRIVER_USER_ORDER,
        &["driver_user_order"],
        WITH_FLAG,
    );
}

/// An NCHW permute, a clamp and a channel split whose first branch goes back
/// to NHWC: the shape RewriteConv leaves around a channel split between two
/// convolutions. Recorded in the venv by probe_channel_split.py (the review's
/// scripts), with rc4's result: the slice cuts W instead of C.
const CHANNEL_SPLIT: &str = "\
case channel_split_after_clamp
before
%0 = placeholder y : torch.float32[1, 3, 5, 8] q=none
%1 = aten.permute_copy.default(%0, [0, 3, 1, 2]) : torch.float32[1, 8, 3, 5] q=none
%2 = aten.clamp.default(%1, 0.0, 6.0) : torch.float32[1, 8, 3, 5] q=none
%3 = aten.slice_copy.Tensor(%2, 1, 0, 4) : torch.float32[1, 4, 3, 5] q=none
%4 = aten.permute_copy.default(%3, [0, 2, 3, 1]) : torch.float32[1, 3, 5, 4] q=none
%5 = aten.neg.default(%3) : torch.float32[1, 4, 3, 5] q=none
output([%4, %5])
after modified=true
%0 = placeholder y : torch.float32[1, 3, 5, 8] q=none
%1 = aten.clamp.default(%0, 0.0, 6.0) : torch.float32[1, 3, 5, 8] q=none
%2 = aten.slice_copy.Tensor(%1, 2, 0, 4) : torch.float32[1, 3, 4, 8] q=none
%3 = aten.permute_copy.default(%2, [0, 3, 1, 2]) : torch.float32[1, 8, 3, 4] q=none
%4 = aten.permute_copy.default(%3, [0, 2, 3, 1]) : torch.float32[1, 3, 4, 8] q=none
%5 = aten.neg.default(%3) : torch.float32[1, 8, 3, 4] q=none
output([%4, %5])
end
";

/// rc4's Down walk crosses the clamp and the slice, rewriting the slice's dim
/// from 1 to 3 as it advances (`:234-235`), finds the fork after the slice
/// unsplittable (a permute branch isn't swappable), and falls back to a
/// region move from the original permute (`:1036-1039`), which leaves the
/// slice reading NCHW data with the NHWC dim; the next sweep moves the new
/// permute past the slice and maps the dim again. Both graph outputs change
/// shape, `[1, 3, 5, 4]` to `[1, 3, 4, 8]` and `[1, 4, 3, 5]` to
/// `[1, 8, 3, 4]` (check_split_numerics.py runs both in torch). The port used
/// to replay that; it now takes the walk's assignments back before the
/// region move (`WalkWrites::undo`), a known divergence in the replays
/// (`channel_split_after_clamp`).
#[test]
fn down_pass_keeps_a_channel_split_on_the_channel_axis() {
    let mut graph = before(CHANNEL_SPLIT, "channel_split_after_clamp");
    IrPropagateViewCopyPermutePass::ARM_DOWN.run(&mut graph);
    let output = graph.output_node().expect("an output node");
    let mut ids = Vec::new();
    output.args[0].collect_node_ids(&mut ids);
    let shapes: Vec<Vec<i64>> = ids
        .iter()
        .map(|&id| graph.node_shape(id).expect("a tensor").to_vec())
        .collect();
    assert_eq!(
        shapes,
        [vec![1, 3, 5, 4], vec![1, 4, 3, 5]],
        "the graph outputs must keep their shapes"
    );
}

/// A failed walk is memoized under the version stamps of every node it read.
/// `is_per_tensor_rescale` also reads the RESCALE's input's special dtype
/// (`_is_per_tensor_rescale`,
/// backends/arm/_passes/propagate_view_copy_permute_pass.py:132-144), and a
/// walk that failed there (a view moving up, which `blocks_moving` doesn't
/// read inputs for) used not to record that input. Here the RESCALE's input
/// loses its INT4 marker after a failed walk, so the view can now cross, and
/// the memo must not skip the walk. `CHECK_SKIPPED_WALKS` is switched off to
/// show what a release build does.
#[test]
fn walk_memo_reads_the_rescale_input_special_dtype() {
    use crate::fx_meta::MetaValue;
    use crate::layout::ctx::LayoutCtx;
    use crate::layout::fx_node::SPECIAL_DTYPE_KEY;
    use crate::layout::propagate::{propagate_iteration, Direction, PropagateConfig, CHECK_SKIPPED_WALKS};
    use crate::layout::testing::*;

    let build = |marked: bool| {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], "torch.int8");
        if marked {
            g.graph_mut()
                .node_mut(x)
                .meta
                .set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("INT4".into()));
        }
        let r = g.rescale(x, "torch.int32", 0.5);
        let v = g.view(r, &[3, 2]);
        g.output(&[v]);
        (g, x)
    };
    let config = PropagateConfig::arm(Direction::Up);
    // Control: without the marker and without a memo, the view crosses.
    let (mut control, _) = build(false);
    assert!(propagate_iteration(&mut LayoutCtx::new(control.graph_mut()), &config));
    let (mut g, x) = build(true);
    let moved = {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(
            !propagate_iteration(&mut ctx, &config),
            "the INT4 marker on the input fails `_is_per_tensor_rescale`, so the view stays"
        );
        ctx.meta_mut(x).extras.remove(SPECIAL_DTYPE_KEY);
        let checking = CHECK_SKIPPED_WALKS.with(|check| check.replace(false));
        let moved = propagate_iteration(&mut ctx, &config);
        CHECK_SKIPPED_WALKS.with(|check| check.set(checking));
        moved
    };
    assert!(moved, "with the marker gone the RESCALE is per-tensor and the view crosses it");
}

/// Prints the Down driver's trace (`propagate::TRACE`) on one case of
/// `$RETAB_REVIEW_DIR/down.txt`, named by `$RETAB_TRACE_CASE`, to compare with
/// tests/fixtures/propagate_review/trace_driver.py's log of rc4 (the two
/// print the same graph text, users in torch.fx's order, at the same points,
/// so `diff` finds the first step where they part).
#[test]
#[ignore = "diagnostic: prints the driver trace of $RETAB_TRACE_CASE in $RETAB_REVIEW_DIR/down.txt"]
fn propagate_review_trace() {
    use crate::layout::propagate::TRACE;
    let (Ok(dir), Ok(name)) = (std::env::var("RETAB_REVIEW_DIR"), std::env::var("RETAB_TRACE_CASE")) else {
        println!("set RETAB_REVIEW_DIR and RETAB_TRACE_CASE");
        return;
    };
    let text = std::fs::read_to_string(std::path::Path::new(&dir).join("down.txt")).expect("down.txt");
    let mut graph = before(&text, &name);
    TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
    IrPropagateViewCopyPermutePass::ARM_DOWN.run(&mut graph);
    let log = TRACE.with(|trace| trace.borrow_mut().take()).unwrap_or_default();
    println!("{}", log.join("\n"));
}
