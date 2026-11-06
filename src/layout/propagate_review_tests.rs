//! Adversarial review of the propagation port (APU-159): rc4 behaviours the
//! port misses. Each failing test is `#[ignore = "BUG: ..."]` so the suite
//! stays green; run them with
//! `cargo test --lib propagate_review -- --ignored`.
//!
//! The rc4 evidence for each comes from probe scripts run in the rc4 venv
//! (named in each test), which build the graph with
//! `tests/fixtures/layout_replay/gen_layout_replay.py`'s builder and run the
//! real passes as `arm_pass_manager.py:726-727` builds them.

use crate::fx_ir::{IrGraph, IrPass};
use crate::layout::propagate::IrPropagateViewCopyPermutePass;
use crate::layout::replay::{self, Options};

/// The Arm Down pass, then the Arm Up pass, as phase 6 runs them
/// (`arm_pass_manager.py:726-727`).
struct DownThenUp;

impl IrPass for DownThenUp {
    fn name(&self) -> &str {
        "IrPropagateViewCopyPermuteDown+Up"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let down = IrPropagateViewCopyPermutePass::ARM_DOWN.run(graph);
        let up = IrPropagateViewCopyPermutePass::ARM_UP.run(graph);
        down | up
    }
}

const WITH_FLAG: Options = Options {
    dim_orders: replay::DimOrders::Ignore,
    dead_nodes: true,
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
/// The port treats the pending 16-bit table as transparent (its RESCALE), so
/// the rank-changing view can never cross it moving down; the table keeps
/// reading `[2, 1, 1, 8]`, the operand rc4 steers away from. The closest RETAB
/// form of rc4's result is the table reading `x` with the view after it (a
/// per-tensor RESCALE commutes with the view). With N = 1 rc4's Up pass moves
/// the view back above the table, so the two agree there.
#[test]
#[ignore = "BUG: a 16-bit pending table keeps the [N,1,1,C] operand rc4's Down pass removes (TABLE guard, MLBEDSW-11805)"]
fn int16_table_does_not_keep_the_unsafe_operand_rc4_removes() {
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
/// and after Up rc4 has one permute and two views. The port can't sink the
/// rank-changing view into the pending 16-bit table (it counts as transparent
/// for its RESCALE), so the shared permute keeps two users and the other
/// branch ends as a second permute: two TRANSPOSEs where rc4 has one. Out of
/// 2,000 random graphs with 16-bit tables (fuzz_t16.py), the port's Down+Up
/// differs from rc4's in 29: this one structurally, 24 in node order only
/// and 4 in the `modified` flag; the committed fixtures can't show any of
/// them, since rc4's Down result is unrepresentable in RETAB's form and the
/// random generator draws only int8 tables.
#[test]
#[ignore = "BUG: 16-bit pending tables modelled as transparent: Down+Up leaves an extra TRANSPOSE rc4 doesn't"]
fn int16_tables_behind_a_shared_permute_match_rc4_after_down_and_up() {
    replay::check_cases_with(
        &DownThenUp,
        SHARED_PERMUTE_INT16_TABLES,
        &["shared_permute_into_int16_tables"],
        WITH_FLAG,
    );
}
