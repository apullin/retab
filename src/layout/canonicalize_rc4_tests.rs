//! Fidelity tests for `IrCanonicalizeViewCopyPermutePass` (canonicalize.rs)
//! against rc4's `CanonicalizeViewCopyPermutePass` as Arm configures it
//! (`backends/arm/_passes/canonicalize_view_copy_permute_pass.py` over
//! `backends/transforms/canonicalize_view_copy_permute_pass.py`).
//!
//! - `matches_rc4_fixture` replays tests/fixtures/canonicalize_rc4_cases.txt,
//!   which canonicalize_rc4_cases.py next to it writes by running rc4's pass,
//!   retrace included: the upstream static graphs, chains around the upstream
//!   swap vectors, and seeded random graphs of views, permutes and relus with
//!   forks, dead chains, `-1` sizes, negative dims, rank 0 and int8 values.
//!   Each case compares the surviving nodes, which input node each one is, and
//!   their args, shapes and dtypes.
//! - The `upstream_` tests port the static cases of
//!   `backends/arm/test/passes/test_canonicalize_view_copy_permute_pass.py`;
//!   the two symbolic ones have no counterpart, as RETAB's dims are integers.
//! - The edge-case tests pin behaviour checked against the rc4 venv.
//! - The swap enumerations follow the `generated_cases_are_semantically_valid`
//!   tests of `test_view_permute_swap.py` and `test_permute_view_swap.py`,
//!   through the whole pass.
//!
//! Every graph is also evaluated on integer data before and after the pass,
//! which must not change what it computes.
//!
//! rc4's retrace renames call nodes after their targets, so in rc4 a view that
//! a swap turned into a permute comes out named `aten_permute_copy_default..`.
//! RETAB keeps the node's name; what both keep is the node itself, with its
//! meta. The tests below name nodes by what they were built as.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::time::Duration;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::layout::canonicalize::IrCanonicalizeViewCopyPermutePass;
use crate::layout::testing::{ints, node, LayoutGraph, ADD, ADD_INPLACE, PERMUTE, RELU, VIEW};
use crate::test_ir::qparam;

const PERMUTE_DEFAULT: &str = "aten.permute.default";
const VIEW_DEFAULT: &str = "aten.view.default";
const FIXTURE: &str = include_str!("../../tests/fixtures/canonicalize_rc4_cases.txt");

/// Runs the pass, checking that the graph computes the same outputs after it.
fn run(g: &mut LayoutGraph) -> bool {
    let before = evaluate(g.graph());
    let modified = IrCanonicalizeViewCopyPermutePass.run(g.graph_mut());
    assert_eq!(
        evaluate(g.graph()),
        before,
        "the pass changed the graph's outputs"
    );
    modified
}

/// An `int[]` arg as RETAB's FX import gives it: an empty list arrives as
/// `IrArg::None` (`fx_ir::resolve_arg`).
fn imported_ints(values: &[i64]) -> IrArg {
    if values.is_empty() {
        IrArg::None
    } else {
        ints(values)
    }
}

fn int_list(arg: &IrArg) -> Vec<i64> {
    match arg {
        IrArg::None => Vec::new(),
        arg => arg.as_int_list().expect("an int list"),
    }
}

fn output_ids(graph: &IrGraph) -> Vec<NodeId> {
    let output = graph.live_output_ids()[0];
    match &graph.node(output).args[0] {
        IrArg::List(items) => items
            .iter()
            .map(|item| item.as_node().expect("outputs are nodes"))
            .collect(),
        IrArg::Node(id) => vec![*id],
        other => panic!("unexpected output arg {other:?}"),
    }
}

/// The live call nodes in graph order, then the output, by node name.
fn sig(g: &LayoutGraph) -> Vec<String> {
    let graph = g.graph();
    let name = |id: NodeId| graph.node(id).name.clone();
    let mut lines = Vec::new();
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        if n.op != NodeOp::CallFunction {
            continue;
        }
        let shape = graph.node_shape(id).unwrap_or(&[]);
        let inputs: Vec<String> = n.args.iter().filter_map(IrArg::as_node).map(name).collect();
        lines.push(match n.target.as_str() {
            VIEW | PERMUTE => format!(
                "{} = {}({}, {:?}) -> {shape:?}",
                n.name,
                if n.target == VIEW { "view" } else { "permute" },
                inputs[0],
                int_list(&n.args[1]),
            ),
            target => format!("{} = {target}({}) -> {shape:?}", n.name, inputs.join(", ")),
        });
    }
    let outputs: Vec<String> = output_ids(graph).into_iter().map(name).collect();
    lines.push(format!("output({})", outputs.join(", ")));
    lines
}

// --- A reference evaluator ---------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Tensor {
    shape: Vec<i64>,
    data: Vec<i64>,
}

fn numel(shape: &[i64]) -> i64 {
    shape.iter().product()
}

fn reshape(input: &Tensor, size: &[i64]) -> Tensor {
    let count = numel(&input.shape);
    let known: i64 = size.iter().filter(|&&dim| dim != -1).product();
    let shape: Vec<i64> = size
        .iter()
        .map(|&dim| if dim == -1 { count / known } else { dim })
        .collect();
    assert_eq!(numel(&shape), count, "view {size:?} of {:?}", input.shape);
    Tensor {
        shape,
        data: input.data.clone(),
    }
}

fn permute(input: &Tensor, dims: &[i64]) -> Tensor {
    let rank = input.shape.len();
    let dims: Vec<usize> = dims
        .iter()
        .map(|&dim| usize::try_from(if dim < 0 { dim + rank as i64 } else { dim }).unwrap())
        .collect();
    let mut sorted = dims.clone();
    sorted.sort_unstable();
    assert!(
        sorted.into_iter().eq(0..rank),
        "permute {dims:?} of {:?}",
        input.shape
    );
    let mut strides = vec![1i64; rank];
    for axis in (0..rank.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * input.shape[axis + 1];
    }
    let shape: Vec<i64> = dims.iter().map(|&dim| input.shape[dim]).collect();
    let mut index = vec![0i64; rank];
    let mut data = Vec::with_capacity(input.data.len());
    for _ in 0..numel(&shape) {
        let source: i64 = (0..rank)
            .map(|axis| index[axis] * strides[dims[axis]])
            .sum();
        data.push(input.data[source as usize]);
        for axis in (0..rank).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Tensor { shape, data }
}

/// The graph's outputs on distinct integer inputs.
fn evaluate(graph: &IrGraph) -> Vec<Tensor> {
    let mut values: HashMap<NodeId, Tensor> = HashMap::new();
    let mut placeholders = 0;
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        let tensor = match n.op {
            NodeOp::Placeholder => {
                let shape = graph
                    .node_shape_vec(id)
                    .expect("placeholders carry a shape");
                let count = numel(&shape);
                let base = 1000 * placeholders - count;
                placeholders += 1;
                Tensor {
                    data: (0..count).map(|i| base + 2 * i).collect(),
                    shape,
                }
            }
            NodeOp::Output => {
                return output_ids(graph)
                    .iter()
                    .map(|id| values[id].clone())
                    .collect();
            }
            NodeOp::CallFunction => {
                let input = &values[&n.args[0].as_node().expect("a tensor input")];
                match n.target.as_str() {
                    VIEW | VIEW_DEFAULT => reshape(input, &int_list(&n.args[1])),
                    PERMUTE | PERMUTE_DEFAULT => permute(input, &int_list(&n.args[1])),
                    RELU => Tensor {
                        shape: input.shape.clone(),
                        data: input.data.iter().map(|&value| value.max(0)).collect(),
                    },
                    ADD | ADD_INPLACE => {
                        let other = &values[&n.args[1].as_node().expect("a tensor operand")];
                        assert_eq!(input.shape, other.shape);
                        Tensor {
                            shape: input.shape.clone(),
                            data: input
                                .data
                                .iter()
                                .zip(&other.data)
                                .map(|(a, b)| a + b)
                                .collect(),
                        }
                    }
                    target => panic!("evaluate: no rule for {target}"),
                }
            }
            _ => continue,
        };
        values.insert(id, tensor);
    }
    panic!("evaluate: no output node");
}

// --- rc4 fixture --------------------------------------------------------------

/// The fixture's list syntax: `-` is empty, `_` no argument.
fn fixture_list(token: &str) -> Option<Vec<i64>> {
    match token {
        "_" => None,
        "-" => Some(Vec::new()),
        token => Some(
            token
                .split(',')
                .map(|value| value.parse().unwrap())
                .collect(),
        ),
    }
}

fn fixture_ints(values: &[i64]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn fixture_dtype(dtype: &str) -> &str {
    match dtype {
        "f32" => "torch.float32",
        "i8" => "torch.int8",
        "torch.float32" => "f32",
        "torch.int8" => "i8",
        other => panic!("unknown dtype {other}"),
    }
}

/// Builds one fixture case, runs the pass and returns what differs from rc4.
fn replay_case(case: &[Vec<&str>]) -> Vec<String> {
    let mut g = LayoutGraph::new();
    let mut ids: HashMap<String, NodeId> = HashMap::new();
    let mut lines = case.iter();
    let mut ops = 0;
    let mut problems = Vec::new();
    // The input graph, up to `expect`.
    let modified_in_rc4 = loop {
        let line = lines.next().expect("a case ends");
        match line[0] {
            "input" => {
                let shape = fixture_list(line[2]).unwrap();
                let id = g.typed_input(line[1], &shape, fixture_dtype(line[3]));
                ids.insert(line[1].to_string(), id);
            }
            "op" => {
                let input = ids[line[3]];
                let id = match (line[2], fixture_list(line[4])) {
                    ("view", Some(size)) => g.call(VIEW, vec![node(input), imported_ints(&size)]),
                    ("permute", Some(dims)) => {
                        g.call(PERMUTE, vec![node(input), imported_ints(&dims)])
                    }
                    ("relu", None) => g.call(RELU, vec![node(input)]),
                    other => panic!("bad op {other:?}"),
                };
                // A distinct zero point per node shows whose meta survives.
                let zp = line[1][1..].parse::<i64>().unwrap();
                g.graph_mut()
                    .node_mut(id)
                    .meta
                    .set_single_output_qparam(qparam(0.25, zp, "torch.int8"));
                ids.insert(line[1].to_string(), id);
                ops += 1;
            }
            "output" => {
                let outputs: Vec<NodeId> = line[1..].iter().map(|label| ids[*label]).collect();
                g.output(&outputs);
            }
            "expect" => break line[1] == "modified",
            other => panic!("bad line {other}"),
        }
    };
    assert!(ops > 0);
    let labels: HashMap<String, String> = ids
        .iter()
        .map(|(label, &id)| (g.name(id), label.clone()))
        .collect();

    let before = evaluate(g.graph());
    let modified = IrCanonicalizeViewCopyPermutePass.run(g.graph_mut());
    if modified != modified_in_rc4 {
        problems.push(format!("modified: RETAB {modified}, rc4 {modified_in_rc4}"));
    }
    if evaluate(g.graph()) != before {
        problems.push("RETAB changed the graph's outputs".to_string());
    }

    let graph = g.graph();
    let label = |id: NodeId| labels[&graph.node(id).name].clone();
    let mut actual = Vec::new();
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        if n.op != NodeOp::CallFunction {
            continue;
        }
        let (kind, args) = match n.target.as_str() {
            VIEW => ("view", fixture_ints(&int_list(&n.args[1]))),
            PERMUTE => ("permute", fixture_ints(&int_list(&n.args[1]))),
            RELU => ("relu", "_".to_string()),
            other => panic!("unexpected target {other}"),
        };
        actual.push(format!(
            "node {} {kind} {} {args} {} {}",
            label(id),
            label(n.args[0].as_node().unwrap()),
            fixture_ints(graph.node_shape(id).unwrap()),
            fixture_dtype(graph.node_dtype(id).unwrap()),
        ));
        let zp = n.meta.output_qparam().map(|qparam| qparam.zp);
        if Some(label(id)[1..].parse::<i64>().unwrap()) != zp {
            problems.push(format!("{} carries the qparams of n{zp:?}", label(id)));
        }
    }
    let outputs: Vec<String> = output_ids(graph).into_iter().map(label).collect();
    actual.push(format!("output {}", outputs.join(" ")));

    let expected: Vec<String> = lines
        .take_while(|line| line[0] != "end")
        .map(|line| line.join(" "))
        .collect();
    if actual != expected {
        problems.push(format!("rc4:   {expected:?}\nRETAB: {actual:?}"));
    }
    problems
}

/// Replays tests/fixtures/canonicalize_rc4_cases.txt: rc4's pass on the
/// upstream graphs, chains around the swap vectors and seeded random graphs
/// (see canonicalize_rc4_cases.py to regenerate).
#[test]
fn matches_rc4_fixture() {
    let mut cases: Vec<(String, Vec<Vec<&str>>)> = Vec::new();
    for line in FIXTURE.lines().filter(|line| !line.starts_with('#')) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens[0] == "case" {
            cases.push((tokens[1].to_string(), Vec::new()));
        } else {
            cases.last_mut().expect("a case header").1.push(tokens);
        }
    }
    assert!(cases.len() > 600, "{} cases", cases.len());
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(name, case)| {
            let problems = replay_case(case);
            (!problems.is_empty()).then(|| format!("case {name}:\n{}", problems.join("\n")))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from rc4:\n{}",
        failures.len(),
        cases.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// --- Upstream: test_canonicalize_view_copy_permute_pass.py --------------------

#[test]
fn upstream_direct_permute_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4, 5]);
    let p1 = g.permute(x, &[0, 2, 3, 1]);
    let p2 = g.permute(p1, &[0, 3, 1, 2]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(sig(&g), ["output(x)"]);
}

#[test]
fn upstream_singleton_permute_to_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 4]);
    let p = g.permute(x, &[0, 3, 1, 2]);
    g.output(&[p]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [1, 4, 1, 1]) -> [1, 4, 1, 1]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn upstream_keeps_equal_sized_non_singleton_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 1, 2]);
    let p = g.permute(x, &[2, 1, 0]);
    g.output(&[p]);

    assert!(!run(&mut g));

    g.assert_counts(1, 0);
}

#[test]
fn upstream_pixel_shuffle_view_permute_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 2, 8]);
    let p1 = g.permute(x, &[0, 3, 1, 2]);
    let v1 = g.view(p1, &[1, 2, 2, 2, 2, 2]);
    let p2 = g.permute(v1, &[0, 1, 4, 2, 5, 3]);
    let v2 = g.view(p2, &[1, 2, 4, 4]);
    let p3 = g.permute(v2, &[0, 2, 3, 1]);
    g.output(&[p3]);

    assert!(run(&mut g));

    // The first permute and the last one survive as the views.
    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [1, 2, 2, 2, 2, 2]) -> [1, 2, 2, 2, 2, 2]",
            "view_copy = permute(permute_copy, [0, 1, 4, 2, 5, 3]) -> [1, 2, 2, 2, 2, 2]",
            "permute_copy_2 = view(view_copy, [1, 4, 4, 2]) -> [1, 4, 4, 2]",
            "output(permute_copy_2)",
        ]
    );
}

#[test]
fn upstream_direct_output_axis_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 12, 5]);
    let v1 = g.view(x, &[2, 3, 4, 5]);
    let p1 = g.permute(v1, &[2, 0, 1, 3]);
    let v2 = g.view(p1, &[4, 6, 5]);
    g.output(&[v2]);

    assert!(run(&mut g));

    // (v1, p1) can't swap, so (p1, v2) does; then v1 and p1 fuse into v1.
    assert_eq!(
        sig(&g),
        [
            "view_copy = view(x, [6, 4, 5]) -> [6, 4, 5]",
            "view_copy_1 = permute(view_copy, [1, 0, 2]) -> [4, 6, 5]",
            "output(view_copy_1)",
        ]
    );
}

#[test]
fn upstream_reordered_view_with_grouped_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v1 = g.view(x, &[3, 2, 4]);
    let p1 = g.permute(v1, &[2, 0, 1]);
    let v2 = g.view(p1, &[4, 6]);
    g.output(&[v2]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = permute(x, [2, 0, 1]) -> [4, 2, 3]",
            "permute_copy = view(view_copy, [4, 6]) -> [4, 6]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn upstream_moves_view_before_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p1 = g.permute(x, &[2, 0, 1]);
    let v1 = g.view(p1, &[4, 6]);
    let p2 = g.permute(v1, &[1, 0]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [6, 4]) -> [6, 4]",
            "output(permute_copy)"
        ]
    );
}

#[test]
fn upstream_moves_permute_before_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 10, 10]);
    let v1 = g.view(x, &[1, 2, 5, 2, 5, 2]);
    let p1 = g.permute(v1, &[3, 5, 0, 1, 2, 4]);
    let v2 = g.view(p1, &[4, 2, 5, 5]);
    let p2 = g.permute(v2, &[0, 2, 3, 1]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = view(x, [1, 2, 5, 2, 5, 2]) -> [1, 2, 5, 2, 5, 2]",
            "permute_copy = permute(view_copy, [3, 5, 0, 2, 4, 1]) -> [2, 2, 1, 5, 5, 2]",
            "permute_copy_1 = view(permute_copy, [4, 5, 5, 2]) -> [4, 5, 5, 2]",
            "output(permute_copy_1)",
        ]
    );
}

#[test]
fn upstream_sinks_singleton_view_below_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 8, 64]);
    let v = g.view(x, &[2, 8, 1, 64]);
    let p = g.permute(v, &[0, 2, 3, 1]);
    g.output(&[p]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = permute(x, [0, 2, 1]) -> [2, 64, 8]",
            "permute_copy = view(view_copy, [2, 1, 64, 8]) -> [2, 1, 64, 8]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn upstream_follows_interleaved_chain_users() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[4, 2, 4]);
    let y = g.input("y", &[2, 3]);
    let p1 = g.permute(x, &[1, 0, 2]);
    let unrelated = g.view(y, &[3, 2]);
    let v1 = g.view(p1, &[1, 2, 4, 4]);
    let p2 = g.permute(v1, &[0, 2, 1, 3]);
    g.output(&[p2, unrelated]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [1, 4, 2, 4]) -> [1, 4, 2, 4]",
            "view_copy = view(y, [3, 2]) -> [3, 2]",
            "output(permute_copy, view_copy)",
        ]
    );
}

#[test]
fn upstream_reordered_view_rejects_separating_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v = g.view(x, &[3, 2, 4]);
    let p = g.permute(v, &[0, 2, 1]);
    g.output(&[p]);

    assert!(!run(&mut g));

    g.assert_counts(1, 1);
}

#[test]
fn upstream_view_permute_swap_uses_factor_order() {
    // rc4 calls _view_permute_swap directly; this view only inserts a unit
    // dim, so the lone pair sinks through the same swap.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4, 5, 6, 7, 8, 9]);
    let v = g.view(x, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let p = g.permute(v, &[0, 8, 7, 6, 5, 4, 3, 2, 1]);
    g.output(&[p]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = permute(x, [7, 6, 5, 4, 3, 2, 1, 0]) -> [9, 8, 7, 6, 5, 4, 3, 2]",
            "permute_copy = view(view_copy, [1, 9, 8, 7, 6, 5, 4, 3, 2]) -> [1, 9, 8, 7, 6, 5, 4, 3, 2]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn upstream_view_permute_swap_rejects_reordered_split_axis() {
    // rc4 calls _view_permute_swap directly. A lone pair would not try the
    // swap, so a flattening view follows; (p, flat) can't swap either.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[4]);
    let v = g.view(x, &[2, 2]);
    let p = g.permute(v, &[1, 0]);
    let flat = g.view(p, &[4]);
    g.output(&[flat]);

    assert!(!run(&mut g));

    g.assert_counts(1, 2);
}

#[test]
fn upstream_does_not_cross_multi_user_chain_node() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p1 = g.permute(x, &[0, 2, 1]);
    let p2 = g.permute(p1, &[0, 2, 1]);
    let v = g.view(p1, &[2, 12]);
    g.output(&[p2, v]);

    assert!(!run(&mut g));

    g.assert_counts(2, 1);
}

#[test]
fn upstream_unsupported_start_view_does_not_block_suffix() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v = g.view(x, &[3, 2]);
    let p1 = g.permute(v, &[1, 0]);
    let p2 = g.permute(p1, &[1, 0]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["view_copy = view(x, [3, 2]) -> [3, 2]", "output(view_copy)"]
    );
}

#[test]
fn upstream_unsupported_end_view_does_not_block_prefix() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p1 = g.permute(x, &[1, 0]);
    let p2 = g.permute(p1, &[1, 0]);
    let v = g.view(p2, &[3, 2]);
    g.output(&[v]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["view_copy = view(x, [3, 2]) -> [3, 2]", "output(view_copy)"]
    );
}

// --- Edge cases, each checked against the rc4 venv -----------------------------

#[test]
fn a_singleton_permute_becomes_a_view_before_it_can_fuse_with_the_next_permute() {
    // `_fuse_sequential_ops` checks the node at `index` before fusing it with
    // the next one, so the first permute is already a view by then.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 1, 3]);
    let p0 = g.permute(x, &[1, 0, 2]);
    let p1 = g.permute(p0, &[2, 0, 1]);
    g.output(&[p1]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [1, 2, 3]) -> [1, 2, 3]",
            "permute_copy_1 = permute(permute_copy, [2, 0, 1]) -> [3, 1, 2]",
            "output(permute_copy_1)",
        ]
    );
}

#[test]
fn fusing_permutes_reads_a_negative_next_dim_from_the_end() {
    // rc4 indexes the first permute's dims with the second's raw dims
    // (`:249`) before the second is normalized.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p0 = g.permute(x, &[1, 0, 2]);
    let p1 = g.permute(p0, &[-1, 0, 1]);
    g.output(&[p1]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = permute(x, [2, 1, 0]) -> [4, 3, 2]",
            "output(permute_copy)"
        ]
    );
}

#[test]
fn a_dead_chain_is_canonicalized_then_eliminated() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p0 = g.permute(x, &[1, 0]);
    let _dead = g.permute(p0, &[1, 0]);
    let relu = g.call(RELU, vec![node(x)]);
    g.output(&[relu]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["relu = aten.relu.default(x) -> [2, 3]", "output(relu)"]
    );
}

#[test]
fn dead_code_goes_only_when_the_pass_changed_something() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p = g.permute(x, &[1, 0]);
    let _dead = g.call(RELU, vec![node(x)]);
    g.output(&[p]);

    assert!(!run(&mut g));
    g.assert_targets(&[PERMUTE, RELU]);

    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v = g.view(x, &[2, 3]);
    let _dead = g.call(RELU, vec![node(x)]);
    g.output(&[v]);

    assert!(run(&mut g));
    assert_eq!(sig(&g), ["output(x)"]);
}

#[test]
fn an_identity_chain_feeding_the_output_twice_leaves_the_input() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p0 = g.permute(x, &[1, 0]);
    let p1 = g.permute(p0, &[1, 0]);
    g.output(&[p1, p1]);

    assert!(run(&mut g));

    assert_eq!(sig(&g), ["output(x, x)"]);
}

#[test]
fn a_user_reading_the_chain_end_twice_counts_once() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p0 = g.permute(x, &[1, 0]);
    let p1 = g.permute(p0, &[1, 0]);
    let add = g.call(ADD, vec![node(p1), node(p1)]);
    g.output(&[add]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["add = aten.add.Tensor(x, x) -> [2, 3]", "output(add)"]
    );
}

#[test]
fn removing_a_chain_head_rewires_the_other_chains_at_the_fork() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p0 = g.permute(x, &[0, 1]);
    let p1 = g.permute(p0, &[1, 0]);
    let relu = g.call(RELU, vec![node(p0)]);
    g.output(&[p1, relu]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy_1 = permute(x, [1, 0]) -> [3, 2]",
            "relu = aten.relu.default(x) -> [2, 3]",
            "output(permute_copy_1, relu)",
        ]
    );
}

#[test]
fn a_rank_0_input_sinks_through_an_empty_permute() {
    // The sink turns the view into permute(x, []), an identity that goes next.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[]);
    let v = g.view(x, &[1, 1]);
    let p = g.permute(v, &[1, 0]);
    g.output(&[p]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "permute_copy = view(x, [1, 1]) -> [1, 1]",
            "output(permute_copy)"
        ]
    );
}

#[test]
fn imported_empty_lists_behave_like_rc4s_empty_lists() {
    // RETAB's FX import gives `IrArg::None` for `[]`.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1]);
    let v0 = g.call(VIEW, vec![node(x), IrArg::None]);
    let v1 = g.view(v0, &[1]);
    g.output(&[v1]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["view_copy = view(x, [1]) -> [1]", "output(view_copy)"]
    );

    let mut g = LayoutGraph::new();
    let x = g.input("x", &[]);
    let p = g.call(PERMUTE, vec![node(x), IrArg::None]);
    let v = g.view(p, &[1]);
    g.output(&[v]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["view_copy = view(x, [1]) -> [1]", "output(view_copy)"]
    );
}

#[test]
fn a_view_with_an_inferred_dim_swaps_as_rc4_does() {
    // ViewMap reads the view's args (`-1` included), not its value.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v0 = g.view(x, &[-1, 4]);
    let p = g.permute(v0, &[1, 0]);
    let v1 = g.view(p, &[4, 6, 1]);
    g.output(&[v1]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = permute(x, [2, 0, 1]) -> [4, 2, 3]",
            "permute_copy = view(view_copy, [4, 6, 1]) -> [4, 6, 1]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn a_lone_pair_sinks_but_only_after_its_chain_shrinks_to_two() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v = g.view(x, &[2, 1, 3]);
    let p0 = g.permute(v, &[2, 1, 0]);
    let p1 = g.permute(p0, &[0, 2, 1]);
    g.output(&[p1]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        [
            "view_copy = permute(x, [1, 0]) -> [3, 2]",
            "permute_copy = view(view_copy, [3, 2, 1]) -> [3, 2, 1]",
            "output(permute_copy)",
        ]
    );
}

#[test]
fn permute_default_and_view_default_are_not_chain_members() {
    // Arm counts only permute_copy as a permute, and the base pass only
    // view_copy as a view.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p0 = g.permute(x, &[1, 0]);
    let pd = g.add_unrefreshed(PERMUTE_DEFAULT, vec![node(p0), ints(&[1, 0])]);
    g.graph_mut().node_mut(pd).meta = NodeMeta::tensor(vec![2, 3], "torch.float32");
    let p1 = g.permute(pd, &[1, 0]);
    g.output(&[p1]);

    assert!(!run(&mut g));

    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v0 = g.view(x, &[6]);
    let vd = g.call(VIEW_DEFAULT, vec![node(v0), ints(&[3, 2])]);
    let v1 = g.view(vd, &[6]);
    g.output(&[v1]);

    assert!(!run(&mut g));
}

#[test]
fn rewritten_nodes_keep_their_own_qparams_through_a_swap() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v1 = g.view(x, &[3, 2, 4]);
    let p = g.permute(v1, &[2, 0, 1]);
    let v2 = g.view(p, &[4, 6]);
    g.output(&[v2]);
    for (zp, id) in [v1, p, v2].into_iter().enumerate() {
        g.graph_mut()
            .node_mut(id)
            .meta
            .set_single_output_qparam(qparam(0.5, zp as i64, "torch.int8"));
    }
    let names = [v1, p].map(|id| g.name(id));

    assert!(run(&mut g));

    // v1 is now the permute and p the view; v2 fused into p.
    let [v1, p] = names.map(|name| g.id(&name));
    assert_eq!(g.graph().node(v1).target, PERMUTE);
    assert_eq!(g.graph().node(p).target, VIEW);
    assert_eq!(g.graph().node(v1).meta.output_qparam().unwrap().zp, 0);
    assert_eq!(g.graph().node(p).meta.output_qparam().unwrap().zp, 1);
}

// --- Swap enumerations ----------------------------------------------------------

/// Every shape of `numel` up to `max_rank`, unit dims included
/// (`_factorizations` in the upstream swap tests).
fn factorizations(numel: i64, max_rank: usize) -> Vec<Vec<i64>> {
    fn recurse(remaining: i64, rank: usize, shape: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
        if rank == 0 {
            if remaining == 1 {
                out.push(shape.clone());
            }
            return;
        }
        for dim in (1..=remaining).filter(|dim| remaining % dim == 0) {
            shape.push(dim);
            recurse(remaining / dim, rank - 1, shape, out);
            shape.pop();
        }
    }
    let mut out = Vec::new();
    for rank in 1..=max_rank {
        recurse(numel, rank, &mut Vec::new(), &mut out);
    }
    out
}

fn permutations(rank: usize) -> Vec<Vec<i64>> {
    if rank == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(rank - 1) {
        for at in 0..=rest.len() {
            let mut dims = rest.clone();
            dims.insert(at, rank as i64 - 1);
            out.push(dims);
        }
    }
    out
}

/// What rc4 leaves in a linear chain: no identity view or permute, no permute
/// that only moves unit dims, and no two adjacent views or permutes.
fn assert_canonical_chain(g: &LayoutGraph) {
    let graph = g.graph();
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        if n.op != NodeOp::CallFunction {
            continue;
        }
        let input = n.args[0].as_node().unwrap();
        let input_shape = graph.node_shape(input).unwrap();
        if n.target == VIEW {
            assert_ne!(
                input_shape,
                graph.node_shape(id).unwrap(),
                "identity view in {:?}",
                sig(g)
            );
        } else {
            // Unit dims in order would make it an identity, or a view.
            let kept: Vec<i64> = int_list(&n.args[1])
                .into_iter()
                .filter(|&axis| input_shape[axis as usize] != 1)
                .collect();
            assert!(
                !kept.is_sorted(),
                "permute that moves only unit dims in {:?}",
                sig(g)
            );
        }
        if graph.node(input).op == NodeOp::CallFunction {
            assert_ne!(
                graph.node(input).target,
                n.target,
                "adjacent pair in {:?}",
                sig(g)
            );
        }
    }
}

const SWAP_INPUT_SHAPES: [&[i64]; 8] = [
    &[2, 3],
    &[2, 4],
    &[2, 3, 4],
    &[2, 2, 3],
    &[1, 2, 3],
    &[2, 1, 3],
    &[2, 3, 1],
    &[2, 2, 2, 3],
];

#[test]
fn view_permute_chains_keep_their_numerics() {
    // test_view_permute_swap_generated_cases_are_semantically_valid, as
    // x -> view -> permute -> flatten so that the pass tries the swap.
    let (mut total, mut changed) = (0, 0);
    for shape in SWAP_INPUT_SHAPES {
        for view_shape in factorizations(numel(shape), 4) {
            for dims in permutations(view_shape.len()) {
                let mut g = LayoutGraph::new();
                let x = g.input("x", shape);
                let v = g.view(x, &view_shape);
                let p = g.permute(v, &dims);
                let flat = g.view(p, &[numel(shape)]);
                g.output(&[flat]);
                total += 1;
                changed += usize::from(run(&mut g));
                assert_canonical_chain(&g);
            }
        }
    }
    assert!(
        total > 1000 && changed > 200,
        "{changed} of {total} changed"
    );
}

#[test]
fn permute_view_chains_keep_their_numerics() {
    // test_permute_view_swap_generated_cases_are_semantically_valid, as
    // x -> permute -> view -> reversing permute.
    let (mut total, mut changed) = (0, 0);
    for shape in SWAP_INPUT_SHAPES {
        for dims in permutations(shape.len()) {
            for view_shape in factorizations(numel(shape), 4) {
                let mut g = LayoutGraph::new();
                let x = g.input("x", shape);
                let p0 = g.permute(x, &dims);
                let v = g.view(p0, &view_shape);
                let reverse: Vec<i64> = (0..view_shape.len() as i64).rev().collect();
                let p1 = g.permute(v, &reverse);
                g.output(&[p1]);
                total += 1;
                changed += usize::from(run(&mut g));
                assert_canonical_chain(&g);
            }
        }
    }
    assert!(
        total > 1000 && changed > 200,
        "{changed} of {total} changed"
    );
}

// --- Divergences ----------------------------------------------------------------

/// x[2,3,4] -> permute [2,0,1] -> view [4,6] -> permute [0,5], the last with
/// dims outside its input's rank and a forced value.
fn chain_with_out_of_range_permute() -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p = g.permute(x, &[2, 0, 1]);
    let v = g.view(p, &[4, 6]);
    let bad = g.add_unrefreshed(PERMUTE, vec![node(v), ints(&[0, 5])]);
    g.graph_mut().node_mut(bad).meta = NodeMeta::tensor(vec![4, 6], "torch.float32");
    g.output(&[bad]);
    g
}

#[test]
fn out_of_range_permute_dims_do_not_hang() {
    // rc4: AssertionError "Invalid dim 5 for rank 2" (dim_maps.py:42, from
    // canonicalize_view_copy_permute_pass.py:178-181) in the first
    // _fuse_sequential_ops. The port once skipped the bad permute and then
    // swapped (permute, view) back and forth forever; it now panics as rc4 raises.
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut g = chain_with_out_of_range_permute();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            IrCanonicalizeViewCopyPermutePass.run(g.graph_mut())
        }));
        let _ = sender.send(outcome.is_err());
    });
    let finished = receiver.recv_timeout(Duration::from_secs(10));
    assert_eq!(finished, Ok(true), "the pass must raise as rc4 does, not hang or return");
}

#[test]
fn out_of_range_permute_dims_raise_as_rc4_does() {
    // rc4 raises for both: AssertionError (dim_maps.py:42) for the lone permute,
    // IndexError at canonicalize_view_copy_permute_pass.py:249 for the pair.
    let lone = || {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let bad = g.add_unrefreshed(PERMUTE, vec![node(x), ints(&[0, 5])]);
        g.graph_mut().node_mut(bad).meta = NodeMeta::tensor(vec![2, 3], "torch.float32");
        g.output(&[bad]);
        g
    };
    let pair = || {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let p = g.permute(x, &[1, 0]);
        let bad = g.add_unrefreshed(PERMUTE, vec![node(p), ints(&[0, 5])]);
        g.graph_mut().node_mut(bad).meta = NodeMeta::tensor(vec![3, 2], "torch.float32");
        g.output(&[bad]);
        g
    };
    for (name, build) in [("lone", &lone as &dyn Fn() -> LayoutGraph), ("pair", &pair)] {
        let mut g = build();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            IrCanonicalizeViewCopyPermutePass.run(g.graph_mut())
        }));
        assert!(
            outcome.is_err(),
            "{name}: RETAB returned {outcome:?}; rc4 raises"
        );
    }
}

#[test]
fn dead_code_elimination_keeps_unused_impure_calls() {
    // rc4 (torch 2.15 Graph.eliminate_dead_code): node.is_impure() is true for
    // an edge aten.add_.Tensor, so the unused add_ stays, as it does in
    // LayoutCtx::eliminate_dead_code (ctx.rs is_impure_call).
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let y = g.input("y", &[2, 3]);
    let v = g.view(x, &[2, 3]);
    let _unused = g.call(ADD_INPLACE, vec![node(y), node(x)]);
    g.output(&[v]);

    assert!(run(&mut g));

    assert_eq!(
        sig(&g),
        ["add_ = aten.add_.Tensor(y, x) -> [2, 3]", "output(x)"]
    );
}
