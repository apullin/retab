//! rc4's tests of `CanonicalizeViewCopyPermutePass`, ported to
//! `IrCanonicalizeViewCopyPermutePass` (APU-135). From
//! `backends/arm/test/passes/`: all of `test_canonicalize_view_copy_permute_pass.py`,
//! `test_view_permute_swap.py` and `test_permute_view_swap.py`, and the tests in
//! `test_fuse_view_copy.py` and `test_remove_data_layout_noops.py` that exercise
//! this pass.
//!
//! Every expectation is what rc4 does, not a reading of the Python: each graph
//! here went through rc4's Arm pass in the v1.6.0-rc4 venv and was read just
//! before the pass's closing retrace (`canonicalize_view_copy_permute_pass.py:94-100`),
//! while its nodes were still the originals. Nodes carry the upstream tests'
//! variable names, so `p1 = view(x, [6, 4]) -> [6, 4]` says that the node built
//! as `p1` survives as a view of `x` with shape `[6, 4]`. Like upstream's
//! `_validate_numerics`, `run` checks that every output holds the same elements
//! in the same places after the pass.
//!
//! `_view_permute_swap` and `_permute_view_swap` are private in both ports, so
//! the swap tests run each helper's graph through the pass: as upstream's lone
//! pair, which the pass leaves alone unless the view only inserts unit dims, and
//! inside a three-op chain whose first pair the swap loop offers to that helper.
//!
//! Not ported:
//! - the symbolic-shape tests, since RETAB hands symbolic shapes to stock:
//!   `test_canonicalize_symbolic_pixel_shuffle_view_permute_chain`,
//!   `test_canonicalize_symbolic_singleton_permute_stays_permute`,
//!   `test_view_permute_swap_preserves_symbolic_dimensions` and
//!   `test_permute_view_swap_preserves_symbolic_dimensions`;
//! - `test_fuse_view_copy_transform_tosa_FP`, which runs only `FuseViewCopyTransform`;
//! - `test_data_layout_noop_cleanup_pipeline_order`, which checks where the pass
//!   manager schedules the pass; RETAB's pipeline does not run it yet.

use std::collections::HashMap;
use std::fmt::Write;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::layout::canonicalize::IrCanonicalizeViewCopyPermutePass;
use crate::layout::fuse_view_copy::IrFuseViewCopyTransformPass;
use crate::layout::testing::*;

/// `assert_run`'s expectation that the pass changes nothing.
const UNCHANGED: &[&str] = &[];

type Ints = &'static [i64];

/// Rendered graph lines, as `render` gives them.
type Lines = &'static [&'static str];

/// The input shapes of upstream's generated swap tests.
const GENERATED_INPUT_SHAPES: [&[i64]; 8] = [
    &[2, 3],
    &[2, 4],
    &[2, 3, 4],
    &[2, 2, 3],
    &[1, 2, 3],
    &[2, 1, 3],
    &[2, 3, 1],
    &[2, 2, 2, 3],
];

/// A call named as the upstream test names it.
fn call(g: &mut LayoutGraph, name: &str, target: &str, args: Vec<IrArg>) -> NodeId {
    let id = g
        .graph_mut()
        .add_call_function(name, target, args, NodeMeta::default());
    g.refresh(id);
    id
}

fn permute(g: &mut LayoutGraph, name: &str, input: NodeId, dims: &[i64]) -> NodeId {
    call(g, name, PERMUTE, vec![node(input), ints(dims)])
}

fn view(g: &mut LayoutGraph, name: &str, input: NodeId, size: &[i64]) -> NodeId {
    call(g, name, VIEW, vec![node(input), ints(size)])
}

/// `x -> view(size) -> permute(dims)`, the graph of `_view_permute_swap`'s
/// tests, and with `flatten` the chain that adds `flat = view([numel])`.
fn view_permute(input: &[i64], size: &[i64], dims: &[i64], flatten: bool) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", input);
    let v = view(&mut g, "view", x, size);
    let mut last = permute(&mut g, "permute", v, dims);
    if flatten {
        last = view(&mut g, "flat", last, &[numel(input)]);
    }
    g.output(&[last]);
    g
}

/// `x -> permute(dims) -> view(size)`, the graph of `_permute_view_swap`'s
/// tests, and with `reverse` the chain that adds `rev`, a permute reversing the
/// view's dims.
fn permute_view(input: &[i64], dims: &[i64], size: &[i64], reverse: bool) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", input);
    let p = permute(&mut g, "permute", x, dims);
    let mut last = view(&mut g, "view", p, size);
    if reverse {
        let reversed: Vec<i64> = (0..size.len() as i64).rev().collect();
        last = permute(&mut g, "rev", last, &reversed);
    }
    g.output(&[last]);
    g
}

/// Runs the pass and checks, as rc4's `_validate_numerics` does, that every
/// output holds the same elements in the same places.
fn run(g: &mut LayoutGraph) -> bool {
    let before = evaluate(g.graph());
    let modified = IrCanonicalizeViewCopyPermutePass.run(g.graph_mut());
    assert_eq!(
        evaluate(g.graph()),
        before,
        "the outputs changed: {:?}",
        render(g)
    );
    modified
}

/// Runs the pass on `g` and compares the result with `expected`, or with `g`
/// itself if `expected` is `UNCHANGED`.
fn assert_run(mut g: LayoutGraph, expected: &[&str], case: &str) {
    let before = render(&g);
    let modified = run(&mut g);
    if expected.is_empty() {
        assert!(!modified, "{case}: modified");
        assert_eq!(render(&g), before, "{case}");
    } else {
        assert!(modified, "{case}: not modified");
        assert_eq!(render(&g), expected, "{case}");
    }
}

/// The live calls in graph order as `name = op(args) -> shape`, then the output.
fn render(g: &LayoutGraph) -> Vec<String> {
    let graph = g.graph();
    graph
        .topological_live_node_ids()
        .into_iter()
        .filter_map(|id| {
            let node = graph.node(id);
            match node.op {
                NodeOp::CallFunction => Some(format!(
                    "{} = {}({}) -> {:?}",
                    node.name,
                    op_name(&node.target),
                    render_args(graph, &node.args),
                    g.shape(id)
                )),
                NodeOp::Output => match &node.args[0] {
                    IrArg::List(results) => {
                        Some(format!("output({})", render_args(graph, results)))
                    }
                    result => Some(format!("output({})", render_arg(graph, result))),
                },
                _ => None,
            }
        })
        .collect()
}

fn render_args(graph: &IrGraph, args: &[IrArg]) -> String {
    let args: Vec<String> = args.iter().map(|arg| render_arg(graph, arg)).collect();
    args.join(", ")
}

fn render_arg(graph: &IrGraph, arg: &IrArg) -> String {
    match arg {
        IrArg::Node(id) => graph.node(*id).name.clone(),
        IrArg::Int(value) => value.to_string(),
        IrArg::List(items) => format!("[{}]", render_args(graph, items)),
        other => format!("{other:?}"),
    }
}

/// `view`, `permute`, or the op segment of another target.
fn op_name(target: &str) -> &str {
    match target {
        VIEW => "view",
        PERMUTE => "permute",
        _ => target.rsplit('.').nth(1).unwrap_or(target),
    }
}

/// A tensor of distinct elements, enough to see where each one goes.
#[derive(Debug, Clone, PartialEq)]
struct Values {
    shape: Vec<i64>,
    data: Vec<i64>,
}

/// The graph's outputs when input `k`, in name order, holds
/// `k * 10^6 + 0, 1, 2, ...`. Each value's shape must be its node's meta shape.
fn evaluate(graph: &IrGraph) -> Vec<Values> {
    let mut inputs: Vec<&str> = graph
        .placeholders()
        .filter(|input| input.alive)
        .map(|input| input.name.as_str())
        .collect();
    inputs.sort_unstable();
    let mut values: HashMap<NodeId, Values> = HashMap::new();
    for id in graph.topological_live_node_ids() {
        let node = graph.node(id);
        let value = match node.op {
            NodeOp::Placeholder => {
                let shape = graph.node_shape_vec(id).expect("inputs have a shape");
                let index = inputs
                    .binary_search(&node.name.as_str())
                    .expect("a live input");
                let first = index as i64 * 1_000_000;
                Values {
                    data: (first..first + numel(&shape)).collect(),
                    shape,
                }
            }
            NodeOp::CallFunction => {
                let operand = |index: usize| {
                    let id = node.args[index].as_node().expect("a tensor operand");
                    &values[&id]
                };
                let int_list = || node.args[1].as_int_list().expect("an int list");
                match node.target.as_str() {
                    VIEW => reshaped(operand(0), &int_list()),
                    PERMUTE => permuted(operand(0), &int_list()),
                    ADD => summed(operand(0), operand(1)),
                    target => panic!("cannot evaluate {target}"),
                }
            }
            NodeOp::Output => {
                let IrArg::List(results) = &node.args[0] else {
                    panic!("the output holds a list");
                };
                return results
                    .iter()
                    .map(|result| values[&result.as_node().expect("a tensor")].clone())
                    .collect();
            }
            _ => panic!("cannot evaluate {}", node.name),
        };
        assert_eq!(
            Some(value.shape.as_slice()),
            graph.node_shape(id),
            "{}'s meta shape",
            node.name
        );
        values.insert(id, value);
    }
    panic!("the graph has no output")
}

fn reshaped(input: &Values, size: &[i64]) -> Values {
    assert_eq!(
        numel(size),
        numel(&input.shape),
        "view {:?} as {size:?}",
        input.shape
    );
    Values {
        shape: size.to_vec(),
        data: input.data.clone(),
    }
}

fn permuted(input: &Values, dims: &[i64]) -> Values {
    let rank = input.shape.len();
    let dims: Vec<usize> = dims
        .iter()
        .map(|&dim| if dim < 0 { dim + rank as i64 } else { dim } as usize)
        .collect();
    let shape: Vec<i64> = dims.iter().map(|&dim| input.shape[dim]).collect();
    let mut strides = vec![1; rank];
    for axis in (1..rank).rev() {
        strides[axis - 1] = strides[axis] * input.shape[axis];
    }
    let mut index = vec![0; rank];
    let mut data = Vec::with_capacity(input.data.len());
    for _ in 0..input.data.len() {
        let offset: i64 = index
            .iter()
            .zip(&dims)
            .map(|(&at, &dim)| at * strides[dim])
            .sum();
        data.push(input.data[offset as usize]);
        for axis in (0..rank).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Values { shape, data }
}

fn summed(lhs: &Values, rhs: &Values) -> Values {
    assert_eq!(lhs.shape, rhs.shape);
    Values {
        shape: lhs.shape.clone(),
        data: lhs.data.iter().zip(&rhs.data).map(|(a, b)| a + b).collect(),
    }
}

fn numel(shape: &[i64]) -> i64 {
    shape.iter().product()
}

/// `_factorizations`: every shape of rank 1 to `max_rank` with `numel`
/// elements, in upstream's order.
fn factorizations(numel: i64, max_rank: usize) -> Vec<Vec<i64>> {
    fn extend(remaining: i64, rank: usize, shape: &mut Vec<i64>, shapes: &mut Vec<Vec<i64>>) {
        if rank == 0 {
            if remaining == 1 {
                shapes.push(shape.clone());
            }
            return;
        }
        for dim in (1..=remaining).filter(|dim| remaining % dim == 0) {
            shape.push(dim);
            extend(remaining / dim, rank - 1, shape, shapes);
            shape.pop();
        }
    }
    let mut shapes = Vec::new();
    for rank in 1..=max_rank {
        extend(numel, rank, &mut Vec::new(), &mut shapes);
    }
    shapes
}

/// `itertools.permutations(range(rank))`, in its lexicographic order.
fn permutations(rank: usize) -> Vec<Vec<i64>> {
    if rank == 0 {
        return vec![Vec::new()];
    }
    let mut all = Vec::new();
    for first in 0..rank as i64 {
        for rest in permutations(rank - 1) {
            let mut permutation = vec![first];
            permutation.extend(rest.into_iter().map(|dim| dim + i64::from(dim >= first)));
            all.push(permutation);
        }
    }
    all
}

/// FNV-1a 64.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// `test_canonicalize_direct_permute_chain`: the permutes compose to the
/// identity, so both go.
#[test]
fn direct_permute_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4, 5]);
    let p1 = permute(&mut g, "p1", x, &[0, 2, 3, 1]);
    let p2 = permute(&mut g, "p2", p1, &[0, 3, 1, 2]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(render(&g), ["output(x)"]);
}

/// `test_canonicalize_singleton_permute_to_view`: a permute that only moves
/// unit dims becomes a view.
#[test]
fn singleton_permute_to_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 4]);
    let p = permute(&mut g, "permute", x, &[0, 3, 1, 2]);
    g.output(&[p]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "permute = view(x, [1, 4, 1, 1]) -> [1, 4, 1, 1]",
            "output(permute)"
        ]
    );
}

/// `test_canonicalize_keeps_equal_sized_non_singleton_permute`: swapping the
/// two dims of size 2 moves data.
#[test]
fn keeps_equal_sized_non_singleton_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 1, 2]);
    let p = permute(&mut g, "permute", x, &[2, 1, 0]);
    g.output(&[p]);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "permute = permute(x, [2, 1, 0]) -> [2, 1, 2]",
            "output(permute)"
        ]
    );
}

/// `test_canonicalize_pixel_shuffle_view_permute_chain`.
#[test]
fn pixel_shuffle_view_permute_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 2, 8]);
    let p1 = permute(&mut g, "p1", x, &[0, 3, 1, 2]);
    let v1 = view(&mut g, "v1", p1, &[1, 2, 2, 2, 2, 2]);
    let p2 = permute(&mut g, "p2", v1, &[0, 1, 4, 2, 5, 3]);
    let v2 = view(&mut g, "v2", p2, &[1, 2, 4, 4]);
    let p3 = permute(&mut g, "p3", v2, &[0, 2, 3, 1]);
    g.output(&[p3]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "p1 = view(x, [1, 2, 2, 2, 2, 2]) -> [1, 2, 2, 2, 2, 2]",
            "v1 = permute(p1, [0, 1, 4, 2, 5, 3]) -> [1, 2, 2, 2, 2, 2]",
            "p3 = view(v1, [1, 4, 4, 2]) -> [1, 4, 4, 2]",
            "output(p3)",
        ]
    );
}

/// `test_canonicalize_direct_output_axis_permute`.
#[test]
fn direct_output_axis_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 12, 5]);
    let v1 = view(&mut g, "v1", x, &[2, 3, 4, 5]);
    let p1 = permute(&mut g, "p1", v1, &[2, 0, 1, 3]);
    let v2 = view(&mut g, "v2", p1, &[4, 6, 5]);
    g.output(&[v2]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "v1 = view(x, [6, 4, 5]) -> [6, 4, 5]",
            "v2 = permute(v1, [1, 0, 2]) -> [4, 6, 5]",
            "output(v2)",
        ]
    );
}

/// `test_canonicalize_reordered_view_with_grouped_permute`.
#[test]
fn reordered_view_with_grouped_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v1 = view(&mut g, "v1", x, &[3, 2, 4]);
    let p1 = permute(&mut g, "p1", v1, &[2, 0, 1]);
    let v2 = view(&mut g, "v2", p1, &[4, 6]);
    g.output(&[v2]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "v1 = permute(x, [2, 0, 1]) -> [4, 2, 3]",
            "p1 = view(v1, [4, 6]) -> [4, 6]",
            "output(p1)",
        ]
    );
}

/// `test_canonicalize_moves_view_before_permute`.
#[test]
fn moves_view_before_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p1 = permute(&mut g, "p1", x, &[2, 0, 1]);
    let v1 = view(&mut g, "v1", p1, &[4, 6]);
    let p2 = permute(&mut g, "p2", v1, &[1, 0]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(render(&g), ["p1 = view(x, [6, 4]) -> [6, 4]", "output(p1)"]);
}

/// `test_canonicalize_moves_permute_before_view`.
#[test]
fn moves_permute_before_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 10, 10]);
    let v1 = view(&mut g, "v1", x, &[1, 2, 5, 2, 5, 2]);
    let p1 = permute(&mut g, "p1", v1, &[3, 5, 0, 1, 2, 4]);
    let v2 = view(&mut g, "v2", p1, &[4, 2, 5, 5]);
    let p2 = permute(&mut g, "p2", v2, &[0, 2, 3, 1]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "v1 = view(x, [1, 2, 5, 2, 5, 2]) -> [1, 2, 5, 2, 5, 2]",
            "p1 = permute(v1, [3, 5, 0, 2, 4, 1]) -> [2, 2, 1, 5, 5, 2]",
            "p2 = view(p1, [4, 5, 5, 2]) -> [4, 5, 5, 2]",
            "output(p2)",
        ]
    );
}

/// `test_canonicalize_sinks_singleton_view_below_permute`: the nodes swap
/// roles, so the permute runs at the lower rank.
#[test]
fn sinks_singleton_view_below_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 8, 64]);
    let v1 = view(&mut g, "v1", x, &[2, 8, 1, 64]);
    let p1 = permute(&mut g, "p1", v1, &[0, 2, 3, 1]);
    g.output(&[p1]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "v1 = permute(x, [0, 2, 1]) -> [2, 64, 8]",
            "p1 = view(v1, [2, 1, 64, 8]) -> [2, 1, 64, 8]",
            "output(p1)",
        ]
    );
}

/// `test_canonicalize_follows_interleaved_chain_users`: a view of another
/// input sits between `p1` and its user, and the chain still runs on.
#[test]
fn follows_interleaved_chain_users() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[4, 2, 4]);
    let y = g.input("y", &[2, 3]);
    let p1 = permute(&mut g, "p1", x, &[1, 0, 2]);
    let unrelated = view(&mut g, "unrelated", y, &[3, 2]);
    let v1 = view(&mut g, "v1", p1, &[1, 2, 4, 4]);
    let p2 = permute(&mut g, "p2", v1, &[0, 2, 1, 3]);
    g.output(&[p2, unrelated]);

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "p1 = view(x, [1, 4, 2, 4]) -> [1, 4, 2, 4]",
            "unrelated = view(y, [3, 2]) -> [3, 2]",
            "output(p1, unrelated)",
        ]
    );
}

/// `test_canonicalize_reordered_view_rejects_separating_permute`.
#[test]
fn reordered_view_rejects_separating_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v1 = view(&mut g, "v1", x, &[3, 2, 4]);
    let p1 = permute(&mut g, "p1", v1, &[0, 2, 1]);
    g.output(&[p1]);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "v1 = view(x, [3, 2, 4]) -> [3, 2, 4]",
            "p1 = permute(v1, [0, 2, 1]) -> [3, 4, 2]",
            "output(p1)",
        ]
    );
}

/// `test_canonicalize_view_permute_swap_uses_factor_order`, through the pass:
/// the view only inserts a unit dim, so the lone pair sinks it, and the nodes
/// take exactly the args upstream expects of `_view_permute_swap`,
/// `([7, 6, 5, 4, 3, 2, 1, 0], [1, 9, 8, 7, 6, 5, 4, 3, 2])`.
#[test]
fn view_permute_swap_uses_factor_order() {
    let mut g = view_permute(
        &[2, 3, 4, 5, 6, 7, 8, 9],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[0, 8, 7, 6, 5, 4, 3, 2, 1],
        false,
    );

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "view = permute(x, [7, 6, 5, 4, 3, 2, 1, 0]) -> [9, 8, 7, 6, 5, 4, 3, 2]",
            "permute = view(view, [1, 9, 8, 7, 6, 5, 4, 3, 2]) -> [1, 9, 8, 7, 6, 5, 4, 3, 2]",
            "output(permute)",
        ]
    );
}

/// `test_canonicalize_view_permute_swap_rejects_reordered_split_axis`, through
/// the pass. The lone pair stays. In the chain the swap loop offers the view
/// and the permute to `_view_permute_swap`, which rejects them as upstream
/// expects, then the permute and `flat` to `_permute_view_swap`, which rejects
/// them too.
#[test]
fn view_permute_swap_rejects_reordered_split_axis() {
    let mut g = view_permute(&[4], &[2, 2], &[1, 0], false);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "view = view(x, [2, 2]) -> [2, 2]",
            "permute = permute(view, [1, 0]) -> [2, 2]",
            "output(permute)",
        ]
    );

    let mut g = view_permute(&[4], &[2, 2], &[1, 0], true);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "view = view(x, [2, 2]) -> [2, 2]",
            "permute = permute(view, [1, 0]) -> [2, 2]",
            "flat = view(permute, [4]) -> [4]",
            "output(flat)",
        ]
    );
}

/// `test_canonicalize_does_not_cross_multi_user_chain_node`: `p1` has two
/// users, so `p1` and `p2` are separate chains and do not fuse.
#[test]
fn does_not_cross_multi_user_chain_node() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p1 = permute(&mut g, "p1", x, &[0, 2, 1]);
    let p2 = permute(&mut g, "p2", p1, &[0, 2, 1]);
    let v1 = view(&mut g, "v1", p1, &[2, 12]);
    g.output(&[p2, v1]);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "p1 = permute(x, [0, 2, 1]) -> [2, 4, 3]",
            "p2 = permute(p1, [0, 2, 1]) -> [2, 3, 4]",
            "v1 = view(p1, [2, 12]) -> [2, 12]",
            "output(p2, v1)",
        ]
    );
}

/// `test_canonicalize_unsupported_start_view_does_not_block_suffix`.
#[test]
fn unsupported_start_view_does_not_block_suffix() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let v1 = view(&mut g, "v1", x, &[3, 2]);
    let p1 = permute(&mut g, "p1", v1, &[1, 0]);
    let p2 = permute(&mut g, "p2", p1, &[1, 0]);
    g.output(&[p2]);

    assert!(run(&mut g));

    assert_eq!(render(&g), ["v1 = view(x, [3, 2]) -> [3, 2]", "output(v1)"]);
}

/// `test_canonicalize_unsupported_end_view_does_not_block_prefix`.
#[test]
fn unsupported_end_view_does_not_block_prefix() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let p1 = permute(&mut g, "p1", x, &[1, 0]);
    let p2 = permute(&mut g, "p2", p1, &[1, 0]);
    let v1 = view(&mut g, "v1", p2, &[3, 2]);
    g.output(&[v1]);

    assert!(run(&mut g));

    assert_eq!(render(&g), ["v1 = view(x, [3, 2]) -> [3, 2]", "output(v1)"]);
}

/// Input, view, dims, upstream's expected `(P', S')`, then the lone pair after
/// the pass.
type ViewPermuteRow = (Ints, Ints, Ints, (Ints, Ints), Lines);

/// `test_view_permute_swap_expected_rewrites`'s rows.
#[rustfmt::skip]
const VIEW_PERMUTE_EXPECTED: &[ViewPermuteRow] = &[
    (&[2, 3, 4], &[2, 3, 2, 2], &[0, 2, 3, 1], (&[0, 2, 1], &[2, 2, 2, 3]), UNCHANGED),
    (&[2, 3, 4], &[2, 3, 2, 2], &[2, 3, 0, 1], (&[2, 0, 1], &[2, 2, 2, 3]), UNCHANGED),
    (&[2, 3, 4], &[2, 12], &[1, 0], (&[1, 2, 0], &[12, 2]), UNCHANGED),
    (&[2, 3, 4], &[6, 4], &[1, 0], (&[2, 0, 1], &[4, 6]), UNCHANGED),
    (&[2, 3, 4], &[6, 4], &[-1, 0], (&[2, 0, 1], &[4, 6]), &[
        "view = view(x, [6, 4]) -> [6, 4]",
        "permute = permute(view, [1, 0]) -> [4, 6]",
        "output(permute)",
    ]),
    (&[2, 3, 4], &[1, 6, 4], &[0, 2, 1], (&[2, 0, 1], &[1, 4, 6]), UNCHANGED),
    (&[2, 3, 4], &[1, 6, 4], &[2, 0, 1], (&[2, 0, 1], &[4, 1, 6]), UNCHANGED),
    (&[2, 3, 4], &[1, 6, 4], &[2, 1, 0], (&[2, 0, 1], &[4, 6, 1]), UNCHANGED),
    (&[2, 3, 4], &[2, 3, 2, 2], &[1, 2, 3, 0], (&[1, 2, 0], &[3, 2, 2, 2]), UNCHANGED),
    (&[2, 3, 4, 5], &[2, 3, 20], &[0, 2, 1], (&[0, 2, 3, 1], &[2, 20, 3]), UNCHANGED),
    (&[2, 3, 4, 5], &[2, 3, 20], &[2, 0, 1], (&[2, 3, 0, 1], &[20, 2, 3]), UNCHANGED),
    (&[2, 3, 4, 5], &[6, 4, 5], &[1, 2, 0], (&[2, 3, 0, 1], &[4, 5, 6]), UNCHANGED),
    (&[2, 3, 4, 5], &[24, 5], &[1, 0], (&[3, 0, 1, 2], &[5, 24]), UNCHANGED),
    (&[1, 2, 3, 4], &[2, 12], &[1, 0], (&[0, 2, 3, 1], &[12, 2]), UNCHANGED),
    (&[2, 1, 3, 4], &[6, 4], &[1, 0], (&[1, 3, 0, 2], &[4, 6]), UNCHANGED),
    (&[2, 3, 1, 4], &[2, 3, 2, 2], &[0, 2, 3, 1], (&[0, 2, 3, 1], &[2, 2, 2, 3]), UNCHANGED),
    (&[2, 2, 3], &[2, 2, 3], &[2, 0, 1], (&[2, 0, 1], &[3, 2, 2]), &[
        "permute = permute(x, [2, 0, 1]) -> [3, 2, 2]",
        "output(permute)",
    ]),
    (&[2, 2, 3], &[2, 2, 3], &[1, 2, 0], (&[1, 2, 0], &[2, 3, 2]), &[
        "permute = permute(x, [1, 2, 0]) -> [2, 3, 2]",
        "output(permute)",
    ]),
    (&[2, 2, 2, 3], &[8, 3], &[1, 0], (&[3, 0, 1, 2], &[3, 8]), UNCHANGED),
    (&[2, 2, 2, 3], &[2, 2, 2, 3], &[3, 0, 1, 2], (&[3, 0, 1, 2], &[3, 2, 2, 2]), &[
        "permute = permute(x, [3, 0, 1, 2]) -> [3, 2, 2, 2]",
        "output(permute)",
    ]),
    (
        &[2, 3, 4, 5, 6, 7, 8, 9],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[0, 8, 7, 6, 5, 4, 3, 2, 1],
        (&[7, 6, 5, 4, 3, 2, 1, 0], &[1, 9, 8, 7, 6, 5, 4, 3, 2]),
        &[
            "view = permute(x, [7, 6, 5, 4, 3, 2, 1, 0]) -> [9, 8, 7, 6, 5, 4, 3, 2]",
            "permute = view(view, [1, 9, 8, 7, 6, 5, 4, 3, 2]) -> [1, 9, 8, 7, 6, 5, 4, 3, 2]",
            "output(permute)",
        ],
    ),
];

/// `test_view_permute_swap_expected_rewrites`, through the pass.
///
/// In the chain `view -> permute -> flat` the swap loop first offers the view
/// and the permute to `_view_permute_swap`, so the view node becomes
/// `permute(x, P')` with upstream's expected `P'`, and the permute node becomes
/// the view `S'`, which then fuses with `flat`. An identity view goes before any
/// swap. The lone pair, upstream's graph, stays as it is, except that the pass
/// normalizes negative dims, removes an identity view, and sinks a view that
/// only inserts unit dims, with upstream's expected args.
#[test]
fn view_permute_swap_expected_rewrites() {
    for &(input, size, dims, (swapped_dims, swapped_size), pair) in VIEW_PERMUTE_EXPECTED {
        let case = format!("{input:?} {size:?} {dims:?}");
        let n = numel(input);
        let mut g = view_permute(input, size, dims, true);

        assert!(run(&mut g), "{case}");

        let expected = if size == input {
            [
                format!("permute = permute(x, {dims:?}) -> {swapped_size:?}"),
                format!("flat = view(permute, [{n}]) -> [{n}]"),
                "output(flat)".to_string(),
            ]
        } else {
            let shape: Vec<i64> = swapped_dims
                .iter()
                .map(|&dim| input[dim as usize])
                .collect();
            [
                format!("view = permute(x, {swapped_dims:?}) -> {shape:?}"),
                format!("permute = view(view, [{n}]) -> [{n}]"),
                "output(permute)".to_string(),
            ]
        };
        assert_eq!(render(&g), expected, "{case}");
        assert_run(view_permute(input, size, dims, false), pair, &case);
    }
}

/// `test_view_permute_swap_rejects_unsupported_rewrites`, through the pass: in
/// the chain the swap loop offers the view and the permute to
/// `_view_permute_swap`, then the permute and `flat` to `_permute_view_swap`,
/// and both reject them, so neither the chain nor the lone pair changes.
#[test]
fn view_permute_swap_rejects_unsupported_rewrites() {
    let cases: [(&[i64], &[i64], &[i64]); 2] = [
        (&[2, 3, 4], &[2, 4, 3], &[0, 2, 1]),
        (&[4], &[2, 2], &[1, 0]),
    ];
    for (input, size, dims) in cases {
        for flatten in [true, false] {
            let case = format!("{input:?} {size:?} {dims:?} flatten={flatten}");
            assert_run(view_permute(input, size, dims, flatten), UNCHANGED, &case);
        }
    }
}

/// `test_view_permute_swap_generated_cases_are_semantically_valid`, through the
/// pass: every generated (input, view, dims), as the lone pair and as the chain
/// `view -> permute -> flat`. `run` checks each rewrite, as upstream checks each
/// swap. The 15304 outcomes are rc4's: each run gives the line
/// `<pair|chain><input><view><dims>: ` followed by `render` joined by `; `, and
/// the digest is FNV-1a 64 of those newline-terminated lines from rc4's pass on
/// the same graphs in the same order.
#[test]
fn view_permute_swap_generated_cases_match_rc4() {
    let mut lines = String::new();
    let mut modified = 0;
    for input in GENERATED_INPUT_SHAPES {
        for size in factorizations(numel(input), 4) {
            for dims in permutations(size.len()) {
                for (form, flatten) in [("pair", false), ("chain", true)] {
                    let mut g = view_permute(input, &size, &dims, flatten);
                    modified += usize::from(run(&mut g));
                    let outcome = render(&g).join("; ");
                    writeln!(lines, "{form}{input:?}{size:?}{dims:?}: {outcome}").unwrap();
                }
            }
        }
    }

    assert_eq!((lines.lines().count(), modified), (15304, 9130));
    assert_eq!(fnv1a(lines.as_bytes()), 0xd1f3_3457_1a06_5a68);
}

/// `test_permute_view_swap_expected_rewrites`'s rows, (input, dims, view), each
/// with the chain and then the lone pair after the pass.
#[rustfmt::skip]
const PERMUTE_VIEW_EXPECTED: &[(Ints, Ints, Ints, Lines, Lines)] = &[
    (&[2, 3, 4], &[2, 0, 1], &[4, 6], &["permute = view(x, [6, 4]) -> [6, 4]", "output(permute)"], UNCHANGED),
    (&[2, 3, 4], &[1, 2, 0], &[12, 2], &["permute = view(x, [2, 12]) -> [2, 12]", "output(permute)"], UNCHANGED),
    (&[2, 3, 4], &[0, 2, 1], &[2, 2, 2, 3], &[
        "permute = view(x, [2, 3, 2, 2]) -> [2, 3, 2, 2]",
        "view = permute(permute, [1, 3, 2, 0]) -> [3, 2, 2, 2]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4], &[-1, 0, 1], &[4, 6], &["permute = view(x, [6, 4]) -> [6, 4]", "output(permute)"], &[
        "permute = permute(x, [2, 0, 1]) -> [4, 2, 3]",
        "view = view(permute, [4, 6]) -> [4, 6]",
        "output(view)",
    ]),
    (&[2, 3, 4], &[2, 0, 1], &[1, 4, 6], &["permute = view(x, [6, 4, 1]) -> [6, 4, 1]", "output(permute)"], UNCHANGED),
    (&[2, 3, 4], &[2, 0, 1], &[4, 1, 6], &["permute = view(x, [6, 1, 4]) -> [6, 1, 4]", "output(permute)"], UNCHANGED),
    (&[2, 3, 4], &[2, 0, 1], &[4, 6, 1], &["permute = view(x, [1, 6, 4]) -> [1, 6, 4]", "output(permute)"], UNCHANGED),
    (&[2, 3, 4], &[2, 0, 1], &[2, 2, 2, 3], &[
        "permute = view(x, [2, 3, 2, 2]) -> [2, 3, 2, 2]",
        "view = permute(permute, [1, 0, 3, 2]) -> [3, 2, 2, 2]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4], &[1, 2, 0], &[3, 2, 2, 2], &[
        "permute = view(x, [2, 3, 2, 2]) -> [2, 3, 2, 2]",
        "view = permute(permute, [0, 3, 2, 1]) -> [2, 2, 2, 3]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4, 5], &[0, 2, 3, 1], &[2, 20, 3], &[
        "permute = view(x, [2, 3, 20]) -> [2, 3, 20]",
        "view = permute(permute, [1, 2, 0]) -> [3, 20, 2]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4, 5], &[2, 3, 0, 1], &[20, 2, 3], &[
        "permute = view(x, [2, 3, 20]) -> [2, 3, 20]",
        "view = permute(permute, [1, 0, 2]) -> [3, 2, 20]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4, 5], &[2, 3, 0, 1], &[4, 5, 6], &[
        "permute = view(x, [6, 4, 5]) -> [6, 4, 5]",
        "view = permute(permute, [0, 2, 1]) -> [6, 5, 4]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 3, 4, 5], &[3, 0, 1, 2], &[5, 24], &["permute = view(x, [24, 5]) -> [24, 5]", "output(permute)"], UNCHANGED),
    (&[1, 2, 3, 4], &[2, 3, 0, 1], &[12, 2], &["permute = view(x, [2, 12]) -> [2, 12]", "output(permute)"], UNCHANGED),
    (&[2, 1, 3, 4], &[1, 3, 0, 2], &[4, 6], &["permute = view(x, [6, 4]) -> [6, 4]", "output(permute)"], UNCHANGED),
    (&[2, 3, 1, 4], &[2, 0, 3, 1], &[2, 2, 2, 3], &[
        "permute = view(x, [2, 3, 2, 2]) -> [2, 3, 2, 2]",
        "view = permute(permute, [1, 3, 2, 0]) -> [3, 2, 2, 2]",
        "output(view)",
    ], UNCHANGED),
    (&[2, 2, 3], &[2, 0, 1], &[3, 2, 2],
        &["permute = permute(x, [1, 0, 2]) -> [2, 2, 3]", "output(permute)"],
        &["permute = permute(x, [2, 0, 1]) -> [3, 2, 2]", "output(permute)"]),
    (&[2, 2, 3], &[1, 2, 0], &[2, 3, 2],
        &["permute = permute(x, [0, 2, 1]) -> [2, 3, 2]", "output(permute)"],
        &["permute = permute(x, [1, 2, 0]) -> [2, 3, 2]", "output(permute)"]),
    (&[2, 2, 2, 3], &[3, 0, 1, 2], &[3, 8], &["permute = view(x, [8, 3]) -> [8, 3]", "output(permute)"], UNCHANGED),
    (&[2, 2, 2, 3], &[3, 0, 1, 2], &[3, 2, 2, 2],
        &["permute = permute(x, [2, 1, 0, 3]) -> [2, 2, 2, 3]", "output(permute)"],
        &["permute = permute(x, [3, 0, 1, 2]) -> [3, 2, 2, 2]", "output(permute)"]),
];

/// `test_permute_view_swap_expected_rewrites`, through the pass.
///
/// In the chain `permute -> view -> rev` the swap loop first offers the permute
/// and the view to `_permute_view_swap`, so the permute node becomes the view
/// `S'`, and the view node becomes the permute `P'`, which then fuses with
/// `rev`. Where the fused permute is an identity it goes, and where it only
/// moves unit dims it becomes a view and fuses into `S'`. An identity view goes
/// before any swap. The lone pair, upstream's graph, stays as it is, except that
/// the pass normalizes negative dims and removes an identity view.
#[test]
fn permute_view_swap_expected_rewrites() {
    for &(input, dims, size, chain, pair) in PERMUTE_VIEW_EXPECTED {
        let case = format!("{input:?} {dims:?} {size:?}");
        assert_run(permute_view(input, dims, size, true), chain, &case);
        assert_run(permute_view(input, dims, size, false), pair, &case);
    }
}

/// `test_permute_view_swap_rejects_unsupported_rewrites`'s rows but the last,
/// (input, dims, view), each with the chain after the pass.
#[rustfmt::skip]
const PERMUTE_VIEW_REJECTED: &[(Ints, Ints, Ints, Lines)] = &[
    (&[2, 3, 4], &[1, 0, 2], &[3, 8], &[
        "permute = permute(x, [0, 2, 1]) -> [2, 4, 3]",
        "rev = view(permute, [8, 3]) -> [8, 3]",
        "output(rev)",
    ]),
    (&[2, 3, 4], &[1, 0, 2], &[6, 4], &[
        "permute = permute(x, [2, 1, 0]) -> [4, 3, 2]",
        "rev = view(permute, [4, 6]) -> [4, 6]",
        "output(rev)",
    ]),
    (&[2, 3, 4], &[2, 1, 0], &[4, 6], &[
        "permute = permute(x, [1, 0, 2]) -> [3, 2, 4]",
        "rev = view(permute, [6, 4]) -> [6, 4]",
        "output(rev)",
    ]),
    (&[2, 3, 4], &[0, 2, 1], &[8, 3], &[
        "permute = permute(x, [1, 0, 2]) -> [3, 2, 4]",
        "rev = view(permute, [3, 8]) -> [3, 8]",
        "output(rev)",
    ]),
    (&[2, 3, 4, 5], &[3, 0, 1, 2], &[10, 12], &[
        "permute = permute(x, [1, 2, 3, 0]) -> [3, 4, 5, 2]",
        "rev = view(permute, [12, 10]) -> [12, 10]",
        "output(rev)",
    ]),
    (&[2, 3, 4, 5], &[1, 3, 0, 2], &[15, 8], &[
        "permute = permute(x, [0, 2, 1, 3]) -> [2, 4, 3, 5]",
        "rev = view(permute, [8, 15]) -> [8, 15]",
        "output(rev)",
    ]),
];

/// `test_permute_view_swap_rejects_unsupported_rewrites`, through the pass. In
/// the chain the swap loop offers the permute and the view to
/// `_permute_view_swap`, which rejects them as upstream expects, so the permute
/// node stays a permute; the loop then swaps the view and `rev` instead, and
/// the two permutes fuse. The lone pairs stay.
#[test]
fn permute_view_swap_rejects_unsupported_rewrites() {
    for &(input, dims, size, chain) in PERMUTE_VIEW_REJECTED {
        let case = format!("{input:?} {dims:?} {size:?}");
        assert_run(permute_view(input, dims, size, true), chain, &case);
        assert_run(permute_view(input, dims, size, false), UNCHANGED, &case);
    }

    // Upstream's last case views 24 elements as [5, 5]. As upstream does, the
    // view's value is set by hand, since no shape rule accepts it; the pass
    // cannot map the view, so nothing changes. The evaluator cannot run it.
    for reverse in [true, false] {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let p = permute(&mut g, "permute", x, &[2, 0, 1]);
        let mut last = g.graph_mut().add_call_function(
            "view",
            VIEW,
            vec![node(p), ints(&[5, 5])],
            NodeMeta::tensor(vec![5, 5], "torch.float32"),
        );
        if reverse {
            last = permute(&mut g, "rev", last, &[1, 0]);
        }
        g.output(&[last]);
        let before = render(&g);

        assert!(
            !IrCanonicalizeViewCopyPermutePass.run(g.graph_mut()),
            "reverse={reverse}"
        );

        assert_eq!(render(&g), before, "reverse={reverse}");
    }
}

/// `test_permute_view_swap_generated_cases_are_semantically_valid`, through the
/// pass: every generated (input, dims, view), as the lone pair and as the chain
/// `permute -> view -> rev`. `run` checks each rewrite, as upstream checks each
/// swap. The 9260 outcomes are rc4's, digested as in
/// `view_permute_swap_generated_cases_match_rc4` from the lines
/// `<pair|chain><input><dims><view>: ` followed by `render` joined by `; `.
#[test]
fn permute_view_swap_generated_cases_match_rc4() {
    let mut lines = String::new();
    let mut modified = 0;
    for input in GENERATED_INPUT_SHAPES {
        let sizes = factorizations(numel(input), 4);
        for dims in permutations(input.len()) {
            for size in &sizes {
                for (form, reverse) in [("pair", false), ("chain", true)] {
                    let mut g = permute_view(input, &dims, size, reverse);
                    modified += usize::from(run(&mut g));
                    let outcome = render(&g).join("; ");
                    writeln!(lines, "{form}{input:?}{dims:?}{size:?}: {outcome}").unwrap();
                }
            }
        }
    }

    assert_eq!((lines.lines().count(), modified), (9260, 3242));
    assert_eq!(fnv1a(lines.as_bytes()), 0x4b5d_d237_4fe6_19d1);
}

/// `test_canonicalize_fuses_view_introduced_from_permute_tosa_FP`, which runs
/// `FuseViewCopyTransform` first, as here. That leaves an identity view before
/// the permute, which the pass removes; the permute then only moves a unit dim,
/// so it becomes a view.
#[test]
fn fuses_view_introduced_from_permute() {
    // x.view(1, 2, 1, 3, 4).view(2, 1, 3, 4).permute(0, 2, 3, 1), named as
    // to_edge names it.
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 1, 3, 4]);
    let v0 = view(&mut g, "aten_view_copy_default", x, &[1, 2, 1, 3, 4]);
    let v1 = view(&mut g, "aten_view_copy_default_1", v0, &[2, 1, 3, 4]);
    let p = permute(&mut g, "aten_permute_copy_default", v1, &[0, 2, 3, 1]);
    g.output(&[p]);

    assert!(IrFuseViewCopyTransformPass.run(g.graph_mut()));
    // `_AssertAfterInitialFusePass`; rc4's pass receives this graph.
    assert_eq!(
        render(&g),
        [
            "aten_view_copy_default = view(x, [2, 1, 3, 4]) -> [2, 1, 3, 4]",
            "aten_permute_copy_default = permute(aten_view_copy_default, [0, 2, 3, 1]) -> [2, 3, 4, 1]",
            "output(aten_permute_copy_default)",
        ]
    );

    assert!(run(&mut g));

    assert_eq!(
        render(&g),
        [
            "aten_permute_copy_default = view(x, [2, 3, 4, 1]) -> [2, 3, 4, 1]",
            "output(aten_permute_copy_default)",
        ]
    );
}

/// `test_noop_removal_exposes_transpose_cleanup`, from the graph
/// `RemoveNoopPass` hands the pass: once the zero PAD between them has gone,
/// the two transposes cancel.
#[test]
fn noop_removal_exposes_transpose_cleanup() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let first = permute(&mut g, "first_permute", x, &[1, 0]);
    let second = permute(&mut g, "second_permute", first, &[1, 0]);
    g.output(&[second]);

    assert!(run(&mut g));

    assert_eq!(render(&g), ["output(x)"]);
}

/// `test_data_layout_noop_cleanup_backend_pipeline`: rc4's pipeline runs the
/// pass five times on this program, and each time the only view or permute is
/// a lone view (the permutes are gone before the first run), so the pass
/// changes nothing. This is the graph of the fifth run.
#[test]
fn data_layout_noop_cleanup_backend_pipeline() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 1, 3]);
    let y = g.input("y", &[2, 1, 3]);
    let v = view(&mut g, "aten_view_copy_default", x, &[2, 3]);
    let add = call(&mut g, "aten_add_tensor", ADD, vec![node(x), node(y)]);
    let add_1 = call(&mut g, "aten_add_tensor_1", ADD, vec![node(x), node(y)]);
    g.output(&[v, x, add, add_1]);

    assert!(!run(&mut g));

    assert_eq!(
        render(&g),
        [
            "aten_view_copy_default = view(x, [2, 3]) -> [2, 3]",
            "aten_add_tensor = add(x, y) -> [2, 1, 3]",
            "aten_add_tensor_1 = add(x, y) -> [2, 1, 3]",
            "output(aten_view_copy_default, x, aten_add_tensor, aten_add_tensor_1)",
        ]
    );
}
