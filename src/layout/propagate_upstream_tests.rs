//! Ports of rc4's own tests for the propagation passes (APU-159):
//! `backends/transforms/test/test_propagate_view_copy_permute_pass.py` (the
//! shared pass), `backends/arm/test/passes/test_propagate_permutes_views_pass.py`
//! (Arm's) and `backends/arm/test/passes/test_rank_match_after_propagation.py`.
//!
//! Each test names the rc4 test it ports and asserts what rc4 asserts (all pass
//! under ExecuTorch v1.6.0-rc4 in the venv; rc4's one strict xfail is ported
//! with the result rc4 actually gives). Two layers:
//!
//! - **Recorded calls.** `tests/fixtures/propagate_upstream/rc4_test_calls.txt`
//!   holds every graph the rc4 tests hand the passes' `call`, and what rc4
//!   returned, recorded by `record_rc4_test_calls.py` in that directory; the
//!   case name says how the test built the pass. [`rc4_test_calls_match`]
//!   replays each one through the port built the same way, with the replay
//!   harness's full comparison. That covers the pipeline tests too, whose
//!   graphs come from `torch.export` (PassPipeline builds the Arm pass with
//!   the exported program, so with its fusion pass), and
//!   `test_per_channel_affine_rank4_u55_INT`, which runs both passes inside
//!   the whole U55 pipeline.
//! - **Assertions.** The tests below rebuild each graph and check rc4's
//!   assertions directly, including those on internal methods
//!   (`_propagate_iteration`, `_propagate`, `_move_node`, `is_swappable`) the
//!   recorder can't see. rc4's `torch.testing.assert_close` checks use the
//!   small reference evaluator here ([`evaluate`]).
//!
//! Not ported: `test_local_refresh_preserves_symbolic_view_permute_meta`
//! (both parametrizations): it checks that SymInt dims keep their identity,
//! and RETAB hands symbolic shapes to stock.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::MetaValue;
use crate::layout::ctx::LayoutCtx;
use crate::layout::fx_node::SPECIAL_DTYPE_KEY;
use crate::layout::propagate::{
    is_swappable, move_node, propagate_iteration, propagate_node, Direction, IrPropagateViewCopyPermutePass,
    PropagateConfig, LOCAL_REFRESHES, RETRACES,
};
use crate::layout::replay::{self, Options};
use crate::layout::shape::int_list;
use crate::layout::testing::*;

const RC4_TEST_CALLS: &str = include_str!("../../tests/fixtures/propagate_upstream/rc4_test_calls.txt");

const SUB: &str = "aten.sub.Tensor";
const GE: &str = "aten.ge.Tensor";
const FLOOR: &str = "aten.floor.default";
const CEIL: &str = "aten.ceil.default";
const MM: &str = "aten.mm.default";
const ANY_DIM: &str = "aten.any.dim";
const TABLE: &str = "backend.tosa.TABLE.default";
const SCATTER: &str = "backend.tosa.SCATTER.default";

const DOWN: Direction = Direction::Down;
const UP: Direction = Direction::Up;

/// `PropagateViewCopyPermuteDownPass()` / `UpPass()` from
/// backends/transforms (the shared pass).
fn shared(direction: Direction) -> PropagateConfig {
    PropagateConfig::shared(direction)
}

/// Arm's pass built without arguments, as `_run_pass_on_graph_module` builds it.
fn arm(direction: Direction) -> PropagateConfig {
    PropagateConfig::arm_without_program(direction)
}

/// Arm's pass as PassPipeline builds it: `pass_cls(exported_program)`, which
/// the constructor takes as the exported program (`:85-87`).
fn arm_with_program(direction: Direction) -> PropagateConfig {
    PropagateConfig::arm(direction)
}

fn run(g: &mut LayoutGraph, config: PropagateConfig) -> bool {
    IrPropagateViewCopyPermutePass { config }.run(g.graph_mut())
}

/// `[node.target for node in graph.nodes if node.op == "call_function"]`
/// after running the pass.
fn run_targets(g: &mut LayoutGraph, config: PropagateConfig) -> Vec<String> {
    run(g, config);
    g.targets().into_iter().map(str::to_string).collect()
}

fn index_of(targets: &[String], target: &str) -> usize {
    targets
        .iter()
        .position(|t| t == target)
        .unwrap_or_else(|| panic!("{target} is not in {targets:?}"))
}

fn count_of(targets: &[String], target: &str) -> usize {
    targets.iter().filter(|t| *t == target).count()
}

/// Live calls in graph order.
fn calls(g: &LayoutGraph) -> Vec<NodeId> {
    g.graph()
        .topological_live_node_ids()
        .into_iter()
        .filter(|&id| g.graph().node(id).op == NodeOp::CallFunction)
        .collect()
}

fn calls_to(g: &LayoutGraph, target: &str) -> Vec<NodeId> {
    calls(g)
        .into_iter()
        .filter(|&id| g.graph().node(id).target == target)
        .collect()
}

fn only(g: &LayoutGraph, target: &str) -> NodeId {
    let ids = calls_to(g, target);
    assert_eq!(ids.len(), 1, "live {target} nodes in {:?}", g.targets());
    ids[0]
}

fn arg_node(g: &LayoutGraph, id: NodeId, index: usize) -> NodeId {
    g.graph().node(id).args[index].as_node().expect("a node argument")
}

fn arg_ints(g: &LayoutGraph, id: NodeId, index: usize) -> Vec<i64> {
    int_list(&g.graph().node(id).args[index]).expect("an int list")
}

fn arg_int(g: &LayoutGraph, id: NodeId, index: usize) -> i64 {
    g.graph().node(id).args[index].as_int().expect("an int")
}

fn output_nodes(graph: &IrGraph) -> Vec<NodeId> {
    let output = graph.output_node().expect("an output node");
    let mut ids = Vec::new();
    output.args[0].collect_node_ids(&mut ids);
    ids
}

/// A call whose value the test gives, for ops the local refresh has no rule
/// for (TABLE, SCATTER, `mm`, `any.dim`).
fn call_val(g: &mut LayoutGraph, target: &str, args: Vec<IrArg>, shape: &[i64], dtype: &str) -> NodeId {
    let id = g.add_unrefreshed(target, args);
    g.graph_mut().node_mut(id).meta = NodeMeta::tensor(shape.to_vec(), dtype);
    id
}

fn set_int48(g: &mut LayoutGraph, id: NodeId) {
    g.graph_mut()
        .node_mut(id)
        .meta
        .set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("TosaSpecialDtype.INT48".to_string()));
}

fn rescale_with(g: &mut LayoutGraph, x: NodeId, dtype: &str, scales: &[f64]) -> NodeId {
    g.call(
        RESCALE,
        vec![
            node(x),
            IrArg::Dtype(dtype.to_string()),
            IrArg::List(scales.iter().map(|&s| IrArg::Float(s)).collect()),
            IrArg::Int(0),
            IrArg::Int(0),
        ],
    )
}

fn sum(g: &mut LayoutGraph, x: NodeId, dims: &[i64], keepdim: Option<bool>) -> NodeId {
    let mut args = vec![node(x), ints(dims)];
    if let Some(keepdim) = keepdim {
        args.push(IrArg::Bool(keepdim));
    }
    g.call(SUM, args)
}

fn mean(g: &mut LayoutGraph, x: NodeId, dims: &[i64], keepdim: bool) -> NodeId {
    g.call(MEAN, vec![node(x), ints(dims), IrArg::Bool(keepdim)])
}

// ---------------------------------------------------------------------------
// A reference evaluator, for rc4's `torch.testing.assert_close`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Tensor {
    shape: Vec<i64>,
    data: Vec<f64>,
}

impl Tensor {
    fn numel(shape: &[i64]) -> usize {
        shape.iter().product::<i64>() as usize
    }

    fn strides(shape: &[i64]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1] as usize;
        }
        strides
    }

    fn coords(flat: usize, shape: &[i64]) -> Vec<usize> {
        let strides = Self::strides(shape);
        (0..shape.len())
            .map(|axis| (flat / strides[axis]) % shape[axis] as usize)
            .collect()
    }

    /// The element at `coords` (one per axis); a size-1 axis broadcasts.
    fn at(&self, coords: &[usize]) -> f64 {
        let strides = Self::strides(&self.shape);
        let flat: usize = self
            .shape
            .iter()
            .enumerate()
            .filter(|(_, &extent)| extent != 1)
            .map(|(axis, _)| coords[axis] * strides[axis])
            .sum();
        self.data[flat]
    }

    fn map_n(inputs: &[&Tensor], shape: &[i64], f: impl Fn(&[f64]) -> f64) -> Tensor {
        let data = (0..Self::numel(shape))
            .map(|flat| {
                let coords = Self::coords(flat, shape);
                let values: Vec<f64> = inputs
                    .iter()
                    .map(|input| input.at(&coords[coords.len() - input.shape.len()..]))
                    .collect();
                f(&values)
            })
            .collect();
        Tensor {
            shape: shape.to_vec(),
            data,
        }
    }

    fn permute(&self, dims: &[i64]) -> Tensor {
        let rank = self.shape.len() as i64;
        let dims: Vec<usize> = dims.iter().map(|&d| d.rem_euclid(rank.max(1)) as usize).collect();
        let shape: Vec<i64> = dims.iter().map(|&d| self.shape[d]).collect();
        let data = (0..Self::numel(&shape))
            .map(|flat| {
                let out = Self::coords(flat, &shape);
                let mut source = vec![0; dims.len()];
                for (axis, &d) in dims.iter().enumerate() {
                    source[d] = out[axis];
                }
                self.at(&source)
            })
            .collect();
        Tensor { shape, data }
    }

    fn reduce(&self, dims: &[i64], keepdim: bool, mean: bool) -> Tensor {
        if self.shape.is_empty() {
            // torch reduces a 0-dim tensor over dim 0 or -1 to itself.
            return self.clone();
        }
        let rank = self.shape.len() as i64;
        let dims: Vec<usize> = if dims.is_empty() {
            (0..self.shape.len()).collect()
        } else {
            dims.iter().map(|&d| d.rem_euclid(rank.max(1)) as usize).collect()
        };
        let kept: Vec<i64> = self
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &extent)| if dims.contains(&axis) { 1 } else { extent })
            .collect();
        let mut data = vec![0.0; Self::numel(&kept)];
        let count: i64 = dims.iter().map(|&d| self.shape[d]).product();
        for flat in 0..Self::numel(&self.shape) {
            let mut coords = Self::coords(flat, &self.shape);
            for &d in &dims {
                coords[d] = 0;
            }
            let strides = Self::strides(&kept);
            let index: usize = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            data[index] += self.data[flat];
        }
        if mean {
            for value in &mut data {
                *value /= count as f64;
            }
        }
        let shape = if keepdim {
            kept
        } else {
            self.shape
                .iter()
                .enumerate()
                .filter(|(axis, _)| !dims.contains(axis))
                .map(|(_, &extent)| extent)
                .collect()
        };
        Tensor { shape, data }
    }

    fn slice(&self, dim: i64, start: i64, end: i64, step: i64) -> Tensor {
        let dim = dim.rem_euclid(self.shape.len() as i64) as usize;
        let end = end.min(self.shape[dim]);
        let picked: Vec<i64> = (start..end).step_by(step as usize).collect();
        let mut shape = self.shape.clone();
        shape[dim] = picked.len() as i64;
        let data = (0..Self::numel(&shape))
            .map(|flat| {
                let mut coords = Self::coords(flat, &shape);
                coords[dim] = picked[coords[dim]] as usize;
                self.at(&coords)
            })
            .collect();
        Tensor { shape, data }
    }

    fn cat(inputs: &[Tensor], dim: i64) -> Tensor {
        let rank = inputs[0].shape.len() as i64;
        let dim = dim.rem_euclid(rank) as usize;
        let mut shape = inputs[0].shape.clone();
        shape[dim] = inputs.iter().map(|input| input.shape[dim]).sum();
        let data = (0..Self::numel(&shape))
            .map(|flat| {
                let mut coords = Self::coords(flat, &shape);
                let mut offset = coords[dim] as i64;
                for input in inputs {
                    if offset < input.shape[dim] {
                        coords[dim] = offset as usize;
                        return input.at(&coords);
                    }
                    offset -= input.shape[dim];
                }
                unreachable!("cat coordinate out of range")
            })
            .collect();
        Tensor { shape, data }
    }
}

fn broadcast(shapes: &[&[i64]]) -> Vec<i64> {
    let shapes: Vec<Vec<i64>> = shapes.iter().map(|shape| shape.to_vec()).collect();
    crate::fx_ir::broadcast_shapes(&shapes).expect("evaluated shapes broadcast")
}

/// The graph's outputs on fixed inputs: distinct values per placeholder.
fn evaluate(graph: &IrGraph) -> Vec<Tensor> {
    let mut values: HashMap<NodeId, Tensor> = HashMap::new();
    let mut placeholders = 0usize;
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        let tensor = match n.op {
            NodeOp::Placeholder => {
                let shape = graph.node_shape_vec(id).expect("placeholders carry a shape");
                let boolean = graph.node_dtype(id) == Some("torch.bool");
                placeholders += 1;
                let data = (0..Tensor::numel(&shape))
                    .map(|i| {
                        if boolean {
                            ((i + placeholders) % 2) as f64
                        } else {
                            ((i * 37 + placeholders * 101) % 97) as f64 / 4.0 - 10.0
                        }
                    })
                    .collect();
                Tensor { shape, data }
            }
            NodeOp::CallFunction => {
                let input = |index: usize| values[&n.args[index].as_node().expect("a tensor operand")].clone();
                let unary = |f: fn(f64) -> f64| {
                    let x = input(0);
                    Tensor::map_n(&[&x], &x.shape.clone(), |v| f(v[0]))
                };
                let binary = |f: fn(f64, f64) -> f64| {
                    let (x, y) = (input(0), input(1));
                    let shape = broadcast(&[&x.shape, &y.shape]);
                    Tensor::map_n(&[&x, &y], &shape, |v| f(v[0], v[1]))
                };
                match n.target.as_str() {
                    VIEW => {
                        let shape = graph.node_shape_vec(id).expect("views carry a shape");
                        Tensor { shape, data: input(0).data }
                    }
                    PERMUTE => input(0).permute(&int_list(&n.args[1]).expect("permute dims")),
                    SUM | MEAN => {
                        let dims = int_list(&n.args[1]).unwrap_or_default();
                        let keepdim = n.args.get(2).and_then(IrArg::as_bool).unwrap_or(false);
                        input(0).reduce(&dims, keepdim, n.target == MEAN)
                    }
                    SLICE => {
                        let step = n.args.get(4).and_then(IrArg::as_int).unwrap_or(1);
                        input(0).slice(
                            n.args[1].as_int().expect("a slice dim"),
                            n.args[2].as_int().expect("a slice start"),
                            n.args[3].as_int().expect("a slice end"),
                            step,
                        )
                    }
                    CAT => {
                        let IrArg::List(items) = &n.args[0] else { panic!("cat takes a list") };
                        let inputs: Vec<Tensor> = items
                            .iter()
                            .map(|item| values[&item.as_node().expect("a tensor")].clone())
                            .collect();
                        Tensor::cat(&inputs, n.args.get(1).and_then(IrArg::as_int).unwrap_or(0))
                    }
                    WHERE => {
                        let (c, x, y) = (input(0), input(1), input(2));
                        let shape = broadcast(&[&c.shape, &x.shape, &y.shape]);
                        Tensor::map_n(&[&c, &x, &y], &shape, |v| if v[0] != 0.0 { v[1] } else { v[2] })
                    }
                    RELU => unary(|v| v.max(0.0)),
                    NEG => unary(|v| -v),
                    ABS => unary(f64::abs),
                    FLOOR => unary(f64::floor),
                    CEIL => unary(f64::ceil),
                    SIGMOID => unary(|v| 1.0 / (1.0 + (-v).exp())),
                    ADD => binary(|a, b| a + b),
                    SUB => binary(|a, b| a - b),
                    MUL => binary(|a, b| a * b),
                    GE => binary(|a, b| f64::from(a >= b)),
                    ANY_DIM => {
                        let x = input(0);
                        let dim = n.args[1].as_int().expect("a dim");
                        let keepdim = n.args.get(2).and_then(IrArg::as_bool).unwrap_or(false);
                        let summed = Tensor::map_n(&[&x], &x.shape.clone(), |v| f64::from(v[0] != 0.0))
                            .reduce(&[dim], keepdim, false);
                        Tensor::map_n(&[&summed], &summed.shape.clone(), |v| f64::from(v[0] != 0.0))
                    }
                    other => panic!("evaluate: no rule for {other}"),
                }
            }
            _ => continue,
        };
        values.insert(id, tensor);
    }
    output_nodes(graph).iter().map(|id| values[id].clone()).collect()
}

/// rc4's `torch.testing.assert_close(result(values), expected)`: the pass
/// keeps the graph's outputs.
fn assert_same_outputs(before: &[Tensor], g: &LayoutGraph) {
    let after = evaluate(g.graph());
    assert_eq!(before.len(), after.len());
    for (b, a) in before.iter().zip(&after) {
        assert_eq!(b.shape, a.shape, "output shapes");
        for (x, y) in b.data.iter().zip(&a.data) {
            assert!((x - y).abs() < 1e-9, "outputs differ: {:?} vs {:?}", b.data, a.data);
        }
    }
}

#[test]
fn the_evaluator_tells_layouts_apart() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p = g.permute(x, &[2, 0, 1]);
    let s = sum(&mut g, p, &[1], None);
    g.output(&[s]);
    let reference = evaluate(g.graph());
    let mut h = LayoutGraph::new();
    let x = h.input("x", &[2, 3, 4]);
    let s = sum(&mut h, x, &[0], None);
    let p = h.permute(s, &[1, 0]);
    h.output(&[p]);
    assert_eq!(evaluate(h.graph()), reference);
    let mut wrong = LayoutGraph::new();
    let x = wrong.input("x", &[2, 3, 4]);
    let s = sum(&mut wrong, x, &[1], None);
    let p = wrong.permute(s, &[1, 0]);
    wrong.output(&[p]);
    assert_ne!(evaluate(wrong.graph()), reference);
}

// ---------------------------------------------------------------------------
// Recorded calls
// ---------------------------------------------------------------------------

/// The pass a recorded case's config names (see the recorder's docstring).
struct RecordedPass {
    pass: IrPropagateViewCopyPermutePass,
    name: String,
}

impl IrPass for RecordedPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        self.pass.run(graph)
    }
}

fn block_add_to_add(graph: &IrGraph, _moving: NodeId, frontier: NodeId, next_nodes: &[NodeId]) -> bool {
    graph.node(frontier).target == ADD && next_nodes.iter().any(|&n| graph.node(n).target == ADD)
}

fn block_floor_to_sub(graph: &IrGraph, _moving: NodeId, frontier: NodeId, next_nodes: &[NodeId]) -> bool {
    graph.node(frontier).target == FLOOR && next_nodes.iter().any(|&n| graph.node(n).target == SUB)
}

fn recorded_config(config: &str) -> PropagateConfig {
    let (config, hook) = match config.split_once('~') {
        Some((config, hook)) => (config, Some(hook)),
        None => (config, None),
    };
    let (config, program) = match config.strip_suffix("+ep") {
        Some(config) => (config, true),
        None => (config, false),
    };
    let mut built = match config {
        "arm.down" => arm(DOWN),
        "arm.up" => arm(UP),
        "shared.down" => shared(DOWN),
        "shared.up" => shared(UP),
        other => panic!("unknown recorded config {other}"),
    };
    built.exported_program = program;
    built.blocks_moving_hook = match hook {
        None => None,
        Some("_BlockAddToAddPass") => Some(block_add_to_add),
        Some("_BlockFloorToSubPass") => Some(block_floor_to_sub),
        Some(other) => panic!("unknown recorded subclass {other}"),
    };
    built
}

/// Every call rc4's tests make, replayed through the port built the same way.
#[test]
fn rc4_test_calls_match() {
    let cases = replay::parse(RC4_TEST_CALLS).expect("the recorded calls parse");
    assert!(cases.len() >= 80, "{} recorded calls", cases.len());
    let options = Options {
        modified: true,
        dead_nodes: true,
        ..Options::default()
    };
    let mut failures = String::new();
    let mut configs: HashMap<String, usize> = HashMap::new();
    for case in &cases {
        let (_, config) = case.name.rsplit_once('|').expect("a recorded config");
        *configs.entry(config.to_string()).or_default() += 1;
        let pass = RecordedPass {
            pass: IrPropagateViewCopyPermutePass {
                config: recorded_config(config),
            },
            name: config.to_string(),
        };
        let result = replay::replay_case(&pass, case, options);
        if result.mismatch.is_some() {
            failures.push_str(&result.report);
        }
    }
    assert!(failures.is_empty(), "recorded rc4 test calls differ:\n{failures}");
    // Every way the tests build the pass is exercised.
    for config in ["shared.down", "shared.up", "arm.down", "arm.up", "arm.up+ep", "arm.down+ep"] {
        assert!(configs.contains_key(config), "no recorded call with {config}: {configs:?}");
    }
}

// ---------------------------------------------------------------------------
// backends/transforms/test/test_propagate_view_copy_permute_pass.py
// ---------------------------------------------------------------------------

const SOURCE_SHAPE: [i64; 4] = [1, 2, 3, 4];
const PERMUTATION: [i64; 4] = [0, 2, 3, 1];

/// `_forked_permute`: a permute feeding `branches` pointwise users.
fn forked_permute(g: &mut LayoutGraph, branches: usize) -> Vec<NodeId> {
    let x = g.input("x", &SOURCE_SHAPE);
    let p = g.permute(x, &PERMUTATION);
    [ABS, NEG, SIGMOID][..branches]
        .iter()
        .map(|&target| g.call(target, vec![node(p)]))
        .collect()
}

/// `_permute_counts`: permutes before, after the shared Down pass and after
/// the shared Up pass.
fn permute_counts(g: &mut LayoutGraph) -> (usize, usize, usize) {
    let before = g.count(PERMUTE);
    run(g, shared(DOWN));
    let after_down = g.count(PERMUTE);
    run(g, shared(UP));
    (before, after_down, g.count(PERMUTE))
}

/// `test_mixed_reconvergence_fork_does_not_strand_a_permute`, a strict xfail
/// expecting `(1, 1, 1)`: rc4 gives `(1, 2, 2)` (one copy below the meeting
/// add, one on the diverging branch), and so does the port.
#[test]
fn mixed_reconvergence_fork_strands_a_permute_as_in_rc4() {
    let mut g = LayoutGraph::new();
    let users = forked_permute(&mut g, 3);
    let rejoin = g.call(ADD, vec![node(users[0]), node(users[1])]);
    g.output(&[rejoin, users[2]]);
    assert_eq!(permute_counts(&mut g), (1, 2, 2));
}

/// `test_fork_split_still_applies_when_every_branch_rejoins`.
#[test]
fn fork_split_still_applies_when_every_branch_rejoins() {
    let mut g = LayoutGraph::new();
    let users = forked_permute(&mut g, 2);
    let rejoin = g.call(ADD, vec![node(users[0]), node(users[1])]);
    g.output(&[rejoin]);
    assert_eq!(permute_counts(&mut g), (1, 1, 1));
}

/// `test_fork_split_still_applies_when_every_branch_diverges`.
#[test]
fn fork_split_still_applies_when_every_branch_diverges() {
    let mut g = LayoutGraph::new();
    let users = forked_permute(&mut g, 2);
    g.output(&users);
    assert_eq!(permute_counts(&mut g), (1, 2, 1));
}

/// `test_down_pass_remembers_path_blocked_before_shared_node`, with
/// `_BlockAddToAddPass`'s `blocks_moving`.
#[test]
fn down_pass_remembers_path_blocked_before_shared_node() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &SOURCE_SHAPE);
    let p = g.permute(x, &PERMUTATION);
    let absolute = g.call(ABS, vec![node(p)]);
    let neg = g.call(NEG, vec![node(p)]);
    let inner = g.call(ADD, vec![node(p), node(absolute)]);
    let outer = g.call(ADD, vec![node(inner), node(neg)]);
    g.output(&[outer]);
    let mut config = shared(DOWN);
    config.blocks_moving_hook = Some(block_add_to_add);
    run(&mut g, config);
    g.assert_targets(&[PERMUTE, ABS, NEG, ADD, ADD]);
}

/// `test_is_swappable_accepts_reduction_that_drops_the_dimension`.
#[test]
fn shared_is_swappable_accepts_reduction_that_drops_the_dimension() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let keep = g.add_unrefreshed(SUM, vec![node(x), ints(&[1]), IrArg::Bool(true)]);
    let drop = g.add_unrefreshed(SUM, vec![node(x), ints(&[1]), IrArg::Bool(false)]);
    let default = g.add_unrefreshed(SUM, vec![node(x), ints(&[1])]);
    let ctx = LayoutCtx::new(g.graph_mut());
    for id in [keep, drop, default] {
        assert!(is_swappable(&ctx, &shared(UP), id));
    }
}

/// `test_permute_crosses_rank_dropping_reduction`, all four parametrizations.
#[test]
fn permute_crosses_rank_dropping_reduction() {
    for (direction, expected_permute, expected_dims) in [(DOWN, vec![1, 0], vec![0]), (UP, vec![0, 2, 1], vec![0])] {
        for keepdim_explicit in [false, true] {
            let keepdim = keepdim_explicit.then_some(false);
            let mut g = LayoutGraph::new();
            let x = g.input("x", &[2, 3, 4]);
            let out = if direction == DOWN {
                let p = g.permute(x, &[2, 0, 1]);
                sum(&mut g, p, &[1], keepdim)
            } else {
                let s = sum(&mut g, x, &[0], keepdim);
                g.permute(s, &[1, 0])
            };
            g.output(&[out]);
            let expected = evaluate(g.graph());
            run(&mut g, shared(direction));
            let order = calls(&g);
            let permute = only(&g, PERMUTE);
            let reduction = only(&g, SUM);
            let (p, r) = (
                order.iter().position(|&id| id == permute).unwrap(),
                order.iter().position(|&id| id == reduction).unwrap(),
            );
            if direction == DOWN {
                assert!(r < p);
            } else {
                assert!(p < r);
            }
            assert_eq!(arg_ints(&g, permute, 1), expected_permute);
            assert_eq!(arg_ints(&g, reduction, 1), expected_dims);
            assert_same_outputs(&expected, &g);
        }
    }
}

/// `test_upward_permute_preserves_scalar_reduction`, all six
/// parametrizations: a rank-0 permute after a rank-0 sum stays.
#[test]
fn upward_permute_preserves_scalar_reduction() {
    for dims in [vec![0], vec![-1], vec![]] {
        for keepdim_explicit in [false, true] {
            let mut g = LayoutGraph::new();
            let x = g.input("x", &[]);
            let s = sum(&mut g, x, &dims, keepdim_explicit.then_some(false));
            let p = g.permute(s, &[]);
            g.output(&[p]);
            let expected = evaluate(g.graph());
            {
                let mut ctx = LayoutCtx::new(g.graph_mut());
                assert!(!propagate_iteration(&mut ctx, &shared(UP)));
            }
            run(&mut g, shared(UP));
            let reduced = only(&g, SUM);
            assert_eq!(arg_ints(&g, reduced, 1), dims);
            assert_eq!(
                g.graph().node(reduced).args.get(2).and_then(IrArg::as_bool),
                keepdim_explicit.then_some(false)
            );
            assert_eq!(g.shape(reduced), &[] as &[i64]);
            assert_same_outputs(&expected, &g);
        }
    }
}

/// `test_downward_permute_crosses_reduction_over_all_dims`.
#[test]
fn downward_permute_crosses_reduction_over_all_dims() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let p = g.permute(x, &[2, 0, 1]);
    let s = sum(&mut g, p, &[], None);
    g.output(&[s]);
    let expected = evaluate(g.graph());
    run(&mut g, shared(DOWN));
    g.assert_targets(&[SUM]);
    assert_same_outputs(&expected, &g);
}

/// `test_upward_permute_crosses_rank_expanding_binary_after_reduction`.
#[test]
fn upward_permute_crosses_rank_expanding_binary_after_reduction() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let bias = g.input("bias", &[1, 1, 1, 1]);
    let add = g.call(ADD, vec![node(x), node(bias)]);
    let s = sum(&mut g, add, &[0], None);
    let p = g.permute(s, &[0, 2, 1]);
    g.output(&[p]);
    let expected = evaluate(g.graph());
    run(&mut g, shared(UP));
    let order = calls(&g);
    let permute = only(&g, PERMUTE);
    let add = only(&g, ADD);
    assert!(order.iter().position(|&id| id == permute) < order.iter().position(|&id| id == add));
    assert_eq!(arg_ints(&g, permute, 1), [1, 0]);
    assert_same_outputs(&expected, &g);
}

/// `test_upward_permute_preserves_broadcast_comparison`, both shapes.
#[test]
fn upward_permute_preserves_broadcast_comparison() {
    for shape in [[1, 3, 5], [1, 5, 5]] {
        let mut g = LayoutGraph::new();
        let scalar = g.input("scalar", &[1, 1, 1]);
        let data = g.input("data", &shape);
        let comparison = g.call(GE, vec![node(scalar), node(data)]);
        let p = g.permute(comparison, &[0, 2, 1]);
        let reduced = call_val(
            &mut g,
            ANY_DIM,
            vec![node(p), IrArg::Int(-1), IrArg::Bool(true)],
            &[1, shape[2], 1],
            "torch.bool",
        );
        let out = g.view(reduced, &[1, shape[2]]);
        g.output(&[out]);
        let expected = evaluate(g.graph());
        run(&mut g, shared(UP));
        assert_same_outputs(&expected, &g);
    }
}

/// `test_upward_permute_moves_through_broadcast_where`.
#[test]
fn upward_permute_moves_through_broadcast_where() {
    let mut g = LayoutGraph::new();
    let condition = g.typed_input("condition", &[1, 1, 1], "torch.bool");
    let data = g.input("data", &[1, 3, 5]);
    let other = g.input("other", &[1, 1, 1]);
    let w = g.call(WHERE, vec![node(condition), node(data), node(other)]);
    let p = g.permute(w, &[0, 2, 1]);
    g.output(&[p]);
    let expected = evaluate(g.graph());
    run(&mut g, shared(UP));
    g.assert_targets(&[PERMUTE, WHERE]);
    let order = calls(&g);
    assert_eq!(arg_node(&g, order[1], 1), order[0]);
    assert_same_outputs(&expected, &g);
}

// ---------------------------------------------------------------------------
// backends/arm/test/passes/test_propagate_permutes_views_pass.py
// ---------------------------------------------------------------------------

const X4: [i64; 4] = [1, 2, 3, 4];
const TO_LAST: [i64; 4] = [0, 2, 3, 1];

/// `x -> permute(0, 2, 3, 1) -> relu -> neg`.
fn permute_relu_neg(g: &mut LayoutGraph) {
    let x = g.input("x", &X4);
    let p = g.permute(x, &TO_LAST);
    let r = g.call(RELU, vec![node(p)]);
    let n = g.call(NEG, vec![node(r)]);
    g.output(&[n]);
}

/// `x -> relu -> neg -> permute(0, 2, 3, 1)`.
fn relu_neg_permute(g: &mut LayoutGraph) {
    let x = g.input("x", &X4);
    let r = g.call(RELU, vec![node(x)]);
    let n = g.call(NEG, vec![node(r)]);
    let p = g.permute(n, &TO_LAST);
    g.output(&[p]);
}

fn assert_order(targets: &[String], order: &[&str]) {
    for pair in order.windows(2) {
        assert!(
            index_of(targets, pair[0]) < index_of(targets, pair[1]),
            "{} must come before {} in {targets:?}",
            pair[0],
            pair[1]
        );
    }
}

/// `test_propagate_permute_down_through_transparent_ops_tosa_FP`: the
/// PassPipeline graph of `DownwardPermute` through the Up pass.
#[test]
fn propagate_permute_down_through_transparent_ops_tosa_fp() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    assert_eq!(g.count(PERMUTE), 1);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[PERMUTE, RELU, NEG]);
    assert_eq!(count_of(&targets, PERMUTE), 1);
}

/// `test_propagate_view_down_through_transparent_ops_tosa_FP` (`DownwardView`).
#[test]
fn propagate_view_down_through_transparent_ops_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v = g.view(x, &[2, 12]);
    let r = g.call(RELU, vec![node(v)]);
    let n = g.call(NEG, vec![node(r)]);
    g.output(&[n]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[VIEW, RELU, NEG]);
    assert_eq!(count_of(&targets, VIEW), 1);
}

/// `test_propagate_scalar_view_down_through_unary_elementwise`.
#[test]
fn propagate_scalar_view_down_through_unary_elementwise() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[]);
    let v = g.view(x, &[1]);
    let r = g.call(RELU, vec![node(v)]);
    g.output(&[r]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[RELU, VIEW]);
    assert_eq!(g.shape(only(&g, RELU)), &[] as &[i64]);
    assert_eq!(g.shape(only(&g, VIEW)), [1]);
}

/// `test_propagate_permute_up_through_transparent_ops_tosa_FP` (`UpwardPermute`).
#[test]
fn propagate_permute_up_through_transparent_ops_tosa_fp() {
    let mut g = LayoutGraph::new();
    relu_neg_permute(&mut g);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[PERMUTE, RELU, NEG]);
    assert_eq!(count_of(&targets, PERMUTE), 1);
}

/// `test_propagate_permute_up_swaps_with_binary_transparent_op_tosa_FP`
/// (`UpwardBinaryPermute`).
#[test]
fn propagate_permute_up_swaps_with_binary_transparent_op_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let add = g.call(ADD, vec![node(x), node(y)]);
    let p = g.permute(add, &TO_LAST);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[ADD, PERMUTE]);
}

/// `test_propagate_view_up_through_transparent_ops_tosa_FP` (`UpwardView`).
#[test]
fn propagate_view_up_through_transparent_ops_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let r = g.call(RELU, vec![node(x)]);
    let n = g.call(NEG, vec![node(r)]);
    let v = g.view(n, &[2, 12]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[VIEW, RELU, NEG]);
    assert_eq!(count_of(&targets, VIEW), 1);
}

/// `test_propagate_stops_at_non_transparent_ops_tosa_FP` (`StopAtNonTransparent`).
#[test]
fn propagate_stops_at_non_transparent_ops_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[3, 2]);
    let weight = g.input("weight", &[3, 4]);
    let p = g.permute(x, &[1, 0]);
    let mm = call_val(&mut g, MM, vec![node(p), node(weight)], &[2, 4], "torch.float32");
    g.output(&[mm]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[PERMUTE, MM]);
    assert_eq!(count_of(&targets, PERMUTE), 1);
}

/// `test_propagate_stops_at_branches_tosa_FP` (`StopAtBranch`).
#[test]
fn propagate_stops_at_branches_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.permute(x, &TO_LAST);
    let r = g.call(RELU, vec![node(y)]);
    let n = g.call(NEG, vec![node(y)]);
    let add = g.call(ADD, vec![node(r), node(n)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[PERMUTE, RELU]);
    assert_order(&targets, &[PERMUTE, NEG]);
    assert_eq!(count_of(&targets, PERMUTE), 1);
}

/// `test_propagate_moves_before_parameter_tosa_FP` (`StopAtParameter`): the
/// exported graph's `p_weight` is a parameter placeholder.
#[test]
fn propagate_moves_before_parameter_tosa_fp() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let weight = crate::test_ir::parameter(
        g.graph_mut(),
        "p_weight",
        crate::test_ir::f32_tensor(X4.to_vec(), &[0.5; 24]),
    );
    g.graph_mut().node_mut(weight).meta = NodeMeta::tensor(X4.to_vec(), "torch.float32");
    let add = g.call(ADD, vec![node(x), node(weight)]);
    let p = g.permute(add, &TO_LAST);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm_with_program(UP));
    assert_order(&targets, &[ADD, PERMUTE]);
    assert_eq!(count_of(&targets, PERMUTE), 1);
}

/// `test_propagation_retraces_once_after_reaching_fixed_point`, with
/// `_CountingDownPass`'s counters.
#[test]
fn propagation_retraces_once_after_reaching_fixed_point() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    RETRACES.with(|count| count.set(0));
    LOCAL_REFRESHES.with(|count| count.set(0));
    assert!(run(&mut g, arm(DOWN)));
    assert_eq!(RETRACES.with(std::cell::Cell::get), 1);
    assert_eq!(LOCAL_REFRESHES.with(std::cell::Cell::get), 1);
}

/// `test_down_propagation_visits_nodes_in_reverse_order`: one sweep moves the
/// permute, then the identity view behind it; a second changes nothing.
#[test]
fn down_propagation_visits_nodes_in_reverse_order() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let v = g.view(x, &X4);
    let r = g.call(RELU, vec![node(v)]);
    let p = g.permute(r, &TO_LAST);
    let n = g.call(NEG, vec![node(p)]);
    g.output(&[n]);
    {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(propagate_iteration(&mut ctx, &arm(DOWN)));
        assert!(!propagate_iteration(&mut ctx, &arm(DOWN)));
        ctx.finish();
    }
    g.assert_targets(&[RELU, NEG, VIEW, PERMUTE]);
}

/// `test_down_propagation_refreshes_metadata_in_dependency_order`.
#[test]
fn down_propagation_refreshes_metadata_in_dependency_order() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    let permute = only(&g, PERMUTE);
    {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(propagate_node(&mut ctx, &arm(DOWN), permute));
        ctx.finish();
    }
    assert_eq!(g.shape(only(&g, RELU)), [1, 2, 3, 4]);
    assert_eq!(g.shape(only(&g, NEG)), [1, 2, 3, 4]);
    assert_eq!(g.shape(only(&g, PERMUTE)), [1, 3, 4, 2]);
}

/// `test_up_propagation_refreshes_metadata_in_dependency_order`.
#[test]
fn up_propagation_refreshes_metadata_in_dependency_order() {
    let mut g = LayoutGraph::new();
    relu_neg_permute(&mut g);
    let permute = only(&g, PERMUTE);
    {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(propagate_node(&mut ctx, &arm(UP), permute));
        ctx.finish();
    }
    assert_eq!(g.shape(only(&g, PERMUTE)), [1, 3, 4, 2]);
    assert_eq!(g.shape(only(&g, RELU)), [1, 3, 4, 2]);
    assert_eq!(g.shape(only(&g, NEG)), [1, 3, 4, 2]);
}

/// `test_is_swappable_rejects_unnormalized_keep_dim_operator`.
#[test]
#[should_panic(expected = "expects keep_dim=True for reduction ops to simplify propagation logic, got")]
fn is_swappable_rejects_unnormalized_keep_dim_operator() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let mean = g.add_unrefreshed(MEAN, vec![node(x), ints(&[1]), IrArg::Bool(false)]);
    let ctx = LayoutCtx::new(g.graph_mut());
    is_swappable(&ctx, &arm(UP), mean);
}

/// `test_down_pass_moves_permute_after_transparent_chain`.
#[test]
fn down_pass_moves_permute_after_transparent_chain() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[RELU, NEG, PERMUTE]);
}

/// `test_down_pass_propagates_for_u85_like_tosa_int_cf`: the TOSA context
/// (INT+cf) doesn't change the pass.
#[test]
fn down_pass_propagates_for_u85_like_tosa_int_cf() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[RELU, NEG, PERMUTE]);
}

/// `test_down_pass_still_canonicalizes_for_u85_like_tosa_int_cf`.
#[test]
fn down_pass_still_canonicalizes_for_u85_like_tosa_int_cf() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 2, 3]);
    let first = g.permute(x, &[0, 2, 1]);
    let second = g.permute(first, &[0, 2, 1]);
    g.output(&[second]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 0);
}

/// `test_down_pass_moves_view_after_transparent_chain`.
#[test]
fn down_pass_moves_view_after_transparent_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let v = g.view(x, &[2, 12]);
    let r = g.call(RELU, vec![node(v)]);
    let n = g.call(NEG, vec![node(r)]);
    g.output(&[n]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[RELU, NEG, VIEW]);
}

/// `test_down_pass_moves_permute_to_graph_output`.
#[test]
fn down_pass_moves_permute_to_graph_output() {
    let mut g = LayoutGraph::new();
    permute_relu_neg(&mut g);
    run(&mut g, arm(DOWN));
    let permute = only(&g, PERMUTE);
    let neg = only(&g, NEG);
    assert_eq!(output_nodes(g.graph()), [permute]);
    assert_eq!(arg_node(&g, permute, 0), neg);
    let order = g.graph().topological_live_node_ids();
    let position = |id| order.iter().position(|&n| n == id).unwrap();
    let output = g.graph().live_output_ids()[0];
    assert!(position(neg) < position(permute) && position(permute) < position(output));
}

/// `test_down_pass_moves_permute_to_matching_output_branch`: `_move_node`
/// with the output as the frontier.
#[test]
fn down_pass_moves_permute_to_matching_output_branch() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let left = g.call(RELU, vec![node(x)]);
    let p = g.permute(x, &TO_LAST);
    let r = g.call(RELU, vec![node(p)]);
    let n = g.call(NEG, vec![node(r)]);
    let output = g.output(&[left, n]);
    let (left_name, p_name, r_name, n_name) = (g.name(left), g.name(p), g.name(r), g.name(n));
    {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        move_node(&mut ctx, &arm(DOWN), p, output, n);
        assert!(ctx.is_topological());
        ctx.finish();
    }
    let (left, p, r, n) = (g.id(&left_name), g.id(&p_name), g.id(&r_name), g.id(&n_name));
    assert_eq!(output_nodes(g.graph()), [left, p]);
    assert_eq!(arg_node(&g, r, 0), g.id("x"));
    assert_eq!(arg_node(&g, p, 0), n);
}

/// `test_up_pass_moves_permute_to_graph_input`.
#[test]
fn up_pass_moves_permute_to_graph_input() {
    let mut g = LayoutGraph::new();
    relu_neg_permute(&mut g);
    run(&mut g, arm(UP));
    let x = g.id("x");
    let permute = only(&g, PERMUTE);
    let relu = only(&g, RELU);
    assert_eq!(arg_node(&g, permute, 0), x);
    assert_eq!(arg_node(&g, relu, 0), permute);
    let order = g.graph().topological_live_node_ids();
    let position = |id| order.iter().position(|&n| n == id).unwrap();
    assert!(position(x) < position(permute) && position(permute) < position(relu));
}

fn two_slices(g: &mut LayoutGraph, x: NodeId, dim: i64, bounds: [(i64, i64); 2]) -> [NodeId; 2] {
    [g.slice(x, dim, bounds[0].0, bounds[0].1), g.slice(x, dim, bounds[1].0, bounds[1].1)]
}

fn slice_args(g: &LayoutGraph, id: NodeId) -> (NodeId, i64, i64, i64) {
    (arg_node(g, id, 0), arg_int(g, id, 1), arg_int(g, id, 2), arg_int(g, id, 3))
}

/// `test_up_pass_fuses_duplicate_permutes_at_placeholder`.
#[test]
fn up_pass_fuses_duplicate_permutes_at_placeholder() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 4, 3, 3]);
    let [left, right] = two_slices(&mut g, x, 1, [(0, 2), (2, 4)]);
    let lp = g.permute(left, &TO_LAST);
    let rp = g.permute(right, &TO_LAST);
    g.output(&[lp, rp]);
    run(&mut g, arm(UP));
    let permutes = calls_to(&g, PERMUTE);
    let slices = calls_to(&g, SLICE);
    assert_eq!((permutes.len(), slices.len()), (1, 2));
    assert_eq!(arg_node(&g, permutes[0], 0), g.id("x"));
    assert_eq!(arg_ints(&g, permutes[0], 1), TO_LAST);
    assert_eq!(slice_args(&g, slices[0]), (permutes[0], 3, 0, 2));
    assert_eq!(slice_args(&g, slices[1]), (permutes[0], 3, 2, 4));
}

/// `test_up_pass_refreshes_permute_meta_before_view_slice_swap`.
#[test]
fn up_pass_refreshes_permute_meta_before_view_slice_swap() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[3, 2, 8, 16]);
    let s = g.slice(x, 0, 0, 1);
    let p = g.permute(s, &[0, 3, 1, 2]);
    let v = g.view(p, &[1, 32, 8]);
    g.output(&[v]);
    run(&mut g, arm(UP));
    let (permute, view, slice) = (only(&g, PERMUTE), only(&g, VIEW), only(&g, SLICE));
    assert_eq!((arg_node(&g, permute, 0), arg_ints(&g, permute, 1)), (g.id("x"), vec![0, 3, 1, 2]));
    assert_eq!(g.shape(permute), [3, 16, 2, 8]);
    assert_eq!((arg_node(&g, view, 0), arg_ints(&g, view, 1)), (permute, vec![3, 32, 8]));
    assert_eq!(slice_args(&g, slice), (view, 0, 0, 1));
}

/// `test_up_pass_preserves_slice_permute_view_output`
/// (`ViewPermutePropagationMRE`): only numerics are checked.
#[test]
fn up_pass_preserves_slice_permute_view_output() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let s = g.call(SLICE, vec![node(x), IrArg::Int(1), IrArg::Int(0), IrArg::Int(1), IrArg::Int(1)]);
    let p = g.permute(s, &[2, 1, 0]);
    let v = g.view(p, &[8, 1]);
    g.output(&[v]);
    let expected = evaluate(g.graph());
    run(&mut g, arm_with_program(UP));
    assert_same_outputs(&expected, &g);
}

/// `test_up_pass_keeps_scatter_input_view_after_slice`.
#[test]
fn up_pass_keeps_scatter_input_view_after_slice() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[3, 16, 2, 8]);
    let indices = g.typed_input("indices", &[1, 4], "torch.int32");
    let data = g.input("data", &[1, 4, 16]);
    let s = g.slice(x, 0, 0, 1);
    let v = g.view(s, &[1, 32, 8]);
    let scatter = call_val(&mut g, SCATTER, vec![node(v), node(indices), node(data)], &[1, 32, 8], "torch.float32");
    g.output(&[scatter]);
    run(&mut g, arm(UP));
    let (view, slice, scatter) = (only(&g, VIEW), only(&g, SLICE), only(&g, SCATTER));
    assert_eq!((arg_node(&g, view, 0), arg_ints(&g, view, 1)), (slice, vec![1, 32, 8]));
    assert_eq!(arg_node(&g, scatter, 0), view);
}

/// `test_up_pass_hoists_matching_transform_chain_across_slice_fanout`.
#[test]
fn up_pass_hoists_matching_transform_chain_across_slice_fanout() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 4, 3, 3]);
    let [left, right] = two_slices(&mut g, x, 1, [(0, 2), (2, 4)]);
    let lv = g.view(left, &[1, 2, 9]);
    let rv = g.view(right, &[1, 2, 9]);
    let lp = g.permute(lv, &[0, 2, 1]);
    let rp = g.permute(rv, &[0, 2, 1]);
    g.output(&[lp, rp]);
    run(&mut g, arm(UP));
    let (views, permutes, slices) = (calls_to(&g, VIEW), calls_to(&g, PERMUTE), calls_to(&g, SLICE));
    assert_eq!((views.len(), permutes.len(), slices.len()), (1, 1, 2));
    assert_eq!((arg_node(&g, views[0], 0), arg_ints(&g, views[0], 1)), (g.id("x"), vec![1, 4, 9]));
    assert_eq!((arg_node(&g, permutes[0], 0), arg_ints(&g, permutes[0], 1)), (views[0], vec![0, 2, 1]));
    assert_eq!(slice_args(&g, slices[0]), (permutes[0], 2, 0, 2));
    assert_eq!(slice_args(&g, slices[1]), (permutes[0], 2, 2, 4));
}

/// `test_up_pass_hoists_unit_slice_views_with_different_args`.
#[test]
fn up_pass_hoists_unit_slice_views_with_different_args() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[5, 2, 3, 4, 6]);
    let [left, right] = two_slices(&mut g, x, 2, [(0, 1), (1, 2)]);
    let lv = g.view(left, &[5, 2, 4, 6]);
    let rv = g.view(right, &[5, 2, 24]);
    g.output(&[lv, rv]);
    run(&mut g, arm(UP));
    let (views, slices) = (calls_to(&g, VIEW), calls_to(&g, SLICE));
    let x = g.id("x");
    assert_eq!((arg_node(&g, views[0], 0), arg_ints(&g, views[0], 1)), (x, vec![5, 2, 12, 6]));
    assert_eq!((arg_node(&g, views[1], 0), arg_ints(&g, views[1], 1)), (x, vec![5, 2, 72]));
    assert_eq!(slice_args(&g, slices[0]), (views[0], 2, 0, 4));
    assert_eq!(slice_args(&g, slices[1]), (views[1], 2, 24, 48));
}

/// `test_up_pass_keeps_mismatched_transform_slice_fanout_split`.
#[test]
fn up_pass_keeps_mismatched_transform_slice_fanout_split() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 4, 3, 3]);
    let [left, right] = two_slices(&mut g, x, 1, [(0, 2), (2, 4)]);
    let lv = g.view(left, &[1, 2, 9]);
    let rp = g.permute(right, &TO_LAST);
    g.output(&[lv, rp]);
    run(&mut g, arm(UP));
    let slices = calls_to(&g, SLICE);
    assert_eq!(slices.len(), 2);
    assert_ne!(arg_node(&g, slices[0], 0), arg_node(&g, slices[1], 0));
}

/// `test_down_pass_moves_matching_input_permutations_after_binary_op`.
#[test]
fn down_pass_moves_matching_input_permutations_after_binary_op() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let xp = g.permute(x, &TO_LAST);
    let yp = g.permute(y, &TO_LAST);
    let add = g.call(ADD, vec![node(xp), node(yp)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[ADD, PERMUTE]);
}

/// `P(x) -> floor; sub(P, floor)`, returning `sub` (and `floor` with
/// `both_exits`).
fn floor_sub_region(g: &mut LayoutGraph, both_exits: bool) {
    let x = g.input("x", &X4);
    let p = g.permute(x, &TO_LAST);
    let floor = g.call(FLOOR, vec![node(p)]);
    let sub = g.call(SUB, vec![node(p), node(floor)]);
    if both_exits {
        g.output(&[floor, sub]);
    } else {
        g.output(&[sub]);
    }
}

/// `test_down_pass_moves_permute_through_single_exit_pointwise_region`.
#[test]
fn down_pass_moves_permute_through_single_exit_pointwise_region() {
    let mut g = LayoutGraph::new();
    floor_sub_region(&mut g, false);
    let expected = evaluate(g.graph());
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[FLOOR, SUB, PERMUTE]);
    assert_same_outputs(&expected, &g);
}

/// `test_down_pass_checks_each_path_into_shared_pointwise_node`, with
/// `_BlockFloorToSubPass`'s `blocks_moving`.
#[test]
fn down_pass_checks_each_path_into_shared_pointwise_node() {
    let mut g = LayoutGraph::new();
    floor_sub_region(&mut g, false);
    let mut config = arm(DOWN);
    config.blocks_moving_hook = Some(block_floor_to_sub);
    let targets = run_targets(&mut g, config);
    assert_order(&targets, &[PERMUTE, FLOOR, SUB]);
}

/// `test_down_pass_keeps_permute_before_multiple_exit_pointwise_region`.
#[test]
fn down_pass_keeps_permute_before_multiple_exit_pointwise_region() {
    let mut g = LayoutGraph::new();
    floor_sub_region(&mut g, true);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[PERMUTE, FLOOR, SUB]);
}

/// `test_down_pass_keeps_lower_rank_singleton_input_before_binary_op`.
#[test]
fn down_pass_keeps_lower_rank_singleton_input_before_binary_op() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &[1, 1]);
    let p = g.permute(x, &TO_LAST);
    let add = g.call(ADD, vec![node(p), node(y)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[PERMUTE, ADD]);
}

/// `test_down_pass_keeps_sunk_view_before_rank_reducing_permute`.
#[test]
fn down_pass_keeps_sunk_view_before_rank_reducing_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 8, 1, 32]);
    let y = g.input("y", &[2, 8, 1, 32]);
    let xv = g.view(x, &[2, 8, 32]);
    let yv = g.view(y, &[2, 8, 32]);
    let add = g.call(ADD, vec![node(xv), node(yv)]);
    let ov = g.view(add, &[2, 8, 32]);
    let p = g.permute(ov, &[0, 2, 1]);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, VIEW), 1);
    assert_order(&targets, &[ADD, VIEW, PERMUTE]);
    let (add, view, permute) = (only(&g, ADD), only(&g, VIEW), only(&g, PERMUTE));
    assert_eq!(g.shape(add), [2, 8, 1, 32]);
    assert_eq!(arg_node(&g, view, 0), add);
    assert_eq!(arg_node(&g, permute, 0), view);
}

/// `test_down_pass_canonicalizes_horizontally_fused_singleton_permute`.
#[test]
fn down_pass_canonicalizes_horizontally_fused_singleton_permute() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 1]);
    let y = g.input("y", &[1, 1, 1, 1]);
    let xp = g.permute(x, &TO_LAST);
    let yp = g.permute(y, &TO_LAST);
    let add = g.call(ADD, vec![node(xp), node(yp)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!((count_of(&targets, PERMUTE), count_of(&targets, VIEW)), (0, 0));
    let add = only(&g, ADD);
    let names: Vec<String> = crate::layout::fx_node::all_input_nodes(g.graph().node(add))
        .into_iter()
        .map(|id| g.name(id))
        .collect();
    assert_eq!(names, ["x", "y"]);
    assert_eq!(output_nodes(g.graph()), [add]);
}

/// `x, y -> permute(0, 2, 3, 1) -> cat(dim 3)`.
fn permuted_cat(g: &mut LayoutGraph) {
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let xp = g.permute(x, &TO_LAST);
    let yp = g.permute(y, &TO_LAST);
    let cat = g.cat(&[xp, yp], 3);
    g.output(&[cat]);
}

/// `test_down_pass_moves_matching_input_permutations_after_cat`.
#[test]
fn down_pass_moves_matching_input_permutations_after_cat() {
    let mut g = LayoutGraph::new();
    permuted_cat(&mut g);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[CAT, PERMUTE]);
    assert_eq!(arg_int(&g, only(&g, CAT), 1), 1);
}

/// `test_down_pass_swaps_concat_with_matching_input_permutations`.
#[test]
fn down_pass_swaps_concat_with_matching_input_permutations() {
    let mut g = LayoutGraph::new();
    permuted_cat(&mut g);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[CAT, PERMUTE]);
    let (cat, permute) = (only(&g, CAT), only(&g, PERMUTE));
    let IrArg::List(items) = &g.graph().node(cat).args[0] else {
        panic!("cat takes a list")
    };
    let names: Vec<String> = items.iter().map(|item| g.name(item.as_node().unwrap())).collect();
    assert_eq!(names, ["x", "y"]);
    assert_eq!(arg_int(&g, cat, 1), 1);
    assert_eq!(g.shape(cat), [1, 4, 3, 4]);
    assert_eq!((arg_node(&g, permute, 0), arg_ints(&g, permute, 1)), (cat, TO_LAST.to_vec()));
}

/// `x, y: [1, 1, 3, 4] -> cat(dim 1) -> permute(0, 2, 3, 1)`.
fn noop_permute_after_cat(g: &mut LayoutGraph) {
    let x = g.input("x", &[1, 1, 3, 4]);
    let y = g.input("y", &[1, 1, 3, 4]);
    let cat = g.cat(&[x, y], 1);
    let p = g.permute(cat, &TO_LAST);
    g.output(&[p]);
}

fn assert_cat_inputs_became_views(g: &LayoutGraph, targets: &[String]) {
    assert_eq!((count_of(targets, PERMUTE), count_of(targets, VIEW)), (0, 2));
    let cat = only(g, CAT);
    assert_eq!(arg_int(g, cat, 1), 3);
    assert_eq!(g.shape(cat), [1, 3, 4, 2]);
    let IrArg::List(items) = &g.graph().node(cat).args[0] else {
        panic!("cat takes a list")
    };
    assert!(items
        .iter()
        .all(|item| g.graph().node(item.as_node().unwrap()).target == VIEW));
}

/// `test_up_pass_moves_noop_input_permutations_before_cat`.
#[test]
fn up_pass_moves_noop_input_permutations_before_cat() {
    let mut g = LayoutGraph::new();
    noop_permute_after_cat(&mut g);
    let targets = run_targets(&mut g, arm(UP));
    assert_cat_inputs_became_views(&g, &targets);
}

/// `test_up_pass_swaps_concat_with_noop_output_permutation`.
#[test]
fn up_pass_swaps_concat_with_noop_output_permutation() {
    let mut g = LayoutGraph::new();
    noop_permute_after_cat(&mut g);
    let targets = run_targets(&mut g, arm(UP));
    assert_cat_inputs_became_views(&g, &targets);
}

/// `test_down_pass_keeps_shared_input_permutations_before_cat`.
#[test]
fn down_pass_keeps_shared_input_permutations_before_cat() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let xp = g.permute(x, &TO_LAST);
    let yp = g.permute(y, &TO_LAST);
    let r = g.call(RELU, vec![node(xp)]);
    let cat = g.cat(&[xp, yp], 3);
    g.output(&[cat, r]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 2);
    let cat = only(&g, CAT);
    let IrArg::List(items) = &g.graph().node(cat).args[0] else {
        panic!("cat takes a list")
    };
    assert!(items
        .iter()
        .all(|item| g.graph().node(item.as_node().unwrap()).target == PERMUTE));
    assert_eq!(arg_int(&g, cat, 1), 3);
}

/// `test_down_pass_keeps_permutation_before_sum`.
#[test]
fn down_pass_keeps_permutation_before_sum() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let p = g.permute(x, &TO_LAST);
    let s = sum(&mut g, p, &[3], Some(true));
    g.output(&[s]);
    let targets = run_targets(&mut g, arm(DOWN));
    let sum = only(&g, SUM);
    let transform = calls(&g)
        .into_iter()
        .find(|&id| matches!(g.graph().node(id).target.as_str(), PERMUTE | VIEW))
        .expect("a transform");
    assert!(index_of(&targets, &g.graph().node(transform).target) < index_of(&targets, SUM));
    assert_eq!(arg_ints(&g, sum, 1), [3]);
    assert_eq!(g.shape(transform), [1, 3, 4, 2]);
}

/// `test_down_pass_keeps_permute_before_reduction_with_layout_dependent_user`,
/// both parametrizations.
#[test]
fn down_pass_keeps_permute_before_reduction_with_layout_dependent_user() {
    for mean_first in [false, true] {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 512, 796]);
        let direct = g.input("direct", &[1, 796, 512]);
        let p = g.permute(x, &[0, 2, 1]);
        let r = g.call(RELU, vec![node(p)]);
        let m = mean(&mut g, r, &[1], true);
        let args = if mean_first { vec![node(m), node(direct)] } else { vec![node(direct), node(m)] };
        let s = g.call(SUB, args);
        g.output(&[s]);
        run(&mut g, arm(DOWN));
        let (mean, sub) = (only(&g, MEAN), only(&g, SUB));
        let mean_input = arg_node(&g, mean, 0);
        let input_shape = g.shape(mean_input).to_vec();
        let output_shape = g.shape(mean).to_vec();
        let rank = input_shape.len() as i64;
        let dims: Vec<usize> = arg_ints(&g, mean, 1).iter().map(|d| d.rem_euclid(rank) as usize).collect();
        for (dim, (input, output)) in input_shape.iter().zip(&output_shape).enumerate() {
            if dims.contains(&dim) {
                assert_eq!(*output, 1);
            } else {
                assert_eq!(output, input);
            }
        }
        let sub_inputs = crate::layout::fx_node::all_input_nodes(g.graph().node(sub));
        let shapes: Vec<Vec<i64>> = sub_inputs.iter().map(|&id| g.shape(id).to_vec()).collect();
        assert_eq!(crate::fx_ir::broadcast_shapes(&shapes).unwrap(), g.shape(sub));
        assert!(!shapes.contains(&vec![1, 512, 1]));
    }
}

/// `test_down_pass_splits_permute_over_elementwise_fanout`.
#[test]
fn down_pass_splits_permute_over_elementwise_fanout() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let p = g.permute(x, &TO_LAST);
    let r = g.call(RELU, vec![node(p)]);
    let n = g.call(NEG, vec![node(p)]);
    g.output(&[r, n]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 2);
    let mut sources: Vec<String> = calls_to(&g, PERMUTE)
        .into_iter()
        .map(|id| g.graph().node(arg_node(&g, id, 0)).target.clone())
        .collect();
    sources.sort();
    assert_eq!(sources, [NEG, RELU]);
}

/// `test_down_pass_splits_permute_over_slice_fanout`.
#[test]
fn down_pass_splits_permute_over_slice_fanout() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 4, 3, 3]);
    let p = g.permute(x, &TO_LAST);
    let [left, right] = two_slices(&mut g, p, 3, [(0, 2), (2, 4)]);
    let lv = g.view(left, &[1, 9, 2]);
    let rv = g.view(right, &[1, 9, 2]);
    let lp = g.permute(lv, &[0, 2, 1]);
    let rp = g.permute(rv, &[0, 2, 1]);
    g.output(&[lp, rp]);
    run(&mut g, arm(DOWN));
    let x = g.id("x");
    let slices = calls_to(&g, SLICE);
    assert_eq!(slice_args(&g, slices[0]), (x, 1, 0, 2));
    assert_eq!(slice_args(&g, slices[1]), (x, 1, 2, 4));
    assert!(calls_to(&g, PERMUTE)
        .into_iter()
        .all(|id| g.graph().node(arg_node(&g, id, 0)).target == SLICE));
}

/// `test_down_pass_stops_when_fanout_branch_has_nontransparent_op`.
#[test]
fn down_pass_stops_when_fanout_branch_has_nontransparent_op() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let weight = g.input("weight", &[2, 2]);
    let p = g.permute(x, &[1, 0]);
    let r = g.call(RELU, vec![node(p)]);
    let mm = call_val(&mut g, MM, vec![node(p), node(weight)], &[3, 2], "torch.float32");
    let add = g.call(ADD, vec![node(r), node(mm)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[PERMUTE, RELU]);
    assert_order(&targets, &[PERMUTE, MM]);
}

/// `test_down_pass_stops_when_convergence_has_untracked_input`.
#[test]
fn down_pass_stops_when_convergence_has_untracked_input() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &[1, 3, 4, 2]);
    let p = g.permute(x, &TO_LAST);
    let r = g.call(RELU, vec![node(p)]);
    let n = g.call(NEG, vec![node(p)]);
    let cat = g.cat(&[r, n, y], 3);
    g.output(&[cat]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 2);
    let mut sources: Vec<String> = calls_to(&g, PERMUTE)
        .into_iter()
        .map(|id| g.graph().node(arg_node(&g, id, 0)).target.clone())
        .collect();
    sources.sort();
    assert_eq!(sources, [NEG, RELU]);
}

/// `test_down_pass_stops_view_before_cat_converging_fanout`.
#[test]
fn down_pass_stops_view_before_cat_converging_fanout() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let v = g.view(x, &[1, 3, 4, 2]);
    let r = g.call(RELU, vec![node(v)]);
    let n = g.call(NEG, vec![node(v)]);
    let cat = g.cat(&[r, n], 3);
    g.output(&[cat]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, VIEW), 1);
    assert_order(&targets, &[VIEW, RELU]);
    assert_order(&targets, &[VIEW, NEG]);
}

/// `test_up_pass_fuses_equivalent_output_permutations_before_fan_out`.
#[test]
fn up_pass_fuses_equivalent_output_permutations_before_fan_out() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let r = g.call(RELU, vec![node(x)]);
    let first = g.permute(r, &TO_LAST);
    let second = g.permute(r, &TO_LAST);
    let add = g.call(ADD, vec![node(first), node(second)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(UP));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[PERMUTE, RELU, ADD]);
}

/// `test_up_pass_fusion_enables_further_propagation`.
#[test]
fn up_pass_fusion_enables_further_propagation() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let r = g.call(RELU, vec![node(x)]);
    let n = g.call(NEG, vec![node(r)]);
    let first = g.permute(n, &TO_LAST);
    let second = g.permute(r, &TO_LAST);
    let add = g.call(ADD, vec![node(first), node(second)]);
    g.output(&[add]);
    run(&mut g, arm(UP));
    let permutes = calls_to(&g, PERMUTE);
    assert_eq!(permutes.len(), 1);
    let (relu, neg, add) = (only(&g, RELU), only(&g, NEG), only(&g, ADD));
    assert_eq!(arg_node(&g, permutes[0], 0), g.id("x"));
    assert_eq!(arg_node(&g, relu, 0), permutes[0]);
    assert_eq!(arg_node(&g, neg, 0), relu);
    let mut inputs = crate::layout::fx_node::all_input_nodes(g.graph().node(add));
    inputs.sort_unstable();
    let mut expected = vec![relu, neg];
    expected.sort_unstable();
    assert_eq!(inputs, expected);
}

/// `test_propagate_moves_before_dtype_changing_rescale`.
#[test]
fn propagate_moves_before_dtype_changing_rescale() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &X4, "torch.int32");
    let r = rescale_with(&mut g, x, "torch.int8", &[1.0]);
    let p = g.permute(r, &TO_LAST);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[PERMUTE, RESCALE]);
}

/// `x -> permute -> mul(scale)`, then a comparison, two arithmetic
/// branches through floor and ceil, and a where (or with views).
fn elementwise_fork(g: &mut LayoutGraph, view: bool) {
    let (shape, ones): (&[i64], &[i64]) = if view { (&[1, 2, 12], &[1, 1, 1]) } else { (&X4, &[1, 1, 1, 1]) };
    let x = g.input("x", shape);
    let scale = g.input("scale", ones);
    let bias = g.input("bias", ones);
    let t = if view { g.view(x, &[1, 3, 8]) } else { g.permute(x, &TO_LAST) };
    let scaled = g.call(MUL, vec![node(t), node(scale)]);
    let condition = g.call(GE, vec![node(scaled), node(bias)]);
    let add = g.call(ADD, vec![node(scaled), node(bias)]);
    let sub = g.call(SUB, vec![node(scaled), node(bias)]);
    let floor = g.call(FLOOR, vec![node(add)]);
    let ceil = g.call(CEIL, vec![node(sub)]);
    let w = g.call(WHERE, vec![node(condition), node(floor), node(ceil)]);
    g.output(&[w]);
}

/// `test_propagate_moves_through_elementwise_fork`.
#[test]
fn propagate_moves_through_elementwise_fork() {
    let mut g = LayoutGraph::new();
    elementwise_fork(&mut g, false);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[WHERE, PERMUTE]);
}

/// `test_propagate_stops_at_sum_reduction_fork`.
#[test]
fn propagate_stops_at_sum_reduction_fork() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let bias = g.input("bias", &[1, 1, 1, 1]);
    let p = g.permute(x, &TO_LAST);
    let condition_sum = sum(&mut g, p, &[1], Some(true));
    let condition = g.call(GE, vec![node(condition_sum), node(bias)]);
    let data_sum = sum(&mut g, p, &[1], Some(true));
    let data_mean = mean(&mut g, p, &[1], true);
    let floor = g.call(FLOOR, vec![node(data_sum)]);
    let ceil = g.call(CEIL, vec![node(data_mean)]);
    let w = g.call(WHERE, vec![node(condition), node(floor), node(ceil)]);
    g.output(&[w]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    let reductions: Vec<NodeId> = calls(&g)
        .into_iter()
        .filter(|&id| matches!(g.graph().node(id).target.as_str(), SUM | MEAN))
        .collect();
    let first_reduction = reductions
        .iter()
        .map(|&id| index_of(&targets, &g.graph().node(id).target))
        .min()
        .unwrap();
    assert!(index_of(&targets, PERMUTE) < first_reduction);
    for id in reductions {
        assert_eq!(arg_ints(&g, id, 1), [1]);
    }
}

/// `test_propagate_view_moves_through_elementwise_fork`.
#[test]
fn propagate_view_moves_through_elementwise_fork() {
    let mut g = LayoutGraph::new();
    elementwise_fork(&mut g, true);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_eq!(count_of(&targets, VIEW), 1);
    assert_order(&targets, &[MUL, VIEW]);
    for target in [GE, ADD, SUB] {
        assert_order(&targets, &[VIEW, target]);
    }
}

fn table_input(g: &mut LayoutGraph) -> NodeId {
    g.typed_input("table", &[256], "torch.int8")
}

/// `test_propagate_fuses_permute_view_around_table`.
#[test]
fn propagate_fuses_permute_view_around_table() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3, 4], "torch.int8");
    let table = table_input(&mut g);
    let p = g.permute(x, &[1, 0, 2]);
    let v = g.view(p, &[3, 8]);
    let t = call_val(&mut g, TABLE, vec![node(v), node(table)], &[3, 8], "torch.int8");
    let ov = g.view(t, &[3, 2, 4]);
    let op = g.permute(ov, &[1, 0, 2]);
    g.output(&[op]);
    run(&mut g, arm(DOWN));
    g.assert_targets(&[TABLE]);
}

/// `test_propagate_up_moves_permute_before_table`.
#[test]
fn propagate_up_moves_permute_before_table() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3, 4], "torch.int8");
    let table = table_input(&mut g);
    let t = call_val(&mut g, TABLE, vec![node(x), node(table)], &[2, 3, 4], "torch.int8");
    let p = g.permute(t, &[1, 0, 2]);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[PERMUTE, TABLE]);
}

/// `test_propagate_down_keeps_view_before_table_with_batched_singleton_spatial`.
#[test]
fn propagate_down_keeps_view_before_table_with_batched_singleton_spatial() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[990, 1, 1, 64], "torch.int8");
    let table = table_input(&mut g);
    let v = g.view(x, &[99, 10, 64]);
    let t = call_val(&mut g, TABLE, vec![node(v), node(table)], &[99, 10, 64], "torch.int8");
    g.output(&[t]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[VIEW, TABLE]);
}

/// `test_propagate_up_keeps_view_after_table_with_batched_singleton_spatial`.
#[test]
fn propagate_up_keeps_view_after_table_with_batched_singleton_spatial() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[99, 10, 64], "torch.int8");
    let table = table_input(&mut g);
    let t = call_val(&mut g, TABLE, vec![node(x), node(table)], &[99, 10, 64], "torch.int8");
    let v = g.view(t, &[990, 1, 1, 64]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[TABLE, VIEW]);
}

/// `test_propagate_down_still_sinks_view_through_table_for_safe_shapes`.
#[test]
fn propagate_down_still_sinks_view_through_table_for_safe_shapes() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 1, 1, 64], "torch.int8");
    let table = table_input(&mut g);
    let v = g.view(x, &[1, 64]);
    let t = call_val(&mut g, TABLE, vec![node(v), node(table)], &[1, 64], "torch.int8");
    g.output(&[t]);
    let targets = run_targets(&mut g, arm(DOWN));
    assert_order(&targets, &[TABLE, VIEW]);
}

/// `test_propagate_stops_at_per_channel_rescale`.
#[test]
fn propagate_stops_at_per_channel_rescale() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3, 4, 5], "torch.int32");
    let r = rescale_with(&mut g, x, "torch.int8", &[1.0; 5]);
    let p = g.permute(r, &[0, 3, 1, 2]);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// `test_propagate_stops_at_rescale_changing_special_dtype`.
#[test]
fn propagate_stops_at_rescale_changing_special_dtype() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 1, 1, 15], "torch.int32");
    set_int48(&mut g, x);
    let r = rescale_with(&mut g, x, "torch.int32", &[1.0]);
    let v = g.view(r, &[15]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[RESCALE, VIEW]);
}

/// `test_propagate_up_stops_at_shared_rescale_producer`.
#[test]
fn propagate_up_stops_at_shared_rescale_producer() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[10, 80, 16], "torch.int8");
    let y = g.typed_input("y", &[10, 80, 16], "torch.int32");
    let r = rescale_with(&mut g, x, "torch.int32", &[1.0]);
    let p = g.permute(r, &[1, 0, 2]);
    let add = g.call(ADD, vec![node(r), node(y)]);
    g.output(&[p, add]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// The Up pass, then `MoveDataMovementOpsToSmallerDtypePass()`, as the three
/// `test_smaller_dtype_pass_restores_permute_*` tests run them.
fn up_then_smaller_dtype(g: &mut LayoutGraph) -> Vec<String> {
    run(g, arm(UP));
    crate::layout::move_data_movement::IrMoveDataMovementOpsToSmallerDtypePass.run(g.graph_mut());
    g.targets().into_iter().map(str::to_string).collect()
}

fn int_binary_rescale_permute(g: &mut LayoutGraph, input: &str, output: &str) {
    let x = g.typed_input("x", &X4, input);
    let y = g.typed_input("y", &X4, input);
    let add = g.call(ADD, vec![node(x), node(y)]);
    let r = rescale_with(g, add, output, &[1.0]);
    let p = g.permute(r, &TO_LAST);
    g.output(&[p]);
}

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_fed_by_binary_op`.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_fed_by_binary_op() {
    let mut g = LayoutGraph::new();
    int_binary_rescale_permute(&mut g, "torch.int32", "torch.int8");
    let targets = up_then_smaller_dtype(&mut g);
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// `test_propagate_up_crosses_same_width_rescale_fed_by_binary_op`.
#[test]
fn propagate_up_crosses_same_width_rescale_fed_by_binary_op() {
    let mut g = LayoutGraph::new();
    int_binary_rescale_permute(&mut g, "torch.int32", "torch.int32");
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[PERMUTE, RESCALE]);
}

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_from_shared_placeholder`.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_from_shared_placeholder() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &X4, "torch.int32");
    let r = rescale_with(&mut g, x, "torch.int8", &[1.0]);
    let other = g.call(NEG, vec![node(x)]);
    let p = g.permute(r, &TO_LAST);
    g.output(&[p, other]);
    let targets = up_then_smaller_dtype(&mut g);
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// `test_propagate_up_crosses_widening_rescale_fed_by_binary_op`.
#[test]
fn propagate_up_crosses_widening_rescale_fed_by_binary_op() {
    let mut g = LayoutGraph::new();
    int_binary_rescale_permute(&mut g, "torch.int8", "torch.int32");
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[PERMUTE, RESCALE]);
}

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_behind_unary`.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_behind_unary() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &X4, "torch.int32");
    let clone = g.call(CLONE, vec![node(x)]);
    let r = rescale_with(&mut g, clone, "torch.int8", &[1.0]);
    let p = g.permute(r, &TO_LAST);
    g.output(&[p]);
    let targets = up_then_smaller_dtype(&mut g);
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// `test_tagged_int48_permute_does_not_propagate`.
#[test]
fn tagged_int48_permute_does_not_propagate() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &X4, "torch.int32");
    set_int48(&mut g, x);
    let r = g.call(RELU, vec![node(x)]);
    set_int48(&mut g, r);
    let p = g.permute(r, &TO_LAST);
    set_int48(&mut g, p);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[RELU, PERMUTE]);
}

/// `test_permute_does_not_cross_int48_rescale_input`.
#[test]
fn permute_does_not_cross_int48_rescale_input() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[2, 3, 5], "torch.int32");
    set_int48(&mut g, x);
    let r = rescale_with(&mut g, x, "torch.int8", &[0.25]);
    let p = g.permute(r, &[1, 0, 2]);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[RESCALE, PERMUTE]);
}

/// `test_propagate_keeps_output_view_after_sum_with_split_dim_remap`.
#[test]
fn propagate_keeps_output_view_after_sum_with_split_dim_remap() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[6, 4]);
    let s = sum(&mut g, x, &[0], Some(true));
    let v = g.view(s, &[1, 1, 4]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SUM, VIEW]);
    assert_eq!(arg_ints(&g, only(&g, SUM), 1), [0]);
    assert_eq!(arg_ints(&g, only(&g, VIEW), 1), [1, 1, 4]);
}

/// `test_propagate_stops_view_map_update_before_sum`.
#[test]
fn propagate_stops_view_map_update_before_sum() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[6, 4]);
    let s = g.slice(x, 0, 0, 4);
    let sm = sum(&mut g, s, &[0], Some(true));
    let v = g.view(sm, &[1, 1, 4]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SLICE, SUM, VIEW]);
    assert_eq!(arg_ints(&g, only(&g, VIEW), 1), [1, 1, 4]);
    assert_eq!(arg_int(&g, only(&g, SLICE), 1), 0);
    assert_eq!(arg_ints(&g, only(&g, SUM), 1), [0]);
}

/// `test_propagate_moves_output_view_before_mean_with_split_dim_remap`.
#[test]
fn propagate_moves_output_view_before_mean_with_split_dim_remap() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[6, 4]);
    let m = mean(&mut g, x, &[0], true);
    let v = g.view(m, &[1, 1, 4]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[VIEW, MEAN]);
    assert_eq!(arg_ints(&g, only(&g, MEAN), 1), [0, 1]);
    assert_eq!(arg_ints(&g, only(&g, VIEW), 1), [6, 1, 4]);
}

/// `test_propagate_keeps_reduction_squeeze_after_sum`.
#[test]
fn propagate_keeps_reduction_squeeze_after_sum() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 50, 10, 20]);
    let s = sum(&mut g, x, &[-1], Some(true));
    let v = g.view(s, &[1, 50, 10]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SUM, VIEW]);
    assert_eq!(arg_ints(&g, only(&g, SUM), 1), [-1]);
    assert_eq!(arg_ints(&g, only(&g, VIEW), 1), [1, 50, 10]);
}

/// `test_propagate_keeps_unit_slice_before_reordering_view`.
#[test]
fn propagate_keeps_unit_slice_before_reordering_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 3, 1, 7]);
    let s = g.slice(x, 3, 2, 3);
    let v = g.view(s, &[1, 1, 1, 3]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SLICE, VIEW]);
    let (slice, view) = (only(&g, SLICE), only(&g, VIEW));
    assert_eq!(slice_args(&g, slice).1..=slice_args(&g, slice).3, 3..=3);
    assert_eq!((arg_int(&g, slice, 1), arg_int(&g, slice, 2), arg_int(&g, slice, 3)), (3, 2, 3));
    assert_eq!((arg_node(&g, view, 0), arg_ints(&g, view, 1)), (slice, vec![1, 1, 1, 3]));
}

/// `test_propagate_stops_when_downward_inputs_are_not_equivalent_transforms`.
#[test]
fn propagate_stops_when_downward_inputs_are_not_equivalent_transforms() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &[1, 3, 4, 2]);
    let p = g.permute(x, &TO_LAST);
    let add = g.call(ADD, vec![node(p), node(y)]);
    g.output(&[add]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[PERMUTE, ADD]);
}

/// A slice of `shape` along `dim`, viewed to `view`; the view stays after it.
fn assert_view_stays_after_slice(shape: &[i64], dim: i64, bounds: (i64, i64), view: &[i64]) {
    let mut g = LayoutGraph::new();
    let x = g.input("x", shape);
    let s = g.slice(x, dim, bounds.0, bounds.1);
    let v = g.view(s, view);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SLICE, VIEW]);
}

/// `test_propagate_stops_split_dim_view_at_slice`.
#[test]
fn propagate_stops_split_dim_view_at_slice() {
    assert_view_stays_after_slice(&[6, 4], 0, (0, 4), &[2, 2, 4]);
}

/// `test_propagate_stops_merged_trailing_dim_view_at_slice`.
#[test]
fn propagate_stops_merged_trailing_dim_view_at_slice() {
    assert_view_stays_after_slice(&[25, 5, 13, 7], 1, (0, 2), &[1, 25, 182]);
}

/// `test_propagate_stops_split_dim_view_at_cat`.
#[test]
fn propagate_stops_split_dim_view_at_cat() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[3, 4]);
    let y = g.input("y", &[3, 4]);
    let cat = g.cat(&[x, y], 0);
    let v = g.view(cat, &[2, 3, 4]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[CAT, VIEW]);
}

/// `test_propagate_keeps_channel_unit_slice_before_reordering_view`.
#[test]
fn propagate_keeps_channel_unit_slice_before_reordering_view() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let s = g.slice(x, 1, 0, 1);
    let v = g.view(s, &[1, 3, 4, 1]);
    g.output(&[v]);
    let targets = run_targets(&mut g, arm(UP));
    assert_order(&targets, &[SLICE, VIEW]);
    let (slice, view) = (only(&g, SLICE), only(&g, VIEW));
    assert_eq!((arg_int(&g, slice, 1), arg_int(&g, slice, 2), arg_int(&g, slice, 3)), (1, 0, 1));
    assert_eq!((arg_node(&g, view, 0), arg_ints(&g, view, 1)), (slice, vec![1, 3, 4, 1]));
}

/// `test_propagate_up_stops_at_multiple_distinct_edge_nodes`.
#[test]
fn propagate_up_stops_at_multiple_distinct_edge_nodes() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let add = g.call(ADD, vec![node(x), node(y)]);
    let p = g.permute(add, &TO_LAST);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[ADD, PERMUTE]);
}

/// `test_propagate_up_moves_to_top_node_before_distinct_edge_nodes`.
#[test]
fn propagate_up_moves_to_top_node_before_distinct_edge_nodes() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &X4);
    let y = g.input("y", &X4);
    let add = g.call(ADD, vec![node(x), node(y)]);
    let r = g.call(RELU, vec![node(add)]);
    let p = g.permute(r, &TO_LAST);
    g.output(&[p]);
    let targets = run_targets(&mut g, arm(UP));
    assert_eq!(count_of(&targets, PERMUTE), 1);
    assert_order(&targets, &[ADD, PERMUTE, RELU]);
}

/// `test_propagate_stops_rank_changing_view_at_slice`.
#[test]
fn propagate_stops_rank_changing_view_at_slice() {
    assert_view_stays_after_slice(&[1, 2, 3], 0, (0, 1), &[2, 3]);
}

// ---------------------------------------------------------------------------
// backends/arm/test/passes/test_rank_match_after_propagation.py
// ---------------------------------------------------------------------------

/// `test_per_channel_affine_rank4_u55_INT`, at the two propagation passes
/// inside the U55 pipeline (`arm_pass_manager.py:726-727`): the recorded
/// graphs go through the port's passes, which fold the singleton permute
/// into the rank-2 view, leave no permute, and keep every binary op's
/// operands rank-matched, which is what the test guards (MatchArgRanksPass
/// then has nothing to do). The whole result must equal rc4's too
/// ([`rc4_test_calls_match`]).
///
/// Not ported: the Vela compile that follows.
#[test]
fn per_channel_affine_rank4_u55_int_propagation() {
    let cases = replay::parse(RC4_TEST_CALLS).expect("the recorded calls parse");
    let mut seen = 0;
    for case in cases
        .iter()
        .filter(|case| case.name.starts_with("test_per_channel_affine_rank4_u55_INT#"))
    {
        let (_, config) = case.name.rsplit_once('|').unwrap();
        let mut graph = replay::build(&case.before);
        IrPropagateViewCopyPermutePass {
            config: recorded_config(config),
        }
        .run(&mut graph);
        for id in graph.live_call_function_ids() {
            let n = graph.node(id);
            assert_ne!(n.target, PERMUTE, "a permute is left");
            if matches!(n.target.as_str(), ADD | MUL) {
                let ranks: Vec<usize> = crate::layout::fx_node::all_input_nodes(n)
                    .into_iter()
                    .map(|input| graph.node_shape(input).unwrap().len())
                    .collect();
                assert!(ranks.windows(2).all(|pair| pair[0] == pair[1]), "{} has ranks {ranks:?}", n.name);
            }
        }
        seen += 1;
    }
    assert_eq!(seen, 2, "the Down and Up calls are recorded");
}
