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

fn placeholder(g: &LayoutGraph, name: &str) -> NodeId {
    g.id(name)
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
