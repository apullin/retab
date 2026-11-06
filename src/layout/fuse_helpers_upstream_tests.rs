//! Ports of rc4's own tests for FuseDuplicateUsers, FuseIdenticalInputTransforms
//! and MoveDataMovementOpsToSmallerDtype (APU-160).
//!
//! Each test names the rc4 test it ports, and asserts what rc4 asserts (all
//! 41 pass under ExecuTorch v1.6.0-rc4 in the venv), plus details checked by
//! running rc4 on the same graph; most of these graphs are also structured
//! cases of the replay fixtures, which compare the whole result.
//!
//! - rc4's `_validate_numerics` runs both graphs on random inputs. Here a small
//!   reference evaluator ([`evaluate`]) runs the graph before and after the
//!   pass on fixed inputs.
//! - rc4's tests of `NormalizeTransformInputPlaceholdersPass` also check the
//!   graph signature's input specs and `process_placeholder`; RETAB keys inputs
//!   by node name and has no signature, so only the placeholder targets are
//!   checked.
//! - The tests that run other passes first
//!   (`test_fuse_duplicate_users_runs_after_tosa_transformations`,
//!   `test_noop_removal_exposes_duplicate_user_cleanup`,
//!   `test_splits_merged_permute_at_distinct_narrowing_rescales` and the three
//!   `test_smaller_dtype_pass_restores_permute_*` of
//!   test_propagate_permutes_views_pass.py) are ported at the pass level: the
//!   fixture is the graph this pass receives there, captured by running the
//!   passes before it in the venv.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::MetaValue;
use crate::layout::fuse_duplicate_users::{IrFuseDuplicateUsersPass, DO_NOT_FUSE_DUPLICATE_META_KEY};
use crate::layout::ctx::LayoutCtx;
use crate::layout::fuse_identical_input_transforms::{
    normalize_transform_input_placeholders, IrFuseIdenticalInputTransformsPass, DELEGATION_TAG_META_KEY,
    PERMUTE_TARGET,
};
use crate::layout::move_data_movement::IrMoveDataMovementOpsToSmallerDtypePass;
use crate::layout::shape::int_list;
use crate::layout::testing::*;

const AVG_POOL: &str = "aten.avg_pool2d.default";
const SUB: &str = "aten.sub.Tensor";
const GT: &str = "aten.gt.Tensor";

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

fn output_nodes(graph: &IrGraph) -> Vec<NodeId> {
    let output = graph.output_node().expect("an output node");
    let mut ids = Vec::new();
    output.args[0].collect_node_ids(&mut ids);
    ids
}

fn input_of(g: &LayoutGraph, id: NodeId) -> NodeId {
    g.graph().node(id).arg_node(0).expect("a tensor input")
}

fn set_tag(g: &mut LayoutGraph, id: NodeId, tag: &str) {
    g.graph_mut()
        .node_mut(id)
        .meta
        .set_extra(DELEGATION_TAG_META_KEY, MetaValue::Str(tag.to_string()));
}

fn tag(g: &LayoutGraph, id: NodeId) -> Option<String> {
    match g.graph().node(id).meta.extras.get(DELEGATION_TAG_META_KEY) {
        Some(MetaValue::Str(tag)) => Some(tag.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// A reference evaluator, for rc4's `_validate_numerics`
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

    /// The flat indices of `self` read at each position of `shape`, which
    /// `self.shape` broadcasts to.
    fn broadcast_indices(&self, shape: &[i64]) -> Vec<usize> {
        let offset = shape.len() - self.shape.len();
        let strides = Self::strides(&self.shape);
        let out_strides = Self::strides(shape);
        (0..Self::numel(shape))
            .map(|flat| {
                (0..self.shape.len())
                    .map(|axis| {
                        let coordinate = (flat / out_strides[axis + offset]) % shape[axis + offset] as usize;
                        if self.shape[axis] == 1 {
                            0
                        } else {
                            coordinate * strides[axis]
                        }
                    })
                    .sum()
            })
            .collect()
    }

    fn map_n(inputs: &[&Tensor], shape: &[i64], f: impl Fn(&[f64]) -> f64) -> Tensor {
        let indices: Vec<Vec<usize>> = inputs.iter().map(|input| input.broadcast_indices(shape)).collect();
        let data = (0..Self::numel(shape))
            .map(|flat| {
                let values: Vec<f64> = inputs
                    .iter()
                    .zip(&indices)
                    .map(|(input, index)| input.data[index[flat]])
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
        let dims: Vec<usize> = dims.iter().map(|&dim| ((dim + rank) % rank.max(1)) as usize).collect();
        let shape: Vec<i64> = dims.iter().map(|&dim| self.shape[dim]).collect();
        let strides = Self::strides(&self.shape);
        let out_strides = Self::strides(&shape);
        let data = (0..Self::numel(&shape))
            .map(|flat| {
                let source: usize = (0..shape.len())
                    .map(|axis| ((flat / out_strides[axis]) % shape[axis] as usize) * strides[dims[axis]])
                    .sum();
                self.data[source]
            })
            .collect();
        Tensor { shape, data }
    }

    fn cat(inputs: &[&Tensor], dim: i64) -> Tensor {
        let rank = inputs[0].shape.len() as i64;
        let axis = ((dim + rank) % rank) as usize;
        let mut shape = inputs[0].shape.clone();
        shape[axis] = inputs.iter().map(|input| input.shape[axis]).sum();
        let outer: usize = shape[..axis].iter().product::<i64>() as usize;
        let mut data = Vec::with_capacity(Self::numel(&shape));
        for block in 0..outer {
            for input in inputs {
                let chunk: usize = input.shape[axis..].iter().product::<i64>() as usize;
                data.extend_from_slice(&input.data[block * chunk..(block + 1) * chunk]);
            }
        }
        Tensor { shape, data }
    }
}

fn broadcast(shapes: &[&[i64]]) -> Vec<i64> {
    let shapes: Vec<Vec<i64>> = shapes.iter().map(|shape| shape.to_vec()).collect();
    crate::fx_ir::broadcast_shapes(&shapes).expect("evaluated shapes broadcast")
}

/// The graph's outputs, on fixed inputs: distinct values per placeholder, and
/// alternating truth values for a bool placeholder.
fn evaluate(graph: &IrGraph) -> Vec<Tensor> {
    let mut values: HashMap<NodeId, Tensor> = HashMap::new();
    let mut placeholders = 0usize;
    for id in graph.topological_live_node_ids() {
        let n = graph.node(id);
        let tensor = match n.op {
            NodeOp::Placeholder => {
                let shape = graph.node_shape_vec(id).expect("placeholders carry a shape");
                let boolean = graph.node_dtype(id) == Some("torch.bool");
                let count = Tensor::numel(&shape);
                placeholders += 1;
                let data = (0..count)
                    .map(|i| {
                        if boolean {
                            ((i + placeholders) % 2) as f64
                        } else {
                            ((i * 37 + placeholders * 101) % 97) as f64 - 40.0
                        }
                    })
                    .collect();
                Tensor { shape, data }
            }
            NodeOp::CallFunction => {
                let node = |arg: &IrArg| values[&arg.as_node().expect("a tensor operand")].clone();
                match n.target.as_str() {
                    VIEW => {
                        let input = node(&n.args[0]);
                        let size = int_list(&n.args[1]).expect("a view size");
                        let shape = graph.node_shape_vec(id).expect("views carry a shape");
                        assert_eq!(size.len(), shape.len());
                        Tensor {
                            shape,
                            data: input.data,
                        }
                    }
                    PERMUTE => node(&n.args[0]).permute(&int_list(&n.args[1]).expect("permute dims")),
                    CAT | "aten.concatenate.default" => {
                        let IrArg::List(items) = &n.args[0] else {
                            panic!("cat takes a list")
                        };
                        let inputs: Vec<Tensor> = items.iter().map(node).collect();
                        let dim = n
                            .args
                            .get(1)
                            .or_else(|| n.kwargs.get("dim"))
                            .and_then(IrArg::as_int)
                            .unwrap_or(0);
                        Tensor::cat(&inputs.iter().collect::<Vec<_>>(), dim)
                    }
                    WHERE => {
                        let (c, x, y) = (node(&n.args[0]), node(&n.args[1]), node(&n.args[2]));
                        let shape = broadcast(&[&c.shape, &x.shape, &y.shape]);
                        Tensor::map_n(&[&c, &x, &y], &shape, |v| if v[0] != 0.0 { v[1] } else { v[2] })
                    }
                    NEG => {
                        let x = node(&n.args[0]);
                        Tensor::map_n(&[&x], &x.shape.clone(), |v| -v[0])
                    }
                    RELU => {
                        let x = node(&n.args[0]);
                        Tensor::map_n(&[&x], &x.shape.clone(), |v| v[0].max(0.0))
                    }
                    target => {
                        let (x, y) = (node(&n.args[0]), node(&n.args[1]));
                        let alpha = n.kwargs.get("alpha").and_then(IrArg::as_numeric_f64).unwrap_or(1.0);
                        let f: fn(f64, f64, f64) -> f64 = match target {
                            ADD => |a, b, alpha| a + alpha * b,
                            SUB => |a, b, alpha| a - alpha * b,
                            MUL => |a, b, _| a * b,
                            GT => |a, b, _| f64::from(a > b),
                            "aten.bitwise_and.Tensor" => |a, b, _| ((a as i64) & (b as i64)) as f64,
                            "aten.bitwise_left_shift.Tensor" => {
                                |a, b, _| ((a as i64).wrapping_shl((b as i64).rem_euclid(8) as u32)) as f64
                            }
                            other => panic!("evaluate: no rule for {other}"),
                        };
                        let shape = broadcast(&[&x.shape, &y.shape]);
                        Tensor::map_n(&[&x, &y], &shape, |v| f(v[0], v[1], alpha))
                    }
                }
            }
            _ => continue,
        };
        values.insert(id, tensor);
    }
    output_nodes(graph).iter().map(|id| values[id].clone()).collect()
}

/// The evaluator tells layouts apart: `[[1, 2, 3], [4, 5, 6]]`-shaped
/// inputs, a transpose, a broadcast and a concat give rc4's numbers, and a
/// wrong permutation gives different ones.
#[test]
fn the_reference_evaluator_catches_a_wrong_layout() {
    let transposed = |dims: &[i64]| {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let ones = g.input("ones", &[3, 1]);
        let p = g.permute(x, dims);
        let shape = g.shape(p).to_vec();
        let ones = if shape == [3, 2] { ones } else { g.view(ones, &[1, 3]) };
        let add = g.call(ADD, vec![node(p), node(ones)]);
        let cat = g.cat(&[add, add], 0);
        g.output(&[cat]);
        evaluate(g.graph())
    };
    let identity = transposed(&[0, 1]);
    let swapped = transposed(&[1, 0]);
    assert_ne!(identity[0].data, swapped[0].data);
    assert_eq!(swapped[0].shape, vec![6, 2]);
    // x is ((i * 37 + 101) % 97) - 40 for i in 0..6, ones (i * 37 + 202) % 97 - 40.
    let x: Vec<f64> = (0..6).map(|i| ((i * 37 + 101) % 97) as f64 - 40.0).collect();
    let ones: Vec<f64> = (0..3).map(|i| ((i * 37 + 202) % 97) as f64 - 40.0).collect();
    let row0: Vec<f64> = vec![x[0] + ones[0], x[3] + ones[0]];
    assert_eq!(swapped[0].data[..2], row0[..]);
}

/// rc4's `_apply_pass` and `_validate_numerics`: run the Arm pass, and check
/// the outputs are unchanged.
fn apply_fuse_identical_input_transforms(g: &mut LayoutGraph) -> bool {
    let before = evaluate(g.graph());
    let modified = IrFuseIdenticalInputTransformsPass.run(g.graph_mut());
    assert_eq!(evaluate(g.graph()), before, "the pass changed the numerics");
    modified
}

// ---------------------------------------------------------------------------
// FuseDuplicateUsersPass (backends/arm/test/passes/test_fuse_duplicate_users_pass.py)
// ---------------------------------------------------------------------------

/// `AvgPool2d(1)` as edge lowers it.
fn avg_pool(g: &mut LayoutGraph, x: NodeId) -> NodeId {
    g.add_unrefreshed(
        AVG_POOL,
        vec![
            node(x),
            ints(&[1, 1]),
            ints(&[1, 1]),
            ints(&[0, 0]),
            IrArg::Bool(false),
            IrArg::Bool(true),
        ],
    )
}

/// `test_fuse_duplicate_users_tosa_FP[fuse_avg_pool]`: `avg(x) + avg(x) +
/// avg(x)` keeps one of its three `avg_pool2d`s.
#[test]
fn fuse_duplicate_users_tosa_fp_fuse_avg_pool() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 1]);
    let (a0, a1) = (avg_pool(&mut g, x), avg_pool(&mut g, x));
    let s = g.add_unrefreshed(ADD, vec![node(a0), node(a1)]);
    let a2 = avg_pool(&mut g, x);
    let out = g.add_unrefreshed(ADD, vec![node(s), node(a2)]);
    g.output(&[out]);
    assert_eq!(g.count(AVG_POOL), 3);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(g.count(AVG_POOL), 1);
}

/// `test_fuse_duplicate_users_tosa_FP[fuse_avg_pool_chain]`: three
/// `avg(avg(x))` chains keep one chain, two `avg_pool2d`s.
#[test]
fn fuse_duplicate_users_tosa_fp_fuse_avg_pool_chain() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1, 1, 1, 1]);
    let chains: Vec<NodeId> = (0..3)
        .map(|_| {
            let inner = avg_pool(&mut g, x);
            avg_pool(&mut g, inner)
        })
        .collect();
    let s = g.add_unrefreshed(ADD, vec![node(chains[0]), node(chains[1])]);
    let out = g.add_unrefreshed(ADD, vec![node(s), node(chains[2])]);
    g.output(&[out]);
    assert_eq!(g.count(AVG_POOL), 6);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(g.count(AVG_POOL), 2);
}

/// `_graph_with_users_not_in_node_order`: `later` is created first, then
/// `earlier` and its consumer are inserted before it.
fn users_not_in_node_order() -> (LayoutGraph, String, String, String) {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1]);
    let y = g.input("y", &[1]);
    let later = g.call(ADD, vec![node(x), node(y)]);
    let earlier = g.call(ADD, vec![node(x), node(y)]);
    g.graph_mut().anchor_to(earlier, later);
    let consumer = g.call(NEG, vec![node(earlier)]);
    g.graph_mut().anchor_to(consumer, later);
    g.output(&[consumer]);
    let names = (g.name(earlier), g.name(later), g.name(consumer));
    (g, names.0, names.1, names.2)
}

/// `test_fuse_duplicate_users_preserves_graph_order_for_representative`: one
/// add is left. rc4 keeps the one first in graph order, which its consumer
/// reads (checked in the venv), so the graph stays topologically sorted.
#[test]
fn fuse_duplicate_users_preserves_graph_order_for_representative() {
    let (mut g, earlier, later, consumer) = users_not_in_node_order();
    let adds = calls_to(&g, ADD);
    assert_eq!(adds, vec![g.id(&earlier), g.id(&later)]);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(calls_to(&g, ADD), vec![g.id(&earlier)]);
    assert!(g.graph().by_name(&later).is_none_or(|node| !node.alive));
    assert_eq!(input_of(&g, g.id(&consumer)), g.id(&earlier));
}

/// `test_fuse_duplicate_users_honors_do_not_fuse_marker`: marking the first
/// add in graph order blocks the fusion.
#[test]
fn fuse_duplicate_users_honors_do_not_fuse_marker() {
    let (mut g, earlier, _, _) = users_not_in_node_order();
    let marked = g.id(&earlier);
    g.graph_mut()
        .node_mut(marked)
        .meta
        .set_extra(DO_NOT_FUSE_DUPLICATE_META_KEY, MetaValue::Bool(true));

    assert!(!IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(g.count(ADD), 2);
}

/// `test_fuse_duplicate_users_skips_erased_queued_nodes`: fusing the two
/// `neg`s makes the two `relu`s duplicates; the erased `neg` is still queued.
#[test]
fn fuse_duplicate_users_skips_erased_queued_nodes() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[1]);
    let n0 = g.call(NEG, vec![node(x)]);
    let n1 = g.call(NEG, vec![node(x)]);
    let r0 = g.call(RELU, vec![node(n0)]);
    let r1 = g.call(RELU, vec![node(n1)]);
    g.output(&[r0, r1]);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    let (negs, relus) = (calls_to(&g, NEG), calls_to(&g, RELU));
    assert_eq!((negs.len(), relus.len()), (1, 1));
    assert_eq!(input_of(&g, relus[0]), negs[0]);
    assert_eq!(output_nodes(g.graph()), vec![relus[0], relus[0]]);
}

/// `test_fuse_duplicate_users_keeps_identical_rescale_users`: two identical
/// int8-to-int32 RESCALEs stay, as Arm excludes RESCALE.
#[test]
fn fuse_duplicate_users_keeps_identical_rescale_users() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1], "torch.int8");
    let rescale = |g: &mut LayoutGraph| {
        g.call(
            RESCALE,
            vec![
                node(x),
                IrArg::Dtype("torch.int32".into()),
                IrArg::List(vec![IrArg::Float(1.0)]),
                IrArg::Int(16),
                IrArg::Int(0),
            ],
        )
    };
    let (first, second) = (rescale(&mut g), rescale(&mut g));
    g.output(&[first, second]);

    assert!(!IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(g.count(RESCALE), 2);
}

/// `test_fuse_duplicate_users_runs_after_tosa_transformations`, at the pass's
/// position after the post-processing RemoveNoopPass (`arm_pass_manager.py:765`):
/// FuseEqualPlaceholdersPass has merged the two equal buffers, so the TOSA
/// `ADD`s of `x + first` and `x + second` are duplicates and one feeds both
/// outputs. Not ported: EnsureUniqueOutputNodesPass's two IDENTITY nodes and
/// the pipeline-order check, which need the rest of the pipeline.
#[test]
fn fuse_duplicate_users_runs_after_tosa_transformations() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let buffer = g.input("b_first", &[2, 3]);
    let first = g.add_unrefreshed("tosa.ADD.default", vec![node(x), node(buffer)]);
    let second = g.add_unrefreshed("tosa.ADD.default", vec![node(x), node(buffer)]);
    g.output(&[first, second]);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    let adds = calls_to(&g, "tosa.ADD.default");
    assert_eq!(adds.len(), 1);
    assert_eq!(output_nodes(g.graph()), vec![adds[0], adds[0]]);
}

/// `test_remove_data_layout_noops.py::test_noop_removal_exposes_duplicate_user_cleanup`,
/// at FuseDuplicateUsersPass: RemoveNoopPass dropped the zero PAD, leaving
/// two `add(x, y)`s (confirmed in the venv); one remains and feeds both
/// outputs, which EnsureUniqueOutputNodesPass (not ported) then separates.
#[test]
fn noop_removal_exposes_duplicate_user_cleanup() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let y = g.input("y", &[2, 3]);
    let first = g.call(ADD, vec![node(x), node(y)]);
    let second = g.call(ADD, vec![node(x), node(y)]);
    g.output(&[first, second]);

    assert!(IrFuseDuplicateUsersPass.run(g.graph_mut()));

    assert_eq!(g.count(ADD), 1);
    let outputs = output_nodes(g.graph());
    assert_eq!(outputs[0], outputs[1]);
}

// ---------------------------------------------------------------------------
// FuseIdenticalInputTransformsPass
// (backends/arm/test/passes/test_fuse_identical_input_transforms_pass.py)
// ---------------------------------------------------------------------------

enum Transform<'a> {
    View(&'a [i64]),
    Permute(&'a [i64]),
}

fn apply(g: &mut LayoutGraph, x: NodeId, transforms: &[Transform<'_>]) -> NodeId {
    transforms.iter().fold(x, |value, transform| match transform {
        Transform::View(shape) => g.view(value, shape),
        Transform::Permute(dims) => g.permute(value, dims),
    })
}

/// `_binary_transform_graph`: `add(x transforms, y transforms)`.
fn binary_transform_graph(
    x_shape: &[i64],
    y_shape: &[i64],
    x_transforms: &[Transform<'_>],
    y_transforms: &[Transform<'_>],
) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", x_shape);
    let y = g.input("y", y_shape);
    let tx = apply(&mut g, x, x_transforms);
    let ty = apply(&mut g, y, y_transforms);
    let add = g.call(ADD, vec![node(tx), node(ty)]);
    g.output(&[add]);
    g
}

/// `_concat_transform_graph`: `cat([x transform, y transform], dim)`.
fn concat_transform_graph(
    x_shape: &[i64],
    y_shape: &[i64],
    x_transform: Transform<'_>,
    y_transform: Transform<'_>,
    dim: i64,
) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", x_shape);
    let y = g.input("y", y_shape);
    let tx = apply(&mut g, x, &[x_transform]);
    let ty = apply(&mut g, y, &[y_transform]);
    let cat = g.cat(&[tx, ty], dim);
    g.output(&[cat]);
    g
}

/// `test_fuse_identical_input_transforms_sinks_view`.
#[test]
fn fuse_identical_input_transforms_sinks_view() {
    let mut g = binary_transform_graph(&[6], &[6], &[Transform::View(&[2, 3])], &[Transform::View(&[2, 3])]);

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(g.count(VIEW), 1);
    g.assert_targets(&[ADD, VIEW]);
}

/// `test_fuse_identical_input_transforms_sunk_view_keeps_delegation_tag`.
#[test]
fn fuse_identical_input_transforms_sunk_view_keeps_delegation_tag() {
    let mut g = binary_transform_graph(&[6], &[6], &[Transform::View(&[2, 3])], &[Transform::View(&[2, 3])]);
    let add = calls_to(&g, ADD)[0];
    set_tag(&mut g, add, "tag0");

    assert!(apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[ADD, VIEW]);
    assert_eq!(tag(&g, calls_to(&g, ADD)[0]).as_deref(), Some("tag0"));
    assert_eq!(tag(&g, calls_to(&g, VIEW)[0]).as_deref(), Some("tag0"));
}

/// `test_fuse_identical_input_transforms_sinks_permute`.
#[test]
fn fuse_identical_input_transforms_sinks_permute() {
    let mut g = binary_transform_graph(
        &[2, 3, 4],
        &[2, 3, 4],
        &[Transform::Permute(&[0, 2, 1])],
        &[Transform::Permute(&[0, 2, 1])],
    );

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(g.count(PERMUTE), 1);
    g.assert_targets(&[ADD, PERMUTE]);
}

/// `test_fuse_identical_input_transforms_sunk_permute_replaces_stale_tag`.
#[test]
fn fuse_identical_input_transforms_sunk_permute_replaces_stale_tag() {
    let mut g = binary_transform_graph(
        &[2, 3, 4],
        &[2, 3, 4],
        &[Transform::Permute(&[0, 2, 1])],
        &[Transform::Permute(&[0, 2, 1])],
    );
    for id in calls_to(&g, PERMUTE) {
        set_tag(&mut g, id, "old_tag");
    }
    let add = calls_to(&g, ADD)[0];
    set_tag(&mut g, add, "tag0");

    assert!(apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[ADD, PERMUTE]);
    assert_eq!(tag(&g, calls_to(&g, PERMUTE)[0]).as_deref(), Some("tag0"));
}

/// Not an upstream test: an op without a tag drops the transform's tag
/// (`_new_transform_meta`, `:275`).
#[test]
fn fuse_identical_input_transforms_sunk_transform_drops_a_tag_the_op_lacks() {
    let mut g = binary_transform_graph(
        &[2, 3, 4],
        &[2, 3, 4],
        &[Transform::Permute(&[0, 2, 1])],
        &[Transform::Permute(&[0, 2, 1])],
    );
    for id in calls_to(&g, PERMUTE) {
        set_tag(&mut g, id, "old_tag");
    }

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(tag(&g, calls_to(&g, PERMUTE)[0]), None);
}

/// `test_fuse_identical_input_transforms_sinks_where_input_permutes`.
#[test]
fn fuse_identical_input_transforms_sinks_where_input_permutes() {
    let mut g = LayoutGraph::new();
    let c = g.typed_input("condition", &[2, 3, 4], "torch.bool");
    let x = g.input("x", &[2, 3, 4]);
    let y = g.input("y", &[2, 3, 4]);
    let pc = g.permute(c, &[0, 2, 1]);
    let px = g.permute(x, &[0, 2, 1]);
    let py = g.permute(y, &[0, 2, 1]);
    let w = g.call(WHERE, vec![node(pc), node(px), node(py)]);
    g.output(&[w]);

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(g.count(PERMUTE), 1);
    g.assert_targets(&[WHERE, PERMUTE]);
}

/// `cat([view(x, [6, 4]), view(y, [6, 4])], 1)` on two `[2, 3, 4]` inputs.
fn cat_input_views() -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let y = g.input("y", &[2, 3, 4]);
    let vx = g.view(x, &[6, 4]);
    let vy = g.view(y, &[6, 4]);
    let cat = g.cat(&[vx, vy], 1);
    g.output(&[cat]);
    g
}

/// `test_fuse_identical_input_transforms_sinks_cat_input_views`: the cat dim
/// maps to 2, the cat reads the inputs and the view goes to `[6, 8]`.
#[test]
fn fuse_identical_input_transforms_sinks_cat_input_views() {
    let mut g = cat_input_views();

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(g.count(VIEW), 1);
    g.assert_targets(&[CAT, VIEW]);
    let cat = calls_to(&g, CAT)[0];
    let op = g.graph().node(cat);
    let IrArg::List(items) = &op.args[0] else {
        panic!("cat takes a list");
    };
    let names: Vec<String> = items
        .iter()
        .map(|item| g.name(item.as_node().unwrap()))
        .collect();
    assert_eq!(names, ["x", "y"]);
    assert_eq!(op.args[1].as_int(), Some(2));
    assert_eq!(g.shape(cat), &[2, 3, 8]);
    let view = calls_to(&g, VIEW)[0];
    let args = &g.graph().node(view).args;
    assert_eq!((args[0].as_node(), args[1].as_int_list()), (Some(cat), Some(vec![6, 8])));
}

/// `test_fuse_identical_input_transforms_sunk_cat_view_keeps_delegation_tag`.
#[test]
fn fuse_identical_input_transforms_sunk_cat_view_keeps_delegation_tag() {
    let mut g = cat_input_views();
    let cat = calls_to(&g, CAT)[0];
    set_tag(&mut g, cat, "tag0");

    assert!(apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[CAT, VIEW]);
    assert_eq!(tag(&g, calls_to(&g, VIEW)[0]).as_deref(), Some("tag0"));
}

/// `x` and `y` placeholders with `add(x, y)` (and `sub(x, y)`), whose `x` has
/// the given name and target, as rc4's normalization tests set them up.
fn placeholder_graph(x_name: &str, x_target: &str, with_sub: bool) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input(x_name, &[2, 3]);
    let y = g.input("y", &[2, 3]);
    g.graph_mut().node_mut(x).target = x_target.to_string();
    let add = g.call(ADD, vec![node(x), node(y)]);
    let mut outputs = vec![add];
    if with_sub {
        outputs.push(g.call(SUB, vec![node(x), node(y)]));
    }
    g.output(&outputs);
    g
}

fn placeholder_target(g: &LayoutGraph, name: &str) -> String {
    g.graph().node(g.id(name)).target.clone()
}

/// `test_fuse_identical_input_transforms_preserves_transform_user_input_names`:
/// placeholders named `aten_view_copy_default` and `aten_view_copy_default_1`,
/// both with target `aten_view_copy_default`, end with targets equal to their
/// names. Not ported: the input specs and `process_placeholder` (see the
/// module docs).
#[test]
fn fuse_identical_input_transforms_preserves_transform_user_input_names() {
    let mut g = LayoutGraph::new();
    let x = g.input("aten_view_copy_default", &[2, 3]);
    let y = g.input("aten_view_copy_default_1", &[2, 3]);
    g.graph_mut().node_mut(y).target = "aten_view_copy_default".to_string();
    let add = g.call(ADD, vec![node(x), node(y)]);
    g.output(&[add]);

    assert!(IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));

    assert_eq!(placeholder_target(&g, "aten_view_copy_default"), "aten_view_copy_default");
    assert_eq!(placeholder_target(&g, "aten_view_copy_default_1"), "aten_view_copy_default_1");
}

/// `test_fuse_identical_input_transforms_normalizes_multi_user_transform_input`.
#[test]
fn fuse_identical_input_transforms_normalizes_multi_user_transform_input() {
    let mut g = placeholder_graph("aten_view_copy_default_2", "aten_view_copy_default", true);

    assert!(IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));

    assert_eq!(placeholder_target(&g, "aten_view_copy_default_2"), "aten_view_copy_default_2");
}

/// `test_normalize_transform_input_placeholders_does_not_sink_transforms`:
/// `NormalizeTransformInputPlaceholdersPass` alone renames the placeholder
/// target and leaves both views.
#[test]
fn normalize_transform_input_placeholders_does_not_sink_transforms() {
    let mut g = binary_transform_graph(&[6], &[6], &[Transform::View(&[2, 3])], &[Transform::View(&[2, 3])]);
    let x = g.id("x");
    g.graph_mut().node_mut(x).name = "aten_view_copy_default_2".to_string();
    g.graph_mut().node_mut(x).target = "aten_view_copy_default".to_string();

    let mut ctx = LayoutCtx::new(g.graph_mut());
    assert!(normalize_transform_input_placeholders(&mut ctx, &[PERMUTE_TARGET]));
    ctx.finish();

    assert_eq!(g.graph().node(x).target, "aten_view_copy_default_2");
    assert_eq!(g.count(VIEW), 2);
    g.assert_targets(&[VIEW, VIEW, ADD]);
}

/// `test_fuse_identical_input_transforms_normalizes_transform_input_signature`.
#[test]
fn fuse_identical_input_transforms_normalizes_transform_input_signature() {
    let mut g = placeholder_graph("aten_view_copy_default_2", "aten_view_copy_default", false);

    assert!(IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));

    assert_eq!(placeholder_target(&g, "aten_view_copy_default_2"), "aten_view_copy_default_2");
}

/// `test_fuse_identical_input_transforms_normalizes_non_user_transform_placeholder`:
/// the same rename when the placeholder is not a user input (in rc4 its spec
/// is absent; RETAB has none either way).
#[test]
fn fuse_identical_input_transforms_normalizes_non_user_transform_placeholder() {
    let mut g = placeholder_graph("aten_view_copy_default_2", "aten_view_copy_default", false);

    assert!(IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));

    assert_eq!(placeholder_target(&g, "aten_view_copy_default_2"), "aten_view_copy_default_2");
}

/// `test_fuse_identical_input_transforms_ignores_non_transform_placeholder_names`.
#[test]
fn fuse_identical_input_transforms_ignores_non_transform_placeholder_names() {
    let mut g = placeholder_graph("custom_view_copy_default", "custom_view_copy_default_base", false);

    assert!(!IrFuseIdenticalInputTransformsPass.run(g.graph_mut()));

    assert_eq!(placeholder_target(&g, "custom_view_copy_default"), "custom_view_copy_default_base");
}

/// `test_fuse_identical_input_transforms_rejects_cat_mixed_transforms`.
#[test]
fn fuse_identical_input_transforms_rejects_cat_mixed_transforms() {
    let mut g = concat_transform_graph(
        &[2, 3, 4],
        &[2, 3, 4],
        Transform::View(&[2, 3, 4]),
        Transform::Permute(&[0, 1, 2]),
        0,
    );

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[VIEW, PERMUTE, CAT]);
}

/// `test_fuse_identical_input_transforms_rejects_cat_different_permutes`.
#[test]
fn fuse_identical_input_transforms_rejects_cat_different_permutes() {
    let mut g = concat_transform_graph(
        &[2, 3, 4],
        &[2, 4, 3],
        Transform::Permute(&[0, 1, 2]),
        Transform::Permute(&[0, 2, 1]),
        0,
    );

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[PERMUTE, PERMUTE, CAT]);
}

/// `test_fuse_identical_input_transforms_rejects_shared_binary_input_transform`.
#[test]
fn fuse_identical_input_transforms_rejects_shared_binary_input_transform() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[6]);
    let y = g.input("y", &[6]);
    let vx = g.view(x, &[2, 3]);
    let vy = g.view(y, &[2, 3]);
    let add = g.call(ADD, vec![node(vx), node(vy)]);
    g.output(&[add, vx]);

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[VIEW, VIEW, ADD]);
}

/// `test_fuse_identical_input_transforms_rejects_shared_cat_input_transform`.
#[test]
fn fuse_identical_input_transforms_rejects_shared_cat_input_transform() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let y = g.input("y", &[2, 3, 4]);
    let vx = g.view(x, &[6, 4]);
    let vy = g.view(y, &[6, 4]);
    let cat = g.cat(&[vx, vy], 1);
    g.output(&[cat, vx]);

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[VIEW, VIEW, CAT]);
}

/// `test_fuse_identical_input_transforms_rejects_different_views`.
#[test]
fn fuse_identical_input_transforms_rejects_different_views() {
    let mut g = binary_transform_graph(&[6], &[6], &[Transform::View(&[2, 3])], &[Transform::View(&[1, 2, 3])]);

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[VIEW, VIEW, ADD]);
}

/// `test_fuse_identical_input_transforms_rejects_different_permutes`.
#[test]
fn fuse_identical_input_transforms_rejects_different_permutes() {
    let mut g = binary_transform_graph(
        &[2, 1, 3],
        &[2, 1, 3],
        &[Transform::Permute(&[0, 2, 1])],
        &[Transform::Permute(&[0, 1, 2])],
    );

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[PERMUTE, PERMUTE, ADD]);
}

/// `test_fuse_identical_input_transforms_sinks_view_permute_chain`.
#[test]
fn fuse_identical_input_transforms_sinks_view_permute_chain() {
    let chain = [Transform::View(&[2, 3, 4]), Transform::Permute(&[0, 2, 1])];
    let mut g = binary_transform_graph(&[6, 4], &[6, 4], &chain, &chain);

    assert!(apply_fuse_identical_input_transforms(&mut g));

    g.assert_counts(1, 1);
    g.assert_targets(&[ADD, VIEW, PERMUTE]);
}

/// `test_fuse_identical_input_transforms_sinks_permute_view_chain`.
#[test]
fn fuse_identical_input_transforms_sinks_permute_view_chain() {
    let chain = [Transform::Permute(&[0, 2, 1]), Transform::View(&[2, 2, 6])];
    let mut g = binary_transform_graph(&[2, 3, 4], &[2, 3, 4], &chain, &chain);

    assert!(apply_fuse_identical_input_transforms(&mut g));

    g.assert_counts(1, 1);
    g.assert_targets(&[ADD, PERMUTE, VIEW]);
}

/// `test_fuse_identical_input_transforms_sinks_views_through_rank_broadcasts`.
#[test]
fn fuse_identical_input_transforms_sinks_views_through_rank_broadcasts() {
    for (x_shape, y_shape, view) in [
        (&[1, 6][..], &[6][..], &[2, 3][..]),
        (&[1, 1, 6], &[1, 6], &[3, 2]),
    ] {
        let mut g = binary_transform_graph(x_shape, y_shape, &[Transform::View(view)], &[Transform::View(view)]);

        assert!(apply_fuse_identical_input_transforms(&mut g), "{x_shape:?} {y_shape:?}");

        g.assert_targets(&[ADD, VIEW]);
    }
}

/// `test_fuse_identical_input_transforms_rejects_views_through_expanding_broadcast`.
#[test]
fn fuse_identical_input_transforms_rejects_views_through_expanding_broadcast() {
    for (x_shape, y_shape, view) in [
        (&[1, 6][..], &[6, 1][..], &[2, 3][..]),
        (&[1, 1, 6], &[1, 6, 1], &[3, 2]),
        (&[6], &[6, 1], &[3, 2]),
    ] {
        let mut g = binary_transform_graph(x_shape, y_shape, &[Transform::View(view)], &[Transform::View(view)]);

        assert!(!apply_fuse_identical_input_transforms(&mut g), "{x_shape:?} {y_shape:?}");

        g.assert_targets(&[VIEW, VIEW, ADD]);
    }
}

/// `test_fuse_identical_input_transforms_sinks_permutes_through_broadcasts`.
#[test]
fn fuse_identical_input_transforms_sinks_permutes_through_broadcasts() {
    for (x_shape, y_shape) in [
        (&[1, 2, 3][..], &[4, 2, 3][..]),
        (&[4, 1, 3], &[1, 2, 3]),
        (&[4, 2, 1], &[1, 1, 3]),
    ] {
        let mut g = binary_transform_graph(
            x_shape,
            y_shape,
            &[Transform::Permute(&[0, 2, 1])],
            &[Transform::Permute(&[0, 2, 1])],
        );

        assert!(apply_fuse_identical_input_transforms(&mut g), "{x_shape:?} {y_shape:?}");

        g.assert_targets(&[ADD, PERMUTE]);
    }
}

/// `test_cat_rejects_different_view_inputs`.
#[test]
fn cat_rejects_different_view_inputs() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3, 4]);
    let y = g.input("y", &[1, 6, 4]);
    let vx = g.view(x, &[6, 4]);
    let vy = g.view(y, &[6, 4]);
    let cat = g.cat(&[vx, vy], 1);
    g.output(&[cat]);

    assert!(!apply_fuse_identical_input_transforms(&mut g));

    g.assert_targets(&[VIEW, VIEW, CAT]);
}

/// Not an upstream test: the evaluator also checks the sinks the rc4 tests
/// don't run numerics for, through `where`, comparisons, kwargs and integer
/// ops.
#[test]
fn fuse_identical_input_transforms_keeps_numerics_across_ops() {
    let mut g = LayoutGraph::new();
    let x = g.input("x", &[2, 3]);
    let y = g.input("y", &[2, 3]);
    let ones = g.input("ones", &[1, 1]);
    let px = g.permute(x, &[1, 0]);
    let py = g.permute(y, &[1, 0]);
    let gt = g.call(GT, vec![node(px), node(py)]);
    let pz = g.permute(x, &[1, 0]);
    let pw = g.permute(y, &[1, 0]);
    let sub = g.call_with_kwargs(SUB, vec![node(pz), node(pw)], &[("alpha", IrArg::Float(2.0))]);
    let scaled = g.call(MUL, vec![node(sub), node(ones)]);
    let a = g.typed_input("a", &[2, 3], "torch.int32");
    let b = g.typed_input("b", &[2, 3], "torch.int32");
    let va = g.view(a, &[6]);
    let vb = g.view(b, &[6]);
    let and = g.call("aten.bitwise_and.Tensor", vec![node(va), node(vb)]);
    g.output(&[gt, scaled, and]);

    assert!(apply_fuse_identical_input_transforms(&mut g));

    assert_eq!(g.count(PERMUTE), 2);
    assert_eq!(g.count(VIEW), 1);
}

// ---------------------------------------------------------------------------
// MoveDataMovementOpsToSmallerDtypePass
// (backends/arm/test/passes/test_move_data_movement_ops_to_smaller_dtype_pass.py)
// ---------------------------------------------------------------------------

fn rescale_with(g: &mut LayoutGraph, x: NodeId, dtype: &str, scales: &[f64]) -> NodeId {
    g.call(
        RESCALE,
        vec![
            node(x),
            IrArg::Dtype(dtype.to_string()),
            IrArg::List(scales.iter().map(|&scale| IrArg::Float(scale)).collect()),
            IrArg::Int(0),
            IrArg::Int(0),
        ],
    )
}

fn run_mdm(g: &mut LayoutGraph) -> bool {
    IrMoveDataMovementOpsToSmallerDtypePass.run(g.graph_mut())
}

/// `test_moves_permute_after_narrowing_rescale`.
#[test]
fn moves_permute_after_narrowing_rescale() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let permute = g.permute(x, &[0, 2, 3, 1]);
    let rescale = rescale_with(&mut g, permute, "torch.int8", &[1.0]);
    g.output(&[rescale]);

    assert!(run_mdm(&mut g));

    let [rescale, permute] = calls(&g)[..] else {
        panic!("two calls");
    };
    assert_eq!(g.graph().node(rescale).target, RESCALE);
    assert_eq!(g.graph().node(permute).target, PERMUTE);
    assert_eq!(g.shape(rescale), &[1, 2, 3, 4]);
    assert_eq!(g.dtype(permute), "torch.int8");
}

/// `test_moves_permute_before_widening_rescale`.
#[test]
fn moves_permute_before_widening_rescale() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int8");
    let rescale = rescale_with(&mut g, x, "torch.int32", &[1.0]);
    let permute = g.permute(rescale, &[0, 2, 3, 1]);
    g.output(&[permute]);

    assert!(run_mdm(&mut g));

    let [permute, rescale] = calls(&g)[..] else {
        panic!("two calls");
    };
    assert_eq!(g.graph().node(permute).target, PERMUTE);
    assert_eq!(g.graph().node(rescale).target, RESCALE);
    assert_eq!(g.dtype(permute), "torch.int8");
    assert_eq!(g.shape(rescale), &[1, 3, 4, 2]);
}

/// `test_keeps_per_channel_rescale_order`.
#[test]
fn keeps_per_channel_rescale_order() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let permute = g.permute(x, &[0, 2, 3, 1]);
    let rescale = rescale_with(&mut g, permute, "torch.int8", &[1.0, 2.0]);
    g.output(&[rescale]);

    assert!(!run_mdm(&mut g));

    g.assert_targets(&[PERMUTE, RESCALE]);
}

/// `test_splits_merged_permute_at_distinct_narrowing_rescales`, from the graph
/// MoveDataMovementOps receives: rc4's PropagateViewCopyPermuteDown/Up leave
/// this graph's one permute and its two RESCALE users as they are (checked
/// in the venv; the propagation passes are ported separately). Each RESCALE
/// gets its own int8 permute.
#[test]
fn splits_merged_permute_at_distinct_narrowing_rescales() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let permute = g.permute(x, &[0, 2, 3, 1]);
    let r1 = rescale_with(&mut g, permute, "torch.int8", &[1.0]);
    let r2 = rescale_with(&mut g, permute, "torch.int8", &[2.0]);
    g.output(&[r1, r2]);
    assert_eq!(g.count(PERMUTE), 1);

    assert!(run_mdm(&mut g));

    let permutes = calls_to(&g, PERMUTE);
    assert_eq!(permutes.len(), 2);
    for permute in permutes {
        assert_eq!(g.dtype(permute), "torch.int8");
        assert_eq!(g.graph().node(input_of(&g, permute)).target, RESCALE);
    }
}

// From backends/arm/test/passes/test_propagate_permutes_views_pass.py, which
// runs PropagateViewCopyPermuteUpPass and then MoveDataMovementOps: each port
// starts from the graph the Up pass hands MoveDataMovementOps there (captured
// in the venv) and checks what it returns.

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_fed_by_binary_op`:
/// the Up pass leaves `add -> permute -> RESCALE:i8`; the permute moves back
/// below the RESCALE.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_fed_by_binary_op() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let y = g.typed_input("y", &[1, 2, 3, 4], "torch.int32");
    let add = g.call(ADD, vec![node(x), node(y)]);
    let permute = g.permute(add, &[0, 2, 3, 1]);
    let rescale = rescale_with(&mut g, permute, "torch.int8", &[1.0]);
    g.output(&[rescale]);

    assert!(run_mdm(&mut g));

    g.assert_targets(&[ADD, RESCALE, PERMUTE]);
    let [_, rescale, permute] = calls(&g)[..] else {
        panic!("three calls");
    };
    assert_eq!((g.shape(rescale), g.dtype(rescale)), (&[1, 2, 3, 4][..], "torch.int8"));
    assert_eq!((g.shape(permute), g.dtype(permute)), (&[1, 3, 4, 2][..], "torch.int8"));
}

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_from_shared_placeholder`:
/// the placeholder's other user, a `neg`, is left alone.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_from_shared_placeholder() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let permute = g.permute(x, &[0, 2, 3, 1]);
    let rescale = rescale_with(&mut g, permute, "torch.int8", &[1.0]);
    let neg = g.call(NEG, vec![node(x)]);
    g.output(&[rescale, neg]);

    assert!(run_mdm(&mut g));

    g.assert_targets(&[RESCALE, PERMUTE, NEG]);
    let [rescale, permute, neg] = calls(&g)[..] else {
        panic!("three calls");
    };
    assert_eq!(input_of(&g, rescale), g.id("x"));
    assert_eq!(input_of(&g, permute), rescale);
    assert_eq!((input_of(&g, neg), g.dtype(neg)), (g.id("x"), "torch.int32"));
    assert_eq!(output_nodes(g.graph()), vec![permute, neg]);
}

/// `test_smaller_dtype_pass_restores_permute_after_narrowing_rescale_behind_unary`:
/// the Up pass leaves `permute -> clone -> RESCALE:i8`; the clone moves below
/// the RESCALE first, then the permute, and the clone ends up on int8.
#[test]
fn smaller_dtype_pass_restores_permute_after_narrowing_rescale_behind_unary() {
    let mut g = LayoutGraph::new();
    let x = g.typed_input("x", &[1, 2, 3, 4], "torch.int32");
    let permute = g.permute(x, &[0, 2, 3, 1]);
    let clone = g.call(CLONE, vec![node(permute)]);
    let rescale = rescale_with(&mut g, clone, "torch.int8", &[1.0]);
    g.output(&[rescale]);

    assert!(run_mdm(&mut g));

    g.assert_targets(&[RESCALE, PERMUTE, CLONE]);
    let [rescale, permute, clone] = calls(&g)[..] else {
        panic!("three calls");
    };
    assert_eq!((g.shape(rescale), g.dtype(rescale)), (&[1, 2, 3, 4][..], "torch.int8"));
    assert_eq!((g.shape(permute), g.dtype(permute)), (&[1, 3, 4, 2][..], "torch.int8"));
    assert_eq!((g.shape(clone), g.dtype(clone)), (&[1, 3, 4, 2][..], "torch.int8"));
}
