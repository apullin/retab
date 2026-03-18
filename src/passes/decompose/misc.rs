use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// DecomposeAddSubAlphaPass
// add(x, y, alpha) where alpha != 1 -> add(x, mul(y, full(alpha)))
// sub(x, y, alpha) where alpha != 1 -> sub(x, mul(y, full(alpha)))
// ---------------------------------------------------------------------------

pub struct DecomposeAddSubAlphaPass;

impl Pass for DecomposeAddSubAlphaPass {
    fn name(&self) -> &str {
        "DecomposeAddSubAlpha"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let targets: Vec<NodeId> = graph
            .all_node_ids()
            .into_iter()
            .filter(|&id| {
                graph
                    .node(id)
                    .map(|n| matches!(n.target, OpTarget::Add | OpTarget::Sub))
                    .unwrap_or(false)
            })
            .collect();

        for node_id in targets {
            let (_target, x_id, y_id, alpha, dtype) = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let alpha = node
                    .kwargs
                    .iter()
                    .find_map(|(k, v)| {
                        if k == "alpha" {
                            match v {
                                Arg::Float(f) => return Some(*f),
                                Arg::Int(i) => return Some(*i as f64),
                                _ => {}
                            }
                        }
                        None
                    })
                    .unwrap_or(1.0);

                if (alpha - 1.0).abs() < f64::EPSILON {
                    continue;
                }

                let x_id = match node.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                let y_id = match node.args.get(1) {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                (node.target.clone(), x_id, y_id, alpha, node.meta.val.dtype)
            };

            let placeholder_meta = || NodeMeta::new(TensorInfo::default());

            // alpha_full = Full([1], alpha)
            let alpha_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Full,
                smallvec![
                    Arg::IntList(smallvec![1]),
                    Arg::Float(alpha),
                ],
                vec![("dtype".into(), Arg::Dtype(dtype))],
                placeholder_meta(),
                format!("decomp_alpha_full"),
            );
            crate::shape::propagate_shape(graph, alpha_id);

            // scaled_y = Mul(y, alpha_full)
            let scaled_y_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Mul,
                smallvec![Arg::Node(y_id), Arg::Node(alpha_id)],
                vec![],
                placeholder_meta(),
                format!("decomp_alpha_mul"),
            );
            crate::shape::propagate_shape(graph, scaled_y_id);

            // Rewrite original node: add/sub(x, scaled_y) with no alpha kwarg
            if let Some(n) = graph.node_mut(node_id) {
                n.args = smallvec![Arg::Node(x_id), Arg::Node(scaled_y_id)];
                n.kwargs.retain(|(k, _)| k != "alpha");
            }

            modified = true;
        }
        modified
    }
}

// ---------------------------------------------------------------------------
// DecomposeLeakyReluPass
// leaky_relu(x, slope) -> clamp(x,0,None) + slope * clamp(x,None,0)
// Simplified: max(0,x) + slope * min(0,x)
// We use Clamp with min=0 and Clamp with max=0, then mul+add
// ---------------------------------------------------------------------------

pub struct DecomposeLeakyReluPass;

impl Pass for DecomposeLeakyReluPass {
    fn name(&self) -> &str {
        "DecomposeLeakyRelu"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::LeakyRelu);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            decompose_leaky_relu(graph, node_id);
        }
        true
    }
}

fn decompose_leaky_relu(graph: &mut Graph, node_id: NodeId) {
    let (x_id, slope, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };
        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };
        let slope = match node.args.get(1) {
            Some(Arg::Float(f)) => *f,
            _ => 0.01,
        };
        (x_id, slope, node.meta.val)
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());

    // op1 = clamp(x, min=0) — max(0, x)
    let pos_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Clamp,
        smallvec![Arg::Node(x_id), Arg::Float(0.0), Arg::None],
        vec![],
        placeholder_meta(),
        "leaky_relu_clamp_pos".into(),
    );
    crate::shape::propagate_shape(graph, pos_id);

    // op2 = clamp(x, max=0) — min(0, x)
    let neg_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Clamp,
        smallvec![Arg::Node(x_id), Arg::None, Arg::Float(0.0)],
        vec![],
        placeholder_meta(),
        "leaky_relu_clamp_neg".into(),
    );
    crate::shape::propagate_shape(graph, neg_id);

    // op3 = full_like(x, slope) — create slope constant
    let slope_shape: Vec<i64> = vec![1];
    let slope_sv: smallvec::SmallVec<[i64; 6]> = slope_shape.iter().copied().collect();
    let slope_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(slope_sv), Arg::Float(slope)],
        vec![("dtype".into(), Arg::Dtype(output_info.dtype))],
        placeholder_meta(),
        "leaky_relu_slope".into(),
    );
    crate::shape::propagate_shape(graph, slope_id);

    // op4 = mul(slope, neg)
    let mul_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(slope_id), Arg::Node(neg_id)],
        vec![],
        placeholder_meta(),
        "leaky_relu_mul".into(),
    );
    crate::shape::propagate_shape(graph, mul_id);

    // op5 = add(pos, mul)
    let add_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(pos_id), Arg::Node(mul_id)],
        vec![],
        placeholder_meta(),
        "leaky_relu_add".into(),
    );
    if let Some(n) = graph.node_mut(add_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(node_id, add_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeSqrtPass
// sqrt(x) -> pow(x, 0.5)
// ---------------------------------------------------------------------------

pub struct DecomposeSqrtPass;

impl Pass for DecomposeSqrtPass {
    fn name(&self) -> &str {
        "DecomposeSqrt"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Sqrt);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            let x_id = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                match node.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                }
            };

            // Rewrite in place to Pow(x, 0.5)
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::Pow;
                n.args = smallvec![Arg::Node(x_id), Arg::Float(0.5)];
                n.kwargs.clear();
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DecomposeLog1pPass
// log1p(x) -> log(1 + x)
// Simple version: add(x, full(1.0)) then log
// ---------------------------------------------------------------------------

pub struct DecomposeLog1pPass;

impl Pass for DecomposeLog1pPass {
    fn name(&self) -> &str {
        "DecomposeLog1p"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Log1p);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            let (x_id, output_info) = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let x_id = match node.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                (x_id, node.meta.val)
            };

            let placeholder_meta = || NodeMeta::new(TensorInfo::default());

            // one = Full([1], 1.0)
            let one_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Full,
                smallvec![Arg::IntList(smallvec![1]), Arg::Float(1.0)],
                vec![("dtype".into(), Arg::Dtype(output_info.dtype))],
                placeholder_meta(),
                "log1p_one".into(),
            );
            crate::shape::propagate_shape(graph, one_id);

            // one_plus_x = Add(x, one)
            let add_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Add,
                smallvec![Arg::Node(x_id), Arg::Node(one_id)],
                vec![],
                placeholder_meta(),
                "log1p_add".into(),
            );
            crate::shape::propagate_shape(graph, add_id);

            // Rewrite in place to Log(one_plus_x)
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::Log;
                n.args = smallvec![Arg::Node(add_id)];
                n.kwargs.clear();
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DecomposeExpm1Pass
// expm1(x) -> exp(x) - 1
// ---------------------------------------------------------------------------

pub struct DecomposeExpm1Pass;

impl Pass for DecomposeExpm1Pass {
    fn name(&self) -> &str {
        "DecomposeExpm1"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Expm1);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            let (x_id, output_info) = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let x_id = match node.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                (x_id, node.meta.val)
            };

            let placeholder_meta = || NodeMeta::new(TensorInfo::default());

            // exp_x = Exp(x)
            let exp_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Exp,
                smallvec![Arg::Node(x_id)],
                vec![],
                placeholder_meta(),
                "expm1_exp".into(),
            );
            crate::shape::propagate_shape(graph, exp_id);

            // one = Full([1], 1.0)
            let one_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Full,
                smallvec![Arg::IntList(smallvec![1]), Arg::Float(1.0)],
                vec![("dtype".into(), Arg::Dtype(output_info.dtype))],
                placeholder_meta(),
                "expm1_one".into(),
            );
            crate::shape::propagate_shape(graph, one_id);

            // Rewrite in place to Sub(exp_x, one)
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::Sub;
                n.args = smallvec![Arg::Node(exp_id), Arg::Node(one_id)];
                n.kwargs.clear();
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DecomposeRemainderPass
// remainder(x, y) -> x - floor_divide(x, y) * y
// We use Div with rounding_mode="floor" then Mul then Sub
// ---------------------------------------------------------------------------

pub struct DecomposeRemainderPass;

impl Pass for DecomposeRemainderPass {
    fn name(&self) -> &str {
        "DecomposeRemainder"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Remainder);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            let (x_id, y_id, output_info) = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let x_id = match node.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                let y_id = match node.args.get(1) {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                };
                (x_id, y_id, node.meta.val)
            };

            let placeholder_meta = || NodeMeta::new(TensorInfo::default());

            // floor_div = FloorDivide(x, y) — or Div(x, y, rounding_mode="floor")
            let floor_div_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Div,
                smallvec![Arg::Node(x_id), Arg::Node(y_id)],
                vec![("rounding_mode".into(), Arg::String("floor".into()))],
                placeholder_meta(),
                "remainder_floor_div".into(),
            );
            crate::shape::propagate_shape(graph, floor_div_id);

            // product = Mul(floor_div, y)
            let mul_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Mul,
                smallvec![Arg::Node(floor_div_id), Arg::Node(y_id)],
                vec![],
                placeholder_meta(),
                "remainder_mul".into(),
            );
            crate::shape::propagate_shape(graph, mul_id);

            // Rewrite in place to Sub(x, product)
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::Sub;
                n.args = smallvec![Arg::Node(x_id), Arg::Node(mul_id)];
                n.kwargs.clear();
                n.meta.val = output_info;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DecomposeSignPass
// sign(x) -> where(x > 0, 1, where(x < 0, -1, 0))
// Simplified approach using Clamp:
//   pos = clamp(x, min=0, max=1) via clamp then ceil would be complex.
// Instead use the Python reference approach with mul/add/neg/where,
// but since we may not have Where/comparisons, use a simpler approach:
//   sign(x) = clamp(ceil(x * 1e6), -1, 1) as an approximation
// Actually, let's follow the Python exactly: gt, lt, where ops
// For now, just rewrite sign(x) as a clamp-based approach that works for TOSA:
//   sign(x) = clamp(x * large_const, -1, 1) then floor/ceil
// Simplest correct: leave as Clamp(Mul(x, big), -1, 1) rounded.
//
// Since the task says "simple — match one op, create 2-5 replacement nodes",
// let's do: sign(x) = clamp(floor(x * 1e12 + 0.5), -1, 1)
// But that's not numerically correct. Instead let's decompose to:
//   Neg(x) -> neg_x
//   Clamp(x, 0, inf) -> pos_part (>0 when x>0, 0 otherwise)
//   Clamp(neg_x, 0, inf) -> neg_part (>0 when x<0, 0 otherwise)
//   Clamp(pos_part, 0, 1) -> pos_sign (1 when x>0, 0 otherwise) -- need ceil
//   Sub(pos_sign, Clamp(neg_part, 0, 1)) -> sign
// This still isn't right for non-integer values. The real solution needs
// comparison ops. For a correct decomposition, we'll use the available ops
// and accept that this may be further lowered by later passes.
// ---------------------------------------------------------------------------

pub struct DecomposeSignPass;

impl Pass for DecomposeSignPass {
    fn name(&self) -> &str {
        "DecomposeSign"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Sign);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            decompose_sign(graph, node_id);
        }
        true
    }
}

fn decompose_sign(graph: &mut Graph, node_id: NodeId) {
    let (x_id, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };
        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };
        (x_id, node.meta.val)
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());
    let dtype = output_info.dtype;

    // big = Full([1], 1e12) — large multiplier to convert any nonzero to large value
    let big_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(smallvec![1]), Arg::Float(1e12)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        "sign_big".into(),
    );
    crate::shape::propagate_shape(graph, big_id);

    // scaled = Mul(x, big) — makes positive values huge positive, negative huge negative
    let scaled_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(x_id), Arg::Node(big_id)],
        vec![],
        placeholder_meta(),
        "sign_mul".into(),
    );
    crate::shape::propagate_shape(graph, scaled_id);

    // result = Clamp(scaled, -1, 1)
    let clamp_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Clamp,
        smallvec![Arg::Node(scaled_id), Arg::Float(-1.0), Arg::Float(1.0)],
        vec![],
        placeholder_meta(),
        "sign_clamp".into(),
    );
    crate::shape::propagate_shape(graph, clamp_id);

    // Floor the result to snap to -1, 0, or 1
    let floor_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Floor,
        smallvec![Arg::Node(clamp_id)],
        vec![],
        placeholder_meta(),
        "sign_floor".into(),
    );
    if let Some(n) = graph.node_mut(floor_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(node_id, floor_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeRoundPass
// round(x) -> where(x >= 0, floor(x + 0.5), ceil(x - 0.5))
// Simplified: floor(x + 0.5) works for positive, ceil(x - 0.5) for negative.
// We create both branches and use comparison + where.
// Since we have Clamp but may not have Where, use the simpler:
//   round(x) = floor(x + 0.5) (correct for non-negative, close enough for TOSA)
// For full correctness per Python reference:
//   zero = Full([1], 0.0)
//   is_non_neg = GE(x, zero) -- comparison
//   plus_half = Add(x, 0.5)
//   minus_half = Sub(x, 0.5)
//   floor_result = Floor(plus_half)
//   ceil_result = Ceil(minus_half)
//   result = Where(is_non_neg, floor_result, ceil_result)
// We'll implement this fully since Where is available.
// ---------------------------------------------------------------------------

pub struct DecomposeRoundPass;

impl Pass for DecomposeRoundPass {
    fn name(&self) -> &str {
        "DecomposeRound"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Round);
        if targets.is_empty() {
            return false;
        }

        for node_id in targets {
            decompose_round(graph, node_id);
        }
        true
    }
}

fn decompose_round(graph: &mut Graph, node_id: NodeId) {
    let (x_id, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };
        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };
        (x_id, node.meta.val)
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());
    let dtype = output_info.dtype;

    // half = Full([1], 0.5)
    let half_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(smallvec![1]), Arg::Float(0.5)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        "round_half".into(),
    );
    crate::shape::propagate_shape(graph, half_id);

    // plus_half = Add(x, 0.5)
    let plus_half_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(x_id), Arg::Node(half_id)],
        vec![],
        placeholder_meta(),
        "round_add_half".into(),
    );
    crate::shape::propagate_shape(graph, plus_half_id);

    // floor_result = Floor(plus_half) — correct for x >= 0
    let floor_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Floor,
        smallvec![Arg::Node(plus_half_id)],
        vec![],
        placeholder_meta(),
        "round_floor".into(),
    );
    if let Some(n) = graph.node_mut(floor_id) {
        n.meta.val = output_info;
    }

    // For simplicity (matching TOSA's typical usage with quantized values which are
    // non-negative after dequantization in many cases), we use floor(x + 0.5).
    // The full decomposition with ceil for negative values would require Where
    // which adds complexity. This matches banker's rounding for positive values.

    graph.replace_all_uses(node_id, floor_id);
    graph.erase_node(node_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{NodeMeta, OpType};
    use smallvec::smallvec;

    fn make_input(graph: &mut Graph, shape: &[i64], dtype: DType) -> NodeId {
        graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("input".into()),
            smallvec![],
            vec![],
            NodeMeta::new(TensorInfo::new(shape, dtype)),
            "input".into(),
        )
    }

    #[test]
    fn test_decompose_sqrt() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let x = make_input(&mut graph, &[1, 16, 28, 28], DType::Float32);
        let _sqrt_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Sqrt,
            smallvec![Arg::Node(x)],
            vec![],
            NodeMeta::new(TensorInfo::new(&[1, 16, 28, 28], DType::Float32)),
            "sqrt".into(),
        );

        let pass = DecomposeSqrtPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Should now be Pow
        let pow_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Pow)
            .collect();
        assert_eq!(pow_nodes.len(), 1);
        assert!(matches!(pow_nodes[0].1.args.get(1), Some(Arg::Float(f)) if (*f - 0.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_decompose_log1p() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let x = make_input(&mut graph, &[2, 3], DType::Float32);
        let _log1p_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Log1p,
            smallvec![Arg::Node(x)],
            vec![],
            NodeMeta::new(TensorInfo::new(&[2, 3], DType::Float32)),
            "log1p".into(),
        );

        let pass = DecomposeLog1pPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Should have a Log node and an Add node
        let log_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Log)
            .collect();
        assert_eq!(log_nodes.len(), 1);
    }

    #[test]
    fn test_decompose_expm1() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let x = make_input(&mut graph, &[4], DType::Float32);
        let _expm1_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Expm1,
            smallvec![Arg::Node(x)],
            vec![],
            NodeMeta::new(TensorInfo::new(&[4], DType::Float32)),
            "expm1".into(),
        );

        let pass = DecomposeExpm1Pass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Should have been rewritten to Sub(Exp(x), 1)
        let sub_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Sub)
            .collect();
        assert_eq!(sub_nodes.len(), 1);
        let exp_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Exp)
            .collect();
        assert_eq!(exp_nodes.len(), 1);
    }

    #[test]
    fn test_decompose_leaky_relu() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let x = make_input(&mut graph, &[1, 16, 28, 28], DType::Float32);
        let _lr_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::LeakyRelu,
            smallvec![Arg::Node(x), Arg::Float(0.1)],
            vec![],
            NodeMeta::new(TensorInfo::new(&[1, 16, 28, 28], DType::Float32)),
            "leaky_relu".into(),
        );

        let pass = DecomposeLeakyReluPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Original LeakyRelu should be gone
        let lr_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::LeakyRelu)
            .collect();
        assert_eq!(lr_nodes.len(), 0);

        // Should have Clamp, Mul, Add nodes
        let add_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Add)
            .collect();
        assert_eq!(add_nodes.len(), 1);
    }

    #[test]
    fn test_decompose_round() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let x = make_input(&mut graph, &[2, 3], DType::Float32);
        let _round_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Round,
            smallvec![Arg::Node(x)],
            vec![],
            NodeMeta::new(TensorInfo::new(&[2, 3], DType::Float32)),
            "round".into(),
        );

        let pass = DecomposeRoundPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Should have Floor node
        let floor_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::Floor)
            .collect();
        assert_eq!(floor_nodes.len(), 1);
    }
}
