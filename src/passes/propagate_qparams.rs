use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Propagate quantization parameters through nodes that are missing them.
///
/// After `FoldAndAnnotateQParamsPass`, decomposed sub-ops (from LayerNorm, Var,
/// MeanDim decomposition) may lack `output_qparams` or `input_qparams` because
/// no Q/DQ nodes existed between them. This pass infers the missing qparams:
///
/// 1. **Forward pass**: for each node with `input_qparams` but no
///    `output_qparams`, set `output_qparams[0] = input_qparams[0]`.
///    (Most element-wise and reduction ops preserve scale.)
///
/// 2. **Backward pass**: for each node that has `output_qparams` but is
///    missing `input_qparams` for a specific arg position, look at the
///    producer's `output_qparams` and use those.
///
/// 3. **Shape-only pass-through**: nodes like Reshape, Transpose, Clamp that
///    do not change the quantization domain propagate qparams from input to
///    output even when they have no input_qparams yet — as long as their
///    single Node-typed arg producer has output_qparams.
///
/// The pass iterates until convergence so that chains of ops all get annotated.
pub struct PropagateQParamsPass;

impl Pass for PropagateQParamsPass {
    fn name(&self) -> &str {
        "PropagateQParams"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        // Iterate until convergence
        for _iteration in 0..20 {
            let mut changed_this_round = false;

            let all_ids = graph.all_node_ids();

            // Forward: propagate output_qparams from input_qparams.
            // For most ops, output_scale = input_scale[0] (pass-through).
            for &node_id in &all_ids {
                let (should_propagate_output, qp_to_set) = {
                    let node = match graph.node(node_id) {
                        Some(n) => n,
                        None => continue,
                    };

                    // Only propagate for CallFunction nodes (skip Placeholder, Output)
                    if node.op != OpType::CallFunction {
                        continue;
                    }

                    // Skip accumulator ops (they produce int32, not int8)
                    if is_accumulator_like(&node.target) {
                        continue;
                    }

                    // Skip nodes that already have output_qparams
                    if !node.meta.output_qparams.is_empty() {
                        continue;
                    }

                    // If we have input_qparams, propagate the first one to output
                    if let Some((_, qp)) = node.meta.input_qparams.first() {
                        (true, *qp)
                    } else {
                        continue;
                    }
                };

                if should_propagate_output {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.output_qparams.push((0, qp_to_set));
                        changed_this_round = true;
                        if debug {
                            eprintln!(
                                "[PropagateQParams] forward: set output_qparams on '{}' ({:?}) (scale={})",
                                node.name, node.target, qp_to_set.scale,
                            );
                        }
                    }
                }
            }

            // Backward: propagate input_qparams from producer's output_qparams.
            // Collect all updates first, then apply them to avoid borrow conflicts.
            let mut backward_updates: Vec<(NodeId, u32, QuantArgs)> = Vec::new();

            for &node_id in &all_ids {
                let (args_snapshot, existing_positions, is_call) = {
                    let node = match graph.node(node_id) {
                        Some(n) => n,
                        None => continue,
                    };
                    let is_call = node.op == OpType::CallFunction;
                    let args: Vec<(usize, Arg)> =
                        node.args.iter().enumerate().map(|(i, a)| (i, a.clone())).collect();
                    let existing: Vec<u32> =
                        node.meta.input_qparams.iter().map(|(p, _)| *p).collect();
                    (args, existing, is_call)
                };

                if !is_call {
                    continue;
                }

                for (pos, arg) in &args_snapshot {
                    let pos_u32 = *pos as u32;
                    if existing_positions.contains(&pos_u32) {
                        continue;
                    }

                    let producer_id = match arg {
                        Arg::Node(id) => *id,
                        _ => continue,
                    };

                    let producer_oqp = graph
                        .node(producer_id)
                        .and_then(|pn| {
                            pn.meta
                                .output_qparams
                                .iter()
                                .find(|(p, _)| *p == 0)
                                .map(|(_, qp)| *qp)
                        });

                    if let Some(qp) = producer_oqp {
                        backward_updates.push((node_id, pos_u32, qp));
                    }
                }
            }

            for (node_id, pos, qp) in backward_updates {
                if let Some(n) = graph.node_mut(node_id) {
                    // Double-check it wasn't already added in a prior iteration
                    if !n.meta.input_qparams.iter().any(|(p, _)| *p == pos) {
                        n.meta.input_qparams.push((pos, qp));
                        changed_this_round = true;
                        if debug {
                            eprintln!(
                                "[PropagateQParams] backward: set input_qparams[{}] on '{}' (scale={})",
                                pos, n.name, qp.scale,
                            );
                        }
                    }
                }
            }

            // Reverse-output: propagate output_qparams from consumer back to
            // producer. This handles the case where an accumulator op (Conv2d)
            // sits behind a pass-through op (Reshape) that already has
            // output_qparams. We walk forward through the user chain to find
            // the first node with output_qparams and copy it back.
            let mut reverse_updates: Vec<(NodeId, QuantArgs)> = Vec::new();

            for &node_id in &all_ids {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };

                if node.op != OpType::CallFunction {
                    continue;
                }

                // Only apply to nodes missing output_qparams
                if !node.meta.output_qparams.is_empty() {
                    continue;
                }

                // Must have input_qparams (i.e., is a quantized op)
                if node.meta.input_qparams.is_empty() {
                    continue;
                }

                // Skip accumulator-like ops — their output_qparams were
                // intentionally cleared by InsertRescalePass after inserting
                // an accumulator rescale.  Re-populating them here would cause
                // the second InsertRescalePass run to insert duplicates.
                if is_accumulator_like(&node.target) {
                    continue;
                }

                // Walk forward through single-user pass-through ops to find
                // a node with output_qparams.
                let qp = walk_users_for_output_qparams(graph, node_id);
                if let Some(qp) = qp {
                    reverse_updates.push((node_id, qp));
                }
            }

            for (node_id, qp) in reverse_updates {
                if let Some(n) = graph.node_mut(node_id) {
                    if n.meta.output_qparams.is_empty() {
                        n.meta.output_qparams.push((0, qp));
                        changed_this_round = true;
                        if debug {
                            eprintln!(
                                "[PropagateQParams] reverse-output: set output_qparams on '{}' ({:?}) (scale={})",
                                n.name, n.target, qp.scale,
                            );
                        }
                    }
                }
            }

            if changed_this_round {
                modified = true;
            } else {
                break;
            }
        }

        modified
    }
}

/// Walk forward from a node through pass-through ops (Reshape, Transpose, etc.)
/// to find the first node that has output_qparams. Returns those qparams so
/// they can be propagated back to the original node.
fn walk_users_for_output_qparams(graph: &Graph, start_id: NodeId) -> Option<QuantArgs> {
    let mut current = start_id;
    for _ in 0..20 {
        let users = match graph.node(current) {
            Some(n) => n.users.to_vec(),
            None => return None,
        };
        // Only follow single-user chains to avoid ambiguity
        if users.len() != 1 {
            return None;
        }
        let user_id = users[0];
        let user = match graph.node(user_id) {
            Some(n) => n,
            None => return None,
        };
        // Skip accumulator ops — their output_qparams represent a different
        // scale domain (post-accumulation) and must not be propagated backwards
        // to their inputs.
        if is_accumulator_like(&user.target) {
            return None;
        }
        // If the user has output_qparams, we found what we need
        if let Some((_, qp)) = user.meta.output_qparams.iter().find(|(p, _)| *p == 0) {
            return Some(*qp);
        }
        // If the user is a pass-through op, keep walking
        if is_passthrough(&user.target) {
            current = user_id;
            continue;
        }
        return None;
    }
    None
}

/// Returns true for ops that pass through quantization parameters unchanged
/// (they don't change the data domain, just reshape/reorder it).
fn is_passthrough(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Reshape
            | OpTarget::View
            | OpTarget::Permute
            | OpTarget::TosaReshape
            | OpTarget::TosaTranspose
    )
}

/// Ops that produce accumulator outputs (int32) and should NOT have their
/// input qparams propagated to output.
fn is_accumulator_like(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaFullyConnected
            | OpTarget::Linear
            | OpTarget::Addmm
            | OpTarget::Matmul
            | OpTarget::Bmm
            | OpTarget::TosaMatmul
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use smallvec::{smallvec, SmallVec};

    fn make_qp(scale: f64) -> QuantArgs {
        QuantArgs {
            scale,
            zero_point: 0,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        }
    }

    #[test]
    fn test_forward_propagation() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let input_id = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("input".into()),
            smallvec![],
            vec![],
            NodeMeta::new(TensorInfo::new(&[1, 16], DType::Int8)),
            "input".into(),
        );

        // Node with input_qparams but no output_qparams
        let mut meta = NodeMeta::new(TensorInfo::new(&[1, 16], DType::Int8));
        meta.input_qparams = SmallVec::from_vec(vec![(0, make_qp(0.01))]);
        let sub_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Sub,
            smallvec![Arg::Node(input_id), Arg::Node(input_id)],
            vec![],
            meta,
            "sub".into(),
        );

        let pass = PropagateQParamsPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        let sub_node = graph.node(sub_id).unwrap();
        assert_eq!(sub_node.meta.output_qparams.len(), 1);
        assert_eq!(sub_node.meta.output_qparams[0].1.scale, 0.01);
    }

    #[test]
    fn test_backward_propagation() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        // Producer with output_qparams
        let mut prod_meta = NodeMeta::new(TensorInfo::new(&[1, 16], DType::Int8));
        prod_meta.output_qparams = SmallVec::from_vec(vec![(0, make_qp(0.02))]);
        let producer_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![],
            vec![],
            prod_meta,
            "producer".into(),
        );

        // Consumer with no input_qparams
        let meta = NodeMeta::new(TensorInfo::new(&[1, 16], DType::Int8));
        let consumer_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![Arg::Node(producer_id)],
            vec![],
            meta,
            "consumer".into(),
        );

        let pass = PropagateQParamsPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        let consumer_node = graph.node(consumer_id).unwrap();
        assert!(consumer_node.meta.input_qparams.iter().any(|(p, _)| *p == 0));
    }
}
