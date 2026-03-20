use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::smallvec;

/// For binary ops (Add, Sub, Mul, Div and their TOSA equivalents), check if
/// input shapes need broadcasting. If one input is a scalar or lower-rank
/// tensor, insert a Reshape node to add leading 1-dims so that ranks match.
pub struct BroadcastArgsPass;

fn is_binary_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Add
            | OpTarget::Sub
            | OpTarget::Mul
            | OpTarget::Div
            | OpTarget::TosaAdd
            | OpTarget::TosaSub
            | OpTarget::TosaMul
    )
}

impl Pass for BroadcastArgsPass {
    fn name(&self) -> &str {
        "BroadcastArgs"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            if !is_binary_op(&target) {
                continue;
            }

            // Get the two input node IDs
            let (lhs_id, rhs_id) = match graph.node(node_id) {
                Some(n) => {
                    let lhs = match n.args.first() {
                        Some(Arg::Node(id)) => *id,
                        _ => continue,
                    };
                    let rhs = match n.args.get(1) {
                        Some(Arg::Node(id)) => *id,
                        _ => continue,
                    };
                    (lhs, rhs)
                }
                None => continue,
            };

            let lhs_rank = graph
                .node(lhs_id)
                .map(|n| n.meta.val.shape_len)
                .unwrap_or(0);
            let rhs_rank = graph
                .node(rhs_id)
                .map(|n| n.meta.val.shape_len)
                .unwrap_or(0);

            if lhs_rank == rhs_rank || lhs_rank == 0 || rhs_rank == 0 {
                continue;
            }

            // Determine which input needs reshaping (the lower-rank one)
            let (smaller_id, smaller_rank, larger_id, target_rank, arg_idx) = if lhs_rank < rhs_rank {
                (lhs_id, lhs_rank, rhs_id, rhs_rank, 0usize)
            } else {
                (rhs_id, rhs_rank, lhs_id, lhs_rank, 1usize)
            };

            // Build the new shape: prepend 1-dims
            let smaller_info = match graph.node(smaller_id) {
                Some(n) => n.meta.val,
                None => continue,
            };
            let old_shape = smaller_info.shape_slice();
            let leading_ones = (target_rank - smaller_rank) as usize;
            let mut new_shape_vec: Vec<i64> = vec![1; leading_ones];
            new_shape_vec.extend_from_slice(old_shape);

            // Preserve the surviving operand's logical layout. In NHWC regions
            // this keeps channel-affine reshapes channel-last for serialization.
            let larger_order = graph
                .node(larger_id)
                .and_then(|n| n.meta.tosa_dim_order.clone())
                .or_else(|| graph.node(node_id).and_then(|n| n.meta.tosa_dim_order.clone()));

            let new_shape_len = new_shape_vec.len().min(4);

            let node_name = graph
                .node(node_id)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            let reshape_name = format!("{}_broadcast_reshape_{}", node_name, arg_idx);

            let mut reshape_meta = NodeMeta::new(TensorInfo::new(
                &new_shape_vec[..new_shape_len],
                smaller_info.dtype,
            ));
            reshape_meta.tosa_dim_order = larger_order;
            let shape_arg: smallvec::SmallVec<[i64; 6]> =
                new_shape_vec[..new_shape_len].iter().copied().collect();

            let reshape_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::Reshape,
                smallvec![Arg::Node(smaller_id), Arg::IntList(shape_arg)],
                vec![],
                reshape_meta,
                reshape_name,
            );

            shape::propagate_shape(graph, reshape_id);

            // Rewire: replace smaller_id with reshape_id in the binary op's arg
            if let Some(node) = graph.node_mut(node_id) {
                if let Some(Arg::Node(id)) = node.args.get_mut(arg_idx) {
                    *id = reshape_id;
                }
            }

            // Update user lists
            if let Some(r_node) = graph.node_mut(reshape_id) {
                if !r_node.users.contains(&node_id) {
                    r_node.users.push(node_id);
                }
            }
            if let Some(s_node) = graph.node_mut(smaller_id) {
                s_node.users.retain(|uid| *uid != node_id);
                if !s_node.users.contains(&reshape_id) {
                    s_node.users.push(reshape_id);
                }
            }

            modified = true;
        }

        modified
    }
}
