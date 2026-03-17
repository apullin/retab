use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::smallvec;

/// Ensure binary ops have matching dtypes on both inputs. If one input is Int8
/// and the other is Int32, insert a TosaRescale (cast) to promote the narrower
/// type to Int32. This is especially important after quantization where scales
/// may differ.
pub struct MatchArgDtypePass;

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

/// Return the "wider" dtype when promoting for a binary op.
fn promote_dtype(a: DType, b: DType) -> DType {
    use DType::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Float32, _) | (_, Float32) => Float32,
        (Int32, _) | (_, Int32) => Int32,
        (Int64, _) | (_, Int64) => Int64,
        (Int16, _) | (_, Int16) => Int16,
        _ => a,
    }
}

impl Pass for MatchArgDtypePass {
    fn name(&self) -> &str {
        "MatchArgDtype"
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

            let lhs_dtype = graph.node(lhs_id).map(|n| n.meta.val.dtype).unwrap_or(DType::Float32);
            let rhs_dtype = graph.node(rhs_id).map(|n| n.meta.val.dtype).unwrap_or(DType::Float32);

            if lhs_dtype == rhs_dtype {
                continue;
            }

            let target_dtype = promote_dtype(lhs_dtype, rhs_dtype);

            // Insert a cast for the input that needs promotion
            let (cast_input_id, arg_idx) = if lhs_dtype != target_dtype {
                (lhs_id, 0usize)
            } else {
                (rhs_id, 1usize)
            };

            let cast_input_info = match graph.node(cast_input_id) {
                Some(n) => n.meta.val,
                None => continue,
            };

            let node_name = graph
                .node(node_id)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            let cast_name = format!("{}_dtype_cast_{}", node_name, arg_idx);

            let cast_meta = NodeMeta::new(TensorInfo::new(
                cast_input_info.shape_slice(),
                target_dtype,
            ));

            // Insert a TosaRescale node that casts the dtype
            let cast_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::TosaRescale,
                smallvec![Arg::Node(cast_input_id)],
                vec![
                    ("input_zp".into(), Arg::Int(0)),
                    ("output_zp".into(), Arg::Int(0)),
                ],
                cast_meta,
                cast_name,
            );

            shape::propagate_shape(graph, cast_id);

            // Rewire
            if let Some(node) = graph.node_mut(node_id) {
                if let Some(Arg::Node(id)) = node.args.get_mut(arg_idx) {
                    *id = cast_id;
                }
            }

            // Update user lists
            if let Some(c_node) = graph.node_mut(cast_id) {
                if !c_node.users.contains(&node_id) {
                    c_node.users.push(node_id);
                }
            }
            if let Some(src_node) = graph.node_mut(cast_input_id) {
                src_node.users.retain(|uid| *uid != node_id);
                if !src_node.users.contains(&cast_id) {
                    src_node.users.push(cast_id);
                }
            }

            modified = true;
        }

        modified
    }
}
