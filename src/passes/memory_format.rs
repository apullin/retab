use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::{smallvec, SmallVec};

/// Insert TosaTranspose nodes to convert between NCHW (ATen) and NHWC (TOSA)
/// memory formats around TOSA conv and pool operations.
pub struct ToTosaMemoryFormatPass;

/// Targets that operate in NHWC format in TOSA.
fn is_nhwc_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaAvgPool2d
            | OpTarget::TosaMaxPool2d
    )
}

impl Pass for ToTosaMemoryFormatPass {
    fn name(&self) -> &str {
        "ToTosaMemoryFormat"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            if !is_nhwc_op(&target) {
                continue;
            }

            // Get the input node (arg 0)
            let input_id = match graph.node(node_id) {
                Some(n) => match n.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                },
                None => continue,
            };

            // Check if input already produces NHWC data (either a transpose or
            // another NHWC op). If so, skip inserting another transpose.
            let input_already_nhwc = graph
                .node(input_id)
                .map(|n| n.target == OpTarget::TosaTranspose || is_nhwc_op(&n.target))
                .unwrap_or(false);

            if !input_already_nhwc {
                // Insert NCHW->NHWC transpose before this node
                let input_info = graph.node(input_id).map(|n| n.meta.val).unwrap_or_default();

                if input_info.shape_len == 4 {
                    let nchw_shape = input_info.shape_slice();
                    let nhwc_shape = [nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1]];

                    let perm_name = format!("{}_nchw_to_nhwc", graph.node(node_id).map(|n| n.name.as_str()).unwrap_or("node"));

                    let mut transpose_meta = NodeMeta::new(TensorInfo::new(&nhwc_shape, input_info.dtype));
                    transpose_meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));

                    let transpose_id = graph.insert_before(
                        node_id,
                        OpType::CallFunction,
                        OpTarget::TosaTranspose,
                        smallvec![
                            Arg::Node(input_id),
                            Arg::IntList(smallvec![0, 2, 3, 1]),
                        ],
                        vec![],
                        transpose_meta,
                        perm_name,
                    );

                    shape::propagate_shape(graph, transpose_id);

                    // Rewire: replace input_id with transpose_id in this node's arg 0
                    if let Some(node) = graph.node_mut(node_id) {
                        if let Some(Arg::Node(id)) = node.args.first_mut() {
                            *id = transpose_id;
                        }
                    }

                    // Update user lists
                    if let Some(t_node) = graph.node_mut(transpose_id) {
                        if !t_node.users.contains(&node_id) {
                            t_node.users.push(node_id);
                        }
                    }
                    if let Some(in_node) = graph.node_mut(input_id) {
                        in_node.users.retain(|uid| *uid != node_id);
                        if !in_node.users.contains(&transpose_id) {
                            in_node.users.push(transpose_id);
                        }
                    }

                    // Set dim order on the TOSA op node
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));
                    }

                    modified = true;
                }
            }

            // Check users: if any user is NOT an NHWC op, insert a single
            // shared NHWC->NCHW transpose and rewire all such users to it.
            let users: Vec<NodeId> = graph
                .node(node_id)
                .map(|n| n.users.to_vec())
                .unwrap_or_default();

            let output_info = graph.node(node_id).map(|n| n.meta.val).unwrap_or_default();

            // Collect non-NHWC users that need a transpose
            let non_nhwc_users: Vec<NodeId> = users
                .iter()
                .copied()
                .filter(|&uid| {
                    graph
                        .node(uid)
                        .map(|n| !is_nhwc_op(&n.target))
                        .unwrap_or(false)
                })
                .collect();

            if !non_nhwc_users.is_empty() && output_info.shape_len == 4 {
                let nhwc_shape = output_info.shape_slice();
                let nchw_shape = [nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]];

                let perm_name = format!(
                    "{}_nhwc_to_nchw",
                    graph
                        .node(node_id)
                        .map(|n| n.name.as_str())
                        .unwrap_or("node")
                );

                let mut transpose_meta = NodeMeta::new(TensorInfo::new(&nchw_shape, output_info.dtype));
                transpose_meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 3, 1, 2]));

                let transpose_id = graph.insert_after(
                    node_id,
                    OpType::CallFunction,
                    OpTarget::TosaTranspose,
                    smallvec![
                        Arg::Node(node_id),
                        Arg::IntList(smallvec![0, 3, 1, 2]),
                    ],
                    vec![],
                    transpose_meta,
                    perm_name,
                );

                shape::propagate_shape(graph, transpose_id);

                // Rewire ALL non-NHWC users to use the single transpose
                for &user_id in &non_nhwc_users {
                    if let Some(user) = graph.node_mut(user_id) {
                        for arg in &mut user.args {
                            if let Arg::Node(id) = arg {
                                if *id == node_id {
                                    *id = transpose_id;
                                }
                            }
                        }
                    }

                    if let Some(t_node) = graph.node_mut(transpose_id) {
                        if !t_node.users.contains(&user_id) {
                            t_node.users.push(user_id);
                        }
                    }
                    if let Some(src_node) = graph.node_mut(node_id) {
                        src_node.users.retain(|uid| *uid != user_id);
                    }
                }

                // Ensure the source node lists the transpose as a user
                if let Some(src_node) = graph.node_mut(node_id) {
                    if !src_node.users.contains(&transpose_id) {
                        src_node.users.push(transpose_id);
                    }
                }

                modified = true;
            }
        }

        modified
    }
}
