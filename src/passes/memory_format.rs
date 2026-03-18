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

/// Returns true for ops that are "format-transparent".
fn is_format_transparent(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaRescale
            | OpTarget::TosaClamp
            | OpTarget::TosaAdd
            | OpTarget::TosaSub
            | OpTarget::TosaMul
            | OpTarget::TosaNeg
            | OpTarget::TosaAbs
            | OpTarget::TosaTable
            | OpTarget::TosaReduceSum
            | OpTarget::TosaReduceMax
            | OpTarget::Add
            | OpTarget::Sub
            | OpTarget::Mul
            | OpTarget::Clamp
            | OpTarget::Relu
    )
}

/// Walk backwards through format-transparent ops to check if the chain
/// ultimately originates from an NHWC source.
fn input_produces_nhwc(graph: &Graph, start_id: NodeId) -> bool {
    let mut current = start_id;
    let mut depth = 0;
    loop {
        if depth > 20 {
            return false;
        }
        let node = match graph.node(current) {
            Some(n) => n,
            None => return false,
        };
        if node.target == OpTarget::TosaTranspose || is_nhwc_op(&node.target) {
            return true;
        }
        if !is_format_transparent(&node.target) {
            return false;
        }
        match node.args.first() {
            Some(Arg::Node(id)) => {
                current = *id;
                depth += 1;
            }
            _ => return false,
        }
    }
}

/// Walk forwards through format-transparent ops to check if ALL paths
/// eventually reach an NHWC op. If so, no NHWC→NCHW transpose is needed.
fn all_users_consume_nhwc(graph: &Graph, node_id: NodeId) -> bool {
    let users = match graph.node(node_id) {
        Some(n) => n.users.to_vec(),
        None => return false,
    };
    if users.is_empty() {
        return false;
    }
    for &uid in &users {
        let user = match graph.node(uid) {
            Some(n) => n,
            None => return false,
        };
        if is_nhwc_op(&user.target) {
            continue;
        }
        if is_format_transparent(&user.target) {
            if all_users_consume_nhwc(graph, uid) {
                continue;
            }
        }
        return false;
    }
    true
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

            let input_id = match graph.node(node_id) {
                Some(n) => match n.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                },
                None => continue,
            };

            // Skip when spatial dimensions are 1x1
            {
                let input_shape = graph.node(input_id).map(|n| n.meta.val).unwrap_or_default();
                if input_shape.shape_len == 4 {
                    let s = input_shape.shape_slice();
                    if s[2] == 1 && s[3] == 1 {
                        continue;
                    }
                }
            }

            let input_already_nhwc = input_produces_nhwc(graph, input_id);

            if !input_already_nhwc {
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

                    if let Some(node) = graph.node_mut(node_id) {
                        if let Some(Arg::Node(id)) = node.args.first_mut() {
                            *id = transpose_id;
                        }
                    }

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

                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.tosa_dim_order = Some(SmallVec::from_slice(&[0, 2, 3, 1]));
                    }

                    modified = true;
                }
            }

            // Output transposes
            let users: Vec<NodeId> = graph
                .node(node_id)
                .map(|n| n.users.to_vec())
                .unwrap_or_default();

            let output_info = graph.node(node_id).map(|n| n.meta.val).unwrap_or_default();

            let non_nhwc_users: Vec<NodeId> = users
                .iter()
                .copied()
                .filter(|&uid| {
                    let user = match graph.node(uid) {
                        Some(n) => n,
                        None => return false,
                    };
                    if is_nhwc_op(&user.target) {
                        return false;
                    }
                    if is_format_transparent(&user.target)
                        && all_users_consume_nhwc(graph, uid)
                    {
                        return false;
                    }
                    true
                })
                .collect();

            if !non_nhwc_users.is_empty() && output_info.shape_len == 4 {
                let nchw_shape = output_info.shape_slice();

                let perm_name = format!(
                    "{}_nhwc_to_nchw",
                    graph.node(node_id).map(|n| n.name.as_str()).unwrap_or("node")
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
