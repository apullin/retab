use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::SmallVec;

/// Extract an IntList from a positional arg, returning empty vec on mismatch.
fn extract_intlist_arg(node: &Node, idx: usize) -> Vec<i64> {
    match node.args.get(idx) {
        Some(Arg::IntList(v)) => v.to_vec(),
        Some(Arg::Int(v)) => vec![*v],
        _ => Vec::new(),
    }
}

/// Remove dead (unused) nodes from the graph. A node is dead if it has no
/// users and is not the output node.
///
/// Also rewrites remaining ATen ops that have direct TOSA equivalents:
/// - MaxPool2d -> TosaMaxPool2d
/// - Reshape / View -> TosaReshape
pub struct CleanupPass;

impl Pass for CleanupPass {
    fn name(&self) -> &str {
        "Cleanup"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;

        // Pass 1: Rewrite remaining ATen ops to TOSA equivalents
        let all_ids = graph.all_node_ids();
        for node_id in &all_ids {
            let target = match graph.node(*node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };
            match target {
                OpTarget::MaxPool2d => {
                    // ATen max_pool2d args: [input, kernel_size, stride, padding, dilation, ceil_mode]
                    // Restructure into kwargs for TOSA emission
                    let node = graph.node(*node_id).unwrap();
                    let kernel = extract_intlist_arg(node, 1);
                    let stride = extract_intlist_arg(node, 2);
                    let padding = extract_intlist_arg(node, 3);
                    // Convert padding from [pH, pW] to TOSA [top, bottom, left, right]
                    let tosa_pad = if padding.len() >= 2 {
                        vec![padding[0], padding[0], padding[1], padding[1]]
                    } else if padding.len() == 1 {
                        vec![padding[0], padding[0], padding[0], padding[0]]
                    } else {
                        vec![0, 0, 0, 0]
                    };
                    if let Some(n) = graph.node_mut(*node_id) {
                        n.target = OpTarget::TosaMaxPool2d;
                        // Keep only the input arg
                        n.args.truncate(1);
                        n.kwargs = vec![
                            ("kernel".into(), Arg::IntList(SmallVec::from_vec(kernel))),
                            ("stride".into(), Arg::IntList(SmallVec::from_vec(stride))),
                            ("pad".into(), Arg::IntList(SmallVec::from_vec(tosa_pad))),
                        ];
                        modified = true;
                    }
                }
                OpTarget::AvgPool2d => {
                    // ATen avg_pool2d args: [input, kernel_size, stride, padding, ...]
                    let node = graph.node(*node_id).unwrap();
                    let kernel = extract_intlist_arg(node, 1);
                    let stride = extract_intlist_arg(node, 2);
                    let padding = extract_intlist_arg(node, 3);
                    let tosa_pad = if padding.len() >= 2 {
                        vec![padding[0], padding[0], padding[1], padding[1]]
                    } else if padding.len() == 1 {
                        vec![padding[0], padding[0], padding[0], padding[0]]
                    } else {
                        vec![0, 0, 0, 0]
                    };
                    if let Some(n) = graph.node_mut(*node_id) {
                        n.target = OpTarget::TosaAvgPool2d;
                        n.args.truncate(1);
                        n.kwargs = vec![
                            ("kernel".into(), Arg::IntList(SmallVec::from_vec(kernel))),
                            ("stride".into(), Arg::IntList(SmallVec::from_vec(stride))),
                            ("pad".into(), Arg::IntList(SmallVec::from_vec(tosa_pad))),
                        ];
                        modified = true;
                    }
                }
                OpTarget::Reshape | OpTarget::View | OpTarget::Flatten => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaReshape;
                        modified = true;
                    }
                }
                OpTarget::Add => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaAdd;
                        modified = true;
                    }
                }
                OpTarget::Sub => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaSub;
                        modified = true;
                    }
                }
                OpTarget::Mul => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaMul;
                        modified = true;
                    }
                }
                OpTarget::Permute => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaTranspose;
                        modified = true;
                    }
                }
                OpTarget::Relu => {
                    // Convert remaining Relu to TosaClamp with default bounds
                    if let Some(n) = graph.node_mut(*node_id) {
                        n.target = OpTarget::TosaClamp;
                        while n.args.len() > 1 {
                            n.args.pop();
                        }
                        n.kwargs = vec![
                            ("min_val".into(), Arg::Int(0)),
                            ("max_val".into(), Arg::Int(127)),
                        ];
                        modified = true;
                    }
                }
                OpTarget::Clamp => {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.target = OpTarget::TosaClamp;
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        // Pass 2: Dead code elimination
        // Iterate until no more nodes can be removed
        loop {
            let mut removed_any = false;
            let ids = graph.all_node_ids();
            for node_id in ids {
                let is_dead = match graph.node(node_id) {
                    Some(n) => {
                        // Keep output nodes
                        if n.op == OpType::Output {
                            false
                        // Keep nodes that have users
                        } else if !n.users.is_empty() {
                            false
                        // Keep nodes that are the graph output
                        } else {
                            graph.output != Some(node_id)
                        }
                    }
                    None => false,
                };
                if is_dead {
                    graph.erase_node(node_id);
                    removed_any = true;
                    modified = true;
                }
            }
            if !removed_any {
                break;
            }
        }

        modified
    }
}
