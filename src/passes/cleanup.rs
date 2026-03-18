use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::{smallvec, SmallVec};

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
                OpTarget::Sum => {
                    if rewrite_sum_to_reduce(graph, *node_id) {
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        // Pass 2: Dead code elimination
        // Iterate until no more nodes can be removed
        let mut dce_iterations = 0;
        loop {
            dce_iterations += 1;
            if dce_iterations > 10 {
                eprintln!("[WARN] CleanupPass DCE hit max iterations");
                break;
            }
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

/// Rewrite an ATen Sum node into one or more chained TosaReduceSum nodes.
///
/// TOSA REDUCE_SUM reduces one axis at a time. If the ATen Sum reduces multiple
/// axes (e.g., dims=[2,3]), we chain reductions from highest axis to lowest so
/// that earlier indices remain stable.
fn rewrite_sum_to_reduce(graph: &mut Graph, node_id: NodeId) -> bool {
    let (input_id, dims, node_meta) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };
        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let dims = extract_intlist_arg(node, 1);
        if dims.is_empty() {
            return false;
        }
        (input_id, dims, node.meta.clone())
    };

    // Sort dims descending so reducing higher axes first keeps lower indices stable
    let mut sorted_dims = dims;
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    // Build the chain of TosaReduceSum nodes
    let mut current_input = input_id;
    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    for (i, &dim) in sorted_dims.iter().enumerate() {
        let is_last = i == sorted_dims.len() - 1;

        // Compute output shape: input shape with this dim set to 1
        let input_shape = graph
            .node(current_input)
            .map(|n| n.meta.val.shape_slice().to_vec())
            .unwrap_or_default();
        let input_dtype = graph
            .node(current_input)
            .map(|n| n.meta.val.dtype)
            .unwrap_or(DType::Int8);

        let mut out_shape = input_shape;
        let dim_idx = dim as usize;
        if dim_idx < out_shape.len() {
            out_shape[dim_idx] = 1;
        }

        let reduce_name = format!("{}_reduce_axis{}", node_name, dim);
        let mut reduce_meta = NodeMeta::new(TensorInfo::new(&out_shape, input_dtype));
        // Carry forward qparams from the original node for the last reduction
        if is_last {
            reduce_meta.input_qparams = node_meta.input_qparams.clone();
            reduce_meta.output_qparams = node_meta.output_qparams.clone();
        }

        if is_last {
            // Rewrite the original node in place for the last reduction
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::TosaReduceSum;
                n.args = smallvec![Arg::Node(current_input)];
                n.kwargs = vec![("axis".into(), Arg::Int(dim))];
                n.meta = reduce_meta;
            }
            // Update user lists: remove old input, add new
            if current_input != input_id {
                // The original node's input was input_id; we already removed it implicitly
                // by changing args. Fix up user lists.
                if let Some(orig_input) = graph.node_mut(input_id) {
                    orig_input.users.retain(|u| *u != node_id);
                }
                if let Some(new_input) = graph.node_mut(current_input) {
                    if !new_input.users.contains(&node_id) {
                        new_input.users.push(node_id);
                    }
                }
            }
        } else {
            // Insert intermediate reduction before the original node
            let reduce_id = graph.insert_before(
                node_id,
                OpType::CallFunction,
                OpTarget::TosaReduceSum,
                smallvec![Arg::Node(current_input)],
                vec![("axis".into(), Arg::Int(dim))],
                reduce_meta,
                reduce_name,
            );
            shape::propagate_shape(graph, reduce_id);
            current_input = reduce_id;
        }
    }

    true
}
