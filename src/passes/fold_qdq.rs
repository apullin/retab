use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Fold QuantizePerTensor / DequantizePerTensor / DequantizePerChannel nodes
/// by extracting their quantization parameters and annotating downstream/upstream
/// nodes, then bypassing the Q/DQ nodes.
pub struct FoldAndAnnotateQParamsPass;

impl Pass for FoldAndAnnotateQParamsPass {
    fn name(&self) -> &str {
        "FoldAndAnnotateQParams"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;

        // ------------------------------------------------------------------
        // Pass 1: Collect DQ nodes and annotate their users' input_qparams
        // ------------------------------------------------------------------
        let dq_nodes = collect_dq_nodes(graph);
        for (dq_id, qargs) in &dq_nodes {
            let users: Vec<NodeId> = graph
                .node(*dq_id)
                .map(|n| n.users.clone())
                .unwrap_or_default()
                .to_vec();

            for &user_id in &users {
                // Find which arg position in the user references this DQ node
                if let Some(user_node) = graph.node(user_id) {
                    let positions = find_arg_positions(user_node, *dq_id);
                    for pos in positions {
                        if let Some(user_node) = graph.node_mut(user_id) {
                            // Only add if not already present for this position
                            if !user_node
                                .meta
                                .input_qparams
                                .iter()
                                .any(|(p, _)| *p == pos)
                            {
                                user_node.meta.input_qparams.push((pos, *qargs));
                            }
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pass 2: Collect Q nodes and annotate their inputs' output_qparams
        // ------------------------------------------------------------------
        let q_nodes = collect_q_nodes(graph);
        for (q_id, qargs) in &q_nodes {
            // The input to the Q node is its first arg
            if let Some(input_id) = graph.node(*q_id).and_then(|n| match n.args.first() {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            }) {
                if let Some(input_node) = graph.node_mut(input_id) {
                    if !input_node
                        .meta
                        .output_qparams
                        .iter()
                        .any(|(p, _)| *p == 0)
                    {
                        input_node.meta.output_qparams.push((0, *qargs));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pass 3: Bypass all DQ and Q nodes
        // ------------------------------------------------------------------
        for (dq_id, _) in dq_nodes {
            graph.bypass_node(dq_id);
            modified = true;
        }
        for (q_id, _) in q_nodes {
            graph.bypass_node(q_id);
            modified = true;
        }

        // ------------------------------------------------------------------
        // Pass 4: Update dtypes on nodes that now operate in quantized domain
        // ------------------------------------------------------------------
        // After bypassing Q/DQ nodes, nodes that have both input_qparams and
        // output_qparams should operate in the quantized dtype (typically Int8).
        // Also update nodes that only have input_qparams (they consume int8).
        if modified {
            let all_ids = graph.all_node_ids();
            for node_id in all_ids {
                let should_update = {
                    let node = match graph.node(node_id) {
                        Some(n) => n,
                        None => continue,
                    };
                    // Skip placeholders and outputs
                    if node.op != OpType::CallFunction {
                        continue;
                    }
                    // If the node has input qparams, it operates in quantized domain
                    if !node.meta.input_qparams.is_empty() {
                        // Use the dtype from input qparams (typically Int8)
                        Some(node.meta.input_qparams[0].1.dtype)
                    } else if !node.meta.output_qparams.is_empty() {
                        Some(node.meta.output_qparams[0].1.dtype)
                    } else {
                        None
                    }
                };
                if let Some(new_dtype) = should_update {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.val.dtype = new_dtype;
                    }
                }
            }
        }

        modified
    }
}

/// Collect all DequantizePerTensor / DequantizePerChannel nodes and extract
/// their QuantArgs.
fn collect_dq_nodes(graph: &Graph) -> Vec<(NodeId, QuantArgs)> {
    let mut result = Vec::new();
    for id in graph.all_node_ids() {
        let node = match graph.node(id) {
            Some(n) => n,
            None => continue,
        };
        let qargs = match &node.target {
            OpTarget::DequantizePerTensor => extract_dq_per_tensor_qargs(node),
            OpTarget::DequantizePerChannel => extract_dq_per_channel_qargs(node),
            _ => continue,
        };
        if let Some(qa) = qargs {
            result.push((id, qa));
        }
    }
    result
}

/// Collect all QuantizePerTensor nodes and extract their QuantArgs.
fn collect_q_nodes(graph: &Graph) -> Vec<(NodeId, QuantArgs)> {
    let mut result = Vec::new();
    for id in graph.all_node_ids() {
        let node = match graph.node(id) {
            Some(n) => n,
            None => continue,
        };
        if node.target != OpTarget::QuantizePerTensor {
            continue;
        }
        if let Some(qa) = extract_q_per_tensor_qargs(node) {
            result.push((id, qa));
        }
    }
    result
}

/// DequantizePerTensor args: [input, scale, zero_point, quant_min, quant_max, dtype]
fn extract_dq_per_tensor_qargs(node: &Node) -> Option<QuantArgs> {
    let scale = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        _ => return None,
    };
    let zero_point = match node.args.get(2) {
        Some(Arg::Int(i)) => *i,
        _ => 0,
    };
    let qmin = match node.args.get(3) {
        Some(Arg::Int(i)) => *i,
        _ => -128,
    };
    let qmax = match node.args.get(4) {
        Some(Arg::Int(i)) => *i,
        _ => 127,
    };
    let dtype = match node.args.get(5) {
        Some(Arg::Dtype(d)) => *d,
        _ => DType::Int8,
    };
    Some(QuantArgs {
        scale,
        zero_point,
        dtype,
        qmin,
        qmax,
    })
}

/// DequantizePerChannel args: [input, scale, zero_point, axis, quant_min, quant_max, dtype]
/// For per-channel, we extract the first scale/zp as a representative (used for rescale math).
fn extract_dq_per_channel_qargs(node: &Node) -> Option<QuantArgs> {
    let scale = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        Some(Arg::FloatList(fl)) => fl.first().copied().unwrap_or(1.0),
        _ => return None,
    };
    let zero_point = match node.args.get(2) {
        Some(Arg::Int(i)) => *i,
        Some(Arg::IntList(il)) => il.first().copied().unwrap_or(0),
        _ => 0,
    };
    // axis at index 3, quant_min at index 4, quant_max at index 5, dtype at index 6
    let qmin = match node.args.get(4) {
        Some(Arg::Int(i)) => *i,
        _ => -128,
    };
    let qmax = match node.args.get(5) {
        Some(Arg::Int(i)) => *i,
        _ => 127,
    };
    let dtype = match node.args.get(6) {
        Some(Arg::Dtype(d)) => *d,
        _ => DType::Int8,
    };
    Some(QuantArgs {
        scale,
        zero_point,
        dtype,
        qmin,
        qmax,
    })
}

/// QuantizePerTensor args: [input, scale, zero_point, quant_min, quant_max, dtype]
fn extract_q_per_tensor_qargs(node: &Node) -> Option<QuantArgs> {
    let scale = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        _ => return None,
    };
    let zero_point = match node.args.get(2) {
        Some(Arg::Int(i)) => *i,
        _ => 0,
    };
    let qmin = match node.args.get(3) {
        Some(Arg::Int(i)) => *i,
        _ => -128,
    };
    let qmax = match node.args.get(4) {
        Some(Arg::Int(i)) => *i,
        _ => 127,
    };
    let dtype = match node.args.get(5) {
        Some(Arg::Dtype(d)) => *d,
        _ => DType::Int8,
    };
    Some(QuantArgs {
        scale,
        zero_point,
        dtype,
        qmin,
        qmax,
    })
}

/// Find which arg positions in a node reference the given NodeId.
fn find_arg_positions(node: &Node, target_id: NodeId) -> Vec<u32> {
    let mut positions = Vec::new();
    for (i, arg) in node.args.iter().enumerate() {
        match arg {
            Arg::Node(id) if *id == target_id => {
                positions.push(i as u32);
            }
            Arg::NodeList(ids) => {
                if ids.contains(&target_id) {
                    positions.push(i as u32);
                }
            }
            Arg::OptionalNode(Some(id)) if *id == target_id => {
                positions.push(i as u32);
            }
            _ => {}
        }
    }
    positions
}
