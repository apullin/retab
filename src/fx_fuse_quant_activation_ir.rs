//! Pure Rust IrPass version of FxFuseQuantizedActivationPass.
//!
//! Removes relu/hardtanh nodes when:
//!   1. The single user is a quantize_per_tensor with zp == qmin
//!   2. The input traces back through Q/DQ to a conv/linear producer
//! Runs BEFORE fold_qdq, so Q/DQ nodes are still present.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

fn is_relu(target: &str) -> bool { target.contains("relu.default") }
fn is_hardtanh(target: &str) -> bool { target.contains("hardtanh.default") }
fn is_quantize(target: &str) -> bool { target.contains("quantize_per_tensor") }
fn is_dequantize(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}
fn is_conv_or_linear(target: &str) -> bool {
    target.contains("convolution.default")
        || target.contains("conv2d.default")
        || target.contains("linear.default")
}

/// Find all alive call_function users of `producer_id`.
fn call_function_users(graph: &IrGraph, producer_id: NodeId) -> Vec<NodeId> {
    graph.nodes.iter()
        .filter(|n| n.alive && n.op == NodeOp::CallFunction)
        .filter(|n| n.args.iter().any(|a| matches!(a, IrArg::Node(id) if *id == producer_id)))
        .map(|n| n.id)
        .collect()
}

pub struct IrFuseQuantActivationPass;

impl IrPass for IrFuseQuantActivationPass {
    fn name(&self) -> &str { "IrFuseQuantActivation" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let is_relu_node = is_relu(&graph.nodes[id].target);
            let is_hardtanh_node = is_hardtanh(&graph.nodes[id].target);
            if !is_relu_node && !is_hardtanh_node { continue; }

            // For hardtanh, check min_val == 0
            if is_hardtanh_node {
                let min_val = graph.nodes[id].args.get(1)
                    .and_then(|a| match a {
                        IrArg::Float(v) => Some(*v),
                        IrArg::Int(v) => Some(*v as f64),
                        _ => None,
                    });
                if min_val != Some(0.0) { continue; }
            }

            // Get relu/hardtanh's input
            let input_id = match graph.nodes[id].first_node_arg() {
                Some(nid) => nid,
                None => continue,
            };

            // Single user must be quantize_per_tensor
            let users = call_function_users(graph, id);
            if users.len() != 1 { continue; }
            let q_id = users[0];
            if !is_quantize(&graph.nodes[q_id].target) { continue; }

            // Check zp == qmin on the Q node: args [input, scale, zp, qmin, qmax, dtype]
            let zp = graph.nodes[q_id].args.get(2).and_then(|a| a.as_int()).unwrap_or(0);
            let qmin = graph.nodes[q_id].args.get(3).and_then(|a| a.as_int()).unwrap_or(-128);
            if zp != qmin { continue; }

            // Trace back through Q/DQ chain to find conv/linear producer
            let mut current = input_id;
            for _ in 0..4 {
                if !graph.nodes[current].alive || graph.nodes[current].op != NodeOp::CallFunction {
                    break;
                }
                let t = &graph.nodes[current].target;
                if is_dequantize(t) || is_quantize(t) {
                    match graph.nodes[current].first_node_arg() {
                        Some(nid) => current = nid,
                        None => break,
                    }
                } else {
                    break;
                }
            }

            // Producer must be conv/linear with a single user
            if !graph.nodes[current].alive
                || graph.nodes[current].op != NodeOp::CallFunction
                || !is_conv_or_linear(&graph.nodes[current].target)
            {
                continue;
            }
            let producer_users = call_function_users(graph, current);
            if producer_users.len() != 1 { continue; }

            // Bypass: replace relu/hardtanh with its input
            graph.replace_and_erase(id, input_id);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrFuseQuantActivation] fused quantized relu/hardtanh"));
        }
        modified
    }
}
