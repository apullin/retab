//! Pure Rust IrPass version of FxFuseQuantizedActivationPass.
//!
//! Removes relu/hardtanh nodes when:
//!   1. The single user is a quantize_per_tensor with zp == qmin
//!   2. The input traces back through Q/DQ to a conv/linear producer
//! Runs BEFORE fold_qdq, so Q/DQ nodes are still present.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

pub struct IrFuseQuantActivationPass;

impl IrPass for IrFuseQuantActivationPass {
    fn name(&self) -> &str {
        "IrFuseQuantActivation"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let is_relu_node = graph.nodes[id].is_relu();
            let is_hardtanh_node = graph.nodes[id].is_hardtanh();
            if !is_relu_node && !is_hardtanh_node {
                continue;
            }

            // For hardtanh, check min_val == 0
            if is_hardtanh_node {
                let min_val = graph.nodes[id].args.get(1).and_then(IrArg::as_numeric_f64);
                if min_val != Some(0.0) {
                    continue;
                }
            }

            // Get relu/hardtanh's input
            let input_id = match graph.nodes[id].first_node_arg() {
                Some(nid) => nid,
                None => continue,
            };

            // Single user must be quantize_per_tensor
            let Some(q_id) = graph.single_live_call_function_user(id) else {
                continue;
            };
            if !graph.nodes[q_id].is_quantize() {
                continue;
            }

            // Check zp == qmin on the Q node: args [input, scale, zp, qmin, qmax, dtype]
            let zp = graph.nodes[q_id]
                .args
                .get(2)
                .and_then(|a| a.as_int())
                .unwrap_or(0);
            let qmin = graph.nodes[q_id]
                .args
                .get(3)
                .and_then(|a| a.as_int())
                .unwrap_or(-128);
            if zp != qmin {
                continue;
            }

            // Trace back through Q/DQ chain to find conv/linear producer
            let mut current = input_id;
            for _ in 0..4 {
                if !graph.nodes[current].alive || graph.nodes[current].op != NodeOp::CallFunction {
                    break;
                }
                if graph.nodes[current].is_dequantize_any() || graph.nodes[current].is_quantize() {
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
                || !graph.nodes[current].is_conv_or_linear()
            {
                continue;
            }
            if graph.single_live_call_function_user(current).is_none() {
                continue;
            }

            // Bypass: replace relu/hardtanh with its input
            graph.replace_and_erase(id, input_id);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrFuseQuantActivation] fused quantized relu/hardtanh"
            ));
        }
        modified
    }
}
