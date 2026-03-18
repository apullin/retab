use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FuseQuantizedActivationPass`.
///
/// Removes relu/hardtanh nodes when they sit between a conv/linear and a Q node
/// whose qmin equals the zero-point (meaning the activation lower-bound is
/// already captured by the quantization range).
pub struct FxFuseQuantizedActivationPass;

fn is_relu(target: &str) -> bool {
    target.contains("relu.default")
}

fn is_hardtanh(target: &str) -> bool {
    target.contains("hardtanh.default")
}

fn is_quantize_per_tensor(target: &str) -> bool {
    target.contains("quantize_per_tensor")
}

fn is_conv_or_linear(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("linear.default")
}

impl FxPass for FxFuseQuantizedActivationPass {
    fn name(&self) -> &str {
        "FxFuseQuantizedActivation"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let mut modified = false;
        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            // Must be relu or hardtanh
            let is_relu_node = is_relu(&target);
            let is_hardtanh_node = is_hardtanh(&target);
            if !is_relu_node && !is_hardtanh_node {
                continue;
            }

            // For hardtanh, check min_val == 0
            if is_hardtanh_node {
                let args = FxGraphModule::node_args(node)?;
                if let Some(min_arg) = args.get(1) {
                    let min_val: f64 = match min_arg.extract() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if min_val != 0.0 {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Single user must be a quantize_per_tensor node
            let users = FxGraphModule::node_users(node)?;
            if users.len() != 1 {
                continue;
            }
            let q_user = &users[0];
            let user_op = FxGraphModule::node_op(q_user)?;
            if user_op != "call_function" {
                continue;
            }
            let user_target = FxGraphModule::node_target_str(q_user)?;
            if !is_quantize_per_tensor(&user_target) {
                continue;
            }

            // Check that zero_point == qmin on the Q node
            let q_args = FxGraphModule::node_args(q_user)?;
            // Args: [input, scale, zero_point, qmin, qmax, dtype]
            let zp: i64 = q_args
                .get(2)
                .and_then(|a| a.extract().ok())
                .unwrap_or(0);
            let qmin: i64 = q_args
                .get(3)
                .and_then(|a| a.extract().ok())
                .unwrap_or(-128);
            if zp != qmin {
                continue;
            }

            // Input to the relu/hardtanh must be a conv/linear with a single user
            let args = FxGraphModule::node_args(node)?;
            let input_node = match args.first() {
                Some(n) if FxGraphModule::is_node(n) => n,
                _ => continue,
            };
            let input_op = FxGraphModule::node_op(input_node)?;
            if input_op != "call_function" {
                continue;
            }
            let input_target = FxGraphModule::node_target_str(input_node)?;
            if !is_conv_or_linear(&input_target) {
                continue;
            }
            let input_users = FxGraphModule::node_users(input_node)?;
            if input_users.len() != 1 {
                continue;
            }

            // Bypass: replace all uses of relu/hardtanh with its input
            FxGraphModule::replace_all_uses_with(node, input_node)?;
            gm.erase_node(node)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
