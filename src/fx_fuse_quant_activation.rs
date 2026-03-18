use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FuseQuantizedActivationPass`.
///
/// Removes relu/hardtanh nodes when:
///   1. The relu/hardtanh's single user is a quantize_per_tensor node
///   2. The Q node's zero_point == quant_min (the quantization range already
///      clips at the zero-point, making relu redundant)
///   3. The relu/hardtanh's input traces back through DQ to a conv/linear
///      producer with a single user
///
/// This pass runs BEFORE fold_qdq so the Q/DQ nodes are still present.
/// On the pre-fold graph, the pattern is: conv -> DQ -> relu -> Q.
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

fn is_dequantize(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_conv_or_linear(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("linear.default")
}

impl FxPass for FxFuseQuantizedActivationPass {
    fn name(&self) -> &str {
        "FxFuseQuantizedActivation"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
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

            let node_name = FxGraphModule::node_name(node)?;

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
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {} has {} users, skipping",
                        node_name,
                        users.len()
                    );
                }
                continue;
            }
            let q_user = &users[0];
            let user_op = FxGraphModule::node_op(q_user)?;
            if user_op != "call_function" {
                continue;
            }
            let user_target = FxGraphModule::node_target_str(q_user)?;
            if !is_quantize_per_tensor(&user_target) {
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {}'s user is not Q: {}",
                        node_name, user_target
                    );
                }
                continue;
            }

            // Check that zero_point == quant_min on the Q node (matching
            // the Python FuseQuantizedActivationPass exactly).
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

            if debug {
                eprintln!(
                    "[FuseQuantActivation] {} -> Q: zp={}, qmin={}",
                    node_name, zp, qmin
                );
            }

            if zp != qmin {
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {} NOT fusing: zp={} != qmin={}",
                        node_name, zp, qmin
                    );
                }
                continue;
            }

            // The relu/hardtanh's input must lead back to a conv/linear.
            // On the pre-fold graph the pattern is: conv -> DQ -> relu -> Q
            // so relu's input is the DQ node. Look through DQ to find conv/linear.
            let args = FxGraphModule::node_args(node)?;
            let relu_input = match args.first() {
                Some(n) if FxGraphModule::is_node(n) => n,
                _ => continue,
            };

            // Look through the Q/DQ chain to find the actual producer.
            // Pre-fold graph pattern: conv -> Q -> DQ -> relu -> Q
            // So relu's input is DQ, DQ's input is Q, Q's input is conv.
            // We look through DQ and Q nodes to find the ultimate producer.
            let mut current = relu_input.clone();
            for _ in 0..4 {
                // Safety limit on chain depth
                let cur_op = FxGraphModule::node_op(&current)?;
                if cur_op != "call_function" {
                    break;
                }
                let cur_target = FxGraphModule::node_target_str(&current)?;
                if is_dequantize(&cur_target) || is_quantize_per_tensor(&cur_target) {
                    let cur_args = FxGraphModule::node_args(&current)?;
                    match cur_args.first() {
                        Some(n) if FxGraphModule::is_node(n) => {
                            current = n.clone();
                        }
                        _ => break,
                    }
                } else {
                    break;
                }
            }
            let producer = current;

            // Producer must be conv/linear with a single user
            let producer_op = FxGraphModule::node_op(&producer)?;
            if producer_op != "call_function" {
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {} producer op is {}, skipping",
                        node_name, producer_op
                    );
                }
                continue;
            }
            let producer_target = FxGraphModule::node_target_str(&producer)?;
            if !is_conv_or_linear(&producer_target) {
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {} producer is {}, not conv/linear",
                        node_name, producer_target
                    );
                }
                continue;
            }
            let producer_users = FxGraphModule::node_users(&producer)?;
            if producer_users.len() != 1 {
                if debug {
                    eprintln!(
                        "[FuseQuantActivation] {} producer has {} users, skipping",
                        node_name,
                        producer_users.len()
                    );
                }
                continue;
            }

            if debug {
                eprintln!("[FuseQuantActivation] fusing {} (zp==qmin={})", node_name, qmin);
            }

            // Bypass: replace all uses of relu/hardtanh with its input (the DQ node)
            FxGraphModule::replace_all_uses_with(node, relu_input)?;
            gm.erase_node(node)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
