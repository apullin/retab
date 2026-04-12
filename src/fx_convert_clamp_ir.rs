//! Pure Rust IrPass version of FxConvertToClampPass.
//!
//! Rewrites `relu.default` → `clamp(x, min=0.0, max=None)`
//! and `hardtanh.default` → `clamp(x, min_val, max_val)`.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeOp};
use crate::runner_logging::log_line;

fn is_relu(target: &str) -> bool {
    target.contains("relu.default")
}

fn is_hardtanh(target: &str) -> bool {
    target.contains("hardtanh.default")
}

/// Extract a float from an IrArg, coercing Int to f64 if needed.
fn arg_as_f64(arg: &IrArg) -> Option<f64> {
    match arg {
        IrArg::Float(v) => Some(*v),
        IrArg::Int(v) => Some(*v as f64),
        _ => None,
    }
}

pub struct IrConvertClampPass;

impl IrPass for IrConvertClampPass {
    fn name(&self) -> &str {
        "IrConvertClamp"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<usize> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }

            let target = &graph.nodes[id].target;

            if is_relu(target) {
                // relu(x) -> clamp(x, min=0.0, max=None)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => continue,
                };

                let new_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_clamp", graph.nodes[id].name),
                    "aten.clamp.default".to_string(),
                    vec![IrArg::Node(input_id), IrArg::Float(0.0), IrArg::None],
                    graph.nodes[id].meta.clone(),
                );
                graph.replace_and_erase(id, new_id);
                modified = true;
            } else if is_hardtanh(target) {
                // hardtanh(x, min_val, max_val) -> clamp(x, min_val, max_val)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => continue,
                };

                let min_val = graph.nodes[id]
                    .args
                    .get(1)
                    .and_then(arg_as_f64)
                    .unwrap_or(-1.0);
                let max_val = graph.nodes[id]
                    .args
                    .get(2)
                    .and_then(arg_as_f64)
                    .unwrap_or(1.0);

                let new_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_clamp", graph.nodes[id].name),
                    "aten.clamp.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::Float(min_val),
                        IrArg::Float(max_val),
                    ],
                    graph.nodes[id].meta.clone(),
                );
                graph.replace_and_erase(id, new_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrConvertClamp] rewrote relu/hardtanh ops to clamp"
            ));
        }
        modified
    }
}
