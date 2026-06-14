//! Pure Rust IrPass version of FxConvertToClampPass.
//!
//! Rewrites `relu.default` → `clamp(x, min=0.0, max=None)`
//! and `hardtanh.default` → `clamp(x, min_val, max_val)`.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeOp};
use crate::runner_logging::log_line;

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

            if graph.nodes[id].is_relu() {
                // relu(x) -> clamp(x, min=0.0, max=None)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => continue,
                };

                graph.replace_with_call_function(
                    id,
                    format!("{}_clamp", graph.nodes[id].name),
                    "aten.clamp.default".to_string(),
                    vec![IrArg::Node(input_id), IrArg::Float(0.0), IrArg::None],
                    graph.nodes[id].meta.clone(),
                );
                modified = true;
            } else if graph.nodes[id].is_hardtanh() {
                // hardtanh(x, min_val, max_val) -> clamp(x, min_val, max_val)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => continue,
                };

                let min_val = graph.nodes[id]
                    .args
                    .get(1)
                    .and_then(IrArg::as_numeric_f64)
                    .unwrap_or(-1.0);
                let max_val = graph.nodes[id]
                    .args
                    .get(2)
                    .and_then(IrArg::as_numeric_f64)
                    .unwrap_or(1.0);

                graph.replace_with_call_function(
                    id,
                    format!("{}_clamp", graph.nodes[id].name),
                    "aten.clamp.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::Float(min_val),
                        IrArg::Float(max_val),
                    ],
                    graph.nodes[id].meta.clone(),
                );
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
