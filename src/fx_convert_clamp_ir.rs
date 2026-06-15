//! Pure Rust IrPass version of FxConvertToClampPass.
//!
//! Rewrites `relu.default` → `clamp(x, min=0.0, max=None)`
//! and `hardtanh.default` → `clamp(x, min_val, max_val)`.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

pub struct IrConvertClampPass;

impl IrPass for IrConvertClampPass {
    fn name(&self) -> &str {
        "IrConvertClamp"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if graph.nodes[id].is_relu() {
                // relu(x) -> clamp(x, min=0.0, max=None)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => return false,
                };

                let node_name = graph.nodes[id].name.clone();
                let node_meta = graph.nodes[id].meta.clone();
                let mut cursor = graph.rewrite_cursor(id);
                let clamp_id = cursor.call_function(
                    format!("{}_clamp", node_name),
                    "aten.clamp.default".to_string(),
                    vec![IrArg::Node(input_id), IrArg::Float(0.0), IrArg::None],
                    node_meta,
                );
                cursor.replace_anchor_with(clamp_id);
                return true;
            } else if graph.nodes[id].is_hardtanh() {
                // hardtanh(x, min_val, max_val) -> clamp(x, min_val, max_val)
                let input_id = match graph.nodes[id].first_node_arg() {
                    Some(id) => id,
                    None => return false,
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

                let node_name = graph.nodes[id].name.clone();
                let node_meta = graph.nodes[id].meta.clone();
                let mut cursor = graph.rewrite_cursor(id);
                let clamp_id = cursor.call_function(
                    format!("{}_clamp", node_name),
                    "aten.clamp.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::Float(min_val),
                        IrArg::Float(max_val),
                    ],
                    node_meta,
                );
                cursor.replace_anchor_with(clamp_id);
                return true;
            }
            false
        });

        if modified {
            log_line(format_args!(
                "[IrConvertClamp] rewrote relu/hardtanh ops to clamp"
            ));
        }
        modified
    }
}
