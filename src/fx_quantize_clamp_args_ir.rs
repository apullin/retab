//! Pure Rust IrPass version of FxQuantizeClampArgsPass.
//!
//! After fold_qdq annotates output_qparams on clamp nodes, this pass quantizes
//! the min/max scalar arguments to match the output quantization scale.

use crate::fx_ir::{IrArg, IrGraph, IrPass, QParam};
use crate::fx_target::is_clamp_target;
use crate::runner_logging::log_line;

fn quantize_value(qp: &QParam, value: f64) -> i64 {
    let q = (value / qp.scale + qp.zp as f64).round() as i64;
    q.max(qp.qmin).min(qp.qmax)
}

pub struct IrQuantizeClampArgsPass;

impl IrPass for IrQuantizeClampArgsPass {
    fn name(&self) -> &str {
        "IrQuantizeClampArgs"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_clamp_target(&graph.nodes[id].target) {
                continue;
            }

            // Need output_qparams to quantize
            let qp = match graph.nodes[id].meta.output_qparam() {
                Some(qp) => qp.clone(),
                None => continue,
            };

            // args: [input, min_val, max_val]
            let args = &graph.nodes[id].args;
            if args.is_empty() {
                continue;
            }

            let mut new_args = args.clone();
            let mut did_update = false;

            // Quantize min_val (args[1]) if present and not None
            if let Some(min_val) = args.get(1).and_then(IrArg::as_numeric_f64) {
                new_args[1] = IrArg::Int(quantize_value(&qp, min_val));
                did_update = true;
            }

            // Quantize max_val (args[2]) if present and not None
            if let Some(max_val) = args.get(2).and_then(IrArg::as_numeric_f64) {
                new_args[2] = IrArg::Int(quantize_value(&qp, max_val));
                did_update = true;
            }

            if did_update {
                // Update args in-place so conversion sees the quantized bounds.
                graph.set_args(id, new_args);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrQuantizeClampArgs] quantized clamp min/max args"
            ));
        }
        modified
    }
}
