//! IrPass port of ExecuTorch's FuseQuantizedActivationPass (v1.6.0-rc4, PR #21974).
//!
//! Removes `relu`, and `hardtanh` with a 0 lower bound, when:
//!   1. its only user is a `quantize_per_tensor` (per-channel quantization has no
//!      scalar qmin to rewrite);
//!   2. for `hardtanh`, its upper bound quantizes to that node's qmax;
//!   3. its input is a convolution or linear whose only user is the activation.
//! The quantize node's qmin is then set to its zero point, so the fused output keeps
//! the activation's lower bound. Runs before fold_qdq, while Q/DQ nodes are present.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

pub struct IrFuseQuantActivationPass;

// quantize_per_tensor args: [input, scale, zero_point, quant_min, quant_max, dtype].
const QUANT_SCALE: usize = 1;
const QUANT_ZP: usize = 2;
const QUANT_QMIN: usize = 3;
const QUANT_QMAX: usize = 4;

/// Mirrors `quantized_decomposed.quantize_per_tensor` on one value: float32
/// arithmetic, round half to even, then clamp to [qmin, qmax].
fn quantize_value(value: f64, scale: f64, zp: i64, qmin: i64, qmax: i64) -> i64 {
    let inv_scale = (1.0 / scale) as f32;
    let rounded = ((value as f32) * inv_scale).round_ties_even() as f64;
    (rounded + zp as f64).clamp(qmin as f64, qmax as f64) as i64
}

/// The quantize_per_tensor node whose qmin should become its zero point, if the
/// activation at `id` is fuseable.
fn fuseable_quant_user(graph: &IrGraph, id: NodeId) -> Option<NodeId> {
    let node = &graph.nodes[id];
    let is_hardtanh = node.is_hardtanh();
    if !node.is_relu() && !is_hardtanh {
        return None;
    }
    if is_hardtanh && node.arg_numeric_f64(1) != Some(0.0) {
        return None;
    }

    let q_id = graph.single_live_user(id)?;
    let quant = &graph.nodes[q_id];
    if !quant.is_quantize() {
        return None;
    }
    if is_hardtanh {
        let max_val = node.arg_numeric_f64(2)?;
        let scale = quant.arg_numeric_f64(QUANT_SCALE)?;
        let zp = quant.arg_int(QUANT_ZP)?;
        let qmin = quant.arg_int(QUANT_QMIN)?;
        let qmax = quant.arg_int(QUANT_QMAX)?;
        if quantize_value(max_val, scale, zp, qmin, qmax) != qmax {
            return None;
        }
    }

    let input_id = node.first_node_arg()?;
    let input = &graph.nodes[input_id];
    if !input.is_conv_or_linear() || graph.single_live_user(input_id) != Some(id) {
        return None;
    }
    Some(q_id)
}

impl IrPass for IrFuseQuantActivationPass {
    fn name(&self) -> &str {
        "IrFuseQuantActivation"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let Some(q_id) = fuseable_quant_user(graph, id) else {
                return false;
            };
            let Some(zp) = graph.nodes[q_id].arg_int(QUANT_ZP) else {
                return false;
            };
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                return false;
            };
            graph.set_arg(q_id, QUANT_QMIN, IrArg::Int(zp));
            graph.replace_and_erase(id, input_id);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrFuseQuantActivation] fused quantized relu/hardtanh"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::quantize_value;

    #[test]
    fn quantize_value_rounds_half_to_even_and_clamps() {
        assert_eq!(quantize_value(6.0, 6.0 / 127.0, 0, -128, 127), 127);
        assert_eq!(quantize_value(6.0, 0.1, -10, -128, 127), 50);
        assert_eq!(quantize_value(100.0, 0.1, 0, -128, 127), 127);
        assert_eq!(quantize_value(0.25, 0.5, 0, -128, 127), 0);
        assert_eq!(quantize_value(0.75, 0.5, 0, -128, 127), 2);
    }
}
