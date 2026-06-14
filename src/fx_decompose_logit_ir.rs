//! Pure Rust IrPass version of FxDecomposeLogitPass.
//!
//! Decomposes `aten.logit` into primitive ops:
//!   logit(x, eps=None) ->
//!     if eps is not None:
//!       x = clamp(x, eps, 1 - eps)
//!     neg_x = mul(x, -1.0)          # full(-1.0) then mul
//!     denom = add(neg_x, 1.0)       # full(1.0) then add  => (1 - x)
//!     frac = reciprocal(denom)       # 1 / (1 - x)
//!     log_input = mul(x, frac)       # x / (1 - x)
//!     result = log(log_input)

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

pub struct IrDecomposeLogitPass;

impl IrPass for IrDecomposeLogitPass {
    fn name(&self) -> &str {
        "IrDecomposeLogit"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("logit") {
                continue;
            }

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(id) => id,
                None => continue,
            };
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => vec![1],
            };
            let rank = input_shape.len();

            // Extract eps: args[1] if present, otherwise None
            let eps: Option<f64> = graph.nodes[id].args.get(1).and_then(|a| match a {
                IrArg::None => None,
                other => other.as_float(),
            });

            let meta = graph.nodes[id].meta.clone();
            let base_name = graph.nodes[id].name.clone();

            // Track the current x — may be updated by clamp
            let mut x = input_id;

            // If eps is not None: x = clamp(x, eps, 1 - eps)
            if let Some(eps_val) = eps {
                x = graph.add_anchored_call_function(
                    id,
                    format!("{base_name}_clamp"),
                    "aten.clamp.default",
                    vec![
                        IrArg::Node(x),
                        IrArg::Float(eps_val),
                        IrArg::Float(1.0 - eps_val),
                    ],
                    meta.clone(),
                );
            }

            // full(-1.0) with broadcast shape [1; rank]
            let shape_ones: Vec<IrArg> = vec![IrArg::Int(1); rank];
            let neg_one = graph.add_anchored_call_function(
                id,
                format!("{base_name}_full_neg1"),
                "aten.full.default",
                vec![IrArg::List(shape_ones.clone()), IrArg::Float(-1.0)],
                meta.clone(),
            );

            // neg_x = mul(x, -1.0)
            let neg_x = graph.add_anchored_call_function(
                id,
                format!("{base_name}_neg_x"),
                "aten.mul.Tensor",
                vec![IrArg::Node(x), IrArg::Node(neg_one)],
                meta.clone(),
            );

            // full(1.0) with broadcast shape [1; rank]
            let one = graph.add_anchored_call_function(
                id,
                format!("{base_name}_full_one"),
                "aten.full.default",
                vec![IrArg::List(shape_ones), IrArg::Float(1.0)],
                meta.clone(),
            );

            // denom = add(neg_x, 1.0) => (1 - x)
            let denom = graph.add_anchored_call_function(
                id,
                format!("{base_name}_denom"),
                "aten.add.Tensor",
                vec![IrArg::Node(neg_x), IrArg::Node(one)],
                meta.clone(),
            );

            // frac = reciprocal(denom) => 1 / (1 - x)
            let frac = graph.add_anchored_call_function(
                id,
                format!("{base_name}_frac"),
                "aten.reciprocal.default",
                vec![IrArg::Node(denom)],
                meta.clone(),
            );

            // log_input = mul(x, frac) => x / (1 - x)
            let log_input = graph.add_anchored_call_function(
                id,
                format!("{base_name}_log_input"),
                "aten.mul.Tensor",
                vec![IrArg::Node(x), IrArg::Node(frac)],
                meta.clone(),
            );

            // result = log(log_input)
            graph.replace_with_call_function(
                id,
                format!("{base_name}_log"),
                "aten.log.default",
                vec![IrArg::Node(log_input)],
                meta,
            );
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeLogit] decomposed logit ops"));
        }
        modified
    }
}
