//! Pure Rust IrPass version of ReplaceInfAndLimitValuesPass for literal args.
//!
//! The Python annotation pipeline clamps `+/-inf` and float32 limit literals to
//! quantization-friendly `+/-255.0` before passes like masked_fill expand them
//! into constant-producing ops. The direct Rust parent pipeline needs the same
//! normalization or the TOSA path can leak `inf` literals while the reference
//! path saturates them to finite quantized values.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

const POS_LIMIT_REPLACEMENT: f64 = 255.0;
const NEG_LIMIT_REPLACEMENT: f64 = -255.0;
const FP32_MAX_AS_F64: f64 = f32::MAX as f64;
const FP32_MIN_AS_F64: f64 = f32::MIN as f64;

fn sanitize_scalar(value: f64) -> Option<f64> {
    if value.is_infinite() {
        return Some(if value.is_sign_negative() {
            NEG_LIMIT_REPLACEMENT
        } else {
            POS_LIMIT_REPLACEMENT
        });
    }
    if value == FP32_MAX_AS_F64 {
        return Some(POS_LIMIT_REPLACEMENT);
    }
    if value == FP32_MIN_AS_F64 {
        return Some(NEG_LIMIT_REPLACEMENT);
    }
    None
}

fn sanitize_arg(arg: &mut IrArg) -> bool {
    match arg {
        IrArg::Float(value) => {
            if let Some(sanitized) = sanitize_scalar(*value) {
                *value = sanitized;
                true
            } else {
                false
            }
        }
        IrArg::List(items) => items.iter_mut().any(sanitize_arg),
        _ => false,
    }
}

fn should_sanitize_node(node: &crate::fx_ir::IrNode) -> bool {
    node.meta.has_qparam_metadata()
}

pub struct IrReplaceInfAndLimitValuesPass;

impl IrPass for IrReplaceInfAndLimitValuesPass {
    fn name(&self) -> &str {
        "IrReplaceInfAndLimitValues"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;

        for node in &mut graph.nodes {
            if !node.alive {
                continue;
            }
            if !should_sanitize_node(node) {
                continue;
            }
            if node
                .meta
                .extra_bool("arm_preserve_inf_literal")
                .unwrap_or(false)
            {
                continue;
            }
            let mut node_changed = false;
            for arg in &mut node.args {
                node_changed |= sanitize_arg(arg);
            }
            for arg in node.kwargs.values_mut() {
                node_changed |= sanitize_arg(arg);
            }
            if node_changed {
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrReplaceInfAndLimitValues] clamped inf/float32 limit literals"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::IrReplaceInfAndLimitValuesPass;
    use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeMeta, NodeOp, QParam};

    #[test]
    fn clamps_inf_literals_in_args_and_kwargs_for_quantized_nodes() {
        let mut graph = IrGraph::empty();
        let mut meta = NodeMeta::default();
        meta.set_single_output_qparam(QParam::per_tensor(
            1.0,
            0,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        let id = graph.add_node(
            NodeOp::CallFunction,
            "masked_fill".into(),
            "aten.masked_fill.Scalar".into(),
            vec![
                IrArg::Int(0),
                IrArg::Bool(true),
                IrArg::Float(f64::INFINITY),
                IrArg::List(vec![IrArg::Float(f64::NEG_INFINITY)]),
            ],
            meta,
        );
        graph.set_kwarg(id, "max_value", IrArg::Float(f32::MAX as f64));
        graph.set_kwarg(id, "min_value", IrArg::Float(f32::MIN as f64));

        let modified = IrReplaceInfAndLimitValuesPass.run(&mut graph);

        assert!(modified);
        assert_eq!(graph.nodes[id].args[2].as_float(), Some(255.0));
        match &graph.nodes[id].args[3] {
            IrArg::List(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].as_float(), Some(-255.0));
            }
            other => panic!("expected List arg, got {other:?}"),
        }
        assert_eq!(
            graph.nodes[id]
                .kwargs
                .get("max_value")
                .and_then(IrArg::as_float),
            Some(255.0)
        );
        assert_eq!(
            graph.nodes[id]
                .kwargs
                .get("min_value")
                .and_then(IrArg::as_float),
            Some(-255.0)
        );
    }

    #[test]
    fn preserves_inf_literals_for_unquantized_nodes() {
        let mut graph = IrGraph::empty();
        let id = graph.add_node(
            NodeOp::CallFunction,
            "masked_fill".into(),
            "aten.masked_fill.Scalar".into(),
            vec![IrArg::Float(f64::INFINITY)],
            NodeMeta::default(),
        );

        let modified = IrReplaceInfAndLimitValuesPass.run(&mut graph);

        assert!(!modified);
        assert_eq!(graph.nodes[id].args[0].as_float(), Some(f64::INFINITY));
    }
}
