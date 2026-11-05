//! Pure Rust IrPass version of ReplaceInfAndLimitValuesPass for literal args.
//!
//! The Python annotation pipeline clamps `+/-inf` and float32 limit literals to
//! configured quantization-friendly values before passes like masked_fill expand them
//! into constant-producing ops. The direct Rust parent pipeline needs the same
//! normalization or the TOSA path can leak `inf` literals while the reference
//! path saturates them to finite quantized values.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::pipeline_config::QuantizeInfConfig;
use crate::runner_logging::log_line;

const FP32_MAX_AS_F64: f64 = f32::MAX as f64;
const FP32_MIN_AS_F64: f64 = f32::MIN as f64;

fn sanitize_scalar(value: f64, config: &QuantizeInfConfig) -> Option<f64> {
    if value.is_infinite() {
        return Some(if value.is_sign_negative() {
            config.neg_inf
        } else {
            config.pos_inf
        });
    }
    if value == FP32_MAX_AS_F64 {
        return Some(config.pos_inf);
    }
    if value == FP32_MIN_AS_F64 {
        return Some(config.neg_inf);
    }
    None
}

fn sanitize_arg(arg: &mut IrArg, config: &QuantizeInfConfig) -> bool {
    match arg {
        IrArg::Float(value) => {
            if let Some(sanitized) = sanitize_scalar(*value, config) {
                *value = sanitized;
                true
            } else {
                false
            }
        }
        IrArg::List(items) => items
            .iter_mut()
            .fold(false, |changed, item| sanitize_arg(item, config) | changed),
        _ => false,
    }
}

fn should_sanitize_node(node: &crate::fx_ir::IrNode) -> bool {
    node.meta.has_qparam_metadata()
}

#[derive(Default)]
pub struct IrReplaceInfAndLimitValuesPass(pub QuantizeInfConfig);

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
                node_changed |= sanitize_arg(arg, &self.0);
            }
            for arg in node.kwargs.values_mut() {
                node_changed |= sanitize_arg(arg, &self.0);
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
    use crate::fx_ir::{IrArg, IrGraph, IrPass, MetaValue, NodeMeta, QParam};
    use crate::pipeline_config::QuantizeInfConfig;
    use crate::test_ir::{call_default, call_with_meta};

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
        let id = call_with_meta(
            &mut graph,
            "masked_fill",
            "aten.masked_fill.Scalar",
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

        let modified = IrReplaceInfAndLimitValuesPass::default().run(&mut graph);

        assert!(modified);
        assert_eq!(graph.nodes[id].args[2].as_float(), Some(255.0));
        match &graph.nodes[id].args[3] {
            IrArg::List(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].as_float(), Some(-256.0));
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
            Some(-256.0)
        );
    }

    #[test]
    fn clamps_every_inf_in_a_list() {
        let mut graph = IrGraph::empty();
        let mut meta = NodeMeta::default();
        meta.set_single_output_qparam(QParam::per_tensor(
            1.0,
            0,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        let id = call_with_meta(
            &mut graph,
            "clamp",
            "aten.clamp.default",
            vec![IrArg::List(vec![
                IrArg::Float(f64::NEG_INFINITY),
                IrArg::Float(1.5),
                IrArg::Float(f64::INFINITY),
            ])],
            meta,
        );

        assert!(IrReplaceInfAndLimitValuesPass::default().run(&mut graph));
        let IrArg::List(items) = &graph.nodes[id].args[0] else {
            panic!("expected List arg");
        };
        let values: Vec<_> = items.iter().map(IrArg::as_float).collect();
        assert_eq!(values, vec![Some(-256.0), Some(1.5), Some(255.0)]);
    }

    #[test]
    fn preserves_inf_literals_for_unquantized_nodes() {
        let mut graph = IrGraph::empty();
        let id = call_default(
            &mut graph,
            "masked_fill",
            "aten.masked_fill.Scalar",
            vec![IrArg::Float(f64::INFINITY)],
        );

        let modified = IrReplaceInfAndLimitValuesPass::default().run(&mut graph);

        assert!(!modified);
        assert_eq!(graph.nodes[id].args[0].as_float(), Some(f64::INFINITY));
    }

    #[test]
    fn uses_configured_inf_values_and_preserves_guard_literals() {
        let mut graph = IrGraph::empty();
        let mut meta = NodeMeta::default();
        meta.set_single_output_qparam(QParam::per_tensor(
            1.0,
            0,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        let id = call_with_meta(
            &mut graph,
            "fill",
            "aten.full.default",
            vec![IrArg::Float(f64::NEG_INFINITY), IrArg::Float(f64::INFINITY)],
            meta.clone(),
        );
        meta.set_extra("arm_preserve_inf_literal", MetaValue::Bool(true));
        let guard = call_with_meta(
            &mut graph,
            "guard",
            "aten.eq.Scalar",
            vec![IrArg::Float(f64::NEG_INFINITY)],
            meta,
        );
        let pass = IrReplaceInfAndLimitValuesPass(QuantizeInfConfig {
            neg_inf: -100.0,
            pos_inf: 200.0,
        });

        assert!(pass.run(&mut graph));
        assert_eq!(graph.nodes[id].args[0].as_float(), Some(-100.0));
        assert_eq!(graph.nodes[id].args[1].as_float(), Some(200.0));
        assert_eq!(
            graph.nodes[guard].args[0].as_float(),
            Some(f64::NEG_INFINITY)
        );
    }
}
