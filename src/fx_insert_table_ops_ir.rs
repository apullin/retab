use crate::fx_ep_effects::{FxEpEffect, FxEpEffectBatch, FxInsertTableOpEffect};
use crate::fx_ir::{IrArg, IrEpPass, IrEpPassResult, IrGraph, IrNode, NodeOp};

pub struct IrInsertTableOpsEpPass;

#[derive(Clone, Debug, PartialEq)]
struct TableOpSpec {
    kind: &'static str,
    scalar_arg: Option<f64>,
}

fn node_scalar_arg(node: &IrNode, idx: usize) -> Option<f64> {
    match node.args.get(idx) {
        Some(IrArg::Float(value)) => Some(*value),
        Some(IrArg::Int(value)) => Some(*value as f64),
        _ => None,
    }
}

fn table_op_spec(node: &IrNode) -> Option<TableOpSpec> {
    let target = node.target.as_str();
    let kind = if target.contains("rsqrt.default") {
        "rsqrt"
    } else if target.contains("ceil.default") {
        "ceil"
    } else if target.contains("floor.default") {
        "floor"
    } else if target.contains("round.default") {
        "round"
    } else if target.contains("reciprocal.default") {
        "reciprocal"
    } else if target.contains("exp.default") {
        "exp"
    } else if target.contains("log.default") {
        "log"
    } else if target.contains("hardsigmoid.default") {
        "hardsigmoid"
    } else if target.contains("sigmoid.default") {
        "sigmoid"
    } else if target.contains("atanh.default") {
        "atanh"
    } else if target.contains("asinh.default") {
        "asinh"
    } else if target.contains("acosh.default") {
        "acosh"
    } else if target.contains("atan.default") {
        "atan"
    } else if target.contains("tanh.default") {
        "tanh"
    } else if target.contains("gelu.default") {
        "gelu"
    } else if target.contains("silu.default") {
        "silu"
    } else if target.contains("erf.default") && !target.contains("erfinv") {
        "erf"
    } else if target.contains("aten.sin.") && !target.contains("sinh") && !target.contains("asin") {
        "sin"
    } else if target.contains("aten.cos.") && !target.contains("cosh") && !target.contains("acos") {
        "cos"
    } else if target.contains("sinh.default") {
        "sinh"
    } else if target.contains("cosh.default") {
        "cosh"
    } else if target.contains("tan.default") {
        "tan"
    } else if target.contains("acos.default") {
        "acos"
    } else if target.contains("asin.default") {
        "asin"
    } else if target.contains("expm1.default") {
        "expm1"
    } else if target.contains("hardswish.default") {
        "hardswish"
    } else if target.contains("erfinv.default") {
        "erfinv"
    } else if target.contains("log1p.default") {
        "log1p"
    } else if target.contains("sqrt.default") {
        "sqrt"
    } else {
        return if target.contains("pow.Tensor_Scalar") {
            node_scalar_arg(node, 1).map(|scalar_arg| TableOpSpec {
                kind: "pow",
                scalar_arg: Some(scalar_arg),
            })
        } else if target.contains("elu.default") || target.contains("elu_") {
            Some(TableOpSpec {
                kind: "elu",
                scalar_arg: Some(node_scalar_arg(node, 1).unwrap_or(1.0)),
            })
        } else {
            None
        };
    };

    Some(TableOpSpec {
        kind,
        scalar_arg: None,
    })
}

fn has_qparams(node: &IrNode) -> bool {
    let has_input = node
        .meta
        .input_qparams
        .as_ref()
        .is_some_and(|qparams| !qparams.is_empty());
    let has_output = node
        .meta
        .output_qparams
        .as_ref()
        .is_some_and(|qparams| !qparams.is_empty());
    has_input && has_output
}

impl IrEpPass for IrInsertTableOpsEpPass {
    fn name(&self) -> &str {
        "IrInsertTableOpsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult {
        let effects = graph
            .nodes
            .iter()
            .filter(|node| node.alive && node.op == NodeOp::CallFunction && has_qparams(node))
            .filter_map(|node| {
                let spec = table_op_spec(node)?;
                Some(FxEpEffect::InsertTableOp(FxInsertTableOpEffect {
                    node_name: node.name.clone(),
                    kind: spec.kind.to_string(),
                    scalar_arg: spec.scalar_arg,
                    allow_missing: true,
                }))
            })
            .collect::<Vec<_>>();

        IrEpPassResult {
            modified: !effects.is_empty(),
            effects: FxEpEffectBatch { effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::fx_control_flow_metadata::QParam;
    use crate::fx_ir::{IrGraph, NodeMeta};

    use super::*;

    fn qparams() -> HashMap<usize, QParam> {
        HashMap::from([(
            0,
            QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string()),
        )])
    }

    fn table_node(graph: &mut IrGraph, name: &str, target: &str, args: Vec<IrArg>) {
        let mut meta = NodeMeta::default();
        meta.input_qparams = Some(qparams());
        meta.output_qparams = Some(qparams());
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            target.to_string(),
            args,
            meta,
        );
    }

    #[test]
    fn plans_quantized_table_ops() {
        let mut graph = IrGraph::empty();
        table_node(&mut graph, "sigmoid", "aten.sigmoid.default", Vec::new());
        table_node(
            &mut graph,
            "pow",
            "aten.pow.Tensor_Scalar",
            vec![IrArg::Int(0), IrArg::Float(2.0)],
        );
        table_node(
            &mut graph,
            "elu",
            "aten.elu.default",
            vec![IrArg::Int(0), IrArg::Float(0.25)],
        );

        let result = IrInsertTableOpsEpPass.run(&mut graph);

        assert!(result.modified);
        let effects = result
            .effects
            .effects
            .iter()
            .map(|effect| match effect {
                FxEpEffect::InsertTableOp(effect) => (
                    effect.node_name.as_str(),
                    effect.kind.as_str(),
                    effect.scalar_arg,
                ),
                _ => panic!("unexpected effect"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            effects,
            vec![
                ("sigmoid", "sigmoid", None),
                ("pow", "pow", Some(2.0)),
                ("elu", "elu", Some(0.25)),
            ]
        );
    }

    #[test]
    fn skips_unquantized_or_unsupported_nodes() {
        let mut graph = IrGraph::empty();
        graph.add_node(
            NodeOp::CallFunction,
            "sigmoid".to_string(),
            "aten.sigmoid.default".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        table_node(&mut graph, "add", "aten.add.Tensor", Vec::new());

        let result = IrInsertTableOpsEpPass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.effects.is_empty());
    }
}
