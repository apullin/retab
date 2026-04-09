use crate::fx_ep_effects::{
    FxEpEffect, FxEpEffectBatch, FxMaterializeZeroInputConstantEffect,
};
use crate::fx_fuse_constant_args::ZeroInputConstantKind;
use crate::fx_ir::{IrArg, IrEpPass, IrEpPassResult, IrGraph, IrNode, NodeOp};

pub struct IrFuseZeroInputConstantsEpPass;

fn arg_has_node(arg: &IrArg) -> bool {
    match arg {
        IrArg::Node(_) => true,
        IrArg::List(values) => values.iter().any(arg_has_node),
        _ => false,
    }
}

fn has_tensor_inputs(node: &IrNode) -> bool {
    node.args.iter().any(arg_has_node) || node.kwargs.values().any(arg_has_node)
}

fn planned_zero_input_constant_kind(node: &IrNode) -> Option<ZeroInputConstantKind> {
    if !node.alive || node.op != NodeOp::CallFunction || has_tensor_inputs(node) {
        return None;
    }
    if node.target.contains("aten.full.default") {
        return Some(ZeroInputConstantKind::Full);
    }
    if node.target.contains("arange")
        || node.target.contains("linspace")
        || node.target.contains("eye")
    {
        return Some(ZeroInputConstantKind::FactoryCall);
    }
    None
}

impl IrEpPass for IrFuseZeroInputConstantsEpPass {
    fn name(&self) -> &str { "IrFuseZeroInputConstantsEp" }

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult {
        let effects = graph
            .nodes
            .iter()
            .filter_map(|node| {
                let kind = planned_zero_input_constant_kind(node)?;
                Some(FxEpEffect::MaterializeZeroInputConstant(
                    FxMaterializeZeroInputConstantEffect {
                        node_name: node.name.clone(),
                        kind,
                        allow_missing: true,
                    },
                ))
            })
            .collect::<Vec<_>>();
        let modified = !effects.is_empty();
        IrEpPassResult {
            modified,
            effects: FxEpEffectBatch { effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrGraph, NodeMeta};

    use super::*;

    fn add_call(graph: &mut IrGraph, name: &str, target: &str, args: Vec<IrArg>) -> usize {
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            target.to_string(),
            args,
            NodeMeta::default(),
        )
    }

    #[test]
    fn plans_zero_input_factory_constants() {
        let mut graph = IrGraph::empty();
        add_call(&mut graph, "full", "aten.full.default", vec![IrArg::List(vec![IrArg::Int(2)]), IrArg::Float(3.0)]);
        add_call(&mut graph, "arange", "aten.arange.start", vec![IrArg::Int(7)]);
        add_call(&mut graph, "linspace", "aten.linspace.default", vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)]);
        add_call(&mut graph, "eye", "aten.eye.default", vec![IrArg::Int(4)]);

        let pass = IrFuseZeroInputConstantsEpPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.effects.len(), 4);
    }

    #[test]
    fn skips_nodes_with_tensor_inputs() {
        let mut graph = IrGraph::empty();
        let placeholder = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        add_call(
            &mut graph,
            "not_zero_input",
            "aten.full.default",
            vec![IrArg::Node(placeholder), IrArg::Int(3)],
        );

        let pass = IrFuseZeroInputConstantsEpPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.effects.is_empty());
    }
}
