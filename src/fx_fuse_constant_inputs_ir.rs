use std::collections::HashSet;

use crate::fx_ep_effects::{FxEpEffect, FxEpEffectBatch, FxEpInputKind, FxFuseConstantNodeEffect};
use crate::fx_ir::{IrArg, IrEpPass, IrEpPassResult, IrGraph, IrNode, NodeOp, TensorSource};

pub struct IrFuseConstantInputsEpPass {
    include_data_slices: bool,
}

impl IrFuseConstantInputsEpPass {
    pub const fn matmul_only() -> Self {
        Self { include_data_slices: false }
    }

    pub const fn with_data_slices() -> Self {
        Self { include_data_slices: true }
    }
}

fn arg_has_node(arg: &IrArg) -> bool {
    match arg {
        IrArg::Node(_) => true,
        IrArg::List(values) => values.iter().any(arg_has_node),
        _ => false,
    }
}

fn arg_nodes_are_constant(arg: &IrArg, constant_nodes: &HashSet<usize>) -> bool {
    match arg {
        IrArg::Node(node_id) => constant_nodes.contains(node_id),
        IrArg::List(values) => values.iter().all(|value| arg_nodes_are_constant(value, constant_nodes)),
        _ => true,
    }
}

fn node_inputs_are_constant(node: &IrNode, constant_nodes: &HashSet<usize>) -> bool {
    node.args
        .iter()
        .all(|arg| arg_nodes_are_constant(arg, constant_nodes))
        && node
            .kwargs
            .values()
            .all(|arg| arg_nodes_are_constant(arg, constant_nodes))
}

fn placeholder_is_constant_input(node: &IrNode) -> bool {
    if !node.alive || node.op != NodeOp::Placeholder {
        return false;
    }
    matches!(
        node.tensor.as_ref().map(|tensor| &tensor.source),
        Some(TensorSource::ExportedProgramInput(binding))
            if matches!(
                binding.input_kind,
                FxEpInputKind::Parameter
                    | FxEpInputKind::Buffer
                    | FxEpInputKind::LiftedTensorConstant
            )
    )
}

fn is_zero_input_constant_target(target: &str) -> bool {
    target.contains("aten.full.default")
        || target.contains("aten.arange")
        || target.contains("aten.linspace")
        || target.contains("aten.eye")
}

fn is_shape_only_constant_target(target: &str) -> bool {
    target.contains("aten.view_copy.default")
        || target.contains("aten.unsqueeze_copy.default")
        || target.contains("aten.permute_copy.default")
}

fn is_safe_constant_input_target(target: &str, include_data_slices: bool) -> bool {
    target.contains("aten.addmm.default")
        || target.contains("aten.mm.default")
        || target.contains("aten.bmm.default")
        || (include_data_slices && target.contains("aten.slice_copy.Tensor"))
}

impl IrEpPass for IrFuseConstantInputsEpPass {
    fn name(&self) -> &str { "IrFuseConstantInputsEp" }

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult {
        let mut constant_nodes: HashSet<usize> = graph
            .nodes
            .iter()
            .filter(|node| placeholder_is_constant_input(node))
            .map(|node| node.id)
            .collect();

        let mut effects = Vec::new();
        for node in &graph.nodes {
            if !node.alive || node.op != NodeOp::CallFunction {
                continue;
            }
            let has_tensor_inputs =
                node.args.iter().any(arg_has_node) || node.kwargs.values().any(arg_has_node);
            if !has_tensor_inputs {
                if is_zero_input_constant_target(&node.target) {
                    constant_nodes.insert(node.id);
                }
                continue;
            }
            if !node_inputs_are_constant(node, &constant_nodes) {
                continue;
            }
            if is_shape_only_constant_target(&node.target) {
                constant_nodes.insert(node.id);
                continue;
            }
            if !is_safe_constant_input_target(&node.target, self.include_data_slices) {
                continue;
            }
            constant_nodes.insert(node.id);
            effects.push(FxEpEffect::FuseConstantNode(FxFuseConstantNodeEffect {
                node_name: node.name.clone(),
                allow_missing: true,
            }));
        }

        let modified = !effects.is_empty();
        IrEpPassResult {
            modified,
            effects: FxEpEffectBatch { effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrGraph, NodeMeta, NodeTensor, PlaceholderTensorBinding};
    use crate::graph::ir::DType;
    use crate::store::TensorData;

    use super::*;

    fn constant_placeholder(graph: &mut IrGraph, name: &str) -> usize {
        let node_id = graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph.node_mut(node_id).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: FxEpInputKind::Parameter,
                target_name: format!("state.{name}"),
            }),
            data: TensorData {
                data: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::Float32,
            },
        });
        node_id
    }

    #[test]
    fn plans_matmul_ops_after_zero_input_and_shape_only_constant_chain() {
        let mut graph = IrGraph::empty();
        let weight = constant_placeholder(&mut graph, "weight");
        let bias = constant_placeholder(&mut graph, "bias");
        let full = graph.add_node(
            NodeOp::CallFunction,
            "full".to_string(),
            "aten.full.default".to_string(),
            vec![IrArg::List(vec![IrArg::Int(1)]), IrArg::Float(1.0)],
            NodeMeta::default(),
        );
        let view = graph.add_node(
            NodeOp::CallFunction,
            "view".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(full), IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)])],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "addmm".to_string(),
            "aten.addmm.default".to_string(),
            vec![IrArg::Node(bias), IrArg::Node(view), IrArg::Node(weight)],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "mm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(view), IrArg::Node(weight)],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "bmm".to_string(),
            "aten.bmm.default".to_string(),
            vec![IrArg::Node(view), IrArg::Node(weight)],
            NodeMeta::default(),
        );

        let pass = IrFuseConstantInputsEpPass::matmul_only();
        let result = pass.run(&mut graph);

        assert!(result.modified);
        let names = result
            .effects
            .effects
            .iter()
            .map(|effect| match effect {
                FxEpEffect::FuseConstantNode(effect) => effect.node_name.as_str(),
                _ => panic!("unexpected effect kind"),
            })
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["addmm", "mm", "bmm"]);
    }

    #[test]
    fn skips_runtime_or_unsupported_constant_nodes() {
        let mut graph = IrGraph::empty();
        let runtime = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        let weight = constant_placeholder(&mut graph, "weight");
        graph.add_node(
            NodeOp::CallFunction,
            "runtime_addmm".to_string(),
            "aten.addmm.default".to_string(),
            vec![IrArg::Node(weight), IrArg::Node(runtime), IrArg::Node(weight)],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "unsupported".to_string(),
            "aten.cat.default".to_string(),
            vec![IrArg::List(vec![IrArg::Node(weight), IrArg::Node(weight)])],
            NodeMeta::default(),
        );

        let pass = IrFuseConstantInputsEpPass::matmul_only();
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.effects.is_empty());
    }

    #[test]
    fn data_slice_target_is_opt_in() {
        let mut graph = IrGraph::empty();
        let weight = constant_placeholder(&mut graph, "weight");
        graph.add_node(
            NodeOp::CallFunction,
            "weight_slice".to_string(),
            "aten.slice_copy.Tensor".to_string(),
            vec![IrArg::Node(weight), IrArg::Int(0), IrArg::Int(0), IrArg::Int(4)],
            NodeMeta::default(),
        );

        let disabled = IrFuseConstantInputsEpPass::matmul_only().run(&mut graph);
        assert!(!disabled.modified);

        let enabled = IrFuseConstantInputsEpPass::with_data_slices().run(&mut graph);
        assert!(enabled.modified);
        assert_eq!(enabled.effects.effects.len(), 1);
    }
}
