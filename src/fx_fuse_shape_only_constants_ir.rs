use std::collections::{HashMap, HashSet};

use crate::fx_constant_tensor_eval::plan_shape_only_tensor as shared_plan_shape_only_tensor;
use crate::fx_ep_effects::{
    FxEpEffect, FxEpEffectBatch, FxEpInputKind, FxMaterializeConstantNodeEffect,
};
use crate::fx_ir::{IrArg, IrEpPass, IrEpPassResult, IrGraph, IrNode, NodeOp, TensorSource};
use crate::store::TensorData;

pub struct IrFuseShapeOnlyConstantsEpPass;

fn arg_nodes_are_constant(arg: &IrArg, constant_nodes: &HashSet<usize>) -> bool {
    match arg {
        IrArg::Node(node_id) => constant_nodes.contains(node_id),
        IrArg::List(values) => values
            .iter()
            .all(|value| arg_nodes_are_constant(value, constant_nodes)),
        _ => true,
    }
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

fn is_shape_only_constant_target(target: &str) -> bool {
    target.contains("aten.reshape.default")
        || target.contains("aten.view_copy.default")
        || target.contains("aten.unsqueeze_copy.default")
        || target.contains("aten.permute_copy.default")
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

fn first_constant_tensor<'a>(
    node: &IrNode,
    constant_tensors: &'a HashMap<usize, TensorData>,
) -> Option<&'a TensorData> {
    node.args.iter().find_map(|arg| match arg {
        IrArg::Node(node_id) => constant_tensors.get(node_id),
        _ => None,
    })
}

fn plan_shape_only_tensor(
    node: &IrNode,
    constant_tensors: &HashMap<usize, TensorData>,
) -> Option<TensorData> {
    first_constant_tensor(node, constant_tensors)
        .and_then(|input| shared_plan_shape_only_tensor(node, input))
}

impl IrEpPass for IrFuseShapeOnlyConstantsEpPass {
    fn name(&self) -> &str {
        "IrFuseShapeOnlyConstantsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult {
        let mut constant_nodes: HashSet<usize> = graph
            .nodes
            .iter()
            .filter(|node| placeholder_is_constant_input(node))
            .map(|node| node.id)
            .collect();
        let mut constant_tensors: HashMap<usize, TensorData> = graph
            .nodes
            .iter()
            .filter(|node| placeholder_is_constant_input(node))
            .filter_map(|node| Some((node.id, node.tensor.as_ref()?.data.clone())))
            .collect();

        let mut effects = Vec::new();
        for node in &graph.nodes {
            if !node.alive || node.op != NodeOp::CallFunction {
                continue;
            }
            if node.meta.extra_bool("arm_preserve_shape_node") == Some(true) {
                continue;
            }
            if !is_shape_only_constant_target(&node.target) {
                continue;
            }
            if !node_inputs_are_constant(node, &constant_nodes) {
                continue;
            }
            let tensor = match plan_shape_only_tensor(node, &constant_tensors) {
                Some(tensor) => tensor,
                None => continue,
            };
            constant_nodes.insert(node.id);
            constant_tensors.insert(node.id, tensor.clone());
            effects.push(FxEpEffect::MaterializeConstantNode(
                FxMaterializeConstantNodeEffect {
                    node_name: node.name.clone(),
                    data: tensor.data,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    allow_missing: true,
                },
            ));
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

    use super::*;

    fn constant_placeholder(graph: &mut IrGraph, name: &str, data: TensorData) -> usize {
        let node_id = graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph.node_mut(node_id).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: FxEpInputKind::Buffer,
                target_name: format!("state.{name}"),
            }),
            data,
        });
        node_id
    }

    #[test]
    fn plans_shape_only_constant_chain() {
        let mut graph = IrGraph::empty();
        let weight = constant_placeholder(
            &mut graph,
            "weight",
            TensorData {
                data: [1i32, 2, 3, 4]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
                shape: vec![4],
                dtype: DType::Int32,
            },
        );
        let view = graph.add_node(
            NodeOp::CallFunction,
            "reshape".to_string(),
            "aten.reshape.default".to_string(),
            vec![
                IrArg::Node(weight),
                IrArg::List(vec![IrArg::Int(2), IrArg::Int(2)]),
            ],
            NodeMeta::default(),
        );
        let perm = graph.add_node(
            NodeOp::CallFunction,
            "perm".to_string(),
            "aten.permute_copy.default".to_string(),
            vec![
                IrArg::Node(view),
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(0)]),
            ],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "unsq".to_string(),
            "aten.unsqueeze_copy.default".to_string(),
            vec![IrArg::Node(perm), IrArg::Int(0)],
            NodeMeta::default(),
        );

        let pass = IrFuseShapeOnlyConstantsEpPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.effects.len(), 3);
        let effects = result
            .effects
            .effects
            .iter()
            .map(|effect| match effect {
                FxEpEffect::MaterializeConstantNode(effect) => effect.clone(),
                _ => panic!("unexpected effect kind"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            effects
                .iter()
                .map(|effect| effect.node_name.as_str())
                .collect::<Vec<_>>(),
            vec!["reshape", "perm", "unsq"]
        );
        assert_eq!(effects[0].shape, vec![2, 2]);
        assert_eq!(effects[1].shape, vec![2, 2]);
        assert_eq!(
            effects[1]
                .data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![1, 3, 2, 4]
        );
        assert_eq!(effects[2].shape, vec![1, 2, 2]);
    }

    #[test]
    fn skips_non_constant_inputs_and_unsupported_targets() {
        let mut graph = IrGraph::empty();
        let runtime = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "view".to_string(),
            "aten.view_copy.default".to_string(),
            vec![
                IrArg::Node(runtime),
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(4)]),
            ],
            NodeMeta::default(),
        );
        let weight = constant_placeholder(
            &mut graph,
            "weight",
            TensorData {
                data: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::Float32,
            },
        );
        graph.add_node(
            NodeOp::CallFunction,
            "other".to_string(),
            "aten.slice_copy.Tensor".to_string(),
            vec![
                IrArg::Node(weight),
                IrArg::Int(0),
                IrArg::Int(0),
                IrArg::Int(1),
            ],
            NodeMeta::default(),
        );

        let pass = IrFuseShapeOnlyConstantsEpPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.effects.is_empty());
    }
}
