use crate::fx_constant_tensor_eval::{
    constant_input_seeds, node_inputs_are_constant, plan_shape_only_tensor_from_constants,
};
use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, NodeOp};
use crate::fx_target::is_shape_only_constant_target;
use crate::lowering_effects::{LoweringEffect, LoweringEffectBatch, MaterializeTensorNodeEffect};

pub struct IrFuseShapeOnlyConstantsEffectPass;

impl EffectfulIrPass for IrFuseShapeOnlyConstantsEffectPass {
    fn name(&self) -> &str {
        "IrFuseShapeOnlyConstantsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let (mut constant_nodes, mut constant_tensors) = constant_input_seeds(graph);

        let mut effects = Vec::new();
        for node in &graph.nodes {
            if !node.alive || node.op != NodeOp::CallFunction {
                continue;
            }
            if !is_shape_only_constant_target(&node.target) {
                continue;
            }
            if !node_inputs_are_constant(node, &constant_nodes) {
                continue;
            }
            let tensor = match plan_shape_only_tensor_from_constants(node, &constant_tensors) {
                Some(tensor) => tensor,
                None => continue,
            };
            constant_nodes.insert(node.id);
            constant_tensors.insert(node.id, tensor.clone());
            effects.push(LoweringEffect::MaterializeTensorNode(
                MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
            ));
        }

        let modified = !effects.is_empty();
        EffectfulIrPassResult {
            modified,
            effects: LoweringEffectBatch { items: effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, NodeOp};
    use crate::graph::ir::DType;
    use crate::store::TensorData;
    use crate::test_ir::buffer as constant_placeholder;

    use super::*;

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

        let pass = IrFuseShapeOnlyConstantsEffectPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.items.len(), 3);
        let effects = result
            .effects
            .items
            .iter()
            .map(|effect| match effect {
                LoweringEffect::MaterializeTensorNode(effect) => effect.clone(),
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

        let pass = IrFuseShapeOnlyConstantsEffectPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }
}
