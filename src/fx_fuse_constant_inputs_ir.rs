use std::collections::{HashMap, HashSet};

use crate::fx_constant_tensor_eval::{
    plan_constant_matmul_tensor as shared_plan_constant_matmul_tensor,
    plan_constant_slice_tensor as shared_plan_constant_slice_tensor,
    plan_shape_only_tensor as shared_plan_shape_only_tensor,
};
use crate::fx_ep_effects::{
    FxEpEffect, FxEpEffectBatch, FxEpInputKind, FxFuseConstantNodeEffect,
    FxMaterializeConstantNodeEffect,
};
use crate::fx_ir::{IrArg, IrEpPass, IrEpPassResult, IrGraph, IrNode, NodeOp, TensorSource};
use crate::store::TensorData;

pub struct IrFuseConstantInputsEpPass {
    include_data_slices: bool,
}

impl IrFuseConstantInputsEpPass {
    pub const fn matmul_only() -> Self {
        Self {
            include_data_slices: false,
        }
    }

    pub const fn with_data_slices() -> Self {
        Self {
            include_data_slices: true,
        }
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
        IrArg::List(values) => values
            .iter()
            .all(|value| arg_nodes_are_constant(value, constant_nodes)),
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

fn input_constant_tensor<'a>(
    node: &IrNode,
    constant_tensors: &'a HashMap<usize, TensorData>,
) -> Option<&'a TensorData> {
    match node.args.first()? {
        IrArg::Node(node_id) => constant_tensors.get(node_id),
        _ => None,
    }
}

fn plan_shape_only_tensor(
    node: &IrNode,
    constant_tensors: &HashMap<usize, TensorData>,
) -> Option<TensorData> {
    input_constant_tensor(node, constant_tensors)
        .and_then(|input| shared_plan_shape_only_tensor(node, input))
}

fn plan_constant_slice_tensor(
    node: &IrNode,
    constant_tensors: &HashMap<usize, TensorData>,
) -> Option<TensorData> {
    input_constant_tensor(node, constant_tensors)
        .and_then(|input| shared_plan_constant_slice_tensor(node, input))
}

fn plan_constant_matmul_tensor(
    node: &IrNode,
    constant_tensors: &HashMap<usize, TensorData>,
) -> Option<TensorData> {
    shared_plan_constant_matmul_tensor(node, constant_tensors)
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
    target.contains("aten.reshape.default")
        || target.contains("aten.view_copy.default")
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
    fn name(&self) -> &str {
        "IrFuseConstantInputsEp"
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
                if let Some(tensor) = plan_shape_only_tensor(node, &constant_tensors) {
                    constant_tensors.insert(node.id, tensor);
                }
                continue;
            }
            if !is_safe_constant_input_target(&node.target, self.include_data_slices) {
                continue;
            }
            constant_nodes.insert(node.id);
            if let Some(tensor) = plan_constant_matmul_tensor(node, &constant_tensors) {
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
                continue;
            }
            if self.include_data_slices && node.target.contains("aten.slice_copy.Tensor") {
                if let Some(tensor) = plan_constant_slice_tensor(node, &constant_tensors) {
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
                    continue;
                }
            }
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
    use crate::fx_ir::{IrGraph, NodeMeta, NodeTensor, PlaceholderTensorBinding, QParam};
    use crate::graph::ir::DType;
    use crate::store::TensorData;

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
                input_kind: FxEpInputKind::Parameter,
                target_name: format!("state.{name}"),
            }),
            data,
        });
        node_id
    }

    #[test]
    fn plans_matmul_ops_after_zero_input_and_shape_only_constant_chain() {
        let mut graph = IrGraph::empty();
        let weight = constant_placeholder(
            &mut graph,
            "weight",
            TensorData {
                data: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::Float32,
            },
        );
        let bias = constant_placeholder(
            &mut graph,
            "bias",
            TensorData {
                data: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::Float32,
            },
        );
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
            vec![
                IrArg::Node(full),
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),
            ],
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
            "runtime_addmm".to_string(),
            "aten.addmm.default".to_string(),
            vec![
                IrArg::Node(weight),
                IrArg::Node(runtime),
                IrArg::Node(weight),
            ],
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
            "weight_slice".to_string(),
            "aten.slice_copy.Tensor".to_string(),
            vec![
                IrArg::Node(weight),
                IrArg::Int(0),
                IrArg::Int(0),
                IrArg::Int(4),
            ],
            NodeMeta::default(),
        );

        let disabled = IrFuseConstantInputsEpPass::matmul_only().run(&mut graph);
        assert!(!disabled.modified);

        let enabled = IrFuseConstantInputsEpPass::with_data_slices().run(&mut graph);
        assert!(enabled.modified);
        assert_eq!(enabled.effects.effects.len(), 1);
        match &enabled.effects.effects[0] {
            FxEpEffect::MaterializeConstantNode(effect) => {
                assert_eq!(effect.node_name, "weight_slice");
                assert_eq!(effect.shape, vec![4]);
            }
            other => panic!("unexpected effect kind: {other:?}"),
        }
    }

    #[test]
    fn materializes_slice_after_shape_only_constant_chain() {
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
            "view".to_string(),
            "aten.reshape.default".to_string(),
            vec![
                IrArg::Node(weight),
                IrArg::List(vec![IrArg::Int(2), IrArg::Int(2)]),
            ],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "view_slice".to_string(),
            "aten.slice_copy.Tensor".to_string(),
            vec![
                IrArg::Node(view),
                IrArg::Int(0),
                IrArg::Int(1),
                IrArg::Int(2),
            ],
            NodeMeta::default(),
        );

        let result = IrFuseConstantInputsEpPass::with_data_slices().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.effects[0] {
            FxEpEffect::MaterializeConstantNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "view_slice");
        assert_eq!(effect.shape, vec![1, 2]);
        let values = effect
            .data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![3, 4]);
    }

    fn float32_tensor(shape: Vec<i64>, values: &[f32]) -> TensorData {
        TensorData {
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            shape,
            dtype: DType::Float32,
        }
    }

    #[test]
    fn materializes_constant_mm_and_addmm() {
        let mut graph = IrGraph::empty();
        let lhs = constant_placeholder(
            &mut graph,
            "lhs",
            float32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let rhs = constant_placeholder(
            &mut graph,
            "rhs",
            float32_tensor(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        let bias = constant_placeholder(&mut graph, "bias", float32_tensor(vec![2], &[0.5, 1.5]));

        graph.add_node(
            NodeOp::CallFunction,
            "mm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            NodeMeta::default(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "addmm".to_string(),
            "aten.addmm.default".to_string(),
            vec![IrArg::Node(bias), IrArg::Node(lhs), IrArg::Node(rhs)],
            NodeMeta::default(),
        );

        let result = IrFuseConstantInputsEpPass::matmul_only().run(&mut graph);
        assert!(result.modified);

        let mut seen = HashMap::new();
        for effect in &result.effects.effects {
            let FxEpEffect::MaterializeConstantNode(effect) = effect else {
                continue;
            };
            seen.insert(effect.node_name.clone(), effect.clone());
        }

        let mm = seen.get("mm").expect("missing mm effect");
        let mm_values = mm
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(mm.shape, vec![2, 2]);
        assert_eq!(mm_values, vec![58.0, 64.0, 139.0, 154.0]);

        let addmm = seen.get("addmm").expect("missing addmm effect");
        let addmm_values = addmm
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(addmm.shape, vec![2, 2]);
        assert_eq!(addmm_values, vec![58.5, 65.5, 139.5, 155.5]);
    }

    #[test]
    fn materializes_quantized_constant_mm() {
        let mut graph = IrGraph::empty();
        let lhs = constant_placeholder(
            &mut graph,
            "qlhs",
            TensorData {
                data: vec![1u8, 2u8],
                shape: vec![1, 2],
                dtype: DType::Int8,
            },
        );
        let rhs = constant_placeholder(
            &mut graph,
            "qrhs",
            TensorData {
                data: vec![3u8, 4u8],
                shape: vec![2, 1],
                dtype: DType::Int8,
            },
        );

        let mut meta = NodeMeta::default();
        let in_qp = QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string());
        let out_qp = QParam::per_tensor(0.25, 0, -128, 127, "torch.int8".to_string());
        meta.set_input_qparams(HashMap::from([(0, in_qp.clone()), (1, in_qp)]));
        meta.set_single_output_qparam(out_qp);
        graph.add_node(
            NodeOp::CallFunction,
            "qmm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            meta,
        );

        let result = IrFuseConstantInputsEpPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.effects[0] {
            FxEpEffect::MaterializeConstantNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "qmm");
        assert_eq!(effect.shape, vec![1, 1]);
        assert_eq!(effect.dtype, DType::Int8);
        assert_eq!(effect.data, vec![11u8]);
    }
}
