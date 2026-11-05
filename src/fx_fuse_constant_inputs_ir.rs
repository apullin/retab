
use crate::fx_constant_tensor_eval::{
    constant_input_seeds, ConstantTensors, first_constant_input_tensor, node_has_tensor_inputs,
    node_inputs_are_constant, plan_constant_cast_tensor, plan_constant_comparison_tensors,
    plan_constant_conv_tensor, plan_constant_linear_tensor, plan_constant_matmul_tensor,
    plan_constant_reciprocal_tensor, plan_constant_slice_tensor_from_constants,
    plan_shape_only_tensor_from_constants,
};
use crate::fx_fuse_zero_input_constants_ir::zero_input_tensor;
use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrArg, IrGraph, IrNode, NodeOp};
use crate::fx_target::{
    is_comparison_target, is_reciprocal_target, is_shape_only_constant_target,
    is_slice_copy_tensor_target, is_to_copy_target, is_zero_input_constant_target,
};
use crate::lowering_effects::{LoweringEffect, LoweringEffectBatch, MaterializeTensorNodeEffect};
use crate::store::TensorData;

pub struct IrFuseConstantInputsEffectPass {
    include_data_slices: bool,
}

impl IrFuseConstantInputsEffectPass {
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

fn is_safe_constant_input_target(node: &IrNode, include_data_slices: bool) -> bool {
    node.is_addmm()
        || node.is_mm()
        || node.is_bmm()
        || node.is_linear()
        || crate::fx_conv_args::ConvArgs::parse(node).is_some()
        || (include_data_slices && is_slice_copy_tensor_target(&node.target))
        || is_to_copy_target(&node.target)
        || is_comparison_target(&node.target)
        || is_reciprocal_target(&node.target)
}

/// Stock's FuseConstantArgsPass skips fusing when the node carries the SHAPE
/// special dtype; the folded bytes would lose that marker.
fn has_shape_special_dtype(node: &IrNode) -> bool {
    node.meta.extra_str("tosa_special_dtype") == Some("SHAPE")
}

fn plan_elementwise_constant_tensor(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    if has_shape_special_dtype(node) {
        return None;
    }
    if is_to_copy_target(&node.target) {
        let input = first_constant_input_tensor(node, constant_tensors)?;
        return plan_constant_cast_tensor(node, input);
    }
    if is_reciprocal_target(&node.target) {
        let input = first_constant_input_tensor(node, constant_tensors)?;
        return plan_constant_reciprocal_tensor(node, input);
    }
    if is_comparison_target(&node.target) {
        let mut inputs = node.args.iter().filter_map(|arg| match arg {
            IrArg::Node(node_id) => constant_tensors.get(node_id),
            _ => None,
        });
        let lhs = inputs.next()?;
        let rhs = inputs.next()?;
        return plan_constant_comparison_tensors(node, lhs, rhs);
    }
    None
}

impl EffectfulIrPass for IrFuseConstantInputsEffectPass {
    fn name(&self) -> &str {
        "IrFuseConstantInputsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let (mut constant_nodes, mut constant_tensors) = constant_input_seeds(graph);
        // Seed zero-input factories (full/arange/.../scalar_tensor) with their
        // planned bytes so folds chain through them the way stock's
        // ComputeConstantOpsAOTPass + FuseConstantArgsPass sequence does.
        for node in &graph.nodes {
            if !node.alive
                || node.op != NodeOp::CallFunction
                || node_has_tensor_inputs(node)
                || !is_zero_input_constant_target(&node.target)
            {
                continue;
            }
            if let Some(tensor) = zero_input_tensor(graph, node) {
                constant_nodes.insert(node.id);
                constant_tensors.insert(node.id, tensor);
            }
        }

        let mut effects = Vec::new();
        for node in &graph.nodes {
            if !node.alive || node.op != NodeOp::CallFunction {
                continue;
            }
            if !node_has_tensor_inputs(node) {
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
                if let Some(tensor) = plan_shape_only_tensor_from_constants(node, &constant_tensors)
                {
                    // Materialize here as well as in the shape-only sweep: an
                    // intermediate reshape (e.g. over a folded `_to_copy`) is
                    // only known constant inside this sweep. Re-materializing
                    // a node the shape-only sweep already covered is harmless
                    // (the lowering effect is idempotent).
                    constant_tensors.insert(node.id, tensor.clone());
                    effects.push(LoweringEffect::MaterializeTensorNode(
                        MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                    ));
                }
                continue;
            }
            if !is_safe_constant_input_target(node, self.include_data_slices) {
                continue;
            }
            constant_nodes.insert(node.id);
            if let Some(tensor) = plan_elementwise_constant_tensor(node, &constant_tensors) {
                constant_tensors.insert(node.id, tensor.clone());
                effects.push(LoweringEffect::MaterializeTensorNode(
                    MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                ));
                continue;
            }
            if let Some(tensor) = plan_constant_matmul_tensor(node, &constant_tensors) {
                constant_tensors.insert(node.id, tensor.clone());
                effects.push(LoweringEffect::MaterializeTensorNode(
                    MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                ));
                continue;
            }
            if let Some(tensor) = plan_constant_linear_tensor(node, &constant_tensors) {
                constant_tensors.insert(node.id, tensor.clone());
                effects.push(LoweringEffect::MaterializeTensorNode(
                    MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                ));
                continue;
            }
            if let Some(tensor) = plan_constant_conv_tensor(node, &constant_tensors) {
                constant_tensors.insert(node.id, tensor.clone());
                effects.push(LoweringEffect::MaterializeTensorNode(
                    MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                ));
                continue;
            }
            if self.include_data_slices && is_slice_copy_tensor_target(&node.target) {
                if let Some(tensor) =
                    plan_constant_slice_tensor_from_constants(node, &constant_tensors)
                {
                    constant_tensors.insert(node.id, tensor.clone());
                    effects.push(LoweringEffect::MaterializeTensorNode(
                        MaterializeTensorNodeEffect::from_tensor(node.name.clone(), tensor, true),
                    ));
                    continue;
                }
            }
            crate::fold_fallback_log(&format!(
                "ConstantInputPrecomputeSkipped target={}",
                node.target
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
    use std::collections::HashMap;

    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, NodeOp, QParam};
    use crate::graph::ir::DType;
    use crate::store::TensorData;
    use crate::test_ir::{f32_tensor as float32_tensor, parameter as constant_placeholder};

    use super::*;

    #[test]
    fn folds_scalar_cast_reshape_and_gt_chain() {
        // Integer div floor-correction pattern with a scalar divisor: the
        // scalar `_to_copy` chain, the broadcast reshape, and the
        // all-constant `gt` all fold in one sweep.
        let mut graph = IrGraph::empty();
        let scalar = constant_placeholder(
            &mut graph,
            "scalar",
            TensorData {
                data: 9i64.to_le_bytes().to_vec(),
                shape: vec![],
                dtype: DType::Int64,
            },
        );
        let cast = graph.add_node(
            NodeOp::CallFunction,
            "rhs_int32".to_string(),
            "aten._to_copy.default".to_string(),
            vec![IrArg::Node(scalar)],
            NodeMeta::tensor(vec![], "torch.int32"),
        );
        graph.set_kwarg(cast, "dtype", IrArg::Dtype("torch.int32".to_string()));
        let reshape = graph.add_node(
            NodeOp::CallFunction,
            "rhs_broadcast".to_string(),
            "aten.reshape.default".to_string(),
            vec![
                IrArg::Node(cast),
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),
            ],
            NodeMeta::tensor(vec![1, 1], "torch.int32"),
        );
        let zero = graph.add_node(
            NodeOp::CallFunction,
            "zero".to_string(),
            "aten.full.default".to_string(),
            vec![
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),
                IrArg::Int(0),
            ],
            NodeMeta::tensor(vec![1, 1], "torch.int32"),
        );
        graph.set_kwarg(zero, "dtype", IrArg::Dtype("torch.int32".to_string()));
        graph.add_node(
            NodeOp::CallFunction,
            "rhs_negative".to_string(),
            "aten.gt.Tensor".to_string(),
            vec![IrArg::Node(zero), IrArg::Node(reshape)],
            NodeMeta::tensor(vec![1, 1], "torch.bool"),
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let mut seen = HashMap::new();
        for effect in &result.effects.items {
            let LoweringEffect::MaterializeTensorNode(effect) = effect else {
                continue;
            };
            seen.insert(effect.node_name.clone(), effect.clone());
        }
        let cast_effect = seen.get("rhs_int32").expect("missing cast effect");
        assert_eq!(cast_effect.dtype, DType::Int32);
        assert_eq!(
            cast_effect.data,
            9i32.to_le_bytes().to_vec(),
            "scalar survives the int64->int32 cast"
        );
        let reshape_effect = seen.get("rhs_broadcast").expect("missing reshape effect");
        assert_eq!(reshape_effect.shape, vec![1, 1]);
        assert_eq!(reshape_effect.data, 9i32.to_le_bytes().to_vec());
        // 0 > 9 is false.
        let gt_effect = seen.get("rhs_negative").expect("missing gt effect");
        assert_eq!(gt_effect.dtype, DType::Bool);
        assert_eq!(gt_effect.data, vec![0]);
    }

    #[test]
    fn folds_quantized_reciprocal_scalar() {
        let mut graph = IrGraph::empty();
        let divisor = constant_placeholder(
            &mut graph,
            "divisor",
            TensorData {
                data: 2.0f32.to_le_bytes().to_vec(),
                shape: vec![1, 1],
                dtype: DType::Float32,
            },
        );
        let mut meta = NodeMeta::tensor(vec![1, 1], "torch.float32");
        meta.set_single_output_qparam(QParam::per_tensor(
            0.5,
            0,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        graph.add_node(
            NodeOp::CallFunction,
            "recip".to_string(),
            "aten.reciprocal.default".to_string(),
            vec![IrArg::Node(divisor)],
            meta,
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "recip");
        assert_eq!(effect.dtype, DType::Int8);
        assert_eq!(effect.data, vec![1u8]);
    }

    #[test]
    fn skips_shape_marked_constant_cast() {
        let mut graph = IrGraph::empty();
        let scalar = constant_placeholder(
            &mut graph,
            "scalar",
            TensorData {
                data: 9i64.to_le_bytes().to_vec(),
                shape: vec![],
                dtype: DType::Int64,
            },
        );
        let cast = graph.add_node(
            NodeOp::CallFunction,
            "rhs_int32".to_string(),
            "aten._to_copy.default".to_string(),
            vec![IrArg::Node(scalar)],
            NodeMeta::tensor(vec![], "torch.int32"),
        );
        graph.set_kwarg(cast, "dtype", IrArg::Dtype("torch.int32".to_string()));
        graph.node_mut(cast).meta.set_extra(
            "tosa_special_dtype",
            crate::fx_ir::MetaValue::Str("SHAPE".to_string()),
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
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

        let pass = IrFuseConstantInputsEffectPass::matmul_only();
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
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

        let pass = IrFuseConstantInputsEffectPass::matmul_only();
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
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

        let disabled = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(!disabled.modified);

        let enabled = IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph);
        assert!(enabled.modified);
        assert_eq!(enabled.effects.items.len(), 1);
        match &enabled.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => {
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

        let result = IrFuseConstantInputsEffectPass::with_data_slices().run(&mut graph);
        assert!(result.modified);
        // The intermediate reshape materializes in the same sweep so the
        // slice sees a constant even though the reshape's producer folded here.
        let mut seen = HashMap::new();
        for effect in &result.effects.items {
            let LoweringEffect::MaterializeTensorNode(effect) = effect else {
                continue;
            };
            seen.insert(effect.node_name.clone(), effect.clone());
        }
        let view_effect = seen.get("view").expect("missing view effect");
        assert_eq!(view_effect.shape, vec![2, 2]);
        let effect = seen.get("view_slice").expect("missing view_slice effect");
        assert_eq!(effect.shape, vec![1, 2]);
        let values = effect
            .data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![3, 4]);
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

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);

        let mut seen = HashMap::new();
        for effect in &result.effects.items {
            let LoweringEffect::MaterializeTensorNode(effect) = effect else {
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
        meta.set_binary_input_output_qparams(in_qp.clone(), in_qp, out_qp);
        graph.add_node(
            NodeOp::CallFunction,
            "qmm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            meta,
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "qmm");
        assert_eq!(effect.shape, vec![1, 1]);
        assert_eq!(effect.dtype, DType::Int8);
        assert_eq!(effect.data, vec![11u8]);
    }

    #[test]
    fn materializes_quantized_constant_linear() {
        let mut graph = IrGraph::empty();
        let input = constant_placeholder(
            &mut graph,
            "qinput",
            TensorData {
                data: vec![1u8, 2u8],
                shape: vec![1, 2],
                dtype: DType::Int8,
            },
        );
        let weight = constant_placeholder(
            &mut graph,
            "qweight",
            TensorData {
                data: vec![3u8, 4u8],
                shape: vec![1, 2],
                dtype: DType::Int8,
            },
        );

        let mut meta = NodeMeta::default();
        let in_qp = QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string());
        let out_qp = QParam::per_tensor(0.25, 0, -128, 127, "torch.int8".to_string());
        meta.set_binary_input_output_qparams(in_qp.clone(), in_qp, out_qp);
        graph.add_node(
            NodeOp::CallFunction,
            "qlinear".to_string(),
            "aten.linear.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            meta,
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        // out = 0.5 * 1.5 + 1.0 * 2.0 = 2.75, quantized at scale 0.25 -> 11.
        assert_eq!(effect.node_name, "qlinear");
        assert_eq!(effect.shape, vec![1, 1]);
        assert_eq!(effect.dtype, DType::Int8);
        assert_eq!(effect.data, vec![11u8]);
    }

    #[test]
    fn skips_linear_with_runtime_input() {
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
                data: vec![3u8, 4u8],
                shape: vec![1, 2],
                dtype: DType::Int8,
            },
        );
        let bias = constant_placeholder(
            &mut graph,
            "bias",
            TensorData {
                data: vec![0u8],
                shape: vec![1],
                dtype: DType::Int8,
            },
        );
        graph.add_node(
            NodeOp::CallFunction,
            "runtime_linear".to_string(),
            "aten.linear.default".to_string(),
            vec![IrArg::Node(runtime), IrArg::Node(weight), IrArg::Node(bias)],
            NodeMeta::default(),
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }

    #[test]
    fn materializes_quantized_constant_conv() {
        let mut graph = IrGraph::empty();
        let input = constant_placeholder(
            &mut graph,
            "qinput",
            TensorData {
                data: vec![2u8, 4u8],
                shape: vec![1, 1, 1, 2],
                dtype: DType::Int8,
            },
        );
        let weight = constant_placeholder(
            &mut graph,
            "qweight",
            TensorData {
                data: vec![1u8, 2u8],
                shape: vec![1, 1, 1, 2],
                dtype: DType::Int8,
            },
        );

        let mut meta = NodeMeta::default();
        let in_qp = QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string());
        let weight_qp = QParam::per_tensor(0.25, 0, -128, 127, "torch.int8".to_string());
        let out_qp = QParam::per_tensor(0.25, 0, -128, 127, "torch.int8".to_string());
        meta.set_binary_input_output_qparams(in_qp, weight_qp, out_qp);
        graph.add_node(
            NodeOp::CallFunction,
            "qconv".to_string(),
            "aten.convolution.default".to_string(),
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::None,
                crate::fx_ir::i64_list_arg(&[1, 1]),
                crate::fx_ir::i64_list_arg(&[0, 0]),
                crate::fx_ir::i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                crate::fx_ir::i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            meta,
        );

        let result = IrFuseConstantInputsEffectPass::matmul_only().run(&mut graph);
        assert!(result.modified);
        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        // out = 1.0 * 0.25 + 2.0 * 0.5 = 1.25, quantized at scale 0.25 -> 5.
        assert_eq!(effect.node_name, "qconv");
        assert_eq!(effect.shape, vec![1, 1, 1, 1]);
        assert_eq!(effect.dtype, DType::Int8);
        assert_eq!(effect.data, vec![5u8]);
    }
}
