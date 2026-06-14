use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, TensorSource};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    CastInt64InputToInt32Effect, LoweringEffect, LoweringEffectBatch, LoweringInputKind,
};

pub struct IrCastInt64ToInt32EffectPass;

impl EffectfulIrPass for IrCastInt64ToInt32EffectPass {
    fn name(&self) -> &str {
        "IrCastInt64ToInt32Ep"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let candidate_ids: Vec<usize> = graph
            .live_placeholder_ids()
            .into_iter()
            .filter(|&id| graph.has_live_users(id))
            .filter(|&id| graph.node(id).meta.val_dtype() == Some("torch.int64"))
            .filter(|&id| {
                matches!(
                    graph.node(id).tensor.as_ref().map(|tensor| &tensor.source),
                    Some(TensorSource::ExportedProgramInput(binding))
                        if binding.input_kind == LoweringInputKind::Buffer
                )
            })
            .collect();

        let mut modified = false;
        let mut effects = LoweringEffectBatch::default();

        for node_id in candidate_ids {
            let node = graph.node(node_id).clone();
            let Some(tensor) = node.tensor.as_ref() else {
                continue;
            };
            let TensorSource::ExportedProgramInput(binding) = &tensor.source else {
                continue;
            };
            if binding.input_kind != LoweringInputKind::Buffer || tensor.data.dtype != DType::Int64
            {
                continue;
            }
            if node.meta.val_shape().is_none() {
                continue;
            }

            graph
                .node_mut(node_id)
                .meta
                .set_val_dtype_for_existing_shape("torch.int32");
            effects.push(LoweringEffect::CastInt64InputToInt32(
                CastInt64InputToInt32Effect {
                    placeholder_name: node.name.clone(),
                    input_kind: binding.input_kind,
                    target_name: Some(binding.target_name.clone()),
                    allow_missing: true,
                    update_placeholder_meta: false,
                },
            ));
            modified = true;
        }

        EffectfulIrPassResult { modified, effects }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{
        IrArg, IrGraph, MetaValue, NodeMeta, NodeOp, NodeTensor, PlaceholderTensorBinding,
    };
    use crate::store::TensorData;

    use super::*;

    #[test]
    fn effectful_cast_plans_buffer_cast_and_updates_meta() {
        let mut graph = IrGraph::empty();
        let placeholder = graph.add_node(
            NodeOp::Placeholder,
            "buf".to_string(),
            "buf".to_string(),
            Vec::<IrArg>::new(),
            NodeMeta::default(),
        );
        graph.node_mut(placeholder).meta = NodeMeta::tensor([2], "torch.int64");
        graph.node_mut(placeholder).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: LoweringInputKind::Buffer,
                target_name: "state.buf".to_string(),
            }),
            data: TensorData {
                data: [1i64.to_le_bytes(), 2i64.to_le_bytes()].concat(),
                shape: vec![2],
                dtype: DType::Int64,
            },
        });
        let user = graph.add_node(
            NodeOp::CallFunction,
            "use_buf".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(placeholder)],
            NodeMeta::default(),
        );
        graph
            .node_mut(user)
            .meta
            .set_extra("arm_quantized_annotation", MetaValue::Bool(true));

        let pass = IrCastInt64ToInt32EffectPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(
            graph.node(placeholder).meta.val_dtype(),
            Some("torch.int32")
        );
        assert_eq!(result.effects.items.len(), 1);
        let LoweringEffect::CastInt64InputToInt32(effect) = &result.effects.items[0] else {
            panic!("expected cast effect");
        };
        assert_eq!(effect.placeholder_name, "buf");
        assert_eq!(effect.input_kind, LoweringInputKind::Buffer);
        assert_eq!(effect.target_name.as_deref(), Some("state.buf"));
        assert!(!effect.update_placeholder_meta);
    }

    #[test]
    fn skips_non_buffer_or_non_int64_placeholders() {
        let mut graph = IrGraph::empty();
        let param = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::<IrArg>::new(),
            NodeMeta::default(),
        );
        graph.node_mut(param).meta = NodeMeta::tensor([1], "torch.int64");
        graph.node_mut(param).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: LoweringInputKind::Parameter,
                target_name: "state.weight".to_string(),
            }),
            data: TensorData {
                data: 1i64.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Int64,
            },
        });
        let user = graph.add_node(
            NodeOp::CallFunction,
            "use_weight".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(param)],
            NodeMeta::default(),
        );
        graph
            .node_mut(user)
            .meta
            .set_extra("arm_quantized_annotation", MetaValue::Bool(true));

        let pass = IrCastInt64ToInt32EffectPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
        assert_eq!(graph.node(param).meta.val_dtype(), Some("torch.int64"));
    }
}
