use crate::fx_ep_effects::{
    FxCastInt64InputToInt32Effect, FxEpEffect, FxEpEffectBatch, FxEpInputKind,
};
use crate::fx_ir::{
    FakeTensorMeta, IrEpPass, IrEpPassResult, IrGraph, NodeOp, TensorSource,
};
use crate::graph::ir::DType;

pub struct IrCastInt64ToInt32EpPass;

impl IrEpPass for IrCastInt64ToInt32EpPass {
    fn name(&self) -> &str { "IrCastInt64ToInt32Ep" }

    fn run(&self, graph: &mut IrGraph) -> IrEpPassResult {
        let candidate_ids: Vec<usize> = graph
            .nodes
            .iter()
            .filter(|node| node.alive && node.op == NodeOp::Placeholder)
            .filter(|node| graph.has_live_users(node.id))
            .filter(|node| node.meta.val_dtype() == Some("torch.int64"))
            .filter(|node| {
                matches!(
                    node.tensor.as_ref().map(|tensor| &tensor.source),
                    Some(TensorSource::ExportedProgramInput(binding))
                        if binding.input_kind == FxEpInputKind::Buffer
                )
            })
            .map(|node| node.id)
            .collect();

        let mut modified = false;
        let mut effects = FxEpEffectBatch::default();

        for node_id in candidate_ids {
            let node = graph.node(node_id).clone();
            let Some(tensor) = node.tensor.as_ref() else {
                continue;
            };
            let TensorSource::ExportedProgramInput(binding) = &tensor.source else {
                continue;
            };
            if binding.input_kind != FxEpInputKind::Buffer || tensor.data.dtype != DType::Int64 {
                continue;
            }
            let Some(shape) = node.meta.val_shape() else {
                continue;
            };

            graph.node_mut(node_id).meta.set_val(FakeTensorMeta {
                shape: shape.to_vec(),
                dtype: "torch.int32".to_string(),
            });
            effects.push(FxEpEffect::CastInt64InputToInt32(
                FxCastInt64InputToInt32Effect {
                    placeholder_name: node.name.clone(),
                    input_kind: binding.input_kind,
                    target_name: Some(binding.target_name.clone()),
                    allow_missing: true,
                    update_placeholder_meta: false,
                },
            ));
            modified = true;
        }

        IrEpPassResult { modified, effects }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{
        IrArg, IrGraph, MetaValue, NodeMeta, NodeTensor, PlaceholderTensorBinding,
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
        graph.node_mut(placeholder).meta.set_val(FakeTensorMeta {
            shape: vec![2],
            dtype: "torch.int64".to_string(),
        });
        graph.node_mut(placeholder).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: FxEpInputKind::Buffer,
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
        graph.node_mut(user)
            .meta
            .set_extra("arm_quantized_annotation", MetaValue::Bool(true));

        let pass = IrCastInt64ToInt32EpPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(graph.node(placeholder).meta.val_dtype(), Some("torch.int32"));
        assert_eq!(result.effects.effects.len(), 1);
        let FxEpEffect::CastInt64InputToInt32(effect) = &result.effects.effects[0] else {
            panic!("expected cast effect");
        };
        assert_eq!(effect.placeholder_name, "buf");
        assert_eq!(effect.input_kind, FxEpInputKind::Buffer);
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
        graph.node_mut(param).meta.set_val(FakeTensorMeta {
            shape: vec![1],
            dtype: "torch.int64".to_string(),
        });
        graph.node_mut(param).tensor = Some(NodeTensor {
            source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: FxEpInputKind::Parameter,
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
        graph.node_mut(user)
            .meta
            .set_extra("arm_quantized_annotation", MetaValue::Bool(true));

        let pass = IrCastInt64ToInt32EpPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.effects.is_empty());
        assert_eq!(graph.node(param).meta.val_dtype(), Some("torch.int64"));
    }
}
