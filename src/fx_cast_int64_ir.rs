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

        // Bounded index sources (upstream ConvertInt64OutputOpsToInt32): TOSA ARGMAX
        // already yields INT32, so a following int32 dim-order copy becomes a no-op.
        for node_id in graph.live_call_function_ids() {
            let node = graph.node(node_id);
            if node.target == "aten.argmax.default"
                && node.meta.val_dtype() == Some("torch.int64")
            {
                graph
                    .node_mut(node_id)
                    .meta
                    .set_val_dtype_for_existing_shape("torch.int32");
                modified = true;
            }
        }

        EffectfulIrPassResult { modified, effects }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, MetaValue, NodeMeta};
    use crate::store::TensorData;
    use crate::test_ir::{buffer, call_default, parameter};

    use super::*;

    #[test]
    fn effectful_cast_plans_buffer_cast_and_updates_meta() {
        let mut graph = IrGraph::empty();
        let placeholder = buffer(
            &mut graph,
            "buf",
            TensorData {
                data: [1i64.to_le_bytes(), 2i64.to_le_bytes()].concat(),
                shape: vec![2],
                dtype: DType::Int64,
            },
        );
        graph.node_mut(placeholder).meta = NodeMeta::tensor([2], "torch.int64");
        let user = call_default(
            &mut graph,
            "use_buf",
            "aten.add.Tensor",
            vec![IrArg::Node(placeholder)],
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
        let param = parameter(
            &mut graph,
            "weight",
            TensorData {
                data: 1i64.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Int64,
            },
        );
        graph.node_mut(param).meta = NodeMeta::tensor([1], "torch.int64");
        let user = call_default(
            &mut graph,
            "use_weight",
            "aten.add.Tensor",
            vec![IrArg::Node(param)],
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
