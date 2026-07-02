use crate::fx_ir::{
    EffectfulIrPass, EffectfulIrPassResult, IrArg, IrGraph, IrNode, NodeOp, TensorSource,
};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    CastInt64InputToInt32Effect, LoweringEffect, LoweringEffectBatch, LoweringInputKind,
};
use crate::store::TensorData;
use crate::tensor_data::{encode_bf16, encode_f16};

pub struct IrCastInt64ToInt32EffectPass;

impl EffectfulIrPass for IrCastInt64ToInt32EffectPass {
    fn name(&self) -> &str {
        "IrCastInt64ToInt32Ep"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let candidate_ids: Vec<usize> = graph
            .live_nodes()
            .filter(|node| graph.has_live_users(node.id))
            .filter(|node| node.meta.val_dtype() == Some("torch.int64"))
            .filter(|node| {
                node.tensor
                    .as_ref()
                    .is_some_and(|tensor| tensor.data.dtype == DType::Int64)
            })
            .map(|node| node.id)
            .collect();

        let mut modified = false;
        let mut effects = LoweringEffectBatch::default();

        for node_id in candidate_ids {
            let node = graph.node(node_id).clone();
            let Some(tensor) = node.tensor.as_ref() else {
                continue;
            };
            if !tensor.data.int64_values_fit_i32() {
                if let Some(float_dtype) = all_users_float_cast_dtype(graph, node_id) {
                    let Some(float_data) = cast_int64_tensor_to_float(&tensor.data, float_dtype)
                    else {
                        continue;
                    };
                    let node = graph.node_mut(node_id);
                    if let Some(tensor) = node.tensor.as_mut() {
                        tensor.data = float_data;
                    }
                    node.meta
                        .set_val_dtype_for_existing_shape(float_dtype.torch_str());
                    modified = true;
                }
                continue;
            }
            if node.meta.val_shape().is_none() {
                continue;
            }

            let exported_buffer_binding = match &tensor.source {
                TensorSource::ExportedProgramInput(binding)
                    if binding.input_kind == LoweringInputKind::Buffer =>
                {
                    Some(binding.clone())
                }
                _ => None,
            };
            if let Some(binding) = exported_buffer_binding {
                graph
                    .node_mut(node_id)
                    .meta
                    .set_val_dtype_for_existing_shape("torch.int32");
                effects.push(LoweringEffect::CastInt64InputToInt32(
                    CastInt64InputToInt32Effect {
                        placeholder_name: node.name.clone(),
                        input_kind: binding.input_kind,
                        target_name: Some(binding.target_name),
                        allow_missing: true,
                        update_placeholder_meta: false,
                    },
                ));
            } else if let Some(int32_data) = cast_int64_tensor_to_int32(&tensor.data) {
                let node = graph.node_mut(node_id);
                if let Some(tensor) = node.tensor.as_mut() {
                    tensor.data = int32_data;
                }
                node.meta.set_val_dtype_for_existing_shape("torch.int32");
            } else {
                continue;
            }
            modified = true;
        }

        EffectfulIrPassResult { modified, effects }
    }
}

fn all_users_float_cast_dtype(graph: &IrGraph, node_id: usize) -> Option<DType> {
    let mut dtype = None;
    let mut saw_user = false;
    for &user_id in graph.users(node_id) {
        let user = graph.node(user_id);
        if !user.alive {
            continue;
        }
        saw_user = true;
        let user_dtype = float_cast_dtype(user)?;
        if dtype.is_some_and(|existing| existing != user_dtype) {
            return None;
        }
        dtype = Some(user_dtype);
    }
    saw_user.then_some(dtype?)
}

fn float_cast_dtype(node: &IrNode) -> Option<DType> {
    if node.op != NodeOp::CallFunction
        || !(node.target.contains("_to_copy.default")
            || node.target.contains("_to_dim_order_copy.default"))
    {
        return None;
    }
    let dtype = node
        .kwargs
        .get("dtype")
        .or_else(|| node.args.get(1))
        .and_then(dtype_from_arg)?;
    dtype.is_float().then_some(dtype)
}

fn dtype_from_arg(arg: &IrArg) -> Option<DType> {
    match arg {
        IrArg::Dtype(dtype) | IrArg::Str(dtype) => Some(DType::from_torch_str(dtype)),
        _ => None,
    }
}

fn cast_int64_tensor_to_int32(tensor: &TensorData) -> Option<TensorData> {
    if tensor.dtype != DType::Int64 || !tensor.int64_values_fit_i32() {
        return None;
    }
    let mut data = Vec::with_capacity(tensor.element_count() * DType::Int32.storage_size_bytes());
    for chunk in tensor.data.chunks_exact(8) {
        let value = i64::from_le_bytes(chunk.try_into().ok()?);
        data.extend_from_slice(&i32::try_from(value).ok()?.to_le_bytes());
    }
    Some(TensorData {
        data,
        shape: tensor.shape.clone(),
        dtype: DType::Int32,
    })
}

fn cast_int64_tensor_to_float(tensor: &TensorData, dtype: DType) -> Option<TensorData> {
    if tensor.dtype != DType::Int64 || tensor.data.len() % 8 != 0 {
        return None;
    }
    let mut data = Vec::with_capacity(tensor.element_count() * dtype.storage_size_bytes());
    for chunk in tensor.data.chunks_exact(8) {
        let value = i64::from_le_bytes(chunk.try_into().ok()?);
        match dtype {
            DType::Float32 => data.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::Float16 => data.extend_from_slice(&encode_f16(value as f32).to_le_bytes()),
            DType::BFloat16 => data.extend_from_slice(&encode_bf16(value as f32).to_le_bytes()),
            _ => return None,
        }
    }
    Some(TensorData {
        data,
        shape: tensor.shape.clone(),
        dtype,
    })
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{dtype_kwarg, IrArg, IrGraph, MetaValue, NodeMeta};
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
    fn narrows_non_buffer_constant_without_lowering_effect() {
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

        assert!(result.modified);
        assert!(result.effects.items.is_empty());
        assert_eq!(graph.node(param).meta.val_dtype(), Some("torch.int32"));
        assert_eq!(
            graph.node(param).tensor.as_ref().unwrap().data.dtype,
            DType::Int32
        );
    }

    #[test]
    fn skips_int64_buffers_outside_int32_range() {
        let mut graph = IrGraph::empty();
        let placeholder = buffer(
            &mut graph,
            "buf",
            TensorData {
                data: (i64::from(i32::MAX) + 1).to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Int64,
            },
        );
        graph.node_mut(placeholder).meta = NodeMeta::tensor([1], "torch.int64");
        call_default(
            &mut graph,
            "use_buf",
            "aten._to_copy.default",
            vec![IrArg::Node(placeholder)],
        );

        let pass = IrCastInt64ToInt32EffectPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
        assert_eq!(
            graph.node(placeholder).meta.val_dtype(),
            Some("torch.int64")
        );
    }

    #[test]
    fn materializes_out_of_range_int64_buffer_as_float_when_all_users_cast_to_float() {
        let mut graph = IrGraph::empty();
        let placeholder = buffer(
            &mut graph,
            "buf",
            TensorData {
                data: [i64::from(i32::MAX) + 1, i64::from(i32::MAX) + 2]
                    .map(i64::to_le_bytes)
                    .concat(),
                shape: vec![2],
                dtype: DType::Int64,
            },
        );
        graph.node_mut(placeholder).meta = NodeMeta::tensor([2], "torch.int64");
        graph.add_node_with_kwargs(
            NodeOp::CallFunction,
            "buf_to_f32".to_string(),
            "aten._to_copy.default".to_string(),
            vec![IrArg::Node(placeholder)],
            dtype_kwarg("torch.float32"),
            NodeMeta::tensor([2], "torch.float32"),
        );

        let pass = IrCastInt64ToInt32EffectPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert!(result.effects.items.is_empty());
        let node = graph.node(placeholder);
        assert_eq!(node.meta.val_dtype(), Some("torch.float32"));
        let data = &node.tensor.as_ref().unwrap().data;
        assert_eq!(data.dtype, DType::Float32);
        assert_eq!(
            data.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                (i64::from(i32::MAX) + 1) as f32,
                (i64::from(i32::MAX) + 2) as f32
            ]
        );
    }
}
