use super::*;

pub(super) fn register(
    table: &mut std::collections::HashMap<
        crate::graph::ir::OpTarget,
        crate::serialize::visitors::emitter::EmitFn,
    >,
) {
    use crate::graph::ir::OpTarget;
    table.insert(OpTarget::Any, emit_aten_any);
    table.insert(OpTarget::Amax, emit_aten_amax);
    table.insert(OpTarget::Amin, emit_aten_amin);
    table.insert(OpTarget::TosaReduceMax, emit_aten_amax);
    table.insert(OpTarget::TosaReduceMin, emit_aten_amin);
    table.insert(OpTarget::TosaReduceAny, emit_aten_any);
}

fn emit_reduce_extrema(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    op: Op,
    kind: ReduceExtremaKind,
) -> EmitResult {
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();
    let base_dtype = node_output_dtype(node);
    let mut current_input = spec.input_name;
    let mut current_shape = spec.input_shape.clone();

    for (idx, axis) in spec.axes.iter().enumerate() {
        let mut out_shape = current_shape.clone();
        if let Some(slot) = out_shape.get_mut(axis.tosa_axis as usize) {
            *slot = 1;
        }

        let is_last_reduce = idx == spec.axes.len() - 1 && spec.keepdim;
        let out_name = if is_last_reduce {
            node.name.clone()
        } else {
            let name = match kind {
                ReduceExtremaKind::Any => "reduce_any",
                ReduceExtremaKind::Max => "reduce_max",
                ReduceExtremaKind::Min => "reduce_min",
            };
            format!("{}_{}_axis{}", node.name, name, axis.raw_dim)
        };

        result.tensors.push(TosaTensor::new(
            out_name.clone(),
            out_shape.clone(),
            base_dtype,
        ));
        result.operators.push(TosaOperator::new(
            op,
            match kind {
                ReduceExtremaKind::Any => Attribute::ReduceAny(ReduceAnyAttribute {
                    axis: axis.tosa_axis,
                }),
                ReduceExtremaKind::Min => Attribute::ReduceMin(ReduceMinAttribute {
                    axis: axis.tosa_axis,
                    nan_mode: NanPropagationMode::Propagate,
                }),
                ReduceExtremaKind::Max => Attribute::ReduceMax(ReduceMaxAttribute {
                    axis: axis.tosa_axis,
                    nan_mode: NanPropagationMode::Propagate,
                }),
            },
            vec![current_input.clone()],
            vec![out_name.clone()],
        ));

        current_input = out_name;
        current_shape = out_shape;
    }

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            current_input,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            base_dtype,
        );
    }

    result
}

pub(in crate::serialize::visitors) fn emit_aten_amax(ctx: &EmitContext) -> EmitResult {
    emit_reduce_extrema(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::ReduceMax,
        ReduceExtremaKind::Max,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_any(ctx: &EmitContext) -> EmitResult {
    emit_reduce_extrema(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::ReduceAny,
        ReduceExtremaKind::Any,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_amin(ctx: &EmitContext) -> EmitResult {
    emit_reduce_extrema(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::ReduceMin,
        ReduceExtremaKind::Min,
    )
}
