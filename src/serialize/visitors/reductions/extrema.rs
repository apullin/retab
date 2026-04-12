use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::Any, emit_aten_any as EmitFn),
        (OpTarget::Amax, emit_aten_amax),
        (OpTarget::Amin, emit_aten_amin),
        (OpTarget::TosaReduceMax, emit_aten_amax),
        (OpTarget::TosaReduceMin, emit_aten_amin),
        (OpTarget::TosaReduceAny, emit_aten_any),
    ]);
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

    let intermediate_prefix = match kind {
        ReduceExtremaKind::Any => "reduce_any_axis",
        ReduceExtremaKind::Max => "reduce_max_axis",
        ReduceExtremaKind::Min => "reduce_min_axis",
    };

    let make_attr: Box<dyn Fn(i32) -> Attribute> = match kind {
        ReduceExtremaKind::Any => {
            Box::new(|tosa_axis| Attribute::ReduceAny(ReduceAnyAttribute { axis: tosa_axis }))
        }
        ReduceExtremaKind::Min => Box::new(|tosa_axis| {
            Attribute::ReduceMin(ReduceMinAttribute {
                axis: tosa_axis,
                nan_mode: NanPropagationMode::Propagate,
            })
        }),
        ReduceExtremaKind::Max => Box::new(|tosa_axis| {
            Attribute::ReduceMax(ReduceMaxAttribute {
                axis: tosa_axis,
                nan_mode: NanPropagationMode::Propagate,
            })
        }),
    };

    let (current_input, _current_shape) = super::emit_multi_axis_reduce_loop(
        &mut result,
        &node.name,
        &spec.axes,
        spec.keepdim,
        spec.input_name,
        spec.input_shape.clone(),
        base_dtype,
        op,
        make_attr,
        intermediate_prefix,
        node.name.clone(),
        false, // final_on_last: only use node.name when keepdim
    );

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
