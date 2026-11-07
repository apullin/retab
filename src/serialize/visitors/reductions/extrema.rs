use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::Any, emit_aten_any as EmitFn),
        (OpTarget::Amax, emit_aten_amax),
        (OpTarget::Amin, emit_aten_amin),
        (OpTarget::TosaReduceMax, emit_aten_amax),
        (OpTarget::TosaReduceMin, emit_aten_amin),
        (OpTarget::ArgMax, emit_argmax),
    ]);
}

/// `aten.argmax` -> TOSA ARGMAX with INT32 indices (upstream ArgMaxVisitor, ExecuTorch
/// v1.6). ARGMAX drops the reduced axis, so a RESHAPE re-inserts it for keepdim.
fn emit_argmax(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let input = first_input_node(graph, node)
        .unwrap_or_else(|| panic!("emit_argmax: missing input for node '{}'", node.name));
    let rank = input.meta.val.shape_len as i32;
    let raw_dim = node
        .args
        .get(1)
        .and_then(Arg::as_i64)
        .or_else(|| {
            node.kwargs
                .iter()
                .find(|(key, _)| key == "dim")
                .and_then(|(_, value)| value.as_i64())
        })
        .expect("argmax requires an explicit dim") as i32;
    let dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
    let output_shape: Vec<i32> = node.meta.val.shape[..node.meta.val.shape_len as usize]
        .iter()
        .map(|&d| d as i32)
        .collect();
    let keepdim = output_shape.len() == rank as usize;
    let reduced_shape: Vec<i32> = if keepdim {
        output_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as i32 != dim)
            .map(|(_, &d)| d)
            .collect()
    } else {
        output_shape.clone()
    };

    let attr = Attribute::ArgMax(ArgMaxAttribute {
        axis: dim,
        nan_mode: NanPropagationMode::Propagate,
    });
    let mut result = EmitResult::default();
    let argmax_name = if keepdim {
        format!("{}_argmax", node.name)
    } else {
        node.name.clone()
    };
    result.push_output_op(
        TosaTensor::new(&argmax_name, reduced_shape, DType::Int32),
        Op::ArgMax,
        attr,
        ctx.input_names(),
    );
    if keepdim {
        push_reshape_with_shape(
            &mut result,
            argmax_name,
            node.name.clone(),
            format!("{}_keepdim_shape", node.name),
            output_shape,
            DType::Int32,
        );
    }
    result
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
            Box::new(|axis| Attribute::ReduceAny(ReduceAnyAttribute { axis }))
        }
        ReduceExtremaKind::Min => Box::new(|axis| {
            Attribute::ReduceMin(ReduceMinAttribute {
                axis,
                nan_mode: NanPropagationMode::Propagate,
            })
        }),
        ReduceExtremaKind::Max => Box::new(|axis| {
            Attribute::ReduceMax(ReduceMaxAttribute {
                axis,
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
