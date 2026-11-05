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
        (OpTarget::ArgMax, emit_argmax),
    ]);
}

/// `aten.argmax` -> TOSA ARGMAX with INT32 indices (upstream ArgMaxVisitor, ExecuTorch
/// v1.6). ARGMAX drops the reduced axis, so its result comes out in the input's physical
/// order minus that axis; a TRANSPOSE restores logical order when the two differ, and a
/// RESHAPE re-inserts the axis for keepdim.
fn emit_argmax(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let input = first_input_node(graph, node)
        .unwrap_or_else(|| panic!("emit_argmax: missing input for node '{}'", node.name));
    let rank = input.meta.val.shape_len as i32;
    let raw_dim = match node.args.get(1) {
        Some(Arg::Int(dim)) => *dim as i32,
        _ => kwarg_i32(node, "dim", 0),
    };
    let dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
    let axis = map_axis_to_tosa_dim_order(graph, input, dim);
    let physical_order: Vec<i32> = match &input.meta.tosa_dim_order {
        Some(order) if order.len() == rank as usize => order.iter().map(|&d| d as i32).collect(),
        _ => (0..rank).collect(),
    };
    // Logical dims of the ARGMAX result, in the order they come out.
    let remaining: Vec<i32> = physical_order
        .iter()
        .copied()
        .filter(|&d| d != dim)
        .collect();
    let mut logical = remaining.clone();
    logical.sort_unstable();

    let mut physical_shape = actual_tensor_shape(graph, ctx.store, input);
    physical_shape.remove(axis as usize);
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
        axis,
        nan_mode: NanPropagationMode::Propagate,
    });
    let needs_transpose = remaining != logical;
    let mut result = EmitResult::default();
    let argmax_name = if needs_transpose || keepdim {
        format!("{}_argmax", node.name)
    } else {
        node.name.clone()
    };
    let argmax_shape = if needs_transpose {
        physical_shape
    } else {
        reduced_shape.clone()
    };
    result.push_output_op(
        // Not make_tensor: a rank-1 argmax yields a rank-0 result, which TOSA ARGMAX
        // requires and make_tensor would promote to [1].
        TosaTensor::new(&argmax_name, argmax_shape, DType::Int32),
        Op::ArgMax,
        attr,
        ctx.input_names(),
    );
    let mut current = argmax_name;
    if needs_transpose {
        let transposed = if keepdim {
            format!("{}_logical", node.name)
        } else {
            node.name.clone()
        };
        let perms = logical
            .iter()
            .map(|d| remaining.iter().position(|r| r == d).unwrap() as i32)
            .collect();
        result.push_output_op(
            make_tensor(transposed.clone(), reduced_shape, DType::Int32, vec![]),
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute { perms }),
            vec![current],
        );
        current = transposed;
    }
    if keepdim {
        push_reshape_with_shape(
            &mut result,
            current,
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
