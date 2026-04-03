use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaPad, emit_pad as EmitFn),
        (OpTarget::Pad, emit_pad),
    ]);
}

fn pad_fill_value(node: &Node) -> i32 {
    let kwarg_value = kwarg_i32(node, "value", i32::MIN);
    if kwarg_value != i32::MIN {
        return kwarg_value;
    }

    // aten.pad.default uses positional args:
    //   (input, pad, mode, value)
    // so the fill value often arrives as the first numeric arg after the
    // padding spec rather than as a kwarg.
    node.args
        .iter()
        .skip(2)
        .find_map(|arg| match arg {
            Arg::Int(value) => Some(*value as i32),
            Arg::Float(value) => Some(value.round() as i32),
            _ => None,
        })
        .unwrap_or(0)
}

pub(in crate::serialize::visitors) fn emit_pad(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let input_shape = node
        .args
        .first()
        .and_then(|arg| match arg {
            Arg::Node(id) => graph.node(*id),
            _ => None,
        })
        .map(|input_node| actual_tensor_shape(graph, store, input_node))
        .unwrap_or_else(|| raw_i32_shape(node));
    let pad = arg_i32_list(node, 1);
    let rank = input_shape.len();
    let pad_pairs: Vec<(i32, i32)> = pad
        .chunks(2)
        .filter_map(|pair| match pair {
            [before, after] => Some((*before, *after)),
            _ => None,
        })
        .collect();
    let mut per_dim_pad = vec![(0i32, 0i32); rank];
    for (i, &pair) in pad_pairs.iter().enumerate() {
        let dim = rank.saturating_sub(1 + i);
        if dim < rank {
            per_dim_pad[dim] = pair;
        }
    }

    let dim_order = node.meta.tosa_dim_order.as_ref();
    let reordered_pad = if let Some(order) = dim_order {
        if order.len() == rank {
            order
                .iter()
                .map(|&idx| per_dim_pad.get(idx as usize).copied().unwrap_or((0, 0)))
                .collect()
        } else {
            per_dim_pad.clone()
        }
    } else {
        per_dim_pad.clone()
    };

    let mut output_shape = input_shape.clone();
    for (i, (before, after)) in reordered_pad.iter().enumerate() {
        if let Some(slot) = output_shape.get_mut(i) {
            *slot += before + after;
        }
    }

    result.tensors.push(TosaTensor::new(
        node.name.clone(),
        output_shape,
        ir_dtype_to_tosa(node.meta.val.dtype),
    ));

    let mut pad_shape = Vec::with_capacity(rank * 2);
    for (before, after) in &reordered_pad {
        pad_shape.push(*before);
        pad_shape.push(*after);
    }
    push_const_shape(&mut result, format!("{}_pad_shape", node.name), pad_shape);

    let pad_const_name = format!("{}_pad_const", node.name);
    let raw_pad_value = pad_fill_value(node);
    let pad_dtype = ir_dtype_to_tosa(node.meta.val.dtype);
    let pad_value = match (pad_dtype, node.meta.output_qparams.first()) {
        (DType::Int8 | DType::Int16, Some((_, qp))) => {
            let quantized =
                ((raw_pad_value as f64 / qp.scale) + qp.zero_point as f64).round() as i32;
            quantized.clamp(qp.qmin as i32, qp.qmax as i32)
        }
        _ => raw_pad_value,
    };
    result.tensors.push(TosaTensor::with_data(
        pad_const_name.clone(),
        vec![1],
        pad_dtype,
        encode_zp_for_dtype(pad_value, pad_dtype),
    ));
    result.operators.push(TosaOperator::new(
        Op::Const,
        Attribute::Const(ConstAttribute),
        vec![],
        vec![pad_const_name.clone()],
    ));

    result.operators.push(TosaOperator::new(
        Op::Pad,
        Attribute::Pad(PadAttribute),
        vec![
            input_names.first().cloned().unwrap_or_default(),
            format!("{}_pad_shape", node.name),
            pad_const_name,
        ],
        vec![node.name.clone()],
    ));
    result
}
