use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaReduceSum, emit_reduce_sum as EmitFn),
        (OpTarget::Sum, emit_aten_sum),
    ]);
}

fn reduction_pre_reshape_perms(input_node: &Node, node: &Node, raw_axes: &[i32]) -> Option<Vec<i32>> {
    let input_rank = input_node.meta.val.shape_len as usize;
    if input_rank == 0 || raw_axes.is_empty() {
        return None;
    }

    let input_order: Vec<usize> = input_node
        .meta
        .tosa_dim_order
        .clone()
        .map(|order| order.into_iter().map(|axis| axis as usize).collect())
        .unwrap_or_else(|| (0..input_rank).collect());
    let output_order: Vec<usize> = node
        .meta
        .tosa_dim_order
        .clone()
        .map(|order| order.into_iter().map(|axis| axis as usize).collect())
        .unwrap_or_else(|| (0..input_rank.saturating_sub(raw_axes.len())).collect());

    let reduced_axes: Vec<usize> = raw_axes
        .iter()
        .map(|&axis| if axis < 0 { axis + input_rank as i32 } else { axis })
        .filter(|&axis| axis >= 0 && axis < input_rank as i32)
        .map(|axis| axis as usize)
        .collect();
    if reduced_axes.is_empty() {
        return None;
    }

    let remaining_axes: Vec<usize> = (0..input_rank)
        .filter(|axis| !reduced_axes.contains(axis))
        .collect();
    let mut desired_axes: Vec<usize> = output_order
        .iter()
        .filter_map(|&output_axis| remaining_axes.get(output_axis).copied())
        .collect();
    desired_axes.extend(reduced_axes.iter().copied());

    let perms: Vec<i32> = desired_axes
        .iter()
        .filter_map(|axis| input_order.iter().position(|&candidate| candidate == *axis))
        .map(|pos| pos as i32)
        .collect();
    if perms.len() != input_rank
        || !perms.iter().enumerate().any(|(idx, &perm)| perm != idx as i32)
    {
        return None;
    }

    Some(perms)
}

pub(in crate::serialize::visitors) fn emit_reduce_sum(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedSingleAxisReduceSpec::parse(graph, node, store);
    let mut result = EmitResult::default();

    let mut reduce_shape = spec.input_shape.clone();
    if let Some(slot) = reduce_shape.get_mut(spec.tosa_axis as usize) {
        *slot = 1;
    }

    let reduce_output = if spec.keepdim {
        node.name.clone()
    } else {
        format!("{}_reduce_axis{}", node.name, spec.raw_axis)
    };

    result.push_output_op(
        TosaTensor::new(reduce_output.clone(), reduce_shape.clone(), spec.reduce_dtype),
        Op::ReduceSum,
        Attribute::ReduceSum(ReduceSumAttribute {
            axis: spec.tosa_axis,
        }),
        spec.input_names,
    );

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            reduce_output,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            node_output_dtype(node),
        );
    }

    result
}

pub(in crate::serialize::visitors) fn emit_aten_sum(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();

    if spec.input_name.is_empty() {
        result.push_output_op(
            make_output_tensor(node),
            Op::ReduceSum,
            Attribute::ReduceSum(ReduceSumAttribute { axis: 0 }),
            vec![spec.input_name],
        );
        return result;
    }

    let input_node = match node.args.first() {
        Some(Arg::Node(id)) => graph.node(*id),
        _ => None,
    };
    let input_dtype = input_node
        .map(|n| passthrough_input_dtype(graph, n))
        .unwrap_or_else(|| node_output_dtype(node));
    let semantic_output_dtype = node_output_dtype(node);
    let mut base_dtype = match input_dtype {
        DType::Int8 | DType::Int16 => DType::Int32,
        _ => semantic_output_dtype,
    };
    let mut current_input = spec.input_name.clone();
    let mut current_shape = spec.input_shape.clone();

    if matches!(input_dtype, DType::Int8 | DType::Int16) && is_float_dtype(semantic_output_dtype) {
        let ir_output_dtype = node.meta.val.dtype;
        let input_qparam = input_node.and_then(single_qparam);
        let scale = input_qparam.map(|qp| qp.scale).unwrap_or(1.0);
        let zero_point = input_qparam.map(|qp| qp.zero_point as f64).unwrap_or(0.0);

        let cast_name = format!("{}_sum_input_cast", node.name);
        result.push_output_op(
            TosaTensor::new(cast_name.clone(), current_shape.clone(), semantic_output_dtype),
            Op::Cast,
            Attribute::Cast(CastAttribute),
            vec![current_input.clone()],
        );
        current_input = cast_name;

        if zero_point != 0.0 {
            let zp_name = format!("{}_sum_input_zp", node.name);
            if let Some(zp_tensor) = make_filled_const_tensor(
                zp_name.clone(),
                current_shape.clone(),
                ir_output_dtype,
                zero_point,
            ) {
                push_inline_const(&mut result, zp_tensor);
                let sub_name = format!("{}_sum_input_sub", node.name);
                result.push_output_op(
                    TosaTensor::new(sub_name.clone(), current_shape.clone(), semantic_output_dtype),
                    Op::Sub,
                    Attribute::Sub(SubAttribute),
                    vec![current_input, zp_name],
                );
                current_input = sub_name;
            }
        }

        if (scale - 1.0).abs() > f64::EPSILON {
            let scale_name = format!("{}_sum_input_scale", node.name);
            if let Some(scale_tensor) = make_filled_const_tensor(
                scale_name.clone(),
                current_shape.clone(),
                ir_output_dtype,
                scale,
            ) {
                push_inline_const(&mut result, scale_tensor);
                let shift_name = push_shift_const(&mut result, &format!("{}_sum_input", node.name));
                let mul_name = format!("{}_sum_input_float", node.name);
                result.push_output_op(
                    TosaTensor::new(mul_name.clone(), current_shape.clone(), semantic_output_dtype),
                    Op::Mul,
                    Attribute::Mul(MulAttribute { shift: 0 }),
                    vec![current_input, scale_name, shift_name],
                );
                current_input = mul_name;
            }
        }

        base_dtype = semantic_output_dtype;
    }

    for (idx, axis) in spec.axes.iter().enumerate() {
        let mut out_shape = current_shape.clone();
        if let Some(slot) = out_shape.get_mut(axis.tosa_axis as usize) {
            *slot = 1;
        }

        let is_last_reduce = idx == spec.axes.len() - 1 && spec.keepdim;
        let out_name = if is_last_reduce {
            node.name.clone()
        } else {
            format!("{}_reduce_axis{}", node.name, axis.raw_dim)
        };

        result.push_output_op(
            TosaTensor::new(out_name.clone(), out_shape.clone(), base_dtype),
            Op::ReduceSum,
            Attribute::ReduceSum(ReduceSumAttribute {
                axis: axis.tosa_axis,
            }),
            vec![current_input.clone()],
        );

        current_input = out_name;
        current_shape = out_shape;
    }

    if !spec.keepdim {
        let mut reshape_input = current_input;
        if let Some(input_node) = input_node {
            let raw_axes: Vec<i32> = spec.axes.iter().map(|axis| axis.raw_dim).collect();
            if let Some(perms) = reduction_pre_reshape_perms(input_node, node, &raw_axes) {
                let transpose_name = format!("{}_layout", node.name);
                let transpose_shape: Vec<i32> = perms
                    .iter()
                    .filter_map(|&axis| current_shape.get(axis as usize).copied())
                    .collect();
                if transpose_shape.len() == current_shape.len() {
                    result.push_output_op(
                        TosaTensor::new(transpose_name.clone(), transpose_shape, base_dtype),
                        Op::Transpose,
                        Attribute::Transpose(TransposeAttribute { perms }),
                        vec![reshape_input],
                    );
                    reshape_input = transpose_name;
                }
            }
        }

        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            reshape_input,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            base_dtype,
        );
    }

    result
}
