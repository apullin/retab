use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([(OpTarget::Mean, emit_aten_mean as EmitFn)]);
}

pub(in crate::serialize::visitors) fn emit_aten_mean(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();

    let mut reduce_count: f64 = 1.0;
    for axis in &spec.axes {
        if let Some(&dim_size) = spec.input_shape.get(axis.tosa_axis as usize) {
            reduce_count *= dim_size as f64;
        }
    }

    let input_node = match node.args.first() {
        Some(Arg::Node(id)) => graph.node(*id),
        _ => None,
    };
    let input_dtype = input_node
        .map(|n| passthrough_input_dtype(graph, n))
        .unwrap_or_else(|| node_output_dtype(node));
    let semantic_output_dtype = node_output_dtype(node);
    let mut base_dtype = semantic_output_dtype;
    let sum_name = format!("{}_sum", node.name);
    let mut current_input = spec.input_name;
    let mut current_shape = spec.input_shape.clone();

    if matches!(input_dtype, DType::Int8 | DType::Int16) && is_float_dtype(semantic_output_dtype) {
        let ir_output_dtype = node.meta.val.dtype;
        let input_qparam = input_node.and_then(single_qparam);
        let scale = input_qparam.map(|qp| qp.scale).unwrap_or(1.0);
        let zero_point = input_qparam.map(|qp| qp.zero_point as f64).unwrap_or(0.0);

        let cast_name = format!("{}_mean_input_cast", node.name);
        result.push_output_op(
            TosaTensor::new(cast_name.clone(), current_shape.clone(), semantic_output_dtype),
            Op::Cast,
            Attribute::Cast(CastAttribute),
            vec![current_input.clone()],
        );
        current_input = cast_name;

        if zero_point != 0.0 {
            let zp_name = format!("{}_mean_input_zp", node.name);
            if let Some(zp_tensor) = make_filled_const_tensor(
                zp_name.clone(),
                current_shape.clone(),
                ir_output_dtype,
                zero_point,
            ) {
                push_inline_const(&mut result, zp_tensor);
                let sub_name = format!("{}_mean_input_sub", node.name);
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
            let scale_name = format!("{}_mean_input_scale", node.name);
            if let Some(scale_tensor) = make_filled_const_tensor(
                scale_name.clone(),
                current_shape.clone(),
                ir_output_dtype,
                scale,
            ) {
                push_inline_const(&mut result, scale_tensor);
                let shift_name =
                    push_shift_const(&mut result, &format!("{}_mean_input", node.name));
                let mul_name = format!("{}_mean_input_float", node.name);
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

        let is_last = idx == spec.axes.len() - 1;
        let out_name = if is_last {
            sum_name.clone()
        } else {
            format!("{}_reduce_sum_axis{}", node.name, axis.raw_dim)
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

    let recip_val = 1.0 / reduce_count;
    let recip_name = format!("{}_mean_recip", node.name);
    let ir_dtype = node.meta.val.dtype;
    if let Some(recip_tensor) =
        make_filled_const_tensor(recip_name.clone(), vec![1], ir_dtype, recip_val)
    {
        push_inline_const(&mut result, recip_tensor);
    }

    let final_name = if spec.keepdim {
        node.name.clone()
    } else {
        format!("{}_mean_pre_reshape", node.name)
    };
    let shift_name = format!("{}_mul_shift", node.name);
    result.push_const(TosaTensor::with_data(
        shift_name.clone(),
        vec![1],
        DType::Int8,
        vec![0u8],
    ));
    result.push_output_op(
        TosaTensor::new(final_name.clone(), current_shape.clone(), base_dtype),
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        vec![sum_name, recip_name, shift_name],
    );

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            final_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            base_dtype,
        );
    }

    result
}
