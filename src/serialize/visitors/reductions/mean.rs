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
    let current_shape = spec.input_shape.clone();

    if matches!(input_dtype, DType::Int8 | DType::Int16) && is_float_dtype(semantic_output_dtype) {
        let ir_output_dtype = node.meta.val.dtype;
        let input_qparam = input_node.and_then(single_qparam);
        let scale = input_qparam.map(|qp| qp.scale).unwrap_or(1.0);
        let zero_point = input_qparam.map(|qp| qp.zero_point as f64).unwrap_or(0.0);

        current_input = super::emit_quant_to_float_chain(
            &mut result,
            &node.name,
            current_input,
            &current_shape,
            ir_output_dtype,
            semantic_output_dtype,
            scale,
            zero_point,
            "mean",
        );

        base_dtype = semantic_output_dtype;
    }

    let (reduce_output, reduce_shape) = super::emit_multi_axis_reduce_loop(
        &mut result,
        &node.name,
        &spec.axes,
        spec.keepdim,
        current_input,
        current_shape,
        base_dtype,
        Op::ReduceSum,
        |tosa_axis| Attribute::ReduceSum(ReduceSumAttribute { axis: tosa_axis }),
        "reduce_sum_axis",
        sum_name.clone(),
        true, // final_on_last: always use sum_name on last iteration
    );
    let _current_input = reduce_output;
    let current_shape = reduce_shape;

    let recip_val = 1.0 / reduce_count;
    let recip_name = format!("{}_mean_recip", node.name);
    let ir_dtype = node.meta.val.dtype;
    if let Some(recip_tensor) = make_filled_const_tensor(
        recip_name.clone(),
        current_shape.clone(),
        ir_dtype,
        recip_val,
    ) {
        push_inline_const(&mut result, recip_tensor);
    }

    let final_name = if spec.keepdim {
        node.name.clone()
    } else {
        format!("{}_mean_pre_reshape", node.name)
    };
    let shift_name = format!("{}_mul_shift", node.name);
    result.push_const(TosaTensor::with_data(
        &shift_name,
        vec![1],
        DType::Int8,
        vec![0u8],
    ));
    result.push_output_op(
        TosaTensor::new(&final_name, current_shape.clone(), base_dtype),
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        vec![sum_name, recip_name, shift_name],
    );

    if !spec.keepdim {
        let final_shape = ctx.shape();
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
