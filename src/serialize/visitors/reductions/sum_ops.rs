use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaReduceSum, emit_reduce_sum as EmitFn),
        (OpTarget::Sum, emit_aten_sum),
    ]);
}

fn reduction_pre_reshape_perms(
    graph: &Graph,
    input_node: &Node,
    node: &Node,
    raw_axes: &[i32],
) -> Option<Vec<i32>> {
    let input_rank = input_node.meta.val.shape_len as usize;
    if input_rank == 0 || raw_axes.is_empty() {
        return None;
    }

    let input_order: Vec<usize> = {
        let mut order = vec![0usize; input_rank];
        let mut seen = vec![false; input_rank];
        let mut valid = true;
        for raw_axis in 0..input_rank {
            let pos = map_axis_to_tosa_dim_order(graph, input_node, raw_axis as i32) as usize;
            if pos >= input_rank || seen[pos] {
                valid = false;
                break;
            }
            order[pos] = raw_axis;
            seen[pos] = true;
        }
        if valid {
            order
        } else {
            input_node
                .meta
                .tosa_dim_order
                .clone()
                .map(|order| order.into_iter().map(|axis| axis as usize).collect())
                .unwrap_or_else(|| (0..input_rank).collect())
        }
    };
    let output_order: Vec<usize> = node
        .meta
        .tosa_dim_order
        .clone()
        .map(|order| order.into_iter().map(|axis| axis as usize).collect())
        .unwrap_or_else(|| (0..input_rank.saturating_sub(raw_axes.len())).collect());

    let reduced_axes: Vec<usize> = raw_axes
        .iter()
        .map(|&axis| {
            if axis < 0 {
                axis + input_rank as i32
            } else {
                axis
            }
        })
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
        || !perms
            .iter()
            .enumerate()
            .any(|(idx, &perm)| perm != idx as i32)
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
        TosaTensor::new(&reduce_output, reduce_shape.clone(), spec.reduce_dtype),
        Op::ReduceSum,
        Attribute::ReduceSum(ReduceSumAttribute {
            axis: spec.tosa_axis,
        }),
        spec.input_names,
    );

    if !spec.keepdim {
        let final_shape = ctx.shape();
        push_reshape_with_shape(
            &mut result,
            reduce_output,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            ctx.dtype(),
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
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!(
            "[SumDebug] node={} raw_shape={:?} actual_input_shape={:?} axes={:?} keepdim={} input_order={:?} node_order={:?} final_shape={:?}",
            node.name,
            raw_i32_shape(node),
            spec.input_shape,
            spec.axes
                .iter()
                .map(|axis| (axis.raw_dim, axis.tosa_axis))
                .collect::<Vec<_>>(),
            spec.keepdim,
            input_node.and_then(|n| n.meta.tosa_dim_order.clone()),
            node.meta.tosa_dim_order.clone(),
            ctx.shape(),
        );
    }
    let input_dtype = input_node
        .map(|n| passthrough_input_dtype(graph, n))
        .unwrap_or_else(|| ctx.dtype());
    let semantic_output_dtype = ctx.dtype();
    let mut base_dtype = match input_dtype {
        DType::Int8 | DType::Int16 => DType::Int32,
        _ => semantic_output_dtype,
    };
    let mut current_input = spec.input_name.clone();
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
            "sum",
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
        "reduce_axis",
        node.name.clone(),
        false, // final_on_last: only use node.name when keepdim
    );
    let current_input = reduce_output;
    let current_shape = reduce_shape;

    if !spec.keepdim {
        let mut reshape_input = current_input;
        if let Some(input_node) = input_node {
            let raw_axes: Vec<i32> = spec.axes.iter().map(|axis| axis.raw_dim).collect();
            if let Some(perms) = reduction_pre_reshape_perms(graph, input_node, node, &raw_axes) {
                let transpose_name = format!("{}_layout", node.name);
                let transpose_shape: Vec<i32> = perms
                    .iter()
                    .filter_map(|&axis| current_shape.get(axis as usize).copied())
                    .collect();
                if transpose_shape.len() == current_shape.len() {
                    result.push_output_op(
                        TosaTensor::new(&transpose_name, transpose_shape, base_dtype),
                        Op::Transpose,
                        Attribute::Transpose(TransposeAttribute { perms }),
                        vec![reshape_input],
                    );
                    reshape_input = transpose_name;
                }
            }
        }

        let final_shape = ctx.shape();
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
