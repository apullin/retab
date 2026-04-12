use super::emitter::EmitContext;
use super::*;
use crate::graph::ir::Arg;

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaSlice, emit_slice as EmitFn),
        (OpTarget::TosaGather, emit_gather),
        (OpTarget::TosaScatter, emit_scatter),
        (OpTarget::TosaSelect, emit_select),
        (OpTarget::Slice, emit_slice),
        (OpTarget::Select, emit_select),
        (OpTarget::SelectCopy, emit_select_copy),
        (OpTarget::IndexSelect, emit_index_select),
        (OpTarget::Where, emit_select),
    ]);
}

#[derive(Debug)]
struct ParsedSelectCopySpec {
    input_name: String,
    input_shape: Vec<i32>,
    mapped_dim: usize,
    index: i32,
    dtype: DType,
    pre_reshape_perms: Option<Vec<i32>>,
}

#[derive(Debug)]
struct ParsedSliceSpec {
    input_name: String,
    input_shape: Vec<i32>,
    raw_dim: i64,
    mapped_dim: usize,
    start: i64,
    end: i64,
    step: i64,
    start_shape: Vec<i32>,
    size_shape: Vec<i32>,
    input_target: String,
    input_order: String,
}

impl ParsedSelectCopySpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_name = input_names.first().cloned().unwrap_or_default();
        let input_node = match node.args.first() {
            Some(Arg::Node(id)) => graph.node(*id),
            _ => None,
        };
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let normalized_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
        let mapped_dim = input_node
            .map(|n| map_axis_to_tosa_dim_order(graph, n, normalized_dim as i32) as usize)
            .unwrap_or(normalized_dim as usize);

        let input_order: Vec<usize> = input_node
            .map(|n| {
                let rank = input_shape.len();
                let mut order = vec![0usize; rank];
                let mut seen = vec![false; rank];
                let mut valid = true;
                for raw_axis in 0..rank {
                    let pos = map_axis_to_tosa_dim_order(graph, n, raw_axis as i32) as usize;
                    if pos >= rank || seen[pos] {
                        valid = false;
                        break;
                    }
                    order[pos] = raw_axis;
                    seen[pos] = true;
                }
                if valid {
                    order
                } else {
                    n.meta
                        .tosa_dim_order
                        .clone()
                        .map(|order| order.into_iter().map(|axis| axis as usize).collect())
                        .unwrap_or_else(|| (0..rank).collect())
                }
            })
            .unwrap_or_else(|| (0..input_shape.len()).collect());
        let normalized_dim = normalized_dim as usize;
        let remaining_axes: Vec<usize> = (0..input_shape.len())
            .filter(|&axis| axis != normalized_dim)
            .collect();
        let output_rank = raw_i32_shape(node).len();
        let output_order: Vec<usize> = node
            .meta
            .tosa_dim_order
            .clone()
            .map(|order| order.into_iter().map(|axis| axis as usize).collect())
            .unwrap_or_else(|| (0..remaining_axes.len()).collect());
        // `select_copy` removes one logical axis and then collapses the
        // selected size-1 dimension with a reshape. For outputs below rank 4,
        // keep the surviving axes in raw logical order so the user-visible
        // tensor is not silently channels-last. For rank-4 outputs, keep using
        // the node's internal dim order so the serializer can still append its
        // normal output-layout transpose back to raw order.
        let mut desired_axes: Vec<usize> = if output_rank < 4 {
            remaining_axes.clone()
        } else {
            output_order
                .iter()
                .filter_map(|&output_axis| remaining_axes.get(output_axis).copied())
                .collect()
        };
        desired_axes.push(normalized_dim);
        let perms: Vec<i32> = desired_axes
            .iter()
            .filter_map(|axis| input_order.iter().position(|&candidate| candidate == *axis))
            .map(|pos| pos as i32)
            .collect();
        let pre_reshape_perms = if perms.len() == input_shape.len()
            && perms
                .iter()
                .enumerate()
                .any(|(idx, &perm)| perm != idx as i32)
        {
            Some(perms)
        } else {
            None
        };

        let raw_index = arg_i64(node, 2, 0);
        let dim_size = input_shape.get(mapped_dim).copied().unwrap_or(1) as i64;
        let index = if raw_index < 0 {
            (dim_size + raw_index).max(0) as i32
        } else {
            raw_index.min(dim_size - 1) as i32
        };

        Self {
            input_name,
            input_shape,
            mapped_dim,
            index,
            dtype: node_output_dtype(node),
            pre_reshape_perms,
        }
    }
}

impl ParsedSliceSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_name = input_names.first().cloned().unwrap_or_default();
        let input_node = match node.args.first() {
            Some(Arg::Node(id)) => graph.node(*id),
            _ => None,
        };
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let raw_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
        let raw_dim = raw_dim.clamp(0, rank.saturating_sub(1));
        let mapped_dim = node
            .meta
            .tosa_dim_order
            .as_ref()
            .map(|_| map_axis_to_tosa_dim_order(graph, node, raw_dim as i32) as i64)
            .or_else(|| {
                input_node.map(|n| map_axis_to_tosa_dim_order(graph, n, raw_dim as i32) as i64)
            })
            .unwrap_or(raw_dim) as usize;
        let start = arg_i64(node, 2, 0);
        let end = arg_i64(
            node,
            3,
            input_shape.get(mapped_dim).copied().unwrap_or(0) as i64,
        );
        let step = arg_i64(node, 4, 1);
        let input_dim = input_shape.get(mapped_dim).copied().unwrap_or(0) as i64;
        let start_idx = if start < 0 {
            (input_dim + start).max(0)
        } else {
            start.min(input_dim)
        };
        let mut end_idx = if end < 0 { input_dim + end } else { end };
        end_idx = end_idx.clamp(start_idx, input_dim);
        let slice_len = if step == 1 {
            (end_idx - start_idx).max(0)
        } else {
            ((end_idx - start_idx).max(0) + step - 1) / step
        };

        let mut start_shape = vec![0; input_shape.len()];
        start_shape[mapped_dim] = start_idx as i32;
        let mut size_shape = input_shape.clone();
        size_shape[mapped_dim] = slice_len as i32;

        let input_target = input_node
            .map(|n| format!("{:?}", n.target))
            .unwrap_or_else(|| "<none>".to_string());
        let input_order = input_node
            .and_then(|n| {
                n.meta
                    .tosa_dim_order
                    .as_ref()
                    .map(|order| format!("{:?}", order))
            })
            .unwrap_or_else(|| "<none>".to_string());

        Self {
            input_name,
            input_shape,
            raw_dim,
            mapped_dim,
            start,
            end,
            step,
            start_shape,
            size_shape,
            input_target,
            input_order,
        }
    }
}

pub(super) fn emit_select(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let mut input_names = ctx.input_names();
    maybe_normalize_select_inputs(graph, node, store, &mut result, &mut input_names);
    result.push_output_op(
        make_output_tensor(node),
        Op::Select,
        Attribute::Select(SelectAttribute {}),
        input_names,
    );
    result
}

fn select_scalar_const_value(store: &TensorStore, node: &Node) -> Option<f64> {
    match node.target {
        OpTarget::ScalarTensor => node.args.first().map(Arg::as_f64),
        _ => store.get(&node.name).and_then(|tensor| tensor.scalar_f64()),
    }
}

fn maybe_normalize_select_inputs(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    result: &mut EmitResult,
    input_names: &mut [String],
) {
    let output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() {
        return;
    }
    let output_dtype = node_output_dtype(node);
    let output_ir_dtype = node.meta.val.dtype;

    for (idx, arg) in node.args.iter().enumerate() {
        if idx == 0 {
            continue;
        }
        let Arg::Node(input_id) = arg else {
            continue;
        };
        let Some(input_node) = graph.node(*input_id) else {
            continue;
        };
        let input_shape = actual_tensor_shape(graph, store, input_node);
        if input_shape != output_shape {
            let Some(raw_value) = select_scalar_const_value(store, input_node) else {
                continue;
            };
            let fill_value = match (
                input_node.meta.val.dtype,
                input_node.meta.output_qparams.first(),
            ) {
                (crate::graph::ir::DType::Int8 | crate::graph::ir::DType::Int16, Some((_, qp))) => {
                    let quantized = ((raw_value / qp.scale) + qp.zero_point as f64).round() as i64;
                    quantized.clamp(qp.qmin, qp.qmax) as f64
                }
                _ => raw_value,
            };

            let const_name = format!("{}_input{}_broadcast", node.name, idx);
            let Some(tensor) = make_filled_const_tensor(
                const_name.clone(),
                output_shape.clone(),
                output_ir_dtype,
                fill_value,
            ) else {
                continue;
            };
            push_inline_const(result, tensor);
            input_names[idx] = const_name;
            continue;
        }

        let input_dtype = actual_tensor_dtype(graph, input_node);
        if input_dtype != output_dtype {
            if input_dtype == DType::Bool {
                let true_name = format!("{}_input{}_true", node.name, idx);
                if let Some(tensor) = make_filled_const_tensor(
                    true_name.clone(),
                    output_shape.clone(),
                    output_ir_dtype,
                    1.0,
                ) {
                    push_inline_const(result, tensor);
                }

                let false_name = format!("{}_input{}_false", node.name, idx);
                if let Some(tensor) = make_filled_const_tensor(
                    false_name.clone(),
                    output_shape.clone(),
                    output_ir_dtype,
                    0.0,
                ) {
                    push_inline_const(result, tensor);
                }

                let select_name = format!("{}_input{}_bool_cast", node.name, idx);
                result.push_output_op(
                    temp_tensor(select_name.clone(), output_shape.clone(), output_dtype),
                    Op::Select,
                    Attribute::Select(SelectAttribute {}),
                    vec![input_names[idx].clone(), true_name, false_name],
                );
                input_names[idx] = select_name;
                continue;
            }

            let cast_name = format!("{}_input{}_cast", node.name, idx);
            result.tensors.push(temp_tensor(
                cast_name.clone(),
                output_shape.clone(),
                output_dtype,
            ));
            result.operators.push(TosaOperator::new(
                Op::Cast,
                Attribute::Cast(CastAttribute {}),
                vec![input_names[idx].clone()],
                vec![cast_name.clone()],
            ));
            input_names[idx] = cast_name;
        }
    }
}

pub(super) fn emit_select_copy(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedSelectCopySpec::parse(graph, node, store);
    let mut result = EmitResult::default();

    let slice_name = format!("{}_slice", node.name);
    let mut start_vec = vec![0i32; spec.input_shape.len()];
    start_vec[spec.mapped_dim] = spec.index;
    let mut size_vec = spec.input_shape.clone();
    size_vec[spec.mapped_dim] = 1;

    push_const_shape(&mut result, format!("{}_start", slice_name), start_vec);
    push_const_shape(
        &mut result,
        format!("{}_size", slice_name),
        size_vec.clone(),
    );
    result.push_output_op(
        TosaTensor::new(&slice_name, size_vec.clone(), spec.dtype),
        Op::Slice,
        Attribute::Slice(SliceAttribute {}),
        vec![
            spec.input_name,
            format!("{}_start", slice_name),
            format!("{}_size", slice_name),
        ],
    );

    let mut reshape_input_name = slice_name;
    let reshape_input_shape = size_vec;
    if let Some(perms) = spec.pre_reshape_perms {
        let transpose_name = format!("{}_layout", node.name);
        let transpose_shape: Vec<i32> = perms
            .iter()
            .map(|&axis| reshape_input_shape[axis as usize])
            .collect();
        result.push_output_op(
            TosaTensor::new(&transpose_name, transpose_shape, spec.dtype),
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute { perms }),
            vec![reshape_input_name],
        );
        reshape_input_name = transpose_name;
    }

    let output_shape = ctx.shape();
    push_reshape_with_shape(
        &mut result,
        reshape_input_name,
        node.name.clone(),
        format!("{}_shape", node.name),
        output_shape,
        spec.dtype,
    );

    result
}

pub(super) fn emit_index_select(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();

    if input_names.len() < 2 {
        eprintln!(
            "[emit_index_select] WARNING: expected 2 node inputs, got {}",
            input_names.len()
        );
        return result;
    }

    result.push_output_op(
        make_output_tensor(node),
        Op::Gather,
        Attribute::Gather(GatherAttribute {}),
        input_names,
    );
    result
}

pub(super) fn emit_gather(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Gather,
        Attribute::Gather(GatherAttribute {}),
        ctx.input_names(),
    );
    result
}

pub(super) fn emit_scatter(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Scatter,
        Attribute::Scatter(ScatterAttribute {}),
        ctx.input_names(),
    );
    result
}

pub(super) fn emit_slice(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let spec = ParsedSliceSpec::parse(ctx.graph, node, ctx.store);
    let mut result = EmitResult::default();

    if std::env::var("ARM_TOSA_ENGINE_DEBUG_SLICE").ok().as_deref() == Some("1") {
        eprintln!(
            "[SliceDebug] node={} input_target={} input_order={} input_shape={:?} raw_dim={} mapped_dim={} start={} end={} step={} start_shape={:?} size_shape={:?}",
            node.name,
            spec.input_target,
            spec.input_order,
            spec.input_shape,
            spec.raw_dim,
            spec.mapped_dim,
            spec.start,
            spec.end,
            spec.step,
            spec.start_shape,
            spec.size_shape
        );
    }

    push_const_shape(
        &mut result,
        format!("{}_slice_start", node.name),
        spec.start_shape,
    );
    push_const_shape(
        &mut result,
        format!("{}_slice_size", node.name),
        spec.size_shape.clone(),
    );
    result.push_output_op(
        TosaTensor::new(
            &node.name,
            spec.size_shape,
            ir_dtype_to_tosa(node.meta.val.dtype),
        ),
        Op::Slice,
        Attribute::Slice(SliceAttribute {}),
        vec![
            spec.input_name,
            format!("{}_slice_start", node.name),
            format!("{}_slice_size", node.name),
        ],
    );

    result
}
