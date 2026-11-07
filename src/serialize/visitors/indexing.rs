use super::emitter::EmitContext;
use super::*;
use crate::constant_materialization::quantize_scalar_with_qparam;
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
    dim: usize,
    index: i32,
    dtype: DType,
}

#[derive(Debug)]
struct ParsedSliceSpec {
    input_name: String,
    input_shape: Vec<i32>,
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
    start_shape: Vec<i32>,
    size_shape: Vec<i32>,
    input_target: String,
}

impl ParsedSelectCopySpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_name = input_name(graph, node, 0).unwrap_or_default();
        let input_node = node
            .args
            .first()
            .and_then(Arg::as_node)
            .and_then(|id| graph.node(id));
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let dim = (if raw_dim < 0 { raw_dim + rank } else { raw_dim }) as usize;

        let raw_index = arg_i64(node, 2, 0);
        let dim_size = input_shape.get(dim).copied().unwrap_or(1) as i64;
        let index = if raw_index < 0 {
            (dim_size + raw_index).max(0) as i32
        } else {
            raw_index.min(dim_size - 1) as i32
        };

        Self {
            input_name,
            input_shape,
            dim,
            index,
            dtype: node_output_dtype(node),
        }
    }
}

impl ParsedSliceSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_name = input_name(graph, node, 0).unwrap_or_default();
        let input_node = node
            .args
            .first()
            .and_then(Arg::as_node)
            .and_then(|id| graph.node(id));
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        if node.target == OpTarget::TosaSlice {
            return Self {
                input_name,
                input_shape,
                dim: 0,
                start: 0,
                end: 0,
                step: 1,
                start_shape: shape_arg_i32_list(graph, node, 1),
                size_shape: shape_arg_i32_list(graph, node, 2),
                input_target: input_node
                    .map(|n| format!("{:?}", n.target))
                    .unwrap_or_default(),
            };
        }

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let raw_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
        let dim = raw_dim.clamp(0, rank.saturating_sub(1)) as usize;
        let start = arg_i64(node, 2, 0);
        let end = arg_i64(node, 3, input_shape.get(dim).copied().unwrap_or(0) as i64);
        let step = arg_i64(node, 4, 1);
        let input_dim = input_shape.get(dim).copied().unwrap_or(0) as i64;
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
        start_shape[dim] = start_idx as i32;
        let mut size_shape = input_shape.clone();
        size_shape[dim] = slice_len as i32;

        let input_target = input_node
            .map(|n| format!("{:?}", n.target))
            .unwrap_or_else(|| "<none>".to_string());

        Self {
            input_name,
            input_shape,
            dim,
            start,
            end,
            step,
            start_shape,
            size_shape,
            input_target,
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

/// The scalar operand's value, and whether it is already quantized (an integer
/// constant, e.g. a scalar_tensor folded ahead of time with its output qparams).
fn select_scalar_const_value(store: &TensorStore, node: &Node) -> Option<(f64, bool)> {
    match node.target {
        OpTarget::ScalarTensor => node.args.first().map(|arg| (arg.as_f64(), false)),
        _ => store.get(&node.name).and_then(|tensor| {
            let quantized = matches!(
                tensor.dtype,
                crate::graph::ir::DType::Int8 | crate::graph::ir::DType::Int16
            );
            tensor.scalar_f64().map(|value| (value, quantized))
        }),
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
            let Some((raw_value, already_quantized)) = select_scalar_const_value(store, input_node)
            else {
                continue;
            };
            let fill_value = match (
                input_node.meta.val.dtype,
                input_node.meta.output_qparams.first(),
            ) {
                (crate::graph::ir::DType::Int8 | crate::graph::ir::DType::Int16, Some((_, qp)))
                    if !already_quantized =>
                {
                    quantize_scalar_with_qparam(raw_value, *qp)
                }
                _ => raw_value,
            };

            let const_name = format!("{}_input{}_broadcast", node.name, idx);
            if !push_filled_const_if_supported(
                result,
                const_name.clone(),
                output_shape.clone(),
                output_ir_dtype,
                fill_value,
            ) {
                continue;
            }
            input_names[idx] = const_name;
            continue;
        }

        let input_dtype = actual_tensor_dtype(graph, input_node);
        if input_dtype != output_dtype {
            if input_dtype == DType::Bool {
                let true_name = format!("{}_input{}_true", node.name, idx);
                push_filled_const_if_supported(
                    result,
                    true_name.clone(),
                    output_shape.clone(),
                    output_ir_dtype,
                    1.0,
                );

                let false_name = format!("{}_input{}_false", node.name, idx);
                push_filled_const_if_supported(
                    result,
                    false_name.clone(),
                    output_shape.clone(),
                    output_ir_dtype,
                    0.0,
                );

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
            result.push_output_op(
                temp_tensor(cast_name.clone(), output_shape.clone(), output_dtype),
                Op::Cast,
                Attribute::Cast(CastAttribute {}),
                vec![input_names[idx].clone()],
            );
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
    start_vec[spec.dim] = spec.index;
    let mut size_vec = spec.input_shape.clone();
    size_vec[spec.dim] = 1;

    push_const_shape(&mut result, format!("{}_start", slice_name), start_vec);
    push_const_shape(
        &mut result,
        format!("{}_size", slice_name),
        size_vec.clone(),
    );
    result.push_output_op(
        TosaTensor::new(&slice_name, size_vec, spec.dtype),
        Op::Slice,
        Attribute::Slice(SliceAttribute {}),
        vec![
            spec.input_name,
            format!("{}_start", slice_name),
            format!("{}_size", slice_name),
        ],
    );

    let output_shape = ctx.shape();
    push_reshape_with_shape(
        &mut result,
        slice_name,
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

    if crate::config::debug_slice_enabled() {
        eprintln!(
            "[SliceDebug] node={} input_target={} input_shape={:?} dim={} start={} end={} step={} start_shape={:?} size_shape={:?}",
            node.name,
            spec.input_target,
            spec.input_shape,
            spec.dim,
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
