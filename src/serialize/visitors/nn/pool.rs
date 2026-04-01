use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaMaxPool2d, emit_max_pool2d as EmitFn),
        (OpTarget::TosaAvgPool2d, emit_avg_pool2d),
        (OpTarget::MaxPool2d, emit_aten_max_pool2d),
        (OpTarget::AvgPool2d, emit_avg_pool2d),
    ]);
}

// Parse pool args once at the function boundary. Downstream max-pool
// emitters then work with fully-normalised, typed fields instead of
// repeatedly fishing values out of raw node args.
#[derive(Debug)]
struct ParsedPoolSpec {
    input_names: Vec<String>,
    kernel: Vec<i32>,
    stride: Vec<i32>,
    /// Fully-adjusted 4-element pad (already accounts for ceil_mode).
    pad: Vec<i32>,
}

enum PoolArgSource {
    /// aten positional args (arg_i32_list / arg_bool)
    Positional,
    /// tosa kwargs (kwarg_i32_list / kwarg_bool)
    Keyword,
}

impl ParsedPoolSpec {
    fn parse(
        graph: &Graph,
        node: &Node,
        store: &TensorStore,
        source: PoolArgSource,
    ) -> Self {
        let input_names = collect_input_names(graph, node);

        let (raw_kernel, raw_stride, raw_padding, ceil_mode) = match source {
            PoolArgSource::Positional => (
                arg_i32_list(node, 1),
                arg_i32_list(node, 2),
                arg_i32_list(node, 3),
                arg_bool(node, 5, false),
            ),
            PoolArgSource::Keyword => (
                kwarg_i32_list(node, "kernel"),
                kwarg_i32_list(node, "stride"),
                kwarg_i32_list(node, "pad"),
                kwarg_bool(node, "ceil_mode", false),
            ),
        };

        let kernel = if raw_kernel.is_empty() {
            vec![2, 2]
        } else {
            raw_kernel
        };
        let stride = if raw_stride.is_empty() {
            kernel.clone()
        } else {
            raw_stride
        };
        let pad = adjusted_max_pool_pad(
            graph,
            store,
            node,
            &kernel,
            &stride,
            &raw_padding,
            ceil_mode,
        );

        Self {
            input_names,
            kernel,
            stride,
            pad,
        }
    }
}

fn adjusted_pool_post_pad(
    input_size: i32,
    kernel_size: i32,
    stride: i32,
    pad: i32,
    ceil_mode: bool,
) -> i32 {
    let numer = input_size - kernel_size + 2 * pad;
    let output_size = if ceil_mode {
        ((numer + stride - 1) / stride) + 1
    } else {
        (numer / stride) + 1
    };
    (output_size - 1) * stride - input_size + kernel_size - pad
}

fn adjusted_max_pool_pad(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    kernel: &[i32],
    stride: &[i32],
    padding: &[i32],
    ceil_mode: bool,
) -> Vec<i32> {
    if padding.len() == 4 {
        return padding.to_vec();
    }

    let symmetric_pad = if padding.len() >= 2 {
        vec![padding[0], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0]]
    } else {
        vec![0, 0]
    };

    let Some(Arg::Node(input_id)) = node.args.first() else {
        return vec![
            symmetric_pad[0],
            symmetric_pad[0],
            symmetric_pad[1],
            symmetric_pad[1],
        ];
    };
    let Some(input_node) = graph.node(*input_id) else {
        return vec![
            symmetric_pad[0],
            symmetric_pad[0],
            symmetric_pad[1],
            symmetric_pad[1],
        ];
    };
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.len() != 4 {
        return vec![
            symmetric_pad[0],
            symmetric_pad[0],
            symmetric_pad[1],
            symmetric_pad[1],
        ];
    }

    let post_pad_h = adjusted_pool_post_pad(
        input_shape[1],
        *kernel.first().unwrap_or(&2),
        *stride.first().unwrap_or(&2),
        symmetric_pad[0],
        ceil_mode,
    );
    let post_pad_w = adjusted_pool_post_pad(
        input_shape[2],
        *kernel.get(1).unwrap_or(&2),
        *stride.get(1).unwrap_or(&2),
        symmetric_pad[1],
        ceil_mode,
    );
    vec![symmetric_pad[0], post_pad_h, symmetric_pad[1], post_pad_w]
}

pub(in crate::serialize::visitors) fn emit_aten_max_pool2d(ctx: &EmitContext) -> EmitResult {
    let spec = ParsedPoolSpec::parse(ctx.graph, ctx.node, ctx.store, PoolArgSource::Positional);
    emit_max_pool2d_from_spec(ctx, spec)
}

pub(in crate::serialize::visitors) fn emit_max_pool2d(ctx: &EmitContext) -> EmitResult {
    let spec = ParsedPoolSpec::parse(ctx.graph, ctx.node, ctx.store, PoolArgSource::Keyword);
    emit_max_pool2d_from_spec(ctx, spec)
}

fn emit_max_pool2d_from_spec(ctx: &EmitContext, spec: ParsedPoolSpec) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(ctx.graph, ctx.node);
    let mut output_tensor = compute_pool_nhwc_output(
        ctx.graph,
        ctx.store,
        ctx.node,
        &spec.kernel,
        &spec.stride,
        &spec.pad,
    );
    output_tensor.name = output_name.clone();
    result.push_output_op(
        output_tensor,
        Op::MaxPool2d,
        Attribute::MaxPool2d(MaxPool2dAttribute {
            kernel: spec.kernel,
            stride: spec.stride,
            pad: spec.pad,
            nan_mode: NanPropagationMode::Propagate,
        }),
        spec.input_names,
    );
    result
}

pub(in crate::serialize::visitors) fn emit_avg_pool2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let kernel = {
        let kw = kwarg_i32_list(node, "kernel");
        if kw.is_empty() {
            arg_i32_list(node, 1)
        } else {
            kw
        }
    };
    let mut stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 2)
        } else {
            kw
        }
    };
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if !kw.is_empty() {
            kw
        } else {
            let padding = arg_i32_list(node, 3);
            if padding.len() >= 2 {
                vec![padding[0], padding[0], padding[1], padding[1]]
            } else if padding.len() == 1 {
                vec![padding[0], padding[0], padding[0], padding[0]]
            } else {
                vec![0, 0, 0, 0]
            }
        }
    };
    if stride.is_empty() {
        stride = kernel.clone();
    }

    result.tensors.push(compute_pool_nhwc_output(
        graph, store, node, &kernel, &stride, &pad,
    ));
    let acc_type = DType::Int32;
    let input_zp = node_input_zero_point(graph, node, 0, "input_zp", 0);
    let output_zp = node_output_zero_point(node, "output_zp", 0);
    let input_dtype = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|n| ir_dtype_to_tosa(n.meta.val.dtype))
            .unwrap_or(DType::Int8),
        _ => DType::Int8,
    };
    let output_dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    let input_zp_name = format!("{}_input_zp", node.name);
    push_inline_const(
        &mut result,
        TosaTensor::with_data(
            input_zp_name.clone(),
            vec![1],
            input_dtype,
            encode_zp_for_dtype(input_zp, input_dtype),
        ),
    );

    let output_zp_name = format!("{}_output_zp", node.name);
    push_inline_const(
        &mut result,
        TosaTensor::with_data(
            output_zp_name.clone(),
            vec![1],
            output_dtype,
            encode_zp_for_dtype(output_zp, output_dtype),
        ),
    );

    let mut pool_inputs = input_names;
    pool_inputs.push(input_zp_name);
    pool_inputs.push(output_zp_name);
    result.operators.push(TosaOperator::new(
        Op::AvgPool2d,
        Attribute::AvgPool2d(AvgPool2dAttribute {
            kernel,
            stride,
            pad,
            acc_type,
        }),
        pool_inputs,
        vec![node.name.clone()],
    ));
    result
}
