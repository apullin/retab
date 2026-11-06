use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaMaxPool2d, emit_max_pool2d as EmitFn),
        (OpTarget::TosaAvgPool2d, emit_tosa_avg_pool2d),
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
    /// rc4's `MAX_POOL2D(x, kernel, stride, pad)`: positional, pad already adjusted.
    TosaPositional,
}

impl ParsedPoolSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore, source: PoolArgSource) -> Self {
        let input_names = collect_input_names(graph, node);

        let (raw_kernel, raw_stride, raw_padding, ceil_mode) = match source {
            PoolArgSource::Positional => (
                arg_i32_list(node, 1),
                arg_i32_list(node, 2),
                arg_i32_list(node, 3),
                arg_bool(node, 5, false),
            ),
            // A 4-element pad passes through `adjusted_max_pool_pad` unchanged.
            PoolArgSource::TosaPositional => (
                arg_i32_list(node, 1),
                arg_i32_list(node, 2),
                arg_i32_list(node, 3),
                false,
            ),
        };

        let kernel = i32_list_or(raw_kernel, vec![2, 2]);
        let stride = i32_list_or(raw_stride, kernel.clone());
        let input_shape = first_input_node(graph, node)
            .map(|input_node| actual_tensor_shape(graph, store, input_node))
            .unwrap_or_default();
        let pad = adjusted_max_pool_pad(&input_shape, &kernel, &stride, &raw_padding, ceil_mode);

        Self {
            input_names,
            kernel,
            stride,
            pad,
        }
    }
}

pub(in crate::serialize::visitors) fn emit_aten_max_pool2d(ctx: &EmitContext) -> EmitResult {
    let spec = ParsedPoolSpec::parse(ctx.graph, ctx.node, ctx.store, PoolArgSource::Positional);
    emit_max_pool2d_from_spec(ctx, spec)
}

pub(in crate::serialize::visitors) fn emit_max_pool2d(ctx: &EmitContext) -> EmitResult {
    // MAX_POOL2D nodes come from IrRewriteMaxPool2d in rc4's positional form, without
    // kwargs (P2 design X2).
    let spec = ParsedPoolSpec::parse(
        ctx.graph,
        ctx.node,
        ctx.store,
        PoolArgSource::TosaPositional,
    );
    emit_max_pool2d_from_spec(ctx, spec)
}

fn emit_max_pool2d_from_spec(ctx: &EmitContext, spec: ParsedPoolSpec) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(ctx.graph, ctx.node);
    let mut output_tensor = pool_output_tensor(ctx.node);
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

/// rc4's `AVG_POOL2D(x, input_zp, output_zp, kernel, stride, pad, acc_type)` (P2 design
/// X3). The zero points are graph constants that the pipeline materializes, already
/// quantized the way rc4's ComputeConstantOpsAOTPass does, so the operator reads them as
/// its 2nd and 3rd inputs and nothing is recomputed here.
pub(in crate::serialize::visitors) fn emit_tosa_avg_pool2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let kernel = arg_i32_list(node, 3);
    let stride = arg_i32_list(node, 4);
    let pad = arg_i32_list(node, 5);
    let acc_type = match node.args.get(6) {
        Some(Arg::Dtype(dtype)) => ir_dtype_to_tosa(*dtype),
        _ => match first_input_node(graph, node).map(|input| input.meta.val.dtype) {
            Some(dtype) if dtype.is_float() => DType::Fp32,
            _ => DType::Int32,
        },
    };
    let output_tensor = pool_output_tensor(node);
    let mut result = EmitResult::default();
    result.push_output_op(
        output_tensor,
        Op::AvgPool2d,
        Attribute::AvgPool2d(AvgPool2dAttribute {
            kernel,
            stride,
            pad,
            acc_type,
        }),
        ctx.input_names(),
    );
    result
}

pub(in crate::serialize::visitors) fn emit_avg_pool2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let kernel = kwarg_or_arg_i32_list(node, "kernel", 1);
    let stride = kwarg_or_arg_i32_list_or(node, "stride", 2, kernel.clone());
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if !kw.is_empty() {
            kw
        } else {
            let input_shape = first_input_node(graph, node)
                .map(|input_node| actual_tensor_shape(graph, store, input_node))
                .unwrap_or_default();
            let ceil_mode = arg_bool(node, 4, false);
            adjusted_max_pool_pad(
                &input_shape,
                &kernel,
                &stride,
                &arg_i32_list(node, 3),
                ceil_mode,
            )
        }
    };
    let output_tensor = pool_output_tensor(node);
    // rc4's RewriteAvgPool2dPass materializes both zero points with
    // `call_scalar(zp, meta)`. The `full` node inherits the avg-pool meta, output_qparams
    // included, so ComputeConstantOpsAOTPass quantizes the zero point as if it were a real
    // value (a negative zp usually saturates to qmin). Both operands get the same
    // treatment and AVG_POOL2D is linear in them, so only the rounding changes; mirror it
    // for numeric parity with stock.
    let input_zp =
        quantize_fill_value(node, node_input_zero_point(graph, node, 0, "input_zp", 0) as f64)
            as i32;
    let output_zp =
        quantize_fill_value(node, node_output_zero_point(node, "output_zp", 0) as f64) as i32;
    let input_dtype = first_input_node(graph, node)
        .map(|input_node| ir_dtype_to_tosa(input_node.meta.val.dtype))
        .unwrap_or(DType::Int8);
    let output_dtype = ir_dtype_to_tosa(node.meta.val.dtype);
    let acc_type = if is_float_dtype(input_dtype) {
        DType::Fp32
    } else {
        DType::Int32
    };

    let input_zp_name = format!("{}_input_zp", node.name);
    push_zero_point_const(&mut result, input_zp_name.clone(), input_dtype, input_zp);

    let output_zp_name = format!("{}_output_zp", node.name);
    push_zero_point_const(&mut result, output_zp_name.clone(), output_dtype, output_zp);

    let mut pool_inputs = input_names;
    pool_inputs.push(input_zp_name);
    pool_inputs.push(output_zp_name);
    result.push_output_op(
        output_tensor,
        Op::AvgPool2d,
        Attribute::AvgPool2d(AvgPool2dAttribute {
            kernel,
            stride,
            pad,
            acc_type,
        }),
        pool_inputs,
    );
    result
}
