use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaMaxPool2d, emit_max_pool2d as EmitFn),
        (OpTarget::TosaAvgPool2d, emit_tosa_avg_pool2d),
    ]);
}

/// rc4's `MAX_POOL2D(x, kernel, stride, pad)`, as IrRewriteMaxPool2d writes it: positional,
/// without kwargs, the pad already adjusted for ceil mode (P2 design X2).
pub(in crate::serialize::visitors) fn emit_max_pool2d(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let mut output_tensor = pool_output_tensor(node);
    output_tensor.name = primary_output_name(ctx.graph, node);
    result.push_output_op(
        output_tensor,
        Op::MaxPool2d,
        Attribute::MaxPool2d(MaxPool2dAttribute {
            kernel: arg_i32_list(node, 1),
            stride: arg_i32_list(node, 2),
            pad: arg_i32_list(node, 3),
            nan_mode: NanPropagationMode::Propagate,
        }),
        collect_input_names(ctx.graph, node),
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
