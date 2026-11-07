use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([(
        OpTarget::TosaTransposeConv2d,
        emit_transpose_conv2d as EmitFn,
    )]);
}

pub(in crate::serialize::visitors) fn emit_transpose_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let output_shape = raw_i32_shape(node);
    let acc_type = conv_accumulator_dtype(node);
    let output_dtype = synthetic_bias_dtype(node, acc_type);
    let output_tensor = make_empty_tensor(node.name.clone(), output_shape, output_dtype);

    // rc4's `TRANSPOSE_CONV2D(x, w, b, out_pad, stride)`, as IrRewriteConv writes it.
    let out_pad = kwarg_or_arg_i32_list(node, "out_pad", 3);
    let stride = kwarg_or_arg_i32_list(node, "stride", 4);
    let local_bound = kwarg_bool(node, "local_bound", false);

    let (input_zp, weight_zp) = input_weight_zero_points(node);
    let conv_inputs = prepare_conv_inputs(
        &mut result,
        graph,
        node,
        input_names,
        acc_type,
        input_zp,
        weight_zp,
    );

    result.push_output_op(
        output_tensor,
        Op::TransposeConv2d,
        Attribute::TransposeConv2d(TransposeConv2dAttribute {
            out_pad,
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            local_bound,
            acc_type,
        }),
        conv_inputs,
    );
    result
}
