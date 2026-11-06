use super::*;
use crate::graph::ir::OpTarget;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (
            OpTarget::TosaTransposeConv2d,
            emit_transpose_conv2d as EmitFn,
        ),
        (OpTarget::ConvTranspose2d, emit_transpose_conv2d),
    ]);
}

fn expand_pair(values: Vec<i32>) -> Vec<i32> {
    match values.as_slice() {
        [] => vec![0, 0],
        [value] => vec![*value, *value],
        [first, second, ..] => vec![*first, *second],
    }
}

fn raw_transpose_conv_output_padding(node: &Node) -> Vec<i32> {
    let arg_idx = if node.args.len() > 7 { 7 } else { 5 };
    arg_or_kwarg_i32_list(node, arg_idx, "output_padding")
}

fn transpose_conv_out_pad(node: &Node) -> Vec<i32> {
    match &node.target {
        OpTarget::TosaTransposeConv2d => kwarg_or_arg_i32_list(node, "out_pad", 3),
        _ => {
            let raw_out_pad = raw_transpose_conv_output_padding(node);
            if raw_out_pad.len() == 4 {
                return raw_out_pad;
            }
            let raw_padding = kwarg_or_arg_i32_list(node, "padding", 4);
            let padding = expand_pair(raw_padding);
            let output_padding = expand_pair(raw_out_pad);
            vec![
                -padding[0],
                -padding[0] + output_padding[0],
                -padding[1],
                -padding[1] + output_padding[1],
            ]
        }
    }
}

fn transpose_conv_stride(node: &Node) -> Vec<i32> {
    match &node.target {
        OpTarget::TosaTransposeConv2d => kwarg_or_arg_i32_list(node, "stride", 4),
        _ => kwarg_or_arg_i32_list(node, "stride", 3),
    }
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

    let out_pad = transpose_conv_out_pad(node);
    let stride = transpose_conv_stride(node);
    let local_bound = kwarg_bool(node, "local_bound", false);

    let (input_zp, weight_zp) = input_weight_zero_points(node, true);
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
