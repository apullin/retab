use super::*;

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

pub(in crate::serialize::visitors) fn emit_transpose_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let output_shape = apply_tosa_dim_order(node, raw_i32_shape(node));
    let acc_type = conv_accumulator_dtype(node);
    result
        .tensors
        .push(make_empty_tensor(node.name.clone(), output_shape, acc_type));

    let out_pad = {
        let args = arg_i32_list(node, 3);
        if args.is_empty() {
            kwarg_i32_list(node, "out_pad")
        } else {
            args
        }
    };
    let stride = {
        let args = arg_i32_list(node, 4);
        if args.is_empty() {
            kwarg_i32_list(node, "stride")
        } else {
            args
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);

    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let input_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
        let (input_zp_name, weight_zp_name) =
            push_input_weight_zp_tensors(&mut result, graph, node, input_zp_val, weight_zp_val);
        extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
    };

    result.operators.push(TosaOperator::new(
        Op::TransposeConv2d,
        Attribute::TransposeConv2d(TransposeConv2dAttribute {
            out_pad: if out_pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                out_pad
            },
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            local_bound,
            acc_type,
        }),
        conv_inputs,
        vec![node.name.clone()],
    ));
    result
}
