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
    let args = if node.args.len() > 7 {
        arg_i32_list(node, 7)
    } else {
        arg_i32_list(node, 5)
    };
    if args.is_empty() {
        kwarg_i32_list(node, "output_padding")
    } else {
        args
    }
}

fn transpose_conv_out_pad(node: &Node) -> Vec<i32> {
    match &node.target {
        OpTarget::TosaTransposeConv2d => {
            let args = arg_i32_list(node, 3);
            if args.is_empty() {
                kwarg_i32_list(node, "out_pad")
            } else {
                args
            }
        }
        _ => {
            let raw_out_pad = raw_transpose_conv_output_padding(node);
            if raw_out_pad.len() == 4 {
                return raw_out_pad;
            }
            let raw_padding = {
                let args = arg_i32_list(node, 4);
                if args.is_empty() {
                    kwarg_i32_list(node, "padding")
                } else {
                    args
                }
            };
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
        OpTarget::TosaTransposeConv2d => {
            let args = arg_i32_list(node, 4);
            if args.is_empty() {
                kwarg_i32_list(node, "stride")
            } else {
                args
            }
        }
        _ => {
            let args = arg_i32_list(node, 3);
            if args.is_empty() {
                kwarg_i32_list(node, "stride")
            } else {
                args
            }
        }
    }
}

fn output_channels(node: &Node) -> i32 {
    node.meta
        .val
        .shape_slice()
        .get(1)
        .copied()
        .or_else(|| node.meta.val.shape_slice().first().copied())
        .unwrap_or(0) as i32
}

fn zero_bytes(dtype: DType, len: usize) -> Vec<u8> {
    let bytes_per_elem = match dtype {
        DType::Bool | DType::Int4 | DType::Int8 => 1,
        DType::Int16 | DType::Fp16 | DType::Bf16 => 2,
        DType::Int32 | DType::Fp32 => 4,
        DType::Int48 => 6,
        DType::Int64 | DType::Shape => 8,
        _ => 4,
    };
    vec![0; bytes_per_elem * len]
}

fn push_zero_bias_tensor(result: &mut EmitResult, node: &Node, dtype: DType) -> String {
    let channels = output_channels(node).max(0) as usize;
    result.push_const(TosaTensor::with_data(
        &format!("{}_bias", node.name),
        vec![channels as i32],
        dtype,
        zero_bytes(dtype, channels),
    ))
}

fn synthetic_bias_dtype(node: &Node, acc_type: DType) -> DType {
    let semantic_dtype = node_output_dtype(node);
    if node.meta.input_qparams.is_empty()
        && matches!(semantic_dtype, DType::Fp32 | DType::Fp16 | DType::Bf16)
    {
        semantic_dtype
    } else {
        acc_type
    }
}

pub(in crate::serialize::visitors) fn emit_transpose_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let output_shape = apply_tosa_dim_order(node, raw_i32_shape(node));
    let acc_type = conv_accumulator_dtype(node);
    let output_dtype = if node.meta.input_qparams.is_empty()
        && matches!(node_output_dtype(node), DType::Fp32 | DType::Fp16 | DType::Bf16)
    {
        node_output_dtype(node)
    } else {
        acc_type
    };
    result
        .tensors
        .push(make_empty_tensor(node.name.clone(), output_shape, output_dtype));

    let out_pad = transpose_conv_out_pad(node);
    let stride = transpose_conv_stride(node);
    let local_bound = kwarg_bool(node, "local_bound", false);

    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let mut input_names = input_names;
        if matches!(input_names.len(), 2 | 4) {
            input_names.insert(
                2,
                push_zero_bias_tensor(&mut result, node, synthetic_bias_dtype(node, acc_type)),
            );
        }
        if input_names.len() >= 5 {
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
        }
    };

    result.operators.push(TosaOperator::new(
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
        vec![node.name.clone()],
    ));
    result
}
