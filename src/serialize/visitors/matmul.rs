use super::emitter::EmitContext;
use super::*;

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaMatmul, emit_matmul as EmitFn),
        (OpTarget::Matmul, emit_matmul),
        (OpTarget::Bmm, emit_matmul),
    ]);
}

pub(super) fn emit_matmul(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let out_shape = ctx.shape();
    let promoted_shape = promoted_2d_matmul_shape(graph, store, node);
    let promote_2d = promoted_shape.is_some();
    let defer_reshape_to_rescale = promote_2d && has_only_tosa_rescale_users(graph, node);
    let a_dtype = matmul_input_dtype(graph, node, 0);
    let b_dtype = matmul_input_dtype(graph, node, 1);
    let matmul_out_dtype = match (a_dtype, b_dtype) {
        (DType::Int16, _) | (_, DType::Int16) => DType::Int48,
        (DType::Int8, DType::Int8) => DType::Int32,
        _ => accumulator_output_dtype(node),
    };

    if !promote_2d {
        result.tensors.push(TosaTensor::new(
            node.name.clone(),
            out_shape.clone(),
            matmul_out_dtype,
        ));
    }

    let a_zp = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 0)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or_else(|| kwarg_i32(node, "A_zp", 0));
    let b_zp = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 1)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or_else(|| kwarg_i32(node, "B_zp", 0));

    let mut matmul_inputs = if promote_2d {
        let (a_shape, b_shape) = match (node.args.first(), node.args.get(1)) {
            (Some(Arg::Node(a_id)), Some(Arg::Node(b_id))) => {
                let a_shape = graph
                    .node(*a_id)
                    .map(|n| actual_tensor_shape(graph, store, n))
                    .unwrap_or_default();
                let b_shape = graph
                    .node(*b_id)
                    .map(|n| actual_tensor_shape(graph, store, n))
                    .unwrap_or_default();
                (a_shape, b_shape)
            }
            _ => (Vec::new(), Vec::new()),
        };

        let a_name = input_names.first().cloned().unwrap_or_default();
        let b_name = input_names.get(1).cloned().unwrap_or_default();
        let a_reshape_name = format!("{}_A_reshape", node.name);
        let b_reshape_name = format!("{}_B_reshape", node.name);

        if a_shape.len() == 2 && b_shape.len() == 2 {
            let a_out_shape = vec![1, a_shape[0], a_shape[1]];
            let b_out_shape = vec![1, b_shape[0], b_shape[1]];
            let temp_out_shape = promoted_shape
                .clone()
                .unwrap_or_else(|| vec![1, a_shape[0], b_shape[1]]);
            let temp_out_name = if defer_reshape_to_rescale {
                node.name.clone()
            } else {
                format!("{}_batched", node.name)
            };

            push_reshape_with_shape(
                &mut result,
                a_name,
                a_reshape_name.clone(),
                format!("{}_shape", a_reshape_name),
                a_out_shape,
                a_dtype,
            );
            push_reshape_with_shape(
                &mut result,
                b_name,
                b_reshape_name.clone(),
                format!("{}_shape", b_reshape_name),
                b_out_shape,
                b_dtype,
            );

            result.tensors.push(TosaTensor::new(
                temp_out_name.clone(),
                temp_out_shape,
                matmul_out_dtype,
            ));

            vec![a_reshape_name, b_reshape_name]
        } else {
            vec![]
        }
    } else {
        input_names
    };

    let a_zp_name = format!("{}_A_zp", node.name);
    push_inline_const(
        &mut result,
        TosaTensor::with_data(
            a_zp_name.clone(),
            vec![1],
            a_dtype,
            encode_zp_for_dtype(a_zp, a_dtype),
        ),
    );

    let b_zp_name = format!("{}_B_zp", node.name);
    push_inline_const(
        &mut result,
        TosaTensor::with_data(
            b_zp_name.clone(),
            vec![1],
            b_dtype,
            encode_zp_for_dtype(b_zp, b_dtype),
        ),
    );

    matmul_inputs.push(a_zp_name);
    matmul_inputs.push(b_zp_name);

    let matmul_output_name = if promote_2d {
        if defer_reshape_to_rescale {
            node.name.clone()
        } else {
            format!("{}_batched", node.name)
        }
    } else {
        node.name.clone()
    };

    result.operators.push(TosaOperator::new(
        Op::MatMul,
        Attribute::MatMul(MatMulAttribute),
        matmul_inputs,
        vec![matmul_output_name.clone()],
    ));

    if promote_2d && !defer_reshape_to_rescale {
        push_reshape_with_shape(
            &mut result,
            matmul_output_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            out_shape,
            matmul_out_dtype,
        );
    }

    result
}
