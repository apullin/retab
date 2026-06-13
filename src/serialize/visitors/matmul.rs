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
    let float_output_cast_dtype = if matches!(a_dtype, DType::Fp16 | DType::Bf16)
        && matches!(b_dtype, DType::Fp16 | DType::Bf16)
    {
        Some(a_dtype)
    } else {
        None
    };
    let needs_float_output_cast = float_output_cast_dtype.is_some();
    let matmul_out_dtype = if needs_float_output_cast {
        DType::Fp32
    } else {
        match (a_dtype, b_dtype) {
            (DType::Int16, _) | (_, DType::Int16) => DType::Int48,
            (DType::Int8, DType::Int8) => DType::Int32,
            _ => accumulator_output_dtype(node),
        }
    };

    let a_zp = input_qparam_zero_point_or_kwarg(node, 0, "A_zp", 0);
    let b_zp = input_qparam_zero_point_or_kwarg(node, 1, "B_zp", 0);

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

            vec![a_reshape_name, b_reshape_name]
        } else {
            vec![]
        }
    } else {
        input_names
    };

    let a_zp_name = format!("{}_A_zp", node.name);
    push_zero_point_const(&mut result, a_zp_name.clone(), a_dtype, a_zp);

    let b_zp_name = format!("{}_B_zp", node.name);
    push_zero_point_const(&mut result, b_zp_name.clone(), b_dtype, b_zp);

    matmul_inputs.push(a_zp_name);
    matmul_inputs.push(b_zp_name);

    let matmul_output_name = if promote_2d {
        if defer_reshape_to_rescale {
            node.name.clone()
        } else if needs_float_output_cast {
            format!("{}_batched_matmul", node.name)
        } else {
            format!("{}_batched", node.name)
        }
    } else if needs_float_output_cast {
        format!("{}_matmul", node.name)
    } else {
        node.name.clone()
    };

    let matmul_output_shape = if promote_2d {
        promoted_shape.clone().unwrap_or_else(|| out_shape.clone())
    } else {
        out_shape.clone()
    };

    result.push_output_op(
        TosaTensor::new(
            &matmul_output_name,
            matmul_output_shape.clone(),
            matmul_out_dtype,
        ),
        Op::MatMul,
        Attribute::MatMul(MatMulAttribute {}),
        matmul_inputs,
    );

    let mut final_input_name = matmul_output_name;
    let mut final_input_dtype = matmul_out_dtype;
    if needs_float_output_cast {
        let cast_dtype = float_output_cast_dtype.expect("low-precision float cast dtype");
        let cast_output_name = if promote_2d {
            format!("{}_batched", node.name)
        } else {
            node.name.clone()
        };
        result.push_output_op(
            TosaTensor::new(&cast_output_name, matmul_output_shape.clone(), cast_dtype),
            Op::Cast,
            Attribute::Cast(CastAttribute {}),
            vec![final_input_name.clone()],
        );
        final_input_name = cast_output_name;
        final_input_dtype = cast_dtype;
    }

    if promote_2d && !defer_reshape_to_rescale {
        push_reshape_with_shape(
            &mut result,
            final_input_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            out_shape,
            final_input_dtype,
        );
    }

    result
}
