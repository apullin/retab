use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::Div, emit_aten_div as EmitFn),
        (OpTarget::Sqrt, emit_aten_sqrt),
        (OpTarget::FullLike, emit_full_like),
        (OpTarget::Sign, emit_aten_sign),
        (OpTarget::Expm1, emit_aten_expm1),
        (OpTarget::Log1p, emit_aten_log1p),
        (OpTarget::FloorDivide, emit_aten_floor_divide),
        (OpTarget::LeakyRelu, emit_aten_leaky_relu),
    ]);
}

pub(in crate::serialize::visitors) fn emit_aten_div(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec =
        ParsedBinaryTensorSpec::parse(graph, store, node, &mut result).unwrap_or_else(|| {
            panic!(
                "emit_aten_div: missing binary inputs for node '{}'",
                node.name
            )
        });

    let recip_name = format!("{}_recip", node.name);
    result.push_output_op(
        temp_tensor(
            recip_name.clone(),
            spec.output_shape.clone(),
            spec.base_dtype,
        ),
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute),
        vec![spec.rhs],
    );

    let shift_name = push_shift_const(&mut result, &node.name);
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        vec![spec.lhs, recip_name, shift_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_aten_sqrt(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_aten_sqrt: missing input for node '{}'", node.name));
    let rsqrt_name = format!("{}_rsqrt", node.name);
    result.push_output_op(
        temp_tensor(rsqrt_name.clone(), spec.output_shape, spec.base_dtype),
        Op::Rsqrt,
        Attribute::Rsqrt(RsqrtAttribute),
        vec![spec.input_name],
    );
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute),
        vec![rsqrt_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_full_like(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let output_shape = actual_tensor_shape(graph, store, node);
    let fill_val = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        Some(Arg::Int(i)) => *i as f64,
        _ => 0.0,
    };
    let fill_val = quantize_fill_value(node, fill_val);

    if let Some(tensor) = make_filled_const_tensor(
        node.name.clone(),
        output_shape,
        node.meta.val.dtype,
        fill_val,
    ) {
        push_inline_const(&mut result, tensor);
    } else {
        result
            .tensors
            .push(make_actual_output_tensor(graph, store, node));
    }
    result
}

pub(in crate::serialize::visitors) fn emit_aten_sign(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_aten_sign: missing input for node '{}'", node.name));

    let zero_name = format!("{}_sign_zero", node.name);
    let one_name = format!("{}_sign_one", node.name);
    let neg_one_name = format!("{}_sign_neg_one", node.name);
    for (name, value) in [
        (zero_name.clone(), 0.0),
        (one_name.clone(), 1.0),
        (neg_one_name.clone(), -1.0),
    ] {
        let fill_value = quantize_fill_value(node, value);
        if let Some(t) =
            make_filled_const_tensor(name, spec.output_shape.clone(), spec.ir_dtype, fill_value)
        {
            push_inline_const(&mut result, t);
        }
    }

    let gt_name = format!("{}_sign_gt", node.name);
    let ge_name = format!("{}_sign_ge", node.name);
    for name in [gt_name.clone(), ge_name.clone()] {
        result
            .tensors
            .push(temp_tensor(name, spec.output_shape.clone(), DType::Bool));
    }

    result.push_op(
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        vec![spec.input_name.clone(), zero_name.clone()],
        vec![gt_name.clone()],
    );
    result.push_op(
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        vec![spec.input_name.clone(), zero_name.clone()],
        vec![ge_name.clone()],
    );

    let inner_name = format!("{}_sign_inner", node.name);
    result.push_output_op(
        temp_tensor(inner_name.clone(), spec.output_shape, spec.base_dtype),
        Op::Select,
        Attribute::Select(SelectAttribute),
        vec![gt_name, one_name, zero_name],
    );
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Select,
        Attribute::Select(SelectAttribute),
        vec![ge_name, inner_name, neg_one_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_aten_expm1(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_aten_expm1: missing input for node '{}'", node.name));

    let exp_name = format!("{}_exp", node.name);
    result.push_output_op(
        temp_tensor(exp_name.clone(), spec.output_shape.clone(), spec.base_dtype),
        Op::Exp,
        Attribute::Exp(ExpAttribute),
        vec![spec.input_name],
    );

    let one_name = format!("{}_one", node.name);
    if let Some(t) =
        make_filled_const_tensor(one_name.clone(), spec.output_shape, spec.ir_dtype, 1.0)
    {
        push_inline_const(&mut result, t);
    }
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Sub,
        Attribute::Sub(SubAttribute),
        vec![exp_name, one_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_aten_log1p(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_aten_log1p: missing input for node '{}'", node.name));

    let one_name = format!("{}_one", node.name);
    if let Some(t) = make_filled_const_tensor(
        one_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        1.0,
    ) {
        push_inline_const(&mut result, t);
    }

    let add_name = format!("{}_add_one", node.name);
    result.push_output_op(
        temp_tensor(add_name.clone(), spec.output_shape, spec.base_dtype),
        Op::Add,
        Attribute::Add(AddAttribute),
        vec![spec.input_name, one_name],
    );
    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Log,
        Attribute::Log(LogAttribute),
        vec![add_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_aten_floor_divide(ctx: &EmitContext) -> EmitResult {
    emit_aten_div(ctx)
}

pub(in crate::serialize::visitors) fn emit_aten_leaky_relu(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node).unwrap_or_else(|| {
        panic!(
            "emit_aten_leaky_relu: missing input for node '{}'",
            node.name
        )
    });
    let alpha = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        _ => 0.01,
    };

    let zero_name = format!("{}_zero", node.name);
    if let Some(t) = make_filled_const_tensor(
        zero_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.0,
    ) {
        push_inline_const(&mut result, t);
    }

    let ge_name = format!("{}_ge", node.name);
    result.push_output_op(
        temp_tensor(ge_name.clone(), spec.output_shape.clone(), DType::Bool),
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        vec![spec.input_name.clone(), zero_name],
    );

    let alpha_name = format!("{}_alpha", node.name);
    if let Some(t) = make_filled_const_tensor(
        alpha_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        alpha,
    ) {
        push_inline_const(&mut result, t);
    }

    let scaled_name = format!("{}_scaled", node.name);
    let shift_name = push_shift_const(&mut result, &node.name);
    result.push_output_op(
        temp_tensor(scaled_name.clone(), spec.output_shape, spec.base_dtype),
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        vec![spec.input_name.clone(), alpha_name, shift_name],
    );

    result.push_output_op(
        make_actual_output_tensor(graph, store, node),
        Op::Select,
        Attribute::Select(SelectAttribute),
        vec![ge_name, spec.input_name, scaled_name],
    );
    result
}
