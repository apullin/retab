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
        (OpTarget::Remainder, emit_aten_remainder),
        (OpTarget::LeakyRelu, emit_aten_leaky_relu),
    ]);
}

fn emit_div_quotient(
    graph: &Graph,
    store: &TensorStore,
    _node: &Node,
    spec: &ParsedBinaryTensorSpec,
    result: &mut EmitResult,
    output_name: String,
) {
    let recip_name = format!("{}_recip", output_name);
    let recip_shape =
        emitted_or_graph_tensor_shape(result, graph, store, &spec.rhs, &spec.output_shape);
    result.unary(
        recip_name.clone(),
        recip_shape,
        spec.base_dtype,
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute {}),
        spec.rhs.clone(),
    );

    let shift_name = push_shift_const(result, &output_name);
    result.ternary(
        output_name,
        spec.output_shape.clone(),
        spec.base_dtype,
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        spec.lhs.clone(),
        recip_name,
        shift_name,
    );
}

fn emitted_or_graph_tensor_shape(
    result: &EmitResult,
    graph: &Graph,
    store: &TensorStore,
    name: &str,
    fallback: &[i32],
) -> Vec<i32> {
    result
        .tensors
        .iter()
        .rev()
        .find(|tensor| tensor.name == name)
        .map(|tensor| tensor.shape.clone())
        .or_else(|| {
            graph
                .nodes
                .values()
                .find(|node| node.name == name)
                .map(|node| actual_tensor_shape(graph, store, node))
        })
        .unwrap_or_else(|| fallback.to_vec())
}

fn emit_div_with_rounding(ctx: &EmitContext, rounding_mode: Option<&str>) -> EmitResult {
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

    let output_name = primary_output_name(graph, node);
    let quotient_name = match rounding_mode {
        Some("floor") | Some("trunc") => format!("{}_div", node.name),
        _ => output_name,
    };
    emit_div_quotient(
        graph,
        store,
        node,
        &spec,
        &mut result,
        quotient_name.clone(),
    );

    match rounding_mode {
        None => {}
        Some("floor") => {
            result.push_output_op(
                ctx.make_output(),
                Op::Floor,
                Attribute::Floor(FloorAttribute {}),
                vec![quotient_name],
            );
        }
        Some("trunc") => {
            let zero_name = format!("{}_zero", node.name);
            push_filled_const_if_supported(
                &mut result,
                zero_name.clone(),
                spec.output_shape.clone(),
                node.meta.val.dtype,
                0.0,
            );

            let is_neg_name = format!("{}_is_neg", node.name);
            result.tensors.push(temp_tensor(
                is_neg_name.clone(),
                spec.output_shape.clone(),
                DType::Bool,
            ));
            result.push_op(
                Op::Greater,
                Attribute::Greater(GreaterAttribute {}),
                vec![zero_name, quotient_name.clone()],
                vec![is_neg_name.clone()],
            );

            let ceil_name = format!("{}_ceil", node.name);
            result.unary(
                ceil_name.clone(),
                spec.output_shape.clone(),
                spec.base_dtype,
                Op::Ceil,
                Attribute::Ceil(CeilAttribute {}),
                quotient_name.clone(),
            );

            let floor_name = format!("{}_floor", node.name);
            result.unary(
                floor_name.clone(),
                spec.output_shape.clone(),
                spec.base_dtype,
                Op::Floor,
                Attribute::Floor(FloorAttribute {}),
                quotient_name,
            );

            result.push_output_op(
                ctx.make_output(),
                Op::Select,
                Attribute::Select(SelectAttribute {}),
                vec![is_neg_name, ceil_name, floor_name],
            );
        }
        Some(_) => {}
    }

    result
}

pub(in crate::serialize::visitors) fn emit_aten_div(ctx: &EmitContext) -> EmitResult {
    let rounding_mode = kwarg_string(ctx.node, "rounding_mode")
        .or_else(|| arg_string(ctx.node, 2))
        .unwrap_or_default();
    if rounding_mode.is_empty() {
        emit_div_with_rounding(ctx, None)
    } else {
        emit_div_with_rounding(ctx, Some(rounding_mode.as_str()))
    }
}

pub(in crate::serialize::visitors) fn emit_aten_sqrt(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_aten_sqrt: missing input for node '{}'", node.name));
    let rsqrt_name = format!("{}_rsqrt", node.name);
    result.unary(
        rsqrt_name.clone(),
        spec.output_shape,
        spec.base_dtype,
        Op::Rsqrt,
        Attribute::Rsqrt(RsqrtAttribute {}),
        spec.input_name,
    );
    result.push_output_op(
        ctx.make_output(),
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute {}),
        vec![rsqrt_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_full_like(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let output_shape = ctx.shape();
    let fill_val = match node.args.get(1) {
        Some(Arg::Float(f)) => *f,
        Some(Arg::Int(i)) => *i as f64,
        _ => 0.0,
    };
    let fill_val = quantize_fill_value(node, fill_val);

    if push_filled_const_if_supported(
        &mut result,
        node.name.clone(),
        output_shape,
        node.meta.val.dtype,
        fill_val,
    ) {
    } else {
        result.tensors.push(ctx.make_output());
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
        push_filled_const_if_supported(
            &mut result,
            name,
            spec.output_shape.clone(),
            spec.ir_dtype,
            fill_value,
        );
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
        Attribute::Greater(GreaterAttribute {}),
        vec![spec.input_name.clone(), zero_name.clone()],
        vec![gt_name.clone()],
    );
    result.push_op(
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute {}),
        vec![spec.input_name.clone(), zero_name.clone()],
        vec![ge_name.clone()],
    );

    let inner_name = format!("{}_sign_inner", node.name);
    result.ternary(
        inner_name.clone(),
        spec.output_shape,
        spec.base_dtype,
        Op::Select,
        Attribute::Select(SelectAttribute {}),
        gt_name,
        one_name,
        zero_name,
    );
    result.push_output_op(
        ctx.make_output(),
        Op::Select,
        Attribute::Select(SelectAttribute {}),
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
    result.unary(
        exp_name.clone(),
        spec.output_shape.clone(),
        spec.base_dtype,
        Op::Exp,
        Attribute::Exp(ExpAttribute {}),
        spec.input_name,
    );

    let one_name = format!("{}_one", node.name);
    push_filled_const_if_supported(
        &mut result,
        one_name.clone(),
        spec.output_shape,
        spec.ir_dtype,
        1.0,
    );
    result.push_output_op(
        ctx.make_output(),
        Op::Sub,
        Attribute::Sub(SubAttribute {}),
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
    push_filled_const_if_supported(
        &mut result,
        one_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        1.0,
    );

    let add_name = format!("{}_add_one", node.name);
    result.binary(
        add_name.clone(),
        spec.output_shape,
        spec.base_dtype,
        Op::Add,
        Attribute::Add(AddAttribute {}),
        spec.input_name,
        one_name,
    );
    result.push_output_op(
        ctx.make_output(),
        Op::Log,
        Attribute::Log(LogAttribute {}),
        vec![add_name],
    );
    result
}

pub(in crate::serialize::visitors) fn emit_aten_floor_divide(ctx: &EmitContext) -> EmitResult {
    emit_div_with_rounding(ctx, Some("floor"))
}

pub(in crate::serialize::visitors) fn emit_aten_remainder(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let spec =
        ParsedBinaryTensorSpec::parse(graph, store, node, &mut result).unwrap_or_else(|| {
            panic!(
                "emit_aten_remainder: missing binary inputs for node '{}'",
                node.name
            )
        });

    let floor_div_name = format!("{}_floor_div", node.name);
    emit_div_quotient(
        graph,
        store,
        node,
        &spec,
        &mut result,
        floor_div_name.clone(),
    );
    let floor_name = format!("{}_floor", node.name);
    result.unary(
        floor_name.clone(),
        spec.output_shape.clone(),
        spec.base_dtype,
        Op::Floor,
        Attribute::Floor(FloorAttribute {}),
        floor_div_name,
    );

    let prod_name = format!("{}_product", node.name);
    let shift_name = push_shift_const(&mut result, &prod_name);
    result.ternary(
        prod_name.clone(),
        spec.output_shape.clone(),
        spec.base_dtype,
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        floor_name,
        spec.rhs.clone(),
        shift_name,
    );

    result.push_output_op(
        ctx.make_output(),
        Op::Sub,
        Attribute::Sub(SubAttribute {}),
        vec![spec.lhs, prod_name],
    );
    result
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
    push_filled_const_if_supported(
        &mut result,
        zero_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.0,
    );

    let ge_name = format!("{}_ge", node.name);
    result.binary(
        ge_name.clone(),
        spec.output_shape.clone(),
        DType::Bool,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute {}),
        spec.input_name.clone(),
        zero_name,
    );

    let alpha_name = format!("{}_alpha", node.name);
    push_filled_const_if_supported(
        &mut result,
        alpha_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        alpha,
    );

    let scaled_name = format!("{}_scaled", node.name);
    let shift_name = push_shift_const(&mut result, &node.name);
    result.ternary(
        scaled_name.clone(),
        spec.output_shape,
        spec.base_dtype,
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        spec.input_name.clone(),
        alpha_name,
        shift_name,
    );

    result.push_output_op(
        ctx.make_output(),
        Op::Select,
        Attribute::Select(SelectAttribute {}),
        vec![ge_name, spec.input_name, scaled_name],
    );
    result
}
