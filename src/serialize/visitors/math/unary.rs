use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaNeg, emit_negate as EmitFn),
        (OpTarget::TosaNegate, emit_negate),
        (OpTarget::TosaCeil, emit_ceil),
        (OpTarget::TosaFloor, emit_floor),
        (OpTarget::Neg, emit_negate),
        (OpTarget::Ceil, emit_ceil),
        (OpTarget::Floor, emit_floor),
        (OpTarget::Round, emit_round),
    ]);
}

fn emit_integral_passthrough_or_unary(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    if is_integral_dtype(passthrough_input_dtype(graph, node)) {
        emit_identity(graph, store, node)
    } else {
        emit_direct_unary(graph, store, node, op, attribute)
    }
}

pub(in crate::serialize::visitors) fn emit_negate(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let input_dtype = ctx.passthrough_dtype();
    result.tensors.push(ctx.make_output());

    let (input_zp, output_zp) = if input_dtype == DType::Int8 {
        let input_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        let output_zp = node
            .meta
            .output_qparams
            .first()
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        (input_zp, output_zp)
    } else {
        (0, 0)
    };

    let input_zp_name = format!("{}_input_zp", node.name);
    let output_zp_name = format!("{}_output_zp", node.name);
    for (name, value) in [
        (input_zp_name.clone(), input_zp),
        (output_zp_name.clone(), output_zp),
    ] {
        result.tensors.push(TosaTensor::with_data(
            &name,
            vec![1],
            input_dtype,
            encode_zp_for_dtype(value, input_dtype),
        ));
        result.operators.push(TosaOperator::new(
            Op::Const,
            Attribute::Const(ConstAttribute {}),
            vec![],
            vec![name],
        ));
    }

    let mut negate_inputs = input_names;
    negate_inputs.push(input_zp_name);
    negate_inputs.push(output_zp_name);
    result.operators.push(TosaOperator::new(
        Op::Negate,
        Attribute::Negate(NegateAttribute {}),
        negate_inputs,
        vec![node.name.clone()],
    ));
    result
}

pub(in crate::serialize::visitors) fn emit_ceil(ctx: &EmitContext) -> EmitResult {
    emit_integral_passthrough_or_unary(
        ctx.graph,
        ctx.store,
        ctx.node,
        Op::Ceil,
        Attribute::Ceil(CeilAttribute {}),
    )
}

pub(in crate::serialize::visitors) fn emit_floor(ctx: &EmitContext) -> EmitResult {
    emit_integral_passthrough_or_unary(
        ctx.graph,
        ctx.store,
        ctx.node,
        Op::Floor,
        Attribute::Floor(FloorAttribute {}),
    )
}

pub(in crate::serialize::visitors) fn emit_round(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let input_dtype = ctx.passthrough_dtype();
    if is_integral_dtype(input_dtype) {
        return ctx.emit_identity();
    }

    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_round: missing input for node '{}'", node.name));

    let half_name = format!("{}_round_half", node.name);
    let one_name = format!("{}_round_one", node.name);
    let two_name = format!("{}_round_two", node.name);
    let floor_x_name = format!("{}_round_floor_x", node.name);
    let frac_name = format!("{}_round_frac", node.name);
    let is_tie_name = format!("{}_round_is_tie", node.name);
    let plus_half_name = format!("{}_round_plus_half", node.name);
    let round_up_name = format!("{}_round_up", node.name);
    let half_floor_mul_name = format!("{}_round_half_floor_mul", node.name);
    let half_floor_name = format!("{}_round_half_floor", node.name);
    let even_floor_name = format!("{}_round_even_floor", node.name);
    let floor_is_even_name = format!("{}_round_floor_is_even", node.name);
    let floor_plus_one_name = format!("{}_round_floor_plus_one", node.name);
    let tie_result_name = format!("{}_round_tie_result", node.name);

    for (name, value, label) in [
        (half_name.clone(), 0.5, "half"),
        (one_name.clone(), 1.0, "one"),
        (two_name.clone(), 2.0, "two"),
    ] {
        let tensor = make_filled_const_tensor(
            name,
            spec.output_shape.clone(),
            spec.ir_dtype,
            quantize_fill_value(node, value),
        )
        .unwrap_or_else(|| {
            panic!(
                "emit_round: unsupported {}-fill dtype {:?} for node '{}'",
                label, spec.ir_dtype, node.name
            )
        });
        push_inline_const(&mut result, tensor);
    }

    let s = spec.output_shape.clone();
    let d = spec.base_dtype;
    let shift_name = push_shift_const(&mut result, &node.name);

    // 12-op banker's-rounding decomposition using the chainable builder API.
    result
        .unary(
            floor_x_name.clone(),
            s.clone(),
            d,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            spec.input_name.clone(),
        )
        .binary(
            frac_name.clone(),
            s.clone(),
            d,
            Op::Sub,
            Attribute::Sub(SubAttribute {}),
            spec.input_name.clone(),
            floor_x_name.clone(),
        )
        .binary(
            is_tie_name.clone(),
            s.clone(),
            DType::Bool,
            Op::Equal,
            Attribute::Equal(EqualAttribute {}),
            frac_name,
            half_name.clone(),
        )
        .binary(
            plus_half_name.clone(),
            s.clone(),
            d,
            Op::Add,
            Attribute::Add(AddAttribute {}),
            spec.input_name.clone(),
            half_name.clone(),
        )
        .unary(
            round_up_name.clone(),
            s.clone(),
            d,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            plus_half_name,
        )
        .ternary(
            half_floor_mul_name.clone(),
            s.clone(),
            d,
            Op::Mul,
            Attribute::Mul(MulAttribute { shift: 0 }),
            floor_x_name.clone(),
            half_name,
            shift_name.clone(),
        )
        .unary(
            half_floor_name.clone(),
            s.clone(),
            d,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            half_floor_mul_name,
        )
        .ternary(
            even_floor_name.clone(),
            s.clone(),
            d,
            Op::Mul,
            Attribute::Mul(MulAttribute { shift: 0 }),
            half_floor_name,
            two_name,
            shift_name,
        )
        .binary(
            floor_is_even_name.clone(),
            s.clone(),
            DType::Bool,
            Op::Equal,
            Attribute::Equal(EqualAttribute {}),
            floor_x_name.clone(),
            even_floor_name,
        )
        .binary(
            floor_plus_one_name.clone(),
            s.clone(),
            d,
            Op::Add,
            Attribute::Add(AddAttribute {}),
            floor_x_name.clone(),
            one_name,
        )
        .ternary(
            tie_result_name.clone(),
            s.clone(),
            d,
            Op::Select,
            Attribute::Select(SelectAttribute {}),
            floor_is_even_name,
            floor_x_name,
            floor_plus_one_name,
        );

    // Final output uses the actual output tensor (not a temp), so use push_output_op directly.
    result.push_output_op(
        ctx.make_output(),
        Op::Select,
        Attribute::Select(SelectAttribute {}),
        vec![is_tie_name, tie_result_name, round_up_name],
    );
    result
}
