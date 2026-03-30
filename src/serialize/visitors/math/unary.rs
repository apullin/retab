use super::*;

pub(super) fn register(table: &mut std::collections::HashMap<crate::graph::ir::OpTarget, crate::serialize::visitors::emitter::EmitFn>) {
    use crate::graph::ir::OpTarget;
    // TOSA-lowered
    table.insert(OpTarget::TosaNeg, emit_negate);
    table.insert(OpTarget::TosaNegate, emit_negate);
    table.insert(OpTarget::TosaCeil, emit_ceil);
    table.insert(OpTarget::TosaFloor, emit_floor);
    // ATen fallbacks
    table.insert(OpTarget::Neg, emit_negate);
    table.insert(OpTarget::Ceil, emit_ceil);
    table.insert(OpTarget::Floor, emit_floor);
    table.insert(OpTarget::Round, emit_round);
}

pub(in crate::serialize::visitors) fn emit_negate(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let input_dtype = passthrough_input_dtype(graph, node);
    result.tensors.push(make_output_tensor(node));

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
            name.clone(),
            vec![1],
            input_dtype,
            encode_zp_for_dtype(value, input_dtype),
        ));
        result.operators.push(TosaOperator::new(
            Op::Const,
            Attribute::Const(ConstAttribute),
            vec![],
            vec![name],
        ));
    }

    let mut negate_inputs = input_names;
    negate_inputs.push(input_zp_name);
    negate_inputs.push(output_zp_name);
    result.operators.push(TosaOperator::new(
        Op::Negate,
        Attribute::Negate(NegateAttribute),
        negate_inputs,
        vec![node.name.clone()],
    ));
    result
}

pub(in crate::serialize::visitors) fn emit_ceil(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        emit_identity(graph, node)
    } else {
        let mut result = EmitResult::default();
        let input_names = collect_input_names(graph, node);
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator::new(
            Op::Ceil,
            Attribute::Ceil(CeilAttribute),
            input_names,
            vec![node.name.clone()],
        ));
        result
    }
}

pub(in crate::serialize::visitors) fn emit_floor(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        emit_identity(graph, node)
    } else {
        let mut result = EmitResult::default();
        let input_names = collect_input_names(graph, node);
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator::new(
            Op::Floor,
            Attribute::Floor(FloorAttribute),
            input_names,
            vec![node.name.clone()],
        ));
        result
    }
}

pub(in crate::serialize::visitors) fn emit_round(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        return emit_identity(graph, node);
    }

    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_round: missing input for node '{}'", node.name));

    let zero_name = format!("{}_round_zero", node.name);
    let half_name = format!("{}_round_half", node.name);
    let ge_name = format!("{}_round_ge", node.name);
    let plus_half_name = format!("{}_round_plus_half", node.name);
    let minus_half_name = format!("{}_round_minus_half", node.name);
    let floor_name = format!("{}_round_floor", node.name);
    let ceil_name = format!("{}_round_ceil", node.name);

    let zero_tensor = make_filled_const_tensor(
        zero_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.0,
    )
    .unwrap_or_else(|| {
        panic!(
            "emit_round: unsupported zero-fill dtype {:?} for node '{}'",
            spec.ir_dtype, node.name
        )
    });
    let half_tensor = make_filled_const_tensor(
        half_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.5,
    )
    .unwrap_or_else(|| {
        panic!(
            "emit_round: unsupported half-fill dtype {:?} for node '{}'",
            spec.ir_dtype, node.name
        )
    });
    push_inline_const(&mut result, zero_tensor);
    push_inline_const(&mut result, half_tensor);

    for tensor_name in [
        ge_name.clone(),
        plus_half_name.clone(),
        minus_half_name.clone(),
    ] {
        let is_bool = tensor_name == ge_name;
        result.tensors.push(TosaTensor::new(
            tensor_name,
            spec.output_shape.clone(),
            if is_bool {
                DType::Bool
            } else {
                ir_dtype_to_tosa(spec.ir_dtype)
            },
        ));
    }

    result.operators.push(TosaOperator::new(
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        vec![spec.input_name.clone(), zero_name],
        vec![ge_name.clone()],
    ));
    result.operators.push(TosaOperator::new(
        Op::Add,
        Attribute::Add(AddAttribute),
        vec![spec.input_name.clone(), half_name.clone()],
        vec![plus_half_name.clone()],
    ));
    result.operators.push(TosaOperator::new(
        Op::Sub,
        Attribute::Sub(SubAttribute),
        vec![spec.input_name.clone(), half_name],
        vec![minus_half_name.clone()],
    ));

    for (tensor_name, op, input) in [
        (floor_name.clone(), Op::Floor, plus_half_name),
        (ceil_name.clone(), Op::Ceil, minus_half_name),
    ] {
        result.tensors.push(TosaTensor::new(
            tensor_name.clone(),
            spec.output_shape.clone(),
            ir_dtype_to_tosa(spec.ir_dtype),
        ));
        result.operators.push(TosaOperator::new(
            op,
            if op == Op::Floor {
                Attribute::Floor(FloorAttribute)
            } else {
                Attribute::Ceil(CeilAttribute)
            },
            vec![input],
            vec![tensor_name],
        ));
    }

    result.tensors.push(TosaTensor::new(
        node.name.clone(),
        spec.output_shape,
        ir_dtype_to_tosa(spec.ir_dtype),
    ));
    result.operators.push(TosaOperator::new(
        Op::Select,
        Attribute::Select(SelectAttribute),
        vec![ge_name, floor_name, ceil_name],
        vec![node.name.clone()],
    ));
    result
}
