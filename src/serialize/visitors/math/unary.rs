use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaNegate, emit_negate as EmitFn),
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
        (
            input_qparam_zero_point(node, 0).unwrap_or(0),
            output_qparam_zero_point(node).unwrap_or(0),
        )
    } else {
        (0, 0)
    };

    let input_zp_name = format!("{}_input_zp", node.name);
    let output_zp_name = format!("{}_output_zp", node.name);
    for (name, value) in [
        (input_zp_name.clone(), input_zp),
        (output_zp_name.clone(), output_zp),
    ] {
        push_zero_point_const(&mut result, name, input_dtype, value);
    }

    let mut negate_inputs = input_names;
    negate_inputs.push(input_zp_name);
    negate_inputs.push(output_zp_name);
    result.push_op(
        Op::Negate,
        Attribute::Negate(NegateAttribute {}),
        negate_inputs,
        vec![node.name.clone()],
    );
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

    emit_bankers_round_with_fill(
        &mut result,
        &node.name,
        &node.name,
        spec.input_name,
        spec.output_shape,
        spec.base_dtype,
        spec.ir_dtype,
        ctx.make_output(),
        |value| quantize_fill_value(node, value),
        "emit_round",
    );
    result
}
