use super::*;

pub(super) fn register(table: &mut std::collections::HashMap<crate::graph::ir::OpTarget, crate::serialize::visitors::emitter::EmitFn>) {
    use crate::graph::ir::OpTarget;
    // TOSA-lowered comparison ops
    table.insert(OpTarget::TosaEqual, emit_equal);
    table.insert(OpTarget::TosaGreater, emit_greater);
    table.insert(OpTarget::TosaGreaterEqual, emit_greater_equal);
    table.insert(OpTarget::TosaMaximum, emit_maximum);
    table.insert(OpTarget::TosaMinimum, emit_minimum);
    // ATen comparison fallbacks
    table.insert(OpTarget::Gt, emit_aten_gt);
    table.insert(OpTarget::Ge, emit_aten_ge);
    table.insert(OpTarget::Lt, emit_aten_lt);
    table.insert(OpTarget::Le, emit_aten_le);
    table.insert(OpTarget::Eq, emit_aten_eq);
    table.insert(OpTarget::Pow, emit_pow);
}

pub(in crate::serialize::visitors) fn emit_equal(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Equal,
        Attribute::Equal(EqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_greater(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_greater_equal(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_pow(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(ctx.graph, ctx.node, ctx.store, Op::Pow, Attribute::Pow(PowAttribute))
}

pub(in crate::serialize::visitors) fn emit_maximum(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Maximum,
        Attribute::Maximum(MaximumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        }),
    )
}

pub(in crate::serialize::visitors) fn emit_minimum(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Minimum,
        Attribute::Minimum(MinimumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        }),
    )
}

pub(in crate::serialize::visitors) fn emit_aten_eq(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Equal,
        Attribute::Equal(EqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_gt(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_ge(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_lt(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        true,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_le(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        true,
    )
}
