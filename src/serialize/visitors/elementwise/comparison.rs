use super::*;

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
