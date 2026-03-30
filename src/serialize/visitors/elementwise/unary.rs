use super::*;

pub(in crate::serialize::visitors) fn emit_table(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Table, Attribute::Table(TableAttribute))
}

pub(in crate::serialize::visitors) fn emit_sigmoid(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Sigmoid,
        Attribute::Sigmoid(SigmoidAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_tanh(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Tanh, Attribute::Tanh(TanhAttribute))
}

pub(in crate::serialize::visitors) fn emit_exp(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Exp, Attribute::Exp(ExpAttribute))
}

pub(in crate::serialize::visitors) fn emit_log(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Log, Attribute::Log(LogAttribute))
}

pub(in crate::serialize::visitors) fn emit_abs(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Abs, Attribute::Abs(AbsAttribute))
}

pub(in crate::serialize::visitors) fn emit_reciprocal(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_erf(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Erf, Attribute::Erf(ErfAttribute))
}

pub(in crate::serialize::visitors) fn emit_cos(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Cos, Attribute::Cos(CosAttribute))
}

pub(in crate::serialize::visitors) fn emit_sin(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Sin, Attribute::Sin(SinAttribute))
}

pub(in crate::serialize::visitors) fn emit_rsqrt(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Rsqrt, Attribute::Rsqrt(RsqrtAttribute))
}

pub(in crate::serialize::visitors) fn emit_cast(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(ctx.graph, ctx.node, Op::Cast, Attribute::Cast(CastAttribute))
}
