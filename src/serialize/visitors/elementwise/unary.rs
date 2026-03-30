use super::*;

pub(super) fn register(
    table: &mut std::collections::HashMap<
        crate::graph::ir::OpTarget,
        crate::serialize::visitors::emitter::EmitFn,
    >,
) {
    use crate::graph::ir::OpTarget;
    // TOSA-lowered unary ops
    table.insert(OpTarget::TosaTable, emit_table);
    table.insert(OpTarget::TosaSigmoid, emit_sigmoid);
    table.insert(OpTarget::TosaTanh, emit_tanh);
    table.insert(OpTarget::TosaExp, emit_exp);
    table.insert(OpTarget::TosaLog, emit_log);
    table.insert(OpTarget::TosaAbs, emit_abs);
    table.insert(OpTarget::TosaReciprocal, emit_reciprocal);
    table.insert(OpTarget::TosaErf, emit_erf);
    table.insert(OpTarget::TosaCos, emit_cos);
    table.insert(OpTarget::TosaSin, emit_sin);
    table.insert(OpTarget::TosaRsqrt, emit_rsqrt);
    table.insert(OpTarget::TosaCast, emit_cast);
    // ATen fallbacks
    table.insert(OpTarget::Sigmoid, emit_sigmoid);
    table.insert(OpTarget::Tanh, emit_tanh);
    table.insert(OpTarget::Exp, emit_exp);
    table.insert(OpTarget::Log, emit_log);
    table.insert(OpTarget::Abs, emit_abs);
    table.insert(OpTarget::Reciprocal, emit_reciprocal);
    table.insert(OpTarget::Rsqrt, emit_rsqrt);
    table.insert(OpTarget::Sin, emit_sin);
    table.insert(OpTarget::Cos, emit_cos);
    table.insert(OpTarget::To, emit_cast);
}

pub(in crate::serialize::visitors) fn emit_table(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Table,
        Attribute::Table(TableAttribute),
    )
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
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Tanh,
        Attribute::Tanh(TanhAttribute),
    )
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
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Rsqrt,
        Attribute::Rsqrt(RsqrtAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_cast(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::Cast,
        Attribute::Cast(CastAttribute),
    )
}
