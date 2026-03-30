use super::*;

fn is_bool_output(node: &Node) -> bool {
    node_output_dtype(node) == DType::Bool
}

pub(in crate::serialize::visitors) fn emit_bitwise_not(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return emit_direct_unary(
            ctx.graph,
            ctx.node,
            Op::LogicalNot,
            Attribute::LogicalNot(LogicalNotAttribute),
        );
    }
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::BitwiseNot,
        Attribute::BitwiseNot(BitwiseNotAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_not(ctx: &EmitContext) -> EmitResult {
    emit_direct_unary(
        ctx.graph,
        ctx.node,
        Op::LogicalNot,
        Attribute::LogicalNot(LogicalNotAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_and(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return emit_bool_binary(
            ctx.graph,
            ctx.node,
            ctx.store,
            Op::LogicalAnd,
            Attribute::LogicalAnd(LogicalAndAttribute),
            false,
        );
    }
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::BitwiseAnd,
        Attribute::BitwiseAnd(BitwiseAndAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_or(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return emit_bool_binary(
            ctx.graph,
            ctx.node,
            ctx.store,
            Op::LogicalOr,
            Attribute::LogicalOr(LogicalOrAttribute),
            false,
        );
    }
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::BitwiseOr,
        Attribute::BitwiseOr(BitwiseOrAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_xor(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return emit_bool_binary(
            ctx.graph,
            ctx.node,
            ctx.store,
            Op::LogicalXor,
            Attribute::LogicalXor(LogicalXorAttribute),
            false,
        );
    }
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::BitwiseXor,
        Attribute::BitwiseXor(BitwiseXorAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_left_shift(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::LogicalLeftShift,
        Attribute::LogicalLeftShift(LogicalLeftShiftAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_arithmetic_right_shift(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::ArithmeticRightShift,
        Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_and(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::LogicalAnd,
        Attribute::LogicalAnd(LogicalAndAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_or(ctx: &EmitContext) -> EmitResult {
    emit_direct_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::LogicalOr,
        Attribute::LogicalOr(LogicalOrAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_xor(ctx: &EmitContext) -> EmitResult {
    emit_bool_binary(
        ctx.graph,
        ctx.node,
        ctx.store,
        Op::LogicalXor,
        Attribute::LogicalXor(LogicalXorAttribute),
        false,
    )
}
