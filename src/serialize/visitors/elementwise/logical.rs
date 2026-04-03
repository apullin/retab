use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::BitwiseNot, emit_bitwise_not as EmitFn),
        (OpTarget::BitwiseAnd, emit_bitwise_and),
        (OpTarget::BitwiseOr, emit_bitwise_or),
        (OpTarget::BitwiseXor, emit_bitwise_xor),
        (OpTarget::BitwiseLeftShift, emit_logical_left_shift),
        (OpTarget::BitwiseRightShift, emit_arithmetic_right_shift),
        (OpTarget::LogicalNot, emit_logical_not),
        (OpTarget::LogicalAnd, emit_logical_and),
        (OpTarget::LogicalOr, emit_logical_or),
        (OpTarget::LogicalXor, emit_logical_xor),
    ]);
}

fn is_bool_output(node: &Node) -> bool {
    node_output_dtype(node) == DType::Bool
}

pub(in crate::serialize::visitors) fn emit_bitwise_not(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_unary(Op::LogicalNot, Attribute::LogicalNot(LogicalNotAttribute));
    }
    ctx.emit_unary(Op::BitwiseNot, Attribute::BitwiseNot(BitwiseNotAttribute))
}

direct_unary_emitters! {
    pub(in crate::serialize::visitors) fn emit_logical_not => (
        Op::LogicalNot,
        Attribute::LogicalNot(LogicalNotAttribute)
    );
}

pub(in crate::serialize::visitors) fn emit_bitwise_and(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(Op::LogicalAnd, Attribute::LogicalAnd(LogicalAndAttribute), false);
    }
    ctx.emit_binary(Op::BitwiseAnd, Attribute::BitwiseAnd(BitwiseAndAttribute))
}

pub(in crate::serialize::visitors) fn emit_bitwise_or(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(Op::LogicalOr, Attribute::LogicalOr(LogicalOrAttribute), false);
    }
    ctx.emit_binary(Op::BitwiseOr, Attribute::BitwiseOr(BitwiseOrAttribute))
}

pub(in crate::serialize::visitors) fn emit_bitwise_xor(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(Op::LogicalXor, Attribute::LogicalXor(LogicalXorAttribute), false);
    }
    ctx.emit_binary(Op::BitwiseXor, Attribute::BitwiseXor(BitwiseXorAttribute))
}

direct_binary_emitters! {
    pub(in crate::serialize::visitors) fn emit_logical_left_shift => (
        Op::LogicalLeftShift,
        Attribute::LogicalLeftShift(LogicalLeftShiftAttribute)
    );
    pub(in crate::serialize::visitors) fn emit_arithmetic_right_shift => (
        Op::ArithmeticRightShift,
        Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute)
    );
    pub(in crate::serialize::visitors) fn emit_logical_and => (
        Op::LogicalAnd,
        Attribute::LogicalAnd(LogicalAndAttribute)
    );
    pub(in crate::serialize::visitors) fn emit_logical_or => (
        Op::LogicalOr,
        Attribute::LogicalOr(LogicalOrAttribute)
    );
}

bool_binary_emitters! {
    pub(in crate::serialize::visitors) fn emit_logical_xor => (
        Op::LogicalXor,
        Attribute::LogicalXor(LogicalXorAttribute),
        false
    );
}
