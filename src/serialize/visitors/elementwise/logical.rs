use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;

    // Custom emitters with bool-dispatch logic
    table.extend([
        (OpTarget::BitwiseNot, emit_bitwise_not as EmitFn),
        (OpTarget::BitwiseAnd, emit_bitwise_and),
        (OpTarget::BitwiseOr, emit_bitwise_or),
        (OpTarget::BitwiseXor, emit_bitwise_xor),
    ]);

    // Direct unary ops
    register_unary!(table;
        OpTarget::LogicalNot => Op::LogicalNot, Attribute::LogicalNot(LogicalNotAttribute {});
    );

    // Direct binary ops
    register_binary!(table;
        OpTarget::BitwiseLeftShift  => Op::LogicalLeftShift,      Attribute::LogicalLeftShift(LogicalLeftShiftAttribute {});
        OpTarget::BitwiseRightShift => Op::ArithmeticRightShift,  Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute {});
        OpTarget::LogicalAnd        => Op::LogicalAnd,            Attribute::LogicalAnd(LogicalAndAttribute {});
        OpTarget::LogicalOr         => Op::LogicalOr,             Attribute::LogicalOr(LogicalOrAttribute {});
    );

    // Bool-output binary ops
    register_bool_binary!(table;
        OpTarget::LogicalXor => Op::LogicalXor, Attribute::LogicalXor(LogicalXorAttribute {}), false;
    );
}

fn is_bool_output(node: &Node) -> bool {
    node_output_dtype(node) == DType::Bool
}

fn emit_bitwise_not(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_unary(
            Op::LogicalNot,
            Attribute::LogicalNot(LogicalNotAttribute {}),
        );
    }
    ctx.emit_unary(
        Op::BitwiseNot,
        Attribute::BitwiseNot(BitwiseNotAttribute {}),
    )
}

fn emit_bitwise_and(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(
            Op::LogicalAnd,
            Attribute::LogicalAnd(LogicalAndAttribute {}),
            false,
        );
    }
    ctx.emit_binary(
        Op::BitwiseAnd,
        Attribute::BitwiseAnd(BitwiseAndAttribute {}),
    )
}

fn emit_bitwise_or(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(
            Op::LogicalOr,
            Attribute::LogicalOr(LogicalOrAttribute {}),
            false,
        );
    }
    ctx.emit_binary(Op::BitwiseOr, Attribute::BitwiseOr(BitwiseOrAttribute {}))
}

fn emit_bitwise_xor(ctx: &EmitContext) -> EmitResult {
    if is_bool_output(ctx.node) {
        return ctx.emit_bool_binary(
            Op::LogicalXor,
            Attribute::LogicalXor(LogicalXorAttribute {}),
            false,
        );
    }
    ctx.emit_binary(
        Op::BitwiseXor,
        Attribute::BitwiseXor(BitwiseXorAttribute {}),
    )
}
