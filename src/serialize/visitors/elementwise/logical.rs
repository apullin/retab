use super::*;

fn is_bool_output(node: &Node) -> bool {
    node_output_dtype(node) == DType::Bool
}

pub(in crate::serialize::visitors) fn emit_bitwise_not(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    if is_bool_output(node) {
        return emit_direct_unary(
            graph,
            node,
            Op::LogicalNot,
            Attribute::LogicalNot(LogicalNotAttribute),
        );
    }
    emit_direct_unary(
        graph,
        node,
        Op::BitwiseNot,
        Attribute::BitwiseNot(BitwiseNotAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_not(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(
        graph,
        node,
        Op::LogicalNot,
        Attribute::LogicalNot(LogicalNotAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_and(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    if is_bool_output(node) {
        return emit_bool_binary(
            graph,
            node,
            store,
            Op::LogicalAnd,
            Attribute::LogicalAnd(LogicalAndAttribute),
            false,
        );
    }
    emit_direct_binary(
        graph,
        node,
        store,
        Op::BitwiseAnd,
        Attribute::BitwiseAnd(BitwiseAndAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_or(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    if is_bool_output(node) {
        return emit_bool_binary(
            graph,
            node,
            store,
            Op::LogicalOr,
            Attribute::LogicalOr(LogicalOrAttribute),
            false,
        );
    }
    emit_direct_binary(
        graph,
        node,
        store,
        Op::BitwiseOr,
        Attribute::BitwiseOr(BitwiseOrAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_bitwise_xor(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    if is_bool_output(node) {
        return emit_bool_binary(
            graph,
            node,
            store,
            Op::LogicalXor,
            Attribute::LogicalXor(LogicalXorAttribute),
            false,
        );
    }
    emit_direct_binary(
        graph,
        node,
        store,
        Op::BitwiseXor,
        Attribute::BitwiseXor(BitwiseXorAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_left_shift(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::LogicalLeftShift,
        Attribute::LogicalLeftShift(LogicalLeftShiftAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_arithmetic_right_shift(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::ArithmeticRightShift,
        Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_and(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::LogicalAnd,
        Attribute::LogicalAnd(LogicalAndAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_or(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::LogicalOr,
        Attribute::LogicalOr(LogicalOrAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_logical_xor(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::LogicalXor,
        Attribute::LogicalXor(LogicalXorAttribute),
        false,
    )
}
