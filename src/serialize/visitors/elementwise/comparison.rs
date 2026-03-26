use super::*;

pub(in crate::serialize::visitors) fn emit_equal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::Equal,
        Attribute::Equal(EqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_greater(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_greater_equal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_pow(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(graph, node, store, Op::Pow, Attribute::Pow(PowAttribute))
}

pub(in crate::serialize::visitors) fn emit_maximum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::Maximum,
        Attribute::Maximum(MaximumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        }),
    )
}

pub(in crate::serialize::visitors) fn emit_minimum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_direct_binary(
        graph,
        node,
        store,
        Op::Minimum,
        Attribute::Minimum(MinimumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        }),
    )
}

pub(in crate::serialize::visitors) fn emit_aten_eq(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::Equal,
        Attribute::Equal(EqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_gt(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_ge(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        false,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_lt(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::Greater,
        Attribute::Greater(GreaterAttribute),
        true,
    )
}

pub(in crate::serialize::visitors) fn emit_aten_le(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_bool_binary(
        graph,
        node,
        store,
        Op::GreaterEqual,
        Attribute::GreaterEqual(GreaterEqualAttribute),
        true,
    )
}
