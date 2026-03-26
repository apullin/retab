use super::*;

pub(in crate::serialize::visitors) fn emit_table(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Table, Attribute::Table(TableAttribute))
}

pub(in crate::serialize::visitors) fn emit_sigmoid(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Sigmoid, Attribute::Sigmoid(SigmoidAttribute))
}

pub(in crate::serialize::visitors) fn emit_tanh(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Tanh, Attribute::Tanh(TanhAttribute))
}

pub(in crate::serialize::visitors) fn emit_exp(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Exp, Attribute::Exp(ExpAttribute))
}

pub(in crate::serialize::visitors) fn emit_log(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Log, Attribute::Log(LogAttribute))
}

pub(in crate::serialize::visitors) fn emit_abs(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Abs, Attribute::Abs(AbsAttribute))
}

pub(in crate::serialize::visitors) fn emit_reciprocal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(
        graph,
        node,
        Op::Reciprocal,
        Attribute::Reciprocal(ReciprocalAttribute),
    )
}

pub(in crate::serialize::visitors) fn emit_erf(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Erf, Attribute::Erf(ErfAttribute))
}

pub(in crate::serialize::visitors) fn emit_cos(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Cos, Attribute::Cos(CosAttribute))
}

pub(in crate::serialize::visitors) fn emit_sin(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Sin, Attribute::Sin(SinAttribute))
}

pub(in crate::serialize::visitors) fn emit_rsqrt(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Rsqrt, Attribute::Rsqrt(RsqrtAttribute))
}

pub(in crate::serialize::visitors) fn emit_cast(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_direct_unary(graph, node, Op::Cast, Attribute::Cast(CastAttribute))
}
