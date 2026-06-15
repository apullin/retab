use crate::fx_ir::{
    IrArg, IrGraph, IrNode, NodeId, NodeMeta, NodeOp, NodeTensor, PlaceholderTensorBinding,
    TensorSource,
};
use crate::graph::ir::DType;
use crate::lowering_effects::LoweringInputKind;
use crate::store::TensorData;

pub(crate) fn meta(shape: impl Into<Vec<i64>>, dtype: &str) -> NodeMeta {
    NodeMeta::tensor(shape, dtype)
}

pub(crate) fn f32_meta(shape: impl Into<Vec<i64>>) -> NodeMeta {
    meta(shape, "torch.float32")
}

pub(crate) fn placeholder(
    graph: &mut IrGraph,
    name: impl Into<String>,
    shape: impl Into<Vec<i64>>,
    dtype: &str,
) -> NodeId {
    let name = name.into();
    graph.add_node(
        NodeOp::Placeholder,
        name.clone(),
        name,
        Vec::new(),
        meta(shape, dtype),
    )
}

pub(crate) fn f32_placeholder(
    graph: &mut IrGraph,
    name: &str,
    shape: impl Into<Vec<i64>>,
) -> NodeId {
    placeholder(graph, name, shape, "torch.float32")
}

pub(crate) fn placeholder_default(graph: &mut IrGraph, name: &str) -> NodeId {
    graph.add_node(
        NodeOp::Placeholder,
        name.to_string(),
        name.to_string(),
        Vec::new(),
        NodeMeta::default(),
    )
}

pub(crate) fn call_with_meta(
    graph: &mut IrGraph,
    name: &str,
    target: &str,
    args: Vec<IrArg>,
    meta: NodeMeta,
) -> NodeId {
    graph.add_node(
        NodeOp::CallFunction,
        name.to_string(),
        target.to_string(),
        args,
        meta,
    )
}

pub(crate) fn call(
    graph: &mut IrGraph,
    name: &str,
    target: &str,
    args: Vec<IrArg>,
    shape: impl Into<Vec<i64>>,
    dtype: &str,
) -> NodeId {
    call_with_meta(graph, name, target, args, meta(shape, dtype))
}

pub(crate) fn f32_call(
    graph: &mut IrGraph,
    name: &str,
    target: &str,
    args: Vec<IrArg>,
    shape: impl Into<Vec<i64>>,
) -> NodeId {
    call(graph, name, target, args, shape, "torch.float32")
}

pub(crate) fn call_default(
    graph: &mut IrGraph,
    name: &str,
    target: &str,
    args: Vec<IrArg>,
) -> NodeId {
    call_with_meta(graph, name, target, args, NodeMeta::default())
}

pub(crate) fn output(graph: &mut IrGraph, input: NodeId) -> NodeId {
    graph.add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        vec![IrArg::Node(input)],
        NodeMeta::default(),
    )
}

pub(crate) fn live_node<'a>(graph: &'a IrGraph, name: &str) -> &'a IrNode {
    graph
        .nodes
        .iter()
        .find(|node| node.alive && node.name == name)
        .unwrap_or_else(|| panic!("missing live node {name}"))
}

pub(crate) fn has_live_target(graph: &IrGraph, name: &str, target: &str) -> bool {
    graph
        .nodes
        .iter()
        .any(|node| node.alive && node.name == name && node.target == target)
}

pub(crate) fn constant_placeholder(
    graph: &mut IrGraph,
    name: &str,
    data: TensorData,
    input_kind: LoweringInputKind,
) -> NodeId {
    let node_id = graph.add_node(
        NodeOp::Placeholder,
        name.to_string(),
        name.to_string(),
        Vec::new(),
        NodeMeta::default(),
    );
    graph.node_mut(node_id).tensor = Some(NodeTensor {
        source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
            input_kind,
            target_name: format!("state.{name}"),
        }),
        data,
    });
    node_id
}

pub(crate) fn parameter(graph: &mut IrGraph, name: &str, data: TensorData) -> NodeId {
    constant_placeholder(graph, name, data, LoweringInputKind::Parameter)
}

pub(crate) fn buffer(graph: &mut IrGraph, name: &str, data: TensorData) -> NodeId {
    constant_placeholder(graph, name, data, LoweringInputKind::Buffer)
}

pub(crate) fn f32_tensor(shape: Vec<i64>, values: &[f32]) -> TensorData {
    TensorData {
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        shape,
        dtype: DType::Float32,
    }
}
