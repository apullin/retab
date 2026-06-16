use crate::fx_ir::{
    IrArg, IrGraph, IrNode, IrPass, NodeId, NodeMeta, NodeOp, NodeTensor, PlaceholderTensorBinding,
    QParam, QParamMap, TensorSource,
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

/// Small Rust-only fixture builder for pass/golden tests.
///
/// Keep this module focused on IR construction and assertions. It deliberately
/// avoids Python, ExportedProgram, and flatbuffer serialization so pass contract
/// tests stay cheap and easy to debug.
pub(crate) struct IrFixture {
    graph: IrGraph,
}

impl IrFixture {
    pub(crate) fn new() -> Self {
        Self {
            graph: IrGraph::empty(),
        }
    }

    pub(crate) fn placeholder(
        &mut self,
        name: &str,
        shape: impl Into<Vec<i64>>,
        dtype: &str,
    ) -> NodeId {
        placeholder(&mut self.graph, name, shape, dtype)
    }

    pub(crate) fn f32_placeholder(&mut self, name: &str, shape: impl Into<Vec<i64>>) -> NodeId {
        f32_placeholder(&mut self.graph, name, shape)
    }

    pub(crate) fn call(
        &mut self,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: impl Into<Vec<i64>>,
        dtype: &str,
    ) -> NodeId {
        call(&mut self.graph, name, target, args, shape, dtype)
    }

    pub(crate) fn f32_call(
        &mut self,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: impl Into<Vec<i64>>,
    ) -> NodeId {
        f32_call(&mut self.graph, name, target, args, shape)
    }

    pub(crate) fn output(&mut self, input: NodeId) -> NodeId {
        output(&mut self.graph, input)
    }

    pub(crate) fn run<P: IrPass>(&mut self, pass: &P) -> bool {
        pass.run(&mut self.graph)
    }

    pub(crate) fn graph(&self) -> &IrGraph {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut IrGraph {
        &mut self.graph
    }
}

pub(crate) fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
    QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
}

pub(crate) fn assert_live_signature(graph: &IrGraph, expected: &[&str]) {
    let actual = live_signature(graph);
    let expected: Vec<String> = expected.iter().map(|line| (*line).to_string()).collect();
    assert_eq!(actual, expected);
}

pub(crate) fn live_signature(graph: &IrGraph) -> Vec<String> {
    topological_live_node_ids(graph)
        .into_iter()
        .map(|id| graph.node(id))
        .map(|node| {
            format!(
                "{}|{:?}|{}|args=[{}]|shape={}|dtype={}|iqp={}|oqp={}",
                node.name,
                node.op,
                node.target,
                node.args
                    .iter()
                    .map(|arg| arg_signature(graph, arg))
                    .collect::<Vec<_>>()
                    .join(","),
                shape_signature(&node.meta),
                node.meta.val_dtype().unwrap_or("-"),
                qparam_map_signature(node.meta.input_qparams()),
                qparam_map_signature(node.meta.output_qparams()),
            )
        })
        .collect()
}

fn topological_live_node_ids(graph: &IrGraph) -> Vec<NodeId> {
    let mut visited = std::collections::HashSet::new();
    let mut result = Vec::new();
    for node in graph.live_nodes() {
        visit_live_node(graph, node.id, &mut visited, &mut result);
    }
    result
}

fn visit_live_node(
    graph: &IrGraph,
    id: NodeId,
    visited: &mut std::collections::HashSet<NodeId>,
    result: &mut Vec<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }
    for arg in &graph.node(id).args {
        visit_arg_deps(graph, arg, visited, result);
    }
    for arg in graph.node(id).kwargs.values() {
        visit_arg_deps(graph, arg, visited, result);
    }
    if graph.node(id).alive {
        result.push(id);
    }
}

fn visit_arg_deps(
    graph: &IrGraph,
    arg: &IrArg,
    visited: &mut std::collections::HashSet<NodeId>,
    result: &mut Vec<NodeId>,
) {
    match arg {
        IrArg::Node(id) if graph.node(*id).alive => visit_live_node(graph, *id, visited, result),
        IrArg::List(items) => {
            for item in items {
                visit_arg_deps(graph, item, visited, result);
            }
        }
        _ => {}
    }
}

fn arg_signature(graph: &IrGraph, arg: &IrArg) -> String {
    match arg {
        IrArg::Node(id) => format!("%{}", graph.node(*id).name),
        IrArg::Int(value) => value.to_string(),
        IrArg::Float(value) => float_signature(*value),
        IrArg::Bool(value) => value.to_string(),
        IrArg::Str(value) => format!("{value:?}"),
        IrArg::Dtype(value) => format!("dtype({value})"),
        IrArg::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| arg_signature(graph, item))
                .collect::<Vec<_>>()
                .join(",")
        ),
        IrArg::None => "None".to_string(),
    }
}

fn float_signature(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn shape_signature(meta: &NodeMeta) -> String {
    meta.val_shape()
        .map(|shape| {
            format!(
                "[{}]",
                shape
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .unwrap_or_else(|| "-".to_string())
}

fn qparam_map_signature(qparams: Option<&QParamMap>) -> String {
    let Some(qparams) = qparams else {
        return "-".to_string();
    };
    if qparams.is_empty() {
        return "{}".to_string();
    }
    format!(
        "{{{}}}",
        qparams
            .iter()
            .map(|(index, qp)| format!("{index}:{}@{}:{}", qp.dtype_str, qp.scale, qp.zp))
            .collect::<Vec<_>>()
            .join(",")
    )
}
