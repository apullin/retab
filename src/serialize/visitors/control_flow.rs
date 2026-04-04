use super::emitter::EmitContext;
use super::*;
use crate::graph::ir::Arg;
use crate::serialize::tosa_generated::{CondIfAttribute, WhileLoopAttribute};

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::CondIf, emit_cond_if as EmitFn),
        (OpTarget::WhileLoop, emit_while_loop as EmitFn),
    ]);
}

// Parse the raw control-flow node once at the boundary, then emit from a
// typed shape rather than repeatedly indexing loosely-structured args.
#[derive(Debug)]
struct CondIfSpec {
    input_names: Vec<String>,
    output_tensors: Vec<TosaTensor>,
    then_graph: String,
    else_graph: String,
}

#[derive(Debug)]
struct WhileLoopSpec {
    input_names: Vec<String>,
    output_tensors: Vec<TosaTensor>,
    cond_graph: String,
    body_graph: String,
}

impl CondIfSpec {
    fn parse(graph: &Graph, node: &Node) -> Self {
        Self {
            input_names: collect_input_names(graph, node),
            output_tensors: cond_output_tensors(graph, node),
            then_graph: resolve_region_name(graph, node.args.get(1)),
            else_graph: resolve_region_name(graph, node.args.get(2)),
        }
    }
}

impl WhileLoopSpec {
    fn parse(ctx: &EmitContext) -> Self {
        let graph = ctx.graph;
        let node = ctx.node;
        let carried_input_ids = carried_input_ids(node);
        let input_names = carried_input_ids
            .iter()
            .filter_map(|id| resolve_input_name_from_node_id(graph, *id))
            .collect();
        let mut output_tensors = cond_output_tensors(graph, node);
        extend_missing_while_outputs(
            &mut output_tensors,
            graph,
            ctx.store,
            node,
            &carried_input_ids,
        );

        Self {
            input_names,
            output_tensors,
            cond_graph: resolve_region_name(graph, node.args.first()),
            body_graph: resolve_region_name(graph, node.args.get(1)),
        }
    }
}

fn resolve_region_name(graph: &Graph, arg: Option<&Arg>) -> String {
    match arg {
        Some(Arg::String(name)) => name.clone(),
        Some(Arg::Node(id)) => graph.node(*id).map(|n| n.name.clone()).unwrap_or_default(),
        _ => String::new(),
    }
}

fn cond_output_tensors(graph: &Graph, node: &Node) -> Vec<TosaTensor> {
    let mut outputs: Vec<(i64, TosaTensor)> = Vec::new();
    for user_id in &node.users {
        let Some(user) = graph.node(*user_id) else {
            continue;
        };
        if !is_getitem_node(user) {
            continue;
        }
        let output_idx = match user.args.get(1) {
            Some(Arg::Int(idx)) => *idx,
            _ => 0,
        };
        outputs.push((output_idx, make_output_tensor(user)));
    }

    outputs.sort_by_key(|(idx, _)| *idx);
    if outputs.is_empty() {
        vec![make_output_tensor(node)]
    } else {
        outputs.into_iter().map(|(_, tensor)| tensor).collect()
    }
}

fn carried_input_ids(node: &Node) -> Vec<NodeId> {
    match node.args.get(2) {
        Some(Arg::NodeList(ids)) => ids.iter().copied().collect(),
        Some(Arg::Node(id)) => vec![*id],
        _ => Vec::new(),
    }
}

fn extend_missing_while_outputs(
    output_tensors: &mut Vec<TosaTensor>,
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    carried_input_ids: &[NodeId],
) {
    if output_tensors.len() >= carried_input_ids.len() {
        return;
    }

    for (idx, input_id) in carried_input_ids.iter().enumerate().skip(output_tensors.len()) {
        let Some(input_node) = graph.node(*input_id) else {
            continue;
        };
        let name = format!("{}_dummy_{}", node.name, idx);
        output_tensors.push(TosaTensor::new(
            &name,
            actual_tensor_shape(graph, store, input_node),
            actual_tensor_dtype(graph, input_node),
        ));
    }
}

pub(super) fn emit_cond_if(ctx: &EmitContext) -> EmitResult {
    let spec = CondIfSpec::parse(ctx.graph, ctx.node);
    let output_names = spec
        .output_tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();

    let mut result = EmitResult::default();
    result.tensors.extend(spec.output_tensors);
    result.operators.push(TosaOperator::new(
        Op::CondIf,
        Attribute::CondIf(CondIfAttribute {
            then_graph: spec.then_graph,
            else_graph: spec.else_graph,
        }),
        spec.input_names,
        output_names,
    ));
    result
}

pub(super) fn emit_while_loop(ctx: &EmitContext) -> EmitResult {
    let spec = WhileLoopSpec::parse(ctx);
    let output_names = spec
        .output_tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();

    let mut result = EmitResult::default();
    result.tensors.extend(spec.output_tensors);
    result.operators.push(TosaOperator::new(
        Op::WhileLoop,
        Attribute::WhileLoop(WhileLoopAttribute {
            cond_branch: spec.cond_graph,
            body_branch: spec.body_graph,
        }),
        spec.input_names,
        output_names,
    ));
    result
}
