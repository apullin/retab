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
        let additional_input_ids = additional_input_ids(node);
        let loop_input_ids: Vec<NodeId> = carried_input_ids
            .iter()
            .chain(additional_input_ids.iter())
            .copied()
            .collect();
        let input_names = loop_input_ids
            .iter()
            .filter_map(|id| resolve_input_name_from_node_id(graph, *id))
            .collect();
        let output_tensors = while_output_tensors(graph, ctx.store, node, &loop_input_ids);

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
    let outputs = indexed_output_tensors(graph, node);
    if outputs.is_empty() {
        vec![make_output_tensor(node)]
    } else {
        outputs.into_iter().map(|(_, tensor)| tensor).collect()
    }
}

fn indexed_output_tensors(graph: &Graph, node: &Node) -> Vec<(usize, TosaTensor)> {
    let mut outputs: Vec<(usize, TosaTensor)> = Vec::new();
    for user_id in &node.users {
        let Some(user) = graph.node(*user_id) else {
            continue;
        };
        if !is_getitem_node(user) {
            continue;
        }
        let output_idx = match user.args.get(1) {
            Some(Arg::Int(idx)) if *idx >= 0 => *idx as usize,
            _ => 0,
        };
        outputs.push((output_idx, make_output_tensor(user)));
    }

    outputs.sort_by_key(|(idx, _)| *idx);
    outputs
}

fn carried_input_ids(node: &Node) -> Vec<NodeId> {
    match node.args.get(2) {
        Some(Arg::NodeList(ids)) => ids.iter().copied().collect(),
        Some(Arg::Node(id)) => vec![*id],
        _ => Vec::new(),
    }
}

fn additional_input_ids(node: &Node) -> Vec<NodeId> {
    match node.args.get(3) {
        Some(Arg::NodeList(ids)) => ids.iter().copied().collect(),
        Some(Arg::Node(id)) => vec![*id],
        _ => Vec::new(),
    }
}

fn while_output_tensors(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    carried_input_ids: &[NodeId],
) -> Vec<TosaTensor> {
    let indexed_outputs = indexed_output_tensors(graph, node);
    let slot_count = carried_input_ids.len().max(
        indexed_outputs
            .iter()
            .map(|(idx, _)| idx + 1)
            .max()
            .unwrap_or(0),
    );
    if slot_count == 0 {
        return vec![make_output_tensor(node)];
    }

    let mut slots: Vec<Option<TosaTensor>> =
        std::iter::repeat_with(|| None).take(slot_count).collect();
    for (idx, tensor) in indexed_outputs {
        if idx < slots.len() {
            slots[idx] = Some(tensor);
        }
    }

    slots
        .into_iter()
        .enumerate()
        .map(|(idx, tensor)| {
            if let Some(tensor) = tensor {
                return tensor;
            }
            let Some(input_node) = carried_input_ids
                .get(idx)
                .and_then(|input_id| graph.node(*input_id))
            else {
                return make_output_tensor(node);
            };
            let name = format!("{}_dummy_{}", node.name, idx);
            TosaTensor::new(
                &name,
                actual_tensor_shape(graph, store, input_node),
                actual_tensor_dtype(graph, input_node),
            )
        })
        .collect()
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
