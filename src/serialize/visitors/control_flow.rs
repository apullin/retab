use super::*;
use super::emitter::EmitContext;
use crate::graph::ir::Arg;
use crate::serialize::tosa_generated::CondIfAttribute;

pub fn register(table: &mut std::collections::HashMap<crate::graph::ir::OpTarget, super::emitter::EmitFn>) {
    use crate::graph::ir::OpTarget;
    table.insert(OpTarget::CondIf, emit_cond_if);
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

pub(super) fn emit_cond_if(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let spec = CondIfSpec::parse(graph, node);
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
