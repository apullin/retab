use super::emitter::EmitContext;
use super::*;
use crate::graph::ir::Arg;

mod extrema;
mod mean;
mod sum_ops;

pub fn register(table: &mut EmitterTable) {
    for register in [
        extrema::register as fn(&mut EmitterTable),
        mean::register,
        sum_ops::register,
    ] {
        register(table);
    }
}

#[derive(Debug, Clone, Copy)]
struct ReduceAxis {
    raw_dim: i32,
    tosa_axis: i32,
}

#[derive(Debug)]
struct ParsedReduceSpec {
    input_name: String,
    input_shape: Vec<i32>,
    axes: Vec<ReduceAxis>,
    keepdim: bool,
}

#[derive(Debug)]
struct ParsedSingleAxisReduceSpec {
    input_names: Vec<String>,
    input_shape: Vec<i32>,
    raw_axis: i32,
    tosa_axis: i32,
    keepdim: bool,
    reduce_dtype: DType,
}

impl ParsedReduceSpec {
    // Parse reduction metadata once at the boundary. Downstream emitters can
    // then operate on typed axes instead of repeatedly validating raw args.
    fn parse(graph: &Graph, node: &Node, store: &TensorStore, keepdim_default: bool) -> Self {
        let input_name = collect_input_names(graph, node)
            .first()
            .cloned()
            .unwrap_or_default();
        let input_node = match node.args.first() {
            Some(Arg::Node(id)) => graph.node(*id),
            _ => None,
        };
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_default();
        let keepdim = arg_bool(node, 2, keepdim_default);
        let mut dims = parse_reduce_dims(graph, node, store, input_node);
        dims.sort_unstable();
        dims.reverse();

        let axes = dims
            .into_iter()
            .map(|raw_dim| ReduceAxis {
                raw_dim,
                tosa_axis: input_node
                    .map(|n| map_axis_to_tosa_dim_order(graph, n, raw_dim))
                    .unwrap_or(raw_dim),
            })
            .collect();

        Self {
            input_name,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl ParsedSingleAxisReduceSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_node = match node.args.first() {
            Some(Arg::Node(input_id)) => graph.node(*input_id),
            _ => None,
        };

        let raw_axis = {
            let dims = arg_i32_list(node, 1);
            if let Some(axis) = dims.first().copied() {
                axis
            } else {
                kwarg_i32(node, "axis", 0)
            }
        };
        let tosa_axis = input_node
            .map(|input_node| map_axis_to_tosa_dim_order(graph, input_node, raw_axis))
            .unwrap_or(raw_axis);
        let keepdim = arg_bool(node, 2, true);
        let input_shape = input_node
            .map(|input_node| actual_tensor_shape(graph, store, input_node))
            .unwrap_or_default();
        let input_dtype = input_node
            .map(|n| passthrough_input_dtype(graph, n))
            .unwrap_or_else(|| node_output_dtype(node));
        let reduce_dtype = match input_dtype {
            DType::Int8 | DType::Int16 => DType::Int32,
            _ => node_output_dtype(node),
        };

        Self {
            input_names,
            input_shape,
            raw_axis,
            tosa_axis,
            keepdim,
            reduce_dtype,
        }
    }
}

fn parse_reduce_dims(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    input_node: Option<&Node>,
) -> Vec<i32> {
    let mut dims = arg_i32_list(node, 1);
    if dims.is_empty() {
        if let Some(Arg::Int(dim)) = node.args.get(1) {
            dims.push(*dim as i32);
        }
    }
    if dims.is_empty() {
        dims = input_node
            .map(|n| {
                actual_tensor_shape(graph, store, n)
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| idx as i32)
                    .collect()
            })
            .unwrap_or_else(|| vec![0]);
    }
    dims
}

enum ReduceExtremaKind {
    Any,
    Max,
    Min,
}
