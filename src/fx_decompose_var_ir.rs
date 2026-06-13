//! Pure Rust IrPass: var.correction / var.dim decomposition.

use std::collections::HashMap;

use crate::fx_ir::{
    i64_list_arg as i64_list, normalized_reduction_dims, reduced_shape, IrArg, IrGraph, IrPass,
    NodeId, NodeOp,
};
use crate::fx_target::{is_var_dim_target, is_var_target};
use crate::runner_logging::log_line;

fn bool_arg(arg: Option<&IrArg>, default: bool) -> bool {
    arg.and_then(IrArg::as_bool).unwrap_or(default)
}

fn int_arg(arg: Option<&IrArg>, default: i64) -> i64 {
    arg.and_then(IrArg::as_int).unwrap_or(default)
}

fn kw_bool(kwargs: &HashMap<String, IrArg>, key: &str, default: bool) -> bool {
    kwargs.get(key).and_then(IrArg::as_bool).unwrap_or(default)
}

fn kw_int(kwargs: &HashMap<String, IrArg>, key: &str, default: i64) -> i64 {
    kwargs.get(key).and_then(IrArg::as_int).unwrap_or(default)
}

pub struct IrDecomposeVarPass;

impl IrPass for IrDecomposeVarPass {
    fn name(&self) -> &str {
        "IrDecomposeVar"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive
                || graph.nodes[id].op != NodeOp::CallFunction
                || !is_var_target(&graph.nodes[id].target)
            {
                continue;
            }
            if decompose_var(graph, id) {
                modified = true;
            }
        }

        if modified {
            log_line(format_args!("[IrDecomposeVar] decomposed var ops"));
        }
        modified
    }
}

fn decompose_var(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(node_id) => node_id,
        None => return false,
    };
    let input_shape = match graph.nodes[x_id].meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let target = graph.nodes[id].target.clone();
    let name = graph.nodes[id].name.clone();
    let base_meta = graph.nodes[id].meta.clone();

    let dims = normalized_reduction_dims(graph.nodes[id].args.get(1), &input_shape, false);
    if dims.is_empty() {
        graph.replace_and_erase(id, x_id);
        return true;
    }

    let (keepdim, correction) = if is_var_dim_target(&target) {
        (
            bool_arg(graph.nodes[id].args.get(3), false),
            int_arg(graph.nodes[id].args.get(2), 1),
        )
    } else {
        (
            kw_bool(&graph.nodes[id].kwargs, "keepdim", false),
            kw_int(&graph.nodes[id].kwargs, "correction", 1),
        )
    };

    let mean_shape = reduced_shape(&input_shape, &dims, true);
    let output_shape = graph.nodes[id]
        .meta
        .val_shape()
        .map(|shape| shape.to_vec())
        .unwrap_or_else(|| reduced_shape(&input_shape, &dims, keepdim));
    let coefficient_shape = vec![1; output_shape.len().max(1)];
    let count: i64 = dims
        .iter()
        .map(|&dim| input_shape[dim as usize].max(1))
        .product();
    let denom = (count - correction).max(1) as f64;

    let mean = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mean", name),
        "aten.mean.dim".to_string(),
        vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
        base_meta.cloned_with_val(mean_shape.clone(), &dtype),
    );
    graph.anchor_to(mean, id);

    let diff = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_diff", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(mean)],
        base_meta.cloned_with_val(input_shape.clone(), &dtype),
    );
    graph.anchor_to(diff, id);

    let squared = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_squared", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(diff), IrArg::Node(diff)],
        base_meta.cloned_with_val(input_shape.clone(), &dtype),
    );
    graph.anchor_to(squared, id);

    let sum = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.sum.dim_IntList".to_string(),
        vec![IrArg::Node(squared), i64_list(&dims), IrArg::Bool(keepdim)],
        base_meta.cloned_with_val(output_shape.clone(), &dtype),
    );
    graph.anchor_to(sum, id);

    let coefficient = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv_count", name),
        "aten.full.default".to_string(),
        vec![i64_list(&coefficient_shape), IrArg::Float(1.0 / denom)],
        base_meta.cloned_with_val(coefficient_shape, &dtype),
    );
    graph.anchor_to(coefficient, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mul", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(sum), IrArg::Node(coefficient)],
        base_meta.cloned_with_val(output_shape, &dtype),
    );
    graph.anchor_to(result, id);

    graph.replace_and_erase(id, result);
    true
}
