//! Pure Rust IrPass: var.correction / var.dim decomposition.

use crate::fx_ir::{
    i64_list_arg as i64_list, normalized_reduction_dims, reduced_shape, IrArg, IrGraph, IrPass,
    NodeId,
};
use crate::fx_target::{is_var_dim_target, is_var_target};
use crate::runner_logging::log_line;

pub struct IrDecomposeVarPass;

impl IrPass for IrDecomposeVarPass {
    fn name(&self) -> &str {
        "IrDecomposeVar"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            is_var_target(&graph.nodes[id].target) && decompose_var(graph, id)
        });

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
    let input_shape = match graph.node_shape_vec(x_id) {
        Some(shape) => shape,
        None => return false,
    };
    let target = graph.nodes[id].target.clone();
    let name = graph.nodes[id].name.clone();
    let base_meta = graph.nodes[id].meta.clone();
    let x_meta = graph.nodes[x_id].meta.clone();
    let meta = base_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);
    let dtype = meta.dtype();

    let dims = normalized_reduction_dims(graph.nodes[id].args.get(1), &input_shape, false);
    if dims.is_empty() {
        graph.replace_and_erase(id, x_id);
        return true;
    }

    let (keepdim, correction) = if is_var_dim_target(&target) {
        (
            graph.nodes[id]
                .args
                .get(3)
                .and_then(IrArg::as_bool)
                .unwrap_or(false),
            graph.nodes[id]
                .args
                .get(2)
                .and_then(IrArg::as_int)
                .unwrap_or(1),
        )
    } else {
        (
            graph.nodes[id]
                .kwargs
                .get("keepdim")
                .and_then(IrArg::as_bool)
                .unwrap_or(false),
            graph.nodes[id]
                .kwargs
                .get("correction")
                .and_then(IrArg::as_int)
                .unwrap_or(1),
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
    let mut cursor = graph.rewrite_cursor(id);

    let mean = cursor.call_function(
        format!("{}_mean", name),
        "aten.mean.dim",
        vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
        base_meta.cloned_with_val(mean_shape.clone(), dtype),
    );

    let diff = cursor.call_function(
        format!("{}_diff", name),
        "aten.sub.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(mean)],
        base_meta.cloned_with_val(input_shape.clone(), dtype),
    );

    let squared = cursor.call_function(
        format!("{}_squared", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(diff), IrArg::Node(diff)],
        base_meta.cloned_with_val(input_shape.clone(), dtype),
    );

    let sum = cursor.call_function(
        format!("{}_sum", name),
        "aten.sum.dim_IntList",
        vec![IrArg::Node(squared), i64_list(&dims), IrArg::Bool(keepdim)],
        base_meta.cloned_with_val(output_shape.clone(), dtype),
    );

    let coefficient = cursor.call_function(
        format!("{}_inv_count", name),
        "aten.full.default",
        vec![i64_list(&coefficient_shape), IrArg::Float(1.0 / denom)],
        base_meta.cloned_with_val(coefficient_shape, dtype),
    );

    let result = cursor.call_function(
        format!("{}_mul", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(sum), IrArg::Node(coefficient)],
        base_meta.cloned_with_val(output_shape, dtype),
    );
    cursor.replace_anchor_with(result);
    true
}
