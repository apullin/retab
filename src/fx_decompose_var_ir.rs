//! Pure Rust IrPass: var.correction / var.dim decomposition.

use std::collections::HashMap;

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp,
};
use crate::runner_logging::log_line;

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut meta = base.clone();
    meta.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    meta
}

fn is_var(target: &str) -> bool {
    target.contains("var.correction")
        || target.contains("var_correction")
        || target.contains("var.dim")
        || target.contains("var_dim")
}

fn list_i64(arg: Option<&IrArg>) -> Option<Vec<i64>> {
    match arg {
        Some(IrArg::List(items)) => {
            let dims: Vec<i64> = items.iter().filter_map(IrArg::as_int).collect();
            Some(dims)
        }
        Some(IrArg::Int(dim)) => Some(vec![*dim]),
        Some(IrArg::None) | None => None,
        _ => None,
    }
}

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

fn normalize_dims(dims: Option<Vec<i64>>, rank: usize) -> Vec<i64> {
    let rank_i64 = rank as i64;
    let mut dims = dims.unwrap_or_else(|| (0..rank_i64).collect());
    for dim in &mut dims {
        if *dim < 0 {
            *dim += rank_i64;
        }
    }
    dims.sort_unstable();
    dims.dedup();
    dims.into_iter()
        .filter(|&dim| dim >= 0 && (dim as usize) < rank)
        .collect()
}

fn reduced_shape(input_shape: &[i64], dims: &[i64], keepdim: bool) -> Vec<i64> {
    if keepdim {
        let mut shape = input_shape.to_vec();
        for &dim in dims {
            shape[dim as usize] = 1;
        }
        return shape;
    }

    let mut shape: Vec<i64> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &extent)| {
            if dims.contains(&(idx as i64)) {
                None
            } else {
                Some(extent)
            }
        })
        .collect();
    if shape.is_empty() {
        shape.push(1);
    }
    shape
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
                || !is_var(&graph.nodes[id].target)
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
    let rank = input_shape.len();
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let target = graph.nodes[id].target.clone();
    let name = graph.nodes[id].name.clone();
    let base_meta = graph.nodes[id].meta.clone();

    let dims = normalize_dims(list_i64(graph.nodes[id].args.get(1)), rank);
    if dims.is_empty() {
        graph.replace_and_erase(id, x_id);
        return true;
    }

    let (keepdim, correction) = if target.contains("var.dim") {
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
    let coefficient_shape = vec![1; rank];
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
        meta_with(&base_meta, mean_shape.clone(), &dtype),
    );
    graph.anchor_to(mean, id);

    let diff = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_diff", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(mean)],
        meta_with(&base_meta, input_shape.clone(), &dtype),
    );
    graph.anchor_to(diff, id);

    let squared = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_squared", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(diff), IrArg::Node(diff)],
        meta_with(&base_meta, input_shape.clone(), &dtype),
    );
    graph.anchor_to(squared, id);

    let sum = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.sum.dim_IntList".to_string(),
        vec![IrArg::Node(squared), i64_list(&dims), IrArg::Bool(keepdim)],
        meta_with(&base_meta, output_shape.clone(), &dtype),
    );
    graph.anchor_to(sum, id);

    let coefficient = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv_count", name),
        "aten.full.default".to_string(),
        vec![i64_list(&coefficient_shape), IrArg::Float(1.0 / denom)],
        meta_with(&base_meta, coefficient_shape, &dtype),
    );
    graph.anchor_to(coefficient, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mul", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(sum), IrArg::Node(coefficient)],
        meta_with(&base_meta, output_shape, &dtype),
    );
    graph.anchor_to(result, id);

    graph.replace_and_erase(id, result);
    true
}
