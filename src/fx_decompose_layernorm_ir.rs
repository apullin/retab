//! Pure Rust IrPass: layer_norm / native_layer_norm decomposition.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::is_layer_norm_target;
use crate::runner_logging::log_line;

fn first_output_user_meta(graph: &IrGraph, id: NodeId) -> Option<NodeMeta> {
    graph
        .live_getitem_users(id)
        .into_iter()
        .find_map(|(node_id, idx)| {
            (idx == 0 && graph.nodes[node_id].meta.val_shape().is_some())
                .then(|| graph.nodes[node_id].meta.clone())
        })
}

fn replace_getitem_users(graph: &mut IrGraph, id: NodeId, replacement: NodeId) {
    let users: Vec<NodeId> = graph
        .live_getitem_users(id)
        .into_iter()
        .map(|(node_id, _)| node_id)
        .collect();

    for user in users {
        if graph.nodes[user].alive {
            graph.replace_and_erase(user, replacement);
        }
    }
}

pub struct IrDecomposeLayerNormPass;

impl IrPass for IrDecomposeLayerNormPass {
    fn name(&self) -> &str {
        "IrDecomposeLayerNorm"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive
                || graph.nodes[id].op != NodeOp::CallFunction
                || !is_layer_norm_target(&graph.nodes[id].target)
            {
                continue;
            }
            if decompose_layer_norm(graph, id) {
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeLayerNorm] decomposed layer_norm ops"
            ));
        }
        modified
    }
}

fn decompose_layer_norm(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(node_id) => node_id,
        None => return false,
    };
    let input_shape = match graph.nodes[x_id].meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let rank = input_shape.len();
    let output_meta = if graph.nodes[id].meta.val_shape().is_some() {
        graph.nodes[id].meta.clone()
    } else {
        match first_output_user_meta(graph, id) {
            Some(meta) => meta,
            None => return false,
        }
    };
    let output_shape = match output_meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let dtype = output_meta
        .val_dtype()
        .or_else(|| graph.nodes[x_id].meta.val_dtype())
        .unwrap_or("torch.float32")
        .to_string();

    let normalized_shape = match graph.nodes[id]
        .args
        .get(1)
        .and_then(IrArg::as_int_list_or_scalar)
    {
        Some(shape) if !shape.is_empty() => shape,
        _ => return false,
    };
    let n_dims = normalized_shape.len();
    if n_dims > rank {
        return false;
    }
    let dims: Vec<i64> = ((rank - n_dims)..rank).map(|dim| dim as i64).collect();
    let mut stat_shape = input_shape.clone();
    for &dim in &dims {
        stat_shape[dim as usize] = 1;
    }
    // Plain logical [1, ..., normalized_shape] param shape, matching upstream
    // decompose_layernorm_pass.py (`weights_reshaped_shape`). The weight/bias
    // views flow through normal NHWC annotation — no physical channels-last
    // shape, no permute, no layout-neutral marks.
    let param_shape: Vec<i64> = (0..rank)
        .map(|idx| {
            if dims.contains(&(idx as i64)) {
                input_shape[idx]
            } else {
                1
            }
        })
        .collect();
    let scalar_shape = vec![1; rank];

    let weights = graph.nodes[id].args.get(2).and_then(|arg| match arg {
        IrArg::Node(node_id) => Some(*node_id),
        _ => None,
    });
    let bias = graph.nodes[id].args.get(3).and_then(|arg| match arg {
        IrArg::Node(node_id) => Some(*node_id),
        _ => None,
    });
    let epsilon = graph.nodes[id]
        .args
        .get(4)
        .and_then(IrArg::as_numeric_f64)
        .unwrap_or(f32::EPSILON as f64);
    let name = graph.nodes[id].name.clone();
    let target = graph.nodes[id].target.clone();
    let view_target = if target.contains("layer_norm.default") && !target.contains("native") {
        "aten.reshape.default"
    } else {
        "aten.view_copy.default"
    };

    let mean = graph.add_anchored_call_function(
        id,
        format!("{}_mean", name),
        "aten.mean.dim",
        vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
        output_meta.cloned_with_val(stat_shape.clone(), &dtype),
    );

    let numerator = graph.add_anchored_call_function(
        id,
        format!("{}_sub", name),
        "aten.sub.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(mean)],
        output_meta.cloned_with_val(input_shape.clone(), &dtype),
    );

    let mut var_kwargs = HashMap::new();
    var_kwargs.insert("correction".to_string(), IrArg::Int(0));
    var_kwargs.insert("keepdim".to_string(), IrArg::Bool(true));
    let var = graph.add_anchored_call_function_with_kwargs(
        id,
        format!("{}_var", name),
        "aten.var.correction",
        vec![IrArg::Node(x_id), i64_list(&dims)],
        var_kwargs,
        output_meta.cloned_with_val(stat_shape.clone(), &dtype),
    );

    let eps = graph.add_anchored_call_function(
        id,
        format!("{}_eps", name),
        "aten.full.default",
        vec![i64_list(&scalar_shape), IrArg::Float(epsilon)],
        output_meta.cloned_with_val(scalar_shape.clone(), &dtype),
    );

    let variance_eps = graph.add_anchored_call_function(
        id,
        format!("{}_var_eps", name),
        "aten.add.Tensor",
        vec![IrArg::Node(var), IrArg::Node(eps)],
        output_meta.cloned_with_val(stat_shape.clone(), &dtype),
    );

    let rsqrt = graph.add_anchored_call_function(
        id,
        format!("{}_rsqrt", name),
        "aten.rsqrt.default",
        vec![IrArg::Node(variance_eps)],
        output_meta.cloned_with_val(stat_shape, &dtype),
    );

    let normalized = graph.add_anchored_call_function(
        id,
        format!("{}_normalized", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(numerator), IrArg::Node(rsqrt)],
        output_meta.cloned_with_val(output_shape.clone(), &dtype),
    );

    let mut output = normalized;
    if let Some(weight_id) = weights {
        let weight_reshaped = graph.add_anchored_call_function(
            id,
            format!("{}_weight_view", name),
            view_target,
            vec![IrArg::Node(weight_id), i64_list(&param_shape)],
            output_meta.cloned_with_val(param_shape.clone(), &dtype),
        );

        output = graph.add_anchored_call_function(
            id,
            format!("{}_weighted", name),
            "aten.mul.Tensor",
            vec![IrArg::Node(output), IrArg::Node(weight_reshaped)],
            output_meta.cloned_with_val(output_shape.clone(), &dtype),
        );
    }

    if let Some(bias_id) = bias {
        let bias_reshaped = graph.add_anchored_call_function(
            id,
            format!("{}_bias_view", name),
            view_target,
            vec![IrArg::Node(bias_id), i64_list(&param_shape)],
            output_meta.cloned_with_val(param_shape.clone(), &dtype),
        );

        output = graph.add_anchored_call_function(
            id,
            format!("{}_biased", name),
            "aten.add.Tensor",
            vec![IrArg::Node(output), IrArg::Node(bias_reshaped)],
            output_meta.cloned_with_val(output_shape, &dtype),
        );
    }

    replace_getitem_users(graph, id, output);
    graph.replace_and_erase(id, output);
    true
}
