//! Pure Rust IrPass: layer_norm / native_layer_norm decomposition.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn channels_last_order(rank: usize) -> Vec<usize> {
    if rank <= 2 {
        return (0..rank).collect();
    }
    let spatial_rank = rank.saturating_sub(2);
    let channel_axis = rank - (spatial_rank + 1);
    let mut order: Vec<usize> = (0..channel_axis).collect();
    order.extend(channel_axis + 1..rank);
    order.push(channel_axis);
    order
}

fn permuted_shape(shape: &[i64], perm: &[usize]) -> Vec<i64> {
    perm.iter()
        .filter_map(|&idx| shape.get(idx).copied())
        .collect()
}

fn normalized_physical_perm(rank: usize, dims: &[i64]) -> Vec<usize> {
    let dims_usize: Vec<usize> = dims.iter().map(|&dim| dim as usize).collect();
    channels_last_order(rank)
        .into_iter()
        .filter(|axis| dims_usize.contains(axis))
        .filter_map(|axis| dims_usize.iter().position(|&dim| dim == axis))
        .collect()
}

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut meta = base.clone();
    meta.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    meta
}

fn is_layer_norm(target: &str) -> bool {
    target.contains("layer_norm") || target.contains("native_layer_norm")
}

fn list_i64(arg: Option<&IrArg>) -> Option<Vec<i64>> {
    match arg {
        Some(IrArg::List(items)) => {
            let vals: Vec<i64> = items.iter().filter_map(IrArg::as_int).collect();
            Some(vals)
        }
        Some(IrArg::Int(v)) => Some(vec![*v]),
        _ => None,
    }
}

fn float_arg(arg: Option<&IrArg>, default: f64) -> f64 {
    match arg {
        Some(IrArg::Float(v)) => *v,
        Some(IrArg::Int(v)) => *v as f64,
        _ => default,
    }
}

fn first_output_user_meta(graph: &IrGraph, id: NodeId) -> Option<NodeMeta> {
    graph
        .nodes
        .iter()
        .find(|node| {
            node.alive
                && node.target.contains("getitem")
                && matches!(node.args.first(), Some(IrArg::Node(node_id)) if *node_id == id)
                && matches!(node.args.get(1), Some(IrArg::Int(0)))
                && node.meta.val_shape().is_some()
        })
        .map(|node| node.meta.clone())
}

fn replace_getitem_users(graph: &mut IrGraph, id: NodeId, replacement: NodeId) {
    let users: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|node| {
            node.alive
                && node.target.contains("getitem")
                && matches!(node.args.first(), Some(IrArg::Node(node_id)) if *node_id == id)
        })
        .map(|node| node.id)
        .collect();

    for user in users {
        if graph.nodes[user].alive {
            graph.replace_and_erase(user, replacement);
        }
    }
}

fn anchor_tensor_meta(graph: &mut IrGraph, node_id: NodeId, anchor: NodeId, meta_source: NodeId) {
    graph.anchor_to(node_id, anchor);
    graph.copy_meta_from(node_id, meta_source);
}

fn preserve_shape_node(graph: &mut IrGraph, node_id: NodeId) {
    let meta = &mut graph.node_mut(node_id).meta;
    meta.set_extra("arm_preserve_shape_node", MetaValue::Bool(true));
    meta.set_extra("arm_layout_neutral", MetaValue::Bool(true));
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
                || !is_layer_norm(&graph.nodes[id].target)
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

    let normalized_shape = match list_i64(graph.nodes[id].args.get(1)) {
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
    let param_shape: Vec<i64> = (0..rank)
        .map(|idx| {
            if dims.contains(&(idx as i64)) {
                input_shape[idx]
            } else {
                1
            }
        })
        .collect();
    let physical_param_shape = if rank >= 4 {
        permuted_shape(&param_shape, &channels_last_order(rank))
    } else {
        param_shape.clone()
    };
    let affine_perm = if rank >= 4 {
        normalized_physical_perm(rank, &dims)
    } else {
        (0..normalized_shape.len()).collect()
    };
    let affine_needs_permute = affine_perm.iter().enumerate().any(|(idx, &dim)| idx != dim);
    let scalar_shape = vec![1; rank];

    let weights = graph.nodes[id].args.get(2).and_then(|arg| match arg {
        IrArg::Node(node_id) => Some(*node_id),
        _ => None,
    });
    let bias = graph.nodes[id].args.get(3).and_then(|arg| match arg {
        IrArg::Node(node_id) => Some(*node_id),
        _ => None,
    });
    let epsilon = float_arg(graph.nodes[id].args.get(4), f32::EPSILON as f64);
    let name = graph.nodes[id].name.clone();
    let target = graph.nodes[id].target.clone();
    let view_target = if target.contains("layer_norm.default") && !target.contains("native") {
        "aten.reshape.default"
    } else {
        "aten.view_copy.default"
    };

    let mean = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mean", name),
        "aten.mean.dim".to_string(),
        vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
        meta_with(&output_meta, stat_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, mean, id, x_id);

    let numerator = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sub", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(mean)],
        meta_with(&output_meta, input_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, numerator, id, x_id);

    let mut var_kwargs = HashMap::new();
    var_kwargs.insert("correction".to_string(), IrArg::Int(0));
    var_kwargs.insert("keepdim".to_string(), IrArg::Bool(true));
    let var = graph.add_node_with_kwargs(
        NodeOp::CallFunction,
        format!("{}_var", name),
        "aten.var.correction".to_string(),
        vec![IrArg::Node(x_id), i64_list(&dims)],
        var_kwargs,
        meta_with(&output_meta, stat_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, var, id, x_id);

    let eps = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_eps", name),
        "aten.full.default".to_string(),
        vec![i64_list(&scalar_shape), IrArg::Float(epsilon)],
        meta_with(&output_meta, scalar_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, eps, id, x_id);

    let variance_eps = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_var_eps", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(var), IrArg::Node(eps)],
        meta_with(&output_meta, stat_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, variance_eps, id, x_id);

    let rsqrt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_rsqrt", name),
        "aten.rsqrt.default".to_string(),
        vec![IrArg::Node(variance_eps)],
        meta_with(&output_meta, stat_shape, &dtype),
    );
    anchor_tensor_meta(graph, rsqrt, id, x_id);

    let normalized = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_normalized", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(numerator), IrArg::Node(rsqrt)],
        meta_with(&output_meta, output_shape.clone(), &dtype),
    );
    anchor_tensor_meta(graph, normalized, id, x_id);

    let mut output = normalized;
    if let Some(weight_id) = weights {
        let weight_input = if affine_needs_permute {
            let weight_permute = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_weight_permute", name),
                "aten.permute_copy.default".to_string(),
                vec![
                    IrArg::Node(weight_id),
                    i64_list(
                        &affine_perm
                            .iter()
                            .map(|&dim| dim as i64)
                            .collect::<Vec<_>>(),
                    ),
                ],
                meta_with(
                    &output_meta,
                    permuted_shape(&normalized_shape, &affine_perm),
                    &dtype,
                ),
            );
            anchor_tensor_meta(graph, weight_permute, id, x_id);
            preserve_shape_node(graph, weight_permute);
            weight_permute
        } else {
            weight_id
        };
        let weight_reshaped = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_weight_view", name),
            view_target.to_string(),
            vec![IrArg::Node(weight_input), i64_list(&physical_param_shape)],
            meta_with(&output_meta, physical_param_shape.clone(), &dtype),
        );
        anchor_tensor_meta(graph, weight_reshaped, id, x_id);
        preserve_shape_node(graph, weight_reshaped);

        output = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_weighted", name),
            "aten.mul.Tensor".to_string(),
            vec![IrArg::Node(output), IrArg::Node(weight_reshaped)],
            meta_with(&output_meta, output_shape.clone(), &dtype),
        );
        anchor_tensor_meta(graph, output, id, x_id);
    }

    if let Some(bias_id) = bias {
        let bias_input = if affine_needs_permute {
            let bias_permute = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_bias_permute", name),
                "aten.permute_copy.default".to_string(),
                vec![
                    IrArg::Node(bias_id),
                    i64_list(
                        &affine_perm
                            .iter()
                            .map(|&dim| dim as i64)
                            .collect::<Vec<_>>(),
                    ),
                ],
                meta_with(
                    &output_meta,
                    permuted_shape(&normalized_shape, &affine_perm),
                    &dtype,
                ),
            );
            anchor_tensor_meta(graph, bias_permute, id, x_id);
            preserve_shape_node(graph, bias_permute);
            bias_permute
        } else {
            bias_id
        };
        let bias_reshaped = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_bias_view", name),
            view_target.to_string(),
            vec![IrArg::Node(bias_input), i64_list(&physical_param_shape)],
            meta_with(&output_meta, physical_param_shape, &dtype),
        );
        anchor_tensor_meta(graph, bias_reshaped, id, x_id);
        preserve_shape_node(graph, bias_reshaped);

        output = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_biased", name),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(output), IrArg::Node(bias_reshaped)],
            meta_with(&output_meta, output_shape, &dtype),
        );
        anchor_tensor_meta(graph, output, id, x_id);
    }

    replace_getitem_users(graph, id, output);
    graph.replace_and_erase(id, output);
    true
}
