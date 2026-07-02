//! Pure Rust IrPass: layer_norm / native_layer_norm decomposition.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta};
use crate::fx_target::{is_layer_norm_target, is_permute_target, layer_norm_param_view_target};
use crate::runner_logging::log_line;

fn first_output_user_meta(graph: &IrGraph, id: NodeId) -> Option<NodeMeta> {
    graph
        .live_getitem_users(id)
        .into_iter()
        .find_map(|(node_id, idx)| {
            (idx == 0 && graph.node_shape(node_id).is_some())
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

fn channels_last_permute(rank: usize) -> Vec<i64> {
    if rank <= 2 {
        return (0..rank).map(|axis| axis as i64).collect();
    }
    let mut order = vec![0_i64];
    order.extend((2..rank).map(|axis| axis as i64));
    order.push(1);
    order
}

fn physical_identity_layer_norm_input(
    graph: &IrGraph,
    x_id: NodeId,
    rank: usize,
) -> Option<NodeId> {
    if !is_permute_target(&graph.nodes[x_id].target) {
        return None;
    }
    let perm = graph.nodes[x_id].args.get(1)?.as_int_list()?;
    (perm == channels_last_permute(rank)).then(|| graph.nodes[x_id].first_node_arg())?
}

pub struct IrDecomposeLayerNormPass;

impl IrPass for IrDecomposeLayerNormPass {
    fn name(&self) -> &str {
        "IrDecomposeLayerNorm"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            is_layer_norm_target(&graph.nodes[id].target) && decompose_layer_norm(graph, id)
        });

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
    let input_shape = match graph.node_shape_vec(x_id) {
        Some(shape) => shape,
        None => return false,
    };
    let rank = input_shape.len();
    let output_meta = if graph.node_shape(id).is_some() {
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
    let input_meta = graph.nodes[x_id].meta.clone();
    let meta = output_meta.tensor_stamp_same_dtype_or_dtype_of_or_float32(&input_meta);

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
    let reduction_count = normalized_shape.iter().product::<i64>().max(1).to_string();
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

    let weights = graph.nodes[id].arg_node(2);
    let bias = graph.nodes[id].arg_node(3);
    let epsilon = graph.nodes[id]
        .arg_numeric_f64(4)
        .unwrap_or(f32::EPSILON as f64);
    let name = graph.nodes[id].name.clone();
    let Some(view_target) = layer_norm_param_view_target(&graph.nodes[id].target) else {
        return false;
    };
    let original_x_id = x_id;
    let physical_identity_source = if rank >= 4 {
        physical_identity_layer_norm_input(graph, x_id, rank)
    } else {
        None
    };
    let x_id = physical_identity_source.unwrap_or(x_id);

    let (output, force_identity_layout_ids) = {
        let mut cursor = graph.rewrite_cursor(id);
        let mut force_identity_layout_ids = Vec::new();

        let mean = cursor.call_function(
            format!("{}_mean", name),
            "aten.mean.dim",
            vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
            meta.preserve_qparams(stat_shape.clone()),
        );
        cursor.graph_mut().nodes[mean].meta.set_extra(
            "retab_reduction_element_count",
            MetaValue::Str(reduction_count.clone()),
        );
        if physical_identity_source.is_some() {
            cursor.graph_mut().nodes[mean]
                .meta
                .set_extra("retab_reduce_axes_are_physical", MetaValue::Bool(true));
        }
        force_identity_layout_ids.push(mean);

        let numerator = cursor.call_function(
            format!("{}_sub", name),
            "aten.sub.Tensor",
            vec![IrArg::Node(x_id), IrArg::Node(mean)],
            meta.preserve_qparams(input_shape.clone()),
        );
        force_identity_layout_ids.push(numerator);

        let mut var_kwargs = HashMap::new();
        var_kwargs.insert("correction".to_string(), IrArg::Int(0));
        var_kwargs.insert("keepdim".to_string(), IrArg::Bool(true));
        let var = cursor.call_function_with_kwargs(
            format!("{}_var", name),
            "aten.var.correction",
            vec![IrArg::Node(x_id), i64_list(&dims)],
            var_kwargs,
            meta.preserve_qparams(stat_shape.clone()),
        );
        cursor.graph_mut().nodes[var].meta.set_extra(
            "retab_reduction_element_count",
            MetaValue::Str(reduction_count.clone()),
        );
        if physical_identity_source.is_some() {
            cursor.graph_mut().nodes[var]
                .meta
                .set_extra("retab_reduce_axes_are_physical", MetaValue::Bool(true));
        }
        force_identity_layout_ids.push(var);

        let eps = cursor.call_function(
            format!("{}_eps", name),
            "aten.full.default",
            vec![i64_list(&scalar_shape), IrArg::Float(epsilon)],
            meta.preserve_qparams(scalar_shape.clone()),
        );
        force_identity_layout_ids.push(eps);

        let variance_eps = cursor.call_function(
            format!("{}_var_eps", name),
            "aten.add.Tensor",
            vec![IrArg::Node(var), IrArg::Node(eps)],
            meta.preserve_qparams(stat_shape.clone()),
        );
        force_identity_layout_ids.push(variance_eps);

        let rsqrt = cursor.call_function(
            format!("{}_rsqrt", name),
            "aten.rsqrt.default",
            vec![IrArg::Node(variance_eps)],
            meta.preserve_qparams(stat_shape),
        );
        force_identity_layout_ids.push(rsqrt);

        let normalized = cursor.call_function(
            format!("{}_normalized", name),
            "aten.mul.Tensor",
            vec![IrArg::Node(numerator), IrArg::Node(rsqrt)],
            meta.preserve_qparams(output_shape.clone()),
        );
        force_identity_layout_ids.push(normalized);

        let mut output = normalized;
        if let Some(weight_id) = weights {
            let weight_reshaped = cursor.call_function(
                format!("{}_weight_view", name),
                view_target,
                vec![IrArg::Node(weight_id), i64_list(&param_shape)],
                meta.preserve_qparams(param_shape.clone()),
            );
            force_identity_layout_ids.push(weight_reshaped);

            output = cursor.call_function(
                format!("{}_weighted", name),
                "aten.mul.Tensor",
                vec![IrArg::Node(output), IrArg::Node(weight_reshaped)],
                meta.preserve_qparams(output_shape.clone()),
            );
            force_identity_layout_ids.push(output);
        }

        if let Some(bias_id) = bias {
            let bias_reshaped = cursor.call_function(
                format!("{}_bias_view", name),
                view_target,
                vec![IrArg::Node(bias_id), i64_list(&param_shape)],
                meta.preserve_qparams(param_shape.clone()),
            );
            force_identity_layout_ids.push(bias_reshaped);

            output = cursor.call_function(
                format!("{}_biased", name),
                "aten.add.Tensor",
                vec![IrArg::Node(output), IrArg::Node(bias_reshaped)],
                meta.preserve_qparams(output_shape),
            );
            force_identity_layout_ids.push(output);
        }
        (output, force_identity_layout_ids)
    };

    if physical_identity_source.is_some() {
        for node_id in force_identity_layout_ids {
            graph.nodes[node_id]
                .meta
                .set_extra("retab_force_identity_dim_order", MetaValue::Bool(true));
        }
    }
    replace_getitem_users(graph, id, output);
    graph.replace_and_erase(id, output);
    if physical_identity_source.is_some() && graph.live_user_ids(original_x_id).is_empty() {
        graph.erase(original_x_id);
    }
    true
}
