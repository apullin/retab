use std::collections::{HashMap, HashSet};

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, QParam, RescaleRewrite};
use crate::fx_target::{
    is_conv_accumulator_target, is_layout_passthrough_target, is_rank_changing_layout_target,
    is_raw_aten_convolution_target,
};
use crate::runner_logging::log_line;

fn changes_rank_across_layout_chain(graph: &IrGraph, current: NodeId, user: NodeId) -> bool {
    if !is_rank_changing_layout_target(&graph.nodes[user].target) {
        return false;
    }

    graph.nodes[current]
        .meta
        .rank()
        .zip(graph.nodes[user].meta.rank())
        .map(|(before, after)| before != after)
        .unwrap_or(false)
}

fn is_int32_rescale(node: &crate::fx_ir::IrNode) -> bool {
    node.args
        .get(1)
        .and_then(|arg| match arg {
            IrArg::Dtype(dtype) => Some(dtype.as_str()),
            _ => None,
        })
        .is_some_and(|dtype| dtype.contains("int32"))
        || node
            .meta
            .output_qparam()
            .is_some_and(|qparam| qparam.dtype_str.contains("int32"))
        || node
            .meta
            .val_dtype()
            .is_some_and(|dtype| dtype.contains("int32"))
}

fn has_downstream_output_rescale(graph: &IrGraph, start: NodeId) -> bool {
    let mut current = start;
    let mut visited = HashSet::new();

    while visited.insert(current) {
        let Some(user) = graph.single_live_user(current) else {
            return false;
        };
        let user_node = &graph.nodes[user];
        if user_node.target.to_ascii_lowercase().contains("rescale") {
            if is_int32_rescale(user_node) {
                return false;
            }
            return true;
        }
        if !is_layout_passthrough_target(&user_node.target)
            || changes_rank_across_layout_chain(graph, current, user)
        {
            return false;
        }
        current = user;
    }

    false
}

fn recover_downstream_output_qparam(graph: &IrGraph, start: NodeId) -> Option<QParam> {
    let mut current = start;
    let mut visited = HashSet::new();

    while visited.insert(current) {
        let user = graph.single_live_user(current)?;
        let user_node = &graph.nodes[user];
        if let Some(qparam) = user_node.meta.output_qparam() {
            return Some(qparam.clone());
        }
        if !is_layout_passthrough_target(&user_node.target) {
            return None;
        }
        current = user;
    }

    None
}

fn recover_consumer_input_qparam(graph: &IrGraph, start: NodeId) -> Option<QParam> {
    for user_id in graph.live_user_ids(start) {
        let user = graph.node(user_id);
        let Some(input_qparams) = user.meta.input_qparams.as_ref() else {
            continue;
        };
        for (arg_idx, arg) in user.args.iter().enumerate() {
            if arg.contains_node(start) {
                if let Some(qparam) = input_qparams.get(&arg_idx) {
                    return Some(qparam.clone());
                }
            }
        }
    }
    None
}

fn output_qparam(graph: &IrGraph, id: NodeId) -> Option<QParam> {
    graph.nodes[id]
        .meta
        .output_qparam()
        .cloned()
        .or_else(|| recover_downstream_output_qparam(graph, id))
        .or_else(|| recover_consumer_input_qparam(graph, id))
}

fn weight_scales(qparam: &QParam) -> Vec<f64> {
    if qparam.is_per_channel() {
        qparam.per_channel_scales.clone()
    } else {
        vec![qparam.scale]
    }
}

fn plain_tensor_meta_from(source_meta: &NodeMeta, shape: &[i64], dtype: &str) -> NodeMeta {
    let mut meta = source_meta.cloned_with_val(shape.to_vec(), dtype);
    meta.set_input_qparams(HashMap::new());
    meta.clear_output_qparams();
    meta.extras.remove("tosa_special_dtype");
    meta
}

fn activation_rank(graph: &IrGraph, node: &crate::fx_ir::IrNode) -> Option<usize> {
    match node.args.first() {
        Some(IrArg::Node(input_id)) => graph.node(*input_id).meta.rank(),
        _ => node.meta.rank(),
    }
}

fn needs_int16_bias_decomposition(graph: &IrGraph, conv_id: NodeId) -> bool {
    let node = graph.node(conv_id);
    if !node.alive
        || node.op != NodeOp::CallFunction
        || !is_raw_aten_convolution_target(&node.target)
        || !matches!(node.args.get(2), Some(IrArg::Node(_)))
    {
        return false;
    }
    if matches!(node.args.get(6), Some(IrArg::Bool(true))) {
        return false;
    }
    let Some(input_qp) = node.meta.input_qparam_at(0) else {
        return false;
    };
    let Some(bias_qp) = node.meta.input_qparam_at(2) else {
        return false;
    };
    let Some(output_qp) = output_qparam(graph, conv_id) else {
        return false;
    };
    matches!(activation_rank(graph, node), Some(4 | 5))
        && input_qp.dtype_str.contains("int16")
        && bias_qp.dtype_str.contains("int32")
        && output_qp.dtype_str.contains("int16")
}

fn decompose_int16_bias_conv(graph: &mut IrGraph, conv_id: NodeId) -> bool {
    if !needs_int16_bias_decomposition(graph, conv_id) {
        return false;
    }

    let node = graph.node(conv_id).clone();
    let Some(IrArg::Node(bias_id)) = node.args.get(2) else {
        return false;
    };
    let Some(output_qp) = output_qparam(graph, conv_id) else {
        return false;
    };
    if output_qp.scale == 0.0 {
        return false;
    }
    let Some(bias_qp) = node.meta.input_qparam_at(2).cloned() else {
        return false;
    };
    let Some(rank) = activation_rank(graph, &node) else {
        return false;
    };
    let Some(conv_shape) = node.meta.val_shape().map(|shape| shape.to_vec()) else {
        return false;
    };
    let Some(&channels) = graph
        .node(*bias_id)
        .meta
        .val_shape()
        .and_then(|shape| shape.first())
    else {
        return false;
    };

    let mut bias_view_shape = vec![1_i64; rank];
    if rank < 2 {
        return false;
    }
    bias_view_shape[1] = channels;

    let bias_scales = if bias_qp.is_per_channel() {
        bias_qp.per_channel_scales.clone()
    } else {
        vec![bias_qp.scale]
    };
    if bias_scales.is_empty() {
        return false;
    }

    {
        let conv = graph.node_mut(conv_id);
        conv.args[2] = IrArg::None;
    }

    let int32_rescale = graph.add_after_rescale(
        conv_id,
        RescaleRewrite::new(
            format!("{}_bias_int32_rescale", node.name),
            conv_id,
            "torch.int32",
            vec![1.0; bias_scales.len()],
            0,
            0,
            plain_tensor_meta_from(&node.meta, &conv_shape, "torch.int32"),
        ),
    );

    let mut bias_view_meta = plain_tensor_meta_from(&node.meta, &bias_view_shape, "torch.int32");
    if let Some(order) = node.meta.tosa_dim_order.clone() {
        bias_view_meta.set_tosa_dim_order(order);
    }
    if let Some(orders) = node.meta.original_dim_orders.clone() {
        bias_view_meta.set_original_dim_orders(orders);
    }
    if let Some(rank) = node.meta.tosa_spatial_rank {
        bias_view_meta.tosa_spatial_rank = Some(rank);
    }

    let bias_view = graph.add_after_call_function(
        int32_rescale,
        format!("{}_bias_view", node.name),
        "aten.view_copy.default",
        vec![
            IrArg::Node(*bias_id),
            IrArg::List(bias_view_shape.iter().copied().map(IrArg::Int).collect()),
        ],
        bias_view_meta,
    );

    let bias_add = graph.add_after_call_function(
        bias_view,
        format!("{}_bias_add", node.name),
        "aten.add.Tensor",
        vec![IrArg::Node(int32_rescale), IrArg::Node(bias_view)],
        plain_tensor_meta_from(&node.meta, &conv_shape, "torch.int32"),
    );

    let output_rescale = graph.add_after_rescale(
        bias_add,
        RescaleRewrite::new(
            format!("{}_bias_output_rescale", node.name),
            bias_add,
            output_qp.dtype_str.clone(),
            bias_scales
                .into_iter()
                .map(|scale| scale / output_qp.scale)
                .collect(),
            0,
            output_qp.zp,
            node.meta.rescale_output_meta(output_qp),
        ),
    );

    graph.replace_live_uses_except(conv_id, output_rescale, int32_rescale)
}

fn insert_conv_rescale(graph: &mut IrGraph, conv_id: NodeId) -> bool {
    if decompose_int16_bias_conv(graph, conv_id) {
        return true;
    }
    let node = &graph.nodes[conv_id];
    if !node.alive || node.op != NodeOp::CallFunction || !is_conv_accumulator_target(&node.target) {
        return false;
    }
    if has_downstream_output_rescale(graph, conv_id) {
        return false;
    }

    let Some(input_qp) = node.meta.input_qparam_at(0) else {
        return false;
    };
    let Some(weight_qp) = node.meta.input_qparam_at(1) else {
        return false;
    };
    let Some(output_qp) = output_qparam(graph, conv_id) else {
        return false;
    };
    if output_qp.scale == 0.0 {
        return false;
    }

    let scales: Vec<f64> = weight_scales(weight_qp)
        .into_iter()
        .map(|weight_scale| (input_qp.scale * weight_scale) / output_qp.scale)
        .collect();
    if scales.is_empty() {
        return false;
    }

    let users = graph.live_user_ids(conv_id);
    if users.is_empty() {
        return false;
    }

    let rescale = graph.add_after_rescale(
        conv_id,
        RescaleRewrite::new(
            format!("{}_accumulator_rescale", node.name),
            conv_id,
            output_qp.dtype_str.clone(),
            scales,
            0,
            output_qp.zp,
            node.meta.rescale_output_meta(output_qp.clone()),
        ),
    );

    graph.replace_live_uses_except(conv_id, rescale, rescale)
}

pub struct IrInsertConvAccumulatorRescalePass;

impl IrPass for IrInsertConvAccumulatorRescalePass {
    fn name(&self) -> &str {
        "IrInsertConvAccumulatorRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= insert_conv_rescale(graph, id);
        }
        if modified {
            log_line(format_args!(
                "[IrInsertConvAccumulatorRescale] inserted conv accumulator output rescale ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    fn per_channel_qparam(scales: &[f64], dtype: &str) -> QParam {
        let mut qparam = QParam::per_tensor(1.0, 0, -128, 127, dtype.to_string());
        qparam.per_channel_scales = scales.to_vec();
        qparam
    }

    fn quantized_conv_meta() -> NodeMeta {
        let mut meta = NodeMeta::tensor([1, 4, 8, 8], "torch.int32");
        let mut input_qparams = HashMap::new();
        input_qparams.insert(0, qparam(0.5, 1, "torch.int8"));
        input_qparams.insert(1, per_channel_qparam(&[0.25, 0.5, 0.75, 1.0], "torch.int8"));
        meta.set_input_qparams(input_qparams);
        let mut output_qparams = HashMap::new();
        output_qparams.insert(0, qparam(0.125, 7, "torch.int8"));
        meta.set_output_qparams(output_qparams);
        meta
    }

    #[test]
    fn inserts_per_channel_rescale_after_tosa_conv() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "backend.tosa.CONV2D.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            quantized_conv_meta(),
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(conv)],
            NodeMeta::tensor(&[1, 4, 8, 8], "torch.int8"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(user)],
            NodeMeta::default(),
        );

        assert!(IrInsertConvAccumulatorRescalePass.run(&mut graph));

        let rescale = graph.by_name("conv_accumulator_rescale").unwrap();
        assert_eq!(rescale.target, "backend.tosa.RESCALE.default");
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == conv));
        assert!(matches!(rescale.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(matches!(
            rescale.args[2],
            IrArg::List(ref scales)
                if matches!(scales.as_slice(), [
                    IrArg::Float(a),
                    IrArg::Float(b),
                    IrArg::Float(c),
                    IrArg::Float(d),
                ] if (*a - 1.0).abs() < 1e-12
                    && (*b - 2.0).abs() < 1e-12
                    && (*c - 3.0).abs() < 1e-12
                    && (*d - 4.0).abs() < 1e-12)
        ));
        assert!(matches!(rescale.args[3], IrArg::Int(0)));
        assert!(matches!(rescale.args[4], IrArg::Int(7)));
        assert_eq!(rescale.meta.val_dtype(), Some("torch.int8"));

        let user = graph.by_name("user").unwrap();
        assert!(matches!(user.args[0], IrArg::Node(id) if id == rescale.id));
    }

    #[test]
    fn inserts_after_raw_conv_using_consumer_input_qparam() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let mut conv_meta = quantized_conv_meta();
        conv_meta.output_qparams = None;
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "aten.conv2d.padding".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            conv_meta,
        );
        let mut table_meta = NodeMeta::tensor(&[1, 4, 8, 8], "torch.int8");
        table_meta.set_input_qparams(HashMap::from([(0, qparam(0.125, 7, "torch.int8"))]));
        let table = graph.add_node(
            NodeOp::CallFunction,
            "table".to_string(),
            "backend.tosa.TABLE.default".to_string(),
            vec![IrArg::Node(conv)],
            table_meta,
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(table)],
            NodeMeta::default(),
        );

        assert!(IrInsertConvAccumulatorRescalePass.run(&mut graph));

        let rescale = graph.by_name("conv_accumulator_rescale").unwrap();
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == conv));
        assert!(matches!(rescale.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        let table = graph.by_name("table").unwrap();
        assert!(matches!(table.args[0], IrArg::Node(id) if id == rescale.id));
    }

    #[test]
    fn prefers_downstream_activation_output_qparam_over_layout_input_qparam() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let mut conv_meta = quantized_conv_meta();
        conv_meta.output_qparams = None;
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "backend.tosa.CONV2D.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            conv_meta,
        );
        let mut transpose_meta = NodeMeta::tensor(&[1, 4, 8, 8], "torch.int8");
        transpose_meta.set_input_qparams(HashMap::from([(0, qparam(0.125, 7, "torch.int8"))]));
        let transpose = graph.add_node(
            NodeOp::CallFunction,
            "transpose".to_string(),
            "aten.permute_copy.default".to_string(),
            vec![IrArg::Node(conv)],
            transpose_meta,
        );
        let mut clamp_meta = NodeMeta::tensor(&[1, 4, 8, 8], "torch.int8");
        clamp_meta.set_input_qparams(HashMap::from([(0, qparam(0.125, 7, "torch.int8"))]));
        clamp_meta.set_output_qparams(HashMap::from([(0, qparam(0.25, -128, "torch.int8"))]));
        let clamp = graph.add_node(
            NodeOp::CallFunction,
            "clamp".to_string(),
            "aten.clamp.default".to_string(),
            vec![IrArg::Node(transpose), IrArg::Int(0), IrArg::None],
            clamp_meta,
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(clamp)],
            NodeMeta::default(),
        );

        assert!(IrInsertConvAccumulatorRescalePass.run(&mut graph));

        let rescale = graph.by_name("conv_accumulator_rescale").unwrap();
        assert!(matches!(
            rescale.args[2],
            IrArg::List(ref scales)
                if matches!(scales.as_slice(), [
                    IrArg::Float(a),
                    IrArg::Float(b),
                    IrArg::Float(c),
                    IrArg::Float(d),
                ] if (*a - 0.5).abs() < 1e-12
                    && (*b - 1.0).abs() < 1e-12
                    && (*c - 1.5).abs() < 1e-12
                    && (*d - 2.0).abs() < 1e-12)
        ));
        assert!(matches!(rescale.args[4], IrArg::Int(-128)));
        assert_eq!(rescale.meta.output_qparam().unwrap().zp, -128);
    }

    #[test]
    fn skips_when_downstream_rescale_already_exists() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "backend.tosa.CONV2D.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            quantized_conv_meta(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "existing_rescale".to_string(),
            "backend.tosa.RESCALE.default".to_string(),
            vec![IrArg::Node(conv)],
            NodeMeta::tensor(&[1, 4, 8, 8], "torch.int8"),
        );

        assert!(!IrInsertConvAccumulatorRescalePass.run(&mut graph));
    }

    #[test]
    fn inserts_before_downstream_int32_rescale() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "backend.tosa.CONV2D.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::None],
            quantized_conv_meta(),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "add_arg1_int32_rescale".to_string(),
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                IrArg::Node(conv),
                IrArg::Dtype("torch.int32".to_string()),
                IrArg::List(vec![IrArg::Float(0.5)]),
                IrArg::Int(7),
                IrArg::Int(0),
            ],
            NodeMeta::tensor(&[1, 4, 8, 8], "torch.int32"),
        );

        assert!(IrInsertConvAccumulatorRescalePass.run(&mut graph));

        let conv_rescale = graph.by_name("conv_accumulator_rescale").unwrap();
        let int32_rescale = graph.by_name("add_arg1_int32_rescale").unwrap();
        assert!(matches!(conv_rescale.args[0], IrArg::Node(id) if id == conv));
        assert!(matches!(
            int32_rescale.args[0],
            IrArg::Node(id) if id == conv_rescale.id
        ));
    }

    #[test]
    fn decomposes_int16_bias_convolution_into_add_chain() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[1, 3, 8, 8], "torch.int16"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4, 3, 3, 3], "torch.int8"),
        );
        let bias = graph.add_node(
            NodeOp::Placeholder,
            "bias".to_string(),
            "bias".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[4], "torch.int32"),
        );
        let mut conv_meta = NodeMeta::tensor(&[1, 4, 8, 8], "torch.int16");
        conv_meta.set_input_qparams(HashMap::from([
            (0, qparam(0.5, 0, "torch.int16")),
            (1, per_channel_qparam(&[0.25, 0.5, 0.75, 1.0], "torch.int8")),
            (
                2,
                per_channel_qparam(&[0.125, 0.25, 0.375, 0.5], "torch.int32"),
            ),
        ]));
        conv_meta.set_output_qparams(HashMap::from([(0, qparam(0.0625, 7, "torch.int16"))]));
        let conv = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "aten.convolution.default".to_string(),
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::Node(bias)],
            conv_meta,
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(conv)],
            NodeMeta::tensor(&[1, 4, 8, 8], "torch.int16"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(user)],
            NodeMeta::default(),
        );

        assert!(IrInsertConvAccumulatorRescalePass.run(&mut graph));

        let conv = graph.by_name("conv").unwrap();
        assert!(matches!(conv.args[2], IrArg::None));

        let int32_rescale = graph.by_name("conv_bias_int32_rescale").unwrap();
        assert!(matches!(int32_rescale.args[0], IrArg::Node(id) if id == conv.id));
        assert!(matches!(
            int32_rescale.args[1],
            IrArg::Dtype(ref dtype) if dtype == "torch.int32"
        ));

        let bias_view = graph.by_name("conv_bias_view").unwrap();
        assert_eq!(bias_view.meta.val_shape(), Some(&[1, 4, 1, 1][..]));

        let bias_add = graph.by_name("conv_bias_add").unwrap();
        assert_eq!(bias_add.target, "aten.add.Tensor");

        let final_rescale = graph.by_name("conv_bias_output_rescale").unwrap();
        assert!(matches!(
            final_rescale.args[1],
            IrArg::Dtype(ref dtype) if dtype == "torch.int16"
        ));

        let user = graph.by_name("user").unwrap();
        assert!(matches!(user.args[0], IrArg::Node(id) if id == final_rescale.id));
    }
}
