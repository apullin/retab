use std::collections::{HashMap, HashSet};

use crate::fx_control_flow_metadata::QParam;
use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn is_tosa_accumulator_op(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    if target.contains("tosa") {
        return target.contains("conv2d")
            || target.contains("depthwise_conv2d")
            || target.contains("conv3d");
    }
    target.contains("aten.convolution") || target.contains("aten.conv2d")
}

fn is_layout_passthrough(target: &str) -> bool {
    let target = target.to_ascii_lowercase();
    target.contains("view")
        || target.contains("reshape")
        || target.contains("transpose")
        || target.contains("permute")
        || target.contains("squeeze")
        || target.contains("unsqueeze")
        || target.contains("flatten")
        || target.contains("contiguous")
}

fn changes_rank_across_layout_chain(graph: &IrGraph, current: NodeId, user: NodeId) -> bool {
    let target = graph.nodes[user].target.to_ascii_lowercase();
    if !(target.contains("view")
        || target.contains("reshape")
        || target.contains("flatten")
        || target.contains("squeeze")
        || target.contains("unsqueeze"))
    {
        return false;
    }

    graph.nodes[current]
        .meta
        .rank()
        .zip(graph.nodes[user].meta.rank())
        .map(|(before, after)| before != after)
        .unwrap_or(false)
}

fn single_live_user(graph: &IrGraph, id: NodeId) -> Option<NodeId> {
    let users = graph.live_user_ids(id);
    match users.as_slice() {
        [user] => Some(*user),
        _ => None,
    }
}

fn has_downstream_rescale(graph: &IrGraph, start: NodeId) -> bool {
    let mut current = start;
    let mut visited = HashSet::new();

    while visited.insert(current) {
        let Some(user) = single_live_user(graph, current) else {
            return false;
        };
        let user_node = &graph.nodes[user];
        if user_node.target.to_ascii_lowercase().contains("rescale") {
            return true;
        }
        if !is_layout_passthrough(&user_node.target)
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
        let user = single_live_user(graph, current)?;
        let user_node = &graph.nodes[user];
        if let Some(qparam) = user_node.meta.output_qparam() {
            return Some(qparam.clone());
        }
        if !is_layout_passthrough(&user_node.target) {
            return None;
        }
        current = user;
    }

    None
}

fn arg_contains_node(arg: &IrArg, target: NodeId) -> bool {
    match arg {
        IrArg::Node(id) => *id == target,
        IrArg::List(items) => items.iter().any(|item| arg_contains_node(item, target)),
        _ => false,
    }
}

fn recover_consumer_input_qparam(graph: &IrGraph, start: NodeId) -> Option<QParam> {
    for user_id in graph.live_user_ids(start) {
        let user = graph.node(user_id);
        let Some(input_qparams) = user.meta.input_qparams.as_ref() else {
            continue;
        };
        for (arg_idx, arg) in user.args.iter().enumerate() {
            if arg_contains_node(arg, start) {
                if let Some(qparam) = input_qparams.get(&arg_idx) {
                    return Some(qparam.clone());
                }
            }
        }
    }
    None
}

fn input_qparam(node: &crate::fx_ir::IrNode, index: usize) -> Option<&QParam> {
    node.meta.input_qparams.as_ref()?.get(&index)
}

fn output_qparam(graph: &IrGraph, id: NodeId) -> Option<QParam> {
    graph.nodes[id]
        .meta
        .output_qparam()
        .cloned()
        .or_else(|| recover_consumer_input_qparam(graph, id))
        .or_else(|| recover_downstream_output_qparam(graph, id))
}

fn weight_scales(qparam: &QParam) -> Vec<f64> {
    if qparam.is_per_channel() {
        qparam.per_channel_scales.clone()
    } else {
        vec![qparam.scale]
    }
}

fn rescale_meta_from(conv_meta: &NodeMeta, output_qparam: QParam) -> NodeMeta {
    let mut meta = conv_meta.clone();
    meta.set_input_qparams(HashMap::new());
    meta.set_single_output_qparam(output_qparam.clone());
    if let Some(order) = meta.tosa_dim_order.clone() {
        meta.set_tosa_dim_order(order);
    }
    if let Some(orders) = meta.original_dim_orders.clone() {
        meta.set_original_dim_orders(orders);
    }
    if let Some(rank) = meta.tosa_spatial_rank {
        meta.set_extra("tosa_spatial_rank", MetaValue::Int(rank as i64));
    }
    meta.extras.remove("tosa_special_dtype");
    if let Some(shape) = conv_meta.val_shape() {
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: output_qparam.dtype_str,
        });
    }
    meta
}

fn insert_conv_rescale(graph: &mut IrGraph, conv_id: NodeId) -> bool {
    let node = &graph.nodes[conv_id];
    if !node.alive
        || node.op != NodeOp::CallFunction
        || !is_tosa_accumulator_op(&node.target)
    {
        return false;
    }
    if has_downstream_rescale(graph, conv_id) {
        return false;
    }

    let Some(input_qp) = input_qparam(node, 0) else {
        return false;
    };
    let Some(weight_qp) = input_qparam(node, 1) else {
        return false;
    };
    let Some(output_qp) = output_qparam(graph, conv_id) else {
        return false;
    };
    if output_qp.scale == 0.0 {
        return false;
    }

    let scales: Vec<IrArg> = weight_scales(weight_qp)
        .into_iter()
        .map(|weight_scale| IrArg::Float((input_qp.scale * weight_scale) / output_qp.scale))
        .collect();
    if scales.is_empty() {
        return false;
    }

    let users = graph.live_user_ids(conv_id);
    if users.is_empty() {
        return false;
    }

    let copy_source_id = node.first_node_arg().unwrap_or(conv_id);
    let rescale_meta = rescale_meta_from(&node.meta, output_qp.clone());
    let rescale = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_accumulator_rescale", node.name),
        "backend.tosa.RESCALE.default".to_string(),
        vec![
            IrArg::Node(conv_id),
            IrArg::Dtype(output_qp.dtype_str.clone()),
            IrArg::List(scales),
            IrArg::Int(0),
            IrArg::Int(output_qp.zp),
        ],
        rescale_meta,
    );
    graph.anchor_after(rescale, conv_id);
    graph.copy_meta_from(rescale, copy_source_id);

    let mut rewired = false;
    for user_id in users {
        if user_id == rescale {
            continue;
        }
        rewired |= graph.replace_input_use(user_id, conv_id, rescale);
    }

    rewired
}

pub struct IrInsertConvAccumulatorRescalePass;

impl IrPass for IrInsertConvAccumulatorRescalePass {
    fn name(&self) -> &str {
        "IrInsertConvAccumulatorRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();
        let mut modified = false;
        for id in ids {
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

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    fn quantized_conv_meta() -> NodeMeta {
        let mut meta = meta(&[1, 4, 8, 8], "torch.int32");
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
            meta(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            meta(&[4, 3, 3, 3], "torch.int8"),
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
            meta(&[1, 4, 8, 8], "torch.int8"),
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
            meta(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            meta(&[4, 3, 3, 3], "torch.int8"),
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
        let mut table_meta = meta(&[1, 4, 8, 8], "torch.int8");
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
    fn skips_when_downstream_rescale_already_exists() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            meta(&[1, 3, 8, 8], "torch.int8"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            Vec::new(),
            meta(&[4, 3, 3, 3], "torch.int8"),
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
            meta(&[1, 4, 8, 8], "torch.int8"),
        );

        assert!(!IrInsertConvAccumulatorRescalePass.run(&mut graph));
    }
}
