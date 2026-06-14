use std::collections::HashSet;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp, QParam, RescaleRewrite};
use crate::fx_target::{
    is_abs_target, is_add_sub_target, is_comparison_target, is_minmax_target, is_mul_target,
    is_sum_dim_int_list_target, is_where_target,
};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

const I32_MIN: i64 = -2_147_483_648;
const I32_MAX: i64 = 2_147_483_647;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Int32OpKind {
    MinScale,
    AddSub,
    Passthrough,
}

fn classify_int32_op(target: &str) -> Option<Int32OpKind> {
    if is_abs_target(target) || is_comparison_target(target) || is_minmax_target(target) {
        Some(Int32OpKind::MinScale)
    } else if is_add_sub_target(target) {
        Some(Int32OpKind::AddSub)
    } else if is_mul_target(target) || is_sum_dim_int_list_target(target) {
        Some(Int32OpKind::Passthrough)
    } else {
        None
    }
}

fn int32_qparam(scale: f64) -> QParam {
    QParam::per_tensor(scale, 0, I32_MIN, I32_MAX, "torch.int32".to_string())
}

fn compute_rescaled_input_scales(kind: Int32OpKind, scales: &[f64], dtypes: &[DType]) -> Vec<f64> {
    match kind {
        Int32OpKind::MinScale => {
            let min_scale = scales.iter().copied().fold(f64::INFINITY, f64::min);
            vec![min_scale; scales.len()]
        }
        Int32OpKind::AddSub => {
            let max_scale = scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let is_int16 = dtypes.first() == Some(&DType::Int16);
            let shift_bits: u32 = if is_int16 { 12 } else { 20 };
            vec![(2.0 * max_scale) / (1u64 << shift_bits) as f64; scales.len()]
        }
        Int32OpKind::Passthrough => scales.to_vec(),
    }
}

fn compute_output_int32_scale(target: &str, rescaled_input_scales: &[f64]) -> Option<f64> {
    if is_comparison_target(target) {
        return None;
    }
    if is_minmax_target(target) {
        return Some(
            rescaled_input_scales
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
        );
    }
    if is_mul_target(target) {
        return Some(rescaled_input_scales.iter().product());
    }
    rescaled_input_scales.first().copied()
}

fn producer_old_zero_point(graph: &IrGraph, producer_id: NodeId, fallback_zp: i64) -> i64 {
    let producer = graph.node(producer_id);
    if producer.op == NodeOp::CallFunction && classify_int32_op(&producer.target).is_some() {
        return 0;
    }
    if let Some(qparam) = producer.meta.output_qparam() {
        if qparam.dtype() == DType::Int32 {
            return 0;
        }
        return qparam.zp;
    }
    if producer
        .meta
        .val_dtype_kind()
        .is_some_and(|dtype| dtype == DType::Int32)
    {
        0
    } else {
        fallback_zp
    }
}

fn insert_input_rescale(
    graph: &mut IrGraph,
    consumer_id: NodeId,
    source_id: NodeId,
    arg_pos: usize,
    old_scale: f64,
    old_zp: i64,
    new_scale: f64,
) -> NodeId {
    let output_qparam = int32_qparam(new_scale);
    let meta = graph
        .node(source_id)
        .meta
        .rescale_output_meta(output_qparam);
    let consumer_name = graph.node(consumer_id).name.clone();
    let rescale = graph.add_after_rescale(
        source_id,
        RescaleRewrite::new(
            format!("{consumer_name}_arg{arg_pos}_int32_rescale"),
            source_id,
            "torch.int32",
            vec![old_scale / new_scale],
            old_zp,
            0,
            meta,
        ),
    );
    graph.replace_input_use(consumer_id, source_id, rescale);
    rescale
}

fn insert_output_rescale(
    graph: &mut IrGraph,
    source_id: NodeId,
    output_int32_scale: f64,
    output_qparam: QParam,
) -> bool {
    if output_qparam.scale == 0.0 || !output_qparam.is_quantized_integer_dtype() {
        return false;
    }
    let users = graph.live_user_ids(source_id);
    if users.is_empty() {
        return false;
    }

    let meta = graph
        .node(source_id)
        .meta
        .rescale_output_meta(output_qparam.clone());
    let rescale = graph.add_after_rescale(
        source_id,
        RescaleRewrite::new(
            format!("{}_output_rescale", graph.node(source_id).name),
            source_id,
            output_qparam.dtype_str.clone(),
            vec![output_int32_scale / output_qparam.scale],
            0,
            output_qparam.zp,
            meta,
        ),
    );
    graph.replace_live_uses_except(source_id, rescale, rescale);
    true
}

fn insert_int32_rescales_for_node(graph: &mut IrGraph, node_id: NodeId) -> bool {
    let node = graph.node(node_id).clone();
    if !node.alive || node.op != NodeOp::CallFunction {
        return false;
    }
    let Some(kind) = classify_int32_op(&node.target) else {
        return false;
    };

    let input_qparams = node.meta.sorted_input_qparams();
    if input_qparams.is_empty()
        || !input_qparams
            .iter()
            .any(|(_, qparam)| qparam.is_quantized_integer_dtype())
    {
        return false;
    }

    let scales: Vec<f64> = input_qparams
        .iter()
        .map(|(_, qparam)| qparam.scale)
        .collect();
    let zps: Vec<i64> = input_qparams.iter().map(|(_, qparam)| qparam.zp).collect();
    let dtypes: Vec<DType> = input_qparams
        .iter()
        .map(|(_, qparam)| qparam.dtype())
        .collect();
    let rescaled_scales = compute_rescaled_input_scales(kind, &scales, &dtypes);
    let output_int32_scale = compute_output_int32_scale(&node.target, &rescaled_scales);

    let mut rescaled_inputs = false;
    let mut seen_sources = HashSet::new();
    for (key_idx, (pos, qparam)) in input_qparams.iter().enumerate() {
        if !qparam.is_quantized_integer_dtype() {
            continue;
        }
        let Some(source_id) = node.args.get(*pos).and_then(IrArg::as_node) else {
            continue;
        };
        if !seen_sources.insert(source_id) {
            continue;
        }
        let old_zp = producer_old_zero_point(graph, source_id, zps[key_idx]);
        insert_input_rescale(
            graph,
            node_id,
            source_id,
            *pos,
            scales[key_idx],
            old_zp,
            rescaled_scales[key_idx],
        );
        rescaled_inputs = true;
    }

    if !rescaled_inputs {
        return false;
    }

    graph.node_mut(node_id).meta.set_input_qparams_from(
        input_qparams
            .iter()
            .enumerate()
            .map(|(key_idx, (pos, _))| (*pos, int32_qparam(rescaled_scales[key_idx]))),
    );

    if let Some(output_int32_scale) = output_int32_scale {
        if let Some(output_qparam) = node.meta.output_qparam().cloned() {
            if insert_output_rescale(graph, node_id, output_int32_scale, output_qparam) {
                graph
                    .node_mut(node_id)
                    .meta
                    .set_single_output_qparam(int32_qparam(output_int32_scale));
                if graph.node(node_id).kwargs.contains_key("dtype") {
                    graph.set_kwarg(node_id, "dtype", IrArg::Dtype("torch.int32".to_string()));
                }
            }
        }
    }

    if output_int32_scale.is_some() {
        graph
            .node_mut(node_id)
            .meta
            .set_val_dtype_for_existing_shape("torch.int32");
    }

    true
}

fn qparam_signature(qparam: &QParam) -> (f64, i64, i64, i64, &str) {
    (
        qparam.scale,
        qparam.zp,
        qparam.qmin,
        qparam.qmax,
        qparam.dtype_str.as_str(),
    )
}

fn insert_where_output_rescale(graph: &mut IrGraph, node_id: NodeId) -> bool {
    let node = graph.node(node_id).clone();
    if !node.alive || node.op != NodeOp::CallFunction || !is_where_target(&node.target) {
        return false;
    }

    let Some(then_qparam) = node.meta.input_qparam_at(1).cloned() else {
        return false;
    };
    let Some(else_qparam) = node.meta.input_qparam_at(2) else {
        return false;
    };
    let Some(output_qparam) = node.meta.output_qparam().cloned() else {
        return false;
    };
    if qparam_signature(&then_qparam) != qparam_signature(else_qparam) {
        return false;
    }

    let output_dtype = node
        .meta
        .val_dtype()
        .unwrap_or(then_qparam.dtype_str.as_str())
        .to_string();
    let then_sig = qparam_signature(&then_qparam);
    let out_sig = qparam_signature(&output_qparam);
    let needs_noop_quant_boundary = then_sig == out_sig && output_dtype != output_qparam.dtype_str;

    if then_sig == out_sig && !needs_noop_quant_boundary {
        return false;
    }
    if !output_qparam.is_quantized_integer_dtype() {
        return false;
    }

    if needs_noop_quant_boundary {
        graph
            .node_mut(node_id)
            .meta
            .set_single_output_qparam(then_qparam.clone());
        graph
            .node_mut(node_id)
            .meta
            .set_val_dtype_for_existing_shape(output_qparam.dtype_str);
        return true;
    }

    let users = graph.live_user_ids(node_id);
    if users.is_empty() {
        return false;
    }
    let meta = node.meta.rescale_output_meta(output_qparam.clone());
    let rescale = graph.add_after_rescale(
        node_id,
        RescaleRewrite::new(
            format!("{}_output_rescale", node.name),
            node_id,
            output_qparam.dtype_str.clone(),
            vec![then_qparam.scale / output_qparam.scale],
            then_qparam.zp,
            output_qparam.zp,
            meta,
        ),
    );
    graph.replace_live_uses_except(node_id, rescale, rescale);
    graph
        .node_mut(node_id)
        .meta
        .set_single_output_qparam(then_qparam);
    true
}

pub struct IrInsertInt32RescalePass;

impl IrPass for IrInsertInt32RescalePass {
    fn name(&self) -> &str {
        "IrInsertInt32Rescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= insert_int32_rescales_for_node(graph, id);
        }

        for id in graph.live_call_function_ids() {
            modified |= insert_where_output_rescale(graph, id);
        }

        if modified {
            log_line(format_args!(
                "[IrInsertInt32Rescale] inserted INT32 rescale repairs"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeMeta;

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    #[test]
    fn rescales_add_inputs_and_output() {
        let mut graph = IrGraph::empty();
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int8"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int8"),
        );
        let mut add_meta = NodeMeta::tensor(&[2], "torch.int8");
        add_meta.set_input_qparams_from([
            (0, qparam(0.25, 3, "torch.int8")),
            (1, qparam(0.5, -4, "torch.int8")),
        ]);
        add_meta.set_single_output_qparam(qparam(0.125, 7, "torch.int8"));
        let add = graph.add_node(
            NodeOp::CallFunction,
            "add".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            add_meta,
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(add)],
            NodeMeta::tensor(&[2], "torch.int8"),
        );

        assert!(IrInsertInt32RescalePass.run(&mut graph));

        let add_node = graph.node(add);
        assert_eq!(add_node.meta.val_dtype(), Some("torch.int32"));
        assert_eq!(add_node.meta.input_qparams.as_ref().unwrap().len(), 2);
        assert!(matches!(add_node.args[0], IrArg::Node(id) if graph.node(id).is_rescale()));
        assert!(matches!(add_node.args[1], IrArg::Node(id) if graph.node(id).is_rescale()));

        let output = graph.by_name("add_output_rescale").unwrap();
        assert!(matches!(output.args[0], IrArg::Node(id) if id == add));
        assert!(matches!(output.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == output.id));
    }

    #[test]
    fn rescales_int32_add_inputs_and_output() {
        let mut graph = IrGraph::empty();
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int32"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int32"),
        );
        let mut add_meta = NodeMeta::tensor(&[2], "torch.int32");
        add_meta.set_input_qparams_from([(0, int32_qparam(0.25)), (1, int32_qparam(0.5))]);
        add_meta.set_single_output_qparam(int32_qparam(0.125));
        let add = graph.add_node(
            NodeOp::CallFunction,
            "add".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            add_meta,
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(add)],
            NodeMeta::tensor(&[2], "torch.int32"),
        );

        assert!(IrInsertInt32RescalePass.run(&mut graph));

        let add_node = graph.node(add);
        assert_eq!(add_node.meta.val_dtype(), Some("torch.int32"));
        assert!(matches!(add_node.args[0], IrArg::Node(id) if graph.node(id).is_rescale()));
        assert!(matches!(add_node.args[1], IrArg::Node(id) if graph.node(id).is_rescale()));

        let output = graph.by_name("add_output_rescale").unwrap();
        assert!(matches!(output.args[0], IrArg::Node(id) if id == add));
        assert!(matches!(output.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int32"));
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == output.id));
    }

    #[test]
    fn where_with_mismatched_output_domain_gets_output_rescale() {
        let mut graph = IrGraph::empty();
        let cond = graph.add_node(
            NodeOp::Placeholder,
            "cond".to_string(),
            "cond".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.bool"),
        );
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int8"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor(&[2], "torch.int8"),
        );
        let mut where_meta = NodeMeta::tensor(&[2], "torch.int8");
        where_meta.set_input_qparams_from([
            (1, qparam(0.5, 3, "torch.int8")),
            (2, qparam(0.5, 3, "torch.int8")),
        ]);
        where_meta.set_single_output_qparam(qparam(0.25, -7, "torch.int8"));
        let where_id = graph.add_node(
            NodeOp::CallFunction,
            "where".to_string(),
            "aten.where.self".to_string(),
            vec![IrArg::Node(cond), IrArg::Node(lhs), IrArg::Node(rhs)],
            where_meta,
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(where_id)],
            NodeMeta::tensor(&[2], "torch.int8"),
        );

        assert!(IrInsertInt32RescalePass.run(&mut graph));

        let rescale = graph.by_name("where_output_rescale").unwrap();
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == where_id));
        assert!(matches!(rescale.args[3], IrArg::Int(3)));
        assert!(matches!(rescale.args[4], IrArg::Int(-7)));
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == rescale.id));
        assert_eq!(
            graph.node(where_id).meta.output_qparam().unwrap().scale,
            0.5
        );
    }
}
