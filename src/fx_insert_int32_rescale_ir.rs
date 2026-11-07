use std::collections::HashSet;

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp, QParam, RescaleRewrite};
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

/// Stock InsertRescaleInt32Pass only rewraps narrow (INT8/INT16) operands:
/// INT32-domain compute such as the INT_i32 add needs no extra rescales.
fn is_narrow_quantized_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Int8 | DType::Int16)
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
    let rescale = {
        let mut cursor = graph.rewrite_cursor_after(consumer_id, source_id);
        cursor.rescale(RescaleRewrite::new(
            format!("{consumer_name}_arg{arg_pos}_int32_rescale"),
            source_id,
            "torch.int32",
            vec![old_scale / new_scale],
            old_zp,
            0,
            meta,
        ))
    };
    graph.replace_input_use(consumer_id, source_id, rescale);
    rescale
}

fn insert_output_rescale(
    graph: &mut IrGraph,
    source_id: NodeId,
    output_int32_scale: f64,
    output_qparam: QParam,
) -> bool {
    if output_qparam.scale == 0.0 || !is_narrow_quantized_dtype(output_qparam.dtype()) {
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
    let rescale_name = format!("{}_output_rescale", graph.node(source_id).name);
    let output_dtype = output_qparam.dtype_str.clone();
    let output_zp = output_qparam.zp;
    let rescale = {
        let mut cursor = graph.rewrite_cursor_after(source_id, source_id);
        cursor.rescale(RescaleRewrite::new(
            rescale_name,
            source_id,
            output_dtype,
            vec![output_int32_scale / output_qparam.scale],
            0,
            output_zp,
            meta,
        ))
    };
    graph.replace_live_uses_except(source_id, rescale, rescale);
    true
}

fn direct_input_node_ids(node: &IrNode) -> Vec<NodeId> {
    fn visit(arg: &IrArg, ids: &mut Vec<NodeId>) {
        match arg {
            IrArg::Node(id) => ids.push(*id),
            IrArg::List(items) => items.iter().for_each(|item| visit(item, ids)),
            _ => {}
        }
    }
    let mut ids = Vec::new();
    node.args.iter().for_each(|arg| visit(arg, &mut ids));
    node.kwargs.values().for_each(|arg| visit(arg, &mut ids));
    ids
}

fn insert_int32_rescales_for_node(graph: &mut IrGraph, node_id: NodeId) -> bool {
    let node = graph.node(node_id).clone();
    if !node.alive || node.op != NodeOp::CallFunction {
        return false;
    }
    let Some(kind) = classify_int32_op(&node.target) else {
        return false;
    };

    // A direct DQ input or Q user means Q/DQ folding deliberately preserved
    // the quantization boundary, so this is not an INT32-domain op.
    if direct_input_node_ids(&node)
        .iter()
        .any(|id| graph.node(*id).is_dequantize_any())
        || graph
            .live_user_ids(node_id)
            .iter()
            .any(|&user| graph.node(user).is_quantize_any())
    {
        return false;
    }

    let input_qparams = node.meta.sorted_input_qparams();
    if input_qparams.is_empty()
        || !input_qparams
            .iter()
            .any(|(_, qparam)| is_narrow_quantized_dtype(qparam.dtype()))
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
        if !is_narrow_quantized_dtype(qparam.dtype()) {
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
    if !then_qparam.same_exact_affine_qrange_dtype(else_qparam) {
        return false;
    }

    let output_dtype = node
        .meta
        .val_dtype()
        .unwrap_or(then_qparam.dtype_str.as_str())
        .to_string();
    let same_output_qparams = then_qparam.same_exact_affine_qrange_dtype(&output_qparam);
    let needs_noop_quant_boundary = same_output_qparams && output_dtype != output_qparam.dtype_str;

    if same_output_qparams && !needs_noop_quant_boundary {
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
    let rescale = {
        let mut cursor = graph.rewrite_cursor_after(node_id, node_id);
        cursor.rescale(RescaleRewrite::new(
            format!("{}_output_rescale", node.name),
            node_id,
            output_qparam.dtype_str.clone(),
            vec![then_qparam.scale / output_qparam.scale],
            then_qparam.zp,
            output_qparam.zp,
            meta,
        ))
    };
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
        let modified = graph.rewrite_live_call_functions(insert_int32_rescales_for_node)
            | graph.rewrite_live_call_functions(insert_where_output_rescale);

        if modified {
            log_line(format_args!(
                "[IrInsertInt32Rescale] inserted INT32 rescale repairs"
            ));
        }
        modified
    }
}

pub struct IrFuseConsecutiveRescalesPass;

impl IrPass for IrFuseConsecutiveRescalesPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveRescales"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        graph.rewrite_live_call_functions(|graph, id| {
            let node = graph.node(id);
            if !node.is_rescale()
                || !matches!(node.arg(1), Some(IrArg::Dtype(dtype) | IrArg::Str(dtype))
                    if dtype == "torch.int8" || dtype == "torch.int16")
            {
                return false;
            }
            let Some(input) = node.first_node_arg() else {
                return false;
            };
            if graph.node_dtype(input) != Some("torch.int32") {
                return false;
            }
            let scale = match node.arg(2) {
                Some(IrArg::List(scales)) if scales.len() == 1 => scales[0].as_numeric_f64(),
                _ => None,
            };
            let (Some(scale), Some(input_zp), Some(output_zp)) =
                (scale, node.arg_int(3), node.arg_int(4))
            else {
                return false;
            };

            let mut modified = false;
            for user_id in graph.live_user_ids(id) {
                let user = graph.node(user_id);
                if !user.is_rescale()
                    || !matches!(user.arg(1), Some(IrArg::Dtype(dtype) | IrArg::Str(dtype))
                        if dtype == "torch.int32")
                    || user.arg_int(3) != Some(output_zp)
                    || user.arg_int(4) != Some(input_zp)
                {
                    continue;
                }
                let user_scale = match user.arg(2) {
                    Some(IrArg::List(scales)) if scales.len() == 1 => scales[0].as_numeric_f64(),
                    _ => None,
                };
                // rc4 only bypasses identity round trips; Vela cannot lower the
                // INT32 -> INT32 rescale that a non-identity fusion would need.
                if user_scale.is_some_and(|user_scale| (scale * user_scale - 1.0).abs() < 1e-6) {
                    graph.replace_and_erase(user_id, input);
                    modified = true;
                }
            }
            if modified && !graph.has_live_users(id) {
                graph.erase(id);
            }
            modified
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeMeta;
    use crate::test_ir::{call_with_meta as call, placeholder};

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    fn view_user(graph: &mut IrGraph, input: NodeId, dtype: &str) -> NodeId {
        call(
            graph,
            "user",
            "aten.view_copy.default",
            vec![IrArg::Node(input)],
            NodeMeta::tensor([2], dtype),
        )
    }

    #[test]
    fn fuses_identity_int8_and_int16_rescale_round_trips() {
        for dtype in ["torch.int8", "torch.int16"] {
            let mut graph = IrGraph::empty();
            let input = placeholder(&mut graph, "input", [2], "torch.int32");
            let r1 = graph.add_rescale(RescaleRewrite::new(
                "r1",
                input,
                dtype,
                vec![0.25],
                0,
                0,
                NodeMeta::tensor([2], dtype),
            ));
            let r2 = graph.add_rescale(RescaleRewrite::new(
                "r2",
                r1,
                "torch.int32",
                vec![4.0],
                0,
                0,
                NodeMeta::tensor([2], "torch.int32"),
            ));
            let user = view_user(&mut graph, r2, "torch.int32");

            assert!(IrFuseConsecutiveRescalesPass.run(&mut graph));
            assert!(!graph.node(r1).alive);
            assert!(!graph.node(r2).alive);
            assert_eq!(graph.node(user).first_node_arg(), Some(input));
        }
    }

    #[test]
    fn rescale_fusion_preserves_other_users() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [2], "torch.int32");
        let r1 = graph.add_rescale(RescaleRewrite::new(
            "r1",
            input,
            "torch.int8",
            vec![0.25],
            0,
            3,
            NodeMeta::tensor([2], "torch.int8"),
        ));
        let r2 = graph.add_rescale(RescaleRewrite::new(
            "r2",
            r1,
            "torch.int32",
            vec![4.0],
            3,
            0,
            NodeMeta::tensor([2], "torch.int32"),
        ));
        let other = graph.add_rescale(RescaleRewrite::new(
            "other",
            r1,
            "torch.int32",
            vec![2.0],
            3,
            0,
            NodeMeta::tensor([2], "torch.int32"),
        ));
        let narrow_user = view_user(&mut graph, r1, "torch.int8");
        let user = call(
            &mut graph,
            "int32_user",
            "aten.abs.default",
            vec![IrArg::Node(r2)],
            NodeMeta::tensor([2], "torch.int32"),
        );

        assert!(IrFuseConsecutiveRescalesPass.run(&mut graph));
        assert!(graph.node(r1).alive);
        assert!(graph.node(other).alive);
        assert!(!graph.node(r2).alive);
        assert_eq!(graph.node(narrow_user).first_node_arg(), Some(r1));
        assert_eq!(graph.node(user).first_node_arg(), Some(input));
    }

    #[test]
    fn rescale_fusion_rejects_nonidentity_zero_point_and_per_channel_pairs() {
        for (scales, input_zp, output_zp) in [
            (vec![2.0], 3, 0),
            (vec![4.0], 2, 0),
            (vec![4.0], 3, 1),
            (vec![4.0, 4.0], 3, 0),
        ] {
            let mut graph = IrGraph::empty();
            let input = placeholder(&mut graph, "input", [2], "torch.int32");
            let r1 = graph.add_rescale(RescaleRewrite::new(
                "r1",
                input,
                "torch.int8",
                vec![0.25],
                0,
                3,
                NodeMeta::tensor([2], "torch.int8"),
            ));
            let r2 = graph.add_rescale(RescaleRewrite::new(
                "r2",
                r1,
                "torch.int32",
                scales,
                input_zp,
                output_zp,
                NodeMeta::tensor([2], "torch.int32"),
            ));
            view_user(&mut graph, r2, "torch.int32");

            assert!(!IrFuseConsecutiveRescalesPass.run(&mut graph));
            assert!(graph.node(r1).alive);
            assert!(graph.node(r2).alive);
        }
    }

    #[test]
    fn rescales_add_inputs_and_output() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [2], "torch.int8");
        let rhs = placeholder(&mut graph, "rhs", [2], "torch.int8");
        let mut add_meta = NodeMeta::tensor([2], "torch.int8");
        add_meta.set_binary_input_output_qparams(
            qparam(0.25, 3, "torch.int8"),
            qparam(0.5, -4, "torch.int8"),
            qparam(0.125, 7, "torch.int8"),
        );
        let add = call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            add_meta,
        );
        let user = view_user(&mut graph, add, "torch.int8");

        assert!(IrInsertInt32RescalePass.run(&mut graph));

        let add_node = graph.node(add);
        assert_eq!(add_node.meta.val_dtype(), Some("torch.int32"));
        assert_eq!(add_node.meta.input_qparams().unwrap().iter().count(), 2);
        assert!(matches!(add_node.args[0], IrArg::Node(id) if graph.node(id).is_rescale()));
        assert!(matches!(add_node.args[1], IrArg::Node(id) if graph.node(id).is_rescale()));

        let output = graph.by_name("add_output_rescale").unwrap();
        assert!(matches!(output.args[0], IrArg::Node(id) if id == add));
        assert!(matches!(output.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == output.id));
    }

    #[test]
    fn leaves_int32_domain_add_without_rescales() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [2], "torch.int32");
        let rhs = placeholder(&mut graph, "rhs", [2], "torch.int32");
        let mut add_meta = NodeMeta::tensor([2], "torch.int32");
        add_meta.set_binary_input_output_qparams(
            int32_qparam(0.25),
            int32_qparam(0.5),
            int32_qparam(0.125),
        );
        let add = call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            add_meta,
        );
        let user = view_user(&mut graph, add, "torch.int32");

        assert!(!IrInsertInt32RescalePass.run(&mut graph));

        let add_node = graph.node(add);
        assert!(matches!(add_node.args[0], IrArg::Node(id) if id == lhs));
        assert!(matches!(add_node.args[1], IrArg::Node(id) if id == rhs));
        assert!(graph.by_name("add_output_rescale").is_none());
        assert_eq!(graph.node(user).first_node_arg(), Some(add));
    }

    #[test]
    fn skips_int32_rescale_when_qdq_boundary_is_preserved() {
        use crate::test_ir::f32_call;

        for target in ["aten.add.Tensor", "aten.mul.Tensor"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2], "torch.int8");
            let y = placeholder(&mut graph, "y", [2], "torch.float32");
            let x_q = call(
                &mut graph,
                "x_q",
                "quantized_decomposed.quantize_per_tensor.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Float(0.5),
                    IrArg::Int(0),
                    IrArg::Int(-128),
                    IrArg::Int(127),
                    IrArg::Dtype("torch.int8".to_string()),
                ],
                NodeMeta::tensor([2], "torch.int8"),
            );
            let x_dq = f32_call(
                &mut graph,
                "x_dq",
                "quantized_decomposed.dequantize_per_tensor.default",
                vec![
                    IrArg::Node(x_q),
                    IrArg::Float(0.5),
                    IrArg::Int(0),
                    IrArg::Int(-128),
                    IrArg::Int(127),
                    IrArg::Dtype("torch.int8".to_string()),
                ],
                [2],
            );
            let mut binary_meta = NodeMeta::tensor([2], "torch.int8");
            binary_meta.set_input_qparams_from([(0, qparam(0.5, 0, "torch.int8"))]);
            binary_meta.set_single_output_qparam(qparam(0.5, 0, "torch.int8"));
            let binary = call(
                &mut graph,
                "binary",
                target,
                vec![IrArg::Node(x_dq), IrArg::Node(y)],
                binary_meta,
            );
            let out_q = call(
                &mut graph,
                "out_q",
                "quantized_decomposed.quantize_per_tensor.default",
                vec![
                    IrArg::Node(binary),
                    IrArg::Float(0.5),
                    IrArg::Int(0),
                    IrArg::Int(-128),
                    IrArg::Int(127),
                    IrArg::Dtype("torch.int8".to_string()),
                ],
                NodeMeta::tensor([2], "torch.int8"),
            );
            let user = view_user(&mut graph, out_q, "torch.int8");

            assert!(!IrInsertInt32RescalePass.run(&mut graph));
            assert!(matches!(graph.node(binary).args[0], IrArg::Node(id) if id == x_dq));
            assert!(graph.users(binary).contains(&out_q));
            assert_eq!(graph.node(user).first_node_arg(), Some(out_q));
            assert!(graph.live_nodes().all(|node| !node.is_rescale()));
        }
    }

    #[test]
    fn where_with_mismatched_output_domain_gets_output_rescale() {
        let mut graph = IrGraph::empty();
        let cond = placeholder(&mut graph, "cond", [2], "torch.bool");
        let lhs = placeholder(&mut graph, "lhs", [2], "torch.int8");
        let rhs = placeholder(&mut graph, "rhs", [2], "torch.int8");
        let mut where_meta = NodeMeta::tensor([2], "torch.int8");
        where_meta.set_input_qparams_from([
            (1, qparam(0.5, 3, "torch.int8")),
            (2, qparam(0.5, 3, "torch.int8")),
        ]);
        where_meta.set_single_output_qparam(qparam(0.25, -7, "torch.int8"));
        let where_id = call(
            &mut graph,
            "where",
            "aten.where.self",
            vec![IrArg::Node(cond), IrArg::Node(lhs), IrArg::Node(rhs)],
            where_meta,
        );
        let user = view_user(&mut graph, where_id, "torch.int8");

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
