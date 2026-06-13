use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

fn insert_qdq_rescale(graph: &mut IrGraph, dq_id: NodeId, q_id: NodeId) -> bool {
    let dq = graph.node(dq_id);
    let q = graph.node(q_id);
    if !dq.alive
        || !q.alive
        || dq.op != NodeOp::CallFunction
        || q.op != NodeOp::CallFunction
        || !dq.is_dequantize()
        || !q.is_quantize()
    {
        return false;
    }

    let Some(source_id) = dq.args.first().and_then(IrArg::as_node) else {
        return false;
    };
    let Some(dq_qp) = dq.parse_qdq_qparam() else {
        return false;
    };
    let Some(q_qp) = q.parse_qdq_qparam() else {
        return false;
    };
    if q_qp.scale == 0.0 {
        return false;
    }

    let scale = dq_qp.scale / q_qp.scale;
    let meta = q.meta.rescale_meta_with_input_qparam(
        &graph.node(source_id).meta,
        dq_qp.clone(),
        q_qp.clone(),
    );
    let rescale = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_rescale", q.name),
        "backend.tosa.RESCALE.default".to_string(),
        vec![
            IrArg::Node(source_id),
            IrArg::Dtype(q_qp.dtype_str.clone()),
            IrArg::List(vec![IrArg::Float(scale)]),
            IrArg::Int(dq_qp.zp),
            IrArg::Int(q_qp.zp),
        ],
        meta,
    );
    graph.anchor_to(rescale, dq_id);

    graph.replace_and_erase(q_id, rescale);
    if !graph.has_live_users(dq_id) {
        graph.erase(dq_id);
    }

    true
}

pub struct IrInsertQdqRescalePass;

impl IrPass for IrInsertQdqRescalePass {
    fn name(&self) -> &str {
        "IrInsertQdqRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut pairs = Vec::new();
        for node in &graph.nodes {
            if !node.alive || node.op != NodeOp::CallFunction || !node.is_dequantize() {
                continue;
            }
            for user_id in graph.live_user_ids(node.id) {
                let user = graph.node(user_id);
                if user.alive && user.op == NodeOp::CallFunction && user.is_quantize() {
                    pairs.push((node.id, user_id));
                }
            }
        }

        let mut modified = false;
        for (dq_id, q_id) in pairs {
            modified |= insert_qdq_rescale(graph, dq_id, q_id);
        }
        if modified {
            log_line(format_args!(
                "[IrInsertQdqRescale] folded DQ->Q pairs into RESCALE ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeMeta;

    #[test]
    fn folds_dq_q_pair_into_rescale() {
        let mut graph = IrGraph::empty();
        let source = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1, 3, 4, 4], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![
                IrArg::Node(source),
                IrArg::Float(0.5),
                IrArg::Int(3),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::tensor([1, 3, 4, 4], "torch.float32"),
        );
        let q = graph.add_node(
            NodeOp::CallFunction,
            "q".to_string(),
            "quantized_decomposed.quantize_per_tensor.default".to_string(),
            vec![
                IrArg::Node(dq),
                IrArg::Float(0.25),
                IrArg::Int(-7),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::tensor([1, 3, 4, 4], "torch.int8"),
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(q)],
            NodeMeta::tensor([1, 48], "torch.int8"),
        );

        assert!(IrInsertQdqRescalePass.run(&mut graph));

        assert!(!graph.node(dq).alive);
        assert!(!graph.node(q).alive);

        let rescale = graph.by_name("q_rescale").unwrap();
        assert_eq!(rescale.target, "backend.tosa.RESCALE.default");
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == source));
        assert!(matches!(rescale.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(matches!(
            rescale.args[2],
            IrArg::List(ref scales)
                if matches!(scales.as_slice(), [IrArg::Float(scale)] if (*scale - 2.0).abs() < 1e-12)
        ));
        assert!(matches!(rescale.args[3], IrArg::Int(3)));
        assert!(matches!(rescale.args[4], IrArg::Int(-7)));
        assert_eq!(rescale.meta.val_dtype(), Some("torch.int8"));

        let user = graph.node(user);
        assert!(matches!(user.args[0], IrArg::Node(id) if id == rescale.id));
    }

    #[test]
    fn skips_non_q_user() {
        let mut graph = IrGraph::empty();
        let source = graph.add_node(
            NodeOp::Placeholder,
            "input".to_string(),
            "input".to_string(),
            Vec::new(),
            NodeMeta::tensor([1], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![
                IrArg::Node(source),
                IrArg::Float(0.5),
                IrArg::Int(0),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::tensor([1], "torch.float32"),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.relu.default".to_string(),
            vec![IrArg::Node(dq)],
            NodeMeta::tensor([1], "torch.float32"),
        );

        assert!(!IrInsertQdqRescalePass.run(&mut graph));
    }
}
