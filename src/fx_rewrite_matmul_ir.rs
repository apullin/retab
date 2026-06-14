use crate::fx_ir::{IrGraph, IrPass, NodeId, NodeOp, RescaleRewrite};
use crate::runner_logging::log_line;

fn insert_matmul_rescale(graph: &mut IrGraph, matmul_id: NodeId) -> bool {
    let node = &graph.nodes[matmul_id];
    if !node.alive || node.op != NodeOp::CallFunction || !node.is_mm_or_bmm() {
        return false;
    }

    let Some(qp_a) = node.meta.input_qparam_at(0) else {
        return false;
    };
    let Some(qp_b) = node.meta.input_qparam_at(1) else {
        return false;
    };
    let Some(out_qp) = node.meta.output_qparam() else {
        return false;
    };

    let rescale_scale = (qp_a.scale * qp_b.scale) / out_qp.scale;
    let rescale_meta = node.meta.rescale_output_meta(out_qp.clone());
    let rescale = graph.add_after_rescale(
        matmul_id,
        RescaleRewrite::new(
            format!("{}_output_rescale", node.name),
            matmul_id,
            out_qp.dtype_str.clone(),
            vec![rescale_scale],
            0,
            out_qp.zp,
            rescale_meta,
        ),
    );
    graph.replace_live_uses_except(matmul_id, rescale, rescale)
}

pub struct IrRewriteMatmulRescalePass;

impl IrPass for IrRewriteMatmulRescalePass {
    fn name(&self) -> &str {
        "IrRewriteMatmulRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= insert_matmul_rescale(graph, id);
        }
        if modified {
            log_line(format_args!(
                "[IrRewriteMatmulRescale] inserted matmul output rescale ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, NodeOp, QParam};
    use std::collections::HashMap;

    use super::*;

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    #[test]
    fn inserts_output_rescale_and_rewires_existing_users() {
        let mut graph = IrGraph::empty();
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([2, 3], "torch.int8"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([3, 4], "torch.int8"),
        );
        let mut matmul_meta = NodeMeta::tensor([2, 4], "torch.int32");
        let mut input_qparams = HashMap::new();
        input_qparams.insert(0, qparam(0.5, 3, "torch.int8"));
        input_qparams.insert(1, qparam(0.25, -4, "torch.int8"));
        matmul_meta.set_input_qparams(input_qparams);
        let mut output_qparams = HashMap::new();
        output_qparams.insert(0, qparam(0.125, 7, "torch.int8"));
        matmul_meta.set_output_qparams(output_qparams);
        let matmul = graph.add_node(
            NodeOp::CallFunction,
            "mm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            matmul_meta,
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "consumer".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(matmul), IrArg::Node(matmul)],
            NodeMeta::tensor([2, 4], "torch.int8"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(user)],
            NodeMeta::default(),
        );

        assert!(IrRewriteMatmulRescalePass.run(&mut graph));
        let rescale = graph.by_name("mm_output_rescale").unwrap();
        assert_eq!(rescale.target, "backend.tosa.RESCALE.default");
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == matmul));
        assert!(matches!(rescale.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(
            matches!(rescale.args[2], IrArg::List(ref scales) if matches!(scales.as_slice(), [IrArg::Float(scale)] if (*scale - 1.0).abs() < 1e-12))
        );
        assert!(matches!(rescale.args[3], IrArg::Int(0)));
        assert!(matches!(rescale.args[4], IrArg::Int(7)));
        assert_eq!(rescale.meta.val_dtype(), Some("torch.int8"));

        let user = graph.by_name("consumer").unwrap();
        let rescale_id = rescale.id;
        assert!(matches!(user.args[0], IrArg::Node(id) if id == rescale_id));
        assert!(matches!(user.args[1], IrArg::Node(id) if id == rescale_id));
    }

    #[test]
    fn skips_matmul_without_quantization_metadata() {
        let mut graph = IrGraph::empty();
        let lhs = graph.add_node(
            NodeOp::Placeholder,
            "lhs".to_string(),
            "lhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([2, 3], "torch.float32"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            NodeMeta::tensor([3, 4], "torch.float32"),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "mm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            NodeMeta::tensor([2, 4], "torch.float32"),
        );

        assert!(!IrRewriteMatmulRescalePass.run(&mut graph));
    }
}
