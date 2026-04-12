use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn is_matmul(target: &str) -> bool {
    target.contains("aten.mm.") || target.contains("aten.bmm.")
}

fn qparam_at(meta: &NodeMeta, index: usize) -> Option<&crate::fx_control_flow_metadata::QParam> {
    meta.input_qparams.as_ref()?.get(&index)
}

fn output_qparam(meta: &NodeMeta) -> Option<&crate::fx_control_flow_metadata::QParam> {
    meta.output_qparams.as_ref()?.get(&0)
}

fn rescale_meta_from(matmul_meta: &NodeMeta, output_dtype: &str) -> NodeMeta {
    let mut meta = matmul_meta.clone();
    meta.set_input_qparams(HashMap::new());
    if let Some(output_qparams) = matmul_meta.output_qparams.clone() {
        meta.set_output_qparams(output_qparams);
    }
    if let Some(shape) = matmul_meta.val_shape() {
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: output_dtype.to_string(),
        });
    }
    meta
}

fn insert_matmul_rescale(graph: &mut IrGraph, matmul_id: NodeId) -> bool {
    let node = &graph.nodes[matmul_id];
    if !node.alive || node.op != NodeOp::CallFunction || !is_matmul(&node.target) {
        return false;
    }

    let Some(qp_a) = qparam_at(&node.meta, 0) else {
        return false;
    };
    let Some(qp_b) = qparam_at(&node.meta, 1) else {
        return false;
    };
    let Some(out_qp) = output_qparam(&node.meta) else {
        return false;
    };

    let rescale_scale = (qp_a.scale * qp_b.scale) / out_qp.scale;
    let rescale_meta = rescale_meta_from(&node.meta, &out_qp.dtype_str);
    let rescale = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_output_rescale", node.name),
        "backend.tosa.RESCALE.default".to_string(),
        vec![
            IrArg::Node(matmul_id),
            IrArg::Dtype(out_qp.dtype_str.clone()),
            IrArg::List(vec![IrArg::Float(rescale_scale)]),
            IrArg::Int(0),
            IrArg::Int(out_qp.zp),
        ],
        rescale_meta,
    );
    graph.anchor_after(rescale, matmul_id);

    let users = graph.live_user_ids(matmul_id);
    let mut rewired = false;
    for user_id in users {
        if user_id == rescale {
            continue;
        }
        rewired |= graph.replace_input_use(user_id, matmul_id, rescale);
    }

    rewired
}

pub struct IrRewriteMatmulRescalePass;

impl IrPass for IrRewriteMatmulRescalePass {
    fn name(&self) -> &str {
        "IrRewriteMatmulRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();
        let mut modified = false;
        for id in ids {
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
    use crate::fx_control_flow_metadata::QParam;
    use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, NodeMeta, NodeOp};

    use super::*;

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

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
            meta(&[2, 3], "torch.int8"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            meta(&[3, 4], "torch.int8"),
        );
        let mut matmul_meta = meta(&[2, 4], "torch.int32");
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
            meta(&[2, 4], "torch.int8"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let rhs = graph.add_node(
            NodeOp::Placeholder,
            "rhs".to_string(),
            "rhs".to_string(),
            Vec::new(),
            meta(&[3, 4], "torch.float32"),
        );
        graph.add_node(
            NodeOp::CallFunction,
            "mm".to_string(),
            "aten.mm.default".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            meta(&[2, 4], "torch.float32"),
        );

        assert!(!IrRewriteMatmulRescalePass.run(&mut graph));
    }
}
