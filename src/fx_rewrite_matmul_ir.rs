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
        let modified = graph.rewrite_live_call_functions(insert_matmul_rescale);
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
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, QParam};
    use crate::test_ir::{call, call_with_meta, f32_call, f32_placeholder, output, placeholder};

    use super::*;

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    #[test]
    fn inserts_output_rescale_and_rewires_existing_users() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [2, 3], "torch.int8");
        let rhs = placeholder(&mut graph, "rhs", [3, 4], "torch.int8");
        let mut matmul_meta = NodeMeta::tensor([2, 4], "torch.int32");
        matmul_meta.set_binary_input_output_qparams(
            qparam(0.5, 3, "torch.int8"),
            qparam(0.25, -4, "torch.int8"),
            qparam(0.125, 7, "torch.int8"),
        );
        let matmul = call_with_meta(
            &mut graph,
            "mm",
            "aten.mm.default",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            matmul_meta,
        );
        let user = call(
            &mut graph,
            "consumer",
            "aten.add.Tensor",
            vec![IrArg::Node(matmul), IrArg::Node(matmul)],
            [2, 4],
            "torch.int8",
        );
        output(&mut graph, user);

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
        let lhs = f32_placeholder(&mut graph, "lhs", [2, 3]);
        let rhs = f32_placeholder(&mut graph, "rhs", [3, 4]);
        f32_call(
            &mut graph,
            "mm",
            "aten.mm.default",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            [2, 4],
        );

        assert!(!IrRewriteMatmulRescalePass.run(&mut graph));
    }
}
