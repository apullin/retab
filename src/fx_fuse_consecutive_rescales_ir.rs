use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

const ARG_INPUT: usize = 0;
const ARG_OUTPUT_DTYPE: usize = 1;
const ARG_SCALE: usize = 2;
const ARG_INPUT_ZP: usize = 3;
const ARG_OUTPUT_ZP: usize = 4;

#[derive(Debug, Clone, Copy)]
struct RescaleSpec {
    input: NodeId,
    output_dtype: DType,
    scale: f64,
    input_zp: i64,
    output_zp: i64,
}

fn parse_per_tensor_rescale(graph: &IrGraph, id: NodeId) -> Option<RescaleSpec> {
    let node = graph.node(id);
    if !node.alive || !node.is_rescale() {
        return None;
    }
    let output_dtype = match node.args.get(ARG_OUTPUT_DTYPE)? {
        IrArg::Dtype(dtype) => DType::from_torch_str(dtype),
        _ => return None,
    };
    let scale = match node.args.get(ARG_SCALE)? {
        IrArg::List(items) if items.len() == 1 => items[0].as_numeric_f64()?,
        _ => return None,
    };
    Some(RescaleSpec {
        input: node.args.get(ARG_INPUT)?.as_node()?,
        output_dtype,
        scale,
        input_zp: node.args.get(ARG_INPUT_ZP)?.as_int()?,
        output_zp: node.args.get(ARG_OUTPUT_ZP)?.as_int()?,
    })
}

fn is_fusable_r1(graph: &IrGraph, id: NodeId, spec: &RescaleSpec) -> bool {
    matches!(spec.output_dtype, DType::Int8 | DType::Int16)
        && graph
            .node(spec.input)
            .meta
            .val_dtype_kind()
            .is_some_and(|dtype| dtype == DType::Int32)
        && graph.node(id).alive
}

fn is_identity_r2(r1: &RescaleSpec, r2: &RescaleSpec) -> bool {
    r2.output_dtype == DType::Int32
        && r1.output_zp == r2.input_zp
        && r1.input_zp == r2.output_zp
        && (r1.scale * r2.scale - 1.0).abs() < 1e-6
}

fn fuse_r1_users(graph: &mut IrGraph, r1_id: NodeId, r1: RescaleSpec) -> usize {
    let users = graph.live_user_ids(r1_id);
    let mut fused = 0;
    for r2_id in users {
        let Some(r2) = parse_per_tensor_rescale(graph, r2_id) else {
            continue;
        };
        if is_identity_r2(&r1, &r2) {
            graph.replace_and_erase(r2_id, r1.input);
            fused += 1;
        }
    }
    if fused > 0 && !graph.has_live_users(r1_id) {
        graph.erase(r1_id);
    }
    fused
}

pub struct IrFuseConsecutiveRescalesPass;

impl IrPass for IrFuseConsecutiveRescalesPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveRescales"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut fused = 0;
        for r1_id in graph.live_call_function_ids() {
            let Some(r1) = parse_per_tensor_rescale(graph, r1_id) else {
                continue;
            };
            if is_fusable_r1(graph, r1_id, &r1) {
                fused += fuse_r1_users(graph, r1_id, r1);
            }
        }
        if fused > 0 {
            log_line(format_args!(
                "[IrFuseConsecutiveRescales] fused {fused} identity RESCALE pair(s)"
            ));
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::RescaleRewrite;
    use crate::test_ir::{call, placeholder};

    fn qparam(scale: f64, zp: i64, dtype: &str) -> crate::fx_ir::QParam {
        crate::fx_ir::QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    fn rescale(
        graph: &mut IrGraph,
        name: &str,
        source: NodeId,
        dtype: &str,
        scale: f64,
        input_zp: i64,
        output_zp: i64,
    ) -> NodeId {
        let output_qparam = qparam(scale, output_zp, dtype);
        let meta = graph.node(source).meta.rescale_output_meta(output_qparam);
        graph.add_rescale(RescaleRewrite::new(
            name.to_string(),
            source,
            dtype,
            vec![scale],
            input_zp,
            output_zp,
            meta,
        ))
    }

    #[test]
    fn fuses_identity_int32_int8_int32_pair() {
        let mut graph = IrGraph::empty();
        let src = placeholder(&mut graph, "src", [2], "torch.int32");
        graph
            .node_mut(src)
            .meta
            .set_single_output_qparam(qparam(0.25, 0, "torch.int32"));
        let r1 = rescale(&mut graph, "r1", src, "torch.int8", 0.5, 0, -7);
        let r2 = rescale(&mut graph, "r2", r1, "torch.int32", 2.0, -7, 0);
        let user = call(
            &mut graph,
            "user",
            "aten.add.Tensor",
            vec![IrArg::Node(r2), IrArg::Node(src)],
            [2],
            "torch.int32",
        );

        assert!(IrFuseConsecutiveRescalesPass.run(&mut graph));

        assert!(!graph.node(r1).alive);
        assert!(!graph.node(r2).alive);
        assert_eq!(graph.node(user).args[0].as_node(), Some(src));
    }

    #[test]
    fn preserves_non_identity_pair() {
        let mut graph = IrGraph::empty();
        let src = placeholder(&mut graph, "src", [2], "torch.int32");
        let r1 = rescale(&mut graph, "r1", src, "torch.int8", 0.5, 0, -7);
        let r2 = rescale(&mut graph, "r2", r1, "torch.int32", 1.5, -7, 0);
        let user = call(
            &mut graph,
            "user",
            "aten.add.Tensor",
            vec![IrArg::Node(r2), IrArg::Node(src)],
            [2],
            "torch.int32",
        );

        assert!(!IrFuseConsecutiveRescalesPass.run(&mut graph));

        assert!(graph.node(r1).alive);
        assert!(graph.node(r2).alive);
        assert_eq!(graph.node(user).args[0].as_node(), Some(r2));
    }

    #[test]
    fn preserves_r1_when_it_has_other_live_users() {
        let mut graph = IrGraph::empty();
        let src = placeholder(&mut graph, "src", [2], "torch.int32");
        let r1 = rescale(&mut graph, "r1", src, "torch.int8", 0.5, 0, -7);
        let r2 = rescale(&mut graph, "r2", r1, "torch.int32", 2.0, -7, 0);
        let direct_user = call(
            &mut graph,
            "direct_user",
            "aten.view_copy.default",
            vec![IrArg::Node(r1)],
            [2],
            "torch.int8",
        );
        let int32_user = call(
            &mut graph,
            "int32_user",
            "aten.add.Tensor",
            vec![IrArg::Node(r2), IrArg::Node(src)],
            [2],
            "torch.int32",
        );

        assert!(IrFuseConsecutiveRescalesPass.run(&mut graph));

        assert!(graph.node(r1).alive);
        assert!(!graph.node(r2).alive);
        assert_eq!(graph.node(direct_user).args[0].as_node(), Some(r1));
        assert_eq!(graph.node(int32_user).args[0].as_node(), Some(src));
    }
}
