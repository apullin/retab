//! Pure Rust IrPass: `aten.ne.*` -> `aten.eq.*` + `aten.logical_not.default`.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

#[cfg(test)]
use crate::fx_ir::{FakeTensorMeta, NodeMeta};

fn eq_target_for(ne_target: &str) -> Option<&'static str> {
    if ne_target.contains("aten.ne.Scalar") {
        Some("aten.eq.Scalar")
    } else if ne_target.contains("aten.ne.Tensor") || ne_target.contains("aten.ne_.Tensor") {
        Some("aten.eq.Tensor")
    } else {
        None
    }
}

pub struct IrDecomposeNotEqualPass;

impl IrPass for IrDecomposeNotEqualPass {
    fn name(&self) -> &str {
        "IrDecomposeNotEqual"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            let node = &graph.nodes[id];
            if !node.alive || node.op != NodeOp::CallFunction {
                continue;
            }

            let Some(eq_target) = eq_target_for(&node.target) else {
                continue;
            };
            if node.args.len() < 2 {
                continue;
            }

            let name = node.name.clone();
            let meta = node.meta.clone();
            let args = node.args[..2].to_vec();

            let eq = graph.add_node(
                NodeOp::CallFunction,
                format!("{name}_eq"),
                eq_target.to_string(),
                args,
                meta.clone(),
            );
            graph.anchor_to(eq, id);

            let not = graph.add_node(
                NodeOp::CallFunction,
                format!("{name}_logical_not"),
                "aten.logical_not.default".to_string(),
                vec![IrArg::Node(eq)],
                meta,
            );
            graph.anchor_to(not, id);

            graph.replace_and_erase(id, not);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeNotEqual] decomposed aten.ne ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    fn placeholder(graph: &mut IrGraph, name: &str, shape: &[i64], dtype: &str) -> NodeId {
        graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::new(),
            meta(shape, dtype),
        )
    }

    #[test]
    fn decomposes_ne_tensor_to_eq_then_logical_not() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", &[2, 3], "torch.float32");
        let rhs = placeholder(&mut graph, "rhs", &[2, 3], "torch.float32");
        let ne = graph.add_node(
            NodeOp::CallFunction,
            "ne".to_string(),
            "aten.ne.Tensor".to_string(),
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            meta(&[2, 3], "torch.bool"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(ne)],
            NodeMeta::default(),
        );

        assert!(IrDecomposeNotEqualPass.run(&mut graph));
        assert!(!graph.nodes[ne].alive);

        let eq = graph.by_name("ne_eq").unwrap();
        assert_eq!(eq.target, "aten.eq.Tensor");
        assert_eq!(eq.args[0].as_node(), Some(lhs));
        assert_eq!(eq.args[1].as_node(), Some(rhs));
        assert_eq!(eq.meta.val_shape(), Some(&[2, 3][..]));
        assert_eq!(eq.meta.val_dtype(), Some("torch.bool"));

        let not = graph.by_name("ne_logical_not").unwrap();
        assert_eq!(not.target, "aten.logical_not.default");
        assert_eq!(not.args[0].as_node(), Some(eq.id));
        assert_eq!(graph.replacements.get(&ne), Some(&not.id));
    }

    #[test]
    fn decomposes_ne_scalar_to_eq_scalar_then_logical_not() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", &[4], "torch.float32");
        let ne = graph.add_node(
            NodeOp::CallFunction,
            "ne_scalar".to_string(),
            "aten.ne.Scalar".to_string(),
            vec![IrArg::Node(lhs), IrArg::Float(1.0)],
            meta(&[4], "torch.bool"),
        );

        assert!(IrDecomposeNotEqualPass.run(&mut graph));
        assert!(!graph.nodes[ne].alive);

        let eq = graph.by_name("ne_scalar_eq").unwrap();
        assert_eq!(eq.target, "aten.eq.Scalar");
        assert_eq!(eq.args[0].as_node(), Some(lhs));
        assert_eq!(eq.args[1].as_float(), Some(1.0));

        let not = graph.by_name("ne_scalar_logical_not").unwrap();
        assert_eq!(not.target, "aten.logical_not.default");
        assert_eq!(not.args[0].as_node(), Some(eq.id));
    }
}
