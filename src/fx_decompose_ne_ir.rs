//! Pure Rust IrPass: `aten.ne.*` -> `aten.eq.*` + `aten.logical_not.default`.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::not_equal_replacement_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeNotEqualPass;

impl IrPass for IrDecomposeNotEqualPass {
    fn name(&self) -> &str {
        "IrDecomposeNotEqual"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            let Some(eq_target) = not_equal_replacement_target(&node.target) else {
                return false;
            };
            if node.args.len() < 2 {
                return false;
            }

            let name = node.name.clone();
            let meta = node.meta.clone();
            let args = node.args[..2].to_vec();
            let mut cursor = graph.rewrite_cursor(id);

            let eq = cursor.call_function(format!("{name}_eq"), eq_target, args, meta.clone());

            let result = cursor.call_function(
                format!("{name}_logical_not"),
                "aten.logical_not.default",
                vec![IrArg::Node(eq)],
                meta,
            );
            cursor.replace_anchor_with(result);
            true
        });

        if modified {
            log_line(format_args!("[IrDecomposeNotEqual] decomposed aten.ne ops"));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, output, placeholder};

    #[test]
    fn decomposes_ne_tensor_to_eq_then_logical_not() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [2, 3], "torch.float32");
        let rhs = placeholder(&mut graph, "rhs", [2, 3], "torch.float32");
        let ne = call(
            &mut graph,
            "ne",
            "aten.ne.Tensor",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            [2, 3],
            "torch.bool",
        );
        output(&mut graph, ne);

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
        assert_eq!(graph.replacement_of(ne), Some(not.id));
    }

    #[test]
    fn decomposes_ne_scalar_to_eq_scalar_then_logical_not() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [4], "torch.float32");
        let ne = call(
            &mut graph,
            "ne_scalar",
            "aten.ne.Scalar",
            vec![IrArg::Node(lhs), IrArg::Float(1.0)],
            [4],
            "torch.bool",
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
