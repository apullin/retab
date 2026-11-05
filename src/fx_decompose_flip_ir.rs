//! IrPass port of ExecuTorch's DecomposeFlipPass (v1.6.0-rc4).
//!
//! TOSA REVERSE reverses a single axis, so a flip over N dims becomes a chain of N
//! single-axis flips, each serialized as one REVERSE. Flip is shared-qparam data
//! movement, so every chained flip keeps the original node's metadata. A flip over no
//! dims is a no-op and is replaced by its input.

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

pub struct IrDecomposeFlipPass;

impl IrPass for IrDecomposeFlipPass {
    fn name(&self) -> &str {
        "IrDecomposeFlip"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            if node.target != "aten.flip.default" {
                return false;
            }
            let (Some(input_id), Some(dims)) = (node.first_node_arg(), node.arg_int_list(1)) else {
                return false;
            };
            if dims.len() == 1 {
                return false;
            }
            let name = node.name.clone();
            let meta = node.meta.clone();
            let mut cursor = graph.rewrite_cursor(id);
            let mut out = input_id;
            for (i, dim) in dims.iter().enumerate() {
                out = cursor.call_function(
                    format!("{name}_axis{i}"),
                    "aten.flip.default",
                    vec![IrArg::Node(out), i64_list_arg(&[*dim])],
                    meta.clone(),
                );
            }
            cursor.replace_anchor_with(out);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrDecomposeFlip] split multi-axis flips into single-axis chains"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, output, placeholder};

    fn flip_graph(dims: &[i64]) -> (IrGraph, crate::fx_ir::NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 2, 3, 4], "torch.int8");
        let flip = call(
            &mut graph,
            "flip",
            "aten.flip.default",
            vec![IrArg::Node(x), i64_list_arg(dims)],
            [1, 2, 3, 4],
            "torch.int8",
        );
        output(&mut graph, flip);
        (graph, flip)
    }

    #[test]
    fn splits_multi_axis_flip_into_single_axis_chain() {
        let (mut graph, flip) = flip_graph(&[1, 3]);
        assert!(IrDecomposeFlipPass.run(&mut graph));
        assert!(!graph.nodes[flip].alive);
        let last = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.nodes[last].arg_int_list(1), Some(vec![3]));
        let first = graph.nodes[last].first_node_arg().unwrap();
        assert_eq!(graph.nodes[first].target, "aten.flip.default");
        assert_eq!(graph.nodes[first].arg_int_list(1), Some(vec![1]));
        assert_eq!(graph.node_shape_vec(last), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn leaves_single_axis_flip_alone() {
        let (mut graph, _) = flip_graph(&[2]);
        assert!(!IrDecomposeFlipPass.run(&mut graph));
    }

    #[test]
    fn empty_flip_is_replaced_by_its_input() {
        let (mut graph, flip) = flip_graph(&[]);
        assert!(IrDecomposeFlipPass.run(&mut graph));
        assert!(!graph.nodes[flip].alive);
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.nodes[out].name, "x");
    }
}
