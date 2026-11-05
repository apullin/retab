use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, IrRewriteCursor, NodeId};

pub struct IrEnsureUniqueOutputNodesPass;

fn replace_repeated_outputs(
    value: IrArg,
    counts: &HashMap<NodeId, usize>,
    cursor: &mut IrRewriteCursor<'_>,
    index: &mut usize,
) -> IrArg {
    match value {
        IrArg::Node(id) if counts[&id] > 1 => {
            let node = cursor.graph().node(id);
            let meta = node.meta.clone().without_qparams();
            let name = format!("{}_output{}", node.name, *index);
            *index += 1;
            IrArg::Node(cursor.call_function(
                name,
                "backend.tosa.IDENTITY.default",
                vec![IrArg::Node(id)],
                meta,
            ))
        }
        IrArg::List(values) => IrArg::List(
            values
                .into_iter()
                .map(|value| replace_repeated_outputs(value, counts, cursor, index))
                .collect(),
        ),
        value => value,
    }
}

impl IrPass for IrEnsureUniqueOutputNodesPass {
    fn name(&self) -> &str {
        "IrEnsureUniqueOutputNodes"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for output in graph.live_output_ids() {
            let args = graph.node(output).args.clone();
            let mut nodes = Vec::new();
            for arg in &args {
                arg.collect_node_ids(&mut nodes);
            }
            let mut counts = HashMap::new();
            for id in nodes {
                *counts.entry(id).or_insert(0) += 1;
            }
            if counts.values().all(|&count| count == 1) {
                continue;
            }
            let mut cursor = graph.rewrite_cursor(output);
            let mut index = 0;
            let args = args
                .into_iter()
                .map(|arg| replace_repeated_outputs(arg, &counts, &mut cursor, &mut index))
                .collect();
            cursor.graph_mut().set_args(output, args);
            modified = true;
        }
        if modified {
            graph.eliminate_dead_code();
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{f32_call, f32_placeholder, output, qparam};

    #[test]
    fn inserts_identity_for_each_repeated_output() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2]);
        let y = f32_call(
            &mut graph,
            "y",
            "aten.add.Scalar",
            vec![IrArg::Node(x), IrArg::Float(1.0)],
            [2, 2],
        );
        graph
            .node_mut(y)
            .meta
            .set_both_qparams(qparam(0.25, 3, "torch.int8"));
        let out = output(&mut graph, y);
        graph.set_args(out, vec![IrArg::List(vec![IrArg::Node(y), IrArg::Node(y)])]);
        assert!(IrEnsureUniqueOutputNodesPass.run(&mut graph));
        let IrArg::List(outputs) = &graph.node(out).args[0] else {
            panic!("output list")
        };
        let first = outputs[0].as_node().unwrap();
        let second = outputs[1].as_node().unwrap();
        assert_ne!(first, second);
        for id in [first, second] {
            let identity = graph.node(id);
            assert_eq!(identity.target, "backend.tosa.IDENTITY.default");
            assert_eq!(identity.arg_node(0), Some(y));
            assert_eq!(graph.node_shape(id), Some([2, 2].as_slice()));
            assert!(!identity.meta.has_nonempty_qparam_metadata());
        }
        assert!(!IrEnsureUniqueOutputNodesPass.run(&mut graph));
    }

    #[test]
    fn keeps_unique_outputs_unchanged() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2]);
        let y = f32_call(
            &mut graph,
            "y",
            "aten.add.Scalar",
            vec![IrArg::Node(x), IrArg::Float(1.0)],
            [2, 2],
        );
        let z = f32_call(
            &mut graph,
            "z",
            "aten.add.Scalar",
            vec![IrArg::Node(x), IrArg::Float(2.0)],
            [2, 2],
        );
        let out = output(&mut graph, y);
        graph.set_args(out, vec![IrArg::List(vec![IrArg::Node(y), IrArg::Node(z)])]);
        assert!(!IrEnsureUniqueOutputNodesPass.run(&mut graph));
        assert_eq!(graph.live_call_function_ids(), vec![y, z]);
    }

    #[test]
    fn preserves_nested_output_structure_and_non_node_leaves() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2]);
        let y = f32_placeholder(&mut graph, "y", [2]);
        let out = output(&mut graph, x);
        graph.set_args(
            out,
            vec![IrArg::List(vec![
                IrArg::Node(x),
                IrArg::List(vec![IrArg::Node(y), IrArg::Node(x), IrArg::None]),
            ])],
        );
        assert!(IrEnsureUniqueOutputNodesPass.run(&mut graph));
        let IrArg::List(outputs) = &graph.node(out).args[0] else {
            panic!("output list")
        };
        let IrArg::List(nested) = &outputs[1] else {
            panic!("nested list")
        };
        assert_eq!(nested[0].as_node(), Some(y));
        assert!(matches!(nested[2], IrArg::None));
        assert_ne!(outputs[0].as_node(), nested[1].as_node());
    }
}
