use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp, QParam};

pub struct IrFuseConsecutiveConcatsPass;

pub(crate) fn cat_inputs(node: &IrNode) -> Option<Vec<NodeId>> {
    if node.op != NodeOp::CallFunction || node.target != "aten.cat.default" {
        return None;
    }
    let IrArg::List(inputs) = node.args.first()? else {
        return None;
    };
    inputs.iter().map(IrArg::as_node).collect()
}

pub(crate) fn cat_dim(node: &IrNode) -> Option<i64> {
    node.arg_or_kwarg_int(1, "dim")
}

impl IrPass for IrFuseConsecutiveConcatsPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveConcats"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.topological_live_node_ids() {
            let node = graph.node(id);
            let (Some(inputs), Some(dim)) = (cat_inputs(node), cat_dim(node)) else {
                continue;
            };
            let mut new_inputs = Vec::new();
            let mut fused = vec![id];
            for input in inputs {
                let nested = graph.node(input);
                if graph.single_live_user(input) == Some(id) && cat_dim(nested) == Some(dim) {
                    if let Some(nested_inputs) = cat_inputs(nested) {
                        new_inputs.extend(nested_inputs);
                        fused.push(input);
                        continue;
                    }
                }
                new_inputs.push(input);
            }
            if fused.len() == 1 {
                continue;
            }
            let qparams: Vec<&QParam> = fused
                .iter()
                .flat_map(|&id| {
                    let meta = &graph.node(id).meta;
                    [meta.input_qparams(), meta.output_qparams()]
                        .into_iter()
                        .flatten()
                        .flat_map(|map| map.values())
                })
                .collect();
            if qparams.iter().any(|qp| *qp != qparams[0]) {
                continue;
            }
            let common_qparam = qparams.first().copied().cloned();
            let mut args = graph.node(id).args.clone();
            args[0] = IrArg::List(new_inputs.into_iter().map(IrArg::Node).collect());
            graph.set_args(id, args);
            if let Some(qparam) = common_qparam {
                graph.node_mut(id).meta.set_single_input_qparam(qparam);
            }
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

    fn nested_concat(keyword_dim: bool) -> (IrGraph, [NodeId; 3], NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let inputs = ["a", "b", "c"].map(|name| f32_placeholder(&mut graph, name, [1, 2]));
        let inner = f32_call(
            &mut graph,
            "inner",
            "aten.cat.default",
            vec![
                IrArg::List(inputs[..2].iter().copied().map(IrArg::Node).collect()),
                IrArg::Int(1),
            ],
            [1, 4],
        );
        let mut args = vec![IrArg::List(vec![
            IrArg::Node(inner),
            IrArg::Node(inputs[2]),
        ])];
        if !keyword_dim {
            args.push(IrArg::Int(1));
        }
        let outer = f32_call(&mut graph, "outer", "aten.cat.default", args, [1, 6]);
        if keyword_dim {
            graph.set_kwarg(outer, "dim", IrArg::Int(1));
        }
        output(&mut graph, outer);
        (graph, inputs, inner, outer)
    }

    #[test]
    fn flattens_single_use_concat_and_preserves_uniform_qparams() {
        let (mut graph, inputs, inner, outer) = nested_concat(false);
        let qp = qparam(0.25, 3, "torch.int8");
        for id in [inner, outer] {
            graph.node_mut(id).meta.set_both_qparams(qp.clone());
        }
        assert!(IrFuseConsecutiveConcatsPass.run(&mut graph));
        assert!(!graph.node(inner).alive);
        assert_eq!(cat_inputs(graph.node(outer)), Some(inputs.to_vec()));
        assert_eq!(graph.node(outer).meta.input_qparam(), Some(&qp));
        assert_eq!(graph.node(outer).meta.output_qparam(), Some(&qp));
    }

    #[test]
    fn preserves_keyword_dim() {
        let (mut graph, inputs, inner, outer) = nested_concat(true);
        assert!(IrFuseConsecutiveConcatsPass.run(&mut graph));
        assert!(!graph.node(inner).alive);
        assert_eq!(cat_inputs(graph.node(outer)), Some(inputs.to_vec()));
        assert_eq!(graph.node(outer).args.len(), 1);
        assert_eq!(graph.node(outer).kwarg_int("dim"), Some(1));
    }

    #[test]
    fn rejects_mismatched_qparams() {
        let (mut graph, _, inner, outer) = nested_concat(false);
        let qp = qparam(0.25, 3, "torch.int8");
        graph.node_mut(inner).meta.set_both_qparams(qp.clone());
        graph.node_mut(outer).meta.set_both_qparams(qp);
        graph
            .node_mut(outer)
            .meta
            .set_single_output_qparam(qparam(0.5, 3, "torch.int8"));
        assert!(!IrFuseConsecutiveConcatsPass.run(&mut graph));
        assert!(graph.node(inner).alive && graph.node(outer).alive);
    }

    #[test]
    fn rejects_multi_use_or_different_dimensions() {
        for shared in [false, true] {
            let (mut graph, _, inner, outer) = nested_concat(false);
            if shared {
                let out = graph.output_node().unwrap().id;
                graph.set_args(
                    out,
                    vec![IrArg::List(vec![IrArg::Node(inner), IrArg::Node(outer)])],
                );
            } else {
                graph.set_arg(outer, 1, IrArg::Int(0));
            }
            assert!(!IrFuseConsecutiveConcatsPass.run(&mut graph));
            assert!(graph.node(inner).alive && graph.node(outer).alive);
        }
    }
}
