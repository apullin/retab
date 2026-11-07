use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp};

pub struct IrFuseConsecutiveClampsPass;

fn clamp_input(node: &IrNode) -> Option<NodeId> {
    node.args
        .first()
        .or_else(|| node.kwargs.get("self"))?
        .as_node()
}

fn clamp_bound(node: &IrNode, index: usize, key: &str) -> Option<Option<f64>> {
    match node.args.get(index).or_else(|| node.kwargs.get(key)) {
        None | Some(IrArg::None) => Some(None),
        Some(arg) => arg.as_numeric_f64().map(Some),
    }
}

impl IrPass for IrFuseConsecutiveClampsPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveClamps"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        loop {
            let mut changed = false;
            for first in graph.topological_live_node_ids() {
                let node = graph.node(first);
                if !node.alive
                    || node.op != NodeOp::CallFunction
                    || node.target != "aten.clamp.default"
                {
                    continue;
                }
                let Some(second) = graph.single_live_user(first) else {
                    continue;
                };
                let user = graph.node(second);
                if user.op != NodeOp::CallFunction
                    || user.target != "aten.clamp.default"
                    || clamp_input(user) != Some(first)
                {
                    continue;
                }
                let Some(input) = clamp_input(node) else {
                    continue;
                };
                let (Some(min1), Some(max1), Some(min2), Some(max2)) = (
                    clamp_bound(node, 1, "min"),
                    clamp_bound(node, 2, "max"),
                    clamp_bound(user, 1, "min"),
                    clamp_bound(user, 2, "max"),
                ) else {
                    continue;
                };
                let min = match (min1, min2) {
                    (Some(a), Some(b)) => Some(if b > a { b } else { a }),
                    (a, b) => a.or(b),
                };
                let max = match (max1, max2) {
                    (Some(a), Some(b)) => Some(if b < a { b } else { a }),
                    (a, b) => a.or(b),
                };
                if min.zip(max).is_some_and(|(min, max)| min > max) {
                    continue;
                }
                let input_qparams = node.meta.cloned_input_qparams();
                graph.set_args(
                    second,
                    vec![
                        IrArg::Node(input),
                        min.map(IrArg::Float).unwrap_or(IrArg::None),
                        max.map(IrArg::Float).unwrap_or(IrArg::None),
                    ],
                );
                for key in ["min", "max", "self"] {
                    graph.remove_kwarg(second, key);
                }
                let survivor = graph.node_mut(second);
                if let Some(qparams) = input_qparams.filter(|qparams| !qparams.is_empty()) {
                    if survivor.meta.input_qparams().is_some() {
                        survivor.meta.set_input_qparams(qparams);
                    }
                }
                graph.erase(first);
                changed = true;
            }
            modified |= changed;
            if !changed {
                break;
            }
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
    use crate::fx_convert_clamp_ir::IrConvertClampPass;
    use crate::test_ir::{f32_call, f32_placeholder, output, qparam};

    #[test]
    fn fuses_rc4_activation_and_explicit_clamp_pairs() {
        for (first_target, first_bounds, second_target, second_bounds, expected) in [
            (
                "aten.hardtanh.default",
                [-1.0, 1.0],
                "aten.relu.default",
                [0.0, f64::INFINITY],
                [0.0, 1.0],
            ),
            (
                "aten.relu.default",
                [0.0, f64::INFINITY],
                "aten.relu.default",
                [0.0, f64::INFINITY],
                [0.0, f64::INFINITY],
            ),
            (
                "aten.hardtanh.default",
                [-2.0, 2.0],
                "aten.hardtanh.default",
                [-1.0, 3.0],
                [-1.0, 2.0],
            ),
            (
                "aten.hardtanh.default",
                [0.0, 6.0],
                "aten.relu.default",
                [0.0, f64::INFINITY],
                [0.0, 6.0],
            ),
            (
                "aten.clamp.default",
                [-2.0, 2.0],
                "aten.clamp.default",
                [-1.0, 3.0],
                [-1.0, 2.0],
            ),
        ] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [1, 8, 8, 3]);
            let mut current = x;
            for (name, target, bounds) in [
                ("first", first_target, first_bounds),
                ("second", second_target, second_bounds),
            ] {
                let mut args = vec![IrArg::Node(current)];
                if target != "aten.relu.default" {
                    args.extend(bounds.map(IrArg::Float));
                }
                current = f32_call(&mut graph, name, target, args, [1, 8, 8, 3]);
            }
            let out = output(&mut graph, current);
            IrConvertClampPass.run(&mut graph);
            assert_eq!(
                graph
                    .live_nodes()
                    .filter(|n| n.target == "aten.clamp.default")
                    .count(),
                2
            );
            assert!(IrFuseConsecutiveClampsPass.run(&mut graph));
            let clamp = graph.node(graph.node(out).arg_node(0).unwrap());
            assert_eq!(clamp.arg_node(0), Some(x));
            assert_eq!(
                clamp.arg(1).and_then(IrArg::as_numeric_f64),
                Some(expected[0])
            );
            assert_eq!(
                clamp.arg(2).and_then(IrArg::as_numeric_f64),
                expected[1].is_finite().then_some(expected[1])
            );
            assert_eq!(
                graph
                    .live_nodes()
                    .filter(|n| n.target == "aten.clamp.default")
                    .count(),
                1
            );
        }
    }

    #[test]
    fn three_clamps_collapse_with_first_input_and_last_output_qparams() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3]);
        let mut current = x;
        let input_qp = qparam(0.25, 3, "torch.int8");
        let output_qp = qparam(0.5, -7, "torch.int8");
        let mut first = None;
        for (name, min, max) in [
            ("hardtanh", -1.0, Some(1.0)),
            ("relu", 0.0, None),
            ("relu2", 0.0, None),
        ] {
            current = f32_call(
                &mut graph,
                name,
                "aten.clamp.default",
                vec![
                    IrArg::Node(current),
                    IrArg::Float(min),
                    max.map(IrArg::Float).unwrap_or(IrArg::None),
                ],
                [2, 3],
            );
            graph
                .node_mut(current)
                .meta
                .set_both_qparams(output_qp.clone());
            if first.is_none() {
                graph
                    .node_mut(current)
                    .meta
                    .set_single_input_qparam(input_qp.clone());
                first = Some(current);
            }
        }
        output(&mut graph, current);
        assert!(IrFuseConsecutiveClampsPass.run(&mut graph));
        let survivor = graph.node(current);
        assert_eq!(survivor.arg_node(0), Some(x));
        assert_eq!(survivor.arg(1).and_then(IrArg::as_numeric_f64), Some(0.0));
        assert_eq!(survivor.arg(2).and_then(IrArg::as_numeric_f64), Some(1.0));
        assert!(survivor
            .meta
            .input_qparam()
            .unwrap()
            .same_exact_affine_qrange_dtype(&input_qp));
        assert!(survivor
            .meta
            .output_qparam()
            .unwrap()
            .same_exact_affine_qrange_dtype(&output_qp));
    }

    #[test]
    fn conflicting_bounds_and_branching_clamps_are_kept() {
        for branching in [false, true] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [2]);
            let first = f32_call(
                &mut graph,
                "first",
                "aten.clamp.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Float(if branching { -1.0 } else { 5.0 }),
                    IrArg::Float(if branching { 1.0 } else { 10.0 }),
                ],
                [2],
            );
            let second = f32_call(
                &mut graph,
                "second",
                "aten.clamp.default",
                vec![IrArg::Node(first), IrArg::Float(0.0), IrArg::Float(3.0)],
                [2],
            );
            let out = output(&mut graph, second);
            if branching {
                graph.set_args(
                    out,
                    vec![IrArg::List(vec![IrArg::Node(first), IrArg::Node(second)])],
                );
            }
            assert!(!IrFuseConsecutiveClampsPass.run(&mut graph));
            assert!(graph.node(first).alive && graph.node(second).alive);
        }
    }

    #[test]
    fn composes_keyword_bounds_and_canonicalizes_survivor() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2]);
        let first = f32_call(
            &mut graph,
            "first",
            "aten.clamp.default",
            vec![IrArg::Node(x)],
            [2],
        );
        graph.set_kwarg(first, "min", IrArg::Float(-2.0));
        graph.set_kwarg(first, "max", IrArg::Float(2.0));
        let second = f32_call(
            &mut graph,
            "second",
            "aten.clamp.default",
            vec![IrArg::Node(first)],
            [2],
        );
        graph.set_kwarg(second, "min", IrArg::Float(-1.0));
        graph.set_kwarg(second, "max", IrArg::Float(3.0));
        output(&mut graph, second);
        assert!(IrFuseConsecutiveClampsPass.run(&mut graph));
        assert!(graph.node(second).kwargs.is_empty());
        assert_eq!(
            graph.node(second).arg(1).and_then(IrArg::as_numeric_f64),
            Some(-1.0)
        );
        assert_eq!(
            graph.node(second).arg(2).and_then(IrArg::as_numeric_f64),
            Some(2.0)
        );
    }
}
