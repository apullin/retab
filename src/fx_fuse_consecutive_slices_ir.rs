use crate::fx_ir::{i64_list_arg, normalize_dim, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::{is_slice_tensor_target, is_tosa_slice_target};

pub struct IrFuseConsecutiveSlicesPass;

pub(crate) struct StaticSlice {
    pub input: NodeId,
    pub start: Vec<i64>,
    pub size: Vec<i64>,
}

pub(crate) fn static_slice(graph: &IrGraph, id: NodeId) -> Option<StaticSlice> {
    let node = graph.node(id);
    if !node.alive || node.op != NodeOp::CallFunction {
        return None;
    }
    let input = node.arg_node(0)?;
    if is_tosa_slice_target(&node.target) {
        let start = graph.constant_shape(node.arg(1)?)?;
        let size = graph.constant_shape(node.arg(2)?)?;
        return Some(StaticSlice { input, start, size });
    }
    if !is_slice_tensor_target(&node.target)
        || node.args.len() < 4
        || node.arg(4).is_some_and(|step| step.as_int() != Some(1))
    {
        return None;
    }
    // SizeAdjustInput's intermediate slices retain the original input metadata.
    let mut size = if let Some(slice) = static_slice(graph, input) {
        graph.shape_in_node_order(id, input, &slice.size)?
    } else {
        graph.logical_input_shape(id, input)?
    };
    let dim = normalize_dim(node.arg_int(1)?, size.len())?;
    let dim_size = size[dim];
    if size.iter().any(|&dim| dim <= 0) {
        return None;
    }
    let normalize_bound = |arg: &IrArg, default: i64| match arg {
        IrArg::None => Some(default),
        IrArg::Int(bound) => {
            Some(if *bound < 0 { dim_size + bound } else { *bound }.clamp(0, dim_size))
        }
        _ => None,
    };
    let start_index = normalize_bound(node.arg(2)?, 0)?;
    let end_index = normalize_bound(node.arg(3)?, dim_size)?;
    size[dim] = (end_index - start_index).max(0);
    if size[dim] == 0 {
        return None;
    }
    let mut start = vec![0; size.len()];
    start[dim] = start_index;
    Some(StaticSlice { input, start, size })
}

impl IrPass for IrFuseConsecutiveSlicesPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveSlices"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.topological_live_node_ids() {
            if !is_tosa_slice_target(&graph.node(id).target) {
                continue;
            }
            while let Some(second) = static_slice(graph, id) {
                let first_node = graph.node(second.input);
                let first = if first_node.alive
                    && first_node.op == NodeOp::CallFunction
                    && is_tosa_slice_target(&first_node.target)
                {
                    first_node
                        .arg_node(0)
                        .zip(first_node.arg(1).and_then(|arg| graph.constant_shape(arg)))
                } else {
                    None
                };
                let Some((input, first_start)) = first else {
                    break;
                };
                let Some(first_start) = graph.shape_in_node_order(id, second.input, &first_start)
                else {
                    break;
                };
                if first_start.len() != second.start.len()
                    || second.start.len() != second.size.len()
                {
                    break;
                }
                let Some(start) = first_start
                    .iter()
                    .zip(&second.start)
                    .map(|(&a, &b)| a.checked_add(b))
                    .collect::<Option<Vec<_>>>()
                else {
                    break;
                };
                graph.set_args(
                    id,
                    vec![
                        IrArg::Node(input),
                        i64_list_arg(&start),
                        i64_list_arg(&second.size),
                    ],
                );
                let node = graph.node_mut(id);
                node.target = "backend.tosa.SLICE.default".into();
                node.meta = node.meta.cloned_with_same_dtype_or_float32(second.size);
                modified = true;
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
    use crate::fx_ir::NodeMeta;
    use crate::test_ir::{call_default, f32_call, f32_placeholder, output, qparam};

    fn tosa_slice(
        graph: &mut IrGraph,
        name: &str,
        input: NodeId,
        start: &[i64],
        size: &[i64],
    ) -> NodeId {
        let start_node = call_default(
            graph,
            &format!("{name}_start"),
            "backend.tosa.CONST_SHAPE.default",
            vec![i64_list_arg(start)],
        );
        let size_node = call_default(
            graph,
            &format!("{name}_size"),
            "backend.tosa.CONST_SHAPE.default",
            vec![i64_list_arg(size)],
        );
        f32_call(
            graph,
            name,
            "backend.tosa.SLICE.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(start_node),
                IrArg::Node(size_node),
            ],
            size,
        )
    }

    #[test]
    fn combines_static_starts_and_removes_unused_shape_operands() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 10]);
        let first = tosa_slice(&mut graph, "first", x, &[0, 1, 0], &[1, 6, 10]);
        let second = tosa_slice(&mut graph, "second", first, &[0, 2, 3], &[1, 2, 4]);
        let qp = qparam(0.5, 3, "torch.int8");
        graph.node_mut(second).meta.set_both_qparams(qp.clone());
        output(&mut graph, second);
        assert!(IrFuseConsecutiveSlicesPass.run(&mut graph));
        assert_eq!(graph.node(second).arg_node(0), Some(x));
        assert_eq!(graph.node(second).arg_int_list(1), Some(vec![0, 3, 3]));
        assert_eq!(graph.node(second).arg_int_list(2), Some(vec![1, 2, 4]));
        assert_eq!(graph.node(second).meta.input_qparam(), Some(&qp));
        assert_eq!(graph.node(second).meta.output_qparam(), Some(&qp));
        assert_eq!(graph.live_call_function_ids(), vec![second]);
    }

    #[test]
    fn consecutive_aten_slices_become_one_multidimensional_tosa_slice() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 10]);
        let first = f32_call(
            &mut graph,
            "first",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(x),
                IrArg::Int(1),
                IrArg::Int(1),
                IrArg::Int(7),
                IrArg::Int(1),
            ],
            [1, 6, 10],
        );
        let second = f32_call(
            &mut graph,
            "second",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(first),
                IrArg::Int(2),
                IrArg::Int(2),
                IrArg::Int(5),
                IrArg::Int(1),
            ],
            [1, 6, 3],
        );
        output(&mut graph, second);
        assert!(crate::fx_rewrite_slice_ir::IrRewriteSlicePass.run(&mut graph));
        assert!(IrFuseConsecutiveSlicesPass.run(&mut graph));
        assert_eq!(graph.node(second).target, "backend.tosa.SLICE.default");
        assert_eq!(graph.node(second).arg_node(0), Some(x));
        assert_eq!(graph.node(second).arg_int_list(1), Some(vec![0, 1, 2]));
        assert_eq!(graph.node(second).arg_int_list(2), Some(vec![1, 6, 3]));
        assert!(!graph.node(first).alive);
    }

    #[test]
    fn preserves_a_shared_first_slice_and_fuses_a_three_slice_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 10]);
        let first = tosa_slice(&mut graph, "first", x, &[0, 1, 0], &[1, 6, 10]);
        let second = tosa_slice(&mut graph, "second", first, &[0, 2, 3], &[1, 2, 4]);
        let third = tosa_slice(&mut graph, "third", second, &[0, 1, 1], &[1, 1, 2]);
        let out = output(&mut graph, third);
        graph.set_args(
            out,
            vec![IrArg::List(vec![IrArg::Node(first), IrArg::Node(third)])],
        );
        assert!(IrFuseConsecutiveSlicesPass.run(&mut graph));
        assert!(graph.node(first).alive);
        assert!(!graph.node(second).alive);
        assert_eq!(graph.node(third).arg_node(0), Some(x));
        assert_eq!(graph.node(third).arg_int_list(1), Some(vec![0, 4, 4]));
        assert_eq!(graph.node(third).arg_int_list(2), Some(vec![1, 1, 2]));
    }

    #[test]
    fn normalizes_negative_bounds_and_uses_actual_intermediate_shapes() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 10]);
        let first = f32_call(
            &mut graph,
            "first",
            "aten.slice_copy.Tensor",
            vec![IrArg::Node(x), IrArg::Int(1), IrArg::Int(1), IrArg::Int(7)],
            [1, 8, 10],
        );
        let second = f32_call(
            &mut graph,
            "second",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(first),
                IrArg::Int(-2),
                IrArg::Int(-4),
                IrArg::Int(-2),
            ],
            [1, 8, 10],
        );
        output(&mut graph, second);
        assert!(crate::fx_rewrite_slice_ir::IrRewriteSlicePass.run(&mut graph));
        assert!(IrFuseConsecutiveSlicesPass.run(&mut graph));
        assert_eq!(graph.node(second).arg_int_list(1), Some(vec![0, 3, 0]));
        assert_eq!(graph.node_shape(second), Some([1, 2, 10].as_slice()));
    }

    #[test]
    fn leaves_dynamic_or_malformed_shape_operands_unchanged() {
        for dynamic in [false, true] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [1, 8, 10]);
            let first = tosa_slice(&mut graph, "first", x, &[0, 1, 0], &[1, 6, 10]);
            let second = tosa_slice(&mut graph, "second", first, &[0, 2, 3], &[1, 2, 4]);
            if dynamic {
                let start = graph.add_node(
                    NodeOp::Placeholder,
                    "dynamic".into(),
                    "dynamic".into(),
                    Vec::new(),
                    NodeMeta::default(),
                );
                graph.set_arg(second, 1, IrArg::Node(start));
            } else {
                graph.set_arg(second, 1, i64_list_arg(&[0, 2]));
            }
            output(&mut graph, second);
            assert!(!IrFuseConsecutiveSlicesPass.run(&mut graph));
            assert!(graph.node(first).alive && graph.node(second).alive);
        }
    }
}
