use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass};
use crate::fx_target::{is_add_sub_target, is_mul_target};
use crate::runner_logging::log_line;

pub struct IrBroadcastArgsPass;

impl IrPass for IrBroadcastArgsPass {
    fn name(&self) -> &str {
        "IrBroadcastArgs"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            if !is_add_sub_target(&node.target) && !is_mul_target(&node.target) {
                return false;
            }
            let Some(output_shape) = graph.node_shape_vec(id) else {
                return false;
            };
            let mut broadcasts = Vec::new();
            for (pos, arg) in node.args.iter().enumerate() {
                let Some(input) = arg.as_node() else {
                    continue;
                };
                let Some(shape) = graph.node_shape(input) else {
                    return false;
                };
                if shape.len() > output_shape.len() {
                    return false;
                }
                let mut aligned_shape = vec![1; output_shape.len() - shape.len()];
                aligned_shape.extend_from_slice(shape);
                if !aligned_shape
                    .iter()
                    .zip(&output_shape)
                    .all(|(&a, &b)| a > 0 && (a == 1 || a == b))
                {
                    return false;
                }
                if aligned_shape != output_shape {
                    broadcasts.push((pos, input, aligned_shape, shape.len()));
                }
            }
            if broadcasts.len() < 2 {
                return false;
            }

            let name = node.name.clone();
            let target = node.target.clone();
            let meta = node.meta.clone();
            let mut args = node.args.clone();
            let kwargs = node.kwargs.clone();
            let repeat_meta: Vec<_> = broadcasts
                .into_iter()
                .skip(1)
                .map(|(pos, input, shape, rank)| {
                    let stamp =
                        meta.tensor_stamp_with_dtype_of_or_float32(&graph.nodes[input].meta);
                    (
                        pos,
                        input,
                        stamp.drop_qparams(shape.clone()),
                        stamp.drop_qparams(output_shape.clone()),
                        shape,
                        rank,
                    )
                })
                .collect();

            let mut cursor = graph.rewrite_cursor(id);
            for (pos, mut input, view_meta, repeat_meta, shape, rank) in repeat_meta {
                if rank != shape.len() {
                    input =
                        cursor.view_copy(format!("{name}_arg{pos}_view"), input, &shape, view_meta);
                }
                let multiples: Vec<_> = output_shape
                    .iter()
                    .zip(&shape)
                    .map(|(&out, &input)| out / input)
                    .collect();
                args[pos] = IrArg::Node(cursor.call_function(
                    format!("{name}_arg{pos}_repeat"),
                    "aten.repeat.default",
                    vec![IrArg::Node(input), i64_list_arg(&multiples)],
                    repeat_meta,
                ));
            }
            let result = cursor.call_function_with_kwargs(
                format!("{name}_broadcast"),
                target,
                args,
                kwargs,
                meta,
            );
            cursor.replace_anchor_with(result);
            true
        });
        if modified {
            log_line(format_args!(
                "[IrBroadcastArgs] repeated additional broadcast operands"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, live_node, placeholder, qparam};

    #[test]
    fn two_broadcast_operands_repeat_only_the_second_without_qparams() {
        for target in ["aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2, 1], "torch.int8");
            let y = placeholder(&mut graph, "y", [1, 3], "torch.int8");
            let op = call(
                &mut graph,
                "op",
                target,
                vec![IrArg::Node(x), IrArg::Node(y)],
                [2, 3],
                "torch.int8",
            );
            graph.nodes[op]
                .meta
                .set_both_qparams(qparam(0.1, 0, "torch.int8"));
            assert!(IrBroadcastArgsPass.run(&mut graph));
            let repeat = live_node(&graph, "op_arg1_repeat");
            assert_eq!(repeat.target, "aten.repeat.default");
            assert_eq!(repeat.first_node_arg(), Some(y));
            assert_eq!(repeat.arg_int_list(1), Some(vec![2, 1]));
            assert_eq!(repeat.meta.val_shape(), Some([2, 3].as_slice()));
            assert_eq!(repeat.meta.val_dtype(), Some("torch.int8"));
            assert!(!repeat.meta.has_nonempty_qparam_metadata());
            let result = &graph.nodes[graph.replacement_of(op).unwrap()];
            assert_eq!(result.args[0].as_node(), Some(x));
            assert_eq!(result.args[1].as_node(), Some(repeat.id));
            assert!(result.meta.has_nonempty_qparam_metadata());
            assert!(!IrBroadcastArgsPass.run(&mut graph));
        }
    }

    #[test]
    fn lower_rank_second_operand_is_aligned_before_repeat() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 1, 1], "torch.int32");
        let y = placeholder(&mut graph, "y", [3], "torch.int32");
        call(
            &mut graph,
            "op",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(y)],
            [2, 1, 3],
            "torch.int32",
        );
        assert!(IrBroadcastArgsPass.run(&mut graph));
        let view = live_node(&graph, "op_arg1_view");
        assert_eq!(view.meta.val_shape(), Some([1, 1, 3].as_slice()));
        let repeat = live_node(&graph, "op_arg1_repeat");
        assert_eq!(repeat.first_node_arg(), Some(view.id));
        assert_eq!(repeat.arg_int_list(1), Some(vec![2, 1, 1]));
    }

    #[test]
    fn one_broadcast_and_other_targets_are_unchanged() {
        for (target, left_shape, right_shape) in [
            ("aten.add.Tensor", vec![2, 3], vec![1, 3]),
            ("aten.sub.Tensor", vec![2, 1], vec![2, 3]),
            ("aten.mul.Tensor", vec![2, 3], vec![2, 3]),
            ("aten.bitwise_left_shift.Tensor", vec![2, 1], vec![1, 3]),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", left_shape, "torch.int32");
            let y = placeholder(&mut graph, "y", right_shape, "torch.int32");
            call(
                &mut graph,
                "op",
                target,
                vec![IrArg::Node(x), IrArg::Node(y)],
                [2, 3],
                "torch.int32",
            );
            assert!(!IrBroadcastArgsPass.run(&mut graph));
        }
    }
}
