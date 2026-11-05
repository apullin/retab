use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

pub struct IrCastToInt32Pass;

impl IrPass for IrCastToInt32Pass {
    fn name(&self) -> &str {
        "IrCastToInt32"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            if !matches!(
                node.target.as_str(),
                "aten.bitwise_left_shift.Tensor" | "aten.bitwise_right_shift.Tensor"
            ) || node.args.len() != 2
            {
                return false;
            }
            let inputs: Vec<_> = node.args.iter().filter_map(IrArg::as_node).collect();
            if inputs.len() != node.args.len()
                || inputs.iter().any(|&input| {
                    graph.node_shape(input).is_none() || graph.node_dtype(input).is_none()
                })
                || inputs
                    .iter()
                    .all(|&input| graph.nodes[input].meta.val_dtype_kind() == Some(DType::Int32))
            {
                return false;
            }

            let name = node.name.clone();
            let target = node.target.clone();
            let meta = node.meta.clone();
            let kwargs = node.kwargs.clone();
            let Some(output_shape) = graph.node_shape_vec(id) else {
                return false;
            };
            let output_dtype = graph.node_dtype(inputs[0]).unwrap().to_string();
            let input_meta: Vec<_> = inputs
                .iter()
                .map(|&input| {
                    graph.nodes[input]
                        .meta
                        .cloned_with_val(graph.node_shape_vec(input).unwrap(), "torch.int32")
                })
                .collect();

            let mut cursor = graph.rewrite_cursor(id);
            let args = inputs
                .into_iter()
                .zip(input_meta)
                .enumerate()
                .map(|(pos, (input, meta))| {
                    IrArg::Node(cursor.dtype_cast_if_needed(
                        format!("{name}_arg{pos}_int32"),
                        input,
                        "torch.int32",
                        meta,
                    ))
                })
                .collect();
            let shift = cursor.call_function_with_kwargs(
                format!("{name}_int32"),
                target,
                args,
                kwargs,
                meta.cloned_with_val(output_shape.clone(), "torch.int32"),
            );
            let result = cursor.dtype_cast_if_needed(
                format!("{name}_output_cast"),
                shift,
                &output_dtype,
                meta.cloned_with_val(output_shape, &output_dtype),
            );
            cursor.replace_anchor_with(result);
            true
        });
        if modified {
            log_line(format_args!(
                "[IrCastToInt32] promoted shift operands to int32"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, live_node, placeholder};

    #[test]
    fn tensor_shifts_widen_operands_and_restore_first_input_dtype() {
        for target in [
            "aten.bitwise_left_shift.Tensor",
            "aten.bitwise_right_shift.Tensor",
        ] {
            for dtype in ["torch.int8", "torch.int16", "torch.int64"] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [2, 3], dtype);
                let y = placeholder(&mut graph, "y", [1, 3], dtype);
                let shift = call(
                    &mut graph,
                    "shift",
                    target,
                    vec![IrArg::Node(x), IrArg::Node(y)],
                    [2, 3],
                    dtype,
                );
                assert!(IrCastToInt32Pass.run(&mut graph));
                let widened = live_node(&graph, "shift_int32");
                assert_eq!(widened.meta.val_dtype(), Some("torch.int32"));
                for (arg, shape) in widened.args.iter().zip([&[2, 3][..], &[1, 3][..]]) {
                    let cast = &graph.nodes[arg.as_node().unwrap()];
                    assert_eq!(cast.meta.val_dtype(), Some("torch.int32"));
                    assert_eq!(cast.meta.val_shape(), Some(shape));
                }
                let result = &graph.nodes[graph.replacement_of(shift).unwrap()];
                assert_eq!(result.meta.val_dtype(), Some(dtype));
                assert_eq!(result.first_node_arg(), Some(widened.id));
                assert!(!IrCastToInt32Pass.run(&mut graph));
            }
        }
    }

    #[test]
    fn mixed_operands_keep_int32_inputs_and_use_first_operand_for_result_dtype() {
        for (left, right, result_dtype, casts) in [
            ("torch.int8", "torch.int32", "torch.int8", 2),
            ("torch.int32", "torch.int16", "torch.int32", 1),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [3], left);
            let y = placeholder(&mut graph, "y", [3], right);
            let shift = call(
                &mut graph,
                "shift",
                "aten.bitwise_right_shift.Tensor",
                vec![IrArg::Node(x), IrArg::Node(y)],
                [3],
                "torch.int32",
            );
            assert!(IrCastToInt32Pass.run(&mut graph));
            let widened = live_node(&graph, "shift_int32");
            assert_eq!(
                widened.args[if left == "torch.int32" { 0 } else { 1 }].as_node(),
                Some(if left == "torch.int32" { x } else { y })
            );
            assert_eq!(
                graph.node_dtype(graph.replacement_of(shift).unwrap()),
                Some(result_dtype)
            );
            assert_eq!(
                graph
                    .live_nodes()
                    .filter(|node| node.target == "aten._to_copy.default")
                    .count(),
                casts
            );
        }
    }

    #[test]
    fn int32_shifts_and_other_overloads_are_unchanged() {
        for (target, dtype) in [
            ("aten.bitwise_left_shift.Tensor", "torch.int32"),
            ("aten.bitwise_right_shift.Tensor", "torch.int32"),
            ("aten.bitwise_left_shift.Tensor_Scalar", "torch.int8"),
            ("aten.bitwise_right_shift.Scalar_Tensor", "torch.int16"),
            ("aten.add.Tensor", "torch.int8"),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [3], dtype);
            let y = placeholder(&mut graph, "y", [3], dtype);
            call(
                &mut graph,
                "op",
                target,
                vec![IrArg::Node(x), IrArg::Node(y)],
                [3],
                dtype,
            );
            assert!(!IrCastToInt32Pass.run(&mut graph));
        }
    }
}
