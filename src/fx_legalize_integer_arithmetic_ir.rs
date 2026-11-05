use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::{is_add_sub_target, is_sum_dim_int_list_target};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

pub struct IrLegalizeIntegerArithmeticPass;

impl IrPass for IrLegalizeIntegerArithmeticPass {
    fn name(&self) -> &str {
        "IrLegalizeIntegerArithmetic"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            let is_sum = is_sum_dim_int_list_target(&node.target);
            if !is_sum && !is_add_sub_target(&node.target) {
                return false;
            }
            if node.args.len() < 2 || node.meta.has_nonempty_qparam_metadata() {
                return false;
            }
            let Some(output_dtype) = node.meta.val_dtype() else {
                return false;
            };
            if !matches!(
                DType::from_torch_str(output_dtype),
                DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
            ) {
                return false;
            }
            let input_count = if is_sum { 1 } else { 2 };
            let inputs: Vec<_> = node.args[..input_count]
                .iter()
                .enumerate()
                .filter_map(|(pos, arg)| Some((pos, arg.as_node()?)))
                .collect();
            if inputs.is_empty()
                || inputs.iter().any(|&(_, input)| {
                    graph.nodes[input].meta.has_nonempty_qparam_metadata()
                        || graph.nodes[input].meta.val_shape().is_none()
                        || !matches!(
                            graph.nodes[input].meta.val_dtype_kind(),
                            Some(DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64)
                        )
                })
                || !inputs.iter().any(|&(_, input)| {
                    matches!(
                        graph.nodes[input].meta.val_dtype_kind(),
                        Some(DType::Int8 | DType::Int16)
                    )
                })
            {
                return false;
            }

            let name = node.name.clone();
            let target = node.target.clone();
            let meta = node.meta.clone();
            let output_dtype = output_dtype.to_string();
            let Some(output_shape) = meta.val_shape().map(<[i64]>::to_vec) else {
                return false;
            };
            let mut args = node.args.clone();
            let mut kwargs = node.kwargs.clone();
            let input_meta: Vec<_> = inputs
                .into_iter()
                .map(|(pos, input)| {
                    let meta = &graph.nodes[input].meta;
                    (
                        pos,
                        input,
                        meta.cloned_with_val(meta.val_shape().unwrap().to_vec(), "torch.int32"),
                    )
                })
                .collect();

            let mut cursor = graph.rewrite_cursor(id);
            for (pos, input, meta) in input_meta {
                args[pos] = IrArg::Node(cursor.dtype_cast_if_needed(
                    format!("{name}_arg{pos}_int32"),
                    input,
                    "torch.int32",
                    meta,
                ));
            }
            if is_sum {
                kwargs.insert("dtype".to_string(), IrArg::Dtype("torch.int32".to_string()));
            }
            let arithmetic = cursor.call_function_with_kwargs(
                format!("{name}_int32"),
                target,
                args,
                kwargs,
                meta.cloned_with_val(output_shape, "torch.int32"),
            );
            let result = cursor.dtype_cast_if_needed(
                format!("{name}_output_cast"),
                arithmetic,
                &output_dtype,
                meta,
            );
            cursor.replace_anchor_with(result);
            true
        });
        if modified {
            log_line(format_args!(
                "[IrLegalizeIntegerArithmetic] promoted raw integer arithmetic to int32"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_convert_sum_ir::IrConvertSumPass;
    use crate::fx_ir::{i64_list_arg, QParamMap};
    use crate::test_ir::{call, live_node, placeholder, qparam};

    #[test]
    fn narrow_add_sub_cast_operands_and_restore_output_dtype() {
        for target in ["aten.add.Tensor", "aten.sub.Tensor"] {
            for dtype in ["torch.int8", "torch.int16"] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [2, 3], dtype);
                let y = placeholder(&mut graph, "y", [1, 3], dtype);
                let arithmetic = call(
                    &mut graph,
                    "arithmetic",
                    target,
                    vec![IrArg::Node(x), IrArg::Node(y)],
                    [2, 3],
                    dtype,
                );

                assert!(IrLegalizeIntegerArithmeticPass.run(&mut graph));
                let x_cast = live_node(&graph, "arithmetic_arg0_int32");
                let y_cast = live_node(&graph, "arithmetic_arg1_int32");
                assert_eq!(x_cast.meta.val_shape(), Some([2, 3].as_slice()));
                assert_eq!(y_cast.meta.val_shape(), Some([1, 3].as_slice()));
                let int32 = live_node(&graph, "arithmetic_int32");
                assert_eq!(int32.args[0].as_node(), Some(x_cast.id));
                assert_eq!(int32.args[1].as_node(), Some(y_cast.id));
                assert_eq!(int32.meta.val_dtype(), Some("torch.int32"));
                let result = graph.replacement_of(arithmetic).unwrap();
                assert_eq!(graph.nodes[result].meta.val_dtype(), Some(dtype));
                assert_eq!(graph.nodes[result].first_node_arg(), Some(int32.id));
                assert!(!IrLegalizeIntegerArithmeticPass.run(&mut graph));
            }
        }
    }

    #[test]
    fn mixed_integer_add_keeps_existing_int32_operand() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [3], "torch.int8");
        let y = placeholder(&mut graph, "y", [3], "torch.int32");
        let add = call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(y)],
            [3],
            "torch.int32",
        );

        assert!(IrLegalizeIntegerArithmeticPass.run(&mut graph));
        let int32 = live_node(&graph, "add_int32");
        assert_eq!(int32.args[1].as_node(), Some(y));
        assert_eq!(graph.replacement_of(add), Some(int32.id));
    }

    #[test]
    fn narrow_sum_promotes_before_decomposing_axes() {
        for output_dtype in ["torch.int8", "torch.int16", "torch.int32", "torch.int64"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2, 3, 4], "torch.int8");
            let sum = call(
                &mut graph,
                "sum",
                "aten.sum.dim_IntList",
                vec![IrArg::Node(x), i64_list_arg(&[0, 2]), IrArg::Bool(false)],
                [3],
                output_dtype,
            );

            assert!(IrLegalizeIntegerArithmeticPass.run(&mut graph));
            assert!(IrConvertSumPass.run(&mut graph));
            for node in graph
                .live_nodes()
                .filter(|node| is_sum_dim_int_list_target(&node.target))
            {
                assert_eq!(node.meta.val_dtype(), Some("torch.int32"));
                assert_eq!(
                    graph.node_dtype(node.first_node_arg().unwrap()),
                    Some("torch.int32")
                );
                assert!(
                    matches!(node.kwargs.get("dtype"), Some(IrArg::Dtype(dtype)) if dtype == "torch.int32")
                );
            }
            assert_eq!(
                graph.node_dtype(graph.replacement_of(sum).unwrap()),
                Some(output_dtype)
            );
        }
    }

    #[test]
    fn quantized_arithmetic_keeps_rescale_lowering() {
        for annotate_input in [false, true] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [3], "torch.int8");
            let add = call(
                &mut graph,
                "add",
                "aten.add.Tensor",
                vec![IrArg::Node(x), IrArg::Int(1)],
                [3],
                "torch.int8",
            );
            let mut qparams = QParamMap::new();
            qparams.insert(0, qparam(0.1, 0, "torch.int8"));
            graph.nodes[if annotate_input { x } else { add }]
                .meta
                .set_output_qparams(qparams);
            assert!(!IrLegalizeIntegerArithmeticPass.run(&mut graph));
            assert!(graph.nodes[add].alive);
        }
    }
}
