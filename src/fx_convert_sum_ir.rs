//! Pure Rust IrPass version of FxConvertSumPass.
//!
//! Rewrites `aten.sum.dim_IntList` nodes into one or more chained
//! `aten.sum.dim_IntList` operations (one per reduction axis), since TOSA
//! REDUCE_SUM only reduces a single axis at a time.
//!
//! Each reduction preserves rank; a final view handles keepdim=False.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::fx_reduction::{ReductionDomain, SumQParamChain};
use crate::fx_target::is_sum_dim_int_list_target;
use crate::runner_logging::log_line;

pub struct IrConvertSumPass;

/// The Rust preprocess pipeline is used only for INT-only specifications.
pub struct IrConvertBoolSumPass;

impl IrPass for IrConvertBoolSumPass {
    fn name(&self) -> &str {
        "IrConvertBoolSum"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        graph.rewrite_live_call_functions(|graph, id| {
            if !is_sum_dim_int_list_target(&graph.nodes[id].target) {
                return false;
            }
            let Some(input_id) = graph.nodes[id].first_node_arg() else {
                return false;
            };
            if graph.node_dtype(input_id) != Some("torch.bool") {
                return false;
            }
            let Some(input_shape) = graph.node_shape_vec(input_id) else {
                return false;
            };
            let domain = ReductionDomain::for_sum(input_shape, graph.nodes[id].args.get(1), false);
            let reduced_elements = domain.dims_i64().into_iter().try_fold(1i64, |count, dim| {
                let extent = domain.input_shape()[dim as usize];
                (extent >= 0)
                    .then_some(extent)
                    .and_then(|extent| count.checked_mul(extent))
            });
            if !reduced_elements.is_some_and(|count| count <= i32::MAX as i64) {
                return false;
            }

            let name = graph.nodes[id].name.clone();
            let input_meta = graph.nodes[input_id].meta.clone();
            let meta = graph.nodes[id].meta.clone();
            let Some(output_shape) = meta.val_shape().map(|shape| shape.to_vec()) else {
                return false;
            };
            let output_dtype = meta.val_dtype_or("torch.int64").to_string();
            let mut args = graph.nodes[id].args.clone();
            let mut kwargs = graph.nodes[id].kwargs.clone();
            kwargs.insert("dtype".into(), IrArg::Dtype("torch.int32".into()));

            let mut cursor = graph.rewrite_cursor(id);
            let cast = cursor.dtype_cast(
                format!("{name}_accumulator_input"),
                input_id,
                "torch.int32",
                input_meta.cloned_with_val(domain.input_shape().to_vec(), "torch.int32"),
            );
            args[0] = IrArg::Node(cast);
            let sum = cursor.call_function_with_kwargs(
                format!("{name}_accumulator_sum"),
                "aten.sum.dim_IntList",
                args,
                kwargs,
                meta.cloned_with_val(output_shape.clone(), "torch.int32"),
            );
            let output = cursor.dtype_cast_if_needed(
                format!("{name}_output_cast"),
                sum,
                &output_dtype,
                meta,
            );
            cursor.replace_anchor_with(output);
            true
        })
    }
}

impl IrPass for IrConvertSumPass {
    fn name(&self) -> &str {
        "IrConvertSum"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_sum_dim_int_list_target(&graph.nodes[id].target) {
                continue;
            }

            // args[0] = input node, args[1] = dims list/int
            if graph.nodes[id].args.len() < 2 {
                continue;
            }
            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };

            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };

            // keepdim: args[2] if present, default false.
            let keepdim = graph.nodes[id]
                .args
                .get(2)
                .and_then(|a| a.as_bool())
                .unwrap_or(false);
            let domain =
                ReductionDomain::for_sum(input_shape, graph.nodes[id].args.get(1), keepdim);
            if domain.is_empty() || (domain.dim_count() == 1 && keepdim) {
                continue;
            }
            let steps = domain.keepdim_single_axis_steps();

            let orig_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_meta = graph.nodes[input_id].meta.clone();
            let kwargs = graph.nodes[id].kwargs.clone();
            let output_shape = orig_meta
                .val_shape()
                .unwrap_or(domain.output_shape())
                .to_vec();
            let mut qparams = SumQParamChain::new(&orig_meta, &input_meta);

            let mut current_input = input_id;
            let mut last_reduce: Option<NodeId> = None;
            let mut cursor = graph.rewrite_cursor(id);

            for (index, step) in steps.iter().enumerate() {
                let reduce_id = cursor.call_function_with_kwargs(
                    format!("{}_reduce_dim{}", orig_name, step.dim),
                    "aten.sum.dim_IntList",
                    vec![
                        IrArg::Node(current_input),
                        IrArg::List(vec![IrArg::Int(step.dim)]),
                        IrArg::Bool(true),
                    ],
                    kwargs.clone(),
                    qparams.step_meta(&step.output_shape, index + 1 == steps.len()),
                );

                current_input = reduce_id;
                last_reduce = Some(reduce_id);
            }

            // Replace original node with the final chained reduce.
            if let Some(final_id) = last_reduce {
                let final_id = if domain.keepdim() {
                    final_id
                } else {
                    cursor.view_copy(
                        format!("{orig_name}_view"),
                        final_id,
                        &output_shape,
                        orig_meta
                            .cloned_with_dtype_of_or_float32(output_shape.clone(), &input_meta),
                    )
                };
                cursor.replace_anchor_with(final_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!("[IrConvertSum] rewrote sum.dim_IntList ops"));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::test_ir::{call, live_node, placeholder};

    #[test]
    fn multi_axis_sum_squeezes_only_after_all_reductions() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3, 4], "torch.int32");
        let sum = call(
            &mut graph,
            "sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(x), i64_list_arg(&[0, -1]), IrArg::Bool(false)],
            [3],
            "torch.int32",
        );

        assert!(IrConvertSumPass.run(&mut graph));
        let first = live_node(&graph, "sum_reduce_dim0");
        let second = live_node(&graph, "sum_reduce_dim2");
        assert_eq!(first.meta.val_shape(), Some([1, 3, 4].as_slice()));
        assert_eq!(second.meta.val_shape(), Some([1, 3, 1].as_slice()));
        assert_eq!(first.args[2].as_bool(), Some(true));
        assert_eq!(second.args[2].as_bool(), Some(true));
        let view = live_node(&graph, "sum_view");
        assert_eq!(view.first_node_arg(), Some(second.id));
        assert_eq!(graph.replacement_of(sum), Some(view.id));
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.alive && node.target == "aten.view_copy.default")
                .count(),
            1
        );
    }

    #[test]
    fn bool_sum_uses_int32_and_restores_requested_dtype() {
        for output_dtype in ["torch.int16", "torch.int32", "torch.int64"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2, 3, 4], "torch.bool");
            let sum = call(
                &mut graph,
                "sum",
                "aten.sum.dim_IntList",
                vec![IrArg::Node(x), i64_list_arg(&[0, -1]), IrArg::Bool(true)],
                [1, 3, 1],
                output_dtype,
            );

            assert!(IrConvertBoolSumPass.run(&mut graph));
            let cast = live_node(&graph, "sum_accumulator_input");
            let accumulator_sum = live_node(&graph, "sum_accumulator_sum");
            assert_eq!(cast.meta.val_dtype(), Some("torch.int32"));
            assert_eq!(accumulator_sum.first_node_arg(), Some(cast.id));
            assert_eq!(accumulator_sum.meta.val_dtype(), Some("torch.int32"));
            assert!(
                matches!(accumulator_sum.kwargs.get("dtype"), Some(IrArg::Dtype(dtype)) if dtype == "torch.int32")
            );
            let replacement = graph.replacement_of(sum).unwrap();
            assert_eq!(
                graph.nodes[replacement].meta.val_dtype(),
                Some(output_dtype)
            );
            assert!(IrConvertSumPass.run(&mut graph));
            assert_eq!(
                live_node(&graph, "sum_accumulator_sum_reduce_dim2")
                    .meta
                    .val_dtype(),
                Some("torch.int32")
            );
        }
    }

    #[test]
    fn bool_sum_with_empty_dims_counts_all_elements() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.bool");
        call(
            &mut graph,
            "sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(x), i64_list_arg(&[]), IrArg::Bool(false)],
            [],
            "torch.int32",
        );
        assert!(IrConvertBoolSumPass.run(&mut graph));
        assert!(IrConvertSumPass.run(&mut graph));
        assert_eq!(
            live_node(&graph, "sum_accumulator_sum_view")
                .meta
                .val_shape(),
            Some([].as_slice())
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.alive && node.target == "aten.sum.dim_IntList")
                .count(),
            2
        );
    }

    #[test]
    fn bool_sum_skips_inexact_accumulators_and_other_dtypes() {
        for (shape, dtype) in [
            (vec![i32::MAX as i64 + 1], "torch.bool"),
            (vec![i64::MAX, i64::MAX], "torch.bool"),
            (vec![2, 3], "torch.int32"),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", shape, dtype);
            let sum = call(
                &mut graph,
                "sum",
                "aten.sum.dim_IntList",
                vec![IrArg::Node(x), IrArg::None],
                [],
                "torch.int64",
            );
            assert!(!IrConvertBoolSumPass.run(&mut graph));
            assert!(graph.nodes[sum].alive);
        }
    }
}
