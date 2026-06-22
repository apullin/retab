//! Pure Rust IrPass version of FxConvertSumPass.
//!
//! Rewrites `aten.sum.dim_IntList` nodes into one or more chained
//! `aten.sum.dim_IntList` operations (one per reduction axis), since TOSA
//! REDUCE_SUM only reduces a single axis at a time.
//!
//! For multi-axis reductions the dims are processed in descending order
//! so that earlier reductions don't shift the indices of later ones.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::fx_reduction::{ReductionDomain, SumQParamChain};
use crate::fx_target::is_sum_dim_int_list_target;
use crate::runner_logging::log_line;

pub struct IrConvertSumPass;

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
            let domain = ReductionDomain::from_arg(
                input_shape,
                graph.nodes[id].args.get(1),
                keepdim,
                false,
                None,
            );
            if domain.dim_count() <= 1 {
                // The native serializer already handles single-axis sums.
                // Only multi-axis reductions need explicit decomposition.
                continue;
            }
            let steps = domain.chained_single_axis_steps();

            let orig_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_meta = graph.nodes[input_id].meta.clone();
            let mut qparams = SumQParamChain::new(&orig_meta, &input_meta);

            let mut current_input = input_id;
            let mut last_reduce: Option<NodeId> = None;
            let mut cursor = graph.rewrite_cursor(id);

            for (index, step) in steps.iter().enumerate() {
                // Build args: (current_input, [dim], keepdim)
                let reduce_id = cursor.call_function(
                    format!("{}_reduce_dim{}", orig_name, step.dim),
                    "aten.sum.dim_IntList",
                    vec![
                        IrArg::Node(current_input),
                        IrArg::List(vec![IrArg::Int(step.dim)]),
                        IrArg::Bool(domain.keepdim()),
                    ],
                    qparams.step_meta(&step.output_shape, index + 1 == steps.len()),
                );

                current_input = reduce_id;
                last_reduce = Some(reduce_id);
            }

            // Replace original node with the final chained reduce.
            if let Some(final_id) = last_reduce {
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
