//! Pure Rust IrPass version of FxConvertSumPass.
//!
//! Rewrites `aten.sum.dim_IntList` nodes into one or more chained
//! `aten.sum.dim_IntList` operations (one per reduction axis), since TOSA
//! REDUCE_SUM only reduces a single axis at a time.
//!
//! For multi-axis reductions the dims are processed in descending order
//! so that earlier reductions don't shift the indices of later ones.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
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

            let dims = match graph.nodes[id].args[1].as_int_list_or_scalar() {
                Some(dims) => dims,
                None => continue,
            };
            if dims.is_empty() {
                continue;
            }
            if dims.len() <= 1 {
                // The native serializer already handles single-axis sums.
                // Only multi-axis reductions need explicit decomposition.
                continue;
            }

            // Normalize negative dims.
            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };
            let input_rank = input_shape.len() as i64;
            let mut dims: Vec<i64> = dims
                .into_iter()
                .map(|d| if d < 0 { d + input_rank } else { d })
                .collect();

            // keepdim: args[2] if present, default false.
            let keepdim = graph.nodes[id]
                .args
                .get(2)
                .and_then(|a| a.as_bool())
                .unwrap_or(false);

            // Sort dims descending to keep indices stable during chained reductions.
            dims.sort_unstable_by(|a, b| b.cmp(a));

            let orig_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_meta = graph.nodes[input_id].meta.clone();

            // Track qparams for chaining: first reduce inherits the original
            // node's input_qparams; subsequent reduces use the previous
            // output_qparams as their input_qparams.  (Mirrors the FxPass.)
            let mut current_input_qparams = orig_meta.cloned_input_qparams();
            let current_output_qparams = orig_meta.cloned_output_qparams();

            let mut current_input = input_id;
            let mut last_reduce: Option<NodeId> = None;

            for &dim in &dims {
                // Get current input shape for computing output shape.
                let cur_shape = match graph.node_shape_vec(current_input) {
                    Some(s) => s,
                    None => continue,
                };
                let rank = cur_shape.len();
                let normalized_dim = (dim as usize) % rank;

                // Compute output shape for this single-dim reduction.
                let output_shape = if keepdim {
                    let mut s = cur_shape;
                    s[normalized_dim] = 1;
                    s
                } else {
                    let mut s = cur_shape;
                    s.remove(normalized_dim);
                    s
                };

                // Preserve layout metadata from the source sum node. Some
                // upstream decompositions (for example group_norm) mark their
                // reduction chain as layout-neutral, and rebuilding from a
                // blank NodeMeta here silently drops that contract.
                let mut step_meta =
                    orig_meta.cloned_with_dtype_of_or_float32(output_shape.clone(), &input_meta);
                step_meta.clear_tosa_dim_order_if_rank_mismatch(output_shape.len());
                if let Some(ref iqp) = current_input_qparams {
                    step_meta.set_input_qparams(iqp.clone());
                }
                if let Some(ref oqp) = current_output_qparams {
                    step_meta.set_output_qparams(oqp.clone());
                }

                // Build args: (current_input, [dim], keepdim)
                let reduce_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_reduce_dim{}", orig_name, dim),
                    "aten.sum.dim_IntList",
                    vec![
                        IrArg::Node(current_input),
                        IrArg::List(vec![IrArg::Int(dim)]),
                        IrArg::Bool(keepdim),
                    ],
                    step_meta,
                );

                current_input = reduce_id;
                // Chain qparams: next reduce's input_qparams = current output_qparams
                current_input_qparams = current_output_qparams.clone();
                last_reduce = Some(reduce_id);
            }

            // Replace original node with the final chained reduce.
            if let Some(final_id) = last_reduce {
                // Restore original node's qparams on the final reduce.
                let final_meta = &mut graph.node_mut(final_id).meta;
                final_meta.copy_qparams_from(&orig_meta);
                graph.replace_and_erase(id, final_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!("[IrConvertSum] rewrote sum.dim_IntList ops"));
        }
        modified
    }
}
