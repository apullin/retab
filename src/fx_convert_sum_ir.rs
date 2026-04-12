//! Pure Rust IrPass version of FxConvertSumPass.
//!
//! Rewrites `aten.sum.dim_IntList` nodes into one or more chained
//! `aten.sum.dim_IntList` operations (one per reduction axis), since TOSA
//! REDUCE_SUM only reduces a single axis at a time.
//!
//! For multi-axis reductions the dims are processed in descending order
//! so that earlier reductions don't shift the indices of later ones.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

pub struct IrConvertSumPass;

impl IrPass for IrConvertSumPass {
    fn name(&self) -> &str {
        "IrConvertSum"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("sum.dim_IntList") {
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

            // Extract dims: may be List([Int, ...]) or a single Int.
            let dims: Vec<i64> = match &graph.nodes[id].args[1] {
                IrArg::List(items) => {
                    match items.iter().map(|a| a.as_int()).collect::<Option<Vec<_>>>() {
                        Some(v) => v,
                        None => continue,
                    }
                }
                IrArg::Int(d) => vec![*d],
                _ => continue,
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
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
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
            let orig_dtype = graph.nodes[input_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

            // Track qparams for chaining: first reduce inherits the original
            // node's input_qparams; subsequent reduces use the previous
            // output_qparams as their input_qparams.  (Mirrors the FxPass.)
            let mut current_input_qparams: Option<HashMap<usize, _>> =
                orig_meta.input_qparams.clone();
            let current_output_qparams: Option<HashMap<usize, _>> =
                orig_meta.output_qparams.clone();

            let mut current_input = input_id;
            let mut last_reduce: Option<NodeId> = None;

            for &dim in &dims {
                // Get current input shape for computing output shape.
                let cur_shape = match graph.nodes[current_input].meta.val_shape() {
                    Some(s) => s.to_vec(),
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
                let mut step_meta = orig_meta.clone();
                step_meta.set_val(FakeTensorMeta {
                    shape: output_shape.clone(),
                    dtype: orig_dtype.clone(),
                });
                if step_meta
                    .tosa_dim_order
                    .as_ref()
                    .is_some_and(|order| order.len() != output_shape.len())
                {
                    step_meta.tosa_dim_order = None;
                    step_meta.dirty.tosa_dim_order = false;
                }
                if let Some(ref iqp) = current_input_qparams {
                    step_meta.set_input_qparams(iqp.clone());
                }
                if let Some(ref oqp) = current_output_qparams {
                    step_meta.set_output_qparams(oqp.clone());
                }

                // Build args: (current_input, [dim], keepdim)
                let reduce_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_reduce_dim{}", orig_name, dim),
                    "aten.sum.dim_IntList".to_string(),
                    vec![
                        IrArg::Node(current_input),
                        IrArg::List(vec![IrArg::Int(dim)]),
                        IrArg::Bool(keepdim),
                    ],
                    step_meta,
                );
                // Anchor intermediate nodes to the original sum node so
                // write_back_with_surgery places them in the right position.
                graph.anchor_to(reduce_id, id);

                current_input = reduce_id;
                // Chain qparams: next reduce's input_qparams = current output_qparams
                current_input_qparams = current_output_qparams.clone();
                last_reduce = Some(reduce_id);
            }

            // Replace original node with the final chained reduce.
            if let Some(final_id) = last_reduce {
                // Restore original node's qparams on the final reduce.
                if let Some(ref iqp) = orig_meta.input_qparams {
                    graph.node_mut(final_id).meta.set_input_qparams(iqp.clone());
                }
                if let Some(ref oqp) = orig_meta.output_qparams {
                    graph
                        .node_mut(final_id)
                        .meta
                        .set_output_qparams(oqp.clone());
                }
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
