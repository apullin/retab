//! Pure Rust IrPass: aten.gather → backend.tosa.GATHER with shape canonicalization.
//!
//! Handles 2D (N,K → GATHER → N,W) and 3D (N,K,C → permute+reshape → GATHER → reshape+permute).
//! Bool inputs are cast to int8 before GATHER, then cast back.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeOp};
use crate::fx_target::is_gather_target;
use crate::runner_logging::log_line;

pub struct IrCanonicalizeGatherPass;

impl IrPass for IrCanonicalizeGatherPass {
    fn name(&self) -> &str {
        "IrCanonicalizeGather"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_gather_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            let x_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let dim = match graph.nodes[id].args[1].as_int() {
                Some(v) => v,
                None => continue,
            };
            let index_id = match graph.nodes[id].args[2].as_node() {
                Some(n) => n,
                None => continue,
            };

            let x_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let index_shape = match graph.nodes[index_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };

            // Only handle rank 2 or 3, dim must normalize to 1, batch dims must match
            let x_rank = x_shape.len() as i64;
            if !matches!(x_shape.len(), 2 | 3) || !matches!(index_shape.len(), 2 | 3) {
                continue;
            }
            let dim_norm = dim.rem_euclid(x_rank) as usize;
            if dim_norm != 1 || index_shape.first() != x_shape.first() {
                continue;
            }
            if x_shape.len() == 3 && (index_shape.len() != 3 || index_shape[2] != x_shape[2]) {
                continue;
            }

            let n = x_shape[0];
            let k = x_shape[1];
            let w = index_shape[1];
            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_dtype = graph.nodes[x_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let needs_bool_cast = x_dtype == "torch.bool";
            let work_dtype = if needs_bool_cast {
                "torch.int8"
            } else {
                &x_dtype
            };

            let mut values_in = x_id;

            // Optional bool→int8 cast
            if needs_bool_cast {
                let cast_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_bool_to_int8", node_name),
                    "aten._to_copy.default",
                    vec![IrArg::Node(x_id)],
                    orig_meta.cloned_with_val(x_shape.clone(), work_dtype),
                );
                values_in = cast_id;
            }

            let result = if x_shape.len() == 2 {
                // 2D path: view to 3D → GATHER → view back to 2D
                let v3d = graph.add_anchored_call_function(
                    id,
                    format!("{}_values_3d", node_name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(values_in), i64_list(&[n, k, 1])],
                    orig_meta.cloned_with_val(vec![n, k, 1], work_dtype),
                );

                let gathered = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather", node_name),
                    "backend.tosa.GATHER.default",
                    vec![IrArg::Node(v3d), IrArg::Node(index_id)],
                    orig_meta.cloned_with_val(vec![n, w, 1], work_dtype),
                );

                graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_out_2d", node_name),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(gathered), i64_list(&[n, w])],
                    orig_meta.cloned_with_val(vec![n, w], work_dtype),
                )
            } else {
                // 3D path: permute(0,2,1) → reshape(N*C,K,1) → GATHER → reshape(N,C,W) → permute(0,2,1)
                let c = x_shape[2];

                let vperm = graph.add_anchored_call_function(
                    id,
                    format!("{}_values_perm", node_name),
                    "aten.permute_copy.default",
                    vec![IrArg::Node(values_in), i64_list(&[0, 2, 1])],
                    orig_meta.cloned_with_val(vec![n, c, k], work_dtype),
                );

                let vresh = graph.add_anchored_call_function(
                    id,
                    format!("{}_values_flat", node_name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(vperm), i64_list(&[n * c, k, 1])],
                    orig_meta.cloned_with_val(vec![n * c, k, 1], work_dtype),
                );

                let iperm = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx_perm", node_name),
                    "aten.permute_copy.default",
                    vec![IrArg::Node(index_id), i64_list(&[0, 2, 1])],
                    orig_meta.cloned_with_val(vec![n, c, w], "torch.int64"),
                );

                let iresh = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx_flat", node_name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(iperm), i64_list(&[n * c, w])],
                    orig_meta.cloned_with_val(vec![n * c, w], "torch.int64"),
                );

                let gathered = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather", node_name),
                    "backend.tosa.GATHER.default",
                    vec![IrArg::Node(vresh), IrArg::Node(iresh)],
                    orig_meta.cloned_with_val(vec![n * c, w, 1], work_dtype),
                );

                let gresh = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather_3d", node_name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(gathered), i64_list(&[n, c, w])],
                    orig_meta.cloned_with_val(vec![n, c, w], work_dtype),
                );

                graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_out_perm", node_name),
                    "aten.permute_copy.default".to_string(),
                    vec![IrArg::Node(gresh), i64_list(&[0, 2, 1])],
                    orig_meta.cloned_with_val(vec![n, w, c], work_dtype),
                )
            };

            // Optional int8→bool cast back
            let final_result = if needs_bool_cast {
                graph.anchor_to(result, id);
                graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_int8_to_bool", node_name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(result)],
                    orig_meta,
                )
            } else {
                result
            };

            graph.replace_and_erase(id, final_result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrCanonicalizeGather] canonicalized gather ops"
            ));
        }
        modified
    }
}
