//! Pure Rust IrPass: aten.gather → backend.tosa.GATHER with shape canonicalization.
//!
//! Handles 2D (N,K → GATHER → N,W) and 3D (N,K,C → permute+reshape → GATHER → reshape+permute).
//! Bool inputs are cast to int8 before GATHER, then cast back.

use crate::fx_ir::{wrap_dim, IrArg, IrGraph, IrPass};
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

            let x_shape = match graph.node_shape_vec(x_id) {
                Some(s) => s,
                None => continue,
            };
            let index_shape = match graph.node_shape_vec(index_id) {
                Some(s) => s,
                None => continue,
            };

            // Only handle rank 2 or 3, dim must normalize to 1, batch dims must match
            if !matches!(x_shape.len(), 2 | 3) || !matches!(index_shape.len(), 2 | 3) {
                continue;
            }
            let Some(dim_norm) = wrap_dim(dim, x_shape.len()) else {
                continue;
            };
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
            let x_meta = graph.nodes[x_id].meta.clone();
            let (work_meta, needs_bool_cast) =
                orig_meta.tensor_stamp_bool_as_int8_work_dtype_of_or_float32(&x_meta);
            let index_meta = orig_meta.tensor_stamp("torch.int64");

            let mut values_in = x_id;

            // Optional bool→int8 cast
            if needs_bool_cast {
                let cast_id = graph.add_anchored_dtype_cast(
                    id,
                    format!("{}_bool_to_int8", node_name),
                    x_id,
                    work_meta.dtype(),
                    work_meta.preserve_qparams(x_shape.clone()),
                );
                values_in = cast_id;
            }

            let result = if x_shape.len() == 2 {
                // 2D path: view to 3D → GATHER → view back to 2D
                let v3d = graph.add_anchored_view_copy(
                    id,
                    format!("{}_values_3d", node_name),
                    values_in,
                    &[n, k, 1],
                    work_meta.preserve_qparams(vec![n, k, 1]),
                );

                let gathered = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather", node_name),
                    "backend.tosa.GATHER.default",
                    vec![IrArg::Node(v3d), IrArg::Node(index_id)],
                    work_meta.preserve_qparams(vec![n, w, 1]),
                );

                graph.add_anchored_view_copy(
                    id,
                    format!("{}_out_2d", node_name),
                    gathered,
                    &[n, w],
                    work_meta.preserve_qparams(vec![n, w]),
                )
            } else {
                // 3D path: permute(0,2,1) → reshape(N*C,K,1) → GATHER → reshape(N,C,W) → permute(0,2,1)
                let c = x_shape[2];

                let vperm = graph.add_anchored_permute_copy(
                    id,
                    format!("{}_values_perm", node_name),
                    values_in,
                    &[0, 2, 1],
                    work_meta.preserve_qparams(vec![n, c, k]),
                );

                let vresh = graph.add_anchored_view_copy(
                    id,
                    format!("{}_values_flat", node_name),
                    vperm,
                    &[n * c, k, 1],
                    work_meta.preserve_qparams(vec![n * c, k, 1]),
                );

                let iperm = graph.add_anchored_permute_copy(
                    id,
                    format!("{}_idx_perm", node_name),
                    index_id,
                    &[0, 2, 1],
                    index_meta.preserve_qparams(vec![n, c, w]),
                );

                let iresh = graph.add_anchored_view_copy(
                    id,
                    format!("{}_idx_flat", node_name),
                    iperm,
                    &[n * c, w],
                    index_meta.preserve_qparams(vec![n * c, w]),
                );

                let gathered = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather", node_name),
                    "backend.tosa.GATHER.default",
                    vec![IrArg::Node(vresh), IrArg::Node(iresh)],
                    work_meta.preserve_qparams(vec![n * c, w, 1]),
                );

                let gresh = graph.add_anchored_view_copy(
                    id,
                    format!("{}_gather_3d", node_name),
                    gathered,
                    &[n, c, w],
                    work_meta.preserve_qparams(vec![n, c, w]),
                );

                graph.add_permute_copy(
                    format!("{}_out_perm", node_name),
                    gresh,
                    &[0, 2, 1],
                    work_meta.preserve_qparams(vec![n, w, c]),
                )
            };

            // Optional int8→bool cast back
            let final_result = if needs_bool_cast {
                graph.add_anchored_dtype_cast(
                    id,
                    format!("{}_int8_to_bool", node_name),
                    result,
                    "torch.bool",
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
