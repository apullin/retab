//! Pure Rust IrPass: index_select → view + unsqueeze + expand + GATHER + view.

use crate::fx_ir::{i64_list_arg as i64_list, shape_product, wrap_dim, IrArg, IrGraph, IrPass};
use crate::fx_target::is_index_select_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeIndexSelectPass;

impl IrPass for IrDecomposeIndexSelectPass {
    fn name(&self) -> &str {
        "IrDecomposeIndexSelect"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_index_select_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() != 3 {
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
            if x_shape.is_empty() || index_shape.len() != 1 {
                continue;
            }

            let Some(dim_norm) = wrap_dim(dim, x_shape.len()) else {
                continue;
            };
            let before: Vec<i64> = x_shape[..dim_norm].to_vec();
            let after: Vec<i64> = x_shape[dim_norm + 1..].to_vec();
            let n = shape_product(&before);
            let k = x_shape[dim_norm];
            let c = shape_product(&after);
            let w = index_shape[0];

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let idx_dtype = graph
                .node_dtype(index_id)
                .unwrap_or("torch.int64")
                .to_string();

            let x_meta = graph.nodes[x_id].meta.clone();
            let (gather_meta, needs_bool_cast) =
                orig_meta.tensor_stamp_bool_as_int8_work_dtype_of_or_float32(&x_meta);
            let index_meta = orig_meta.tensor_stamp(&idx_dtype);
            let bool_meta = orig_meta.tensor_stamp("torch.bool");

            let x_for_gather = if needs_bool_cast {
                graph.add_anchored_dtype_cast(
                    id,
                    format!("{}_bool_to_i8", name),
                    x_id,
                    gather_meta.dtype(),
                    gather_meta.preserve_qparams(x_shape.clone()),
                )
            } else {
                x_id
            };

            // Flatten x to (N, K, C)
            let values = graph.add_anchored_view_copy(
                id,
                format!("{}_values", name),
                x_for_gather,
                &[n, k, c],
                gather_meta.preserve_qparams(vec![n, k, c]),
            );

            // Index: unsqueeze(0) → (1, W), then expand to (N, W)
            let idx_unsq = graph.add_anchored_call_function(
                id,
                format!("{}_idx_unsq", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(index_id), IrArg::Int(0)],
                index_meta.preserve_qparams(vec![1, w]),
            );

            let idx_exp = graph.add_anchored_call_function(
                id,
                format!("{}_idx_exp", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[n, w])],
                index_meta.preserve_qparams(vec![n, w]),
            );

            // GATHER(values, indices) → (N, W, C)
            let gathered = graph.add_anchored_call_function(
                id,
                format!("{}_gather", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                gather_meta.preserve_qparams(vec![n, w, c]),
            );

            // Reshape to (...before, W, ...after)
            let out_shape: Vec<i64> = before
                .iter()
                .copied()
                .chain(std::iter::once(w))
                .chain(after.iter().copied())
                .collect();
            if needs_bool_cast {
                let result = graph.add_anchored_view_copy(
                    id,
                    format!("{}_out", name),
                    gathered,
                    &out_shape,
                    gather_meta.preserve_qparams(out_shape.clone()),
                );
                graph.replace_with_dtype_cast(
                    id,
                    format!("{}_i8_to_bool", name),
                    result,
                    "torch.bool",
                    bool_meta.preserve_qparams(out_shape),
                );
            } else {
                graph.replace_with_view_copy(
                    id,
                    format!("{}_out", name),
                    gathered,
                    &out_shape,
                    gather_meta.preserve_qparams(out_shape.clone()),
                );
            }
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeIndexSelect] decomposed index_select ops"
            ));
        }
        modified
    }
}
