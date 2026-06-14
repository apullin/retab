//! Pure Rust IrPass: index_select → view + unsqueeze + expand + GATHER + view.

use crate::fx_ir::{dtype_kwarg, i64_list_arg as i64_list, shape_product, IrArg, IrGraph, IrPass};
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

            let x_rank = x_shape.len() as i64;
            let dim_norm = dim.rem_euclid(x_rank) as usize;
            let before: Vec<i64> = x_shape[..dim_norm].to_vec();
            let after: Vec<i64> = x_shape[dim_norm + 1..].to_vec();
            let n = shape_product(&before);
            let k = x_shape[dim_norm];
            let c = shape_product(&after);
            let w = index_shape[0];

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_dtype = graph.node_dtype_or_float32(x_id).to_string();
            let idx_dtype = graph
                .node_dtype(index_id)
                .unwrap_or("torch.int64")
                .to_string();

            let needs_bool_cast = x_dtype == "torch.bool";
            let gather_dtype = if needs_bool_cast {
                "torch.int8"
            } else {
                x_dtype.as_str()
            };

            let x_for_gather = if needs_bool_cast {
                let cast = graph.add_anchored_call_function_with_kwargs(
                    id,
                    format!("{}_bool_to_i8", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(x_id)],
                    dtype_kwarg("torch.int8"),
                    orig_meta.cloned_with_val(x_shape.clone(), "torch.int8"),
                );
                cast
            } else {
                x_id
            };

            // Flatten x to (N, K, C)
            let values = graph.add_anchored_call_function(
                id,
                format!("{}_values", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(x_for_gather), i64_list(&[n, k, c])],
                orig_meta.cloned_with_val(vec![n, k, c], gather_dtype),
            );

            // Index: unsqueeze(0) → (1, W), then expand to (N, W)
            let idx_unsq = graph.add_anchored_call_function(
                id,
                format!("{}_idx_unsq", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(index_id), IrArg::Int(0)],
                orig_meta.cloned_with_val(vec![1, w], &idx_dtype),
            );

            let idx_exp = graph.add_anchored_call_function(
                id,
                format!("{}_idx_exp", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[n, w])],
                orig_meta.cloned_with_val(vec![n, w], &idx_dtype),
            );

            // GATHER(values, indices) → (N, W, C)
            let gathered = graph.add_anchored_call_function(
                id,
                format!("{}_gather", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                orig_meta.cloned_with_val(vec![n, w, c], gather_dtype),
            );

            // Reshape to (...before, W, ...after)
            let out_shape: Vec<i64> = before
                .iter()
                .copied()
                .chain(std::iter::once(w))
                .chain(after.iter().copied())
                .collect();
            if needs_bool_cast {
                let result = graph.add_anchored_call_function(
                    id,
                    format!("{}_out", name),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(gathered), i64_list(&out_shape)],
                    orig_meta.cloned_with_val(out_shape.clone(), gather_dtype),
                );
                graph.replace_with_call_function_with_kwargs(
                    id,
                    format!("{}_i8_to_bool", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(result)],
                    dtype_kwarg("torch.bool"),
                    orig_meta.cloned_with_val(out_shape, "torch.bool"),
                );
            } else {
                graph.replace_with_call_function(
                    id,
                    format!("{}_out", name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(gathered), i64_list(&out_shape)],
                    orig_meta.cloned_with_val(out_shape, gather_dtype),
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
