//! Pure Rust IrPass: select_scatter → arange + view + full + eq + unsqueeze + expand + where.

use crate::fx_index_domain::{bool_stamp, int32_index_stamp, TOSA_INDEX_DTYPE};
use crate::fx_ir::{i64_list_arg as i64_list, wrap_dim, IrArg, IrGraph, IrPass};
use crate::fx_target::is_select_scatter_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeSelectScatterPass;

impl IrPass for IrDecomposeSelectScatterPass {
    fn name(&self) -> &str {
        "IrDecomposeSelectScatter"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_select_scatter_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() != 4 {
                continue;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let src_id = match graph.nodes[id].args[1].as_node() {
                Some(n) => n,
                None => continue,
            };
            let dim = match graph.nodes[id].args[2].as_int() {
                Some(v) => v,
                None => continue,
            };
            let mut index = match graph.nodes[id].args[3].as_int() {
                Some(v) => v,
                None => continue,
            };

            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };
            if input_shape.is_empty() {
                continue;
            }
            let Some(dim_norm) = wrap_dim(dim, input_shape.len()) else {
                continue;
            };
            let dim_size = input_shape[dim_norm];
            if index < 0 {
                index += dim_size;
            }

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_meta = graph.nodes[input_id].meta.clone();
            let src_meta = graph.nodes[src_id].meta.clone();
            let index_meta = int32_index_stamp(&orig_meta);
            let bool_meta = bool_stamp(&orig_meta);
            let src_shape = src_meta.val_shape().map(|s| s.to_vec()).unwrap_or_default();
            let mut unsqueezed_shape = src_shape.clone();
            unsqueezed_shape.insert(dim_norm, 1);
            let unsqueeze_meta =
                src_meta.cloned_with_dtype_of_or_float32(unsqueezed_shape, &input_meta);
            let expand_meta =
                src_meta.cloned_with_dtype_of_or_float32(input_shape.clone(), &input_meta);

            let mut cursor = graph.rewrite_cursor(id);

            // 1. arange(0, dim_size) -> shape [dim_size], dtype int32
            let arange_id = cursor.arange(
                format!("{}_arange", node_name),
                0,
                dim_size,
                1,
                TOSA_INDEX_DTYPE,
                index_meta.preserve_qparams(vec![dim_size]),
            );

            // 2. view(arange, [1, ..., dim_size, ..., 1])
            let mut mask_shape = vec![1i64; input_shape.len()];
            mask_shape[dim_norm] = dim_size;
            let viewed_arange = cursor.view_copy(
                format!("{}_arange_view", node_name),
                arange_id,
                &mask_shape,
                index_meta.preserve_qparams(mask_shape.clone()),
            );

            // 3. full([1], index, dtype=int32) - scalar index tensor
            let index_tensor = cursor.full_int(
                format!("{}_index_full", node_name),
                &[1],
                index,
                TOSA_INDEX_DTYPE,
                index_meta.preserve_qparams(vec![1]),
            );

            // 4. eq(viewed_arange, index_tensor) -> boolean mask
            let eq_mask = cursor.call_function(
                format!("{}_eq", node_name),
                "aten.eq.Tensor",
                vec![IrArg::Node(viewed_arange), IrArg::Node(index_tensor)],
                bool_meta.preserve_qparams(mask_shape),
            );

            // 5. unsqueeze(src, dim) - restore dimension that select removed
            let unsqueeze_node = cursor.call_function(
                format!("{}_unsqueeze", node_name),
                "aten.unsqueeze_copy.default",
                vec![IrArg::Node(src_id), IrArg::Int(dim_norm as i64)],
                unsqueeze_meta,
            );

            // 6. expand(unsqueezed, input_shape)
            let expand_node = cursor.call_function(
                format!("{}_expand", node_name),
                "aten.expand_copy.default",
                vec![IrArg::Node(unsqueeze_node), i64_list(&input_shape)],
                expand_meta,
            );

            // 7. where(mask, expanded_src, input) -> result
            let result = cursor.call_function(
                format!("{}_where", node_name),
                "aten.where.self",
                vec![
                    IrArg::Node(eq_mask),
                    IrArg::Node(expand_node),
                    IrArg::Node(input_id),
                ],
                orig_meta,
            );
            cursor.replace_anchor_with(result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeSelectScatter] decomposed select_scatter ops"
            ));
        }
        modified
    }
}
