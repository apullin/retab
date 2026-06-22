//! Pure Rust IrPass: index_put → view + scatter + view for TOSA.

use crate::fx_index_domain::{first_index_dtype, index_stamp, node_index_dtype_or_default};
use crate::fx_index_linear::{build_linear_index, contiguous_strides, LinearIndexSpec};
use crate::fx_ir::{
    broadcast_shapes, i64_list_arg as i64_list, shape_product, IrArg, IrGraph, IrPass, NodeId,
};
use crate::fx_target::is_index_put_default_target;
use crate::runner_logging::log_line;

pub struct IrRewriteIndexPutPass;

impl IrPass for IrRewriteIndexPutPass {
    fn name(&self) -> &str {
        "IrRewriteIndexPut"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_index_put_default_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            let source_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let indices_list = match &graph.nodes[id].args[1] {
                IrArg::List(items) => items.clone(),
                _ => continue,
            };
            let values_id = match graph.nodes[id].args[2].as_node() {
                Some(n) => n,
                None => continue,
            };
            let accumulate = graph.nodes[id]
                .args
                .get(3)
                .and_then(|a| a.as_bool())
                .unwrap_or(false);

            let source_shape = match graph.node_shape_vec(source_id) {
                Some(s) => s,
                None => continue,
            };
            if source_shape.is_empty() {
                continue;
            }
            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let source_base_meta = graph.nodes[source_id].meta.clone();
            let source_meta = source_base_meta.tensor_stamp_same_dtype_or_float32();

            let index_dtype = first_index_dtype(graph, &indices_list);
            let Some(index_dtype) = index_dtype else {
                continue;
            };
            let index_meta = index_stamp(&orig_meta, &index_dtype);

            // Collect index node IDs and shapes, expanding None to arange.
            let (idx_ids, idx_shapes, invalid_indices) = {
                let mut cursor = graph.rewrite_cursor(id);
                let mut idx_ids: Vec<NodeId> = Vec::new();
                let mut idx_shapes: Vec<Vec<i64>> = Vec::new();
                let mut invalid_indices = false;
                for (dim_idx, item) in indices_list.iter().enumerate() {
                    match item {
                        IrArg::Node(nid) => {
                            let Some(s) = cursor.graph().node_shape(*nid) else {
                                invalid_indices = true;
                                break;
                            };
                            idx_ids.push(*nid);
                            idx_shapes.push(s.to_vec());
                        }
                        IrArg::None => {
                            let Some(&dim_size) = source_shape.get(dim_idx) else {
                                invalid_indices = true;
                                break;
                            };
                            let arange_meta = index_meta.drop_qparams(vec![dim_size]);
                            let arange = cursor.arange(
                                format!("{}_in{}", name, dim_idx),
                                0,
                                dim_size,
                                1,
                                &index_dtype,
                                arange_meta,
                            );
                            idx_ids.push(arange);
                            idx_shapes.push(vec![dim_size]);
                        }
                        _ => {
                            invalid_indices = true;
                            break;
                        }
                    }
                }
                (idx_ids, idx_shapes, invalid_indices)
            };
            if invalid_indices || idx_ids.is_empty() {
                continue;
            }

            let Some(bcast) = broadcast_shapes(&idx_shapes) else {
                continue;
            };
            let num_idx = idx_ids.len();
            let w = shape_product(&bcast).max(1);
            let mut k = 1i64;
            for i in 0..num_idx {
                k *= source_shape.get(i).copied().unwrap_or(1);
            }
            let total = shape_product(&source_shape);
            let c = if k != 0 { total / k } else { 1 };
            if !(k > 0 && w > 0 && c > 0) {
                continue;
            }

            let strides = contiguous_strides(&source_shape);
            let lin_scales: Vec<i64> = strides.iter().take(num_idx).map(|s| s / c).collect();

            let idx_dtype = node_index_dtype_or_default(graph, idx_ids[0]);

            let flat_idx = {
                let mut cursor = graph.rewrite_cursor(id);
                build_linear_index(
                    &mut cursor,
                    LinearIndexSpec {
                        name: &name,
                        meta: &orig_meta,
                        index_ids: &idx_ids,
                        index_shapes: &idx_shapes,
                        broadcast_shape: &bcast,
                        linear_scales: &lin_scales,
                        index_dtype: &idx_dtype,
                        flat_len: w,
                    },
                )
            };
            let Some(flat_idx) = flat_idx else {
                continue;
            };

            // Broadcast values to (W, *C_j) before the (1, W, C) view: a bare view
            // is invalid when values needs broadcasting (e.g. scalar data). Rank-
            // align with a leading-1 view first, as the index path does. Mirrors
            // master RewriteIndexPutPass.
            let c_j: Vec<i64> = source_shape[num_idx..].to_vec();
            let mut bcast_val_shape: Vec<i64> = Vec::with_capacity(1 + c_j.len());
            bcast_val_shape.push(w);
            bcast_val_shape.extend_from_slice(&c_j);
            let values_shape = graph.nodes[values_id]
                .meta
                .val_shape()
                .map(|s| s.to_vec())
                .unwrap_or_default();

            let result = {
                let mut cursor = graph.rewrite_cursor_after(id, flat_idx);

                // Reshape: indices (1, W), source (1, K, C), values (1, W, C).
                let idx_1w = cursor.view_copy(
                    format!("{}_idx", name),
                    flat_idx,
                    &[1, w],
                    index_meta.drop_qparams(vec![1, w]),
                );

                let src_1kc = cursor.view_copy(
                    format!("{}_src", name),
                    source_id,
                    &[1, k, c],
                    source_meta.preserve_qparams(vec![1, k, c]),
                );

                let mut val_cur = values_id;
                if values_shape.len() < bcast_val_shape.len() {
                    let mut aligned = vec![1i64; bcast_val_shape.len() - values_shape.len()];
                    aligned.extend_from_slice(&values_shape);
                    let aligned_meta = source_meta.preserve_qparams(aligned.clone());
                    val_cur =
                        cursor.view_copy(format!("{}_vala", name), val_cur, &aligned, aligned_meta);
                }
                let val_bcast = cursor.call_function(
                    format!("{}_valb", name),
                    "aten.expand_copy.default",
                    vec![IrArg::Node(val_cur), i64_list(&bcast_val_shape)],
                    source_meta.preserve_qparams(bcast_val_shape.clone()),
                );
                let val_1wc = cursor.view_copy(
                    format!("{}_val", name),
                    val_bcast,
                    &[1, w, c],
                    source_meta.preserve_qparams(vec![1, w, c]),
                );

                let scatter_values = if accumulate {
                    let gathered = cursor.call_function(
                        format!("{}_gather_prev", name),
                        "backend.tosa.GATHER.default",
                        vec![IrArg::Node(src_1kc), IrArg::Node(idx_1w)],
                        source_meta.preserve_qparams(vec![1, w, c]),
                    );

                    cursor.call_function(
                        format!("{}_accum", name),
                        "aten.add.Tensor",
                        vec![IrArg::Node(gathered), IrArg::Node(val_1wc)],
                        source_meta.preserve_qparams(vec![1, w, c]),
                    )
                } else {
                    val_1wc
                };

                // SCATTER(source_flat, indices_flat, values_flat), then reshape back.
                let scattered = cursor.call_function(
                    format!("{}_scatter", name),
                    "backend.tosa.SCATTER.default",
                    vec![
                        IrArg::Node(src_1kc),
                        IrArg::Node(idx_1w),
                        IrArg::Node(scatter_values),
                    ],
                    source_meta.preserve_qparams(vec![1, k, c]),
                );

                cursor.view_copy(
                    format!("{}_out", name),
                    scattered,
                    &source_shape,
                    source_meta.preserve_qparams(source_shape.clone()),
                )
            };
            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrRewriteIndexPut] rewrote index_put to SCATTER"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call_with_meta, f32_meta, f32_placeholder, output, placeholder};

    fn graph_with_index_put_none() -> IrGraph {
        let mut graph = IrGraph::empty();

        let source_meta = f32_meta([2, 5, 4]);
        let source = f32_placeholder(&mut graph, "source", [2, 5, 4]);
        let index = placeholder(&mut graph, "index", [1], "torch.int64");
        let values = f32_placeholder(&mut graph, "values", [2, 4]);
        let index_put = call_with_meta(
            &mut graph,
            "index_put",
            "index_put_.default",
            vec![
                IrArg::Node(source),
                IrArg::List(vec![IrArg::None, IrArg::Node(index)]),
                IrArg::Node(values),
                IrArg::Bool(false),
            ],
            source_meta,
        );

        output(&mut graph, index_put);

        graph
    }

    #[test]
    fn rewrites_index_put_underscore_with_none_index() {
        let mut graph = graph_with_index_put_none();

        let modified = IrRewriteIndexPutPass.run(&mut graph);

        assert!(modified);
        assert!(!graph.nodes[3].alive);
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| { node.alive && node.target == "aten.arange.start_step" })
                .count(),
            1
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| { node.alive && node.target == "backend.tosa.SCATTER.default" })
                .count(),
            1
        );
        assert_eq!(
            graph.output_node().and_then(|node| node.first_node_arg()),
            graph.by_name("index_put_out").map(|node| node.id),
        );
        assert_eq!(
            graph
                .by_name("index_put_idx")
                .and_then(|node| node.meta.val_shape()),
            Some(&[1, 2][..]),
        );
        assert_eq!(
            graph
                .by_name("index_put_src")
                .and_then(|node| node.meta.val_shape()),
            Some(&[1, 10, 4][..]),
        );
        assert_eq!(
            graph
                .by_name("index_put_val")
                .and_then(|node| node.meta.val_shape()),
            Some(&[1, 2, 4][..]),
        );
        assert_eq!(
            graph
                .by_name("index_put_out")
                .and_then(|node| node.meta.val_shape()),
            Some(&[2, 5, 4][..]),
        );
    }
}
