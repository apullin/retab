//! Pure Rust IrPass: index_put → view + scatter + view for TOSA.

use crate::fx_ir::{
    broadcast_shapes, i64_list_arg as i64_list, shape_product, IrArg, IrGraph, IrPass, NodeId,
    NodeOp,
};
use crate::fx_target::is_index_put_default_target;
use crate::runner_logging::log_line;

fn shape_to_stride(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

pub struct IrRewriteIndexPutPass;

impl IrPass for IrRewriteIndexPutPass {
    fn name(&self) -> &str {
        "IrRewriteIndexPut"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
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

            let source_shape = match graph.nodes[source_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            if source_shape.is_empty() {
                continue;
            }
            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let src_dtype = graph.nodes[source_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

            let index_dtype = indices_list.iter().find_map(|item| match item {
                IrArg::Node(nid) => graph.nodes[*nid].meta.val_dtype().map(str::to_string),
                _ => None,
            });
            let Some(index_dtype) = index_dtype else {
                continue;
            };

            // Collect index node IDs and shapes, expanding None to arange
            let mut idx_ids: Vec<NodeId> = Vec::new();
            let mut idx_shapes: Vec<Vec<i64>> = Vec::new();
            let mut invalid_indices = false;
            for (dim_idx, item) in indices_list.iter().enumerate() {
                match item {
                    IrArg::Node(nid) => {
                        let Some(s) = graph.nodes[*nid].meta.val_shape() else {
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
                        let arange_meta =
                            orig_meta.cloned_without_qparams(vec![dim_size], &index_dtype);
                        let arange = graph.add_anchored_arange(
                            id,
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

            let strides = shape_to_stride(&source_shape);
            let lin_scales: Vec<i64> = strides.iter().take(num_idx).map(|s| s / c).collect();

            let idx_dtype = graph.nodes[idx_ids[0]]
                .meta
                .val_dtype()
                .unwrap_or("torch.int64")
                .to_string();

            // Build flat index
            let mut lin: Option<NodeId> = None;
            for (i, (&idx_id, lin_scale)) in idx_ids.iter().zip(lin_scales.iter()).enumerate() {
                let idx_shape = &idx_shapes[i];
                let mut current = idx_id;

                if idx_shape.len() != bcast.len() {
                    let mut aligned: Vec<i64> = vec![1; bcast.len() - idx_shape.len()];
                    aligned.extend(idx_shape);
                    let aligned_meta =
                        orig_meta.cloned_without_qparams(aligned.clone(), &idx_dtype);
                    let a = graph.add_anchored_call_function(
                        id,
                        format!("{}_ia{}", name, i),
                        "aten.view_copy.default".to_string(),
                        vec![IrArg::Node(current), i64_list(&aligned)],
                        aligned_meta,
                    );
                    current = a;
                }

                let bcast_meta = orig_meta.cloned_without_qparams(bcast.clone(), &idx_dtype);
                let b = graph.add_anchored_call_function(
                    id,
                    format!("{}_ib{}", name, i),
                    "aten.expand_copy.default".to_string(),
                    vec![IrArg::Node(current), i64_list(&bcast)],
                    bcast_meta,
                );

                let flat_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                let flat = graph.add_anchored_call_function(
                    id,
                    format!("{}_if{}", name, i),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(b), i64_list(&[w])],
                    flat_meta,
                );

                let scale_meta = orig_meta.cloned_without_qparams(vec![1], &idx_dtype);
                let scale = graph.add_anchored_full_int(
                    id,
                    format!("{}_sc{}", name, i),
                    &[1],
                    *lin_scale,
                    &idx_dtype,
                    scale_meta,
                );

                let scaled_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                let scaled = graph.add_anchored_call_function(
                    id,
                    format!("{}_sm{}", name, i),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(flat), IrArg::Node(scale)],
                    scaled_meta,
                );

                lin = Some(match lin {
                    None => scaled,
                    Some(prev) => {
                        let add_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                        let add = graph.add_anchored_call_function(
                            id,
                            format!("{}_la{}", name, i),
                            "aten.add.Tensor".to_string(),
                            vec![IrArg::Node(prev), IrArg::Node(scaled)],
                            add_meta,
                        );
                        add
                    }
                });
            }
            let Some(flat_idx) = lin else { continue };

            // Reshape: indices (1, W), source (1, K, C), values (1, W, C)
            let idx_meta = orig_meta.cloned_without_qparams(vec![1, w], &idx_dtype);
            let idx_1w = graph.add_anchored_call_function(
                id,
                format!("{}_idx", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(flat_idx), i64_list(&[1, w])],
                idx_meta,
            );

            let src_meta = graph.nodes[source_id]
                .meta
                .cloned_with_val(vec![1, k, c], &src_dtype)
                .with_qparams_from(&graph.nodes[source_id].meta);
            let src_1kc = graph.add_anchored_call_function(
                id,
                format!("{}_src", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(source_id), i64_list(&[1, k, c])],
                src_meta,
            );

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
            let mut val_cur = values_id;
            if values_shape.len() < bcast_val_shape.len() {
                let mut aligned = vec![1i64; bcast_val_shape.len() - values_shape.len()];
                aligned.extend_from_slice(&values_shape);
                let aligned_meta = graph.nodes[source_id]
                    .meta
                    .cloned_with_val(aligned.clone(), &src_dtype)
                    .with_qparams_from(&graph.nodes[source_id].meta);
                val_cur = graph.add_anchored_call_function(
                    id,
                    format!("{}_vala", name),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(val_cur), i64_list(&aligned)],
                    aligned_meta,
                );
            }
            let bcast_val_meta = graph.nodes[source_id]
                .meta
                .cloned_with_val(bcast_val_shape.clone(), &src_dtype)
                .with_qparams_from(&graph.nodes[source_id].meta);
            let val_bcast = graph.add_anchored_call_function(
                id,
                format!("{}_valb", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(val_cur), i64_list(&bcast_val_shape)],
                bcast_val_meta,
            );
            let val_meta = graph.nodes[source_id]
                .meta
                .cloned_with_val(vec![1, w, c], &src_dtype)
                .with_qparams_from(&graph.nodes[source_id].meta);
            let val_1wc = graph.add_anchored_call_function(
                id,
                format!("{}_val", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(val_bcast), i64_list(&[1, w, c])],
                val_meta,
            );

            let scatter_values = if accumulate {
                let gathered_meta = graph.nodes[source_id]
                    .meta
                    .cloned_with_val(vec![1, w, c], &src_dtype)
                    .with_qparams_from(&graph.nodes[source_id].meta);
                let gathered = graph.add_anchored_call_function(
                    id,
                    format!("{}_gather_prev", name),
                    "backend.tosa.GATHER.default".to_string(),
                    vec![IrArg::Node(src_1kc), IrArg::Node(idx_1w)],
                    gathered_meta,
                );

                let added_meta = graph.nodes[source_id]
                    .meta
                    .cloned_with_val(vec![1, w, c], &src_dtype)
                    .with_qparams_from(&graph.nodes[source_id].meta);
                let added = graph.add_anchored_call_function(
                    id,
                    format!("{}_accum", name),
                    "aten.add.Tensor".to_string(),
                    vec![IrArg::Node(gathered), IrArg::Node(val_1wc)],
                    added_meta,
                );
                added
            } else {
                val_1wc
            };

            // SCATTER(source_flat, indices_flat, values_flat)
            let scattered_meta = graph.nodes[source_id]
                .meta
                .cloned_with_val(vec![1, k, c], &src_dtype)
                .with_qparams_from(&graph.nodes[source_id].meta);
            let scattered = graph.add_anchored_call_function(
                id,
                format!("{}_scatter", name),
                "backend.tosa.SCATTER.default".to_string(),
                vec![
                    IrArg::Node(src_1kc),
                    IrArg::Node(idx_1w),
                    IrArg::Node(scatter_values),
                ],
                scattered_meta,
            );

            // Reshape back
            let result_meta = graph.nodes[source_id]
                .meta
                .cloned_with_val(source_shape.clone(), &src_dtype)
                .with_qparams_from(&graph.nodes[source_id].meta);
            graph.replace_with_call_function(
                id,
                format!("{}_out", name),
                "aten.view_copy.default",
                vec![IrArg::Node(scattered), i64_list(&source_shape)],
                result_meta,
            );
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
    use crate::fx_ir::NodeMeta;

    fn graph_with_index_put_none() -> IrGraph {
        let mut graph = IrGraph::empty();

        let source_meta = NodeMeta::tensor([2, 5, 4], "torch.float32");
        let source = graph.add_node(
            NodeOp::Placeholder,
            "source".to_string(),
            "source".to_string(),
            vec![],
            source_meta.clone(),
        );

        let index_meta = NodeMeta::tensor([1], "torch.int64");
        let index = graph.add_node(
            NodeOp::Placeholder,
            "index".to_string(),
            "index".to_string(),
            vec![],
            index_meta,
        );

        let values_meta = NodeMeta::tensor([2, 4], "torch.float32");
        let values = graph.add_node(
            NodeOp::Placeholder,
            "values".to_string(),
            "values".to_string(),
            vec![],
            values_meta,
        );

        let index_put = graph.add_node(
            NodeOp::CallFunction,
            "index_put".to_string(),
            "index_put_.default".to_string(),
            vec![
                IrArg::Node(source),
                IrArg::List(vec![IrArg::None, IrArg::Node(index)]),
                IrArg::Node(values),
                IrArg::Bool(false),
            ],
            source_meta,
        );

        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(index_put)],
            NodeMeta::default(),
        );

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
