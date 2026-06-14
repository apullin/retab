//! Pure Rust IrPass: aten.index.Tensor → linearized GATHER.

use crate::fx_ir::{
    broadcast_shapes, i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeOp,
};
use crate::runner_logging::log_line;

fn compute_index_output_shape(
    batch_shape: &[i64],
    broadcast: &[i64],
    kept_shape: &[i64],
) -> Vec<i64> {
    let mut out = batch_shape.to_vec();
    out.extend_from_slice(broadcast);
    out.extend_from_slice(kept_shape);
    out
}

fn is_identity_perm(perm: &[i64]) -> bool {
    perm.iter().enumerate().all(|(i, &value)| value == i as i64)
}

fn packed_lin_scales(tensor_sizes: &[i64]) -> Vec<i64> {
    let mut scales = vec![1; tensor_sizes.len()];
    let mut running = 1i64;
    for (i, size) in tensor_sizes.iter().enumerate().rev() {
        scales[i] = running;
        running *= *size;
    }
    scales
}

pub struct IrDecomposeIndexTensorPass;

impl IrPass for IrDecomposeIndexTensorPass {
    fn name(&self) -> &str {
        "IrDecomposeIndexTensor"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("aten.index.Tensor") {
                continue;
            }
            if graph.nodes[id].args.len() != 2 {
                continue;
            }

            let x_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let index_list = match &graph.nodes[id].args[1] {
                IrArg::List(items) => items.clone(),
                _ => continue,
            };
            if index_list.is_empty() {
                continue;
            }

            let x_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            if x_shape.is_empty() {
                continue;
            }

            let prefix_len = index_list.len();
            if prefix_len > x_shape.len() {
                continue;
            }

            // Collect tensor index node IDs/shapes; None markers preserve the
            // corresponding prefix dimension in the output.
            let mut idx_ids: Vec<NodeId> = Vec::new();
            let mut idx_shapes: Vec<Vec<i64>> = Vec::new();
            let mut idx_positions: Vec<usize> = Vec::new();
            let mut all_valid = true;
            for (pos, item) in index_list.iter().enumerate() {
                let Some(nid) = item.as_node() else {
                    if matches!(item, IrArg::None) {
                        continue;
                    }
                    all_valid = false;
                    break;
                };
                let Some(s) = graph.nodes[nid].meta.val_shape() else {
                    all_valid = false;
                    break;
                };
                idx_ids.push(nid);
                idx_shapes.push(s.to_vec());
                idx_positions.push(pos);
            }
            if !all_valid || idx_ids.is_empty() {
                continue;
            }

            let Some(broadcast) = broadcast_shapes(&idx_shapes) else {
                continue;
            };
            let first_tensor_pos = idx_positions[0];
            let batch_positions: Vec<usize> = (0..first_tensor_pos).collect();
            let kept_positions: Vec<usize> = (first_tensor_pos..x_shape.len())
                .filter(|pos| *pos >= prefix_len || !idx_positions.contains(pos))
                .collect();
            let batch_shape: Vec<i64> = batch_positions.iter().map(|&pos| x_shape[pos]).collect();
            let tensor_sizes: Vec<i64> = idx_positions.iter().map(|&pos| x_shape[pos]).collect();
            let kept_shape: Vec<i64> = kept_positions.iter().map(|&pos| x_shape[pos]).collect();
            let batch = batch_shape.iter().copied().product::<i64>().max(1);
            let w = broadcast.iter().copied().product::<i64>().max(1);
            let k = tensor_sizes.iter().copied().product::<i64>().max(1);
            let c = kept_shape.iter().copied().product::<i64>().max(1);
            let lin_scales = packed_lin_scales(&tensor_sizes);
            let perm: Vec<i64> = batch_positions
                .iter()
                .chain(idx_positions.iter())
                .chain(kept_positions.iter())
                .map(|&pos| pos as i64)
                .collect();

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_dtype = graph.nodes[x_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let idx_dtype = graph.nodes[idx_ids[0]]
                .meta
                .val_dtype()
                .unwrap_or("torch.int64")
                .to_string();

            let mut values_in = x_id;
            if !is_identity_perm(&perm) {
                let permuted_shape: Vec<i64> =
                    perm.iter().map(|&pos| x_shape[pos as usize]).collect();
                let perm_meta = graph.nodes[x_id]
                    .meta
                    .cloned_with_val(permuted_shape, &x_dtype)
                    .with_qparams_from(&graph.nodes[x_id].meta);
                let perm_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_x_perm", node_name),
                    "aten.permute_copy.default".to_string(),
                    vec![IrArg::Node(values_in), i64_list(&perm)],
                    perm_meta,
                );
                values_in = perm_id;
            }

            // Flatten x to (B, K, C)
            let x_bkc_meta = graph.nodes[x_id]
                .meta
                .cloned_with_val(vec![batch, k, c], &x_dtype)
                .with_qparams_from(&graph.nodes[x_id].meta);
            let x_bkc = graph.add_anchored_call_function(
                id,
                format!("{}_x_flat", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(values_in), i64_list(&[batch, k, c])],
                x_bkc_meta,
            );

            // TOSA GATHER does not support BOOL data; gather as INT8 and cast the
            // result back to BOOL below.
            let x_is_bool = x_dtype.contains("bool");
            let gather_dtype: String = if x_is_bool {
                "torch.int8".to_string()
            } else {
                x_dtype.clone()
            };
            let gather_x = if x_is_bool {
                let mut kwargs = std::collections::HashMap::new();
                kwargs.insert("dtype".to_string(), IrArg::Dtype(gather_dtype.clone()));
                let cast_meta = graph.nodes[x_id]
                    .meta
                    .cloned_without_qparams(vec![batch, k, c], &gather_dtype);
                graph.add_anchored_call_function_with_kwargs(
                    id,
                    format!("{}_x_i8", node_name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(x_bkc)],
                    kwargs,
                    cast_meta,
                )
            } else {
                x_bkc
            };

            // Build linearized flat index
            let mut lin_w: Option<NodeId> = None;
            for (i, (&idx_id, (idx_shape, &lin_scale))) in idx_ids
                .iter()
                .zip(idx_shapes.iter().zip(lin_scales.iter()))
                .enumerate()
            {
                let mut current = idx_id;

                // Align rank if needed
                if idx_shape.len() != broadcast.len() {
                    let mut aligned: Vec<i64> = vec![1; broadcast.len() - idx_shape.len()];
                    aligned.extend(idx_shape);
                    let aligned_meta =
                        orig_meta.cloned_without_qparams(aligned.clone(), &idx_dtype);
                    let aligned_id = graph.add_anchored_call_function(
                        id,
                        format!("{}_idx{}_align", node_name, i),
                        "aten.view_copy.default".to_string(),
                        vec![IrArg::Node(current), i64_list(&aligned)],
                        aligned_meta,
                    );
                    current = aligned_id;
                }

                // Broadcast to common shape
                let bcast_meta = orig_meta.cloned_without_qparams(broadcast.clone(), &idx_dtype);
                let bcast_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx{}_bcast", node_name, i),
                    "aten.expand_copy.default".to_string(),
                    vec![IrArg::Node(current), i64_list(&broadcast)],
                    bcast_meta,
                );

                // Flatten to (W,)
                let flat_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                let flat_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx{}_flat", node_name, i),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(bcast_id), i64_list(&[w])],
                    flat_meta,
                );

                // Scale by linearized stride
                let scale_meta = orig_meta.cloned_without_qparams(vec![1], &idx_dtype);
                let scale_id = graph.add_anchored_full_int(
                    id,
                    format!("{}_scale{}", node_name, i),
                    &[1],
                    lin_scale,
                    &idx_dtype,
                    scale_meta,
                );

                let scaled_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                let scaled_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx{}_scaled", node_name, i),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(flat_id), IrArg::Node(scale_id)],
                    scaled_meta,
                );

                lin_w = Some(match lin_w {
                    None => scaled_id,
                    Some(prev) => {
                        let add_meta = orig_meta.cloned_without_qparams(vec![w], &idx_dtype);
                        let add_id = graph.add_anchored_call_function(
                            id,
                            format!("{}_lin_add{}", node_name, i),
                            "aten.add.Tensor".to_string(),
                            vec![IrArg::Node(prev), IrArg::Node(scaled_id)],
                            add_meta,
                        );
                        add_id
                    }
                });
            }

            let Some(lin_w_id) = lin_w else { continue };

            // Unsqueeze to (1, W), then tile across batch if needed.
            let idx_1w_meta = orig_meta.cloned_without_qparams(vec![1, w], &idx_dtype);
            let idx_bw_seed = graph.add_anchored_call_function(
                id,
                format!("{}_idx_1w", node_name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(lin_w_id), IrArg::Int(0)],
                idx_1w_meta,
            );

            let idx_bw = if batch == 1 {
                idx_bw_seed
            } else {
                let expand_meta = orig_meta.cloned_without_qparams(vec![batch, w], &idx_dtype);
                let expand_id = graph.add_anchored_call_function(
                    id,
                    format!("{}_idx_bw", node_name),
                    "aten.expand_copy.default".to_string(),
                    vec![IrArg::Node(idx_bw_seed), i64_list(&[batch, w])],
                    expand_meta,
                );
                expand_id
            };

            // GATHER(x_bkc, idx_bw) → (B, W, C)
            let gathered_meta = graph.nodes[x_id]
                .meta
                .cloned_with_val(vec![batch, w, c], &gather_dtype)
                .with_qparams_from(&graph.nodes[x_id].meta);
            let gathered = graph.add_anchored_call_function(
                id,
                format!("{}_gather", node_name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(gather_x), IrArg::Node(idx_bw)],
                gathered_meta,
            );
            let gathered = if x_is_bool {
                let mut kwargs = std::collections::HashMap::new();
                kwargs.insert("dtype".to_string(), IrArg::Dtype(x_dtype.clone()));
                let cast_meta = graph.nodes[x_id]
                    .meta
                    .cloned_without_qparams(vec![batch, w, c], &x_dtype);
                graph.add_anchored_call_function_with_kwargs(
                    id,
                    format!("{}_gather_bool", node_name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(gathered)],
                    kwargs,
                    cast_meta,
                )
            } else {
                gathered
            };

            // Reshape to (...batch, ...broadcast, ...kept)
            let out_shape = compute_index_output_shape(&batch_shape, &broadcast, &kept_shape);
            let result_meta = graph.nodes[x_id]
                .meta
                .cloned_with_val(out_shape.clone(), &x_dtype)
                .with_qparams_from(&graph.nodes[x_id].meta);
            graph.replace_with_call_function(
                id,
                format!("{}_out", node_name),
                "aten.view_copy.default",
                vec![IrArg::Node(gathered), i64_list(&out_shape)],
                result_meta,
            );
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeIndexTensor] decomposed index.Tensor ops"
            ));
        }
        modified
    }
}
