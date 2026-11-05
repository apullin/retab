//! Pure Rust IrPass: aten.index.Tensor → SLICE/CONCAT on U55, otherwise linearized GATHER.

use crate::fx_constant_tensor_eval::placeholder_is_constant_input;
use crate::fx_index_domain::{
    gather_work_stamp_or_float32, index_stamp, node_index_dtype_or_default,
    slice_concat_for_indices,
};
use crate::fx_index_linear::{build_linear_index, packed_linear_scales, LinearIndexSpec};
use crate::fx_ir::{
    broadcast_shapes, i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeOp,
};
use crate::fx_target::is_index_tensor_target;
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

const MAX_U55_INDEX_TENSOR_ELEMENTS: usize = 64;

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

pub struct IrDecomposeIndexTensorPass {
    pub decompose_constant_indices: bool,
}

impl IrPass for IrDecomposeIndexTensorPass {
    fn name(&self) -> &str {
        "IrDecomposeIndexTensor"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_index_tensor_target(&graph.nodes[id].target) {
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

            let x_shape = match graph.node_shape_vec(x_id) {
                Some(s) => s,
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
                let Some(s) = graph.node_shape(nid) else {
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

            if self.decompose_constant_indices
                && idx_ids.len() == 1
                && idx_shapes[0].len() == 1
                && x_shape.iter().all(|&size| size >= 0)
            {
                let index_node = &graph.nodes[idx_ids[0]];
                if placeholder_is_constant_input(index_node) || index_node.op == NodeOp::GetAttr {
                    if let Some(mut indices) = index_node
                        .tensor
                        .as_ref()
                        .filter(|tensor| {
                            tensor.data.shape.len() == 1
                                && matches!(tensor.data.dtype, DType::Int32 | DType::Int64)
                        })
                        .and_then(|tensor| tensor.data.vec_i64())
                        .filter(|indices| {
                            !indices.is_empty() && indices.len() <= MAX_U55_INDEX_TENSOR_ELEMENTS
                        })
                    {
                        let dim = idx_positions[0];
                        let dim_size = x_shape[dim];
                        for index in &mut indices {
                            let original = *index;
                            if *index < 0 {
                                *index += dim_size;
                            }
                            assert!(
                                *index >= 0 && *index < dim_size,
                                "index {original} is out of bounds for dimension {dim} with size {dim_size}"
                            );
                        }
                        let result = slice_concat_for_indices(graph, id, x_id, dim, &indices);
                        graph.replace_and_erase(id, result);
                        modified = true;
                        continue;
                    }
                }
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
            let lin_scales = packed_linear_scales(&tensor_sizes);
            let perm: Vec<i64> = batch_positions
                .iter()
                .chain(idx_positions.iter())
                .chain(kept_positions.iter())
                .map(|&pos| pos as i64)
                .collect();

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let idx_dtype = node_index_dtype_or_default(graph, idx_ids[0]);
            let x_base_meta = graph.nodes[x_id].meta.clone();
            let x_meta = x_base_meta.tensor_stamp_same_dtype_or_float32();
            let index_meta = index_stamp(&orig_meta, &idx_dtype);

            // Reshape to (...batch, ...broadcast, ...kept)
            let out_shape = compute_index_output_shape(&batch_shape, &broadcast, &kept_shape);
            let result_meta = x_meta.preserve_qparams(out_shape.clone());

            let result = {
                let mut cursor = graph.rewrite_cursor(id);
                let mut values_in = x_id;
                if !is_identity_perm(&perm) {
                    let permuted_shape: Vec<i64> =
                        perm.iter().map(|&pos| x_shape[pos as usize]).collect();
                    values_in = cursor.permute_copy(
                        format!("{}_x_perm", node_name),
                        values_in,
                        &perm,
                        x_meta.preserve_qparams(permuted_shape),
                    );
                }

                // Flatten x to (B, K, C).
                let x_bkc = cursor.view_copy(
                    format!("{}_x_flat", node_name),
                    values_in,
                    &[batch, k, c],
                    x_meta.preserve_qparams(vec![batch, k, c]),
                );

                // TOSA GATHER does not support BOOL data; gather as INT8 and cast
                // the result back to BOOL below.
                let (gather_meta, x_is_bool) = gather_work_stamp_or_float32(&x_base_meta);
                let gather_x = if x_is_bool {
                    cursor.dtype_cast(
                        format!("{}_x_i8", node_name),
                        x_bkc,
                        gather_meta.dtype(),
                        gather_meta.drop_qparams(vec![batch, k, c]),
                    )
                } else {
                    x_bkc
                };

                let Some(lin_w_id) = build_linear_index(
                    &mut cursor,
                    LinearIndexSpec {
                        name: &node_name,
                        meta: &orig_meta,
                        index_ids: &idx_ids,
                        index_shapes: &idx_shapes,
                        broadcast_shape: &broadcast,
                        linear_scales: &lin_scales,
                        index_dtype: &idx_dtype,
                        flat_len: w,
                    },
                ) else {
                    continue;
                };

                // Unsqueeze to (1, W), then tile across batch if needed.
                let idx_bw_seed = cursor.call_function(
                    format!("{}_idx_1w", node_name),
                    "aten.unsqueeze_copy.default",
                    vec![IrArg::Node(lin_w_id), IrArg::Int(0)],
                    index_meta.drop_qparams(vec![1, w]),
                );

                let idx_bw = if batch == 1 {
                    idx_bw_seed
                } else {
                    cursor.call_function(
                        format!("{}_idx_bw", node_name),
                        "aten.expand_copy.default",
                        vec![IrArg::Node(idx_bw_seed), i64_list(&[batch, w])],
                        index_meta.drop_qparams(vec![batch, w]),
                    )
                };

                // GATHER(x_bkc, idx_bw) -> (B, W, C).
                let gathered = cursor.call_function(
                    format!("{}_gather", node_name),
                    "backend.tosa.GATHER.default",
                    vec![IrArg::Node(gather_x), IrArg::Node(idx_bw)],
                    gather_meta.preserve_qparams(vec![batch, w, c]),
                );
                let gathered = if x_is_bool {
                    cursor.dtype_cast(
                        format!("{}_gather_bool", node_name),
                        gathered,
                        "torch.bool",
                        x_meta.drop_qparams(vec![batch, w, c]),
                    )
                } else {
                    gathered
                };

                cursor.view_copy(
                    format!("{}_out", node_name),
                    gathered,
                    &out_shape,
                    result_meta,
                )
            };
            graph.replace_and_erase(id, result);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeMeta;
    use crate::store::TensorData;
    use crate::test_ir::{buffer, call, output, placeholder, qparam};

    fn constant_index_graph(indices: &[i64]) -> (IrGraph, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 5, 3], "torch.int16");
        let index = buffer(
            &mut graph,
            "index",
            TensorData {
                data: indices
                    .iter()
                    .flat_map(|index| index.to_le_bytes())
                    .collect(),
                shape: vec![indices.len() as i64],
                dtype: DType::Int64,
            },
        );
        graph.nodes[index].meta = NodeMeta::tensor([indices.len() as i64], "torch.int64");
        let indexed = call(
            &mut graph,
            "indexed",
            "aten.index.Tensor",
            vec![
                IrArg::Node(x),
                IrArg::List(vec![IrArg::None, IrArg::Node(index), IrArg::None]),
            ],
            [2, indices.len() as i64, 3],
            "torch.int16",
        );
        output(&mut graph, indexed);
        (graph, indexed, index)
    }

    #[test]
    fn contiguous_negative_indices_become_one_slice() {
        let (mut graph, indexed, _) = constant_index_graph(&[-4, -3, -2]);
        assert!(IrDecomposeIndexTensorPass {
            decompose_constant_indices: true
        }
        .run(&mut graph));
        let slice = &graph.nodes[graph.replacement_of(indexed).unwrap()];
        assert_eq!(slice.target, "aten.slice_copy.Tensor");
        assert_eq!(slice.arg_int(1), Some(1));
        assert_eq!(slice.arg_int(2), Some(1));
        assert_eq!(slice.arg_int(3), Some(4));
        assert_eq!(slice.meta.val_shape(), Some([2, 3, 3].as_slice()));
        assert_eq!(slice.meta.val_dtype(), Some("torch.int16"));
        assert_eq!(graph.live_call_function_ids(), vec![slice.id]);
    }

    #[test]
    fn reordered_negative_indices_reuse_slices_and_keep_qparams() {
        let (mut graph, indexed, _) = constant_index_graph(&[-1, 1, -1]);
        graph.nodes[indexed]
            .meta
            .set_both_qparams(qparam(0.2, 0, "torch.int16"));
        assert!(IrDecomposeIndexTensorPass {
            decompose_constant_indices: true
        }
        .run(&mut graph));
        let cat = &graph.nodes[graph.replacement_of(indexed).unwrap()];
        assert_eq!(cat.target, "aten.cat.default");
        assert_eq!(cat.arg_int(1), Some(1));
        assert!(cat.meta.has_nonempty_qparam_metadata());
        let IrArg::List(slices) = &cat.args[0] else {
            panic!("cat inputs")
        };
        assert_eq!(slices[0].as_node(), slices[2].as_node());
        for (slice, value) in slices.iter().zip([4, 1, 4]) {
            let slice = &graph.nodes[slice.as_node().unwrap()];
            assert_eq!(slice.arg_int(2), Some(value));
            assert_eq!(slice.arg_int(3), Some(value + 1));
            assert_eq!(slice.meta.val_shape(), Some([2, 1, 3].as_slice()));
            assert!(slice.meta.has_nonempty_qparam_metadata());
        }
        assert!(!graph
            .live_nodes()
            .any(|node| node.target == "backend.tosa.GATHER.default"));
    }

    #[test]
    fn u55_constant_index_limit_is_inclusive() {
        for count in [64, 65] {
            let indices: Vec<_> = (0..count).collect();
            let (mut graph, indexed, _) = constant_index_graph(&indices);
            let x = graph.nodes[indexed].first_node_arg().unwrap();
            graph.nodes[x].meta = NodeMeta::tensor([2, 70, 3], "torch.int16");
            assert!(IrDecomposeIndexTensorPass {
                decompose_constant_indices: true
            }
            .run(&mut graph));
            assert_eq!(
                graph
                    .live_nodes()
                    .any(|node| node.target == "backend.tosa.GATHER.default"),
                count > 64
            );
            assert_eq!(
                graph
                    .live_nodes()
                    .any(|node| node.target == "aten.slice_copy.Tensor"),
                count == 64
            );
        }
    }

    #[test]
    fn non_u55_constants_keep_gather() {
        let (mut graph, _, _) = constant_index_graph(&[1, 2]);
        assert!(IrDecomposeIndexTensorPass {
            decompose_constant_indices: false
        }
        .run(&mut graph));
        assert!(graph
            .live_nodes()
            .any(|node| node.target == "backend.tosa.GATHER.default"));
        assert!(!graph
            .live_nodes()
            .any(|node| node.target == "aten.slice_copy.Tensor"));
    }

    #[test]
    fn runtime_empty_multidimensional_and_multiple_indices_keep_gather() {
        let (mut runtime, _, index) = constant_index_graph(&[1, 2]);
        runtime.nodes[index].tensor = None;
        let (empty, _, _) = constant_index_graph(&[]);
        let (mut multidimensional, _, index) = constant_index_graph(&[1, 2]);
        multidimensional.nodes[index].meta = NodeMeta::tensor([1, 2], "torch.int64");
        multidimensional.nodes[index]
            .tensor
            .as_mut()
            .unwrap()
            .data
            .shape = vec![1, 2];
        let (mut multiple, indexed, index) = constant_index_graph(&[0, 1]);
        multiple.nodes[indexed].args[1] = IrArg::List(vec![IrArg::Node(index), IrArg::Node(index)]);
        let (mut symbolic, indexed, _) = constant_index_graph(&[1, 2]);
        let x = symbolic.nodes[indexed].first_node_arg().unwrap();
        symbolic.nodes[x].meta = NodeMeta::tensor([2, -1, 3], "torch.int16");
        for mut graph in [runtime, empty, multidimensional, multiple, symbolic] {
            assert!(IrDecomposeIndexTensorPass {
                decompose_constant_indices: true
            }
            .run(&mut graph));
            assert!(graph
                .live_nodes()
                .any(|node| node.target == "backend.tosa.GATHER.default"));
            assert!(!graph
                .live_nodes()
                .any(|node| node.target == "aten.slice_copy.Tensor"));
        }
    }

    #[test]
    fn constant_indices_are_bounds_checked_after_normalization() {
        for indices in [&[5][..], &[-6][..]] {
            let (mut graph, _, _) = constant_index_graph(indices);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                IrDecomposeIndexTensorPass {
                    decompose_constant_indices: true,
                }
                .run(&mut graph);
            }));
            assert!(result.is_err());
        }
    }
}
