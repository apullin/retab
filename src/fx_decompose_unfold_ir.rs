//! Pure Rust IrPass: unfold_copy → slices on U55, otherwise a GATHER-based sliding window.

use crate::fx_index_domain::{
    bool_stamp, gather_work_stamp_or_float32, int32_index_stamp, TOSA_INDEX_DTYPE,
};
use crate::fx_ir::{i64_list_arg as i64_list, shape_product, wrap_dim, IrArg, IrGraph, IrPass};
use crate::fx_target::is_unfold_copy_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeUnfoldPass {
    pub use_slice: bool,
}

impl IrPass for IrDecomposeUnfoldPass {
    fn name(&self) -> &str {
        "IrDecomposeUnfold"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_unfold_copy_target(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() != 4 {
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
            let size = match graph.nodes[id].args[2].as_int() {
                Some(v) => v,
                None => continue,
            };
            let step = match graph.nodes[id].args[3].as_int() {
                Some(v) => v,
                None => continue,
            };
            if size <= 0 || step <= 0 {
                continue;
            }

            let input_shape = match graph.node_shape_vec(x_id) {
                Some(s) => s,
                None => continue,
            };
            let Some(dim_norm) = wrap_dim(dim, input_shape.len()) else {
                continue;
            };
            let k = input_shape[dim_norm];
            if k < size {
                continue;
            }

            let u = (k - size) / step + 1;
            if self.use_slice {
                let name = graph.nodes[id].name.clone();
                let orig_meta = graph.nodes[id].meta.clone();
                let input_meta = graph.nodes[x_id].meta.clone();
                let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&input_meta);
                let perm: Vec<i64> = (0..dim_norm)
                    .chain(dim_norm + 1..input_shape.len())
                    .chain(std::iter::once(dim_norm))
                    .map(|dim| dim as i64)
                    .collect();
                let perm_shape: Vec<_> =
                    perm.iter().map(|&dim| input_shape[dim as usize]).collect();
                let mut window_shape = perm_shape.clone();
                window_shape.insert(dim_norm, 1);
                let mut cursor = graph.rewrite_cursor(id);
                // Share layout moves, as rc4's later propagation passes do.
                let transposed = if dim_norm + 1 == input_shape.len() {
                    x_id
                } else {
                    cursor.permute_copy(
                        format!("{name}_permute"),
                        x_id,
                        &perm,
                        meta.preserve_qparams(perm_shape),
                    )
                };
                let prepared = cursor.call_function(
                    format!("{name}_windows"),
                    "aten.unsqueeze_copy.default",
                    vec![IrArg::Node(transposed), IrArg::Int(dim_norm as i64)],
                    meta.preserve_qparams(window_shape.clone()),
                );
                window_shape[input_shape.len()] = size;
                let mut windows = Vec::new();
                for window in 0..u {
                    let start = window * step;
                    let sliced = cursor.slice_copy(
                        format!("{name}_slice{window}"),
                        prepared,
                        input_shape.len() as i64,
                        start,
                        start + size,
                        None,
                        meta.preserve_qparams(window_shape.clone()),
                    );
                    windows.push(IrArg::Node(sliced));
                }
                let result = if windows.len() == 1 {
                    windows[0].as_node().unwrap()
                } else {
                    window_shape[dim_norm] = u;
                    cursor.call_function(
                        format!("{name}_cat"),
                        "aten.cat.default",
                        vec![IrArg::List(windows), IrArg::Int(dim_norm as i64)],
                        meta.preserve_qparams(window_shape),
                    )
                };
                cursor.replace_anchor_with(result);
                modified = true;
                continue;
            }

            let uc = u * size;
            let pre: Vec<i64> = input_shape[..dim_norm].to_vec();
            let post: Vec<i64> = input_shape[dim_norm + 1..].to_vec();
            let p = shape_product(&pre);
            let q = shape_product(&post);

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let (work_meta_seed, needs_bool_cast) = gather_work_stamp_or_float32(&x_meta);
            let index_meta = int32_index_stamp(&orig_meta);
            let bool_meta = bool_stamp(&orig_meta);
            let mut values_in = x_id;
            let work_dtype = work_meta_seed.dtype().to_string();
            let work_meta = work_meta_seed;
            let mut cursor = graph.rewrite_cursor(id);

            if needs_bool_cast {
                let cast_meta = work_meta.drop_qparams(input_shape.clone());
                let cast_id = cursor.dtype_cast(
                    format!("{}_bool_to_int8", name),
                    x_id,
                    &work_dtype,
                    cast_meta,
                );
                values_in = cast_id;
            }

            // Flatten x to (P, K, Q)
            let values_meta = if needs_bool_cast {
                work_meta.drop_qparams(vec![p, k, q])
            } else {
                work_meta.preserve_qparams(vec![p, k, q])
            };
            let values =
                cursor.view_copy(format!("{}_flat", name), values_in, &[p, k, q], values_meta);

            // Build sliding-window index: arange(0,U)*step + arange(0,size) → (U*size,)
            let idx_u_meta = index_meta.drop_qparams(vec![u]);
            let idx_u = cursor.arange(
                format!("{}_au", name),
                0,
                u,
                1,
                TOSA_INDEX_DTYPE,
                idx_u_meta,
            );

            let step_meta = index_meta.drop_qparams(vec![1]);
            let step_t = cursor.full_int(
                format!("{}_step", name),
                &[1],
                step,
                TOSA_INDEX_DTYPE,
                step_meta,
            );

            let idx_scaled_meta = index_meta.drop_qparams(vec![u]);
            let idx_scaled = cursor.call_function(
                format!("{}_us", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(idx_u), IrArg::Node(step_t)],
                idx_scaled_meta,
            );

            let idx_u2d_meta = index_meta.drop_qparams(vec![u, 1]);
            let idx_u2d = cursor.call_function(
                format!("{}_u2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_scaled), IrArg::Int(1)],
                idx_u2d_meta,
            );

            let idx_c_meta = index_meta.drop_qparams(vec![size]);
            let idx_c = cursor.arange(
                format!("{}_ac", name),
                0,
                size,
                1,
                TOSA_INDEX_DTYPE,
                idx_c_meta,
            );

            let idx_c2d_meta = index_meta.drop_qparams(vec![1, size]);
            let idx_c2d = cursor.call_function(
                format!("{}_c2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_c), IrArg::Int(0)],
                idx_c2d_meta,
            );

            let idx2d_meta = index_meta.drop_qparams(vec![u, size]);
            let idx2d = cursor.call_function(
                format!("{}_i2d", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(idx_u2d), IrArg::Node(idx_c2d)],
                idx2d_meta,
            );

            let idx1d_meta = index_meta.drop_qparams(vec![uc]);
            let idx1d = cursor.view_copy(format!("{}_i1d", name), idx2d, &[uc], idx1d_meta);

            let idx_unsq_meta = index_meta.drop_qparams(vec![1, uc]);
            let idx_unsq = cursor.call_function(
                format!("{}_iu", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx1d), IrArg::Int(0)],
                idx_unsq_meta,
            );

            let idx_exp_meta = index_meta.drop_qparams(vec![p, uc]);
            let idx_exp = cursor.call_function(
                format!("{}_ie", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[p, uc])],
                idx_exp_meta,
            );

            // GATHER
            let gathered_meta = if needs_bool_cast {
                work_meta.drop_qparams(vec![p, uc, q])
            } else {
                work_meta.preserve_qparams(vec![p, uc, q])
            };
            let gathered = cursor.call_function(
                format!("{}_g", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                gathered_meta,
            );

            // Reshape (P, U*size, Q) → (P, U, size, Q)
            let r4d_meta = if needs_bool_cast {
                work_meta.drop_qparams(vec![p, u, size, q])
            } else {
                work_meta.preserve_qparams(vec![p, u, size, q])
            };
            let r4d =
                cursor.view_copy(format!("{}_r4", name), gathered, &[p, u, size, q], r4d_meta);

            // Reshape to (...pre, U, size, ...post)
            let mut full_shape: Vec<i64> = pre.clone();
            full_shape.push(u);
            full_shape.push(size);
            full_shape.extend(post.iter().copied());
            let reshaped_meta = if needs_bool_cast {
                work_meta.drop_qparams(full_shape.clone())
            } else {
                work_meta.preserve_qparams(full_shape.clone())
            };
            let reshaped =
                cursor.view_copy(format!("{}_rs", name), r4d, &full_shape, reshaped_meta);

            // Permute: move size dim to end
            let pre_len = pre.len() as i64;
            let post_len = post.len() as i64;
            let rank_cur = pre_len + 2 + post_len;
            let c_idx = pre_len + 1;
            let perm: Vec<i64> = (0..pre_len + 1)
                .chain(c_idx + 1..rank_cur)
                .chain(std::iter::once(c_idx))
                .collect();
            let mut perm_shape = Vec::new();
            for &p in &perm {
                perm_shape.push(full_shape[p as usize]);
            }
            let cast_back_shape = perm_shape.clone();
            let permuted_meta = if needs_bool_cast {
                work_meta.drop_qparams(perm_shape)
            } else {
                work_meta.preserve_qparams(perm_shape)
            };
            let permuted =
                cursor.permute_copy(format!("{}_pm", name), reshaped, &perm, permuted_meta);

            if needs_bool_cast {
                let cast_back_meta = bool_meta.drop_qparams(cast_back_shape);
                let cast_back = cursor.dtype_cast(
                    format!("{}_int8_to_bool", name),
                    permuted,
                    "torch.bool",
                    cast_back_meta,
                );
                cursor.replace_anchor_with(cast_back);
            } else {
                cursor.replace_anchor_with(permuted);
            }
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeUnfold] decomposed unfold_copy ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, placeholder, qparam};

    #[test]
    fn slice_windows_keep_unfold_layout_for_first_middle_and_negative_dimensions() {
        for (shape, dim, size, step, expected_shape) in [
            (vec![5], 0, 3, 2, vec![2, 3]),
            (vec![2, 5, 3], 1, 2, 2, vec![2, 2, 3, 2]),
            (vec![2, 3, 5], -1, 3, 4, vec![2, 3, 1, 3]),
            (vec![17], 0, 2, 1, vec![16, 2]),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", shape.clone(), "torch.int8");
            let unfolded = call(
                &mut graph,
                "unfold",
                "aten.unfold_copy.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Int(dim),
                    IrArg::Int(size),
                    IrArg::Int(step),
                ],
                expected_shape.clone(),
                "torch.int8",
            );
            graph.nodes[unfolded]
                .meta
                .set_both_qparams(qparam(0.25, 0, "torch.int8"));
            assert!(IrDecomposeUnfoldPass { use_slice: true }.run(&mut graph));
            let result = &graph.nodes[graph.replacement_of(unfolded).unwrap()];
            let dim_norm = wrap_dim(dim, shape.len()).unwrap();
            let windows = expected_shape[dim_norm];
            assert_eq!(result.meta.val_shape(), Some(expected_shape.as_slice()));
            assert_eq!(result.meta.val_dtype(), Some("torch.int8"));
            assert!(result.meta.has_nonempty_qparam_metadata());
            assert_eq!(
                result.target,
                if windows == 1 {
                    "aten.slice_copy.Tensor"
                } else {
                    "aten.cat.default"
                }
            );
            let slices: Vec<_> = graph
                .live_nodes()
                .filter(|node| node.target == "aten.slice_copy.Tensor")
                .collect();
            assert_eq!(slices.len() as i64, windows);
            let prepared = graph
                .live_nodes()
                .find(|node| node.name == "unfold_windows")
                .unwrap();
            assert_eq!(
                graph
                    .live_nodes()
                    .filter(|node| node.target == "aten.permute_copy.default")
                    .count(),
                usize::from(dim_norm + 1 != shape.len())
            );
            for (window, slice) in slices.iter().enumerate() {
                assert_eq!(slice.first_node_arg(), Some(prepared.id));
                assert_eq!(slice.arg_int(1), Some(shape.len() as i64));
                assert_eq!(slice.arg_int(2), Some(window as i64 * step));
                assert_eq!(slice.arg_int(3), Some(window as i64 * step + size));
            }
            assert!(!graph
                .live_nodes()
                .any(|node| node.target == "backend.tosa.GATHER.default"
                    || node.target == "aten._to_copy.default"));
            assert!(!IrDecomposeUnfoldPass { use_slice: true }.run(&mut graph));
        }
    }

    #[test]
    fn generic_unfold_keeps_gather_and_bool_work_casts() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 5, 3], "torch.bool");
        let unfolded = call(
            &mut graph,
            "unfold",
            "aten.unfold_copy.default",
            vec![IrArg::Node(x), IrArg::Int(1), IrArg::Int(2), IrArg::Int(2)],
            [2, 2, 3, 2],
            "torch.bool",
        );
        assert!(IrDecomposeUnfoldPass { use_slice: false }.run(&mut graph));
        assert!(graph
            .live_nodes()
            .any(|node| node.target == "backend.tosa.GATHER.default"));
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten._to_copy.default")
                .count(),
            2
        );
        assert!(!graph
            .live_nodes()
            .any(|node| node.target == "aten.slice_copy.Tensor"));
        let result = &graph.nodes[graph.replacement_of(unfolded).unwrap()];
        assert_eq!(result.meta.val_shape(), Some([2, 2, 3, 2].as_slice()));
        assert_eq!(result.meta.val_dtype(), Some("torch.bool"));
    }
}
