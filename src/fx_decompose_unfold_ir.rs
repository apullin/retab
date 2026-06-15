//! Pure Rust IrPass: unfold_copy → GATHER-based sliding window.

use crate::fx_ir::{i64_list_arg as i64_list, shape_product, IrArg, IrGraph, IrPass};
use crate::fx_target::is_unfold_copy_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeUnfoldPass;

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
            let rank = input_shape.len() as i64;
            if rank < 1 {
                continue;
            }

            let dim_norm = dim.rem_euclid(rank) as usize;
            let k = input_shape[dim_norm];
            if k < size {
                continue;
            }

            let u = (k - size) / step + 1;
            let uc = u * size;
            let pre: Vec<i64> = input_shape[..dim_norm].to_vec();
            let post: Vec<i64> = input_shape[dim_norm + 1..].to_vec();
            let p = shape_product(&pre);
            let q = shape_product(&post);

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let (work_meta_seed, needs_bool_cast) =
                x_meta.tensor_stamp_bool_as_int8_work_or_float32();
            let index_meta = orig_meta.tensor_stamp("torch.int32");
            let bool_meta = orig_meta.tensor_stamp("torch.bool");
            let mut values_in = x_id;

            if needs_bool_cast {
                let cast_meta = work_meta_seed.drop_qparams(input_shape.clone());
                let cast_id = graph.add_anchored_dtype_cast(
                    id,
                    format!("{}_bool_to_int8", name),
                    x_id,
                    work_meta_seed.dtype(),
                    cast_meta,
                );
                values_in = cast_id;
            }

            let values_base_meta = graph.nodes[values_in].meta.clone();
            let work_meta = values_base_meta.tensor_stamp(work_meta_seed.dtype());

            // Flatten x to (P, K, Q)
            let values_meta = if needs_bool_cast {
                work_meta.drop_qparams(vec![p, k, q])
            } else {
                work_meta.preserve_qparams(vec![p, k, q])
            };
            let values = graph.add_anchored_view_copy(
                id,
                format!("{}_flat", name),
                values_in,
                &[p, k, q],
                values_meta,
            );

            // Build sliding-window index: arange(0,U)*step + arange(0,size) → (U*size,)
            let idx_u_meta = index_meta.drop_qparams(vec![u]);
            let idx_u = graph.add_anchored_arange(
                id,
                format!("{}_au", name),
                0,
                u,
                1,
                "torch.int32",
                idx_u_meta,
            );

            let step_meta = index_meta.drop_qparams(vec![1]);
            let step_t = graph.add_anchored_full_int(
                id,
                format!("{}_step", name),
                &[1],
                step,
                "torch.int32",
                step_meta,
            );

            let idx_scaled_meta = index_meta.drop_qparams(vec![u]);
            let idx_scaled = graph.add_anchored_call_function(
                id,
                format!("{}_us", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(idx_u), IrArg::Node(step_t)],
                idx_scaled_meta,
            );

            let idx_u2d_meta = index_meta.drop_qparams(vec![u, 1]);
            let idx_u2d = graph.add_anchored_call_function(
                id,
                format!("{}_u2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_scaled), IrArg::Int(1)],
                idx_u2d_meta,
            );

            let idx_c_meta = index_meta.drop_qparams(vec![size]);
            let idx_c = graph.add_anchored_arange(
                id,
                format!("{}_ac", name),
                0,
                size,
                1,
                "torch.int32",
                idx_c_meta,
            );

            let idx_c2d_meta = index_meta.drop_qparams(vec![1, size]);
            let idx_c2d = graph.add_anchored_call_function(
                id,
                format!("{}_c2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_c), IrArg::Int(0)],
                idx_c2d_meta,
            );

            let idx2d_meta = index_meta.drop_qparams(vec![u, size]);
            let idx2d = graph.add_anchored_call_function(
                id,
                format!("{}_i2d", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(idx_u2d), IrArg::Node(idx_c2d)],
                idx2d_meta,
            );

            let idx1d_meta = index_meta.drop_qparams(vec![uc]);
            let idx1d =
                graph.add_anchored_view_copy(id, format!("{}_i1d", name), idx2d, &[uc], idx1d_meta);

            let idx_unsq_meta = index_meta.drop_qparams(vec![1, uc]);
            let idx_unsq = graph.add_anchored_call_function(
                id,
                format!("{}_iu", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx1d), IrArg::Int(0)],
                idx_unsq_meta,
            );

            let idx_exp_meta = index_meta.drop_qparams(vec![p, uc]);
            let idx_exp = graph.add_anchored_call_function(
                id,
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
            let gathered = graph.add_anchored_call_function(
                id,
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
            let r4d = graph.add_anchored_view_copy(
                id,
                format!("{}_r4", name),
                gathered,
                &[p, u, size, q],
                r4d_meta,
            );

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
            let reshaped = graph.add_anchored_view_copy(
                id,
                format!("{}_rs", name),
                r4d,
                &full_shape,
                reshaped_meta,
            );

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
            let permuted_meta = if needs_bool_cast {
                work_meta.drop_qparams(perm_shape)
            } else {
                work_meta.preserve_qparams(perm_shape)
            };
            let permuted = graph.add_anchored_permute_copy(
                id,
                format!("{}_pm", name),
                reshaped,
                &perm,
                permuted_meta,
            );

            if needs_bool_cast {
                let cast_back_shape = graph.nodes[permuted]
                    .meta
                    .val_shape()
                    .unwrap_or(&[])
                    .to_vec();
                let cast_back_meta = bool_meta.drop_qparams(cast_back_shape);
                graph.replace_with_dtype_cast(
                    id,
                    format!("{}_int8_to_bool", name),
                    permuted,
                    "torch.bool",
                    cast_back_meta,
                );
            } else {
                graph.replace_and_erase(id, permuted);
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
