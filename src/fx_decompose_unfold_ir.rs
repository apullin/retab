//! Pure Rust IrPass: unfold_copy → GATHER-based sliding window.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_neutral(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta { shape: shape.clone(), dtype: dtype.to_string() });
    m.set_extra("arm_layout_neutral", MetaValue::Bool(true));
    m.set_extra("tosa_spatial_rank", MetaValue::Int(0));
    let order: Vec<usize> = (0..shape.len()).collect();
    m.set_tosa_dim_order(order);
    m
}

fn mark_layout_neutral_in_place(graph: &mut IrGraph, id: NodeId) {
    graph.node_mut(id).meta.set_extra("arm_layout_neutral", MetaValue::Bool(true));
    graph.node_mut(id).meta.set_extra("tosa_spatial_rank", MetaValue::Int(0));
    if let Some(shape) = graph.nodes[id].meta.val_shape() {
        let order: Vec<usize> = (0..shape.len()).collect();
        graph.node_mut(id).meta.set_tosa_dim_order(order);
    }
}

fn dtype_kwarg(dtype_str: &str) -> HashMap<String, IrArg> {
    let mut kw = HashMap::new();
    kw.insert("dtype".into(), IrArg::Dtype(dtype_str.into()));
    kw
}

fn copy_qparams_from_node(graph: &IrGraph, src_id: NodeId, dst_meta: &mut NodeMeta) {
    if let Some(iqp) = graph.nodes[src_id].meta.input_qparams.clone() {
        dst_meta.set_input_qparams(iqp);
    }
    if let Some(oqp) = graph.nodes[src_id].meta.output_qparams.clone() {
        dst_meta.set_output_qparams(oqp);
    }
}

fn clear_qparams(dst_meta: &mut NodeMeta) {
    dst_meta.set_input_qparams(HashMap::new());
    dst_meta.set_output_qparams(HashMap::new());
}

fn add_full_int(
    graph: &mut IrGraph,
    name: String,
    shape: &[i64],
    value: i64,
    dtype: &str,
    meta: NodeMeta,
) -> NodeId {
    let mut kwargs = HashMap::new();
    kwargs.insert("dtype".to_string(), IrArg::Dtype(dtype.to_string()));
    graph.add_node_with_kwargs(
        NodeOp::CallFunction,
        name,
        "aten.full.default".to_string(),
        vec![i64_list(shape), IrArg::Int(value)],
        kwargs,
        meta,
    )
}

fn product(s: &[i64]) -> i64 { if s.is_empty() { 1 } else { s.iter().copied().product() } }

pub struct IrDecomposeUnfoldPass;

impl IrPass for IrDecomposeUnfoldPass {
    fn name(&self) -> &str { "IrDecomposeUnfold" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("unfold_copy") { continue; }
            if graph.nodes[id].args.len() != 4 { continue; }

            let x_id = match graph.nodes[id].args[0].as_node() { Some(n) => n, None => continue };
            let dim = match graph.nodes[id].args[1].as_int() { Some(v) => v, None => continue };
            let size = match graph.nodes[id].args[2].as_int() { Some(v) => v, None => continue };
            let step = match graph.nodes[id].args[3].as_int() { Some(v) => v, None => continue };
            if size <= 0 || step <= 0 { continue; }

            let input_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(), None => continue,
            };
            let rank = input_shape.len() as i64;
            if rank < 1 { continue; }

            let dim_norm = dim.rem_euclid(rank) as usize;
            let k = input_shape[dim_norm];
            if k < size { continue; }

            let u = (k - size) / step + 1;
            let uc = u * size;
            let pre: Vec<i64> = input_shape[..dim_norm].to_vec();
            let post: Vec<i64> = input_shape[dim_norm + 1..].to_vec();
            let p = product(&pre);
            let q = product(&post);

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let original_users = graph.users(id).to_vec();
            let x_dtype = graph.nodes[x_id].meta.val_dtype().unwrap_or("torch.float32").to_string();
            let needs_bool_cast = x_dtype == "torch.bool";
            let work_dtype = if needs_bool_cast { "torch.int8" } else { x_dtype.as_str() };
            let mut values_in = x_id;

            // Mark input as layout-neutral
            mark_layout_neutral_in_place(graph, x_id);

            if needs_bool_cast {
                let mut cast_meta = meta_neutral(&graph.nodes[x_id].meta, input_shape.clone(), work_dtype);
                clear_qparams(&mut cast_meta);
                let cast_id = graph.add_node_with_kwargs(
                    NodeOp::CallFunction,
                    format!("{}_bool_to_int8", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(x_id)],
                    dtype_kwarg(work_dtype),
                    cast_meta,
                );
                graph.anchor_to(cast_id, id);
                values_in = cast_id;
            }

            // Flatten x to (P, K, Q)
            let mut values_meta = meta_neutral(&graph.nodes[values_in].meta, vec![p, k, q], work_dtype);
            if !needs_bool_cast {
                copy_qparams_from_node(graph, x_id, &mut values_meta);
            } else {
                clear_qparams(&mut values_meta);
            }
            let values = graph.add_node(NodeOp::CallFunction, format!("{}_flat", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(values_in), i64_list(&[p, k, q])],
                values_meta);
            graph.anchor_to(values, id);

            // Build sliding-window index: arange(0,U)*step + arange(0,size) → (U*size,)
            let mut idx_u_meta = meta_neutral(&orig_meta, vec![u], "torch.int32");
            clear_qparams(&mut idx_u_meta);
            let idx_u = graph.add_node_with_kwargs(NodeOp::CallFunction, format!("{}_au", name),
                "aten.arange.start_step".to_string(),
                vec![IrArg::Int(0), IrArg::Int(u)],
                dtype_kwarg("torch.int32"),
                idx_u_meta);
            graph.anchor_to(idx_u, id);

            let mut step_meta = meta_neutral(&orig_meta, vec![1], "torch.int32");
            clear_qparams(&mut step_meta);
            let step_t = add_full_int(
                graph,
                format!("{}_step", name),
                &[1],
                step,
                "torch.int32",
                step_meta,
            );
            graph.anchor_to(step_t, id);

            let mut idx_scaled_meta = meta_neutral(&orig_meta, vec![u], "torch.int32");
            clear_qparams(&mut idx_scaled_meta);
            let idx_scaled = graph.add_node(NodeOp::CallFunction, format!("{}_us", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(idx_u), IrArg::Node(step_t)],
                idx_scaled_meta);
            graph.anchor_to(idx_scaled, id);

            let mut idx_u2d_meta = meta_neutral(&orig_meta, vec![u, 1], "torch.int32");
            clear_qparams(&mut idx_u2d_meta);
            let idx_u2d = graph.add_node(NodeOp::CallFunction, format!("{}_u2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_scaled), IrArg::Int(1)],
                idx_u2d_meta);
            graph.anchor_to(idx_u2d, id);

            let mut idx_c_meta = meta_neutral(&orig_meta, vec![size], "torch.int32");
            clear_qparams(&mut idx_c_meta);
            let idx_c = graph.add_node_with_kwargs(NodeOp::CallFunction, format!("{}_ac", name),
                "aten.arange.start_step".to_string(),
                vec![IrArg::Int(0), IrArg::Int(size)],
                dtype_kwarg("torch.int32"),
                idx_c_meta);
            graph.anchor_to(idx_c, id);

            let mut idx_c2d_meta = meta_neutral(&orig_meta, vec![1, size], "torch.int32");
            clear_qparams(&mut idx_c2d_meta);
            let idx_c2d = graph.add_node(NodeOp::CallFunction, format!("{}_c2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_c), IrArg::Int(0)],
                idx_c2d_meta);
            graph.anchor_to(idx_c2d, id);

            let mut idx2d_meta = meta_neutral(&orig_meta, vec![u, size], "torch.int32");
            clear_qparams(&mut idx2d_meta);
            let idx2d = graph.add_node(NodeOp::CallFunction, format!("{}_i2d", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(idx_u2d), IrArg::Node(idx_c2d)],
                idx2d_meta);
            graph.anchor_to(idx2d, id);

            let mut idx1d_meta = meta_neutral(&orig_meta, vec![uc], "torch.int32");
            clear_qparams(&mut idx1d_meta);
            let idx1d = graph.add_node(NodeOp::CallFunction, format!("{}_i1d", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(idx2d), i64_list(&[uc])],
                idx1d_meta);
            graph.anchor_to(idx1d, id);

            let mut idx_unsq_meta = meta_neutral(&orig_meta, vec![1, uc], "torch.int32");
            clear_qparams(&mut idx_unsq_meta);
            let idx_unsq = graph.add_node(NodeOp::CallFunction, format!("{}_iu", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx1d), IrArg::Int(0)],
                idx_unsq_meta);
            graph.anchor_to(idx_unsq, id);

            let mut idx_exp_meta = meta_neutral(&orig_meta, vec![p, uc], "torch.int32");
            clear_qparams(&mut idx_exp_meta);
            let idx_exp = graph.add_node(NodeOp::CallFunction, format!("{}_ie", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[p, uc])],
                idx_exp_meta);
            graph.anchor_to(idx_exp, id);

            // GATHER
            let mut gathered_meta = meta_neutral(&graph.nodes[values_in].meta, vec![p, uc, q], work_dtype);
            if !needs_bool_cast {
                copy_qparams_from_node(graph, x_id, &mut gathered_meta);
            } else {
                clear_qparams(&mut gathered_meta);
            }
            let gathered = graph.add_node(NodeOp::CallFunction, format!("{}_g", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                gathered_meta);
            graph.anchor_to(gathered, id);

            // Reshape (P, U*size, Q) → (P, U, size, Q)
            let mut r4d_meta = meta_neutral(&graph.nodes[values_in].meta, vec![p, u, size, q], work_dtype);
            if !needs_bool_cast {
                copy_qparams_from_node(graph, x_id, &mut r4d_meta);
            } else {
                clear_qparams(&mut r4d_meta);
            }
            let r4d = graph.add_node(NodeOp::CallFunction, format!("{}_r4", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(gathered), i64_list(&[p, u, size, q])],
                r4d_meta);
            graph.anchor_to(r4d, id);

            // Reshape to (...pre, U, size, ...post)
            let mut full_shape: Vec<i64> = pre.clone();
            full_shape.push(u);
            full_shape.push(size);
            full_shape.extend(post.iter().copied());
            let mut reshaped_meta = meta_neutral(&graph.nodes[values_in].meta, full_shape.clone(), work_dtype);
            if !needs_bool_cast {
                copy_qparams_from_node(graph, x_id, &mut reshaped_meta);
            } else {
                clear_qparams(&mut reshaped_meta);
            }
            let reshaped = graph.add_node(NodeOp::CallFunction, format!("{}_rs", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(r4d), i64_list(&full_shape)],
                reshaped_meta);
            graph.anchor_to(reshaped, id);

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
            for &p in &perm { perm_shape.push(full_shape[p as usize]); }
            let mut permuted_meta = meta_neutral(&graph.nodes[values_in].meta, perm_shape, work_dtype);
            if !needs_bool_cast {
                copy_qparams_from_node(graph, x_id, &mut permuted_meta);
            } else {
                clear_qparams(&mut permuted_meta);
            }
            let permuted = graph.add_node(NodeOp::CallFunction, format!("{}_pm", name),
                "aten.permute_copy.default".to_string(),
                vec![IrArg::Node(reshaped), i64_list(&perm)],
                permuted_meta);
            graph.anchor_to(permuted, id);

            let final_result = if needs_bool_cast {
                let mut cast_back_meta = meta_neutral(&orig_meta, graph.nodes[permuted].meta.val_shape().unwrap_or(&[]).to_vec(), "torch.bool");
                clear_qparams(&mut cast_back_meta);
                let cast_back = graph.add_node_with_kwargs(
                    NodeOp::CallFunction,
                    format!("{}_int8_to_bool", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(permuted)],
                    dtype_kwarg("torch.bool"),
                    cast_back_meta,
                );
                graph.anchor_to(cast_back, id);
                cast_back
            } else {
                permuted
            };

            for user_id in original_users {
                if graph.nodes[user_id].alive {
                    mark_layout_neutral_in_place(graph, user_id);
                }
            }
            graph.replace_and_erase(id, final_result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeUnfold] decomposed unfold_copy ops"));
        }
        modified
    }
}
