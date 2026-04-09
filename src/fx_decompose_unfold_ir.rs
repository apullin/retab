//! Pure Rust IrPass: unfold_copy → GATHER-based sliding window.

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
            let x_dtype = graph.nodes[x_id].meta.val_dtype().unwrap_or("torch.float32").to_string();

            // Skip bool inputs (need dtype cast kwargs)
            if x_dtype == "torch.bool" { continue; }

            // Mark input as layout-neutral
            graph.node_mut(x_id).meta.set_extra("arm_layout_neutral", MetaValue::Bool(true));
            graph.node_mut(x_id).meta.set_extra("tosa_spatial_rank", MetaValue::Int(0));
            let order: Vec<usize> = (0..input_shape.len()).collect();
            graph.node_mut(x_id).meta.set_tosa_dim_order(order);

            // Flatten x to (P, K, Q)
            let values = graph.add_node(NodeOp::CallFunction, format!("{}_flat", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(x_id), i64_list(&[p, k, q])],
                meta_neutral(&orig_meta, vec![p, k, q], &x_dtype));
            graph.anchor_to(values, id);

            // Build sliding-window index: arange(0,U)*step + arange(0,size) → (U*size,)
            let idx_u = graph.add_node(NodeOp::CallFunction, format!("{}_au", name),
                "aten.arange.start_step".to_string(),
                vec![IrArg::Int(0), IrArg::Int(u)],
                meta_neutral(&orig_meta, vec![u], "torch.int32"));
            graph.anchor_to(idx_u, id);

            let step_t = graph.add_node(NodeOp::CallFunction, format!("{}_step", name),
                "aten.full.default".to_string(),
                vec![i64_list(&[1]), IrArg::Int(step)],
                meta_neutral(&orig_meta, vec![1], "torch.int32"));
            graph.anchor_to(step_t, id);

            let idx_scaled = graph.add_node(NodeOp::CallFunction, format!("{}_us", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(idx_u), IrArg::Node(step_t)],
                meta_neutral(&orig_meta, vec![u], "torch.int32"));
            graph.anchor_to(idx_scaled, id);

            let idx_u2d = graph.add_node(NodeOp::CallFunction, format!("{}_u2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_scaled), IrArg::Int(1)],
                meta_neutral(&orig_meta, vec![u, 1], "torch.int32"));
            graph.anchor_to(idx_u2d, id);

            let idx_c = graph.add_node(NodeOp::CallFunction, format!("{}_ac", name),
                "aten.arange.start_step".to_string(),
                vec![IrArg::Int(0), IrArg::Int(size)],
                meta_neutral(&orig_meta, vec![size], "torch.int32"));
            graph.anchor_to(idx_c, id);

            let idx_c2d = graph.add_node(NodeOp::CallFunction, format!("{}_c2d", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx_c), IrArg::Int(0)],
                meta_neutral(&orig_meta, vec![1, size], "torch.int32"));
            graph.anchor_to(idx_c2d, id);

            let idx2d = graph.add_node(NodeOp::CallFunction, format!("{}_i2d", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(idx_u2d), IrArg::Node(idx_c2d)],
                meta_neutral(&orig_meta, vec![u, size], "torch.int32"));
            graph.anchor_to(idx2d, id);

            let idx1d = graph.add_node(NodeOp::CallFunction, format!("{}_i1d", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(idx2d), i64_list(&[uc])],
                meta_neutral(&orig_meta, vec![uc], "torch.int32"));
            graph.anchor_to(idx1d, id);

            let idx_unsq = graph.add_node(NodeOp::CallFunction, format!("{}_iu", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(idx1d), IrArg::Int(0)],
                meta_neutral(&orig_meta, vec![1, uc], "torch.int32"));
            graph.anchor_to(idx_unsq, id);

            let idx_exp = graph.add_node(NodeOp::CallFunction, format!("{}_ie", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[p, uc])],
                meta_neutral(&orig_meta, vec![p, uc], "torch.int32"));
            graph.anchor_to(idx_exp, id);

            // GATHER
            let gathered = graph.add_node(NodeOp::CallFunction, format!("{}_g", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                meta_neutral(&orig_meta, vec![p, uc, q], &x_dtype));
            graph.anchor_to(gathered, id);

            // Reshape (P, U*size, Q) → (P, U, size, Q)
            let r4d = graph.add_node(NodeOp::CallFunction, format!("{}_r4", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(gathered), i64_list(&[p, u, size, q])],
                meta_neutral(&orig_meta, vec![p, u, size, q], &x_dtype));
            graph.anchor_to(r4d, id);

            // Reshape to (...pre, U, size, ...post)
            let mut full_shape: Vec<i64> = pre.clone();
            full_shape.push(u);
            full_shape.push(size);
            full_shape.extend(post.iter().copied());
            let reshaped = graph.add_node(NodeOp::CallFunction, format!("{}_rs", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(r4d), i64_list(&full_shape)],
                meta_neutral(&orig_meta, full_shape.clone(), &x_dtype));
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
            let permuted = graph.add_node(NodeOp::CallFunction, format!("{}_pm", name),
                "aten.permute_copy.default".to_string(),
                vec![IrArg::Node(reshaped), i64_list(&perm)],
                meta_neutral(&orig_meta, perm_shape, &x_dtype));

            graph.replace_and_erase(id, permuted);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeUnfold] decomposed unfold_copy ops"));
        }
        modified
    }
}
