//! Pure Rust IrPass: aten.index.Tensor → linearized GATHER.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta { shape, dtype: dtype.to_string() });
    m
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

fn broadcast_shape(shapes: &[Vec<i64>]) -> Option<Vec<i64>> {
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(max_rank);
    for axis in 0..max_rank {
        let mut chosen = 1i64;
        for shape in shapes {
            let dim = *shape.iter().rev().nth(axis).unwrap_or(&1);
            if dim != 1 {
                if chosen == 1 { chosen = dim; }
                else if chosen != dim { return None; }
            }
        }
        out.push(chosen);
    }
    out.reverse();
    Some(out)
}

fn shape_to_stride(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= *dim;
    }
    strides
}

pub struct IrDecomposeIndexTensorPass;

impl IrPass for IrDecomposeIndexTensorPass {
    fn name(&self) -> &str { "IrDecomposeIndexTensor" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("aten.index.Tensor") { continue; }
            if graph.nodes[id].args.len() != 2 { continue; }

            let x_id = match graph.nodes[id].args[0].as_node() { Some(n) => n, None => continue };
            let index_list = match &graph.nodes[id].args[1] {
                IrArg::List(items) => items.clone(),
                _ => continue,
            };
            if index_list.is_empty() { continue; }

            let x_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(), None => continue,
            };
            if x_shape.is_empty() { continue; }

            let m = index_list.len();
            if m > x_shape.len() { continue; }

            // Collect index node IDs and shapes
            let mut idx_ids: Vec<NodeId> = Vec::new();
            let mut idx_shapes: Vec<Vec<i64>> = Vec::new();
            let mut all_valid = true;
            for item in &index_list {
                let Some(nid) = item.as_node() else { all_valid = false; break; };
                let Some(s) = graph.nodes[nid].meta.val_shape() else { all_valid = false; break; };
                idx_ids.push(nid);
                idx_shapes.push(s.to_vec());
            }
            if !all_valid { continue; }

            let Some(broadcast) = broadcast_shape(&idx_shapes) else { continue };
            let w = broadcast.iter().copied().product::<i64>().max(1);
            let leading: Vec<i64> = x_shape.iter().take(m).copied().collect();
            let trailing: Vec<i64> = x_shape.iter().skip(m).copied().collect();
            let k = leading.iter().copied().product::<i64>().max(1);
            let c = trailing.iter().copied().product::<i64>().max(1);
            let strides = shape_to_stride(&x_shape);
            let lin_scales: Vec<i64> = strides.iter().take(m).map(|s| s / c).collect();

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_dtype = graph.nodes[x_id].meta.val_dtype()
                .unwrap_or("torch.float32").to_string();
            let idx_dtype = graph.nodes[idx_ids[0]].meta.val_dtype()
                .unwrap_or("torch.int64").to_string();

            // Flatten x to (1, K, C)
            let mut x_1kc_meta = meta_with(&graph.nodes[x_id].meta, vec![1, k, c], &x_dtype);
            copy_qparams_from_node(graph, x_id, &mut x_1kc_meta);
            let x_1kc = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_x_flat", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(x_id), i64_list(&[1, k, c])],
                x_1kc_meta,
            );
            graph.anchor_to(x_1kc, id);

            // Build linearized flat index
            let mut lin_w: Option<NodeId> = None;
            for (i, (&idx_id, (idx_shape, &lin_scale))) in idx_ids.iter()
                .zip(idx_shapes.iter().zip(lin_scales.iter()))
                .enumerate()
            {
                let mut current = idx_id;

                // Align rank if needed
                if idx_shape.len() != broadcast.len() {
                    let mut aligned: Vec<i64> = vec![1; broadcast.len() - idx_shape.len()];
                    aligned.extend(idx_shape);
                    let mut aligned_meta = meta_with(&orig_meta, aligned.clone(), &idx_dtype);
                    clear_qparams(&mut aligned_meta);
                    let aligned_id = graph.add_node(
                        NodeOp::CallFunction,
                        format!("{}_idx{}_align", node_name, i),
                        "aten.view_copy.default".to_string(),
                        vec![IrArg::Node(current), i64_list(&aligned)],
                        aligned_meta,
                    );
                    graph.anchor_to(aligned_id, id);
                    current = aligned_id;
                }

                // Broadcast to common shape
                let mut bcast_meta = meta_with(&orig_meta, broadcast.clone(), &idx_dtype);
                clear_qparams(&mut bcast_meta);
                let bcast_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_idx{}_bcast", node_name, i),
                    "aten.expand_copy.default".to_string(),
                    vec![IrArg::Node(current), i64_list(&broadcast)],
                    bcast_meta,
                );
                graph.anchor_to(bcast_id, id);

                // Flatten to (W,)
                let mut flat_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                clear_qparams(&mut flat_meta);
                let flat_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_idx{}_flat", node_name, i),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(bcast_id), i64_list(&[w])],
                    flat_meta,
                );
                graph.anchor_to(flat_id, id);

                // Scale by linearized stride
                let mut scale_meta = meta_with(&orig_meta, vec![1], &idx_dtype);
                clear_qparams(&mut scale_meta);
                let scale_id = add_full_int(
                    graph,
                    format!("{}_scale{}", node_name, i),
                    &[1],
                    lin_scale,
                    &idx_dtype,
                    scale_meta,
                );
                graph.anchor_to(scale_id, id);

                let mut scaled_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                clear_qparams(&mut scaled_meta);
                let scaled_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_idx{}_scaled", node_name, i),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(flat_id), IrArg::Node(scale_id)],
                    scaled_meta,
                );
                graph.anchor_to(scaled_id, id);

                lin_w = Some(match lin_w {
                    None => scaled_id,
                    Some(prev) => {
                        let mut add_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                        clear_qparams(&mut add_meta);
                        let add_id = graph.add_node(
                            NodeOp::CallFunction,
                            format!("{}_lin_add{}", node_name, i),
                            "aten.add.Tensor".to_string(),
                            vec![IrArg::Node(prev), IrArg::Node(scaled_id)],
                            add_meta,
                        );
                        graph.anchor_to(add_id, id);
                        add_id
                    }
                });
            }

            let Some(lin_w_id) = lin_w else { continue };

            // Unsqueeze to (1, W) for GATHER
            let mut idx_1w_meta = meta_with(&orig_meta, vec![1, w], &idx_dtype);
            clear_qparams(&mut idx_1w_meta);
            let idx_1w = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_idx_1w", node_name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(lin_w_id), IrArg::Int(0)],
                idx_1w_meta,
            );
            graph.anchor_to(idx_1w, id);

            // GATHER(x_1kc, idx_1w) → (1, W, C)
            let mut gathered_meta = meta_with(&graph.nodes[x_id].meta, vec![1, w, c], &x_dtype);
            copy_qparams_from_node(graph, x_id, &mut gathered_meta);
            let gathered = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_gather", node_name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(x_1kc), IrArg::Node(idx_1w)],
                gathered_meta,
            );
            graph.anchor_to(gathered, id);

            // Reshape to (...broadcast, ...trailing)
            let mut out_shape: Vec<i64> = broadcast.clone();
            out_shape.extend(trailing.iter().copied());
            let mut result_meta = meta_with(&graph.nodes[x_id].meta, out_shape.clone(), &x_dtype);
            copy_qparams_from_node(graph, x_id, &mut result_meta);
            let result = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_out", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(gathered), i64_list(&out_shape)],
                result_meta,
            );

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeIndexTensor] decomposed index.Tensor ops"));
        }
        modified
    }
}
