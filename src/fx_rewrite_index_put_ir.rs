//! Pure Rust IrPass: index_put → view + scatter + view for TOSA.

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

fn product(s: &[i64]) -> i64 { if s.is_empty() { 1 } else { s.iter().copied().product() } }

fn shape_to_stride(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for i in (0..shape.len()).rev() { strides[i] = stride; stride *= shape[i]; }
    strides
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

pub struct IrRewriteIndexPutPass;

impl IrPass for IrRewriteIndexPutPass {
    fn name(&self) -> &str { "IrRewriteIndexPut" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("index_put.default") { continue; }
            if graph.nodes[id].args.len() < 3 { continue; }

            let source_id = match graph.nodes[id].args[0].as_node() { Some(n) => n, None => continue };
            let indices_list = match &graph.nodes[id].args[1] {
                IrArg::List(items) => items.clone(),
                _ => continue,
            };
            let values_id = match graph.nodes[id].args[2].as_node() { Some(n) => n, None => continue };
            let accumulate = graph.nodes[id].args.get(3).and_then(|a| a.as_bool()).unwrap_or(false);

            let source_shape = match graph.nodes[source_id].meta.val_shape() {
                Some(s) => s.to_vec(), None => continue,
            };
            if source_shape.is_empty() { continue; }

            // Collect index node IDs and shapes
            let mut idx_ids: Vec<NodeId> = Vec::new();
            let mut idx_shapes: Vec<Vec<i64>> = Vec::new();
            let mut has_none = false;
            for item in &indices_list {
                match item {
                    IrArg::Node(nid) => {
                        let Some(s) = graph.nodes[*nid].meta.val_shape() else { continue };
                        idx_ids.push(*nid);
                        idx_shapes.push(s.to_vec());
                    }
                    IrArg::None => {
                        has_none = true;
                        // Need arange for None indices — skip for now
                        break;
                    }
                    _ => break,
                }
            }
            if has_none || idx_ids.is_empty() { continue; }

            let Some(bcast) = broadcast_shape(&idx_shapes) else { continue };
            let num_idx = idx_ids.len();
            let w = product(&bcast).max(1);
            let mut k = 1i64;
            for i in 0..num_idx { k *= source_shape.get(i).copied().unwrap_or(1); }
            let total = product(&source_shape);
            let c = if k != 0 { total / k } else { 1 };
            if !(k > 0 && w > 0 && c > 0) { continue; }

            let strides = shape_to_stride(&source_shape);
            let lin_scales: Vec<i64> = strides.iter().take(num_idx).map(|s| s / c).collect();

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let src_dtype = graph.nodes[source_id].meta.val_dtype()
                .unwrap_or("torch.float32").to_string();
            let idx_dtype = graph.nodes[idx_ids[0]].meta.val_dtype()
                .unwrap_or("torch.int64").to_string();

            // Build flat index
            let mut lin: Option<NodeId> = None;
            for (i, (&idx_id, lin_scale)) in idx_ids.iter().zip(lin_scales.iter()).enumerate() {
                let idx_shape = &idx_shapes[i];
                let mut current = idx_id;

                if idx_shape.len() != bcast.len() {
                    let mut aligned: Vec<i64> = vec![1; bcast.len() - idx_shape.len()];
                    aligned.extend(idx_shape);
                    let mut aligned_meta = meta_with(&orig_meta, aligned.clone(), &idx_dtype);
                    clear_qparams(&mut aligned_meta);
                    let a = graph.add_node(NodeOp::CallFunction, format!("{}_ia{}", name, i),
                        "aten.view_copy.default".to_string(),
                        vec![IrArg::Node(current), i64_list(&aligned)],
                        aligned_meta);
                    graph.anchor_to(a, id);
                    current = a;
                }

                let mut bcast_meta = meta_with(&orig_meta, bcast.clone(), &idx_dtype);
                clear_qparams(&mut bcast_meta);
                let b = graph.add_node(NodeOp::CallFunction, format!("{}_ib{}", name, i),
                    "aten.expand_copy.default".to_string(),
                    vec![IrArg::Node(current), i64_list(&bcast)],
                    bcast_meta);
                graph.anchor_to(b, id);

                let mut flat_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                clear_qparams(&mut flat_meta);
                let flat = graph.add_node(NodeOp::CallFunction, format!("{}_if{}", name, i),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(b), i64_list(&[w])],
                    flat_meta);
                graph.anchor_to(flat, id);

                let mut scale_meta = meta_with(&orig_meta, vec![1], &idx_dtype);
                clear_qparams(&mut scale_meta);
                let scale = add_full_int(
                    graph,
                    format!("{}_sc{}", name, i),
                    &[1],
                    *lin_scale,
                    &idx_dtype,
                    scale_meta,
                );
                graph.anchor_to(scale, id);

                let mut scaled_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                clear_qparams(&mut scaled_meta);
                let scaled = graph.add_node(NodeOp::CallFunction, format!("{}_sm{}", name, i),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(flat), IrArg::Node(scale)],
                    scaled_meta);
                graph.anchor_to(scaled, id);

                lin = Some(match lin {
                    None => scaled,
                    Some(prev) => {
                        let mut add_meta = meta_with(&orig_meta, vec![w], &idx_dtype);
                        clear_qparams(&mut add_meta);
                        let add = graph.add_node(NodeOp::CallFunction, format!("{}_la{}", name, i),
                            "aten.add.Tensor".to_string(),
                            vec![IrArg::Node(prev), IrArg::Node(scaled)],
                            add_meta);
                        graph.anchor_to(add, id);
                        add
                    }
                });
            }
            let Some(flat_idx) = lin else { continue };

            // Reshape: indices (1, W), source (1, K, C), values (1, W, C)
            let mut idx_meta = meta_with(&orig_meta, vec![1, w], &idx_dtype);
            clear_qparams(&mut idx_meta);
            let idx_1w = graph.add_node(NodeOp::CallFunction, format!("{}_idx", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(flat_idx), i64_list(&[1, w])],
                idx_meta);
            graph.anchor_to(idx_1w, id);

            let mut src_meta = meta_with(&graph.nodes[source_id].meta, vec![1, k, c], &src_dtype);
            copy_qparams_from_node(graph, source_id, &mut src_meta);
            let src_1kc = graph.add_node(NodeOp::CallFunction, format!("{}_src", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(source_id), i64_list(&[1, k, c])],
                src_meta);
            graph.anchor_to(src_1kc, id);

            let mut val_meta = meta_with(&graph.nodes[source_id].meta, vec![1, w, c], &src_dtype);
            copy_qparams_from_node(graph, source_id, &mut val_meta);
            let val_1wc = graph.add_node(NodeOp::CallFunction, format!("{}_val", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(values_id), i64_list(&[1, w, c])],
                val_meta);
            graph.anchor_to(val_1wc, id);

            let scatter_values = if accumulate {
                let mut gathered_meta = meta_with(&graph.nodes[source_id].meta, vec![1, w, c], &src_dtype);
                copy_qparams_from_node(graph, source_id, &mut gathered_meta);
                let gathered = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_gather_prev", name),
                    "backend.tosa.GATHER.default".to_string(),
                    vec![IrArg::Node(src_1kc), IrArg::Node(idx_1w)],
                    gathered_meta,
                );
                graph.anchor_to(gathered, id);

                let mut added_meta = meta_with(&graph.nodes[source_id].meta, vec![1, w, c], &src_dtype);
                copy_qparams_from_node(graph, source_id, &mut added_meta);
                let added = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_accum", name),
                    "aten.add.Tensor".to_string(),
                    vec![IrArg::Node(gathered), IrArg::Node(val_1wc)],
                    added_meta,
                );
                graph.anchor_to(added, id);
                added
            } else {
                val_1wc
            };

            // SCATTER(source_flat, indices_flat, values_flat)
            let mut scattered_meta = meta_with(&graph.nodes[source_id].meta, vec![1, k, c], &src_dtype);
            copy_qparams_from_node(graph, source_id, &mut scattered_meta);
            let scattered = graph.add_node(NodeOp::CallFunction, format!("{}_scatter", name),
                "backend.tosa.SCATTER.default".to_string(),
                vec![IrArg::Node(src_1kc), IrArg::Node(idx_1w), IrArg::Node(scatter_values)],
                scattered_meta);
            graph.anchor_to(scattered, id);

            // Reshape back
            let mut result_meta = meta_with(&graph.nodes[source_id].meta, source_shape.clone(), &src_dtype);
            copy_qparams_from_node(graph, source_id, &mut result_meta);
            let result = graph.add_node(NodeOp::CallFunction, format!("{}_out", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(scattered), i64_list(&source_shape)],
                result_meta);

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrRewriteIndexPut] rewrote index_put to SCATTER"));
        }
        modified
    }
}
