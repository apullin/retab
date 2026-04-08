//! Pure Rust IrPass: strided slice_copy (step > 1) → unit slice + pad + view + slice + view.

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_with_shape(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta { shape, dtype: dtype.to_string() });
    m
}

fn fixup_start(start: Option<i64>, dim_size: i64) -> i64 {
    let mut s = start.unwrap_or(0);
    if s < 0 { s = s.rem_euclid(dim_size); }
    s.clamp(0, dim_size)
}

fn fixup_end(end: Option<i64>, dim_size: i64) -> i64 {
    let mut e = end.unwrap_or(dim_size);
    if e > dim_size { e = dim_size; }
    if e < 0 { e = e.rem_euclid(dim_size); }
    e.clamp(0, dim_size)
}

fn optional_int(arg: Option<&IrArg>) -> Option<i64> {
    match arg {
        Some(IrArg::Int(v)) => Some(*v),
        Some(IrArg::None) | None => None,
        _ => None,
    }
}

pub struct IrDecomposeStridedSlicePass;

impl IrPass for IrDecomposeStridedSlicePass {
    fn name(&self) -> &str { "IrDecomposeStridedSlice" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let t = &graph.nodes[id].target;
            if !((t.contains("slice_copy.Tensor") || t.contains("slice.Tensor"))
                && !t.contains("slice_scatter"))
            {
                continue;
            }
            if graph.nodes[id].args.len() != 5 { continue; }

            let input_id = match graph.nodes[id].args[0].as_node() { Some(n) => n, None => continue };
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(), None => continue,
            };
            if input_shape.is_empty() { continue; }

            let step = match graph.nodes[id].args[4].as_int() { Some(v) => v, None => continue };
            if step <= 1 { continue; }

            let rank = input_shape.len() as i64;
            let dim_raw = match graph.nodes[id].args[1].as_int() { Some(v) => v, None => continue };
            let dim = dim_raw.rem_euclid(rank) as usize;
            let dim_size = input_shape[dim];
            if dim_size <= 0 { continue; }

            let start = fixup_start(optional_int(graph.nodes[id].args.get(2)), dim_size);
            let end = fixup_end(optional_int(graph.nodes[id].args.get(3)), dim_size);
            let span_len = end - start;
            if span_len <= 0 { continue; }

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let dtype = graph.nodes[input_id].meta.val_dtype()
                .unwrap_or("torch.float32").to_string();

            // 1. Unit-step slice: slice_copy(input, dim, start, end, 1)
            let mut slice_shape = input_shape.clone();
            slice_shape[dim] = span_len;
            let unit_slice = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_unit_slice", node_name),
                "aten.slice_copy.Tensor".to_string(),
                vec![
                    IrArg::Node(input_id),
                    IrArg::Int(dim as i64),
                    IrArg::Int(start),
                    IrArg::Int(end),
                    IrArg::Int(1),
                ],
                meta_with_shape(&orig_meta, slice_shape.clone(), &dtype),
            );
            graph.anchor_to(unit_slice, id);

            let u = (span_len + step - 1) / step;
            let pad_right = u * step - span_len;
            let mut y = unit_slice;

            // 2. Optional pad: full(pad_shape, 0) + cat([slice, pad], dim)
            if pad_right > 0 {
                let pad_shape: Vec<i64> = input_shape.iter().enumerate()
                    .map(|(i, &s)| if i == dim { pad_right } else { s })
                    .collect();
                let fill_val = if dtype.contains("bool") { IrArg::Bool(false) } else { IrArg::Int(0) };
                let pad_node = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_pad", node_name),
                    "aten.full.default".to_string(),
                    vec![i64_list(&pad_shape), fill_val],
                    meta_with_shape(&orig_meta, pad_shape, &dtype),
                );
                graph.anchor_to(pad_node, id);

                let mut cat_shape = slice_shape.clone();
                cat_shape[dim] = span_len + pad_right;
                let cat_node = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_cat", node_name),
                    "aten.cat.default".to_string(),
                    vec![
                        IrArg::List(vec![IrArg::Node(y), IrArg::Node(pad_node)]),
                        IrArg::Int(dim as i64),
                    ],
                    meta_with_shape(&orig_meta, cat_shape, &dtype),
                );
                graph.anchor_to(cat_node, id);
                y = cat_node;
            }

            // 3. View: reshape to [..., u, step, ...]
            let mut split_shape = input_shape.clone();
            split_shape[dim] = u;
            split_shape.insert(dim + 1, step);
            let view1 = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_split_view", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(y), i64_list(&split_shape)],
                meta_with_shape(&orig_meta, split_shape.clone(), &dtype),
            );
            graph.anchor_to(view1, id);

            // 4. Inner slice: slice_copy(view1, dim+1, 0, 1, 1) — take first element
            let mut inner_shape = split_shape.clone();
            inner_shape[dim + 1] = 1;
            let inner_slice = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_inner_slice", node_name),
                "aten.slice_copy.Tensor".to_string(),
                vec![
                    IrArg::Node(view1),
                    IrArg::Int((dim + 1) as i64),
                    IrArg::Int(0),
                    IrArg::Int(1),
                    IrArg::Int(1),
                ],
                meta_with_shape(&orig_meta, inner_shape, &dtype),
            );
            graph.anchor_to(inner_slice, id);

            // 5. View: reshape back to output shape [..., u, ...]
            let mut out_shape = input_shape.clone();
            out_shape[dim] = u;
            let result = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_out_view", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(inner_slice), i64_list(&out_shape)],
                meta_with_shape(&orig_meta, out_shape, &dtype),
            );

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeStridedSlice] decomposed strided slice ops"));
        }
        modified
    }
}
