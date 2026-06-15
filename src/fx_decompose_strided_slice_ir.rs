//! Pure Rust IrPass: strided slice_copy (step > 1) → unit slice + pad + view + slice + view.

use crate::fx_ir::{i64_list_arg as i64_list, wrap_dim, IrArg, IrGraph, IrPass};
use crate::fx_target::is_slice_tensor_target;
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

fn fixup_start(start: Option<i64>, dim_size: i64) -> i64 {
    let mut s = start.unwrap_or(0);
    if s < 0 {
        s = s.rem_euclid(dim_size);
    }
    s.clamp(0, dim_size)
}

fn fixup_end(end: Option<i64>, dim_size: i64) -> i64 {
    let mut e = end.unwrap_or(dim_size);
    if e > dim_size {
        e = dim_size;
    }
    if e < 0 {
        e = e.rem_euclid(dim_size);
    }
    e.clamp(0, dim_size)
}

pub struct IrDecomposeStridedSlicePass;

impl IrPass for IrDecomposeStridedSlicePass {
    fn name(&self) -> &str {
        "IrDecomposeStridedSlice"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            let t = &graph.nodes[id].target;
            if !is_slice_tensor_target(t) {
                continue;
            }
            if graph.nodes[id].args.len() != 5 {
                continue;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };
            if input_shape.is_empty() {
                continue;
            }

            let step = match graph.nodes[id].args[4].as_int() {
                Some(v) => v,
                None => continue,
            };
            if step <= 1 {
                continue;
            }

            let dim_raw = match graph.nodes[id].args[1].as_int() {
                Some(v) => v,
                None => continue,
            };
            let Some(dim) = wrap_dim(dim_raw, input_shape.len()) else {
                continue;
            };
            let dim_size = input_shape[dim];
            if dim_size <= 0 {
                continue;
            }

            let start = fixup_start(graph.nodes[id].arg_int(2), dim_size);
            let end = fixup_end(graph.nodes[id].arg_int(3), dim_size);
            let span_len = end - start;
            if span_len <= 0 {
                continue;
            }

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_meta = graph.nodes[input_id].meta.clone();
            let is_bool_dtype = graph.nodes[input_id].meta.val_dtype_is(DType::Bool);
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&input_meta);

            let mut cursor = graph.rewrite_cursor(id);

            // 1. Unit-step slice: slice_copy(input, dim, start, end, 1)
            let mut slice_shape = input_shape.clone();
            slice_shape[dim] = span_len;
            let unit_slice = cursor.slice_copy(
                format!("{}_unit_slice", node_name),
                input_id,
                dim as i64,
                start,
                end,
                Some(1),
                meta.preserve_qparams(slice_shape.clone()),
            );

            let u = (span_len + step - 1) / step;
            let pad_right = u * step - span_len;
            let mut y = unit_slice;

            // 2. Optional pad: full(pad_shape, 0) + cat([slice, pad], dim)
            if pad_right > 0 {
                let pad_shape: Vec<i64> = input_shape
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| if i == dim { pad_right } else { s })
                    .collect();
                let fill_val = if is_bool_dtype {
                    IrArg::Bool(false)
                } else {
                    IrArg::Int(0)
                };
                let pad_node = cursor.call_function(
                    format!("{}_pad", node_name),
                    "aten.full.default",
                    vec![i64_list(&pad_shape), fill_val],
                    meta.preserve_qparams(pad_shape),
                );

                let mut cat_shape = slice_shape.clone();
                cat_shape[dim] = span_len + pad_right;
                let cat_node = cursor.call_function(
                    format!("{}_cat", node_name),
                    "aten.cat.default",
                    vec![
                        IrArg::List(vec![IrArg::Node(y), IrArg::Node(pad_node)]),
                        IrArg::Int(dim as i64),
                    ],
                    meta.preserve_qparams(cat_shape),
                );
                y = cat_node;
            }

            // 3. View: reshape to [..., u, step, ...]
            let mut split_shape = input_shape.clone();
            split_shape[dim] = u;
            split_shape.insert(dim + 1, step);
            let view1 = cursor.view_copy(
                format!("{}_split_view", node_name),
                y,
                &split_shape,
                meta.preserve_qparams(split_shape.clone()),
            );

            // 4. Inner slice: slice_copy(view1, dim+1, 0, 1, 1) — take first element
            let mut inner_shape = split_shape.clone();
            inner_shape[dim + 1] = 1;
            let inner_slice = cursor.slice_copy(
                format!("{}_inner_slice", node_name),
                view1,
                (dim + 1) as i64,
                0,
                1,
                Some(1),
                meta.preserve_qparams(inner_shape),
            );

            // 5. View: reshape back to output shape [..., u, ...]
            let mut out_shape = input_shape.clone();
            out_shape[dim] = u;
            let result = cursor.view_copy(
                format!("{}_out_view", node_name),
                inner_slice,
                &out_shape,
                meta.preserve_qparams(out_shape.clone()),
            );
            cursor.replace_anchor_with(result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeStridedSlice] decomposed strided slice ops"
            ));
        }
        modified
    }
}
