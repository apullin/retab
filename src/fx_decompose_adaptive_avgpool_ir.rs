//! Pure Rust IrPass: adaptive_avg_pool2d → slice + avg_pool2d + cat.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::is_adaptive_avg_pool2d_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeAdaptiveAvgPoolPass;

impl IrPass for IrDecomposeAdaptiveAvgPoolPass {
    fn name(&self) -> &str {
        "IrDecomposeAdaptiveAvgPool"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_adaptive_avg_pool2d_target(&graph.nodes[id].target) {
                continue;
            }

            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.node_shape(x_id) {
                Some(s) if s.len() >= 4 => s.to_vec(),
                _ => continue,
            };
            let (n_batch, c_ch, input_h, input_w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );

            let (output_h, output_w) = match &graph.nodes[id].args.get(1) {
                Some(IrArg::List(items)) if items.len() == 2 => {
                    let h = items[0].as_int().unwrap_or(1);
                    let w = items[1].as_int().unwrap_or(1);
                    (h, w)
                }
                _ => (1, 1),
            };

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let dtype = graph.node_dtype_or_float32(x_id).to_string();
            let meta = orig_meta.tensor_stamp(&dtype);

            let mut rows: Vec<NodeId> = Vec::new();
            for out_i in 0..output_h {
                let mut row: Vec<NodeId> = Vec::new();
                for out_j in 0..output_w {
                    let start_h = out_i * input_h / output_h;
                    let end_h = (out_i + 1) * input_h / output_h
                        + if ((out_i + 1) * input_h) % output_h != 0 {
                            1
                        } else {
                            0
                        };
                    let start_w = out_j * input_w / output_w;
                    let end_w = (out_j + 1) * input_w / output_w
                        + if ((out_j + 1) * input_w) % output_w != 0 {
                            1
                        } else {
                            0
                        };
                    let kh = end_h - start_h;
                    let kw = end_w - start_w;

                    // slice_copy(x, dim=2, start_h, end_h)
                    let slice_h = graph.add_anchored_slice_copy(
                        id,
                        format!("{}_sh_{}_{}", node_name, out_i, out_j),
                        x_id,
                        2,
                        start_h,
                        end_h,
                        None,
                        meta.preserve_qparams(vec![n_batch, c_ch, kh, input_w]),
                    );

                    // slice_copy(slice_h, dim=3, start_w, end_w)
                    let slice_hw = graph.add_anchored_slice_copy(
                        id,
                        format!("{}_sw_{}_{}", node_name, out_i, out_j),
                        slice_h,
                        3,
                        start_w,
                        end_w,
                        None,
                        meta.preserve_qparams(vec![n_batch, c_ch, kh, kw]),
                    );

                    // avg_pool2d(slice_hw, [kh,kw], [1,1], [0,0])
                    let pooled = graph.add_anchored_call_function(
                        id,
                        format!("{}_pool_{}_{}", node_name, out_i, out_j),
                        "aten.avg_pool2d.default",
                        vec![
                            IrArg::Node(slice_hw),
                            i64_list(&[kh, kw]), // kernel
                            i64_list(&[1, 1]),   // stride
                            i64_list(&[0, 0]),   // padding
                        ],
                        meta.preserve_qparams(vec![n_batch, c_ch, 1, 1]),
                    );

                    row.push(pooled);
                }

                // Cat columns along W (dim=3) if multiple
                if row.len() == 1 {
                    rows.push(row[0]);
                } else {
                    let cat_args = IrArg::List(row.iter().map(|&n| IrArg::Node(n)).collect());
                    let row_cat = graph.add_anchored_call_function(
                        id,
                        format!("{}_cat_w_{}", node_name, out_i),
                        "aten.cat.default",
                        vec![cat_args, IrArg::Int(3)],
                        meta.preserve_qparams(vec![n_batch, c_ch, 1, output_w]),
                    );
                    rows.push(row_cat);
                }
            }

            // Cat rows along H (dim=2) if multiple
            let result = if rows.len() == 1 {
                rows[0]
            } else {
                let cat_args = IrArg::List(rows.iter().map(|&n| IrArg::Node(n)).collect());
                graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_cat_h", node_name),
                    "aten.cat.default".to_string(),
                    vec![cat_args, IrArg::Int(2)],
                    meta.preserve_qparams(vec![n_batch, c_ch, output_h, output_w]),
                )
            };

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeAdaptiveAvgPool] decomposed adaptive pool ops"
            ));
        }
        modified
    }
}
