//! Pure Rust IrPass: adaptive_avg_pool2d → slice + avg_pool2d + cat.

use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeOp,
};
use crate::runner_logging::log_line;

pub struct IrDecomposeAdaptiveAvgPoolPass;

impl IrPass for IrDecomposeAdaptiveAvgPoolPass {
    fn name(&self) -> &str {
        "IrDecomposeAdaptiveAvgPool"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("adaptive_avg_pool2d") {
                continue;
            }

            let x_id = match graph.nodes[id].args.first().and_then(|a| a.as_node()) {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.nodes[x_id].meta.val_shape() {
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
            let dtype = graph.nodes[x_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

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
                    let slice_h = graph.add_node(
                        NodeOp::CallFunction,
                        format!("{}_sh_{}_{}", node_name, out_i, out_j),
                        "aten.slice_copy.Tensor".to_string(),
                        vec![
                            IrArg::Node(x_id),
                            IrArg::Int(2),
                            IrArg::Int(start_h),
                            IrArg::Int(end_h),
                        ],
                        orig_meta.cloned_with_val(vec![n_batch, c_ch, kh, input_w], &dtype),
                    );
                    graph.anchor_to(slice_h, id);

                    // slice_copy(slice_h, dim=3, start_w, end_w)
                    let slice_hw = graph.add_node(
                        NodeOp::CallFunction,
                        format!("{}_sw_{}_{}", node_name, out_i, out_j),
                        "aten.slice_copy.Tensor".to_string(),
                        vec![
                            IrArg::Node(slice_h),
                            IrArg::Int(3),
                            IrArg::Int(start_w),
                            IrArg::Int(end_w),
                        ],
                        orig_meta.cloned_with_val(vec![n_batch, c_ch, kh, kw], &dtype),
                    );
                    graph.anchor_to(slice_hw, id);

                    // avg_pool2d(slice_hw, [kh,kw], [1,1], [0,0])
                    let pooled = graph.add_node(
                        NodeOp::CallFunction,
                        format!("{}_pool_{}_{}", node_name, out_i, out_j),
                        "aten.avg_pool2d.default".to_string(),
                        vec![
                            IrArg::Node(slice_hw),
                            i64_list(&[kh, kw]), // kernel
                            i64_list(&[1, 1]),   // stride
                            i64_list(&[0, 0]),   // padding
                        ],
                        orig_meta.cloned_with_val(vec![n_batch, c_ch, 1, 1], &dtype),
                    );
                    graph.anchor_to(pooled, id);

                    row.push(pooled);
                }

                // Cat columns along W (dim=3) if multiple
                if row.len() == 1 {
                    rows.push(row[0]);
                } else {
                    let cat_args = IrArg::List(row.iter().map(|&n| IrArg::Node(n)).collect());
                    let row_cat = graph.add_node(
                        NodeOp::CallFunction,
                        format!("{}_cat_w_{}", node_name, out_i),
                        "aten.cat.default".to_string(),
                        vec![cat_args, IrArg::Int(3)],
                        orig_meta.cloned_with_val(vec![n_batch, c_ch, 1, output_w], &dtype),
                    );
                    graph.anchor_to(row_cat, id);
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
                    orig_meta.cloned_with_val(vec![n_batch, c_ch, output_h, output_w], &dtype),
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
