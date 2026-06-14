//! Pure Rust IrPass: adjust conv/pool input sizes to yield exact integer outputs.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeOp};
use crate::fx_target::is_size_adjust_target;
use crate::runner_logging::log_line;

fn pair_arg(args: &[IrArg], idx: usize, default: Vec<i64>) -> Vec<i64> {
    match args.get(idx) {
        Some(IrArg::List(items)) => items.iter().filter_map(|a| a.as_int()).collect(),
        Some(IrArg::Int(v)) => vec![*v; default.len()],
        _ => default,
    }
}

pub struct IrSizeAdjustInputPass;

impl IrPass for IrSizeAdjustInputPass {
    fn name(&self) -> &str {
        "IrSizeAdjustInput"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            // Skip nodes marked with skip_size_adjust
            if graph.nodes[id].meta.extra_bool("skip_size_adjust") == Some(true) {
                continue;
            }
            if !is_size_adjust_target(&graph.nodes[id].target) {
                continue;
            }

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) if s.len() >= 4 => s.to_vec(),
                _ => continue,
            };

            let target = graph.nodes[id].target.clone();
            let is_conv = target.contains("conv") || target.contains("CONV");
            let is_pool = target.contains("pool") || target.contains("POOL");
            let node_name = graph.nodes[id].name.clone();
            let dtype = graph.nodes[input_id]
                .meta
                .val_dtype_or_float32()
                .to_string();

            let slices = if is_conv {
                let weight_shape = graph.nodes[id]
                    .args
                    .get(1)
                    .and_then(|a| a.as_node())
                    .and_then(|wid| graph.nodes[wid].meta.val_shape())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                if weight_shape.len() < input_shape.len() {
                    continue;
                }
                let spatial_rank = input_shape.len() - 2;
                if spatial_rank == 0 {
                    continue;
                }

                let stride = pair_arg(&graph.nodes[id].args, 3, vec![1; spatial_rank]);
                let raw_pad = pair_arg(&graph.nodes[id].args, 4, vec![0; spatial_rank]);
                let dilation = pair_arg(&graph.nodes[id].args, 5, vec![1; spatial_rank]);
                let pad_per_dim: Vec<(i64, i64)> = if raw_pad.len() == spatial_rank * 2 {
                    (0..spatial_rank)
                        .map(|i| (raw_pad[i * 2], raw_pad[i * 2 + 1]))
                        .collect()
                } else if raw_pad.len() == spatial_rank {
                    raw_pad.iter().map(|&p| (p, p)).collect()
                } else {
                    vec![(0, 0); spatial_rank]
                };

                let mut slices = Vec::new();
                for i in 0..spatial_rank {
                    let dim = i + 2;
                    let s = stride.get(i).copied().unwrap_or(1);
                    let (pad_before, pad_after) = pad_per_dim.get(i).copied().unwrap_or((0, 0));
                    let total_pad = pad_before + pad_after;
                    let d = dilation.get(i).copied().unwrap_or(1);
                    let k = weight_shape.get(dim).copied().unwrap_or(1);
                    let input_len = input_shape.get(dim).copied().unwrap_or(1);
                    let remainder = (input_len + total_pad - d * (k - 1) - 1) % s;
                    if remainder > pad_before {
                        let adjustment = remainder - pad_before;
                        slices.push((dim as i64, 0i64, input_len - adjustment));
                    }
                }
                slices
            } else if is_pool {
                let kernel = pair_arg(&graph.nodes[id].args, 1, vec![2, 2]);
                let stride = pair_arg(&graph.nodes[id].args, 2, kernel.clone());
                let pad = pair_arg(&graph.nodes[id].args, 3, vec![0, 0]);
                let mut slices = Vec::new();
                for i in 0..2 {
                    let dim = i + 2;
                    let k = kernel.get(i).copied().unwrap_or(2);
                    let s = stride.get(i).copied().unwrap_or(k);
                    let p = pad.get(i).copied().unwrap_or(0);
                    let input_len = input_shape.get(dim).copied().unwrap_or(1);
                    let remainder = (input_len + 2 * p - k) % s;
                    if remainder > p {
                        let adjustment = remainder - p;
                        slices.push((dim as i64, 0i64, input_len - adjustment));
                    }
                }
                slices
            } else {
                continue;
            };

            if slices.is_empty() {
                continue;
            }

            // Create slice nodes and update the conv/pool's input arg.
            // The slice inherits meta from a (possibly layout-neutral) input node
            // that may carry no output_qparams — which leaves it FP32 at
            // serialization while the consumer expects int8. Stamp the consumer's
            // input quantization onto each slice so it stays in the input domain.
            let consumer_input_q = graph.nodes[id]
                .meta
                .input_qparams
                .as_ref()
                .and_then(|m| m.get(&0))
                .cloned();
            let mut current = input_id;
            for (dim, start, end) in &slices {
                let slice_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_adj_d{}", node_name, dim),
                    "aten.slice_copy.Tensor".to_string(),
                    vec![
                        IrArg::Node(current),
                        IrArg::Int(*dim),
                        IrArg::Int(*start),
                        IrArg::Int(*end),
                    ],
                    // Match the FX pass: copy the original input metadata onto
                    // the slice node and let the final fake-tensor refresh
                    // reconcile the exact sliced shape.
                    graph.nodes[input_id]
                        .meta
                        .cloned_with_val(input_shape.clone(), &dtype),
                );
                graph.anchor_after(slice_id, current);
                if let Some(ref q) = consumer_input_q {
                    graph.nodes[slice_id].meta.set_both_qparams(q.clone());
                }
                current = slice_id;
            }

            // Update conv/pool's first arg to point to the sliced input
            graph.set_arg(id, 0, IrArg::Node(current));
            if graph.nodes[id].name.ends_with("_conv2d") {
                graph.anchor_after(id, current);
            }
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrSizeAdjustInput] adjusted conv/pool input sizes"
            ));
        }
        modified
    }
}
