//! Pure Rust IrPass: adjust conv/pool input sizes to yield exact integer outputs.

use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::{size_adjust_target_kind, PoolSizeAdjustKind, SizeAdjustTargetKind};
use crate::runner_logging::log_line;

fn pair_arg(args: &[IrArg], idx: usize, default: Vec<i64>) -> Vec<i64> {
    match args.get(idx) {
        Some(IrArg::List(items)) => items.iter().filter_map(|a| a.as_int()).collect(),
        Some(IrArg::Int(v)) => vec![*v; default.len()],
        _ => default,
    }
}

fn pooling_size_adjust_allowed(kind: PoolSizeAdjustKind, node: &crate::fx_ir::IrNode) -> bool {
    match kind {
        PoolSizeAdjustKind::Max => !node.arg_bool_or(5, false),
        PoolSizeAdjustKind::Avg => {
            let ceil_mode = node.arg_bool_or(4, false);
            let count_include_pad = node.arg_bool_or(5, true);
            let divisor_override = node.arg(6).is_some_and(|arg| !matches!(arg, IrArg::None));
            let kernel_area = node.arg_int_pair_or_scalar(1).map(|(h, w)| h * w);
            let divisor_matches_kernel =
                node.arg_int(6).is_some() && node.arg_int(6) == kernel_area;
            let pad = pair_arg(&node.args, 3, vec![0, 0]);
            let no_padding = pad.iter().all(|&value| value == 0);
            !ceil_mode
                && (!count_include_pad || no_padding)
                && (!divisor_override || divisor_matches_kernel)
        }
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
            let Some(target_kind) = size_adjust_target_kind(&graph.nodes[id].target) else {
                continue;
            };

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.node_shape(input_id) {
                Some(s) if s.len() >= 4 => s.to_vec(),
                _ => continue,
            };

            let node_name = graph.nodes[id].name.clone();

            let slices = match target_kind {
                SizeAdjustTargetKind::Conv => {
                    let Some(conv) = ConvArgs::parse(&graph.nodes[id]) else {
                        continue;
                    };
                    if conv.transposed {
                        continue;
                    }
                    let weight_shape = graph
                        .node_shape(conv.weight)
                        .map(|s| s.to_vec())
                        .unwrap_or_default();
                    if weight_shape.len() < input_shape.len() {
                        continue;
                    }
                    let spatial_rank = input_shape.len() - 2;
                    if spatial_rank == 0 {
                        continue;
                    }

                    let stride = &conv.stride;
                    let raw_pad = &conv.padding;
                    let dilation = &conv.dilation;
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
                }
                SizeAdjustTargetKind::Pool(pool_kind) => {
                    if !pooling_size_adjust_allowed(pool_kind, &graph.nodes[id]) {
                        continue;
                    }
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
                }
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
                .input_qparams()
                .and_then(|m| m.get(&0))
                .cloned();
            let mut current = input_id;
            let slice_base_meta = graph.nodes[input_id]
                .meta
                .cloned_with_same_dtype_or_float32(input_shape.clone());
            let mut cursor = graph.rewrite_cursor_after(id, input_id);
            for (dim, start, end) in &slices {
                let mut meta = slice_base_meta.clone();
                if let Some(ref q) = consumer_input_q {
                    meta.set_both_qparams(q.clone());
                }
                let slice_id = cursor.slice_copy(
                    format!("{}_adj_d{}", node_name, dim),
                    current,
                    *dim,
                    *start,
                    *end,
                    None,
                    // Match the FX pass: copy the original input metadata onto
                    // the slice node and let the final fake-tensor refresh
                    // reconcile the exact sliced shape.
                    meta,
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::test_ir::{f32_call, f32_placeholder, output};

    #[test]
    fn convolution_height_and_width_trims_fuse_after_layout_conversion() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 3, 8, 8]);
        let weight = f32_placeholder(&mut graph, "weight", [4, 3, 1, 1]);
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            [1, 4, 4, 4],
        );
        output(&mut graph, conv);
        assert!(IrSizeAdjustInputPass.run(&mut graph));
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten.slice_copy.Tensor")
                .count(),
            2
        );
        crate::fx_memory_format_ir::IrToTosaMemoryFormatPass.run(&mut graph);
        assert!(crate::fx_rewrite_slice_ir::IrRewriteSlicePass.run(&mut graph));
        assert!(crate::fx_fuse_consecutive_slices_ir::IrFuseConsecutiveSlicesPass.run(&mut graph));
        let slice = graph.node(graph.node(conv).arg_node(0).unwrap());
        assert_eq!(slice.target, "backend.tosa.SLICE.default");
        assert_eq!(slice.arg_int_list(1), Some(vec![0, 0, 0, 0]));
        assert_eq!(slice.arg_int_list(2), Some(vec![1, 3, 7, 7]));
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "backend.tosa.SLICE.default")
                .count(),
            1
        );
    }

    #[test]
    fn regression_transposed_convolution_keeps_seven_by_seven_input() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "x", [1, 1, 7, 7]);
        let weight = f32_placeholder(&mut graph, "weight", [1, 1, 2, 2]);
        let conv = f32_call(
            &mut graph,
            "conv_transpose",
            "aten.convolution.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(true),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            [1, 1, 14, 14],
        );
        output(&mut graph, conv);

        assert!(!IrSizeAdjustInputPass.run(&mut graph));
        assert_eq!(graph.node(conv).arg_node(0), Some(input));
        assert_eq!(graph.node_shape(input), Some([1, 1, 7, 7].as_slice()));
        assert_eq!(graph.live_call_function_ids(), vec![conv]);
    }

    #[test]
    fn regression_avg_pool_with_kernel_area_divisor_slices_unused_edges() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "x", [1, 1, 5, 5]);
        let pool = f32_call(
            &mut graph,
            "pool",
            "aten.avg_pool2d.default",
            vec![
                IrArg::Node(input),
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                IrArg::Bool(false),
                IrArg::Bool(true),
                IrArg::Int(4),
            ],
            [1, 1, 2, 2],
        );
        output(&mut graph, pool);
        assert!(!crate::fx_decompose_avgpool_pad_ir::IrDecomposeAvgPoolPadPass.run(&mut graph));

        assert!(IrSizeAdjustInputPass.run(&mut graph));
        let width_slice = graph.node(pool).arg_node(0).unwrap();
        assert_eq!(graph.node(width_slice).target, "aten.slice_copy.Tensor");
        assert_eq!(graph.node(width_slice).arg_int(1), Some(3));
        assert_eq!(graph.node(width_slice).arg_int(3), Some(4));
        let height_slice = graph.node(width_slice).arg_node(0).unwrap();
        assert_eq!(graph.node(height_slice).target, "aten.slice_copy.Tensor");
        assert_eq!(graph.node(height_slice).arg_int(1), Some(2));
        assert_eq!(graph.node(height_slice).arg_int(3), Some(4));
        assert_eq!(graph.node(height_slice).arg_node(0), Some(input));
    }
}
