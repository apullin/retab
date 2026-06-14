//! Pure Rust IrPass: mean.dim → sum(keepdim=True) * (1/count) + view, or avg_pool2d for 4D spatial.

use crate::fx_ir::{
    i64_list_arg as i64_list, normalized_reduction_dims, reduced_shape, IrArg, IrGraph, IrPass,
    NodeOp, QParam,
};
use crate::fx_target::is_mean_target;
use crate::runner_logging::log_line;
use std::collections::HashMap;

pub struct IrDecomposeMeanPass;

impl IrPass for IrDecomposeMeanPass {
    fn name(&self) -> &str {
        "IrDecomposeMean"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_mean_target(&graph.nodes[id].target) {
                continue;
            }

            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let keepdim = graph.nodes[id]
                .args
                .get(2)
                .and_then(|a| a.as_bool())
                .unwrap_or(false);
            let dims = normalized_reduction_dims(graph.nodes[id].args.get(1), &input_shape, true);
            let output_shape = graph.nodes[id]
                .meta
                .val_shape()
                .map(|s| s.to_vec())
                .unwrap_or_else(|| reduced_shape(&input_shape, &dims, keepdim));

            if dims.is_empty() {
                graph.replace_and_erase(id, x_id);
                modified = true;
                continue;
            }

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let dtype = graph.nodes[x_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

            // 4D spatial reduction → avg_pool2d path
            if input_shape.len() == 4 && dims.iter().all(|&d| d == 2 || d == 3) {
                let kernel = match dims.as_slice() {
                    [2, 3] | [3, 2] => vec![input_shape[2], input_shape[3]],
                    [2] => vec![input_shape[2], 1],
                    [3] => vec![1, input_shape[3]],
                    _ => Vec::new(),
                };
                if !kernel.is_empty() {
                    let pool_h = 1i64.max(input_shape[2] / kernel[0].max(1));
                    let pool_w = 1i64.max(input_shape[3] / kernel[1].max(1));
                    let pool_shape = vec![input_shape[0], input_shape[1], pool_h, pool_w];
                    let result = graph.add_anchored_call_function(
                        id,
                        format!("{}_avgpool", name),
                        "aten.avg_pool2d.default",
                        vec![IrArg::Node(x_id), i64_list(&kernel), i64_list(&[1, 1])],
                        orig_meta.cloned_with_val(pool_shape.clone(), &dtype),
                    );

                    if !keepdim {
                        graph.replace_with_call_function(
                            id,
                            format!("{}_view", name),
                            "aten.view_copy.default",
                            vec![IrArg::Node(result), i64_list(&output_shape)],
                            orig_meta.cloned_with_val(output_shape.clone(), &dtype),
                        );
                        modified = true;
                        continue;
                    }
                    graph.replace_and_erase(id, result);
                    modified = true;
                    continue;
                }
            }

            // General path: sum(keepdim=True) * full(1/count) + optional view
            let count: f64 = dims
                .iter()
                .map(|&d| input_shape.get(d as usize).copied().unwrap_or(1) as f64)
                .product();
            if count == 0.0 {
                continue;
            }

            // sum(x, dims, keepdim=True)
            let sum_shape = reduced_shape(&input_shape, &dims, true);
            let sum_id = graph.add_anchored_call_function(
                id,
                format!("{}_sum", name),
                "aten.sum.dim_IntList",
                vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
                orig_meta.cloned_with_val(sum_shape.clone(), &dtype),
            );

            // full([1]*rank, 1/count)
            let ones_shape: Vec<i64> = vec![1; input_shape.len()];
            let count_id = graph.add_anchored_call_function(
                id,
                format!("{}_inv_count", name),
                "aten.full.default",
                vec![i64_list(&ones_shape), IrArg::Float(1.0 / count)],
                orig_meta.cloned_with_val(ones_shape, &dtype),
            );

            // mul(sum, 1/count)
            let mul_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_mul", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(sum_id), IrArg::Node(count_id)],
                orig_meta.cloned_with_val(sum_shape.clone(), &dtype),
            );
            // Note: mul is NOT anchored — it depends on sum_id which may be
            // further decomposed by convert_sum. Conversion re-sorts by def-use,
            // so it will still land after its dependencies.

            // Annotate qparams if quantized
            if let Some(input_qp) = orig_meta.input_qparam().cloned() {
                if let Some(output_qp) = orig_meta.output_qparam().cloned() {
                    let sum_qp = QParam::per_tensor(
                        input_qp.scale * count,
                        input_qp.zp,
                        input_qp.qmin,
                        input_qp.qmax,
                        input_qp.dtype_str.clone(),
                    );
                    let full_qp = QParam::per_tensor(
                        1.0 / (count * input_qp.qmax as f64),
                        0,
                        input_qp.qmin,
                        input_qp.qmax,
                        input_qp.dtype_str.clone(),
                    );
                    graph
                        .node_mut(sum_id)
                        .meta
                        .set_single_input_qparam(input_qp);
                    graph
                        .node_mut(sum_id)
                        .meta
                        .set_single_output_qparam(sum_qp.clone());

                    graph
                        .node_mut(count_id)
                        .meta
                        .set_single_output_qparam(full_qp.clone());

                    let mut mul_iqp = HashMap::new();
                    mul_iqp.insert(0, sum_qp);
                    mul_iqp.insert(1, full_qp);
                    graph.node_mut(mul_id).meta.set_input_qparams(mul_iqp);
                    graph
                        .node_mut(mul_id)
                        .meta
                        .set_single_output_qparam(output_qp);
                }
            }

            // Optional view if keepdim=False
            if !keepdim {
                graph.replace_with_call_function(
                    id,
                    format!("{}_view", name),
                    "aten.view_copy.default",
                    vec![IrArg::Node(mul_id), i64_list(&output_shape)],
                    orig_meta.cloned_with_val(output_shape.clone(), &dtype),
                );
                modified = true;
                continue;
            }

            graph.replace_and_erase(id, mul_id);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeMean] decomposed mean ops"));
        }
        modified
    }
}
