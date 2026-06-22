//! Pure Rust IrPass: mean.dim → sum(keepdim=True) * (1/count) + view, or avg_pool2d for 4D spatial.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass};
use crate::fx_reduction::{MeanQParamPlan, ReductionDomain};
use crate::fx_target::is_mean_target;
use crate::runner_logging::log_line;

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
            let input_shape = match graph.node_shape_vec(x_id) {
                Some(s) => s,
                None => continue,
            };
            let keepdim = graph.nodes[id]
                .args
                .get(2)
                .and_then(|a| a.as_bool())
                .unwrap_or(false);
            let domain = ReductionDomain::from_arg(
                input_shape,
                graph.nodes[id].args.get(1),
                keepdim,
                true,
                graph.nodes[id].meta.val_shape().map(|s| s.to_vec()),
            );
            let dims = domain.dims_i64();
            let output_shape = domain.output_shape().to_vec();

            if domain.is_empty() {
                graph.replace_and_erase(id, x_id);
                modified = true;
                continue;
            }

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);
            let dtype = meta.dtype();

            // 4D spatial reduction → avg_pool2d path
            if let Some(kernel) = domain.spatial_avg_pool_kernel_4d() {
                let pool_shape = domain.avg_pool_output_shape(&kernel);
                let result = {
                    let mut cursor = graph.rewrite_cursor(id);
                    let pool = cursor.call_function(
                        format!("{}_avgpool", name),
                        "aten.avg_pool2d.default",
                        vec![IrArg::Node(x_id), i64_list(&kernel), i64_list(&[1, 1])],
                        orig_meta.cloned_with_val(pool_shape.clone(), dtype),
                    );
                    if keepdim {
                        pool
                    } else {
                        cursor.view_copy(
                            format!("{}_view", name),
                            pool,
                            &output_shape,
                            meta.preserve_qparams(output_shape.clone()),
                        )
                    }
                };
                graph.replace_and_erase(id, result);
                modified = true;
                continue;
            }

            // General path: sum(keepdim=True) * full(1/count) + optional view
            let count = domain.element_count_f64();
            if count == 0.0 {
                continue;
            }

            // sum(x, dims, keepdim=True)
            let sum_shape = domain.shape_with_keepdim(true);

            // full([1]*rank, 1/count)
            let ones_shape = domain.input_rank_unit_shape();
            let (sum_id, count_id, mul_id) = {
                let mut cursor = graph.rewrite_cursor(id);
                let sum = cursor.call_function(
                    format!("{}_sum", name),
                    "aten.sum.dim_IntList",
                    vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
                    orig_meta.cloned_with_val(sum_shape.clone(), dtype),
                );
                let count_node = cursor.call_function(
                    format!("{}_inv_count", name),
                    "aten.full.default",
                    vec![i64_list(&ones_shape), IrArg::Float(1.0 / count)],
                    orig_meta.cloned_with_val(ones_shape, dtype),
                );
                let mul = cursor.call_function(
                    format!("{}_mul", name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(sum), IrArg::Node(count_node)],
                    orig_meta.cloned_with_val(sum_shape.clone(), dtype),
                );
                (sum, count_node, mul)
            };

            // Annotate qparams if quantized
            if let Some(plan) = MeanQParamPlan::from_meta(&orig_meta, count) {
                plan.stamp_sum(&mut graph.node_mut(sum_id).meta);
                plan.stamp_coefficient(&mut graph.node_mut(count_id).meta);
                plan.stamp_mul(&mut graph.node_mut(mul_id).meta);
            }

            // Optional view if keepdim=False
            let result = if keepdim {
                mul_id
            } else {
                graph.rewrite_cursor_after(id, mul_id).view_copy(
                    format!("{}_view", name),
                    mul_id,
                    &output_shape,
                    meta.preserve_qparams(output_shape.clone()),
                )
            };
            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeMean] decomposed mean ops"));
        }
        modified
    }
}
