//! Pure Rust IrPass: mean.dim → sum(keepdim=True) * (1/count) + optional view.

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

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);
            let dtype = meta.dtype();

            if domain.is_empty() {
                let result = if domain.input_shape() == output_shape {
                    x_id
                } else {
                    graph.rewrite_cursor(id).view_copy(
                        format!("{name}_view"),
                        x_id,
                        &output_shape,
                        meta.preserve_qparams(output_shape.clone()),
                    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{f32_call, f32_placeholder, has_live_target, live_node};

    #[test]
    fn sub_rank_four_means_preserve_reduction_rank() {
        for shape in [vec![7], vec![5, 7], vec![3, 5, 7]] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", shape.as_slice());
            let dims: Vec<i64> = (0..shape.len() as i64).collect();
            let output_shape = vec![1; shape.len()];
            f32_call(
                &mut graph,
                "mean",
                "aten.mean.dim",
                vec![IrArg::Node(x), i64_list(&dims), IrArg::Bool(true)],
                output_shape.as_slice(),
            );

            assert!(IrDecomposeMeanPass.run(&mut graph));
            assert_eq!(live_node(&graph, "mean_sum").first_node_arg(), Some(x));
            assert_eq!(
                graph.node_shape(live_node(&graph, "mean_sum").id),
                Some(output_shape.as_slice())
            );
            assert_eq!(
                graph.node_shape(live_node(&graph, "mean_inv_count").id),
                Some(output_shape.as_slice())
            );
            assert!(!graph
                .nodes
                .iter()
                .any(|node| node.alive && node.target == "aten.view_copy.default"));
        }
    }

    #[test]
    fn spatial_means_use_sum_and_mul() {
        for dims in [vec![2], vec![3], vec![2, 3]] {
            let mut graph = IrGraph::empty();
            let shape = [2, 3, 5, 7];
            let x = f32_placeholder(&mut graph, "x", shape);
            let mut output_shape = shape;
            for &dim in &dims {
                output_shape[dim as usize] = 1;
            }
            f32_call(
                &mut graph,
                "mean",
                "aten.mean.dim",
                vec![IrArg::Node(x), i64_list(&dims), IrArg::Bool(true)],
                output_shape,
            );

            assert!(IrDecomposeMeanPass.run(&mut graph));
            assert!(has_live_target(&graph, "mean_sum", "aten.sum.dim_IntList"));
            assert!(has_live_target(&graph, "mean_mul", "aten.mul.Tensor"));
            assert!(!graph
                .nodes
                .iter()
                .any(|node| node.alive && node.target == "aten.avg_pool2d.default"));
        }
    }

    #[test]
    fn mean_without_keepdim_only_reshapes_the_result() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3, 5]);
        let mean = f32_call(
            &mut graph,
            "mean",
            "aten.mean.dim",
            vec![IrArg::Node(x), i64_list(&[-1]), IrArg::Bool(false)],
            [2, 3],
        );

        assert!(IrDecomposeMeanPass.run(&mut graph));
        assert_eq!(
            graph.node_shape(live_node(&graph, "mean_sum").id),
            Some([2, 3, 1].as_slice())
        );
        let view = live_node(&graph, "mean_view");
        assert_eq!(view.target, "aten.view_copy.default");
        assert_eq!(
            view.first_node_arg(),
            Some(live_node(&graph, "mean_mul").id)
        );
        assert_eq!(graph.replacement_of(mean), Some(view.id));
    }

    #[test]
    fn mean_over_unit_axes_keeps_the_output_rank() {
        for (shape, dims, keepdim, output_shape) in [
            (vec![1], vec![0], false, vec![]),
            (vec![1, 1], vec![0, 1], false, vec![]),
            (vec![1, 2], vec![0], false, vec![2]),
            (vec![1], vec![0], true, vec![1]),
            (vec![1, 2], vec![0], true, vec![1, 2]),
            (vec![], vec![], false, vec![]),
        ] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", shape.as_slice());
            let mean = f32_call(
                &mut graph,
                "mean",
                "aten.mean.dim",
                vec![IrArg::Node(x), i64_list(&dims), IrArg::Bool(keepdim)],
                output_shape.as_slice(),
            );
            assert!(IrDecomposeMeanPass.run(&mut graph));
            let replacement = graph.replacement_of(mean).unwrap();
            assert_eq!(graph.node_shape(replacement), Some(output_shape.as_slice()));
            if shape == output_shape {
                assert_eq!(replacement, x);
            } else {
                let view = graph.node(replacement);
                assert_eq!(view.target, "aten.view_copy.default");
                assert_eq!(view.first_node_arg(), Some(x));
                assert_eq!(view.args[1].as_int_list(), Some(output_shape));
            }
        }
    }
}
