//! Pure Rust IrPass: mean.dim → sum(keepdim=True) * (1/count) + optional view.

use std::collections::HashSet;

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeMeta};
use crate::fx_quant::QParam;
use crate::fx_reduction::{MeanQParamPlan, ReductionDomain};
use crate::fx_target::is_mean_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeMeanPass;

fn qdq_args(input: usize, qparam: &QParam) -> Vec<IrArg> {
    vec![
        IrArg::Node(input),
        IrArg::Float(qparam.scale),
        IrArg::Int(qparam.zp),
        IrArg::Int(qparam.qmin),
        IrArg::Int(qparam.qmax),
        IrArg::Dtype(qparam.dtype_str.clone()),
    ]
}

fn propagate_layout_extras(source: &NodeMeta, dest: &mut NodeMeta) {
    for key in [
        "retab_force_identity_dim_order",
        "retab_reduce_axes_are_physical",
        "retab_reduction_element_count",
    ] {
        if let Some(value) = source.extras.get(key).cloned() {
            dest.set_extra(key, value);
        }
    }
}

fn explicit_reduction_count(meta: &NodeMeta) -> Option<f64> {
    meta.extras
        .get("retab_reduction_element_count")
        .and_then(|value| match value {
            crate::fx_ir::MetaValue::Str(raw) => raw.parse::<f64>().ok(),
            _ => None,
        })
        .filter(|count| *count > 0.0)
}

fn effective_source_qparam(graph: &IrGraph, start: usize) -> Option<QParam> {
    let mut current = start;
    let mut seen = HashSet::new();
    while seen.insert(current) {
        let node = graph.node(current);
        if !node.alive {
            return None;
        }
        if let Some(qparam) = node
            .meta
            .output_qparam()
            .cloned()
            .filter(QParam::is_quantized_integer_dtype)
        {
            return Some(qparam);
        }
        current = node.first_node_arg()?;
    }
    None
}

fn mean_sum_qparam(graph: &IrGraph, input_id: usize, count: f64) -> Option<QParam> {
    if count == 0.0 {
        return None;
    }
    let input_qparam = effective_source_qparam(graph, input_id)?;
    Some(QParam::per_tensor(
        input_qparam.scale * count,
        input_qparam.zp,
        input_qparam.qmin,
        input_qparam.qmax,
        input_qparam.dtype_str,
    ))
}

fn fp_output_qdq_qparam(meta: &crate::fx_meta::NodeMeta) -> Option<QParam> {
    if meta.val_dtype_kind().is_some_and(|dtype| !dtype.is_float()) {
        return None;
    }
    meta.output_qparam()
        .cloned()
        .filter(QParam::is_quantized_integer_dtype)
}

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
            if crate::config::debug_node_probe()
                .as_deref()
                .is_some_and(|probe| name.contains(probe))
            {
                eprintln!(
                    "[MeanProbe] name={} dtype={} orig_dtype={:?} input_qparams={:?} output_qparams={:?}",
                    name,
                    dtype,
                    orig_meta.val_dtype(),
                    orig_meta.input_qparams(),
                    orig_meta.output_qparams()
                );
            }

            // General path: sum(keepdim=True) * full(1/count) + optional view
            let count =
                explicit_reduction_count(&orig_meta).unwrap_or_else(|| domain.element_count_f64());
            if count == 0.0 {
                continue;
            }
            let qparam_plan = MeanQParamPlan::from_meta(&orig_meta, count);
            let sum_qdq = if qparam_plan.is_none() {
                mean_sum_qparam(graph, x_id, count)
            } else {
                None
            };
            let output_qdq = if qparam_plan.is_none() {
                fp_output_qdq_qparam(&orig_meta)
            } else {
                None
            };
            if crate::config::debug_node_probe()
                .as_deref()
                .is_some_and(|probe| name.contains(probe))
            {
                eprintln!(
                    "[MeanProbe] count={} plan={} sum_qdq={:?} output_qdq={:?}",
                    count,
                    qparam_plan.is_some(),
                    sum_qdq,
                    output_qdq
                );
            }
            let is_quantized_mean = qparam_plan.is_some();
            let helper_meta = |shape: Vec<i64>| {
                if is_quantized_mean {
                    orig_meta.cloned_with_val(shape, dtype)
                } else {
                    meta.drop_qparams(shape)
                }
            };

            // sum(x, dims, keepdim=True)
            let sum_shape = domain.shape_with_keepdim(true);

            // full([1]*rank, 1/count)
            let ones_shape = domain.input_rank_unit_shape();
            let sum_meta = helper_meta(sum_shape.clone());
            let count_meta = helper_meta(ones_shape.clone());
            let mul_meta = helper_meta(sum_shape.clone());
            let (sum_id, count_id) = {
                let mut cursor = graph.rewrite_cursor(id);
                let sum = cursor.call_function(
                    format!("{}_sum", name),
                    "aten.sum.dim_IntList",
                    vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
                    sum_meta,
                );
                let count_node = cursor.call_function(
                    format!("{}_inv_count", name),
                    "aten.full.default",
                    vec![i64_list(&ones_shape), IrArg::Float(1.0 / count)],
                    count_meta,
                );
                (sum, count_node)
            };

            let sum_for_mul = if let Some(qparam) = sum_qdq {
                let mut cursor = graph.rewrite_cursor_after(id, count_id);
                let quant = cursor.call_function(
                    format!("{}_sum_quantize", name),
                    "quantized_decomposed.quantize_per_tensor.default",
                    qdq_args(sum_id, &qparam),
                    meta.with_dtype(qparam.dtype_str.clone())
                        .drop_qparams(sum_shape.clone()),
                );
                cursor.call_function(
                    format!("{}_sum_dequantize", name),
                    "quantized_decomposed.dequantize_per_tensor.default",
                    qdq_args(quant, &qparam),
                    meta.drop_qparams(sum_shape.clone()),
                )
            } else {
                sum_id
            };

            let mul_id = {
                let mut cursor = graph.rewrite_cursor_after(id, sum_for_mul);
                cursor.call_function(
                    format!("{}_mul", name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(sum_for_mul), IrArg::Node(count_id)],
                    mul_meta,
                )
            };

            // Annotate qparams if quantized
            if let Some(plan) = qparam_plan {
                plan.stamp_sum(&mut graph.node_mut(sum_id).meta);
                plan.stamp_coefficient(&mut graph.node_mut(count_id).meta);
                plan.stamp_mul(&mut graph.node_mut(mul_id).meta);
            }
            for node_id in [sum_id, count_id, sum_for_mul, mul_id] {
                propagate_layout_extras(&orig_meta, &mut graph.node_mut(node_id).meta);
            }

            let mean_value = if let Some(qparam) = output_qdq {
                let mut cursor = graph.rewrite_cursor_after(id, mul_id);
                let quant = cursor.call_function(
                    format!("{}_folded_quantize", name),
                    "quantized_decomposed.quantize_per_tensor.default",
                    qdq_args(mul_id, &qparam),
                    meta.with_dtype(qparam.dtype_str.clone())
                        .drop_qparams(sum_shape.clone()),
                );
                cursor.call_function(
                    format!("{}_folded_dequantize", name),
                    "quantized_decomposed.dequantize_per_tensor.default",
                    qdq_args(quant, &qparam),
                    meta.drop_qparams(sum_shape.clone()),
                )
            } else {
                mul_id
            };
            propagate_layout_extras(&orig_meta, &mut graph.node_mut(mean_value).meta);

            // Optional view if keepdim=False
            let result = if keepdim {
                mean_value
            } else {
                let result_meta = if graph.node(mean_value).meta.output_qparam().is_some() {
                    meta.preserve_qparams(output_shape.clone())
                } else {
                    meta.drop_qparams(output_shape.clone())
                };
                graph.rewrite_cursor_after(id, mean_value).view_copy(
                    format!("{}_view", name),
                    mean_value,
                    &output_shape,
                    result_meta,
                )
            };
            propagate_layout_extras(&orig_meta, &mut graph.node_mut(result).meta);
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
    use crate::fx_meta::NodeMeta;
    use crate::fx_quant::QParam;
    use crate::test_ir::{call_with_meta, f32_placeholder, live_node, output};

    fn qparam(scale: f64) -> QParam {
        QParam::per_tensor(scale, 0, -128, 127, "torch.int8".to_string())
    }

    #[test]
    fn float_mean_decomposition_reifies_output_qdq_without_quantizing_helpers() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 1280, 7, 7]);
        let mut mean_meta = NodeMeta::tensor([1, 1280, 1, 1], "torch.float32");
        mean_meta.set_unary_input_output_qparams(qparam(0.25), qparam(0.5));
        let mean = call_with_meta(
            &mut graph,
            "aten_mean_dim",
            "aten.mean.dim",
            vec![IrArg::Node(x), i64_list(&[2, 3]), IrArg::Bool(true)],
            mean_meta,
        );
        output(&mut graph, mean);

        assert!(IrDecomposeMeanPass.run(&mut graph));

        for name in [
            "aten_mean_dim_sum",
            "aten_mean_dim_inv_count",
            "aten_mean_dim_mul",
        ] {
            let node = live_node(&graph, name);
            assert_eq!(node.meta.val_dtype(), Some("torch.float32"));
            assert!(node.meta.input_qparams().is_none(), "{name} input qparams");
            assert!(
                node.meta.output_qparams().is_none(),
                "{name} output qparams"
            );
        }

        let mul = live_node(&graph, "aten_mean_dim_mul");
        let quant = live_node(&graph, "aten_mean_dim_folded_quantize");
        let dequant = live_node(&graph, "aten_mean_dim_folded_dequantize");
        assert_eq!(
            quant.target,
            "quantized_decomposed.quantize_per_tensor.default"
        );
        assert_eq!(
            dequant.target,
            "quantized_decomposed.dequantize_per_tensor.default"
        );
        assert_eq!(quant.meta.val_dtype(), Some("torch.int8"));
        assert_eq!(dequant.meta.val_dtype(), Some("torch.float32"));
        assert!(matches!(quant.args[0], IrArg::Node(id) if id == mul.id));
        assert!(matches!(dequant.args[0], IrArg::Node(id) if id == quant.id));
        assert!(matches!(
            live_node(&graph, "output").args[0],
            IrArg::Node(id) if id == dequant.id
        ));
    }

    #[test]
    fn float_mean_from_quantized_source_quantizes_sum_before_reciprocal() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 1280, 7, 7]);
        graph
            .node_mut(x)
            .meta
            .set_single_output_qparam(qparam(0.25));
        let mean = call_with_meta(
            &mut graph,
            "aten_mean_dim",
            "aten.mean.dim",
            vec![IrArg::Node(x), i64_list(&[2, 3]), IrArg::Bool(true)],
            NodeMeta::tensor([1, 1280, 1, 1], "torch.float32"),
        );
        output(&mut graph, mean);

        assert!(IrDecomposeMeanPass.run(&mut graph));

        let sum = live_node(&graph, "aten_mean_dim_sum");
        let quant = live_node(&graph, "aten_mean_dim_sum_quantize");
        let dequant = live_node(&graph, "aten_mean_dim_sum_dequantize");
        let mul = live_node(&graph, "aten_mean_dim_mul");

        assert!(matches!(quant.args[0], IrArg::Node(id) if id == sum.id));
        assert!(
            matches!(quant.args[1], IrArg::Float(scale) if (scale - 12.25).abs() < f64::EPSILON)
        );
        assert!(matches!(dequant.args[0], IrArg::Node(id) if id == quant.id));
        assert!(matches!(mul.args[0], IrArg::Node(id) if id == dequant.id));
    }
}
