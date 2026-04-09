//! Pure Rust IrPass: mean.dim → sum(keepdim=True) * (1/count) + view, or avg_pool2d for 4D spatial.

use std::collections::HashMap;
use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, QParam};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta { shape, dtype: dtype.to_string() });
    m
}

pub struct IrDecomposeMeanPass;

impl IrPass for IrDecomposeMeanPass {
    fn name(&self) -> &str { "IrDecomposeMean" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("mean.dim")
                && !graph.nodes[id].target.contains("mean.default") { continue; }

            let x_id = match graph.nodes[id].first_node_arg() { Some(n) => n, None => continue };
            let input_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(), None => continue,
            };
            let input_rank = input_shape.len() as i64;
            let output_shape = graph.nodes[id].meta.val_shape()
                .map(|s| s.to_vec()).unwrap_or_else(|| vec![1]);

            // Extract dims
            let dims: Vec<i64> = match graph.nodes[id].args.get(1) {
                Some(IrArg::List(items)) => items.iter().filter_map(|a| a.as_int()).collect(),
                Some(IrArg::Int(d)) => vec![*d],
                Some(IrArg::None) | None => (0..input_rank).collect(),
                _ => (0..input_rank).collect(),
            };
            let dims: Vec<i64> = dims.into_iter()
                .map(|d| if d < 0 { d + input_rank } else { d })
                .filter(|&d| input_shape.get(d as usize).copied().unwrap_or(1) != 1)
                .collect();

            let keepdim = graph.nodes[id].args.get(2).and_then(|a| a.as_bool()).unwrap_or(false);

            if dims.is_empty() {
                graph.replace_and_erase(id, x_id);
                modified = true;
                continue;
            }

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let dtype = graph.nodes[x_id].meta.val_dtype().unwrap_or("torch.float32").to_string();

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
                    let mut result = graph.add_node(NodeOp::CallFunction,
                        format!("{}_avgpool", name),
                        "aten.avg_pool2d.default".to_string(),
                        vec![IrArg::Node(x_id), i64_list(&kernel), i64_list(&[1, 1])],
                        meta_with(&orig_meta, pool_shape.clone(), &dtype));
                    graph.anchor_to(result, id);

                    if !keepdim {
                        result = graph.add_node(NodeOp::CallFunction,
                            format!("{}_view", name),
                            "aten.view_copy.default".to_string(),
                            vec![IrArg::Node(result), i64_list(&output_shape)],
                            meta_with(&orig_meta, output_shape.clone(), &dtype));
                    }
                    graph.replace_and_erase(id, result);
                    modified = true;
                    continue;
                }
            }

            // General path: sum(keepdim=True) * full(1/count) + optional view
            let count: f64 = dims.iter()
                .map(|&d| input_shape.get(d as usize).copied().unwrap_or(1) as f64)
                .product();
            if count == 0.0 { continue; }

            // sum(x, dims, keepdim=True)
            let mut sum_shape = input_shape.clone();
            for &d in &dims { sum_shape[d as usize] = 1; }
            let sum_id = graph.add_node(NodeOp::CallFunction,
                format!("{}_sum", name),
                "aten.sum.dim_IntList".to_string(),
                vec![IrArg::Node(x_id), i64_list(&dims), IrArg::Bool(true)],
                meta_with(&orig_meta, sum_shape.clone(), &dtype));
            graph.anchor_to(sum_id, id);

            // full([1]*rank, 1/count)
            let ones_shape: Vec<i64> = vec![1; input_shape.len()];
            let count_id = graph.add_node(NodeOp::CallFunction,
                format!("{}_inv_count", name),
                "aten.full.default".to_string(),
                vec![i64_list(&ones_shape), IrArg::Float(1.0 / count)],
                meta_with(&orig_meta, ones_shape, &dtype));
            graph.anchor_to(count_id, id);

            // mul(sum, 1/count)
            let mul_id = graph.add_node(NodeOp::CallFunction,
                format!("{}_mul", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(sum_id), IrArg::Node(count_id)],
                meta_with(&orig_meta, sum_shape.clone(), &dtype));
            // Note: mul is NOT anchored — it depends on sum_id which may be
            // further decomposed by convert_sum. The multi-pass write_back will
            // place it after its dependencies via the fallback (before output).

            // Annotate qparams if quantized
            if let Some(input_qp) = orig_meta.input_qparam().cloned() {
                if let Some(output_qp) = orig_meta.output_qparam().cloned() {
                    let sum_qp = QParam::per_tensor(
                        input_qp.scale * count, input_qp.zp,
                        input_qp.qmin, input_qp.qmax, input_qp.dtype_str.clone(),
                    );
                    let full_qp = QParam::per_tensor(
                        1.0 / (count * input_qp.qmax as f64), 0,
                        input_qp.qmin, input_qp.qmax, input_qp.dtype_str.clone(),
                    );
                    let mut sum_iqp = HashMap::new();
                    sum_iqp.insert(0, input_qp);
                    graph.node_mut(sum_id).meta.set_input_qparams(sum_iqp);
                    let mut sum_oqp = HashMap::new();
                    sum_oqp.insert(0, sum_qp.clone());
                    graph.node_mut(sum_id).meta.set_output_qparams(sum_oqp);

                    let mut full_oqp = HashMap::new();
                    full_oqp.insert(0, full_qp.clone());
                    graph.node_mut(count_id).meta.set_output_qparams(full_oqp);

                    let mut mul_iqp = HashMap::new();
                    mul_iqp.insert(0, sum_qp);
                    mul_iqp.insert(1, full_qp);
                    graph.node_mut(mul_id).meta.set_input_qparams(mul_iqp);
                    let mut mul_oqp = HashMap::new();
                    mul_oqp.insert(0, output_qp);
                    graph.node_mut(mul_id).meta.set_output_qparams(mul_oqp);
                }
            }

            // Optional view if keepdim=False
            let mut result = mul_id;
            if !keepdim {
                result = graph.add_node(NodeOp::CallFunction,
                    format!("{}_view", name),
                    "aten.view_copy.default".to_string(),
                    vec![IrArg::Node(mul_id), i64_list(&output_shape)],
                    meta_with(&orig_meta, output_shape.clone(), &dtype));
            }

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeMean] decomposed mean ops"));
        }
        modified
    }
}
