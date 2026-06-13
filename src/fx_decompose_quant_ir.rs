//! Pure Rust IrPass: decompose remaining Q/DQ nodes into primitive arithmetic.
//!
//! quantize:   cast(clamp(add(round(mul(x, 1/scale)), zp), qmin, qmax), target_dtype)
//! dequantize: mul(cast(sub(cast(x, int32), zp), fp_dtype), scale)
//!
//! Skips DQ→Q pairs (handled by InsertRescalePass).

use crate::fx_ir::{
    dtype_kwarg, i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp,
};
use crate::runner_logging::log_line;

/// Create a _to_copy.default(x) node with dtype kwarg for casting.
fn add_cast(
    graph: &mut IrGraph,
    name: &str,
    x_id: NodeId,
    target_dtype: &str,
    shape: &[i64],
    orig_meta: &NodeMeta,
    anchor: NodeId,
) -> NodeId {
    if graph.node(x_id).meta.val_dtype() == Some(target_dtype) {
        return x_id;
    }
    graph.add_anchored_call_function_with_kwargs(
        anchor,
        name.to_string(),
        "aten._to_copy.default".to_string(),
        vec![IrArg::Node(x_id)],
        dtype_kwarg(target_dtype),
        orig_meta.cloned_without_qparams(shape.to_vec(), target_dtype),
    )
}

pub struct IrDecomposeQuantPass;

impl IrPass for IrDecomposeQuantPass {
    fn name(&self) -> &str {
        "IrDecomposeQuant"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        // Collect candidates (skip DQ→Q pairs)
        let mut candidates: Vec<(NodeId, bool)> = Vec::new(); // (id, is_quantize)
        for &id in &ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if graph.nodes[id].is_dequantize() {
                // Skip if all users are Q nodes (DQ→Q pair)
                let all_users_q = graph
                    .nodes
                    .iter()
                    .filter(|n| {
                        n.alive
                            && n.args
                                .iter()
                                .any(|a| matches!(a, IrArg::Node(nid) if *nid == id))
                    })
                    .all(|n| n.is_quantize());
                let has_users = graph.nodes.iter().any(|n| {
                    n.alive
                        && n.args
                            .iter()
                            .any(|a| matches!(a, IrArg::Node(nid) if *nid == id))
                });
                if has_users && all_users_q {
                    continue;
                }
                candidates.push((id, false));
            } else if graph.nodes[id].is_quantize() {
                // Skip if input is DQ (DQ→Q pair)
                if let Some(IrArg::Node(input_id)) = graph.nodes[id].args.first() {
                    if graph.nodes[*input_id].is_dequantize() {
                        continue;
                    }
                }
                candidates.push((id, true));
            }
        }

        for (id, is_quantize) in candidates {
            let args = &graph.nodes[id].args;
            if args.len() < 6 {
                continue;
            }
            let x_id = match args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let scale = match &args[1] {
                IrArg::Float(v) => *v,
                IrArg::Int(v) => *v as f64,
                _ => continue,
            };
            let zp = match args[2].as_int() {
                Some(v) => v,
                None => continue,
            };
            let qmin = match args[3].as_int() {
                Some(v) => v,
                None => continue,
            };
            let qmax = match args[4].as_int() {
                Some(v) => v,
                None => continue,
            };
            let target_dtype = match &args[5] {
                IrArg::Dtype(s) => s.clone(),
                IrArg::Str(s) => s.clone(),
                _ => {
                    if qmin == -128 && qmax == 127 {
                        "torch.int8".into()
                    } else if qmin == -32768 {
                        "torch.int16".into()
                    } else {
                        continue;
                    }
                }
            };

            let shape = graph.nodes[x_id]
                .meta
                .val_shape()
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![1]);
            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let rank = shape.len();
            let ones: Vec<i64> = vec![1; rank];

            let fp_dtype = if is_quantize {
                graph.nodes[x_id]
                    .meta
                    .val_dtype()
                    .unwrap_or("torch.float32")
                    .to_string()
            } else {
                graph.nodes[id]
                    .meta
                    .val_dtype()
                    .unwrap_or("torch.float32")
                    .to_string()
            };

            let eff_scale = if is_quantize { 1.0 / scale } else { scale };

            let result = if is_quantize {
                // scale_const = full([1..], 1/scale)
                let sc = graph.add_anchored_call_function(
                    id,
                    format!("{}_sc", name),
                    "aten.full.default".to_string(),
                    vec![i64_list(&ones), IrArg::Float(eff_scale)],
                    orig_meta.cloned_without_qparams(ones.clone(), &fp_dtype),
                );

                // zp_const = full([1..], zp) as float
                let zpc = graph.add_anchored_call_function(
                    id,
                    format!("{}_zp", name),
                    "aten.full.default".to_string(),
                    vec![i64_list(&ones), IrArg::Int(zp)],
                    orig_meta.cloned_without_qparams(ones.clone(), &fp_dtype),
                );

                let scaled = graph.add_anchored_call_function(
                    id,
                    format!("{}_mul", name),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(x_id), IrArg::Node(sc)],
                    orig_meta.cloned_without_qparams(shape.clone(), &fp_dtype),
                );

                let rounded = graph.add_anchored_call_function(
                    id,
                    format!("{}_round", name),
                    "aten.round.default".to_string(),
                    vec![IrArg::Node(scaled)],
                    orig_meta.cloned_without_qparams(shape.clone(), &fp_dtype),
                );

                let shifted = graph.add_anchored_call_function(
                    id,
                    format!("{}_add", name),
                    "aten.add.Tensor".to_string(),
                    vec![IrArg::Node(rounded), IrArg::Node(zpc)],
                    orig_meta.cloned_without_qparams(shape.clone(), &fp_dtype),
                );

                let clamped = graph.add_anchored_call_function(
                    id,
                    format!("{}_clamp", name),
                    "aten.clamp.default".to_string(),
                    vec![
                        IrArg::Node(shifted),
                        IrArg::Float(qmin as f64),
                        IrArg::Float(qmax as f64),
                    ],
                    orig_meta.cloned_without_qparams(shape.clone(), &fp_dtype),
                );

                add_cast(
                    graph,
                    &format!("{}_cast", name),
                    clamped,
                    &target_dtype,
                    &shape,
                    &orig_meta,
                    id,
                )
            } else {
                // dequantize: cast(x, int32) → sub(zp) → cast(fp) → mul(scale)
                let x_i32 = add_cast(
                    graph,
                    &format!("{}_to_i32", name),
                    x_id,
                    "torch.int32",
                    &shape,
                    &orig_meta,
                    id,
                );

                let zpc = graph.add_anchored_call_function(
                    id,
                    format!("{}_zp", name),
                    "aten.full.default".to_string(),
                    vec![i64_list(&ones), IrArg::Int(zp)],
                    orig_meta.cloned_without_qparams(ones.clone(), "torch.int32"),
                );

                let shifted = graph.add_anchored_call_function(
                    id,
                    format!("{}_sub", name),
                    "aten.sub.Tensor".to_string(),
                    vec![IrArg::Node(x_i32), IrArg::Node(zpc)],
                    orig_meta.cloned_without_qparams(shape.clone(), "torch.int32"),
                );

                let casted = add_cast(
                    graph,
                    &format!("{}_to_fp", name),
                    shifted,
                    &fp_dtype,
                    &shape,
                    &orig_meta,
                    id,
                );

                let sc = graph.add_anchored_call_function(
                    id,
                    format!("{}_sc", name),
                    "aten.full.default".to_string(),
                    vec![i64_list(&ones), IrArg::Float(eff_scale)],
                    orig_meta.cloned_without_qparams(ones, &fp_dtype),
                );

                graph.add_call_function(
                    format!("{}_mul", name),
                    "aten.mul.Tensor".to_string(),
                    vec![IrArg::Node(casted), IrArg::Node(sc)],
                    orig_meta.cloned_without_qparams(shape, &fp_dtype),
                )
            };

            if is_quantize {
                if let Some(qp) = orig_meta.output_qparam().cloned() {
                    graph.nodes[result].meta.set_single_output_qparam(qp);
                }
            }

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrDecomposeQuant] decomposed Q/DQ nodes"));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_quant_skips_identity_int32_input_casts() {
        let mut graph = IrGraph::empty();
        let input = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor([2], "torch.int32"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![
                IrArg::Node(input),
                IrArg::Float(0.25),
                IrArg::Int(0),
                IrArg::Int(i32::MIN as i64),
                IrArg::Int(i32::MAX as i64),
                IrArg::Dtype("torch.int32".to_string()),
            ],
            NodeMeta::tensor([2], "torch.float32"),
        );
        let _output = graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::Node(dq)],
            NodeMeta::tensor([2], "torch.float32"),
        );

        assert!(IrDecomposeQuantPass.run(&mut graph));
        assert!(graph.by_name("dq_to_i32").is_none());
        assert!(graph.by_name("dq_to_fp").is_some());
    }
}
