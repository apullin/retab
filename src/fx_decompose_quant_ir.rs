//! Pure Rust IrPass: decompose remaining Q/DQ nodes into primitive arithmetic.
//!
//! quantize:   cast(clamp(add(round(mul(x, 1/scale)), zp), qmin, qmax), target_dtype)
//! dequantize: mul(cast(sub(cast(x, int32), zp), fp_dtype), scale)
//!
//! Skips DQ→Q pairs (handled by InsertRescalePass).

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

pub struct IrDecomposeQuantPass;

impl IrPass for IrDecomposeQuantPass {
    fn name(&self) -> &str {
        "IrDecomposeQuant"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        // Collect candidates (skip DQ→Q pairs)
        let mut candidates: Vec<(NodeId, bool)> = Vec::new(); // (id, is_quantize)
        for id in graph.live_call_function_ids() {
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
            let scale = args[1]
                .as_float()
                .or_else(|| args[1].as_int().map(|v| v as f64));
            let Some(scale) = scale else {
                continue;
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

            let shape = graph.node_shape_vec(x_id).unwrap_or_else(|| vec![1]);
            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let rank = shape.len();
            let ones: Vec<i64> = vec![1; rank];

            let x_meta = graph.nodes[x_id].meta.clone();
            let fp_meta = if is_quantize {
                orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta)
            } else {
                orig_meta.tensor_stamp_same_dtype_or_float32()
            };
            let fp_dtype = fp_meta.dtype();

            let eff_scale = if is_quantize { 1.0 / scale } else { scale };
            let int32_meta = orig_meta.tensor_stamp("torch.int32");

            let result = if is_quantize {
                let mut cursor = graph.rewrite_cursor(id);

                // scale_const = full([1..], 1/scale)
                let sc = cursor.call_function(
                    format!("{}_sc", name),
                    "aten.full.default",
                    vec![i64_list(&ones), IrArg::Float(eff_scale)],
                    fp_meta.drop_qparams(ones.clone()),
                );

                // zp_const = full([1..], zp) as float
                let zpc = cursor.call_function(
                    format!("{}_zp", name),
                    "aten.full.default",
                    vec![i64_list(&ones), IrArg::Int(zp)],
                    fp_meta.drop_qparams(ones.clone()),
                );

                let scaled = cursor.call_function(
                    format!("{}_mul", name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(x_id), IrArg::Node(sc)],
                    fp_meta.drop_qparams(shape.clone()),
                );

                let rounded = cursor.call_function(
                    format!("{}_round", name),
                    "aten.round.default",
                    vec![IrArg::Node(scaled)],
                    fp_meta.drop_qparams(shape.clone()),
                );

                let shifted = cursor.call_function(
                    format!("{}_add", name),
                    "aten.add.Tensor",
                    vec![IrArg::Node(rounded), IrArg::Node(zpc)],
                    fp_meta.drop_qparams(shape.clone()),
                );

                let clamped = cursor.call_function(
                    format!("{}_clamp", name),
                    "aten.clamp.default",
                    vec![
                        IrArg::Node(shifted),
                        IrArg::Float(qmin as f64),
                        IrArg::Float(qmax as f64),
                    ],
                    fp_meta.drop_qparams(shape.clone()),
                );

                cursor.dtype_cast_if_needed(
                    format!("{}_cast", name),
                    clamped,
                    &target_dtype,
                    orig_meta
                        .tensor_stamp(&target_dtype)
                        .drop_qparams(shape.clone()),
                )
            } else {
                let mut cursor = graph.rewrite_cursor(id);

                // dequantize: cast(x, int32) → sub(zp) → cast(fp) → mul(scale)
                let x_i32 = cursor.dtype_cast_if_needed(
                    format!("{}_to_i32", name),
                    x_id,
                    "torch.int32",
                    orig_meta
                        .tensor_stamp("torch.int32")
                        .drop_qparams(shape.clone()),
                );

                let zpc = cursor.call_function(
                    format!("{}_zp", name),
                    "aten.full.default",
                    vec![i64_list(&ones), IrArg::Int(zp)],
                    int32_meta.drop_qparams(ones.clone()),
                );

                let shifted = cursor.call_function(
                    format!("{}_sub", name),
                    "aten.sub.Tensor",
                    vec![IrArg::Node(x_i32), IrArg::Node(zpc)],
                    int32_meta.drop_qparams(shape.clone()),
                );

                let casted = cursor.dtype_cast_if_needed(
                    format!("{}_to_fp", name),
                    shifted,
                    fp_dtype,
                    orig_meta.tensor_stamp(fp_dtype).drop_qparams(shape.clone()),
                );

                let sc = cursor.call_function(
                    format!("{}_sc", name),
                    "aten.full.default",
                    vec![i64_list(&ones), IrArg::Float(eff_scale)],
                    fp_meta.drop_qparams(ones),
                );

                cursor.call_function(
                    format!("{}_mul", name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(casted), IrArg::Node(sc)],
                    fp_meta.drop_qparams(shape),
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
    use crate::test_ir::{call, output, placeholder};

    #[test]
    fn decompose_quant_skips_identity_int32_input_casts() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x", [2], "torch.int32");
        let dq = call(
            &mut graph,
            "dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            vec![
                IrArg::Node(input),
                IrArg::Float(0.25),
                IrArg::Int(0),
                IrArg::Int(i32::MIN as i64),
                IrArg::Int(i32::MAX as i64),
                IrArg::Dtype("torch.int32".to_string()),
            ],
            [2],
            "torch.float32",
        );
        output(&mut graph, dq);

        assert!(IrDecomposeQuantPass.run(&mut graph));
        assert!(graph.by_name("dq_to_i32").is_none());
        assert!(graph.by_name("dq_to_fp").is_some());
    }
}
