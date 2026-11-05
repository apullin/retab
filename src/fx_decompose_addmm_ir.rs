//! Pure Rust IrPass version of FxDecomposeAddmmPass.
//!
//! Decomposes `aten.addmm` into mm + (optional mul/full scalars) + add.
//! addmm(bias, mat1, mat2, beta=1, alpha=1) =>
//!   mm_result = mm(mat1, mat2)
//!   scaled_mm = alpha != 1 ? mul(mm_result, full(alpha)) : mm_result
//!   scaled_bias = beta != 1 ? mul(bias, full(beta)) : bias
//!   result = add(scaled_mm, scaled_bias)

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

/// Compute the output shape for mm(mat1, mat2).
/// mat1: [M, K], mat2: [K, N] => [M, N]
fn mm_output_shape(mat1_shape: &[i64], mat2_shape: &[i64]) -> Vec<i64> {
    if mat1_shape.len() == 2 && mat2_shape.len() == 2 {
        vec![mat1_shape[0], mat2_shape[1]]
    } else {
        // Fallback: return empty (caller should skip if shapes unavailable)
        Vec::new()
    }
}

pub struct IrDecomposeAddmmPass;

impl IrPass for IrDecomposeAddmmPass {
    fn name(&self) -> &str {
        "IrDecomposeAddmm"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].is_addmm() {
                continue;
            }

            // addmm args: [bias, mat1, mat2, (beta), (alpha)]
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            let bias_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let mat1_id = match graph.nodes[id].args[1].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let mat2_id = match graph.nodes[id].args[2].as_node() {
                Some(nid) => nid,
                None => continue,
            };

            let beta = graph.nodes[id]
                .arg_or_kwarg_numeric_f64(3, "beta")
                .unwrap_or(1.0);
            let alpha = graph.nodes[id]
                .arg_or_kwarg_numeric_f64(4, "alpha")
                .unwrap_or(1.0);

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();

            // Compute mm output shape from mat1 and mat2 shapes
            let mat1_shape = graph.node_shape_vec(mat1_id);
            let mat2_shape = graph.node_shape_vec(mat2_id);

            let mm_meta = match (&mat1_shape, &mat2_shape) {
                (Some(s1), Some(s2)) => {
                    let shape = mm_output_shape(s1, s2);
                    if shape.is_empty() {
                        orig_meta.clone()
                    } else {
                        orig_meta.cloned_with_same_dtype_or_float32(shape)
                    }
                }
                _ => orig_meta.clone(),
            };

            let bias_meta = graph.nodes[bias_id].meta.clone();
            let mut cursor = graph.rewrite_cursor(id);

            // Step 1: mm(mat1, mat2)
            let mm_id = cursor.call_function(
                format!("{}_mm", node_name),
                "aten.mm.default",
                vec![IrArg::Node(mat1_id), IrArg::Node(mat2_id)],
                mm_meta.clone(),
            );

            // Step 2: optionally scale mm result by alpha
            let scaled_mm_id = if (alpha - 1.0).abs() > 1e-10 {
                let rank = mm_meta.val_shape().map(|s| s.len()).unwrap_or(1);
                let full_shape: Vec<IrArg> = vec![IrArg::Int(1); rank];
                let full_id = cursor.call_function(
                    format!("{}_alpha_full", node_name),
                    "aten.full.default",
                    vec![IrArg::List(full_shape), IrArg::Float(alpha)],
                    mm_meta.clone(),
                );
                cursor.call_function(
                    format!("{}_alpha_mul", node_name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(mm_id), IrArg::Node(full_id)],
                    mm_meta.clone(),
                )
            } else {
                mm_id
            };

            // Step 3: optionally scale bias by beta
            let scaled_bias_id = if (beta - 1.0).abs() > 1e-10 {
                let rank = bias_meta.val_shape().map(|s| s.len()).unwrap_or(1);
                let full_shape: Vec<IrArg> = vec![IrArg::Int(1); rank];
                let full_id = cursor.call_function(
                    format!("{}_beta_full", node_name),
                    "aten.full.default",
                    vec![IrArg::List(full_shape), IrArg::Float(beta)],
                    bias_meta.clone(),
                );
                cursor.call_function(
                    format!("{}_beta_mul", node_name),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(bias_id), IrArg::Node(full_id)],
                    bias_meta,
                )
            } else {
                bias_id
            };

            // Step 4: add(scaled_mm, scaled_bias)
            let result = cursor.call_function(
                format!("{}_add", node_name),
                "aten.add.Tensor",
                vec![IrArg::Node(scaled_mm_id), IrArg::Node(scaled_bias_id)],
                orig_meta,
            );
            cursor.replace_anchor_with(result);
            modified = true;

            log_line(format_args!(
                "[IrDecomposeAddmm] decomposed: {} (beta={}, alpha={})",
                node_name, beta, alpha
            ));
        }

        modified
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::fx_ir::{NodeMeta, NodeOp};
    use crate::test_ir::f32_placeholder as placeholder;

    #[test]
    fn decompose_addmm_reads_alpha_beta_from_kwargs() {
        let mut graph = IrGraph::empty();
        let bias = placeholder(&mut graph, "bias", [2, 2]);
        let mat1 = placeholder(&mut graph, "mat1", [2, 3]);
        let mat2 = placeholder(&mut graph, "mat2", [3, 2]);
        let mut kwargs = HashMap::new();
        kwargs.insert("alpha".to_string(), IrArg::Float(0.0));
        kwargs.insert("beta".to_string(), IrArg::Float(1.0));
        let addmm = graph.add_node_with_kwargs(
            NodeOp::CallFunction,
            "addmm".to_string(),
            "aten.addmm.default".to_string(),
            vec![IrArg::Node(bias), IrArg::Node(mat1), IrArg::Node(mat2)],
            kwargs,
            NodeMeta::tensor([2, 2], "torch.float32"),
        );

        assert!(IrDecomposeAddmmPass.run(&mut graph));
        assert!(!graph.nodes[addmm].alive);

        let mm = graph.by_name("addmm_mm").unwrap();
        let alpha_full = graph.by_name("addmm_alpha_full").unwrap();
        let alpha_mul = graph.by_name("addmm_alpha_mul").unwrap();
        let add = graph.by_name("addmm_add").unwrap();
        // alpha=0.0 (!= the default 1.0) is read from kwargs, so mm is scaled by full(alpha)
        assert_eq!(alpha_full.args[1].as_float(), Some(0.0));
        assert_eq!(alpha_mul.args[0].as_node(), Some(mm.id));
        assert_eq!(alpha_mul.args[1].as_node(), Some(alpha_full.id));
        assert_eq!(add.args[0].as_node(), Some(alpha_mul.id));
        assert_eq!(add.args[1].as_node(), Some(bias));
        // beta=1.0 is the default, so the bias is not scaled
        assert!(graph.by_name("addmm_beta_mul").is_none());
    }
}
