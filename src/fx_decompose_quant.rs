use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `DecomposeQuantNodesPass`.
///
/// After fold_qdq + insert_rescale, there may be remaining Q/DQ nodes that
/// weren't folded (e.g., at graph boundaries or in non-annotated sub-graphs).
///
/// The full Python pass decomposes these into primitive arithmetic (mul, round,
/// add, clamp, cast for Q; cast, sub, cast, mul for DQ). However, the Python
/// pass skips Q/DQ nodes that still form DQ->Q pairs (those are handled by
/// InsertRescalePass).
///
/// This pass decomposes the remaining stragglers by delegating to the Python
/// helper `decompose_quant_node()` which performs the actual graph surgery
/// with proper FakeTensor metadata propagation.
pub struct FxDecomposeQuantNodesPass;

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor.default")
}

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor.default")
}

fn is_q_or_dq(target: &str) -> bool {
    is_q(target) || is_dq(target)
}

impl FxPass for FxDecomposeQuantNodesPass {
    fn name(&self) -> &str {
        "FxDecomposeQuantNodes"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_q_or_dq(&target) {
                continue;
            }

            // Skip DQ nodes whose users are all Q nodes (these are DQ->Q pairs
            // that should have been handled by InsertRescalePass).
            if is_dq(&target) {
                let users = FxGraphModule::node_users(node)?;
                let all_users_are_q = !users.is_empty()
                    && users.iter().all(|u| {
                        FxGraphModule::node_op(u)
                            .ok()
                            .map(|op| op == "call_function")
                            .unwrap_or(false)
                            && FxGraphModule::node_target_str(u)
                                .ok()
                                .map(|t| is_q(&t))
                                .unwrap_or(false)
                    });
                if all_users_are_q {
                    continue;
                }
            }

            // Skip Q nodes whose input is a DQ node (these are DQ->Q pairs).
            if is_q(&target) {
                let args = FxGraphModule::node_args(node)?;
                if let Some(input) = args.first() {
                    if FxGraphModule::is_node(input) {
                        let input_target = FxGraphModule::node_target_str(input)
                            .unwrap_or_default();
                        if is_dq(&input_target) {
                            continue;
                        }
                    }
                }
            }

            candidates.push(node.clone());
        }

        for node in &candidates {
            let target = FxGraphModule::node_target_str(node)?;
            let is_quant = is_q(&target);
            helpers.call_method1(
                "decompose_quant_node",
                (gm.gm_ref(), node, is_quant),
            )?;
            modified = true;
        }

        if modified {
            // Eliminate dead code (the original Q/DQ nodes are erased by the helper)
            let graph = gm.gm_ref().getattr("graph")?;
            graph.call_method0("eliminate_dead_code")?;
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
