use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `DecomposeLinearPass`.
///
/// Decomposes `aten.linear.default(x, weight, bias)` into:
///   1. Reshape input to 4D: [N, ..., Ci] -> [N*..., Ci, 1, 1]
///   2. Reshape weight to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
///   3. Conv2d with 1x1 kernel (inherits qparams from linear)
///   4. Reshape output back: [N*..., Co, 1, 1] -> [N, ..., Co]
///
/// The graph surgery (node creation with proper FakeTensor metadata) is
/// delegated to `fx_helpers.decompose_linear_node()`.
pub struct FxDecomposeLinearPass;

fn is_linear(target: &str) -> bool {
    target.contains("linear.default")
}

impl FxPass for FxDecomposeLinearPass {
    fn name(&self) -> &str {
        "FxDecomposeLinear"
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
            if is_linear(&target) {
                candidates.push(node.clone());
            }
        }

        for node in &candidates {
            helpers.call_method1("decompose_linear_node", (gm.gm_ref(), node))?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
