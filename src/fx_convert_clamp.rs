use pyo3::prelude::*;

use crate::fx_graph::{arg_extract, FxGraphModule};
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `ConvertToClampPass`.
///
/// Rewrites `relu.default` and `hardtanh.default` nodes to `aten.clamp.default`
/// by calling into the Python helper to swap the target and args in-place.
pub struct FxConvertToClampPass;

fn is_relu(target: &str) -> bool {
    target.contains("relu.default")
}

fn is_hardtanh(target: &str) -> bool {
    target.contains("hardtanh.default")
}

impl FxPass for FxConvertToClampPass {
    fn name(&self) -> &str {
        "FxConvertToClamp"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;
        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_relu(&target) {
                // relu(x) -> clamp(x, min=0.0, max=None)
                helpers.call_method1(
                    "replace_target_with_clamp",
                    (gm.gm_ref(), node, 0.0_f64, py.None()),
                )?;
                modified = true;
            } else if is_hardtanh(&target) {
                // hardtanh(x, min_val, max_val) -> clamp(x, min_val, max_val)
                let args = FxGraphModule::node_args(node)?;
                let min_val: f64 = arg_extract(&args, 1, -1.0);
                let max_val: f64 = arg_extract(&args, 2, 1.0);
                helpers.call_method1(
                    "replace_target_with_clamp",
                    (gm.gm_ref(), node, min_val, max_val),
                )?;
                modified = true;
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
