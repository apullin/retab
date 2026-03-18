use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of sum-to-REDUCE_SUM conversion.
///
/// Rewrites `aten.sum.dim_IntList` nodes into one or more chained
/// `tosa.REDUCE_SUM` operations (one per reduction axis), since TOSA
/// REDUCE_SUM only reduces a single axis at a time.
///
/// For multi-axis reductions the dims are processed in descending order
/// so that earlier reductions don't shift the indices of later ones.
pub struct FxConvertSumPass;

fn is_sum(target: &str) -> bool {
    target.contains("sum.dim_IntList")
}

impl FxPass for FxConvertSumPass {
    fn name(&self) -> &str {
        "FxConvertSum"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        // Collect candidates to avoid mutation during iteration.
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_sum(&target) {
                candidates.push(node.clone());
            }
        }

        for node in &candidates {
            let args = FxGraphModule::node_args(node)?;

            // args[0] = input tensor, args[1] = dims list
            if args.len() < 2 {
                continue;
            }

            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            // Extract the dims list.  It may be a Python list/tuple of ints.
            let dims_obj = &args[1];
            let dims: Vec<i64> = match dims_obj.extract::<Vec<i64>>() {
                Ok(d) => d,
                Err(_) => {
                    // Might be a single int
                    match dims_obj.extract::<i64>() {
                        Ok(d) => vec![d],
                        Err(_) => continue,
                    }
                }
            };

            if dims.is_empty() {
                continue;
            }

            // Check for keepdim (args[2] if present, default True for TOSA).
            let keepdim: bool = args
                .get(2)
                .and_then(|a| a.extract().ok())
                .unwrap_or(false);

            // Delegate to Python helper which:
            //   1. Sorts dims descending
            //   2. Creates chained REDUCE_SUM nodes (one per dim)
            //   3. Replaces all uses of the original sum with the final REDUCE_SUM
            //   4. Erases the original sum node
            helpers.call_method1(
                "convert_sum_to_reduce_sum",
                (gm.gm_ref(), node, input_node, dims, keepdim),
            )?;
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
