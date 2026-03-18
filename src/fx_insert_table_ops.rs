use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `InsertTableOpsPass`.
///
/// Finds unary math ops (`rsqrt`, `exp`, `log`, `sigmoid`, `tanh`) that have
/// both `input_qparams` and `output_qparams` in their node metadata, and
/// replaces them with a TOSA TABLE op backed by a precomputed 513-entry
/// lookup table.
///
/// The heavy matching logic lives in Rust; the actual table computation and
/// node creation are delegated to `fx_helpers.py` to avoid reimplementing
/// NumPy / FakeTensor bookkeeping.
pub struct FxInsertTableOpsPass;

/// Recognised unary targets that can be lowered to TABLE ops.
fn table_op_kind(target: &str) -> Option<&'static str> {
    if target.contains("rsqrt.default") {
        Some("rsqrt")
    } else if target.contains("exp.default") {
        Some("exp")
    } else if target.contains("log.default") {
        Some("log")
    } else if target.contains("sigmoid.default") {
        Some("sigmoid")
    } else if target.contains("tanh.default") {
        Some("tanh")
    } else {
        None
    }
}

impl FxPass for FxInsertTableOpsPass {
    fn name(&self) -> &str {
        "FxInsertTableOps"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        // Collect candidates first to avoid mutation during iteration.
        struct TableCandidate<'py> {
            node: Bound<'py, PyAny>,
            kind: &'static str,
        }
        let mut candidates: Vec<TableCandidate> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            let kind = match table_op_kind(&target) {
                Some(k) => k,
                None => continue,
            };

            // Only convert when the node has *both* input and output qparams
            // (i.e. it sits inside a quantised region).
            let meta = node.getattr("meta")?;

            let has_input_qp = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => {
                    let len: usize = py
                        .import_bound("builtins")?
                        .call_method1("len", (&v,))?
                        .extract()
                        .unwrap_or(0);
                    len > 0
                }
                _ => false,
            };

            let has_output_qp = match meta.call_method1("get", ("output_qparams",)) {
                Ok(v) if !v.is_none() => {
                    let len: usize = py
                        .import_bound("builtins")?
                        .call_method1("len", (&v,))?
                        .extract()
                        .unwrap_or(0);
                    len > 0
                }
                _ => false,
            };

            if has_input_qp && has_output_qp {
                candidates.push(TableCandidate {
                    node: node.clone(),
                    kind,
                });
            }
        }

        // Apply replacements.
        for cand in &candidates {
            // Delegate to Python helper which:
            //   1. Reads input_qparams[0] and output_qparams[0]
            //   2. Computes the 513-entry table
            //   3. Creates a const node + TABLE call_function node
            //   4. Replaces all uses of the original node with the TABLE node
            //   5. Erases the original node
            helpers.call_method1(
                "insert_table_op",
                (gm.gm_ref(), &cand.node, cand.kind),
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
    // Integration tests live in Python; Rust unit tests would need a live
    // Python FX graph which is non-trivial to set up.
}
