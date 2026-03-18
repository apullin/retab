use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FuseConstantArgsPass`.
///
/// Finds call_function nodes where **all** input nodes are constant
/// placeholders (parameters, buffers, or lifted tensor constants). For each
/// such node, evaluates the op eagerly with the actual tensor data and
/// replaces the node with a single pre-computed constant placeholder.
///
/// This is the most expensive pass in the Python pipeline (~13s) because it
/// must resolve parameters from the ExportedProgram and execute ops. The Rust
/// side handles the iteration and filtering; the Python helper does the
/// actual evaluation and constant-placeholder creation (which requires
/// ExportedProgram mutation).
///
/// Skips certain TOSA ops (MATMUL, RESCALE, RESIZE, TABLE, TRANSPOSE) that
/// should not be folded, matching the Python implementation.
pub struct FxFuseConstantArgsPass {
    exported_program: Option<PyObject>,
}

impl FxFuseConstantArgsPass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self { exported_program }
    }
}

/// Targets that must not be fused (matching Python's skip list).
fn is_skip_target(target: &str) -> bool {
    target.contains("tosa.MATMUL")
        || target.contains("tosa.RESCALE")
        || target.contains("tosa.RESIZE")
        || target.contains("tosa.TABLE")
        || target.contains("tosa.TRANSPOSE")
}

impl FxPass for FxFuseConstantArgsPass {
    fn name(&self) -> &str {
        "FxFuseConstantArgs"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;

        // We need the ExportedProgram to check which nodes are constant
        // (params/buffers/lifted constants) and to create new constant
        // placeholders.
        let ep = match &self.exported_program {
            Some(ep) => ep.bind(py).clone(),
            None => return Ok(false),
        };

        let torch_export_utils = py.import_bound("torch._export.utils")?;

        let mut modified = false;
        // Nodes to try deleting after all fusions (dead-code cleanup).
        let mut stale_inputs: Vec<Bound<'_, PyAny>> = Vec::new();

        // We iterate in topological order. After fusing a node we recompile
        // so that chained constant ops can be resolved in the next iteration.
        // Use a simple fixed-point loop (matching the Python pass behavior).
        let mut made_progress = true;
        while made_progress {
            made_progress = false;
            let all_nodes = gm.nodes()?;

            for node in &all_nodes {
                let op = FxGraphModule::node_op(node)?;
                if op != "call_function" {
                    continue;
                }

                let target = FxGraphModule::node_target_str(node)?;
                if is_skip_target(&target) {
                    continue;
                }

                // Gather all input nodes via all_input_nodes.
                let input_nodes: Vec<Bound<'_, PyAny>> = node
                    .getattr("all_input_nodes")?
                    .iter()?
                    .filter_map(|n| n.ok())
                    .collect();

                if input_nodes.is_empty() {
                    continue;
                }

                // Check that every input is a constant placeholder.
                let mut all_constant = true;
                for inp in &input_nodes {
                    let is_param: bool = torch_export_utils
                        .call_method1("is_param", (&ep, inp))?
                        .extract()?;
                    let is_lifted: bool = torch_export_utils
                        .call_method1("is_lifted_tensor_constant", (&ep, inp))?
                        .extract()?;
                    let is_buffer: bool = torch_export_utils
                        .call_method1("is_buffer", (&ep, inp))?
                        .extract()?;
                    if !(is_param || is_lifted || is_buffer) {
                        all_constant = false;
                        break;
                    }
                }

                if !all_constant {
                    continue;
                }

                // Delegate the actual fusion to the Python helper. It:
                //   1. Resolves param tensors and dequantizes if input_qparams present
                //   2. Evaluates the op with those tensors
                //   3. Quantizes the result if output_qparams present
                //   4. Creates a new constant placeholder node
                //   5. Replaces all uses of the original node
                // Returns True if the node was fused, False if skipped (e.g. tensor grew).
                let did_fuse: bool = match helpers.call_method1(
                    "fuse_constant_node",
                    (gm.gm_ref(), &ep, node),
                ) {
                    Ok(result) => result.extract().unwrap_or(false),
                    Err(e) => {
                        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
                        if debug {
                            let name = FxGraphModule::node_name(node).unwrap_or_default();
                            eprintln!(
                                "[FuseConstantArgs] failed to fuse {}: {}",
                                name, e
                            );
                        }
                        false
                    }
                };

                if did_fuse {
                    stale_inputs.extend(input_nodes);
                    modified = true;
                    made_progress = true;
                    // Recompile after each fusion so chained constant ops
                    // become visible in the next iteration.
                    gm.recompile()?;
                    break; // Restart iteration with fresh node list
                }
            }
        }

        if modified {
            // Eliminate dead code and clean up unused constant placeholders.
            let graph = gm.gm_ref().getattr("graph")?;
            graph.call_method0("eliminate_dead_code")?;

            for inp in &stale_inputs {
                // Only delete if the node has no remaining users.
                let users = FxGraphModule::node_users(inp)?;
                if users.is_empty() {
                    let _ = helpers.call_method1(
                        "delete_constant_placeholder",
                        (&ep, inp),
                    );
                }
            }

            gm.recompile()?;
        }
        Ok(modified)
    }
}
