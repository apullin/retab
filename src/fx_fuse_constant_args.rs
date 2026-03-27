use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

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
    mode: FuseMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FuseMode {
    General,
    ZeroInputFullOnly,
}

impl FxFuseConstantArgsPass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self {
            exported_program,
            mode: FuseMode::General,
        }
    }

    pub fn new_zero_input_full_only(exported_program: Option<PyObject>) -> Self {
        Self {
            exported_program,
            mode: FuseMode::ZeroInputFullOnly,
        }
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

fn is_zero_input_full(target: &str, input_nodes: &[Bound<'_, PyAny>]) -> bool {
    input_nodes.is_empty() && target.contains("aten.full.default")
}

fn is_zero_input_range_like(target: &str, input_nodes: &[Bound<'_, PyAny>]) -> bool {
    // Range builders with no tensor inputs should become frozen buffers before
    // lowering. Leaving them as runtime ops leaks unsupported ATen nodes into
    // the TOSA graph even though their values are fully known at compile time.
    input_nodes.is_empty() && (target.contains("arange") || target.contains("linspace"))
}

fn materialize_zero_input_const(
    gm: &FxGraphModule,
    node: &Bound<'_, PyAny>,
    const_tensor: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = gm.py();
    let graph = gm.gm_ref().getattr("graph")?;
    let meta = node.getattr("meta")?;

    let node_name = FxGraphModule::node_name(node)?;
    let mut attr_name = format!("{}_fused_const", node_name);
    let mut counter = 0usize;
    while gm.gm_ref().hasattr(attr_name.as_str())? {
        counter += 1;
        attr_name = format!("{}_fused_const_{}", node_name, counter);
    }

    gm.gm_ref()
        .call_method1("register_buffer", (attr_name.as_str(), const_tensor))?;

    let insert_ctx = graph.call_method1("inserting_before", (node,))?;
    insert_ctx.call_method0("__enter__")?;
    let const_node_result = (|| -> PyResult<Bound<'_, PyAny>> {
        let const_node = graph.call_method1("get_attr", (attr_name.as_str(),))?;
        let const_meta = meta.call_method0("copy")?;
        const_meta.set_item("val", const_tensor)?;
        const_node.setattr("meta", &const_meta)?;
        Ok(const_node)
    })();
    insert_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    let const_node = const_node_result?;

    FxGraphModule::replace_all_uses_with(node, &const_node)?;
    gm.erase_node(node)?;
    Ok(true)
}

fn fold_zero_input_full(gm: &FxGraphModule, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = gm.py();
    let torch = py.import_bound("torch")?;

    let args = FxGraphModule::node_args(node)?;
    if args.len() < 2 {
        return Ok(false);
    }

    let shape = &args[0];
    let fill_value = &args[1];

    let meta = node.getattr("meta")?;
    let val = meta.call_method1("get", ("val",))?;

    // Parse, don't validate: branch-local zero-input FULL nodes may lose their
    // fake tensor metadata during control-flow rewrites. If `meta["val"]` is
    // absent, fall back to eager `torch.full` inference instead of crashing the
    // entire recursive pass stack on a bookkeeping hole.
    let mut const_tensor = if val.is_none() {
        torch.call_method1("full", (shape, fill_value))?
    } else {
        let dtype = val.getattr("dtype")?;
        let full_kwargs = PyDict::new_bound(py);
        full_kwargs.set_item("dtype", &dtype)?;
        torch.call_method("full", (shape, fill_value), Some(&full_kwargs))?
    };

    // Match the general constant-folding helper: if the FULL output is already
    // annotated with output_qparams, materialize the frozen buffer in that
    // quantized domain instead of preserving the original FP32 data.
    if let Ok(output_qparams) = meta.call_method1("get", ("output_qparams",)) {
        if !output_qparams.is_none() && output_qparams.len().unwrap_or(0) > 0 {
            if let Ok(values) = output_qparams.call_method0("values") {
                if let Some(first_qp) = values.iter()?.next() {
                    let first_qp = first_qp?;
                    const_tensor = first_qp.call_method1("quantize_value", (&const_tensor,))?;
                }
            }
        }
    }

    materialize_zero_input_const(gm, node, &const_tensor)
}

fn fold_zero_input_range_like(gm: &FxGraphModule, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = gm.py();
    let args = FxGraphModule::node_args(node)?;
    let kwargs = node.getattr("kwargs")?;
    let target = node.getattr("target")?;
    let args_tuple = PyTuple::new_bound(py, args.iter());

    let mut const_tensor = if let Ok(kwargs_dict) = kwargs.downcast::<PyDict>() {
        target.call(&args_tuple, Some(&kwargs_dict))?
    } else {
        target.call1(&args_tuple)?
    };

    let meta = node.getattr("meta")?;
    if let Ok(output_qparams) = meta.call_method1("get", ("output_qparams",)) {
        if !output_qparams.is_none() && output_qparams.len().unwrap_or(0) > 0 {
            if let Ok(values) = output_qparams.call_method0("values") {
                if let Some(first_qp) = values.iter()?.next() {
                    let first_qp = first_qp?;
                    const_tensor = first_qp.call_method1("quantize_value", (&const_tensor,))?;
                }
            }
        }
    }

    materialize_zero_input_const(gm, node, &const_tensor)
}

impl FxPass for FxFuseConstantArgsPass {
    fn name(&self) -> &str {
        match self.mode {
            FuseMode::General => "FxFuseConstantArgs",
            FuseMode::ZeroInputFullOnly => "FxFoldZeroInputFull",
        }
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

        // Fixed-point loop: fuse as many nodes as possible in each sweep, then
        // recompile once before restarting. This preserves chained constant
        // folding behavior while avoiding per-node `graph_module.recompile()`
        // costs on large graphs.
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

                if is_zero_input_full(&target, &input_nodes) {
                    if fold_zero_input_full(gm, node)? {
                        modified = true;
                        made_progress = true;
                    }
                    continue;
                }

                if is_zero_input_range_like(&target, &input_nodes) {
                    if fold_zero_input_range_like(gm, node)? {
                        modified = true;
                        made_progress = true;
                    }
                    continue;
                }

                if self.mode == FuseMode::ZeroInputFullOnly {
                    continue;
                }

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
                let did_fuse: bool =
                    match helpers.call_method1("fuse_constant_node", (gm.gm_ref(), &ep, node)) {
                        Ok(result) => result.extract().unwrap_or(false),
                        Err(e) => {
                            let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
                            if debug {
                                let name = FxGraphModule::node_name(node).unwrap_or_default();
                                eprintln!("[FuseConstantArgs] failed to fuse {}: {}", name, e);
                            }
                            false
                        }
                    };

                if did_fuse {
                    stale_inputs.extend(input_nodes);
                    modified = true;
                    made_progress = true;
                }
            }

            if made_progress {
                gm.recompile()?;
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
                    let _ = helpers.call_method1("delete_constant_placeholder", (&ep, inp));
                }
            }

            gm.recompile()?;
        }
        Ok(modified)
    }
}
