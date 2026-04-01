use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `RemoveNoopPass`.
///
/// Removes identity-like nodes from the graph: clone_dim_order, raw clone,
/// _to_dim_order_copy, alias_copy, copy, and detach_copy where input and
/// output dtypes match. These nodes become no-ops after earlier passes have
/// resolved memory formats and dtypes.
///
/// For `aten.copy.default`, the pass returns args[1] (the source) since
/// copy(dst, src) copies src into dst. For all others, args[0] is the
/// pass-through value.
pub struct FxRemoveNoopPass;

/// Target strings that are candidates for noop removal.
const NOOP_TARGETS: &[&str] = &[
    "aten.clone.default",
    "_clone_dim_order.default",
    "_to_dim_order_copy.default",
    "alias_copy.default",
    "copy.default",
    "detach_copy.default",
];

fn is_noop_target(target: &str) -> bool {
    NOOP_TARGETS.iter().any(|t| target.contains(t))
}

fn is_copy_target(target: &str) -> bool {
    // Specifically aten.copy.default (not alias_copy or detach_copy)
    target.contains("copy.default")
        && !target.contains("alias_copy")
        && !target.contains("detach_copy")
        && !target.contains("dim_order_copy")
}

impl FxPass for FxRemoveNoopPass {
    fn name(&self) -> &str {
        "FxRemoveNoop"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let debug = FxGraphModule::debug();
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        // Collect candidates first to avoid mutating during iteration
        let mut to_remove: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }

            let target = FxGraphModule::node_target_str(node)?;
            if !is_noop_target(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.is_empty() {
                continue;
            }

            // Check dtype match: input dtype must equal output dtype
            // For _to_dim_order_copy, kwargs may contain a "dtype" override
            let input_node = &args[0];

            // Get input dtype
            let input_dtype = match Self::get_node_dtype(input_node) {
                Some(d) => d,
                None => continue,
            };

            // Get output dtype: check kwargs for explicit dtype, else use node's own dtype
            let output_dtype = {
                let kwargs = FxGraphModule::node_kwargs(node)?;
                let explicit: Option<String> =
                    kwargs.call_method1("get", ("dtype",)).ok().and_then(|v| {
                        if v.is_none() {
                            None
                        } else {
                            v.str().ok().map(|s| s.to_string())
                        }
                    });
                explicit.unwrap_or(input_dtype.clone())
            };

            if input_dtype != output_dtype {
                continue;
            }

            // Determine the replacement node
            let replacement = if is_copy_target(&target) {
                // aten.copy.default(dst, src) -> return src (args[1])
                if args.len() >= 2 && FxGraphModule::is_node(&args[1]) {
                    args[1].clone()
                } else {
                    continue;
                }
            } else {
                // All others: return args[0]
                if FxGraphModule::is_node(&args[0]) {
                    args[0].clone()
                } else {
                    continue;
                }
            };

            if debug {
                let node_name = FxGraphModule::node_name(node)?;
                eprintln!("[RemoveNoop] removing '{}' (target={})", node_name, target);
            }

            to_remove.push((node.clone(), replacement));
        }

        for (noop_node, replacement) in &to_remove {
            FxGraphModule::replace_all_uses_with(noop_node, replacement)?;
            gm.erase_node(noop_node)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

impl FxRemoveNoopPass {
    /// Extract dtype string from a node's meta["val"].dtype.
    fn get_node_dtype(node: &Bound<'_, PyAny>) -> Option<String> {
        let meta = node.getattr("meta").ok()?;
        let val = meta.call_method1("get", ("val",)).ok()?;
        if val.is_none() {
            return None;
        }
        let dtype = val.getattr("dtype").ok()?;
        Some(dtype.str().ok()?.to_string())
    }
}
