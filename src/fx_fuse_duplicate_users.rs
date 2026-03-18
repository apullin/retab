use pyo3::prelude::*;
use std::collections::HashMap;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FuseDuplicateUsersPass`.
///
/// The LayerNorm decomposition (and others) can produce duplicate nodes —
/// two `call_function` nodes with the same target and identical argument
/// lists.  This pass deduplicates them: if two nodes have the same
/// (target, args, kwargs) signature, all uses of the second are replaced
/// with the first and the second is erased.
///
/// The pass iterates until convergence because fusing one pair of
/// duplicates can expose further duplicates.
pub struct FxFuseDuplicateUsersPass;

/// Build a hashable signature string for a `call_function` node.
///
/// Two nodes are considered duplicates iff they share the same target and
/// their positional arguments match element-wise (Node args by name,
/// literal args by value).  kwargs are included in the key as well.
fn node_signature(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let op = FxGraphModule::node_op(node)?;
    if op != "call_function" {
        return Ok(None);
    }

    let target = FxGraphModule::node_target_str(node)?;
    let args = FxGraphModule::node_args(node)?;

    let mut parts: Vec<String> = Vec::with_capacity(args.len() + 2);
    parts.push(target);

    for arg in &args {
        if FxGraphModule::is_node(arg) {
            let name = FxGraphModule::node_name(arg)?;
            parts.push(format!("N:{}", name));
        } else {
            // For lists/tuples inside args, repr them so order matters.
            let repr = arg.repr()?.extract::<String>()?;
            parts.push(format!("V:{}", repr));
        }
    }

    // Include kwargs in the signature for completeness.
    let kwargs = FxGraphModule::node_kwargs(node)?;
    let kwargs_repr = kwargs.repr()?.extract::<String>()?;
    parts.push(format!("KW:{}", kwargs_repr));

    // Suppress unused variable warning — py is needed for the trait bound
    let _ = py;

    Ok(Some(parts.join("|")))
}

impl FxPass for FxFuseDuplicateUsersPass {
    fn name(&self) -> &str {
        "FxFuseDuplicateUsers"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut any_modified = false;

        // Iterate until convergence — fusing one pair may expose more.
        loop {
            let mut modified_this_round = false;

            let all_nodes = gm.nodes()?;

            // Map: signature -> first node with that signature.
            let mut seen: HashMap<String, Bound<'_, PyAny>> = HashMap::new();

            // Collect (duplicate_node, canonical_node) pairs to process.
            let mut to_fuse: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = Vec::new();

            for node in &all_nodes {
                let sig = match node_signature(py, node)? {
                    Some(s) => s,
                    None => continue,
                };

                if let Some(canonical) = seen.get(&sig) {
                    to_fuse.push((node.clone(), canonical.clone()));
                } else {
                    seen.insert(sig, node.clone());
                }
            }

            for (dup, canonical) in &to_fuse {
                FxGraphModule::replace_all_uses_with(dup, canonical)?;
                gm.erase_node(dup)?;
                modified_this_round = true;
            }

            if modified_this_round {
                any_modified = true;
            } else {
                break;
            }
        }

        if any_modified {
            gm.recompile()?;
        }
        Ok(any_modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
