use pyo3::prelude::*;
use std::collections::HashMap;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `FuseEqualPlaceholdersPass`.
///
/// Finds constant placeholder nodes (parameters and buffers) that reference
/// identical tensors (same dtype, shape, and data bytes) and merges them into
/// a single placeholder. This reduces memory usage when the same weight/buffer
/// appears multiple times in the graph (e.g. after decomposition passes).
///
/// Requires an `ExportedProgram` to resolve placeholder tensors and manage
/// the graph signature when deleting duplicates.
pub struct FxFuseEqualPlaceholdersPass {
    exported_program: Option<PyObject>,
}

impl FxFuseEqualPlaceholdersPass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self { exported_program }
    }
}

impl FxPass for FxFuseEqualPlaceholdersPass {
    fn name(&self) -> &str {
        "FxFuseEqualPlaceholders"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = FxGraphModule::debug();

        let ep = match &self.exported_program {
            Some(ep) => ep.bind(py).clone(),
            None => return Ok(false),
        };

        // Import helpers
        let arm_pass_utils = py
            .import_bound("executorch.backends.arm._passes.arm_pass_utils")
            .or_else(|_| {
                // Fallback: if full import path fails, the pass is a no-op
                Err(pyo3::exceptions::PyImportError::new_err(
                    "arm_pass_utils not available",
                ))
            });

        // If we can't import the helper module, skip gracefully
        let arm_pass_utils = match arm_pass_utils {
            Ok(m) => m,
            Err(_) => {
                if debug {
                    eprintln!("[FuseEqualPlaceholders] arm_pass_utils not available, skipping");
                }
                return Ok(false);
            }
        };

        let is_param_node_fn = arm_pass_utils.getattr("is_param_node")?;
        let get_param_tensor_fn = arm_pass_utils.getattr("get_param_tensor")?;

        let hashlib = py.import_bound("hashlib")?;
        let torch = py.import_bound("torch")?;
        let uint8_dtype = torch.getattr("uint8")?;

        let all_nodes = gm.nodes()?;
        let mut modified = false;

        // Build hash buckets: fingerprint -> list of (node, tensor)
        let mut hash_buckets: HashMap<String, Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>> =
            HashMap::new();

        for node in &all_nodes {
            let is_param: bool = match is_param_node_fn.call1((&ep, node)) {
                Ok(v) => v.extract().unwrap_or(false),
                Err(_) => false,
            };
            if !is_param {
                continue;
            }

            let tensor = match get_param_tensor_fn.call1((&ep, node)) {
                Ok(t) if !t.is_none() => t,
                _ => continue,
            };

            // Check for special dtype metadata (int48 tensors must not merge with int32)
            let meta = node.getattr("meta")?;
            let special_dtype: String = meta
                .call_method1("get", ("tosa_special_dtype",))?
                .str()
                .map(|s| s.to_string())
                .unwrap_or_default();

            // Build fingerprint: special_dtype + dtype + shape + SHA1(data)
            let dtype_str: String = tensor.getattr("dtype")?.str()?.extract()?;
            let shape_str: String = tensor.getattr("shape")?.repr()?.extract()?;

            // Flatten to uint8 bytes for hashing
            let t_cpu = tensor
                .call_method0("cpu")?
                .call_method0("contiguous")?
                .call_method0("flatten")?
                .call_method1("view", (&uint8_dtype,))?;
            let numpy_bytes = t_cpu.call_method0("numpy")?.call_method0("tobytes")?;

            let sha1 = hashlib.call_method1("sha1", (&numpy_bytes,))?;
            let hex_digest: String = sha1.call_method1("hexdigest", ())?.extract()?;

            let key = format!(
                "{}|{}|{}|{}",
                special_dtype, dtype_str, shape_str, hex_digest
            );
            hash_buckets
                .entry(key)
                .or_default()
                .push((node.clone(), tensor));
        }

        // Import fx_helpers once (its delete wrapper gracefully handles
        // get_attr nodes like TABLE constants that aren't registered in
        // the ExportedProgram's graph signature).
        let helpers = py.import_bound("fx_helpers")?;

        // For each bucket with duplicates, fuse them
        for nodes_tensors in hash_buckets.values() {
            if nodes_tensors.len() < 2 {
                continue;
            }

            // Keep the first node as the canonical one
            let (canonical, _) = &nodes_tensors[0];

            if debug {
                let canonical_name = FxGraphModule::node_name(canonical)?;
                eprintln!(
                    "[FuseEqualPlaceholders] merging {} duplicates into '{}'",
                    nodes_tensors.len() - 1,
                    canonical_name
                );
            }

            // Replace all duplicate nodes' uses with the canonical node
            for (dup_node, _) in &nodes_tensors[1..] {
                FxGraphModule::replace_all_uses_with(dup_node, canonical)?;

                let _ = helpers.call_method1("delete_constant_placeholder", (&ep, dup_node));
                modified = true;
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
