use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `CastInt64BuffersToInt32Pass`.
///
/// Finds buffer placeholder nodes whose `meta["val"]` has dtype int64 and
/// casts them (both the FakeTensor in meta and the real tensor in the
/// ExportedProgram's state_dict) to int32. The values must fit in int32
/// range or the pass raises an error.
///
/// The Python version requires an `ExportedProgram` to look up buffer names
/// and mutate `state_dict`. We pass it through as an optional PyObject.
pub struct FxCastInt64ToInt32Pass {
    exported_program: Option<PyObject>,
}

impl FxCastInt64ToInt32Pass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self { exported_program }
    }
}

impl FxPass for FxCastInt64ToInt32Pass {
    fn name(&self) -> &str {
        "FxCastInt64ToInt32"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = FxGraphModule::debug();
        let mut modified = false;

        let ep = match &self.exported_program {
            Some(ep) => ep.bind(py).clone(),
            None => return Ok(false),
        };

        let torch = py.import_bound("torch")?;
        let int64_dtype = torch.getattr("int64")?;
        let int32_dtype = torch.getattr("int32")?;

        // Access the is_buffer helper
        let is_buffer_fn = py
            .import_bound("torch._export.utils")?
            .getattr("is_buffer")?;

        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            // Skip nodes with no users (dead code)
            let users = FxGraphModule::node_users(node)?;
            if users.is_empty() {
                continue;
            }

            let meta = node.getattr("meta")?;

            // Skip nodes without a "val" in meta
            let val = match meta.call_method1("get", ("val",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };

            // Check if it's a FakeTensor (has dtype attribute)
            let Ok(dtype) = val.getattr("dtype") else {
                continue;
            };

            // Only process int64 tensors
            if !dtype.eq(&int64_dtype)? {
                continue;
            }

            // Only process buffer nodes
            let is_buf: bool = is_buffer_fn.call1((&ep, node))?.extract()?;
            if !is_buf {
                continue;
            }

            // Get the buffer name from graph_signature.inputs_to_buffers
            let graph_sig = ep.getattr("graph_signature")?;
            let inputs_to_buffers = graph_sig.getattr("inputs_to_buffers")?;
            let node_name = FxGraphModule::node_name(node)?;
            let Ok(buffer_name_obj) = inputs_to_buffers.get_item(&node_name) else {
                continue;
            };
            let buffer_name: String = buffer_name_obj.extract()?;

            // Get the actual buffer tensor from state_dict
            let state_dict = ep.getattr("state_dict")?;
            let buffer = state_dict.get_item(&buffer_name)?;

            // Assert values are within int32 range
            let min_val: i64 = torch
                .call_method1("min", (&buffer,))?
                .call_method0("item")?
                .extract()?;
            let max_val: i64 = torch
                .call_method1("max", (&buffer,))?
                .call_method0("item")?
                .extract()?;

            let i32_min: i64 = -2_147_483_648;
            let i32_max: i64 = 2_147_483_647;

            if min_val < i32_min {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Node {} has value < {}",
                    node_name, i32_min
                )));
            }
            if max_val > i32_max {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Node {} has value > {}",
                    node_name, i32_max
                )));
            }

            if debug {
                eprintln!(
                    "[CastInt64] casting buffer '{}' from int64 to int32",
                    node_name
                );
            }

            // Cast the FakeTensor in meta["val"] to int32
            let new_val = val.call_method1("to", (&int32_dtype,))?;
            meta.set_item("val", &new_val)?;

            // Cast the real buffer in state_dict to int32
            let buffer_int32 = buffer.call_method1("to", (&int32_dtype,))?;
            state_dict.set_item(&buffer_name, &buffer_int32)?;

            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
