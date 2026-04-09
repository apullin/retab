use pyo3::prelude::*;

use crate::fx_ep_effects::{
    FxCastInt64InputToInt32Effect, FxEpApplyContext, FxEpEffect, FxEpEffectBatch,
    FxEpInputKind,
};
use crate::fx_graph::{meta_val, FxGraphModule};
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

        let ep = match &self.exported_program {
            Some(ep) => ep.bind(py).clone(),
            None => return Ok(false),
        };

        let torch = py.import_bound("torch")?;
        let int64_dtype = torch.getattr("int64")?;
        // Access the is_buffer helper
        let is_buffer_fn = py
            .import_bound("torch._export.utils")?
            .getattr("is_buffer")?;

        let all_nodes = gm.nodes()?;
        let mut effects = FxEpEffectBatch::default();

        for node in &all_nodes {
            // Skip nodes with no users (dead code)
            let users = FxGraphModule::node_users(node)?;
            if users.is_empty() {
                continue;
            }

            let meta = node.getattr("meta")?;

            // Skip nodes without a "val" in meta
            let Some(val) = meta_val(&meta, "val") else { continue };

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
            let node_name = FxGraphModule::node_name(node)?;

            if debug {
                eprintln!("[CastInt64] planning int64->int32 cast for '{}'", node_name);
            }

            effects.push(FxEpEffect::CastInt64InputToInt32(
                FxCastInt64InputToInt32Effect {
                    placeholder_name: node_name,
                    input_kind: FxEpInputKind::Buffer,
                    target_name: None,
                    allow_missing: true,
                    update_placeholder_meta: true,
                },
            ));
        }

        if effects.effects.is_empty() {
            return Ok(false);
        }

        let ctx = FxEpApplyContext::new(gm.gm_ref(), &ep)?;
        let stats = effects.apply(&ctx)?;
        let modified = stats.applied > 0;

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
