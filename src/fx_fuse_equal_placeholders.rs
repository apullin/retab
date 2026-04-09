use pyo3::prelude::*;

use crate::fx_ep_effects::{
    FxDeleteConstantPlaceholderEffect, FxEpApplyContext, FxEpEffect, FxEpEffectBatch,
};
use crate::fx_fuse_equal_placeholders_ir::find_duplicate_placeholder_pairs;
use crate::fx_graph::FxGraphModule;
use crate::fx_ir::IrGraph;
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

        let mut ir = IrGraph::from_fx_with_tensors(gm.gm_ref(), Some(&ep))?;
        let duplicate_pairs = find_duplicate_placeholder_pairs(&ir);
        if duplicate_pairs.is_empty() {
            return Ok(false);
        }

        let mut effects = FxEpEffectBatch::default();
        let mut modified = false;

        for (dup, canonical) in duplicate_pairs {
            if debug {
                eprintln!(
                    "[FuseEqualPlaceholders] merging '{}' into '{}'",
                    ir.node(dup).name,
                    ir.node(canonical).name
                );
            }
            ir.replace_all_uses(dup, canonical);
            effects.push(FxEpEffect::DeleteConstantPlaceholder(
                FxDeleteConstantPlaceholderEffect {
                    placeholder_name: ir.node(dup).name.clone(),
                    allow_missing: true,
                },
            ));
            modified = true;
        }

        if modified {
            ir.write_back_with_surgery(py, gm.gm_ref())?;
            let ctx = FxEpApplyContext::new(gm.gm_ref(), &ep)?;
            let _ = effects.apply(&ctx)?;
            let graph = gm.gm_ref().getattr("graph")?;
            graph.call_method0("eliminate_dead_code")?;
            graph.call_method0("lint")?;
            gm.recompile()?;
        }
        Ok(modified)
    }
}
