use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

pub struct FxRewriteIndexPutPass;

fn is_index_put(target: &str) -> bool {
    target.contains("index_put.default")
}

impl FxPass for FxRewriteIndexPutPass {
    fn name(&self) -> &str {
        "FxRewriteIndexPut"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let all_nodes = gm.nodes()?;
        let mut candidates = Vec::new();

        for node in &all_nodes {
            if FxGraphModule::node_op(node)? != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_index_put(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        for node in &candidates {
            helpers.call_method1("rewrite_index_put_to_tosa", (gm.gm_ref(), node))?;
        }

        gm.recompile()?;
        Ok(true)
    }
}
