use pyo3::prelude::*;

crate::fx_pass!(FxRewriteIndexPutPass, "FxRewriteIndexPut",
    matches: |target| target.contains("index_put.default"),
    transform: |py, graph, gm, node, debug| {
        let _ = (&graph, &debug); // suppress unused warnings
        let helpers = py.import_bound("fx_helpers")?;
        helpers.call_method1("rewrite_index_put_to_tosa", (gm.gm_ref(), node))?;
        Ok(())
    }
);
