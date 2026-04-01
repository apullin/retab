use pyo3::prelude::*;

crate::fx_pass!(
    /// FX-graph-direct implementation of `DecomposeLinearPass`.
    ///
    /// Decomposes `aten.linear.default(x, weight, bias)` into:
    ///   1. Reshape input to 4D: [N, ..., Ci] -> [N*..., Ci, 1, 1]
    ///   2. Reshape weight to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
    ///   3. Conv2d with 1x1 kernel (inherits qparams from linear)
    ///   4. Reshape output back: [N*..., Co, 1, 1] -> [N, ..., Co]
    ///
    /// The graph surgery (node creation with proper FakeTensor metadata) is
    /// delegated to `fx_helpers.decompose_linear_node()`.
    FxDecomposeLinearPass, "FxDecomposeLinear",
    matches: |target, _node| target.contains("linear.default"),
    transform: |py, graph, gm, node, debug| {
        let _ = (&graph, &debug); // suppress unused warnings
        let helpers = py.import_bound("fx_helpers")?;
        helpers.call_method1("decompose_linear_node", (gm.gm_ref(), node))?;
        Ok(())
    }
);

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
