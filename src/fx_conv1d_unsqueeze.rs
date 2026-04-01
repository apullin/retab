use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;

fn is_conv1d(target: &str, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !(target.contains("convolution.default") || target.contains("conv1d.default")) {
        return Ok(false);
    }

    // For conv1d.default, the op is always 1D regardless of arg count
    // (optional args stride/padding/dilation/groups may be omitted).
    if target.contains("conv1d.default") {
        return Ok(true);
    }

    // For convolution.default, check stride length to identify 1D.
    let args = FxGraphModule::node_args(node)?;
    if let Some(stride) = args.get(3) {
        let py = node.py();
        let builtins = py.import_bound("builtins")?;
        let stride_len: usize = builtins
            .call_method1("len", (stride,))?
            .extract()
            .unwrap_or(0);
        Ok(stride_len == 1)
    } else {
        Ok(false)
    }
}

crate::fx_pass!(
    /// FX-graph-direct implementation of Arm's Conv1dUnsqueezePass.
    ///
    /// TOSA has no native 1D convolution. The reference Arm pass rewrites conv1d
    /// into:
    ///   view_copy(input[..., W] -> [..., W, 1])
    ///   view_copy(weight[..., KW] -> [..., KW, 1])
    ///   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
    ///   view_copy(output[..., W_out, 1] -> [..., W_out])
    ///
    /// This pass mirrors that shape legalization before `FxRewriteConvPass`.
    FxConv1dUnsqueezePass, "FxConv1dUnsqueeze",
    matches: |target, node| is_conv1d(&target, &node).unwrap_or(false),
    transform: |py, graph, gm, node, debug| {
        let _ = (&graph, &debug); // suppress unused warnings
        let helpers = py.import_bound("fx_helpers")?;
        helpers.call_method1("unsqueeze_conv1d_node", (gm.gm_ref(), node))?;
        Ok(())
    }
);
