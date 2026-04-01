use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct implementation of `DecomposeLinearPass`.
    ///
    /// Decomposes `aten.linear.default(x, weight, bias)` into:
    ///   1. Reshape input to 4D: [N, ..., Ci] -> [N*..., Ci, 1, 1]
    ///   2. Reshape weight to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
    ///   3. Conv2d with 1x1 kernel (inherits qparams from linear)
    ///   4. Reshape output back: [N*..., Co, 1, 1] -> [N, ..., Co]
    ///
    /// All graph surgery is performed in Rust via the `FxDecomposer` DSL.
    FxDecomposeLinearPass, "FxDecomposeLinear",
    matches: |target, _node| target.contains("linear.default"),
    transform: |py, graph, gm, node, debug| decompose_linear(py, &graph, gm, node, debug)
);

fn decompose_linear<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 2 {
        return Ok(());
    }

    let input_node = &args[0];
    let weights_node = &args[1];
    let bias_node: Option<&Bound<'py, PyAny>> = if args.len() > 2 && !args[2].is_none() {
        Some(&args[2])
    } else {
        None
    };

    let output_shape = FxGraphModule::node_shape(node)?;
    let input_shape = FxGraphModule::node_shape(input_node)?;
    let weights_shape = FxGraphModule::node_shape(weights_node)?;

    // Compute batch dimension: product of all dims except last.
    let batches: i64 = input_shape[..input_shape.len() - 1].iter().product();

    // input (..., Ci) -> (N, Ci, 1, 1)
    let input_reshaped_shape = [batches, input_shape[input_shape.len() - 1], 1, 1];
    // weights (Co, Ci) -> (Co, Ci, 1, 1)
    let weights_reshaped_shape = [weights_shape[0], weights_shape[1], 1, 1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeLinear] {} input={:?} weights={:?} output={:?} batches={}",
            name, input_shape, weights_shape, output_shape, batches
        );
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    // 1. Reshape input to 4D
    let input_reshaped = d.view(input_node, &input_reshaped_shape)?;

    // 2. Reshape weights to 4D
    let weights_reshaped = d.view(weights_node, &weights_reshaped_shape)?;

    // 3. Conv2d with 1x1 kernel (inherits qparams from linear)
    let conv = d.convolution(
        &input_reshaped,
        &weights_reshaped,
        bias_node,
        &[1, 1],   // stride
        &[0, 0],   // padding
        &[1, 1],   // dilation
        false,      // transposed
        &[0, 0],   // output_padding
        1,          // groups
    )?;

    // 4. Reshape output back to original shape
    let output = d.view(&conv, &output_shape)?;

    d.replace(&output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
