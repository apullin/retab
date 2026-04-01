use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.logit` into primitive ops.
    ///
    /// Mirrors Python `DecomposeLogitPass`:
    ///   logit(x, eps=None) ->
    ///     if eps is not None:
    ///       x = clamp(x, eps, 1 - eps)
    ///     neg_x = mul(x, -1.0)
    ///     denom = add(neg_x, 1.0)       # (1 - x)
    ///     frac = reciprocal(denom)       # 1 / (1 - x)
    ///     log_input = mul(x, frac)       # x / (1 - x)
    ///     result = log(log_input)
    ///
    /// Must run before fold_qdq so the decomposed ops get quantization handling.
    FxDecomposeLogitPass, "FxDecomposeLogit",
    matches: |target, _node| target.contains("logit"),
    transform: |py, graph, gm, node, debug| {
        decompose_logit(py, &graph, gm, node, debug)
    }
);

fn decompose_logit<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }

    let mut x = args[0].clone();

    // Extract eps from args[1] or kwargs
    let eps: Option<f64> = if args.len() > 1 && !args[1].is_none() {
        args[1].extract().ok()
    } else {
        let kwargs = node.getattr("kwargs")?;
        match kwargs.call_method1("get", ("eps",)) {
            Ok(v) if !v.is_none() => v.extract().ok(),
            _ => None,
        }
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeLogit] decomposing: {} (eps={:?})", name, eps);
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    let input_shape = FxGraphModule::node_shape(&x).unwrap_or_else(|_| vec![1]);
    let rank = input_shape.len();

    // If eps is not None: x = clamp(x, eps, 1 - eps)
    if let Some(eps_val) = eps {
        x = d.clamp(&x, eps_val, 1.0 - eps_val)?;
    }

    // neg_x = mul(x, -1.0)
    let neg_one = d.full_scalar(rank, -1.0)?;
    let neg_x = d.mul(&x, &neg_one)?;

    // denom = add(neg_x, 1.0)  => (1 - x)
    let one = d.full_scalar(rank, 1.0)?;
    let denom = d.add(&neg_x, &one)?;

    // frac = reciprocal(denom)  => 1 / (1 - x)
    let frac = d.reciprocal(&denom)?;

    // log_input = mul(x, frac)  => x / (1 - x)
    let log_input = d.mul(&x, &frac)?;

    // result = log(log_input)
    let result = d.log(&log_input)?;

    d.replace(&result)?;
    Ok(())
}
