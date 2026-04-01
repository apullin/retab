use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// Decompose `aten.addmm` into mm + mul + add.
    /// Mirrors Python `DecomposeAddmmPass`.
    FxDecomposeAddmmPass, "FxDecomposeAddmm",
    matches: |target, _node| target.contains("addmm"),
    transform: |py, graph, gm, node, debug| decompose_addmm(py, &graph, gm, node, debug)
);

fn decompose_addmm<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 3 {
        return Ok(());
    }

    let input = &args[0];
    let mat1 = &args[1];
    let mat2 = &args[2];

    let kwargs = node.getattr("kwargs")?;

    let beta: f64 = if args.len() > 3 {
        args[3].extract().unwrap_or(1.0)
    } else {
        match kwargs.call_method1("get", ("beta", 1.0f64)) {
            Ok(v) => v.extract().unwrap_or(1.0),
            Err(_) => 1.0,
        }
    };

    let alpha: f64 = if args.len() > 4 {
        args[4].extract().unwrap_or(1.0)
    } else {
        match kwargs.call_method1("get", ("alpha", 1.0f64)) {
            Ok(v) => v.extract().unwrap_or(1.0),
            Err(_) => 1.0,
        }
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeAddmm] decomposing: {} (beta={}, alpha={})",
            name, beta, alpha
        );
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    let mm_result = d.mm(mat1, mat2)?;

    let scaled_mm = if (alpha - 1.0).abs() > 1e-10 {
        let rank = FxGraphModule::node_shape(&mm_result)
            .unwrap_or_else(|_| vec![1])
            .len();
        let alpha_node = d.full_scalar(rank, alpha)?;
        d.mul(&mm_result, &alpha_node)?
    } else {
        mm_result
    };

    let scaled_input = if (beta - 1.0).abs() > 1e-10 {
        let rank = FxGraphModule::node_shape(input)
            .unwrap_or_else(|_| vec![1])
            .len();
        let beta_node = d.full_scalar(rank, beta)?;
        d.mul(input, &beta_node)?
    } else {
        input.clone()
    };

    let result = d.add(&scaled_mm, &scaled_input)?;

    d.replace(&result)?;
    Ok(())
}
