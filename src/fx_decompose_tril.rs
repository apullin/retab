use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::{arg_extract, FxGraphModule};

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.tril` into
    /// arange + view + add(diagonal) + ge + where / bitwise_and,
    /// mirroring the Python Arm pass.
    FxDecomposeTrilPass, "FxDecomposeTril",
    matches: |target, _node| target.contains("aten.tril")
        || target.contains("aten_tril")
        || target.contains("aten.tril_copy")
        || target.contains("aten_tril_copy"),
    transform: |py, graph, gm, node, debug| decompose_tril(py, &graph, gm, node, debug).map(|_| ())
);

fn decompose_tril<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(false);
    }

    let input = &args[0];
    if !FxGraphModule::is_node(input) {
        return Ok(false);
    }

    let diagonal: i64 = arg_extract(&args, 1, 0);

    let shape = FxGraphModule::node_shape(input)?;
    if shape.len() < 2 {
        return Ok(false);
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeTril] {} diagonal={} shape={:?}", name, diagonal, shape);
    }

    let d = FxDecomposer::new(py, graph, gm, node);
    let float32 = py.import_bound("torch")?.getattr("float32")?;

    // row indices: arange(0, rows) reshaped to [rows, 1]
    let mut row = d.view(&d.arange(0, rows, &float32)?, &[rows, 1])?;
    if diagonal != 0 {
        row = d.add_scalar(&row, diagonal)?;
    }

    // col indices: arange(0, cols) reshaped to [1, cols]
    let col = d.view(&d.arange(0, cols, &float32)?, &[1, cols])?;

    // mask: row >= col
    let mask = d.ge(&row, &col)?;

    let result = if FxGraphModule::node_dtype_str(input)?.contains("bool") {
        d.bitwise_and(input, &mask)?
    } else {
        let zero = d.scalar_tensor_quantized(node, input, debug)?;
        d.where_(&mask, input, &zero)?
    };

    d.replace(&result)
}
