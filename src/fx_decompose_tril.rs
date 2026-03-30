use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.tril` into
    /// arange + view + add(diagonal) + ge + where / bitwise_and,
    /// mirroring the Python Arm pass.
    FxDecomposeTrilPass, "FxDecomposeTril",
    matches: |target| target.contains("aten.tril")
        || target.contains("aten_tril")
        || target.contains("aten.tril_copy")
        || target.contains("aten_tril_copy"),
    transform: |py, graph, gm, node, debug| decompose_tril(py, &graph, gm, node, debug).map(|_| ())
);

fn copied_meta<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    clear_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let copy_mod = py.import_bound("copy")?;
    let meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    if clear_qparams {
        meta.set_item("input_qparams", PyDict::new_bound(py))?;
        meta.set_item("output_qparams", PyDict::new_bound(py))?;
    }
    Ok(meta)
}

fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    helpers.call_method1("refresh_node_fake_tensor", (node,))?;
    Ok(())
}

fn insert_call_before_with_kwargs<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
    meta_src: &Bound<'py, PyAny>,
    clear_qparams: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = graph.call_method1("call_function", (target, args, kwargs))?;
    let meta_copy = copied_meta(py, meta_src, clear_qparams)?;
    new_node.setattr("meta", &meta_copy)?;
    let _ = refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn zero_scalar_value_and_kwargs<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    input_tensor: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let builtins = py.import_bound("builtins")?;
    let kwargs = PyDict::new_bound(py);

    let meta = node.getattr("meta")?;
    let output_qparams = meta.call_method1("get", ("output_qparams",))?;
    let is_quantized = !output_qparams.is_none()
        && output_qparams
            .downcast::<PyDict>()
            .map(|d| !d.is_empty())
            .unwrap_or(false);

    if is_quantized {
        if let Ok(qparams) = output_qparams.call_method1("get", (0,)) {
            if !qparams.is_none() {
                let zp: i64 = qparams
                    .getattr("zp")
                    .or_else(|_| qparams.getattr("zero_point"))
                    .and_then(|v| v.extract())
                    .unwrap_or(0);
                let qmin: i64 = qparams
                    .getattr("qmin")
                    .and_then(|v| v.extract())
                    .unwrap_or(-128);
                let qmax: i64 = qparams
                    .getattr("qmax")
                    .and_then(|v| v.extract())
                    .unwrap_or(127);
                let output_dtype = qparams.getattr("dtype")?;
                let clamped = zp.max(qmin).min(qmax);
                if debug {
                    eprintln!(
                        "[DecomposeTril] quantizing zero fill -> {} (zp={}, qmin={}, qmax={})",
                        clamped, zp, qmin, qmax
                    );
                }
                kwargs.set_item("dtype", output_dtype)?;
                return Ok((builtins.call_method1("int", (clamped,))?, kwargs));
            }
        }
    }

    let input_dtype = input_tensor
        .getattr("meta")?
        .get_item("val")?
        .getattr("dtype")?;
    kwargs.set_item("dtype", input_dtype)?;
    Ok((builtins.call_method1("int", (0,))?, kwargs))
}

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

    let input_tensor = &args[0];
    if !FxGraphModule::is_node(input_tensor) {
        return Ok(false);
    }

    let diagonal = match args.get(1) {
        Some(arg) => arg.extract::<i64>().unwrap_or(0),
        None => 0,
    };

    let input_shape = FxGraphModule::node_shape(input_tensor)?;
    if input_shape.len() < 2 {
        return Ok(false);
    }
    let rows = input_shape[input_shape.len() - 2];
    let cols = input_shape[input_shape.len() - 1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeTril] {} diagonal={} input_shape={:?}",
            name, diagonal, input_shape
        );
    }

    let torch = py.import_bound("torch")?;
    let float32_dtype = torch.getattr("float32")?;

    let arange_op = resolve_aten_op(py, "arange.start_step")?;
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let add_scalar_op = resolve_aten_op(py, "add.Scalar")?;
    let ge_tensor_op = resolve_aten_op(py, "ge.Tensor")?;
    let where_op = resolve_aten_op(py, "where.self")?;
    let scalar_tensor_op = resolve_aten_op(py, "scalar_tensor.default")?;
    let bitwise_and_op = resolve_aten_op(py, "bitwise_and.Tensor")?;

    let arange_kwargs = PyDict::new_bound(py);
    arange_kwargs.set_item("dtype", float32_dtype)?;

    let row_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            rows.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let row_1d = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &arange_op,
        row_args.as_ref(),
        &arange_kwargs,
        node,
        true,
    )?;
    let row_shape = PyList::new_bound(py, [rows, 1i64]);
    let row_view_args = PyTuple::new_bound(py, &[row_1d.as_ref(), row_shape.as_ref()]);
    let mut row = insert_call_before(py, graph, node, &view_op, row_view_args.as_ref(), &row_1d)?;

    if diagonal != 0 {
        let add_args = PyTuple::new_bound(
            py,
            &[row.as_ref(), diagonal.into_py(py).into_bound(py).as_ref()],
        );
        row = insert_call_before(py, graph, node, &add_scalar_op, add_args.as_ref(), &row)?;
    }

    let col_args = PyTuple::new_bound(
        py,
        &[
            0i64.into_py(py).into_bound(py).as_ref(),
            cols.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let col_1d = insert_call_before_with_kwargs(
        py,
        graph,
        node,
        &arange_op,
        col_args.as_ref(),
        &arange_kwargs,
        node,
        true,
    )?;
    let col_shape = PyList::new_bound(py, [1i64, cols]);
    let col_view_args = PyTuple::new_bound(py, &[col_1d.as_ref(), col_shape.as_ref()]);
    let col = insert_call_before(py, graph, node, &view_op, col_view_args.as_ref(), &col_1d)?;

    let mask_args = PyTuple::new_bound(py, &[row.as_ref(), col.as_ref()]);
    let mask = insert_call_before(py, graph, node, &ge_tensor_op, mask_args.as_ref(), node)?;

    let input_dtype = FxGraphModule::node_dtype_str(input_tensor)?;
    let result_node = if input_dtype.contains("bool") {
        let bitwise_and_args = PyTuple::new_bound(py, &[input_tensor.as_ref(), mask.as_ref()]);
        insert_call_before(
            py,
            graph,
            node,
            &bitwise_and_op,
            bitwise_and_args.as_ref(),
            node,
        )?
    } else {
        let (zero_value, zero_kwargs) =
            zero_scalar_value_and_kwargs(py, node, input_tensor, debug)?;
        let zero_args = PyTuple::new_bound(py, &[zero_value.as_ref()]);
        let zero_node = insert_call_before_with_kwargs(
            py,
            graph,
            node,
            &scalar_tensor_op,
            zero_args.as_ref(),
            &zero_kwargs,
            node,
            true,
        )?;
        let where_args = PyTuple::new_bound(
            py,
            &[mask.as_ref(), input_tensor.as_ref(), zero_node.as_ref()],
        );
        insert_call_before(py, graph, node, &where_op, where_args.as_ref(), node)?
    };

    let final_meta = copied_meta(py, node, false)?;
    result_node.setattr("meta", &final_meta)?;
    let _ = refresh_fake_tensor(py, &result_node);

    FxGraphModule::replace_all_uses_with(node, &result_node)?;
    gm.erase_node(node)?;
    Ok(true)
}
