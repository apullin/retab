use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// Mirror the Python Arm backend's non-unit-step slice_copy decomposition:
    /// unit-step slice + optional right pad + view + slice + view.
    FxDecomposeStridedSlicePass, "FxDecomposeStridedSlice",
    matches: |target, _node| {
        (target.contains("slice_copy.Tensor") || target.contains("slice.Tensor"))
            && !target.contains("slice_scatter")
    },
    transform: |py, graph, gm, node, debug| {
        decompose_strided_slice(py, &graph, gm, node, debug)?;
        Ok(())
    }
);

fn copied_meta<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let copy_mod = py.import_bound("copy")?;
    let meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    meta.set_item("input_qparams", PyDict::new_bound(py))?;
    meta.set_item("output_qparams", PyDict::new_bound(py))?;
    Ok(meta)
}

fn copy_first_output_qparams<'py>(
    py: Python<'py>,
    dst_meta: &Bound<'py, PyAny>,
    src_nodes: &[&Bound<'py, PyAny>],
) -> PyResult<()> {
    let copy_mod = py.import_bound("copy")?;
    for src_node in src_nodes {
        let src_meta = src_node.getattr("meta")?;
        let output_qparams = match src_meta.call_method1("get", ("output_qparams",)) {
            Ok(v) if !v.is_none() => v,
            _ => continue,
        };
        if output_qparams.len().unwrap_or(0) == 0 {
            continue;
        }
        let qp_copy = copy_mod.call_method1("copy", (&output_qparams,))?;
        dst_meta.set_item("output_qparams", qp_copy)?;
        return Ok(());
    }
    Ok(())
}

fn insert_call_before<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    meta_src: &Bound<'py, PyAny>,
    qparam_sources: &[&Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = match kwargs {
        Some(kwargs) => graph.call_method1("call_function", (target, args, kwargs))?,
        None => graph.call_method1("call_function", (target, args))?,
    };
    let meta_copy = copied_meta(py, meta_src)?;
    copy_first_output_qparams(py, &meta_copy, qparam_sources)?;
    new_node.setattr("meta", &meta_copy)?;
    let _ = fx_utils::refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn extract_optional_i64(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i64>> {
    match arg {
        Some(value) if value.is_none() => Ok(None),
        Some(value) => Ok(Some(value.extract()?)),
        None => Ok(None),
    }
}

fn fixup_start(start: Option<i64>, dim_size: i64) -> i64 {
    let mut s = start.unwrap_or(0);
    if s < 0 {
        s = s.rem_euclid(dim_size);
    }
    s.clamp(0, dim_size)
}

fn fixup_end(end: Option<i64>, dim_size: i64) -> i64 {
    let mut e = end.unwrap_or(dim_size);
    if e > dim_size {
        e = dim_size;
    }
    if e < 0 {
        e = e.rem_euclid(dim_size);
    }
    e.clamp(0, dim_size)
}

fn decompose_strided_slice<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 5 || !FxGraphModule::is_node(&args[0]) {
        return Ok(false);
    }

    let input = &args[0];
    let input_shape = FxGraphModule::node_shape(input)?;
    if input_shape.is_empty() {
        return Ok(false);
    }

    let rank = input_shape.len() as i64;
    let step: i64 = args[4].extract()?;
    if step <= 1 {
        return Ok(false);
    }

    let dim_raw: i64 = args[1].extract()?;
    let dim = dim_raw.rem_euclid(rank) as usize;
    let dim_size = input_shape[dim];
    if dim_size <= 0 {
        return Ok(false);
    }

    let start = fixup_start(extract_optional_i64(args.get(2))?, dim_size);
    let end = fixup_end(extract_optional_i64(args.get(3))?, dim_size);
    let span_len = end - start;
    if span_len <= 0 {
        return Ok(false);
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let cat_op = resolve_aten_op(py, "cat.default")?;
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    let dim_py = (dim as i64).into_py(py).into_bound(py);
    let start_py = start.into_py(py).into_bound(py);
    let end_py = end.into_py(py).into_bound(py);
    let one_py = 1i64.into_py(py).into_bound(py);
    let zero_py = 0i64.into_py(py).into_bound(py);

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeStridedSlice] {} dim={} start={} end={} step={} shape={:?}",
            name, dim, start, end, step, input_shape
        );
    }

    let unit_slice_args = PyTuple::new_bound(
        py,
        &[
            input.as_ref(),
            dim_py.as_ref(),
            start_py.as_ref(),
            end_py.as_ref(),
            one_py.as_ref(),
        ],
    );
    let qparam_sources = [node, input];
    let mut y = insert_call_before(
        py,
        graph,
        node,
        &slice_op,
        &unit_slice_args,
        None,
        node,
        &qparam_sources,
    )?;

    let u = (span_len + step - 1) / step;
    let pad_right = u * step - span_len;

    if pad_right > 0 {
        let pad_shape: Vec<i64> = input_shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| if axis == dim { pad_right } else { size })
            .collect();

        let input_meta = input.getattr("meta")?;
        let input_val = input_meta.call_method1("get", ("val",))?;
        let dtype = if !input_val.is_none() {
            input_val.getattr("dtype").ok()
        } else {
            None
        };

        let fill_value = match dtype.as_ref() {
            Some(dtype) if dtype.str()?.to_str()? == "torch.bool" => {
                false.into_py(py).into_bound(py)
            }
            _ => zero_py.clone(),
        };

        let pad_shape_py = PyTuple::new_bound(py, pad_shape.iter().copied());
        let full_args = PyTuple::new_bound(py, &[pad_shape_py.as_ref(), fill_value.as_ref()]);
        let full_kwargs = PyDict::new_bound(py);
        if let Some(dtype) = dtype {
            full_kwargs.set_item("dtype", dtype)?;
        }
        if let Ok(device) = input_val.getattr("device") {
            if !device.is_none() {
                full_kwargs.set_item("device", device)?;
            }
        }
        let z = insert_call_before(
            py,
            graph,
            node,
            &full_op,
            &full_args,
            Some(&full_kwargs),
            node,
            &qparam_sources,
        )?;

        let cat_inputs = PyList::new_bound(py, &[y.as_ref(), z.as_ref()]);
        let cat_args = PyTuple::new_bound(py, &[cat_inputs.as_ref(), dim_py.as_ref()]);
        y = insert_call_before(
            py,
            graph,
            node,
            &cat_op,
            &cat_args,
            None,
            node,
            &qparam_sources,
        )?;
    }

    let mut split_shape = input_shape.clone();
    split_shape[dim] = u;
    split_shape.insert(dim + 1, step);
    let split_shape_py = PyList::new_bound(py, split_shape.iter().copied());
    let split_view_args = PyTuple::new_bound(py, &[y.as_ref(), split_shape_py.as_ref()]);
    let y2 = insert_call_before(
        py,
        graph,
        node,
        &view_op,
        &split_view_args,
        None,
        node,
        &qparam_sources,
    )?;

    let inserted_dim_py = ((dim + 1) as i64).into_py(py).into_bound(py);
    let inner_slice_args = PyTuple::new_bound(
        py,
        &[
            y2.as_ref(),
            inserted_dim_py.as_ref(),
            zero_py.as_ref(),
            one_py.as_ref(),
            one_py.as_ref(),
        ],
    );
    let y3 = insert_call_before(
        py,
        graph,
        node,
        &slice_op,
        &inner_slice_args,
        None,
        node,
        &qparam_sources,
    )?;

    let mut out_shape = input_shape.clone();
    out_shape[dim] = u;
    let out_shape_py = PyList::new_bound(py, out_shape.iter().copied());
    let out_args = PyTuple::new_bound(py, &[y3.as_ref(), out_shape_py.as_ref()]);
    let result = insert_call_before(
        py,
        graph,
        node,
        &view_op,
        &out_args,
        None,
        node,
        &qparam_sources,
    )?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(true)
}
