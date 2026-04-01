use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_utils;

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.mean.dim` into
    /// `sum(keepdim=True) + mul(1/count) + view_copy(output_shape)`.
    ///
    /// Always uses keepdim=True for the sum to keep tensors at input rank,
    /// then reshapes to the target output shape.  For 4D inputs with spatial
    /// dim reductions, uses avg_pool2d (matching Python DecomposeMeanDimPass).
    FxDecomposeMeanPass, "FxDecomposeMean",
    matches: |target, _node| target.contains("mean.dim") || target.contains("mean.default"),
    transform: |py, graph, gm, node, debug| decompose_mean(py, &graph, gm, node, debug)
);

fn get_qparam_field<'py, T: for<'a> FromPyObject<'a>>(
    qparam: &Bound<'py, PyAny>,
    attr: &str,
    fallback_key: Option<&str>,
) -> Option<T> {
    if let Ok(value) = qparam.getattr(attr) {
        if let Ok(parsed) = value.extract::<T>() {
            return Some(parsed);
        }
    }
    if let Some(key) = fallback_key {
        if let Ok(value) = qparam.call_method1("get", (key,)) {
            if let Ok(parsed) = value.extract::<T>() {
                return Some(parsed);
            }
        }
    }
    if let Ok(value) = qparam.call_method1("get", (attr,)) {
        if let Ok(parsed) = value.extract::<T>() {
            return Some(parsed);
        }
    }
    None
}

fn get_qparam_dtype<'py>(qparam: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    qparam
        .getattr("dtype")
        .ok()
        .or_else(|| qparam.call_method1("get", ("dtype",)).ok())
}

fn first_meta_qparam<'py>(meta: &Bound<'py, PyAny>, key: &str) -> Option<Bound<'py, PyAny>> {
    let qparams = meta.call_method1("get", (key,)).ok()?;
    if qparams.is_none() {
        return None;
    }
    qparams.get_item(0).ok()
}

fn make_qparam<'py>(
    py: Python<'py>,
    scale: f64,
    zero_point: i64,
    qmin: i64,
    qmax: i64,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")?
        .call1((scale, zero_point, qmin, qmax, dtype))
}

fn set_qparam_map<'py>(
    py: Python<'py>,
    meta: &Bound<'py, PyAny>,
    key: &str,
    entries: &[(usize, Bound<'py, PyAny>)],
) -> PyResult<()> {
    let qparams = PyDict::new_bound(py);
    for (index, qparam) in entries {
        qparams.set_item(*index, qparam)?;
    }
    meta.set_item(key, qparams)?;
    Ok(())
}

fn insert_call_before_with_kwargs<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: &Bound<'py, PyDict>,
    meta_src: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = graph.call_method1("call_function", (target, args, kwargs))?;
    fx_utils::copy_meta(py, meta_src, &new_node)?;
    // Clear qparams for intermediate node
    let meta_copy = new_node.getattr("meta")?;
    meta_copy.set_item("input_qparams", PyDict::new_bound(py))?;
    meta_copy.set_item("output_qparams", PyDict::new_bound(py))?;
    fx_utils::refresh_fake_tensor(py, &new_node)?;
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn insert_call_before_preserving_meta<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    meta_src: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = graph.call_method1("call_function", (target, args))?;
    fx_utils::copy_meta_and_refresh(py, meta_src, &new_node)?;
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

fn resolve_edge_quantized_op<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let parts: Vec<&str> = name.split('.').collect();
    let op_name = parts[0];
    let overload = if parts.len() > 1 { parts[1] } else { "default" };

    py.import_bound("executorch.exir.dialects._ops")?
        .getattr("ops")?
        .getattr("edge")?
        .getattr("quantized_decomposed")?
        .getattr(op_name)?
        .getattr(overload)
}

fn maybe_insert_q_dq_after<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    op: &Bound<'py, PyAny>,
    meta_src: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let input_nodes = op.getattr("all_input_nodes")?;
    let mut input_iter = input_nodes.iter()?;
    let Some(first_input) = input_iter.next() else {
        return Ok(op.clone());
    };
    let input_node = first_input?;
    if input_iter.next().is_some() {
        return Ok(op.clone());
    }

    let input_target = FxGraphModule::node_target_str(&input_node).unwrap_or_default();
    let (quant_op, dequant_op) = if input_target.contains("dequantize_per_tensor.default") {
        (
            resolve_edge_quantized_op(py, "quantize_per_tensor.default")?,
            resolve_edge_quantized_op(py, "dequantize_per_tensor.default")?,
        )
    } else if input_target.contains("dequantize_per_channel.default") {
        (
            resolve_edge_quantized_op(py, "quantize_per_channel.default")?,
            resolve_edge_quantized_op(py, "dequantize_per_channel.default")?,
        )
    } else {
        return Ok(op.clone());
    };

    let input_args = FxGraphModule::node_args(&input_node)?;
    if input_args.len() < 2 {
        return Ok(op.clone());
    }

    let mut quant_args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(input_args.len());
    quant_args.push(op.clone());
    quant_args.extend(input_args.iter().skip(1).cloned());
    let quant_args_tuple = PyTuple::new_bound(py, quant_args.iter().map(|arg| arg.as_ref()));
    let quant_node = insert_call_before_preserving_meta(
        py,
        graph,
        meta_src,
        &quant_op,
        &quant_args_tuple,
        meta_src,
    )?;

    let mut dequant_args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(input_args.len());
    dequant_args.push(quant_node.clone());
    dequant_args.extend(input_args.iter().skip(1).cloned());
    let dequant_args_tuple = PyTuple::new_bound(py, dequant_args.iter().map(|arg| arg.as_ref()));
    insert_call_before_preserving_meta(
        py,
        graph,
        meta_src,
        &dequant_op,
        &dequant_args_tuple,
        meta_src,
    )
}

fn annotate_quantized_mean<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    sum_node: &Bound<'py, PyAny>,
    full_node: &Bound<'py, PyAny>,
    mul_node: &Bound<'py, PyAny>,
    count: f64,
) -> PyResult<()> {
    let node_meta = node.getattr("meta")?;
    let Some(input_qp) = first_meta_qparam(&node_meta, "input_qparams") else {
        return Ok(());
    };
    let Some(output_qp) = first_meta_qparam(&node_meta, "output_qparams") else {
        return Ok(());
    };

    let input_scale = get_qparam_field::<f64>(&input_qp, "scale", None).unwrap_or(1.0);
    let input_zp = get_qparam_field::<i64>(&input_qp, "zp", Some("zero_point")).unwrap_or(0);
    let qmin = get_qparam_field::<i64>(&input_qp, "qmin", None).unwrap_or(-128);
    let qmax = get_qparam_field::<i64>(&input_qp, "qmax", None).unwrap_or(127);
    let Some(dtype) = get_qparam_dtype(&input_qp) else {
        return Ok(());
    };

    let sum_qp = make_qparam(py, input_scale * count, input_zp, qmin, qmax, &dtype)?;
    let full_qp = make_qparam(py, 1.0 / (count * qmax as f64), 0, qmin, qmax, &dtype)?;

    let sum_meta = sum_node.getattr("meta")?;
    set_qparam_map(py, &sum_meta, "input_qparams", &[(0, input_qp.clone())])?;
    set_qparam_map(py, &sum_meta, "output_qparams", &[(0, sum_qp.clone())])?;

    let full_meta = full_node.getattr("meta")?;
    set_qparam_map(py, &full_meta, "output_qparams", &[(0, full_qp.clone())])?;

    let mul_meta = mul_node.getattr("meta")?;
    set_qparam_map(
        py,
        &mul_meta,
        "input_qparams",
        &[(0, sum_qp.clone()), (1, full_qp)],
    )?;
    set_qparam_map(py, &mul_meta, "output_qparams", &[(0, output_qp)])?;
    Ok(())
}

fn decompose_mean<'py>(
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
    let x = &args[0];

    let input_shape = FxGraphModule::node_shape(x).unwrap_or_default();
    let input_rank = input_shape.len() as i64;
    let output_shape = FxGraphModule::node_shape(node).unwrap_or_else(|_| vec![1]);

    // Extract dims
    let dims: Vec<i64> = if args.len() > 1 {
        let dims_obj = &args[1];
        if dims_obj.is_none() {
            (0..input_rank).collect()
        } else {
            match dims_obj.extract::<Vec<i64>>() {
                Ok(d) => d,
                Err(_) => match dims_obj.extract::<i64>() {
                    Ok(d) => vec![d],
                    Err(_) => (0..input_rank).collect(),
                },
            }
        }
    } else {
        (0..input_rank).collect()
    };

    // Normalize negative dims
    let dims: Vec<i64> = dims
        .into_iter()
        .map(|d| if d < 0 { d + input_rank } else { d })
        .collect();

    // Filter out dims where input_shape[dim] == 1 (matches Python)
    let dims: Vec<i64> = dims
        .into_iter()
        .filter(|&d| input_shape.get(d as usize).copied().unwrap_or(1) != 1)
        .collect();

    let keepdim: bool = if args.len() > 2 {
        args[2].extract().unwrap_or(false)
    } else {
        false
    };

    if dims.is_empty() {
        FxGraphModule::replace_all_uses_with(node, x)?;
        gm.erase_node(node)?;
        return Ok(());
    }

    // Important semantic fast path:
    // for 4D spatial reductions, the Python Arm pass lowers mean via avg_pool2d,
    // not via sum * (1/count). Matching that here avoids a fragile quantized
    // reduce/multiply sequence and keeps direct lowering aligned with the
    // backend path we are trying to reproduce.
    if input_shape.len() == 4 && dims.iter().all(|&d| d == 2 || d == 3) {
        let avgpool_op = resolve_aten_op(py, "avg_pool2d.default")?;
        let kernel = match dims.as_slice() {
            [2, 3] | [3, 2] => vec![input_shape[2], input_shape[3]],
            [2] => vec![input_shape[2], 1],
            [3] => vec![1, input_shape[3]],
            _ => Vec::new(),
        };
        if !kernel.is_empty() {
            let stride = vec![1i64, 1];
            let kernel_py = PyList::new_bound(py, kernel.iter().copied());
            let stride_py = PyList::new_bound(py, stride.iter().copied());
            let pool_args =
                PyTuple::new_bound(py, &[x.as_ref(), kernel_py.as_ref(), stride_py.as_ref()]);
            let mut result =
                insert_call_before_preserving_meta(py, graph, node, &avgpool_op, &pool_args, node)?;
            result = maybe_insert_q_dq_after(py, graph, &result, node)?;

            if !keepdim {
                // avg_pool2d lowers through NHWC. Before collapsing the trailing
                // singleton spatial dim, rotate the tensor back into the
                // logical NCHW view so view_copy preserves the channel-major
                // layout expected by mean(dim=[2] / dim=[3]).
                let permute_op = resolve_aten_op(py, "permute_copy.default")?;
                let perm = PyList::new_bound(py, [0i64, 3, 1, 2]);
                let permute_args = PyTuple::new_bound(py, &[result.as_ref(), perm.as_ref()]);
                result = insert_call_before_preserving_meta(
                    py, graph, node, &permute_op, &permute_args, node,
                )?;

                let view_op = resolve_aten_op(py, "view_copy.default")?;
                let shape_list = PyList::new_bound(py, output_shape.iter().copied());
                let view_args = PyTuple::new_bound(py, &[result.as_ref(), shape_list.as_ref()]);
                result = insert_call_before_preserving_meta(
                    py, graph, node, &view_op, &view_args, node,
                )?;
            }

            FxGraphModule::replace_all_uses_with(node, &result)?;
            gm.erase_node(node)?;
            return Ok(());
        }
    }

    // Compute count = product of dim sizes
    let count: f64 = dims
        .iter()
        .map(|&d| input_shape.get(d as usize).copied().unwrap_or(1) as f64)
        .product();
    if count == 0.0 {
        return Ok(());
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeMean] {} dims={:?} keepdim={} count={} shape={:?}->{:?}",
            name, dims, keepdim, count, input_shape, output_shape
        );
    }

    let sum_op = resolve_aten_op(py, "sum.dim_IntList")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // sum(x, dims, keepdim=True) — always keepdim=True to preserve rank
    let dims_py = pyo3::types::PyList::new_bound(py, dims.iter().copied());
    let keepdim_true = true.into_py(py).into_bound(py);
    let sum_args = PyTuple::new_bound(py, &[x.as_ref(), dims_py.as_ref(), keepdim_true.as_ref()]);
    let sum_node = insert_call_before(py, graph, node, &sum_op, sum_args.as_ref(), node)?;

    // full([1]*input_rank, 1/count, dtype=node.dtype)
    let ones_shape = PyTuple::new_bound(py, vec![1i64; input_shape.len()].iter().copied());
    let builtins = py.import_bound("builtins")?;
    let inv_count = builtins.call_method1("float", (1.0 / count,))?;
    let full_args = PyTuple::new_bound(py, &[ones_shape.as_ref(), inv_count.as_ref()]);
    let full_kwargs = PyDict::new_bound(py);
    let result_dtype = node.getattr("meta")?.get_item("val")?.getattr("dtype")?;
    full_kwargs.set_item("dtype", result_dtype)?;
    if let Ok(device) = x.getattr("meta")?.get_item("val")?.getattr("device") {
        if !device.is_none() {
            full_kwargs.set_item("device", device)?;
        }
    }
    let count_node =
        insert_call_before_with_kwargs(py, graph, node, &full_op, &full_args, &full_kwargs, node)?;

    // mul(sum, 1/count)
    let mul_args = PyTuple::new_bound(py, &[sum_node.as_ref(), count_node.as_ref()]);
    let mut result = insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?;

    // If keepdim=False, reshape to target output shape
    if !keepdim {
        let view_op = resolve_aten_op(py, "view_copy.default")?;
        let shape_list = pyo3::types::PyList::new_bound(py, output_shape.iter().copied());
        let view_args = PyTuple::new_bound(py, &[result.as_ref(), shape_list.as_ref()]);
        result = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), node)?;
    }

    // Copy original node's meta onto the final result
    fx_utils::copy_meta(py, node, &result)?;
    annotate_quantized_mean(py, node, &sum_node, &count_node, &result, count)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
