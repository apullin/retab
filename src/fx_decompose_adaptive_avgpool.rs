use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of adaptive_avg_pool2d → avg_pool2d.
///
/// For the common case where input_h % output_h == 0 and input_w % output_w == 0,
/// this is a single avg_pool2d with kernel = (input/output, input/output).
///
/// For the general case, mirrors Python DecomposeAdaptiveAvgPool2dPass:
/// slice into pooling regions, avg_pool2d each, concatenate.
pub struct FxDecomposeAdaptiveAvgPoolPass;

fn is_adaptive_avgpool(target: &str) -> bool {
    target.contains("adaptive_avg_pool2d")
}

fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    helpers.call_method1("refresh_node_fake_tensor", (node,))?;
    Ok(())
}

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

fn set_meta_and_refresh(
    py: Python<'_>,
    inserted: &Bound<'_, PyAny>,
    meta: &Bound<'_, PyAny>,
) -> PyResult<()> {
    inserted.setattr("meta", meta)?;
    refresh_fake_tensor(py, inserted)
}

fn set_meta_with_value(
    inserted: &Bound<'_, PyAny>,
    meta: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    meta.set_item("val", value)?;
    inserted.setattr("meta", meta)?;
    Ok(())
}

fn cat_fake_tensor<'py>(
    py: Python<'py>,
    inputs: &[Bound<'py, PyAny>],
    dim: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import_bound("torch")?;
    let fake_inputs = PyList::empty_bound(py);
    for input in inputs {
        let meta = input.getattr("meta")?;
        fake_inputs.append(meta.get_item("val")?)?;
    }
    torch.call_method1("cat", (fake_inputs.as_ref(), dim))
}

impl FxPass for FxDecomposeAdaptiveAvgPoolPass {
    fn name(&self) -> &str {
        "FxDecomposeAdaptiveAvgPool"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_adaptive_avgpool(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_adaptive(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_adaptive<'py>(
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
    let input_shape = FxGraphModule::node_shape(x)?;
    if input_shape.len() < 4 {
        return Ok(());
    }

    let input_h = input_shape[2];
    let input_w = input_shape[3];

    // args[1] = output_size (tuple of 2 ints)
    let output_size: Vec<i64> = if args.len() > 1 {
        args[1].extract().unwrap_or_else(|_| vec![1, 1])
    } else {
        vec![1, 1]
    };
    let output_h = output_size[0];
    let output_w = output_size[1];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeAdaptiveAvgPool] {} input=({},{}) output=({},{})",
            name, input_h, input_w, output_h, output_w
        );
    }

    let avgpool_op = resolve_aten_op(py, "avg_pool2d.default")?;
    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    // Match the Python Arm pass literally: always slice pooling regions,
    // apply avg_pool2d with stride 1, then concatenate. The old “single pool
    // on divisible shapes” shortcut leaks oversized stride values and diverges
    // from the backend path we are trying to reproduce.
    let inserting = graph.call_method1("inserting_before", (node,))?;
    inserting.call_method0("__enter__")?;

    let meta_with_qparams = copied_meta(py, node, false)?;
    let meta_without_qparams = copied_meta(py, node, true)?;

    let mut rows: Vec<Bound<'py, PyAny>> = Vec::new();
    for out_i in 0..output_h {
        let mut row: Vec<Bound<'py, PyAny>> = Vec::new();
        for out_j in 0..output_w {
            let start_h = (out_i * input_h / output_h) as i64;
            let end_h = ((out_i + 1) * input_h / output_h
                + if ((out_i + 1) * input_h) % output_h != 0 {
                    1
                } else {
                    0
                }) as i64;
            let start_w = (out_j * input_w / output_w) as i64;
            let end_w = ((out_j + 1) * input_w / output_w
                + if ((out_j + 1) * input_w) % output_w != 0 {
                    1
                } else {
                    0
                }) as i64;

            // Slice along H
            let h_args = PyTuple::new_bound(
                py,
                &[
                    x.as_ref(),
                    2i64.into_py(py).into_bound(py).as_ref(),
                    start_h.into_py(py).into_bound(py).as_ref(),
                    end_h.into_py(py).into_bound(py).as_ref(),
                ],
            );
            let x_h = graph.call_method1("call_function", (&slice_op, &h_args))?;
            set_meta_and_refresh(py, &x_h, &meta_without_qparams)?;

            // Slice along W
            let w_args = PyTuple::new_bound(
                py,
                &[
                    x_h.as_ref(),
                    3i64.into_py(py).into_bound(py).as_ref(),
                    start_w.into_py(py).into_bound(py).as_ref(),
                    end_w.into_py(py).into_bound(py).as_ref(),
                ],
            );
            let x_hw = graph.call_method1("call_function", (&slice_op, &w_args))?;
            set_meta_and_refresh(py, &x_hw, &meta_without_qparams)?;

            // avg_pool2d with kernel = region size
            let kh = end_h - start_h;
            let kw = end_w - start_w;
            let kernel = pyo3::types::PyList::new_bound(py, [kh, kw].iter().copied());
            let stride = pyo3::types::PyList::new_bound(py, [1i64, 1].iter().copied());
            let padding = pyo3::types::PyList::new_bound(py, [0i64, 0].iter().copied());
            let pool_args = PyTuple::new_bound(
                py,
                &[
                    x_hw.as_ref(),
                    kernel.as_ref(),
                    stride.as_ref(),
                    padding.as_ref(),
                ],
            );
            let pooled = graph.call_method1("call_function", (&avgpool_op, &pool_args))?;
            set_meta_and_refresh(py, &pooled, &meta_with_qparams)?;

            row.push(pooled);
        }

        // Avoid emitting invalid single-input CONCATs; otherwise attach an
        // exact fake tensor instead of relying on generic refresh logic.
        if row.len() == 1 {
            rows.push(row.into_iter().next().unwrap());
        } else {
            let row_list = pyo3::types::PyList::new_bound(py, &row);
            let cat_args = PyTuple::new_bound(
                py,
                &[row_list.as_ref(), 3i64.into_py(py).into_bound(py).as_ref()],
            );
            let row_tensor = graph.call_method1("call_function", (&cat_op, &cat_args))?;
            let row_meta = copied_meta(py, node, true)?;
            let row_fake_val = cat_fake_tensor(py, &row, 3)?;
            set_meta_with_value(&row_tensor, &row_meta, &row_fake_val)?;
            rows.push(row_tensor);
        }
    }

    // Cat all rows along H (dim=2)
    let result = if rows.len() == 1 {
        rows.into_iter().next().unwrap()
    } else {
        let rows_list = pyo3::types::PyList::new_bound(py, &rows);
        let cat_args = PyTuple::new_bound(
            py,
            &[rows_list.as_ref(), 2i64.into_py(py).into_bound(py).as_ref()],
        );
        let result = graph.call_method1("call_function", (&cat_op, &cat_args))?;
        let result_meta = copied_meta(py, node, true)?;
        let result_fake_val = cat_fake_tensor(py, &rows, 2)?;
        set_meta_with_value(&result, &result_meta, &result_fake_val)?;
        result
    };

    inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
