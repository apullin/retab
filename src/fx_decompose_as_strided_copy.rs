use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;
use crate::fx_utils;

pub struct FxDecomposeAsStridedCopyPass;

fn is_as_strided_copy(target: &str) -> bool {
    target.contains("as_strided_copy.default")
}

fn extract_static_tuple(arg: &Bound<'_, PyAny>) -> Option<Vec<i64>> {
    arg.extract::<Vec<i64>>().ok()
}

fn extract_storage_offset(args: &[Bound<'_, PyAny>], kwargs: &Bound<'_, PyDict>) -> Option<i64> {
    if let Ok(value) = kwargs.get_item("storage_offset") {
        if let Some(value) = value {
            return value.extract::<i64>().ok();
        }
    }
    args.get(3)
        .and_then(|value| value.extract::<i64>().ok())
        .or(Some(0))
}

fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    let mut running = 1;
    for (idx, dim) in shape.iter().enumerate().rev() {
        strides[idx] = running;
        running *= *dim;
    }
    strides
}

fn stride_matches(shape: &[i64], stride: &[i64]) -> bool {
    if shape.len() != stride.len() {
        return false;
    }
    let expected = contiguous_strides(shape);
    shape
        .iter()
        .zip(stride.iter())
        .zip(expected.iter())
        .enumerate()
        .all(|(idx, ((dim, actual), expected))| {
            if *actual < 0 {
                return false;
            }
            if idx == shape.len() - 1 {
                return *actual >= *expected;
            }
            if *dim == 1 || *expected == 0 {
                return true;
            }
            *actual == *expected
        })
}

fn insert_view_copy_before<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
    size: &[i64],
) -> PyResult<Bound<'py, PyAny>> {
    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let shape = PyTuple::new_bound(py, size.iter().copied());
    let args = PyTuple::new_bound(py, &[input.as_ref(), shape.as_ref()]);

    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let result = (|| -> PyResult<Bound<'py, PyAny>> {
        let new_node = graph.call_method1("call_function", (&view_op, &args))?;
        fx_utils::copy_meta_and_refresh(py, ref_node, &new_node)?;
        Ok(new_node)
    })();
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    result
}

impl FxPass for FxDecomposeAsStridedCopyPass {
    fn name(&self) -> &str {
        "FxDecomposeAsStridedCopy"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let graph = gm.gm_ref().getattr("graph")?;
        let mut modified = false;

        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();
        for node in gm.nodes()? {
            if FxGraphModule::node_op(&node)? != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(&node)?;
            if is_as_strided_copy(&target) {
                candidates.push(node);
            }
        }

        for node in &candidates {
            let args = FxGraphModule::node_args(node)?;
            if args.len() < 3 || !FxGraphModule::is_node(&args[0]) {
                continue;
            }

            let Some(size) = extract_static_tuple(&args[1]) else {
                continue;
            };
            let Some(stride) = extract_static_tuple(&args[2]) else {
                continue;
            };
            let kwargs = FxGraphModule::node_kwargs(node)?;
            let kwargs = kwargs.downcast::<PyDict>()?;
            let Some(storage_offset) = extract_storage_offset(&args, &kwargs) else {
                continue;
            };
            if storage_offset != 0 || !stride_matches(&size, &stride) {
                continue;
            }

            let replacement = insert_view_copy_before(py, &graph, node, &args[0], &size)?;
            FxGraphModule::replace_all_uses_with(node, &replacement)?;
            gm.erase_node(node)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
