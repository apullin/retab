//! Common utilities for FX graph transformation passes.
//!
//! These "verbs" abstract repeated PyO3 patterns across FX pass files:
//! extracting metadata, copying/refreshing FakeTensors, and checking
//! quantization.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Check whether a node has quantization parameters.
pub fn is_quantized(node: &Bound<'_, PyAny>) -> bool {
    let Ok(meta) = node.getattr("meta") else {
        return false;
    };
    let Ok(iqp) = meta.call_method1("get", ("input_qparams",)) else {
        return false;
    };
    if iqp.is_none() {
        return false;
    }
    iqp.downcast::<PyDict>()
        .map(|d| !d.is_empty())
        .unwrap_or(false)
}

/// Get `output_qparams[0]` from a node, if it exists.
pub fn get_output_qparam<'py>(node: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    let meta = node.getattr("meta").ok()?;
    let oqp = meta.call_method1("get", ("output_qparams",)).ok()?;
    if oqp.is_none() {
        return None;
    }
    let qp = oqp.call_method1("get", (0,)).ok()?;
    if qp.is_none() {
        None
    } else {
        Some(qp)
    }
}

/// Copy metadata from `src` to `dst` node using `copy.copy()`.
pub fn copy_meta<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    dst: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let copy_mod = py.import_bound("copy")?;
    let meta = copy_mod.call_method1("copy", (src.getattr("meta")?,))?;
    dst.setattr("meta", &meta)?;
    Ok(())
}

/// Copy metadata and refresh FakeTensor on the destination node.
pub fn copy_meta_and_refresh<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    dst: &Bound<'py, PyAny>,
) -> PyResult<()> {
    copy_meta(py, src, dst)?;
    refresh_fake_tensor(py, dst)
}

/// Refresh a node's FakeTensor metadata via `fx_helpers.refresh_node_fake_tensor`.
pub fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    let _ = helpers.call_method1("refresh_node_fake_tensor", (node,));
    Ok(())
}
