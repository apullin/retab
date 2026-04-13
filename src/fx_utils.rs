//! Common utilities for FX graph transformation passes.
//!
//! These helpers centralize small repeated PyO3 patterns that are still shared
//! across the legacy FX-side passes.

use pyo3::prelude::*;

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
