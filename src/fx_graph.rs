use pyo3::prelude::*;

use crate::fx_py_attr::attrs;

pub fn debug() -> bool {
    static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG.get_or_init(crate::config::tosa_engine_debug_enabled)
}

/// `node.args` as a vector of Python objects.
pub fn node_args<'py>(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let args = attrs(node).object("args")?;
    let mut result = Vec::new();
    for arg in args.iter()? {
        result.push(arg?);
    }
    Ok(result)
}

/// `node.kwargs` as a Python dict.
pub fn node_kwargs<'py>(node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    attrs(node).object("kwargs")
}
