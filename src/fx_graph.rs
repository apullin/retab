use pyo3::prelude::*;

/// Extract a typed attribute from a Python object, trying multiple names.
/// Returns `default` if none of the names exist or extraction fails.
pub fn py_attr<'py, T: pyo3::FromPyObject<'py>>(
    obj: &pyo3::Bound<'py, pyo3::PyAny>,
    names: &[&str],
    default: T,
) -> T {
    names
        .iter()
        .find_map(|name| obj.getattr(*name).ok())
        .and_then(|v| v.extract().ok())
        .unwrap_or(default)
}

/// Get a Python value from a dict, returning None if missing or None.
pub fn meta_val<'py>(
    dict: &pyo3::Bound<'py, pyo3::PyAny>,
    key: &str,
) -> Option<pyo3::Bound<'py, pyo3::PyAny>> {
    dict.call_method1("get", (key,))
        .ok()
        .filter(|v| !v.is_none())
}

/// Thin wrapper around a Python `torch.fx.GraphModule`, operating directly
/// on the live Python FX graph via PyO3.  No Rust-side IR copy is made —
/// every read/write goes straight through to Python.
pub struct FxGraphModule<'py> {
    gm: Bound<'py, PyAny>,
}

impl<'py> FxGraphModule<'py> {
    pub fn new(gm: &Bound<'py, PyAny>) -> Self {
        Self { gm: gm.clone() }
    }

    pub fn py(&self) -> Python<'py> {
        self.gm.py()
    }

    pub fn debug() -> bool {
        static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DEBUG.get_or_init(|| std::env::var("TOSA_ENGINE_DEBUG").is_ok())
    }

    /// Borrow the underlying Python `GraphModule` object.
    pub fn gm_ref(&self) -> &Bound<'py, PyAny> {
        &self.gm
    }

    /// `str(node.target)`.
    pub fn node_target_str(node: &Bound<'py, PyAny>) -> PyResult<String> {
        let target = node.getattr("target")?;
        target.str()?.extract()
    }

    /// `node.name`.
    pub fn node_name(node: &Bound<'py, PyAny>) -> PyResult<String> {
        node.getattr("name")?.extract()
    }

    /// `node.args` as a vector of Python objects.
    pub fn node_args(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let args = node.getattr("args")?;
        let mut result = Vec::new();
        for arg in args.iter()? {
            result.push(arg?);
        }
        Ok(result)
    }

    /// `node.kwargs` as a Python dict.
    pub fn node_kwargs(node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        node.getattr("kwargs")
    }

    /// Returns `true` when `obj` looks like an FX `Node` (duck-typed).
    pub fn is_node(obj: &Bound<'py, PyAny>) -> bool {
        obj.getattr("op").is_ok() && obj.getattr("target").is_ok()
    }

    /// `graph_module.recompile()`.
    pub fn recompile(&self) -> PyResult<()> {
        self.gm.call_method0("recompile")?;
        Ok(())
    }
}
