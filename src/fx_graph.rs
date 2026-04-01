use pyo3::prelude::*;
use std::time::Instant;

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

    // ------------------------------------------------------------------
    // Read helpers
    // ------------------------------------------------------------------

    /// All nodes in topological order (Python node objects).
    pub fn nodes(&self) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let graph = self.gm.getattr("graph")?;
        let nodes = graph.getattr("nodes")?;
        let mut result = Vec::new();
        for node in nodes.iter()? {
            result.push(node?);
        }
        Ok(result)
    }

    /// `node.op` as a Rust string.
    pub fn node_op(node: &Bound<'py, PyAny>) -> PyResult<String> {
        node.getattr("op")?.extract()
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

    /// `node.users` keys (i.e. the consumer nodes).
    pub fn node_users(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let users = node.getattr("users")?;
        let keys = users.call_method0("keys")?;
        let mut result = Vec::new();
        for k in keys.iter()? {
            result.push(k?);
        }
        Ok(result)
    }

    /// Returns `true` when `obj` looks like an FX `Node` (duck-typed).
    pub fn is_node(obj: &Bound<'py, PyAny>) -> bool {
        obj.getattr("op").is_ok() && obj.getattr("target").is_ok()
    }

    /// `node.meta["val"].shape` as `Vec<i64>`.
    pub fn node_shape(node: &Bound<'py, PyAny>) -> PyResult<Vec<i64>> {
        let meta = node.getattr("meta")?;
        let val = meta.get_item("val")?;
        let shape = val.getattr("shape")?;
        shape.extract()
    }

    /// `str(node.meta["val"].dtype)`.
    pub fn node_dtype_str(node: &Bound<'py, PyAny>) -> PyResult<String> {
        let meta = node.getattr("meta")?;
        let val = meta.get_item("val")?;
        let dtype = val.getattr("dtype")?;
        dtype.str()?.extract()
    }

    /// Write `node.meta[key] = value`.
    pub fn set_meta(
        node: &Bound<'py, PyAny>,
        key: &str,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let meta = node.getattr("meta")?;
        meta.set_item(key, value)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Mutation helpers (delegate to Python FX graph APIs)
    // ------------------------------------------------------------------

    /// `node.replace_all_uses_with(new_node)`.
    pub fn replace_all_uses_with(
        node: &Bound<'py, PyAny>,
        new_node: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        node.call_method1("replace_all_uses_with", (new_node,))?;
        Ok(())
    }

    /// `graph.erase_node(node)`.
    pub fn erase_node(&self, node: &Bound<'py, PyAny>) -> PyResult<()> {
        let graph = self.gm.getattr("graph")?;
        graph.call_method1("erase_node", (node,))?;
        Ok(())
    }

    /// `graph_module.recompile()`.
    pub fn recompile(&self) -> PyResult<()> {
        let t0 = if crate::fx_profile::enabled() {
            Some(Instant::now())
        } else {
            None
        };
        self.gm.call_method0("recompile")?;
        if let Some(t0) = t0 {
            crate::fx_profile::record_recompile(t0.elapsed());
        }
        Ok(())
    }

    /// Iterator over call_function nodes in the graph.
    pub fn call_functions(&self) -> PyResult<Vec<(Bound<'py, PyAny>, String)>> {
        let all_nodes = self.nodes()?;
        let mut result = Vec::new();
        for node in all_nodes {
            let op = Self::node_op(&node)?;
            if op != "call_function" {
                continue;
            }
            let target = Self::node_target_str(&node)?;
            result.push((node, target));
        }
        Ok(result)
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> PyResult<usize> {
        let graph = self.gm.getattr("graph")?;
        let nodes = graph.getattr("nodes")?;
        // torch.fx.Graph.nodes supports len()
        let len = self
            .gm
            .py()
            .import_bound("builtins")?
            .call_method1("len", (&nodes,))?
            .extract::<usize>()?;
        Ok(len)
    }
}
