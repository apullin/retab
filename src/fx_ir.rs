//! Rust-native graph IR types for FX graph metadata.
//!
//! Phase 1: NodeMeta extraction — read metadata from Python FX nodes into
//! typed Rust structs, operate in pure Rust, write back changed fields.
//! The FX graph structure stays Python-managed.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_control_flow_metadata::QParam;

// ---------------------------------------------------------------------------
// NodeMeta — typed metadata extracted from a Python FX node
// ---------------------------------------------------------------------------

/// Metadata fields extracted from `node.meta`. The common fields are typed;
/// rare fields stay in the Python meta dict and are not touched.
#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    pub input_qparams: Option<HashMap<usize, QParam>>,
    pub output_qparams: Option<HashMap<usize, QParam>>,
    pub val_dtype: Option<String>,
    pub val_shape: Option<Vec<i64>>,
    pub tosa_dim_order: Option<Vec<usize>>,
    pub tosa_spatial_rank: Option<usize>,

    // Pre-pass snapshot of parent arg qparams (stored on cond/while nodes)
    pub pre_pass_arg_qparams: Option<HashMap<usize, QParam>>,

    // Track which fields were modified so write_back only touches what changed
    dirty: DirtyFlags,
}

#[derive(Debug, Clone, Default)]
struct DirtyFlags {
    input_qparams: bool,
    output_qparams: bool,
    tosa_dim_order: bool,
}

// ---------------------------------------------------------------------------
// QParam dict extraction helpers
// ---------------------------------------------------------------------------

fn extract_qparam_dict(py_dict: &Bound<'_, PyAny>) -> Option<HashMap<usize, QParam>> {
    if py_dict.is_none() || !py_dict.is_truthy().unwrap_or(false) {
        return None;
    }
    let mut map = HashMap::new();
    if let Ok(items) = py_dict.call_method0("items") {
        if let Ok(iter) = items.iter() {
            for item in iter.filter_map(|i| i.ok()) {
                if let (Ok(key), Ok(val)) = (
                    item.get_item(0).and_then(|k| k.extract::<usize>()),
                    item.get_item(1),
                ) {
                    map.insert(key, QParam::from_py(&val));
                }
            }
        }
    }
    if map.is_empty() { None } else { Some(map) }
}

fn qparam_dict_to_py<'py>(
    py: Python<'py>,
    map: &HashMap<usize, QParam>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (&idx, qp) in map {
        dict.set_item(idx, qp.to_py(py)?)?;
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// NodeMeta import / export
// ---------------------------------------------------------------------------

impl NodeMeta {
    /// Extract typed metadata from a Python FX node.
    pub fn from_fx(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        let meta = node.getattr("meta")?;

        let input_qparams = meta.get_item("input_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        let output_qparams = meta.get_item("output_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        let val = meta.get_item("val").ok();
        let val_dtype = val.as_ref()
            .and_then(|v| v.getattr("dtype").ok())
            .map(|d| format!("{d}"));
        let val_shape = val.as_ref()
            .and_then(|v| v.getattr("shape").ok())
            .and_then(|s| s.extract::<Vec<i64>>().ok());

        let tosa_dim_order = meta.get_item("tosa_dim_order")
            .ok()
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let tosa_spatial_rank = meta.get_item("tosa_spatial_rank")
            .ok()
            .and_then(|v| v.extract::<usize>().ok());

        let pre_pass_arg_qparams = meta.get_item("_pre_pass_arg_qparams")
            .ok()
            .and_then(|v| extract_qparam_dict(&v));

        Ok(Self {
            input_qparams,
            output_qparams,
            val_dtype,
            val_shape,
            tosa_dim_order,
            tosa_spatial_rank,
            pre_pass_arg_qparams,
            dirty: DirtyFlags::default(),
        })
    }

    /// Write back only the changed metadata fields to a Python FX node.
    pub fn write_back(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
        let meta = node.getattr("meta")?;

        if self.dirty.output_qparams {
            match &self.output_qparams {
                Some(map) => meta.set_item("output_qparams", qparam_dict_to_py(py, map)?)?,
                None => { let _ = meta.del_item("output_qparams"); },
            }
        }

        if self.dirty.input_qparams {
            match &self.input_qparams {
                Some(map) => meta.set_item("input_qparams", qparam_dict_to_py(py, map)?)?,
                None => { let _ = meta.del_item("input_qparams"); },
            }
        }

        if self.dirty.tosa_dim_order {
            if let Some(ref order) = self.tosa_dim_order {
                let tuple = pyo3::types::PyTuple::new_bound(py, order);
                meta.set_item("tosa_dim_order", tuple)?;
            }
        }

        Ok(())
    }

    // --- Mutation helpers that set dirty flags ---

    pub fn set_output_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.output_qparams = Some(map);
        self.dirty.output_qparams = true;
    }

    pub fn set_input_qparams(&mut self, map: HashMap<usize, QParam>) {
        self.input_qparams = Some(map);
        self.dirty.input_qparams = true;
    }

    pub fn set_single_output_qparam(&mut self, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(0, qp);
        self.set_output_qparams(map);
    }

    pub fn set_single_input_qparam(&mut self, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(0, qp);
        self.set_input_qparams(map);
    }

    pub fn set_both_qparams(&mut self, qp: QParam) {
        self.set_single_output_qparam(qp.clone());
        self.set_single_input_qparam(qp);
    }

    pub fn clear_output_qparams(&mut self) {
        self.output_qparams = None;
        self.dirty.output_qparams = true;
    }

    pub fn set_tosa_dim_order(&mut self, order: Vec<usize>) {
        self.tosa_dim_order = Some(order);
        self.dirty.tosa_dim_order = true;
    }

    /// Get the single output qparam (key 0), if present.
    pub fn output_qparam(&self) -> Option<&QParam> {
        self.output_qparams.as_ref()?.get(&0)
    }

    /// Get the single input qparam (key 0), if present.
    pub fn input_qparam(&self) -> Option<&QParam> {
        self.input_qparams.as_ref()?.get(&0)
    }
}
