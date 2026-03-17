use crate::graph::ir::DType;
use std::collections::HashMap;

/// Raw tensor data stored as bytes alongside shape and dtype metadata.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
}

/// A map of named tensor buffers (weights, bias, constants, etc.).
#[derive(Debug, Clone)]
pub struct TensorStore {
    pub buffers: HashMap<String, TensorData>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Insert or overwrite a tensor buffer.
    pub fn insert(&mut self, name: String, data: TensorData) {
        self.buffers.insert(name, data);
    }

    /// Get a reference to a tensor buffer by name.
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.buffers.get(name)
    }

    /// Remove a tensor buffer, returning it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<TensorData> {
        self.buffers.remove(name)
    }

    /// Number of stored buffers.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}
