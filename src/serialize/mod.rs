pub mod tosa_generated;
pub mod visitors;

mod attribute_flatbuffer;
mod collection;
mod constant_pool;
mod flatbuffer;
pub(crate) mod op_semantics;
mod output_layout;
mod tensor_layout;
mod transpose_peephole;

use crate::graph::Graph;
use crate::pipeline_config::PipelineConfig;
use crate::store::TensorStore;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub fn serialize_graphs_to_tosa_py(
    py: Python<'_>,
    graphs: &[(String, Graph)],
    store: &TensorStore,
    config: &PipelineConfig,
) -> PyResult<Py<PyBytes>> {
    let tensor_bytes: usize = store.buffers.values().map(|t| t.data.len() + 256).sum();
    let node_count: usize = graphs.iter().map(|(_, g)| g.order.len()).sum();
    let estimated = tensor_bytes + node_count * 512 + 1024 * 1024;
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!(
            "[Serialize] pre-allocating {}MB for {} tensors, {} nodes",
            estimated / (1024 * 1024),
            store.buffers.len(),
            node_count
        );
    }
    let regions = collection::collect_regions(graphs, store).map_err(PyRuntimeError::new_err)?;
    Ok(flatbuffer::serialize_regions_to_tosa_py(
        py, &regions, estimated, config,
    ))
}
