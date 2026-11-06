pub mod tosa_types;
pub mod visitors;

mod attribute_flatbuffer;
mod collection;
mod constant_pool;
#[cfg(test)]
mod explicit_control_flow_tests;
#[cfg(test)]
mod explicit_layout_tests;
#[cfg(test)]
mod explicit_pool_rewrite_tests;
#[cfg(test)]
mod explicit_rewrite_conv_tests;
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
    if crate::config::retab_debug_enabled() {
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
