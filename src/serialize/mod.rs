pub mod tosa_generated;
pub mod visitors;

mod attribute_flatbuffer;
mod collection;
mod flatbuffer;
pub(crate) mod op_semantics;
mod output_layout;
mod tensor_decl;
mod tensor_layout;

use crate::graph::Graph;
use crate::store::TensorStore;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn log_serialize_profile_phase(profile: bool, phase: &str, start: std::time::Instant) {
    if profile {
        eprintln!(
            "[RETAB][serialize-profile] phase={phase} time_s={:.6}",
            start.elapsed().as_secs_f64()
        );
    }
}

pub fn serialize_graphs_to_tosa_py(
    py: Python<'_>,
    graphs: &[(String, Graph)],
    store: &TensorStore,
) -> Py<PyBytes> {
    let profile = crate::config::profile_ingest_serialize_enabled();
    let tensor_bytes: usize = store.buffers.values().map(|t| t.data.len() + 256).sum();
    let node_count: usize = graphs.iter().map(|(_, g)| g.order.len()).sum();
    let estimated = tensor_bytes + node_count * 512 + 1024 * 1024;
    let phase_start = std::time::Instant::now();
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    log_serialize_profile_phase(profile, "clear_caches", phase_start);
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!(
            "[Serialize] pre-allocating {}MB for {} tensors, {} nodes",
            estimated / (1024 * 1024),
            store.buffers.len(),
            node_count
        );
    }
    let phase_start = std::time::Instant::now();
    let regions = collection::collect_regions(graphs, store);
    log_serialize_profile_phase(profile, "collect_regions", phase_start);
    if profile {
        let tensors: usize = regions.iter().map(|r| r.items.tensors.len()).sum();
        let shapes: usize = regions.iter().map(|r| r.items.shapes.len()).sum();
        let operators: usize = regions.iter().map(|r| r.items.operators.len()).sum();
        eprintln!(
            "[RETAB][serialize-profile] regions={} tensors={} shapes={} operators={} tensor_bytes={} estimated_capacity={}",
            regions.len(),
            tensors,
            shapes,
            operators,
            tensor_bytes,
            estimated
        );
    }
    let phase_start = std::time::Instant::now();
    let py_bytes = flatbuffer::serialize_regions_to_tosa_py(py, &regions, estimated);
    log_serialize_profile_phase(profile, "flatbuffer_total", phase_start);
    py_bytes
}
