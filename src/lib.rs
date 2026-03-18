use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod compile_spec;
mod graph;
mod ingest;
mod passes;
mod serialize;
mod shape;
mod store;

#[pyfunction]
fn ingest_graph(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let (graph, tensor_store) =
        ingest::fx_bridge::ingest_fx_graph(graph_module, state_dict, graph_signature)?;
    Ok(format!(
        "Ingested {} nodes, {} tensors",
        graph.order.len(),
        tensor_store.buffers.len()
    ))
}

#[pyfunction]
fn lower_to_tosa(
    py: Python<'_>,
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    // 1. Ingest FX graph
    let (mut graph, mut store) =
        ingest::fx_bridge::ingest_fx_graph(graph_module, state_dict, graph_signature)?;

    // 2. Always use phase2 pipeline — it's a superset of phase1.
    //    The quantized graph has already decomposed LayerNorm/Var/etc.
    //    into primitive ops, so detection by op name doesn't work.
    let pipeline = passes::build_phase2_pipeline();
    pipeline.run_all(&mut graph, &mut store);

    // 3. Serialize to TOSA
    let tosa_bytes = serialize::serialize_to_tosa(&graph, &store);

    // 4. Return bytes to Python
    Ok(PyBytes::new_bound(py, &tosa_bytes).unbind())
}

#[pymodule]
fn arm_tosa_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ingest_graph, m)?)?;
    m.add_function(wrap_pyfunction!(lower_to_tosa, m)?)?;
    Ok(())
}
