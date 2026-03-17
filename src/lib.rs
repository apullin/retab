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

    // 2. Run passes — detect whether phase2 is needed based on the graph
    //    containing ops that require decomposition (LayerNorm, Var, etc.).
    use crate::graph::ir::OpTarget;
    let needs_phase2 = graph.iter_nodes().any(|(_, n)| {
        matches!(
            n.target,
            OpTarget::LayerNorm | OpTarget::Var | OpTarget::BatchNorm
        )
    });
    let pipeline = if needs_phase2 {
        passes::build_phase2_pipeline()
    } else {
        passes::build_phase1_pipeline()
    };
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
