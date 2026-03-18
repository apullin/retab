use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod compile_spec;
mod fx_fold_qdq;
mod fx_graph;
mod fx_pass;
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

// ---------------------------------------------------------------------------
// FX-graph-direct: fold Q/DQ pass operating on live Python FX graph
// ---------------------------------------------------------------------------

/// Run the FoldAndAnnotateQParams pass directly on a Python FX GraphModule.
///
/// This mutates the graph in-place (removes Q/DQ nodes, annotates qparams
/// in node.meta) and returns True if any modifications were made.
#[pyfunction]
fn fold_qdq_fx(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let pass = fx_fold_qdq::FxFoldAndAnnotateQParamsPass;
    fx_pass::FxPass::run(&pass, &gm)
}

/// Return a summary of the FX graph: node count plus a list of
/// (name, op, target_str) triples.  Useful for quick sanity checks.
#[pyfunction]
fn fx_graph_summary(graph_module: &Bound<'_, PyAny>) -> PyResult<Vec<(String, String, String)>> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let nodes = gm.nodes()?;
    let mut result = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let name = fx_graph::FxGraphModule::node_name(node)?;
        let op = fx_graph::FxGraphModule::node_op(node)?;
        let target = fx_graph::FxGraphModule::node_target_str(node)?;
        result.push((name, op, target));
    }
    Ok(result)
}

#[pymodule]
fn arm_tosa_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ingest_graph, m)?)?;
    m.add_function(wrap_pyfunction!(lower_to_tosa, m)?)?;
    m.add_function(wrap_pyfunction!(fold_qdq_fx, m)?)?;
    m.add_function(wrap_pyfunction!(fx_graph_summary, m)?)?;
    Ok(())
}
