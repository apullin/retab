use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod compile_spec;
mod fx_cast_int64;
mod fx_convert_clamp;
mod fx_convert_full_like;
mod fx_decompose_adaptive_avgpool;
mod fx_convert_split_to_slice;
mod fx_conv1d_unsqueeze;
mod fx_convert_sum;
mod fx_decompose;
mod fx_decompose_addmm;
mod fx_decompose_linear;
mod fx_decompose_logit;
mod fx_decompose_masked_fill;
mod fx_decompose_mean;
mod fx_decompose_ops;
mod fx_decompose_quant;
mod fx_fold_qdq;
mod fx_fuse_constant_args;
mod fx_fuse_duplicate_users;
mod fx_fuse_equal_placeholders;
mod fx_fuse_quant_activation;
mod fx_graph;
mod fx_insert_rescale;
mod fx_insert_table_ops;
mod fx_memory_format;
mod fx_pass;
mod fx_pipeline;
mod fx_profile;
mod fx_python_bridge;
mod fx_quantize_clamp_args;
mod fx_remove_noop;
mod fx_rewrite_conv;
mod fx_rewrite_matmul;
mod fx_rewrite_squeeze;
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
///
/// The optional `exported_program` is needed to resolve parameter tensor
/// nodes (per-channel scales/zero-points stored as graph parameters).
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None))]
fn fold_qdq_fx(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let ep = exported_program.map(|ep| ep.clone().unbind());
    let pass = fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(ep);
    fx_pass::FxPass::run(&pass, &gm)
}

/// Run only the Rust InsertRescale pass on a Python FX GraphModule.
///
/// This is for diagnostic comparison against the Python InsertRescale passes.
/// The graph must already be Q/DQ-folded (with QuantArgs in node.meta).
#[pyfunction]
fn insert_rescale_fx(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let pass = fx_insert_rescale::FxInsertRescalePass;
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

/// Run the full FX-direct lowering pipeline on a Python FX GraphModule.
///
/// This executes all passes in order: fuse activations, Rust fold Q/DQ,
/// Rust insert rescales, insert table ops, convert to clamp, rewrite conv,
/// annotate memory format. Returns True if any pass modified the graph.
///
/// The optional `exported_program` is forwarded to the Rust fold pass
/// so it can resolve parameter tensor nodes (per-channel scales/zps).
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None))]
fn run_fx_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let ep = exported_program.map(|ep| ep.clone().unbind());
    let pipeline = fx_pipeline::build_fx_pipeline(ep);
    let pass_refs: Vec<&dyn fx_pass::FxPass> = pipeline.iter().map(|p| p.as_ref()).collect();
    fx_pass::run_fx_passes(&gm, &pass_refs)
}

/// Hybrid pipeline: FX-direct passes (correct graph mutations via Python FX)
/// followed by Rust ingestion + TOSA serialization (fast).
///
/// This is the recommended entry point. It combines:
/// 1. FX-direct passes on the live Python graph (correct by construction)
/// 2. Ingest the post-pass graph into Rust IR (one-time, no mutations needed)
/// 3. Serialize to TOSA flatbuffer in Rust (fast)
///
/// The optional `exported_program` is forwarded to Python's
/// `FoldAndAnnotateQParamsPass` for resolving parameter tensor nodes.
#[pyfunction]
#[pyo3(signature = (graph_module, state_dict, graph_signature, exported_program=None))]
fn lower_to_tosa_v2(
    py: Python<'_>,
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyBytes>> {
    // 1. Run FX-direct passes (mutates Python graph in place)
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let ep = exported_program.map(|ep| ep.clone().unbind());
    let pipeline = fx_pipeline::build_fx_pipeline(ep);
    let pass_refs: Vec<&dyn fx_pass::FxPass> = pipeline.iter().map(|p| p.as_ref()).collect();
    fx_pass::run_fx_passes(&gm, &pass_refs)?;

    // 2. Ingest the post-pass graph into Rust IR (read-only, no mutations)
    let (graph, store) =
        ingest::fx_bridge::ingest_fx_graph(graph_module, state_dict, graph_signature)?;

    // 3. Serialize to TOSA flatbuffer
    let tosa_bytes = serialize::serialize_to_tosa(&graph, &store);

    Ok(PyBytes::new_bound(py, &tosa_bytes).unbind())
}

/// Ingest-and-serialize only: convert an already-lowered FX graph to TOSA
/// flatbuffer without running any IR passes.
///
/// Use this after `run_fx_passes()` has already transformed the graph on the
/// Python side.  Unlike `lower_to_tosa` (which runs the full phase-2 IR
/// pipeline), this function only ingests the FX graph into Rust IR and
/// serializes it -- no duplicate pass execution.
#[pyfunction]
fn ingest_and_serialize(
    py: Python<'_>,
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    // 1. Ingest FX graph into Rust IR (read-only snapshot)
    let (mut graph, store) =
        ingest::fx_bridge::ingest_fx_graph(graph_module, state_dict, graph_signature)?;

    // 2. Rebuild user lists to fix stale edges from FX pass mutations
    graph.rebuild_user_lists();

    // 3. Serialize directly to TOSA flatbuffer -- no IR passes
    let tosa_bytes = serialize::serialize_to_tosa(&graph, &store);

    Ok(PyBytes::new_bound(py, &tosa_bytes).unbind())
}

#[pymodule]
fn arm_tosa_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ingest_graph, m)?)?;
    m.add_function(wrap_pyfunction!(lower_to_tosa, m)?)?;
    m.add_function(wrap_pyfunction!(lower_to_tosa_v2, m)?)?;
    m.add_function(wrap_pyfunction!(ingest_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(fold_qdq_fx, m)?)?;
    m.add_function(wrap_pyfunction!(insert_rescale_fx, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_passes, m)?)?;
    m.add_function(wrap_pyfunction!(fx_graph_summary, m)?)?;
    Ok(())
}
