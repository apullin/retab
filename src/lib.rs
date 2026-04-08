use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::graph::Graph;
use crate::store::TensorStore;

mod fx_annotate_output_dim_order;
mod fx_canonicalize_gather;
mod fx_cast_int64;
mod fx_conv1d_unsqueeze;
mod fx_conv1d_unsqueeze_ir;
mod fx_control_flow_metadata;
mod fx_ir;
mod fx_convert_clamp;
mod fx_convert_clamp_ir;
mod fx_convert_full_like;
mod fx_convert_full_like_ir;
mod fx_convert_split_to_slice;
mod fx_convert_split_to_slice_ir;
mod fx_convert_sum;
mod fx_convert_sum_ir;
mod fx_decompose_adaptive_avgpool;
mod fx_decompose_addmm;
mod fx_decompose_addmm_ir;
mod fx_decompose_as_strided_copy;
mod fx_decompose_as_strided_copy_ir;
mod fx_decompose_cumsum;
mod fx_decompose_cumsum_ir;
mod fx_decompose_grouped_conv;
mod fx_decompose_index_select;
mod fx_decompose_index_tensor;
mod fx_decompose_linear;
mod fx_decompose_linear_ir;
mod fx_decompose_logit;
mod fx_decompose_logit_ir;
mod fx_decompose_masked_fill;
mod fx_decompose_masked_fill_ir;
mod fx_decompose_maxpool_dilation;
mod fx_decompose_mean;
mod fx_decompose_ops;
mod fx_decompose_pixel_shuffle;
mod fx_decompose_pixel_shuffle_ir;
mod fx_decompose_quant;
mod fx_decompose_repeat_interleave;
mod fx_decompose_repeat_interleave_ir;
mod fx_decompose_select_scatter;
mod fx_decompose_select_scatter_ir;
mod fx_decompose_strided_slice;
mod fx_decompose_strided_slice_ir;
mod fx_decompose_tril;
mod fx_decompose_tril_ir;
mod fx_decompose_unfold;
mod fx_expand_where_mask;
mod fx_expand_where_mask_ir;
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
mod fx_quantize_clamp_args;
mod fx_remove_noop;
mod fx_remove_noop_ir;
mod fx_rewrite_conv;
mod fx_rewrite_index_put;
mod fx_rewrite_matmul;
mod fx_rewrite_squeeze_ir;
mod fx_size_adjust_input;
mod fx_utils;
mod graph;
mod ingest;
mod runner_logging;
mod serialize;
mod store;

type GraphHierarchy = Vec<(String, Graph)>;

fn clone_bound_pyobject(exported_program: Option<&Bound<'_, PyAny>>) -> Option<PyObject> {
    exported_program.map(|ep| ep.clone().unbind())
}

fn to_py_bytes(py: Python<'_>, data: Vec<u8>) -> Py<PyBytes> {
    PyBytes::new_bound(py, &data).unbind()
}

fn run_single_fx_pass<P: fx_pass::FxPass>(
    graph_module: &Bound<'_, PyAny>,
    pass: P,
) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    fx_pass::FxPass::run(&pass, &gm)
}

fn run_full_fx_pipeline(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<PyObject>,
) -> PyResult<bool> {
    // Use hybrid pipeline: persistent IrGraph for Rust passes,
    // flush/reimport only at FxPass boundaries.
    let py = graph_module.py();
    let pipeline = fx_pipeline::build_hybrid_pipeline(exported_program);
    fx_ir::run_hybrid_pipeline(py, graph_module, &pipeline)
}

fn ingest_graph_hierarchy(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<(GraphHierarchy, TensorStore)> {
    ingest::fx_bridge::ingest_fx_graph_hierarchy(graph_module, state_dict, graph_signature)
}

fn rebuild_graph_users(graphs: &mut [(String, Graph)]) {
    for (_, graph) in graphs {
        graph.rebuild_user_lists();
    }
}

fn lower_fx_graph_to_tosa_bytes(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
    exported_program: Option<PyObject>,
) -> PyResult<Vec<u8>> {
    run_full_fx_pipeline(graph_module, exported_program)?;
    let (graphs, store) = ingest_graph_hierarchy(graph_module, state_dict, graph_signature)?;
    Ok(serialize::serialize_graphs_to_tosa(&graphs, &store))
}

fn ingest_and_serialize_tosa_bytes(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<Vec<u8>> {
    let (mut graphs, store) = ingest_graph_hierarchy(graph_module, state_dict, graph_signature)?;
    rebuild_graph_users(&mut graphs);
    Ok(serialize::serialize_graphs_to_tosa(&graphs, &store))
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
    run_single_fx_pass(
        graph_module,
        fx_fold_qdq::FxFoldAndAnnotateQParamsPass::new(clone_bound_pyobject(exported_program)),
    )
}

/// Run only the Rust InsertRescale pass on a Python FX GraphModule.
///
/// This is for diagnostic comparison against the Python InsertRescale passes.
/// The graph must already be Q/DQ-folded (with QuantArgs in node.meta).
#[pyfunction]
fn insert_rescale_fx(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    run_single_fx_pass(graph_module, fx_insert_rescale::FxInsertRescalePass::new())
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
    run_full_fx_pipeline(graph_module, clone_bound_pyobject(exported_program))
}

/// Run a safe subset of FX passes on a control-flow submodule graph.
///
/// Run the while_loop submodule pipeline (fold_qdq + carried rescale).
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None))]
fn run_fx_submodule_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let ep = exported_program.map(|e| e.clone().unbind());
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let pipeline = fx_pipeline::build_fx_while_submodule_pipeline(ep);
    fx_pass::run_boxed_fx_passes(&gm, &pipeline)
}

/// Extended submodule passes for cond branches (includes fold_qdq).
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None))]
fn run_fx_cond_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let ep = exported_program.map(|e| e.clone().unbind());
    let gm = fx_graph::FxGraphModule::new(graph_module);
    let pipeline = fx_pipeline::build_fx_cond_submodule_pipeline(ep);
    fx_pass::run_boxed_fx_passes(&gm, &pipeline)
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
    Ok(to_py_bytes(
        py,
        lower_fx_graph_to_tosa_bytes(
            graph_module,
            state_dict,
            graph_signature,
            clone_bound_pyobject(exported_program),
        )?,
    ))
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
    Ok(to_py_bytes(
        py,
        ingest_and_serialize_tosa_bytes(graph_module, state_dict, graph_signature)?,
    ))
}

/// Structural-only metadata stamp (no qparam propagation). Run before parent passes.
#[pyfunction]
#[pyo3(signature = (graph_module))]
fn stamp_control_flow_structural(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    fx_pass::FxPass::run(&fx_control_flow_metadata::FxControlFlowStructuralPass, &gm)
}

/// Propagate parent post-pass qparams to submodule placeholders. Run after parent passes.
#[pyfunction]
#[pyo3(signature = (graph_module))]
fn propagate_control_flow_qparams(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    fx_pass::FxPass::run(&fx_control_flow_metadata::FxControlFlowQParamPropagationPass, &gm)
}

/// Insert boundary RESCALE pairs for control-flow submodules.
/// Ports upstream InsertControlFlowRescalesPass.
#[pyfunction]
#[pyo3(signature = (graph_module))]
fn cf_insert_boundary_rescales(graph_module: &Bound<'_, PyAny>) -> PyResult<u32> {
    let py = graph_module.py();
    fx_control_flow_metadata::insert_cf_boundary_rescales(py, graph_module)
}

/// Run the full output normalize phase in one Rust call.
#[pyfunction]
#[pyo3(signature = (graph_module))]
fn cf_run_output_normalize_phase(graph_module: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = graph_module.py();
    fx_control_flow_metadata::run_output_normalize_phase(py, graph_module)
}

/// Run the full post-Rust boundary phase in one Rust call.
#[pyfunction]
#[pyo3(signature = (graph_module, tosa_spec))]
fn cf_run_boundary_phase(
    graph_module: &Bound<'_, PyAny>,
    tosa_spec: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = graph_module.py();
    fx_control_flow_metadata::run_boundary_phase(py, graph_module, tosa_spec)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_function(wrap_pyfunction!(lower_to_tosa_v2, m)?)?;
    m.add_function(wrap_pyfunction!(ingest_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_submodule_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_cond_passes, m)?)?;
    // Control-flow pipeline (new ordering)
    m.add_function(wrap_pyfunction!(stamp_control_flow_structural, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_control_flow_qparams, m)?)?;
    m.add_function(wrap_pyfunction!(cf_insert_boundary_rescales, m)?)?;
    m.add_function(wrap_pyfunction!(cf_run_boundary_phase, m)?)?;
    m.add_function(wrap_pyfunction!(cf_run_output_normalize_phase, m)?)?;
    // Utilities
    m.add_function(wrap_pyfunction!(fold_qdq_fx, m)?)?;
    m.add_function(wrap_pyfunction!(insert_rescale_fx, m)?)?;
    m.add_function(wrap_pyfunction!(fx_graph_summary, m)?)?;
    Ok(())
}
