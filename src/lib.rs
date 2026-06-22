use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::BTreeMap;

use crate::graph::Graph;

mod config;
mod constant_materialization;
mod control_flow;
mod fx_activation;
mod fx_annotate_conv_bias_ir;
mod fx_annotate_output_dim_order_ir;
mod fx_batch_norm_args;
mod fx_canonicalize_gather_ir;
mod fx_cast_int64_ir;
mod fx_constant_tensor_eval;
mod fx_conv1d_unsqueeze_ir;
mod fx_conv_args;
mod fx_convert_clamp_ir;
mod fx_convert_full_like_ir;
mod fx_convert_split_to_slice_ir;
mod fx_convert_sum_ir;
mod fx_decompose_adaptive_avgpool_ir;
mod fx_decompose_addmm_ir;
mod fx_decompose_as_strided_copy_ir;
mod fx_decompose_avgpool_pad_ir;
mod fx_decompose_cumsum_ir;
mod fx_decompose_embedding_ir;
mod fx_decompose_grouped_conv_ir;
mod fx_decompose_index_select_ir;
mod fx_decompose_index_tensor_ir;
mod fx_decompose_layernorm_ir;
mod fx_decompose_linear_ir;
mod fx_decompose_logit_ir;
mod fx_decompose_masked_fill_ir;
mod fx_decompose_maxpool_dilation_ir;
mod fx_decompose_mean_ir;
mod fx_decompose_ne_ir;
mod fx_decompose_ops_ir;
mod fx_decompose_pixel_shuffle_ir;
mod fx_decompose_quant_ir;
mod fx_decompose_repeat_interleave_ir;
mod fx_decompose_select_scatter_ir;
mod fx_decompose_slice_scatter_ir;
mod fx_decompose_strided_slice_ir;
mod fx_decompose_tril_ir;
mod fx_decompose_unfold_ir;
mod fx_decompose_var_ir;
mod fx_decorate_fp_int_cast_ir;
mod fx_expand_where_mask_ir;
mod fx_fold_qdq_ir;
mod fx_fuse_batch_norm_ir;
mod fx_fuse_constant_inputs_ir;
mod fx_fuse_duplicate_users_ir;
mod fx_fuse_equal_placeholders_ir;
mod fx_fuse_quant_activation_ir;
mod fx_fuse_shape_only_constants_ir;
mod fx_fuse_zero_input_constants_ir;
mod fx_group_norm_args;
mod fx_index_domain;
mod fx_index_linear;
mod fx_insert_conv_rescale_ir;
mod fx_insert_int32_rescale_ir;
mod fx_insert_qdq_rescale_ir;
mod fx_insert_table_ops_ir;
mod fx_ir;
mod fx_memory_format_ir;
mod fx_meta;
mod fx_normalize_index_put_bool_mask_ir;
mod fx_normalize_index_put_none_indices_ir;
mod fx_pipeline;
mod fx_py_attr;
mod fx_quant;
mod fx_quantize_clamp_args_ir;
mod fx_reduction;
mod fx_remove_noop_ir;
mod fx_replace_inf_values_ir;
mod fx_replace_scalar_with_tensor_ir;
mod fx_rewrite_index_put_ir;
mod fx_rewrite_matmul_ir;
mod fx_rewrite_pad_ir;
mod fx_rewrite_squeeze_ir;
mod fx_size_adjust_input_ir;
mod fx_target;
mod graph;
mod ingest;
#[cfg(test)]
mod ir_golden_tests;
mod lowering_effects;
mod runner_logging;
mod serialize;
mod store;
mod tensor_data;
#[cfg(test)]
mod test_ir;

fn clone_bound_pyobject(exported_program: Option<&Bound<'_, PyAny>>) -> Option<PyObject> {
    exported_program.map(|ep| ep.clone().unbind())
}

/// The post-pipeline IrGraph plus the lowering effects accumulated during it.
pub(crate) type StashedPipeline = (fx_ir::IrGraph, lowering_effects::LoweringEffectBatch);

#[derive(Default)]
struct StashedPipelineHierarchy {
    parent: Option<StashedPipeline>,
    submodules: BTreeMap<String, StashedPipeline>,
}

/// True if the stashed IrGraph contains a control-flow op (cond / while_loop).
/// Control-flow graphs also need child region graphs assembled from stashed
/// submodule IRs before serialization.
fn ir_has_control_flow(ir: &fx_ir::IrGraph) -> bool {
    use crate::graph::ir::OpTarget;
    ir.live_nodes().any(|n| {
        matches!(
            crate::ingest::target_map::parse_op_target(&n.target),
            OpTarget::CondIf | OpTarget::WhileLoop
        )
    })
}

/// Append a diagnostic constant-fold fallback record to the file named by
/// `ARM_FOLD_FALLBACK_FILE` (no-op when unset). Used to sweep the battery for
/// unsupported Rust constant materialization.
pub(crate) fn fold_fallback_log(msg: &str) {
    use std::io::Write;
    if let Some(path) = crate::config::fold_fallback_file() {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{msg}");
        }
    }
}

thread_local! {
    /// Stash for the final post-pipeline IrGraph + lowering effects, set by the pass
    /// entry points and consumed by serialization without re-reading the Python graph.
    static STASHED_IR: std::cell::RefCell<StashedPipelineHierarchy> =
        std::cell::RefCell::new(StashedPipelineHierarchy::default());
}

fn stash_ir(stashed: Option<StashedPipeline>) {
    STASHED_IR.with(|s| {
        let mut stash = s.borrow_mut();
        stash.parent = stashed;
        stash.submodules.clear();
    });
}

fn stash_submodule_ir(region_name: Option<String>, stashed: Option<StashedPipeline>) {
    let (Some(region_name), Some(stashed)) = (region_name, stashed) else {
        return;
    };
    STASHED_IR.with(|s| {
        s.borrow_mut().submodules.insert(region_name, stashed);
    });
}

fn take_stashed_hierarchy() -> StashedPipelineHierarchy {
    STASHED_IR.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

fn run_full_fx_pipeline(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<PyObject>,
) -> PyResult<(
    bool,
    Option<fx_ir::IrGraph>,
    lowering_effects::LoweringEffectBatch,
)> {
    let py = graph_module.py();
    let exported_program_for_pipeline = exported_program.as_ref().map(|ep| ep.clone_ref(py));
    let pipeline = fx_pipeline::build_hybrid_pipeline(exported_program_for_pipeline);
    let exported_program = exported_program.as_ref().map(|ep| ep.bind(py));
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let structural_changed = control_flow::snapshot_pre_pass_qparams_ir(&mut initial_ir);
    let (modified, ir, effects) = fx_ir::run_hybrid_pipeline_ext_with_initial_ir(
        py,
        graph_module,
        exported_program,
        &pipeline,
        Some(initial_ir),
    )?;
    Ok((modified || structural_changed > 0, ir, effects))
}

fn rebuild_graph_users(graphs: &mut [(String, Graph)]) {
    for (_, graph) in graphs {
        graph.rebuild_user_lists();
    }
}

fn apply_pre_submodule_qparam_propagation_ir(
    region_name: Option<&str>,
    graph: &mut fx_ir::IrGraph,
) -> u32 {
    STASHED_IR.with(|stash| {
        let stash = stash.borrow();
        control_flow::apply_pre_submodule_qparam_propagation_ir(
            region_name,
            graph,
            stash.parent.as_ref(),
            &stash.submodules,
        )
    })
}

fn ingest_and_serialize_tosa_pybytes(
    py: Python<'_>,
    _graph_module: &Bound<'_, PyAny>,
    _state_dict: &Bound<'_, PyDict>,
    _graph_signature: &Bound<'_, PyAny>,
    _exported_program: Option<PyObject>,
) -> PyResult<Py<PyBytes>> {
    let profile = crate::config::profile_ingest_serialize_enabled();
    let mut stashed = take_stashed_hierarchy();
    let Some((mut ir, effects)) = stashed.parent.take() else {
        return Err(PyRuntimeError::new_err(
            "missing stashed IrGraph for serialization",
        ));
    };

    let has_control_flow = ir_has_control_flow(&ir);
    if has_control_flow {
        let changed = control_flow::apply_boundary_contracts(&mut ir, &mut stashed.submodules);
        if crate::config::from_ir_debug_enabled() && changed > 0 {
            eprintln!(
                "[RETAB][cf-boundary-ir] applied {changed} parent/submodule boundary metadata update(s)"
            );
        }
    }

    let mut store = crate::store::TensorStore::new();
    let mut converted = ingest::from_ir::graph_from_ir(&ir, &mut store);
    ingest::from_ir::apply_effects(&mut converted, &mut store, &effects);
    ingest::finalize::finalize_converted_graph(&mut converted);
    if crate::config::from_ir_debug_enabled() {
        ingest::from_ir::debug_dump(&converted, "MAIN FROM IR");
    }
    ingest::from_ir::report_missing_consts(&converted, &store);

    let mut graphs = vec![(String::from("main"), converted)];
    if has_control_flow {
        control_flow::append_subgraphs_from_ir(&mut graphs, &mut store, &ir, &stashed.submodules)?;
    }

    rebuild_graph_users(&mut graphs);
    let py_bytes = serialize::serialize_graphs_to_tosa_py(py, &graphs, &store);
    if profile {
        eprintln!(
            "[RETAB][ingest_and_serialize] from-ir graphs={} nodes={} tensors={}",
            graphs.len(),
            graphs.iter().map(|(_, g)| g.order.len()).sum::<usize>(),
            store.buffers.len(),
        );
    }
    Ok(py_bytes)
}

/// Run the parent lowering pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None))]
fn run_fx_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let (modified, ir, effects) =
        run_full_fx_pipeline(graph_module, clone_bound_pyobject(exported_program))?;
    stash_ir(ir.map(|g| (g, effects)));
    Ok(modified)
}

/// Run the while_loop submodule IR pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, region_name=None))]
fn run_fx_submodule_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    region_name: Option<String>,
) -> PyResult<bool> {
    let pipeline = fx_pipeline::build_ir_while_submodule_pipeline();
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let propagation_changed =
        apply_pre_submodule_qparam_propagation_ir(region_name.as_deref(), &mut initial_ir);
    let (modified, ir, effects) = fx_ir::run_hybrid_pipeline_ext_with_initial_ir(
        graph_module.py(),
        graph_module,
        exported_program,
        &pipeline,
        Some(initial_ir),
    )?;
    stash_submodule_ir(region_name, ir.map(|g| (g, effects)));
    Ok(modified || propagation_changed > 0)
}

/// Run the cond branch submodule IR pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, region_name=None))]
fn run_fx_cond_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    region_name: Option<String>,
) -> PyResult<bool> {
    let pipeline = fx_pipeline::build_ir_cond_submodule_pipeline();
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let propagation_changed =
        apply_pre_submodule_qparam_propagation_ir(region_name.as_deref(), &mut initial_ir);
    let (modified, ir, effects) = fx_ir::run_hybrid_pipeline_ext_with_initial_ir(
        graph_module.py(),
        graph_module,
        exported_program,
        &pipeline,
        Some(initial_ir),
    )?;
    stash_submodule_ir(region_name, ir.map(|g| (g, effects)));
    Ok(modified || propagation_changed > 0)
}

/// Serialize the lowered stashed IR hierarchy to a TOSA flatbuffer.
#[pyfunction]
#[pyo3(signature = (graph_module, state_dict, graph_signature, exported_program=None))]
fn ingest_and_serialize(
    py: Python<'_>,
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyBytes>> {
    ingest_and_serialize_tosa_pybytes(
        py,
        graph_module,
        state_dict,
        graph_signature,
        clone_bound_pyobject(exported_program),
    )
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_function(wrap_pyfunction!(ingest_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_submodule_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_cond_passes, m)?)?;
    Ok(())
}
