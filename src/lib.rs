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
mod fx_broadcast_args_ir;
mod fx_canonicalize_gather_ir;
mod fx_cast_int64_ir;
mod fx_cancel_views_around_unary_ir;
mod fx_cast_to_int32_ir;
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
mod fx_decompose_div_tensor_mode_ir;
mod fx_decompose_embedding_ir;
mod fx_decompose_flip_ir;
mod fx_decompose_grouped_conv_ir;
mod fx_decompose_index_select_ir;
mod fx_decompose_index_tensor_ir;
mod fx_decompose_large_stride_maxpool_ir;
mod fx_decompose_layernorm_ir;
mod fx_decompose_linear_ir;
mod fx_decompose_logit_ir;
mod fx_decompose_masked_fill_ir;
mod fx_decompose_maxpool_dilation_ir;
mod fx_decompose_mean_ir;
mod fx_decompose_ne_ir;
mod fx_decompose_ops_ir;
mod fx_decompose_permute_for_u55_ir;
mod fx_decompose_pixel_shuffle_ir;
mod fx_decompose_quant_ir;
mod fx_decompose_repeat_interleave_ir;
mod fx_decompose_roll_ir;
mod fx_decompose_select_scatter_ir;
mod fx_decompose_slice_scatter_ir;
mod fx_decompose_strided_slice_ir;
mod fx_decompose_tril_ir;
mod fx_decompose_unfold_ir;
mod fx_decompose_var_ir;
mod fx_decorate_fp_int_cast_ir;
mod fx_ensure_unique_output_nodes_ir;
mod fx_expand_where_mask_ir;
mod fx_fold_qdq_ir;
mod fx_fuse_batch_norm_ir;
mod fx_fuse_cascaded_views_ir;
mod fx_fuse_consecutive_clamps_ir;
mod fx_fuse_consecutive_concats_ir;
mod fx_fuse_consecutive_slices_ir;
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
mod fx_legalize_integer_arithmetic_ir;
mod fx_memory_format_ir;
mod fx_meta;
mod fx_normalize_index_put_bool_mask_ir;
mod fx_normalize_index_put_none_indices_ir;
mod fx_normalize_maxpool_rank_ir;
mod fx_pipeline;
mod fx_propagate_slice_views_ir;
mod fx_py_attr;
mod fx_quant;
mod fx_quantize_clamp_args_ir;
mod fx_reduction;
mod fx_remove_noop_ir;
mod fx_replace_inf_values_ir;
mod fx_replace_scalar_with_tensor_ir;
mod fx_rewrite_cat_slice_ir;
mod fx_rewrite_index_put_ir;
mod fx_rewrite_matmul_ir;
mod fx_rewrite_pad_ir;
mod fx_rewrite_slice_ir;
mod fx_rewrite_squeeze_ir;
mod fx_size_adjust_input_ir;
mod fx_target;
mod graph;
mod ingest;
#[cfg(test)]
mod ir_golden_tests;
mod layout;
mod lowering_effects;
mod pipeline_config;
mod runner_logging;
mod serialize;
mod stock_fallback;
mod store;
mod tensor_data;
#[cfg(test)]
mod test_ir;

fn clone_bound_pyobject(exported_program: Option<&Bound<'_, PyAny>>) -> Option<Py<PyAny>> {
    exported_program.map(|ep| ep.clone().unbind())
}

/// The post-pipeline IrGraph plus the lowering effects accumulated during it.
pub(crate) type StashedPipeline = (fx_ir::IrGraph, lowering_effects::LoweringEffectBatch);

#[derive(Default)]
struct StashedPipelineHierarchy {
    parent: Option<StashedPipeline>,
    submodules: BTreeMap<String, StashedPipeline>,
    pipeline_config: Option<pipeline_config::PipelineConfig>,
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

fn stash_ir(stashed: Option<StashedPipeline>, pipeline_config: pipeline_config::PipelineConfig) {
    STASHED_IR.with(|s| {
        let mut stash = s.borrow_mut();
        stash.parent = stashed;
        stash.submodules.clear();
        stash.pipeline_config = Some(pipeline_config);
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

thread_local! {
    /// Main-graph runtime inputs that nothing consumes, from the last serialization.
    static LAST_UNUSED_INPUTS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Main-block inputs that no operator reads and that aren't block outputs. This reads
/// the serialized block because emitters can drop an operand the graph still lists
/// (e.g. the overwritten destination of `copy_`).
fn unused_main_block_inputs(tosa: &[u8]) -> Vec<String> {
    let Ok(graph) = tosa_rs::generated::root_as_tosa_graph(tosa) else {
        return Vec::new();
    };
    let Some(block) = graph
        .regions()
        .and_then(|regions| regions.iter().next())
        .and_then(|region| region.blocks())
        .and_then(|blocks| blocks.iter().next())
    else {
        return Vec::new();
    };
    let mut consumed: std::collections::HashSet<&str> = block
        .operators()
        .into_iter()
        .flatten()
        .filter_map(|op| op.inputs())
        .flatten()
        .collect();
    consumed.extend(block.outputs().into_iter().flatten());
    block
        .inputs()
        .into_iter()
        .flatten()
        .filter(|name| !consumed.contains(name))
        .map(str::to_string)
        .collect()
}

fn take_stashed_hierarchy() -> StashedPipelineHierarchy {
    STASHED_IR.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Fail with a `RETAB_STOCK_FALLBACK:` error (the Python seam then calls stock) when
/// someone called `stock_fallback::request` since the last check.
fn fail_on_stock_fallback_request() -> PyResult<()> {
    match stock_fallback::take() {
        Some(reason) => Err(PyRuntimeError::new_err(format!(
            "RETAB_STOCK_FALLBACK: {reason}"
        ))),
        None => Ok(()),
    }
}

/// Fail with a `RETAB_STOCK_FALLBACK:` error (the Python seam then calls stock) when a
/// pass requested it or a quantized table op has no planned TABLE.
fn ensure_retab_can_lower(
    ir: Option<&fx_ir::IrGraph>,
    effects: &lowering_effects::LoweringEffectBatch,
) -> PyResult<()> {
    fail_on_stock_fallback_request()?;
    let Some(graph) = ir else {
        return Ok(());
    };
    let missing = crate::fx_insert_table_ops_ir::unsupported_table_node_names(graph, effects);
    if missing.is_empty() {
        return Ok(());
    }
    Err(PyRuntimeError::new_err(format!(
        "RETAB_STOCK_FALLBACK: unsupported table op: {} would emit integer TOSA SIGMOID/TANH/etc",
        missing.join(", ")
    )))
}

fn run_full_fx_pipeline(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<Py<PyAny>>,
    pipeline_config: &pipeline_config::PipelineConfig,
) -> PyResult<(
    bool,
    Option<fx_ir::IrGraph>,
    lowering_effects::LoweringEffectBatch,
)> {
    let py = graph_module.py();
    stock_fallback::take();
    let pipeline = fx_pipeline::build_main_pipeline(pipeline_config);
    let exported_program = exported_program.as_ref().map(|ep| ep.bind(py));
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let structural_changed = control_flow::snapshot_pre_pass_qparams_ir(&mut initial_ir);
    let (modified, ir, effects) = fx_ir::run_ir_pipeline(&pipeline, initial_ir);
    ensure_retab_can_lower(Some(&ir), &effects)?;
    Ok((modified || structural_changed > 0, Some(ir), effects))
}

fn rebuild_graph_users(graphs: &mut [(String, Graph)]) {
    for (_, graph) in graphs {
        graph.rebuild_user_lists();
    }
}

fn apply_pre_submodule_qparam_propagation_ir(
    region_name: Option<&str>,
    graph: &mut fx_ir::IrGraph,
    layout: pipeline_config::LayoutMode,
) -> u32 {
    STASHED_IR.with(|stash| {
        let stash = stash.borrow();
        control_flow::apply_pre_submodule_qparam_propagation_ir(
            region_name,
            graph,
            stash.parent.as_ref(),
            &stash.submodules,
            layout,
        )
    })
}

/// Align each control-flow region's IO with the graph that invokes it
/// (`control_flow::apply_boundary_contracts`). Explicit mode's captured_conv runs inside
/// and can ask for a stock fallback: a captured weight that does not resolve to a
/// constant, or a bias that is not int32 data. The pass pipelines' checks are behind
/// us, so this returns that request's reason for the caller to raise (P2 critique B6).
/// Dim-order mode's contracts request nothing.
fn apply_control_flow_contracts(
    ir: &mut fx_ir::IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
    layout: pipeline_config::LayoutMode,
) -> Result<(), String> {
    let changed = control_flow::apply_boundary_contracts(ir, submodules, layout);
    if crate::config::from_ir_debug_enabled() && changed > 0 {
        eprintln!(
            "[RETAB][cf-boundary-ir] applied {changed} parent/submodule boundary metadata update(s)"
        );
    }
    match layout {
        pipeline_config::LayoutMode::Explicit => stock_fallback::take().map_or(Ok(()), Err),
        pipeline_config::LayoutMode::DimOrder => Ok(()),
    }
}

fn ingest_and_serialize_tosa_pybytes(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    let started = std::time::Instant::now();
    let profile = crate::config::profile_ingest_serialize_enabled();
    let mut stashed = take_stashed_hierarchy();
    let Some((mut ir, effects)) = stashed.parent.take() else {
        return Err(PyRuntimeError::new_err(
            "missing stashed IrGraph for serialization",
        ));
    };
    let pipeline_config = stashed.pipeline_config.take().ok_or_else(|| {
        PyRuntimeError::new_err("missing stashed pipeline config for serialization")
    })?;

    let layout = pipeline_config.layout;
    let explicit = layout == pipeline_config::LayoutMode::Explicit;
    let has_control_flow = ir_has_control_flow(&ir);
    if has_control_flow {
        apply_control_flow_contracts(&mut ir, &mut stashed.submodules, layout).map_err(
            |reason| PyRuntimeError::new_err(format!("RETAB_STOCK_FALLBACK: {reason}")),
        )?;
    }

    let mut store = crate::store::TensorStore::new();
    let mut converted = ingest::from_ir::graph_from_ir(&ir, &mut store);
    converted.layout = layout;
    ingest::from_ir::apply_effects(&mut converted, &mut store, &effects);
    ingest::finalize::finalize_converted_graph(&mut converted);
    if crate::config::from_ir_debug_enabled() {
        ingest::from_ir::debug_dump(&converted, "MAIN FROM IR");
    }
    ingest::from_ir::report_missing_consts(&converted, &store);

    let mut graphs = vec![(String::from("main"), converted)];
    if has_control_flow {
        control_flow::append_subgraphs_from_ir(
            &mut graphs,
            &mut store,
            &ir,
            &stashed.submodules,
            layout,
        )?;
    }

    rebuild_graph_users(&mut graphs);
    let serialized = serialize::serialize_graphs_to_tosa_py(py, &graphs, &store, &pipeline_config);
    // A fallback requested during ingest or serialization wins over a later error, which
    // is usually its consequence (B6).
    if explicit {
        fail_on_stock_fallback_request()?;
    }
    let py_bytes = serialized?;
    let unused = unused_main_block_inputs(py_bytes.bind(py).as_bytes());
    LAST_UNUSED_INPUTS.with(|slot| *slot.borrow_mut() = unused);
    if profile {
        eprintln!(
            "[RETAB][ingest_and_serialize] from-ir graphs={} nodes={} tensors={}",
            graphs.len(),
            graphs.iter().map(|(_, g)| g.order.len()).sum::<usize>(),
            store.buffers.len(),
        );
    }
    if std::env::var_os("RETAB_PASS_TIMING").is_some() {
        eprintln!(
            "[RETAB][pass-timing] ingest and serialize: {:.2} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(py_bytes)
}

/// Run the parent lowering pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, *, pipeline_config))]
fn run_fx_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    pipeline_config: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let pipeline_config = pipeline_config::PipelineConfig::from_py(pipeline_config)?;
    let (modified, ir, effects) = run_full_fx_pipeline(
        graph_module,
        clone_bound_pyobject(exported_program),
        &pipeline_config,
    )?;
    stash_ir(ir.map(|g| (g, effects)), pipeline_config);
    Ok(modified)
}

/// Why a region pipeline in `layout` must not join the stashed hierarchy, if it must
/// not: the parent pipeline stashed for serialization ran the other layout mode (P2
/// design §4.1, "Mixed modes"). One mode drives the whole hierarchy: the region
/// pipelines, the boundary contracts, captured_conv, ingest and the serializer all take
/// the stashed parent's.
///
/// Mixed modes cannot arise through `retab.preprocess`, the only caller that answers a
/// `RETAB_STOCK_FALLBACK:` error: `_all_rust_preprocess` builds one pipeline config per
/// delegate, hands that same dict to `run_fx_passes` and to every region entry point,
/// and runs the regions only once `run_fx_passes` has stashed that delegate's parent.
/// A mismatch therefore means a caller drove the native entry points with two configs.
/// That is a bug, not an input stock could take, so it is a hard error: a stock fallback
/// would hide it.
fn mixed_layout_error(region: Option<&str>, layout: pipeline_config::LayoutMode) -> Option<String> {
    let parent = STASHED_IR.with(|s| s.borrow().pipeline_config.as_ref().map(|c| c.layout))?;
    (parent != layout).then(|| {
        format!(
            "control-flow region {} runs the {} layout, but the parent pipeline stashed for it \
             ran {}: a delegate's parent and regions must share one pipeline config",
            region.unwrap_or("<unnamed>"),
            layout.as_str(),
            parent.as_str()
        )
    })
}

/// A graph's live placeholders, by name and in order.
fn placeholder_names(graph: &fx_ir::IrGraph) -> Vec<String> {
    graph
        .live_placeholder_ids()
        .into_iter()
        .map(|id| graph.node(id).name.clone())
        .collect()
}

/// Critique B2: a region's placeholders are its runtime inputs, which the boundary
/// contracts and captured_conv pair with the invoking cond's or while loop's operands
/// by position. A pass that added, dropped or reordered one would shift that pairing
/// and miscompile silently. No explicit pass does so (region constants are `get_attr`
/// nodes, `layout::ctx::GraphScope::Region`), so explicit mode checks that a region
/// pipeline kept them, names and order, and sends the graph to stock if it did not.
/// Returns that fallback's reason.
fn region_placeholders_changed(
    region: Option<&str>,
    before: &[String],
    after: &fx_ir::IrGraph,
) -> Option<String> {
    let after = placeholder_names(after);
    (after != before).then(|| {
        format!(
            "region placeholders changed: region {}: the pipeline turned placeholders {before:?} \
             into {after:?}, which would shift the boundary operands",
            region.unwrap_or("<unnamed>")
        )
    })
}

/// Run a control-flow region's pipeline, from `build`, and stash the resulting IrGraph
/// for serialization.
fn run_region_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    region_name: Option<String>,
    pipeline_config: &Bound<'_, PyDict>,
    build: fn(&pipeline_config::PipelineConfig) -> Vec<fx_ir::PipelineStep>,
) -> PyResult<bool> {
    let pipeline_config = pipeline_config::PipelineConfig::from_py(pipeline_config)?;
    if let Some(error) = mixed_layout_error(region_name.as_deref(), pipeline_config.layout) {
        return Err(PyRuntimeError::new_err(error));
    }
    let pipeline = build(&pipeline_config);
    stock_fallback::take();
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let propagation_changed = apply_pre_submodule_qparam_propagation_ir(
        region_name.as_deref(),
        &mut initial_ir,
        pipeline_config.layout,
    );
    let placeholders = (pipeline_config.layout == pipeline_config::LayoutMode::Explicit)
        .then(|| placeholder_names(&initial_ir));
    let (modified, ir, effects) = fx_ir::run_ir_pipeline(&pipeline, initial_ir);
    if let Some(reason) = placeholders
        .and_then(|before| region_placeholders_changed(region_name.as_deref(), &before, &ir))
    {
        stock_fallback::request(reason);
    }
    ensure_retab_can_lower(Some(&ir), &effects)?;
    stash_submodule_ir(region_name, Some((ir, effects)));
    Ok(modified || propagation_changed > 0)
}

/// Run the while_loop submodule IR pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, region_name=None, *, pipeline_config))]
fn run_fx_submodule_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    region_name: Option<String>,
    pipeline_config: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    run_region_passes(
        graph_module,
        exported_program,
        region_name,
        pipeline_config,
        fx_pipeline::build_ir_while_submodule_pipeline,
    )
}

/// Run the cond branch submodule IR pipeline and stash the resulting IrGraph.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, region_name=None, *, pipeline_config))]
fn run_fx_cond_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    region_name: Option<String>,
    pipeline_config: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    run_region_passes(
        graph_module,
        exported_program,
        region_name,
        pipeline_config,
        fx_pipeline::build_ir_cond_submodule_pipeline,
    )
}

/// Runtime inputs of the last serialized main graph that no op consumes. Ethos-U
/// binds delegate arguments by position after Vela drops unused inputs, so its
/// backend must reject these.
#[pyfunction]
fn last_unused_inputs() -> Vec<String> {
    LAST_UNUSED_INPUTS.with(|slot| slot.borrow().clone())
}

/// Serialize the IR hierarchy stashed by the `run_fx_*` entry points to a TOSA flatbuffer.
#[pyfunction]
fn ingest_and_serialize(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    ingest_and_serialize_tosa_pybytes(py)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_function(wrap_pyfunction!(ingest_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_submodule_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_cond_passes, m)?)?;
    m.add_function(wrap_pyfunction!(last_unused_inputs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_config::{LayoutMode, PipelineConfig};

    #[test]
    fn stashed_hierarchy_carries_the_layout_mode_to_serialization() {
        let lowered = || {
            (
                fx_ir::IrGraph::empty(),
                lowering_effects::LoweringEffectBatch::default(),
            )
        };
        for layout in [LayoutMode::Explicit, LayoutMode::DimOrder] {
            let config = PipelineConfig {
                layout,
                ..PipelineConfig::from_tosa_spec("TOSA-1.0+INT+cf")
            };
            stash_ir(Some(lowered()), config);
            stash_submodule_ir(Some("true_graph".to_owned()), Some(lowered()));
            let stashed = take_stashed_hierarchy();
            assert!(stashed.parent.is_some());
            assert!(stashed.submodules.contains_key("true_graph"));
            assert_eq!(
                stashed.pipeline_config.map(|config| config.layout),
                Some(layout)
            );
        }
    }

    /// P2 design §4.1: a region pipeline in the other layout mode than the stashed
    /// parent's is rejected, with a hard error rather than a stock fallback (only a
    /// caller driving the native entry points with two configs gets there). Without a
    /// stashed parent there is nothing to compare.
    #[test]
    fn region_pipelines_must_run_the_stashed_parents_layout_mode() {
        let config = |layout| PipelineConfig {
            layout,
            ..PipelineConfig::from_tosa_spec("TOSA-1.0+INT+cf")
        };
        take_stashed_hierarchy();
        for layout in [LayoutMode::DimOrder, LayoutMode::Explicit] {
            assert_eq!(mixed_layout_error(Some("true_graph"), layout), None);
        }
        for (parent, region) in [
            (LayoutMode::Explicit, LayoutMode::DimOrder),
            (LayoutMode::DimOrder, LayoutMode::Explicit),
        ] {
            stash_ir(None, config(parent));
            assert_eq!(mixed_layout_error(Some("true_graph"), parent), None);
            let error = mixed_layout_error(Some("true_graph"), region).expect("a mismatch");
            assert_eq!(
                error,
                format!(
                    "control-flow region true_graph runs the {} layout, but the parent pipeline \
                     stashed for it ran {}: a delegate's parent and regions must share one \
                     pipeline config",
                    region.as_str(),
                    parent.as_str()
                )
            );
            assert!(!error.contains("RETAB_STOCK_FALLBACK"), "{error}");
        }
        take_stashed_hierarchy();
    }

    /// Critique B2: explicit mode sends a region to stock when its pipeline added,
    /// dropped or reordered a placeholder, which would shift the boundary operands, as a
    /// region constant made as a placeholder at the graph head would.
    #[test]
    fn region_placeholder_changes_are_reported() {
        use crate::test_ir::{output, placeholder};

        let region = |head_constant: bool| {
            let mut region = fx_ir::IrGraph::empty();
            if head_constant {
                placeholder(&mut region, "avg_pool_zp", [1], "torch.int8");
            }
            let x = placeholder(&mut region, "x_ph", [1, 2, 8], "torch.int8");
            placeholder(&mut region, "w_ph", [4, 2, 3], "torch.int8");
            output(&mut region, x);
            region
        };
        let before = placeholder_names(&region(false));
        assert_eq!(before, ["x_ph", "w_ph"]);
        assert_eq!(
            region_placeholders_changed(Some("true_graph"), &before, &region(false)),
            None
        );

        let reason = region_placeholders_changed(Some("true_graph"), &before, &region(true))
            .expect("a changed placeholder list");
        assert_eq!(
            reason,
            "region placeholders changed: region true_graph: the pipeline turned placeholders \
             [\"x_ph\", \"w_ph\"] into [\"avg_pool_zp\", \"x_ph\", \"w_ph\"], which would shift \
             the boundary operands"
        );
    }
}
