use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::BTreeMap;

use crate::graph::Graph;

mod config;
mod constant_materialization;
mod control_flow;
mod fx_activation;
mod fx_batch_norm_args;
mod fx_broadcast_args_ir;
mod fx_canonicalize_gather_ir;
mod fx_cast_int64_ir;
mod fx_cast_to_int32_ir;
mod fx_constant_tensor_eval;
mod fx_conv_args;
mod fx_convert_clamp_ir;
mod fx_convert_full_like_ir;
mod fx_convert_split_to_slice_ir;
mod fx_convert_sum_ir;
mod fx_decompose_adaptive_avgpool_ir;
mod fx_decompose_addmm_ir;
mod fx_decompose_as_strided_copy_ir;
mod fx_decompose_avgpool_pad_ir;
mod fx_decompose_bool_equality_ir;
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
mod fx_decompose_slice_scatter_ir;
mod fx_decompose_strided_slice_ir;
mod fx_decompose_unfold_ir;
mod fx_decompose_var_ir;
mod fx_decorate_fp_int_cast_ir;
mod fx_ensure_unique_output_nodes_ir;
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
mod fx_insert_int32_rescale_ir;
mod fx_insert_qdq_rescale_ir;
mod fx_insert_table_ops_ir;
mod fx_ir;
mod fx_legalize_integer_arithmetic_ir;
mod fx_meta;
mod fx_normalize_index_put_bool_mask_ir;
mod fx_normalize_index_put_none_indices_ir;
mod fx_normalize_maxpool_rank_ir;
mod fx_pipeline;
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
/// `RETAB_FOLD_FALLBACK_FILE` (no-op when unset). Used to sweep the battery for
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

/// Align each control-flow region's IO with the graph that invokes it
/// (`control_flow::apply_boundary_contracts`). captured_conv runs inside and can ask
/// for a stock fallback: a captured weight that does not resolve to a constant, a bias
/// that is not int32, or a while loop's capture. The pass pipelines' checks are behind
/// us, so this returns that request's reason for the caller to raise (P2 critique B6).
fn apply_control_flow_contracts(
    ir: &mut fx_ir::IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
) -> Result<(), String> {
    let changed = control_flow::apply_boundary_contracts(ir, submodules);
    if crate::config::from_ir_debug_enabled() && changed > 0 {
        eprintln!(
            "[RETAB][cf-boundary-ir] applied {changed} parent/submodule boundary metadata update(s)"
        );
    }
    stock_fallback::take().map_or(Ok(()), Err)
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

    let has_control_flow = ir_has_control_flow(&ir);
    if has_control_flow {
        apply_control_flow_contracts(&mut ir, &mut stashed.submodules)
            .map_err(|reason| PyRuntimeError::new_err(format!("RETAB_STOCK_FALLBACK: {reason}")))?;
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
    let serialized = serialize::serialize_graphs_to_tosa_py(py, &graphs, &store, &pipeline_config);
    // A fallback requested during ingest or serialization wins over a later error, which
    // is usually its consequence (B6).
    fail_on_stock_fallback_request()?;
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

/// `callable(shape, dims, dtype) -> bool`, Python's exact U55 TRANSPOSE check
/// (`preprocess._u55_transpose_probe`), as the U55 split's probe. The pipeline runs on
/// the thread that holds the GIL, so a call reattaches at no cost. The first exception
/// goes to `error`, for `run_fx_passes` to raise once the pipeline returns; the failed
/// call counts as a rejection meanwhile, as a failed probe does in rc4's check.
fn python_transpose_probe(
    callable: Py<PyAny>,
    error: std::rc::Rc<std::cell::RefCell<Option<PyErr>>>,
) -> Box<fx_decompose_permute_for_u55_ir::TransposeProbe> {
    Box::new(move |shape, perm, dtype| {
        Python::attach(|py| {
            callable
                .call1(py, (shape, perm, dtype))
                .and_then(|verdict| verdict.extract::<bool>(py))
                .unwrap_or_else(|err| {
                    error.borrow_mut().get_or_insert(err);
                    true
                })
        })
    })
}

/// Run the parent lowering pipeline and stash the resulting IrGraph. On U55 Python
/// passes `u55_transpose_probe`, its exact U55 TRANSPOSE check, which the U55
/// permute split asks about each live permute that breaks the worst-case rule
/// (`fx_decompose_permute_for_u55_ir`). U55 has no control flow, so the region entry
/// points take none.
#[pyfunction]
#[pyo3(signature = (graph_module, exported_program=None, *, pipeline_config, u55_transpose_probe=None))]
fn run_fx_passes(
    graph_module: &Bound<'_, PyAny>,
    exported_program: Option<&Bound<'_, PyAny>>,
    pipeline_config: &Bound<'_, PyDict>,
    u55_transpose_probe: Option<Py<PyAny>>,
) -> PyResult<bool> {
    let pipeline_config = pipeline_config::PipelineConfig::from_py(pipeline_config)?;
    let probe_error = std::rc::Rc::new(std::cell::RefCell::new(None));
    let probe = u55_transpose_probe
        .map(|callable| python_transpose_probe(callable, std::rc::Rc::clone(&probe_error)));
    let lowered = fx_decompose_permute_for_u55_ir::with_transpose_probe(probe, || {
        run_full_fx_pipeline(
            graph_module,
            clone_bound_pyobject(exported_program),
            &pipeline_config,
        )
    });
    if let Some(err) = probe_error.take() {
        return Err(err);
    }
    let (modified, ir, effects) = lowered?;
    stash_ir(ir.map(|g| (g, effects)), pipeline_config);
    Ok(modified)
}

/// Run a control-flow region's pipeline `steps` over `ir`, as `fx_ir::run_ir_pipeline`
/// does.
///
/// Critique B2: a region's placeholders are its runtime inputs, which the boundary
/// contracts and captured_conv pair with the invoking cond's or while loop's operands
/// by position. A pass that added, dropped or reordered one would shift that pairing
/// and miscompile silently. No pass does so (region constants are `get_attr` nodes,
/// `layout::ctx::GraphScope::Region`), so this checks that the pipeline kept them,
/// names and order, and requests a stock fallback if it did not.
fn run_region_pipeline(
    region: Option<&str>,
    steps: &[fx_ir::PipelineStep],
    ir: fx_ir::IrGraph,
) -> (bool, fx_ir::IrGraph, lowering_effects::LoweringEffectBatch) {
    let before = ir.live_placeholder_names();
    let (modified, ir, effects) = fx_ir::run_ir_pipeline(steps, ir);
    let after = ir.live_placeholder_names();
    if after != before {
        stock_fallback::request(format!(
            "region placeholders changed: region {}: the pipeline turned placeholders \
             {before:?} into {after:?}, which would shift the boundary operands",
            region.unwrap_or("<unnamed>")
        ));
    }
    (modified, ir, effects)
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
    let pipeline = build(&pipeline_config);
    stock_fallback::take();
    let mut initial_ir = fx_ir::IrGraph::from_fx_with_tensors(graph_module, exported_program)?;
    let propagation_changed =
        apply_pre_submodule_qparam_propagation_ir(region_name.as_deref(), &mut initial_ir);
    let (modified, ir, effects) =
        run_region_pipeline(region_name.as_deref(), &pipeline, initial_ir);
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
    use crate::pipeline_config::PipelineConfig;

    #[test]
    fn stashed_hierarchy_carries_the_pipeline_config_to_serialization() {
        let lowered = || {
            (
                fx_ir::IrGraph::empty(),
                lowering_effects::LoweringEffectBatch::default(),
            )
        };
        for spec in ["TOSA-1.0+INT+cf", "TOSA-1.0+INT+int16+int4+u55"] {
            let config = PipelineConfig::from_tosa_spec(spec);
            stash_ir(Some(lowered()), config.clone());
            stash_submodule_ir(Some("true_graph".to_owned()), Some(lowered()));
            let stashed = take_stashed_hierarchy();
            assert!(stashed.parent.is_some());
            assert!(stashed.submodules.contains_key("true_graph"));
            assert_eq!(stashed.pipeline_config, Some(config));
        }
    }

    /// Critique B2: a region pipeline that changes the region's placeholders requests a
    /// stock fallback, as a region-blind pass would: it makes a constant with the
    /// parent's scope, a placeholder, here at the graph head, which shifts the boundary
    /// operands. A pipeline that keeps the placeholders requests nothing.
    #[test]
    fn region_pipelines_that_change_the_placeholders_go_to_stock() {
        use crate::graph::ir::DType;
        use crate::layout::ctx::{GraphScope, InsertPoint, LayoutCtx};
        use crate::lowering_effects::LoweringInputKind;
        use crate::store::TensorData;
        use crate::test_ir::{output, placeholder};

        struct HeadConstantPass;
        impl fx_ir::IrPass for HeadConstantPass {
            fn name(&self) -> &str {
                "HeadConstant"
            }

            fn run(&self, graph: &mut fx_ir::IrGraph) -> bool {
                let mut ctx = LayoutCtx::with_scope(graph, GraphScope::Parent);
                let zero_point = TensorData {
                    data: vec![0],
                    shape: vec![1],
                    dtype: DType::Int8,
                };
                ctx.with_insert_point(InsertPoint::Start, |ctx| {
                    ctx.create_constant("avg_pool_zp", LoweringInputKind::Buffer, zero_point)
                });
                ctx.finish();
                true
            }
        }

        let region = || {
            let mut region = fx_ir::IrGraph::empty();
            let x = placeholder(&mut region, "x_ph", [1, 2, 8], "torch.int8");
            placeholder(&mut region, "w_ph", [4, 2, 3], "torch.int8");
            output(&mut region, x);
            region
        };
        let head_constant = [fx_ir::PipelineStep::Ir(Box::new(HeadConstantPass))];
        stock_fallback::take();

        let (modified, ir, _) = run_region_pipeline(Some("true_graph"), &head_constant, region());
        assert!(modified);
        assert_eq!(ir.live_placeholder_names(), ["avg_pool_zp", "x_ph", "w_ph"]);
        assert_eq!(
            stock_fallback::take().as_deref(),
            Some(
                "region placeholders changed: region true_graph: the pipeline turned \
                 placeholders [\"x_ph\", \"w_ph\"] into [\"avg_pool_zp\", \"x_ph\", \"w_ph\"], \
                 which would shift the boundary operands"
            )
        );

        let (_, ir, _) = run_region_pipeline(Some("true_graph"), &[], region());
        assert_eq!(ir.live_placeholder_names(), ["x_ph", "w_ph"]);
        assert_eq!(stock_fallback::take(), None);
    }
}
