use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{BTreeMap, BTreeSet};

use crate::graph::ir::{Arg, DType, NodeId};
use crate::graph::Graph;
use crate::store::TensorStore;

mod fx_annotate_conv_bias_ir;
mod fx_annotate_output_dim_order_ir;
mod fx_canonicalize_gather_ir;
mod fx_cast_int64_ir;
mod fx_constant_tensor_eval;
mod fx_control_flow_metadata;
mod fx_conv1d_unsqueeze_ir;
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
mod fx_ep_effects;
mod fx_expand_where_mask_ir;
mod fx_fold_qdq_ir;
mod fx_fuse_constant_args;
mod fx_fuse_constant_inputs_ir;
mod fx_fuse_duplicate_users_ir;
mod fx_fuse_equal_placeholders_ir;
mod fx_fuse_quant_activation_ir;
mod fx_fuse_shape_only_constants_ir;
mod fx_fuse_zero_input_constants_ir;
mod fx_graph;
mod fx_insert_conv_rescale_ir;
mod fx_insert_int32_rescale_ir;
mod fx_insert_qdq_rescale_ir;
mod fx_insert_table_ops_ir;
mod fx_ir;
mod fx_memory_format_ir;
mod fx_normalize_index_put_bool_mask_ir;
mod fx_normalize_index_put_none_indices_ir;
mod fx_pass;
mod fx_pipeline;
mod fx_quantize_clamp_args_ir;
mod fx_remove_noop_ir;
mod fx_replace_inf_values_ir;
mod fx_replace_scalar_with_tensor_ir;
mod fx_rewrite_index_put_ir;
mod fx_rewrite_matmul_ir;
mod fx_rewrite_pad_ir;
mod fx_rewrite_squeeze_ir;
mod fx_size_adjust_input_ir;
mod graph;
mod ingest;
mod runner_logging;
mod serialize;
mod store;
mod tensor_data;

fn profile_ingest_serialize_enabled() -> bool {
    std::env::var_os("RETAB_PROFILE_INGEST_SERIALIZE").is_some()
}

fn clone_bound_pyobject(exported_program: Option<&Bound<'_, PyAny>>) -> Option<PyObject> {
    exported_program.map(|ep| ep.clone().unbind())
}

/// The post-pipeline IrGraph plus the EP-effects accumulated during it.
type StashedPipeline = (fx_ir::IrGraph, fx_ep_effects::FxEpEffectBatch);

#[derive(Default)]
struct StashedPipelineHierarchy {
    parent: Option<StashedPipeline>,
    submodules: BTreeMap<String, StashedPipeline>,
}

/// Append a Python-deferred constant-fold fallback record to the file named by
/// `ARM_FOLD_FALLBACK_FILE` (no-op when unset). Used to sweep the battery for any
/// remaining round-trip dependency on Python computation (round-trip Phase 4).
/// True if the stashed IrGraph contains a control-flow op (cond / while_loop).
/// Such graphs still need the Python ingest path: their submodule hierarchy and
/// cf boundary phases are not yet captured in the IrGraph (round-trip Phase 4c).
fn ir_has_control_flow(ir: &fx_ir::IrGraph) -> bool {
    use crate::graph::ir::OpTarget;
    ir.nodes.iter().filter(|n| n.alive).any(|n| {
        matches!(
            crate::ingest::target_map::parse_op_target(&n.target),
            OpTarget::CondIf | OpTarget::WhileLoop
        )
    })
}

pub(crate) fn fold_fallback_log(msg: &str) {
    use std::io::Write;
    if let Ok(path) = std::env::var("ARM_FOLD_FALLBACK_FILE") {
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
    /// Stash for the final post-pipeline IrGraph + EP-effects, set by the pass
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
) -> PyResult<(bool, Option<fx_ir::IrGraph>, fx_ep_effects::FxEpEffectBatch)> {
    // Use hybrid pipeline: persistent IrGraph for Rust passes,
    // flush/reimport only at FxPass boundaries.
    let py = graph_module.py();
    let exported_program_for_pipeline = exported_program.as_ref().map(|ep| ep.clone_ref(py));
    let pipeline = fx_pipeline::build_hybrid_pipeline(exported_program_for_pipeline);
    let exported_program = exported_program.as_ref().map(|ep| ep.bind(py));
    fx_ir::run_hybrid_pipeline(py, graph_module, exported_program, &pipeline)
}

fn rebuild_graph_users(graphs: &mut [(String, Graph)]) {
    for (_, graph) in graphs {
        graph.rebuild_user_lists();
    }
}

#[derive(Clone, Copy)]
struct IrWhileBodyInputCounts {
    carried: usize,
    additional: usize,
}

#[derive(Clone)]
struct IrBoundaryInputMeta {
    meta: fx_ir::NodeMeta,
    boundary_dim_order: Option<Vec<usize>>,
    fallback_qparam: Option<fx_ir::QParam>,
}

#[derive(Clone)]
struct IrCfBoundaryContract {
    region: String,
    inputs: Vec<IrBoundaryInputMeta>,
}

#[derive(Clone)]
struct IrCfOutputContract {
    region: String,
    targets: Vec<(usize, fx_ir::QParam)>,
}

fn qualify_region(parent_region: Option<&str>, local_name: &str) -> String {
    match parent_region {
        Some(prefix) if !local_name.contains("::") => format!("{prefix}::{local_name}"),
        _ => local_name.to_string(),
    }
}

fn region_name_from_ir_arg(
    ir: &fx_ir::IrGraph,
    arg: Option<&fx_ir::IrArg>,
    parent_region: Option<&str>,
) -> Option<String> {
    match arg {
        Some(fx_ir::IrArg::Str(name)) => Some(qualify_region(parent_region, name)),
        Some(fx_ir::IrArg::Node(id)) => ir
            .nodes
            .iter()
            .find(|node| node.id == *id)
            .map(|node| qualify_region(parent_region, &node.name)),
        _ => None,
    }
}

fn node_ids_from_ir_arg(arg: Option<&fx_ir::IrArg>) -> Vec<fx_ir::NodeId> {
    fn collect(arg: &fx_ir::IrArg, out: &mut Vec<fx_ir::NodeId>) {
        match arg {
            fx_ir::IrArg::Node(id) => out.push(*id),
            fx_ir::IrArg::List(items) => {
                for item in items {
                    collect(item, out);
                }
            }
            _ => {}
        }
    }

    let mut ids = Vec::new();
    if let Some(arg) = arg {
        collect(arg, &mut ids);
    }
    ids
}

fn resolve_cf_parent_arg_ids_ir(node: &fx_ir::IrNode) -> Vec<fx_ir::NodeId> {
    if node.target.contains("cond") {
        return node_ids_from_ir_arg(node.args.last());
    }
    if node.target.contains("while_loop") && node.args.len() >= 2 {
        let mut ids = node_ids_from_ir_arg(node.args.get(node.args.len() - 2));
        ids.extend(node_ids_from_ir_arg(node.args.last()));
        return ids;
    }
    Vec::new()
}

fn boundary_placeholder_dim_order(parent_arg: &fx_ir::IrNode) -> Option<Vec<usize>> {
    let rank = parent_arg.meta.rank()?;
    // The memory-format pass inserts physical TOSA-boundary tensors such as
    // `x_input_layout`. Their fake-tensor shape is already the serialized
    // tensor shape, so carrying the producer's non-identity layout metadata
    // onto a subgraph placeholder would transpose the block input a second time.
    if parent_arg.name.ends_with("_input_layout") || parent_arg.target.contains("TRANSPOSE") {
        return Some((0..rank).collect());
    }
    parent_arg.meta.tosa_dim_order.clone()
}

fn collect_ir_boundary_contracts_for_graph(
    ir: &fx_ir::IrGraph,
    parent_region: Option<&str>,
    contracts: &mut Vec<IrCfBoundaryContract>,
    while_body_regions: &mut BTreeSet<String>,
) {
    for node in ir.nodes.iter().filter(|node| node.alive) {
        let is_cond = node.target.contains("cond");
        let is_while = node.target.contains("while_loop");
        if !is_cond && !is_while {
            continue;
        }

        let parent_arg_ids = resolve_cf_parent_arg_ids_ir(node);
        if parent_arg_ids.is_empty() {
            continue;
        }

        let inputs: Vec<IrBoundaryInputMeta> = parent_arg_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &arg_id)| {
                let arg_node = ir.nodes.get(arg_id)?;
                let fallback_qparam = node
                    .meta
                    .pre_pass_arg_qparams
                    .as_ref()
                    .and_then(|saved| saved.get(&idx).cloned())
                    .or_else(|| fx_control_flow_metadata::extract_effective_qparam_ir(ir, arg_id));
                Some(IrBoundaryInputMeta {
                    meta: arg_node.meta.clone(),
                    boundary_dim_order: boundary_placeholder_dim_order(arg_node),
                    fallback_qparam,
                })
            })
            .collect();
        if inputs.is_empty() {
            continue;
        }

        let region_indices: &[(usize, bool)] = if is_while {
            &[(0, false), (1, true)]
        } else {
            &[(1, false), (2, false)]
        };
        for &(arg_idx, is_while_body) in region_indices {
            let Some(region) = region_name_from_ir_arg(ir, node.args.get(arg_idx), parent_region)
            else {
                continue;
            };
            if is_while_body {
                while_body_regions.insert(region.clone());
            }
            contracts.push(IrCfBoundaryContract {
                region,
                inputs: inputs.clone(),
            });
        }
    }
}

fn infer_cf_output_qparams_map_ir(
    graph: &fx_ir::IrGraph,
    cf_id: fx_ir::NodeId,
) -> Vec<(usize, fx_ir::QParam)> {
    let mut user_ids: Vec<fx_ir::NodeId> = graph.users(cf_id).to_vec();
    user_ids.sort_by(|&a, &b| {
        let ak = graph.node(a).args.get(1).and_then(|arg| match arg {
            fx_ir::IrArg::Int(i) => Some(*i),
            _ => None,
        });
        let bk = graph.node(b).args.get(1).and_then(|arg| match arg {
            fx_ir::IrArg::Int(i) => Some(*i),
            _ => None,
        });
        match (ak, bk) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });

    let mut result = Vec::new();
    for user_id in user_ids {
        let user = graph.node(user_id);
        if !user.alive {
            continue;
        }
        let idx = match user.args.get(1) {
            Some(fx_ir::IrArg::Int(i)) => *i as usize,
            _ => continue,
        };
        if let Some(qparam) = user.meta.output_qparam() {
            result.push((idx, qparam.clone()));
            continue;
        }
        let mut found = false;
        for &consumer_id in graph.users(user_id) {
            let consumer = graph.node(consumer_id);
            if !consumer.alive {
                continue;
            }
            if let Some(qparam) = consumer.meta.input_qparam() {
                result.push((idx, qparam.clone()));
                found = true;
                break;
            }
        }
        if found {
            continue;
        }
        for &consumer_id in graph.users(user_id) {
            let consumer = graph.node(consumer_id);
            if !consumer.alive || !consumer.is_dequantize() {
                continue;
            }
            if let Some(qparam) = consumer.parse_qdq_qparam() {
                result.push((idx, qparam));
                break;
            }
        }
    }
    result
}

fn collect_ir_output_contracts_for_graph(
    ir: &fx_ir::IrGraph,
    parent_region: Option<&str>,
    contracts: &mut Vec<IrCfOutputContract>,
) {
    for node in ir.nodes.iter().filter(|node| node.alive) {
        let is_cond = node.target.contains("cond");
        let is_while = node.target.contains("while_loop");
        if !is_cond && !is_while {
            continue;
        }
        let targets = infer_cf_output_qparams_map_ir(ir, node.id);
        if targets.is_empty() {
            continue;
        }
        let region_indices: &[usize] = if is_while {
            &[1usize]
        } else {
            &[1usize, 2usize]
        };
        for &arg_idx in region_indices {
            let Some(region) = region_name_from_ir_arg(ir, node.args.get(arg_idx), parent_region)
            else {
                continue;
            };
            contracts.push(IrCfOutputContract {
                region,
                targets: targets.clone(),
            });
        }
    }
}

fn annotate_cf_parent_rank4_dim_order_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    const NHWC_DIM_ORDER: [usize; 4] = [0, 2, 3, 1];

    let cf_ids: Vec<fx_ir::NodeId> = graph
        .nodes
        .iter()
        .filter(|node| node.alive)
        .filter(|node| node.target.contains("cond") || node.target.contains("while_loop"))
        .map(|node| node.id)
        .collect();

    let mut annotated = 0u32;
    for cf_id in cf_ids {
        let user_ids = graph.users(cf_id).to_vec();
        for user_id in user_ids {
            if !graph.nodes[user_id].alive || graph.nodes[user_id].meta.rank() != Some(4) {
                continue;
            }
            graph.nodes[user_id].meta.set_extra(
                "arm_control_flow_parent_output_user",
                fx_ir::MetaValue::Bool(true),
            );
            if graph.nodes[user_id].meta.tosa_dim_order.as_deref() != Some(&NHWC_DIM_ORDER) {
                graph.nodes[user_id]
                    .meta
                    .set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
                annotated += 1;
            }
        }
    }
    annotated
}

fn apply_ir_boundary_contract_to_submodule(
    graph: &mut fx_ir::IrGraph,
    contract: &IrCfBoundaryContract,
) -> u32 {
    let placeholder_ids: Vec<fx_ir::NodeId> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.op == fx_control_flow_metadata::NodeOp::Placeholder)
        .map(|node| node.id)
        .collect();

    let mut changed = 0u32;
    for (ph_id, input) in placeholder_ids.iter().copied().zip(contract.inputs.iter()) {
        let internal_qparam = graph.nodes[ph_id].meta.output_qparam().cloned();
        let external_qparam = input
            .meta
            .output_qparam()
            .or(input.meta.input_qparam())
            .cloned()
            .or_else(|| input.fallback_qparam.clone());

        let ph_meta = &mut graph.nodes[ph_id].meta;
        ph_meta.set_extra("arm_control_flow_placeholder", fx_ir::MetaValue::Bool(true));
        if let Some(val) = input.meta.val.clone() {
            ph_meta.set_val(val);
            changed += 1;
        }
        if let Some(dim_order) = input.boundary_dim_order.clone() {
            ph_meta.set_tosa_dim_order(dim_order);
            changed += 1;
        }

        let mut found_qparam = false;
        if let Some(output_qparams) = input.meta.output_qparams.clone() {
            ph_meta.set_output_qparams(output_qparams);
            found_qparam = true;
            changed += 1;
        }
        if let Some(input_qparams) = input.meta.input_qparams.clone() {
            ph_meta.set_input_qparams(input_qparams);
            found_qparam = true;
            changed += 1;
        }
        if !found_qparam {
            if let Some(qparam) = input.fallback_qparam.clone() {
                ph_meta.set_both_qparams(qparam);
                changed += 1;
            }
        }

        let Some(external_qp) = external_qparam else {
            continue;
        };
        let Some(internal_qp) = internal_qparam else {
            continue;
        };
        if (external_qp.scale - internal_qp.scale).abs() < 1e-10 && external_qp.zp == internal_qp.zp
        {
            continue;
        }
        if external_qp.scale == 0.0 || internal_qp.scale == 0.0 {
            continue;
        }

        // Control-flow block inputs arrive in the parent/external quantization
        // domain. Preserve the submodule's internal arithmetic domain by
        // inserting the same placeholder-edge RESCALE as the Python boundary
        // pass, rather than overwriting all placeholder users with metadata.
        let mut meta = graph.nodes[ph_id].meta.clone();
        meta.set_single_input_qparam(external_qp.clone());
        meta.set_single_output_qparam(internal_qp.clone());
        meta.extras.remove("tosa_special_dtype");
        if let Some(shape) = graph.nodes[ph_id].meta.val_shape() {
            meta.set_val(fx_ir::FakeTensorMeta {
                shape: shape.to_vec(),
                dtype: internal_qp.dtype_str.clone(),
            });
        }
        let ph_name = graph.nodes[ph_id].name.clone();
        let rescale_args = vec![
            fx_ir::IrArg::Node(ph_id),
            fx_ir::IrArg::Dtype(internal_qp.dtype_str.clone()),
            fx_ir::IrArg::List(vec![fx_ir::IrArg::Float(
                external_qp.scale / internal_qp.scale,
            )]),
            fx_ir::IrArg::Int(external_qp.zp),
            fx_ir::IrArg::Int(internal_qp.zp),
        ];
        let rescale = graph.add_node(
            fx_control_flow_metadata::NodeOp::CallFunction,
            format!("{ph_name}_cf_input_rescale"),
            "backend.tosa.RESCALE.default".to_string(),
            rescale_args.clone(),
            meta,
        );
        graph.anchor_after(rescale, ph_id);
        graph.replace_all_uses(ph_id, rescale);
        graph.set_args(rescale, rescale_args);
        graph.nodes[ph_id].meta.clear_output_qparams();
        changed += 1;
    }

    changed
}

fn apply_ir_qparam_contract_to_submodule(
    graph: &mut fx_ir::IrGraph,
    contract: &IrCfBoundaryContract,
) -> u32 {
    let placeholder_ids: Vec<fx_ir::NodeId> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.op == fx_control_flow_metadata::NodeOp::Placeholder)
        .map(|node| node.id)
        .collect();

    let mut changed = 0u32;
    for (ph_id, input) in placeholder_ids.iter().copied().zip(contract.inputs.iter()) {
        let ph_meta = &mut graph.nodes[ph_id].meta;
        ph_meta.set_extra("arm_control_flow_placeholder", fx_ir::MetaValue::Bool(true));
        changed += 1;

        let mut found_qparam = false;
        if let Some(output_qparams) = input.meta.output_qparams.clone() {
            ph_meta.set_output_qparams(output_qparams);
            found_qparam = true;
            changed += 1;
        }
        if let Some(input_qparams) = input.meta.input_qparams.clone() {
            ph_meta.set_input_qparams(input_qparams);
            found_qparam = true;
            changed += 1;
        }
        if !found_qparam {
            if let Some(qparam) = input.fallback_qparam.clone() {
                ph_meta.set_both_qparams(qparam);
                changed += 1;
            }
        }
    }

    changed
}

fn find_ir_boundary_contract_for_region(region_name: &str) -> Option<(IrCfBoundaryContract, bool)> {
    STASHED_IR.with(|stash| {
        let stash = stash.borrow();
        let mut contracts = Vec::new();
        let mut while_body_regions = BTreeSet::new();

        if let Some((parent_ir, _)) = &stash.parent {
            collect_ir_boundary_contracts_for_graph(
                parent_ir,
                None,
                &mut contracts,
                &mut while_body_regions,
            );
        }
        for (region, (ir, _)) in stash.submodules.iter() {
            collect_ir_boundary_contracts_for_graph(
                ir,
                Some(region.as_str()),
                &mut contracts,
                &mut while_body_regions,
            );
        }

        contracts
            .into_iter()
            .find(|contract| contract.region == region_name)
            .map(|contract| {
                let is_while_body = while_body_regions.contains(region_name);
                (contract, is_while_body)
            })
    })
}

fn apply_pre_submodule_qparam_propagation_ir(
    region_name: Option<&str>,
    graph: &mut fx_ir::IrGraph,
) -> u32 {
    let Some(region_name) = region_name else {
        return 0;
    };
    let Some((contract, is_while_body)) = find_ir_boundary_contract_for_region(region_name) else {
        return 0;
    };

    let mut changed = apply_ir_qparam_contract_to_submodule(graph, &contract);
    if is_while_body {
        changed += fx_control_flow_metadata::stamp_while_body_output_qparams_ir(graph) as u32;
    }
    changed
}

fn rewrite_output_dequantize_boundaries_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    let mut rewritten = 0u32;
    let ids: Vec<fx_ir::NodeId> = (0..graph.nodes.len()).collect();
    for dq_id in ids {
        if !graph.nodes[dq_id].alive || !graph.nodes[dq_id].is_dequantize() {
            continue;
        }
        let Some(source_id) = graph.nodes[dq_id].first_node_arg() else {
            continue;
        };
        let user_ids = graph.users(dq_id).to_vec();
        if user_ids.is_empty() {
            continue;
        }
        let all_rescale_or_output = user_ids.iter().all(|&uid| {
            let user = &graph.nodes[uid];
            user.alive && (user.is_rescale() || user.op == fx_control_flow_metadata::NodeOp::Output)
        });
        if !all_rescale_or_output {
            continue;
        }
        graph.replace_and_erase(dq_id, source_id);
        rewritten += 1;
    }
    rewritten
}

fn rewrite_quantize_boundaries_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    let mut rewritten = 0u32;
    let ids: Vec<fx_ir::NodeId> = (0..graph.nodes.len()).collect();
    for q_id in ids {
        if !graph.nodes[q_id].alive || !graph.nodes[q_id].is_quantize() {
            continue;
        }
        let Some(source_id) = graph.nodes[q_id].first_node_arg() else {
            continue;
        };
        let Some(target_qp) = graph.nodes[q_id].parse_qdq_qparam() else {
            continue;
        };
        let Some(source_qp) =
            fx_control_flow_metadata::extract_effective_qparam_ir(graph, source_id)
        else {
            continue;
        };

        if source_qp.scale == target_qp.scale && source_qp.zp == target_qp.zp {
            graph.replace_and_erase(q_id, source_id);
            rewritten += 1;
            continue;
        }
        if source_qp.scale == 0.0 || target_qp.scale == 0.0 {
            continue;
        }

        let mut meta = graph.nodes[q_id].meta.clone();
        meta.set_single_input_qparam(source_qp.clone());
        meta.set_single_output_qparam(target_qp.clone());
        meta.extras.remove("tosa_special_dtype");
        if let Some(shape) = graph.nodes[source_id].meta.val_shape() {
            meta.set_val(fx_ir::FakeTensorMeta {
                shape: shape.to_vec(),
                dtype: target_qp.dtype_str.clone(),
            });
        }
        if let Some(dim_order) = graph.nodes[source_id].meta.tosa_dim_order.clone() {
            meta.set_tosa_dim_order(dim_order);
        }
        let q_name = graph.nodes[q_id].name.clone();
        let rescale = graph.add_node(
            fx_control_flow_metadata::NodeOp::CallFunction,
            format!("{q_name}_boundary_rescale"),
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                fx_ir::IrArg::Node(source_id),
                fx_ir::IrArg::Dtype(target_qp.dtype_str.clone()),
                fx_ir::IrArg::List(vec![fx_ir::IrArg::Float(source_qp.scale / target_qp.scale)]),
                fx_ir::IrArg::Int(source_qp.zp),
                fx_ir::IrArg::Int(target_qp.zp),
            ],
            meta,
        );
        graph.anchor_to(rescale, q_id);
        graph.copy_meta_from(rescale, source_id);
        graph.replace_and_erase(q_id, rescale);
        rewritten += 1;
    }
    rewritten
}

fn output_arg_node_ids_ir(graph: &fx_ir::IrGraph, out_id: fx_ir::NodeId) -> Vec<fx_ir::NodeId> {
    let output = graph.node(out_id);
    if output.args.len() == 1 {
        if let Some(fx_ir::IrArg::List(items)) = output.args.first() {
            return items
                .iter()
                .filter_map(|item| match item {
                    fx_ir::IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect();
        }
    }
    output
        .args
        .iter()
        .filter_map(|arg| match arg {
            fx_ir::IrArg::Node(id) => Some(*id),
            _ => None,
        })
        .collect()
}

fn replace_output_arg_at_index_ir(
    graph: &mut fx_ir::IrGraph,
    out_id: fx_ir::NodeId,
    index: usize,
    new_id: fx_ir::NodeId,
) -> bool {
    let mut args = graph.node(out_id).args.clone();
    if args.len() == 1 {
        if let Some(fx_ir::IrArg::List(items)) = args.first_mut() {
            let Some(fx_ir::IrArg::Node(slot)) = items.get_mut(index) else {
                return false;
            };
            *slot = new_id;
            graph.set_args(out_id, args);
            return true;
        }
    }
    let Some(fx_ir::IrArg::Node(slot)) = args.get_mut(index) else {
        return false;
    };
    *slot = new_id;
    graph.set_args(out_id, args);
    true
}

fn apply_output_rescale_contract_ir(
    graph: &mut fx_ir::IrGraph,
    contract: &IrCfOutputContract,
) -> u32 {
    let Some(out_id) = graph.output_node().map(|node| node.id) else {
        return 0;
    };
    let output_arg_ids = output_arg_node_ids_ir(graph, out_id);
    let mut inserted = 0u32;
    let mut targets = contract.targets.clone();
    targets.sort_by_key(|(idx, _)| *idx);
    for (idx, target_qp) in targets {
        let Some(&initial_src_id) = output_arg_ids.get(idx) else {
            continue;
        };
        let mut src_id = initial_src_id;
        let mut carried_rescale_id = None;
        if graph.nodes[initial_src_id]
            .name
            .contains("_carried_rescale")
        {
            if let Some(carried_source_id) = graph.nodes[initial_src_id].first_node_arg() {
                src_id = carried_source_id;
                carried_rescale_id = Some(initial_src_id);
            }
        }
        // Match the Python boundary phase: while body outputs are stamped on the
        // output node's input_qparams before inserting external-domain rescales.
        // Prefer actual RESCALE args over metadata; carried-rescale metadata can
        // be stale after boundary rewrites while the op args remain authoritative.
        let source_from_rescale = if graph.nodes[src_id].is_rescale() {
            fx_control_flow_metadata::reverse_engineer_rescale_ir(graph, graph.node(src_id))
        } else {
            None
        };
        let Some(source_qp) = source_from_rescale
            .or_else(|| {
                graph.nodes[out_id]
                    .meta
                    .input_qparams
                    .as_ref()
                    .and_then(|map| map.get(&idx).cloned())
            })
            .or_else(|| fx_control_flow_metadata::extract_effective_qparam_ir(graph, src_id))
        else {
            continue;
        };
        if source_qp.scale == target_qp.scale && source_qp.zp == target_qp.zp {
            if let Some(dead_id) = carried_rescale_id {
                if replace_output_arg_at_index_ir(graph, out_id, idx, src_id) {
                    graph.erase(dead_id);
                    inserted += 1;
                }
            }
            continue;
        }
        if source_qp.scale == 0.0 || target_qp.scale == 0.0 {
            continue;
        }

        let mut meta = graph.nodes[src_id].meta.clone();
        meta.set_single_input_qparam(source_qp.clone());
        meta.set_single_output_qparam(target_qp.clone());
        meta.extras.remove("tosa_special_dtype");
        if let Some(shape) = graph.nodes[src_id].meta.val_shape() {
            meta.set_val(fx_ir::FakeTensorMeta {
                shape: shape.to_vec(),
                dtype: target_qp.dtype_str.clone(),
            });
        }
        let src_name = graph.nodes[src_id].name.clone();
        let rescale = graph.add_node(
            fx_control_flow_metadata::NodeOp::CallFunction,
            format!("{src_name}_cf_output_rescale"),
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                fx_ir::IrArg::Node(src_id),
                fx_ir::IrArg::Dtype(target_qp.dtype_str.clone()),
                fx_ir::IrArg::List(vec![fx_ir::IrArg::Float(source_qp.scale / target_qp.scale)]),
                fx_ir::IrArg::Int(source_qp.zp),
                fx_ir::IrArg::Int(target_qp.zp),
            ],
            meta,
        );
        graph.anchor_after(rescale, src_id);
        graph.copy_meta_from(rescale, src_id);
        if replace_output_arg_at_index_ir(graph, out_id, idx, rescale) {
            if let Some(dead_id) = carried_rescale_id {
                graph.erase(dead_id);
            }
            inserted += 1;
        }
    }
    inserted
}

fn strip_while_cond_output_rescales_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    let Some(out_id) = graph.output_node().map(|node| node.id) else {
        return 0;
    };
    let output_arg_ids = output_arg_node_ids_ir(graph, out_id);
    let mut stripped = 0u32;
    for rescale_id in output_arg_ids {
        if !graph.nodes[rescale_id].alive || !graph.nodes[rescale_id].is_rescale() {
            continue;
        }
        let Some(src_id) = graph.nodes[rescale_id].first_node_arg() else {
            continue;
        };
        let source_is_bool = graph.nodes[src_id]
            .meta
            .val_dtype()
            .is_some_and(|dtype| dtype.contains("bool"));
        if !source_is_bool && !graph.nodes[rescale_id].name.contains("_carried_rescale") {
            continue;
        }
        // WHILE_LOOP condition blocks must return a BOOL predicate. The shared
        // while submodule pipeline can insert carried-domain output rescales
        // because it cannot distinguish cond/body regions at pipeline time.
        graph.replace_and_erase(rescale_id, src_id);
        stripped += 1;
    }
    stripped
}

fn apply_ir_control_flow_boundary_contracts(
    parent_ir: &mut fx_ir::IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
) -> u32 {
    let mut contracts = Vec::new();
    let mut output_contracts = Vec::new();
    let mut while_body_regions = BTreeSet::new();
    let (while_cond_regions, _) = collect_ir_control_flow_region_metadata(parent_ir, submodules);
    collect_ir_boundary_contracts_for_graph(
        parent_ir,
        None,
        &mut contracts,
        &mut while_body_regions,
    );
    collect_ir_output_contracts_for_graph(parent_ir, None, &mut output_contracts);
    for (region, (ir, _)) in submodules.iter() {
        collect_ir_boundary_contracts_for_graph(
            ir,
            Some(region.as_str()),
            &mut contracts,
            &mut while_body_regions,
        );
        collect_ir_output_contracts_for_graph(ir, Some(region.as_str()), &mut output_contracts);
    }
    contracts.sort_by(|a, b| a.region.cmp(&b.region));
    output_contracts.sort_by(|a, b| a.region.cmp(&b.region));

    let mut changed = annotate_cf_parent_rank4_dim_order_ir(parent_ir);
    for (region, (ir, _)) in submodules.iter_mut() {
        if while_cond_regions.contains(region) {
            let stripped = strip_while_cond_output_rescales_ir(ir);
            if stripped > 0 && std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[RETAB][cf-boundary-ir] stripped {stripped} while-cond output RESCALE node(s) in {region}"
                );
            }
            changed += stripped;
        }
        let rewrites = rewrite_output_dequantize_boundaries_ir(ir);
        if rewrites > 0 && std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[RETAB][cf-boundary-ir] rewrote {rewrites} output DQ boundary node(s) in {region}"
            );
        }
        changed += rewrites;
    }
    for contract in &contracts {
        let Some((ir, _)) = submodules.get_mut(&contract.region) else {
            continue;
        };
        changed += apply_ir_boundary_contract_to_submodule(ir, contract);
        if while_body_regions.contains(&contract.region) {
            changed += fx_control_flow_metadata::stamp_while_body_output_qparams_ir(ir) as u32;
        }
    }
    for (region, (ir, _)) in submodules.iter_mut() {
        let rewrites = rewrite_quantize_boundaries_ir(ir);
        if rewrites > 0 && std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[RETAB][cf-boundary-ir] rewrote {rewrites} quantize boundary node(s) in {region}"
            );
        }
        changed += rewrites;
    }
    for contract in &output_contracts {
        let Some((ir, _)) = submodules.get_mut(&contract.region) else {
            continue;
        };
        let inserts = apply_output_rescale_contract_ir(ir, contract);
        if inserts > 0 && std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[RETAB][cf-boundary-ir] inserted {inserts} output RESCALE boundary node(s) in {}",
                contract.region
            );
        }
        changed += inserts;
    }
    changed
}

fn sequence_arg_len_ir(arg: Option<&fx_ir::IrArg>) -> usize {
    match arg {
        Some(fx_ir::IrArg::List(items)) => items.len(),
        Some(fx_ir::IrArg::None) | None => 0,
        Some(_) => 1,
    }
}

fn collect_ir_control_flow_region_metadata(
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> (BTreeSet<String>, BTreeMap<String, IrWhileBodyInputCounts>) {
    let mut while_cond_graph_names = BTreeSet::new();
    let mut while_body_input_counts = BTreeMap::new();

    for (parent_region, ir) in std::iter::once((None, parent_ir)).chain(
        submodules
            .iter()
            .map(|(name, (ir, _))| (Some(name.as_str()), ir)),
    ) {
        for node in ir.nodes.iter().filter(|node| node.alive) {
            if !node.target.contains("while_loop") {
                continue;
            }

            if let Some(cond_name) = region_name_from_ir_arg(ir, node.args.first(), parent_region) {
                while_cond_graph_names.insert(cond_name);
            }
            if let Some(body_name) = region_name_from_ir_arg(ir, node.args.get(1), parent_region) {
                while_body_input_counts.insert(
                    body_name,
                    IrWhileBodyInputCounts {
                        carried: sequence_arg_len_ir(node.args.get(2)),
                        additional: sequence_arg_len_ir(node.args.get(3)),
                    },
                );
            }
        }
    }

    (while_cond_graph_names, while_body_input_counts)
}

fn mark_scalar_control_flow_outputs_ir(graph: &mut Graph) {
    let Some(output_id) = graph.output else {
        return;
    };
    let Some(output_node) = graph.node(output_id) else {
        return;
    };
    let output_args = output_node.args.clone();

    for arg in output_args {
        for node_id in arg.node_refs() {
            let Some(node) = graph.node_mut(node_id) else {
                continue;
            };
            if node.meta.val.dtype == DType::Bool {
                node.meta.preserve_scalar_shape = true;
            }
        }
    }
}

fn append_while_body_passthrough_outputs_ir(graph: &mut Graph, counts: IrWhileBodyInputCounts) {
    let additional_inputs: Vec<NodeId> = graph
        .inputs
        .iter()
        .skip(counts.carried)
        .take(counts.additional)
        .copied()
        .collect();
    if additional_inputs.is_empty() {
        return;
    }

    let Some(output_id) = graph.output else {
        return;
    };
    let Some(output_node) = graph.node_mut(output_id) else {
        return;
    };

    match output_node.args.first_mut() {
        Some(Arg::NodeList(ids)) => ids.extend(additional_inputs),
        Some(Arg::Node(id)) => {
            let mut ids = smallvec::SmallVec::new();
            ids.push(*id);
            ids.extend(additional_inputs);
            output_node.args[0] = Arg::NodeList(ids);
        }
        _ => output_node
            .args
            .push(Arg::NodeList(smallvec::SmallVec::from_vec(
                additional_inputs,
            ))),
    }
    graph.rebuild_user_lists();
}

fn drop_subgraph_input_store_entries(graph: &Graph, store: &mut TensorStore) {
    for &input_id in &graph.inputs {
        let Some(node) = graph.node(input_id) else {
            continue;
        };
        store.buffers.remove(&node.name);
    }
}

fn append_control_flow_subgraphs_from_ir(
    graphs: &mut Vec<(String, Graph)>,
    store: &mut TensorStore,
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> PyResult<()> {
    let (while_cond_graph_names, while_body_input_counts) =
        collect_ir_control_flow_region_metadata(parent_ir, submodules);

    let referenced_regions: BTreeSet<String> = while_cond_graph_names
        .iter()
        .cloned()
        .chain(while_body_input_counts.keys().cloned())
        .chain(submodules.keys().cloned())
        .collect();

    for region in referenced_regions {
        if graphs.iter().any(|(name, _)| name == &region) {
            continue;
        }
        let Some((ir, effects)) = submodules.get(&region) else {
            return Err(PyRuntimeError::new_err(format!(
                "missing stashed IrGraph for control-flow region '{region}'"
            )));
        };

        let mut sub_store = TensorStore::new();
        let mut sub_graph =
            ingest::from_ir::graph_from_ir_preserve_placeholders(ir, &mut sub_store);
        ingest::from_ir::apply_effects_preserve_placeholders(
            &mut sub_graph,
            &mut sub_store,
            effects,
        );
        ingest::fx_bridge::finalize_converted_graph(&mut sub_graph);
        if while_cond_graph_names.contains(&region) {
            mark_scalar_control_flow_outputs_ir(&mut sub_graph);
        }
        if let Some(counts) = while_body_input_counts.get(&region).copied() {
            append_while_body_passthrough_outputs_ir(&mut sub_graph, counts);
        }
        drop_subgraph_input_store_entries(&sub_graph, &mut sub_store);
        if std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
            ingest::from_ir::debug_dump(&sub_graph, &format!("CF SUBGRAPH {region}"));
        }
        ingest::from_ir::prefix_graph(&mut sub_graph, &mut sub_store, &format!("{region}::"));
        for (key, value) in sub_store.buffers {
            if store.get(&key).is_none() {
                store.insert(key, value);
            }
        }
        graphs.push((region, sub_graph));
    }

    Ok(())
}

fn ingest_and_serialize_tosa_pybytes(
    py: Python<'_>,
    _graph_module: &Bound<'_, PyAny>,
    _state_dict: &Bound<'_, PyDict>,
    _graph_signature: &Bound<'_, PyAny>,
    _exported_program: Option<PyObject>,
) -> PyResult<Py<PyBytes>> {
    let profile = profile_ingest_serialize_enabled();
    let mut stashed = take_stashed_hierarchy();
    let Some((mut ir, effects)) = stashed.parent.take() else {
        return Err(PyRuntimeError::new_err(
            "missing stashed IrGraph for serialization",
        ));
    };

    let has_control_flow = ir_has_control_flow(&ir);
    if has_control_flow {
        let changed = apply_ir_control_flow_boundary_contracts(&mut ir, &mut stashed.submodules);
        if std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") && changed > 0 {
            eprintln!(
                "[RETAB][cf-boundary-ir] applied {changed} parent/submodule boundary metadata update(s)"
            );
        }
    }

    let mut store = crate::store::TensorStore::new();
    let mut converted = ingest::from_ir::graph_from_ir(&ir, &mut store);
    ingest::from_ir::apply_effects(&mut converted, &mut store, &effects);
    ingest::fx_bridge::finalize_converted_graph(&mut converted);
    if std::env::var("ARM_FROM_IR_DEBUG").ok().as_deref() == Some("1") {
        ingest::from_ir::debug_dump(&converted, "MAIN FROM IR");
    }
    ingest::from_ir::report_missing_consts(&converted, &store);

    let mut graphs = vec![(String::from("main"), converted)];
    if has_control_flow {
        append_control_flow_subgraphs_from_ir(&mut graphs, &mut store, &ir, &stashed.submodules)?;
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
        true,
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
        true,
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

/// Structural-only metadata stamp (no qparam propagation). Run before parent passes.
#[pyfunction]
#[pyo3(signature = (graph_module))]
fn stamp_control_flow_structural(graph_module: &Bound<'_, PyAny>) -> PyResult<bool> {
    let gm = fx_graph::FxGraphModule::new(graph_module);
    fx_pass::FxPass::run(&fx_control_flow_metadata::FxControlFlowStructuralPass, &gm)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_function(wrap_pyfunction!(ingest_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_submodule_passes, m)?)?;
    m.add_function(wrap_pyfunction!(run_fx_cond_passes, m)?)?;
    // Control-flow pipeline (new ordering)
    m.add_function(wrap_pyfunction!(stamp_control_flow_structural, m)?)?;
    Ok(())
}
