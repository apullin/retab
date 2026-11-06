use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::fx_target::is_tosa_transpose_target;
use crate::graph::ir::DType;
use crate::pipeline_config::LayoutMode;
use crate::StashedPipeline;

use super::{collect_region_metadata, metadata, region_name_from_ir_arg};

#[derive(Clone)]
struct BoundaryInputMeta {
    meta: fx_ir::NodeMeta,
    boundary_dim_order: Option<Vec<usize>>,
    fallback_qparam: Option<fx_ir::QParam>,
}

#[derive(Clone)]
struct BoundaryContract {
    region: String,
    inputs: Vec<BoundaryInputMeta>,
}

#[derive(Clone)]
struct OutputContract {
    region: String,
    targets: Vec<(usize, fx_ir::QParam)>,
}

fn boundary_rescale_meta(
    base_meta: &fx_ir::NodeMeta,
    source_meta: &fx_ir::NodeMeta,
    source_qp: &fx_ir::QParam,
    target_qp: &fx_ir::QParam,
) -> fx_ir::NodeMeta {
    let mut meta =
        base_meta.rescale_meta_with_input_qparam(source_meta, source_qp.clone(), target_qp.clone());
    if let Some(dim_order) = source_meta.cloned_tosa_dim_order() {
        meta.set_tosa_dim_order(dim_order);
    }
    meta
}

fn add_boundary_rescale(
    graph: &mut fx_ir::IrGraph,
    name: String,
    source_id: fx_ir::NodeId,
    base_meta_id: fx_ir::NodeId,
    source_qp: &fx_ir::QParam,
    target_qp: &fx_ir::QParam,
) -> fx_ir::NodeId {
    let meta = boundary_rescale_meta(
        &graph.nodes[base_meta_id].meta,
        &graph.nodes[source_id].meta,
        source_qp,
        target_qp,
    );
    graph.add_rescale(fx_ir::RescaleRewrite::new(
        name,
        source_id,
        target_qp.dtype_str.clone(),
        vec![source_qp.scale / target_qp.scale],
        source_qp.zp,
        target_qp.zp,
        meta,
    ))
}

fn boundary_placeholder_ids(graph: &fx_ir::IrGraph) -> Vec<fx_ir::NodeId> {
    graph.live_placeholder_ids()
}

fn boundary_external_qparam(input: &BoundaryInputMeta) -> Option<fx_ir::QParam> {
    input
        .fallback_qparam
        .clone()
        .or_else(|| boundary_arg_qparam(input))
}

fn stamp_boundary_qparam(ph_meta: &mut fx_ir::NodeMeta, input: &BoundaryInputMeta) -> bool {
    if let Some(qparam) = input.fallback_qparam.clone() {
        ph_meta.set_both_qparams(qparam);
        return true;
    }
    let mut found_qparam = false;
    if ph_meta.copy_output_qparams_from(&input.meta) {
        found_qparam = true;
    }
    if ph_meta.copy_input_qparams_from(&input.meta) {
        found_qparam = true;
    }
    found_qparam
}

fn boundary_arg_qparam(input: &BoundaryInputMeta) -> Option<fx_ir::QParam> {
    input
        .meta
        .output_qparam()
        .or(input.meta.input_qparam())
        .cloned()
}

fn stamp_boundary_placeholder_meta(
    ph_meta: &mut fx_ir::NodeMeta,
    input: &BoundaryInputMeta,
    count_marker_write: bool,
) -> u32 {
    let mut changed = 0u32;
    ph_meta.set_extra("arm_control_flow_placeholder", fx_ir::MetaValue::Bool(true));
    if count_marker_write {
        changed += 1;
    }
    if let Some(val) = input.meta.tensor_facts().cloned() {
        ph_meta.set_val(val);
        changed += 1;
    }
    if let Some(dim_order) = input.boundary_dim_order.clone() {
        ph_meta.set_val_dim_order(dim_order.clone());
        ph_meta.set_tosa_dim_order(dim_order);
        changed += 1;
    }

    let found_qparam = stamp_boundary_qparam(ph_meta, input);
    if found_qparam {
        changed += 1;
    }
    if !found_qparam {
        if let Some(qparam) = boundary_arg_qparam(input) {
            ph_meta.set_both_qparams(qparam);
            changed += 1;
        }
    }

    changed
}

pub(super) fn node_ids_from_ir_arg(arg: Option<&fx_ir::IrArg>) -> Vec<fx_ir::NodeId> {
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

fn resolve_parent_arg_ids_ir(node: &fx_ir::IrNode) -> Vec<fx_ir::NodeId> {
    if node.is_cond() {
        return node_ids_from_ir_arg(node.args.last());
    }
    if node.is_while_loop() && node.args.len() >= 2 {
        let mut ids = node_ids_from_ir_arg(node.args.get(node.args.len() - 2));
        ids.extend(node_ids_from_ir_arg(node.args.last()));
        return ids;
    }
    Vec::new()
}

/// The meta a region placeholder takes from the operand bound to it, and the dim order
/// that the boundary gives it.
///
/// Explicit mode: a val is the tensor as TOSA sees it, on both sides of the boundary,
/// and no node has a dim order. The placeholder takes the operand's val as it is, as
/// rc4's `_regularize_submodule` gives region IO the parent's post-pass vals
/// (`backends/arm/tosa/backend.py:204-249`), and the boundary writes no dim order. The
/// TOSA TRANSPOSE case below reconstructs a logical shape under a physical dim order,
/// which is dim-order logic; it is inert in explicit mode, where permutes stay
/// `aten.permute_copy` until the serializer (P2 design §1c, §4.3.1).
fn boundary_placeholder_meta(
    ir: &fx_ir::IrGraph,
    parent_id: fx_ir::NodeId,
    layout: LayoutMode,
) -> (fx_ir::NodeMeta, Option<Vec<usize>>) {
    let parent_arg = ir.node(parent_id);
    if layout == LayoutMode::Explicit {
        return (parent_arg.meta.clone(), None);
    }
    let fallback = (
        parent_arg.meta.clone(),
        parent_arg.meta.cloned_tosa_dim_order(),
    );
    if !is_tosa_transpose_target(&parent_arg.target) {
        return fallback;
    }
    let (Some(input_id), Some(perms)) = (parent_arg.first_node_arg(), parent_arg.arg_int_list(1))
    else {
        return fallback;
    };
    let (input_meta, input_order) = boundary_placeholder_meta(ir, input_id, layout);
    let Some(shape) = input_meta.val_shape() else {
        return fallback;
    };
    if perms.len() != shape.len() {
        return fallback;
    }
    let input_order = input_order.unwrap_or_else(|| (0..shape.len()).collect());
    let order: Option<Vec<usize>> = perms
        .iter()
        .map(|&dim| {
            fx_ir::normalize_dim(dim, shape.len()).and_then(|dim| input_order.get(dim).copied())
        })
        .collect();
    let Some(order) = order else {
        return fallback;
    };
    // Layout transposes change physical order; region decompositions still use logical axes.
    (
        parent_arg
            .meta
            .cloned_with_same_dtype_or_float32(shape.to_vec()),
        Some(order),
    )
}

fn collect_boundary_contracts_for_graph(
    ir: &fx_ir::IrGraph,
    parent_region: Option<&str>,
    contracts: &mut Vec<BoundaryContract>,
    while_body_regions: &mut BTreeSet<String>,
    layout: LayoutMode,
) {
    for node_id in ir.live_node_ids() {
        let node = ir.node(node_id);
        let is_cond = node.is_cond();
        let is_while = node.is_while_loop();
        if !is_cond && !is_while {
            continue;
        }

        let parent_arg_ids = resolve_parent_arg_ids_ir(node);
        if parent_arg_ids.is_empty() {
            continue;
        }

        let list_input_qparam = if is_while {
            node.meta.input_qparam_at(2).cloned()
        } else {
            node.meta.input_qparam_at(3).cloned()
        };
        let inputs: Vec<BoundaryInputMeta> = parent_arg_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &arg_id)| {
                let arg_node = ir.nodes.get(arg_id)?;
                // The list-level qparam is the one the quantizer shares between the
                // quantized operands. A constant passed as raw integers (no qparams before
                // the pipeline) carries none; the branch's own DQ interprets it.
                let raw_constant = arg_node.tensor.is_some()
                    && node
                        .meta
                        .pre_pass_arg_qparams()
                        .is_some_and(|saved| saved.get(&idx).is_none());
                let fallback_qparam = if raw_constant {
                    None
                } else {
                    list_input_qparam
                        .clone()
                        .or_else(|| {
                            node.meta
                                .pre_pass_arg_qparams()
                                .and_then(|saved| saved.get(&idx).cloned())
                        })
                        .or_else(|| metadata::extract_effective_qparam_ir(ir, arg_id))
                };
                let (meta, boundary_dim_order) = boundary_placeholder_meta(ir, arg_id, layout);
                Some(BoundaryInputMeta {
                    meta,
                    boundary_dim_order,
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
            contracts.push(BoundaryContract {
                region,
                inputs: inputs.clone(),
            });
        }
    }
}

fn infer_output_qparams_map_ir(
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

fn collect_output_contracts_for_graph(
    ir: &fx_ir::IrGraph,
    parent_region: Option<&str>,
    contracts: &mut Vec<OutputContract>,
) {
    for node_id in ir.live_node_ids() {
        let node = ir.node(node_id);
        let is_cond = node.is_cond();
        let is_while = node.is_while_loop();
        if !is_cond && !is_while {
            continue;
        }
        let targets = infer_output_qparams_map_ir(ir, node.id);
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
            contracts.push(OutputContract {
                region,
                targets: targets.clone(),
            });
        }
    }
}

pub(crate) fn snapshot_pre_pass_qparams_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    let cf_ids: Vec<fx_ir::NodeId> = graph
        .live_nodes()
        .filter(|node| node.is_control_flow())
        .map(|node| node.id)
        .collect();

    let mut saved = 0u32;
    for cf_id in cf_ids {
        let parent_arg_ids = resolve_parent_arg_ids_ir(graph.node(cf_id));
        let mut snapshot = std::collections::HashMap::new();

        for (idx, arg_id) in parent_arg_ids.into_iter().enumerate() {
            let arg = graph.node(arg_id);
            let qparam = arg
                .meta
                .output_qparam()
                .cloned()
                .or_else(|| metadata::extract_effective_qparam_ir(graph, arg_id))
                .or_else(|| {
                    if arg.is_q_or_dq() {
                        arg.parse_qdq_qparam()
                    } else {
                        None
                    }
                });
            if let Some(qparam) = qparam {
                snapshot.insert(idx, qparam);
                saved += 1;
            }
        }

        if !snapshot.is_empty() {
            graph
                .node_mut(cf_id)
                .meta
                .set_pre_pass_arg_qparams(snapshot);
        }
    }

    saved
}

/// Dim-order mode lays a region's rank-4 tensors out channels-last, so a parent's rank-4
/// cond or while output arrives NHWC: give each such getitem that dim order, and mark it
/// for the main-output TRANSPOSE back to the user's layout (`serialize::output_layout`).
/// Explicit mode never calls this: there a region's outputs keep their vals, as rc4's do
/// (P2 design §4.3.1).
fn annotate_parent_rank4_dim_order_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    const NHWC_DIM_ORDER: [usize; 4] = [0, 2, 3, 1];

    let cf_ids: Vec<fx_ir::NodeId> = graph
        .live_nodes()
        .filter(|node| node.is_control_flow())
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
            if graph.nodes[user_id].meta.tosa_dim_order() != Some(&NHWC_DIM_ORDER) {
                graph.nodes[user_id]
                    .meta
                    .set_tosa_dim_order(NHWC_DIM_ORDER.to_vec());
                annotated += 1;
            }
        }
    }
    annotated
}

fn apply_boundary_contract_to_submodule(
    graph: &mut fx_ir::IrGraph,
    contract: &BoundaryContract,
) -> u32 {
    let placeholder_ids = boundary_placeholder_ids(graph);

    let mut changed = 0u32;
    for (ph_id, input) in placeholder_ids.iter().copied().zip(contract.inputs.iter()) {
        let internal_qparam = graph.nodes[ph_id].meta.output_qparam().cloned();
        let external_qparam = boundary_external_qparam(input);
        changed += stamp_boundary_placeholder_meta(&mut graph.nodes[ph_id].meta, input, false);

        let Some(external_qp) = external_qparam else {
            continue;
        };
        let Some(internal_qp) = internal_qparam else {
            continue;
        };
        if external_qp.same_scale_zp(&internal_qp) {
            continue;
        }
        if external_qp.scale == 0.0 || internal_qp.scale == 0.0 {
            continue;
        }

        let ph_name = graph.nodes[ph_id].name.clone();
        let rescale = add_boundary_rescale(
            graph,
            format!("{ph_name}_cf_input_rescale"),
            ph_id,
            ph_id,
            &external_qp,
            &internal_qp,
        );
        graph.anchor_after(rescale, ph_id);
        graph.replace_all_uses(ph_id, rescale);
        graph.set_arg(rescale, 0, fx_ir::IrArg::Node(ph_id));
        graph.nodes[ph_id].meta.clear_output_qparams();
        changed += 1;
    }

    changed
}

fn apply_qparam_contract_to_submodule(
    graph: &mut fx_ir::IrGraph,
    contract: &BoundaryContract,
) -> u32 {
    let placeholder_ids = boundary_placeholder_ids(graph);

    let mut changed = 0u32;
    for (ph_id, input) in placeholder_ids.iter().copied().zip(contract.inputs.iter()) {
        changed += stamp_boundary_placeholder_meta(&mut graph.nodes[ph_id].meta, input, true);
    }

    changed
}

fn find_boundary_contract_for_region(
    region_name: &str,
    parent: Option<&StashedPipeline>,
    submodules: &BTreeMap<String, StashedPipeline>,
    layout: LayoutMode,
) -> Option<(BoundaryContract, bool)> {
    let mut contracts = Vec::new();
    let mut while_body_regions = BTreeSet::new();

    if let Some((parent_ir, _)) = parent {
        collect_boundary_contracts_for_graph(
            parent_ir,
            None,
            &mut contracts,
            &mut while_body_regions,
            layout,
        );
    }
    for (region, (ir, _)) in submodules.iter() {
        collect_boundary_contracts_for_graph(
            ir,
            Some(region.as_str()),
            &mut contracts,
            &mut while_body_regions,
            layout,
        );
    }

    contracts
        .into_iter()
        .find(|contract| contract.region == region_name)
        .map(|contract| {
            let is_while_body = while_body_regions.contains(region_name);
            (contract, is_while_body)
        })
}

/// Before a region's pipeline runs: give its placeholders the vals and qparams of the
/// operands bound to them, read from the stashed post-pipeline graphs that invoke it.
/// `layout` is the hierarchy's layout mode (Python passes one pipeline config to the
/// parent and to every region).
pub(crate) fn apply_pre_submodule_qparam_propagation_ir(
    region_name: Option<&str>,
    graph: &mut fx_ir::IrGraph,
    parent: Option<&StashedPipeline>,
    submodules: &BTreeMap<String, StashedPipeline>,
    layout: LayoutMode,
) -> u32 {
    let Some(region_name) = region_name else {
        return 0;
    };
    let Some((contract, is_while_body)) =
        find_boundary_contract_for_region(region_name, parent, submodules, layout)
    else {
        return 0;
    };

    let mut changed = apply_qparam_contract_to_submodule(graph, &contract);
    if is_while_body {
        changed += metadata::stamp_while_body_output_qparams_ir(graph) as u32;
    }
    changed
}

/// Drop the dequantizes that only feed RESCALEs or the region's output: the
/// region hands its outputs over quantized, and the output contracts rescale
/// them into the parent's quantization.
///
/// With `record_output_qparams` (explicit layout mode), a dequantize feeding
/// the output first records its qparam as the output node's input qparam for
/// that output, unless one is recorded, which is rc4's FoldAndAnnotateQParamsPass
/// annotation that InsertControlFlowRescalesPass reads as the region side of
/// the output's RESCALE (`insert_rescales_pass.py`, `_rescale_submodule_outputs`;
/// APU-135 S8). For an output that is not itself a RESCALE, the output
/// contract then takes it rather than walking back from the output's producer
/// (`apply_output_rescale_contract_ir`). Explicit mode needs it: rc4's
/// rewrites end a convolution or a bilinear upsample with a qparam-less
/// RESCALE and permute, and the walk reverse engineers that RESCALE from the
/// TOSA op's inherited output qparam, which gives a wrong scale (for a
/// per-tensor conv, `s_out^2 / (s_in * s_w)` rather than `s_out`).
fn rewrite_output_dequantize_boundaries_ir(
    graph: &mut fx_ir::IrGraph,
    record_output_qparams: bool,
) -> u32 {
    let mut rewritten = 0u32;
    let output_id = graph.output_node().map(|node| node.id);
    let ids = graph.live_node_ids();
    for dq_id in ids {
        if !graph.nodes[dq_id].is_dequantize() {
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
            user.alive && (user.is_rescale() || user.op == fx_ir::NodeOp::Output)
        });
        if !all_rescale_or_output {
            continue;
        }
        if record_output_qparams {
            if let (Some(out_id), Some(qparam)) = (output_id, graph.nodes[dq_id].parse_qdq_qparam())
            {
                record_output_source_qparam(graph, out_id, dq_id, &qparam);
            }
        }
        graph.replace_and_erase(dq_id, source_id);
        rewritten += 1;
    }
    rewritten
}

/// Record `qparam` as the output node's input qparam for each output `value`
/// is, unless that output has one.
fn record_output_source_qparam(
    graph: &mut fx_ir::IrGraph,
    out_id: fx_ir::NodeId,
    value: fx_ir::NodeId,
    qparam: &fx_ir::QParam,
) {
    let slots: Vec<usize> = output_arg_node_ids_ir(graph, out_id)
        .iter()
        .enumerate()
        .filter_map(|(index, &id)| (id == value).then_some(index))
        .collect();
    if slots.is_empty() {
        return;
    }
    let meta = &mut graph.nodes[out_id].meta;
    let mut qparams = meta.cloned_input_qparams().unwrap_or_default();
    for slot in slots {
        if qparams.get(&slot).is_none() {
            qparams.insert(slot, qparam.clone());
        }
    }
    meta.set_input_qparams(qparams);
}

fn rewrite_quantize_boundaries_ir(graph: &mut fx_ir::IrGraph) -> u32 {
    let mut rewritten = 0u32;
    let ids = graph.live_node_ids();
    for q_id in ids {
        if !graph.nodes[q_id].is_quantize() {
            continue;
        }
        let Some(source_id) = graph.nodes[q_id].first_node_arg() else {
            continue;
        };
        let Some(target_qp) = graph.nodes[q_id].parse_qdq_qparam() else {
            continue;
        };
        let Some(source_qp) = metadata::extract_effective_qparam_ir(graph, source_id) else {
            continue;
        };

        if source_qp.same_exact_scale_zp(&target_qp) {
            graph.replace_and_erase(q_id, source_id);
            rewritten += 1;
            continue;
        }
        if source_qp.scale == 0.0 || target_qp.scale == 0.0 {
            continue;
        }

        let q_name = graph.nodes[q_id].name.clone();
        let rescale = add_boundary_rescale(
            graph,
            format!("{q_name}_boundary_rescale"),
            source_id,
            q_id,
            &source_qp,
            &target_qp,
        );
        graph.anchor_to(rescale, q_id);
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

fn apply_output_rescale_contract_ir(graph: &mut fx_ir::IrGraph, contract: &OutputContract) -> u32 {
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
        let source_from_rescale = if graph.nodes[src_id].is_rescale() {
            metadata::reverse_engineer_rescale_ir(graph, graph.node(src_id))
        } else {
            None
        };
        let Some(source_qp) = source_from_rescale
            .or_else(|| graph.nodes[out_id].meta.input_qparam_at(idx).cloned())
            .or_else(|| metadata::extract_effective_qparam_ir(graph, src_id))
        else {
            continue;
        };
        if source_qp.same_exact_scale_zp(&target_qp) {
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

        let src_name = graph.nodes[src_id].name.clone();
        let rescale = add_boundary_rescale(
            graph,
            format!("{src_name}_cf_output_rescale"),
            src_id,
            src_id,
            &source_qp,
            &target_qp,
        );
        graph.anchor_after(rescale, src_id);
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
        let source_is_bool = graph.nodes[src_id].meta.val_dtype_is(DType::Bool);
        if !source_is_bool && !graph.nodes[rescale_id].name.contains("_carried_rescale") {
            continue;
        }
        graph.replace_and_erase(rescale_id, src_id);
        stripped += 1;
    }
    stripped
}

/// At serialization: align every control-flow region's IO with the graph that invokes
/// it (boundary vals and qparams, and RESCALEs for inputs and outputs whose qparams
/// differ), then resolve the conv weights and biases the cond branches capture
/// (`captured_conv`). `layout` is the hierarchy's layout mode. Explicit mode writes no
/// dim order (no rank-4 getitem annotation; boundary placeholders take the operand's
/// val), records the regions' output qparams from their output dequantizes, and uses
/// its own captured_conv (APU-135 S8).
pub(crate) fn apply_boundary_contracts(
    parent_ir: &mut fx_ir::IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
    layout: LayoutMode,
) -> u32 {
    let explicit = layout == LayoutMode::Explicit;
    let mut contracts = Vec::new();
    let mut output_contracts = Vec::new();
    let mut while_body_regions = BTreeSet::new();
    let (while_cond_regions, _) = collect_region_metadata(parent_ir, submodules);
    collect_boundary_contracts_for_graph(
        parent_ir,
        None,
        &mut contracts,
        &mut while_body_regions,
        layout,
    );
    collect_output_contracts_for_graph(parent_ir, None, &mut output_contracts);
    for (region, (ir, _)) in submodules.iter() {
        collect_boundary_contracts_for_graph(
            ir,
            Some(region.as_str()),
            &mut contracts,
            &mut while_body_regions,
            layout,
        );
        collect_output_contracts_for_graph(ir, Some(region.as_str()), &mut output_contracts);
    }
    contracts.sort_by(|a, b| a.region.cmp(&b.region));
    output_contracts.sort_by(|a, b| a.region.cmp(&b.region));

    let mut changed = match layout {
        LayoutMode::DimOrder => annotate_parent_rank4_dim_order_ir(parent_ir),
        LayoutMode::Explicit => 0,
    };
    let cf_debug = crate::config::from_ir_debug_enabled();
    for (region, (ir, _)) in submodules.iter_mut() {
        if while_cond_regions.contains(region) {
            let stripped = strip_while_cond_output_rescales_ir(ir);
            if stripped > 0 && cf_debug {
                eprintln!(
                    "[RETAB][cf-boundary-ir] stripped {stripped} while-cond output RESCALE node(s) in {region}"
                );
            }
            changed += stripped;
        }
        let rewrites = rewrite_output_dequantize_boundaries_ir(ir, explicit);
        if rewrites > 0 && cf_debug {
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
        changed += apply_boundary_contract_to_submodule(ir, contract);
        if while_body_regions.contains(&contract.region) {
            changed += metadata::stamp_while_body_output_qparams_ir(ir) as u32;
        }
    }
    for (region, (ir, _)) in submodules.iter_mut() {
        let rewrites = rewrite_quantize_boundaries_ir(ir);
        if rewrites > 0 && cf_debug {
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
        if inserts > 0 && cf_debug {
            eprintln!(
                "[RETAB][cf-boundary-ir] inserted {inserts} output RESCALE boundary node(s) in {}",
                contract.region
            );
        }
        changed += inserts;
    }
    let rewritten = match layout {
        LayoutMode::DimOrder => {
            super::captured_conv::rewrite_captured_conv_operands(parent_ir, submodules)
        }
        LayoutMode::Explicit => {
            super::captured_conv::rewrite_explicit_captured_conv_operands(parent_ir, submodules)
        }
    };
    if rewritten > 0 && cf_debug {
        eprintln!("[RETAB][cf-boundary-ir] rewrote {rewritten} captured conv weight(s)/bias(es)");
    }
    changed + rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, NodeMeta, NodeOp, NodeTensor, TensorSource};
    use crate::lowering_effects::LoweringEffectBatch;
    use crate::store::TensorData;
    use crate::test_ir::{call, call_with_meta, live_node, output, placeholder, qparam};

    const I8: &str = "torch.int8";

    /// The parent: `x -> cond(true_graph, false_graph) -> getitem -> output`;
    /// the getitem holds the cond output's quantization, `(0.05, 3)`.
    fn parent_graph() -> IrGraph {
        let mut parent = IrGraph::empty();
        let x = placeholder(&mut parent, "x", [1, 2, 8], I8);
        let cond = call_with_meta(
            &mut parent,
            "cond",
            "torch.ops.higher_order.cond",
            vec![
                IrArg::Int(1),
                IrArg::Str("true_graph".into()),
                IrArg::Str("false_graph".into()),
                IrArg::List(vec![IrArg::Node(x)]),
            ],
            NodeMeta::default(),
        );
        let getitem = call(
            &mut parent,
            "getitem",
            "<built-in function getitem>",
            vec![IrArg::Node(cond), IrArg::Int(0)],
            [1, 4, 8],
            I8,
        );
        parent
            .node_mut(getitem)
            .meta
            .set_single_output_qparam(qparam(0.05, 3, I8));
        output(&mut parent, getitem);
        parent
    }

    fn region_constant(graph: &mut IrGraph, name: &str, data: TensorData) -> fx_ir::NodeId {
        let meta = NodeMeta::tensor(data.shape.clone(), data.dtype.torch_str());
        graph.add_constant_node(
            NodeOp::GetAttr,
            name.to_string(),
            name.to_string(),
            NodeTensor {
                source: TensorSource::GraphModuleAttr,
                data,
            },
            meta,
        )
    }

    /// A cond branch ending in explicit IrRewriteConv's conv1d chain, as rc4
    /// builds it: the TOSA conv inherits the conv's qparams (input 0.02 and
    /// weight 0.005, output `(0.04, 7)`); its RESCALE, squeeze view and output
    /// permute have none; the branch output's dequantize carries the output's
    /// quantization.
    fn conv_region() -> IrGraph {
        let mut region = IrGraph::empty();
        let x = placeholder(&mut region, "x_ph", [1, 2, 8], I8);
        let x_nlc = call(
            &mut region,
            "conv_nlc",
            "aten.permute_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[0, 2, 1])],
            [1, 8, 2],
            I8,
        );
        let x_nhwc = call(
            &mut region,
            "conv_nhwc",
            "aten.view_copy.default",
            vec![IrArg::Node(x_nlc), i64_list_arg(&[1, 1, 8, 2])],
            [1, 1, 8, 2],
            I8,
        );
        let weight = region_constant(
            &mut region,
            "conv_weight_ohwi",
            TensorData {
                data: vec![1; 24],
                shape: vec![4, 1, 3, 2],
                dtype: DType::Int8,
            },
        );
        let bias = region_constant(
            &mut region,
            "conv_bias",
            TensorData {
                data: vec![0; 16],
                shape: vec![4],
                dtype: DType::Int32,
            },
        );
        let mut conv_meta = NodeMeta::tensor(vec![1, 1, 8, 4], "torch.int32");
        conv_meta.set_input_qparams_from([(0, qparam(0.02, 0, I8)), (1, qparam(0.005, 0, I8))]);
        conv_meta.set_single_output_qparam(qparam(0.04, 7, I8));
        let conv = call_with_meta(
            &mut region,
            "conv_tosa",
            "backend.tosa.CONV2D.default",
            vec![
                IrArg::Node(x_nhwc),
                IrArg::Node(weight),
                IrArg::Node(bias),
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[0, 0, 1, 1]),
                i64_list_arg(&[1, 1]),
            ],
            conv_meta,
        );
        let no_qparams = |shape: &[i64]| {
            let mut meta = NodeMeta::tensor(shape.to_vec(), I8);
            meta.set_empty_input_qparams();
            meta.set_output_qparams(fx_ir::QParamMap::new());
            meta
        };
        let rescale = call_with_meta(
            &mut region,
            "conv_rescale",
            "backend.tosa.RESCALE.default",
            vec![
                IrArg::Node(conv),
                IrArg::Dtype(I8.into()),
                IrArg::List(vec![IrArg::Float(0.02 * 0.005 / 0.04)]),
                IrArg::Int(0),
                IrArg::Int(7),
            ],
            no_qparams(&[1, 1, 8, 4]),
        );
        let squeeze = call_with_meta(
            &mut region,
            "conv_squeeze",
            "aten.view_copy.default",
            vec![IrArg::Node(rescale), i64_list_arg(&[1, 8, 4])],
            no_qparams(&[1, 8, 4]),
        );
        let permute = call_with_meta(
            &mut region,
            "conv_ncl",
            "aten.permute_copy.default",
            vec![IrArg::Node(squeeze), i64_list_arg(&[0, 2, 1])],
            no_qparams(&[1, 4, 8]),
        );
        let dequantize = call(
            &mut region,
            "output_dequantize",
            "quantized_decomposed.dequantize_per_tensor.default",
            vec![
                IrArg::Node(permute),
                IrArg::Float(0.04),
                IrArg::Int(7),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype(I8.into()),
            ],
            [1, 4, 8],
            "torch.float32",
        );
        region.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::List(vec![IrArg::Node(dequantize)])],
            NodeMeta::default(),
        );
        region
    }

    fn apply(layout: LayoutMode) -> (IrGraph, BTreeMap<String, StashedPipeline>) {
        let mut parent = parent_graph();
        let mut submodules: BTreeMap<String, StashedPipeline> = BTreeMap::from([(
            "true_graph".to_string(),
            (conv_region(), LoweringEffectBatch::default()),
        )]);
        crate::stock_fallback::take();
        apply_boundary_contracts(&mut parent, &mut submodules, layout);
        assert_eq!(crate::stock_fallback::take(), None, "{layout:?}");
        (parent, submodules)
    }

    /// APU-135 S8: explicit mode records a cond branch's output quantization
    /// from its output dequantize, as rc4's FoldAndAnnotateQParamsPass does, so
    /// the output RESCALE takes the branch output from the conv's `(0.04, 7)`
    /// to the parent's `(0.05, 3)`. The walk from the output permute instead
    /// reverse engineers the qparam-less RESCALE from the TOSA conv's inherited
    /// output qparam, `0.04 / (0.02 * 0.005 / 0.04) = 16`, which dim-order mode
    /// would use (it never sees this chain: its convs stay aten until the
    /// serializer).
    #[test]
    fn explicit_region_outputs_take_their_dequantizes_qparams() {
        let (_, submodules) = apply(LayoutMode::Explicit);
        let region = &submodules["true_graph"].0;
        assert!(!region
            .live_nodes()
            .any(|node| node.name == "output_dequantize"));
        let output = region.output_node().expect("an output");
        let recorded = output.meta.input_qparam_at(0).expect("a recorded qparam");
        assert_eq!((recorded.scale, recorded.zp), (0.04, 7));
        let rescale = live_node(region, "conv_ncl_cf_output_rescale");
        assert_eq!(
            format!("{:?}", &rescale.args[1..]),
            format!(
                "{:?}",
                [
                    IrArg::Dtype(I8.into()),
                    IrArg::List(vec![IrArg::Float(0.04 / 0.05)]),
                    IrArg::Int(7),
                    IrArg::Int(3),
                ]
            )
        );
        assert_eq!(
            rescale.first_node_arg(),
            Some(live_node(region, "conv_ncl").id)
        );

        let walked =
            super::metadata::extract_effective_qparam_ir(region, live_node(region, "conv_ncl").id)
                .expect("a qparam");
        assert!((walked.scale - 16.0).abs() < 1e-9, "{}", walked.scale);
    }

    /// Dim-order mode keeps its output contract: nothing is recorded.
    #[test]
    fn dim_order_region_outputs_record_nothing() {
        let (_, submodules) = apply(LayoutMode::DimOrder);
        let region = &submodules["true_graph"].0;
        let output = region.output_node().expect("an output");
        assert_eq!(output.meta.input_qparam_at(0), None);
    }
}
