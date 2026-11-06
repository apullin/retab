use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::graph::ir::{Arg, NodeId};
use crate::graph::Graph;
use crate::store::TensorStore;
use crate::StashedPipeline;

use super::{collect_region_metadata, region_name_from_ir_arg, WhileBodyInputCounts};

fn append_while_body_passthrough_outputs(graph: &mut Graph, counts: WhileBodyInputCounts) {
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

/// Child regions of one graph in stock's emission order: cond `[1, 2]`
/// (then, else), while `[0, 1]` (cond, body).
fn child_regions(ir: &fx_ir::IrGraph, region: Option<&str>) -> Vec<String> {
    let mut children = Vec::new();
    for node in ir.live_nodes() {
        let branches = if node.is_cond() {
            [node.args.get(1), node.args.get(2)]
        } else if node.is_while_loop() {
            [node.args.first(), node.args.get(1)]
        } else {
            continue;
        };
        for arg in branches {
            if let Some(name) = region_name_from_ir_arg(ir, arg, region) {
                children.push(name);
            }
        }
    }
    children
}

/// Regions in stock's pre-order emission order: the parent's children first in
/// `[1, 2]`/`[0, 1]` order, recursing into each before the next sibling. Any
/// stashed region no live node references is appended last, as before.
fn ordered_regions(
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> Vec<String> {
    let mut order = Vec::new();
    let mut visited = BTreeSet::new();
    let mut pending = child_regions(parent_ir, None);
    pending.reverse();
    while let Some(region) = pending.pop() {
        if !visited.insert(region.clone()) {
            continue;
        }
        order.push(region.clone());
        if let Some((ir, _)) = submodules.get(&region) {
            let mut children = child_regions(ir, Some(region.as_str()));
            children.reverse();
            pending.extend(children);
        }
    }
    for region in submodules.keys() {
        if visited.insert(region.clone()) {
            order.push(region.clone());
        }
    }
    order
}

/// Convert one stashed region like the parent: in the parent's layout mode, which
/// finalize depends on.
fn ingest_region(
    (ir, effects): &StashedPipeline,
    layout: crate::pipeline_config::LayoutMode,
) -> (Graph, TensorStore) {
    let mut sub_store = TensorStore::new();
    let mut sub_graph =
        crate::ingest::from_ir::graph_from_ir_preserve_placeholders(ir, &mut sub_store);
    sub_graph.layout = layout;
    crate::ingest::from_ir::apply_effects_preserve_placeholders(
        &mut sub_graph,
        &mut sub_store,
        effects,
    );
    crate::ingest::finalize::finalize_converted_graph(&mut sub_graph);
    (sub_graph, sub_store)
}

pub(crate) fn append_subgraphs_from_ir(
    graphs: &mut Vec<(String, Graph)>,
    store: &mut TensorStore,
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    layout: crate::pipeline_config::LayoutMode,
) -> PyResult<()> {
    let (_, while_body_input_counts) = collect_region_metadata(parent_ir, submodules);

    for region in ordered_regions(parent_ir, submodules) {
        if graphs.iter().any(|(name, _)| name == &region) {
            continue;
        }
        let Some(stashed) = submodules.get(&region) else {
            return Err(PyRuntimeError::new_err(format!(
                "missing stashed IrGraph for control-flow region '{region}'"
            )));
        };

        let (mut sub_graph, mut sub_store) = ingest_region(stashed, layout);
        if let Some(counts) = while_body_input_counts.get(&region).copied() {
            append_while_body_passthrough_outputs(&mut sub_graph, counts);
        }
        drop_subgraph_input_store_entries(&sub_graph, &mut sub_store);
        if crate::config::from_ir_debug_enabled() {
            crate::ingest::from_ir::debug_dump(&sub_graph, &format!("CF SUBGRAPH {region}"));
        }
        crate::ingest::from_ir::suffix_graph(&mut sub_graph, &mut sub_store, &region);
        for (key, value) in sub_store.buffers {
            match store.get(&key) {
                None => store.insert(key, value),
                Some(existing) if existing.data == value.data => {}
                // Stock's `name_region` tensor names can collide with a parent
                // constant's name; stock keeps one tensor list per block, RETAB one
                // store, so hand the graph to stock rather than lose either tensor.
                Some(_) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "RETAB_STOCK_FALLBACK: control-flow constant name collision: \
                         region '{region}' constant '{key}' differs from an earlier one"
                    )));
                }
            }
        }
        graphs.push((region, sub_graph));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, NodeOp};
    use crate::lowering_effects::LoweringEffectBatch;

    fn add_cond(graph: &mut IrGraph, name: &str, then: &str, else_: &str) {
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            "torch.ops.higher_order.cond".to_string(),
            vec![
                IrArg::Int(1),
                IrArg::Str(then.to_string()),
                IrArg::Str(else_.to_string()),
            ],
            NodeMeta::default(),
        );
    }

    fn stash(graph: IrGraph) -> StashedPipeline {
        (graph, LoweringEffectBatch::default())
    }

    #[test]
    fn nested_regions_emit_then_before_else_pre_order() {
        let mut parent = IrGraph::empty();
        add_cond(&mut parent, "outer", "b_branch", "a_branch");

        let mut inner = IrGraph::empty();
        add_cond(&mut inner, "nested", "z_leaf", "a_leaf");

        let mut submodules = BTreeMap::new();
        submodules.insert("b_branch".to_string(), stash(inner));
        submodules.insert("b_branch__z_leaf".to_string(), stash(IrGraph::empty()));
        submodules.insert("b_branch__a_leaf".to_string(), stash(IrGraph::empty()));
        submodules.insert("a_branch".to_string(), stash(IrGraph::empty()));

        // Alphabetical order would put a_branch first; stock emits [1, 2].
        assert_eq!(
            ordered_regions(&parent, &submodules),
            vec![
                "b_branch".to_string(),
                "b_branch__z_leaf".to_string(),
                "b_branch__a_leaf".to_string(),
                "a_branch".to_string(),
            ]
        );
    }

    #[test]
    fn regions_are_ingested_in_the_parent_layout_mode() {
        use crate::graph::ir::{DType, OpTarget};
        use crate::pipeline_config::LayoutMode;
        use crate::test_ir::{call, output, placeholder, qparam};

        // An explicit-mode RESIZE keeps its int32 val through finalize despite the int8
        // qparams it inherits, so its dtype shows which mode finalize ran in.
        let mut region = IrGraph::empty();
        let x = placeholder(&mut region, "x", [1, 5, 5, 3], "torch.int8");
        let resize = call(
            &mut region,
            "resize",
            "backend.tosa.RESIZE.default",
            vec![IrArg::Node(x)],
            [1, 9, 9, 3],
            "torch.int32",
        );
        region
            .node_mut(resize)
            .meta
            .set_both_qparams(qparam(0.05, 0, "torch.int8"));
        output(&mut region, resize);
        let region = stash(region);

        for (layout, resize_dtype) in [
            (LayoutMode::DimOrder, DType::Int8),
            (LayoutMode::Explicit, DType::Int32),
        ] {
            let (graph, _) = ingest_region(&region, layout);
            assert_eq!(graph.layout, layout);
            let resize = graph
                .nodes
                .values()
                .find(|node| node.target == OpTarget::TosaResize)
                .unwrap();
            assert_eq!(resize.meta.val.dtype, resize_dtype, "{layout:?}");
        }
    }
}
