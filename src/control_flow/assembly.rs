use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::graph::ir::{Arg, DType, NodeId};
use crate::graph::Graph;
use crate::store::TensorStore;
use crate::StashedPipeline;

use super::{collect_region_metadata, WhileBodyInputCounts};

fn mark_scalar_outputs(graph: &mut Graph) {
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

pub(crate) fn append_subgraphs_from_ir(
    graphs: &mut Vec<(String, Graph)>,
    store: &mut TensorStore,
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> PyResult<()> {
    let (while_cond_graph_names, while_body_input_counts) =
        collect_region_metadata(parent_ir, submodules);

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
            crate::ingest::from_ir::graph_from_ir_preserve_placeholders(ir, &mut sub_store);
        crate::ingest::from_ir::apply_effects_preserve_placeholders(
            &mut sub_graph,
            &mut sub_store,
            effects,
        );
        crate::ingest::fx_bridge::finalize_converted_graph(&mut sub_graph);
        if while_cond_graph_names.contains(&region) {
            mark_scalar_outputs(&mut sub_graph);
        }
        if let Some(counts) = while_body_input_counts.get(&region).copied() {
            append_while_body_passthrough_outputs(&mut sub_graph, counts);
        }
        drop_subgraph_input_store_entries(&sub_graph, &mut sub_store);
        if crate::config::from_ir_debug_enabled() {
            crate::ingest::from_ir::debug_dump(&sub_graph, &format!("CF SUBGRAPH {region}"));
        }
        crate::ingest::from_ir::prefix_graph(
            &mut sub_graph,
            &mut sub_store,
            &format!("{region}::"),
        );
        for (key, value) in sub_store.buffers {
            if store.get(&key).is_none() {
                store.insert(key, value);
            }
        }
        graphs.push((region, sub_graph));
    }

    Ok(())
}
