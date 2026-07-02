use crate::graph::ir::OpType;
use crate::graph::Graph;
use crate::store::TensorStore;

use super::tensor_layout::TensorDeclarationContext;
use super::tosa_generated::{Attribute, TosaOperator, TosaShape, TosaTensor};
use super::{output_layout, tosa_generated, visitors};

pub(super) struct SerializedGraphItems {
    pub(super) tensors: Vec<TosaTensor>,
    pub(super) shapes: Vec<TosaShape>,
    pub(super) operators: Vec<TosaOperator>,
    pub(super) input_names: Vec<String>,
    pub(super) output_names: Vec<String>,
}

pub(super) struct SerializedRegion {
    pub(super) name: String,
    pub(super) items: SerializedGraphItems,
}

pub(super) fn collect_regions(
    graphs: &[(String, Graph)],
    store: &TensorStore,
) -> Vec<SerializedRegion> {
    graphs
        .iter()
        .map(|(name, graph)| collect_region(name, graph, store))
        .collect()
}

fn collect_region(name: &str, graph: &Graph, store: &TensorStore) -> SerializedRegion {
    // Region outputs are only true external program outputs on the root graph.
    // Control-flow subgraphs feed COND_IF/WHILE_LOOP directly and must retain
    // their internal layout instead of normalizing back to user-facing shape.
    let normalize_outputs = name == "main";
    SerializedRegion {
        name: name.to_string(),
        items: collect_graph_items(graph, store, normalize_outputs),
    }
}

fn collect_graph_items(
    graph: &Graph,
    store: &TensorStore,
    normalize_outputs: bool,
) -> SerializedGraphItems {
    let debug = crate::config::tosa_engine_debug_enabled();
    let profile = crate::config::profile_ingest_serialize_enabled();
    let sorted = graph.all_node_ids();
    if debug {
        eprintln!("[Serialize] using insertion order, {} nodes", sorted.len());
        validate_insertion_order(graph, &sorted);
    }

    let tensor_decls = TensorDeclarationContext::new(graph);

    let mut tensors: Vec<TosaTensor> = Vec::new();
    let mut shapes: Vec<TosaShape> = Vec::new();
    let mut operators: Vec<TosaOperator> = Vec::new();
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();

    let mut const_placeholder_count = 0usize;
    let mut const_placeholder_bytes = 0usize;
    let mut const_placeholder_s = 0.0f64;
    let mut runtime_placeholder_count = 0usize;
    let mut runtime_placeholder_s = 0.0f64;
    let mut call_function_count = 0usize;
    let mut call_function_s = 0.0f64;
    let mut output_count = 0usize;
    let mut output_s = 0.0f64;
    let mut slow_nodes: Vec<(f64, usize, String, String, usize)> = Vec::new();

    for (loop_idx, &node_id) in sorted.iter().enumerate() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let node_start = std::time::Instant::now();

        if debug {
            eprintln!(
                "[Serialize] node '{}' op={:?} target={:?}",
                node.name, node.op, node.target
            );
        }

        match node.op {
            OpType::Placeholder => {
                let name = &node.name;

                if let Some(tensor_data) = store.get(name) {
                    let phase_start = std::time::Instant::now();
                    tensors.push(tensor_decls.const_placeholder(node, tensor_data));
                    operators.push(TosaOperator::new(
                        tosa_generated::Op::Const,
                        Attribute::Const(tosa_generated::ConstAttribute {}),
                        vec![],
                        vec![name.clone()],
                    ));
                    const_placeholder_count += 1;
                    const_placeholder_bytes += tensor_data.data.len();
                    const_placeholder_s += phase_start.elapsed().as_secs_f64();
                } else {
                    let phase_start = std::time::Instant::now();
                    tensors.push(tensor_decls.runtime_placeholder(node));
                    input_names.push(name.clone());
                    runtime_placeholder_count += 1;
                    runtime_placeholder_s += phase_start.elapsed().as_secs_f64();
                }
            }
            OpType::CallFunction => {
                let phase_start = std::time::Instant::now();
                if let Some(emit_result) = visitors::emit_node(graph, node_id, node, store) {
                    tensors.extend(emit_result.tensors);
                    shapes.extend(emit_result.shapes);
                    operators.extend(emit_result.operators);
                } else {
                    eprintln!(
                        "[serialize] WARNING: skipping unhandled call_function node '{}' target {:?}",
                        node.name, node.target
                    );
                }
                call_function_count += 1;
                call_function_s += phase_start.elapsed().as_secs_f64();
            }
            OpType::Output => {
                let phase_start = std::time::Instant::now();
                for arg in &node.args {
                    if normalize_outputs {
                        output_layout::append_normalized_output_names(
                            graph,
                            store,
                            arg,
                            &mut tensors,
                            &mut operators,
                            &mut output_names,
                        );
                    } else {
                        output_layout::append_output_names(graph, arg, &mut output_names);
                    }
                }
                output_count += 1;
                output_s += phase_start.elapsed().as_secs_f64();
            }
        }

        let elapsed = node_start.elapsed();
        if profile && elapsed.as_secs_f64() > 0.005 {
            let bytes = store
                .get(&node.name)
                .map(|tensor| tensor.data.len())
                .unwrap_or(0);
            slow_nodes.push((
                elapsed.as_secs_f64(),
                loop_idx,
                node.name.clone(),
                format!("{:?}", node.target),
                bytes,
            ));
        }
        if debug && elapsed.as_millis() > 100 {
            eprintln!(
                "[Serialize] SLOW node {}/{} name={} target={:?} took {:?}",
                loop_idx,
                sorted.len(),
                node.name,
                node.target,
                elapsed
            );
        }
    }

    if profile {
        eprintln!(
            "[RETAB][collection-profile] nodes={} const_placeholders={} const_bytes={} const_s={:.6} runtime_placeholders={} runtime_s={:.6} call_functions={} call_s={:.6} outputs={} output_s={:.6}",
            sorted.len(),
            const_placeholder_count,
            const_placeholder_bytes,
            const_placeholder_s,
            runtime_placeholder_count,
            runtime_placeholder_s,
            call_function_count,
            call_function_s,
            output_count,
            output_s
        );
        slow_nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (time_s, index, name, target, bytes)) in slow_nodes.iter().take(10).enumerate() {
            eprintln!(
                "[RETAB][collection-profile] slow_node={} index={} time_s={:.6} bytes={} name={} target={}",
                rank + 1,
                index,
                time_s,
                bytes,
                name,
                target
            );
        }
    }

    if debug {
        eprintln!(
            "[Serialize] topo sort done, collected {} tensors, {} operators, {} shapes",
            tensors.len(),
            operators.len(),
            shapes.len()
        );
    }

    SerializedGraphItems {
        tensors,
        shapes,
        operators,
        input_names,
        output_names,
    }
}

fn validate_insertion_order(graph: &Graph, sorted: &[crate::graph::ir::NodeId]) {
    let mut position = std::collections::HashMap::new();
    for (i, &id) in sorted.iter().enumerate() {
        position.insert(id, i);
    }

    let mut violations = 0;
    for (i, &id) in sorted.iter().enumerate() {
        let Some(node) = graph.node(id) else {
            continue;
        };
        for input_id in Graph::collect_input_ids(node) {
            let Some(&input_pos) = position.get(&input_id) else {
                continue;
            };
            if input_pos < i {
                continue;
            }
            if violations < 10 {
                eprintln!(
                    "[Serialize] WARNING: '{}' at position {} depends on input at position {}",
                    node.name, i, input_pos
                );
            }
            violations += 1;
        }
    }

    if violations > 0 {
        eprintln!(
            "[Serialize] WARNING: {} topological order violations detected",
            violations
        );
    }
}
