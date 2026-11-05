use crate::graph::ir::OpType;
use crate::graph::Graph;
use crate::store::TensorStore;

use super::constant_pool::ConstantPool;
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
) -> Result<Vec<SerializedRegion>, String> {
    graphs
        .iter()
        .map(|(name, graph)| collect_region(name, graph, store))
        .collect()
}

fn collect_region(
    name: &str,
    graph: &Graph,
    store: &TensorStore,
) -> Result<SerializedRegion, String> {
    // Region outputs are only true external program outputs on the root graph.
    // Control-flow subgraphs feed COND_IF/WHILE_LOOP directly and must retain
    // their internal layout instead of normalizing back to user-facing shape.
    let normalize_outputs = name == "main";
    Ok(SerializedRegion {
        name: name.to_string(),
        items: collect_graph_items(name, graph, store, normalize_outputs)?,
    })
}

fn collect_graph_items(
    region: &str,
    graph: &Graph,
    store: &TensorStore,
    normalize_outputs: bool,
) -> Result<SerializedGraphItems, String> {
    let debug = crate::config::tosa_engine_debug_enabled();
    let sorted = graph.all_node_ids();
    if debug {
        eprintln!("[Serialize] using insertion order, {} nodes", sorted.len());
        validate_insertion_order(graph, &sorted);
    }

    let tensor_decls = TensorDeclarationContext::new(graph);

    let mut tensors: Vec<TosaTensor> = Vec::new();
    let mut shapes: Vec<TosaShape> = Vec::new();
    let mut operators: Vec<TosaOperator> = Vec::new();
    let mut constant_pool = ConstantPool::default();
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();
    let mut unhandled: Vec<String> = Vec::new();

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
                    tensors.push(tensor_decls.const_placeholder(node, tensor_data));
                    operators.push(TosaOperator::new(
                        tosa_generated::Op::Const,
                        Attribute::Const(tosa_generated::ConstAttribute {}),
                        vec![],
                        vec![name.clone()],
                    ));
                } else {
                    tensors.push(tensor_decls.runtime_placeholder(node));
                    input_names.push(name.clone());
                }
            }
            OpType::CallFunction => {
                if let Some(mut emit_result) = visitors::emit_node(graph, node_id, node, store) {
                    constant_pool.absorb(&mut emit_result);
                    tensors.extend(emit_result.tensors);
                    shapes.extend(emit_result.shapes);
                    operators.extend(emit_result.operators);
                } else {
                    unhandled.push(format!("'{}' ({:?})", node.name, node.target));
                }
            }
            OpType::Output => {
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
            }
        }

        let elapsed = node_start.elapsed();
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

    let simplified = super::transpose_peephole::simplify_transposes(
        &mut tensors,
        &mut shapes,
        &mut operators,
        &input_names,
        &mut output_names,
    );
    if debug && simplified > 0 {
        eprintln!("[Serialize] removed or relabeled {simplified} redundant TRANSPOSE op(s)");
    }

    if debug {
        eprintln!(
            "[Serialize] topo sort done, collected {} tensors, {} operators, {} shapes",
            tensors.len(),
            operators.len(),
            shapes.len()
        );
    }

    if !unhandled.is_empty() {
        return Err(format!(
            "RETAB has no TOSA emitter for {} node(s) in region '{}': {}",
            unhandled.len(),
            region,
            unhandled.join(", ")
        ));
    }

    Ok(SerializedGraphItems {
        tensors,
        shapes,
        operators,
        input_names,
        output_names,
    })
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
