use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};

pub struct IrAnnotateOutputDimOrderPass;

fn collect_output_node_ids(arg: &IrArg, output_nodes: &mut Vec<NodeId>) {
    match arg {
        IrArg::Node(id) => output_nodes.push(*id),
        IrArg::List(items) => {
            for item in items {
                collect_output_node_ids(item, output_nodes);
            }
        }
        _ => {}
    }
}

impl IrPass for IrAnnotateOutputDimOrderPass {
    fn name(&self) -> &str {
        "IrAnnotateOutputDimOrder"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let output_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|node| node.alive && node.op == NodeOp::Output)
            .map(|node| node.id)
            .collect();

        let mut modified = false;
        for output_id in output_ids {
            if graph.nodes[output_id].meta.original_dim_orders.is_some() {
                continue;
            }

            let mut output_nodes = Vec::new();
            for arg in &graph.nodes[output_id].args {
                collect_output_node_ids(arg, &mut output_nodes);
            }

            let output_dim_orders: Vec<Vec<usize>> = output_nodes
                .into_iter()
                .filter_map(|id| {
                    graph.nodes[id]
                        .meta
                        .val_dim_order()
                        .map(|order| order.to_vec())
                })
                .collect();
            if output_dim_orders.is_empty() {
                continue;
            }

            graph.nodes[output_id]
                .meta
                .set_original_dim_orders(output_dim_orders);
            modified = true;
        }

        modified
    }
}
