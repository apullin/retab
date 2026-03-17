use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::{TensorData, TensorStore};

/// Walk placeholder nodes with Int64 dtype and downcast their tensor data in
/// TensorStore from Int64 to Int32. Also updates the node's meta dtype.
///
/// Int64 is not supported by TOSA/Ethos-U55 and must be downcast.
pub struct CastToInt32Pass;

impl Pass for CastToInt32Pass {
    fn name(&self) -> &str {
        "CastToInt32"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let node = match graph.node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Only process placeholders / TosaConst with Int64 dtype
            if node.meta.val.dtype != DType::Int64 {
                continue;
            }

            if node.op != OpType::Placeholder && node.target != OpTarget::TosaConst {
                continue;
            }

            let name = node.name.clone();

            // Convert data in store if present
            if let Some(tensor_data) = store.get(&name) {
                if tensor_data.dtype == DType::Int64 {
                    let i64_vals: Vec<i64> = tensor_data
                        .data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();

                    let i32_data: Vec<u8> = i64_vals
                        .iter()
                        .flat_map(|&v| (v as i32).to_le_bytes())
                        .collect();

                    let new_shape = tensor_data.shape.clone();
                    store.insert(
                        name,
                        TensorData {
                            data: i32_data,
                            shape: new_shape,
                            dtype: DType::Int32,
                        },
                    );
                }
            }

            // Update the node's meta dtype
            if let Some(node) = graph.node_mut(node_id) {
                node.meta.val.dtype = DType::Int32;
                modified = true;
            }
        }

        modified
    }
}

/// Walk ALL buffers in TensorStore (not just those referenced by graph nodes)
/// and convert any Int64 buffers to Int32. This catches stray buffers that may
/// not have a corresponding placeholder node.
pub struct CastInt64BuffersToInt32Pass;

impl Pass for CastInt64BuffersToInt32Pass {
    fn name(&self) -> &str {
        "CastInt64BuffersToInt32"
    }

    fn run(&self, _graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;

        let i64_keys: Vec<String> = store
            .buffers
            .iter()
            .filter(|(_, v)| v.dtype == DType::Int64)
            .map(|(k, _)| k.clone())
            .collect();

        for key in i64_keys {
            if let Some(tensor_data) = store.remove(&key) {
                let i64_vals: Vec<i64> = tensor_data
                    .data
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    })
                    .collect();

                let i32_data: Vec<u8> = i64_vals
                    .iter()
                    .flat_map(|&v| (v as i32).to_le_bytes())
                    .collect();

                store.insert(
                    key,
                    TensorData {
                        data: i32_data,
                        shape: tensor_data.shape,
                        dtype: DType::Int32,
                    },
                );
                modified = true;
            }
        }

        modified
    }
}
