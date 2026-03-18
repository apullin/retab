use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Rewrite ATen Matmul / Bmm nodes to TOSA Matmul.
///
/// TOSA matmul expects 3D inputs: (B, M, K) x (B, K, N) -> (B, M, N).
/// For 2D inputs (M, K) x (K, N), we unsqueeze batch dim, matmul, then squeeze.
/// For quantized inputs, zero-point constants are attached.
pub struct RewriteMatmulPass;

impl Pass for RewriteMatmulPass {
    fn name(&self) -> &str {
        "RewriteMatmul"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            match target {
                OpTarget::Matmul | OpTarget::Bmm => {
                    if rewrite_matmul(graph, store, node_id) {
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        modified
    }
}

fn rewrite_matmul(graph: &mut Graph, store: &mut TensorStore, node_id: NodeId) -> bool {
    let (a_id, b_id, a_rank, node_name, input_qparams) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let a_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let b_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };

        let a_rank = graph
            .node(a_id)
            .map(|n| n.meta.val.shape_len as usize)
            .unwrap_or(0);

        (
            a_id,
            b_id,
            a_rank,
            node.name.clone(),
            node.meta.input_qparams.clone(),
        )
    };

    // Create zero-point constants for quantized matmul
    let a_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 0)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    let b_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 1)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    let azp_name = format!("{}_a_zp", node_name);
    store.insert(
        azp_name.clone(),
        TensorData {
            data: (a_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let azp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        azp_name,
    );

    let bzp_name = format!("{}_b_zp", node_name);
    store.insert(
        bzp_name.clone(),
        TensorData {
            data: (b_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let bzp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        bzp_name,
    );

    // For 2D matmul (mm), inputs are already handled as batched in TOSA
    // by treating as batch=1. The shape propagation handles this.
    // Build args: [A, B, A_zp, B_zp]
    let new_args: smallvec::SmallVec<[Arg; 4]> = smallvec![
        Arg::Node(a_id),
        Arg::Node(b_id),
        Arg::Node(azp_id),
        Arg::Node(bzp_id),
    ];

    if let Some(node) = graph.node_mut(node_id) {
        node.target = OpTarget::TosaMatmul;
        node.args = new_args;
        node.kwargs = vec![];
        // Track whether we need to handle 2D->3D promotion
        if a_rank == 2 {
            node.kwargs
                .push(("promoted_from_2d".into(), Arg::Bool(true)));
        }
    }

    shape::propagate_shape(graph, node_id);

    true
}
