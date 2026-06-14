//! Pure Rust IrPass: aten.embedding(weight, indices, ...) -> aten.index.Tensor(weight, [indices]).
//!
//! Embedding is a row-gather (weight[indices]); routing it through index.Tensor reuses the existing
//! GATHER lowering (IrDecomposeIndexTensor). Without this, aten.embedding is an unhandled serialize
//! target that gets dropped, leaving every downstream consumer with a dangling tensor reference
//! (the ref-model "(!found)" failure that broke the text-encoder transformer models).

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::is_embedding_target;

pub struct IrDecomposeEmbeddingPass;

impl IrPass for IrDecomposeEmbeddingPass {
    fn name(&self) -> &str {
        "IrDecomposeEmbedding"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_embedding_target(&graph.nodes[id].target) {
                continue;
            }
            let Some(weight_id) = graph.nodes[id].arg_node(0) else {
                continue;
            };
            let Some(indices_id) = graph.nodes[id].arg_node(1) else {
                continue;
            };

            // weight[indices]: index dim 0 with the (possibly N-d) indices tensor. The output meta
            // (indices.shape + weight.shape[1:]) is exactly the embedding output, so reuse it.
            let name = graph.nodes[id].name.clone();
            let meta = graph.nodes[id].meta.clone();
            graph.replace_with_call_function(
                id,
                format!("{}_as_index", name),
                "aten.index.Tensor".to_string(),
                vec![
                    IrArg::Node(weight_id),
                    IrArg::List(vec![IrArg::Node(indices_id)]),
                ],
                meta,
            );
            modified = true;
        }
        modified
    }
}
