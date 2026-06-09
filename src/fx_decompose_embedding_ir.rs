//! Pure Rust IrPass: aten.embedding(weight, indices, ...) -> aten.index.Tensor(weight, [indices]).
//!
//! Embedding is a row-gather (weight[indices]); routing it through index.Tensor reuses the existing
//! GATHER lowering (IrDecomposeIndexTensor). Without this, aten.embedding is an unhandled serialize
//! target that gets dropped, leaving every downstream consumer with a dangling tensor reference
//! (the ref-model "(!found)" failure that broke the text-encoder transformer models).

use crate::fx_ir::{IrArg, IrGraph, IrPass};

pub struct IrDecomposeEmbeddingPass;

impl IrPass for IrDecomposeEmbeddingPass {
    fn name(&self) -> &str {
        "IrDecomposeEmbedding"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in 0..graph.nodes.len() {
            if !graph.nodes[id].alive || !graph.nodes[id].target.contains("aten.embedding") {
                continue;
            }
            let args = graph.nodes[id].args.clone();
            let Some(weight_id) = args.first().and_then(IrArg::as_node) else {
                continue;
            };
            let Some(indices_id) = args.get(1).and_then(IrArg::as_node) else {
                continue;
            };

            // weight[indices]: index dim 0 with the (possibly N-d) indices tensor. The output meta
            // (indices.shape + weight.shape[1:]) is exactly the embedding output, so reuse it.
            let name = graph.nodes[id].name.clone();
            let meta = graph.nodes[id].meta.clone();
            let index = graph.add_anchored_call_function(
                id,
                format!("{}_as_index", name),
                "aten.index.Tensor".to_string(),
                vec![
                    IrArg::Node(weight_id),
                    IrArg::List(vec![IrArg::Node(indices_id)]),
                ],
                meta,
            );
            graph.replace_and_erase(id, index);
            modified = true;
        }
        modified
    }
}
