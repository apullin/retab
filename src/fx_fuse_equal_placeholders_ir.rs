use std::collections::HashMap;

use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, NodeId, NodeOp};
use crate::lowering_effects::{
    DeleteConstantPlaceholderEffect, LoweringEffect, LoweringEffectBatch,
};

fn placeholder_fingerprint(
    graph: &IrGraph,
    id: NodeId,
) -> Option<(String, crate::graph::ir::DType, Vec<i64>, Vec<u8>)> {
    let node = &graph.nodes[id];
    if !matches!(node.op, NodeOp::Placeholder | NodeOp::GetAttr) {
        return None;
    }
    let tensor = node.tensor.as_ref()?;
    let special_dtype = node
        .meta
        .extra_str("tosa_special_dtype")
        .unwrap_or_default()
        .to_string();
    Some((
        special_dtype,
        tensor.data.dtype,
        tensor.data.shape.clone(),
        tensor.data.data.clone(),
    ))
}

pub fn find_duplicate_placeholder_pairs(graph: &IrGraph) -> Vec<(NodeId, NodeId)> {
    let mut buckets: HashMap<(String, crate::graph::ir::DType, Vec<i64>, Vec<u8>), Vec<NodeId>> =
        HashMap::new();

    for id in graph.live_node_ids() {
        let Some(key) = placeholder_fingerprint(graph, id) else {
            continue;
        };
        buckets.entry(key).or_default().push(id);
    }

    let mut pairs = Vec::new();
    for ids in buckets.values() {
        let Some((&canonical, duplicates)) = ids.split_first() else {
            continue;
        };
        for &dup in duplicates {
            pairs.push((dup, canonical));
        }
    }
    pairs
}

pub struct IrFuseEqualPlaceholdersEffectPass;

impl EffectfulIrPass for IrFuseEqualPlaceholdersEffectPass {
    fn name(&self) -> &str {
        "IrFuseEqualPlaceholdersEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let mut modified = false;
        let mut effects = LoweringEffectBatch::default();

        for (dup, canonical) in find_duplicate_placeholder_pairs(graph) {
            let placeholder_name = graph.node(dup).name.clone();
            graph.replace_and_erase(dup, canonical);
            effects.push(LoweringEffect::DeleteConstantPlaceholder(
                DeleteConstantPlaceholderEffect {
                    placeholder_name,
                    allow_missing: true,
                },
            ));
            modified = true;
        }

        EffectfulIrPassResult { modified, effects }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, MetaValue, NodeMeta};
    use crate::store::TensorData;

    use super::*;

    fn add_placeholder(graph: &mut IrGraph, name: &str, data: Vec<u8>, shape: Vec<i64>) -> NodeId {
        let id = graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::<IrArg>::new(),
            NodeMeta::default(),
        );
        graph.nodes[id].tensor = Some(crate::fx_ir::NodeTensor {
            source: crate::fx_ir::TensorSource::GraphModuleAttr,
            data: TensorData {
                data,
                shape,
                dtype: crate::graph::ir::DType::Int32,
            },
        });
        id
    }

    #[test]
    fn finds_duplicate_placeholder_pairs() {
        let mut graph = IrGraph::empty();
        let canonical = add_placeholder(&mut graph, "a", 1i32.to_le_bytes().to_vec(), vec![1]);
        let duplicate = add_placeholder(&mut graph, "b", 1i32.to_le_bytes().to_vec(), vec![1]);
        let distinct = add_placeholder(&mut graph, "c", 2i32.to_le_bytes().to_vec(), vec![1]);
        graph.nodes[distinct]
            .meta
            .set_extra("tosa_special_dtype", MetaValue::Str("INT48".to_string()));

        let pairs = find_duplicate_placeholder_pairs(&graph);
        assert_eq!(pairs, vec![(duplicate, canonical)]);
    }

    #[test]
    fn effectful_pass_plans_delete_effects_and_rewrites_graph() {
        let mut graph = IrGraph::empty();
        let canonical = add_placeholder(&mut graph, "a", 1i32.to_le_bytes().to_vec(), vec![1]);
        let duplicate = add_placeholder(&mut graph, "b", 1i32.to_le_bytes().to_vec(), vec![1]);
        let user = graph.add_node(
            NodeOp::CallFunction,
            "use_b".to_string(),
            "aten.add.Tensor".to_string(),
            vec![crate::fx_ir::IrArg::Node(duplicate)],
            NodeMeta::default(),
        );

        let pass = IrFuseEqualPlaceholdersEffectPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(graph.node(user).args[0].as_node(), Some(canonical));
        assert!(!graph.node(duplicate).alive);
        assert_eq!(result.effects.items.len(), 1);
        let LoweringEffect::DeleteConstantPlaceholder(effect) = &result.effects.items[0] else {
            panic!("expected delete effect");
        };
        assert_eq!(effect.placeholder_name, "b");
    }
}
