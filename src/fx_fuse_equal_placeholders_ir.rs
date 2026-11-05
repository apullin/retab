use std::collections::HashMap;

use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, NodeId, NodeOp};
use crate::lowering_effects::{
    DeleteConstantPlaceholderEffect, LoweringEffect, LoweringEffectBatch,
};

/// Pairs `(duplicate, canonical)` of constant placeholders with equal dtype, shape,
/// special dtype and bytes; the canonical one is the first in graph order. Only
/// placeholders that share metadata have their bytes compared, so large unique
/// weights are never hashed or copied.
pub fn find_duplicate_placeholder_pairs(graph: &IrGraph) -> Vec<(NodeId, NodeId)> {
    let mut by_meta: HashMap<(&str, crate::graph::ir::DType, &[i64]), Vec<NodeId>> =
        HashMap::new();
    for node in graph.live_nodes() {
        if !matches!(node.op, NodeOp::Placeholder | NodeOp::GetAttr) {
            continue;
        }
        let Some(tensor) = node.tensor.as_ref() else {
            continue;
        };
        let special_dtype = node.meta.extra_str("tosa_special_dtype").unwrap_or_default();
        by_meta
            .entry((special_dtype, tensor.data.dtype, tensor.data.shape.as_slice()))
            .or_default()
            .push(node.id);
    }

    let mut pairs = Vec::new();
    for ids in by_meta.values().filter(|ids| ids.len() > 1) {
        // Same-shape weights (e.g. one per transformer layer) are usually different:
        // a sampled fingerprint separates them without reading all their bytes, and
        // a byte comparison confirms the rare candidates that match.
        let mut canonicals: HashMap<u64, Vec<(NodeId, &[u8])>> = HashMap::new();
        for &id in ids {
            let Some(data) = graph.nodes[id].tensor.as_ref().map(|t| t.data.data.as_slice())
            else {
                continue;
            };
            let candidates = canonicals.entry(sampled_fingerprint(data)).or_default();
            match candidates.iter().find(|(_, other)| *other == data) {
                Some(&(canonical, _)) => pairs.push((id, canonical)),
                None => candidates.push((id, data)),
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Hash of the length and up to 64 bytes from each of the start, middle and end.
fn sampled_fingerprint(data: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    const SAMPLE: usize = 64;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.len().hash(&mut hasher);
    let middle = data.len() / 2;
    for start in [0, middle.saturating_sub(SAMPLE / 2), data.len().saturating_sub(SAMPLE)] {
        data[start..(start + SAMPLE).min(data.len())].hash(&mut hasher);
    }
    hasher.finish()
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
    fn same_fingerprint_with_different_bytes_is_not_a_duplicate() {
        // 4096 bytes: the sampled fingerprint never reads offset 1000.
        let base = vec![7u8; 4096];
        let mut changed = base.clone();
        changed[1000] = 8;
        assert_eq!(sampled_fingerprint(&base), sampled_fingerprint(&changed));

        let mut graph = IrGraph::empty();
        let first = add_placeholder(&mut graph, "a", base.clone(), vec![1024]);
        add_placeholder(&mut graph, "b", changed, vec![1024]);
        let third = add_placeholder(&mut graph, "c", base, vec![1024]);
        assert_eq!(find_duplicate_placeholder_pairs(&graph), vec![(third, first)]);
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
