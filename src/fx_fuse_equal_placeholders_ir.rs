//! rc4's FuseEqualPlaceholdersPass
//! (`backends/arm/_passes/fuse_equal_placeholders_pass.py`): constants holding
//! equal tensors become one constant that all their users read.
//!
//! rc4 buckets constants by special dtype, `str(tensor.dtype)` and shape, and
//! compares a hash of the bytes (`_get_param_key`, `:40-72`). RETAB compares
//! the stored data: its dtype, shape and bytes, plus the special dtype. The
//! first constant of a bucket in graph order survives, where rc4 replaces the
//! bucket with a `<first>_common` copy (`:105-131`).
//!
//! The two keys part where RETAB's importer narrows int64 data to int32 and
//! keeps the int64 val (`tensor_data::extract_tensor_data`). An unused int64
//! zero point, a per-channel weight's that the Q/DQ fold leaves behind, then
//! has the same data as an all-zero int32 bias. When the zero point comes
//! first, the bias's users read a constant whose val is int64. The multihead
//! attention op tests' Linears did, and IrRewriteConv sent them to stock while
//! this pass still ran before phase 6 (APU-157). The key therefore also holds
//! the val dtype, which is rc4's tensor dtype.
//!
//! rc4 also casts every int64 buffer that has users to int32 just before the
//! merge (CastInt64BuffersToInt32Pass, `arm_pass_manager.py:754`).
//! IrCastInt64ToInt32EffectPass skips a buffer whose data the importer already
//! narrowed, so that buffer's val stays int64. RETAB keeps such a buffer apart
//! from an equal int32 constant that rc4 merges it with. That costs one
//! constant, and every user still reads its own dtype.

use std::collections::HashMap;

use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, NodeId, NodeOp};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    DeleteConstantPlaceholderEffect, LoweringEffect, LoweringEffectBatch,
};

/// What a constant must share with another before their bytes are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConstantKey<'a> {
    special_dtype: &'a str,
    dtype: DType,
    shape: &'a [i64],
    /// The val dtype, rc4's `str(tensor.dtype)`.
    val_dtype: Option<&'a str>,
}

/// Pairs `(duplicate, canonical)` of constant placeholders with equal dtype, shape,
/// special dtype, val dtype and bytes (see the module docs); the canonical one is
/// the first in graph order. Only placeholders that share metadata have their bytes
/// compared, so large unique weights are never hashed or copied.
pub fn find_duplicate_placeholder_pairs(graph: &IrGraph) -> Vec<(NodeId, NodeId)> {
    let mut by_meta: HashMap<ConstantKey<'_>, Vec<NodeId>> = HashMap::new();
    for node in graph.live_nodes() {
        if !matches!(node.op, NodeOp::Placeholder | NodeOp::GetAttr) {
            continue;
        }
        let Some(tensor) = node.tensor.as_ref() else {
            continue;
        };
        let key = ConstantKey {
            special_dtype: node.meta.extra_str("tosa_special_dtype").unwrap_or_default(),
            dtype: tensor.data.dtype,
            shape: tensor.data.shape.as_slice(),
            val_dtype: node.meta.val_dtype(),
        };
        by_meta.entry(key).or_default().push(node.id);
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

/// See the module docs.
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
        assert_eq!(
            find_duplicate_placeholder_pairs(&graph),
            vec![(third, first)]
        );
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

        let result = IrFuseEqualPlaceholdersEffectPass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(graph.node(user).args[0].as_node(), Some(canonical));
        assert!(!graph.node(duplicate).alive);
        assert_eq!(result.effects.items.len(), 1);
        let LoweringEffect::DeleteConstantPlaceholder(effect) = &result.effects.items[0] else {
            panic!("expected delete effect");
        };
        assert_eq!(effect.placeholder_name, "b");
    }

    /// A `[3]` buffer of zeros as RETAB's importer builds it: int64 data is
    /// narrowed to int32, and the val keeps `val_dtype`.
    fn zeros_buffer(graph: &mut IrGraph, name: &str, val_dtype: &str) -> NodeId {
        let id = crate::test_ir::buffer(
            graph,
            name,
            TensorData {
                data: vec![0; 3 * 4],
                shape: vec![3],
                dtype: DType::Int32,
            },
        );
        graph.node_mut(id).meta = NodeMeta::tensor([3], val_dtype);
        id
    }

    /// The constants of rc4's `test_multihead_attention_tosa_INT[rand_2d]` that
    /// hold zeros, in graph order: each projection's per-channel weight zero
    /// point (int64, unused after the Q/DQ fold), its bias's zero point (int32,
    /// unused) and its bias (int32, all zero, read by the projection's TOSA
    /// convolution).
    struct Attention {
        graph: IrGraph,
        weight_zps: [NodeId; 2],
        bias_zps: [NodeId; 2],
        biases: [NodeId; 2],
        convs: [NodeId; 2],
    }

    fn attention() -> Attention {
        let mut graph = IrGraph::empty();
        let mut constants = Vec::new();
        for projection in 0..2 {
            let weight_zp = zeros_buffer(
                &mut graph,
                &format!("b__zero_point_{}", 2 * projection),
                "torch.int64",
            );
            let bias_zp = zeros_buffer(
                &mut graph,
                &format!("b__zero_point_{}", 2 * projection + 1),
                "torch.int32",
            );
            let bias = zeros_buffer(
                &mut graph,
                &format!("b__frozen_param{}", 2 * projection + 1),
                "torch.int32",
            );
            constants.push((weight_zp, bias_zp, bias));
        }
        let convs = [0, 1].map(|projection| {
            crate::test_ir::call(
                &mut graph,
                &format!("conv_{projection}"),
                "backend.tosa.CONV2D.default",
                vec![IrArg::Node(constants[projection].2)],
                [1, 1, 6, 3],
                "torch.int32",
            )
        });
        Attention {
            graph,
            weight_zps: [constants[0].0, constants[1].0],
            bias_zps: [constants[0].1, constants[1].1],
            biases: [constants[0].2, constants[1].2],
            convs,
        }
    }

    /// rc4 merges the int64 zero points among themselves and every int32 zero
    /// constant into the first int32 one (`b__zero_point_0_common`, unused,
    /// and `b__zero_point_1_common`, which every projection's convolution
    /// reads). RETAB keeps the same buckets, so the convolutions read an int32
    /// bias.
    #[test]
    fn keeps_int64_zero_points_off_int32_biases_as_rc4_does() {
        let Attention {
            mut graph,
            weight_zps,
            bias_zps,
            biases,
            convs,
        } = attention();
        assert_eq!(
            find_duplicate_placeholder_pairs(&graph),
            vec![
                (biases[0], bias_zps[0]),
                (weight_zps[1], weight_zps[0]),
                (bias_zps[1], bias_zps[0]),
                (biases[1], bias_zps[0]),
            ]
        );

        let result = IrFuseEqualPlaceholdersEffectPass.run(&mut graph);
        assert!(result.modified);
        for conv in convs {
            let bias = graph.node(conv).args[0].as_node().unwrap();
            assert_eq!(bias, bias_zps[0]);
            assert_eq!(graph.node(bias).meta.val_dtype(), Some("torch.int32"));
        }
        assert!(graph.node(weight_zps[0]).alive);
        assert_eq!(
            graph.node(weight_zps[0]).meta.val_dtype(),
            Some("torch.int64")
        );
    }

    /// The pass compares the val dtype on top of the data: a constant without a
    /// val still merges with another one without a val, never with one that has
    /// a val.
    #[test]
    fn tells_a_missing_val_from_any_val_dtype() {
        let mut graph = IrGraph::empty();
        let bare = add_placeholder(&mut graph, "bare", vec![0; 4], vec![1]);
        let typed = add_placeholder(&mut graph, "typed", vec![0; 4], vec![1]);
        graph.node_mut(typed).meta = NodeMeta::tensor([1], "torch.int32");
        let bare_again = add_placeholder(&mut graph, "bare_again", vec![0; 4], vec![1]);
        assert_eq!(
            find_duplicate_placeholder_pairs(&graph),
            vec![(bare_again, bare)]
        );
    }
}
