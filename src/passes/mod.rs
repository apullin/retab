pub mod annotate_dim_order;
pub mod broadcast_args;
pub mod cast_int;
pub mod cleanup;
pub mod convert_clamp;
pub mod convert_full;
pub mod decompose;
pub mod decompose_quant;
pub mod fold_qdq;
pub mod fuse_constants;
pub mod fuse_quantized_activation;
pub mod insert_rescale;
pub mod insert_table_ops;
pub mod match_arg_dtype;
pub mod memory_format;
pub mod propagate_qparams;
pub mod rewrite_conv;

use crate::graph::Graph;
use crate::store::TensorStore;

// ---------------------------------------------------------------------------
// Pass trait and runner
// ---------------------------------------------------------------------------

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool;
}

pub struct PassRunner {
    passes: Vec<Box<dyn Pass>>,
}

impl PassRunner {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run_all(&self, graph: &mut Graph, store: &mut TensorStore) {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        for pass in &self.passes {
            let before_count = graph.order.len();
            let modified = pass.run(graph, store);
            let after_count = graph.order.len();
            if debug {
                eprintln!(
                    "[PassRunner] {} — modified={}, nodes: {} -> {}",
                    pass.name(),
                    modified,
                    before_count,
                    after_count
                );
                for &nid in &graph.order {
                    if let Some(n) = graph.node(nid) {
                        if n.op == crate::graph::ir::OpType::CallFunction {
                            eprintln!("  node '{}': target={:?}, dtype={:?}", n.name, n.target, n.meta.val.dtype);
                        }
                    }
                }
            }
        }
    }
}

impl Default for PassRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// No-op placeholder passes (to be implemented later)
// ---------------------------------------------------------------------------

/// Fuse nodes that have the same target and identical args into a single node.
///
/// When decomposition passes create duplicate subgraphs (e.g., both LayerNorm
/// and Var decompose to `Sub(x, Mean(x, dims))`), this pass merges the
/// duplicates by replacing all uses of the duplicate with the first occurrence.
pub struct FuseDuplicateUsersPass;

impl Pass for FuseDuplicateUsersPass {
    fn name(&self) -> &str {
        "FuseDuplicateUsers"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        // Collect candidate groups: nodes with the same (target, args, kwargs).
        // We iterate in order and keep the first occurrence of each signature.
        loop {
            let mut fused_any = false;
            let all_ids = graph.all_node_ids();

            // Map from signature -> first NodeId with that signature
            let mut seen: std::collections::HashMap<FuseKey, crate::graph::ir::NodeId> =
                std::collections::HashMap::new();
            let mut to_fuse: Vec<(crate::graph::ir::NodeId, crate::graph::ir::NodeId)> = Vec::new();

            for &nid in &all_ids {
                let node = match graph.node(nid) {
                    Some(n) => n,
                    None => continue,
                };

                // Only fuse CallFunction nodes
                if node.op != crate::graph::ir::OpType::CallFunction {
                    continue;
                }

                // Skip nodes that shouldn't be fused (side-effectful, etc.)
                if !is_fusable(&node.target) {
                    continue;
                }

                let key = FuseKey {
                    target: format!("{:?}", node.target),
                    args: format!("{:?}", node.args),
                    kwargs: format!("{:?}", node.kwargs),
                };

                if let Some(&first_id) = seen.get(&key) {
                    to_fuse.push((nid, first_id));
                } else {
                    seen.insert(key, nid);
                }
            }

            for (dup_id, keep_id) in to_fuse {
                if debug {
                    let dup_name = graph.node(dup_id).map(|n| n.name.clone()).unwrap_or_default();
                    let keep_name = graph.node(keep_id).map(|n| n.name.clone()).unwrap_or_default();
                    eprintln!(
                        "[FuseDuplicateUsers] fusing '{}' into '{}'",
                        dup_name, keep_name,
                    );
                }
                graph.replace_all_uses(dup_id, keep_id);
                graph.erase_node(dup_id);
                fused_any = true;
                modified = true;
            }

            if !fused_any {
                break;
            }
        }

        modified
    }
}

/// Signature key for detecting duplicate nodes.
#[derive(Hash, PartialEq, Eq)]
struct FuseKey {
    target: String,
    args: String,
    kwargs: String,
}

/// Returns true for ops that are safe to fuse (pure, no side effects).
fn is_fusable(target: &crate::graph::ir::OpTarget) -> bool {
    use crate::graph::ir::OpTarget;
    matches!(
        target,
        OpTarget::Add
            | OpTarget::Sub
            | OpTarget::Mul
            | OpTarget::Reshape
            | OpTarget::Mean
            | OpTarget::Sum
            | OpTarget::Full
            | OpTarget::Rsqrt
            | OpTarget::Var
            | OpTarget::Clamp
            | OpTarget::TosaAdd
            | OpTarget::TosaSub
            | OpTarget::TosaMul
            | OpTarget::TosaReshape
            | OpTarget::TosaReduceSum
            | OpTarget::TosaClamp
    )
}

/// Placeholder: fuse placeholders that reference the same underlying buffer.
pub struct FuseEqualPlaceholdersPass;

impl Pass for FuseEqualPlaceholdersPass {
    fn name(&self) -> &str {
        "FuseEqualPlaceholders"
    }

    fn run(&self, _graph: &mut Graph, _store: &mut TensorStore) -> bool {
        // No-op for now
        false
    }
}

/// Placeholder: remove no-op nodes (identity reshapes, etc.).
pub struct RemoveNoopPass;

impl Pass for RemoveNoopPass {
    fn name(&self) -> &str {
        "RemoveNoop"
    }

    fn run(&self, _graph: &mut Graph, _store: &mut TensorStore) -> bool {
        // No-op for now
        false
    }
}

// ---------------------------------------------------------------------------
// Phase 1 pipeline for tiny_cnn
// ---------------------------------------------------------------------------

pub fn build_phase1_pipeline() -> PassRunner {
    let mut runner = PassRunner::new();
    runner.add(Box::new(fold_qdq::FoldAndAnnotateQParamsPass));
    runner.add(Box::new(convert_clamp::ConvertToClampPass));
    runner.add(Box::new(insert_table_ops::InsertTableOpsPass));
    runner.add(Box::new(insert_rescale::InsertRescalePass));
    runner.add(Box::new(decompose_quant::DecomposeQuantNodesPass));
    runner.add(Box::new(rewrite_conv::RewriteConvPass));
    runner.add(Box::new(memory_format::ToTosaMemoryFormatPass));
    runner.add(Box::new(cleanup::CleanupPass));
    runner
}

// ---------------------------------------------------------------------------
// Phase 2 pipeline — full pass ordering per Ticket 2B
// ---------------------------------------------------------------------------

pub fn build_phase2_pipeline() -> PassRunner {
    let mut runner = PassRunner::new();

    // 1. AnnotateOutputDimOrderPass
    runner.add(Box::new(annotate_dim_order::AnnotateOutputDimOrderPass));

    // 2. ConvertToClampPass (from 2A)
    runner.add(Box::new(convert_clamp::ConvertToClampPass));

    // 3. FuseQuantizedActivationPass
    runner.add(Box::new(fuse_quantized_activation::FuseQuantizedActivationPass));

    // 4. Decomposition passes (from 2A)
    runner.add(Box::new(decompose::layernorm::DecomposeLayerNormPass));
    runner.add(Box::new(decompose::var::DecomposeVarPass));
    runner.add(Box::new(decompose::meandim::DecomposeMeanDimPass));
    runner.add(Box::new(convert_full::ConvertFullLikeToFullPass));

    // 5. FoldAndAnnotateQParamsPass (existing)
    runner.add(Box::new(fold_qdq::FoldAndAnnotateQParamsPass));

    // 5a. PropagateQParamsPass — propagate qparams through decomposed ops that
    //     lack them because decomposition happened after quantization.
    runner.add(Box::new(propagate_qparams::PropagateQParamsPass));

    // 5b. InsertTableOpsPass (replace quantized Rsqrt with TABLE)
    runner.add(Box::new(insert_table_ops::InsertTableOpsPass));

    // 6. FuseDuplicateUsersPass (no-op for now)
    runner.add(Box::new(FuseDuplicateUsersPass));

    // 7. InsertRescaleInt32Pass — use existing InsertRescalePass
    runner.add(Box::new(insert_rescale::InsertRescalePass));

    // 8. DecomposeQuantNodesPass (existing)
    runner.add(Box::new(decompose_quant::DecomposeQuantNodesPass));

    // 9. BroadcastArgsPass
    runner.add(Box::new(broadcast_args::BroadcastArgsPass));

    // 10. MatchArgDtypePass
    runner.add(Box::new(match_arg_dtype::MatchArgDtypePass));

    // 11. FuseConstantArgsPass
    runner.add(Box::new(fuse_constants::FuseConstantArgsPass));

    // 12. CastToInt32Pass
    runner.add(Box::new(cast_int::CastToInt32Pass));

    // 13. RewriteConvPass (existing)
    runner.add(Box::new(rewrite_conv::RewriteConvPass));

    // 14. ToTosaMemoryFormatPass (existing)
    runner.add(Box::new(memory_format::ToTosaMemoryFormatPass));

    // 15. RemoveNoopPass (no-op for now)
    runner.add(Box::new(RemoveNoopPass));

    // 16. InsertRescalePass (existing — final rescale insertion)
    runner.add(Box::new(insert_rescale::InsertRescalePass));

    // 17. FuseEqualPlaceholdersPass (no-op for now)
    runner.add(Box::new(FuseEqualPlaceholdersPass));

    // 18. CleanupPass (existing)
    runner.add(Box::new(cleanup::CleanupPass));

    runner
}
