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
pub mod insert_arithmetic_rescales;
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
    pub passes: Vec<Box<dyn Pass>>,
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
        use std::io::Write;
        for (i, pass) in self.passes.iter().enumerate() {
            // Always log pass start to stderr and flush immediately
            let _ = writeln!(std::io::stderr(), "[PassRunner] starting pass {}/{}: {}", i+1, self.passes.len(), pass.name());
            let _ = std::io::stderr().flush();
            let before_count = graph.order.len();
            let modified = pass.run(graph, store);
            let after_count = graph.order.len();
            let _ = writeln!(std::io::stderr(), "[PassRunner] {} — modified={}, nodes: {} -> {}", pass.name(), modified, before_count, after_count);
            let _ = std::io::stderr().flush();
            if debug {
                for &nid in &graph.order {
                    if let Some(n) = graph.node(nid) {
                        if n.op == crate::graph::ir::OpType::CallFunction {
                            eprintln!("  node '{}': target={:?}, dtype={:?}", n.name, n.target, n.meta.val.dtype);
                        }
                    }
                }
            }
        }

        // Rebuild user lists to fix any stale edges from pass mutations
        graph.rebuild_user_lists();
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
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > 5 {
                eprintln!("[WARN] FuseDuplicateUsersPass hit max iterations (5)");
                break;
            }
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
            | OpTarget::View
            | OpTarget::Mean
            | OpTarget::Sum
            | OpTarget::Full
            | OpTarget::FullLike
            | OpTarget::Rsqrt
            | OpTarget::Var
            | OpTarget::Clamp
            | OpTarget::Permute
            | OpTarget::TosaAdd
            | OpTarget::TosaSub
            | OpTarget::TosaMul
            | OpTarget::TosaReshape
            | OpTarget::TosaTranspose
            | OpTarget::TosaReduceSum
            | OpTarget::TosaClamp
            | OpTarget::TosaTable
    )
}

/// Diagnostic pass to count qparam coverage. Gated behind TOSA_ENGINE_DEBUG.
struct DiagnosticPass { label: &'static str }
impl Pass for DiagnosticPass {
    fn name(&self) -> &str { "Diagnostic" }
    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        if std::env::var("TOSA_ENGINE_DEBUG").is_err() {
            return false;
        }
        let mut has_input = 0;
        let mut has_output = 0;
        let mut has_both = 0;
        let mut call_fn_count = 0;
        let mut no_qp = Vec::new();
        for &id in &graph.all_node_ids() {
            if let Some(node) = graph.node(id) {
                if node.op == crate::graph::ir::OpType::CallFunction {
                    call_fn_count += 1;
                    let has_i = !node.meta.input_qparams.is_empty();
                    let has_o = !node.meta.output_qparams.is_empty();
                    if has_i { has_input += 1; }
                    if has_o { has_output += 1; }
                    if has_i && has_o { has_both += 1; }
                    if !has_i && !has_o {
                        no_qp.push(format!("  {} ({:?})", node.name, node.target));
                    }
                }
            }
        }
        eprintln!("[Diagnostic:{}] CallFunction nodes: {call_fn_count}, with input_qp: {has_input}, with output_qp: {has_output}, with both: {has_both}",
            self.label);
        if !no_qp.is_empty() {
            eprintln!("[Diagnostic:{}] Nodes with NO qparams ({}):", self.label, no_qp.len());
            for s in &no_qp { eprintln!("{}", s); }
        }
        for &id in &graph.all_node_ids() {
            if let Some(node) = graph.node(id) {
                if node.op == crate::graph::ir::OpType::CallFunction {
                    let in_scales: Vec<_> = node.meta.input_qparams.iter().map(|(p, q)| format!("{}:{:.6}", p, q.scale)).collect();
                    let out_scales: Vec<_> = node.meta.output_qparams.iter().map(|(p, q)| format!("{}:{:.6}", p, q.scale)).collect();
                    eprintln!("[Diagnostic:{}]   '{}' ({:?}) in=[{}] out=[{}]",
                        self.label, node.name, node.target, in_scales.join(","), out_scales.join(","));
                }
            }
        }
        false
    }
}

/// Pass that rebuilds all user lists from scratch to fix stale edges.
/// Insert this before passes that are sensitive to user-list integrity.
pub struct RebuildUserListsPass;

impl Pass for RebuildUserListsPass {
    fn name(&self) -> &str {
        "RebuildUserLists"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        graph.rebuild_user_lists();
        false // doesn't modify graph structure
    }
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

    // 5a-diag. Diagnostic pass to measure qparam coverage (gated behind TOSA_ENGINE_DEBUG)
    runner.add(Box::new(DiagnosticPass { label: "after_propagate" }));

    // 5b. InsertTableOpsPass (replace quantized Rsqrt with TABLE)
    runner.add(Box::new(insert_table_ops::InsertTableOpsPass));

    // 5c. Rebuild user lists before dedup (fixes stale edges from prior passes)
    runner.add(Box::new(RebuildUserListsPass));

    // 6. FuseDuplicateUsersPass
    runner.add(Box::new(FuseDuplicateUsersPass));

    // 6a. InsertArithmeticRescalesPass — insert rescales around ADD/SUB/MUL
    //     per TOSA requirements (input rescales for ADD/SUB, output rescales
    //     for MUL). This MUST run before InsertRescalePass so that the
    //     mismatch detection can fill in remaining inter-op rescales.
    runner.add(Box::new(
        insert_arithmetic_rescales::InsertArithmeticRescalesPass,
    ));

    // 6b. Rebuild user lists before rescale insertion
    runner.add(Box::new(RebuildUserListsPass));

    // 7. InsertRescaleInt32Pass — accumulator rescales + remaining mismatches
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

    // 15a. PropagateQParamsPass — re-propagate qparams through nodes inserted
    //      by BroadcastArgs, MatchArgDtype, MemoryFormat, etc.
    runner.add(Box::new(propagate_qparams::PropagateQParamsPass));

    // 15b. InsertArithmeticRescalesPass — second pass for newly inserted nodes
    runner.add(Box::new(
        insert_arithmetic_rescales::InsertArithmeticRescalesPass,
    ));

    // 15c. Diagnostic (gated behind TOSA_ENGINE_DEBUG)
    runner.add(Box::new(DiagnosticPass { label: "before_final_rescale" }));

    // 15d. Rebuild user lists before final rescale insertion
    runner.add(Box::new(RebuildUserListsPass));

    // 16. InsertRescalePass (existing — final rescale insertion)
    runner.add(Box::new(insert_rescale::InsertRescalePass));

    // 17. FuseEqualPlaceholdersPass (no-op for now)
    runner.add(Box::new(FuseEqualPlaceholdersPass));

    // 17a. Rebuild user lists before cleanup
    runner.add(Box::new(RebuildUserListsPass));

    // 18. CleanupPass (existing)
    runner.add(Box::new(cleanup::CleanupPass));

    runner
}
