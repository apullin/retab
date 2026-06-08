//! Pure Rust IrPass version of `FuseConsecutiveRescalesPass`
//! (executorch master, backends/arm/_passes/fuse_consecutive_rescales_pass.py).
//!
//! `IrInsertInt32Rescale` wraps each quantized arithmetic/comparison op with
//! input rescales (INT8/INT16 -> INT32) and an output rescale (INT32 ->
//! INT8/INT16). When two such ops are chained, the output rescale of op1 (R1:
//! INT32 -> INT8/INT16) feeds directly into an input rescale of op2 (R2:
//! INT8/INT16 -> INT32), forming a redundant INT32 -> INT8 -> INT32 round-trip
//! that loses precision.
//!
//! This pass fuses only **identity** pairs (composed scale ~= 1.0, matching zero
//! points): it rewires R2's users directly to R1's INT32 input and removes the
//! round-trip. Non-identity pairs are left intact — Vela cannot process an
//! INT32 -> INT32 RESCALE (it produces all-zero NPU outputs), so the INT8/INT16
//! intermediate must be retained.

// DISABLED in fx_pipeline.rs pending the consistent ET test-advance (master's comparison
// util accommodates it, but can't be overlaid piecemeal). Kept for re-enable. See note.
#![allow(dead_code)]

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

// TOSA RESCALE argument positions (see fx_insert_int32_rescale_ir.rs):
//   args[0] = input tensor (Node)
//   args[1] = output dtype  (Dtype, e.g. "torch.int32")
//   args[2] = scale list    (List; per-tensor when len == 1)
//   args[3] = input zero point  (Int)
//   args[4] = output zero point (Int)
const ARG_INPUT: usize = 0;
const ARG_OUTPUT_DTYPE: usize = 1;
const ARG_SCALE: usize = 2;
const ARG_INPUT_ZP: usize = 3;
const ARG_OUTPUT_ZP: usize = 4;

fn rescale_output_dtype(graph: &IrGraph, id: NodeId) -> Option<&str> {
    match graph.node(id).args.get(ARG_OUTPUT_DTYPE)? {
        IrArg::Dtype(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Per-tensor scale (single-element scale list); `None` for per-channel.
fn per_tensor_scale(graph: &IrGraph, id: NodeId) -> Option<f64> {
    match graph.node(id).args.get(ARG_SCALE)? {
        IrArg::List(items) if items.len() == 1 => match &items[0] {
            IrArg::Float(v) => Some(*v),
            IrArg::Int(v) => Some(*v as f64),
            _ => None,
        },
        _ => None,
    }
}

fn int_arg(graph: &IrGraph, id: NodeId, pos: usize) -> Option<i64> {
    match graph.node(id).args.get(pos)? {
        IrArg::Int(v) => Some(*v),
        _ => None,
    }
}

fn input_node_id(graph: &IrGraph, id: NodeId) -> Option<NodeId> {
    match graph.node(id).args.get(ARG_INPUT)? {
        IrArg::Node(n) => Some(*n),
        _ => None,
    }
}

/// R1 candidate: live RESCALE(INT32 -> INT8/INT16), per-tensor scale, INT32 input.
fn is_fuseable_r1(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    if !node.alive || !node.is_rescale() {
        return false;
    }
    if !matches!(
        rescale_output_dtype(graph, id),
        Some("torch.int8") | Some("torch.int16")
    ) {
        return false;
    }
    if per_tensor_scale(graph, id).is_none() {
        return false;
    }
    match input_node_id(graph, id) {
        Some(input_id) => graph.node(input_id).meta.val_dtype() == Some("torch.int32"),
        None => false,
    }
}

/// If `r2` forms an identity pair with `r1`, rewire `r2`'s users to `r1`'s input
/// and erase `r2`. Returns whether the fusion happened.
fn try_fuse_identity_pair(graph: &mut IrGraph, r1: NodeId, r2: NodeId) -> bool {
    if r2 == r1 || !graph.node(r2).alive || !graph.node(r2).is_rescale() {
        return false;
    }
    if rescale_output_dtype(graph, r2) != Some("torch.int32") {
        return false;
    }
    let (r1_scale, r2_scale) = match (per_tensor_scale(graph, r1), per_tensor_scale(graph, r2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    let r1_in_zp = match int_arg(graph, r1, ARG_INPUT_ZP) {
        Some(v) => v,
        None => return false,
    };
    let r1_out_zp = match int_arg(graph, r1, ARG_OUTPUT_ZP) {
        Some(v) => v,
        None => return false,
    };
    let r2_in_zp = match int_arg(graph, r2, ARG_INPUT_ZP) {
        Some(v) => v,
        None => return false,
    };
    let r2_out_zp = match int_arg(graph, r2, ARG_OUTPUT_ZP) {
        Some(v) => v,
        None => return false,
    };
    // Round-trip must compose to identity: scales multiply to 1.0, R1 out zp ==
    // R2 in zp, and R1 in zp == R2 out zp.
    if r1_out_zp != r2_in_zp || r1_in_zp != r2_out_zp {
        return false;
    }
    if (r1_scale * r2_scale - 1.0).abs() >= 1e-6 {
        return false;
    }
    let r1_input = match input_node_id(graph, r1) {
        Some(id) => id,
        None => return false,
    };
    graph.replace_and_erase(r2, r1_input);
    true
}

pub struct IrFuseConsecutiveRescalesPass;

impl IrPass for IrFuseConsecutiveRescalesPass {
    fn name(&self) -> &str {
        "IrFuseConsecutiveRescales"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();
        for r1 in ids {
            if !is_fuseable_r1(graph, r1) {
                continue;
            }
            // A single R1 may feed several R2 rescales (and possibly non-rescale
            // users). Fuse each identity R1->R2 individually.
            let mut fused = false;
            for r2 in graph.live_user_ids(r1) {
                if try_fuse_identity_pair(graph, r1, r2) {
                    fused = true;
                }
            }
            if fused {
                modified = true;
                // Drop R1 once nothing live references it (mirrors the upstream
                // eliminate_dead_code); keep it if non-rescale users remain.
                if !graph.has_live_users(r1) {
                    graph.erase(r1);
                }
            }
        }

        if modified {
            log_line(format_args!(
                "[IrFuseConsecutiveRescales] fused identity rescale pairs"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{FakeTensorMeta, NodeMeta, NodeOp};

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    /// Build src(`src_dtype`) -> R1(->int8, scale=s1) -> R2(int8->int32, scale=s2)
    /// -> user. Returns (graph, src, r1, r2, user).
    fn pair_graph(src_dtype: &str, s1: f64, s2: f64) -> (IrGraph, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let src = graph.add_node(
            NodeOp::CallFunction,
            "src".to_string(),
            "aten.add.Tensor".to_string(),
            Vec::new(),
            meta(&[4], src_dtype),
        );
        let r1 = graph.add_node(
            NodeOp::CallFunction,
            "r1".to_string(),
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                IrArg::Node(src),
                IrArg::Dtype("torch.int8".to_string()),
                IrArg::List(vec![IrArg::Float(s1)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
            meta(&[4], "torch.int8"),
        );
        let r2 = graph.add_node(
            NodeOp::CallFunction,
            "r2".to_string(),
            "backend.tosa.RESCALE.default".to_string(),
            vec![
                IrArg::Node(r1),
                IrArg::Dtype("torch.int32".to_string()),
                IrArg::List(vec![IrArg::Float(s2)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
            meta(&[4], "torch.int32"),
        );
        let user = graph.add_node(
            NodeOp::CallFunction,
            "user".to_string(),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(r2)],
            meta(&[4], "torch.int32"),
        );
        (graph, src, r1, r2, user)
    }

    #[test]
    fn fuses_identity_pair() {
        // s1 * s2 == 1.0 -> identity round-trip, both rescales removed.
        let (mut graph, src, r1, r2, user) = pair_graph("torch.int32", 2.0, 0.5);
        assert!(IrFuseConsecutiveRescalesPass.run(&mut graph));
        // user now reads src directly; both rescales are dead.
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == src));
        assert!(!graph.node(r2).alive);
        assert!(!graph.node(r1).alive);
    }

    #[test]
    fn leaves_non_identity_pair() {
        // s1 * s2 == 4.0 -> not identity; Vela needs the INT8 intermediate.
        let (mut graph, _src, r1, r2, user) = pair_graph("torch.int32", 2.0, 2.0);
        assert!(!IrFuseConsecutiveRescalesPass.run(&mut graph));
        assert!(graph.node(r1).alive);
        assert!(graph.node(r2).alive);
        assert!(matches!(graph.node(user).args[0], IrArg::Node(id) if id == r2));
    }

    #[test]
    fn ignores_non_int32_r1_input() {
        // R1 input is int8 (not int32): not a fuseable R1 even if scales compose.
        let (mut graph, ..) = pair_graph("torch.int8", 2.0, 0.5);
        assert!(!IrFuseConsecutiveRescalesPass.run(&mut graph));
    }
}
