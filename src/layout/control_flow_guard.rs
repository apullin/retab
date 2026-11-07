//! The pipeline's stock fallback for control-flow outputs whose boundary
//! contract cannot find its target qparam (APU-135 P2 and S8).
//!
//! RETAB inserts a cond or while output's RESCALE at serialization
//! (`control_flow::contracts::apply_boundary_contracts`). The target qparam
//! comes from `infer_output_qparams_map_ir`: the getitem's own output qparams,
//! else the input qparams of one of its direct consumers, else a dequantize
//! consumer's. rc4 reads its targets at `arm_pass_manager.py:595`, before any
//! layout pass runs. The layout passes can then put copies without qparams
//! between a getitem and the consumer that held the target, as rc4's do:
//! IrMatchArgRanks' rank views (`layout::match_arg_ranks`),
//! IrSizeAdjustInput's slices, and the input permutes of the upsample and
//! conv rewrites (the pool rewrites' permutes keep the pool's qparams;
//! `layout::meta`). The contract then finds no target,
//! inserts no RESCALE, and the consumers read the region's output in the
//! region's quantization.
//!
//! RETAB now records each output's target before IrFoldQdq, from the
//! quantize that follows the getitem, as rc4 reads it
//! (`control_flow::IrRecordControlFlowRescaleTargetsPass`), and the contract
//! takes the recorded target, so a recorded output is never at risk
//! ([`output_target_recorded`]). What is left is an output the record found
//! no target for, which the contract can only find in the graph as it is at
//! serialization.
//!
//! IrRewriteConv requests the fallback for the convs it would rewrite
//! (`control_flow_hazard` in `layout::rewrite_conv`). This pass runs after
//! every layout pass and requests it for any other such getitem: one
//! without output qparams or a recorded target, with no direct consumer the
//! contract can read, that reaches a node with input qparams through
//! data-movement copies without input qparams. The target exists, but the
//! contract cannot see it. Where no node on the way has qparams, there is no
//! target for the copies to hide, and the graph stays with RETAB.

use std::collections::HashSet;

use crate::fx_ir::{IrGraph, IrNode, IrPass, NodeId, NodeOp};
use crate::fx_target::{
    is_layout_passthrough_target, is_slice_tensor_target, noop_target, NoopTarget,
};

/// See the module docs. It never changes the graph.
pub(crate) struct IrControlFlowOutputGuardPass;

impl IrPass for IrControlFlowOutputGuardPass {
    fn name(&self) -> &str {
        "IrControlFlowOutputGuard"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        if let Some(reason) = hidden_output_target(graph) {
            crate::stock_fallback::request(format!("IrControlFlowOutputGuard: {reason}"));
        }
        false
    }
}

/// The first cond or while output whose contract target only copies without
/// qparams lead to, described for the fallback reason.
fn hidden_output_target(graph: &IrGraph) -> Option<String> {
    for getitem in graph.live_nodes() {
        if !is_unqualified_control_flow_output(graph, getitem) {
            continue;
        }
        let users = graph.live_user_ids(getitem.id);
        if users.iter().any(|&user| contract_reads(graph.node(user))) {
            continue;
        }
        if let Some((copy, holder)) = target_behind_copies(graph, getitem.id, &users) {
            return Some(format!(
                "control-flow output {} reaches {} only through {} and other copies without \
                 qparams, so its boundary contract finds no target qparam (APU-135 S8)",
                getitem.name,
                graph.node(holder).name,
                graph.node(copy).name
            ));
        }
    }
    None
}

/// A getitem of a cond or while without output qparams or a recorded target,
/// whose target the contract looks for among its consumers.
fn is_unqualified_control_flow_output(graph: &IrGraph, node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && node.is_getitem()
        && node.meta.output_qparam().is_none()
        && node
            .first_node_arg()
            .is_some_and(|source| graph.node(source).is_control_flow())
        && !output_target_recorded(graph, node)
}

/// Whether the cond or while that getitem `node` reads carries a recorded
/// target for its output index (`IrRecordControlFlowRescaleTargetsPass`),
/// which the boundary contract takes whatever the graph shows by then.
pub(crate) fn output_target_recorded(graph: &IrGraph, node: &IrNode) -> bool {
    let (Some(source), Some(index)) = (node.first_node_arg(), node.args.get(1)) else {
        return false;
    };
    let Some(index) = index.as_int() else {
        return false;
    };
    usize::try_from(index).is_ok_and(|index| {
        graph
            .node(source)
            .meta
            .control_flow_output_targets()
            .is_some_and(|targets| targets.get(&index).is_some())
    })
}

/// Whether `infer_output_qparams_map_ir` takes a target from this consumer.
fn contract_reads(node: &IrNode) -> bool {
    node.meta.input_qparam().is_some()
        || (node.is_dequantize() && node.parse_qdq_qparam().is_some())
}

fn has_input_qparams(node: &IrNode) -> bool {
    node.meta
        .input_qparams()
        .is_some_and(|qparams| !qparams.is_empty())
}

/// A data-movement copy: its value holds its first input's elements, so they
/// keep that input's quantization (permutes, views and other rank changes,
/// slices, clones). The same set as IrRewriteConv's hazard walk.
fn copies_its_input(node: &IrNode) -> bool {
    let target = node.target.as_str();
    node.op == NodeOp::CallFunction
        && (is_layout_passthrough_target(target)
            || is_slice_tensor_target(target)
            || noop_target(target) == Some(NoopTarget::PassthroughFirstArg))
}

/// The first copy without input qparams that reads `getitem`, and the first
/// node with input qparams that a chain of such copies from it reaches.
fn target_behind_copies(
    graph: &IrGraph,
    getitem: NodeId,
    users: &[NodeId],
) -> Option<(NodeId, NodeId)> {
    let reads_as_copy = |node: &IrNode, input: NodeId| {
        copies_its_input(node) && !has_input_qparams(node) && node.first_node_arg() == Some(input)
    };
    for &first in users {
        if !reads_as_copy(graph.node(first), getitem) {
            continue;
        }
        let mut stack = vec![first];
        let mut seen = HashSet::new();
        while let Some(copy) = stack.pop() {
            if !seen.insert(copy) {
                continue;
            }
            for user in graph.live_user_ids(copy) {
                let node = graph.node(user);
                if has_input_qparams(node) {
                    return Some((first, user));
                }
                if reads_as_copy(node, copy) {
                    stack.push(user);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, IrArg};
    use crate::test_ir::{call, output, placeholder, qparam};

    const F32: &str = "torch.float32";
    const I8: &str = "torch.int8";

    /// `x -> cond -> getitem(0)`; returns the graph and the getitem.
    fn cond_output() -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let pred = placeholder(&mut graph, "pred", [1], "torch.bool");
        let x = placeholder(&mut graph, "x", [4], F32);
        let cond = call(
            &mut graph,
            "cond",
            "torch.ops.higher_order.cond",
            vec![
                IrArg::Node(pred),
                IrArg::Str("true_graph_0".into()),
                IrArg::Str("false_graph_0".into()),
                IrArg::List(vec![IrArg::Node(x)]),
            ],
            [4],
            F32,
        );
        let getitem = call(
            &mut graph,
            "getitem",
            "getitem",
            vec![IrArg::Node(cond), IrArg::Int(0)],
            [4],
            F32,
        );
        assert!(graph.node(cond).is_control_flow(), "fixture: cond target");
        assert!(graph.node(getitem).is_getitem(), "fixture: getitem target");
        (graph, getitem)
    }

    fn add_with_qparams(graph: &mut IrGraph, name: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        let add = call(
            graph,
            name,
            "aten.add.Tensor",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            [1, 4],
            F32,
        );
        let meta = &mut graph.node_mut(add).meta;
        meta.set_input_qparams_from([(0, qparam(0.1, 0, I8)), (1, qparam(0.2, 1, I8))]);
        meta.set_single_output_qparam(qparam(0.3, 2, I8));
        add
    }

    fn guard(graph: &mut IrGraph) -> Option<String> {
        crate::stock_fallback::take();
        assert!(!IrControlFlowOutputGuardPass.run(graph));
        crate::stock_fallback::take()
    }

    /// MatchArgRanks' view sits between the cond output and the add whose
    /// input qparams held the target.
    #[test]
    fn a_rank_view_hiding_the_target_falls_back() {
        let (mut graph, getitem) = cond_output();
        let y = placeholder(&mut graph, "y", [1, 4], F32);
        let view = call(
            &mut graph,
            "getitem_rank_matched",
            "aten.view_copy.default",
            vec![IrArg::Node(getitem), i64_list_arg(&[1, 4])],
            [1, 4],
            F32,
        );
        let add = add_with_qparams(&mut graph, "add", y, view);
        output(&mut graph, add);
        let reason = guard(&mut graph).expect("a fallback");
        assert!(
            reason.starts_with("IrControlFlowOutputGuard: control-flow output getitem reaches add"),
            "{reason}"
        );
    }

    /// The same view, but the cond carries the target the record took before
    /// IrFoldQdq (`IrRecordControlFlowRescaleTargetsPass`), which the contract
    /// uses: nothing to do. A record for another output doesn't count.
    #[test]
    fn a_recorded_target_stays_with_retab() {
        for (recorded_index, falls_back) in [(0, false), (1, true)] {
            let (mut graph, getitem) = cond_output();
            let cond = graph.node(getitem).first_node_arg().unwrap();
            graph
                .node_mut(cond)
                .meta
                .set_control_flow_output_targets(crate::fx_ir::QParamMap::from_pairs([(
                    recorded_index,
                    qparam(0.2, 1, I8),
                )]));
            let y = placeholder(&mut graph, "y", [1, 4], F32);
            let view = call(
                &mut graph,
                "getitem_rank_matched",
                "aten.view_copy.default",
                vec![IrArg::Node(getitem), i64_list_arg(&[1, 4])],
                [1, 4],
                F32,
            );
            let add = add_with_qparams(&mut graph, "add", y, view);
            output(&mut graph, add);
            assert_eq!(guard(&mut graph).is_some(), falls_back, "{recorded_index}");
        }
    }

    /// The contract can read a direct consumer's input qparams, or the
    /// getitem's own output qparams: nothing to do.
    #[test]
    fn a_readable_target_stays_with_retab() {
        let (mut graph, getitem) = cond_output();
        let y = placeholder(&mut graph, "y", [4], F32);
        let add = add_with_qparams(&mut graph, "add", getitem, y);
        let view = call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(getitem), i64_list_arg(&[1, 4])],
            [1, 4],
            F32,
        );
        let other = add_with_qparams(&mut graph, "other", view, y);
        output(&mut graph, add);
        output(&mut graph, other);
        assert_eq!(guard(&mut graph), None);

        let (mut graph, getitem) = cond_output();
        graph
            .node_mut(getitem)
            .meta
            .set_single_output_qparam(qparam(0.1, 0, I8));
        let view = call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(getitem), i64_list_arg(&[1, 4])],
            [1, 4],
            F32,
        );
        let y = placeholder(&mut graph, "y", [1, 4], F32);
        let add = add_with_qparams(&mut graph, "add", view, y);
        output(&mut graph, add);
        assert_eq!(guard(&mut graph), None);
    }

    /// Without qparams anywhere past the copies there is no target for them to
    /// hide; a copy that has input qparams is readable.
    #[test]
    fn no_target_or_a_qualified_copy_stays_with_retab() {
        let (mut graph, getitem) = cond_output();
        let view = call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(getitem), i64_list_arg(&[1, 4])],
            [1, 4],
            F32,
        );
        let abs = call(
            &mut graph,
            "abs",
            "aten.abs.default",
            vec![IrArg::Node(view)],
            [1, 4],
            F32,
        );
        output(&mut graph, abs);
        assert_eq!(guard(&mut graph), None);

        let (mut graph, getitem) = cond_output();
        let permute = call(
            &mut graph,
            "permute",
            "aten.permute_copy.default",
            vec![IrArg::Node(getitem), i64_list_arg(&[0])],
            [4],
            F32,
        );
        graph
            .node_mut(permute)
            .meta
            .set_single_input_qparam(qparam(0.1, 0, I8));
        let y = placeholder(&mut graph, "y", [4], F32);
        let add = add_with_qparams(&mut graph, "add", permute, y);
        output(&mut graph, add);
        assert_eq!(guard(&mut graph), None);
    }

    /// A chain of copies (a slice, then a permute) still hides the target.
    #[test]
    fn a_chain_of_copies_hiding_the_target_falls_back() {
        let (mut graph, getitem) = cond_output();
        let slice = call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(getitem),
                IrArg::Int(0),
                IrArg::Int(0),
                IrArg::Int(4),
            ],
            [4],
            F32,
        );
        let view = call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(slice), i64_list_arg(&[1, 4])],
            [1, 4],
            F32,
        );
        let y = placeholder(&mut graph, "y", [1, 4], F32);
        let add = add_with_qparams(&mut graph, "add", view, y);
        output(&mut graph, add);
        let reason = guard(&mut graph).expect("a fallback");
        assert!(
            reason.contains("reaches add only through slice"),
            "{reason}"
        );
    }
}
