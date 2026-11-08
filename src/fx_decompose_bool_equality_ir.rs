//! Port of rc4's `DecomposeBoolEqualityPass`
//! (`backends/arm/_passes/decompose_bool_equality_pass.py`, ExecuTorch #24083):
//! `eq.Tensor(a, b)` on bool inputs becomes `logical_not(logical_xor(a, b))`.
//!
//! TOSA's EQUAL takes numeric inputs only. RETAB serialized a bool eq as EQUAL,
//! which is invalid TOSA: the reference model rejects the graph, and Vela fails
//! its validation ("Input data type not supported for this operator"). U55
//! never delegates a bool tensor, but generic INT and U85 specs delegate a bool
//! eq, and RETAB lowers both (APU-123).
//!
//! The pipeline runs the port at rc4's slot (`arm_pass_manager.py:672`). That
//! is after MatchArgRanks (`:664`), so a lower-rank operand is rank-matched
//! before the eq splits, and before the constant folds at `:699-700`. rc4 runs
//! DecomposeNotEqual (`:671`) right before it. RETAB runs IrDecomposeNotEqual
//! earlier, before IrReplaceScalarWithTensor, so the eq that a bool ne leaves
//! still reaches this pass, and the ne becomes
//! `logical_not(logical_not(logical_xor(a, b)))`, as in rc4.
//!
//! RETAB's folds differ from rc4's here. rc4's FuseConstantArgs (`:700`)
//! folds the `logical_xor` and `logical_not` of an eq between two constants.
//! RETAB's fold planner, IrFuseConstantInputsEffectPass (which runs just
//! after this pass and inside IrMaterializeConstants), folds comparisons but
//! not logical ops. So a bool eq of two constants, which RETAB folded before
//! this port, now lowers as LOGICAL_XOR and LOGICAL_NOT on two CONSTs: valid,
//! but not folded. Only a graph lowered without the quantizer can show this;
//! the quantized flow folds such an eq before RETAB sees it.
//!
//! The port builds its nodes as rc4's `call_operator(op, args, {}, meta,
//! updated=True)` does ([`meta::call_operator`]):
//!
//! - Both nodes copy the eq's meta, qparams included, and take no kwargs.
//! - Each holds the eq's value (bool, the broadcast shape), which is what
//!   rc4's fake kernels give `logical_xor` and `logical_not`.
//! - The nodes go where the eq was, `logical_xor` first, and the eq's uses
//!   move to the `logical_not`.
//! - They are named `{eq}_logical_xor` and `{eq}_logical_not`, with a suffix
//!   when a name is taken, as the other ports name theirs.
//!
//! rc4 leaves an eq alone when any argument is not bool (`any(arg.data.dtype
//! != torch.bool for arg in args)`). The port also leaves one whose arguments
//! are not two nodes, or that has no value; rc4's fake kernels could not
//! have produced either.
//!
//! An `eq.Scalar` is not a target either. By this slot rc4's
//! ReplaceScalarWithTensorByProfilePass (`:656`) has made a bool one an
//! `eq.Tensor` whose second operand is a bool `full`, and so has RETAB's
//! IrReplaceScalarWithTensor. On the quantized flow the eq arrives that way,
//! because rc4's quantizer runs the same rewrite while it annotates.
//!
//! The pass is not a TFA pass, so `allowed_to_transform` is always true for
//! it. Like every `ArmOpTargetedPass`, it leaves a graph without a target
//! unchanged.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::meta;
use crate::runner_logging::log_line;

const EQ_TENSOR: &str = "aten.eq.Tensor";
const LOGICAL_XOR: &str = "aten.logical_xor.default";
const LOGICAL_NOT: &str = "aten.logical_not.default";
const BOOL: &str = "torch.bool";

/// rc4's DecomposeBoolEqualityPass (`arm_pass_manager.py:672`); see the
/// module docs.
pub(crate) struct IrDecomposeBoolEqualityPass;

impl IrPass for IrDecomposeBoolEqualityPass {
    fn name(&self) -> &str {
        "IrDecomposeBoolEquality"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        // `ArmOpTargetedPass` skips a graph without a target op.
        let targets: HashMap<NodeId, Vec<i64>> = graph
            .live_call_function_ids()
            .into_iter()
            .filter_map(|id| bool_equality_shape(graph, id).map(|shape| (id, shape)))
            .collect();
        if targets.is_empty() {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        for id in ctx.order().to_vec() {
            let Some(shape) = targets.get(&id) else {
                continue;
            };
            let eq = ctx.node(id);
            let name = eq.name.clone();
            let args = eq.args.clone();
            let meta = meta::call_operator(&eq.meta, TensorFacts::new(shape.clone(), BOOL));
            let result = ctx.inserting_before(id, |ctx| {
                let xor = ctx.create_node(
                    NodeOp::CallFunction,
                    format!("{name}_logical_xor"),
                    LOGICAL_XOR,
                    args,
                    HashMap::new(),
                    meta.clone(),
                );
                ctx.create_node(
                    NodeOp::CallFunction,
                    format!("{name}_logical_not"),
                    LOGICAL_NOT,
                    vec![IrArg::Node(xor)],
                    HashMap::new(),
                    meta,
                )
            });
            ctx.replace_all_uses_with(id, result);
            ctx.erase_node(id);
        }
        ctx.finish();
        log_line(format_args!(
            "[IrDecomposeBoolEquality] decomposed {} bool aten.eq.Tensor op(s) into logical_not(logical_xor)",
            targets.len()
        ));
        true
    }
}

/// The value shape of a live `eq.Tensor` whose two arguments are bool nodes,
/// or `None` for any other node.
fn bool_equality_shape(graph: &IrGraph, id: NodeId) -> Option<Vec<i64>> {
    let node = graph.node(id);
    let is_target = node.alive
        && node.op == NodeOp::CallFunction
        && node.target == EQ_TENSOR
        && node.args.len() == 2
        && node.args.iter().all(|arg| {
            arg.as_node()
                .is_some_and(|input| graph.node(input).meta.val_dtype() == Some(BOOL))
        });
    if !is_target {
        return None;
    }
    node.meta.tensor_facts().map(|facts| facts.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_decompose_ne_ir::IrDecomposeNotEqualPass;
    use crate::fx_ir::QParamMap;
    use crate::layout::match_arg_ranks::IrMatchArgRanksPass;
    use crate::test_ir::{call, live_node, output, placeholder, qparam};

    const I8: &str = "torch.int8";
    const F32: &str = "torch.float32";

    /// `(name, target, args as names or literals, val dtype, val shape)`.
    type Call = (String, String, Vec<String>, String, Vec<i64>);

    /// Every live call in graph order.
    fn calls(graph: &IrGraph) -> Vec<Call> {
        graph
            .topological_live_node_ids()
            .into_iter()
            .map(|id| graph.node(id))
            .filter(|node| node.op == NodeOp::CallFunction)
            .map(|node| {
                let args = node
                    .args
                    .iter()
                    .map(|arg| match arg {
                        IrArg::Node(id) => graph.node(*id).name.clone(),
                        other => format!("{other:?}"),
                    })
                    .collect();
                (
                    node.name.clone(),
                    node.target.clone(),
                    args,
                    node.meta.val_dtype().unwrap_or("?").to_owned(),
                    node.meta.val_shape().unwrap_or(&[]).to_vec(),
                )
            })
            .collect()
    }

    fn row(name: &str, target: &str, args: &[&str], shape: &[i64]) -> Call {
        (
            name.to_owned(),
            target.to_owned(),
            args.iter().map(|arg| arg.to_string()).collect(),
            BOOL.to_owned(),
            shape.to_vec(),
        )
    }

    /// The name of the node the output reads. LayoutCtx::finish renumbers
    /// node ids, so tests look nodes up by name after a pass.
    fn output_arg(graph: &IrGraph) -> &str {
        let out = graph
            .live_nodes()
            .find(|node| node.op == NodeOp::Output)
            .expect("an output");
        let IrArg::Node(id) = out.args[0] else {
            panic!("output reads a node")
        };
        &graph.node(id).name
    }

    /// `lhs`, `rhs` -> `target` -> output, with the inputs' dtypes and shapes.
    fn binary_graph(
        target: &str,
        (lhs_dtype, lhs_shape): (&str, &[i64]),
        (rhs_dtype, rhs_shape): (&str, &[i64]),
        out_shape: &[i64],
    ) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", lhs_shape, lhs_dtype);
        let rhs = placeholder(&mut graph, "rhs", rhs_shape, rhs_dtype);
        let op = call(
            &mut graph,
            "eq",
            target,
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            out_shape,
            BOOL,
        );
        output(&mut graph, op);
        (graph, op)
    }

    /// rc4's test_eq.py `eq_bool` operands, [3, 1] and [1, 3]: the eq becomes
    /// logical_xor of its own operands, then logical_not, each holding the
    /// broadcast bool value, and the output reads the logical_not.
    #[test]
    fn bool_eq_becomes_logical_not_of_logical_xor() {
        let (mut graph, _) = binary_graph(EQ_TENSOR, (BOOL, &[3, 1]), (BOOL, &[1, 3]), &[3, 3]);
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        assert!(!graph.live_nodes().any(|node| node.name == "eq"));
        assert_eq!(
            calls(&graph),
            vec![
                row("eq_logical_xor", LOGICAL_XOR, &["lhs", "rhs"], &[3, 3]),
                row("eq_logical_not", LOGICAL_NOT, &["eq_logical_xor"], &[3, 3]),
            ]
        );
        assert_eq!(output_arg(&graph), "eq_logical_not");
        for name in ["eq_logical_xor", "eq_logical_not"] {
            let node = live_node(&graph, name);
            assert!(node.kwargs.is_empty(), "{name}");
            assert_eq!(node.meta.val_dim_order(), Some(&[0, 1][..]), "{name}");
        }
    }

    /// rc4 skips an eq with any non-bool argument, and targets eq.Tensor
    /// only: numeric and mixed eqs, a bool eq.Scalar and a bool ne stay.
    #[test]
    fn leaves_numeric_mixed_scalar_and_ne_comparisons() {
        for (lhs, rhs) in [(F32, F32), (I8, I8), (BOOL, I8), ("torch.int32", BOOL)] {
            let (mut graph, eq) = binary_graph(EQ_TENSOR, (lhs, &[4]), (rhs, &[4]), &[4]);
            assert!(!IrDecomposeBoolEqualityPass.run(&mut graph), "{lhs}, {rhs}");
            assert!(graph.node(eq).alive, "{lhs}, {rhs}");
        }

        let (mut graph, ne) = binary_graph("aten.ne.Tensor", (BOOL, &[4]), (BOOL, &[4]), &[4]);
        assert!(!IrDecomposeBoolEqualityPass.run(&mut graph));
        assert!(graph.node(ne).alive);

        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [4], BOOL);
        let eq = call(
            &mut graph,
            "eq",
            "aten.eq.Scalar",
            vec![IrArg::Node(x), IrArg::Bool(true)],
            [4],
            BOOL,
        );
        output(&mut graph, eq);
        assert!(!IrDecomposeBoolEqualityPass.run(&mut graph));
        assert!(graph.node(eq).alive);
    }

    /// `call_operator(..., meta, updated=True)` copies the eq's meta, qparams
    /// included, onto both nodes.
    #[test]
    fn both_nodes_copy_the_eqs_meta() {
        let (mut graph, eq) = binary_graph(EQ_TENSOR, (BOOL, &[2, 3]), (BOOL, &[2, 3]), &[2, 3]);
        graph
            .node_mut(eq)
            .meta
            .set_single_input_qparam(qparam(0.05, -3, I8));
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        for name in ["eq_logical_xor", "eq_logical_not"] {
            let node = live_node(&graph, name);
            assert_eq!(
                node.meta.input_qparam(),
                Some(&qparam(0.05, -3, I8)),
                "{name}"
            );
            assert!(
                node.meta.output_qparams().is_none_or(QParamMap::is_empty),
                "{name}"
            );
        }
    }

    /// Taken names get a suffix rather than being reused, and a bool eq that
    /// reads another one's result decomposes after it, in graph order.
    #[test]
    fn names_stay_unique_and_chained_eqs_decompose_in_order() {
        let mut graph = IrGraph::empty();
        let a = placeholder(&mut graph, "a", [4], BOOL);
        let b = placeholder(&mut graph, "b", [4], BOOL);
        let taken = call(
            &mut graph,
            "first_logical_xor",
            "aten.logical_or.default",
            vec![IrArg::Node(a), IrArg::Node(b)],
            [4],
            BOOL,
        );
        let first = call(
            &mut graph,
            "first",
            EQ_TENSOR,
            vec![IrArg::Node(taken), IrArg::Node(b)],
            [4],
            BOOL,
        );
        let second = call(
            &mut graph,
            "second",
            EQ_TENSOR,
            vec![IrArg::Node(first), IrArg::Node(a)],
            [4],
            BOOL,
        );
        output(&mut graph, second);
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        assert_eq!(
            calls(&graph),
            vec![
                row(
                    "first_logical_xor",
                    "aten.logical_or.default",
                    &["a", "b"],
                    &[4]
                ),
                row(
                    "first_logical_xor_1",
                    LOGICAL_XOR,
                    &["first_logical_xor", "b"],
                    &[4]
                ),
                row(
                    "first_logical_not",
                    LOGICAL_NOT,
                    &["first_logical_xor_1"],
                    &[4]
                ),
                row(
                    "second_logical_xor",
                    LOGICAL_XOR,
                    &["first_logical_not", "a"],
                    &[4]
                ),
                row(
                    "second_logical_not",
                    LOGICAL_NOT,
                    &["second_logical_xor"],
                    &[4]
                ),
            ]
        );
        assert_eq!(output_arg(&graph), "second_logical_not");
    }

    /// rc4's order for a bool ne: DecomposeNotEqual makes it logical_not(eq),
    /// then this pass splits the eq, leaving two logical_nots.
    #[test]
    fn bool_ne_ends_as_two_logical_nots_after_decompose_not_equal() {
        let (mut graph, _) =
            binary_graph("aten.ne.Tensor", (BOOL, &[3, 1]), (BOOL, &[1, 3]), &[3, 3]);
        assert!(IrDecomposeNotEqualPass.run(&mut graph));
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        assert_eq!(
            calls(&graph),
            vec![
                row("eq_eq_logical_xor", LOGICAL_XOR, &["lhs", "rhs"], &[3, 3]),
                row(
                    "eq_eq_logical_not",
                    LOGICAL_NOT,
                    &["eq_eq_logical_xor"],
                    &[3, 3]
                ),
                row(
                    "eq_logical_not",
                    LOGICAL_NOT,
                    &["eq_eq_logical_not"],
                    &[3, 3]
                ),
            ]
        );
        assert_eq!(output_arg(&graph), "eq_logical_not");
    }

    /// MatchArgRanks (APM:664) runs first, so logical_xor reads the
    /// rank-matched operand, as rc4's does.
    #[test]
    fn a_lower_rank_operand_is_rank_matched_before_the_split() {
        let (mut graph, _) = binary_graph(EQ_TENSOR, (BOOL, &[2, 3, 4]), (BOOL, &[4]), &[2, 3, 4]);
        assert!(IrMatchArgRanksPass.run(&mut graph));
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        let xor = live_node(&graph, "eq_logical_xor");
        assert_eq!(xor.target, LOGICAL_XOR);
        assert_eq!(xor.args[0].as_node(), Some(live_node(&graph, "lhs").id));
        let view = graph.node(xor.args[1].as_node().expect("a node"));
        assert_eq!(view.target, "aten.view_copy.default");
        assert_eq!(view.meta.val_shape(), Some(&[1, 1, 4][..]));
        assert_eq!(view.meta.val_dtype(), Some(BOOL));
        assert_eq!(view.args[0].as_node(), Some(live_node(&graph, "rhs").id));
        assert_eq!(output_arg(&graph), "eq_logical_not");
    }
}
