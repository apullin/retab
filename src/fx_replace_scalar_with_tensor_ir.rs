//! Pure Rust IrPass for scalar binary ops that must be rewritten to tensor form.
//!
//! rc4's ReplaceScalarWithTensorByProfilePass (`arm_pass_manager.py:656`,
//! `replace_scalar_with_tensor_pass.py`) gives every scalar binary op in its
//! table a tensor operand, for every dtype: `op.Scalar(x, s)` becomes
//! `op.Tensor(x, full((1,), float(s), dtype=x.dtype))`. Its INT table holds
//! add, sub, mul, div, both shifts, eq, ne, gt, ge, lt, le, the three bitwise
//! ops and remainder. RETAB rewrites two groups:
//!
//! - the scalar shifts, for every dtype;
//! - eq.Scalar and ne.Scalar on a bool tensor (APU-123).
//!
//! For the other ops the serializer gives the scalar operand its constant,
//! which TOSA accepts for every dtype but one: EQUAL takes no bool inputs. A
//! bool `x == True` that reaches the serializer as an eq.Scalar becomes an
//! EQUAL on bool, which the reference model and Vela reject. As in rc4, the
//! rewrite makes it an eq.Tensor whose second operand is a bool `full`, so
//! IrDecomposeBoolEquality (rc4's DecomposeBoolEqualityPass, `:672`) splits it
//! into LOGICAL_XOR and LOGICAL_NOT. IrMatchArgRanks rank-matches the `full`
//! first, and IrMaterializeConstants folds it into a constant, as rc4's
//! MatchArgRanks (`:664`) and ComputeConstantOpsAOT and FuseConstantArgs
//! (`:699-700`) do. The quantized flow already arrives this way, because
//! rc4's quantizer runs the same rewrite while it annotates.
//!
//! The bool rewrite fills the `full` with `float(s)`, as rc4 does, so any
//! nonzero scalar fills it with true: `x == 2` lowers as `x == True`, where
//! eager compares in int64 and gives all false. rc4 lowers it the same way.
//!
//! ne.Scalar is in the table because rc4's is. RETAB runs
//! IrDecomposeNotEqual before this pass, where rc4 runs DecomposeNotEqual
//! after it (`:671`), so a bool ne.Scalar arrives here as
//! `logical_not(eq.Scalar)`. Either order ends as rc4's
//! `logical_not(logical_not(logical_xor(x, full)))`.

use crate::fx_ir::{dtype_kwarg, IrArg, IrGraph, IrPass, NodeId, NodeMeta};
use crate::fx_target::scalar_shift_replacement_target;
use crate::runner_logging::log_line;

const BOOL: &str = "torch.bool";

fn scalar_arg(arg: &IrArg) -> Option<IrArg> {
    if let Some(value) = arg.as_int() {
        Some(IrArg::Int(value))
    } else {
        arg.as_float().map(IrArg::Float)
    }
}

/// eq.Scalar and ne.Scalar's `.Tensor` targets, from rc4's table.
fn bool_comparison_replacement_target(target: &str) -> Option<&'static str> {
    match target {
        "aten.eq.Scalar" => Some("aten.eq.Tensor"),
        "aten.ne.Scalar" => Some("aten.ne.Tensor"),
        _ => None,
    }
}

/// The `.Tensor` target and the `full`'s fill value for a scalar op this pass
/// rewrites, or `None` for any other op. `lhs` is the op's tensor operand.
fn replacement(
    graph: &IrGraph,
    target: &str,
    lhs: NodeId,
    scalar: &IrArg,
) -> Option<(&'static str, IrArg)> {
    if let Some(new_target) = scalar_shift_replacement_target(target) {
        return Some((new_target, scalar_arg(scalar)?));
    }
    let new_target = bool_comparison_replacement_target(target)?;
    if graph.nodes[lhs].meta.val_dtype() != Some(BOOL) {
        return None;
    }
    // rc4 fills the `full` with `float(args[1])`, which takes a bool too.
    Some((new_target, IrArg::Float(scalar.as_scalar_literal_f64()?)))
}

pub struct IrReplaceScalarWithTensorPass;

impl IrPass for IrReplaceScalarWithTensorPass {
    fn name(&self) -> &str {
        "IrReplaceScalarWithTensor"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if graph.nodes[id].args.len() != 2 {
                return false;
            }
            let Some(lhs_id) = graph.nodes[id].args[0].as_node() else {
                return false;
            };
            let target = graph.nodes[id].target.clone();
            let Some((new_target, scalar_value)) =
                replacement(graph, &target, lhs_id, &graph.nodes[id].args[1])
            else {
                return false;
            };
            let node_name = graph.nodes[id].name.clone();
            let output_meta = graph.nodes[id].meta.clone();
            let lhs_dtype = graph.nodes[lhs_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.int64")
                .to_string();
            let shape = [1];
            let mut cursor = graph.rewrite_cursor(id);

            let full = cursor.call_function_with_kwargs(
                format!("{}_scalar_tensor", node_name),
                "aten.full.default",
                vec![IrArg::List(vec![IrArg::Int(1)]), scalar_value],
                dtype_kwarg(lhs_dtype.clone()),
                NodeMeta::tensor(shape, &lhs_dtype),
            );

            let replacement = cursor.call_function(
                format!("{}_tensor", node_name),
                new_target,
                vec![IrArg::Node(lhs_id), IrArg::Node(full)],
                output_meta,
            );
            cursor.replace_anchor_with(replacement);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrReplaceScalarWithTensor] rewrote scalar ops to tensor form"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_decompose_bool_equality_ir::IrDecomposeBoolEqualityPass;
    use crate::fx_decompose_ne_ir::IrDecomposeNotEqualPass;
    use crate::fx_ir::{IrNode, NodeOp};
    use crate::layout::match_arg_ranks::IrMatchArgRanksPass;
    use crate::test_ir::{call, live_node, output, placeholder};

    #[test]
    fn rewrites_left_shift_scalar_to_tensor_form() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.int16");
        let shift = call(
            &mut graph,
            "lshift",
            "aten.__lshift__.Scalar",
            vec![IrArg::Node(x), IrArg::Int(3)],
            [2, 3],
            "torch.int16",
        );

        assert!(IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(!graph.nodes[shift].alive);

        let scalar_tensor = graph.by_name("lshift_scalar_tensor").unwrap();
        assert_eq!(scalar_tensor.target, "aten.full.default");
        assert_eq!(scalar_tensor.args[1].as_int(), Some(3));
        assert!(matches!(
            scalar_tensor.kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int16"
        ));

        let replacement = graph.by_name("lshift_tensor").unwrap();
        assert_eq!(replacement.target, "aten.bitwise_left_shift.Tensor");
        assert_eq!(replacement.args[0].as_node(), Some(x));
        assert_eq!(replacement.args[1].as_node(), Some(scalar_tensor.id));
    }

    /// `x` (bool, `shape`) -> `target(x, scalar)` -> output.
    fn bool_scalar_graph(target: &str, scalar: IrArg, shape: &[i64]) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", shape, BOOL);
        let op = call(
            &mut graph,
            "cmp",
            target,
            vec![IrArg::Node(x), scalar],
            shape,
            BOOL,
        );
        output(&mut graph, op);
        (graph, op)
    }

    /// The node ids `node`'s args name; a literal arg fails the test.
    fn node_args(node: &IrNode) -> Vec<NodeId> {
        node.args
            .iter()
            .map(|arg| arg.as_node().expect("a node arg"))
            .collect()
    }

    fn assert_bool_full(full: &IrNode, value: f64) {
        assert_eq!(full.target, "aten.full.default");
        assert_eq!(full.args[0].as_int_list(), Some(vec![1]));
        assert_eq!(full.args[1].as_float(), Some(value));
        assert!(matches!(
            full.kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == BOOL
        ));
        assert_eq!(full.meta.val_shape(), Some(&[1][..]));
        assert_eq!(full.meta.val_dtype(), Some(BOOL));
    }

    /// rc4's `get_replacement`: a bool `x == True` becomes
    /// `eq.Tensor(x, full((1,), 1.0, dtype=torch.bool))`, keeping the eq's
    /// value, and the output reads it.
    #[test]
    fn bool_eq_scalar_becomes_eq_tensor_with_a_bool_full() {
        let (mut graph, eq) = bool_scalar_graph("aten.eq.Scalar", IrArg::Bool(true), &[2, 3]);
        assert!(IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(!graph.nodes[eq].alive);

        let full = live_node(&graph, "cmp_scalar_tensor");
        assert_bool_full(full, 1.0);
        let replacement = live_node(&graph, "cmp_tensor");
        assert_eq!(replacement.target, "aten.eq.Tensor");
        assert_eq!(node_args(replacement), [live_node(&graph, "x").id, full.id]);
        assert_eq!(replacement.meta.val_shape(), Some(&[2, 3][..]));
        assert_eq!(replacement.meta.val_dtype(), Some(BOOL));
        assert_eq!(graph.replacement_of(eq), Some(replacement.id));
    }

    /// `float(args[1])`: False, an int and a float scalar fill the `full`
    /// with their float value, and ne.Scalar becomes ne.Tensor.
    #[test]
    fn bool_fill_values_are_rc4s_float_of_the_scalar() {
        for (target, scalar, value, new_target) in [
            ("aten.eq.Scalar", IrArg::Bool(false), 0.0, "aten.eq.Tensor"),
            ("aten.eq.Scalar", IrArg::Int(1), 1.0, "aten.eq.Tensor"),
            ("aten.eq.Scalar", IrArg::Int(2), 2.0, "aten.eq.Tensor"),
            ("aten.eq.Scalar", IrArg::Float(0.5), 0.5, "aten.eq.Tensor"),
            ("aten.ne.Scalar", IrArg::Bool(true), 1.0, "aten.ne.Tensor"),
        ] {
            let (mut graph, _) = bool_scalar_graph(target, scalar.clone(), &[4]);
            assert!(IrReplaceScalarWithTensorPass.run(&mut graph), "{scalar:?}");
            assert_bool_full(live_node(&graph, "cmp_scalar_tensor"), value);
            assert_eq!(live_node(&graph, "cmp_tensor").target, new_target);
        }
    }

    /// RETAB rewrites eq and ne only on a bool tensor: a numeric eq.Scalar
    /// keeps the serializer's constant, and the other comparisons stay on
    /// bool too.
    #[test]
    fn leaves_numeric_scalar_comparisons_and_other_bool_scalar_ops() {
        for dtype in ["torch.int8", "torch.int32", "torch.float32"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [4], dtype);
            let eq = call(
                &mut graph,
                "eq",
                "aten.eq.Scalar",
                vec![IrArg::Node(x), IrArg::Int(1)],
                [4],
                BOOL,
            );
            assert!(!IrReplaceScalarWithTensorPass.run(&mut graph), "{dtype}");
            assert!(graph.nodes[eq].alive, "{dtype}");
        }
        for target in [
            "aten.gt.Scalar",
            "aten.bitwise_and.Scalar",
            "aten.add.Scalar",
        ] {
            let (mut graph, op) = bool_scalar_graph(target, IrArg::Bool(true), &[4]);
            assert!(!IrReplaceScalarWithTensorPass.run(&mut graph), "{target}");
            assert!(graph.nodes[op].alive, "{target}");
        }
        // A scalar that is not a number stays.
        let (mut graph, eq) = bool_scalar_graph("aten.eq.Scalar", IrArg::None, &[4]);
        assert!(!IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(graph.nodes[eq].alive);
    }

    /// Every live call as `(name, target)`, in graph order.
    fn calls(graph: &IrGraph) -> Vec<(String, String)> {
        graph
            .topological_live_node_ids()
            .into_iter()
            .map(|id| graph.node(id))
            .filter(|node| node.op == NodeOp::CallFunction)
            .map(|node| (node.name.clone(), node.target.clone()))
            .collect()
    }

    fn named(rows: &[(&str, &str)]) -> Vec<(String, String)> {
        rows.iter()
            .map(|(name, target)| (name.to_string(), target.to_string()))
            .collect()
    }

    /// RETAB's order, IrDecomposeNotEqual, this pass, IrMatchArgRanks and
    /// IrDecomposeBoolEquality, ends where rc4's does: a rank-3 bool
    /// `x == True` becomes LOGICAL_XOR with a rank-matched `full`, then
    /// LOGICAL_NOT.
    #[test]
    fn bool_eq_scalar_ends_as_logical_not_of_logical_xor() {
        let (mut graph, _) = bool_scalar_graph("aten.eq.Scalar", IrArg::Bool(true), &[2, 3, 4]);
        assert!(!IrDecomposeNotEqualPass.run(&mut graph));
        assert!(IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(IrMatchArgRanksPass.run(&mut graph));
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        let rows = calls(&graph);
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert_eq!(
            rows[0],
            ("cmp_scalar_tensor".into(), "aten.full.default".into())
        );
        assert_eq!(rows[1].1, "aten.view_copy.default");
        assert_eq!(
            rows[2..],
            named(&[
                ("cmp_tensor_logical_xor", "aten.logical_xor.default"),
                ("cmp_tensor_logical_not", "aten.logical_not.default"),
            ])
        );
        let view = live_node(&graph, &rows[1].0);
        assert_eq!(view.meta.val_shape(), Some(&[1, 1, 1][..]));
        let xor = live_node(&graph, "cmp_tensor_logical_xor");
        assert_eq!(node_args(xor), [live_node(&graph, "x").id, view.id]);
    }

    /// A bool ne.Scalar: IrDecomposeNotEqual first, as RETAB orders the
    /// passes, gives rc4's two LOGICAL_NOTs after the LOGICAL_XOR.
    #[test]
    fn bool_ne_scalar_ends_as_two_logical_nots() {
        let (mut graph, _) = bool_scalar_graph("aten.ne.Scalar", IrArg::Bool(true), &[4]);
        assert!(IrDecomposeNotEqualPass.run(&mut graph));
        assert!(IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(!IrMatchArgRanksPass.run(&mut graph));
        assert!(IrDecomposeBoolEqualityPass.run(&mut graph));
        assert_eq!(
            calls(&graph),
            named(&[
                ("cmp_eq_scalar_tensor", "aten.full.default"),
                ("cmp_eq_tensor_logical_xor", "aten.logical_xor.default"),
                ("cmp_eq_tensor_logical_not", "aten.logical_not.default"),
                ("cmp_logical_not", "aten.logical_not.default"),
            ])
        );
    }
}
