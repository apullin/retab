//! Ports of the three rc4 decompositions that run before
//! phase 6 (APU-135 P2 S9, design §4.4): `DecomposeRoundPass`
//! (`backends/arm/_passes/decompose_round_pass.py`, `arm_pass_manager.py:611`),
//! `DecomposeSqrtPass` (`decompose_sqrt_pass.py`, `:617`) and
//! `DecomposeSignPass` (`decompose_sign_pass.py`, `:629`).
//!
//! RETAB lowers `aten.round`, `aten.sqrt` and `aten.sign` in the serializer
//! (or as a TABLE effect), so without these ports each stays one node. rc4 has
//! replaced them by the time its phase-6 layout passes run, and those passes
//! treat the replacement differently from the original: FuseViewCopyTransform
//! fuses a view across a unary op such as `aten.sign`, but not across the
//! `where` its decomposition ends with (`layout::pending`). The pipeline
//! therefore runs the three ports at their rc4 positions: after the Q/DQ fold,
//! because two of them read qparams, and before `IrReplaceScalarWithTensor`
//! (rc4's `:656`), because they create `.Scalar` ops.
//!
//! Each port builds its nodes as rc4's `call_operator(op, args, kwargs, meta,
//! updated=True)` does: every new node copies the decomposed node's meta,
//! qparams included ([`meta::call_operator`]), and holds the value rc4's fake
//! kernel gives it, which the port computes because RETAB does not retrace.
//! The nodes go where the decomposed node was, in rc4's creation order, and
//! its uses move to the last one. rc4 names them after their targets in its
//! retrace; the port names them `{node}_{role}` (with a suffix when taken), so
//! no lowering effect planned under an old name can land on them (design F9).
//!
//! - **Round** (`decompose_round_pass.py:37-75`): round half to even,
//!   `where(gt(d, 0.5) | (eq(d, 0.5) & odd(floor(x))), ceil(x), floor(x))`
//!   with `d = x - floor(x)`. A quantized round (both qparam maps non-empty,
//!   `ArmPass._is_quantized_meta`, `arm_pass.py:48-58`) stays: InsertTableOps
//!   makes it a TABLE.
//! - **Sqrt** (`decompose_sqrt_pass.py:34-46`): `pow.Tensor_Scalar(x, 0.5)`
//!   with no kwargs; a quantized sqrt stays, as for round.
//! - **Sign** (`decompose_sign_pass.py:55-85`): every sign, quantized or not:
//!   `where(x < 0, where(x < 0, -1, 0), where(x > 0, 1, 0))` with the zeros
//!   `x * 0.0` and the ones `zeros + 1.0`. A quantized sign rarely reaches
//!   the backend, since rc4's annotation pipeline decomposes sign first
//!   (`arm_pass_manager.py:816`); this port then copies its qparams onto every
//!   node as rc4 does.
//!
//! These passes are not TFA passes, so `allowed_to_transform` is always true
//! for them. Like every `ArmOpTargetedPass`, each skips a graph without its
//! target.

use std::collections::{HashMap, HashSet};

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::meta;

const ROUND: &str = "aten.round.default";
const SQRT: &str = "aten.sqrt.default";
const SIGN: &str = "aten.sign.default";

const FLOOR: &str = "aten.floor.default";
const CEIL: &str = "aten.ceil.default";
const SUB_TENSOR: &str = "aten.sub.Tensor";
const MUL_SCALAR: &str = "aten.mul.Scalar";
const ADD_SCALAR: &str = "aten.add.Scalar";
const EQ_SCALAR: &str = "aten.eq.Scalar";
const GT_SCALAR: &str = "aten.gt.Scalar";
const LT_SCALAR: &str = "aten.lt.Scalar";
const LOGICAL_AND: &str = "aten.logical_and.default";
const LOGICAL_OR: &str = "aten.logical_or.default";
const NEG: &str = "aten.neg.default";
const WHERE: &str = "aten.where.self";
const POW_TENSOR_SCALAR: &str = "aten.pow.Tensor_Scalar";

const BOOL: &str = "torch.bool";
/// torch's default dtype, the result of an integer tensor times a Python
/// float.
const DEFAULT_FLOAT: &str = "torch.float32";

/// rc4's `ArmPass._is_quantized_meta` (`arm_pass.py:48-58`): both qparam maps
/// non-empty.
fn is_quantized_meta(meta: &NodeMeta) -> bool {
    meta.has_nonempty_input_and_output_qparams()
}

fn is_floating(dtype: &str) -> bool {
    matches!(
        dtype,
        "torch.float16"
            | "torch.bfloat16"
            | "torch.float32"
            | "torch.float64"
            | "torch.float8_e4m3fn"
            | "torch.float8_e5m2"
    )
}

/// The dtype of `tensor op float_scalar` under torch's promotion: a floating
/// tensor keeps its dtype, an integer or bool one becomes the default float.
fn with_float_scalar(dtype: &str) -> &str {
    if is_floating(dtype) {
        dtype
    } else {
        DEFAULT_FLOAT
    }
}

/// Whether `id` is a live call of `target` that the pass may decompose.
fn is_call_of(graph: &IrGraph, id: NodeId, target: &str) -> bool {
    let node = graph.node(id);
    node.alive && node.op == NodeOp::CallFunction && node.target == target
}

/// Run `rewrite` on every live `target` call that `wanted` accepts, inside one
/// LayoutCtx, and finish it if anything changed. `rewrite` gets the node and
/// returns the node its uses move to; it runs with the insert point before the
/// node, and the node is erased afterwards.
fn decompose_each(
    graph: &mut IrGraph,
    target: &str,
    wanted: impl Fn(&NodeMeta) -> bool,
    rewrite: impl Fn(&mut LayoutCtx<'_>, NodeId, &Decomposed) -> NodeId,
) -> bool {
    // `ArmOpTargetedPass` skips a graph without a target op.
    let candidates: HashSet<NodeId> = graph
        .live_call_function_ids()
        .into_iter()
        .filter(|&id| is_call_of(graph, id, target) && wanted(&graph.node(id).meta))
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let mut ctx = LayoutCtx::new(graph);
    let mut modified = false;
    for id in ctx.order().to_vec() {
        if !candidates.contains(&id) {
            continue;
        }
        let Some(decomposed) = Decomposed::of(&ctx, id) else {
            continue;
        };
        let result = ctx.inserting_before(id, |ctx| rewrite(ctx, id, &decomposed));
        ctx.replace_all_uses_with(id, result);
        ctx.erase_node(id);
        modified = true;
    }
    if modified {
        ctx.finish();
    }
    modified
}

/// What a decomposition reads from the node it replaces.
struct Decomposed {
    name: String,
    /// `args[0]`, the decomposed op's tensor input.
    input: IrArg,
    kwargs: HashMap<String, IrArg>,
    meta: NodeMeta,
    /// The node's value: elementwise ops with a scalar keep its shape.
    facts: TensorFacts,
}

impl Decomposed {
    /// `None` for a node without a tensor input or a value, which rc4's fake
    /// kernels could not have produced.
    fn of(ctx: &LayoutCtx<'_>, id: NodeId) -> Option<Self> {
        let node = ctx.node(id);
        let input = node
            .args
            .first()
            .filter(|arg| arg.as_node().is_some())?
            .clone();
        let facts = node.meta.tensor_facts()?.clone();
        Some(Self {
            name: node.name.clone(),
            input,
            kwargs: node.kwargs.clone(),
            meta: node.meta.clone(),
            facts,
        })
    }

    /// `call_operator(op, args, kwargs, meta, updated=True)`: a node named
    /// `{name}_{role}` holding a value of the decomposed node's shape and
    /// `dtype`.
    fn call(
        &self,
        ctx: &mut LayoutCtx<'_>,
        role: &str,
        target: &str,
        args: Vec<IrArg>,
        kwargs: HashMap<String, IrArg>,
        dtype: &str,
    ) -> NodeId {
        let meta = meta::call_operator(
            &self.meta,
            TensorFacts::new(self.facts.shape.clone(), dtype),
        );
        ctx.create_node(
            NodeOp::CallFunction,
            format!("{}_{role}", self.name),
            target,
            args,
            kwargs,
            meta,
        )
    }
}

/// rc4's DecomposeRoundPass (`arm_pass_manager.py:611`); see the module docs.
pub(crate) struct IrDecomposeRoundPass;

impl IrPass for IrDecomposeRoundPass {
    fn name(&self) -> &str {
        "IrDecomposeRound"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        decompose_each(
            graph,
            ROUND,
            |meta| !is_quantized_meta(meta),
            decompose_round,
        )
    }
}

/// `DecomposeRoundPass.call_operator` (`decompose_round_pass.py:37-75`). Every
/// call passes the round's kwargs, as rc4's local `call` does.
fn decompose_round(ctx: &mut LayoutCtx<'_>, _round: NodeId, node: &Decomposed) -> NodeId {
    let x = node.input.clone();
    let dtype = node.facts.dtype.clone();
    let halved_dtype = with_float_scalar(&dtype).to_owned();
    let call =
        |ctx: &mut LayoutCtx<'_>, role: &str, target: &str, args: Vec<IrArg>, dtype: &str| {
            node.call(ctx, role, target, args, node.kwargs.clone(), dtype)
        };

    let floor_x = call(ctx, "floor", FLOOR, vec![x.clone()], &dtype);
    let dist_to_floor = call(
        ctx,
        "dist_to_floor",
        SUB_TENSOR,
        vec![x.clone(), IrArg::Node(floor_x)],
        &dtype,
    );
    // floor_x is odd iff floor_x / 2 has a .5 fractional part.
    let halved = call(
        ctx,
        "halved",
        MUL_SCALAR,
        vec![IrArg::Node(floor_x), IrArg::Float(0.5)],
        &halved_dtype,
    );
    let floor_halved = call(
        ctx,
        "floor_halved",
        FLOOR,
        vec![IrArg::Node(halved)],
        &halved_dtype,
    );
    let halved_frac = call(
        ctx,
        "halved_frac",
        SUB_TENSOR,
        vec![IrArg::Node(halved), IrArg::Node(floor_halved)],
        &halved_dtype,
    );
    let floor_is_odd = call(
        ctx,
        "floor_is_odd",
        EQ_SCALAR,
        vec![IrArg::Node(halved_frac), IrArg::Float(0.5)],
        BOOL,
    );
    let at_half = call(
        ctx,
        "at_half",
        EQ_SCALAR,
        vec![IrArg::Node(dist_to_floor), IrArg::Float(0.5)],
        BOOL,
    );
    let tie_to_even = call(
        ctx,
        "tie_to_even",
        LOGICAL_AND,
        vec![IrArg::Node(at_half), IrArg::Node(floor_is_odd)],
        BOOL,
    );
    let above_half = call(
        ctx,
        "above_half",
        GT_SCALAR,
        vec![IrArg::Node(dist_to_floor), IrArg::Float(0.5)],
        BOOL,
    );
    let take_ceil = call(
        ctx,
        "take_ceil",
        LOGICAL_OR,
        vec![IrArg::Node(above_half), IrArg::Node(tie_to_even)],
        BOOL,
    );
    let ceil_x = call(ctx, "ceil", CEIL, vec![x], &dtype);
    call(
        ctx,
        "result",
        WHERE,
        vec![
            IrArg::Node(take_ceil),
            IrArg::Node(ceil_x),
            IrArg::Node(floor_x),
        ],
        &dtype,
    )
}

/// rc4's DecomposeSqrtPass (`arm_pass_manager.py:617`); see the module docs.
pub(crate) struct IrDecomposeSqrtPass;

impl IrPass for IrDecomposeSqrtPass {
    fn name(&self) -> &str {
        "IrDecomposeSqrt"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        decompose_each(
            graph,
            SQRT,
            |meta| !is_quantized_meta(meta),
            |ctx, _, node| {
                // `pow(x, 0.5)` and `sqrt(x)` promote alike, so the value is the
                // sqrt's.
                node.call(
                    ctx,
                    "pow",
                    POW_TENSOR_SCALAR,
                    vec![node.input.clone(), IrArg::Float(0.5)],
                    HashMap::new(),
                    &node.facts.dtype.clone(),
                )
            },
        )
    }
}

/// rc4's DecomposeSignPass (`arm_pass_manager.py:629`); see the module docs.
pub(crate) struct IrDecomposeSignPass;

impl IrPass for IrDecomposeSignPass {
    fn name(&self) -> &str {
        "IrDecomposeSign"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        decompose_each(graph, SIGN, |_| true, decompose_sign)
    }
}

/// `DecomposeSignPass.call_operator` (`decompose_sign_pass.py:55-85`), with
/// no kwargs on any call.
fn decompose_sign(ctx: &mut LayoutCtx<'_>, _sign: NodeId, node: &Decomposed) -> NodeId {
    let x = node.input.clone();
    // `x * 0.0` promotes an integer x to the default float, and the rest
    // follow the zeros.
    let dtype = with_float_scalar(&node.facts.dtype).to_owned();
    let call =
        |ctx: &mut LayoutCtx<'_>, role: &str, target: &str, args: Vec<IrArg>, dtype: &str| {
            node.call(ctx, role, target, args, HashMap::new(), dtype)
        };

    let gt_mask = call(
        ctx,
        "gt_mask",
        GT_SCALAR,
        vec![x.clone(), IrArg::Float(0.0)],
        BOOL,
    );
    let lt_mask = call(
        ctx,
        "lt_mask",
        LT_SCALAR,
        vec![x.clone(), IrArg::Float(0.0)],
        BOOL,
    );
    let zeros = call(ctx, "zeros", MUL_SCALAR, vec![x, IrArg::Float(0.0)], &dtype);
    let ones = call(
        ctx,
        "ones",
        ADD_SCALAR,
        vec![IrArg::Node(zeros), IrArg::Float(1.0)],
        &dtype,
    );
    let neg_ones = call(ctx, "neg_ones", NEG, vec![IrArg::Node(ones)], &dtype);
    let negative = call(
        ctx,
        "negative",
        WHERE,
        vec![
            IrArg::Node(lt_mask),
            IrArg::Node(neg_ones),
            IrArg::Node(zeros),
        ],
        &dtype,
    );
    let positive = call(
        ctx,
        "positive",
        WHERE,
        vec![IrArg::Node(gt_mask), IrArg::Node(ones), IrArg::Node(zeros)],
        &dtype,
    );
    call(
        ctx,
        "result",
        WHERE,
        vec![
            IrArg::Node(lt_mask),
            IrArg::Node(negative),
            IrArg::Node(positive),
        ],
        &dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::QParamMap;
    use crate::test_ir::{call, live_node, output, placeholder, qparam};

    const F32: &str = "torch.float32";
    const I8: &str = "torch.int8";

    /// `(target, args as text, val dtype)` of every live call in graph order.
    fn calls(graph: &IrGraph) -> Vec<(String, Vec<String>, String)> {
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
                        IrArg::Float(value) => format!("{value}"),
                        other => format!("{other:?}"),
                    })
                    .collect();
                (
                    node.target.clone(),
                    args,
                    node.meta.val_dtype().unwrap_or("?").to_owned(),
                )
            })
            .collect()
    }

    /// `x -> op -> output`; `op` is the target call's id before any pass.
    fn unary_graph(target: &str, dtype: &str) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], dtype);
        let op = call(
            &mut graph,
            "op",
            target,
            vec![IrArg::Node(x)],
            [2, 3],
            dtype,
        );
        output(&mut graph, op);
        (graph, op)
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

    fn is_live(graph: &IrGraph, name: &str) -> bool {
        graph.live_nodes().any(|node| node.name == name)
    }

    fn quantize(graph: &mut IrGraph, id: NodeId, input: bool, output: bool) {
        let meta = &mut graph.node_mut(id).meta;
        if input {
            meta.set_single_input_qparam(qparam(0.05, -3, I8));
        }
        if output {
            meta.set_single_output_qparam(qparam(0.1, 2, I8));
        }
    }

    fn owned(rows: &[(&str, &[&str], &str)]) -> Vec<(String, Vec<String>, String)> {
        rows.iter()
            .map(|(target, args, dtype)| {
                (
                    target.to_string(),
                    args.iter().map(|arg| arg.to_string()).collect(),
                    dtype.to_string(),
                )
            })
            .collect()
    }

    /// rc4's round half to even, node for node in creation order.
    #[test]
    fn round_becomes_rc4s_half_to_even_chain() {
        let (mut graph, _) = unary_graph(ROUND, F32);
        assert!(IrDecomposeRoundPass.run(&mut graph));
        assert!(!is_live(&graph, "op"));
        assert_eq!(
            calls(&graph),
            owned(&[
                (FLOOR, &["x"], F32),
                (SUB_TENSOR, &["x", "op_floor"], F32),
                (MUL_SCALAR, &["op_floor", "0.5"], F32),
                (FLOOR, &["op_halved"], F32),
                (SUB_TENSOR, &["op_halved", "op_floor_halved"], F32),
                (EQ_SCALAR, &["op_halved_frac", "0.5"], BOOL),
                (EQ_SCALAR, &["op_dist_to_floor", "0.5"], BOOL),
                (LOGICAL_AND, &["op_at_half", "op_floor_is_odd"], BOOL),
                (GT_SCALAR, &["op_dist_to_floor", "0.5"], BOOL),
                (LOGICAL_OR, &["op_above_half", "op_tie_to_even"], BOOL),
                (CEIL, &["x"], F32),
                (WHERE, &["op_take_ceil", "op_ceil", "op_floor"], F32),
            ])
        );
        assert_eq!(output_arg(&graph), "op_result");
        assert!(graph
            .live_nodes()
            .filter(|node| node.op == NodeOp::CallFunction)
            .all(|node| node.meta.val_shape() == Some(&[2, 3][..])));
    }

    /// `_is_quantized_meta` needs both maps: a round with both stays for
    /// InsertTableOps, one with input qparams only is decomposed and every node
    /// copies them.
    #[test]
    fn round_and_sqrt_skip_only_fully_quantized_nodes() {
        for (target, pass) in [
            (ROUND, &IrDecomposeRoundPass as &dyn IrPass),
            (SQRT, &IrDecomposeSqrtPass),
        ] {
            let (mut graph, op) = unary_graph(target, F32);
            quantize(&mut graph, op, true, true);
            assert!(!pass.run(&mut graph), "{target}");
            assert!(is_live(&graph, "op"));

            let (mut graph, op) = unary_graph(target, F32);
            quantize(&mut graph, op, true, false);
            assert!(pass.run(&mut graph), "{target}");
            assert!(!is_live(&graph, "op"));
            for node in graph
                .live_nodes()
                .filter(|node| node.op == NodeOp::CallFunction)
            {
                assert_eq!(
                    node.meta.input_qparam(),
                    Some(&qparam(0.05, -3, I8)),
                    "{target}: {}",
                    node.name
                );
                assert!(node.meta.output_qparams().is_none_or(QParamMap::is_empty));
            }
        }
    }

    /// An integer round promotes its halving to the default float, as torch
    /// does for an integer tensor times 0.5.
    #[test]
    fn round_of_an_integer_tensor_halves_in_float() {
        let (mut graph, _) = unary_graph(ROUND, "torch.int32");
        assert!(IrDecomposeRoundPass.run(&mut graph));
        let dtype = |name: &str| live_node(&graph, name).meta.val_dtype().unwrap().to_owned();
        assert_eq!(dtype("op_floor"), "torch.int32");
        assert_eq!(dtype("op_halved"), F32);
        assert_eq!(dtype("op_halved_frac"), F32);
        assert_eq!(dtype("op_result"), "torch.int32");
    }

    #[test]
    fn sqrt_becomes_pow_one_half_without_kwargs() {
        let (mut graph, sqrt) = unary_graph(SQRT, F32);
        graph
            .node_mut(sqrt)
            .kwargs
            .insert("unused".into(), IrArg::Int(1));
        assert!(IrDecomposeSqrtPass.run(&mut graph));
        assert_eq!(
            calls(&graph),
            owned(&[(POW_TENSOR_SCALAR, &["x", "0.5"], F32)])
        );
        let pow = live_node(&graph, "op_pow");
        assert!(pow.kwargs.is_empty());
        assert_eq!(output_arg(&graph), "op_pow");
    }

    /// rc4's sign decomposition, node for node, with the sign's qparams on
    /// every node, a quantized sign included.
    #[test]
    fn every_sign_becomes_rc4s_where_chain_with_its_meta() {
        for quantized in [false, true] {
            let (mut graph, sign) = unary_graph(SIGN, F32);
            if quantized {
                quantize(&mut graph, sign, true, true);
            }
            assert!(IrDecomposeSignPass.run(&mut graph), "quantized={quantized}");
            assert!(!is_live(&graph, "op"));
            assert_eq!(
                calls(&graph),
                owned(&[
                    (GT_SCALAR, &["x", "0"], BOOL),
                    (LT_SCALAR, &["x", "0"], BOOL),
                    (MUL_SCALAR, &["x", "0"], F32),
                    (ADD_SCALAR, &["op_zeros", "1"], F32),
                    (NEG, &["op_ones"], F32),
                    (WHERE, &["op_lt_mask", "op_neg_ones", "op_zeros"], F32),
                    (WHERE, &["op_gt_mask", "op_ones", "op_zeros"], F32),
                    (WHERE, &["op_lt_mask", "op_negative", "op_positive"], F32),
                ])
            );
            assert_eq!(output_arg(&graph), "op_result");
            for node in graph
                .live_nodes()
                .filter(|node| node.op == NodeOp::CallFunction)
            {
                assert_eq!(node.meta.has_nonempty_input_and_output_qparams(), quantized);
                assert!(node.kwargs.is_empty());
            }
        }
    }

    #[test]
    fn integer_sign_follows_the_float_zeros() {
        let (mut graph, _) = unary_graph(SIGN, "torch.int32");
        assert!(IrDecomposeSignPass.run(&mut graph));
        assert_eq!(live_node(&graph, "op_zeros").meta.val_dtype(), Some(F32));
        assert_eq!(live_node(&graph, "op_result").meta.val_dtype(), Some(F32));
        assert_eq!(live_node(&graph, "op_gt_mask").meta.val_dtype(), Some(BOOL));
    }

    /// Graphs without the target are left alone, and taken names get a
    /// suffix rather than being reused.
    #[test]
    fn untargeted_graphs_are_unchanged_and_names_stay_unique() {
        let (mut graph, _) = unary_graph("aten.abs.default", F32);
        for pass in [
            &IrDecomposeRoundPass as &dyn IrPass,
            &IrDecomposeSqrtPass,
            &IrDecomposeSignPass,
        ] {
            assert!(!pass.run(&mut graph), "{}", pass.name());
        }

        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [4], F32);
        let taken = call(
            &mut graph,
            "op_pow",
            "aten.abs.default",
            vec![IrArg::Node(x)],
            [4],
            F32,
        );
        let sqrt = call(&mut graph, "op", SQRT, vec![IrArg::Node(taken)], [4], F32);
        output(&mut graph, sqrt);
        assert!(IrDecomposeSqrtPass.run(&mut graph));
        assert_eq!(live_node(&graph, "op_pow").target, "aten.abs.default");
        assert_eq!(live_node(&graph, "op_pow_1").target, POW_TENSOR_SCALAR);
    }

    /// Two signs in a row: each is decomposed in place, the second reading
    /// the first's result.
    #[test]
    fn chained_signs_decompose_in_graph_order() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [4], F32);
        let first = call(&mut graph, "first", SIGN, vec![IrArg::Node(x)], [4], F32);
        let second = call(
            &mut graph,
            "second",
            SIGN,
            vec![IrArg::Node(first)],
            [4],
            F32,
        );
        output(&mut graph, second);
        assert!(IrDecomposeSignPass.run(&mut graph));
        assert_eq!(
            live_node(&graph, "second_gt_mask").args[0].as_node(),
            Some(live_node(&graph, "first_result").id)
        );
        assert_eq!(output_arg(&graph), "second_result");
        let names: Vec<String> = calls(&graph)
            .into_iter()
            .map(|(target, _, _)| target)
            .collect();
        assert_eq!(names.len(), 16);
    }
}
