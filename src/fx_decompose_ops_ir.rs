//! Pure Rust IrPass: atanh, stack, unbind decompositions.
//! Mirrors FxDecomposeOpsPass (hardsigmoid is intentionally skipped).

use std::f64::consts::FRAC_PI_2;

use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp,
};
use crate::runner_logging::log_line;

type DecomposeFn = fn(&mut IrGraph, NodeId) -> bool;

#[derive(Clone, Copy)]
struct TargetMatcher {
    any: &'static [&'static str],
    none: &'static [&'static str],
}

impl TargetMatcher {
    const fn any(any: &'static [&'static str]) -> Self {
        Self { any, none: &[] }
    }

    const fn any_except(any: &'static [&'static str], none: &'static [&'static str]) -> Self {
        Self { any, none }
    }

    fn matches(self, target: &str) -> bool {
        self.any.iter().any(|marker| target.contains(marker))
            && !self.none.iter().any(|marker| target.contains(marker))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecomposeMode {
    Always,
    SkipQuantizedEnvelope,
}

#[derive(Clone, Copy)]
struct DecomposeRule {
    matcher: TargetMatcher,
    mode: DecomposeMode,
    decompose: DecomposeFn,
}

#[derive(Clone, Copy)]
struct MetaFactory<'a> {
    base: &'a NodeMeta,
    dtype: &'a str,
}

impl<'a> MetaFactory<'a> {
    fn new(base: &'a NodeMeta, dtype: &'a str) -> Self {
        Self { base, dtype }
    }

    fn tensor(self, shape: &[i64]) -> NodeMeta {
        self.base.cloned_with_val(shape.to_vec(), self.dtype)
    }

    fn bool(self, shape: &[i64]) -> NodeMeta {
        self.base.cloned_with_val(shape.to_vec(), "torch.bool")
    }
}

#[derive(Clone)]
struct ApproxCtx {
    anchor_id: NodeId,
    name: String,
    shape: Vec<i64>,
    dtype: String,
    meta: NodeMeta,
}

impl ApproxCtx {
    fn from_input(graph: &IrGraph, anchor_id: NodeId, input_id: NodeId) -> Option<Self> {
        Some(Self {
            anchor_id,
            name: graph.nodes[anchor_id].name.clone(),
            shape: graph.node_shape_vec(input_id)?,
            dtype: graph.node_dtype_or_float32(input_id).to_string(),
            meta: graph.nodes[anchor_id].meta.clone(),
        })
    }

    fn from_first_input(graph: &IrGraph, anchor_id: NodeId) -> Option<(NodeId, Self)> {
        let input_id = graph.nodes[anchor_id].first_node_arg()?;
        let ctx = Self::from_input(graph, anchor_id, input_id)?;
        Some((input_id, ctx))
    }

    fn dtype(&self) -> &str {
        &self.dtype
    }

    fn with_meta(&self, meta: NodeMeta) -> Self {
        Self {
            anchor_id: self.anchor_id,
            name: self.name.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            meta,
        }
    }

    fn metas(&self) -> MetaFactory<'_> {
        MetaFactory::new(&self.meta, &self.dtype)
    }

    fn tensor_meta(&self, shape: &[i64]) -> NodeMeta {
        self.metas().tensor(shape)
    }

    fn bool_meta(&self, shape: &[i64]) -> NodeMeta {
        self.metas().bool(shape)
    }

    fn op_name(&self, suffix: &str) -> String {
        format!("{}_{}", self.name, suffix)
    }

    fn full(&self, graph: &mut IrGraph, suffix: &str, value: f64) -> NodeId {
        self.full_shape(graph, suffix, value, &self.shape)
    }

    fn scalar_shape(&self) -> Vec<i64> {
        vec![1; self.shape.len()]
    }

    fn full_scalar(&self, graph: &mut IrGraph, suffix: &str, value: f64) -> NodeId {
        let shape = self.scalar_shape();
        self.full_shape(graph, suffix, value, &shape)
    }

    fn full_shape(&self, graph: &mut IrGraph, suffix: &str, value: f64, shape: &[i64]) -> NodeId {
        graph.add_anchored_full_float(
            self.anchor_id,
            self.op_name(suffix),
            shape,
            value,
            self.dtype(),
            self.tensor_meta(shape),
        )
    }

    fn tensor(&self, graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>) -> NodeId {
        self.tensor_shape(graph, suffix, target, args, &self.shape)
    }

    fn tensor_nodes(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.tensor(
            graph,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    fn tensor_scalar(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        let shape = self.scalar_shape();
        self.tensor_shape(graph, suffix, target, args, &shape)
    }

    fn tensor_shape(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
    ) -> NodeId {
        graph.add_anchored_call_function_with_val(
            self.anchor_id,
            self.op_name(suffix),
            target,
            args,
            &self.meta,
            shape.to_vec(),
            self.dtype(),
        )
    }

    fn tensor_nodes_shape(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
        shape: &[i64],
    ) -> NodeId {
        self.tensor_shape(
            graph,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
            shape,
        )
    }

    fn bool(&self, graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>) -> NodeId {
        self.bool_shape(graph, suffix, target, args, &self.shape)
    }

    fn bool_nodes(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.bool(
            graph,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    fn bool_shape(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
    ) -> NodeId {
        graph.add_anchored_call_function(
            self.anchor_id,
            self.op_name(suffix),
            target,
            args,
            self.bool_meta(shape),
        )
    }

    fn output(&self, graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>) -> NodeId {
        graph.add_anchored_call_function(
            self.anchor_id,
            self.op_name(suffix),
            target,
            args,
            self.meta.clone(),
        )
    }

    fn replace_output(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        graph.replace_with_call_function(
            self.anchor_id,
            self.op_name(suffix),
            target,
            args,
            self.meta.clone(),
        )
    }

    fn replace_nodes(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.replace_output(
            graph,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    fn view_shape(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        source: NodeId,
        shape: &[i64],
    ) -> NodeId {
        graph.add_anchored_call_function(
            self.anchor_id,
            self.op_name(suffix),
            "aten.view_copy.default",
            vec![IrArg::Node(source), i64_list(shape)],
            self.tensor_meta(shape),
        )
    }

    fn reshape_shape(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        source: NodeId,
        shape: &[i64],
    ) -> NodeId {
        self.tensor_shape(
            graph,
            suffix,
            target,
            vec![IrArg::Node(source), i64_list(shape)],
            shape,
        )
    }

    fn add(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.tensor_nodes(graph, suffix, "aten.add.Tensor", &[lhs, rhs])
    }

    fn sub(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.tensor_nodes(graph, suffix, "aten.sub.Tensor", &[lhs, rhs])
    }

    fn mul(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.tensor_nodes(graph, suffix, "aten.mul.Tensor", &[lhs, rhs])
    }

    fn div(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.tensor_nodes(graph, suffix, "aten.div.Tensor", &[lhs, rhs])
    }

    fn unary(&self, graph: &mut IrGraph, suffix: &str, target: &str, input: NodeId) -> NodeId {
        self.tensor_nodes(graph, suffix, target, &[input])
    }

    fn neg(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.neg.default", input)
    }

    fn exp(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.exp.default", input)
    }

    fn sqrt(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.sqrt.default", input)
    }

    fn log(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.log.default", input)
    }

    fn abs(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.abs.default", input)
    }

    fn reciprocal(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.reciprocal.default", input)
    }

    fn tanh(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.tanh.default", input)
    }

    fn erf(&self, graph: &mut IrGraph, suffix: &str, input: NodeId) -> NodeId {
        self.unary(graph, suffix, "aten.erf.default", input)
    }

    fn ge(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.bool_nodes(graph, suffix, "aten.ge.Tensor", &[lhs, rhs])
    }

    fn gt(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.bool_nodes(graph, suffix, "aten.gt.Tensor", &[lhs, rhs])
    }

    fn where_self(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        condition: NodeId,
        when_true: NodeId,
        when_false: NodeId,
    ) -> NodeId {
        self.tensor_nodes(
            graph,
            suffix,
            "aten.where.self",
            &[condition, when_true, when_false],
        )
    }

    fn poly(
        &self,
        graph: &mut IrGraph,
        prefix: &str,
        coefficients: &[f64],
        variable_id: NodeId,
    ) -> Option<NodeId> {
        self.poly_shape(graph, prefix, coefficients, variable_id, &self.shape)
    }

    fn poly_shape(
        &self,
        graph: &mut IrGraph,
        prefix: &str,
        coefficients: &[f64],
        variable_id: NodeId,
        shape: &[i64],
    ) -> Option<NodeId> {
        let first = *coefficients.first()?;
        let mut result = self.full_shape(graph, &format!("{prefix}_c0"), first, shape);
        let mut current_var = variable_id;

        for (idx, coeff) in coefficients.iter().enumerate().skip(1) {
            let coeff_id = self.full_shape(graph, &format!("{prefix}_c{idx}"), *coeff, shape);
            let term = self.tensor_shape(
                graph,
                &format!("{prefix}_term{idx}"),
                "aten.mul.Tensor",
                vec![IrArg::Node(current_var), IrArg::Node(coeff_id)],
                shape,
            );
            let sum = self.tensor_shape(
                graph,
                &format!("{prefix}_sum{idx}"),
                "aten.add.Tensor",
                vec![IrArg::Node(result), IrArg::Node(term)],
                shape,
            );
            result = sum;

            if idx + 1 < coefficients.len() {
                current_var = self.tensor_shape(
                    graph,
                    &format!("{prefix}_pow{idx}"),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(current_var), IrArg::Node(current_var)],
                    shape,
                );
            }
        }

        Some(result)
    }

    fn horner_nodes(
        &self,
        graph: &mut IrGraph,
        prefix: &str,
        variable_id: NodeId,
        coefficients: &[NodeId],
    ) -> Option<NodeId> {
        self.horner_nodes_shape(graph, prefix, variable_id, coefficients, &self.shape)
    }

    fn horner_nodes_shape(
        &self,
        graph: &mut IrGraph,
        prefix: &str,
        variable_id: NodeId,
        coefficients: &[NodeId],
        shape: &[i64],
    ) -> Option<NodeId> {
        let (&last, rest) = coefficients.split_last()?;
        let mut acc = last;

        for (step, coeff_id) in rest.iter().rev().enumerate() {
            let mul = self.tensor_shape(
                graph,
                &format!("{prefix}_mul{step}"),
                "aten.mul.Tensor",
                vec![IrArg::Node(acc), IrArg::Node(variable_id)],
                shape,
            );
            acc = self.tensor_shape(
                graph,
                &format!("{prefix}_add{step}"),
                "aten.add.Tensor",
                vec![IrArg::Node(mul), IrArg::Node(*coeff_id)],
                shape,
            );
        }

        Some(acc)
    }
}

const MAX_DIM_MATCHER: TargetMatcher = TargetMatcher::any(&["aten.max.dim"]);
const MIN_DIM_MATCHER: TargetMatcher = TargetMatcher::any(&["aten.min.dim"]);

fn has_direct_non_getitem_user(graph: &IrGraph, id: NodeId) -> bool {
    graph.nodes.iter().any(|node| {
        node.alive
            && node.id != id
            && !node.is_getitem()
            && node.args.iter().any(|arg| arg.as_node() == Some(id))
    })
}

fn replace_primary_result_users(graph: &mut IrGraph, id: NodeId, result: NodeId) -> bool {
    let getitem_users = graph.live_getitem_users(id);

    if getitem_users.is_empty() {
        graph.replace_and_erase(id, result);
        return true;
    }

    if has_direct_non_getitem_user(graph, id) {
        return false;
    }

    for (getitem_id, idx) in getitem_users {
        if idx != 0 {
            return false;
        }
        graph.replace_and_erase(getitem_id, result);
    }
    graph.erase(id);
    true
}

fn pow_scalar_numeric_arg(node: &crate::fx_ir::IrNode) -> Option<f64> {
    node.arg_numeric_f64(1)
}

fn pow_scalar_int_arg(node: &crate::fx_ir::IrNode) -> Option<i64> {
    node.arg_int(1)
}

fn alpha_value(node: &crate::fx_ir::IrNode) -> Option<f64> {
    node.arg_or_kwarg_numeric_f64(2, "alpha")
}

fn elu_alpha(node: &crate::fx_ir::IrNode) -> f64 {
    node.arg_or_kwarg_numeric_f64(1, "alpha").unwrap_or(1.0)
}

fn gelu_approximate(node: &crate::fx_ir::IrNode) -> &str {
    node.kwarg_str("approximate").unwrap_or("none")
}

fn batch_norm_training(node: &crate::fx_ir::IrNode) -> bool {
    node.arg_or_kwarg_bool(5, "training").unwrap_or(false)
}

fn batch_norm_eps(node: &crate::fx_ir::IrNode) -> f64 {
    node.kwarg_numeric_f64("eps")
        .or_else(|| node.args.iter().rev().find_map(IrArg::as_numeric_f64))
        .unwrap_or(1.0e-5)
}

/// Check if input is DQ and at least one user is Q (quantized envelope → skip atanh).
fn has_quantized_envelope(graph: &IrGraph, id: NodeId) -> bool {
    // Input must be dequantize
    let input_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    if !graph.nodes[input_id].is_dequantize() {
        return false;
    }
    // At least one user must be quantize
    graph
        .nodes
        .iter()
        .filter(|n| n.alive && n.op == NodeOp::CallFunction)
        .filter(|n| {
            n.args
                .iter()
                .any(|a| matches!(a, IrArg::Node(nid) if *nid == id))
        })
        .any(|n| n.is_quantize())
}

fn decompose_asin(graph: &mut IrGraph, id: NodeId) -> bool {
    decompose_asin_or_acos(graph, id, false)
}

fn decompose_acos(graph: &mut IrGraph, id: NodeId) -> bool {
    decompose_asin_or_acos(graph, id, true)
}

fn decompose_softmax(graph: &mut IrGraph, id: NodeId) -> bool {
    decompose_softmax_like(graph, id, false)
}

fn decompose_log_softmax_wrapper(graph: &mut IrGraph, id: NodeId) -> bool {
    decompose_softmax_like(graph, id, true)
}

fn matching_rule(target: &str, rules: &[DecomposeRule]) -> Option<DecomposeRule> {
    rules
        .iter()
        .find_map(|rule| rule.matcher.matches(target).then_some(*rule))
}

fn try_decompose_rule(graph: &mut IrGraph, id: NodeId, target: &str) -> bool {
    let Some(rule) = matching_rule(target, DECOMPOSE_RULES) else {
        return false;
    };
    if rule.mode == DecomposeMode::SkipQuantizedEnvelope && has_quantized_envelope(graph, id) {
        return false;
    }
    (rule.decompose)(graph, id)
}

const DECOMPOSE_RULES: &[DecomposeRule] = &[
    DecomposeRule {
        matcher: TargetMatcher::any(&["atanh"]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_atanh,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.tan."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_tan,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.sinh."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_sinh,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.cosh."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_cosh,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.asinh."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_asinh,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.asin."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_asin,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.acos."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_acos,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.atan."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_atan,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.acosh."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_acosh,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.elu."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_elu,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.gelu."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_gelu,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.erfinv."]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_erfinv,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["pow.Tensor_Scalar"]),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_pow_scalar,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.group_norm", "aten.native_group_norm"]),
        mode: DecomposeMode::Always,
        decompose: decompose_group_norm,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&[
            "aten.batch_norm",
            "aten.native_batch_norm",
            "aten._native_batch_norm_legit_no_training",
            "aten._native_batch_norm_legit.no_stats",
        ]),
        mode: DecomposeMode::Always,
        decompose: decompose_batch_norm,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.log_softmax.", "aten._log_softmax."]),
        mode: DecomposeMode::Always,
        decompose: decompose_log_softmax_wrapper,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["aten.softmax.", "aten._safe_softmax.", "aten._softmax."]),
        mode: DecomposeMode::Always,
        decompose: decompose_softmax,
    },
    DecomposeRule {
        matcher: MAX_DIM_MATCHER,
        mode: DecomposeMode::Always,
        decompose: decompose_dim_extrema,
    },
    DecomposeRule {
        matcher: MIN_DIM_MATCHER,
        mode: DecomposeMode::Always,
        decompose: decompose_dim_extrema,
    },
    DecomposeRule {
        matcher: TargetMatcher::any_except(&["stack"], &["unstack"]),
        mode: DecomposeMode::Always,
        decompose: decompose_stack,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["unbind"]),
        mode: DecomposeMode::Always,
        decompose: decompose_unbind,
    },
    DecomposeRule {
        matcher: TargetMatcher::any(&["add.Tensor", "sub.Tensor"]),
        mode: DecomposeMode::Always,
        decompose: decompose_add_sub_alpha,
    },
];

pub struct IrDecomposeOpsPass;

impl IrPass for IrDecomposeOpsPass {
    fn name(&self) -> &str {
        "IrDecomposeOps"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let target = graph.nodes[id].target.clone();
            try_decompose_rule(graph, id, &target)
        });

        if modified {
            log_line(format_args!(
                "[IrDecomposeOps] decomposed atanh/tan/sinh/cosh/asinh/acosh/elu/gelu/batch-norm/stack/unbind/add-sub-alpha/pow ops"
            ));
        }
        modified
    }
}

fn decompose_add_sub_alpha(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some(rhs_id) = graph.nodes[id].arg_node(1) else {
        return false;
    };
    let Some(alpha) = alpha_value(&graph.nodes[id]) else {
        return false;
    };
    if (alpha - 1.0).abs() <= 1e-10 {
        return false;
    }

    let Some(rhs_ctx) = ApproxCtx::from_input(graph, id, rhs_id) else {
        return false;
    };
    let rhs_ctx = rhs_ctx.with_meta(graph.nodes[rhs_id].meta.clone());
    let target = if graph.nodes[id].target.contains("sub.Tensor") {
        "aten.sub.Tensor"
    } else {
        "aten.add.Tensor"
    };
    let alpha_full = rhs_ctx.full(graph, "alpha_full", alpha);
    let scaled_rhs = rhs_ctx.output(
        graph,
        "alpha_mul",
        "aten.mul.Tensor",
        vec![IrArg::Node(rhs_id), IrArg::Node(alpha_full)],
    );

    let mut add_args = graph.nodes[id].args.clone();
    if add_args.len() < 2 {
        return false;
    }
    add_args[1] = IrArg::Node(scaled_rhs);
    add_args.truncate(2);
    graph.replace_with_call_function(
        id,
        format!("{}_alpha_result", graph.nodes[id].name),
        target,
        add_args,
        graph.nodes[id].meta.clone(),
    );
    true
}

fn decompose_pow_scalar(graph: &mut IrGraph, id: NodeId) -> bool {
    if graph.nodes[id].meta.has_nonempty_qparam_metadata() {
        return false;
    }

    let Some((input_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    if pow_scalar_numeric_arg(&graph.nodes[id]).is_some_and(|exp| exp == 0.0) {
        let one = ctx.full(graph, "one", 1.0);
        let eq = ctx.bool(
            graph,
            "self_eq",
            "aten.eq.Tensor",
            vec![IrArg::Node(input_id), IrArg::Node(input_id)],
        );
        let ones = ctx.output(
            graph,
            "ones",
            "aten.where.self",
            vec![IrArg::Node(eq), IrArg::Node(one), IrArg::Node(one)],
        );
        graph.replace_and_erase(id, ones);
        return true;
    }

    let Some(mut exp) = pow_scalar_int_arg(&graph.nodes[id]) else {
        return false;
    };

    let mut base_id = input_id;
    if exp < 0 {
        let reciprocal = ctx.output(
            graph,
            "reciprocal",
            "aten.reciprocal.default",
            vec![IrArg::Node(input_id)],
        );
        base_id = reciprocal;
        exp = -exp;
    }

    let mut result_id = base_id;
    for idx in 1..exp {
        let mul = ctx.output(
            graph,
            &format!("mul{idx}"),
            "aten.mul.Tensor",
            vec![IrArg::Node(result_id), IrArg::Node(base_id)],
        );
        result_id = mul;
    }

    graph.replace_and_erase(id, result_id);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_max_dim_rewrites_to_amax_and_squeeze() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );
        let max = graph.add_node(
            NodeOp::CallFunction,
            "max".to_string(),
            "executorch.exir.dialects.edge._ops.aten.max.dim".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            NodeMeta::default(),
        );
        let getitem = graph.add_node(
            NodeOp::CallFunction,
            "getitem".to_string(),
            "getitem".to_string(),
            vec![IrArg::Node(max), IrArg::Int(0)],
            NodeMeta::tensor(&[2, 4], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[max].alive);
        assert!(!graph.nodes[getitem].alive);

        let reduce = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "max_amax")
            .expect("amax node");
        assert_eq!(reduce.target, "aten.amax.default");
        assert_eq!(reduce.args[0].as_node(), Some(x));
        assert_eq!(reduce.args[1].as_int(), Some(1));
        assert_eq!(reduce.args[2].as_bool(), Some(true));
        assert_eq!(reduce.meta.val_shape().unwrap(), &[2, 1, 4]);

        let squeeze = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "max_result")
            .expect("squeeze node");
        assert_eq!(squeeze.target, "aten.squeeze_copy.dims");
        assert_eq!(squeeze.args[0].as_node(), Some(reduce.id));
        assert_eq!(squeeze.meta.val_shape().unwrap(), &[2, 4]);
    }

    #[test]
    fn skips_max_dim_when_index_output_is_live() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );
        let max = graph.add_node(
            NodeOp::CallFunction,
            "max".to_string(),
            "executorch.exir.dialects.edge._ops.aten.max.dim".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            NodeMeta::default(),
        );
        let _values = graph.add_node(
            NodeOp::CallFunction,
            "getitem".to_string(),
            "getitem".to_string(),
            vec![IrArg::Node(max), IrArg::Int(0)],
            NodeMeta::tensor(&[2, 4], "torch.float32"),
        );
        let indices = graph.add_node(
            NodeOp::CallFunction,
            "getitem_1".to_string(),
            "getitem".to_string(),
            vec![IrArg::Node(max), IrArg::Int(1)],
            NodeMeta::tensor(&[2, 4], "torch.int64"),
        );
        let _use_indices = graph.add_node(
            NodeOp::CallFunction,
            "use_indices".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(indices), IrArg::Node(indices)],
            NodeMeta::tensor(&[2, 4], "torch.int64"),
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[max].alive);
        assert!(graph.nodes[indices].alive);
    }

    #[test]
    fn decompose_atan_rewrites_to_piecewise_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let atan = graph.add_node(
            NodeOp::CallFunction,
            "atan".to_string(),
            "aten.atan.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[atan].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "atan_result"
            && node.target == "aten.where.self"));
    }

    #[test]
    fn decompose_acos_rewrites_to_piecewise_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let acos = graph.add_node(
            NodeOp::CallFunction,
            "acos".to_string(),
            "aten.acos.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[acos].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "acos_result"
            && node.target == "aten.where.self"));
    }

    #[test]
    fn decompose_softmax_rewrites_to_reduce_exp_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );
        let softmax = graph.add_node(
            NodeOp::CallFunction,
            "softmax".to_string(),
            "aten._softmax.default".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[softmax].alive);
        // softmax decomposes to amax -> sub -> exp -> sum -> reciprocal -> mul (-> clone)
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.alive && n.target == "aten.amax.default"));
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.alive && n.target == "aten.exp.default"));
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.alive && n.target == "aten.reciprocal.default"));
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "softmax_softmax"
            && node.target == "aten.mul.Tensor"));
    }

    #[test]
    fn decompose_log_softmax_rewrites_to_log_of_softmax_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );
        let log_softmax = graph.add_node(
            NodeOp::CallFunction,
            "log_softmax".to_string(),
            "aten._log_softmax.default".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            NodeMeta::tensor(&[2, 3, 4], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[log_softmax].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "log_softmax_result"
            && node.target == "aten.log.default"));
    }

    #[test]
    fn decompose_add_with_alpha_scales_rhs_before_add() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 4, 5], "torch.float32"),
        );
        let y = graph.add_node(
            NodeOp::Placeholder,
            "y".to_string(),
            "y".to_string(),
            vec![],
            NodeMeta::tensor(&[4, 1], "torch.float32"),
        );
        let add = graph.add_node(
            NodeOp::CallFunction,
            "add".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(x), IrArg::Node(y), IrArg::Float(1.5)],
            NodeMeta::tensor(&[1, 4, 5], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[add].alive);

        let scaled_rhs = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "add_alpha_mul")
            .expect("scaled rhs");
        assert_eq!(scaled_rhs.target, "aten.mul.Tensor");

        let final_add = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "add_alpha_result")
            .expect("final add");
        assert_eq!(final_add.target, "aten.add.Tensor");
        assert_eq!(final_add.args.len(), 2);
        assert_eq!(final_add.args[0].as_node(), Some(x));
        assert_eq!(final_add.args[1].as_node(), Some(scaled_rhs.id));
    }

    #[test]
    fn decompose_tan_rewrites_to_sin_cos_div() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let tan = graph.add_node(
            NodeOp::CallFunction,
            "tan".to_string(),
            "aten.tan.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[tan].alive);

        let sin = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "tan_sin")
            .expect("sin node");
        assert_eq!(sin.target, "aten.sin.default");
        assert_eq!(sin.args[0].as_node(), Some(x));

        let cos = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "tan_cos")
            .expect("cos node");
        assert_eq!(cos.target, "aten.cos.default");
        assert_eq!(cos.args[0].as_node(), Some(x));

        let div = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "tan_result")
            .expect("div node");
        assert_eq!(div.target, "aten.div.Tensor");
        assert_eq!(div.args[0].as_node(), Some(sin.id));
        assert_eq!(div.args[1].as_node(), Some(cos.id));
    }

    #[test]
    fn decompose_pow_scalar_zero_rewrites_to_full_like() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 2, 3, 7], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(x), IrArg::Float(0.0)],
            NodeMeta::tensor(&[1, 2, 3, 7], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[pow].alive);

        let eq = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "pow_self_eq")
            .expect("eq node");
        assert_eq!(eq.target, "aten.eq.Tensor");
        assert_eq!(eq.args[0].as_node(), Some(x));
        assert_eq!(eq.args[1].as_node(), Some(x));

        let one = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "pow_one")
            .expect("one const");
        assert_eq!(one.target, "aten.full.default");

        let ones = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "pow_ones")
            .expect("where node");
        assert_eq!(ones.target, "aten.where.self");
        assert_eq!(ones.args[0].as_node(), Some(eq.id));
        assert_eq!(ones.args[1].as_node(), Some(one.id));
        assert_eq!(ones.args[2].as_node(), Some(one.id));
    }

    #[test]
    fn decompose_pow_scalar_positive_int_rewrites_to_mul_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(x), IrArg::Int(3)],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[pow].alive);

        let mul1 = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "pow_mul1")
            .expect("first mul");
        assert_eq!(mul1.target, "aten.mul.Tensor");
        assert_eq!(mul1.args[0].as_node(), Some(x));
        assert_eq!(mul1.args[1].as_node(), Some(x));

        let mul2 = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "pow_mul2")
            .expect("second mul");
        assert_eq!(mul2.target, "aten.mul.Tensor");
        assert_eq!(mul2.args[0].as_node(), Some(mul1.id));
        assert_eq!(mul2.args[1].as_node(), Some(x));
    }

    #[test]
    fn skips_tan_inside_quantized_envelope() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "x_dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let tan = graph.add_node(
            NodeOp::CallFunction,
            "tan".to_string(),
            "aten.tan.default".to_string(),
            vec![IrArg::Node(dq)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let q = graph.add_node(
            NodeOp::CallFunction,
            "tan_q".to_string(),
            "quantized_decomposed.quantize_per_tensor.default".to_string(),
            vec![IrArg::Node(tan)],
            NodeMeta::tensor(&[2, 3], "torch.int8"),
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[tan].alive);
        assert_eq!(graph.nodes[tan].target, "aten.tan.default");
        assert!(graph.nodes[q].alive);
    }

    #[test]
    fn skips_pow_inside_quantized_envelope() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "x_dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(dq), IrArg::Int(3)],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.float32"),
        );
        let q = graph.add_node(
            NodeOp::CallFunction,
            "pow_q".to_string(),
            "quantized_decomposed.quantize_per_tensor.default".to_string(),
            vec![IrArg::Node(pow)],
            NodeMeta::tensor(&[1, 2, 3, 6], "torch.int8"),
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[pow].alive);
        assert_eq!(graph.nodes[pow].target, "aten.pow.Tensor_Scalar");
        assert!(graph.nodes[q].alive);
    }

    #[test]
    fn decompose_sinh_rewrites_via_exp_sub_mul() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let sinh = graph.add_node(
            NodeOp::CallFunction,
            "sinh".to_string(),
            "aten.sinh.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[sinh].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "sinh_result"
            && node.target == "aten.mul.Tensor"));
    }

    #[test]
    fn decompose_cosh_rewrites_via_exp_add_mul() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let cosh = graph.add_node(
            NodeOp::CallFunction,
            "cosh".to_string(),
            "aten.cosh.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[cosh].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "cosh_result"
            && node.target == "aten.mul.Tensor"));
    }

    #[test]
    fn decompose_asinh_rewrites_via_log_and_sqrt() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let asinh = graph.add_node(
            NodeOp::CallFunction,
            "asinh".to_string(),
            "aten.asinh.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[asinh].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "asinh_result"
            && node.target == "aten.log.default"));
    }

    #[test]
    fn decompose_acosh_rewrites_via_log_and_sqrt() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let acosh = graph.add_node(
            NodeOp::CallFunction,
            "acosh".to_string(),
            "aten.acosh.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[acosh].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "acosh_result"
            && node.target == "aten.log.default"));
    }

    #[test]
    fn decompose_elu_rewrites_via_expm1_ge_where() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let elu = graph.add_node(
            NodeOp::CallFunction,
            "elu".to_string(),
            "aten.elu.default".to_string(),
            vec![IrArg::Node(x), IrArg::Float(2.0)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[elu].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "elu_result"
            && node.target == "aten.where.self"));
    }

    #[test]
    fn decompose_gelu_none_rewrites_via_erf() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let gelu = graph.add_node(
            NodeOp::CallFunction,
            "gelu".to_string(),
            "aten.gelu.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[gelu].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "gelu_result"
            && node.target == "aten.mul.Tensor"));
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "gelu_erf"
            && node.target == "aten.erf.default"));
    }

    #[test]
    fn decompose_gelu_tanh_rewrites_via_tanh() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        let gelu = graph.add_node(
            NodeOp::CallFunction,
            "gelu".to_string(),
            "aten.gelu.default".to_string(),
            vec![IrArg::Node(x)],
            NodeMeta::tensor(&[2, 3], "torch.float32"),
        );
        graph.set_kwarg(gelu, "approximate", IrArg::Str("tanh".to_string()));

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[gelu].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "gelu_tanh"
            && node.target == "aten.tanh.default"));
    }

    #[test]
    fn decompose_batch_norm_rewrites_to_elementwise_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 4, 5, 6], "torch.float32"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            vec![],
            NodeMeta::tensor(&[4], "torch.float32"),
        );
        let bias = graph.add_node(
            NodeOp::Placeholder,
            "bias".to_string(),
            "bias".to_string(),
            vec![],
            NodeMeta::tensor(&[4], "torch.float32"),
        );
        let mean = graph.add_node(
            NodeOp::Placeholder,
            "mean".to_string(),
            "mean".to_string(),
            vec![],
            NodeMeta::tensor(&[4], "torch.float32"),
        );
        let var = graph.add_node(
            NodeOp::Placeholder,
            "var".to_string(),
            "var".to_string(),
            vec![],
            NodeMeta::tensor(&[4], "torch.float32"),
        );
        let bn = graph.add_node(
            NodeOp::CallFunction,
            "bn".to_string(),
            "aten.batch_norm.default".to_string(),
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::Node(bias),
                IrArg::Node(mean),
                IrArg::Node(var),
                IrArg::Bool(false),
                IrArg::Float(0.1),
                IrArg::Float(1.0e-5),
                IrArg::Bool(false),
            ],
            NodeMeta::tensor(&[1, 4, 5, 6], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[bn].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "bn_result"
            && node.target == "aten.add.Tensor"));
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "bn_inv_sqrt"
            && node.target == "aten.rsqrt.default"));
        let result = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "bn_result")
            .expect("result");
        assert_eq!(result.meta.val_shape().unwrap(), &[1, 4, 5, 6]);
        let mean_view = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "bn_mean_view")
            .expect("mean view");
        assert_eq!(mean_view.meta.val_shape().unwrap(), &[1, 4, 1, 1]);
    }

    #[test]
    fn decompose_group_norm_rewrites_to_reduction_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor(&[1, 6, 8], "torch.float32"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            vec![],
            NodeMeta::tensor(&[6], "torch.float32"),
        );
        let bias = graph.add_node(
            NodeOp::Placeholder,
            "bias".to_string(),
            "bias".to_string(),
            vec![],
            NodeMeta::tensor(&[6], "torch.float32"),
        );
        let gn = graph.add_node(
            NodeOp::CallFunction,
            "gn".to_string(),
            "aten.group_norm.default".to_string(),
            vec![
                IrArg::Node(x),
                IrArg::Int(2),
                IrArg::Node(weight),
                IrArg::Node(bias),
                IrArg::Float(1.0e-5),
                IrArg::Bool(true),
            ],
            NodeMeta::tensor(&[1, 6, 8], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[gn].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "gn_grouped"
            && node.target == "aten.reshape.default"));
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "gn_result"
            && node.target == "aten.reshape.default"));
        let grouped = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "gn_grouped")
            .expect("grouped");
        assert_eq!(grouped.meta.val_shape().unwrap(), &[1, 2, 3, 8]);
    }
}

fn decompose_dim_extrema(graph: &mut IrGraph, id: NodeId) -> bool {
    let node_target = graph.nodes[id].target.clone();
    let node_name = graph.nodes[id].name.clone();
    let reduce_target = if MAX_DIM_MATCHER.matches(&node_target) {
        "aten.amax.default"
    } else if MIN_DIM_MATCHER.matches(&node_target) {
        "aten.amin.default"
    } else {
        return false;
    };

    let input_id = match graph.nodes[id].first_node_arg() {
        Some(input_id) => input_id,
        None => return false,
    };
    let dim = match graph.nodes[id].arg_int(1) {
        Some(dim) => dim,
        None => return false,
    };
    let input_shape = match graph.node_shape_vec(input_id) {
        Some(shape) => shape,
        None => return false,
    };
    let rank = input_shape.len() as i64;
    let dim = if dim < 0 { dim + rank } else { dim };
    if dim < 0 || dim >= rank {
        return false;
    }
    let dim = dim as usize;

    let getitem_users = graph.live_getitem_users(id);

    if getitem_users.is_empty() {
        return false;
    }

    if graph
        .nodes
        .iter()
        .any(|n| n.alive && n.id != id && !n.is_getitem() && graph.users(id).contains(&n.id))
    {
        return false;
    }

    let value_getitem = getitem_users
        .iter()
        .find_map(|(user_id, idx)| (*idx == 0).then_some(*user_id));
    let value_getitem = match value_getitem {
        Some(user_id) => user_id,
        None => return false,
    };

    for (user_id, idx) in &getitem_users {
        if *idx != 0 && graph.has_live_users(*user_id) {
            return false;
        }
    }

    let input_meta = graph.nodes[input_id].meta.clone();
    let value_meta = graph.nodes[value_getitem].meta.clone();
    let dtype = value_meta
        .val_dtype()
        .unwrap_or_else(|| input_meta.val_dtype_or_float32())
        .to_string();

    let mut keepdim_shape = input_shape;
    keepdim_shape[dim] = 1;

    let mut reduce_meta = value_meta.cloned_with_val(keepdim_shape, &dtype);
    reduce_meta.val_dim_order = input_meta.val_dim_order.clone();
    reduce_meta.tosa_dim_order = input_meta.tosa_dim_order.clone();
    reduce_meta.original_dim_orders = input_meta.original_dim_orders.clone();

    let reduce_id = graph.add_anchored_call_function(
        id,
        format!(
            "{}_{}",
            node_name,
            if MAX_DIM_MATCHER.matches(&node_target) {
                "amax"
            } else {
                "amin"
            }
        ),
        reduce_target.to_string(),
        vec![
            IrArg::Node(input_id),
            IrArg::Int(dim as i64),
            IrArg::Bool(true),
        ],
        reduce_meta,
    );

    let squeeze_id = graph.add_anchored_call_function(
        reduce_id,
        format!("{}_result", node_name),
        "aten.squeeze_copy.dims".to_string(),
        vec![IrArg::Node(reduce_id), i64_list(&[dim as i64])],
        value_meta,
    );

    graph.replace_and_erase(value_getitem, squeeze_id);
    for (user_id, idx) in getitem_users {
        if idx != 0 {
            graph.erase(user_id);
        }
    }
    graph.erase(id);
    true
}

fn decompose_softmax_like(graph: &mut IrGraph, id: NodeId, log_output: bool) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };
    let shape = ctx.shape.clone();
    let dim = graph.nodes[id].arg_or_kwarg_int(1, "dim").unwrap_or(-1);
    let rank = shape.len() as i64;
    let dim = if dim < 0 { dim + rank } else { dim };
    if dim < 0 || dim >= rank {
        return false;
    }
    let dim = dim as usize;

    let mut reduced_shape = shape.clone();
    reduced_shape[dim] = 1;
    let sum_shape = reduced_shape.clone();

    let reduce = ctx.tensor_shape(
        graph,
        "amax",
        "aten.amax.default",
        vec![IrArg::Node(x_id), IrArg::Int(dim as i64), IrArg::Bool(true)],
        &reduced_shape,
    );

    let shifted = ctx.sub(graph, "shifted", x_id, reduce);

    let exp = ctx.exp(graph, "exp", shifted);

    let sum = ctx.tensor_shape(
        graph,
        "sum",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(exp), i64_list(&[dim as i64]), IrArg::Bool(true)],
        &sum_shape,
    );

    let recip = ctx.tensor_shape(
        graph,
        "recip",
        "aten.reciprocal.default",
        vec![IrArg::Node(sum)],
        &sum_shape,
    );

    let softmax = ctx.mul(graph, "softmax", exp, recip);

    if log_output {
        ctx.replace_nodes(graph, "result", "aten.log.default", &[softmax]);
    } else {
        ctx.replace_nodes(graph, "result", "aten.clone.default", &[softmax]);
    }
    true
}

/// atanh(x) = 0.5 * log((1+x) / (1-x))
fn decompose_atanh(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let one = ctx.full(graph, "one", 1.0);
    let half = ctx.full(graph, "half", 0.5);
    let one_plus = ctx.add(graph, "1px", one, x_id);
    let one_minus = ctx.sub(graph, "1mx", one, x_id);
    let ratio = ctx.div(graph, "ratio", one_plus, one_minus);
    let log_r = ctx.log(graph, "log", ratio);
    ctx.replace_nodes(graph, "result", "aten.mul.Tensor", &[half, log_r]);
    true
}

/// tan(x) = sin(x) / cos(x)
fn decompose_tan(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let sin = ctx.unary(graph, "sin", "aten.sin.default", x_id);
    let cos = ctx.unary(graph, "cos", "aten.cos.default", x_id);
    ctx.replace_nodes(graph, "result", "aten.div.Tensor", &[sin, cos]);
    true
}

/// atan(x) via the same Padé approximation used in the Python pass.
fn decompose_atan(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let one = ctx.full(graph, "one", 1.0);
    let zero = ctx.full(graph, "zero", 0.0);
    let half_pi = ctx.full(graph, "half_pi", FRAC_PI_2);

    let abs_x = ctx.abs(graph, "abs", x_id);
    let mask_hi = ctx.gt(graph, "mask_hi", abs_x, one);
    let inv_x = ctx.reciprocal(graph, "inv", abs_x);
    let z = ctx.where_self(graph, "z", mask_hi, inv_x, abs_x);
    let z2 = ctx.mul(graph, "z2", z, z);
    let num = match ctx.poly(graph, "num", &[1.0, 0.3529666667, -0.0287666667], z2) {
        Some(node) => node,
        None => return false,
    };
    let den = match ctx.poly(graph, "den", &[1.0, 0.6863], z2) {
        Some(node) => node,
        None => return false,
    };
    let inv_den = ctx.reciprocal(graph, "inv_den", den);
    let prod = ctx.mul(graph, "prod", num, inv_den);
    let atan_z = ctx.mul(graph, "atan_z", z, prod);
    let diff = ctx.sub(graph, "diff", half_pi, atan_z);
    let atan_abs = ctx.where_self(graph, "atan_abs", mask_hi, diff, atan_z);
    let mask_pos = ctx.gt(graph, "mask_pos", x_id, zero);
    let neg = ctx.neg(graph, "neg", atan_abs);
    ctx.replace_nodes(
        graph,
        "result",
        "aten.where.self",
        &[mask_pos, atan_abs, neg],
    );
    true
}

/// sinh(x) = 0.5 * (exp(x) - exp(-x))
fn decompose_sinh(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let half = ctx.full(graph, "half", 0.5);
    let exp_pos = ctx.exp(graph, "exp_pos", x_id);
    let neg = ctx.neg(graph, "neg", x_id);
    let exp_neg = ctx.exp(graph, "exp_neg", neg);
    let diff = ctx.sub(graph, "diff", exp_pos, exp_neg);
    ctx.replace_nodes(graph, "result", "aten.mul.Tensor", &[diff, half]);
    true
}

/// cosh(x) = 0.5 * (exp(x) + exp(-x))
fn decompose_cosh(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let half = ctx.full(graph, "half", 0.5);
    let exp_pos = ctx.exp(graph, "exp_pos", x_id);
    let neg = ctx.neg(graph, "neg", x_id);
    let exp_neg = ctx.exp(graph, "exp_neg", neg);
    let sum = ctx.add(graph, "sum", exp_pos, exp_neg);
    ctx.replace_nodes(graph, "result", "aten.mul.Tensor", &[sum, half]);
    true
}

/// asinh(x) = log(x + sqrt(x^2 + 1))
fn decompose_asinh(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let one = ctx.full(graph, "one", 1.0);
    let square = ctx.mul(graph, "square", x_id, x_id);
    let radicand = ctx.add(graph, "radicand", square, one);
    let sqrt = ctx.sqrt(graph, "sqrt", radicand);
    let add = ctx.add(graph, "sum", x_id, sqrt);
    ctx.replace_nodes(graph, "result", "aten.log.default", &[add]);
    true
}

/// acosh(x) = log(x + sqrt((x - 1) * (x + 1)))
fn decompose_acosh(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let one = ctx.full(graph, "one", 1.0);
    let minus = ctx.sub(graph, "minus", x_id, one);
    let plus = ctx.add(graph, "plus", x_id, one);
    let radicand = ctx.mul(graph, "radicand", minus, plus);
    let sqrt = ctx.sqrt(graph, "sqrt", radicand);
    let add = ctx.add(graph, "sum", x_id, sqrt);
    ctx.replace_nodes(graph, "result", "aten.log.default", &[add]);
    true
}

/// asin/acos decomposition matching the Python piecewise rational approximation.
fn decompose_asin_or_acos(graph: &mut IrGraph, id: NodeId, is_acos_op: bool) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let p_coefficients = [
        1.6666667163e-01,
        -3.2556581497e-01,
        2.0121252537e-01,
        -4.0055535734e-02,
        7.9153501429e-04,
    ];
    let q_coefficients = [1.0, -2.4033949375e00, 2.0209457874e00, -6.8828397989e-01];

    let one = ctx.full(graph, "one", 1.0);
    let half = ctx.full(graph, "half", 0.5);
    let neg_half = ctx.full(graph, "neg_half", -0.5);
    let two = ctx.full(graph, "two", 2.0);
    let half_pi = ctx.full(graph, "half_pi", FRAC_PI_2);
    let zero = ctx.full(graph, "zero", 0.0);

    let x_abs = ctx.tensor(graph, "abs", "aten.abs.default", vec![IrArg::Node(x_id)]);
    let y = ctx.tensor(
        graph,
        "y",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(x_abs)],
    );
    let x3 = ctx.tensor(
        graph,
        "x3",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(y)],
    );

    let p_small = match ctx.poly(graph, "p_small", &p_coefficients, x_abs) {
        Some(node_id) => node_id,
        None => return false,
    };
    let q_small = match ctx.poly(graph, "q_small", &q_coefficients, x_abs) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_small = ctx.tensor(
        graph,
        "numer_small",
        "aten.mul.Tensor",
        vec![IrArg::Node(x3), IrArg::Node(p_small)],
    );
    let r_small = ctx.tensor(
        graph,
        "r_small",
        "aten.div.Tensor",
        vec![IrArg::Node(numer_small), IrArg::Node(q_small)],
    );
    let asin_small = ctx.tensor(
        graph,
        "asin_small",
        "aten.add.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(r_small)],
    );
    let tmp = ctx.tensor(
        graph,
        "tmp",
        "aten.sub.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(one)],
    );
    let z = ctx.tensor(
        graph,
        "z",
        "aten.mul.Tensor",
        vec![IrArg::Node(tmp), IrArg::Node(neg_half)],
    );
    let s = ctx.tensor(graph, "s", "aten.sqrt.default", vec![IrArg::Node(z)]);
    let s2 = ctx.tensor(
        graph,
        "s2",
        "aten.mul.Tensor",
        vec![IrArg::Node(s), IrArg::Node(s)],
    );
    let s3 = ctx.tensor(
        graph,
        "s3",
        "aten.mul.Tensor",
        vec![IrArg::Node(s2), IrArg::Node(s)],
    );

    let pz = match ctx.poly(graph, "p_large", &p_coefficients, z) {
        Some(node_id) => node_id,
        None => return false,
    };
    let qz = match ctx.poly(graph, "q_large", &q_coefficients, z) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_large = ctx.tensor(
        graph,
        "numer_large",
        "aten.mul.Tensor",
        vec![IrArg::Node(s3), IrArg::Node(pz)],
    );
    let r_large = ctx.tensor(
        graph,
        "r_large",
        "aten.div.Tensor",
        vec![IrArg::Node(numer_large), IrArg::Node(qz)],
    );
    let t1 = ctx.tensor(
        graph,
        "t1",
        "aten.add.Tensor",
        vec![IrArg::Node(s), IrArg::Node(r_large)],
    );
    let t2 = ctx.tensor(
        graph,
        "t2",
        "aten.mul.Tensor",
        vec![IrArg::Node(t1), IrArg::Node(two)],
    );
    let diff = ctx.tensor(
        graph,
        "diff",
        "aten.sub.Tensor",
        vec![IrArg::Node(t2), IrArg::Node(half_pi)],
    );
    let asin_large = ctx.tensor(
        graph,
        "asin_large",
        "aten.neg.default",
        vec![IrArg::Node(diff)],
    );
    let mask_large = ctx.bool(
        graph,
        "mask_large",
        "aten.gt.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(half)],
    );
    let asin_unsigned = ctx.tensor(
        graph,
        "asin_unsigned",
        "aten.where.self",
        vec![
            IrArg::Node(mask_large),
            IrArg::Node(asin_large),
            IrArg::Node(asin_small),
        ],
    );
    let negated_asin = ctx.tensor(
        graph,
        "negated_asin",
        "aten.neg.default",
        vec![IrArg::Node(asin_unsigned)],
    );
    let mask_neg = ctx.bool(
        graph,
        "mask_neg",
        "aten.lt.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
    );
    let asin = ctx.tensor(
        graph,
        "asin",
        "aten.where.self",
        vec![
            IrArg::Node(mask_neg),
            IrArg::Node(negated_asin),
            IrArg::Node(asin_unsigned),
        ],
    );

    if is_acos_op {
        let acos_small = ctx.tensor(
            graph,
            "acos_small",
            "aten.sub.Tensor",
            vec![IrArg::Node(half_pi), IrArg::Node(asin)],
        );
        let mask_gt_half = ctx.bool(
            graph,
            "mask_gt_half",
            "aten.gt.Tensor",
            vec![IrArg::Node(x_id), IrArg::Node(half)],
        );
        ctx.replace_output(
            graph,
            "result",
            "aten.where.self",
            vec![
                IrArg::Node(mask_gt_half),
                IrArg::Node(t2),
                IrArg::Node(acos_small),
            ],
        );
    } else {
        ctx.replace_output(
            graph,
            "result",
            "aten.clone.default",
            vec![IrArg::Node(asin)],
        );
    }
    true
}

/// elu(x, alpha) = where(x >= 0, x, alpha * expm1(x))
fn decompose_elu(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };
    let alpha = elu_alpha(&graph.nodes[id]);

    let zero = ctx.full(graph, "zero", 0.0);
    let alpha_full = ctx.full(graph, "alpha", alpha);
    let expm1 = ctx.unary(graph, "expm1", "aten.expm1.default", x_id);
    let scaled = ctx.mul(graph, "scaled", expm1, alpha_full);
    let ge = ctx.ge(graph, "ge", x_id, zero);
    ctx.replace_nodes(graph, "result", "aten.where.self", &[ge, x_id, scaled]);
    true
}

/// gelu(x) decomposition matching the Python pass for approximate="none"/"tanh".
fn decompose_gelu(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };
    let approximate = gelu_approximate(&graph.nodes[id]).to_string();
    let half = ctx.full_scalar(graph, "half", 0.5);
    let one = ctx.full_scalar(graph, "one", 1.0);

    match approximate.as_str() {
        "none" => {
            let sqrt_half = ctx.full_scalar(graph, "sqrt_half", std::f64::consts::FRAC_1_SQRT_2);
            let scaled = ctx.mul(graph, "scaled", x_id, sqrt_half);
            let erf = ctx.erf(graph, "erf", scaled);
            let plus_one = ctx.add(graph, "plus_one", erf, one);
            let weighted = ctx.mul(graph, "weighted", plus_one, half);
            ctx.replace_nodes(graph, "result", "aten.mul.Tensor", &[x_id, weighted]);
        }
        "tanh" => {
            let sqrt_two = ctx.full_scalar(graph, "sqrt_two", std::f64::consts::SQRT_2);
            let two_over_sqrt_pi =
                ctx.full_scalar(graph, "two_over_sqrt_pi", 1.128_379_167_095_512_6);
            let cube_coeff = ctx.full_scalar(graph, "cube_coeff", 0.044_715);

            let sqrt_mul = ctx.tensor_scalar(
                graph,
                "sqrt_mul",
                "aten.mul.Tensor",
                vec![IrArg::Node(sqrt_two), IrArg::Node(two_over_sqrt_pi)],
            );
            let sqrt_2_pi = ctx.tensor_scalar(
                graph,
                "sqrt_2_pi",
                "aten.mul.Tensor",
                vec![IrArg::Node(sqrt_mul), IrArg::Node(half)],
            );
            let square = ctx.mul(graph, "square", x_id, x_id);
            let cube = ctx.mul(graph, "cube", square, x_id);
            let cubic_term = ctx.mul(graph, "cubic_term", cube, cube_coeff);
            let inner = ctx.add(graph, "inner", x_id, cubic_term);
            let scaled = ctx.mul(graph, "scaled", inner, sqrt_2_pi);
            let tanh = ctx.tanh(graph, "tanh", scaled);
            let plus_one = ctx.add(graph, "plus_one", tanh, one);
            let first_mul = ctx.mul(graph, "first_mul", x_id, plus_one);
            ctx.replace_nodes(graph, "result", "aten.mul.Tensor", &[first_mul, half]);
        }
        _ => return false,
    }
    true
}

/// erfinv(x) decomposition matching the Python pass.
fn decompose_erfinv(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let central_range = ctx.full_scalar(graph, "central_range", 0.7);
    let a0 = ctx.full_scalar(graph, "a0", 0.886_226_899);
    let a1 = ctx.full_scalar(graph, "a1", -1.645_349_621);
    let a2 = ctx.full_scalar(graph, "a2", 0.914_624_893);
    let a3 = ctx.full_scalar(graph, "a3", -0.140_543_331);
    let b0 = ctx.full_scalar(graph, "b0", -2.118_377_725);
    let b1 = ctx.full_scalar(graph, "b1", 1.442_710_462);
    let b2 = ctx.full_scalar(graph, "b2", -0.329_097_515);
    let b3 = ctx.full_scalar(graph, "b3", 0.012_229_801);
    let c0 = ctx.full_scalar(graph, "c0", -1.970_840_454);
    let c1 = ctx.full_scalar(graph, "c1", -1.624_906_493);
    let c2 = ctx.full_scalar(graph, "c2", 3.429_567_803);
    let c3 = ctx.full_scalar(graph, "c3", 1.641_345_311);
    let d0 = ctx.full_scalar(graph, "d0", 3.543_889_2);
    let d1 = ctx.full_scalar(graph, "d1", 1.637_067_8);
    let half = ctx.full_scalar(graph, "half", 0.5);
    let one = ctx.full_scalar(graph, "one", 1.0);
    let zeros = ctx.full_scalar(graph, "zeros", 0.0);
    let neg_ones = ctx.full_scalar(graph, "neg_ones", -1.0);
    let refine_max_abs = ctx.full_scalar(graph, "refine_max_abs", 0.95);
    let deriv_eps = ctx.full_scalar(graph, "deriv_eps", 1.0e-3);
    let corr_max = ctx.full_scalar(graph, "corr_max", 0.5);
    let two_over_sqrt_pi = ctx.full_scalar(graph, "two_over_sqrt_pi", 1.128_379_167_095_512_6);
    let nan = ctx.full_scalar(graph, "nan", f64::NAN);
    let pos_inf = ctx.full_scalar(graph, "pos_inf", f64::INFINITY);
    let neg_inf = ctx.full_scalar(graph, "neg_inf", f64::NEG_INFINITY);
    graph.nodes[pos_inf]
        .meta
        .set_extra("arm_preserve_inf_literal", MetaValue::Bool(true));
    graph.nodes[neg_inf]
        .meta
        .set_extra("arm_preserve_inf_literal", MetaValue::Bool(true));

    let x_lt0 = ctx.bool(
        graph,
        "x_lt0",
        "aten.lt.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zeros)],
    );

    let s = ctx.tensor(
        graph,
        "sign",
        "aten.where.self",
        vec![IrArg::Node(x_lt0), IrArg::Node(neg_ones), IrArg::Node(one)],
    );

    let a = ctx.tensor(graph, "abs", "aten.abs.default", vec![IrArg::Node(x_id)]);

    let abs_gt1 = ctx.bool(
        graph,
        "abs_gt1",
        "aten.gt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(one)],
    );

    let abs_eq1 = ctx.bool(
        graph,
        "abs_eq1",
        "aten.eq.Tensor",
        vec![IrArg::Node(a), IrArg::Node(one)],
    );

    let a_math_gt = ctx.tensor(
        graph,
        "a_math_gt",
        "aten.where.self",
        vec![IrArg::Node(abs_gt1), IrArg::Node(zeros), IrArg::Node(a)],
    );

    let a_math = ctx.tensor(
        graph,
        "a_math",
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(zeros),
            IrArg::Node(a_math_gt),
        ],
    );

    let z = ctx.tensor(
        graph,
        "z",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(x_id)],
    );
    let p = match ctx.horner_nodes(graph, "p", z, &[a0, a1, a2, a3]) {
        Some(node) => node,
        None => return false,
    };
    let q = match ctx.horner_nodes(graph, "q", z, &[one, b0, b1, b2, b3]) {
        Some(node) => node,
        None => return false,
    };

    let xp = ctx.tensor(
        graph,
        "xp",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(p)],
    );
    let y0_central = ctx.tensor(
        graph,
        "y0_central",
        "aten.div.Tensor",
        vec![IrArg::Node(xp), IrArg::Node(q)],
    );

    let one_minus_a = ctx.tensor(
        graph,
        "one_minus_a",
        "aten.sub.Tensor",
        vec![IrArg::Node(one), IrArg::Node(a_math)],
    );
    let u = ctx.tensor(
        graph,
        "u",
        "aten.mul.Tensor",
        vec![IrArg::Node(one_minus_a), IrArg::Node(half)],
    );
    let u_safe = ctx.tensor(
        graph,
        "u_safe",
        "aten.where.self",
        vec![IrArg::Node(abs_eq1), IrArg::Node(one), IrArg::Node(u)],
    );
    let log_u = ctx.tensor(
        graph,
        "log_u",
        "aten.log.default",
        vec![IrArg::Node(u_safe)],
    );
    let neg_log_u = ctx.tensor(
        graph,
        "neg_log_u",
        "aten.mul.Tensor",
        vec![IrArg::Node(log_u), IrArg::Node(neg_ones)],
    );
    let t = ctx.tensor(
        graph,
        "t",
        "aten.sqrt.default",
        vec![IrArg::Node(neg_log_u)],
    );
    let r = match ctx.horner_nodes(graph, "r", t, &[c0, c1, c2, c3]) {
        Some(node) => node,
        None => return false,
    };
    let s_den = match ctx.horner_nodes(graph, "s_den", t, &[one, d0, d1]) {
        Some(node) => node,
        None => return false,
    };

    let frac = ctx.tensor(
        graph,
        "frac",
        "aten.div.Tensor",
        vec![IrArg::Node(r), IrArg::Node(s_den)],
    );
    let frac_abs = ctx.tensor(
        graph,
        "frac_abs",
        "aten.abs.default",
        vec![IrArg::Node(frac)],
    );
    let y0_tail = ctx.tensor(
        graph,
        "y0_tail",
        "aten.mul.Tensor",
        vec![IrArg::Node(s), IrArg::Node(frac_abs)],
    );

    let in_central = ctx.bool(
        graph,
        "in_central",
        "aten.lt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(central_range)],
    );

    let y0_pre = ctx.tensor(
        graph,
        "y0_pre",
        "aten.where.self",
        vec![
            IrArg::Node(in_central),
            IrArg::Node(y0_central),
            IrArg::Node(y0_tail),
        ],
    );
    let y0_no_gt = ctx.tensor(
        graph,
        "y0_no_gt",
        "aten.where.self",
        vec![
            IrArg::Node(abs_gt1),
            IrArg::Node(zeros),
            IrArg::Node(y0_pre),
        ],
    );
    let y0 = ctx.tensor(
        graph,
        "y0",
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(zeros),
            IrArg::Node(y0_no_gt),
        ],
    );

    let refine_mask = ctx.bool(
        graph,
        "refine_mask",
        "aten.lt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(refine_max_abs)],
    );

    let erf_y = ctx.tensor(graph, "erf_y", "aten.erf.default", vec![IrArg::Node(y0)]);
    let err = ctx.tensor(
        graph,
        "err",
        "aten.sub.Tensor",
        vec![IrArg::Node(erf_y), IrArg::Node(x_id)],
    );
    let y_sq = ctx.tensor(
        graph,
        "y_sq",
        "aten.mul.Tensor",
        vec![IrArg::Node(y0), IrArg::Node(y0)],
    );
    let neg_y_sq = ctx.tensor(
        graph,
        "neg_y_sq",
        "aten.mul.Tensor",
        vec![IrArg::Node(y_sq), IrArg::Node(neg_ones)],
    );
    let exp_term = ctx.tensor(
        graph,
        "exp_term",
        "aten.exp.default",
        vec![IrArg::Node(neg_y_sq)],
    );
    let deriv = ctx.tensor(
        graph,
        "deriv",
        "aten.mul.Tensor",
        vec![IrArg::Node(exp_term), IrArg::Node(two_over_sqrt_pi)],
    );
    let deriv_tiny = ctx.bool(
        graph,
        "deriv_tiny",
        "aten.lt.Tensor",
        vec![IrArg::Node(deriv), IrArg::Node(deriv_eps)],
    );
    let corr = ctx.tensor(
        graph,
        "corr",
        "aten.div.Tensor",
        vec![IrArg::Node(err), IrArg::Node(deriv)],
    );
    let corr_abs = ctx.tensor(
        graph,
        "corr_abs",
        "aten.abs.default",
        vec![IrArg::Node(corr)],
    );
    let corr_huge = ctx.bool(
        graph,
        "corr_huge",
        "aten.gt.Tensor",
        vec![IrArg::Node(corr_abs), IrArg::Node(corr_max)],
    );
    let y1 = ctx.tensor(
        graph,
        "y1",
        "aten.sub.Tensor",
        vec![IrArg::Node(y0), IrArg::Node(corr)],
    );
    let y_safe_small = ctx.tensor(
        graph,
        "y_safe_small",
        "aten.where.self",
        vec![IrArg::Node(deriv_tiny), IrArg::Node(y0), IrArg::Node(y1)],
    );
    let y_safe = ctx.tensor(
        graph,
        "y_safe",
        "aten.where.self",
        vec![
            IrArg::Node(corr_huge),
            IrArg::Node(y0),
            IrArg::Node(y_safe_small),
        ],
    );
    let y = ctx.tensor(
        graph,
        "y",
        "aten.where.self",
        vec![
            IrArg::Node(refine_mask),
            IrArg::Node(y_safe),
            IrArg::Node(y0),
        ],
    );

    let inf_signed = ctx.tensor(
        graph,
        "inf_signed",
        "aten.where.self",
        vec![
            IrArg::Node(x_lt0),
            IrArg::Node(neg_inf),
            IrArg::Node(pos_inf),
        ],
    );
    let out_gt1 = ctx.tensor(
        graph,
        "out_gt1",
        "aten.where.self",
        vec![IrArg::Node(abs_gt1), IrArg::Node(nan), IrArg::Node(y)],
    );
    ctx.replace_output(
        graph,
        "result",
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(inf_signed),
            IrArg::Node(out_gt1),
        ],
    );
    true
}

/// Inference batch norm decomposition for the no-training case.
fn decompose_batch_norm(graph: &mut IrGraph, id: NodeId) -> bool {
    if batch_norm_training(&graph.nodes[id]) {
        return false;
    }

    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };
    let weight_arg = graph.nodes[id].arg_clone_or_none(1);
    let bias_arg = graph.nodes[id].arg_clone_or_none(2);
    let running_mean = match graph.nodes[id].arg_node(3) {
        Some(n) => n,
        None => return false,
    };
    let running_var = match graph.nodes[id].arg_node(4) {
        Some(n) => n,
        None => return false,
    };
    let shape = ctx.shape.clone();
    if shape.len() < 2 {
        return false;
    }
    let channel_axis = 1_usize;
    let Some(&num_features) = shape.get(channel_axis) else {
        return false;
    };
    let mut channel_shape = vec![1_i64; shape.len()];
    channel_shape[channel_axis] = num_features;
    let scalar_shape = ctx.scalar_shape();
    let eps = batch_norm_eps(&graph.nodes[id]);
    let mean_view = ctx.view_shape(graph, "mean_view", running_mean, &channel_shape);
    let var_view = ctx.view_shape(graph, "var_view", running_var, &channel_shape);

    let centered = ctx.sub(graph, "centered", x_id, mean_view);
    let eps_full = ctx.full_shape(graph, "eps", eps, &scalar_shape);

    let var_eps = ctx.tensor_nodes_shape(
        graph,
        "var_eps",
        "aten.add.Tensor",
        &[var_view, eps_full],
        &channel_shape,
    );

    let inv_sqrt = ctx.tensor_nodes_shape(
        graph,
        "inv_sqrt",
        "aten.rsqrt.default",
        &[var_eps],
        &channel_shape,
    );

    let normed = ctx.mul(graph, "normed", centered, inv_sqrt);

    let weight_view = match weight_arg {
        IrArg::Node(weight_id) => ctx.view_shape(graph, "weight_view", weight_id, &channel_shape),
        IrArg::None => {
            let weight_full = ctx.full_shape(graph, "weight_full", 1.0, &[num_features]);
            ctx.view_shape(graph, "weight_view", weight_full, &channel_shape)
        }
        _ => return false,
    };

    let weighted = ctx.mul(graph, "weighted", normed, weight_view);

    let bias_view = match bias_arg {
        IrArg::Node(bias_id) => ctx.view_shape(graph, "bias_view", bias_id, &channel_shape),
        IrArg::None => {
            let bias_full = ctx.full_shape(graph, "bias_full", 0.0, &[num_features]);
            ctx.view_shape(graph, "bias_view", bias_full, &channel_shape)
        }
        _ => return false,
    };

    let result = ctx.add(graph, "result", weighted, bias_view);
    replace_primary_result_users(graph, id, result)
}

fn decompose_group_norm(graph: &mut IrGraph, id: NodeId) -> bool {
    let args = graph.nodes[id].args.clone();
    let Some((x_id, ctx)) = ApproxCtx::from_first_input(graph, id) else {
        return false;
    };

    let (weight_arg, bias_arg, group, eps) = match args.len() {
        8 => (
            args.get(1).cloned().unwrap_or(IrArg::None),
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(6).and_then(IrArg::as_int),
            args.get(7).and_then(IrArg::as_numeric_f64),
        ),
        6 => (
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(3).cloned().unwrap_or(IrArg::None),
            args.get(1).and_then(IrArg::as_int),
            args.get(4).and_then(IrArg::as_numeric_f64),
        ),
        5 => (
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(3).cloned().unwrap_or(IrArg::None),
            args.get(1).and_then(IrArg::as_int),
            args.get(4).and_then(IrArg::as_numeric_f64),
        ),
        4 => (
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(3).cloned().unwrap_or(IrArg::None),
            args.get(1).and_then(IrArg::as_int),
            None,
        ),
        2 => (
            IrArg::None,
            IrArg::None,
            args.get(1).and_then(IrArg::as_int),
            None,
        ),
        _ => (
            graph.nodes[id].kwarg_clone_or_none("weight"),
            graph.nodes[id].kwarg_clone_or_none("bias"),
            graph.nodes[id]
                .kwarg_int("num_groups")
                .or_else(|| graph.nodes[id].kwarg_int("group")),
            graph.nodes[id].kwarg_numeric_f64("eps"),
        ),
    };
    let Some(group) = group else {
        return false;
    };
    if group <= 0 {
        return false;
    }

    let shape = ctx.shape.clone();
    if shape.len() < 2 {
        return false;
    }
    let eps = eps.unwrap_or(1.0e-5);
    let n = shape[0];
    let c = shape[1];
    if c <= 0 || c % group != 0 {
        return false;
    }
    let channels_per_group = c / group;
    let hxw = shape.iter().skip(2).product::<i64>().max(1);
    let grouped_shape = vec![n, group, channels_per_group, hxw];
    let reduce_dims = i64_list(&[2, 3]);
    let scalar_shape = vec![1_i64; grouped_shape.len()];
    let stat_shape = vec![n, group, 1, 1];
    let affine_shape = vec![1, group, channels_per_group, 1];
    let reshape_target = if graph.nodes[id].target.contains("native_group_norm") {
        "aten.view_copy.default"
    } else {
        "aten.reshape.default"
    };
    let grouped = ctx.reshape_shape(graph, "grouped", reshape_target, x_id, &grouped_shape);
    let inv_n = ctx.full_shape(
        graph,
        "inv_n",
        1.0 / ((channels_per_group * hxw) as f64),
        &scalar_shape,
    );
    let sum_x = ctx.tensor_shape(
        graph,
        "sum",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(grouped), reduce_dims.clone(), IrArg::Bool(true)],
        &stat_shape,
    );
    let mean = ctx.tensor_nodes_shape(
        graph,
        "mean",
        "aten.mul.Tensor",
        &[sum_x, inv_n],
        &stat_shape,
    );
    let centered = ctx.tensor_nodes_shape(
        graph,
        "centered",
        "aten.sub.Tensor",
        &[grouped, mean],
        &grouped_shape,
    );
    let squared = ctx.tensor_nodes_shape(
        graph,
        "squared",
        "aten.mul.Tensor",
        &[centered, centered],
        &grouped_shape,
    );
    let sum_squared = ctx.tensor_shape(
        graph,
        "sum_squared",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(squared), reduce_dims, IrArg::Bool(true)],
        &stat_shape,
    );
    let var = ctx.tensor_nodes_shape(
        graph,
        "var",
        "aten.mul.Tensor",
        &[sum_squared, inv_n],
        &stat_shape,
    );
    let eps_full = ctx.full_shape(graph, "eps", eps, &scalar_shape);
    let var_eps = ctx.tensor_nodes_shape(
        graph,
        "var_eps",
        "aten.add.Tensor",
        &[var, eps_full],
        &stat_shape,
    );
    let rsqrt = ctx.tensor_nodes_shape(
        graph,
        "rsqrt",
        "aten.rsqrt.default",
        &[var_eps],
        &stat_shape,
    );
    let normalized = ctx.tensor_nodes_shape(
        graph,
        "normalized",
        "aten.mul.Tensor",
        &[centered, rsqrt],
        &grouped_shape,
    );

    let weighted = match weight_arg {
        IrArg::Node(weight_id) => {
            let weight_view = ctx.reshape_shape(
                graph,
                "weight_view",
                reshape_target,
                weight_id,
                &affine_shape,
            );
            ctx.tensor_nodes_shape(
                graph,
                "weighted",
                "aten.mul.Tensor",
                &[normalized, weight_view],
                &grouped_shape,
            )
        }
        IrArg::None => normalized,
        _ => return false,
    };

    let biased = match bias_arg {
        IrArg::Node(bias_id) => {
            let bias_view =
                ctx.reshape_shape(graph, "bias_view", reshape_target, bias_id, &affine_shape);
            ctx.tensor_nodes_shape(
                graph,
                "biased",
                "aten.add.Tensor",
                &[weighted, bias_view],
                &grouped_shape,
            )
        }
        IrArg::None => weighted,
        _ => return false,
    };

    let result = ctx.reshape_shape(graph, "result", reshape_target, biased, &shape);
    replace_primary_result_users(graph, id, result)
}

/// stack(tensors, dim) → unsqueeze each + cat
fn decompose_stack(graph: &mut IrGraph, id: NodeId) -> bool {
    let args = &graph.nodes[id].args;
    if args.is_empty() {
        return false;
    }

    // args[0] = List of node refs, args[1] = dim (default 0)
    let tensor_ids: Vec<NodeId> = match &args[0] {
        IrArg::List(items) => items.iter().filter_map(|a| a.as_node()).collect(),
        _ => return false,
    };
    if tensor_ids.is_empty() {
        return false;
    }
    let dim = graph.nodes[id].arg_int_or(1, 0);

    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let dtype = graph.nodes[tensor_ids[0]]
        .meta
        .val_dtype_or_float32()
        .to_string();
    let metas = MetaFactory::new(&meta, &dtype);

    // Unsqueeze each tensor at dim
    let mut unsqueezed: Vec<NodeId> = Vec::new();
    for (i, &tid) in tensor_ids.iter().enumerate() {
        let t_shape = graph.nodes[tid]
            .meta
            .val_shape()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let mut u_shape = t_shape;
        let pos = if dim < 0 {
            (dim + u_shape.len() as i64 + 1) as usize
        } else {
            dim as usize
        };
        u_shape.insert(pos, 1);

        let unsq = graph.add_anchored_call_function(
            id,
            format!("{}_unsq_{}", name, i),
            "aten.unsqueeze_copy.default".to_string(),
            vec![IrArg::Node(tid), IrArg::Int(dim)],
            metas.tensor(&u_shape),
        );
        unsqueezed.push(unsq);
    }

    // Cat along dim
    let cat_args = IrArg::List(unsqueezed.iter().map(|&n| IrArg::Node(n)).collect());
    graph.replace_with_call_function(
        id,
        format!("{}_cat", name),
        "aten.cat.default",
        vec![cat_args, IrArg::Int(dim)],
        meta,
    );
    true
}

/// unbind(x, dim) → for each getitem user: slice + squeeze
fn decompose_unbind(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let dim = graph.nodes[id]
        .args
        .get(1)
        .and_then(|a| a.as_int())
        .unwrap_or(0);

    let x_shape = match graph.node_shape_vec(x_id) {
        Some(s) => s,
        None => return false,
    };
    let rank = x_shape.len() as i64;
    let dim_norm = dim.rem_euclid(rank) as usize;
    let dtype = graph.node_dtype_or_float32(x_id).to_string();
    let name = graph.nodes[id].name.clone();
    let orig_meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&orig_meta, &dtype);

    let getitem_users = graph.live_getitem_users(id);

    if getitem_users.is_empty() {
        return false;
    }

    for (getitem_id, idx) in &getitem_users {
        // slice(x, dim, idx, idx+1)
        let mut slice_shape = x_shape.clone();
        slice_shape[dim_norm] = 1;
        let slice_id = graph.add_anchored_call_function(
            id,
            format!("{}_slice_{}", name, idx),
            "aten.slice_copy.Tensor".to_string(),
            vec![
                IrArg::Node(x_id),
                IrArg::Int(dim_norm as i64),
                IrArg::Int(*idx),
                IrArg::Int(idx + 1),
            ],
            metas.tensor(&slice_shape),
        );

        // squeeze(slice, dim)
        let mut sq_shape = x_shape.clone();
        sq_shape.remove(dim_norm);
        let squeeze_id = graph.add_anchored_call_function(
            slice_id,
            format!("{}_sq_{}", name, idx),
            "aten.squeeze_copy.dims".to_string(),
            vec![IrArg::Node(slice_id), i64_list(&[dim_norm as i64])],
            metas.tensor(&sq_shape),
        );

        graph.replace_and_erase(*getitem_id, squeeze_id);
    }

    // Erase the unbind node itself (now has no users)
    graph.erase(id);
    true
}
