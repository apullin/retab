//! Pure Rust IrPass: atanh, stack, unbind decompositions.
//! Mirrors FxDecomposeOpsPass (hardsigmoid is intentionally skipped).

use std::f64::consts::FRAC_PI_2;

use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta, NodeOp,
    TensorMetaStamp,
};
use crate::fx_target::{add_sub_alpha_result_target, group_norm_reshape_target};
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

struct MetaFactory<'a> {
    tensor: TensorMetaStamp<'a>,
    bool_tensor: TensorMetaStamp<'a>,
}

impl<'a> MetaFactory<'a> {
    fn new(base: &'a NodeMeta, dtype: &'a str) -> Self {
        Self {
            tensor: base.tensor_stamp(dtype),
            bool_tensor: base.tensor_stamp("torch.bool"),
        }
    }

    fn from_tensor_stamp(tensor: TensorMetaStamp<'a>) -> Self {
        Self {
            bool_tensor: tensor.with_dtype("torch.bool"),
            tensor,
        }
    }

    fn tensor(&self, shape: &[i64]) -> NodeMeta {
        self.tensor.preserve_qparams(shape.to_vec())
    }

    fn bool(&self, shape: &[i64]) -> NodeMeta {
        self.bool_tensor.preserve_qparams(shape.to_vec())
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
        graph.add_anchored_call_function(
            self.anchor_id,
            self.op_name(suffix),
            target,
            args,
            self.tensor_meta(shape),
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

    fn output_nodes(
        &self,
        graph: &mut IrGraph,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.output(
            graph,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
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
        graph.add_anchored_view_copy(
            self.anchor_id,
            self.op_name(suffix),
            source,
            shape,
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

    fn lt(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.bool_nodes(graph, suffix, "aten.lt.Tensor", &[lhs, rhs])
    }

    fn eq(&self, graph: &mut IrGraph, suffix: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.bool_nodes(graph, suffix, "aten.eq.Tensor", &[lhs, rhs])
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

const fn skip_quantized(any: &'static [&'static str], decompose: DecomposeFn) -> DecomposeRule {
    DecomposeRule {
        matcher: TargetMatcher::any(any),
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose,
    }
}

const fn always(any: &'static [&'static str], decompose: DecomposeFn) -> DecomposeRule {
    DecomposeRule {
        matcher: TargetMatcher::any(any),
        mode: DecomposeMode::Always,
        decompose,
    }
}

const fn always_except(
    any: &'static [&'static str],
    none: &'static [&'static str],
    decompose: DecomposeFn,
) -> DecomposeRule {
    DecomposeRule {
        matcher: TargetMatcher::any_except(any, none),
        mode: DecomposeMode::Always,
        decompose,
    }
}

const fn always_matcher(matcher: TargetMatcher, decompose: DecomposeFn) -> DecomposeRule {
    DecomposeRule {
        matcher,
        mode: DecomposeMode::Always,
        decompose,
    }
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
    skip_quantized(&["atanh"], decompose_atanh),
    skip_quantized(&["aten.tan."], decompose_tan),
    skip_quantized(&["aten.sinh."], decompose_sinh),
    skip_quantized(&["aten.cosh."], decompose_cosh),
    skip_quantized(&["aten.asinh."], decompose_asinh),
    skip_quantized(&["aten.asin."], decompose_asin),
    skip_quantized(&["aten.acos."], decompose_acos),
    skip_quantized(&["aten.atan."], decompose_atan),
    skip_quantized(&["aten.acosh."], decompose_acosh),
    skip_quantized(&["aten.elu."], decompose_elu),
    skip_quantized(&["aten.gelu."], decompose_gelu),
    skip_quantized(&["aten.erfinv."], decompose_erfinv),
    skip_quantized(&["pow.Tensor_Scalar"], decompose_pow_scalar),
    always(
        &["aten.group_norm", "aten.native_group_norm"],
        decompose_group_norm,
    ),
    always(
        &[
            "aten.batch_norm",
            "aten.native_batch_norm",
            "aten._native_batch_norm_legit_no_training",
            "aten._native_batch_norm_legit.no_stats",
        ],
        decompose_batch_norm,
    ),
    always(
        &["aten.log_softmax.", "aten._log_softmax."],
        decompose_log_softmax_wrapper,
    ),
    always(
        &["aten.softmax.", "aten._safe_softmax.", "aten._softmax."],
        decompose_softmax,
    ),
    always_matcher(MAX_DIM_MATCHER, decompose_dim_extrema),
    always_matcher(MIN_DIM_MATCHER, decompose_dim_extrema),
    always_except(&["stack"], &["unstack"], decompose_stack),
    always(&["unbind"], decompose_unbind),
    always(&["add.Tensor", "sub.Tensor"], decompose_add_sub_alpha),
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
    let Some(target) = add_sub_alpha_result_target(&graph.nodes[id].target) else {
        return false;
    };
    let alpha_full = rhs_ctx.full(graph, "alpha_full", alpha);
    let scaled_rhs =
        rhs_ctx.output_nodes(graph, "alpha_mul", "aten.mul.Tensor", &[rhs_id, alpha_full]);

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
        let eq = ctx.eq(graph, "self_eq", input_id, input_id);
        let ones = ctx.output_nodes(graph, "ones", "aten.where.self", &[eq, one, one]);
        graph.replace_and_erase(id, ones);
        return true;
    }

    let Some(mut exp) = pow_scalar_int_arg(&graph.nodes[id]) else {
        return false;
    };

    let mut base_id = input_id;
    if exp < 0 {
        let reciprocal =
            ctx.output_nodes(graph, "reciprocal", "aten.reciprocal.default", &[input_id]);
        base_id = reciprocal;
        exp = -exp;
    }

    let mut result_id = base_id;
    for idx in 1..exp {
        let mul = ctx.output_nodes(
            graph,
            &format!("mul{idx}"),
            "aten.mul.Tensor",
            &[result_id, base_id],
        );
        result_id = mul;
    }

    graph.replace_and_erase(id, result_id);
    true
}

#[cfg(test)]
mod tests {
    use crate::test_ir::{
        call, f32_call, f32_meta, f32_placeholder, has_live_target, live_node, meta as tensor_meta,
        placeholder,
    };

    use super::*;

    #[test]
    fn decompose_max_dim_rewrites_to_amax_and_squeeze() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3, 4]);
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
            f32_meta(&[2, 4]),
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
        let x = f32_placeholder(&mut graph, "x", &[2, 3, 4]);
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
            f32_meta(&[2, 4]),
        );
        let indices = graph.add_node(
            NodeOp::CallFunction,
            "getitem_1".to_string(),
            "getitem".to_string(),
            vec![IrArg::Node(max), IrArg::Int(1)],
            tensor_meta(&[2, 4], "torch.int64"),
        );
        let _use_indices = graph.add_node(
            NodeOp::CallFunction,
            "use_indices".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(indices), IrArg::Node(indices)],
            tensor_meta(&[2, 4], "torch.int64"),
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[max].alive);
        assert!(graph.nodes[indices].alive);
    }

    #[test]
    fn decompose_atan_rewrites_to_piecewise_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let atan = f32_call(
            &mut graph,
            "atan",
            "aten.atan.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[atan].alive);
        assert!(has_live_target(&graph, "atan_result", "aten.where.self"));
    }

    #[test]
    fn decompose_acos_rewrites_to_piecewise_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let acos = f32_call(
            &mut graph,
            "acos",
            "aten.acos.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[acos].alive);
        assert!(has_live_target(&graph, "acos_result", "aten.where.self"));
    }

    #[test]
    fn decompose_softmax_rewrites_to_reduce_exp_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3, 4]);
        let softmax = f32_call(
            &mut graph,
            "softmax",
            "aten._softmax.default",
            vec![IrArg::Node(x), IrArg::Int(1)],
            &[2, 3, 4],
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
        assert!(has_live_target(
            &graph,
            "softmax_softmax",
            "aten.mul.Tensor"
        ));
    }

    #[test]
    fn decompose_log_softmax_rewrites_to_log_of_softmax_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3, 4]);
        let log_softmax = f32_call(
            &mut graph,
            "log_softmax",
            "aten._log_softmax.default",
            vec![IrArg::Node(x), IrArg::Int(1)],
            &[2, 3, 4],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[log_softmax].alive);
        assert!(has_live_target(
            &graph,
            "log_softmax_result",
            "aten.log.default"
        ));
    }

    #[test]
    fn decompose_add_with_alpha_scales_rhs_before_add() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[1, 4, 5]);
        let y = f32_placeholder(&mut graph, "y", &[4, 1]);
        let add = f32_call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(y), IrArg::Float(1.5)],
            &[1, 4, 5],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[add].alive);

        let scaled_rhs = live_node(&graph, "add_alpha_mul");
        assert_eq!(scaled_rhs.target, "aten.mul.Tensor");

        let final_add = live_node(&graph, "add_alpha_result");
        assert_eq!(final_add.target, "aten.add.Tensor");
        assert_eq!(final_add.args.len(), 2);
        assert_eq!(final_add.args[0].as_node(), Some(x));
        assert_eq!(final_add.args[1].as_node(), Some(scaled_rhs.id));
    }

    #[test]
    fn decompose_tan_rewrites_to_sin_cos_div() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let tan = f32_call(
            &mut graph,
            "tan",
            "aten.tan.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[tan].alive);

        let sin = live_node(&graph, "tan_sin");
        assert_eq!(sin.target, "aten.sin.default");
        assert_eq!(sin.args[0].as_node(), Some(x));

        let cos = live_node(&graph, "tan_cos");
        assert_eq!(cos.target, "aten.cos.default");
        assert_eq!(cos.args[0].as_node(), Some(x));

        let div = live_node(&graph, "tan_result");
        assert_eq!(div.target, "aten.div.Tensor");
        assert_eq!(div.args[0].as_node(), Some(sin.id));
        assert_eq!(div.args[1].as_node(), Some(cos.id));
    }

    #[test]
    fn decompose_pow_scalar_zero_rewrites_to_full_like() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[1, 2, 3, 7]);
        let pow = f32_call(
            &mut graph,
            "pow",
            "aten.pow.Tensor_Scalar",
            vec![IrArg::Node(x), IrArg::Float(0.0)],
            &[1, 2, 3, 7],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[pow].alive);

        let eq = live_node(&graph, "pow_self_eq");
        assert_eq!(eq.target, "aten.eq.Tensor");
        assert_eq!(eq.args[0].as_node(), Some(x));
        assert_eq!(eq.args[1].as_node(), Some(x));

        let one = live_node(&graph, "pow_one");
        assert_eq!(one.target, "aten.full.default");

        let ones = live_node(&graph, "pow_ones");
        assert_eq!(ones.target, "aten.where.self");
        assert_eq!(ones.args[0].as_node(), Some(eq.id));
        assert_eq!(ones.args[1].as_node(), Some(one.id));
        assert_eq!(ones.args[2].as_node(), Some(one.id));
    }

    #[test]
    fn decompose_pow_scalar_positive_int_rewrites_to_mul_chain() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[1, 2, 3, 6]);
        let pow = f32_call(
            &mut graph,
            "pow",
            "aten.pow.Tensor_Scalar",
            vec![IrArg::Node(x), IrArg::Int(3)],
            &[1, 2, 3, 6],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[pow].alive);

        let mul1 = live_node(&graph, "pow_mul1");
        assert_eq!(mul1.target, "aten.mul.Tensor");
        assert_eq!(mul1.args[0].as_node(), Some(x));
        assert_eq!(mul1.args[1].as_node(), Some(x));

        let mul2 = live_node(&graph, "pow_mul2");
        assert_eq!(mul2.target, "aten.mul.Tensor");
        assert_eq!(mul2.args[0].as_node(), Some(mul1.id));
        assert_eq!(mul2.args[1].as_node(), Some(x));
    }

    #[test]
    fn skips_tan_inside_quantized_envelope() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[2, 3], "torch.int8");
        let dq = f32_call(
            &mut graph,
            "x_dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );
        let tan = f32_call(
            &mut graph,
            "tan",
            "aten.tan.default",
            vec![IrArg::Node(dq)],
            &[2, 3],
        );
        let q = call(
            &mut graph,
            "tan_q",
            "quantized_decomposed.quantize_per_tensor.default",
            vec![IrArg::Node(tan)],
            &[2, 3],
            "torch.int8",
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[tan].alive);
        assert_eq!(graph.nodes[tan].target, "aten.tan.default");
        assert!(graph.nodes[q].alive);
    }

    #[test]
    fn skips_pow_inside_quantized_envelope() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[1, 2, 3, 6], "torch.int8");
        let dq = f32_call(
            &mut graph,
            "x_dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            vec![IrArg::Node(x)],
            &[1, 2, 3, 6],
        );
        let pow = f32_call(
            &mut graph,
            "pow",
            "aten.pow.Tensor_Scalar",
            vec![IrArg::Node(dq), IrArg::Int(3)],
            &[1, 2, 3, 6],
        );
        let q = call(
            &mut graph,
            "pow_q",
            "quantized_decomposed.quantize_per_tensor.default",
            vec![IrArg::Node(pow)],
            &[1, 2, 3, 6],
            "torch.int8",
        );

        assert!(!IrDecomposeOpsPass.run(&mut graph));
        assert!(graph.nodes[pow].alive);
        assert_eq!(graph.nodes[pow].target, "aten.pow.Tensor_Scalar");
        assert!(graph.nodes[q].alive);
    }

    #[test]
    fn decompose_sinh_rewrites_via_exp_sub_mul() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let sinh = f32_call(
            &mut graph,
            "sinh",
            "aten.sinh.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[sinh].alive);
        assert!(has_live_target(&graph, "sinh_result", "aten.mul.Tensor"));
    }

    #[test]
    fn decompose_cosh_rewrites_via_exp_add_mul() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let cosh = f32_call(
            &mut graph,
            "cosh",
            "aten.cosh.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[cosh].alive);
        assert!(has_live_target(&graph, "cosh_result", "aten.mul.Tensor"));
    }

    #[test]
    fn decompose_asinh_rewrites_via_log_and_sqrt() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let asinh = f32_call(
            &mut graph,
            "asinh",
            "aten.asinh.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[asinh].alive);
        assert!(has_live_target(&graph, "asinh_result", "aten.log.default"));
    }

    #[test]
    fn decompose_acosh_rewrites_via_log_and_sqrt() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let acosh = f32_call(
            &mut graph,
            "acosh",
            "aten.acosh.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[acosh].alive);
        assert!(has_live_target(&graph, "acosh_result", "aten.log.default"));
    }

    #[test]
    fn decompose_elu_rewrites_via_expm1_ge_where() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let elu = f32_call(
            &mut graph,
            "elu",
            "aten.elu.default",
            vec![IrArg::Node(x), IrArg::Float(2.0)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[elu].alive);
        assert!(has_live_target(&graph, "elu_result", "aten.where.self"));
    }

    #[test]
    fn decompose_gelu_none_rewrites_via_erf() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let gelu = f32_call(
            &mut graph,
            "gelu",
            "aten.gelu.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[gelu].alive);
        assert!(has_live_target(&graph, "gelu_result", "aten.mul.Tensor"));
        assert!(has_live_target(&graph, "gelu_erf", "aten.erf.default"));
    }

    #[test]
    fn decompose_gelu_tanh_rewrites_via_tanh() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", &[2, 3]);
        let gelu = f32_call(
            &mut graph,
            "gelu",
            "aten.gelu.default",
            vec![IrArg::Node(x)],
            &[2, 3],
        );
        graph.set_kwarg(gelu, "approximate", IrArg::Str("tanh".to_string()));

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[gelu].alive);
        assert!(has_live_target(&graph, "gelu_tanh", "aten.tanh.default"));
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

    let mut keepdim_shape = input_shape;
    keepdim_shape[dim] = 1;

    let mut reduce_meta = value_meta
        .tensor_stamp_same_dtype_or_dtype_of_or_float32(&input_meta)
        .preserve_qparams(keepdim_shape);
    reduce_meta.copy_dim_order_facts_from(&input_meta);

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

    let x_abs = ctx.abs(graph, "abs", x_id);
    let y = ctx.mul(graph, "y", x_abs, x_abs);
    let x3 = ctx.mul(graph, "x3", x_abs, y);

    let p_small = match ctx.poly(graph, "p_small", &p_coefficients, x_abs) {
        Some(node_id) => node_id,
        None => return false,
    };
    let q_small = match ctx.poly(graph, "q_small", &q_coefficients, x_abs) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_small = ctx.mul(graph, "numer_small", x3, p_small);
    let r_small = ctx.div(graph, "r_small", numer_small, q_small);
    let asin_small = ctx.add(graph, "asin_small", x_abs, r_small);
    let tmp = ctx.sub(graph, "tmp", x_abs, one);
    let z = ctx.mul(graph, "z", tmp, neg_half);
    let s = ctx.sqrt(graph, "s", z);
    let s2 = ctx.mul(graph, "s2", s, s);
    let s3 = ctx.mul(graph, "s3", s2, s);

    let pz = match ctx.poly(graph, "p_large", &p_coefficients, z) {
        Some(node_id) => node_id,
        None => return false,
    };
    let qz = match ctx.poly(graph, "q_large", &q_coefficients, z) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_large = ctx.mul(graph, "numer_large", s3, pz);
    let r_large = ctx.div(graph, "r_large", numer_large, qz);
    let t1 = ctx.add(graph, "t1", s, r_large);
    let t2 = ctx.mul(graph, "t2", t1, two);
    let diff = ctx.sub(graph, "diff", t2, half_pi);
    let asin_large = ctx.neg(graph, "asin_large", diff);
    let mask_large = ctx.gt(graph, "mask_large", x_abs, half);
    let asin_unsigned = ctx.where_self(graph, "asin_unsigned", mask_large, asin_large, asin_small);
    let negated_asin = ctx.neg(graph, "negated_asin", asin_unsigned);
    let mask_neg = ctx.lt(graph, "mask_neg", x_id, zero);
    let asin = ctx.where_self(graph, "asin", mask_neg, negated_asin, asin_unsigned);

    if is_acos_op {
        let acos_small = ctx.sub(graph, "acos_small", half_pi, asin);
        let mask_gt_half = ctx.gt(graph, "mask_gt_half", x_id, half);
        ctx.replace_nodes(
            graph,
            "result",
            "aten.where.self",
            &[mask_gt_half, t2, acos_small],
        );
    } else {
        ctx.replace_nodes(graph, "result", "aten.clone.default", &[asin]);
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

    let x_lt0 = ctx.lt(graph, "x_lt0", x_id, zeros);
    let s = ctx.where_self(graph, "sign", x_lt0, neg_ones, one);
    let a = ctx.abs(graph, "abs", x_id);
    let abs_gt1 = ctx.gt(graph, "abs_gt1", a, one);
    let abs_eq1 = ctx.eq(graph, "abs_eq1", a, one);
    let a_math_gt = ctx.where_self(graph, "a_math_gt", abs_gt1, zeros, a);
    let a_math = ctx.where_self(graph, "a_math", abs_eq1, zeros, a_math_gt);

    let z = ctx.mul(graph, "z", x_id, x_id);
    let p = match ctx.horner_nodes(graph, "p", z, &[a0, a1, a2, a3]) {
        Some(node) => node,
        None => return false,
    };
    let q = match ctx.horner_nodes(graph, "q", z, &[one, b0, b1, b2, b3]) {
        Some(node) => node,
        None => return false,
    };

    let xp = ctx.mul(graph, "xp", x_id, p);
    let y0_central = ctx.div(graph, "y0_central", xp, q);

    let one_minus_a = ctx.sub(graph, "one_minus_a", one, a_math);
    let u = ctx.mul(graph, "u", one_minus_a, half);
    let u_safe = ctx.where_self(graph, "u_safe", abs_eq1, one, u);
    let log_u = ctx.log(graph, "log_u", u_safe);
    let neg_log_u = ctx.mul(graph, "neg_log_u", log_u, neg_ones);
    let t = ctx.sqrt(graph, "t", neg_log_u);
    let r = match ctx.horner_nodes(graph, "r", t, &[c0, c1, c2, c3]) {
        Some(node) => node,
        None => return false,
    };
    let s_den = match ctx.horner_nodes(graph, "s_den", t, &[one, d0, d1]) {
        Some(node) => node,
        None => return false,
    };

    let frac = ctx.div(graph, "frac", r, s_den);
    let frac_abs = ctx.abs(graph, "frac_abs", frac);
    let y0_tail = ctx.mul(graph, "y0_tail", s, frac_abs);

    let in_central = ctx.lt(graph, "in_central", a, central_range);

    let y0_pre = ctx.where_self(graph, "y0_pre", in_central, y0_central, y0_tail);
    let y0_no_gt = ctx.where_self(graph, "y0_no_gt", abs_gt1, zeros, y0_pre);
    let y0 = ctx.where_self(graph, "y0", abs_eq1, zeros, y0_no_gt);

    let refine_mask = ctx.lt(graph, "refine_mask", a, refine_max_abs);

    let erf_y = ctx.erf(graph, "erf_y", y0);
    let err = ctx.sub(graph, "err", erf_y, x_id);
    let y_sq = ctx.mul(graph, "y_sq", y0, y0);
    let neg_y_sq = ctx.mul(graph, "neg_y_sq", y_sq, neg_ones);
    let exp_term = ctx.exp(graph, "exp_term", neg_y_sq);
    let deriv = ctx.mul(graph, "deriv", exp_term, two_over_sqrt_pi);
    let deriv_tiny = ctx.lt(graph, "deriv_tiny", deriv, deriv_eps);
    let corr = ctx.div(graph, "corr", err, deriv);
    let corr_abs = ctx.abs(graph, "corr_abs", corr);
    let corr_huge = ctx.gt(graph, "corr_huge", corr_abs, corr_max);
    let y1 = ctx.sub(graph, "y1", y0, corr);
    let y_safe_small = ctx.where_self(graph, "y_safe_small", deriv_tiny, y0, y1);
    let y_safe = ctx.where_self(graph, "y_safe", corr_huge, y0, y_safe_small);
    let y = ctx.where_self(graph, "y", refine_mask, y_safe, y0);

    let inf_signed = ctx.where_self(graph, "inf_signed", x_lt0, neg_inf, pos_inf);
    let out_gt1 = ctx.where_self(graph, "out_gt1", abs_gt1, nan, y);
    ctx.replace_nodes(
        graph,
        "result",
        "aten.where.self",
        &[abs_eq1, inf_signed, out_gt1],
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
    let Some(reshape_target) = group_norm_reshape_target(&graph.nodes[id].target) else {
        return false;
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
    let first_meta = graph.nodes[tensor_ids[0]].meta.clone();
    let metas =
        MetaFactory::from_tensor_stamp(meta.tensor_stamp_with_dtype_of_or_float32(&first_meta));

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
    let name = graph.nodes[id].name.clone();
    let orig_meta = graph.nodes[id].meta.clone();
    let input_meta = graph.nodes[x_id].meta.clone();
    let metas = MetaFactory::from_tensor_stamp(
        orig_meta.tensor_stamp_with_dtype_of_or_float32(&input_meta),
    );

    let getitem_users = graph.live_getitem_users(id);

    if getitem_users.is_empty() {
        return false;
    }

    for (getitem_id, idx) in &getitem_users {
        // slice(x, dim, idx, idx+1)
        let mut slice_shape = x_shape.clone();
        slice_shape[dim_norm] = 1;
        let slice_id = graph.add_anchored_slice_copy(
            id,
            format!("{}_slice_{}", name, idx),
            x_id,
            dim_norm as i64,
            *idx,
            idx + 1,
            None,
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
