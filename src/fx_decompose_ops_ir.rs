//! Pure Rust IrPass: atanh, stack, unbind decompositions.
//! Mirrors FxDecomposeOpsPass (hardsigmoid is intentionally skipped).

use std::f64::consts::FRAC_PI_2;

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta,
    NodeOp,
};
use crate::runner_logging::log_line;

type TargetPredicate = fn(&str) -> bool;
type DecomposeFn = fn(&mut IrGraph, NodeId) -> bool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecomposeMode {
    Always,
    SkipQuantizedEnvelope,
}

#[derive(Clone, Copy)]
struct DecomposeRule {
    matches: TargetPredicate,
    mode: DecomposeMode,
    decompose: DecomposeFn,
}

fn channels_last_order(rank: usize) -> Vec<usize> {
    if rank <= 2 {
        return (0..rank).collect();
    }
    let spatial_rank = rank.saturating_sub(2);
    let channel_axis = rank - (spatial_rank + 1);
    let mut order: Vec<usize> = (0..channel_axis).collect();
    order.extend(channel_axis + 1..rank);
    order.push(channel_axis);
    order
}

fn permuted_shape(shape: &[i64], perm: &[usize]) -> Vec<i64> {
    perm.iter()
        .filter_map(|&idx| shape.get(idx).copied())
        .collect()
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

fn is_atanh(t: &str) -> bool {
    t.contains("atanh")
}
fn is_tan(t: &str) -> bool {
    t.contains("aten.tan.")
}
fn is_sinh(t: &str) -> bool {
    t.contains("aten.sinh.")
}
fn is_cosh(t: &str) -> bool {
    t.contains("aten.cosh.")
}
fn is_asinh(t: &str) -> bool {
    t.contains("aten.asinh.")
}
fn is_asin(t: &str) -> bool {
    t.contains("aten.asin.")
}
fn is_acos(t: &str) -> bool {
    t.contains("aten.acos.")
}
fn is_atan(t: &str) -> bool {
    t.contains("aten.atan.")
}
fn is_acosh(t: &str) -> bool {
    t.contains("aten.acosh.")
}
fn is_elu(t: &str) -> bool {
    t.contains("aten.elu.")
}
fn is_gelu(t: &str) -> bool {
    t.contains("aten.gelu.")
}
fn is_erfinv(t: &str) -> bool {
    t.contains("aten.erfinv.")
}
fn is_batch_norm(t: &str) -> bool {
    t.contains("aten.batch_norm")
        || t.contains("aten.native_batch_norm")
        || t.contains("aten._native_batch_norm_legit_no_training")
        || t.contains("aten._native_batch_norm_legit.no_stats")
}
fn is_group_norm(t: &str) -> bool {
    t.contains("aten.group_norm") || t.contains("aten.native_group_norm")
}
fn is_softmax(t: &str) -> bool {
    t.contains("aten.softmax.") || t.contains("aten._safe_softmax.") || t.contains("aten._softmax.")
}
fn is_log_softmax(t: &str) -> bool {
    t.contains("aten.log_softmax.") || t.contains("aten._log_softmax.")
}
fn is_max_dim(t: &str) -> bool {
    t.contains("aten.max.dim")
}
fn is_min_dim(t: &str) -> bool {
    t.contains("aten.min.dim")
}
fn is_stack(t: &str) -> bool {
    t.contains("stack") && !t.contains("unstack")
}
fn is_unbind(t: &str) -> bool {
    t.contains("unbind")
}

fn is_add_sub_with_alpha(t: &str) -> bool {
    t.contains("add.Tensor") || t.contains("sub.Tensor")
}

fn is_pow_scalar(t: &str) -> bool {
    t.contains("pow.Tensor_Scalar")
}

fn has_quantized_meta(node: &crate::fx_ir::IrNode) -> bool {
    node.meta
        .input_qparams
        .as_ref()
        .is_some_and(|qparams| !qparams.is_empty())
        || node
            .meta
            .output_qparams
            .as_ref()
            .is_some_and(|qparams| !qparams.is_empty())
}

fn pow_scalar_numeric_arg(node: &crate::fx_ir::IrNode) -> Option<f64> {
    node.args
        .get(1)
        .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
}

fn pow_scalar_int_arg(node: &crate::fx_ir::IrNode) -> Option<i64> {
    node.args.get(1).and_then(IrArg::as_int)
}

fn alpha_value(node: &crate::fx_ir::IrNode) -> Option<f64> {
    node.args
        .get(2)
        .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        .or_else(|| {
            node.kwargs
                .get("alpha")
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        })
}

fn elu_alpha(node: &crate::fx_ir::IrNode) -> f64 {
    node.args
        .get(1)
        .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        .or_else(|| {
            node.kwargs
                .get("alpha")
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        })
        .unwrap_or(1.0)
}

fn gelu_approximate(node: &crate::fx_ir::IrNode) -> &str {
    node.kwargs
        .get("approximate")
        .and_then(IrArg::as_str)
        .unwrap_or("none")
}

fn batch_norm_training(node: &crate::fx_ir::IrNode) -> bool {
    node.kwargs
        .get("training")
        .and_then(IrArg::as_bool)
        .or_else(|| node.args.get(5).and_then(IrArg::as_bool))
        .unwrap_or(false)
}

fn batch_norm_eps(node: &crate::fx_ir::IrNode) -> f64 {
    node.kwargs
        .get("eps")
        .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        .or_else(|| {
            node.args
                .iter()
                .rev()
                .find_map(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64)))
        })
        .unwrap_or(1.0e-5)
}

fn preserve_shape_node(graph: &mut IrGraph, node_id: NodeId) {
    let meta = &mut graph.node_mut(node_id).meta;
    meta.mark_preserve_shape_node();
    meta.mark_layout_neutral_flag();
}

fn preserve_identity_layout(graph: &mut IrGraph, node_id: NodeId) {
    preserve_shape_node(graph, node_id);
    graph.node_mut(node_id).meta.mark_identity_dim_order();
}

fn add_full_const(
    graph: &mut IrGraph,
    anchor_id: NodeId,
    name: &str,
    suffix: &str,
    value: f64,
    shape: &[i64],
    dtype: &str,
    meta: &NodeMeta,
) -> NodeId {
    graph.add_anchored_full_float(
        anchor_id,
        format!("{}_{}", name, suffix),
        shape,
        value,
        dtype,
        meta.cloned_with_val(shape.to_vec(), dtype),
    )
}

fn build_polynomial(
    graph: &mut IrGraph,
    anchor_id: NodeId,
    name: &str,
    prefix: &str,
    coefficients: &[f64],
    variable_id: NodeId,
    shape: &[i64],
    dtype: &str,
    meta: &NodeMeta,
) -> Option<NodeId> {
    let first = *coefficients.first()?;
    let mut result = add_full_const(
        graph,
        anchor_id,
        name,
        &format!("{}_c0", prefix),
        first,
        shape,
        dtype,
        meta,
    );
    let mut current_var = variable_id;

    for (idx, coeff) in coefficients.iter().enumerate().skip(1) {
        let coeff_id = add_full_const(
            graph,
            anchor_id,
            name,
            &format!("{}_c{}", prefix, idx),
            *coeff,
            shape,
            dtype,
            meta,
        );
        let term = graph.add_anchored_call_function_with_val(
            anchor_id,
            format!("{}_{}_term{}", name, prefix, idx),
            "aten.mul.Tensor",
            vec![IrArg::Node(current_var), IrArg::Node(coeff_id)],
            meta,
            shape.to_vec(),
            dtype,
        );
        let sum = graph.add_anchored_call_function_with_val(
            anchor_id,
            format!("{}_{}_sum{}", name, prefix, idx),
            "aten.add.Tensor",
            vec![IrArg::Node(result), IrArg::Node(term)],
            meta,
            shape.to_vec(),
            dtype,
        );
        result = sum;

        if idx + 1 < coefficients.len() {
            let next_var = graph.add_anchored_call_function_with_val(
                anchor_id,
                format!("{}_{}_pow{}", name, prefix, idx),
                "aten.mul.Tensor",
                vec![IrArg::Node(current_var), IrArg::Node(current_var)],
                meta,
                shape.to_vec(),
                dtype,
            );
            current_var = next_var;
        }
    }

    Some(result)
}

/// Check if input is DQ and at least one user is Q (quantized envelope → skip atanh).
fn has_quantized_envelope(graph: &IrGraph, id: NodeId) -> bool {
    // Input must be dequantize
    let input_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    if !graph.nodes[input_id]
        .target
        .contains("dequantize_per_tensor")
    {
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
        .any(|n| n.target.contains("quantize_per_tensor") && !n.target.contains("dequantize"))
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
        .find_map(|rule| (rule.matches)(target).then_some(*rule))
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
        matches: is_atanh,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_atanh,
    },
    DecomposeRule {
        matches: is_tan,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_tan,
    },
    DecomposeRule {
        matches: is_sinh,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_sinh,
    },
    DecomposeRule {
        matches: is_cosh,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_cosh,
    },
    DecomposeRule {
        matches: is_asinh,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_asinh,
    },
    DecomposeRule {
        matches: is_asin,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_asin,
    },
    DecomposeRule {
        matches: is_acos,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_acos,
    },
    DecomposeRule {
        matches: is_atan,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_atan,
    },
    DecomposeRule {
        matches: is_acosh,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_acosh,
    },
    DecomposeRule {
        matches: is_elu,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_elu,
    },
    DecomposeRule {
        matches: is_gelu,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_gelu,
    },
    DecomposeRule {
        matches: is_erfinv,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_erfinv,
    },
    DecomposeRule {
        matches: is_pow_scalar,
        mode: DecomposeMode::SkipQuantizedEnvelope,
        decompose: decompose_pow_scalar,
    },
    DecomposeRule {
        matches: is_group_norm,
        mode: DecomposeMode::Always,
        decompose: decompose_group_norm,
    },
    DecomposeRule {
        matches: is_batch_norm,
        mode: DecomposeMode::Always,
        decompose: decompose_batch_norm,
    },
    DecomposeRule {
        matches: is_log_softmax,
        mode: DecomposeMode::Always,
        decompose: decompose_log_softmax_wrapper,
    },
    DecomposeRule {
        matches: is_softmax,
        mode: DecomposeMode::Always,
        decompose: decompose_softmax,
    },
    DecomposeRule {
        matches: is_max_dim,
        mode: DecomposeMode::Always,
        decompose: decompose_dim_extrema,
    },
    DecomposeRule {
        matches: is_min_dim,
        mode: DecomposeMode::Always,
        decompose: decompose_dim_extrema,
    },
    DecomposeRule {
        matches: is_stack,
        mode: DecomposeMode::Always,
        decompose: decompose_stack,
    },
    DecomposeRule {
        matches: is_unbind,
        mode: DecomposeMode::Always,
        decompose: decompose_unbind,
    },
    DecomposeRule {
        matches: is_add_sub_with_alpha,
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
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let target = graph.nodes[id].target.clone();

            if try_decompose_rule(graph, id, &target) {
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeOps] decomposed atanh/tan/sinh/cosh/asinh/acosh/elu/gelu/batch-norm/stack/unbind/add-sub-alpha/pow ops"
            ));
        }
        modified
    }
}

fn decompose_add_sub_alpha(graph: &mut IrGraph, id: NodeId) -> bool {
    let Some(rhs_id) = graph.nodes[id].args.get(1).and_then(|arg| arg.as_node()) else {
        return false;
    };
    let Some(alpha) = alpha_value(&graph.nodes[id]) else {
        return false;
    };
    if (alpha - 1.0).abs() <= 1e-10 {
        return false;
    }

    let rhs_shape = match graph.nodes[rhs_id].meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let rhs_dtype = graph.nodes[rhs_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let target = if graph.nodes[id].target.contains("sub.Tensor") {
        "aten.sub.Tensor".to_string()
    } else {
        "aten.add.Tensor".to_string()
    };
    let meta = graph.nodes[id].meta.clone();
    let scaled_meta = graph.nodes[rhs_id]
        .meta
        .cloned_with_val(rhs_shape.clone(), &rhs_dtype);

    let alpha_full = graph.add_anchored_full_float(
        id,
        format!("{}_alpha_full", name),
        &rhs_shape,
        alpha,
        &rhs_dtype,
        scaled_meta.clone(),
    );

    let scaled_rhs = graph.add_anchored_call_function(
        id,
        format!("{}_alpha_mul", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(rhs_id), IrArg::Node(alpha_full)],
        scaled_meta,
    );

    let mut add_args = graph.nodes[id].args.clone();
    if add_args.len() < 2 {
        return false;
    }
    add_args[1] = IrArg::Node(scaled_rhs);
    add_args.truncate(2);
    let result = graph.add_anchored_call_function(
        id,
        format!("{}_alpha_result", name),
        target,
        add_args,
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

fn decompose_pow_scalar(graph: &mut IrGraph, id: NodeId) -> bool {
    if has_quantized_meta(&graph.nodes[id]) {
        return false;
    }

    let Some(input_id) = graph.nodes[id].args.first().and_then(IrArg::as_node) else {
        return false;
    };

    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();

    if pow_scalar_numeric_arg(&graph.nodes[id]).is_some_and(|exp| exp == 0.0) {
        let shape = meta
            .val_shape()
            .map(|shape| shape.to_vec())
            .unwrap_or_default();
        let dtype = meta.val_dtype().unwrap_or("torch.float32").to_string();
        let metas = MetaFactory::new(&meta, &dtype);
        let one = graph.add_anchored_full_float(
            id,
            format!("{}_one", name),
            &shape,
            1.0,
            &dtype,
            metas.tensor(&shape),
        );
        let eq = graph.add_anchored_call_function(
            id,
            format!("{}_self_eq", name),
            "aten.eq.Tensor",
            vec![IrArg::Node(input_id), IrArg::Node(input_id)],
            metas.bool(&shape),
        );
        let ones = graph.add_anchored_call_function(
            id,
            format!("{}_ones", name),
            "aten.where.self",
            vec![IrArg::Node(eq), IrArg::Node(one), IrArg::Node(one)],
            meta,
        );
        graph.replace_and_erase(id, ones);
        return true;
    }

    let Some(mut exp) = pow_scalar_int_arg(&graph.nodes[id]) else {
        return false;
    };

    let mut base_id = input_id;
    if exp < 0 {
        let reciprocal = graph.add_anchored_call_function(
            id,
            format!("{}_reciprocal", name),
            "aten.reciprocal.default",
            vec![IrArg::Node(input_id)],
            meta.clone(),
        );
        base_id = reciprocal;
        exp = -exp;
    }

    let mut result_id = base_id;
    for idx in 1..exp {
        let mul = graph.add_anchored_call_function(
            id,
            format!("{}_mul{}", name, idx),
            "aten.mul.Tensor",
            vec![IrArg::Node(result_id), IrArg::Node(base_id)],
            meta.clone(),
        );
        result_id = mul;
    }

    graph.replace_and_erase(id, result_id);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    #[test]
    fn decompose_max_dim_rewrites_to_amax_and_squeeze() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            meta(&[2, 3, 4], "torch.float32"),
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
            meta(&[2, 4], "torch.float32"),
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
            meta(&[2, 3, 4], "torch.float32"),
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
            meta(&[2, 4], "torch.float32"),
        );
        let indices = graph.add_node(
            NodeOp::CallFunction,
            "getitem_1".to_string(),
            "getitem".to_string(),
            vec![IrArg::Node(max), IrArg::Int(1)],
            meta(&[2, 4], "torch.int64"),
        );
        let _use_indices = graph.add_node(
            NodeOp::CallFunction,
            "use_indices".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(indices), IrArg::Node(indices)],
            meta(&[2, 4], "torch.int64"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let atan = graph.add_node(
            NodeOp::CallFunction,
            "atan".to_string(),
            "aten.atan.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let acos = graph.add_node(
            NodeOp::CallFunction,
            "acos".to_string(),
            "aten.acos.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3, 4], "torch.float32"),
        );
        let softmax = graph.add_node(
            NodeOp::CallFunction,
            "softmax".to_string(),
            "aten._softmax.default".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            meta(&[2, 3, 4], "torch.float32"),
        );

        assert!(IrDecomposeOpsPass.run(&mut graph));
        assert!(!graph.nodes[softmax].alive);
        assert!(graph.nodes.iter().any(|node| node.alive
            && node.name == "softmax_result"
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
            meta(&[2, 3, 4], "torch.float32"),
        );
        let log_softmax = graph.add_node(
            NodeOp::CallFunction,
            "log_softmax".to_string(),
            "aten._log_softmax.default".to_string(),
            vec![IrArg::Node(x), IrArg::Int(1)],
            meta(&[2, 3, 4], "torch.float32"),
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
            meta(&[1, 4, 5], "torch.float32"),
        );
        let y = graph.add_node(
            NodeOp::Placeholder,
            "y".to_string(),
            "y".to_string(),
            vec![],
            meta(&[4, 1], "torch.float32"),
        );
        let add = graph.add_node(
            NodeOp::CallFunction,
            "add".to_string(),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(x), IrArg::Node(y), IrArg::Float(1.5)],
            meta(&[1, 4, 5], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let tan = graph.add_node(
            NodeOp::CallFunction,
            "tan".to_string(),
            "aten.tan.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[1, 2, 3, 7], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(x), IrArg::Float(0.0)],
            meta(&[1, 2, 3, 7], "torch.float32"),
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
            meta(&[1, 2, 3, 6], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(x), IrArg::Int(3)],
            meta(&[1, 2, 3, 6], "torch.float32"),
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
            meta(&[2, 3], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "x_dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
        );
        let tan = graph.add_node(
            NodeOp::CallFunction,
            "tan".to_string(),
            "aten.tan.default".to_string(),
            vec![IrArg::Node(dq)],
            meta(&[2, 3], "torch.float32"),
        );
        let q = graph.add_node(
            NodeOp::CallFunction,
            "tan_q".to_string(),
            "quantized_decomposed.quantize_per_tensor.default".to_string(),
            vec![IrArg::Node(tan)],
            meta(&[2, 3], "torch.int8"),
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
            meta(&[1, 2, 3, 6], "torch.int8"),
        );
        let dq = graph.add_node(
            NodeOp::CallFunction,
            "x_dq".to_string(),
            "quantized_decomposed.dequantize_per_tensor.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[1, 2, 3, 6], "torch.float32"),
        );
        let pow = graph.add_node(
            NodeOp::CallFunction,
            "pow".to_string(),
            "aten.pow.Tensor_Scalar".to_string(),
            vec![IrArg::Node(dq), IrArg::Int(3)],
            meta(&[1, 2, 3, 6], "torch.float32"),
        );
        let q = graph.add_node(
            NodeOp::CallFunction,
            "pow_q".to_string(),
            "quantized_decomposed.quantize_per_tensor.default".to_string(),
            vec![IrArg::Node(pow)],
            meta(&[1, 2, 3, 6], "torch.int8"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let sinh = graph.add_node(
            NodeOp::CallFunction,
            "sinh".to_string(),
            "aten.sinh.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let cosh = graph.add_node(
            NodeOp::CallFunction,
            "cosh".to_string(),
            "aten.cosh.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let asinh = graph.add_node(
            NodeOp::CallFunction,
            "asinh".to_string(),
            "aten.asinh.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let acosh = graph.add_node(
            NodeOp::CallFunction,
            "acosh".to_string(),
            "aten.acosh.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let elu = graph.add_node(
            NodeOp::CallFunction,
            "elu".to_string(),
            "aten.elu.default".to_string(),
            vec![IrArg::Node(x), IrArg::Float(2.0)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let gelu = graph.add_node(
            NodeOp::CallFunction,
            "gelu".to_string(),
            "aten.gelu.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[2, 3], "torch.float32"),
        );
        let gelu = graph.add_node(
            NodeOp::CallFunction,
            "gelu".to_string(),
            "aten.gelu.default".to_string(),
            vec![IrArg::Node(x)],
            meta(&[2, 3], "torch.float32"),
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
            meta(&[1, 4, 5, 6], "torch.float32"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            vec![],
            meta(&[4], "torch.float32"),
        );
        let bias = graph.add_node(
            NodeOp::Placeholder,
            "bias".to_string(),
            "bias".to_string(),
            vec![],
            meta(&[4], "torch.float32"),
        );
        let mean = graph.add_node(
            NodeOp::Placeholder,
            "mean".to_string(),
            "mean".to_string(),
            vec![],
            meta(&[4], "torch.float32"),
        );
        let var = graph.add_node(
            NodeOp::Placeholder,
            "var".to_string(),
            "var".to_string(),
            vec![],
            meta(&[4], "torch.float32"),
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
            meta(&[1, 4, 5, 6], "torch.float32"),
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
        assert_eq!(mean_view.meta.val_shape().unwrap(), &[1, 5, 6, 4]);
        assert_eq!(
            mean_view.meta.extra_bool("arm_preserve_shape_node"),
            Some(true)
        );
        assert_eq!(mean_view.meta.extra_bool("arm_layout_neutral"), Some(true));
    }

    #[test]
    fn decompose_group_norm_rewrites_to_reduction_chain() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            meta(&[1, 6, 8], "torch.float32"),
        );
        let weight = graph.add_node(
            NodeOp::Placeholder,
            "weight".to_string(),
            "weight".to_string(),
            vec![],
            meta(&[6], "torch.float32"),
        );
        let bias = graph.add_node(
            NodeOp::Placeholder,
            "bias".to_string(),
            "bias".to_string(),
            vec![],
            meta(&[6], "torch.float32"),
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
            meta(&[1, 6, 8], "torch.float32"),
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
        assert_eq!(
            grouped.meta.extra_bool("arm_preserve_shape_node"),
            Some(true)
        );
        assert_eq!(grouped.meta.extra_bool("arm_layout_neutral"), Some(true));
    }
}

fn decompose_dim_extrema(graph: &mut IrGraph, id: NodeId) -> bool {
    let node_target = graph.nodes[id].target.clone();
    let node_name = graph.nodes[id].name.clone();
    let reduce_target = if is_max_dim(&node_target) {
        "aten.amax.default"
    } else if is_min_dim(&node_target) {
        "aten.amin.default"
    } else {
        return false;
    };

    let input_id = match graph.nodes[id].first_node_arg() {
        Some(input_id) => input_id,
        None => return false,
    };
    let dim = match graph.nodes[id].args.get(1).and_then(IrArg::as_int) {
        Some(dim) => dim,
        None => return false,
    };
    let input_shape = match graph.nodes[input_id].meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let rank = input_shape.len() as i64;
    let dim = if dim < 0 { dim + rank } else { dim };
    if dim < 0 || dim >= rank {
        return false;
    }
    let dim = dim as usize;

    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|n| n.alive && n.target.contains("getitem"))
        .filter(|n| n.args.first().and_then(IrArg::as_node) == Some(id))
        .map(|n| (n.id, n.args.get(1).and_then(IrArg::as_int).unwrap_or(0)))
        .collect();

    if getitem_users.is_empty() {
        return false;
    }

    if graph.nodes.iter().any(|n| {
        n.alive && n.id != id && !n.target.contains("getitem") && graph.users(id).contains(&n.id)
    }) {
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
        .or_else(|| input_meta.val_dtype())
        .unwrap_or("torch.float32")
        .to_string();

    let mut keepdim_shape = input_shape;
    keepdim_shape[dim] = 1;

    let mut reduce_meta = value_meta.clone();
    reduce_meta.val_dim_order = input_meta.val_dim_order.clone();
    reduce_meta.tosa_dim_order = input_meta.tosa_dim_order.clone();
    reduce_meta.original_dim_orders = input_meta.original_dim_orders.clone();
    reduce_meta.set_val(FakeTensorMeta {
        shape: keepdim_shape,
        dtype: dtype.clone(),
    });

    let reduce_id = graph.add_node(
        NodeOp::CallFunction,
        format!(
            "{}_{}",
            node_name,
            if is_max_dim(&node_target) {
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
    graph.anchor_to(reduce_id, id);
    graph.copy_meta_from(reduce_id, value_getitem);

    let squeeze_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", node_name),
        "aten.squeeze_copy.dims".to_string(),
        vec![IrArg::Node(reduce_id), i64_list(&[dim as i64])],
        value_meta,
    );
    graph.anchor_after(squeeze_id, reduce_id);
    graph.copy_meta_from(squeeze_id, value_getitem);

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
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(x_id) => x_id,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(shape) => shape.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let dim = graph.nodes[id]
        .args
        .get(1)
        .and_then(IrArg::as_int)
        .or_else(|| graph.nodes[id].kwargs.get("dim").and_then(IrArg::as_int))
        .unwrap_or(-1);
    let rank = shape.len() as i64;
    let dim = if dim < 0 { dim + rank } else { dim };
    if dim < 0 || dim >= rank {
        return false;
    }
    let dim = dim as usize;

    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let mut reduced_shape = shape.clone();
    reduced_shape[dim] = 1;
    let sum_shape = reduced_shape.clone();

    let reduce = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_amax", name),
        "aten.amax.default",
        vec![IrArg::Node(x_id), IrArg::Int(dim as i64), IrArg::Bool(true)],
        &meta,
        reduced_shape.clone(),
        &dtype,
    );

    let shifted = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_shifted", name),
        "aten.sub.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(reduce)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let exp = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_exp", name),
        "aten.exp.default",
        vec![IrArg::Node(shifted)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let sum = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_sum", name),
        "aten.sum.dim_IntList",
        vec![IrArg::Node(exp), i64_list(&[dim as i64]), IrArg::Bool(true)],
        &meta,
        sum_shape.clone(),
        &dtype,
    );

    let recip = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_recip", name),
        "aten.reciprocal.default",
        vec![IrArg::Node(sum)],
        &meta,
        sum_shape,
        &dtype,
    );

    let softmax = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_softmax", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(exp), IrArg::Node(recip)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let result = if log_output {
        graph.add_anchored_call_function(
            id,
            format!("{}_result", name),
            "aten.log.default",
            vec![IrArg::Node(softmax)],
            meta,
        )
    } else {
        graph.add_anchored_call_function(
            id,
            format!("{}_result", name),
            "aten.clone.default",
            vec![IrArg::Node(softmax)],
            meta,
        )
    };

    graph.replace_and_erase(id, result);
    true
}

/// atanh(x) = 0.5 * log((1+x) / (1-x))
fn decompose_atanh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();

    let one = graph.add_anchored_full_float(
        id,
        format!("{}_one", name),
        &shape,
        1.0,
        &dtype,
        meta.cloned_with_val(shape.clone(), &dtype),
    );

    let half = graph.add_anchored_full_float(
        id,
        format!("{}_half", name),
        &shape,
        0.5,
        &dtype,
        meta.cloned_with_val(shape.clone(), &dtype),
    );

    let one_plus = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_1px", name),
        "aten.add.Tensor",
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let one_minus = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_1mx", name),
        "aten.sub.Tensor",
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let ratio = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_ratio", name),
        "aten.div.Tensor",
        vec![IrArg::Node(one_plus), IrArg::Node(one_minus)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let log_r = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_log", name),
        "aten.log.default",
        vec![IrArg::Node(ratio)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(half), IrArg::Node(log_r)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// tan(x) = sin(x) / cos(x)
fn decompose_tan(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();

    let sin = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_sin", name),
        "aten.sin.default",
        vec![IrArg::Node(x_id)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let cos = graph.add_anchored_call_function_with_val(
        id,
        format!("{}_cos", name),
        "aten.cos.default",
        vec![IrArg::Node(x_id)],
        &meta,
        shape.clone(),
        &dtype,
    );

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(sin), IrArg::Node(cos)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// atan(x) via the same Padé approximation used in the Python pass.
fn decompose_atan(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);
    let bool_meta = metas.bool(&shape);
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let bool_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            bool_meta.clone(),
        )
    };

    let one = add_full_const(graph, id, &name, "one", 1.0, &shape, &dtype, &meta);
    let zero = add_full_const(graph, id, &name, "zero", 0.0, &shape, &dtype, &meta);
    let half_pi = add_full_const(
        graph, id, &name, "half_pi", FRAC_PI_2, &shape, &dtype, &meta,
    );

    let abs_x = tensor_op(graph, "abs", "aten.abs.default", vec![IrArg::Node(x_id)]);
    let mask_hi = bool_op(
        graph,
        "mask_hi",
        "aten.gt.Tensor",
        vec![IrArg::Node(abs_x), IrArg::Node(one)],
    );
    let inv_x = tensor_op(
        graph,
        "inv",
        "aten.reciprocal.default",
        vec![IrArg::Node(abs_x)],
    );
    let z = tensor_op(
        graph,
        "z",
        "aten.where.self",
        vec![IrArg::Node(mask_hi), IrArg::Node(inv_x), IrArg::Node(abs_x)],
    );
    let z2 = tensor_op(
        graph,
        "z2",
        "aten.mul.Tensor",
        vec![IrArg::Node(z), IrArg::Node(z)],
    );
    let num = match build_polynomial(
        graph,
        id,
        &name,
        "num",
        &[1.0, 0.3529666667, -0.0287666667],
        z2,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node) => node,
        None => return false,
    };
    let den = match build_polynomial(
        graph,
        id,
        &name,
        "den",
        &[1.0, 0.6863],
        z2,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node) => node,
        None => return false,
    };
    let inv_den = tensor_op(
        graph,
        "inv_den",
        "aten.reciprocal.default",
        vec![IrArg::Node(den)],
    );
    let prod = tensor_op(
        graph,
        "prod",
        "aten.mul.Tensor",
        vec![IrArg::Node(num), IrArg::Node(inv_den)],
    );
    let atan_z = tensor_op(
        graph,
        "atan_z",
        "aten.mul.Tensor",
        vec![IrArg::Node(z), IrArg::Node(prod)],
    );
    let diff = tensor_op(
        graph,
        "diff",
        "aten.sub.Tensor",
        vec![IrArg::Node(half_pi), IrArg::Node(atan_z)],
    );
    let atan_abs = tensor_op(
        graph,
        "atan_abs",
        "aten.where.self",
        vec![IrArg::Node(mask_hi), IrArg::Node(diff), IrArg::Node(atan_z)],
    );
    let mask_pos = bool_op(
        graph,
        "mask_pos",
        "aten.gt.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
    );
    let neg = tensor_op(
        graph,
        "neg",
        "aten.neg.default",
        vec![IrArg::Node(atan_abs)],
    );

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.where.self",
        vec![
            IrArg::Node(mask_pos),
            IrArg::Node(atan_abs),
            IrArg::Node(neg),
        ],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// sinh(x) = 0.5 * (exp(x) - exp(-x))
fn decompose_sinh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };

    let half = add_full_const(graph, id, &name, "half", 0.5, &shape, &dtype, &meta);
    let exp_pos = tensor_op(graph, "exp_pos", "aten.exp.default", vec![IrArg::Node(x_id)]);
    let neg = tensor_op(graph, "neg", "aten.neg.default", vec![IrArg::Node(x_id)]);
    let exp_neg = tensor_op(graph, "exp_neg", "aten.exp.default", vec![IrArg::Node(neg)]);
    let diff = tensor_op(
        graph,
        "diff",
        "aten.sub.Tensor",
        vec![IrArg::Node(exp_pos), IrArg::Node(exp_neg)],
    );

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(diff), IrArg::Node(half)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// cosh(x) = 0.5 * (exp(x) + exp(-x))
fn decompose_cosh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };

    let half = add_full_const(graph, id, &name, "half", 0.5, &shape, &dtype, &meta);
    let exp_pos = tensor_op(graph, "exp_pos", "aten.exp.default", vec![IrArg::Node(x_id)]);
    let neg = tensor_op(graph, "neg", "aten.neg.default", vec![IrArg::Node(x_id)]);
    let exp_neg = tensor_op(graph, "exp_neg", "aten.exp.default", vec![IrArg::Node(neg)]);
    let sum = tensor_op(
        graph,
        "sum",
        "aten.add.Tensor",
        vec![IrArg::Node(exp_pos), IrArg::Node(exp_neg)],
    );

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.mul.Tensor",
        vec![IrArg::Node(sum), IrArg::Node(half)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// asinh(x) = log(x + sqrt(x^2 + 1))
fn decompose_asinh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };

    let one = add_full_const(graph, id, &name, "one", 1.0, &shape, &dtype, &meta);
    let square = tensor_op(
        graph,
        "square",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(x_id)],
    );
    let radicand = tensor_op(
        graph,
        "radicand",
        "aten.add.Tensor",
        vec![IrArg::Node(square), IrArg::Node(one)],
    );
    let sqrt = tensor_op(graph, "sqrt", "aten.sqrt.default", vec![IrArg::Node(radicand)]);
    let add = tensor_op(
        graph,
        "sum",
        "aten.add.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(sqrt)],
    );

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.log.default",
        vec![IrArg::Node(add)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// acosh(x) = log(x + sqrt((x - 1) * (x + 1)))
fn decompose_acosh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };

    let one = add_full_const(graph, id, &name, "one", 1.0, &shape, &dtype, &meta);
    let minus = tensor_op(
        graph,
        "minus",
        "aten.sub.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(one)],
    );
    let plus = tensor_op(
        graph,
        "plus",
        "aten.add.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(one)],
    );
    let radicand = tensor_op(
        graph,
        "radicand",
        "aten.mul.Tensor",
        vec![IrArg::Node(minus), IrArg::Node(plus)],
    );
    let sqrt = tensor_op(graph, "sqrt", "aten.sqrt.default", vec![IrArg::Node(radicand)]);
    let add = tensor_op(
        graph,
        "sum",
        "aten.add.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(sqrt)],
    );

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.log.default",
        vec![IrArg::Node(add)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// asin/acos decomposition matching the Python piecewise rational approximation.
fn decompose_asin_or_acos(graph: &mut IrGraph, id: NodeId, is_acos_op: bool) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);
    let bool_meta = metas.bool(&shape);
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let bool_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            bool_meta.clone(),
        )
    };

    let p_coefficients = [
        1.6666667163e-01,
        -3.2556581497e-01,
        2.0121252537e-01,
        -4.0055535734e-02,
        7.9153501429e-04,
    ];
    let q_coefficients = [1.0, -2.4033949375e00, 2.0209457874e00, -6.8828397989e-01];

    let one = add_full_const(graph, id, &name, "one", 1.0, &shape, &dtype, &meta);
    let half = add_full_const(graph, id, &name, "half", 0.5, &shape, &dtype, &meta);
    let neg_half = add_full_const(graph, id, &name, "neg_half", -0.5, &shape, &dtype, &meta);
    let two = add_full_const(graph, id, &name, "two", 2.0, &shape, &dtype, &meta);
    let half_pi = add_full_const(
        graph, id, &name, "half_pi", FRAC_PI_2, &shape, &dtype, &meta,
    );
    let zero = add_full_const(graph, id, &name, "zero", 0.0, &shape, &dtype, &meta);

    let x_abs = tensor_op(graph, "abs", "aten.abs.default", vec![IrArg::Node(x_id)]);
    let y = tensor_op(
        graph,
        "y",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(x_abs)],
    );
    let x3 = tensor_op(
        graph,
        "x3",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(y)],
    );

    let p_small = match build_polynomial(
        graph,
        id,
        &name,
        "p_small",
        &p_coefficients,
        x_abs,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node_id) => node_id,
        None => return false,
    };
    let q_small = match build_polynomial(
        graph,
        id,
        &name,
        "q_small",
        &q_coefficients,
        x_abs,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_small = tensor_op(
        graph,
        "numer_small",
        "aten.mul.Tensor",
        vec![IrArg::Node(x3), IrArg::Node(p_small)],
    );
    let r_small = tensor_op(
        graph,
        "r_small",
        "aten.div.Tensor",
        vec![IrArg::Node(numer_small), IrArg::Node(q_small)],
    );
    let asin_small = tensor_op(
        graph,
        "asin_small",
        "aten.add.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(r_small)],
    );
    let tmp = tensor_op(
        graph,
        "tmp",
        "aten.sub.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(one)],
    );
    let z = tensor_op(
        graph,
        "z",
        "aten.mul.Tensor",
        vec![IrArg::Node(tmp), IrArg::Node(neg_half)],
    );
    let s = tensor_op(graph, "s", "aten.sqrt.default", vec![IrArg::Node(z)]);
    let s2 = tensor_op(
        graph,
        "s2",
        "aten.mul.Tensor",
        vec![IrArg::Node(s), IrArg::Node(s)],
    );
    let s3 = tensor_op(
        graph,
        "s3",
        "aten.mul.Tensor",
        vec![IrArg::Node(s2), IrArg::Node(s)],
    );

    let pz = match build_polynomial(
        graph,
        id,
        &name,
        "p_large",
        &p_coefficients,
        z,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node_id) => node_id,
        None => return false,
    };
    let qz = match build_polynomial(
        graph,
        id,
        &name,
        "q_large",
        &q_coefficients,
        z,
        &shape,
        &dtype,
        &meta,
    ) {
        Some(node_id) => node_id,
        None => return false,
    };

    let numer_large = tensor_op(
        graph,
        "numer_large",
        "aten.mul.Tensor",
        vec![IrArg::Node(s3), IrArg::Node(pz)],
    );
    let r_large = tensor_op(
        graph,
        "r_large",
        "aten.div.Tensor",
        vec![IrArg::Node(numer_large), IrArg::Node(qz)],
    );
    let t1 = tensor_op(
        graph,
        "t1",
        "aten.add.Tensor",
        vec![IrArg::Node(s), IrArg::Node(r_large)],
    );
    let t2 = tensor_op(
        graph,
        "t2",
        "aten.mul.Tensor",
        vec![IrArg::Node(t1), IrArg::Node(two)],
    );
    let diff = tensor_op(
        graph,
        "diff",
        "aten.sub.Tensor",
        vec![IrArg::Node(t2), IrArg::Node(half_pi)],
    );
    let asin_large = tensor_op(
        graph,
        "asin_large",
        "aten.neg.default",
        vec![IrArg::Node(diff)],
    );
    let mask_large = bool_op(
        graph,
        "mask_large",
        "aten.gt.Tensor",
        vec![IrArg::Node(x_abs), IrArg::Node(half)],
    );
    let asin_unsigned = tensor_op(
        graph,
        "asin_unsigned",
        "aten.where.self",
        vec![
            IrArg::Node(mask_large),
            IrArg::Node(asin_large),
            IrArg::Node(asin_small),
        ],
    );
    let negated_asin = tensor_op(
        graph,
        "negated_asin",
        "aten.neg.default",
        vec![IrArg::Node(asin_unsigned)],
    );
    let mask_neg = bool_op(
        graph,
        "mask_neg",
        "aten.lt.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
    );
    let asin = tensor_op(
        graph,
        "asin",
        "aten.where.self",
        vec![
            IrArg::Node(mask_neg),
            IrArg::Node(negated_asin),
            IrArg::Node(asin_unsigned),
        ],
    );

    let result = if is_acos_op {
        let acos_small = tensor_op(
            graph,
            "acos_small",
            "aten.sub.Tensor",
            vec![IrArg::Node(half_pi), IrArg::Node(asin)],
        );
        let mask_gt_half = bool_op(
            graph,
            "mask_gt_half",
            "aten.gt.Tensor",
            vec![IrArg::Node(x_id), IrArg::Node(half)],
        );

        let acos = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_result", name),
            "aten.where.self".to_string(),
            vec![
                IrArg::Node(mask_gt_half),
                IrArg::Node(t2),
                IrArg::Node(acos_small),
            ],
            meta,
        );
        graph.anchor_to(acos, id);
        acos
    } else {
        graph.add_anchored_call_function(
            id,
            format!("{}_result", name),
            "aten.clone.default",
            vec![IrArg::Node(asin)],
            meta,
        )
    };

    graph.replace_and_erase(id, result);
    true
}

/// elu(x, alpha) = where(x >= 0, x, alpha * expm1(x))
fn decompose_elu(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let alpha = elu_alpha(&graph.nodes[id]);
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let bool_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            metas.bool(&shape),
        )
    };

    let zero = graph.add_anchored_full_float(
        id,
        format!("{}_zero", name),
        &shape,
        0.0,
        &dtype,
        metas.tensor(&shape),
    );
    let alpha_full = graph.add_anchored_full_float(
        id,
        format!("{}_alpha", name),
        &shape,
        alpha,
        &dtype,
        metas.tensor(&shape),
    );
    let expm1 = tensor_op(graph, "expm1", "aten.expm1.default", vec![IrArg::Node(x_id)]);
    let scaled = tensor_op(
        graph,
        "scaled",
        "aten.mul.Tensor",
        vec![IrArg::Node(expm1), IrArg::Node(alpha_full)],
    );
    let ge = bool_op(
        graph,
        "ge",
        "aten.ge.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
    );

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(ge), IrArg::Node(x_id), IrArg::Node(scaled)],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// gelu(x) decomposition matching the Python pass for approximate="none"/"tanh".
fn decompose_gelu(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let rank_ones = vec![1_i64; shape.len()];
    let approximate = gelu_approximate(&graph.nodes[id]).to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);
    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let scalar_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            rank_ones.clone(),
            &dtype,
        )
    };

    let full = |graph: &mut IrGraph, suffix: &str, value: f64| {
        graph.add_anchored_full_float(
            id,
            format!("{}_{}", name, suffix),
            &rank_ones,
            value,
            &dtype,
            metas.tensor(&rank_ones),
        )
    };

    let half = full(graph, "half", 0.5);
    let one = full(graph, "one", 1.0);

    let result = match approximate.as_str() {
        "none" => {
            let sqrt_half = full(graph, "sqrt_half", std::f64::consts::FRAC_1_SQRT_2);
            let scaled = tensor_op(
                graph,
                "scaled",
                "aten.mul.Tensor",
                vec![IrArg::Node(x_id), IrArg::Node(sqrt_half)],
            );
            let erf = tensor_op(graph, "erf", "aten.erf.default", vec![IrArg::Node(scaled)]);
            let plus_one = tensor_op(
                graph,
                "plus_one",
                "aten.add.Tensor",
                vec![IrArg::Node(erf), IrArg::Node(one)],
            );
            let weighted = tensor_op(
                graph,
                "weighted",
                "aten.mul.Tensor",
                vec![IrArg::Node(plus_one), IrArg::Node(half)],
            );

            graph.add_node(
                NodeOp::CallFunction,
                format!("{}_result", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(weighted)],
                meta,
            )
        }
        "tanh" => {
            let sqrt_two = full(graph, "sqrt_two", std::f64::consts::SQRT_2);
            let two_over_sqrt_pi = full(graph, "two_over_sqrt_pi", 1.128_379_167_095_512_6);
            let cube_coeff = full(graph, "cube_coeff", 0.044_715);

            let sqrt_mul = scalar_op(
                graph,
                "sqrt_mul",
                "aten.mul.Tensor",
                vec![IrArg::Node(sqrt_two), IrArg::Node(two_over_sqrt_pi)],
            );
            let sqrt_2_pi = scalar_op(
                graph,
                "sqrt_2_pi",
                "aten.mul.Tensor",
                vec![IrArg::Node(sqrt_mul), IrArg::Node(half)],
            );
            let square = tensor_op(
                graph,
                "square",
                "aten.mul.Tensor",
                vec![IrArg::Node(x_id), IrArg::Node(x_id)],
            );
            let cube = tensor_op(
                graph,
                "cube",
                "aten.mul.Tensor",
                vec![IrArg::Node(square), IrArg::Node(x_id)],
            );
            let cubic_term = tensor_op(
                graph,
                "cubic_term",
                "aten.mul.Tensor",
                vec![IrArg::Node(cube), IrArg::Node(cube_coeff)],
            );
            let inner = tensor_op(
                graph,
                "inner",
                "aten.add.Tensor",
                vec![IrArg::Node(x_id), IrArg::Node(cubic_term)],
            );
            let scaled = tensor_op(
                graph,
                "scaled",
                "aten.mul.Tensor",
                vec![IrArg::Node(inner), IrArg::Node(sqrt_2_pi)],
            );
            let tanh = tensor_op(graph, "tanh", "aten.tanh.default", vec![IrArg::Node(scaled)]);
            let plus_one = tensor_op(
                graph,
                "plus_one",
                "aten.add.Tensor",
                vec![IrArg::Node(tanh), IrArg::Node(one)],
            );
            let first_mul = tensor_op(
                graph,
                "first_mul",
                "aten.mul.Tensor",
                vec![IrArg::Node(x_id), IrArg::Node(plus_one)],
            );

            graph.add_node(
                NodeOp::CallFunction,
                format!("{}_result", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(first_mul), IrArg::Node(half)],
                meta,
            )
        }
        _ => return false,
    };

    graph.replace_and_erase(id, result);
    true
}

/// erfinv(x) decomposition matching the Python pass.
fn decompose_erfinv(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let rank_ones = vec![1_i64; shape.len()];
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);

    let full = |graph: &mut IrGraph, suffix: &str, value: f64| {
        graph.add_anchored_full_float(
            id,
            format!("{}_{}", name, suffix),
            &rank_ones,
            value,
            &dtype,
            metas.tensor(&rank_ones),
        )
    };

    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let bool_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            metas.bool(&shape),
        )
    };

    let central_range = full(graph, "central_range", 0.7);
    let a0 = full(graph, "a0", 0.886_226_899);
    let a1 = full(graph, "a1", -1.645_349_621);
    let a2 = full(graph, "a2", 0.914_624_893);
    let a3 = full(graph, "a3", -0.140_543_331);
    let b0 = full(graph, "b0", -2.118_377_725);
    let b1 = full(graph, "b1", 1.442_710_462);
    let b2 = full(graph, "b2", -0.329_097_515);
    let b3 = full(graph, "b3", 0.012_229_801);
    let c0 = full(graph, "c0", -1.970_840_454);
    let c1 = full(graph, "c1", -1.624_906_493);
    let c2 = full(graph, "c2", 3.429_567_803);
    let c3 = full(graph, "c3", 1.641_345_311);
    let d0 = full(graph, "d0", 3.543_889_2);
    let d1 = full(graph, "d1", 1.637_067_8);
    let half = full(graph, "half", 0.5);
    let one = full(graph, "one", 1.0);
    let zeros = full(graph, "zeros", 0.0);
    let neg_ones = full(graph, "neg_ones", -1.0);
    let refine_max_abs = full(graph, "refine_max_abs", 0.95);
    let deriv_eps = full(graph, "deriv_eps", 1.0e-3);
    let corr_max = full(graph, "corr_max", 0.5);
    let two_over_sqrt_pi = full(graph, "two_over_sqrt_pi", 1.128_379_167_095_512_6);
    let nan = full(graph, "nan", f64::NAN);
    let pos_inf = full(graph, "pos_inf", f64::INFINITY);
    let neg_inf = full(graph, "neg_inf", f64::NEG_INFINITY);
    graph.nodes[pos_inf]
        .meta
        .set_extra("arm_preserve_inf_literal", MetaValue::Bool(true));
    graph.nodes[neg_inf]
        .meta
        .set_extra("arm_preserve_inf_literal", MetaValue::Bool(true));

    let x_lt0 = bool_op(
        graph,
        "x_lt0",
        "aten.lt.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(zeros)],
    );

    let s = tensor_op(
        graph,
        "sign",
        "aten.where.self",
        vec![IrArg::Node(x_lt0), IrArg::Node(neg_ones), IrArg::Node(one)],
    );

    let a = tensor_op(
        graph,
        "abs",
        "aten.abs.default",
        vec![IrArg::Node(x_id)],
    );

    let abs_gt1 = bool_op(
        graph,
        "abs_gt1",
        "aten.gt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(one)],
    );

    let abs_eq1 = bool_op(
        graph,
        "abs_eq1",
        "aten.eq.Tensor",
        vec![IrArg::Node(a), IrArg::Node(one)],
    );

    let a_math_gt = tensor_op(
        graph,
        "a_math_gt",
        "aten.where.self",
        vec![IrArg::Node(abs_gt1), IrArg::Node(zeros), IrArg::Node(a)],
    );

    let a_math = tensor_op(
        graph,
        "a_math",
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(zeros),
            IrArg::Node(a_math_gt),
        ],
    );

    let z = tensor_op(
        graph,
        "z",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(x_id)],
    );

    let p0 = tensor_op(
        graph,
        "p0",
        "aten.mul.Tensor",
        vec![IrArg::Node(z), IrArg::Node(a3)],
    );
    let p1 = tensor_op(
        graph,
        "p1",
        "aten.add.Tensor",
        vec![IrArg::Node(p0), IrArg::Node(a2)],
    );
    let p2 = tensor_op(
        graph,
        "p2",
        "aten.mul.Tensor",
        vec![IrArg::Node(p1), IrArg::Node(z)],
    );
    let p3 = tensor_op(
        graph,
        "p3",
        "aten.add.Tensor",
        vec![IrArg::Node(p2), IrArg::Node(a1)],
    );
    let p4 = tensor_op(
        graph,
        "p4",
        "aten.mul.Tensor",
        vec![IrArg::Node(p3), IrArg::Node(z)],
    );
    let p = tensor_op(
        graph,
        "p",
        "aten.add.Tensor",
        vec![IrArg::Node(p4), IrArg::Node(a0)],
    );

    let q0 = tensor_op(
        graph,
        "q0",
        "aten.mul.Tensor",
        vec![IrArg::Node(z), IrArg::Node(b3)],
    );
    let q1 = tensor_op(
        graph,
        "q1",
        "aten.add.Tensor",
        vec![IrArg::Node(q0), IrArg::Node(b2)],
    );
    let q2 = tensor_op(
        graph,
        "q2",
        "aten.mul.Tensor",
        vec![IrArg::Node(q1), IrArg::Node(z)],
    );
    let q3 = tensor_op(
        graph,
        "q3",
        "aten.add.Tensor",
        vec![IrArg::Node(q2), IrArg::Node(b1)],
    );
    let q4 = tensor_op(
        graph,
        "q4",
        "aten.mul.Tensor",
        vec![IrArg::Node(q3), IrArg::Node(z)],
    );
    let q5 = tensor_op(
        graph,
        "q5",
        "aten.add.Tensor",
        vec![IrArg::Node(q4), IrArg::Node(b0)],
    );
    let q6 = tensor_op(
        graph,
        "q6",
        "aten.mul.Tensor",
        vec![IrArg::Node(q5), IrArg::Node(z)],
    );
    let q = tensor_op(
        graph,
        "q",
        "aten.add.Tensor",
        vec![IrArg::Node(q6), IrArg::Node(one)],
    );

    let xp = tensor_op(
        graph,
        "xp",
        "aten.mul.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(p)],
    );
    let y0_central = tensor_op(
        graph,
        "y0_central",
        "aten.div.Tensor",
        vec![IrArg::Node(xp), IrArg::Node(q)],
    );

    let one_minus_a = tensor_op(
        graph,
        "one_minus_a",
        "aten.sub.Tensor",
        vec![IrArg::Node(one), IrArg::Node(a_math)],
    );
    let u = tensor_op(
        graph,
        "u",
        "aten.mul.Tensor",
        vec![IrArg::Node(one_minus_a), IrArg::Node(half)],
    );
    let u_safe = tensor_op(
        graph,
        "u_safe",
        "aten.where.self",
        vec![IrArg::Node(abs_eq1), IrArg::Node(one), IrArg::Node(u)],
    );
    let log_u = tensor_op(
        graph,
        "log_u",
        "aten.log.default",
        vec![IrArg::Node(u_safe)],
    );
    let neg_log_u = tensor_op(
        graph,
        "neg_log_u",
        "aten.mul.Tensor",
        vec![IrArg::Node(log_u), IrArg::Node(neg_ones)],
    );
    let t = tensor_op(
        graph,
        "t",
        "aten.sqrt.default",
        vec![IrArg::Node(neg_log_u)],
    );

    let r0 = tensor_op(
        graph,
        "r0",
        "aten.mul.Tensor",
        vec![IrArg::Node(t), IrArg::Node(c3)],
    );
    let r1 = tensor_op(
        graph,
        "r1",
        "aten.add.Tensor",
        vec![IrArg::Node(r0), IrArg::Node(c2)],
    );
    let r2 = tensor_op(
        graph,
        "r2",
        "aten.mul.Tensor",
        vec![IrArg::Node(r1), IrArg::Node(t)],
    );
    let r3 = tensor_op(
        graph,
        "r3",
        "aten.add.Tensor",
        vec![IrArg::Node(r2), IrArg::Node(c1)],
    );
    let r4 = tensor_op(
        graph,
        "r4",
        "aten.mul.Tensor",
        vec![IrArg::Node(r3), IrArg::Node(t)],
    );
    let r = tensor_op(
        graph,
        "r",
        "aten.add.Tensor",
        vec![IrArg::Node(r4), IrArg::Node(c0)],
    );

    let s0 = tensor_op(
        graph,
        "s0",
        "aten.mul.Tensor",
        vec![IrArg::Node(t), IrArg::Node(d1)],
    );
    let s1 = tensor_op(
        graph,
        "s1",
        "aten.add.Tensor",
        vec![IrArg::Node(s0), IrArg::Node(d0)],
    );
    let s2 = tensor_op(
        graph,
        "s2",
        "aten.mul.Tensor",
        vec![IrArg::Node(s1), IrArg::Node(t)],
    );
    let s_den = tensor_op(
        graph,
        "s_den",
        "aten.add.Tensor",
        vec![IrArg::Node(s2), IrArg::Node(one)],
    );

    let frac = tensor_op(
        graph,
        "frac",
        "aten.div.Tensor",
        vec![IrArg::Node(r), IrArg::Node(s_den)],
    );
    let frac_abs = tensor_op(
        graph,
        "frac_abs",
        "aten.abs.default",
        vec![IrArg::Node(frac)],
    );
    let y0_tail = tensor_op(
        graph,
        "y0_tail",
        "aten.mul.Tensor",
        vec![IrArg::Node(s), IrArg::Node(frac_abs)],
    );

    let in_central = bool_op(
        graph,
        "in_central",
        "aten.lt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(central_range)],
    );

    let y0_pre = tensor_op(
        graph,
        "y0_pre",
        "aten.where.self",
        vec![
            IrArg::Node(in_central),
            IrArg::Node(y0_central),
            IrArg::Node(y0_tail),
        ],
    );
    let y0_no_gt = tensor_op(
        graph,
        "y0_no_gt",
        "aten.where.self",
        vec![
            IrArg::Node(abs_gt1),
            IrArg::Node(zeros),
            IrArg::Node(y0_pre),
        ],
    );
    let y0 = tensor_op(
        graph,
        "y0",
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(zeros),
            IrArg::Node(y0_no_gt),
        ],
    );

    let refine_mask = bool_op(
        graph,
        "refine_mask",
        "aten.lt.Tensor",
        vec![IrArg::Node(a), IrArg::Node(refine_max_abs)],
    );

    let erf_y = tensor_op(
        graph,
        "erf_y",
        "aten.erf.default",
        vec![IrArg::Node(y0)],
    );
    let err = tensor_op(
        graph,
        "err",
        "aten.sub.Tensor",
        vec![IrArg::Node(erf_y), IrArg::Node(x_id)],
    );
    let y_sq = tensor_op(
        graph,
        "y_sq",
        "aten.mul.Tensor",
        vec![IrArg::Node(y0), IrArg::Node(y0)],
    );
    let neg_y_sq = tensor_op(
        graph,
        "neg_y_sq",
        "aten.mul.Tensor",
        vec![IrArg::Node(y_sq), IrArg::Node(neg_ones)],
    );
    let exp_term = tensor_op(
        graph,
        "exp_term",
        "aten.exp.default",
        vec![IrArg::Node(neg_y_sq)],
    );
    let deriv = tensor_op(
        graph,
        "deriv",
        "aten.mul.Tensor",
        vec![IrArg::Node(exp_term), IrArg::Node(two_over_sqrt_pi)],
    );
    let deriv_tiny = bool_op(
        graph,
        "deriv_tiny",
        "aten.lt.Tensor",
        vec![IrArg::Node(deriv), IrArg::Node(deriv_eps)],
    );
    let corr = tensor_op(
        graph,
        "corr",
        "aten.div.Tensor",
        vec![IrArg::Node(err), IrArg::Node(deriv)],
    );
    let corr_abs = tensor_op(
        graph,
        "corr_abs",
        "aten.abs.default",
        vec![IrArg::Node(corr)],
    );
    let corr_huge = bool_op(
        graph,
        "corr_huge",
        "aten.gt.Tensor",
        vec![IrArg::Node(corr_abs), IrArg::Node(corr_max)],
    );
    let y1 = tensor_op(
        graph,
        "y1",
        "aten.sub.Tensor",
        vec![IrArg::Node(y0), IrArg::Node(corr)],
    );
    let y_safe_small = tensor_op(
        graph,
        "y_safe_small",
        "aten.where.self",
        vec![IrArg::Node(deriv_tiny), IrArg::Node(y0), IrArg::Node(y1)],
    );
    let y_safe = tensor_op(
        graph,
        "y_safe",
        "aten.where.self",
        vec![
            IrArg::Node(corr_huge),
            IrArg::Node(y0),
            IrArg::Node(y_safe_small),
        ],
    );
    let y = tensor_op(
        graph,
        "y",
        "aten.where.self",
        vec![
            IrArg::Node(refine_mask),
            IrArg::Node(y_safe),
            IrArg::Node(y0),
        ],
    );

    let inf_signed = tensor_op(
        graph,
        "inf_signed",
        "aten.where.self",
        vec![
            IrArg::Node(x_lt0),
            IrArg::Node(neg_inf),
            IrArg::Node(pos_inf),
        ],
    );
    let out_gt1 = tensor_op(
        graph,
        "out_gt1",
        "aten.where.self",
        vec![IrArg::Node(abs_gt1), IrArg::Node(nan), IrArg::Node(y)],
    );
    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.where.self",
        vec![
            IrArg::Node(abs_eq1),
            IrArg::Node(inf_signed),
            IrArg::Node(out_gt1),
        ],
        meta,
    );
    graph.replace_and_erase(id, result);
    true
}

/// Inference batch norm decomposition for the no-training case.
fn decompose_batch_norm(graph: &mut IrGraph, id: NodeId) -> bool {
    if batch_norm_training(&graph.nodes[id]) {
        return false;
    }

    let x_id = match graph.nodes[id].args.first().and_then(IrArg::as_node) {
        Some(n) => n,
        None => return false,
    };
    let weight_arg = graph.nodes[id].args.get(1).cloned().unwrap_or(IrArg::None);
    let bias_arg = graph.nodes[id].args.get(2).cloned().unwrap_or(IrArg::None);
    let running_mean = match graph.nodes[id].args.get(3).and_then(IrArg::as_node) {
        Some(n) => n,
        None => return false,
    };
    let running_var = match graph.nodes[id].args.get(4).and_then(IrArg::as_node) {
        Some(n) => n,
        None => return false,
    };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    if shape.len() < 2 {
        return false;
    }
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let channel_axis = 1_usize;
    let Some(&num_features) = shape.get(channel_axis) else {
        return false;
    };
    let mut channel_shape = vec![1_i64; shape.len()];
    channel_shape[channel_axis] = num_features;
    let physical_channel_shape = if shape.len() >= 4 {
        permuted_shape(&channel_shape, &channels_last_order(shape.len()))
    } else {
        channel_shape.clone()
    };
    let scalar_shape = vec![1_i64; shape.len()];
    let eps = batch_norm_eps(&graph.nodes[id]);
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);

    let tensor_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            shape.clone(),
            &dtype,
        )
    };
    let stats_op = |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>| {
        let node = graph.add_anchored_call_function_with_val(
            id,
            format!("{}_{}", name, suffix),
            target,
            args,
            &meta,
            physical_channel_shape.clone(),
            &dtype,
        );
        preserve_shape_node(graph, node);
        node
    };
    let view_stats = |graph: &mut IrGraph, source: NodeId, suffix: &str| {
        let node = graph.add_anchored_call_function(
            id,
            format!("{}_{}", name, suffix),
            "aten.view_copy.default",
            vec![IrArg::Node(source), i64_list(&physical_channel_shape)],
            metas.tensor(&physical_channel_shape),
        );
        preserve_shape_node(graph, node);
        node
    };

    let mean_view = view_stats(graph, running_mean, "mean_view");
    let var_view = view_stats(graph, running_var, "var_view");

    let centered = tensor_op(
        graph,
        "centered",
        "aten.sub.Tensor",
        vec![IrArg::Node(x_id), IrArg::Node(mean_view)],
    );

    let eps_full = graph.add_anchored_full_float(
        id,
        format!("{}_eps", name),
        &scalar_shape,
        eps,
        &dtype,
        metas.tensor(&scalar_shape),
    );
    preserve_shape_node(graph, eps_full);

    let var_eps = stats_op(
        graph,
        "var_eps",
        "aten.add.Tensor",
        vec![IrArg::Node(var_view), IrArg::Node(eps_full)],
    );

    let inv_sqrt = stats_op(
        graph,
        "inv_sqrt",
        "aten.rsqrt.default",
        vec![IrArg::Node(var_eps)],
    );

    let normed = tensor_op(
        graph,
        "normed",
        "aten.mul.Tensor",
        vec![IrArg::Node(centered), IrArg::Node(inv_sqrt)],
    );

    let weight_view = match weight_arg {
        IrArg::Node(weight_id) => view_stats(graph, weight_id, "weight_view"),
        IrArg::None => {
            let weight_full = graph.add_anchored_full_float(
                id,
                format!("{}_weight_full", name),
                &[num_features],
                1.0,
                &dtype,
                metas.tensor(&[num_features]),
            );
            view_stats(graph, weight_full, "weight_view")
        }
        _ => return false,
    };

    let weighted = tensor_op(
        graph,
        "weighted",
        "aten.mul.Tensor",
        vec![IrArg::Node(normed), IrArg::Node(weight_view)],
    );

    let bias_view = match bias_arg {
        IrArg::Node(bias_id) => view_stats(graph, bias_id, "bias_view"),
        IrArg::None => {
            let bias_full = graph.add_anchored_full_float(
                id,
                format!("{}_bias_full", name),
                &[num_features],
                0.0,
                &dtype,
                metas.tensor(&[num_features]),
            );
            view_stats(graph, bias_full, "bias_view")
        }
        _ => return false,
    };

    let result = graph.add_anchored_call_function(
        id,
        format!("{}_result", name),
        "aten.add.Tensor",
        vec![IrArg::Node(weighted), IrArg::Node(bias_view)],
        metas.tensor(&shape),
    );

    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.target.contains("getitem"))
        .filter(|node| node.args.first().and_then(IrArg::as_node) == Some(id))
        .map(|node| {
            (
                node.id,
                node.args.get(1).and_then(IrArg::as_int).unwrap_or(0),
            )
        })
        .collect();

    if getitem_users.is_empty() {
        graph.replace_and_erase(id, result);
        return true;
    }

    if graph.nodes.iter().any(|node| {
        node.alive
            && node.id != id
            && !node.target.contains("getitem")
            && node.args.iter().any(|arg| arg.as_node() == Some(id))
    }) {
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

fn decompose_group_norm(graph: &mut IrGraph, id: NodeId) -> bool {
    let args = graph.nodes[id].args.clone();
    let x_id = match args.first().and_then(IrArg::as_node) {
        Some(n) => n,
        None => return false,
    };

    let (weight_arg, bias_arg, group, eps) = match args.len() {
        8 => (
            args.get(1).cloned().unwrap_or(IrArg::None),
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(6).and_then(IrArg::as_int),
            args.get(7)
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64))),
        ),
        6 => (
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(3).cloned().unwrap_or(IrArg::None),
            args.get(1).and_then(IrArg::as_int),
            args.get(4)
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64))),
        ),
        5 => (
            args.get(2).cloned().unwrap_or(IrArg::None),
            args.get(3).cloned().unwrap_or(IrArg::None),
            args.get(1).and_then(IrArg::as_int),
            args.get(4)
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64))),
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
            graph.nodes[id]
                .kwargs
                .get("weight")
                .cloned()
                .unwrap_or(IrArg::None),
            graph.nodes[id]
                .kwargs
                .get("bias")
                .cloned()
                .unwrap_or(IrArg::None),
            graph.nodes[id]
                .kwargs
                .get("num_groups")
                .and_then(IrArg::as_int)
                .or_else(|| graph.nodes[id].kwargs.get("group").and_then(IrArg::as_int)),
            graph.nodes[id]
                .kwargs
                .get("eps")
                .and_then(|arg| arg.as_float().or_else(|| arg.as_int().map(|v| v as f64))),
        ),
    };
    let Some(group) = group else {
        return false;
    };
    if group <= 0 {
        return false;
    }

    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    if shape.len() < 2 {
        return false;
    }
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
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
    let affine_shape = vec![1, group, channels_per_group, 1];
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&meta, &dtype);
    let reshape_target = if graph.nodes[id].target.contains("native_group_norm") {
        "aten.view_copy.default"
    } else {
        "aten.reshape.default"
    };
    let shaped_tensor_op =
        |graph: &mut IrGraph, suffix: &str, target: &str, args: Vec<IrArg>, out_shape: &[i64]| {
            let node = graph.add_anchored_call_function_with_val(
                id,
                format!("{}_{}", name, suffix),
                target,
                args,
                &meta,
                out_shape.to_vec(),
                &dtype,
            );
            preserve_identity_layout(graph, node);
            node
        };
    let reshape =
        |graph: &mut IrGraph, suffix: &str, input: NodeId, out_shape: &[i64]| {
            shaped_tensor_op(
                graph,
                suffix,
                reshape_target,
                vec![IrArg::Node(input), i64_list(out_shape)],
                out_shape,
            )
        };
    let full = |graph: &mut IrGraph, suffix: &str, value: f64, out_shape: &[i64]| {
        let node = graph.add_anchored_full_float(
            id,
            format!("{}_{}", name, suffix),
            out_shape,
            value,
            &dtype,
            metas.tensor(out_shape),
        );
        preserve_identity_layout(graph, node);
        node
    };

    let grouped = reshape(graph, "grouped", x_id, &grouped_shape);
    let inv_n = full(
        graph,
        "inv_n",
        1.0 / ((channels_per_group * hxw) as f64),
        &scalar_shape,
    );
    let sum_x = shaped_tensor_op(
        graph,
        "sum",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(grouped), reduce_dims.clone(), IrArg::Bool(true)],
        &scalar_shape,
    );
    let mean = shaped_tensor_op(
        graph,
        "mean",
        "aten.mul.Tensor",
        vec![IrArg::Node(sum_x), IrArg::Node(inv_n)],
        &scalar_shape,
    );
    let centered = shaped_tensor_op(
        graph,
        "centered",
        "aten.sub.Tensor",
        vec![IrArg::Node(grouped), IrArg::Node(mean)],
        &grouped_shape,
    );
    let squared = shaped_tensor_op(
        graph,
        "squared",
        "aten.mul.Tensor",
        vec![IrArg::Node(centered), IrArg::Node(centered)],
        &grouped_shape,
    );
    let sum_squared = shaped_tensor_op(
        graph,
        "sum_squared",
        "aten.sum.dim_IntList",
        vec![IrArg::Node(squared), reduce_dims, IrArg::Bool(true)],
        &scalar_shape,
    );
    let var = shaped_tensor_op(
        graph,
        "var",
        "aten.mul.Tensor",
        vec![IrArg::Node(sum_squared), IrArg::Node(inv_n)],
        &scalar_shape,
    );
    let eps_full = full(graph, "eps", eps, &scalar_shape);
    let var_eps = shaped_tensor_op(
        graph,
        "var_eps",
        "aten.add.Tensor",
        vec![IrArg::Node(var), IrArg::Node(eps_full)],
        &scalar_shape,
    );
    let rsqrt = shaped_tensor_op(
        graph,
        "rsqrt",
        "aten.rsqrt.default",
        vec![IrArg::Node(var_eps)],
        &scalar_shape,
    );
    let normalized = shaped_tensor_op(
        graph,
        "normalized",
        "aten.mul.Tensor",
        vec![IrArg::Node(centered), IrArg::Node(rsqrt)],
        &grouped_shape,
    );

    let weighted = match weight_arg {
        IrArg::Node(weight_id) => {
            let weight_view = reshape(graph, "weight_view", weight_id, &affine_shape);
            let weighted = shaped_tensor_op(
                graph,
                "weighted",
                "aten.mul.Tensor",
                vec![IrArg::Node(normalized), IrArg::Node(weight_view)],
                &grouped_shape,
            );
            weighted
        }
        IrArg::None => normalized,
        _ => return false,
    };

    let biased = match bias_arg {
        IrArg::Node(bias_id) => {
            let bias_view = reshape(graph, "bias_view", bias_id, &affine_shape);
            let biased = shaped_tensor_op(
                graph,
                "biased",
                "aten.add.Tensor",
                vec![IrArg::Node(weighted), IrArg::Node(bias_view)],
                &grouped_shape,
            );
            biased
        }
        IrArg::None => weighted,
        _ => return false,
    };

    let result = reshape(graph, "result", biased, &shape);

    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.target.contains("getitem"))
        .filter(|node| node.args.first().and_then(IrArg::as_node) == Some(id))
        .map(|node| {
            (
                node.id,
                node.args.get(1).and_then(IrArg::as_int).unwrap_or(0),
            )
        })
        .collect();

    if getitem_users.is_empty() {
        graph.replace_and_erase(id, result);
        return true;
    }

    if graph.nodes.iter().any(|node| {
        node.alive
            && node.id != id
            && !node.target.contains("getitem")
            && node.args.iter().any(|arg| arg.as_node() == Some(id))
    }) {
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
    let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0);

    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let dtype = graph.nodes[tensor_ids[0]]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
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

        let unsq = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_unsq_{}", name, i),
            "aten.unsqueeze_copy.default".to_string(),
            vec![IrArg::Node(tid), IrArg::Int(dim)],
            metas.tensor(&u_shape),
        );
        graph.anchor_to(unsq, id);
        unsqueezed.push(unsq);
    }

    // Cat along dim
    let cat_args = IrArg::List(unsqueezed.iter().map(|&n| IrArg::Node(n)).collect());
    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_cat", name),
        "aten.cat.default".to_string(),
        vec![cat_args, IrArg::Int(dim)],
        meta,
    );
    graph.replace_and_erase(id, result);
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

    let x_shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let rank = x_shape.len() as i64;
    let dim_norm = dim.rem_euclid(rank) as usize;
    let dtype = graph.nodes[x_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();
    let name = graph.nodes[id].name.clone();
    let orig_meta = graph.nodes[id].meta.clone();
    let metas = MetaFactory::new(&orig_meta, &dtype);

    // Find getitem users
    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|n| n.alive && n.target.contains("getitem"))
        .filter(|n| n.args.first().and_then(|a| a.as_node()) == Some(id))
        .map(|n| {
            let idx = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0);
            (n.id, idx)
        })
        .collect();

    if getitem_users.is_empty() {
        return false;
    }

    for (getitem_id, idx) in &getitem_users {
        // slice(x, dim, idx, idx+1)
        let mut slice_shape = x_shape.clone();
        slice_shape[dim_norm] = 1;
        let slice_id = graph.add_node(
            NodeOp::CallFunction,
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
        graph.anchor_to(slice_id, id);

        // squeeze(slice, dim)
        let mut sq_shape = x_shape.clone();
        sq_shape.remove(dim_norm);
        let squeeze_id = graph.add_node(
            NodeOp::CallFunction,
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
