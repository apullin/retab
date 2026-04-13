//! Pure Rust IrPass: atanh, stack, unbind decompositions.
//! Mirrors FxDecomposeOpsPass (hardsigmoid is intentionally skipped).

use std::f64::consts::FRAC_PI_2;

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, MetaValue, NodeId,
    NodeMeta, NodeOp,
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

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    m
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
    t.contains("aten.softmax.")
        || t.contains("aten._safe_softmax.")
        || t.contains("aten._softmax.")
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
    meta.set_extra("arm_preserve_shape_node", MetaValue::Bool(true));
    meta.set_extra("arm_layout_neutral", MetaValue::Bool(true));
}

fn preserve_identity_layout(graph: &mut IrGraph, node_id: NodeId) {
    preserve_shape_node(graph, node_id);
    let rank = graph.nodes[node_id]
        .meta
        .val_shape()
        .map(|shape| shape.len())
        .unwrap_or(0);
    if rank > 0 {
        graph.nodes[node_id]
            .meta
            .set_tosa_dim_order((0..rank).collect());
    }
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
    let const_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_{}", name, suffix),
        "aten.full.default".to_string(),
        vec![i64_list(shape), IrArg::Float(value)],
        meta_with(meta, shape.to_vec(), dtype),
    );
    graph.anchor_to(const_id, anchor_id);
    const_id
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
        let term = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_{}_term{}", name, prefix, idx),
            "aten.mul.Tensor".to_string(),
            vec![IrArg::Node(current_var), IrArg::Node(coeff_id)],
            meta_with(meta, shape.to_vec(), dtype),
        );
        graph.anchor_to(term, anchor_id);
        let sum = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_{}_sum{}", name, prefix, idx),
            "aten.add.Tensor".to_string(),
            vec![IrArg::Node(result), IrArg::Node(term)],
            meta_with(meta, shape.to_vec(), dtype),
        );
        graph.anchor_to(sum, anchor_id);
        result = sum;

        if idx + 1 < coefficients.len() {
            let next_var = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_{}_pow{}", name, prefix, idx),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(current_var), IrArg::Node(current_var)],
                meta_with(meta, shape.to_vec(), dtype),
            );
            graph.anchor_to(next_var, anchor_id);
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
    let scaled_meta = meta_with(&graph.nodes[rhs_id].meta, rhs_shape.clone(), &rhs_dtype);

    let alpha_full = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_alpha_full", name),
        "aten.full.default".to_string(),
        vec![i64_list(&rhs_shape), IrArg::Float(alpha)],
        scaled_meta.clone(),
    );
    graph.anchor_to(alpha_full, id);

    let scaled_rhs = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_alpha_mul", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(rhs_id), IrArg::Node(alpha_full)],
        scaled_meta,
    );
    graph.anchor_to(scaled_rhs, id);

    let mut add_args = graph.nodes[id].args.clone();
    if add_args.len() < 2 {
        return false;
    }
    add_args[1] = IrArg::Node(scaled_rhs);
    add_args.truncate(2);
    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_alpha_result", name),
        target,
        add_args,
        meta,
    );
    graph.anchor_to(result, id);
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
        let shape = meta.val_shape().map(|shape| shape.to_vec()).unwrap_or_default();
        let dtype = meta.val_dtype().unwrap_or("torch.float32").to_string();
        let one = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_one", name),
            "aten.full.default".to_string(),
            vec![i64_list(&shape), IrArg::Float(1.0)],
            meta_with(&meta, shape.clone(), &dtype),
        );
        graph.anchor_to(one, id);
        let eq = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_self_eq", name),
            "aten.eq.Tensor".to_string(),
            vec![IrArg::Node(input_id), IrArg::Node(input_id)],
            meta_with(&meta, shape, "torch.bool"),
        );
        graph.anchor_to(eq, id);
        let ones = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_ones", name),
            "aten.where.self".to_string(),
            vec![IrArg::Node(eq), IrArg::Node(one), IrArg::Node(one)],
            meta,
        );
        graph.anchor_to(ones, id);
        graph.replace_and_erase(id, ones);
        return true;
    }

    let Some(mut exp) = pow_scalar_int_arg(&graph.nodes[id]) else {
        return false;
    };

    let mut base_id = input_id;
    if exp < 0 {
        let reciprocal = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_reciprocal", name),
            "aten.reciprocal.default".to_string(),
            vec![IrArg::Node(input_id)],
            meta.clone(),
        );
        graph.anchor_to(reciprocal, id);
        base_id = reciprocal;
        exp = -exp;
    }

    let mut result_id = base_id;
    for idx in 1..exp {
        let mul = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_mul{}", name, idx),
            "aten.mul.Tensor".to_string(),
            vec![IrArg::Node(result_id), IrArg::Node(base_id)],
            meta.clone(),
        );
        graph.anchor_to(mul, id);
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "atan_result" && node.target == "aten.where.self"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "acos_result" && node.target == "aten.where.self"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "softmax_result" && node.target == "aten.mul.Tensor"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "log_softmax_result" && node.target == "aten.log.default"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "sinh_result" && node.target == "aten.mul.Tensor"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "cosh_result" && node.target == "aten.mul.Tensor"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "asinh_result" && node.target == "aten.log.default"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "acosh_result" && node.target == "aten.log.default"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "elu_result" && node.target == "aten.where.self"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "gelu_result" && node.target == "aten.mul.Tensor"));
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "gelu_erf" && node.target == "aten.erf.default"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "gelu_tanh" && node.target == "aten.tanh.default"));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "bn_result" && node.target == "aten.add.Tensor"));
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "bn_inv_sqrt" && node.target == "aten.rsqrt.default"));
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
        assert_eq!(mean_view.meta.extra_bool("arm_preserve_shape_node"), Some(true));
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
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "gn_grouped" && node.target == "aten.reshape.default"));
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == "gn_result" && node.target == "aten.reshape.default"));
        let grouped = graph
            .nodes
            .iter()
            .find(|node| node.alive && node.name == "gn_grouped")
            .expect("grouped");
        assert_eq!(grouped.meta.val_shape().unwrap(), &[1, 2, 3, 8]);
        assert_eq!(grouped.meta.extra_bool("arm_preserve_shape_node"), Some(true));
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

    if graph
        .nodes
        .iter()
        .any(|n| n.alive && n.id != id && !n.target.contains("getitem") && graph.users(id).contains(&n.id))
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
            if is_max_dim(&node_target) { "amax" } else { "amin" }
        ),
        reduce_target.to_string(),
        vec![IrArg::Node(input_id), IrArg::Int(dim as i64), IrArg::Bool(true)],
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

    let reduce = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_amax", name),
        "aten.amax.default".to_string(),
        vec![IrArg::Node(x_id), IrArg::Int(dim as i64), IrArg::Bool(true)],
        meta_with(&meta, reduced_shape.clone(), &dtype),
    );
    graph.anchor_to(reduce, id);

    let shifted = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_shifted", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(reduce)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(shifted, id);

    let exp = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(shifted)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(exp, id);

    let sum = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.sum.dim_IntList".to_string(),
        vec![
            IrArg::Node(exp),
            i64_list(&[dim as i64]),
            IrArg::Bool(true),
        ],
        meta_with(&meta, sum_shape.clone(), &dtype),
    );
    graph.anchor_to(sum, id);

    let recip = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_recip", name),
        "aten.reciprocal.default".to_string(),
        vec![IrArg::Node(sum)],
        meta_with(&meta, sum_shape, &dtype),
    );
    graph.anchor_to(recip, id);

    let softmax = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_softmax", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(exp), IrArg::Node(recip)],
        meta_with(&meta, shape, &dtype),
    );
    graph.anchor_to(softmax, id);

    let result = if log_output {
        let log = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_result", name),
            "aten.log.default".to_string(),
            vec![IrArg::Node(softmax)],
            meta,
        );
        graph.anchor_to(log, id);
        log
    } else {
        let rename = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_result", name),
            "aten.clone.default".to_string(),
            vec![IrArg::Node(softmax)],
            meta,
        );
        graph.anchor_to(rename, id);
        rename
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

    let one = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_one", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(1.0)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(one, id);

    let half = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_half", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(0.5)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(half, id);

    let one_plus = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_1px", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(one_plus, id);

    let one_minus = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_1mx", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(one_minus, id);

    let ratio = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_ratio", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(one_plus), IrArg::Node(one_minus)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(ratio, id);

    let log_r = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_log", name),
        "aten.log.default".to_string(),
        vec![IrArg::Node(ratio)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(log_r, id);

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

    let sin = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sin", name),
        "aten.sin.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(sin, id);

    let cos = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_cos", name),
        "aten.cos.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(cos, id);

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
    let bool_meta = meta_with(&meta, shape.clone(), "torch.bool");

    let one = add_full_const(graph, id, &name, "one", 1.0, &shape, &dtype, &meta);
    let zero = add_full_const(graph, id, &name, "zero", 0.0, &shape, &dtype, &meta);
    let half_pi = add_full_const(graph, id, &name, "half_pi", FRAC_PI_2, &shape, &dtype, &meta);
    let a1 = add_full_const(graph, id, &name, "a1", 0.3529666667, &shape, &dtype, &meta);
    let a2 = add_full_const(graph, id, &name, "a2", -0.0287666667, &shape, &dtype, &meta);
    let b1 = add_full_const(graph, id, &name, "b1", 0.6863, &shape, &dtype, &meta);

    let abs_x = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_abs", name),
        "aten.abs.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(abs_x, id);

    let mask_hi = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mask_hi", name),
        "aten.gt.Tensor".to_string(),
        vec![IrArg::Node(abs_x), IrArg::Node(one)],
        bool_meta.clone(),
    );
    graph.anchor_to(mask_hi, id);

    let inv_x = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv", name),
        "aten.reciprocal.default".to_string(),
        vec![IrArg::Node(abs_x)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(inv_x, id);

    let z = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_z", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(mask_hi), IrArg::Node(inv_x), IrArg::Node(abs_x)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(z, id);

    let z2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_z2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z), IrArg::Node(z)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(z2, id);

    let z4 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_z4", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z2), IrArg::Node(z2)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(z4, id);

    let num1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_num1", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z2), IrArg::Node(a1)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(num1, id);

    let num2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_num2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z4), IrArg::Node(a2)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(num2, id);

    let num_base = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_num_base", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(num1), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(num_base, id);

    let num = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_num", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(num_base), IrArg::Node(num2)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(num, id);

    let den1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_den1", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z2), IrArg::Node(b1)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(den1, id);

    let den = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_den", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(den1), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(den, id);

    let inv_den = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv_den", name),
        "aten.reciprocal.default".to_string(),
        vec![IrArg::Node(den)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(inv_den, id);

    let prod = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_prod", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(num), IrArg::Node(inv_den)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(prod, id);

    let atan_z = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_atan_z", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z), IrArg::Node(prod)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(atan_z, id);

    let diff = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_diff", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(half_pi), IrArg::Node(atan_z)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(diff, id);

    let atan_abs = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_atan_abs", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(mask_hi), IrArg::Node(diff), IrArg::Node(atan_z)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(atan_abs, id);

    let mask_pos = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mask_pos", name),
        "aten.gt.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
        bool_meta,
    );
    graph.anchor_to(mask_pos, id);

    let neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_neg", name),
        "aten.neg.default".to_string(),
        vec![IrArg::Node(atan_abs)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(neg, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(mask_pos), IrArg::Node(atan_abs), IrArg::Node(neg)],
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

    let half = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_half", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(0.5)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(half, id);

    let exp_pos = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp_pos", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(exp_pos, id);

    let neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_neg", name),
        "aten.neg.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(neg, id);

    let exp_neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp_neg", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(neg)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(exp_neg, id);

    let diff = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_diff", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(exp_pos), IrArg::Node(exp_neg)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(diff, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.mul.Tensor".to_string(),
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

    let half = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_half", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(0.5)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(half, id);

    let exp_pos = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp_pos", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(exp_pos, id);

    let neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_neg", name),
        "aten.neg.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(neg, id);

    let exp_neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp_neg", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(neg)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(exp_neg, id);

    let sum = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(exp_pos), IrArg::Node(exp_neg)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(sum, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.mul.Tensor".to_string(),
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

    let one = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_one", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(1.0)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(one, id);

    let square = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_square", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(square, id);

    let radicand = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_radicand", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(square), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(radicand, id);

    let sqrt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sqrt", name),
        "aten.sqrt.default".to_string(),
        vec![IrArg::Node(radicand)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(sqrt, id);

    let add = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(sqrt)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(add, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.log.default".to_string(),
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

    let one = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_one", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(1.0)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(one, id);

    let minus = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_minus", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(minus, id);

    let plus = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_plus", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(plus, id);

    let radicand = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_radicand", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(minus), IrArg::Node(plus)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(radicand, id);

    let sqrt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sqrt", name),
        "aten.sqrt.default".to_string(),
        vec![IrArg::Node(radicand)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(sqrt, id);

    let add = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(sqrt)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(add, id);

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.log.default".to_string(),
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
    let bool_meta = meta_with(&meta, shape.clone(), "torch.bool");

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
    let half_pi = add_full_const(graph, id, &name, "half_pi", FRAC_PI_2, &shape, &dtype, &meta);
    let zero = add_full_const(graph, id, &name, "zero", 0.0, &shape, &dtype, &meta);

    let x_abs = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_abs", name),
        "aten.abs.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(x_abs, id);

    let y = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x_abs), IrArg::Node(x_abs)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(y, id);

    let x3 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_x3", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x_abs), IrArg::Node(y)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(x3, id);

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

    let numer_small = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_numer_small", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x3), IrArg::Node(p_small)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(numer_small, id);

    let r_small = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r_small", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(numer_small), IrArg::Node(q_small)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(r_small, id);

    let asin_small = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_asin_small", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(x_abs), IrArg::Node(r_small)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(asin_small, id);

    let tmp = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_tmp", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_abs), IrArg::Node(one)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(tmp, id);

    let z = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_z", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(tmp), IrArg::Node(neg_half)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(z, id);

    let s = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s", name),
        "aten.sqrt.default".to_string(),
        vec![IrArg::Node(z)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(s, id);

    let s2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(s), IrArg::Node(s)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(s2, id);

    let s3 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s3", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(s2), IrArg::Node(s)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(s3, id);

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

    let numer_large = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_numer_large", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(s3), IrArg::Node(pz)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(numer_large, id);

    let r_large = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r_large", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(numer_large), IrArg::Node(qz)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(r_large, id);

    let t1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_t1", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(s), IrArg::Node(r_large)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(t1, id);

    let t2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_t2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(t1), IrArg::Node(two)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(t2, id);

    let diff = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_diff", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(t2), IrArg::Node(half_pi)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(diff, id);

    let asin_large = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_asin_large", name),
        "aten.neg.default".to_string(),
        vec![IrArg::Node(diff)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(asin_large, id);

    let mask_large = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mask_large", name),
        "aten.gt.Tensor".to_string(),
        vec![IrArg::Node(x_abs), IrArg::Node(half)],
        bool_meta.clone(),
    );
    graph.anchor_to(mask_large, id);

    let asin_unsigned = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_asin_unsigned", name),
        "aten.where.self".to_string(),
        vec![
            IrArg::Node(mask_large),
            IrArg::Node(asin_large),
            IrArg::Node(asin_small),
        ],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(asin_unsigned, id);

    let negated_asin = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_negated_asin", name),
        "aten.neg.default".to_string(),
        vec![IrArg::Node(asin_unsigned)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(negated_asin, id);

    let mask_neg = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mask_neg", name),
        "aten.lt.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
        bool_meta.clone(),
    );
    graph.anchor_to(mask_neg, id);

    let asin = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_asin", name),
        "aten.where.self".to_string(),
        vec![
            IrArg::Node(mask_neg),
            IrArg::Node(negated_asin),
            IrArg::Node(asin_unsigned),
        ],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(asin, id);

    let result = if is_acos_op {
        let acos_small = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_acos_small", name),
            "aten.sub.Tensor".to_string(),
            vec![IrArg::Node(half_pi), IrArg::Node(asin)],
            meta_with(&meta, shape.clone(), &dtype),
        );
        graph.anchor_to(acos_small, id);

        let mask_gt_half = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_mask_gt_half", name),
            "aten.gt.Tensor".to_string(),
            vec![IrArg::Node(x_id), IrArg::Node(half)],
            bool_meta,
        );
        graph.anchor_to(mask_gt_half, id);

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
        let asin_result = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_result", name),
            "aten.clone.default".to_string(),
            vec![IrArg::Node(asin)],
            meta,
        );
        graph.anchor_to(asin_result, id);
        asin_result
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

    let zero = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_zero", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(0.0)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(zero, id);

    let alpha_full = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_alpha", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(alpha)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(alpha_full, id);

    let expm1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_expm1", name),
        "aten.expm1.default".to_string(),
        vec![IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(expm1, id);

    let scaled = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_scaled", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(expm1), IrArg::Node(alpha_full)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(scaled, id);

    let ge = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_ge", name),
        "aten.ge.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(zero)],
        meta_with(&meta, shape.clone(), "torch.bool"),
    );
    graph.anchor_to(ge, id);

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

    let full = |graph: &mut IrGraph, suffix: &str, value: f64| {
        let node = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_{}", name, suffix),
            "aten.full.default".to_string(),
            vec![i64_list(&rank_ones), IrArg::Float(value)],
            meta_with(&meta, rank_ones.clone(), &dtype),
        );
        graph.anchor_to(node, id);
        node
    };

    let half = full(graph, "half", 0.5);
    let one = full(graph, "one", 1.0);

    let result = match approximate.as_str() {
        "none" => {
            let sqrt_half = full(graph, "sqrt_half", std::f64::consts::FRAC_1_SQRT_2);
            let scaled = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_scaled", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(sqrt_half)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(scaled, id);

            let erf = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_erf", name),
                "aten.erf.default".to_string(),
                vec![IrArg::Node(scaled)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(erf, id);

            let plus_one = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_plus_one", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(erf), IrArg::Node(one)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(plus_one, id);

            let weighted = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_weighted", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(plus_one), IrArg::Node(half)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(weighted, id);

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

            let sqrt_mul = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_sqrt_mul", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(sqrt_two), IrArg::Node(two_over_sqrt_pi)],
                meta_with(&meta, rank_ones.clone(), &dtype),
            );
            graph.anchor_to(sqrt_mul, id);

            let sqrt_2_pi = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_sqrt_2_pi", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(sqrt_mul), IrArg::Node(half)],
                meta_with(&meta, rank_ones.clone(), &dtype),
            );
            graph.anchor_to(sqrt_2_pi, id);

            let square = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_square", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(x_id)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(square, id);

            let cube = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_cube", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(square), IrArg::Node(x_id)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(cube, id);

            let cubic_term = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_cubic_term", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(cube), IrArg::Node(cube_coeff)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(cubic_term, id);

            let inner = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_inner", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(cubic_term)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(inner, id);

            let scaled = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_scaled", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(inner), IrArg::Node(sqrt_2_pi)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(scaled, id);

            let tanh = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_tanh", name),
                "aten.tanh.default".to_string(),
                vec![IrArg::Node(scaled)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(tanh, id);

            let plus_one = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_plus_one", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(tanh), IrArg::Node(one)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(plus_one, id);

            let first_mul = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_first_mul", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(plus_one)],
                meta_with(&meta, shape.clone(), &dtype),
            );
            graph.anchor_to(first_mul, id);

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

    let full = |graph: &mut IrGraph, suffix: &str, value: f64| {
        let node = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_{}", name, suffix),
            "aten.full.default".to_string(),
            vec![i64_list(&rank_ones), IrArg::Float(value)],
            meta_with(&meta, rank_ones.clone(), &dtype),
        );
        graph.anchor_to(node, id);
        node
    };

    let bool_meta = meta_with(&meta, shape.clone(), "torch.bool");
    let tensor_meta = meta_with(&meta, shape.clone(), &dtype);

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

    let x_lt0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_x_lt0", name),
        "aten.lt.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(zeros)],
        bool_meta.clone(),
    );
    graph.anchor_to(x_lt0, id);

    let s = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sign", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(x_lt0), IrArg::Node(neg_ones), IrArg::Node(one)],
        tensor_meta.clone(),
    );
    graph.anchor_to(s, id);

    let a = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_abs", name),
        "aten.abs.default".to_string(),
        vec![IrArg::Node(x_id)],
        tensor_meta.clone(),
    );
    graph.anchor_to(a, id);

    let abs_gt1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_abs_gt1", name),
        "aten.gt.Tensor".to_string(),
        vec![IrArg::Node(a), IrArg::Node(one)],
        bool_meta.clone(),
    );
    graph.anchor_to(abs_gt1, id);

    let abs_eq1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_abs_eq1", name),
        "aten.eq.Tensor".to_string(),
        vec![IrArg::Node(a), IrArg::Node(one)],
        bool_meta.clone(),
    );
    graph.anchor_to(abs_eq1, id);

    let a_math_gt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_a_math_gt", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_gt1), IrArg::Node(zeros), IrArg::Node(a)],
        tensor_meta.clone(),
    );
    graph.anchor_to(a_math_gt, id);

    let a_math = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_a_math", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_eq1), IrArg::Node(zeros), IrArg::Node(a_math_gt)],
        tensor_meta.clone(),
    );
    graph.anchor_to(a_math, id);

    let z = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_z", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(x_id)],
        tensor_meta.clone(),
    );
    graph.anchor_to(z, id);

    let p0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p0", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z), IrArg::Node(a3)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p0, id);
    let p1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p1", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(p0), IrArg::Node(a2)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p1, id);
    let p2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(p1), IrArg::Node(z)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p2, id);
    let p3 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p3", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(p2), IrArg::Node(a1)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p3, id);
    let p4 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p4", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(p3), IrArg::Node(z)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p4, id);
    let p = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_p", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(p4), IrArg::Node(a0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(p, id);

    let q0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q0", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(z), IrArg::Node(b3)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q0, id);
    let q1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q1", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(q0), IrArg::Node(b2)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q1, id);
    let q2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(q1), IrArg::Node(z)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q2, id);
    let q3 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q3", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(q2), IrArg::Node(b1)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q3, id);
    let q4 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q4", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(q3), IrArg::Node(z)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q4, id);
    let q5 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q5", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(q4), IrArg::Node(b0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q5, id);
    let q6 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q6", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(q5), IrArg::Node(z)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q6, id);
    let q = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_q", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(q6), IrArg::Node(one)],
        tensor_meta.clone(),
    );
    graph.anchor_to(q, id);

    let xp = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_xp", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(p)],
        tensor_meta.clone(),
    );
    graph.anchor_to(xp, id);
    let y0_central = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y0_central", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(xp), IrArg::Node(q)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y0_central, id);

    let one_minus_a = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_one_minus_a", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(one), IrArg::Node(a_math)],
        tensor_meta.clone(),
    );
    graph.anchor_to(one_minus_a, id);
    let u = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_u", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(one_minus_a), IrArg::Node(half)],
        tensor_meta.clone(),
    );
    graph.anchor_to(u, id);
    let u_safe = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_u_safe", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_eq1), IrArg::Node(one), IrArg::Node(u)],
        tensor_meta.clone(),
    );
    graph.anchor_to(u_safe, id);
    let log_u = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_log_u", name),
        "aten.log.default".to_string(),
        vec![IrArg::Node(u_safe)],
        tensor_meta.clone(),
    );
    graph.anchor_to(log_u, id);
    let neg_log_u = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_neg_log_u", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(log_u), IrArg::Node(neg_ones)],
        tensor_meta.clone(),
    );
    graph.anchor_to(neg_log_u, id);
    let t = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_t", name),
        "aten.sqrt.default".to_string(),
        vec![IrArg::Node(neg_log_u)],
        tensor_meta.clone(),
    );
    graph.anchor_to(t, id);

    let r0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r0", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(t), IrArg::Node(c3)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r0, id);
    let r1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r1", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(r0), IrArg::Node(c2)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r1, id);
    let r2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(r1), IrArg::Node(t)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r2, id);
    let r3 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r3", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(r2), IrArg::Node(c1)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r3, id);
    let r4 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r4", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(r3), IrArg::Node(t)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r4, id);
    let r = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_r", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(r4), IrArg::Node(c0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(r, id);

    let s0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s0", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(t), IrArg::Node(d1)],
        tensor_meta.clone(),
    );
    graph.anchor_to(s0, id);
    let s1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s1", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(s0), IrArg::Node(d0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(s1, id);
    let s2 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s2", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(s1), IrArg::Node(t)],
        tensor_meta.clone(),
    );
    graph.anchor_to(s2, id);
    let s_den = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_s_den", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(s2), IrArg::Node(one)],
        tensor_meta.clone(),
    );
    graph.anchor_to(s_den, id);

    let frac = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_frac", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(r), IrArg::Node(s_den)],
        tensor_meta.clone(),
    );
    graph.anchor_to(frac, id);
    let frac_abs = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_frac_abs", name),
        "aten.abs.default".to_string(),
        vec![IrArg::Node(frac)],
        tensor_meta.clone(),
    );
    graph.anchor_to(frac_abs, id);
    let y0_tail = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y0_tail", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(s), IrArg::Node(frac_abs)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y0_tail, id);

    let in_central = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_in_central", name),
        "aten.lt.Tensor".to_string(),
        vec![IrArg::Node(a), IrArg::Node(central_range)],
        bool_meta.clone(),
    );
    graph.anchor_to(in_central, id);

    let y0_pre = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y0_pre", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(in_central), IrArg::Node(y0_central), IrArg::Node(y0_tail)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y0_pre, id);
    let y0_no_gt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y0_no_gt", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_gt1), IrArg::Node(zeros), IrArg::Node(y0_pre)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y0_no_gt, id);
    let y0 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y0", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_eq1), IrArg::Node(zeros), IrArg::Node(y0_no_gt)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y0, id);

    let refine_mask = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_refine_mask", name),
        "aten.lt.Tensor".to_string(),
        vec![IrArg::Node(a), IrArg::Node(refine_max_abs)],
        bool_meta.clone(),
    );
    graph.anchor_to(refine_mask, id);

    let erf_y = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_erf_y", name),
        "aten.erf.default".to_string(),
        vec![IrArg::Node(y0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(erf_y, id);
    let err = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_err", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(erf_y), IrArg::Node(x_id)],
        tensor_meta.clone(),
    );
    graph.anchor_to(err, id);
    let y_sq = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y_sq", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(y0), IrArg::Node(y0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y_sq, id);
    let neg_y_sq = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_neg_y_sq", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(y_sq), IrArg::Node(neg_ones)],
        tensor_meta.clone(),
    );
    graph.anchor_to(neg_y_sq, id);
    let exp_term = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_exp_term", name),
        "aten.exp.default".to_string(),
        vec![IrArg::Node(neg_y_sq)],
        tensor_meta.clone(),
    );
    graph.anchor_to(exp_term, id);
    let deriv = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_deriv", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(exp_term), IrArg::Node(two_over_sqrt_pi)],
        tensor_meta.clone(),
    );
    graph.anchor_to(deriv, id);
    let deriv_tiny = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_deriv_tiny", name),
        "aten.lt.Tensor".to_string(),
        vec![IrArg::Node(deriv), IrArg::Node(deriv_eps)],
        bool_meta.clone(),
    );
    graph.anchor_to(deriv_tiny, id);
    let corr = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_corr", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(err), IrArg::Node(deriv)],
        tensor_meta.clone(),
    );
    graph.anchor_to(corr, id);
    let corr_abs = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_corr_abs", name),
        "aten.abs.default".to_string(),
        vec![IrArg::Node(corr)],
        tensor_meta.clone(),
    );
    graph.anchor_to(corr_abs, id);
    let corr_huge = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_corr_huge", name),
        "aten.gt.Tensor".to_string(),
        vec![IrArg::Node(corr_abs), IrArg::Node(corr_max)],
        bool_meta.clone(),
    );
    graph.anchor_to(corr_huge, id);
    let y1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y1", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(y0), IrArg::Node(corr)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y1, id);
    let y_safe_small = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y_safe_small", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(deriv_tiny), IrArg::Node(y0), IrArg::Node(y1)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y_safe_small, id);
    let y_safe = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y_safe", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(corr_huge), IrArg::Node(y0), IrArg::Node(y_safe_small)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y_safe, id);
    let y = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_y", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(refine_mask), IrArg::Node(y_safe), IrArg::Node(y0)],
        tensor_meta.clone(),
    );
    graph.anchor_to(y, id);

    let inf_signed = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inf_signed", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(x_lt0), IrArg::Node(neg_inf), IrArg::Node(pos_inf)],
        tensor_meta.clone(),
    );
    graph.anchor_to(inf_signed, id);
    let out_gt1 = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_out_gt1", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_gt1), IrArg::Node(nan), IrArg::Node(y)],
        tensor_meta.clone(),
    );
    graph.anchor_to(out_gt1, id);
    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.where.self".to_string(),
        vec![IrArg::Node(abs_eq1), IrArg::Node(inf_signed), IrArg::Node(out_gt1)],
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

    let view_stats = |graph: &mut IrGraph, source: NodeId, suffix: &str| {
        let node = graph.add_node(
            NodeOp::CallFunction,
            format!("{}_{}", name, suffix),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(source), i64_list(&physical_channel_shape)],
            meta_with(&meta, physical_channel_shape.clone(), &dtype),
        );
        graph.anchor_to(node, id);
        preserve_shape_node(graph, node);
        node
    };

    let mean_view = view_stats(graph, running_mean, "mean_view");
    let var_view = view_stats(graph, running_var, "var_view");

    let centered = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_centered", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(x_id), IrArg::Node(mean_view)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(centered, id);

    let eps_full = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_eps", name),
        "aten.full.default".to_string(),
        vec![i64_list(&scalar_shape), IrArg::Float(eps)],
        meta_with(&meta, scalar_shape, &dtype),
    );
    graph.anchor_to(eps_full, id);
    preserve_shape_node(graph, eps_full);

    let var_eps = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_var_eps", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(var_view), IrArg::Node(eps_full)],
        meta_with(&meta, physical_channel_shape.clone(), &dtype),
    );
    graph.anchor_to(var_eps, id);
    preserve_shape_node(graph, var_eps);

    let inv_sqrt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv_sqrt", name),
        "aten.rsqrt.default".to_string(),
        vec![IrArg::Node(var_eps)],
        meta_with(&meta, physical_channel_shape.clone(), &dtype),
    );
    graph.anchor_to(inv_sqrt, id);
    preserve_shape_node(graph, inv_sqrt);

    let normed = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_normed", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(centered), IrArg::Node(inv_sqrt)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(normed, id);

    let weight_view = match weight_arg {
        IrArg::Node(weight_id) => view_stats(graph, weight_id, "weight_view"),
        IrArg::None => {
            let weight_full = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_weight_full", name),
                "aten.full.default".to_string(),
            vec![i64_list(&[num_features]), IrArg::Float(1.0)],
            meta_with(&meta, vec![num_features], &dtype),
        );
        graph.anchor_to(weight_full, id);
        view_stats(graph, weight_full, "weight_view")
        }
        _ => return false,
    };

    let weighted = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_weighted", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(normed), IrArg::Node(weight_view)],
        meta_with(&meta, shape.clone(), &dtype),
    );
    graph.anchor_to(weighted, id);

    let bias_view = match bias_arg {
        IrArg::Node(bias_id) => view_stats(graph, bias_id, "bias_view"),
        IrArg::None => {
            let bias_full = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_bias_full", name),
                "aten.full.default".to_string(),
            vec![i64_list(&[num_features]), IrArg::Float(0.0)],
            meta_with(&meta, vec![num_features], &dtype),
        );
        graph.anchor_to(bias_full, id);
        view_stats(graph, bias_full, "bias_view")
        }
        _ => return false,
    };

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(weighted), IrArg::Node(bias_view)],
        meta_with(&meta, shape, &dtype),
    );

    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.target.contains("getitem"))
        .filter(|node| node.args.first().and_then(IrArg::as_node) == Some(id))
        .map(|node| (node.id, node.args.get(1).and_then(IrArg::as_int).unwrap_or(0)))
        .collect();

    if getitem_users.is_empty() {
        graph.replace_and_erase(id, result);
        return true;
    }

    if graph
        .nodes
        .iter()
        .any(|node| node.alive
            && node.id != id
            && !node.target.contains("getitem")
            && node.args.iter().any(|arg| arg.as_node() == Some(id)))
    {
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
    let reshape_target = if graph.nodes[id].target.contains("native_group_norm") {
        "aten.view_copy.default"
    } else {
        "aten.reshape.default"
    };

    let grouped = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_grouped", name),
        reshape_target.to_string(),
        vec![IrArg::Node(x_id), i64_list(&grouped_shape)],
        meta_with(&meta, grouped_shape.clone(), &dtype),
    );
    graph.anchor_to(grouped, id);
    preserve_identity_layout(graph, grouped);

    let inv_n = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_inv_n", name),
        "aten.full.default".to_string(),
        vec![
            i64_list(&scalar_shape),
            IrArg::Float(1.0 / ((channels_per_group * hxw) as f64)),
        ],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(inv_n, id);
    preserve_identity_layout(graph, inv_n);

    let sum_x = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum", name),
        "aten.sum.dim_IntList".to_string(),
        vec![IrArg::Node(grouped), reduce_dims.clone(), IrArg::Bool(true)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(sum_x, id);
    preserve_identity_layout(graph, sum_x);

    let mean = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_mean", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(sum_x), IrArg::Node(inv_n)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(mean, id);
    preserve_identity_layout(graph, mean);

    let centered = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_centered", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(grouped), IrArg::Node(mean)],
        meta_with(&meta, grouped_shape.clone(), &dtype),
    );
    graph.anchor_to(centered, id);
    preserve_identity_layout(graph, centered);

    let squared = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_squared", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(centered), IrArg::Node(centered)],
        meta_with(&meta, grouped_shape.clone(), &dtype),
    );
    graph.anchor_to(squared, id);
    preserve_identity_layout(graph, squared);

    let sum_squared = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_sum_squared", name),
        "aten.sum.dim_IntList".to_string(),
        vec![IrArg::Node(squared), reduce_dims, IrArg::Bool(true)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(sum_squared, id);
    preserve_identity_layout(graph, sum_squared);

    let var = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_var", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(sum_squared), IrArg::Node(inv_n)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(var, id);
    preserve_identity_layout(graph, var);

    let eps_full = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_eps", name),
        "aten.full.default".to_string(),
        vec![i64_list(&scalar_shape), IrArg::Float(eps)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(eps_full, id);
    preserve_identity_layout(graph, eps_full);

    let var_eps = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_var_eps", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(var), IrArg::Node(eps_full)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(var_eps, id);
    preserve_identity_layout(graph, var_eps);

    let rsqrt = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_rsqrt", name),
        "aten.rsqrt.default".to_string(),
        vec![IrArg::Node(var_eps)],
        meta_with(&meta, scalar_shape.clone(), &dtype),
    );
    graph.anchor_to(rsqrt, id);
    preserve_identity_layout(graph, rsqrt);

    let normalized = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_normalized", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(centered), IrArg::Node(rsqrt)],
        meta_with(&meta, grouped_shape.clone(), &dtype),
    );
    graph.anchor_to(normalized, id);
    preserve_identity_layout(graph, normalized);

    let weighted = match weight_arg {
        IrArg::Node(weight_id) => {
            let weight_view = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_weight_view", name),
                reshape_target.to_string(),
                vec![IrArg::Node(weight_id), i64_list(&affine_shape)],
                meta_with(&meta, affine_shape.clone(), &dtype),
            );
            graph.anchor_to(weight_view, id);
            preserve_identity_layout(graph, weight_view);

            let weighted = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_weighted", name),
                "aten.mul.Tensor".to_string(),
                vec![IrArg::Node(normalized), IrArg::Node(weight_view)],
                meta_with(&meta, grouped_shape.clone(), &dtype),
            );
            graph.anchor_to(weighted, id);
            preserve_identity_layout(graph, weighted);
            weighted
        }
        IrArg::None => normalized,
        _ => return false,
    };

    let biased = match bias_arg {
        IrArg::Node(bias_id) => {
            let bias_view = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_bias_view", name),
                reshape_target.to_string(),
                vec![IrArg::Node(bias_id), i64_list(&affine_shape)],
                meta_with(&meta, affine_shape.clone(), &dtype),
            );
            graph.anchor_to(bias_view, id);
            preserve_identity_layout(graph, bias_view);

            let biased = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_biased", name),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(weighted), IrArg::Node(bias_view)],
                meta_with(&meta, grouped_shape.clone(), &dtype),
            );
            graph.anchor_to(biased, id);
            preserve_identity_layout(graph, biased);
            biased
        }
        IrArg::None => weighted,
        _ => return false,
    };

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_result", name),
        reshape_target.to_string(),
        vec![IrArg::Node(biased), i64_list(&shape)],
        meta_with(&meta, shape, &dtype),
    );
    graph.anchor_to(result, id);
    preserve_identity_layout(graph, result);

    let getitem_users: Vec<(NodeId, i64)> = graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.target.contains("getitem"))
        .filter(|node| node.args.first().and_then(IrArg::as_node) == Some(id))
        .map(|node| (node.id, node.args.get(1).and_then(IrArg::as_int).unwrap_or(0)))
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
            meta_with(&meta, u_shape, &dtype),
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
            meta_with(&orig_meta, slice_shape, &dtype),
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
            meta_with(&orig_meta, sq_shape, &dtype),
        );

        graph.replace_and_erase(*getitem_id, squeeze_id);
    }

    // Erase the unbind node itself (now has no users)
    graph.erase(id);
    true
}
