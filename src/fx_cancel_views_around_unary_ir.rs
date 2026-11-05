//! Cancel a reshape pair around elementwise unary ops: `view(f(view(x)))`, where the
//! outer view restores `x`'s shape, becomes `f(x)`.
//!
//! A Linear lowers to `view -> 1x1 conv -> view`, so two Linears around an activation
//! (a transformer MLP, ConvNeXt's `Linear -> GELU -> Linear`) reshape the activation
//! back to rank 4 and straight back again. rc4's view propagation never creates that
//! pair, and Vela can then fuse the activation's LUT into the conv. This pass only
//! moves ops whose result doesn't depend on the tensor's shape: per-tensor pointwise
//! ops and per-tensor RESCALEs.

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp};
use crate::fx_target::is_view_like_target;

pub struct IrCancelViewsAroundUnaryPass;

/// Pointwise in the first argument; any other argument is a scalar or a rank-0/1
/// constant that doesn't follow the tensor's shape.
fn is_shape_agnostic_unary(graph: &IrGraph, node: &IrNode) -> bool {
    if node.op != NodeOp::CallFunction {
        return false;
    }
    let pointwise = matches!(
        node.target.as_str(),
        "aten.gelu.default"
            | "aten.relu.default"
            | "aten.relu6.default"
            | "aten.clamp.default"
            | "aten.hardtanh.default"
            | "aten.sigmoid.default"
            | "aten.tanh.default"
            | "aten.hardswish.default"
            | "aten.hardsigmoid.default"
            | "aten.silu.default"
            | "aten.leaky_relu.default"
            | "aten.elu.default"
            | "aten.exp.default"
            | "aten.log.default"
            | "aten.neg.default"
            | "aten.abs.default"
            | "aten.sqrt.default"
            | "aten.rsqrt.default"
            | "aten.reciprocal.default"
            | "aten.erf.default"
    ) || (node.is_rescale()
        && matches!(node.args.get(2), Some(IrArg::List(scales)) if scales.len() == 1));
    if !pointwise {
        return false;
    }
    let rank_at_most_1 = |id: NodeId| graph.node_shape(id).is_some_and(|shape| shape.len() <= 1);
    node.args.iter().skip(1).all(|arg| match arg {
        IrArg::Node(id) => rank_at_most_1(*id),
        IrArg::List(items) => items
            .iter()
            .all(|item| item.as_node().is_none_or(rank_at_most_1)),
        _ => true,
    })
}

/// For `outer = view(f_k(...f_1(inner_view(x))))` with each `f_i` used only by the
/// next: the chain `[f_1, ..., f_k]` and `x`, when `outer` restores `x`'s shape.
fn match_pair(graph: &IrGraph, outer: NodeId) -> Option<(Vec<NodeId>, NodeId)> {
    let target_shape = graph.node_shape_vec(outer)?;
    let mut chain = Vec::new();
    let mut current = graph.node(outer).first_node_arg()?;
    while is_shape_agnostic_unary(graph, graph.node(current)) {
        if graph.live_user_ids(current).len() != 1 {
            return None;
        }
        chain.push(current);
        current = graph.node(current).first_node_arg()?;
    }
    let inner = graph.node(current);
    if chain.is_empty() || inner.op != NodeOp::CallFunction || !is_view_like_target(&inner.target)
    {
        return None;
    }
    let source = inner.first_node_arg()?;
    if graph.node_shape_vec(source)? != target_shape {
        return None;
    }
    chain.reverse();
    Some((chain, source))
}

impl IrPass for IrCancelViewsAroundUnaryPass {
    fn name(&self) -> &str {
        "IrCancelViewsAroundUnary"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for outer in graph.live_call_function_ids() {
            if !graph.nodes[outer].alive || !is_view_like_target(&graph.nodes[outer].target) {
                continue;
            }
            let Some((chain, source)) = match_pair(graph, outer) else {
                continue;
            };
            let shape = graph.node_shape_vec(source).expect("matched on its shape");
            let inner = graph.node(chain[0]).first_node_arg().expect("matched view");
            graph.set_arg(chain[0], 0, IrArg::Node(source));
            for &id in &chain {
                let meta = graph.node(id).meta.cloned_with_same_dtype_or(shape.clone(), "torch.float32");
                graph.node_mut(id).meta = meta;
            }
            graph.replace_and_erase(outer, *chain.last().expect("non-empty chain"));
            if !graph.has_live_users(inner) {
                graph.erase(inner);
            }
            modified = true;
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::test_ir::{f32_call, f32_placeholder, output};

    fn view(graph: &mut IrGraph, name: &str, input: NodeId, shape: &[i64]) -> NodeId {
        f32_call(
            graph,
            name,
            "aten.view_copy.default",
            vec![IrArg::Node(input), i64_list_arg(shape)],
            shape.to_vec(),
        )
    }

    #[test]
    fn cancels_the_reshape_pair_around_an_activation() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [64, 32, 1, 1]);
        let up = view(&mut graph, "up", x, &[1, 8, 8, 32]);
        let gelu = f32_call(&mut graph, "gelu", "aten.gelu.default", vec![IrArg::Node(up)], [1, 8, 8, 32]);
        let down = view(&mut graph, "down", gelu, &[64, 32, 1, 1]);
        let out = output(&mut graph, down);

        assert!(IrCancelViewsAroundUnaryPass.run(&mut graph));
        assert_eq!(graph.node(gelu).first_node_arg(), Some(x));
        assert_eq!(graph.node_shape_vec(gelu), Some(vec![64, 32, 1, 1]));
        assert_eq!(graph.node(out).first_node_arg(), Some(gelu));
        assert!(!graph.node(up).alive && !graph.node(down).alive);
    }

    #[test]
    fn keeps_pairs_that_change_the_shape_or_share_the_activation() {
        // The outer view doesn't restore x's shape.
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [64, 32, 1, 1]);
        let up = view(&mut graph, "up", x, &[1, 8, 8, 32]);
        let relu = f32_call(&mut graph, "relu", "aten.relu.default", vec![IrArg::Node(up)], [1, 8, 8, 32]);
        let down = view(&mut graph, "down", relu, &[8, 256]);
        output(&mut graph, down);
        assert!(!IrCancelViewsAroundUnaryPass.run(&mut graph));

        // The activation has another reader that needs the rank-4 shape.
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [64, 32, 1, 1]);
        let up = view(&mut graph, "up", x, &[1, 8, 8, 32]);
        let relu = f32_call(&mut graph, "relu", "aten.relu.default", vec![IrArg::Node(up)], [1, 8, 8, 32]);
        let down = view(&mut graph, "down", relu, &[64, 32, 1, 1]);
        let other = f32_call(&mut graph, "other", "aten.neg.default", vec![IrArg::Node(relu)], [1, 8, 8, 32]);
        output(&mut graph, down);
        output(&mut graph, other);
        assert!(!IrCancelViewsAroundUnaryPass.run(&mut graph));
    }
}
