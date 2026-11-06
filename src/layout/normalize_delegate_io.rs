//! Port of rc4's `NormalizeDelegateIOLayoutPass`
//! (`backends/arm/_passes/normalize_delegate_io_layout_pass.py`), rc4's first
//! pass (`arm_pass_manager.py:544`).
//!
//! A runtime input whose exported value has a non-identity dim order takes its
//! physical shape, and a permute by the inverse order gives its users the
//! logical tensor back. An output with a non-identity dim order gets a permute
//! into that order. Everything between the boundaries is then logical.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeMeta, NodeOp};
use crate::fx_meta::{QParamMap, TensorFacts};
use crate::layout::ctx::LayoutCtx;
use crate::layout::shape::mark_contiguous;

const PERMUTE_COPY: &str = "aten.permute_copy.default";

pub(crate) struct IrNormalizeDelegateIOLayoutPass;

impl IrPass for IrNormalizeDelegateIOLayoutPass {
    fn name(&self) -> &str {
        "IrNormalizeDelegateIOLayout"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let mut modified = normalize_inputs(&mut ctx);
        modified |= normalize_outputs(&mut ctx);
        if !modified {
            return false;
        }
        // rc4's retrace re-derives every computed value from the now
        // contiguous runtime inputs; placeholders keep theirs, so a
        // channels-last parameter stays channels-last. Its fake kernels can
        // also carry such a parameter's strides into a value (a convolution);
        // RETAB does not model that, and nothing reads these orders later.
        for id in ctx.order().to_vec() {
            if ctx.node(id).op != NodeOp::Placeholder {
                mark_contiguous(ctx.meta_mut(id));
            }
        }
        ctx.finish();
        true
    }
}

/// `_normalize_input_layout` (`:46-77`).
fn normalize_inputs(ctx: &mut LayoutCtx<'_>) -> bool {
    let mut modified = false;
    for id in ctx.order().to_vec() {
        let node = ctx.node(id);
        // `is_param_node`: parameters, buffers and lifted constants carry
        // exported-program tensor data in the IR.
        if node.op != NodeOp::Placeholder || node.tensor.is_some() {
            continue;
        }
        let (Some(order), Some(shape)) = (
            node.meta.val_dim_order().map(<[usize]>::to_vec),
            ctx.graph().node_shape_vec(id),
        ) else {
            continue;
        };
        if is_identity(&order) || order.len() != shape.len() {
            continue;
        }

        // `input_fake.reshape(boundary_shape)` is contiguous.
        let boundary: Vec<i64> = order.iter().map(|&axis| shape[axis]).collect();
        let dtype = dtype_of(&node.meta);
        let meta = ctx.meta_mut(id);
        meta.set_tensor_facts(TensorFacts::new(boundary.clone(), dtype));
        meta.set_val_dim_order(identity(order.len()));

        let transpose = inverse(&order);
        let permute_meta = copied_meta(ctx.node(id).meta.clone(), permuted(&boundary, &transpose));
        let name = format!("{}_logical", ctx.node(id).name);
        let permute = ctx.inserting_after(id, |ctx| {
            ctx.create_node(
                NodeOp::CallFunction,
                name,
                PERMUTE_COPY,
                vec![IrArg::Node(id), i64_list_arg(&to_i64(&transpose))],
                HashMap::new(),
                permute_meta,
            )
        });
        for user in ctx.graph().live_user_ids(id) {
            if user != permute {
                ctx.replace_input_with(user, id, permute);
            }
        }
        modified = true;
    }
    modified
}

/// `_normalize_output_layout` and `_rewrite_output_arg` (`:79-129`).
fn normalize_outputs(ctx: &mut LayoutCtx<'_>) -> bool {
    let Some(output) = ctx.graph().live_output_ids().into_iter().next() else {
        return false;
    };
    let Some(outputs) = ctx.node(output).args.first().cloned() else {
        return false;
    };
    let (rewritten, modified) = rewrite_output_arg(ctx, &outputs);
    if modified {
        let mut args = ctx.node(output).args.clone();
        args[0] = rewritten;
        ctx.set_args(output, args);
    }
    modified
}

fn rewrite_output_arg(ctx: &mut LayoutCtx<'_>, arg: &IrArg) -> (IrArg, bool) {
    match arg {
        IrArg::Node(id) => {
            let id = *id;
            let node = ctx.node(id);
            let (Some(order), Some(shape)) = (
                node.meta.val_dim_order().map(<[usize]>::to_vec),
                ctx.graph().node_shape_vec(id),
            ) else {
                return (arg.clone(), false);
            };
            if is_identity(&order) || order.len() != shape.len() {
                return (arg.clone(), false);
            }
            let meta = copied_meta(node.meta.clone(), permuted(&shape, &order));
            let name = format!("{}_dim_order", node.name);
            // Each output permute goes directly after its producer, so a
            // producer returned twice gets the second permute first.
            let permute = ctx.inserting_after(id, |ctx| {
                ctx.create_node(
                    NodeOp::CallFunction,
                    name,
                    PERMUTE_COPY,
                    vec![IrArg::Node(id), i64_list_arg(&to_i64(&order))],
                    HashMap::new(),
                    meta,
                )
            });
            (IrArg::Node(permute), true)
        }
        IrArg::List(items) => {
            let mut modified = false;
            let rewritten = items
                .iter()
                .map(|item| {
                    let (item, changed) = rewrite_output_arg(ctx, item);
                    modified |= changed;
                    item
                })
                .collect();
            (IrArg::List(rewritten), modified)
        }
        _ => (arg.clone(), false),
    }
}

/// `create_node(..., from_node=node)` (`arm_pass_utils.py:173-182`): every key
/// of the source meta, with present qparam maps emptied. The value is
/// `permute_fake_tensor_metadata` (`arm_pass_utils.py:820-830`): a fresh
/// contiguous tensor, so the dim order is the identity.
fn copied_meta(mut meta: NodeMeta, shape: Vec<i64>) -> NodeMeta {
    if meta.input_qparams().is_some() {
        meta.set_empty_input_qparams();
    }
    if meta.output_qparams().is_some() {
        meta.set_output_qparams(QParamMap::new());
    }
    let rank = shape.len();
    let dtype = dtype_of(&meta);
    meta.set_tensor_facts(TensorFacts::new(shape, dtype));
    meta.set_val_dim_order(identity(rank));
    meta
}

fn dtype_of(meta: &NodeMeta) -> String {
    meta.val_dtype().unwrap_or("torch.float32").to_owned()
}

fn is_identity(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(index, &axis)| index == axis)
}

fn identity(rank: usize) -> Vec<usize> {
    (0..rank).collect()
}

/// `_inverse_permutation` (`:31-36`).
fn inverse(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; permutation.len()];
    for (index, &axis) in permutation.iter().enumerate() {
        inverse[axis] = index;
    }
    inverse
}

fn permuted(shape: &[i64], permutation: &[usize]) -> Vec<i64> {
    permutation.iter().map(|&axis| shape[axis]).collect()
}

fn to_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&dim| dim as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeId;
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    const NHWC: [usize; 4] = [0, 2, 3, 1];

    fn run(g: &mut LayoutGraph) -> bool {
        IrNormalizeDelegateIOLayoutPass.run(g.graph_mut())
    }

    fn set_dim_order(g: &mut LayoutGraph, id: NodeId, order: &[usize]) {
        g.graph_mut().node_mut(id).meta.set_val_dim_order(order.to_vec());
    }

    fn output_args(g: &LayoutGraph) -> Vec<NodeId> {
        let output = g.graph().live_output_ids()[0];
        match &g.graph().node(output).args[0] {
            IrArg::List(items) => items.iter().filter_map(IrArg::as_node).collect(),
            IrArg::Node(id) => vec![*id],
            _ => vec![],
        }
    }

    #[test]
    fn channels_last_input_gets_its_physical_shape_and_an_inverse_permute() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let relu = g.call("aten.relu.default", vec![node(x)]);
        g.output(&[relu]);
        let relu_name = g.name(relu);
        set_dim_order(&mut g, x, &NHWC);
        g.graph_mut()
            .node_mut(x)
            .meta
            .set_single_output_qparam(qparam(0.5, 0, "torch.int8"));

        assert!(run(&mut g));

        let (x, relu) = (g.id("x"), g.id(&relu_name));
        assert_eq!(g.shape(x), &[1, 4, 5, 3]);
        assert_eq!(g.graph().node(x).meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
        let permute = g.graph().node(relu).args[0].as_node().unwrap();
        assert_eq!(g.graph().node(permute).target, PERMUTE_COPY);
        assert_eq!(g.graph().node(permute).args[1].as_int_list(), Some(vec![0, 3, 1, 2]));
        assert_eq!(g.shape(permute), &[1, 3, 4, 5]);
        // `from_node` copies the meta and empties the qparam maps it has.
        let meta = &g.graph().node(permute).meta;
        assert!(meta.output_qparams().is_some_and(QParamMap::is_empty));
        assert!(meta.input_qparams().is_none());
        g.assert_targets(&[PERMUTE_COPY, "aten.relu.default"]);
    }

    #[test]
    fn an_input_permute_sits_directly_after_its_placeholder() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let y = g.input("y", &[1, 3, 4, 5]);
        let add = g.call("aten.add.Tensor", vec![node(x), node(y)]);
        g.output(&[add]);
        set_dim_order(&mut g, x, &NHWC);
        set_dim_order(&mut g, y, &NHWC);

        assert!(run(&mut g));

        let order: Vec<String> = g
            .graph()
            .live_nodes()
            .map(|node| node.name.clone())
            .collect();
        assert_eq!(order, ["x", "x_logical", "y", "y_logical", order[4].as_str(), order[5].as_str()]);
    }

    #[test]
    fn channels_last_outputs_get_a_permute_into_their_dim_order() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let relu = g.call("aten.relu.default", vec![node(x)]);
        let neg = g.call("aten.neg.default", vec![node(x)]);
        g.output(&[relu, neg]);
        set_dim_order(&mut g, relu, &NHWC);
        let neg_name = g.name(neg);

        assert!(run(&mut g));

        let neg = g.id(&neg_name);
        let outputs = output_args(&g);
        assert_eq!(g.graph().node(outputs[0]).target, PERMUTE_COPY);
        assert_eq!(g.graph().node(outputs[0]).args[1].as_int_list(), Some(vec![0, 2, 3, 1]));
        assert_eq!(g.shape(outputs[0]), &[1, 4, 5, 3]);
        assert_eq!(outputs[1], neg);
    }

    #[test]
    fn a_producer_returned_twice_gets_two_permutes_in_reverse_order() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let relu = g.call("aten.relu.default", vec![node(x)]);
        g.output(&[relu, relu]);
        set_dim_order(&mut g, relu, &NHWC);

        assert!(run(&mut g));

        let outputs = output_args(&g);
        assert_ne!(outputs[0], outputs[1]);
        let names: Vec<String> = g.graph().live_nodes().map(|node| node.name.clone()).collect();
        let first = g.graph().node(outputs[0]).name.clone();
        let second = g.graph().node(outputs[1]).name.clone();
        let position = |name: &str| names.iter().position(|n| n == name).unwrap();
        assert!(position(&second) < position(&first));
    }

    #[test]
    fn identity_dim_orders_are_left_alone() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 5]);
        let relu = g.call("aten.relu.default", vec![node(x)]);
        g.output(&[relu]);
        set_dim_order(&mut g, x, &[0, 1, 2, 3]);
        set_dim_order(&mut g, relu, &[0, 1, 2, 3]);

        assert!(!run(&mut g));
        g.assert_targets(&["aten.relu.default"]);
    }

    #[test]
    fn inverse_undoes_the_permutation() {
        assert_eq!(inverse(&NHWC), vec![0, 3, 1, 2]);
        assert_eq!(inverse(&inverse(&NHWC)), NHWC.to_vec());
    }
}
