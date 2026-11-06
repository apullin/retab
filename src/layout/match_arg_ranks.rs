//! Port of rc4's `MatchArgRanksPass`
//! (`backends/arm/_passes/match_arg_ranks_pass.py`).
//!
//! For the broadcasting ops in [`TARGETS`], a lower-rank node argument gets a
//! `view_copy` that prepends size-1 dims up to the highest argument rank:
//! `(4, 3, 2)`, `(2)` and `(3, 1)` become `(4, 3, 2)`, `(1, 1, 2)` and
//! `(1, 3, 1)`.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::shape::mark_contiguous;

const VIEW_COPY: &str = "aten.view_copy.default";

/// rc4's `target_ops` (`:48-87`), with both of RETAB's TOSA spellings.
const TARGETS: &[&str] = &[
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.div.Tensor_mode",
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_left_shift.Tensor",
    "aten.eq.Tensor",
    "aten.gt.Tensor",
    "aten.ge.Tensor",
    "aten.lt.Tensor",
    "aten.le.Tensor",
    "aten.ne.Tensor",
    "aten.pow.Tensor_Tensor",
    "aten.remainder.Tensor",
    "aten.where.self",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_xor.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.maximum.default",
    "aten.minimum.default",
    "backend.tosa.ADD.default",
    "backend.tosa.ARITHMETIC_RIGHT_SHIFT.default",
    "backend.tosa.BITWISE_AND.default",
    "backend.tosa.BITWISE_OR.default",
    "backend.tosa.BITWISE_XOR.default",
    "backend.tosa.EQUAL.default",
    "backend.tosa.GREATER.default",
    "backend.tosa.GREATER_EQUAL.default",
    "backend.tosa.LOGICAL_AND.default",
    "backend.tosa.LOGICAL_LEFT_SHIFT.default",
    "backend.tosa.LOGICAL_OR.default",
    "backend.tosa.LOGICAL_XOR.default",
    "backend.tosa.MAXIMUM.default",
    "backend.tosa.MINIMUM.default",
    "backend.tosa.MUL.default",
    "backend.tosa.POW.default",
    "backend.tosa.SUB.default",
];

fn is_target(target: &str) -> bool {
    let target = target.strip_prefix("tosa.").map_or(target, |rest| {
        // `tosa.ADD.default` is RETAB's other spelling of `backend.tosa.ADD.default`.
        TARGETS
            .iter()
            .find(|known| known.strip_prefix("backend.tosa.") == Some(rest))
            .copied()
            .unwrap_or(target)
    });
    TARGETS.contains(&target)
}

pub(crate) struct IrMatchArgRanksPass;

impl IrPass for IrMatchArgRanksPass {
    fn name(&self) -> &str {
        "IrMatchArgRanks"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        // `ArmOpTargetedPass` skips graphs without a target op.
        if !graph
            .live_call_function_ids()
            .into_iter()
            .any(|id| is_target(&graph.node(id).target))
        {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let mut modified = false;
        for id in ctx.order().to_vec() {
            let node = ctx.node(id);
            if node.op != NodeOp::CallFunction || !is_target(&node.target) {
                continue;
            }
            // Only top-level positional node args count (`:111-127`).
            let args: Vec<NodeId> = node.args.iter().filter_map(IrArg::as_node).collect();
            let Some(shapes) = args
                .iter()
                .map(|&arg| ctx.graph().node_shape_vec(arg))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let max_rank = shapes.iter().map(Vec::len).max().unwrap_or(0);

            let mut matched: Vec<NodeId> = Vec::new();
            for (arg, shape) in args.into_iter().zip(shapes) {
                // rc4 visits a repeated arg again, but its first view already
                // replaced every use (`replace_input_with`), so the second view
                // would be dead. Skipping it gives the same live graph.
                if shape.len() == max_rank || matched.contains(&arg) {
                    continue;
                }
                matched.push(arg);
                let view = insert_rank_view(&mut ctx, id, arg, &shape, max_rank);
                ctx.replace_input_with(id, arg, view);
                modified = true;
            }
        }
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// `_match_op_rank` (`:89-104`): a `view_copy` directly before `node`. rc4's
/// `create_node` gets no `from_node`, so the view's meta holds only the value
/// the retrace gives it: the arg's dtype at the matched rank, without qparams.
fn insert_rank_view(
    ctx: &mut LayoutCtx<'_>,
    node: NodeId,
    arg: NodeId,
    shape: &[i64],
    max_rank: usize,
) -> NodeId {
    let mut new_shape = vec![1; max_rank - shape.len()];
    new_shape.extend_from_slice(shape);
    let dtype = ctx
        .node(arg)
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_owned();
    let mut meta = NodeMeta::default();
    meta.set_tensor_facts(TensorFacts::new(new_shape.clone(), dtype));
    mark_contiguous(&mut meta);
    let name = format!("{}_rank_matched", ctx.node(arg).name);
    ctx.inserting_before(node, |ctx| {
        ctx.create_node(
            NodeOp::CallFunction,
            name,
            VIEW_COPY,
            vec![IrArg::Node(arg), i64_list_arg(&new_shape)],
            HashMap::new(),
            meta,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph) -> bool {
        IrMatchArgRanksPass.run(g.graph_mut())
    }

    fn arg_shapes(g: &LayoutGraph, id: NodeId) -> Vec<Vec<i64>> {
        g.graph()
            .node(id)
            .args
            .iter()
            .filter_map(IrArg::as_node)
            .map(|arg| g.shape(arg).to_vec())
            .collect()
    }

    /// The docstring example (`:28-39`), on `where.self`.
    #[test]
    fn lower_rank_args_get_leading_unit_dims() {
        let mut g = LayoutGraph::new();
        let a = g.typed_input("a", &[4, 3, 2], "torch.bool");
        let b = g.input("b", &[2]);
        let c = g.input("c", &[3, 1]);
        let w = g.call("aten.where.self", vec![node(a), node(b), node(c)]);
        g.output(&[w]);
        let w_name = g.name(w);

        assert!(run(&mut g));

        let w = g.id(&w_name);

        assert_eq!(arg_shapes(&g, w), vec![vec![4, 3, 2], vec![1, 1, 2], vec![1, 3, 1]]);
        g.assert_targets(&[VIEW_COPY, VIEW_COPY, "aten.where.self"]);
        let views: Vec<NodeId> = g.graph().node(w).args[1..]
            .iter()
            .filter_map(IrArg::as_node)
            .collect();
        for view in views {
            assert!(!g.graph().node(view).meta.has_qparam_metadata());
        }
    }

    /// Port of rc4's `test_match_arg_ranks_pass.py`: a rank-1 operand of `add`
    /// becomes rank 4.
    #[test]
    fn add_with_a_rank1_operand() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let y = g.input("y", &[4]);
        let add = g.call("aten.add.Tensor", vec![node(x), node(y)]);
        g.output(&[add]);
        let add_name = g.name(add);

        assert!(run(&mut g));

        let add = g.id(&add_name);
        assert_eq!(arg_shapes(&g, add), vec![vec![1, 2, 3, 4], vec![1, 1, 1, 4]]);
    }

    #[test]
    fn equal_ranks_scalars_and_other_ops_are_left_alone() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let y = g.input("y", &[1, 3]);
        let add = g.call("aten.add.Tensor", vec![node(x), node(y)]);
        let scaled = g.call("aten.mul.Tensor", vec![node(add), IrArg::Float(2.0)]);
        let z = g.input("z", &[3]);
        let cat = g.call("aten.cat.default", vec![IrArg::List(vec![node(scaled), node(scaled)]), IrArg::Int(0)]);
        let _ = (z, cat);
        g.output(&[cat]);

        assert!(!run(&mut g));
    }

    #[test]
    fn a_repeated_arg_gets_one_view() {
        let mut g = LayoutGraph::new();
        let c = g.typed_input("c", &[2, 3], "torch.bool");
        let x = g.input("x", &[3]);
        let w = g.call("aten.where.self", vec![node(c), node(x), node(x)]);
        g.output(&[w]);
        let w_name = g.name(w);

        assert!(run(&mut g));

        let w = g.id(&w_name);

        g.assert_targets(&[VIEW_COPY, "aten.where.self"]);
        let args = &g.graph().node(w).args;
        assert_eq!(args[1].as_node(), args[2].as_node());
    }

    #[test]
    fn both_tosa_spellings_match() {
        assert!(is_target("backend.tosa.MUL.default"));
        assert!(is_target("tosa.MUL.default"));
        assert!(!is_target("tosa.RESCALE.default"));
        assert!(!is_target("aten.cat.default"));
    }
}
