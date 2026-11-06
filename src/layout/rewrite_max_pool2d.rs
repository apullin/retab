//! Port of rc4's `RewriteMaxPool2dPass`
//! (`backends/arm/_passes/rewrite_max_pool2d_pass.py`, line numbers below are
//! that file's), phase-6 slot 2 (`arm_pass_manager.py:718`), with
//! RemoveGetItemPass's effect folded in (APU-135 P2 S7, design §3.2,
//! critique A3/B3).
//!
//! Each max pool becomes
//!
//! ```text
//! x -> permute_copy([0, 2, 3, 1]) -> MAX_POOL2D(x', kernel, stride, pad) -> permute_copy([0, 3, 1, 2])
//! ```
//!
//! **Which nodes.** rc4 converts every `max_pool2d_with_indices` + `getitem(0)`
//! into `max_pool2d` at `arm_pass_manager.py:605`
//! (`backends/transforms/remove_getitem_op.py:16-70`); RETAB has no such pass
//! and keeps the tuple-valued pool up to here. So the port rewrites both
//! `aten.max_pool2d.default` and `aten.max_pool2d_with_indices.default` whose
//! only user is `getitem(.., 0)`. For the second it does what RemoveGetItem
//! did: the new nodes take the pool node's meta, which has input qparams only
//! (the getitem holds the output qparams, which rc4 drops), its users move
//! from the getitem to the output permute, and the getitem and the pool are
//! erased. RETAB's tuple-valued pool has no val, so the values come from x and
//! the arguments, and the getitem's val is only a cross-check. Any other
//! user pattern is RemoveGetItem's AssertionError.
//!
//! **Arguments** (`:79-107`): kernel `args[1]` as given (rc4 indexes it, so it
//! must hold two values), stride `to_2tuple(args[2])` or the kernel when it is
//! falsy, padding `to_2tuple(args[3])` or `(0, 0)`, dilation `to_2tuple(args[4])`
//! or `(1, 1)`, `ceil_mode` `args[5]`. RETAB's `_with_indices` pools keep the
//! edge args with trailing defaults omitted, while rc4's DecomposeMaxPool2dPass
//! spells them out before this pass; both read the same. The TOSA pad is
//! `[p0, p0', p1, p1']`, each post pad from [`adjust_pooling_pad_if_needed`]
//! on the input's val.
//!
//! **Meta** (`:150-170`): `call_operator(..., meta, updated=True)` gives all
//! three nodes the pool's meta, qparams included ([`meta::call_operator`]).
//! The permutes hold their input's dtype (RETAB's, design F2) and MAX_POOL2D
//! the input dtype rc4 sees (`shape/tosa.rs`).
//!
//! The adaptive branch (`:109-147`, MAX_POOL2D_ADAPTIVE) needs symbolic shapes
//! and TOSA 1.1, so it never fires for RETAB (design F1).
//!
//! **Failures.** Where rc4 raises, the port asks for the stock fallback and
//! stops: a dilated pool (`:88-95`; DecomposeMaxPool2dPass should have run), an
//! input that isn't rank 4, arguments that don't unpack as two values, a
//! negative output extent, and the user patterns RemoveGetItem rejects. Like
//! the avg-pool rewrite it also declines a keyword argument with a value
//! (rc4 reads `args` only) and an output shape that differs from the pool's
//! (or the getitem's) val.
//!
//! RETAB's IR spells the TOSA target `backend.tosa.MAX_POOL2D.default`.
//!
//! [`adjust_pooling_pad_if_needed`]: crate::layout::pool::adjust_pooling_pad_if_needed
//! [`meta::call_operator`]: crate::layout::meta::call_operator

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::dtype::operand_dtype;
use crate::layout::meta;
use crate::layout::pool::{
    adjust_pooling_pad_if_needed, ceil_mode_arg, has_no_valued_kwargs, stride_or_kernel,
    to_2tuple, NHWC_INVERSE_ORDER, NHWC_ORDER,
};
use crate::layout::shape::permute_shape;
use crate::layout::shape::tosa::pool2d_shape;

const MAX_POOL2D: &str = "aten.max_pool2d.default";
const MAX_POOL2D_WITH_INDICES: &str = "aten.max_pool2d_with_indices.default";
const TOSA_MAX_POOL2D: &str = "backend.tosa.MAX_POOL2D.default";
const PERMUTE_COPY: &str = "aten.permute_copy.default";

fn is_target(target: &str) -> bool {
    matches!(target, MAX_POOL2D | MAX_POOL2D_WITH_INDICES)
}

/// rc4's RewriteMaxPool2dPass, after its RemoveGetItemPass; see the module docs.
pub(crate) struct IrRewriteMaxPool2dPass;

impl IrPass for IrRewriteMaxPool2dPass {
    fn name(&self) -> &str {
        "IrRewriteMaxPool2d"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
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
            match plan(ctx.graph(), id) {
                Ok(plan) => {
                    rewrite(&mut ctx, id, &plan);
                    modified = true;
                }
                Err(reason) => {
                    crate::stock_fallback::request(format!(
                        "IrRewriteMaxPool2d: {}: {reason}",
                        ctx.node(id).name
                    ));
                    break;
                }
            }
        }
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// Everything one rewrite needs, worked out before the graph changes.
#[derive(Debug)]
struct Plan {
    x: NodeId,
    /// The `getitem(.., 0)` of a `max_pool2d_with_indices` pool.
    getitem: Option<NodeId>,
    kernel: [i64; 2],
    stride: [i64; 2],
    pad: [i64; 4],
    /// Values: the input permute, MAX_POOL2D and the output permute.
    x_nhwc: TensorFacts,
    pooled: TensorFacts,
    output: TensorFacts,
}

/// RemoveGetItemPass's checks (`remove_getitem_op.py:32-62`): the tuple-valued
/// pool's only user is `getitem(pool, 0)`.
fn getitem_of(graph: &IrGraph, id: NodeId) -> Result<NodeId, String> {
    let users = graph.live_user_ids(id);
    let &[user] = users.as_slice() else {
        return Err(format!(
            "max_pool2d_with_indices has {} users; RemoveGetItem needs one",
            users.len()
        ));
    };
    let getitem = graph.node(user);
    if !getitem.is_getitem() {
        return Err(format!(
            "max_pool2d_with_indices is read by {}, not a getitem",
            getitem.target
        ));
    }
    match getitem.args.get(1) {
        Some(IrArg::Int(0)) => Ok(user),
        index => Err(format!(
            "getitem index {index:?}; TOSA MAX_POOL2D gives only the values"
        )),
    }
}

/// `RewriteMaxPool2dPass.call_operator` (`:76-187`, static shapes) up to the
/// graph edits.
fn plan(graph: &IrGraph, id: NodeId) -> Result<Plan, String> {
    let node = graph.node(id);
    let getitem = if node.target == MAX_POOL2D_WITH_INDICES {
        Some(getitem_of(graph, id)?)
    } else {
        None
    };
    if !has_no_valued_kwargs(node) {
        return Err("keyword arguments, which rc4 ignores".into());
    }
    let x = node
        .args
        .first()
        .and_then(IrArg::as_node)
        .ok_or("no input node")?;
    let x_facts = graph
        .node(x)
        .meta
        .tensor_facts()
        .ok_or("the input has no value")?;
    let &[_, _, height, width] = x_facts.shape.as_slice() else {
        return Err(format!("a rank-{} input; rc4 reads shape[2:4]", x_facts.rank()));
    };

    // `kernel = args[1]`: rc4 indexes it as `kernel[0]`, `kernel[1]`.
    let kernel: [i64; 2] = match node.args.get(1).and_then(IrArg::as_int_list) {
        Some(kernel) => kernel
            .try_into()
            .map_err(|kernel| format!("kernel {kernel:?} does not hold two values"))?,
        None => return Err("the kernel is not an int list".into()),
    };
    let stride = stride_or_kernel(node, kernel).ok_or("the stride is not one or two ints")?;
    let padding = match node.args.get(3) {
        None => [0, 0],
        Some(arg) => to_2tuple(arg).ok_or("the padding is not one or two ints")?,
    };
    let dilation = match node.args.get(4) {
        None => [1, 1],
        Some(arg) => to_2tuple(arg).ok_or("the dilation is not one or two ints")?,
    };
    let ceil_mode = ceil_mode_arg(node, 5).ok_or("ceil_mode is not a bool")?;
    // `:88-95`.
    if dilation != [1, 1] {
        return Err(format!(
            "dilation {dilation:?}: tosa.MAX_POOL2D has none (has IrDecomposeMaxPoolDilation run?)"
        ));
    }
    if stride.iter().any(|&s| s < 1) {
        return Err(format!("stride {stride:?}"));
    }
    // `:97-107`.
    let pad = [
        padding[0],
        adjust_pooling_pad_if_needed(height, kernel[0], stride[0], padding[0], ceil_mode),
        padding[1],
        adjust_pooling_pad_if_needed(width, kernel[1], stride[1], padding[1], ceil_mode),
    ];

    let input_dtype = operand_dtype(graph, id, 0).ok_or("the input has no dtype")?;
    let x_nhwc = TensorFacts::new(
        permute_shape(&x_facts.shape, &NHWC_ORDER).map_err(|e| e.to_string())?,
        x_facts.dtype.clone(),
    );
    let pooled = TensorFacts::new(
        pool2d_shape(&x_nhwc.shape, &kernel, &stride, &pad).map_err(|e| e.to_string())?,
        input_dtype,
    );
    let output = TensorFacts::new(
        permute_shape(&pooled.shape, &NHWC_INVERSE_ORDER).map_err(|e| e.to_string())?,
        pooled.dtype.clone(),
    );
    // The pool's own val, or the getitem's: the tuple-valued pool has none.
    let value_node = getitem.unwrap_or(id);
    if let Some(expected) = graph.node(value_node).meta.val_shape() {
        if expected != output.shape.as_slice() {
            return Err(format!(
                "the rewrite gives {:?} where the pool's value is {expected:?}",
                output.shape
            ));
        }
    }
    Ok(Plan {
        x,
        getitem,
        kernel,
        stride,
        pad,
        x_nhwc,
        pooled,
        output,
    })
}

/// The graph edits: the three nodes go where the pool was (RemoveGetItem puts
/// its `max_pool2d` directly before the tuple-valued pool), the uses move to
/// the output permute, and the getitem and the pool are erased.
fn rewrite(ctx: &mut LayoutCtx<'_>, id: NodeId, plan: &Plan) {
    let source = ctx.node(id).meta.clone();
    let name = ctx.node(id).name.clone();
    let post = ctx.inserting_before(id, |ctx| {
        let pre = ctx.call_function(
            format!("{name}_nhwc"),
            PERMUTE_COPY,
            vec![IrArg::Node(plan.x), i64_list_arg(&NHWC_ORDER)],
            meta::call_operator(&source, plan.x_nhwc.clone()),
        );
        let pool = ctx.call_function(
            format!("{name}_tosa"),
            TOSA_MAX_POOL2D,
            vec![
                IrArg::Node(pre),
                i64_list_arg(&plan.kernel),
                i64_list_arg(&plan.stride),
                i64_list_arg(&plan.pad),
            ],
            meta::call_operator(&source, plan.pooled.clone()),
        );
        ctx.call_function(
            format!("{name}_nchw"),
            PERMUTE_COPY,
            vec![IrArg::Node(pool), i64_list_arg(&NHWC_INVERSE_ORDER)],
            meta::call_operator(&source, plan.output.clone()),
        )
    });
    match plan.getitem {
        Some(getitem) => {
            ctx.replace_all_uses_with(getitem, post);
            ctx.erase_node(getitem);
        }
        None => {
            ctx.replace_all_uses_with(id, post);
        }
    }
    ctx.erase_node(id);
}
