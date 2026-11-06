//! Port of rc4's `RewriteAvgPool2dPass`
//! (`backends/arm/_passes/rewrite_avg_pool2d_pass.py`, line numbers below are
//! that file's), which runs at `arm_pass_manager.py:698`, right after
//! SizeAdjustInputPass (APU-135 P2 S7, design §3.1).
//!
//! Each `aten.avg_pool2d` becomes
//!
//! ```text
//! full((1,), input_zp)   full((1,), output_zp)
//! x -> permute_copy([0, 2, 3, 1]) -> AVG_POOL2D(x', in_zp, out_zp, kernel, stride, pad, acc)
//!   -> permute_copy([0, 3, 1, 2])
//! ```
//!
//! - **Arguments** (`:36-55`): kernel `to_2tuple(args[1])`, stride
//!   `to_2tuple(args[2])` or the kernel when it is falsy, padding
//!   `to_2tuple(args[3])` or `(0, 0)`, `ceil_mode` `args[4]`. Arguments 5 and 6
//!   (`count_include_pad`, `divisor_override`) are ignored, as in rc4: the
//!   avg-pool decomposition before (IrDecomposeAvgPoolPad) has already
//!   normalised them. The TOSA pad is `[ph, ph', pw, pw']`, with each post pad
//!   from [`adjust_pooling_pad_if_needed`] on the input's val.
//! - **Zero points** (`:57-67`): rc4's `call_scalar` materializes each as
//!   `full((1,), zp, dtype=val.dtype, device=val.device)` carrying the pool's
//!   meta (`arm_pass.py:177-205`), input zero point first. The port does the
//!   same (critique B2, A7): IrMaterializeConstantsPass, at rc4's
//!   ComputeConstantOpsAOTPass slot right after this pass (`:699`), or else
//!   IrFuseZeroInputConstantsEffectPass's effect at ingest, then folds each
//!   `full` into a constant quantized with the pool's output qparams, which is
//!   rc4's quirk (`fuse_constant_ops_pass.py:327-332`). Because the zero points
//!   are call nodes, the rewrite needs no constant of its own and works
//!   unchanged in a control-flow region. `dtype` is the pool's val dtype as
//!   rc4's fake tensor has it ([`dtype::value_dtype`]); the device, which
//!   RETAB's importer reads as None, is left out.
//! - **Accumulator** (`:69-75`): int32 for an int8 or int16 input, float16 for
//!   fp8 and float32 otherwise, from the input dtype rc4's fake tensor has
//!   ([`dtype::operand_dtype`]).
//! - **Meta** (`:77-106`): every new node is built with
//!   `call_operator(..., meta, updated=True)`, so it copies the pool's whole
//!   meta, qparams included ([`meta::call_operator`]). Each holds the value
//!   rc4's fake kernel gives it: the permutes their input's dtype (RETAB's,
//!   design F2), AVG_POOL2D the input dtype rc4 sees, which keeps it typed when
//!   RETAB holds the input as float32 (`shape/tosa.rs`).
//!
//! The rewrite fails like rc4 where rc4 raises: a per-channel zero point
//! (`get_zp_per_tensor`), an input that isn't rank 4 (`x.data.shape[3]`),
//! arguments that don't unpack as two values, and AVG_POOL2D's own checks
//! (zero-point dtypes, the output extent). It then asks for the stock
//! fallback and stops; nodes it already rewrote stay rewritten. It also
//! declines a pool with a keyword argument that has a value, which rc4 would
//! drop silently (`call_operator` reads `args` only) and exported graphs never
//! carry. An output shape that differs from the pool's val, which rc4's
//! arithmetic rules out, is declined the same way rather than trusted.
//!
//! RETAB's IR spells the TOSA target `backend.tosa.AVG_POOL2D.default`, as its
//! other passes create TOSA ops (`IrGraph::add_rescale`).
//!
//! [`adjust_pooling_pad_if_needed`]: crate::layout::pool::adjust_pooling_pad_if_needed
//! [`dtype::value_dtype`]: crate::layout::dtype::value_dtype
//! [`dtype::operand_dtype`]: crate::layout::dtype::operand_dtype
//! [`meta::call_operator`]: crate::layout::meta::call_operator

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::dtype::{operand_dtype, value_dtype};
use crate::layout::meta;
use crate::layout::pool::{
    adjust_pooling_pad_if_needed, ceil_mode_arg, has_no_valued_kwargs, stride_or_kernel,
    to_2tuple, NHWC_INVERSE_ORDER, NHWC_ORDER,
};
use crate::layout::shape::permute_shape;
use crate::layout::shape::tosa::pool2d_shape;

const AVG_POOL2D: &str = "aten.avg_pool2d.default";
const TOSA_AVG_POOL2D: &str = "backend.tosa.AVG_POOL2D.default";
const PERMUTE_COPY: &str = "aten.permute_copy.default";
const FULL: &str = "aten.full.default";

/// rc4's RewriteAvgPool2dPass; see the module docs.
pub(crate) struct IrRewriteAvgPool2dPass;

impl IrPass for IrRewriteAvgPool2dPass {
    fn name(&self) -> &str {
        "IrRewriteAvgPool2d"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        // `ArmOpTargetedPass` skips a graph without a target op.
        if !graph
            .live_call_function_ids()
            .into_iter()
            .any(|id| graph.node(id).target == AVG_POOL2D)
        {
            return false;
        }
        let mut ctx = LayoutCtx::new(graph);
        let mut modified = false;
        for id in ctx.order().to_vec() {
            let node = ctx.node(id);
            if node.op != NodeOp::CallFunction || node.target != AVG_POOL2D {
                continue;
            }
            match plan(ctx.graph(), id) {
                Ok(plan) => {
                    rewrite(&mut ctx, id, &plan);
                    modified = true;
                }
                Err(reason) => {
                    crate::stock_fallback::request(format!(
                        "IrRewriteAvgPool2d: {}: {reason}",
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
    kernel: [i64; 2],
    stride: [i64; 2],
    pad: [i64; 4],
    input_zp: i64,
    output_zp: i64,
    /// `meta["val"].dtype` of the pool, the zero points' dtype.
    zp_dtype: String,
    acc_type: &'static str,
    /// Values: the input permute, AVG_POOL2D and the output permute.
    x_nhwc: TensorFacts,
    pooled: TensorFacts,
    output: TensorFacts,
}

/// `RewriteAvgPool2dPass.call_operator` (`:30-106`) up to the graph edits.
fn plan(graph: &IrGraph, id: NodeId) -> Result<Plan, String> {
    let node = graph.node(id);
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

    let kernel = node
        .args
        .get(1)
        .and_then(to_2tuple)
        .ok_or("the kernel is not one or two ints")?;
    let stride = stride_or_kernel(node, kernel).ok_or("the stride is not one or two ints")?;
    let [pad_h, pad_w] = match node.args.get(3) {
        None => [0, 0],
        Some(arg) => to_2tuple(arg).ok_or("the padding is not one or two ints")?,
    };
    let ceil_mode = ceil_mode_arg(node, 4).ok_or("ceil_mode is not a bool")?;
    if stride.iter().any(|&s| s < 1) {
        return Err(format!("stride {stride:?}"));
    }
    // `:44-55`.
    let pad = [
        pad_h,
        adjust_pooling_pad_if_needed(height, kernel[0], stride[0], pad_h, ceil_mode),
        pad_w,
        adjust_pooling_pad_if_needed(width, kernel[1], stride[1], pad_w, ceil_mode),
    ];

    // `:57-67`: `get_zp_per_tensor` raises for a per-channel zero point.
    let zero_point = |qparam: Option<&crate::fx_ir::QParam>, which: &str| match qparam {
        None => Ok(0),
        Some(qparam) if qparam.is_per_channel() => {
            Err(format!("a per-channel {which} qparam"))
        }
        Some(qparam) => Ok(qparam.zp),
    };
    let input_zp = zero_point(
        node.meta.input_qparams().and_then(|map| map.get(&0)),
        "input",
    )?;
    let output_zp = zero_point(
        node.meta.output_qparams().and_then(|map| map.get(&0)),
        "output",
    )?;
    let zp_dtype = value_dtype(graph, id).ok_or("the pool has no value")?;

    // `:69-75`.
    let input_dtype = operand_dtype(graph, id, 0).ok_or("the input has no dtype")?;
    let acc_type = match input_dtype.as_str() {
        "torch.int8" | "torch.int16" => "torch.int32",
        "torch.float8_e4m3fn" | "torch.float8_e5m2" => "torch.float16",
        _ => "torch.float32",
    };
    // `validate_avg_pool2d_dtype` (`tosa/dialect/ops/avg_pool2d.py:40-75`):
    // both zero points must have the input's dtype.
    if zp_dtype != input_dtype {
        return Err(format!(
            "zero points of {zp_dtype} for an input of {input_dtype}"
        ));
    }

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
    if let Some(expected) = node.meta.val_shape() {
        if expected != output.shape.as_slice() {
            return Err(format!(
                "the rewrite gives {:?} where the pool's value is {expected:?}",
                output.shape
            ));
        }
    }
    Ok(Plan {
        x,
        kernel,
        stride,
        pad,
        input_zp,
        output_zp,
        zp_dtype,
        acc_type,
        x_nhwc,
        pooled,
        output,
    })
}

/// The graph edits of `call_operator`: the nodes rc4's retrace emits in the
/// pool's place, in its order, then the pool's uses move to the output permute
/// and the pool goes.
fn rewrite(ctx: &mut LayoutCtx<'_>, id: NodeId, plan: &Plan) {
    let source = ctx.node(id).meta.clone();
    let name = ctx.node(id).name.clone();
    let post = ctx.inserting_before(id, |ctx| {
        let mut zero_point = |which: &str, value: i64| {
            // `call_scalar` (`arm_pass.py:177-205`).
            let kwargs = HashMap::from([(
                "dtype".to_string(),
                IrArg::Dtype(plan.zp_dtype.clone()),
            )]);
            ctx.create_node(
                NodeOp::CallFunction,
                format!("{name}_{which}_zero_point"),
                FULL,
                vec![i64_list_arg(&[1]), IrArg::Int(value)],
                kwargs,
                meta::call_operator(&source, TensorFacts::new(vec![1], plan.zp_dtype.clone())),
            )
        };
        let input_zp = zero_point("input", plan.input_zp);
        let output_zp = zero_point("output", plan.output_zp);
        let pre = ctx.call_function(
            format!("{name}_nhwc"),
            PERMUTE_COPY,
            vec![IrArg::Node(plan.x), i64_list_arg(&NHWC_ORDER)],
            meta::call_operator(&source, plan.x_nhwc.clone()),
        );
        let pool = ctx.call_function(
            format!("{name}_tosa"),
            TOSA_AVG_POOL2D,
            vec![
                IrArg::Node(pre),
                IrArg::Node(input_zp),
                IrArg::Node(output_zp),
                i64_list_arg(&plan.kernel),
                i64_list_arg(&plan.stride),
                i64_list_arg(&plan.pad),
                IrArg::Dtype(plan.acc_type.to_string()),
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
    ctx.replace_all_uses_with(id, post);
    ctx.erase_node(id);
}
