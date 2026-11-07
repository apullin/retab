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
//!   same (critique B2, A7). IrMaterializeConstantsPass, which takes rc4's
//!   ComputeConstantOpsAOTPass slot right after this pass (`:699`), then folds
//!   each `full` into a constant as that pass does
//!   (`fuse_constant_ops_pass.py:327-335`, [`FullQParams::Rc4`]): quantized
//!   with the pool's output qparams when it has them, which is rc4's quirk,
//!   and holding the zero point itself when it has none, input qparams or
//!   not. The pipeline needs that pass after this one: the effects of
//!   IrFuseZeroInputConstantsEffectPass, at ingest, would quantize the zero
//!   points of a pool without output qparams with its input qparams. Because
//!   the zero points are call nodes, the rewrite needs no constant of its own
//!   and works unchanged in a control-flow region. `dtype` is the pool's val
//!   dtype as rc4's fake tensor has it ([`dtype::value_dtype`]); the device,
//!   which RETAB's importer reads as None, is left out.
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
//! [`FullQParams::Rc4`]: crate::fx_fuse_zero_input_constants_ir::FullQParams::Rc4
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
    adjust_pooling_pad_if_needed, ceil_mode_arg, has_no_valued_kwargs, stride_or_kernel, to_2tuple,
    NHWC_INVERSE_ORDER, NHWC_ORDER,
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
        return Err(format!(
            "a rank-{} input; rc4 reads shape[2:4]",
            x_facts.rank()
        ));
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
        Some(qparam) if qparam.is_per_channel() => Err(format!("a per-channel {which} qparam")),
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
            let kwargs =
                HashMap::from([("dtype".to_string(), IrArg::Dtype(plan.zp_dtype.clone()))]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{NodeMeta, QParam, QParamMap, TensorSource};
    use crate::graph::ir::DType;
    use crate::layout::constants::IrMaterializeConstantsPass;
    use crate::layout::testing::{ints, node, LayoutGraph, PERMUTE};
    use crate::lowering_effects::LoweringInputKind;
    use crate::test_ir::qparam;

    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";
    const F32: &str = "torch.float32";

    /// `IrArg` has no `PartialEq`; its derived `Debug` is exact.
    fn assert_args(actual: &[IrArg], expected: &[IrArg]) {
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }

    fn int16_qparam(scale: f64, zp: i64) -> QParam {
        QParam::per_tensor(scale, zp, -32767, 32767, I16.to_string())
    }

    /// `x -> avg_pool2d(x, args...) -> output`. A quantized pool holds RETAB's
    /// float32 val (design F2) and `iq`/`oq`; a float pool holds `input_dtype`.
    fn pool_graph(
        input: &[i64],
        input_dtype: &str,
        args: Vec<IrArg>,
        output: &[i64],
        iq: Option<QParam>,
        oq: Option<QParam>,
    ) -> (LayoutGraph, String) {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", input, input_dtype);
        let mut all_args = vec![node(x)];
        all_args.extend(args);
        let pool = g.add_unrefreshed(AVG_POOL2D, all_args);
        let quantized = iq.is_some() || oq.is_some();
        let dtype = if quantized { F32 } else { input_dtype };
        let mut meta = NodeMeta::tensor(output.to_vec(), dtype);
        if let Some(iq) = iq {
            meta.set_single_input_qparam(iq);
        }
        if let Some(oq) = oq {
            meta.set_single_output_qparam(oq);
        }
        g.graph_mut().node_mut(pool).meta = meta;
        g.output(&[pool]);
        let name = g.name(pool);
        (g, name)
    }

    fn by_name<'a>(g: &'a LayoutGraph, name: &str) -> &'a crate::fx_ir::IrNode {
        g.graph().node(g.id(name))
    }

    fn assert_declined(g: &mut LayoutGraph, reason: &str) {
        crate::stock_fallback::take();
        let before: Vec<String> = g.targets().into_iter().map(str::to_string).collect();
        assert!(!IrRewriteAvgPool2dPass.run(g.graph_mut()));
        assert_eq!(g.targets(), before, "a declined rewrite changes nothing");
        let request = crate::stock_fallback::take().expect("a stock fallback request");
        assert!(request.contains(reason), "{request}");
    }

    /// rc4's `avg_pool_k3s2p1` shape with distinct input and output zero points,
    /// which the captured cases never have (their pools share qparams): each
    /// `full` takes its own map's zero point, input first, and every node copies
    /// the pool's meta, qparams included.
    #[test]
    fn quantized_pool_becomes_zero_points_permutes_and_avg_pool2d() {
        let (iq, oq) = (qparam(0.1, -3, I8), qparam(0.2, 5, I8));
        let (mut g, name) = pool_graph(
            &[1, 3, 8, 8],
            I8,
            vec![
                ints(&[3, 3]),
                ints(&[2, 2]),
                ints(&[1, 1]),
                IrArg::Bool(false),
                IrArg::Bool(false),
            ],
            &[1, 3, 4, 4],
            Some(iq.clone()),
            Some(oq.clone()),
        );
        assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
        g.assert_targets(&[FULL, FULL, PERMUTE, TOSA_AVG_POOL2D, PERMUTE]);
        assert!(g.graph().by_name(&name).is_none_or(|node| !node.alive));

        let input_zp = by_name(&g, &format!("{name}_input_zero_point"));
        let output_zp = by_name(&g, &format!("{name}_output_zero_point"));
        for (zp, value) in [(input_zp, -3), (output_zp, 5)] {
            assert_args(&zp.args, &[ints(&[1]), IrArg::Int(value)]);
            assert_eq!(
                format!("{:?}", zp.kwarg("dtype")),
                format!("{:?}", Some(IrArg::Dtype(I8.into())))
            );
            assert_eq!((g.shape(zp.id), g.dtype(zp.id)), (&[1][..], I8));
            assert_eq!(zp.meta.input_qparam(), Some(&iq));
            assert_eq!(zp.meta.output_qparam(), Some(&oq));
        }
        let pre = by_name(&g, &format!("{name}_nhwc"));
        assert_args(&pre.args, &[node(g.id("x")), ints(&[0, 2, 3, 1])]);
        assert_eq!((g.shape(pre.id), g.dtype(pre.id)), (&[1, 8, 8, 3][..], I8));
        let pool = by_name(&g, &format!("{name}_tosa"));
        assert_args(
            &pool.args,
            &[
                node(pre.id),
                node(input_zp.id),
                node(output_zp.id),
                ints(&[3, 3]),
                ints(&[2, 2]),
                ints(&[1, 0, 1, 0]),
                IrArg::Dtype("torch.int32".into()),
            ],
        );
        // The input dtype rc4's kernel keeps, from the input qparam.
        assert_eq!(
            (g.shape(pool.id), g.dtype(pool.id)),
            (&[1, 4, 4, 3][..], I8)
        );
        let post = by_name(&g, &format!("{name}_nchw"));
        assert_args(&post.args, &[node(pool.id), ints(&[0, 3, 1, 2])]);
        assert_eq!(
            (g.shape(post.id), g.dtype(post.id)),
            (&[1, 3, 4, 4][..], I8)
        );
        for id in [pre.id, pool.id, post.id] {
            let meta = &g.graph().node(id).meta;
            assert_eq!(meta.input_qparam(), Some(&iq));
            assert_eq!(meta.output_qparam(), Some(&oq));
            assert_eq!(meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
        }
        assert!(g.graph().output_node().unwrap().args[0].contains_node(post.id));
    }

    /// An omitted stride is the kernel (rc4's
    /// `test_rewrite_avg_pool2d_tosa_empty_stride_uses_kernel_size`), and so is
    /// any falsy one: RETAB's import spells `[]` as None.
    #[test]
    fn a_falsy_stride_is_the_kernel() {
        for stride in [
            None,
            Some(IrArg::None),
            Some(IrArg::List(vec![])),
            Some(IrArg::Int(0)),
        ] {
            let mut args = vec![ints(&[2, 3])];
            args.extend(stride.clone());
            let (mut g, name) = pool_graph(&[1, 3, 8, 9], F32, args, &[1, 3, 4, 3], None, None);
            assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()), "{stride:?}");
            let pool = by_name(&g, &format!("{name}_tosa"));
            assert_eq!(pool.arg_int_list(4), Some(vec![2, 3]), "{stride:?}");
        }
    }

    /// One-element lists apply to both axes (`to_2tuple`), and ceil mode adds a
    /// post pad (`adjust_pooling_pad_if_needed`).
    #[test]
    fn one_element_lists_and_ceil_mode() {
        let q = qparam(0.05, 0, I8);
        let (mut g, name) = pool_graph(
            &[1, 3, 8, 8],
            I8,
            vec![ints(&[3]), ints(&[2]), ints(&[0]), IrArg::Bool(true)],
            &[1, 3, 4, 4],
            Some(q.clone()),
            Some(q),
        );
        assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
        let pool = by_name(&g, &format!("{name}_tosa"));
        assert_eq!(pool.arg_int_list(3), Some(vec![3, 3]));
        assert_eq!(pool.arg_int_list(4), Some(vec![2, 2]));
        assert_eq!(pool.arg_int_list(5), Some(vec![0, 1, 0, 1]));
        assert_eq!(g.shape(pool.id), [1, 4, 4, 3]);
    }

    /// `:69-75`: int16 accumulates in int32 too; a float pool in float32 with
    /// zero points of 0 in its own dtype (rc4's FP path, test only).
    #[test]
    fn accumulator_and_zero_point_dtypes_follow_the_input() {
        let q = int16_qparam(0.000244140625, 0);
        let (mut g, name) = pool_graph(
            &[1, 3, 8, 8],
            I16,
            vec![ints(&[2, 2]), ints(&[2, 2]), ints(&[0, 0])],
            &[1, 3, 4, 4],
            Some(q.clone()),
            Some(q),
        );
        assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
        let pool = by_name(&g, &format!("{name}_tosa"));
        assert_eq!(format!("{:?}", pool.args[6]), "Dtype(\"torch.int32\")");
        assert_eq!(g.dtype(pool.id), I16);
        assert_eq!(g.dtype(g.id(&format!("{name}_input_zero_point"))), I16);

        let (mut g, name) = pool_graph(
            &[1, 3, 8, 8],
            F32,
            vec![ints(&[2, 2])],
            &[1, 3, 4, 4],
            None,
            None,
        );
        assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
        let pool = by_name(&g, &format!("{name}_tosa"));
        assert_eq!(format!("{:?}", pool.args[6]), "Dtype(\"torch.float32\")");
        let zp = by_name(&g, &format!("{name}_input_zero_point"));
        assert_args(&zp.args, &[ints(&[1]), IrArg::Int(0)]);
        assert_eq!(g.dtype(zp.id), F32);
        assert!(!zp.meta.has_qparam_metadata());
    }

    /// With IrMaterializeConstants after it, as rc4 runs ComputeConstantOpsAOT
    /// at `arm_pass_manager.py:699`, each zero point becomes a buffer at the
    /// graph head holding the zero point quantized with the pool's *output*
    /// qparams (rc4's quirk, critique A7): q(-3) = round(-3 / 0.2) + 5 = -10 and
    /// q(5) = 30, output zero point first. In a region they are `get_attr`
    /// nodes and the placeholders stay as they were (critique B2).
    #[test]
    fn zero_points_materialize_quantized_with_the_output_qparams() {
        for in_region in [false, true] {
            let (mut g, name) = pool_graph(
                &[1, 3, 8, 8],
                I8,
                vec![ints(&[2, 2]), ints(&[2, 2])],
                &[1, 3, 4, 4],
                Some(qparam(0.1, -3, I8)),
                Some(qparam(0.2, 5, I8)),
            );
            let placeholders = |g: &LayoutGraph| -> Vec<String> {
                g.graph()
                    .live_placeholder_ids()
                    .into_iter()
                    .map(|id| g.name(id))
                    .collect()
            };
            let before = placeholders(&g);
            assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
            assert_eq!(placeholders(&g), before, "the rewrite adds no placeholder");
            assert!(IrMaterializeConstantsPass { in_region }.run(g.graph_mut()));
            assert_eq!(g.count(FULL), 0);
            let pool = by_name(&g, &format!("{name}_tosa"));
            let constant = |index: usize| {
                let id = pool.args[index].as_node().unwrap();
                g.graph().node(id)
            };
            for (index, value) in [(1, -10i8), (2, 30i8)] {
                let zp = constant(index);
                let tensor = zp.tensor.as_ref().expect("constant data");
                assert_eq!(tensor.data.dtype, DType::Int8);
                assert_eq!(tensor.data.data, vec![value as u8], "zero point {index}");
                if in_region {
                    assert_eq!(zp.op, NodeOp::GetAttr);
                } else {
                    assert_eq!(zp.op, NodeOp::Placeholder);
                    let TensorSource::ExportedProgramInput(binding) = &tensor.source else {
                        panic!("a parent constant is bound to the exported program");
                    };
                    assert_eq!(binding.input_kind, LoweringInputKind::Buffer);
                }
            }
            let order = g.graph().topological_live_node_ids();
            let position = |index: usize| {
                let id = pool.args[index].as_node().unwrap();
                order.iter().position(|&live| live == id).unwrap()
            };
            assert!(
                position(2) < position(1),
                "rc4 puts the output zero point first"
            );
            if in_region {
                assert_eq!(placeholders(&g), before);
            }
        }
    }

    /// rc4's ComputeConstantOpsAOTPass quantizes a `full` only when the node
    /// has output qparams (`fuse_constant_ops_pass.py:327-335`); otherwise it
    /// evaluates `full((1,), zp, dtype=int8)` as it is. rc4's
    /// RewriteAvgPool2dPass then ComputeConstantOpsAOTPass on an int8 pool
    /// give (run in .venv-et16, review of APU-156):
    ///
    /// - input qparams (0.1, -3) and an empty output map: int8 [-3] and [0];
    /// - output qparams (0.2, 5) and an empty input map: q(0) = 5 and
    ///   q(5) = 30.
    ///
    /// IrMaterializeConstants gives the same, with the output map empty or
    /// absent (RETAB's import leaves it out) and in a region. The planner
    /// alone, as the effect passes run it (IrFuseZeroInputConstantsEffectPass's
    /// rule), would quantize the first pool's zero points with its input
    /// qparams: round(-3 / 0.1) - 3 = -33 and round(0 / 0.1) - 3 = -3.
    #[test]
    fn zero_points_of_a_pool_without_output_qparams_stay_unquantized() {
        let zero_points = |iq: Option<QParam>, oq: Option<QParam>, empty_map, in_region| {
            let (mut g, name) = pool_graph(
                &[1, 3, 8, 8],
                I8,
                vec![ints(&[2, 2]), ints(&[2, 2])],
                &[1, 3, 4, 4],
                iq,
                oq,
            );
            if empty_map {
                let id = g.id(&name);
                let meta = &mut g.graph_mut().node_mut(id).meta;
                if meta.input_qparams().is_none() {
                    meta.set_input_qparams(QParamMap::new());
                }
                if meta.output_qparams().is_none() {
                    meta.set_output_qparams(QParamMap::new());
                }
            }
            assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
            assert!(IrMaterializeConstantsPass { in_region }.run(g.graph_mut()));
            let pool = by_name(&g, &format!("{name}_tosa"));
            [1, 2].map(|index| {
                let id = pool.args[index].as_node().unwrap();
                let tensor = g.graph().node(id).tensor.as_ref().expect("constant data");
                let data = &tensor.data;
                assert_eq!(
                    (data.dtype, data.shape.as_slice(), data.data.len()),
                    (DType::Int8, &[1][..], 1),
                    "zero point {index}"
                );
                data.data[0] as i8
            })
        };
        for empty_map in [false, true] {
            for in_region in [false, true] {
                let case = format!("empty_map={empty_map} in_region={in_region}");
                let input_only = zero_points(Some(qparam(0.1, -3, I8)), None, empty_map, in_region);
                assert_eq!(input_only, [-3, 0], "input qparams only, {case}");
                let output_only = zero_points(None, Some(qparam(0.2, 5, I8)), empty_map, in_region);
                assert_eq!(output_only, [5, 30], "output qparams only, {case}");
            }
        }
    }

    #[test]
    fn rc4_errors_go_to_stock() {
        let q = || Some(qparam(0.1, 0, I8));
        let pool = |input: &[i64], args: Vec<IrArg>, output: &[i64], iq, oq| {
            pool_graph(input, I8, args, output, iq, oq).0
        };
        // `get_zp_per_tensor` raises for a per-channel zero point.
        let mut per_channel = qparam(0.1, 0, I8);
        per_channel.per_channel_scales = vec![0.1, 0.2, 0.3];
        per_channel.per_channel_zps = vec![0, 0, 0];
        let mut g = pool(
            &[1, 3, 8, 8],
            vec![ints(&[2, 2])],
            &[1, 3, 4, 4],
            q(),
            Some(per_channel),
        );
        assert_declined(&mut g, "per-channel");
        // rc4 reads `x.data.shape[3]`.
        let mut g = pool(&[3, 8, 8], vec![ints(&[2, 2])], &[3, 4, 4], q(), q());
        assert_declined(&mut g, "rank-3");
        // A kernel that doesn't unpack as two values.
        let mut g = pool(
            &[1, 3, 8, 8],
            vec![ints(&[2, 2, 2])],
            &[1, 3, 4, 4],
            q(),
            q(),
        );
        assert_declined(&mut g, "kernel");
        // AVG_POOL2D's zero points must have the input's dtype.
        let mut g = pool(
            &[1, 3, 8, 8],
            vec![ints(&[2, 2])],
            &[1, 3, 4, 4],
            q(),
            Some(int16_qparam(0.1, 0)),
        );
        assert_declined(&mut g, "zero points of torch.int16");
        // An output the pool's val contradicts.
        let mut g = pool(&[1, 3, 8, 8], vec![ints(&[2, 2])], &[1, 3, 5, 5], q(), q());
        assert_declined(&mut g, "value");
        // A keyword argument with a value, which rc4 would drop.
        let mut g = pool(&[1, 3, 8, 8], vec![ints(&[2, 2])], &[1, 3, 4, 4], q(), q());
        let id = g.id("avg_pool2d");
        g.graph_mut()
            .set_kwarg(id, "divisor_override", IrArg::Int(2));
        assert_declined(&mut g, "keyword");
    }

    /// The first pool rc4 would raise on stops the pass; pools before it stay
    /// rewritten (the graph goes to stock either way).
    #[test]
    fn a_declined_pool_stops_the_pass() {
        crate::stock_fallback::take();
        let q = qparam(0.1, 0, I8);
        let (mut g, first) = pool_graph(
            &[1, 3, 8, 8],
            I8,
            vec![ints(&[2, 2])],
            &[1, 3, 4, 4],
            Some(q.clone()),
            Some(q.clone()),
        );
        let first_id = g.id(&first);
        let second = g.add_unrefreshed(AVG_POOL2D, vec![node(first_id), ints(&[2, 2, 2])]);
        let mut meta = NodeMeta::tensor(vec![1, 3, 2, 2], F32);
        meta.set_both_qparams(q);
        g.graph_mut().node_mut(second).meta = meta;
        let output = g.graph().output_node().unwrap().id;
        g.rewire(output, first_id, second);
        assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
        g.assert_targets(&[FULL, FULL, PERMUTE, TOSA_AVG_POOL2D, PERMUTE, AVG_POOL2D]);
        assert!(crate::stock_fallback::take().is_some());
    }

    #[test]
    fn graphs_without_avg_pools_are_left_alone() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 4, 4]);
        let y = g.permute(x, &[0, 2, 3, 1]);
        g.output(&[y]);
        let before = g.graph().topological_live_node_ids();
        assert!(!IrRewriteAvgPool2dPass.run(g.graph_mut()));
        assert_eq!(g.graph().topological_live_node_ids(), before);
    }
}
