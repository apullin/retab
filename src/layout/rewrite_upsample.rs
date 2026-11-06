//! Port of rc4's `RewriteUpsamplePass`
//! (`backends/arm/_passes/rewrite_upsample.py`, line numbers below are that
//! file's), phase-6 slot 1 (`arm_pass_manager.py:717`; APU-135 P2 S7, design
//! §3.3, critique A1).
//!
//! Each `aten.upsample_nearest2d.vec` or `aten.upsample_bilinear2d.vec`
//! becomes
//!
//! ```text
//! CONST_SHAPE([syn, syd, sxn, sxd])  CONST_SHAPE([oy, ox])  CONST_SHAPE([by, bx])
//! x -> permute_copy([0, 2, 3, 1]) -> RESIZE(x', scale, offset, border, resize_mode=...)
//!   [-> RESCALE(.., out_dtype, [1 / (syn * sxn)], 0, 0)] -> permute_copy([0, 3, 1, 2])
//! ```
//!
//! - **Parameters.** [`resize_parameters_1d`] is `get_resize_parameters_1d`
//!   (`:45-155`) for static sizes, run per axis on the input's and the
//!   upsample's vals. It keeps rc4's nearest-mode branches, which RETAB's
//!   dim-order emitter lacks (`resize_axis_parameters`, design §3.3), and its
//!   `nsimplify(scale_factor, tolerance=1e-6, rational=True)` for a nearest
//!   upsample with explicit scale factors (design risk 6): see
//!   [`nsimplify_scale_factor`].
//! - **Nodes and meta** (`:205-306`). Every node is `create_node(...,
//!   from_node=node)`: the upsample's meta with its qparam maps emptied
//!   ([`meta::from_node`], [`meta::shape_node`] for the CONST_SHAPEs), except
//!   the RESIZE, which inherits the qparams (critique A1). The CONST_SHAPEs,
//!   the input permute and the RESIZE go before the upsample in that order, the
//!   RESCALE right after the RESIZE and the output permute after that.
//! - **Dtypes.** rc4 branches on the input's fake dtype and types the RESCALE
//!   with the upsample's ([`dtype::operand_dtype`](crate::layout::dtype::operand_dtype),
//!   [`dtype::value_dtype`](crate::layout::dtype::value_dtype);
//!   design F2). The RESIZE holds its fake kernel's dtype: int8 for nearest,
//!   int32 for int8 bilinear, int16 for int16 (design F10). An int8 or int16
//!   bilinear RESIZE is followed by the RESCALE, and an int16 one is marked
//!   INT48 (`:258-289`). The permutes hold their input's dtype.
//!
//! **Failures.** Where rc4 raises, the port asks for the stock fallback and
//! stops: arguments that don't unpack (`:191-197`), an input or output that
//! isn't rank 4, a scale factor list with one value, a ratio sympy can't turn
//! into integers, an input dtype the RESIZE kernel rejects, and RESIZE
//! parameters its validation rejects (`resize_utils.py`, critique B10). That
//! last one includes the exact 1/16 bilinear downscale rc4's
//! DecomposeUnsupportedBilinearResizePass (`arm_pass_manager.py:696`), which
//! RETAB lacks, turns into an avg pool first. An output shape that differs from
//! the upsample's val is declined rather than trusted (rc4's border makes the
//! two agree by construction). The `.default` overloads aren't rc4's targets;
//! they stay aten ops, which the serializer's explicit-mode guard (X5) sends
//! to stock.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrNode, IrPass, NodeId, NodeOp};
use crate::fx_meta::TensorFacts;
use crate::layout::ctx::LayoutCtx;
use crate::layout::dtype::{
    operand_dtype, resize_output_dtype, set_special_dtype, value_dtype, SpecialDtype,
};
use crate::layout::meta;
use crate::layout::pool::{NHWC_INVERSE_ORDER, NHWC_ORDER};
use crate::layout::shape::permute_shape;
use crate::layout::shape::tosa::resize_shape;

const UPSAMPLE_NEAREST2D_VEC: &str = "aten.upsample_nearest2d.vec";
const UPSAMPLE_BILINEAR2D_VEC: &str = "aten.upsample_bilinear2d.vec";
const TOSA_RESIZE: &str = "backend.tosa.RESIZE.default";
const TOSA_RESCALE: &str = "backend.tosa.RESCALE.default";
const TOSA_CONST_SHAPE: &str = "backend.tosa.CONST_SHAPE.default";
const PERMUTE_COPY: &str = "aten.permute_copy.default";

/// `nsimplify`'s `tolerance=1e-6`: `_real_to_rational` limits the denominator
/// to `ceiling(1 / tolerance)`, which is exactly 1000000 in double precision
/// (sympy 1.14, `sympy/simplify/simplify.py`).
const SCALE_FACTOR_MAX_DENOMINATOR: i128 = 1_000_000;

/// TOSA RESIZE's mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResizeMode {
    Nearest,
    Bilinear,
}

impl ResizeMode {
    fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
        }
    }
}

/// One axis's `(scale_n, scale_d, offset, border)`, TOSA's doubled-rational
/// encoding of the resize.
pub(crate) type ResizeAxis = [i64; 4];

/// rc4's `get_resize_parameters_1d` (`rewrite_upsample.py:45-155`) for static
/// sizes (the symbolic branches need dynamic shapes, which RETAB never sees):
///
/// 1. The ratio is `(out - 1) / (in - 1)` for `align_corners` when both sizes
///    exceed 1, else `out / in`, reduced; for a nearest resize with an
///    explicit scale factor it is that factor as [`nsimplify_scale_factor`]
///    recovers it (PyTorch samples with the factor, not the size ratio, when
///    `recompute_scale_factor` is False).
/// 2. `scale_n = 2 * num`, `scale_d = 2 * den`.
/// 3. The offset is `scale_d / 2 - scale_n / 2` for a nearest integer upscale
///    (den 1) and for bilinear without `align_corners` (half-pixel centres),
///    `-(scale_n / 2)` for any other nearest resize (cancelling TOSA's
///    half-scale bias so it samples `floor(out * in / out_size)` as PyTorch
///    does), and 0 for bilinear with `align_corners`.
/// 4. `border = scale_d * (out - 1) - scale_n * (in - 1) + offset`, which makes
///    TOSA's output size `out`.
///
/// `Err` where rc4 raises: a zero input size (sympy's ratio is `zoo`), a scale
/// factor that isn't finite and positive, or values outside `i64`.
pub(crate) fn resize_parameters_1d(
    input_size: i64,
    output_size: i64,
    align_corners: bool,
    mode: ResizeMode,
    scale_factor: Option<f64>,
) -> Result<ResizeAxis, String> {
    let (input, output) = (i128::from(input_size), i128::from(output_size));
    let (scale_n, scale_d) = if align_corners && input > 1 && output > 1 {
        (output - 1, input - 1)
    } else {
        (output, input)
    };
    let (num, den) = match (mode, scale_factor) {
        (ResizeMode::Nearest, Some(factor)) => nsimplify_scale_factor(factor)?,
        // `nsimplify(simplify(scale_n / scale_d))` of two sympy Integers is the
        // reduced Rational (checked against sympy for sizes up to 300).
        _ => reduced(scale_n, scale_d)
            .ok_or_else(|| format!("resize ratio {scale_n}/{scale_d} is not a number"))?,
    };
    let (scale_n, scale_d) = (2 * num, 2 * den);
    let offset = match mode {
        ResizeMode::Nearest if den == 1 => scale_d.div_euclid(2) - scale_n.div_euclid(2),
        ResizeMode::Nearest => -scale_n.div_euclid(2),
        ResizeMode::Bilinear if align_corners => 0,
        ResizeMode::Bilinear => scale_d.div_euclid(2) - scale_n.div_euclid(2),
    };
    let border = scale_d * (output - 1) - scale_n * (input - 1) + offset;
    let narrow = |value: i128| {
        i64::try_from(value).map_err(|_| format!("resize parameter {value} overflows"))
    };
    Ok([
        narrow(scale_n)?,
        narrow(scale_d)?,
        narrow(offset)?,
        narrow(border)?,
    ])
}

/// `num / den` in lowest terms with a positive denominator, as sympy's Rational
/// normalizes it; `None` for a zero denominator.
fn reduced(num: i128, den: i128) -> Option<(i128, i128)> {
    if den == 0 {
        return None;
    }
    let divisor = gcd(num, den);
    let sign = if den < 0 { -1 } else { 1 };
    Some((sign * num / divisor, sign * den / divisor))
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// `sympy.nsimplify(factor, tolerance=1e-6, rational=True)` as `(num, den)`.
///
/// For a float, `nsimplify` with `rational=True` is `_real_to_rational`
/// (sympy 1.14, `sympy/simplify/simplify.py`): the float's exact binary value
/// as a Rational, then `limit_denominator(ceiling(1 / tolerance))`, which is
/// Python's `fractions.Fraction.limit_denominator(1000000)`: the closest
/// fraction with a denominator of at most 10^6. So 1.6 and its float32 value
/// 1.600000023841858 both give 8/5, and float32 1/3 gives 1/3 (critique risk 6;
/// `p2_replay_tests::review_upsample_covers_float32_rounded_scale_factors`).
/// An integral factor gives itself.
///
/// `Err` for a factor that isn't finite and positive (rc4 would build a
/// non-positive RESIZE scale, or fail converting infinity to an int), or one
/// too large or too small for [`exact_fraction`].
pub(crate) fn nsimplify_scale_factor(factor: f64) -> Result<(i128, i128), String> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(format!("scale factor {factor}"));
    }
    let (num, den) =
        exact_fraction(factor).ok_or_else(|| format!("scale factor {factor} is out of range"))?;
    Ok(limit_denominator(num, den, SCALE_FACTOR_MAX_DENOMINATOR))
}

/// A positive finite double as `num / den` in lowest terms, exactly, or
/// `None` when either needs more than [`MAX_EXACT_BITS`] bits. Every factor a
/// RESIZE can take (its scale is at most 2048 and above 1/16) fits easily.
fn exact_fraction(value: f64) -> Option<(i128, i128)> {
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i64;
    let fraction_bits = bits & ((1u64 << 52) - 1);
    let (mantissa, exponent) = if exponent_bits == 0 {
        (fraction_bits, -1074)
    } else {
        (fraction_bits | (1u64 << 52), exponent_bits - 1075)
    };
    if mantissa == 0 {
        return None;
    }
    // An odd mantissa over a power of two is in lowest terms.
    let shift = i64::from(mantissa.trailing_zeros());
    let (mantissa, exponent) = (i128::from(mantissa >> shift), exponent + shift);
    let mantissa_bits = 128 - i64::from(mantissa.leading_zeros());
    if exponent >= 0 {
        (mantissa_bits + exponent <= MAX_EXACT_BITS).then(|| (mantissa << exponent, 1))
    } else {
        (-exponent <= MAX_EXACT_BITS).then(|| (mantissa, 1i128 << -exponent))
    }
}

/// The bit length [`exact_fraction`] allows, which keeps every product in
/// [`limit_denominator`] (at most 2 * 2^100 * 10^6) inside `i128`.
const MAX_EXACT_BITS: i64 = 100;

/// Python's `fractions.Fraction.limit_denominator` (CPython 3.12,
/// `Lib/fractions.py`) for a positive fraction in lowest terms: the closest
/// fraction with a denominator of at most `max_denominator`, the one with the
/// smaller denominator on a tie.
fn limit_denominator(num: i128, den: i128, max_denominator: i128) -> (i128, i128) {
    if den <= max_denominator {
        return (num, den);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (num, den);
    loop {
        if d == 0 {
            break;
        }
        let a = n.div_euclid(d);
        let q2 = q0 + a * q1;
        if q2 > max_denominator {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (max_denominator - q0).div_euclid(q1);
    if 2 * d * (q0 + k * q1) <= den {
        (p1, q1)
    } else {
        (p0 + k * p1, q0 + k * q1)
    }
}

/// rc4's RewriteUpsamplePass; see the module docs.
pub(crate) struct IrRewriteUpsamplePass;

fn is_target(target: &str) -> bool {
    matches!(target, UPSAMPLE_NEAREST2D_VEC | UPSAMPLE_BILINEAR2D_VEC)
}

impl IrPass for IrRewriteUpsamplePass {
    fn name(&self) -> &str {
        "IrRewriteUpsample"
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
                        "IrRewriteUpsample: {}: {reason}",
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

/// The RESCALE an int8 or int16 bilinear RESIZE needs (`:258-289`).
#[derive(Debug)]
struct Rescale {
    dtype: String,
    scale: f64,
    value: TensorFacts,
}

/// Everything one rewrite needs, worked out before the graph changes.
#[derive(Debug)]
struct Plan {
    x: NodeId,
    mode: ResizeMode,
    scale: [i64; 4],
    offset: [i64; 2],
    border: [i64; 2],
    x_nhwc: TensorFacts,
    resized: TensorFacts,
    int48: bool,
    rescale: Option<Rescale>,
    output: TensorFacts,
}

/// The upsample's positional arguments (`:191-197`): `x, output_size,
/// align_corners, scale_factors` for bilinear and `x, output_size,
/// scale_factors` for nearest, whose `align_corners` is False.
fn unpack(node: &IrNode) -> Result<(NodeId, ResizeMode, bool, &IrArg), String> {
    if node
        .kwargs
        .values()
        .any(|value| !matches!(value, IrArg::None))
    {
        return Err("keyword arguments; rc4 unpacks node.args".into());
    }
    let x = |arg: &IrArg| arg.as_node().ok_or_else(|| "no input node".to_string());
    match (node.target.as_str(), node.args.as_slice()) {
        (UPSAMPLE_BILINEAR2D_VEC, [input, _, align_corners, scale_factors]) => {
            let align_corners = match align_corners {
                IrArg::Bool(value) => *value,
                IrArg::None => false,
                other => return Err(format!("align_corners {other:?}")),
            };
            Ok((
                x(input)?,
                ResizeMode::Bilinear,
                align_corners,
                scale_factors,
            ))
        }
        (UPSAMPLE_NEAREST2D_VEC, [input, _, scale_factors]) => {
            Ok((x(input)?, ResizeMode::Nearest, false, scale_factors))
        }
        (_, args) => Err(format!("{} positional arguments", args.len())),
    }
}

/// `scale_factors or (None, None)`, then `scale_factors_yx[0]` and `[1]`.
fn scale_factors_yx(arg: &IrArg) -> Result<[Option<f64>; 2], String> {
    let items = match arg {
        IrArg::None => return Ok([None, None]),
        IrArg::List(items) if items.is_empty() => return Ok([None, None]),
        IrArg::List(items) => items,
        other => return Err(format!("scale_factors {other:?}")),
    };
    let factor = |item: &IrArg| {
        item.as_numeric_f64()
            .ok_or_else(|| format!("scale factor {item:?}"))
    };
    match items.as_slice() {
        [y, x, ..] => Ok([Some(factor(y)?), Some(factor(x)?)]),
        _ => Err("one scale factor for two axes".into()),
    }
}

/// `RewriteUpsamplePass.call` (`:183-306`) for one node, up to the graph edits.
fn plan(graph: &IrGraph, id: NodeId) -> Result<Plan, String> {
    let node = graph.node(id);
    let (x, mode, align_corners, scale_factors) = unpack(node)?;
    let x_facts = graph
        .node(x)
        .meta
        .tensor_facts()
        .ok_or("the input has no value")?;
    let output_facts = node
        .meta
        .tensor_facts()
        .ok_or("the upsample has no value")?;
    let (&[_, _, in_h, in_w], &[_, _, out_h, out_w]) =
        (x_facts.shape.as_slice(), output_facts.shape.as_slice())
    else {
        return Err(format!(
            "input {:?} and output {:?} must be rank 4",
            x_facts.shape, output_facts.shape
        ));
    };
    let [factor_y, factor_x] = scale_factors_yx(scale_factors)?;
    let [syn, syd, offset_y, border_y] =
        resize_parameters_1d(in_h, out_h, align_corners, mode, factor_y)?;
    let [sxn, sxd, offset_x, border_x] =
        resize_parameters_1d(in_w, out_w, align_corners, mode, factor_x)?;
    let scale = [syn, syd, sxn, sxd];
    let offset = [offset_y, offset_x];
    let border = [border_y, border_x];

    let x_nhwc = TensorFacts::new(
        permute_shape(&x_facts.shape, &NHWC_ORDER).map_err(|e| e.to_string())?,
        x_facts.dtype.clone(),
    );
    // `:258-263` reads the input's fake dtype; the RESIZE kernel types its
    // output from it (`tosa/dialect/ops/resize.py:28-55`).
    let input_dtype = operand_dtype(graph, id, 0).ok_or("the input has no dtype")?;
    let bilinear = mode == ResizeMode::Bilinear;
    let resize_dtype = resize_output_dtype(&input_dtype, bilinear)
        .ok_or_else(|| format!("RESIZE does not take {input_dtype}"))?;
    let resized = TensorFacts::new(
        resize_shape(&x_nhwc.shape, &scale, &offset, &border).map_err(|e| e.to_string())?,
        resize_dtype,
    );
    let integer_input = matches!(input_dtype.as_str(), "torch.int8" | "torch.int16");
    let rescale = if integer_input && bilinear {
        let dtype = value_dtype(graph, id).ok_or("the upsample has no dtype")?;
        Some(Rescale {
            scale: 1.0 / ((syn * sxn) as f64),
            value: TensorFacts::new(resized.shape.clone(), dtype.clone()),
            dtype,
        })
    } else {
        None
    };
    let last = rescale.as_ref().map_or(&resized, |rescale| &rescale.value);
    let output = TensorFacts::new(
        permute_shape(&last.shape, &NHWC_INVERSE_ORDER).map_err(|e| e.to_string())?,
        last.dtype.clone(),
    );
    if output.shape != output_facts.shape {
        return Err(format!(
            "the rewrite gives {:?} where the upsample's value is {:?}",
            output.shape, output_facts.shape
        ));
    }
    Ok(Plan {
        x,
        mode,
        scale,
        offset,
        border,
        x_nhwc,
        resized,
        int48: bilinear && input_dtype == "torch.int16",
        rescale,
        output,
    })
}

/// The graph edits (`:205-306`).
fn rewrite(ctx: &mut LayoutCtx<'_>, id: NodeId, plan: &Plan) {
    let source = ctx.node(id).meta.clone();
    let name = ctx.node(id).name.clone();
    let resize = ctx.inserting_before(id, |ctx| {
        let mut shape = |suffix: &str, values: &[i64]| {
            // `create_shape_node` (`arm_pass_utils.py:195-216`).
            ctx.call_function(
                format!("{name}_{suffix}"),
                TOSA_CONST_SHAPE,
                vec![i64_list_arg(values)],
                meta::shape_node(&source),
            )
        };
        let scale = shape("scale", &plan.scale);
        let offset = shape("offset", &plan.offset);
        let border = shape("border", &plan.border);
        let pre = ctx.call_function(
            format!("{name}_nhwc"),
            PERMUTE_COPY,
            vec![IrArg::Node(plan.x), i64_list_arg(&NHWC_ORDER)],
            meta::from_node(&source, plan.x_nhwc.clone()),
        );
        ctx.create_node(
            NodeOp::CallFunction,
            format!("{name}_tosa"),
            TOSA_RESIZE,
            vec![
                IrArg::Node(pre),
                IrArg::Node(scale),
                IrArg::Node(offset),
                IrArg::Node(border),
            ],
            HashMap::from([(
                "resize_mode".to_string(),
                IrArg::Str(plan.mode.name().to_string()),
            )]),
            meta::inheriting_qparams(&source, plan.resized.clone()),
        )
    });
    if plan.int48 {
        set_special_dtype(ctx.meta_mut(resize), SpecialDtype::Int48);
    }
    let replacement = match &plan.rescale {
        Some(rescale) => ctx.inserting_after(resize, |ctx| {
            ctx.call_function(
                format!("{name}_rescale"),
                TOSA_RESCALE,
                vec![
                    IrArg::Node(resize),
                    IrArg::Dtype(rescale.dtype.clone()),
                    IrArg::List(vec![IrArg::Float(rescale.scale)]),
                    IrArg::Int(0),
                    IrArg::Int(0),
                ],
                meta::from_node(&source, rescale.value.clone()),
            )
        }),
        None => resize,
    };
    let post = ctx.inserting_after(replacement, |ctx| {
        ctx.call_function(
            format!("{name}_nchw"),
            PERMUTE_COPY,
            vec![IrArg::Node(replacement), i64_list_arg(&NHWC_INVERSE_ORDER)],
            meta::from_node(&source, plan.output.clone()),
        )
    });
    ctx.replace_all_uses_with(id, post);
    ctx.erase_node(id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{NodeMeta, QParam};
    use crate::layout::dtype::special_dtype;
    use crate::layout::testing::{ints, node, LayoutGraph, PERMUTE, RESCALE};
    use crate::test_ir::qparam;
    use std::f64::consts::{E, PI, SQRT_2};

    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";
    const I32: &str = "torch.int32";
    const F32: &str = "torch.float32";
    const N: ResizeMode = ResizeMode::Nearest;
    const B: ResizeMode = ResizeMode::Bilinear;

    /// `IrArg` has no `PartialEq`; its derived `Debug` is exact.
    fn assert_args(actual: &[IrArg], expected: &[IrArg]) {
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }

    fn assert_arg(actual: Option<&IrArg>, expected: &IrArg) {
        assert_eq!(format!("{actual:?}"), format!("{:?}", Some(expected)));
    }

    fn int16_qparam(scale: f64) -> QParam {
        QParam::per_tensor(scale, 0, -32767, 32767, I16.to_string())
    }

    /// `x -> <target>(x, args...) -> output`: the upsample holds RETAB's float32
    /// val for a quantized activation and `qparam` on both maps.
    fn upsample_graph(
        target: &str,
        input: &[i64],
        input_dtype: &str,
        args: Vec<IrArg>,
        output: &[i64],
        qparam: Option<QParam>,
    ) -> (LayoutGraph, String) {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", input, input_dtype);
        let mut all_args = vec![node(x)];
        all_args.extend(args);
        let upsample = g.add_unrefreshed(target, all_args);
        let mut meta = NodeMeta::tensor(output.to_vec(), F32);
        if let Some(qparam) = qparam {
            meta.set_both_qparams(qparam);
        } else {
            meta.set_tensor_facts(TensorFacts::new(output.to_vec(), input_dtype));
        }
        g.graph_mut().node_mut(upsample).meta = meta;
        g.output(&[upsample]);
        let name = g.name(upsample);
        (g, name)
    }

    fn floats(values: &[f64]) -> IrArg {
        IrArg::List(values.iter().map(|&v| IrArg::Float(v)).collect())
    }

    fn by_name<'a>(g: &'a LayoutGraph, name: &str) -> &'a IrNode {
        g.graph().node(g.id(name))
    }

    fn qparam_keys(map: Option<&crate::fx_ir::QParamMap>) -> Option<Vec<usize>> {
        map.map(|map| map.iter().map(|(&key, _)| key).collect())
    }

    fn assert_declined(g: &mut LayoutGraph, reason: &str) {
        crate::stock_fallback::take();
        let before = g
            .targets()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(!IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_eq!(g.targets(), before, "a declined rewrite changes nothing");
        let request = crate::stock_fallback::take().expect("a stock fallback request");
        assert!(request.contains(reason), "{request}");
    }

    /// `(input, output, align_corners, mode, scale_factor, expected)`.
    type ParityRow = (i64, i64, bool, ResizeMode, Option<f64>, ResizeAxis);

    /// `get_resize_parameters_1d` against rc4's own function, run in
    /// .venv-et16 (sympy 1.14): every branch (nearest integer upscale,
    /// fractional and downscale; bilinear half-pixel and `align_corners`,
    /// including its size-1 fallback to `out / in`), and nearest scale factors
    /// as doubles and float32 values, near-integers and irrationals, whose
    /// `nsimplify(tolerance=1e-6)` takes `limit_denominator`'s semiconvergent
    /// path. Each row's expectation is
    /// `RewriteUpsamplePass.get_resize_parameters_1d(input, output,
    /// align_corners, mode, scale_factor)` evaluated there; a scale-factor row
    /// has `output = int(input * scale_factor)`, as torch sizes it.
    #[test]
    fn resize_parameters_match_rc4() {
        #[rustfmt::skip]
        let rows: &[ParityRow] = &[
        (1, 1, false, N, None, [2, 2, 0, 0]),
        (1, 2, false, N, None, [4, 2, -1, 1]),
        (1, 5, false, N, None, [10, 2, -4, 4]),
        (2, 1, false, N, None, [2, 4, -1, -3]),
        (2, 2, false, N, None, [2, 2, 0, 0]),
        (2, 3, false, N, None, [6, 4, -3, -1]),
        (2, 4, false, N, None, [4, 2, -1, 1]),
        (2, 7, false, N, None, [14, 4, -7, 3]),
        (3, 1, false, N, None, [2, 6, -1, -5]),
        (3, 2, false, N, None, [4, 6, -2, -4]),
        (3, 6, false, N, None, [4, 2, -1, 1]),
        (3, 7, false, N, None, [14, 6, -7, 1]),
        (3, 12, false, N, None, [8, 2, -3, 3]),
        (3, 24, false, N, None, [16, 2, -7, 7]),
        (4, 2, false, N, None, [2, 4, -1, -3]),
        (4, 3, false, N, None, [6, 8, -3, -5]),
        (4, 6, false, N, None, [6, 4, -3, -1]),
        (4, 8, false, N, None, [4, 2, -1, 1]),
        (4, 10, false, N, None, [10, 4, -5, 1]),
        (5, 3, false, N, None, [6, 10, -3, -7]),
        (5, 8, false, N, None, [16, 10, -8, -2]),
        (5, 9, false, N, None, [18, 10, -9, -1]),
        (5, 10, false, N, None, [4, 2, -1, 1]),
        (6, 4, false, N, None, [4, 6, -2, -4]),
        (6, 9, false, N, None, [6, 4, -3, -1]),
        (7, 3, false, N, None, [6, 14, -3, -11]),
        (7, 10, false, N, None, [20, 14, -10, -4]),
        (8, 3, false, N, None, [6, 16, -3, -13]),
        (8, 12, false, N, None, [6, 4, -3, -1]),
        (8, 16, false, N, None, [4, 2, -1, 1]),
        (9, 3, false, N, None, [2, 6, -1, -5]),
        (9, 4, false, N, None, [8, 18, -4, -14]),
        (12, 5, false, N, None, [10, 24, -5, -19]),
        (16, 2, false, N, None, [2, 16, -1, -15]),
        (16, 33, false, N, None, [66, 32, -33, 1]),
        (32, 3, false, N, None, [6, 64, -3, -61]),
        (1, 1, false, B, None, [2, 2, 0, 0]),
        (1, 2, false, B, None, [4, 2, -1, 1]),
        (1, 5, false, B, None, [10, 2, -4, 4]),
        (2, 1, false, B, None, [2, 4, 1, -1]),
        (2, 2, false, B, None, [2, 2, 0, 0]),
        (2, 3, false, B, None, [6, 4, -1, 1]),
        (2, 4, false, B, None, [4, 2, -1, 1]),
        (2, 7, false, B, None, [14, 4, -5, 5]),
        (3, 1, false, B, None, [2, 6, 2, -2]),
        (3, 2, false, B, None, [4, 6, 1, -1]),
        (3, 6, false, B, None, [4, 2, -1, 1]),
        (3, 7, false, B, None, [14, 6, -4, 4]),
        (3, 12, false, B, None, [8, 2, -3, 3]),
        (3, 24, false, B, None, [16, 2, -7, 7]),
        (4, 2, false, B, None, [2, 4, 1, -1]),
        (4, 3, false, B, None, [6, 8, 1, -1]),
        (4, 6, false, B, None, [6, 4, -1, 1]),
        (4, 8, false, B, None, [4, 2, -1, 1]),
        (4, 10, false, B, None, [10, 4, -3, 3]),
        (5, 3, false, B, None, [6, 10, 2, -2]),
        (5, 8, false, B, None, [16, 10, -3, 3]),
        (5, 9, false, B, None, [18, 10, -4, 4]),
        (5, 10, false, B, None, [4, 2, -1, 1]),
        (6, 4, false, B, None, [4, 6, 1, -1]),
        (6, 9, false, B, None, [6, 4, -1, 1]),
        (7, 3, false, B, None, [6, 14, 4, -4]),
        (7, 10, false, B, None, [20, 14, -3, 3]),
        (8, 3, false, B, None, [6, 16, 5, -5]),
        (8, 12, false, B, None, [6, 4, -1, 1]),
        (8, 16, false, B, None, [4, 2, -1, 1]),
        (9, 3, false, B, None, [2, 6, 2, -2]),
        (9, 4, false, B, None, [8, 18, 5, -5]),
        (12, 5, false, B, None, [10, 24, 7, -7]),
        (16, 2, false, B, None, [2, 16, 7, -7]),
        (16, 33, false, B, None, [66, 32, -17, 17]),
        (32, 3, false, B, None, [6, 64, 29, -29]),
        (1, 1, true, B, None, [2, 2, 0, 0]),
        (1, 2, true, B, None, [4, 2, 0, 2]),
        (1, 5, true, B, None, [10, 2, 0, 8]),
        (2, 1, true, B, None, [2, 4, 0, -2]),
        (2, 2, true, B, None, [2, 2, 0, 0]),
        (2, 3, true, B, None, [4, 2, 0, 0]),
        (2, 4, true, B, None, [6, 2, 0, 0]),
        (2, 7, true, B, None, [12, 2, 0, 0]),
        (3, 1, true, B, None, [2, 6, 0, -4]),
        (3, 2, true, B, None, [2, 4, 0, 0]),
        (3, 6, true, B, None, [10, 4, 0, 0]),
        (3, 7, true, B, None, [6, 2, 0, 0]),
        (3, 12, true, B, None, [22, 4, 0, 0]),
        (3, 24, true, B, None, [46, 4, 0, 0]),
        (4, 2, true, B, None, [2, 6, 0, 0]),
        (4, 3, true, B, None, [4, 6, 0, 0]),
        (4, 6, true, B, None, [10, 6, 0, 0]),
        (4, 8, true, B, None, [14, 6, 0, 0]),
        (4, 10, true, B, None, [6, 2, 0, 0]),
        (5, 3, true, B, None, [2, 4, 0, 0]),
        (5, 8, true, B, None, [14, 8, 0, 0]),
        (5, 9, true, B, None, [4, 2, 0, 0]),
        (5, 10, true, B, None, [18, 8, 0, 0]),
        (6, 4, true, B, None, [6, 10, 0, 0]),
        (6, 9, true, B, None, [16, 10, 0, 0]),
        (7, 3, true, B, None, [2, 6, 0, 0]),
        (7, 10, true, B, None, [6, 4, 0, 0]),
        (8, 3, true, B, None, [4, 14, 0, 0]),
        (8, 12, true, B, None, [22, 14, 0, 0]),
        (8, 16, true, B, None, [30, 14, 0, 0]),
        (9, 3, true, B, None, [2, 8, 0, 0]),
        (9, 4, true, B, None, [6, 16, 0, 0]),
        (12, 5, true, B, None, [8, 22, 0, 0]),
        (16, 2, true, B, None, [2, 30, 0, 0]),
        (16, 33, true, B, None, [64, 30, 0, 0]),
        (32, 3, true, B, None, [4, 62, 0, 0]),
        (4, 8, false, N, Some(2.0), [4, 2, -1, 1]),
        (4, 6, false, N, Some(1.5), [6, 4, -3, -1]),
        (5, 8, false, N, Some(1.6), [16, 10, -8, -2]),
        (5, 8, false, N, Some(1.600000023841858), [16, 10, -8, -2]),
        (4, 6, false, N, Some(1.6), [16, 10, -8, -6]),
        (4, 6, false, N, Some(1.600000023841858), [16, 10, -8, -6]),
        (4, 10, false, N, Some(2.5), [10, 4, -5, 1]),
        (9, 3, false, N, Some(0.3333333333333333), [2, 6, -1, -5]),
        (9, 3, false, N, Some(0.3333333432674408), [2, 6, -1, -5]),
        (8, 4, false, N, Some(0.5), [2, 4, -1, -3]),
        (8, 2, false, N, Some(0.25), [2, 8, -1, -7]),
        (8, 6, false, N, Some(0.75), [6, 8, -3, -5]),
        (30, 3, false, N, Some(0.10000000149011612), [2, 20, -1, -19]),
        (30, 3, false, N, Some(0.1), [2, 20, -1, -19]),
        (8, 10, false, N, Some(1.25), [10, 8, -5, -3]),
        (5, 6, false, N, Some(1.2000000476837158), [12, 10, -6, -4]),
        (5, 6, false, N, Some(1.2), [12, 10, -6, -4]),
        (3, 9, false, N, Some(3.0), [6, 2, -2, 2]),
        (9, 6, false, N, Some(0.6666666865348816), [4, 6, -2, -4]),
        (9, 6, false, N, Some(0.6666666666666666), [4, 6, -2, -4]),
        (4, 7, false, N, Some(1.75), [14, 8, -7, -1]),
        (10, 3, false, N, Some(0.30000001192092896), [6, 20, -3, -17]),
        (10, 3, false, N, Some(0.3), [6, 20, -3, -17]),
        (5, 5, false, N, Some(1.0), [2, 2, 0, 0]),
        (3, 7, false, N, Some(2.3333333333333335), [14, 6, -7, 1]),
        (3, 6, false, N, Some(2.3333332538604736), [14, 6, -7, -5]),
        (2, 8, false, N, Some(4.0), [8, 2, -3, 3]),
        (3, 9, false, N, Some(PI), [6253070, 1990414, -3126535, 290637]),
        (3, 8, false, N, Some(E), [2168966, 797918, -1084483, 163011]),
        (9, 9, false, N, Some(1.0000001), [2, 2, 0, 0]),
        (7, 9, false, N, Some(SQRT_2), [1331714, 941664, -665857, -1122829]),
        (9, 8, false, N, Some(0.999999), [1999998, 2000000, -999999, -2999983]),
        (7, 9, false, N, Some(1.4142135381698608), [109726, 77588, -54863, -92515]),
        (9, 9, false, N, Some(1.0000005), [2000002, 2000000, -1000001, -1000017]),
        ];
        assert_eq!(rows.len(), 142);
        for &(input, output, align_corners, mode, factor, expected) in rows {
            assert_eq!(
                resize_parameters_1d(input, output, align_corners, mode, factor),
                Ok(expected),
                "({input}, {output}, {align_corners}, {mode:?}, {factor:?})"
            );
        }
    }

    /// Python's own `fractions.Fraction.limit_denominator` doctests, and the
    /// float32 scale factors design risk 6 names.
    #[test]
    fn limit_denominator_and_nsimplify_match_python() {
        let pi = exact_fraction(PI).unwrap();
        assert_eq!(limit_denominator(pi.0, pi.1, 10), (22, 7));
        assert_eq!(limit_denominator(pi.0, pi.1, 100), (311, 99));
        assert_eq!(limit_denominator(4321, 8765, 10000), (4321, 8765));
        for (factor, expected) in [
            (1.6, (8, 5)),
            (f64::from(1.6f32), (8, 5)),
            (2.5, (5, 2)),
            (1.0 / 3.0, (1, 3)),
            (f64::from(1.0f32 / 3.0), (1, 3)),
            (2.0, (2, 1)),
            (0.5, (1, 2)),
            (1.0000001, (1, 1)),
        ] {
            assert_eq!(nsimplify_scale_factor(factor), Ok(expected), "{factor}");
        }
        for factor in [0.0, -2.0, f64::INFINITY, f64::NAN, 1e-300, 1e300] {
            assert!(nsimplify_scale_factor(factor).is_err(), "{factor}");
        }
        assert_eq!(exact_fraction(0.75), Some((3, 4)));
        assert_eq!(exact_fraction(6.0), Some((6, 1)));
    }

    /// rc4's nearest 5 -> 8 (design §3.3, critique C): scale [16, 10], offset
    /// -8, border -2 per axis, where RETAB's dim-order emitter computes -3 and 3.
    #[test]
    fn nearest_int8_upsample_becomes_const_shapes_permutes_and_resize() {
        let q = qparam(0.017, 4, I8);
        let (mut g, name) = upsample_graph(
            UPSAMPLE_NEAREST2D_VEC,
            &[1, 3, 5, 5],
            I8,
            vec![ints(&[8, 8]), IrArg::None],
            &[1, 3, 8, 8],
            Some(q.clone()),
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        g.assert_targets(&[
            TOSA_CONST_SHAPE,
            TOSA_CONST_SHAPE,
            TOSA_CONST_SHAPE,
            PERMUTE,
            TOSA_RESIZE,
            PERMUTE,
        ]);
        assert!(g.graph().by_name(&name).is_none_or(|node| !node.alive));
        for (suffix, values) in [
            ("scale", &[16, 10, 16, 10][..]),
            ("offset", &[-8, -8][..]),
            ("border", &[-2, -2][..]),
        ] {
            let shape = by_name(&g, &format!("{name}_{suffix}"));
            assert_args(&shape.args, &[ints(values)]);
            assert!(shape.meta.tensor_facts().is_none());
            assert_eq!(special_dtype(&shape.meta), Some(SpecialDtype::Shape));
            assert_eq!(qparam_keys(shape.meta.input_qparams()), Some(vec![]));
            assert_eq!(qparam_keys(shape.meta.output_qparams()), Some(vec![]));
        }
        let pre = by_name(&g, &format!("{name}_nhwc"));
        assert_args(&pre.args, &[node(g.id("x")), ints(&[0, 2, 3, 1])]);
        assert_eq!(g.shape(pre.id), [1, 5, 5, 3]);
        assert_eq!(g.dtype(pre.id), I8);
        // Critique A1: only the RESIZE keeps the upsample's qparams.
        assert_eq!(qparam_keys(pre.meta.input_qparams()), Some(vec![]));
        let resize = by_name(&g, &format!("{name}_tosa"));
        assert_args(
            &resize.args,
            &[
                node(pre.id),
                node(g.id(&format!("{name}_scale"))),
                node(g.id(&format!("{name}_offset"))),
                node(g.id(&format!("{name}_border"))),
            ],
        );
        assert_eq!(
            resize.kwarg("resize_mode").and_then(IrArg::as_str),
            Some("nearest")
        );
        assert_eq!(
            (g.shape(resize.id), g.dtype(resize.id)),
            (&[1, 8, 8, 3][..], I8)
        );
        assert_eq!(resize.meta.input_qparam(), Some(&q));
        assert_eq!(resize.meta.output_qparam(), Some(&q));
        assert_eq!(special_dtype(&resize.meta), None);
        let post = by_name(&g, &format!("{name}_nchw"));
        assert_args(&post.args, &[node(resize.id), ints(&[0, 3, 1, 2])]);
        assert_eq!(
            (g.shape(post.id), g.dtype(post.id)),
            (&[1, 3, 8, 8][..], I8)
        );
        assert_eq!(qparam_keys(post.meta.output_qparams()), Some(vec![]));
        for id in [pre.id, resize.id, post.id] {
            assert_eq!(
                g.graph().node(id).meta.val_dim_order(),
                Some(&[0, 1, 2, 3][..])
            );
        }
        let output = g.graph().output_node().unwrap();
        assert!(output.args[0].contains_node(post.id));
    }

    /// Design F10 and `rewrite_upsample.py:258-289`: an int8 bilinear RESIZE
    /// produces int32, and a RESCALE by `1 / (scale_y_n * scale_x_n)` brings it
    /// back to the upsample's int8, with empty qparam maps.
    #[test]
    fn int8_bilinear_resize_is_int32_then_rescaled() {
        let q = qparam(0.016, 12, I8);
        let (mut g, name) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![IrArg::None, IrArg::Bool(false), floats(&[2.0, 2.0])],
            &[1, 3, 8, 8],
            Some(q.clone()),
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        g.assert_targets(&[
            TOSA_CONST_SHAPE,
            TOSA_CONST_SHAPE,
            TOSA_CONST_SHAPE,
            PERMUTE,
            TOSA_RESIZE,
            RESCALE,
            PERMUTE,
        ]);
        let resize = by_name(&g, &format!("{name}_tosa"));
        assert_args(
            &by_name(&g, &format!("{name}_scale")).args,
            &[ints(&[4, 2, 4, 2])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_offset")).args,
            &[ints(&[-1, -1])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_border")).args,
            &[ints(&[1, 1])],
        );
        assert_eq!(
            resize.kwarg("resize_mode").and_then(IrArg::as_str),
            Some("bilinear")
        );
        assert_eq!(
            (g.shape(resize.id), g.dtype(resize.id)),
            (&[1, 8, 8, 3][..], I32)
        );
        assert_eq!(special_dtype(&resize.meta), None);
        let rescale = by_name(&g, &format!("{name}_rescale"));
        assert_args(
            &rescale.args,
            &[
                node(resize.id),
                IrArg::Dtype(I8.into()),
                IrArg::List(vec![IrArg::Float(0.0625)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
        );
        assert_eq!(
            (g.shape(rescale.id), g.dtype(rescale.id)),
            (&[1, 8, 8, 3][..], I8)
        );
        assert_eq!(qparam_keys(rescale.meta.input_qparams()), Some(vec![]));
        assert_eq!(qparam_keys(rescale.meta.output_qparams()), Some(vec![]));
        let post = by_name(&g, &format!("{name}_nchw"));
        assert_arg(post.args.first(), &node(rescale.id));
        assert_eq!(
            (g.shape(post.id), g.dtype(post.id)),
            (&[1, 3, 8, 8][..], I8)
        );
    }

    /// `:274-277`: an int16 bilinear RESIZE stays int16 but accumulates in
    /// INT48, and its RESCALE brings it back to int16.
    #[test]
    fn int16_bilinear_resize_is_int48() {
        let (mut g, name) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            I16,
            vec![IrArg::None, IrArg::Bool(false), floats(&[2.0, 2.0])],
            &[1, 3, 8, 8],
            Some(int16_qparam(0.000244140625)),
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        let resize = by_name(&g, &format!("{name}_tosa"));
        assert_eq!(g.dtype(resize.id), I16);
        assert_eq!(special_dtype(&resize.meta), Some(SpecialDtype::Int48));
        let rescale = by_name(&g, &format!("{name}_rescale"));
        assert_arg(rescale.args.get(1), &IrArg::Dtype(I16.into()));
        assert_eq!(g.dtype(rescale.id), I16);
        assert_eq!(special_dtype(&rescale.meta), None);
    }

    /// rc4's `align_corners` 5 -> 9: ratio 8/4 = 2, offset and border 0.
    #[test]
    fn align_corners_uses_the_corner_ratio() {
        let (mut g, name) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 5, 5],
            I8,
            vec![ints(&[9, 9]), IrArg::Bool(true), IrArg::None],
            &[1, 3, 9, 9],
            Some(qparam(0.017, 4, I8)),
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_args(
            &by_name(&g, &format!("{name}_scale")).args,
            &[ints(&[4, 2, 4, 2])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_offset")).args,
            &[ints(&[0, 0])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_border")).args,
            &[ints(&[0, 0])],
        );
        let rescale = by_name(&g, &format!("{name}_rescale"));
        assert_arg(
            rescale.args.get(2),
            &IrArg::List(vec![IrArg::Float(1.0 / 16.0)]),
        );
    }

    /// A float32-rounded nearest factor gives the same compact rational as the
    /// double (design risk 6): 1.600000023841858 is 8/5, not 13421773/8388608.
    #[test]
    fn nearest_scale_factors_are_recovered_like_nsimplify() {
        let factor = f64::from(1.6f32);
        let (mut g, name) = upsample_graph(
            UPSAMPLE_NEAREST2D_VEC,
            &[1, 3, 5, 5],
            I8,
            vec![IrArg::None, floats(&[factor, factor])],
            &[1, 3, 8, 8],
            Some(qparam(0.017, 4, I8)),
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_args(
            &by_name(&g, &format!("{name}_scale")).args,
            &[ints(&[16, 10, 16, 10])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_offset")).args,
            &[ints(&[-8, -8])],
        );
        assert_args(
            &by_name(&g, &format!("{name}_border")).args,
            &[ints(&[-2, -2])],
        );
    }

    /// A float bilinear upsample needs no RESCALE (rc4's FP path; RETAB lowers
    /// INT only, design F1).
    #[test]
    fn float_bilinear_resize_has_no_rescale() {
        let (mut g, name) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            F32,
            vec![IrArg::None, IrArg::Bool(false), floats(&[2.0, 2.0])],
            &[1, 3, 8, 8],
            None,
        );
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_eq!(g.count(RESCALE), 0);
        let resize = by_name(&g, &format!("{name}_tosa"));
        assert_eq!(g.dtype(resize.id), F32);
        // The upsample has no qparam maps, so neither does anything built from it.
        assert!(!resize.meta.has_qparam_metadata());
    }

    /// rc4's ChainedBilinearInterpolate (`test_upsample_nearest2d.py`): the
    /// second upsample reads the first's output permute.
    #[test]
    fn chained_upsamples_are_both_rewritten() {
        let q = qparam(0.016, 12, I8);
        let (mut g, first) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![IrArg::None, IrArg::Bool(false), floats(&[2.0, 2.0])],
            &[1, 3, 8, 8],
            Some(q.clone()),
        );
        let first_id = g.id(&first);
        let second = g.add_unrefreshed(
            UPSAMPLE_BILINEAR2D_VEC,
            vec![
                node(first_id),
                IrArg::None,
                IrArg::Bool(false),
                floats(&[2.0, 2.0]),
            ],
        );
        let mut meta = NodeMeta::tensor(vec![1, 3, 16, 16], F32);
        meta.set_both_qparams(q);
        g.graph_mut().node_mut(second).meta = meta;
        let output = g.graph().output_node().unwrap().id;
        g.rewire(output, first_id, second);
        let second = g.name(second);
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_eq!(g.count(TOSA_RESIZE), 2);
        let second_pre = by_name(&g, &format!("{second}_nhwc"));
        assert_arg(
            second_pre.args.first(),
            &node(g.id(&format!("{first}_nchw"))),
        );
        assert_eq!(
            g.shape(by_name(&g, &format!("{second}_tosa")).id),
            [1, 16, 16, 3]
        );
    }

    /// RESIZE parameters rc4's validation rejects go to stock (critique B10):
    /// the exact 1/16 bilinear downscale, which rc4's
    /// DecomposeUnsupportedBilinearResizePass turns into an avg pool first.
    #[test]
    fn parameters_rc4_rejects_go_to_stock() {
        let (mut g, _) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 32, 32],
            I8,
            vec![ints(&[2, 2]), IrArg::Bool(false), IrArg::None],
            &[1, 3, 2, 2],
            Some(qparam(0.016, 12, I8)),
        );
        assert_declined(&mut g, "downscale");
    }

    #[test]
    fn malformed_upsamples_go_to_stock() {
        let q = || Some(qparam(0.016, 12, I8));
        // One scale factor for two axes: rc4's `scale_factors_yx[1]` raises.
        let (mut g, _) = upsample_graph(
            UPSAMPLE_NEAREST2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![IrArg::None, floats(&[2.0])],
            &[1, 3, 8, 8],
            q(),
        );
        assert_declined(&mut g, "one scale factor");
        // A rank-3 input: the RESIZE kernel needs rank 4.
        let (mut g, _) = upsample_graph(
            UPSAMPLE_NEAREST2D_VEC,
            &[3, 4, 4],
            I8,
            vec![ints(&[8, 8]), IrArg::None],
            &[3, 8, 8],
            q(),
        );
        assert_declined(&mut g, "rank 4");
        // A missing argument: rc4 unpacks four.
        let (mut g, _) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![ints(&[8, 8]), IrArg::Bool(false)],
            &[1, 3, 8, 8],
            q(),
        );
        assert_declined(&mut g, "3 positional arguments");
        // An output that disagrees with the upsample's val. rc4's border makes
        // the spatial sizes agree by construction (a nearest 2x factor with a
        // 9-wide val gives border 3 and 9 columns), so only a malformed val,
        // here one with four channels, can.
        let (mut g, _) = upsample_graph(
            UPSAMPLE_NEAREST2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![IrArg::None, floats(&[2.0, 2.0])],
            &[1, 4, 8, 8],
            q(),
        );
        assert_declined(&mut g, "value");
    }

    /// rc4 targets the `.vec` overloads only; a `.default` stays an aten op,
    /// which the serializer's explicit-mode guard sends to stock (X5).
    #[test]
    fn default_overloads_are_left_to_the_serializer_guard() {
        crate::stock_fallback::take();
        let (mut g, _) = upsample_graph(
            "aten.upsample_nearest2d.default",
            &[1, 3, 4, 4],
            I8,
            vec![ints(&[8, 8])],
            &[1, 3, 8, 8],
            Some(qparam(0.016, 12, I8)),
        );
        assert!(!IrRewriteUpsamplePass.run(g.graph_mut()));
        g.assert_targets(&["aten.upsample_nearest2d.default"]);
        assert_eq!(crate::stock_fallback::take(), None);
    }

    /// The rewrite creates call nodes only, so a region's placeholders, which
    /// the boundary contracts pair with the parent's operands by position, stay
    /// as they are (critique B2).
    #[test]
    fn the_rewrite_adds_no_placeholders() {
        let (mut g, _) = upsample_graph(
            UPSAMPLE_BILINEAR2D_VEC,
            &[1, 3, 4, 4],
            I8,
            vec![IrArg::None, IrArg::Bool(false), floats(&[2.0, 2.0])],
            &[1, 3, 8, 8],
            Some(qparam(0.016, 12, I8)),
        );
        let names = |g: &LayoutGraph| -> Vec<String> {
            g.graph()
                .live_placeholder_ids()
                .into_iter()
                .map(|id| g.name(id))
                .collect()
        };
        let before = names(&g);
        assert!(IrRewriteUpsamplePass.run(g.graph_mut()));
        assert_eq!(names(&g), before);
    }
}
