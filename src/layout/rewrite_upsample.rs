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
//!   with the upsample's ([`dtype::operand_dtype`], [`dtype::value_dtype`];
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
        _ => reduced(scale_n, scale_d).ok_or_else(|| {
            format!("resize ratio {scale_n}/{scale_d} is not a number")
        })?,
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
    let (num, den) = exact_fraction(factor)
        .ok_or_else(|| format!("scale factor {factor} is out of range"))?;
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
    if node.kwargs.values().any(|value| !matches!(value, IrArg::None)) {
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
            Ok((x(input)?, ResizeMode::Bilinear, align_corners, scale_factors))
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
