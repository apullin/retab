use crate::fx_activation::EluParams;
use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, IrNode, MetaValue, QParam};
use crate::fx_meta::POW_LOG_POSITIVE_LOWER_BOUND;
use crate::fx_target::{table_op_target_kind, EluTargetKind, TableOpTargetKind};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    InsertPrecomputedTableOpEffect, LoweringEffect, LoweringEffectBatch, TableTensorSpec,
};

pub struct IrInsertTableOpsEffectPass;

#[derive(Clone, Debug, PartialEq)]
struct TableOpSpec {
    kind: &'static str,
    scalar_arg: Option<f64>,
    scale: Option<f64>,
    input_scale: Option<f64>,
}

impl TableOpSpec {
    fn unary(kind: &'static str) -> Self {
        Self {
            kind,
            scalar_arg: None,
            scale: None,
            input_scale: None,
        }
    }

    fn scalar(kind: &'static str, scalar_arg: f64) -> Self {
        Self {
            kind,
            scalar_arg: Some(scalar_arg),
            scale: None,
            input_scale: None,
        }
    }

    /// softplus(x, beta, threshold) (upstream PR #23522): beta in `scalar_arg`,
    /// threshold in `scale`.
    fn softplus(beta: f64, threshold: f64) -> Self {
        Self {
            kind: "softplus",
            scalar_arg: Some(beta),
            scale: Some(threshold),
            input_scale: None,
        }
    }

    fn elu(alpha: f64, scale: f64, input_scale: f64) -> Self {
        Self {
            kind: "elu",
            scalar_arg: Some(alpha),
            scale: Some(scale),
            input_scale: Some(input_scale),
        }
    }
}

fn table_op_spec(node: &IrNode) -> Option<TableOpSpec> {
    match table_op_target_kind(&node.target)? {
        TableOpTargetKind::Unary(kind) => Some(TableOpSpec::unary(kind)),
        TableOpTargetKind::Gelu => match node.kwarg_str("approximate").unwrap_or("none") {
            "none" => Some(TableOpSpec::unary("gelu")),
            "tanh" => Some(TableOpSpec::unary("gelu_tanh")),
            _ => None,
        },
        TableOpTargetKind::PowTensorScalar => node
            .arg_numeric_f64(1)
            .map(|arg| TableOpSpec::scalar("pow", arg)),
        TableOpTargetKind::RemainderScalar => node
            .arg_numeric_f64(1)
            .map(|arg| TableOpSpec::scalar("remainder", arg)),
        TableOpTargetKind::Softplus => Some(TableOpSpec::softplus(
            node.arg_or_kwarg_numeric_f64(1, "beta").unwrap_or(1.0),
            node.arg_or_kwarg_numeric_f64(2, "threshold")
                .unwrap_or(20.0),
        )),
        TableOpTargetKind::LeakyRelu => Some(TableOpSpec::scalar(
            "leaky_relu",
            node.arg_or_kwarg_numeric_f64(1, "negative_slope")
                .unwrap_or(0.01),
        )),
        TableOpTargetKind::Elu(kind) => Some(elu_table_op_spec(node, kind)),
    }
}

fn elu_table_op_spec(node: &IrNode, kind: EluTargetKind) -> TableOpSpec {
    let params = EluParams::from_node(node, kind);
    TableOpSpec::elu(params.alpha, params.scale, params.input_scale)
}

fn table_quant_spec(qp: &QParam) -> Option<TableTensorSpec> {
    if qp.is_per_channel() || qp.scale == 0.0 {
        return None;
    }
    let dtype = qp.dtype();
    Some(TableTensorSpec {
        scale: qp.scale,
        zp: qp.zp,
        qmin: qp.qmin,
        qmax: qp.qmax,
        dtype,
    })
}

fn quantize_to_spec(value: f64, spec: &TableTensorSpec) -> Option<i64> {
    if value.is_nan() {
        // Domain-restricted ops (acosh for x<1, logit outside (0,1), rsqrt for
        // x<=0) yield NaN for out-of-domain dequantized inputs. torch casts
        // NaN->int as 0, and those table entries are never hit at runtime, so
        // emit 0 rather than aborting the whole precompute.
        return Some(0);
    }
    if value.is_infinite() {
        return Some(if value.is_sign_positive() {
            spec.qmax
        } else {
            spec.qmin
        });
    }
    Some(crate::constant_materialization::quantize_value_stock(
        value,
        spec.scale,
        spec.zp,
        spec.qmin,
        spec.qmax,
    ))
}

fn dequantize_code(code: i64, zp: i64, scale: f64) -> f64 {
    ((code - zp) as f32 * scale as f32) as f64
}

/// rc4's InsertTableOpsPass._validate_pow_log_positive_lower_bound. A LOG made from
/// `pow(base, exp)` carries the base's proven lower bound
/// (`fx_meta::POW_LOG_POSITIVE_LOWER_BOUND`), and LOG needs a positive input, so the
/// bound must stay positive after QDQ with the LOG input's qparams. `Err` holds a stock
/// fallback reason wherever rc4 raises.
fn check_pow_log_lower_bound(node: &IrNode) -> Result<(), String> {
    let bound = match node.meta.extras.get(POW_LOG_POSITIVE_LOWER_BOUND) {
        None => return Ok(()),
        Some(MetaValue::Float(bound)) => *bound,
        Some(invalid) => {
            return Err(format!(
                "invalid POW LOG lower bound: {}: {invalid:?}",
                node.name
            ))
        }
    };
    let Some(input) = node.meta.input_qparam() else {
        return Ok(());
    };
    if input.is_per_channel() {
        return Err(format!(
            "per-channel POW LOG input: {}: the lower-bound check needs per-tensor qparams",
            node.name
        ));
    }
    let code = crate::constant_materialization::quantize_value_stock(
        bound,
        input.scale,
        input.zp,
        input.qmin,
        input.qmax,
    );
    let dequantized = dequantize_code(code, input.zp, input.scale);
    if dequantized.is_finite() && dequantized > 0.0 {
        return Ok(());
    }
    Err(format!(
        "POW LOG lower bound not positive after quantization: {}: the base's proven lower \
         bound {bound} becomes {dequantized} at the LOG input (scale={}, zero_point={})",
        node.name, input.scale, input.zp
    ))
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn hardsigmoid(x: f64) -> f64 {
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

fn erf_approx(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// Inverse error function (Giles, 2010 -- single-precision accurate, ~1e-7,
/// well within INT8/INT16 quantization tolerance). Saturates domain edges.
fn erfinv_approx(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    let mut w = -((1.0 - x) * (1.0 + x)).ln();
    let mut p;
    if w < 5.0 {
        w -= 2.5;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_64 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
    } else {
        w = w.sqrt() - 3.0;
        p = -0.000_200_214_257;
        p = 0.000_100_950_558 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_44 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
    }
    p * x
}

fn eval_precomputable_table(spec: &TableOpSpec, x: f64) -> Option<f64> {
    match spec.kind {
        "rsqrt" => Some(x.sqrt().recip()),
        "ceil" => Some(x.ceil()),
        "erf" => Some(erf_approx(x)),
        "erfinv" => Some(erfinv_approx(x)),
        "floor" => Some(x.floor()),
        "round" => Some(x.round_ties_even()),
        "reciprocal" => Some(x.recip()),
        "logit" => Some((x / (1.0 - x)).ln()),
        "exp" => Some(x.exp()),
        "log" => Some(x.ln()),
        "log10" => Some(x.log10()),
        "sigmoid" => Some(sigmoid(x)),
        "atanh" => Some(x.atanh()),
        "acosh" => Some(x.acosh()),
        "tanh" => Some(x.tanh()),
        "tan" => Some(x.tan()),
        "acos" => Some(x.acos()),
        "asin" => Some(x.asin()),
        "sin" => Some(x.sin()),
        "cos" => Some(x.cos()),
        "sinh" => Some(x.sinh()),
        "cosh" => Some(x.cosh()),
        "atan" => Some(x.atan()),
        "asinh" => Some(x.asinh()),
        "expm1" => Some(x.exp_m1()),
        "log1p" => Some(x.ln_1p()),
        "sqrt" => Some(x.sqrt()),
        "hardsigmoid" => Some(hardsigmoid(x)),
        "hardswish" => Some(x * hardsigmoid(x)),
        "silu" => Some(x * sigmoid(x)),
        "gelu" => Some(0.5 * x * (1.0 + erf_approx(x * std::f64::consts::FRAC_1_SQRT_2))),
        "gelu_tanh" => {
            const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
            Some(0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x.powi(3))).tanh()))
        }
        "pow" => {
            let exponent = spec.scalar_arg?;
            let rounded = exponent.round();
            if (exponent - rounded).abs() < 1e-12
                && rounded >= i32::MIN as f64
                && rounded <= i32::MAX as f64
            {
                Some(x.powi(rounded as i32))
            } else {
                Some(x.powf(exponent))
            }
        }
        "elu" => {
            let alpha = spec.scalar_arg.unwrap_or(1.0);
            let scale = spec.scale.unwrap_or(1.0);
            let input_scale = spec.input_scale.unwrap_or(1.0);
            let selected = if x >= 0.0 {
                x
            } else {
                alpha * (input_scale * x).exp_m1()
            };
            Some(scale * selected)
        }
        "softplus" => {
            let (beta, threshold) = (spec.scalar_arg?, spec.scale?);
            // torch.nn.functional.softplus: linear above the threshold.
            Some(if beta * x > threshold {
                x
            } else {
                (beta * x).exp().ln_1p() / beta
            })
        }
        "leaky_relu" => {
            let negative_slope = spec.scalar_arg?;
            Some(if x > 0.0 { x } else { x * negative_slope })
        }
        "remainder" => {
            let divisor = spec.scalar_arg?;
            if divisor == 0.0 {
                return None;
            }
            Some(x - divisor * (x / divisor).floor())
        }
        _ => None,
    }
}

fn plan_precomputed_int8_table(
    node: &IrNode,
    spec: &TableOpSpec,
) -> Option<InsertPrecomputedTableOpEffect> {
    let input = table_quant_spec(node.meta.input_qparam()?)?;
    let output = table_quant_spec(node.meta.output_qparam()?)?;
    if input.dtype != DType::Int8 || output.dtype != DType::Int8 {
        return None;
    }

    // Stock clamps the canonical INT8 domain to the input range, so narrowed
    // domains still yield a full 256-entry table with duplicated edge values.
    let mut data = Vec::with_capacity(256);
    for q in i8::MIN as i64..=i8::MAX as i64 {
        let clamped = q.clamp(input.qmin, input.qmax);
        let dequant = dequantize_code(clamped, input.zp, input.scale);
        let value = eval_precomputable_table(spec, dequant)?;
        let q_out = quantize_to_spec(value, &output)?;
        let q_i8 = i8::try_from(q_out).ok()?;
        data.push(q_i8 as u8);
    }

    Some(InsertPrecomputedTableOpEffect {
        node_name: node.name.clone(),
        data,
        shape: vec![256],
        table_dtype: DType::Int8,
        input,
        output,
        rescale_lshift: 0,
        allow_missing: true,
    })
}

fn ceil_log2_u64(value: u64) -> i64 {
    if value <= 1 {
        0
    } else {
        (u64::BITS - (value - 1).leading_zeros()) as i64
    }
}

fn plan_precomputed_int16_table(
    node: &IrNode,
    spec: &TableOpSpec,
) -> Option<InsertPrecomputedTableOpEffect> {
    let dbg = crate::config::table_debug_enabled();
    let input = match node.meta.input_qparam().and_then(table_quant_spec) {
        Some(s) => s,
        None => {
            if dbg {
                eprintln!("[T16-bail] {} no/invalid input_qparam", spec.kind);
            }
            return None;
        }
    };
    let output = match node.meta.output_qparam().and_then(table_quant_spec) {
        Some(s) => s,
        None => {
            if dbg {
                eprintln!("[T16-bail] {} no/invalid output_qparam", spec.kind);
            }
            return None;
        }
    };
    if input.dtype != DType::Int16 {
        if dbg {
            eprintln!("[T16-bail] {} input.dtype={:?}", spec.kind, input.dtype);
        }
        return None;
    }
    if !matches!(output.dtype, DType::Int16 | DType::Int32) {
        if dbg {
            eprintln!("[T16-bail] {} output.dtype={:?}", spec.kind, output.dtype);
        }
        return None;
    }
    // Full INT16 range: 65535 (asymmetric, qmin=-32768) or 65534 (symmetric,
    // qmin=-32767, used by 16a8w). The 513-entry / 128-stride table covers both
    // (the clamp handles the top edge).
    if !matches!(input.qmax - input.qmin, 65_534 | 65_535) {
        if dbg {
            eprintln!(
                "[T16-bail] {} qrange={}",
                spec.kind,
                input.qmax - input.qmin
            );
        }
        return None;
    }

    let span = (input.qmax + 1 - input.qmin) as f64;
    let has_full_upper = input.qmax == i16::MAX as i64;
    let mut lut_values = Vec::with_capacity(513);
    for step in 0..=512 {
        // Stock linspaces qmin..=qmax+1 in 513 steps (int32, truncated).
        let raw_q = (input.qmin as f64 + step as f64 * span / 512.0) as i64;
        let clamped = raw_q.clamp(input.qmin, input.qmax);
        let masked = (((clamped as i16) as i32) & !0x7f) as i64;
        // TOSA interpolates raw codes [32640, 32767] against a virtual right
        // endpoint at affine code 32768; evaluate its real value there.
        let dequant = if step == 512 && has_full_upper {
            dequantize_code(i16::MAX as i64 + 1, input.zp, input.scale)
        } else {
            dequantize_code(masked, input.zp, input.scale)
        };
        let value = eval_precomputable_table(spec, dequant)?;
        lut_values.push(quantize_to_spec(value, &output)?);
    }

    let max_abs = lut_values
        .iter()
        .map(|&value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    // Small outputs already fit in signed int16; stock clamps the shift to 0.
    let rshift = (ceil_log2_u64(max_abs) + 1 - 16).max(0);
    let rescale_lshift = rshift - 7;
    let data = lut_values
        .into_iter()
        .flat_map(|value| ((value >> rshift) as i16).to_le_bytes())
        .collect();

    Some(InsertPrecomputedTableOpEffect {
        node_name: node.name.clone(),
        data,
        shape: vec![513],
        table_dtype: DType::Int16,
        input,
        output,
        rescale_lshift,
        allow_missing: true,
    })
}

fn plan_precomputed_table(
    node: &IrNode,
    spec: &TableOpSpec,
) -> Option<InsertPrecomputedTableOpEffect> {
    plan_precomputed_int8_table(node, spec).or_else(|| plan_precomputed_int16_table(node, spec))
}

/// The TABLE this pass plans for `node`, as `run` plans it: the table
/// constant's bytes, shape and dtype, and the left shift of the RESCALE that
/// follows a 16-bit table (0: none). `None` when `node` is not a quantized
/// table op or its table can't be precomputed. The layout port of
/// FuseDuplicateUsers compares these where stock compares TABLE nodes and
/// their table constants (`layout::fuse_duplicate_users`).
pub(crate) fn planned_table(node: &IrNode) -> Option<InsertPrecomputedTableOpEffect> {
    if !node.meta.has_nonempty_input_and_output_qparams() {
        return None;
    }
    plan_precomputed_table(node, &table_op_spec(node)?)
}

pub(crate) fn unsupported_table_node_names(
    graph: &IrGraph,
    effects: &LoweringEffectBatch,
) -> Vec<String> {
    use std::collections::HashSet;
    let planned: HashSet<&str> = effects
        .items
        .iter()
        .filter_map(|effect| match effect {
            LoweringEffect::InsertPrecomputedTableOp(effect) => Some(effect.node_name.as_str()),
            _ => None,
        })
        .collect();
    graph
        .live_call_function_ids()
        .into_iter()
        .filter_map(|id| {
            let node = graph.node(id);
            if !node.meta.has_nonempty_input_and_output_qparams() {
                return None;
            }
            table_op_spec(node)?;
            if planned.contains(node.name.as_str()) {
                None
            } else {
                Some(node.name.clone())
            }
        })
        .collect()
}

impl EffectfulIrPass for IrInsertTableOpsEffectPass {
    fn name(&self) -> &str {
        "IrInsertTableOpsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let effects = graph
            .live_call_function_ids()
            .into_iter()
            .filter_map(|id| {
                let node = graph.node(id);
                if !node.meta.has_nonempty_input_and_output_qparams() {
                    return None;
                }
                let spec = table_op_spec(node)?;
                // Stock raises here; let it.
                if let Err(reason) = check_pow_log_lower_bound(node) {
                    crate::stock_fallback::request(reason);
                    return None;
                }
                if let Some(effect) = plan_precomputed_table(node, &spec) {
                    return Some(LoweringEffect::InsertPrecomputedTableOp(effect));
                }
                crate::fold_fallback_log(&format!(
                    "TablePrecomputeSkipped target={} kind={} node={}",
                    node.target, spec.kind, node.name
                ));
                None
            })
            .collect::<Vec<_>>();

        EffectfulIrPassResult {
            modified: !effects.is_empty(),
            effects: LoweringEffectBatch { items: effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, QParam};
    use crate::test_ir::{call_default, call_with_meta};

    use super::*;

    fn qparam() -> QParam {
        QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string())
    }

    fn table_meta(qparam: QParam) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_both_qparams(qparam);
        meta
    }

    fn table_node(graph: &mut IrGraph, name: &str, target: &str, args: Vec<IrArg>) -> usize {
        call_with_meta(graph, name, target, args, table_meta(qparam()))
    }

    fn table_node_with_qparam(
        graph: &mut IrGraph,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        qparam: QParam,
    ) -> usize {
        call_with_meta(graph, name, target, args, table_meta(qparam))
    }

    fn table_node_with_approximate(graph: &mut IrGraph, name: &str, approximate: &str) {
        let id = table_node(graph, name, "aten.gelu.default", Vec::new());
        graph.set_kwarg(id, "approximate", IrArg::Str(approximate.to_string()));
    }

    fn int16_qparam(scale: f64, qmin: i64, qmax: i64) -> QParam {
        QParam::per_tensor(scale, 0, qmin, qmax, "torch.int16".to_string())
    }

    fn int16_table_node(graph: &mut IrGraph, name: &str, scale: f64, qmin: i64, qmax: i64) {
        table_node_with_qparam(
            graph,
            name,
            "aten.sigmoid.default",
            Vec::new(),
            int16_qparam(scale, qmin, qmax),
        );
    }

    fn table_node_with_io_qparams(
        graph: &mut IrGraph,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        input: QParam,
        output: QParam,
    ) -> usize {
        let mut meta = NodeMeta::default();
        meta.set_input_qparam_at(0, input);
        meta.set_output_qparam_at(0, output);
        call_with_meta(graph, name, target, args, meta)
    }

    fn int16_io_qparam(scale: f64, zp: i64, qmin: i64, qmax: i64) -> QParam {
        QParam::per_tensor(scale, zp, qmin, qmax, "torch.int16".to_string())
    }

    fn int8_io_qparam(scale: f64, zp: i64, qmin: i64, qmax: i64) -> QParam {
        QParam::per_tensor(scale, zp, qmin, qmax, "torch.int8".to_string())
    }

    fn table_i16_bytes(effect: &InsertPrecomputedTableOpEffect) -> Vec<i16> {
        effect
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn plans_quantized_table_ops() {
        let mut graph = IrGraph::empty();
        table_node(&mut graph, "sigmoid", "aten.sigmoid.default", Vec::new());
        table_node(
            &mut graph,
            "pow",
            "aten.pow.Tensor_Scalar",
            vec![IrArg::Int(0), IrArg::Float(2.0)],
        );
        table_node(
            &mut graph,
            "elu",
            "aten.elu.default",
            vec![IrArg::Int(0), IrArg::Float(0.25)],
        );
        table_node_with_approximate(&mut graph, "gelu_none", "none");
        table_node_with_approximate(&mut graph, "gelu_tanh", "tanh");

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        assert!(result.modified);
        let effects = result
            .effects
            .items
            .iter()
            .map(|effect| match effect {
                LoweringEffect::InsertPrecomputedTableOp(effect) => effect.node_name.as_str(),
                _ => panic!("unexpected effect"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            effects,
            vec!["sigmoid", "pow", "elu", "gelu_none", "gelu_tanh"]
        );
    }

    #[test]
    fn plans_extended_precomputed_table_ops() {
        let mut graph = IrGraph::empty();
        let positive_qparam = QParam::per_tensor(0.01, -128, -128, 127, "torch.int8".to_string());
        let unit_domain_qparam = QParam::per_tensor(0.003, 0, -128, 127, "torch.int8".to_string());
        table_node_with_qparam(
            &mut graph,
            "reciprocal",
            "aten.reciprocal.default",
            Vec::new(),
            positive_qparam.clone(),
        );
        table_node_with_qparam(
            &mut graph,
            "log",
            "aten.log.default",
            Vec::new(),
            positive_qparam.clone(),
        );
        table_node_with_qparam(
            &mut graph,
            "log10",
            "aten.log10.default",
            Vec::new(),
            positive_qparam.clone(),
        );
        table_node_with_qparam(
            &mut graph,
            "sqrt",
            "aten.sqrt.default",
            Vec::new(),
            positive_qparam,
        );
        table_node(&mut graph, "tan", "aten.tan.default", Vec::new());
        table_node_with_qparam(
            &mut graph,
            "asin",
            "aten.asin.default",
            Vec::new(),
            unit_domain_qparam,
        );
        table_node(&mut graph, "expm1", "aten.expm1.default", Vec::new());

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.items.len(), 7);
        for effect in &result.effects.items {
            match effect {
                LoweringEffect::InsertPrecomputedTableOp(effect) => {
                    assert!(matches!(
                        effect.node_name.as_str(),
                        "reciprocal" | "log" | "log10" | "sqrt" | "tan" | "asin" | "expm1"
                    ));
                }
                other => panic!("expected precomputed table effect, got {other:?}"),
            }
        }
    }

    #[test]
    fn precomputes_nan_domain_table_values() {
        // log1p is precomputable in Rust (NaN samples quantize to 0), so it takes
        // the precomputed-table lane.
        let mut graph = IrGraph::empty();
        table_node(&mut graph, "log1p_nan", "aten.log1p.default", Vec::new());

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        assert!(result.modified);
        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::InsertPrecomputedTableOp(_)
        ));
    }

    #[test]
    fn precomputes_erfinv() {
        // erfinv is precomputable in Rust (Giles approximation), so it takes the
        // precomputed-table lane.
        let mut graph = IrGraph::empty();
        table_node(&mut graph, "erfinv", "aten.erfinv.default", Vec::new());

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        assert!(result.modified);
        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::InsertPrecomputedTableOp(_)
        ));
    }

    #[test]
    fn erfinv_table_saturates_exact_domain_edges() {
        let qparam = QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string());
        let mut graph = IrGraph::empty();
        table_node_with_qparam(
            &mut graph,
            "erfinv",
            "aten.erfinv.default",
            Vec::new(),
            qparam,
        );

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                let neg_one_index = (-2 - effect.input.qmin) as usize;
                let pos_one_index = (2 - effect.input.qmin) as usize;
                assert_eq!(effect.data[neg_one_index] as i8, -128);
                assert_eq!(effect.data[pos_one_index] as i8, 127);
                assert_eq!(effect.data[0] as i8, -128);
                assert_eq!(effect.data[255] as i8, 127);
            }
            other => panic!("expected precomputed table effect, got {other:?}"),
        }
    }

    #[test]
    fn skips_unquantized_or_unsupported_nodes() {
        let mut graph = IrGraph::empty();
        call_default(&mut graph, "sigmoid", "aten.sigmoid.default", Vec::new());
        table_node(&mut graph, "add", "aten.add.Tensor", Vec::new());

        let result = IrInsertTableOpsEffectPass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }

    #[test]
    fn plans_precomputed_int16_table_lane() {
        let mut graph = IrGraph::empty();
        int16_table_node(&mut graph, "sigmoid16", 1.0 / 32768.0, -32768, 32767);

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                assert_eq!(effect.shape, vec![513]);
                assert_eq!(effect.table_dtype, DType::Int16);
                assert_eq!(effect.output.dtype, DType::Int16);
                assert_eq!(effect.rescale_lshift, -7);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn falls_back_for_narrow_int16_range() {
        let mut graph = IrGraph::empty();
        int16_table_node(&mut graph, "sigmoid16_narrow", 0.25, -1024, 1023);

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }

    #[test]
    fn clamps_negative_shift_to_zero() {
        let mut graph = IrGraph::empty();
        int16_table_node(&mut graph, "sigmoid16_neg_shift", 0.25, -32768, 32767);

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                assert_eq!(effect.shape, vec![513]);
                assert_eq!(effect.rescale_lshift, -7);
                let table = table_i16_bytes(effect);
                assert_eq!(table.len(), 513);
                assert!(table.iter().take(256).all(|&v| v == 0));
                assert_eq!(table[256], 2);
                assert!(table.iter().skip(257).all(|&v| v == 4));
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn int16_softplus_uses_virtual_endpoint() {
        let mut graph = IrGraph::empty();
        table_node_with_io_qparams(
            &mut graph,
            "softplus16",
            "aten.softplus.default",
            vec![IrArg::Int(0)],
            int16_io_qparam(0.001, 0, -32768, 32767),
            int16_io_qparam(0.001, 0, -32768, 32767),
        );

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                assert_eq!(effect.shape, vec![513]);
                assert_eq!(effect.rescale_lshift, -7);
                let table = table_i16_bytes(effect);
                assert_eq!(table.len(), 513);
                assert_eq!(table[511], 32640);
                assert_eq!(table[512], 32767);
                assert_eq!(table[510], 32512);
                #[rustfmt::skip]
                let expected: [i16; 513] = [
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,
                    2,2,3,3,4,4,5,5,6,7,8,9,10,11,13,15,
                    17,19,21,24,27,31,35,40,45,51,58,66,74,84,95,108,
                    121,137,154,173,195,219,245,275,307,342,381,423,470,519,573,631,
                    693,759,829,903,982,1063,1149,1238,1331,1427,1525,1627,1731,1837,1946,2057,
                    2169,2284,2399,2516,2634,2754,2874,2995,3117,3240,3363,3487,3611,3736,3861,3987,
                    4113,4239,4365,4491,4618,4745,4872,4999,5126,5253,5381,5508,5636,5763,5891,6018,
                    6146,6274,6402,6529,6657,6785,6913,7041,7169,7297,7425,7553,7680,7808,7936,8064,
                    8192,8320,8448,8576,8704,8832,8960,9088,9216,9344,9472,9600,9728,9856,9984,10112,
                    10240,10368,10496,10624,10752,10880,11008,11136,11264,11392,11520,11648,11776,11904,12032,12160,
                    12288,12416,12544,12672,12800,12928,13056,13184,13312,13440,13568,13696,13824,13952,14080,14208,
                    14336,14464,14592,14720,14848,14976,15104,15232,15360,15488,15616,15744,15872,16000,16128,16256,
                    16384,16512,16640,16768,16896,17024,17152,17280,17408,17536,17664,17792,17920,18048,18176,18304,
                    18432,18560,18688,18816,18944,19072,19200,19328,19456,19584,19712,19840,19968,20096,20224,20352,
                    20480,20608,20736,20864,20992,21120,21248,21376,21504,21632,21760,21888,22016,22144,22272,22400,
                    22528,22656,22784,22912,23040,23168,23296,23424,23552,23680,23808,23936,24064,24192,24320,24448,
                    24576,24704,24832,24960,25088,25216,25344,25472,25600,25728,25856,25984,26112,26240,26368,26496,
                    26624,26752,26880,27008,27136,27264,27392,27520,27648,27776,27904,28032,28160,28288,28416,28544,
                    28672,28800,28928,29056,29184,29312,29440,29568,29696,29824,29952,30080,30208,30336,30464,30592,
                    30720,30848,30976,31104,31232,31360,31488,31616,31744,31872,32000,32128,32256,32384,32512,32640,
                    32767,
                ];
                assert_eq!(table.as_slice(), &expected);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn int16_sigmoid_negative_shift_matches_stock() {
        let mut graph = IrGraph::empty();
        table_node_with_io_qparams(
            &mut graph,
            "sigmoid16_neg",
            "aten.sigmoid.default",
            Vec::new(),
            int16_io_qparam(0.01, 0, -32768, 32767),
            int16_io_qparam(0.001, 0, -32768, 32767),
        );

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                assert_eq!(effect.rescale_lshift, -7);
                let table = table_i16_bytes(effect);
                #[rustfmt::skip]
                let expected: [i16; 513] = [
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,2,6,21,72,218,
                    500,782,928,979,994,998,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,
                    1000,
                ];
                assert_eq!(table.as_slice(), &expected);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn unsupported_helper_flags_unplanned_quantized_table_ops() {
        let mut graph = IrGraph::empty();
        int16_table_node(&mut graph, "sigmoid16_narrow", 0.25, -1024, 1023);
        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(!result.modified);
        let missing = unsupported_table_node_names(&graph, &result.effects);
        assert_eq!(missing, vec!["sigmoid16_narrow".to_string()]);

        let mut ok_graph = IrGraph::empty();
        int16_table_node(&mut ok_graph, "sigmoid16_ok", 0.25, -32768, 32767);
        let ok_result = IrInsertTableOpsEffectPass.run(&mut ok_graph);
        assert!(ok_result.modified);
        assert!(unsupported_table_node_names(&ok_graph, &ok_result.effects).is_empty());
    }

    #[test]
    fn pow_table_supports_negative_base_integer_exponents() {
        let spec = TableOpSpec::scalar("pow", 3.0);
        assert_eq!(eval_precomputable_table(&spec, -11.0), Some(-1331.0));
        assert_eq!(eval_precomputable_table(&spec, -2.0), Some(-8.0));
    }

    #[test]
    fn int8_narrow_domain_clamps_canonical_codes() {
        let mut graph = IrGraph::empty();
        table_node_with_io_qparams(
            &mut graph,
            "sigmoid_narrow",
            "aten.sigmoid.default",
            Vec::new(),
            int8_io_qparam(0.1, 0, -10, 10),
            int8_io_qparam(0.1, 0, -128, 127),
        );

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                assert_eq!(effect.shape, vec![256]);
                assert_eq!(effect.rescale_lshift, 0);
                let table: Vec<i8> =
                    effect.data.iter().map(|&b| b as i8).collect();
                #[rustfmt::skip]
                let expected: [i8; 256] = [
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,4,4,4,4,5,5,
                    5,5,5,6,6,6,6,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                ];
                assert_eq!(table.as_slice(), &expected);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn int8_softplus_beta2_uses_float32_quantize() {
        let mut graph = IrGraph::empty();
        table_node_with_io_qparams(
            &mut graph,
            "softplus_beta2",
            "aten.softplus.default",
            vec![
                IrArg::Int(0),
                IrArg::Float(2.0),
                IrArg::Float(5.0),
            ],
            int8_io_qparam(0.01, -128, -128, 127),
            int8_io_qparam(0.02, -128, -128, 127),
        );

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(result.modified);
        match &result.effects.items[0] {
            LoweringEffect::InsertPrecomputedTableOp(effect) => {
                let table: Vec<i8> =
                    effect.data.iter().map(|&b| b as i8).collect();
                assert_eq!(table.len(), 256);
                assert_eq!(table[125 + 128], -2);
                #[rustfmt::skip]
                let expected: [i8; 256] = [
                    -111,-110,-110,-110,-110,-109,-109,-109,-109,-108,-108,-108,-107,-107,-107,-107,
                    -106,-106,-106,-105,-105,-105,-105,-104,-104,-104,-103,-103,-103,-102,-102,-102,
                    -101,-101,-101,-100,-100,-100,-99,-99,-99,-98,-98,-98,-97,-97,-97,-96,
                    -96,-96,-95,-95,-94,-94,-94,-93,-93,-93,-92,-92,-91,-91,-91,-90,
                    -90,-89,-89,-89,-88,-88,-87,-87,-87,-86,-86,-85,-85,-85,-84,-84,
                    -83,-83,-83,-82,-82,-81,-81,-80,-80,-80,-79,-79,-78,-78,-77,-77,
                    -77,-76,-76,-75,-75,-74,-74,-74,-73,-73,-72,-72,-71,-71,-70,-70,
                    -69,-69,-69,-68,-68,-67,-67,-66,-66,-65,-65,-64,-64,-64,-63,-63,
                    -62,-62,-61,-61,-60,-60,-59,-59,-58,-58,-57,-57,-57,-56,-56,-55,
                    -55,-54,-54,-53,-53,-52,-52,-51,-51,-50,-50,-49,-49,-48,-48,-47,
                    -47,-47,-46,-46,-45,-45,-44,-44,-43,-43,-42,-42,-41,-41,-40,-40,
                    -39,-39,-38,-38,-37,-37,-36,-36,-35,-35,-34,-34,-33,-33,-32,-32,
                    -31,-31,-30,-30,-30,-29,-29,-28,-28,-27,-27,-26,-26,-25,-25,-24,
                    -24,-23,-23,-22,-22,-21,-21,-20,-20,-19,-19,-18,-18,-17,-17,-16,
                    -16,-15,-15,-14,-14,-13,-13,-12,-12,-11,-11,-10,-10,-9,-9,-8,
                    -8,-7,-7,-6,-6,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,0,
                ];
                assert_eq!(table.as_slice(), &expected);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    fn pow_log_node(graph: &mut IrGraph, input: QParam, output: QParam, bound: MetaValue) -> usize {
        let id = table_node_with_io_qparams(
            graph,
            "aten_log_default",
            "aten.log.default",
            Vec::new(),
            input,
            output,
        );
        graph
            .node_mut(id)
            .meta
            .set_extra(POW_LOG_POSITIVE_LOWER_BOUND, bound);
        id
    }

    fn run_and_take_fallback(graph: &mut IrGraph) -> (EffectfulIrPassResult, Option<String>) {
        crate::stock_fallback::take();
        let result = IrInsertTableOpsEffectPass.run(graph);
        (result, crate::stock_fallback::take())
    }

    fn planned_names(result: &EffectfulIrPassResult) -> Vec<&str> {
        result
            .effects
            .items
            .iter()
            .filter_map(|effect| match effect {
                LoweringEffect::InsertPrecomputedTableOp(effect) => Some(effect.node_name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// rc4 test_pow.py::test_pow_tensor_tensor_int_rejects_base_lower_bound_quantized_to_zero,
    /// with the LOG qparams stock derives for it. abs(x) + 0.25 is at least 0.25 by
    /// construction, but calibrating x on [0, 128] gives the LOG input an INT8 scale that
    /// maps 0.25 to the zero code. rc4 raises there; RETAB plans no table and asks for
    /// stock, which raises its own error.
    #[test]
    fn pow_log_rejects_base_lower_bound_quantized_to_zero() {
        let mut graph = IrGraph::empty();
        pow_log_node(
            &mut graph,
            int8_io_qparam(0.5026960968971252, -128, -128, 127),
            int8_io_qparam(0.0244597215205431, -71, -128, 127),
            MetaValue::Float(0.25),
        );

        let (result, fallback) = run_and_take_fallback(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
        assert_eq!(
            fallback.as_deref(),
            Some(
                "POW LOG lower bound not positive after quantization: aten_log_default: the \
                 base's proven lower bound 0.25 becomes 0 at the LOG input \
                 (scale=0.5026960968971252, zero_point=-128)"
            )
        );
    }

    /// test_pow.py's tosa_INT positive-base case, with stock's LOG qparams for base
    /// samples on [0.25, 1.75]: 0.25 dequantizes positive, so the LOG becomes a TABLE.
    #[test]
    fn pow_log_lowers_when_the_lower_bound_stays_positive() {
        let mut graph = IrGraph::empty();
        pow_log_node(
            &mut graph,
            int8_io_qparam(0.007840264588594437, -128, -128, 127),
            int8_io_qparam(0.0054337941110134125, 0, -128, 127),
            MetaValue::Float(0.25),
        );

        let (result, fallback) = run_and_take_fallback(&mut graph);

        assert_eq!(fallback, None);
        assert_eq!(planned_names(&result), vec!["aten_log_default"]);
    }

    /// The check quantizes in float32 and rounds half to even, as torch's
    /// quantize_per_tensor does: at scale 0.5, 0.25 sits on a half step and rounds to
    /// the zero code, while 0.2500001 and 0.75 round to positive codes.
    #[test]
    fn pow_log_bound_rounds_half_steps_like_torch() {
        for (bound, rejected) in [(0.25, true), (0.250_000_1, false), (0.75, false)] {
            let mut graph = IrGraph::empty();
            pow_log_node(
                &mut graph,
                int8_io_qparam(0.5, -128, -128, 127),
                int8_io_qparam(0.5, -128, -128, 127),
                MetaValue::Float(bound),
            );
            let (_, fallback) = run_and_take_fallback(&mut graph);
            assert_eq!(fallback.is_some(), rejected, "bound {bound}: {fallback:?}");
        }
    }

    /// Like rc4, the check skips a LOG without the bound and an unquantized LOG.
    #[test]
    fn pow_log_check_needs_the_bound_and_a_quantized_log() {
        let mut graph = IrGraph::empty();
        table_node_with_io_qparams(
            &mut graph,
            "plain_log",
            "aten.log.default",
            Vec::new(),
            int8_io_qparam(0.5026960968971252, -128, -128, 127),
            int8_io_qparam(0.0244597215205431, -71, -128, 127),
        );
        let unquantized = call_default(&mut graph, "float_log", "aten.log.default", Vec::new());
        graph
            .node_mut(unquantized)
            .meta
            .set_extra(POW_LOG_POSITIVE_LOWER_BOUND, MetaValue::Float(0.0));

        let (result, fallback) = run_and_take_fallback(&mut graph);

        assert_eq!(fallback, None);
        assert_eq!(planned_names(&result), vec!["plain_log"]);
    }

    /// rc4 also raises for a bound that isn't a number and for a per-channel LOG input.
    #[test]
    fn pow_log_with_invalid_bound_or_per_channel_input_asks_for_stock() {
        let mut graph = IrGraph::empty();
        pow_log_node(
            &mut graph,
            int8_io_qparam(0.01, -128, -128, 127),
            int8_io_qparam(0.01, 0, -128, 127),
            MetaValue::Str("True".to_string()),
        );
        let (result, fallback) = run_and_take_fallback(&mut graph);
        assert!(result.effects.items.is_empty());
        assert_eq!(
            fallback.as_deref(),
            Some(r#"invalid POW LOG lower bound: aten_log_default: Str("True")"#)
        );

        let per_channel = QParam {
            per_channel_scales: vec![0.01, 0.02],
            per_channel_zps: vec![-128, -128],
            ..int8_io_qparam(0.01, -128, -128, 127)
        };
        let mut graph = IrGraph::empty();
        pow_log_node(
            &mut graph,
            per_channel,
            int8_io_qparam(0.01, 0, -128, 127),
            MetaValue::Float(0.25),
        );
        let (result, fallback) = run_and_take_fallback(&mut graph);
        assert!(result.effects.items.is_empty());
        assert!(
            fallback.as_deref().is_some_and(
                |reason| reason.starts_with("per-channel POW LOG input: aten_log_default")
            ),
            "{fallback:?}"
        );
    }
}
