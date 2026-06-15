use crate::fx_ir::{EffectfulIrPass, EffectfulIrPassResult, IrGraph, IrNode, QParam};
use crate::fx_target::{table_op_target_kind, TableOpTargetKind};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    InsertPrecomputedTableOpEffect, LoweringEffect, LoweringEffectBatch, TableTensorSpec,
};

pub struct IrInsertTableOpsEffectPass;

#[derive(Clone, Debug, PartialEq)]
struct TableOpSpec {
    kind: &'static str,
    scalar_arg: Option<f64>,
}

fn table_op_spec(node: &IrNode) -> Option<TableOpSpec> {
    match table_op_target_kind(&node.target)? {
        TableOpTargetKind::Unary(kind) => Some(TableOpSpec {
            kind,
            scalar_arg: None,
        }),
        TableOpTargetKind::Gelu => match node.kwarg_str("approximate").unwrap_or("none") {
            "none" => Some(TableOpSpec {
                kind: "gelu",
                scalar_arg: None,
            }),
            "tanh" => Some(TableOpSpec {
                kind: "gelu_tanh",
                scalar_arg: None,
            }),
            _ => None,
        },
        TableOpTargetKind::PowTensorScalar => {
            node.arg_numeric_f64(1).map(|scalar_arg| TableOpSpec {
                kind: "pow",
                scalar_arg: Some(scalar_arg),
            })
        }
        TableOpTargetKind::RemainderScalar => {
            node.arg_numeric_f64(1).map(|scalar_arg| TableOpSpec {
                kind: "remainder",
                scalar_arg: Some(scalar_arg),
            })
        }
        TableOpTargetKind::Elu => Some(TableOpSpec {
            kind: "elu",
            scalar_arg: Some(node.arg_numeric_f64(1).unwrap_or(1.0)),
        }),
    }
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
    let rounded = (value / spec.scale).round_ties_even() as i64 + spec.zp;
    Some(rounded.clamp(spec.qmin, spec.qmax))
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
/// well within INT8/INT16 quantization tolerance). NaN out of [-1, 1].
fn erfinv_approx(x: f64) -> f64 {
    if x <= -1.0 || x >= 1.0 {
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

fn eval_precomputable_table(kind: &str, scalar_arg: Option<f64>, x: f64) -> Option<f64> {
    match kind {
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
            let exponent = scalar_arg?;
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
            let alpha = scalar_arg.unwrap_or(1.0);
            Some(if x >= 0.0 { x } else { alpha * x.exp_m1() })
        }
        "remainder" => {
            let divisor = scalar_arg?;
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
    if input.qmax - input.qmin != 255 {
        return None;
    }

    let mut data = Vec::with_capacity(256);
    for q in input.qmin..=input.qmax {
        let dequant = ((q - input.zp) as f64) * input.scale;
        let value = eval_precomputable_table(spec.kind, spec.scalar_arg, dequant)?;
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

    let mut lut_values = Vec::with_capacity(513);
    for step in 0..=512 {
        let raw_q = input.qmin + (step * 128);
        let clamped = raw_q.clamp(input.qmin, input.qmax);
        let masked = (((clamped as i16) as i32) & !0x7f) as i64;
        let dequant = ((masked - input.zp) as f64) * input.scale;
        let value = eval_precomputable_table(spec.kind, spec.scalar_arg, dequant)?;
        lut_values.push(quantize_to_spec(value, &output)?);
    }

    let max_abs = lut_values
        .iter()
        .map(|&value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    let rshift = ceil_log2_u64(max_abs) + 1 - 16;
    if rshift < 0 {
        if crate::config::table_debug_enabled() {
            eprintln!(
                "[TABLE16-None] kind={} max_abs={} rshift={} in(scale={},zp={},qmin={},qmax={}) out(scale={},zp={},dtype={:?})",
                spec.kind, max_abs, rshift, input.scale, input.zp, input.qmin, input.qmax,
                output.scale, output.zp, output.dtype,
            );
        }
        return None;
    }
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
        assert_eq!(result.effects.items.len(), 6);
        for effect in &result.effects.items {
            match effect {
                LoweringEffect::InsertPrecomputedTableOp(effect) => {
                    assert!(matches!(
                        effect.node_name.as_str(),
                        "reciprocal" | "log" | "sqrt" | "tan" | "asin" | "expm1"
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
    fn falls_back_when_int16_table_requires_negative_shift() {
        let mut graph = IrGraph::empty();
        int16_table_node(&mut graph, "sigmoid16_neg_shift", 0.25, -32768, 32767);

        let result = IrInsertTableOpsEffectPass.run(&mut graph);
        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }

    #[test]
    fn pow_table_supports_negative_base_integer_exponents() {
        let spec = TableOpSpec {
            kind: "pow",
            scalar_arg: Some(3.0),
        };
        assert_eq!(
            eval_precomputable_table(spec.kind, spec.scalar_arg, -11.0),
            Some(-1331.0)
        );
        assert_eq!(
            eval_precomputable_table(spec.kind, spec.scalar_arg, -2.0),
            Some(-8.0)
        );
    }
}
