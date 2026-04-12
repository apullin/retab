use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::fx_graph::{arg_extract, meta_val, FxGraphModule};
use crate::fx_pass::FxPass;
use crate::fx_utils;
use crate::runner_logging::log_line;

fn qp_get_value<'py>(
    qp: &Bound<'py, PyAny>,
    attr: &str,
    dict_key: Option<&str>,
) -> Option<Bound<'py, PyAny>> {
    qp.getattr(attr).ok().or_else(|| {
        dict_key
            .and_then(|key| meta_val(qp, key))
            .or_else(|| meta_val(qp, attr))
    })
}

macro_rules! qp_get_scalar {
    ($name:ident, $ty:ty) => {
        fn $name(qp: &Bound<'_, PyAny>, attr: &str, dict_key: Option<&str>, default: $ty) -> $ty {
            qp_get_value(qp, attr, dict_key)
                .and_then(|value| {
                    value.extract::<$ty>().ok().or_else(|| {
                        value
                            .get_item(0_usize)
                            .ok()
                            .and_then(|item| item.extract::<$ty>().ok())
                    })
                })
                .unwrap_or(default)
        }
    };
}

qp_get_scalar!(qp_get_f64, f64);
qp_get_scalar!(qp_get_i64, i64);

fn qp_get_f64_list(qp: &Bound<'_, PyAny>, attr: &str, default: f64) -> Vec<f64> {
    if let Some(val) = qp_get_value(qp, attr, None) {
        if let Ok(list) = val.extract::<Vec<f64>>() {
            if !list.is_empty() {
                return list;
            }
        }
        if let Ok(v) = val.extract::<f64>() {
            return vec![v];
        }
        if let Ok(len) = val.len() {
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                if let Ok(item) = val.get_item(i) {
                    if let Ok(v) = item.extract::<f64>() {
                        result.push(v);
                    }
                }
            }
            if !result.is_empty() {
                return result;
            }
        }
    }
    vec![default]
}

fn qp_get_dtype_str(qp: &Bound<'_, PyAny>, default: &str) -> String {
    if let Some(val) = qp_get_value(qp, "dtype", None) {
        if let Ok(s) = val.extract::<String>() {
            return s;
        }
        if let Ok(s) = val.str() {
            return s.to_string();
        }
    }
    default.to_string()
}

fn py_len(value: &Bound<'_, PyAny>) -> usize {
    value.len().unwrap_or(0)
}

fn meta_non_empty_item<'py>(
    meta: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(value) = meta_val(meta, key) else {
        return Ok(None);
    };
    Ok((py_len(&value) > 0).then_some(value))
}

fn first_qparam<'py>(qparams: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    qparams.get_item(0_usize).ok()
}

fn qparam_scale_zp_dtype(
    qp: &Bound<'_, PyAny>,
    fallback_zp: i64,
    default_dtype: &str,
) -> (f64, i64, String) {
    (
        qp_get_f64(qp, "scale", None, 1.0),
        qp_get_i64(qp, "zp", Some("zero_point"), fallback_zp),
        qp_get_dtype_str(qp, default_dtype),
    )
}

const I32_MIN: i64 = -2_147_483_648;
const I32_MAX: i64 = 2_147_483_647;

fn debug_skip(node_name: Option<&str>, message: impl std::fmt::Display) {
    if let Some(name) = node_name {
        eprintln!("[InsertRescale] skip '{}': {}", name, message);
    }
}

enum LayoutChainStep<T> {
    Return(T),
    Continue,
    Stop,
}

fn walk_single_user_layout_chain<'py, T>(
    start: &Bound<'py, PyAny>,
    mut visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>, &str) -> PyResult<LayoutChainStep<T>>,
) -> PyResult<Option<T>> {
    let mut current = start.clone();
    let mut visited: HashSet<String> = HashSet::new();

    loop {
        let name = FxGraphModule::node_name(&current)?;
        if !visited.insert(name) {
            return Ok(None);
        }

        let users = FxGraphModule::node_users(&current)?;
        if users.len() != 1 {
            return Ok(None);
        }
        let user = users[0].clone();
        let target = FxGraphModule::node_target_str(&user)?;
        match visit(&current, &user, &target)? {
            LayoutChainStep::Return(value) => return Ok(Some(value)),
            LayoutChainStep::Continue => current = user,
            LayoutChainStep::Stop => return Ok(None),
        }
    }
}

fn changes_rank_across_layout_chain(
    current: &Bound<'_, PyAny>,
    user: &Bound<'_, PyAny>,
    target: &str,
) -> bool {
    if !(target.contains("view")
        || target.contains("reshape")
        || target.contains("flatten")
        || target.contains("squeeze")
        || target.contains("unsqueeze"))
    {
        return false;
    }
    FxGraphModule::node_shape(current)
        .ok()
        .zip(FxGraphModule::node_shape(user).ok())
        .map(|(before, after)| before.len() != after.len())
        .unwrap_or(false)
}

struct QuantEnv<'py> {
    quantargs_cls: Bound<'py, PyAny>,
    int8: Bound<'py, PyAny>,
    int16: Bound<'py, PyAny>,
    int32: Bound<'py, PyAny>,
}

impl<'py> QuantEnv<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let quantargs_cls = py
            .import_bound("executorch.backends.arm._passes.quant_args")?
            .getattr("QuantArgs")?;
        let torch = py.import_bound("torch")?;
        Ok(Self {
            quantargs_cls,
            int8: torch.getattr("int8")?,
            int16: torch.getattr("int16")?,
            int32: torch.getattr("int32")?,
        })
    }

    fn dtype_for_str(&self, dtype_str: &str) -> &Bound<'py, PyAny> {
        if dtype_str.contains("int16") {
            &self.int16
        } else {
            &self.int8
        }
    }

    fn make_int32_qparam(&self, scale: f64) -> PyResult<Bound<'py, PyAny>> {
        self.quantargs_cls
            .call1((scale, 0_i64, I32_MIN, I32_MAX, &self.int32))
    }

    fn set_int32_qparams(
        &self,
        py: Python<'py>,
        meta: &Bound<'py, PyAny>,
        key: &str,
        entries: impl IntoIterator<Item = (usize, f64)>,
    ) -> PyResult<()> {
        let qparams = PyDict::new_bound(py);
        for (index, scale) in entries {
            qparams.set_item(index, self.make_int32_qparam(scale)?)?;
        }
        meta.set_item(key, &qparams)?;
        Ok(())
    }

    fn update_node_dtype_kwarg(&self, node: &Bound<'py, PyAny>) -> PyResult<()> {
        let kwargs = FxGraphModule::node_kwargs(node)?;
        let Ok(true) = kwargs.contains("dtype") else {
            return Ok(());
        };
        let Ok(kwargs) = kwargs.downcast::<PyDict>() else {
            return Ok(());
        };
        kwargs.set_item("dtype", &self.int32)?;
        node.setattr("kwargs", kwargs)?;
        Ok(())
    }
}

struct RescaleOps<'py, 'gm> {
    gm: &'gm FxGraphModule<'py>,
    helpers: Bound<'py, PyModule>,
}

impl<'py, 'gm> RescaleOps<'py, 'gm> {
    fn new(gm: &'gm FxGraphModule<'py>, helpers: Bound<'py, PyModule>) -> Self {
        Self { gm, helpers }
    }

    fn insert(
        &self,
        ref_node: &Bound<'py, PyAny>,
        input_node: &Bound<'py, PyAny>,
        dtype: &Bound<'py, PyAny>,
        scales: Vec<f64>,
        in_zp: i64,
        out_zp: i64,
        after: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.helpers.call_method1(
            "insert_rescale_node",
            (
                self.gm.gm_ref(),
                ref_node,
                input_node,
                dtype,
                scales,
                in_zp,
                out_zp,
                after,
            ),
        )
    }

    fn restamp_output_meta(
        &self,
        py: Python<'py>,
        meta_source: &Bound<'py, PyAny>,
        rescale: &Bound<'py, PyAny>,
        input_node: &Bound<'py, PyAny>,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        fx_utils::copy_meta(py, meta_source, rescale)?;
        self.helpers
            .call_method1("restamp_rescale_output_meta", (rescale, input_node, dtype))?;
        Ok(())
    }

    fn rewire_users(
        &self,
        old_input: &Bound<'py, PyAny>,
        replacement: &Bound<'py, PyAny>,
        users: &[Bound<'py, PyAny>],
    ) -> PyResult<()> {
        for user in users {
            if !user.is(replacement) {
                user.call_method1("replace_input_with", (old_input, replacement))?;
            }
        }
        Ok(())
    }

    fn refresh_fake_tensor(&self, node: &Bound<'py, PyAny>) -> PyResult<()> {
        self.helpers
            .call_method1("refresh_node_fake_tensor", (node,))?;
        Ok(())
    }
}

/// Where output rescale qparams should come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputQParamSource {
    /// Resolve from downstream consumers (default for parent graph + cond branches).
    Consumer,
    /// Resolve from this graph's own placeholders (while_loop body:
    /// output[i] feeds back as placeholder[i] on the next iteration).
    Carried,
}

/// FX-graph-direct implementation of Python's InsertRescalePass,
/// InsertRescaleInt32Pass, and the post-conv accumulator rescale step.
pub struct FxInsertRescalePass {
    pub output_qparam_source: OutputQParamSource,
}

impl FxInsertRescalePass {
    pub fn new() -> Self {
        Self {
            output_qparam_source: OutputQParamSource::Consumer,
        }
    }
    pub fn carried() -> Self {
        Self {
            output_qparam_source: OutputQParamSource::Carried,
        }
    }
}

/// Returns true for TOSA accumulator-producing ops that need INT32 -> INT8 rescale.
fn is_tosa_accumulator_op(target: &str) -> bool {
    let lowered = target.to_ascii_lowercase();
    lowered.contains("conv2d") || lowered.contains("depthwise_conv2d") || lowered.contains("conv3d")
}

fn is_layout_passthrough_user(target: &str) -> bool {
    let lowered = target.to_ascii_lowercase();
    lowered.contains("view")
        || lowered.contains("reshape")
        || lowered.contains("transpose")
        || lowered.contains("permute")
        || lowered.contains("squeeze")
        || lowered.contains("unsqueeze")
        || lowered.contains("flatten")
        || lowered.contains("contiguous")
}

fn extract_output_qparams<'py>(meta: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    meta_non_empty_item(meta, "output_qparams")
}

fn recover_downstream_output_qparams<'py>(
    start: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    walk_single_user_layout_chain(start, |_, user, target| {
        let user_meta = user.getattr("meta")?;
        if let Some(output_qparams) = extract_output_qparams(&user_meta)? {
            return Ok(LayoutChainStep::Return(output_qparams));
        }
        Ok(if is_layout_passthrough_user(target) {
            LayoutChainStep::Continue
        } else {
            LayoutChainStep::Stop
        })
    })
}

fn has_downstream_rescale(start: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(
        walk_single_user_layout_chain(start, |current, user, target| {
            let lowered = target.to_ascii_lowercase();
            if lowered.contains("rescale") {
                return Ok(LayoutChainStep::Return(true));
            }
            if !is_layout_passthrough_user(target)
                || changes_rank_across_layout_chain(current, user, &lowered)
            {
                return Ok(LayoutChainStep::Stop);
            }
            Ok(LayoutChainStep::Continue)
        })?
        .unwrap_or(false),
    )
}

fn producer_old_zero_point(arg_node: &Bound<'_, PyAny>, fallback_zp: i64) -> PyResult<i64> {
    let op = arg_node
        .getattr("op")
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default();
    if op == "call_function" {
        let target = FxGraphModule::node_target_str(arg_node)?;
        if classify_int32_op(&target).is_some() {
            return Ok(0);
        }
    }

    let meta = arg_node.getattr("meta")?;
    if let Some(output_qparams) = extract_output_qparams(&meta)? {
        if let Some(qp) = first_qparam(&output_qparams) {
            let (_, zp, dtype_str) = qparam_scale_zp_dtype(&qp, fallback_zp, "torch.int8");
            if dtype_str.contains("int32") {
                return Ok(0);
            }
            return Ok(zp);
        }
    }

    let Some(arg_fake_val) = meta_val(&meta, "val") else {
        return Ok(fallback_zp);
    };
    let is_int32 = arg_fake_val
        .getattr("dtype")
        .ok()
        .and_then(|dtype| dtype.str().ok())
        .map(|dtype| dtype.to_string_lossy().into_owned())
        .map(|dtype| dtype.contains("int32"))
        .unwrap_or(false);
    Ok(if is_int32 { 0 } else { fallback_zp })
}

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor")
}

/// Extract (scale, zero_point) from a DQ/Q node's args.
fn extract_scale_zp(args: &[Bound<'_, PyAny>]) -> PyResult<(f64, i64)> {
    let scale: f64 = arg_extract(&args, 1, 1.0);
    let zp: i64 = arg_extract(&args, 2, 0);
    Ok((scale, zp))
}

/// Op classification for INT32 rescale scale computation.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Int32OpKind {
    /// abs, eq, ge, gt, le, lt, maximum, minimum: use min_scale for all inputs
    MinScale,
    /// add, sub: use 2*max(lhs,rhs) / (1<<shift_bits)
    AddSub,
    /// mul, sum: keep each input's own scale
    Passthrough,
}

/// Check if a target is an INT32 accumulator op and return its kind.
fn classify_int32_op(target: &str) -> Option<Int32OpKind> {
    if target.contains("abs.default")
        || target.contains("eq.Tensor")
        || target.contains("ge.Tensor")
        || target.contains("gt.Tensor")
        || target.contains("le.Tensor")
        || target.contains("lt.Tensor")
        || target.contains("maximum.default")
        || target.contains("minimum.default")
    {
        Some(Int32OpKind::MinScale)
    } else if target.contains("add.Tensor") || target.contains("sub.Tensor") {
        Some(Int32OpKind::AddSub)
    } else if target.contains("mul.Tensor") || target.contains("sum.dim_IntList") {
        Some(Int32OpKind::Passthrough)
    } else {
        None
    }
}

/// Whether this target produces boolean output (no output rescale needed).
fn is_comparison_op(target: &str) -> bool {
    target.contains("eq.Tensor")
        || target.contains("ge.Tensor")
        || target.contains("gt.Tensor")
        || target.contains("le.Tensor")
        || target.contains("lt.Tensor")
}

/// Whether this target is min/max (output uses min_scale of rescaled inputs).
fn is_minmax_op(target: &str) -> bool {
    target.contains("maximum.default") || target.contains("minimum.default")
}

/// Whether this target is mul (output scale = product of input scales).
fn is_mul_op(target: &str) -> bool {
    target.contains("mul.Tensor")
}

/// Walk a node's producer chain to find output_qparams.
fn recover_producer_qparam<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let mut current = node.clone();
    for _ in 0..8 {
        let meta = current.getattr("meta")?;
        if let Some(oqp) = meta_non_empty_item(&meta, "output_qparams")? {
            return Ok(Some(oqp));
        }
        // Walk through rescale/layout-passthrough nodes
        let Ok(op) = current.getattr("op").and_then(|v| v.extract::<String>()) else {
            break;
        };
        if op != "call_function" && op != "placeholder" {
            break;
        }
        let Ok(args) = FxGraphModule::node_args(&current) else {
            break;
        };
        let Some(first_arg) = args.first() else { break };
        if !FxGraphModule::is_node(first_arg) {
            break;
        }
        current = first_arg.clone();
    }
    Ok(None)
}

/// Read input_qparams from node.meta, returning (keys, scales, zps, dtype_strs).
/// If a binary node has missing input_qparams, tries to recover them from
/// the actual arg nodes' producer chains (handles buffer/captured args in
/// control-flow submodules).
fn read_input_qparams(
    node: &Bound<'_, PyAny>,
) -> PyResult<Option<(Vec<usize>, Vec<f64>, Vec<i64>, Vec<String>)>> {
    read_input_qparams_impl(node, false)
}

fn read_input_qparams_carried(
    node: &Bound<'_, PyAny>,
) -> PyResult<Option<(Vec<usize>, Vec<f64>, Vec<i64>, Vec<String>)>> {
    read_input_qparams_impl(node, true)
}

fn read_input_qparams_impl(
    node: &Bound<'_, PyAny>,
    extract_from_q_literals: bool,
) -> PyResult<Option<(Vec<usize>, Vec<f64>, Vec<i64>, Vec<String>)>> {
    let meta = node.getattr("meta")?;
    let input_qp = meta_non_empty_item(&meta, "input_qparams")?;

    // If input_qparams exists, use it directly
    if let Some(ref qp_dict) = input_qp {
        let keys: Vec<usize> = qp_dict
            .call_method0("keys")?
            .iter()?
            .filter_map(|k| k.ok().and_then(|k| k.extract().ok()))
            .collect();

        let mut scales = Vec::with_capacity(keys.len());
        let mut zps = Vec::with_capacity(keys.len());
        let mut dtype_strs = Vec::with_capacity(keys.len());

        for &idx in &keys {
            let qp = qp_dict.get_item(idx)?;
            let (scale, zp, dtype_str) = qparam_scale_zp_dtype(&qp, 0, "torch.int8");
            scales.push(scale);
            zps.push(zp);
            dtype_strs.push(dtype_str);
        }

        return Ok(Some((keys, scales, zps, dtype_strs)));
    }

    // Fallback: try to build input_qparams from the actual arg nodes.
    // This handles buffer/captured args in control-flow submodules where
    // fold_qdq didn't annotate the binary node because a placeholder
    // was missing qparams.
    let Ok(args) = FxGraphModule::node_args(node) else {
        return Ok(None);
    };
    let mut keys = Vec::new();
    let mut scales = Vec::new();
    let mut zps = Vec::new();
    let mut dtype_strs = Vec::new();

    let debug_carried = std::env::var("TOSA_TRACE").as_deref() == Ok("1");
    for (idx, arg) in args.iter().enumerate().take(2) {
        if !FxGraphModule::is_node(arg) {
            continue;
        }
        if debug_carried {
            let arg_name = FxGraphModule::node_name(arg).unwrap_or_default();
            let arg_op = FxGraphModule::node_op(arg).unwrap_or_default();
            let arg_meta = arg.getattr("meta").ok();
            let has_oqp = arg_meta
                .as_ref()
                .and_then(|m| m.get_item("output_qparams").ok())
                .map(|v| v.is_truthy().unwrap_or(false))
                .unwrap_or(false);
            let has_iqp = arg_meta
                .as_ref()
                .and_then(|m| m.get_item("input_qparams").ok())
                .map(|v| v.is_truthy().unwrap_or(false))
                .unwrap_or(false);
            log_line(format_args!(
                "[InsertRescale-fallback] arg {} name={} op={} has_output_qparams={} has_input_qparams={}",
                idx, arg_name, arg_op, has_oqp, has_iqp
            ));
        }
        if let Some(oqp) = recover_producer_qparam(arg)? {
            if let Some(qp) = first_qparam(&oqp) {
                let (scale, zp, dtype_str) = qparam_scale_zp_dtype(&qp, 0, "torch.int8");
                if debug_carried {
                    log_line(format_args!(
                        "[InsertRescale-fallback] arg {} recovered scale={} zp={} dtype={}",
                        idx, scale, zp, dtype_str
                    ));
                }
                if dtype_str.contains("int8") || dtype_str.contains("int16") {
                    keys.push(idx);
                    scales.push(scale);
                    zps.push(zp);
                    dtype_strs.push(dtype_str);
                }
            }
        } else if extract_from_q_literals {
            // Walk through DQ to find Q node with literal scale/zp args
            let mut walk = arg.clone();
            for _ in 0..4 {
                let target = FxGraphModule::node_target_str(&walk).unwrap_or_default();
                if target.contains("quantize_per_tensor") && !target.contains("dequantize") {
                    let q_args = FxGraphModule::node_args(&walk)?;
                    if q_args.len() >= 6 {
                        let scale: f64 = q_args[1].extract().unwrap_or(1.0);
                        let zp: i64 = q_args[2].extract().unwrap_or(0);
                        let dtype_str = format!("{}", q_args[5]);
                        if dtype_str.contains("int8") || dtype_str.contains("int16") {
                            keys.push(idx);
                            scales.push(scale);
                            zps.push(zp);
                            dtype_strs.push(dtype_str);
                            if debug_carried {
                                log_line(format_args!(
                                    "[InsertRescale-fallback] arg {} Q-literal scale={} zp={}",
                                    idx, scale, zp
                                ));
                            }
                        }
                    }
                    break;
                }
                let w_args = FxGraphModule::node_args(&walk).unwrap_or_default();
                if w_args.is_empty() || !FxGraphModule::is_node(&w_args[0]) {
                    break;
                }
                walk = w_args[0].clone();
            }
        } else if debug_carried {
            log_line(format_args!(
                "[InsertRescale-fallback] arg {} no producer qparam found",
                idx
            ));
        }
    }

    if keys.is_empty() {
        return Ok(None);
    }
    Ok(Some((keys, scales, zps, dtype_strs)))
}

/// Compute the INT32 rescaled input scales based on op kind.
/// Returns a Vec of new scales (one per input key, in same order as keys).
fn compute_rescaled_input_scales(
    kind: Int32OpKind,
    scales: &[f64],
    dtype_strs: &[String],
) -> Vec<f64> {
    match kind {
        Int32OpKind::MinScale => {
            let min_scale = scales.iter().cloned().fold(f64::INFINITY, f64::min);
            vec![min_scale; scales.len()]
        }
        Int32OpKind::AddSub => {
            let max_scale = scales.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let max_scale_2x = 2.0 * max_scale;
            let is_int16 = dtype_strs
                .first()
                .map(|s| s.contains("int16"))
                .unwrap_or(false);
            let shift_bits: u32 = if is_int16 { 12 } else { 20 };
            let new_scale = max_scale_2x / (1u64 << shift_bits) as f64;
            vec![new_scale; scales.len()]
        }
        Int32OpKind::Passthrough => scales.to_vec(),
    }
}

fn compute_output_int32_scale(target: &str, rescaled_input_scales: &[f64]) -> Option<f64> {
    if is_comparison_op(target) {
        return None;
    }
    if is_minmax_op(target) {
        let min_scale = rescaled_input_scales
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        return Some(min_scale);
    }
    if is_mul_op(target) {
        let product: f64 = rescaled_input_scales.iter().product();
        return Some(product);
    }
    Some(rescaled_input_scales[0])
}

fn qparam_signature(qp: &Bound<'_, PyAny>, default_dtype: &str) -> (f64, i64, i64, i64, String) {
    (
        qp_get_f64(qp, "scale", None, 1.0),
        qp_get_i64(qp, "zp", Some("zero_point"), 0),
        qp_get_i64(qp, "qmin", None, -128),
        qp_get_i64(qp, "qmax", None, 127),
        qp_get_dtype_str(qp, default_dtype),
    )
}

impl FxPass for FxInsertRescalePass {
    fn name(&self) -> &str {
        "FxInsertRescale"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let rescale_ops = RescaleOps::new(gm, py.import_bound("fx_helpers")?);
        let quant = QuantEnv::new(py)?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut dq_q_pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_dq(&target) {
                continue;
            }
            let users = FxGraphModule::node_users(node)?;
            for user in &users {
                let user_op = FxGraphModule::node_op(user)?;
                if user_op != "call_function" {
                    continue;
                }
                let user_target = FxGraphModule::node_target_str(user)?;
                if is_q(&user_target) {
                    dq_q_pairs.push((node.clone(), user.clone()));
                }
            }
        }

        for (dq_node, q_node) in &dq_q_pairs {
            let dq_args = FxGraphModule::node_args(dq_node)?;
            let q_args = FxGraphModule::node_args(q_node)?;

            let (dq_scale, dq_zp) = extract_scale_zp(&dq_args)?;
            let (q_scale, q_zp) = extract_scale_zp(&q_args)?;

            let new_scale = dq_scale / q_scale;

            let dq_input = match dq_args.first() {
                Some(n) if FxGraphModule::is_node(n) => n.clone(),
                _ => continue,
            };

            let q_dtype = q_args.get(5).cloned().unwrap_or_else(|| quant.int8.clone());

            let rescale_node = rescale_ops.insert(
                dq_node,
                &dq_input,
                &q_dtype,
                vec![new_scale],
                dq_zp,
                q_zp,
                false,
            )?;

            rescale_ops.restamp_output_meta(py, q_node, &rescale_node, &dq_input, &q_dtype)?;

            FxGraphModule::replace_all_uses_with(q_node, &rescale_node)?;
            gm.erase_node(q_node)?;

            let dq_users = FxGraphModule::node_users(dq_node)?;
            if dq_users.is_empty() {
                gm.erase_node(dq_node)?;
            }

            modified = true;
        }

        // Phase 2: insert INT32 rescales around elementwise ops.
        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            let Some(op_kind) = classify_int32_op(&target) else {
                continue;
            };

            let qparams_result = if self.output_qparam_source == OutputQParamSource::Carried {
                read_input_qparams_carried(node)?
            } else {
                read_input_qparams(node)?
            };
            let Some((keys, scales, zps, dtype_strs)) = qparams_result else {
                if self.output_qparam_source == OutputQParamSource::Carried {
                    let name = FxGraphModule::node_name(node).unwrap_or_default();
                    log_line(format_args!(
                        "[InsertRescale-Carried] Phase 2 SKIP {}: no input_qparams for {}",
                        name, target
                    ));
                }
                continue;
            };

            let has_quantized_input = dtype_strs
                .iter()
                .any(|d| d.contains("int8") || d.contains("int16"));
            if !has_quantized_input {
                continue;
            }

            let rescaled_scales = compute_rescaled_input_scales(op_kind, &scales, &dtype_strs);
            let output_int32_scale = compute_output_int32_scale(&target, &rescaled_scales);

            let args = FxGraphModule::node_args(node)?;
            let mut rescaled_inputs = false;
            let mut seen_arg_names: HashSet<String> = HashSet::new();

            for (key_idx, &pos) in keys.iter().enumerate() {
                if pos >= args.len() {
                    continue;
                }

                let dtype_str = &dtype_strs[key_idx];
                if !dtype_str.contains("int8") && !dtype_str.contains("int16") {
                    continue;
                }

                let arg_node = &args[pos];
                if !FxGraphModule::is_node(arg_node) {
                    continue;
                }

                let arg_name = FxGraphModule::node_name(arg_node)?;
                if seen_arg_names.contains(&arg_name) {
                    continue;
                }
                seen_arg_names.insert(arg_name);

                let old_scale = scales[key_idx];
                let new_scale = rescaled_scales[key_idx];
                let rescale_scale = old_scale / new_scale;
                let old_zp = producer_old_zero_point(arg_node, zps[key_idx])?;
                let new_zp = 0_i64;

                let rescale = rescale_ops.insert(
                    arg_node,
                    arg_node,
                    &quant.int32,
                    vec![rescale_scale],
                    old_zp,
                    new_zp,
                    true,
                )?;

                node.call_method1("replace_input_with", (arg_node, &rescale))?;
                rescaled_inputs = true;
            }

            if !rescaled_inputs {
                continue;
            }

            modified = true;

            let meta = node.getattr("meta")?;
            quant.set_int32_qparams(
                py,
                &meta,
                "input_qparams",
                keys.iter()
                    .enumerate()
                    .map(|(key_idx, &pos)| (pos, rescaled_scales[key_idx])),
            )?;

            let mut outputs_was_rescaled = false;
            if let Some(out_int32_scale) = output_int32_scale {
                let Some(output_qp) = meta_non_empty_item(&meta, "output_qparams")? else {
                    continue;
                };

                let Some(oqp) = first_qparam(&output_qp) else {
                    continue;
                };
                let (out_orig_scale, out_orig_zp, out_orig_dtype_str) =
                    qparam_scale_zp_dtype(&oqp, 0, "torch.int8");

                if out_orig_dtype_str.contains("int8") || out_orig_dtype_str.contains("int16") {
                    let rescale_scale = out_int32_scale / out_orig_scale;
                    let users = FxGraphModule::node_users(node)?;
                    let rescale_out = rescale_ops.insert(
                        node,
                        node,
                        quant.dtype_for_str(&out_orig_dtype_str),
                        vec![rescale_scale],
                        0_i64,
                        out_orig_zp,
                        true,
                    )?;
                    rescale_ops.rewire_users(node, &rescale_out, &users)?;

                    outputs_was_rescaled = true;
                }
            }

            if outputs_was_rescaled {
                if let Some(out_int32_scale) = output_int32_scale {
                    quant.set_int32_qparams(
                        py,
                        &meta,
                        "output_qparams",
                        [(0_usize, out_int32_scale)],
                    )?;
                    quant.update_node_dtype_kwarg(node)?;
                }
            }

            rescale_ops.refresh_fake_tensor(node)?;
        }

        // Phase 2b: insert output rescales after quantized `where` nodes when
        // the branch tensors share a quantization domain that differs from the
        // annotated output domain.
        let all_nodes = gm.nodes()?;
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !target.contains("where.self") {
                continue;
            }

            let meta = node.getattr("meta")?;
            let Some(input_qparams) = meta_non_empty_item(&meta, "input_qparams")? else {
                continue;
            };
            let Some(output_qparams) = meta_non_empty_item(&meta, "output_qparams")? else {
                continue;
            };

            let Ok(then_qp) = input_qparams.get_item(1_usize) else {
                continue;
            };
            let Ok(else_qp) = input_qparams.get_item(2_usize) else {
                continue;
            };
            let Some(out_qp) = first_qparam(&output_qparams) else {
                continue;
            };

            let then_sig = qparam_signature(&then_qp, "torch.int8");
            let else_sig = qparam_signature(&else_qp, "torch.int8");
            if then_sig != else_sig {
                continue;
            }

            let out_sig = qparam_signature(&out_qp, "torch.int8");
            let output_dtype = meta_val(&meta, "val")
                .and_then(|v| v.getattr("dtype").ok())
                .and_then(|dt| dt.str().ok())
                .and_then(|s| s.extract::<String>().ok())
                .unwrap_or_else(|| out_sig.4.clone());

            let needs_noop_quant_boundary = then_sig == out_sig && output_dtype != out_sig.4;
            if then_sig == out_sig && !needs_noop_quant_boundary {
                continue;
            }

            if !out_sig.4.contains("int8") && !out_sig.4.contains("int16") {
                continue;
            }

            let branch_output_qparams = PyDict::new_bound(py);
            branch_output_qparams.set_item(0_usize, &then_qp)?;

            if needs_noop_quant_boundary {
                let output_dtype = quant.dtype_for_str(&out_sig.4);
                rescale_ops
                    .helpers
                    .call_method1("restamp_node_output_meta", (node, node, output_dtype))?;
                meta.set_item("output_qparams", &branch_output_qparams)?;
                modified = true;
                continue;
            }

            let users = FxGraphModule::node_users(node)?;
            let rescale_out = rescale_ops.insert(
                node,
                node,
                quant.dtype_for_str(&out_sig.4),
                vec![then_sig.0 / out_sig.0],
                then_sig.1,
                out_sig.1,
                true,
            )?;
            rescale_ops.restamp_output_meta(
                py,
                node,
                &rescale_out,
                node,
                quant.dtype_for_str(&out_sig.4),
            )?;
            rescale_ops.rewire_users(node, &rescale_out, &users)?;

            meta.set_item("output_qparams", &branch_output_qparams)?;
            rescale_ops.refresh_fake_tensor(node)?;
            modified = true;
        }

        // Phase 3: insert accumulator rescales after lowered TOSA conv ops.
        let all_nodes = gm.nodes()?;
        let debug = FxGraphModule::debug();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_tosa_accumulator_op(&target) {
                continue;
            }
            rescale_ops.refresh_fake_tensor(node)?;
            let node_name = if debug {
                Some(FxGraphModule::node_name(node)?)
            } else {
                None
            };

            let meta = node.getattr("meta")?;
            let input_qparams = match meta_non_empty_item(&meta, "input_qparams")? {
                Some(v) => v,
                _ => {
                    debug_skip(
                        node_name.as_deref(),
                        format_args!("missing input_qparams (target={})", target),
                    );
                    continue;
                }
            };
            if has_downstream_rescale(node)? {
                debug_skip(node_name.as_deref(), "downstream rescale already present");
                continue;
            }
            if debug {
                eprintln!(
                    "[InsertRescale] inspect '{}': target={}, input_qparams={}",
                    node_name.as_deref().unwrap_or("<unknown>"),
                    target,
                    py_len(&input_qparams)
                );
            }

            let input_scale = match input_qparams.get_item(0_usize) {
                Ok(qp) => qp_get_f64(&qp, "scale", None, 1.0),
                Err(_) => {
                    debug_skip(node_name.as_deref(), "missing input qparam[0]");
                    continue;
                }
            };
            let weight_scales: Vec<f64> = match input_qparams.get_item(1_usize) {
                Ok(qp) => qp_get_f64_list(&qp, "scale", 1.0),
                Err(_) => {
                    debug_skip(node_name.as_deref(), "missing weight qparam[1]");
                    continue;
                }
            };

            let output_qparams = if let Some(qparams) = extract_output_qparams(&meta)? {
                qparams
            } else if let Some(qparams) = recover_downstream_output_qparams(node)? {
                if let Some(name) = &node_name {
                    eprintln!(
                        "[InsertRescale] recovered output_qparams for '{}' from downstream user",
                        name
                    );
                }
                qparams
            } else {
                debug_skip(
                    node_name.as_deref(),
                    "no output_qparams on node or downstream layout chain",
                );
                continue;
            };
            let Some(oqp) = first_qparam(&output_qparams) else {
                debug_skip(node_name.as_deref(), "missing output qparam[0]");
                continue;
            };
            let (output_scale, output_zp, output_dtype_str) =
                qparam_scale_zp_dtype(&oqp, 0, "torch.int8");
            let output_dtype = quant.dtype_for_str(&output_dtype_str);

            if output_scale == 0.0 {
                continue;
            }

            let rescale_scales: Vec<f64> = weight_scales
                .iter()
                .map(|&ws| (input_scale * ws) / output_scale)
                .collect();

            if debug {
                let name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[InsertRescale] accumulator rescale after '{}': in_s={}, w_s={:?}, out_s={}, num_scales={}",
                    name, input_scale, &weight_scales[..weight_scales.len().min(4)], output_scale, rescale_scales.len(),
                );
            }

            let users = FxGraphModule::node_users(node)?;
            let rescale_node = rescale_ops.insert(
                node,
                node,
                output_dtype,
                rescale_scales,
                0_i64,
                output_zp,
                true,
            )?;

            rescale_ops.restamp_output_meta(py, node, &rescale_node, node, output_dtype)?;
            rescale_ops.rewire_users(node, &rescale_node, &users)?;

            modified = true;
        }

        // Phase 4: loop-carried output rescaling.
        // For while_loop bodies, ensure each output arg has the same qparams
        // as its corresponding placeholder (output[i] feeds back as input[i]).
        // The placeholder qparams are the source of truth — no inference needed.
        if self.output_qparam_source == OutputQParamSource::Carried {
            let placeholders: Vec<Bound<'_, PyAny>> = gm
                .nodes()?
                .into_iter()
                .filter(|n| {
                    FxGraphModule::node_op(n)
                        .map(|s| s == "placeholder")
                        .unwrap_or(false)
                })
                .collect();

            let output_node = gm.nodes()?.into_iter().find(|n| {
                FxGraphModule::node_op(n)
                    .map(|s| s == "output")
                    .unwrap_or(false)
            });

            if let Some(output) = output_node {
                let output_args = FxGraphModule::node_args(&output)?;
                let output_vals: Vec<Bound<'_, PyAny>> = if output_args.len() == 1 {
                    if let Ok(iter) = output_args[0].iter() {
                        iter.filter_map(|v| v.ok()).collect()
                    } else {
                        vec![output_args[0].clone()]
                    }
                } else {
                    output_args
                };

                for (idx, (out_val, ph)) in output_vals.iter().zip(placeholders.iter()).enumerate()
                {
                    if !FxGraphModule::is_node(out_val) {
                        continue;
                    }

                    let ph_meta = ph.getattr("meta")?;
                    let ph_qp = if let Ok(oqp) = ph_meta.get_item("output_qparams") {
                        first_qparam(&oqp)
                    } else if let Ok(iqp) = ph_meta.get_item("input_qparams") {
                        first_qparam(&iqp)
                    } else {
                        None
                    };
                    let ph_qp = match ph_qp {
                        Some(qp) => qp,
                        None => continue,
                    };
                    let ph_scale = qp_get_f64(&ph_qp, "scale", None, 1.0);
                    let ph_zp: i64 = qp_get_i64(&ph_qp, "zp", None, 0);
                    let ph_dtype_str = ph_qp
                        .getattr("dtype")
                        .map(|d| format!("{}", d))
                        .unwrap_or("torch.int8".into());
                    let ph_dtype = quant.dtype_for_str(&ph_dtype_str);

                    if ph_scale == 0.0 {
                        continue;
                    }

                    // Get output arg's scale from its fake tensor or from the
                    // RESCALE node that produced it. Walk the producer to find scale.
                    let out_scale;
                    let out_zp;
                    let out_target = FxGraphModule::node_target_str(out_val).unwrap_or_default();

                    if out_target.contains("RESCALE") || out_target.contains("rescale") {
                        // Output arg is already a RESCALE — read its output zp from args[4]
                        let r_args = FxGraphModule::node_args(out_val)?;
                        if r_args.len() >= 5 {
                            let mult: f64 = r_args[2]
                                .get_item(0_usize)
                                .ok()
                                .and_then(|v| v.extract().ok())
                                .unwrap_or(1.0);
                            // Source of the rescale
                            let source_meta = if FxGraphModule::is_node(&r_args[0]) {
                                r_args[0].getattr("meta").ok()
                            } else {
                                None
                            };
                            let source_oqp = source_meta
                                .as_ref()
                                .and_then(|m| m.get_item("output_qparams").ok())
                                .and_then(|oqp| first_qparam(&oqp));
                            let source_scale = source_oqp
                                .as_ref()
                                .map(|q| qp_get_f64(q, "scale", None, 1.0))
                                .unwrap_or(1.0);
                            out_scale = if mult != 0.0 {
                                source_scale / mult
                            } else {
                                1.0
                            };
                            out_zp = r_args[4].extract::<i64>().unwrap_or(0);
                        } else {
                            out_scale = 1.0;
                            out_zp = 0;
                        }
                    } else {
                        // Not a RESCALE — try output_qparams metadata
                        let out_meta = out_val.getattr("meta")?;
                        let oqp = out_meta
                            .get_item("output_qparams")
                            .ok()
                            .and_then(|oqp| first_qparam(&oqp));
                        out_scale = oqp
                            .as_ref()
                            .map(|q| qp_get_f64(q, "scale", None, 1.0))
                            .unwrap_or(ph_scale); // default to ph_scale = no-op
                        out_zp = oqp
                            .as_ref()
                            .map(|q| qp_get_i64(q, "zp", None, 0))
                            .unwrap_or(ph_zp);
                    }

                    // Skip if already matching
                    if (out_scale - ph_scale).abs() < 1e-10 && out_zp == ph_zp {
                        continue;
                    }

                    if out_scale == 0.0 {
                        continue;
                    }

                    let rescale_ratio = out_scale / ph_scale;
                    let rescale = rescale_ops.insert(
                        out_val,
                        out_val,
                        ph_dtype,
                        vec![rescale_ratio],
                        out_zp,
                        ph_zp,
                        true,
                    )?;
                    rescale_ops.restamp_output_meta(py, out_val, &rescale, out_val, ph_dtype)?;

                    if output_vals.len() == 1 {
                        output.call_method("update_arg", (0i64, &rescale), None)?;
                    } else {
                        output.call_method("update_arg", (idx as i64, &rescale), None)?;
                    }

                    log_line(format_args!(
                        "[InsertRescale] carried output rescale at idx {}: scale {:.6} -> {:.6}",
                        idx, out_scale, ph_scale
                    ));
                    modified = true;
                }
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
