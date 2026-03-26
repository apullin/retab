use std::collections::HashSet;

use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

// ---------------------------------------------------------------------------
// Dual-format qparam field access (QuantArgs namedtuple or legacy dict)
// ---------------------------------------------------------------------------

/// Read a float field from either a QuantArgs namedtuple or a plain dict.
fn qp_get_f64(qp: &Bound<'_, PyAny>, attr: &str, dict_key: Option<&str>, default: f64) -> f64 {
    if let Ok(val) = qp.getattr(attr) {
        if let Ok(v) = val.extract::<f64>() {
            return v;
        }
        if let Ok(item) = val.get_item(0_usize) {
            if let Ok(v) = item.extract::<f64>() {
                return v;
            }
        }
    }
    if let Some(key) = dict_key {
        if let Ok(val) = qp.call_method1("get", (key,)) {
            if let Ok(v) = val.extract::<f64>() {
                return v;
            }
        }
    }
    if let Ok(val) = qp.call_method1("get", (attr,)) {
        if let Ok(v) = val.extract::<f64>() {
            return v;
        }
    }
    default
}

/// Read an int field from either a QuantArgs namedtuple or a plain dict.
fn qp_get_i64(qp: &Bound<'_, PyAny>, attr: &str, dict_key: Option<&str>, default: i64) -> i64 {
    if let Ok(val) = qp.getattr(attr) {
        if let Ok(v) = val.extract::<i64>() {
            return v;
        }
        if let Ok(item) = val.get_item(0_usize) {
            if let Ok(v) = item.extract::<i64>() {
                return v;
            }
        }
    }
    if let Some(key) = dict_key {
        if let Ok(val) = qp.call_method1("get", (key,)) {
            if let Ok(v) = val.extract::<i64>() {
                return v;
            }
        }
    }
    if let Ok(val) = qp.call_method1("get", (attr,)) {
        if let Ok(v) = val.extract::<i64>() {
            return v;
        }
    }
    default
}

/// Read a float list from a QuantArgs namedtuple or plain dict.
///
/// Returns all scale values for per-channel quant, or a single-element list
/// for per-tensor quant. Falls back to `[default]` if extraction fails.
fn qp_get_f64_list(qp: &Bound<'_, PyAny>, attr: &str, default: f64) -> Vec<f64> {
    if let Ok(val) = qp.getattr(attr) {
        // Try extracting as a list first (per-channel)
        if let Ok(list) = val.extract::<Vec<f64>>() {
            if !list.is_empty() {
                return list;
            }
        }
        // Try as a single float (per-tensor)
        if let Ok(v) = val.extract::<f64>() {
            return vec![v];
        }
        // Try indexing (could be a tuple or other sequence)
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

/// Read dtype string from either a QuantArgs namedtuple or a plain dict.
fn qp_get_dtype_str(qp: &Bound<'_, PyAny>, default: &str) -> String {
    if let Ok(val) = qp.getattr("dtype") {
        if let Ok(s) = val.str() {
            return s.to_string();
        }
    }
    if let Ok(val) = qp.call_method1("get", ("dtype",)) {
        if let Ok(s) = val.extract::<String>() {
            return s;
        }
        if let Ok(s) = val.str() {
            return s.to_string();
        }
    }
    default.to_string()
}

/// FX-graph-direct implementation matching Python's InsertRescalePass +
/// InsertRescaleInt32Pass exactly.
///
/// Phase 1: Replace surviving DQ->Q pairs with TOSA RESCALE (InsertRescalePass)
/// Phase 2: Insert INT32 rescales around elementwise ops (InsertRescaleInt32Pass)
///
/// Phase 3: Insert accumulator rescales after tosa.CONV2D / tosa.DEPTHWISE_CONV2D.
///   After RewriteConvPass, conv nodes produce INT32 accumulators that must be
///   rescaled to INT8 using: scale = (input_scale * weight_scale) / output_scale.
pub struct FxInsertRescalePass;

/// Returns true for TOSA accumulator-producing ops that need INT32 -> INT8 rescale.
fn is_tosa_accumulator_op(target: &str) -> bool {
    let lowered = target.to_ascii_lowercase();
    lowered.contains("conv2d")
        || lowered.contains("depthwise_conv2d")
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

fn extract_output_qparams<'py>(
    py: Python<'py>,
    meta: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let output_qparams = match meta.call_method1("get", ("output_qparams",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(None),
    };
    let len: usize = py
        .import_bound("builtins")?
        .call_method1("len", (&output_qparams,))?
        .extract()
        .unwrap_or(0);
    if len == 0 {
        return Ok(None);
    }
    Ok(Some(output_qparams))
}

fn recover_downstream_output_qparams<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
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
        let user_meta = user.getattr("meta")?;
        if let Some(output_qparams) = extract_output_qparams(py, &user_meta)? {
            return Ok(Some(output_qparams));
        }

        let target = FxGraphModule::node_target_str(&user)?;
        if !is_layout_passthrough_user(&target) {
            return Ok(None);
        }
        current = user;
    }
}

fn has_downstream_rescale(start: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut current = start.clone();
    let mut visited: HashSet<String> = HashSet::new();

    loop {
        let name = FxGraphModule::node_name(&current)?;
        if !visited.insert(name) {
            return Ok(false);
        }

        let users = FxGraphModule::node_users(&current)?;
        if users.len() != 1 {
            return Ok(false);
        }
        let user = users[0].clone();
        let target = FxGraphModule::node_target_str(&user)?;
        let lowered = target.to_ascii_lowercase();
        if lowered.contains("rescale") {
            return Ok(true);
        }
        if !is_layout_passthrough_user(&target) {
            return Ok(false);
        }
        if (lowered.contains("view")
            || lowered.contains("reshape")
            || lowered.contains("flatten")
            || lowered.contains("squeeze")
            || lowered.contains("unsqueeze"))
            && FxGraphModule::node_shape(&current)
                .ok()
                .zip(FxGraphModule::node_shape(&user).ok())
                .map(|(before, after)| before.len() != after.len())
                .unwrap_or(false)
        {
            return Ok(false);
        }
        current = user;
    }
}

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor")
}

/// Extract (scale, zero_point) from a DQ/Q node's args.
fn extract_scale_zp(args: &[Bound<'_, PyAny>]) -> PyResult<(f64, i64)> {
    let scale: f64 = args.get(1).and_then(|a| a.extract().ok()).unwrap_or(1.0);
    let zp: i64 = args.get(2).and_then(|a| a.extract().ok()).unwrap_or(0);
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

/// Read input_qparams from node.meta, returning (keys, scales, zps, dtype_strs).
fn read_input_qparams(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
) -> PyResult<Option<(Vec<usize>, Vec<f64>, Vec<i64>, Vec<String>)>> {
    let meta = node.getattr("meta")?;
    let input_qp = match meta.call_method1("get", ("input_qparams",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(None),
    };
    let len: usize = py
        .import_bound("builtins")?
        .call_method1("len", (&input_qp,))?
        .extract()
        .unwrap_or(0);
    if len == 0 {
        return Ok(None);
    }

    let keys: Vec<usize> = input_qp
        .call_method0("keys")?
        .iter()?
        .filter_map(|k| k.ok().and_then(|k| k.extract().ok()))
        .collect();

    let mut scales = Vec::with_capacity(keys.len());
    let mut zps = Vec::with_capacity(keys.len());
    let mut dtype_strs = Vec::with_capacity(keys.len());

    for &idx in &keys {
        let qp = input_qp.get_item(idx)?;
        scales.push(qp_get_f64(&qp, "scale", None, 1.0));
        zps.push(qp_get_i64(&qp, "zp", Some("zero_point"), 0));
        dtype_strs.push(qp_get_dtype_str(&qp, "torch.int8"));
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

/// Compute the output rescale scale (INT32 domain) for the given op.
/// Returns None if no output rescale is needed (comparison ops).
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
    // abs, sum, add, sub: output scale == first input's rescaled scale
    Some(rescaled_input_scales[0])
}

impl FxPass for FxInsertRescalePass {
    fn name(&self) -> &str {
        "FxInsertRescale"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;

        // ------------------------------------------------------------------
        // Phase 1: Replace surviving DQ -> Q pairs with TOSA RESCALE
        // ------------------------------------------------------------------
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

            let q_dtype = q_args.get(5).cloned().unwrap_or_else(|| {
                py.eval_bound("__import__('torch').int8", None, None)
                    .unwrap()
            });

            // Insert RESCALE before the DQ node (matching Python: inserting_before(node))
            let rescale_node = helpers.call_method1(
                "insert_rescale_node",
                (
                    gm.gm_ref(),
                    dq_node,
                    &dq_input,
                    &q_dtype,
                    vec![new_scale],
                    dq_zp,
                    q_zp,
                    false, // before
                ),
            )?;

            // Copy Q node's meta onto the rescale (Python: rescale_node.meta = copy(user.meta))
            let q_meta = q_node.getattr("meta")?;
            let copy_mod = py.import_bound("copy")?;
            let meta_copy = copy_mod.call_method1("copy", (&q_meta,))?;
            rescale_node.setattr("meta", &meta_copy)?;
            helpers.call_method1(
                "restamp_rescale_output_meta",
                (&rescale_node, &dq_input, &q_dtype),
            )?;

            FxGraphModule::replace_all_uses_with(q_node, &rescale_node)?;
            gm.erase_node(q_node)?;

            let dq_users = FxGraphModule::node_users(dq_node)?;
            if dq_users.is_empty() {
                gm.erase_node(dq_node)?;
            }

            modified = true;
        }

        // ------------------------------------------------------------------
        // Phase 2: Insert INT32 rescales around element-wise ops
        // Matches Python InsertRescaleInt32Pass exactly.
        // ------------------------------------------------------------------
        let all_nodes = gm.nodes()?;
        let torch_int32 = py.eval_bound("__import__('torch').int32", None, None)?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            let op_kind = match classify_int32_op(&target) {
                Some(kind) => kind,
                None => continue,
            };

            // Read input_qparams
            let (keys, scales, zps, dtype_strs) = match read_input_qparams(py, node)? {
                Some(v) => v,
                None => continue,
            };

            // Check that at least one input is int8/int16
            let has_quantized_input = dtype_strs
                .iter()
                .any(|d| d.contains("int8") || d.contains("int16"));
            if !has_quantized_input {
                continue;
            }

            // Compute the rescaled INT32 input scales
            let rescaled_scales = compute_rescaled_input_scales(op_kind, &scales, &dtype_strs);

            // Compute the output INT32 scale
            let output_int32_scale = compute_output_int32_scale(&target, &rescaled_scales);

            // --- Insert input rescales ---
            let args = FxGraphModule::node_args(node)?;
            let mut new_args: Vec<Bound<'_, PyAny>> = args.clone();
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

                // Python's seen_args dedup: skip if we already inserted a rescale
                // for this exact arg node (e.g., add(x, x) only inserts one rescale).
                let arg_name = FxGraphModule::node_name(arg_node)?;
                if seen_arg_names.contains(&arg_name) {
                    continue;
                }
                seen_arg_names.insert(arg_name);

                let old_scale = scales[key_idx];
                let new_scale = rescaled_scales[key_idx];
                let rescale_scale = old_scale / new_scale;
                let old_zp = zps[key_idx];
                let new_zp = 0_i64; // INT32 always has zp=0

                let rescale = helpers.call_method1(
                    "insert_rescale_node",
                    (
                        gm.gm_ref(),
                        arg_node, // ref_node: insert after the arg
                        arg_node, // input_node: the arg itself
                        &torch_int32,
                        vec![rescale_scale],
                        old_zp,
                        new_zp,
                        true, // after
                    ),
                )?;

                // Python uses node.replace_input_with(arg_node, rescale_node)
                node.call_method1("replace_input_with", (arg_node, &rescale))?;

                // Also update new_args for tracking (though replace_input_with
                // already handles the graph edge)
                new_args[pos] = rescale;
                rescaled_inputs = true;
            }

            if !rescaled_inputs {
                continue;
            }

            // Inputs were rescaled, so this node was modified
            modified = true;

            // Update input_qparams metadata to reflect rescaled INT32 params
            let meta = node.getattr("meta")?;
            let quantargs_cls = py
                .import_bound("executorch.backends.arm._passes.quant_args")?
                .getattr("QuantArgs")?;
            let torch_mod = py.import_bound("torch")?;
            let torch_int32_dtype = torch_mod.getattr("int32")?;
            let i32_min: i64 = -2_147_483_648;
            let i32_max: i64 = 2_147_483_647;

            let new_input_qp = pyo3::types::PyDict::new_bound(py);
            for (key_idx, &pos) in keys.iter().enumerate() {
                let new_qa = quantargs_cls.call1((
                    rescaled_scales[key_idx],
                    0_i64,
                    i32_min,
                    i32_max,
                    &torch_int32_dtype,
                ))?;
                new_input_qp.set_item(pos, &new_qa)?;
            }
            meta.set_item("input_qparams", &new_input_qp)?;

            // --- Insert output rescale (only if op produces quantized output) ---
            let mut outputs_was_rescaled = false;
            if let Some(out_int32_scale) = output_int32_scale {
                let output_qp = match meta.call_method1("get", ("output_qparams",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => {
                        // No output_qparams -- skip output rescale but input
                        // rescales are still valid.
                        continue;
                    }
                };
                let oqp_len: usize = py
                    .import_bound("builtins")?
                    .call_method1("len", (&output_qp,))?
                    .extract()
                    .unwrap_or(0);
                if oqp_len == 0 {
                    continue;
                }

                let oqp = output_qp.get_item(0)?;
                let out_orig_scale = qp_get_f64(&oqp, "scale", None, 1.0);
                let out_orig_zp = qp_get_i64(&oqp, "zp", Some("zero_point"), 0);
                let out_orig_dtype_str = qp_get_dtype_str(&oqp, "torch.int8");

                if out_orig_dtype_str.contains("int8") || out_orig_dtype_str.contains("int16") {
                    // Output rescale: INT32 -> original dtype
                    // Scale = computed_int32_scale / original_output_scale
                    let rescale_scale = out_int32_scale / out_orig_scale;
                    let out_dtype = if out_orig_dtype_str.contains("int16") {
                        py.eval_bound("__import__('torch').int16", None, None)?
                    } else {
                        py.eval_bound("__import__('torch').int8", None, None)?
                    };

                    let users = FxGraphModule::node_users(node)?;

                    let rescale_out = helpers.call_method1(
                        "insert_rescale_node",
                        (
                            gm.gm_ref(),
                            node,
                            node,
                            &out_dtype,
                            vec![rescale_scale],
                            0_i64,       // in_zp: 0 (INT32 accumulator)
                            out_orig_zp, // out_zp: original output zp
                            true,        // after
                        ),
                    )?;

                    for user in &users {
                        if !user.is(&rescale_out) {
                            user.call_method1("replace_input_with", (node, &rescale_out))?;
                        }
                    }

                    outputs_was_rescaled = true;
                }
            }

            // Update output_qparams metadata if output was rescaled
            if outputs_was_rescaled {
                if let Some(out_int32_scale) = output_int32_scale {
                    let new_output_qp = pyo3::types::PyDict::new_bound(py);
                    let out_qa = quantargs_cls.call1((
                        out_int32_scale,
                        0_i64,
                        i32_min,
                        i32_max,
                        &torch_int32_dtype,
                    ))?;
                    new_output_qp.set_item(0_usize, &out_qa)?;
                    meta.set_item("output_qparams", &new_output_qp)?;

                    // If "dtype" is in node.kwargs, change it to torch.int32
                    let kwargs = FxGraphModule::node_kwargs(node)?;
                    if kwargs.contains("dtype").unwrap_or(false) {
                        let new_kwargs = kwargs.downcast::<pyo3::types::PyDict>().ok();
                        if let Some(kw) = new_kwargs {
                            kw.set_item("dtype", &torch_int32)?;
                            node.setattr("kwargs", kw)?;
                        }
                    }
                }
            }

            // Python retraces the modified graph after InsertRescaleInt32Pass.
            // Refresh the touched node's FakeTensor eagerly so downstream
            // passes and serialization observe the INT32 accumulator dtype.
            helpers.call_method1("refresh_node_fake_tensor", (node,))?;
        }

        // ------------------------------------------------------------------
        // Phase 3: Insert accumulator rescales after TOSA CONV2D / DEPTHWISE_CONV2D
        // These ops produce INT32 accumulators; rescale to INT8 using
        // scale = (input_scale * weight_scale) / output_scale.
        // ------------------------------------------------------------------
        let all_nodes = gm.nodes()?;
        let torch_int8 = py.eval_bound("__import__('torch').int8", None, None)?;
        let torch_int16 = py.eval_bound("__import__('torch').int16", None, None)?;
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_tosa_accumulator_op(&target) {
                continue;
            }
            let node_name = if debug {
                Some(FxGraphModule::node_name(node)?)
            } else {
                None
            };

            // Read input_qparams to get input_scale (pos 0) and weight_scales (pos 1)
            let meta = node.getattr("meta")?;
            let input_qparams = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => v,
                _ => {
                    if let Some(name) = &node_name {
                        eprintln!(
                            "[InsertRescale] skip '{}': missing input_qparams (target={})",
                            name, target
                        );
                    }
                    continue;
                }
            };
            if has_downstream_rescale(node)? {
                if let Some(name) = &node_name {
                    eprintln!(
                        "[InsertRescale] skip '{}': downstream rescale already present",
                        name
                    );
                }
                continue;
            }
            if debug {
                let input_qp_len: usize = py
                    .import_bound("builtins")?
                    .call_method1("len", (&input_qparams,))?
                    .extract()
                    .unwrap_or(0);
                eprintln!(
                    "[InsertRescale] inspect '{}': target={}, input_qparams={}",
                    node_name.as_deref().unwrap_or("<unknown>"),
                    target,
                    input_qp_len
                );
            }

            let input_scale = match input_qparams.get_item(0_usize) {
                Ok(qp) => qp_get_f64(&qp, "scale", None, 1.0),
                Err(_) => {
                    if let Some(name) = &node_name {
                        eprintln!(
                            "[InsertRescale] skip '{}': missing input qparam[0]",
                            name
                        );
                    }
                    continue;
                }
            };
            // Read per-channel weight scales (list for per-channel, single for per-tensor)
            let weight_scales: Vec<f64> = match input_qparams.get_item(1_usize) {
                Ok(qp) => qp_get_f64_list(&qp, "scale", 1.0),
                Err(_) => {
                    if let Some(name) = &node_name {
                        eprintln!(
                            "[InsertRescale] skip '{}': missing weight qparam[1]",
                            name
                        );
                    }
                    continue;
                }
            };

            // Read output_qparams to get output_scale(s) and output_zp
            let output_qparams = if let Some(qparams) = extract_output_qparams(py, &meta)? {
                qparams
            } else if let Some(qparams) = recover_downstream_output_qparams(py, node)? {
                if let Some(name) = &node_name {
                    eprintln!(
                        "[InsertRescale] recovered output_qparams for '{}' from downstream user",
                        name
                    );
                }
                qparams
            } else {
                if let Some(name) = &node_name {
                    eprintln!(
                        "[InsertRescale] skip '{}': no output_qparams on node or downstream layout chain",
                        name
                    );
                }
                continue;
            };
            let oqp = match output_qparams.get_item(0_usize) {
                Ok(qp) => qp,
                Err(_) => {
                    if let Some(name) = &node_name {
                        eprintln!(
                            "[InsertRescale] skip '{}': missing output qparam[0]",
                            name
                        );
                    }
                    continue;
                }
            };
            let output_scale = qp_get_f64(&oqp, "scale", None, 1.0);
            let output_zp = qp_get_i64(&oqp, "zp", Some("zero_point"), 0);
            let output_dtype_str = qp_get_dtype_str(&oqp, "torch.int8");
            let output_dtype = if output_dtype_str.contains("int16") {
                &torch_int16
            } else {
                &torch_int8
            };

            if output_scale == 0.0 {
                continue;
            }

            // Compute per-channel rescale scales:
            //   rescale_scale[i] = (input_scale * weight_scale[i]) / output_scale
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

            // Capture users before inserting the rescale
            let users = FxGraphModule::node_users(node)?;

            // Insert RESCALE after the conv node: INT32 -> quantized output dtype
            // input_zp = 0 (accumulator is zero-centered)
            let rescale_node = helpers.call_method1(
                "insert_rescale_node",
                (
                    gm.gm_ref(),
                    node, // ref_node
                    node, // input_node
                    output_dtype,
                    rescale_scales,
                    0_i64,     // input_zp (accumulator is zero-centered)
                    output_zp, // output_zp from output qparams
                    true,      // after
                ),
            )?;

            // Copy the conv node's meta onto the rescale node
            let conv_meta = node.getattr("meta")?;
            let copy_mod = py.import_bound("copy")?;
            let meta_copy = copy_mod.call_method1("copy", (&conv_meta,))?;
            rescale_node.setattr("meta", &meta_copy)?;
            helpers.call_method1(
                "restamp_rescale_output_meta",
                (&rescale_node, node, output_dtype),
            )?;

            // Rewire all users of the conv to use the rescale instead
            for user in &users {
                if !user.is(&rescale_node) {
                    user.call_method1("replace_input_with", (node, &rescale_node))?;
                }
            }

            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
