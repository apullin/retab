use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `InsertRescalePass`.
///
/// Finds adjacent DQ -> Q pairs that survived fold_qdq (because they represent
/// a type-domain change, e.g. int8 -> int32 or vice versa) and replaces them
/// with a single `tosa.RESCALE` node.
///
/// Also handles the INT32-rescale insertion for element-wise ops (add, mul,
/// sub, etc.) that need INT32 accumulators via `InsertRescaleInt32Pass` logic.
pub struct FxInsertRescalePass;

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor")
}

/// Extract (scale, zero_point) from a DQ/Q node's args.
/// Args layout: [input, scale, zero_point, qmin, qmax, dtype]
fn extract_scale_zp(args: &[Bound<'_, PyAny>]) -> PyResult<(f64, i64)> {
    let scale: f64 = args
        .get(1)
        .and_then(|a| a.extract().ok())
        .unwrap_or(1.0);
    let zp: i64 = args
        .get(2)
        .and_then(|a| a.extract().ok())
        .unwrap_or(0);
    Ok((scale, zp))
}

/// Check if a target string matches one of the int32-accumulator ops.
fn is_int32_accumulator_op(target: &str) -> bool {
    target.contains("add.Tensor")
        || target.contains("sub.Tensor")
        || target.contains("mul.Tensor")
        || target.contains("sum.dim_IntList")
        || target.contains("abs.default")
        || target.contains("maximum.default")
        || target.contains("minimum.default")
        || target.contains("eq.Tensor")
        || target.contains("ge.Tensor")
        || target.contains("gt.Tensor")
        || target.contains("le.Tensor")
        || target.contains("lt.Tensor")
}

/// Check if a target is a conv/depthwise_conv/linear/matmul that produces INT32 accumulators.
fn is_accumulator_producer(target: &str) -> bool {
    target.contains("convolution.default")
        || target.contains("linear.default")
        || target.contains("addmm.default")
        || target.contains("mm.default")
        || target.contains("bmm.default")
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
        // After fold_qdq, any remaining DQ nodes whose users are Q nodes
        // represent a requantization that should become a RESCALE.
        let all_nodes = gm.nodes()?;

        // Collect (dq_node, q_node) pairs first to avoid mutation during iteration.
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
            let q_args = FxGraphModule::node_args(&q_node)?;

            let (dq_scale, dq_zp) = extract_scale_zp(&dq_args)?;
            let (q_scale, q_zp) = extract_scale_zp(&q_args)?;

            let new_scale = dq_scale / q_scale;

            // The input to the DQ node (the actual quantized tensor)
            let dq_input = match dq_args.first() {
                Some(n) if FxGraphModule::is_node(n) => n.clone(),
                _ => continue,
            };

            // Get output dtype from Q node args
            let q_dtype = q_args
                .get(5)
                .cloned()
                .unwrap_or_else(|| {
                    py.eval_bound("__import__('torch').int8", None, None)
                        .unwrap()
                });

            // Insert RESCALE before the DQ node
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
                    false, // before, not after
                ),
            )?;

            // Replace all uses of Q with the rescale node, then erase Q
            FxGraphModule::replace_all_uses_with(q_node, &rescale_node)?;
            gm.erase_node(q_node)?;

            // If the DQ node has no more users, erase it too
            let dq_users = FxGraphModule::node_users(dq_node)?;
            if dq_users.is_empty() {
                gm.erase_node(dq_node)?;
            }

            modified = true;
        }

        // ------------------------------------------------------------------
        // Phase 2: Insert INT32 rescales around element-wise ops
        // ------------------------------------------------------------------
        // Re-snapshot nodes since we may have mutated the graph.
        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_int32_accumulator_op(&target) {
                continue;
            }

            // Check if node has input_qparams metadata
            let meta = node.getattr("meta")?;
            let input_qp = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            // Must be a non-empty dict
            let len: usize = match py
                .import_bound("builtins")?
                .call_method1("len", (&input_qp,))
            {
                Ok(v) => v.extract().unwrap_or(0),
                Err(_) => continue,
            };
            if len == 0 {
                continue;
            }

            // For each quantized input arg, insert a RESCALE int8 -> int32
            let args = FxGraphModule::node_args(node)?;
            let torch_int32 = py.eval_bound("__import__('torch').int32", None, None)?;

            let keys: Vec<usize> = input_qp
                .call_method0("keys")?
                .iter()?
                .filter_map(|k| k.ok().and_then(|k| k.extract().ok()))
                .collect();

            let mut new_args: Vec<Bound<'_, PyAny>> = args.clone();
            let mut rescaled_inputs = false;

            for &idx in &keys {
                if idx >= args.len() {
                    continue;
                }
                let arg_node = &args[idx];
                if !FxGraphModule::is_node(arg_node) {
                    continue;
                }

                // Get qparams for this input position
                let qp = input_qp.get_item(idx)?;
                let scale: f64 = qp
                    .call_method1("get", ("scale",))?
                    .extract()
                    .unwrap_or(1.0);
                let zp: i64 = qp
                    .call_method1("get", ("zero_point",))?
                    .extract()
                    .unwrap_or(0);
                let dtype_str: String = qp
                    .call_method1("get", ("dtype",))?
                    .extract()
                    .unwrap_or_else(|_| "torch.int8".to_string());

                // Only rescale int8/int16 inputs
                if !dtype_str.contains("int8") && !dtype_str.contains("int16") {
                    continue;
                }

                // Insert rescale: int8 -> int32 with scale=1.0 (identity rescale
                // to upcast). The actual scale handling is done by the TOSA runtime.
                let rescale = helpers.call_method1(
                    "insert_rescale_node",
                    (
                        gm.gm_ref(),
                        arg_node,
                        arg_node,
                        &torch_int32,
                        vec![scale],
                        zp,
                        0_i64,
                        true, // after the arg node
                    ),
                )?;

                new_args[idx] = rescale;
                rescaled_inputs = true;
            }

            if rescaled_inputs {
                // Update the node's args via Python
                let args_tuple =
                    pyo3::types::PyTuple::new_bound(py, new_args.iter().map(|a| a.as_ref()));
                node.setattr("args", args_tuple)?;

                // Insert output rescale: int32 -> int8
                let output_qp = match meta.call_method1("get", ("output_qparams",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => continue,
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
                let out_scale: f64 = oqp
                    .call_method1("get", ("scale",))?
                    .extract()
                    .unwrap_or(1.0);
                let out_zp: i64 = oqp
                    .call_method1("get", ("zero_point",))?
                    .extract()
                    .unwrap_or(0);
                let out_dtype_str: String = oqp
                    .call_method1("get", ("dtype",))?
                    .extract()
                    .unwrap_or_else(|_| "torch.int8".to_string());

                let out_dtype = if out_dtype_str.contains("int16") {
                    py.eval_bound("__import__('torch').int16", None, None)?
                } else {
                    py.eval_bound("__import__('torch').int8", None, None)?
                };

                let rescale_out = helpers.call_method1(
                    "insert_rescale_node",
                    (
                        gm.gm_ref(),
                        node,
                        node,
                        &out_dtype,
                        vec![out_scale],
                        0_i64,
                        out_zp,
                        true, // after the node
                    ),
                )?;

                // Replace all users of node (except the rescale itself) with the rescale
                let users = FxGraphModule::node_users(node)?;
                for user in &users {
                    if !user.is(&rescale_out) {
                        user.call_method1("replace_input_with", (node, &rescale_out))?;
                    }
                }

                modified = true;
            }
        }

        // ------------------------------------------------------------------
        // Phase 3: Insert accumulator rescales after conv/linear
        // ------------------------------------------------------------------
        // After fold_qdq, conv2d/depthwise_conv2d/linear nodes have
        // output_qparams. They produce INT32 accumulators that need RESCALE
        // to INT8. The rescale scale = (input_scale * weight_scale) / output_scale.
        let all_nodes = gm.nodes()?;
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();

        // Collect candidates first to avoid mutation during iteration.
        let mut accum_nodes: Vec<Bound<'_, PyAny>> = Vec::new();
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_accumulator_producer(&target) {
                continue;
            }

            let node_name = FxGraphModule::node_name(node)?;

            // Must have output_qparams (set by fold_qdq from the downstream Q node)
            let meta = node.getattr("meta")?;
            let has_output_qp;
            let output_qp = match meta.call_method1("get", ("output_qparams",)) {
                Ok(v) if !v.is_none() => {
                    has_output_qp = true;
                    v
                }
                _ => {
                    has_output_qp = false;
                    if debug {
                        eprintln!(
                            "[AccumRescale] SKIP {} (target='{}'): no output_qparams",
                            node_name, target
                        );
                    }
                    // Try to infer output_qparams from the node's users:
                    // If any user has input_qparams for the position that references
                    // this node, use those qparams as our output_qparams.
                    let users = FxGraphModule::node_users(node)?;
                    let mut inferred = false;
                    for user in &users {
                        let user_meta = user.getattr("meta")?;
                        let user_iqp = match user_meta.call_method1("get", ("input_qparams",)) {
                            Ok(v) if !v.is_none() => v,
                            _ => continue,
                        };
                        // Find which position of user's args references our node
                        let user_args = FxGraphModule::node_args(user)?;
                        for (pos, arg) in user_args.iter().enumerate() {
                            if FxGraphModule::is_node(arg) && arg.is(node) {
                                if let Ok(qp) = user_iqp.get_item(pos) {
                                    if !qp.is_none() {
                                        // Use this as output_qparams
                                        let oqp_dict = PyDict::new_bound(py);
                                        oqp_dict.set_item(0, &qp)?;
                                        meta.set_item("output_qparams", &oqp_dict)?;
                                        inferred = true;
                                        if debug {
                                            eprintln!(
                                                "[AccumRescale] INFERRED output_qparams for {} from user '{}' pos {}",
                                                node_name, FxGraphModule::node_name(user)?, pos
                                            );
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        if inferred {
                            break;
                        }
                    }
                    if !inferred {
                        continue;
                    }
                    meta.call_method1("get", ("output_qparams",))?
                }
            };
            let oqp_len: usize = py
                .import_bound("builtins")?
                .call_method1("len", (&output_qp,))?
                .extract()
                .unwrap_or(0);
            if oqp_len == 0 {
                if debug {
                    eprintln!(
                        "[AccumRescale] SKIP {} (target='{}'): output_qparams empty",
                        node_name, target
                    );
                }
                continue;
            }

            if debug && has_output_qp {
                eprintln!(
                    "[AccumRescale] ACCEPT {} (target='{}'): has output_qparams (len={})",
                    node_name, target, oqp_len
                );
            }
            accum_nodes.push(node.clone());
        }

        for node in &accum_nodes {
            let meta = node.getattr("meta")?;
            let node_name = FxGraphModule::node_name(node)?;

            // Get input_qparams to compute the accumulator scale
            let input_qp = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => v,
                _ => {
                    if debug {
                        eprintln!(
                            "[AccumRescale] SKIP processing {}: no input_qparams",
                            node_name
                        );
                    }
                    continue;
                }
            };

            // Position 0 = activation input, position 1 = weight
            let input_scale: f64 = match input_qp.get_item(0_usize) {
                Ok(qp0) => qp0
                    .call_method1("get", ("scale",))?
                    .extract()
                    .unwrap_or(1.0),
                Err(_) => {
                    if debug {
                        eprintln!(
                            "[AccumRescale] SKIP processing {}: no input_qparams[0]",
                            node_name
                        );
                    }
                    continue;
                }
            };
            let weight_scale: f64 = match input_qp.get_item(1_usize) {
                Ok(qp1) => qp1
                    .call_method1("get", ("scale",))?
                    .extract()
                    .unwrap_or(1.0),
                Err(_) => {
                    if debug {
                        eprintln!(
                            "[AccumRescale] SKIP processing {}: no input_qparams[1]",
                            node_name
                        );
                    }
                    continue;
                }
            };

            // Output qparams from the downstream Q node
            let output_qp = meta.call_method1("get", ("output_qparams",))?;
            let oqp = output_qp.get_item(0)?;
            let out_scale: f64 = oqp
                .call_method1("get", ("scale",))?
                .extract()
                .unwrap_or(1.0);
            let out_zp: i64 = oqp
                .call_method1("get", ("zero_point",))?
                .extract()
                .unwrap_or(0);
            let out_dtype_str: String = oqp
                .call_method1("get", ("dtype",))?
                .extract()
                .unwrap_or_else(|_| "torch.int8".to_string());

            // Accumulator rescale: INT32 -> INT8
            // scale = (input_scale * weight_scale) / output_scale
            let rescale_scale = (input_scale * weight_scale) / out_scale;

            let out_dtype = if out_dtype_str.contains("int16") {
                py.eval_bound("__import__('torch').int16", None, None)?
            } else {
                py.eval_bound("__import__('torch').int8", None, None)?
            };

            if debug {
                eprintln!(
                    "[InsertRescale] accumulator rescale for {}: scale={:.6} (in={:.6} * wt={:.6} / out={:.6}), out_zp={}",
                    node_name, rescale_scale, input_scale, weight_scale, out_scale, out_zp
                );
            }

            let rescale_node = helpers.call_method1(
                "insert_rescale_node",
                (
                    gm.gm_ref(),
                    node,
                    node,
                    &out_dtype,
                    vec![rescale_scale],
                    0_i64,   // in_zp: 0 (accumulators are zero-centered)
                    out_zp,  // out_zp: from output_qparams
                    true,    // after the conv node
                ),
            )?;

            // Rewire: all users of conv (except the rescale) should use the rescale
            let users = FxGraphModule::node_users(node)?;
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
