use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct pass that inserts a TOSA RESCALE after matmul ops
/// (aten.mm, aten.bmm) when the quantized matmul produces INT32 output
/// but the consumer expects INT8/INT16.
///
/// Mirrors Python `RewriteMatmulPass._insert_output_rescale`:
///   scale = (input_A_scale * input_B_scale) / output_scale
///   rescale(matmul_out, target_dtype, [scale], input_zp=0, output_zp)
///
/// Must run AFTER fold_qdq (needs qparams) and AFTER InsertRescalePass
/// (so we can replace the generic rescale with the correct one).
pub struct FxRewriteMatmulRescalePass;

fn is_matmul(target: &str) -> bool {
    target.contains("aten.mm.") || target.contains("aten.bmm.")
}

impl FxPass for FxRewriteMatmulRescalePass {
    fn name(&self) -> &str {
        "FxRewriteMatmulRescale"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_matmul(&target) {
                continue;
            }

            // Check if the matmul has qparams (quantized path)
            let meta = node.getattr("meta")?;
            let input_qparams = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let output_qparams = match meta.call_method1("get", ("output_qparams",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };

            // Get input_qparams[0] and input_qparams[1] scales
            let qp_a = match input_qparams.call_method1("get", (0,)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let qp_b = match input_qparams.call_method1("get", (1,)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let out_qp = match output_qparams.call_method1("get", (0,)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };

            let scale_a: f64 = qp_a.getattr("scale")?.extract().unwrap_or(1.0);
            let scale_b: f64 = qp_b.getattr("scale")?.extract().unwrap_or(1.0);
            let scale_out: f64 = out_qp.getattr("scale")?.extract().unwrap_or(1.0);
            let out_zp: i64 = out_qp
                .getattr("zp")
                .or_else(|_| out_qp.getattr("zero_point"))
                .and_then(|v| v.extract())
                .unwrap_or(0);
            let out_dtype = out_qp.getattr("dtype")?;

            // Accumulator scale = A_scale * B_scale
            // Rescale: scale = accumulator_scale / output_scale
            let rescale_scale = (scale_a * scale_b) / scale_out;

            if debug {
                let name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[RewriteMatmulRescale] {} scale_a={} scale_b={} scale_out={} rescale={}",
                    name, scale_a, scale_b, scale_out, rescale_scale
                );
            }

            // Find the next RESCALE node after this matmul and update its scale.
            // The InsertRescalePass already inserted a RESCALE, but with the wrong
            // scale (it used input[0].scale / output.scale instead of
            // input[0].scale * input[1].scale / output.scale).
            let users = FxGraphModule::node_users(node)?;
            for user in &users {
                let user_target = FxGraphModule::node_target_str(user)?;
                if !user_target.contains("RESCALE") && !user_target.contains("rescale") {
                    continue;
                }

                // Update ONLY the scale in the rescale node's args.
                // Keep the original dtype/zps intact to not break the type chain.
                // tosa.RESCALE(input, out_dtype, [scale], input_zp, output_zp)
                let rescale_args = FxGraphModule::node_args(user)?;
                if rescale_args.len() >= 5 {
                    let scale_list = PyList::new_bound(py, [rescale_scale]);

                    // Preserve original args[0] (input), args[1] (dtype),
                    // args[3] (input_zp), args[4] (output_zp) — only replace args[2] (scale)
                    let new_args = pyo3::types::PyTuple::new_bound(
                        py,
                        &[
                            rescale_args[0].as_ref(),  // original input
                            rescale_args[1].as_ref(),  // original output dtype
                            scale_list.as_ref(),       // NEW scale
                            rescale_args[3].as_ref(),  // original input_zp
                            rescale_args[4].as_ref(),  // original output_zp
                        ],
                    );
                    user.setattr("args", new_args)?;

                    if debug {
                        let user_name = FxGraphModule::node_name(user)?;
                        eprintln!(
                            "[RewriteMatmulRescale] updated rescale '{}' scale={}",
                            user_name, rescale_scale
                        );
                    }
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
