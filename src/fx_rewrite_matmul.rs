use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::fx_graph::{meta_val, py_attr, FxGraphModule};
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
        let debug = FxGraphModule::debug();
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
            let Some(input_qparams) = meta_val(&meta, "input_qparams") else {
                continue;
            };
            let Some(output_qparams) = meta_val(&meta, "output_qparams") else {
                continue;
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

            let scale_a: f64 = py_attr(&qp_a, &["scale"], 1.0);
            let scale_b: f64 = py_attr(&qp_b, &["scale"], 1.0);
            let scale_out: f64 = py_attr(&out_qp, &["scale"], 1.0);
            let out_zp: i64 = py_attr(&out_qp, &["zp", "zero_point"], 0);
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

            // Always insert a NEW RESCALE (INT32→INT8) after the matmul and
            // rewire ALL users through it.  Any existing RESCALE inserted by
            // InsertRescalePass (INT8→INT32 for sub/add) becomes a consumer of
            // the new one, giving the correct chain:
            //   matmul(INT32) → new_RESCALE(INT32→INT8) → old_RESCALE(INT8→INT32) → sub
            let helpers = py.import_bound("fx_helpers")?;
            let scale_list = PyList::new_bound(py, [rescale_scale]);
            let builtins = py.import_bound("builtins")?;
            let input_zp = builtins.call_method1("int", (0i64,))?;
            let output_zp = builtins.call_method1("int", (out_zp,))?;

            let rescale_node = helpers.call_method1(
                "insert_rescale_node",
                (
                    gm.gm_ref(),
                    node,        // ref_node (insert after)
                    node,        // input_node
                    &out_dtype,  // out_dtype
                    &scale_list, // scales
                    &input_zp,   // input_zp = 0 (accumulator)
                    &output_zp,  // output_zp
                    true,        // after=True
                ),
            )?;

            // Propagate the matmul's output_qparams to the RESCALE node so
            // downstream passes (InsertRescalePass) see the correct scales.
            let rescale_meta = rescale_node.getattr("meta")?;
            if let Ok(oqp) = output_qparams.call_method0("copy") {
                rescale_meta.set_item("output_qparams", &oqp)?;
            } else {
                rescale_meta.set_item("output_qparams", &output_qparams)?;
            }

            // Rewire: all users of matmul (except the new rescale) use the rescale
            FxGraphModule::replace_all_uses_with(node, &rescale_node)?;
            // Fix: replace_all_uses also changed the rescale's own input.
            // Restore args[0] to point back to the matmul node.
            let rescale_args = FxGraphModule::node_args(&rescale_node)?;
            if !rescale_args.is_empty() {
                let mut arg_refs: Vec<&pyo3::Bound<'_, pyo3::PyAny>> = Vec::new();
                arg_refs.push(node); // args[0] = matmul
                for a in &rescale_args[1..] {
                    arg_refs.push(a);
                }
                let new_rescale_args = pyo3::types::PyTuple::new_bound(py, &arg_refs);
                rescale_node.setattr("args", new_rescale_args)?;
            }

            if debug {
                let name = FxGraphModule::node_name(node)?;
                let rescale_name = FxGraphModule::node_name(&rescale_node)?;
                eprintln!(
                    "[RewriteMatmulRescale] inserted rescale '{}' after '{}' scale={} zp={}",
                    rescale_name, name, rescale_scale, out_zp
                );
            }
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
