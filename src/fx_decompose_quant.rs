use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `DecomposeQuantNodesPass`.
///
/// After fold_qdq + insert_rescale, there may be remaining Q/DQ nodes that
/// weren't folded (e.g., at graph boundaries or in non-annotated sub-graphs).
///
/// The pass decomposes these into primitive arithmetic:
///   - quantize:   `cast(clamp(add(round(mul(x, 1/scale)), zp), qmin, qmax), dtype)`
///   - dequantize: `mul(cast(sub(cast(x, int32), zp), fp_dtype), scale)`
///
/// Q/DQ nodes that still form DQ->Q pairs are skipped (those are handled by
/// InsertRescalePass).
pub struct FxDecomposeQuantNodesPass;

fn is_q(target: &str) -> bool {
    target.contains("quantize_per_tensor.default")
}

fn is_dq(target: &str) -> bool {
    target.contains("dequantize_per_tensor.default")
}

fn is_q_or_dq(target: &str) -> bool {
    is_q(target) || is_dq(target)
}

/// Decompose a single quantize or dequantize node into primitive arithmetic.
///
/// For quantize_per_tensor (`is_quantize = true`):
///   scaled      = mul(x, full(1/scale, fp_dtype))
///   rounded     = round(scaled)
///   shifted     = add(rounded, full(zp, fp_dtype))
///   clamped     = clamp(shifted, qmin, qmax)
///   output      = cast_dim_order(clamped, target_dtype)
///
/// For dequantize_per_tensor (`is_quantize = false`):
///   input_cast  = cast_dim_order(x, int32)
///   shifted     = sub(input_cast, full(zp, int32))
///   casted_fp   = cast_dim_order(shifted, fp_dtype)
///   output      = mul(casted_fp, full(scale, fp_dtype))
fn decompose_quant_node(
    py: Python<'_>,
    gm: &FxGraphModule<'_>,
    node: &Bound<'_, PyAny>,
    is_quantize: bool,
) -> PyResult<()> {
    let graph = gm.gm_ref().getattr("graph")?;
    let d = FxDecomposer::new(py, &graph, gm, node);

    // ---- extract args: (x, scale, zp, qmin, qmax, dtype) ----
    let args = FxGraphModule::node_args(node)?;
    let x = &args[0];
    let scale: f64 = args[1].extract()?;
    let zero_point: i64 = args[2].extract()?;
    let qmin: i64 = args[3].extract()?;
    let qmax: i64 = args[4].extract()?;
    let target_dtype = &args[5]; // Python dtype object for the cast

    // ---- derive shapes and dtypes ----
    let input_shape = FxGraphModule::node_shape(x)?;
    let input_rank = input_shape.len();
    let shape_ones: Vec<i64> = vec![1; input_rank];

    let torch = py.import_bound("torch")?;
    let int32_dtype = torch.getattr("int32")?;

    // fp_dtype = output dtype (for DQ) or input dtype (for Q)
    let fp_dtype = if is_quantize {
        x.getattr("meta")?.get_item("val")?.getattr("dtype")?
    } else {
        node.getattr("meta")?.get_item("val")?.getattr("dtype")?
    };

    let effective_scale = if is_quantize { 1.0 / scale } else { scale };

    // ---- build the decomposed subgraph ----
    let output = if is_quantize {
        // scale constant (fp)
        let scale_const = d.full(&shape_ones, effective_scale, &fp_dtype)?;
        // zero-point constant (fp, so add works without a cast)
        let zp_const = d.full_int(&shape_ones, zero_point, &fp_dtype)?;

        let scaled = d.mul(x, &scale_const)?;
        let rounded = d.round(&scaled)?;
        let shifted = d.add(&rounded, &zp_const)?;
        let clamped = d.clamp(&shifted, qmin as f64, qmax as f64)?;
        d.cast_dim_order(&clamped, target_dtype)?
    } else {
        // scale constant (fp)
        let scale_const = d.full(&shape_ones, effective_scale, &fp_dtype)?;
        // zero-point constant (int32, matching the cast of x)
        let zp_const = d.full_int(&shape_ones, zero_point, &int32_dtype)?;

        let input_casted = d.cast_dim_order(x, &int32_dtype)?;
        let shifted = d.sub(&input_casted, &zp_const)?;
        let casted_float = d.cast_dim_order(&shifted, &fp_dtype)?;
        d.mul(&casted_float, &scale_const)?
    };

    d.replace(&output)?;
    Ok(())
}

impl FxPass for FxDecomposeQuantNodesPass {
    fn name(&self) -> &str {
        "FxDecomposeQuantNodes"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_q_or_dq(&target) {
                continue;
            }

            // Skip DQ nodes whose users are all Q nodes (these are DQ->Q pairs
            // that should have been handled by InsertRescalePass).
            if is_dq(&target) {
                let users = FxGraphModule::node_users(node)?;
                let all_users_are_q = !users.is_empty()
                    && users.iter().all(|u| {
                        FxGraphModule::node_op(u)
                            .ok()
                            .map(|op| op == "call_function")
                            .unwrap_or(false)
                            && FxGraphModule::node_target_str(u)
                                .ok()
                                .map(|t| is_q(&t))
                                .unwrap_or(false)
                    });
                if all_users_are_q {
                    continue;
                }
            }

            // Skip Q nodes whose input is a DQ node (these are DQ->Q pairs).
            if is_q(&target) {
                let args = FxGraphModule::node_args(node)?;
                if let Some(input) = args.first() {
                    if FxGraphModule::is_node(input) {
                        let input_target =
                            FxGraphModule::node_target_str(input).unwrap_or_default();
                        if is_dq(&input_target) {
                            continue;
                        }
                    }
                }
            }

            candidates.push(node.clone());
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            let target = FxGraphModule::node_target_str(node)?;
            let is_quant = is_q(&target);
            decompose_quant_node(py, gm, node, is_quant)?;
            modified = true;
        }

        if modified {
            // Eliminate dead code (the original Q/DQ nodes are erased by replace)
            graph.call_method0("eliminate_dead_code")?;
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
