use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `ToTosaMemoryFormatPass`.
///
/// Annotates each node with `tosa_dim_order` and `tosa_spatial_rank` metadata,
/// then inserts TOSA TRANSPOSE nodes at NCHW/NHWC boundaries:
///   - After 4D+ input placeholders: NCHW->NHWC transpose (perm [0,2,3,1])
///   - Before the output node for 4D+ args: NHWC->NCHW transpose (perm [0,3,1,2])
pub struct FxToTosaMemoryFormatPass;

/// Compute the channels-last permutation for a tensor of given rank and spatial rank.
/// E.g. rank=4, spatial_rank=2 -> (0, 2, 3, 1) for NCHW -> NHWC.
fn channels_last_order(rank: usize, spatial_rank: usize) -> Vec<usize> {
    if rank <= 2 {
        return (0..rank).collect();
    }
    let spatial_rank = spatial_rank.min(rank.saturating_sub(2));
    let channel_axis = rank - (spatial_rank + 1);
    let mut order: Vec<usize> = (0..channel_axis).collect();
    order.extend(channel_axis + 1..rank);
    order.push(channel_axis);
    order
}

/// Compute the inverse of channels_last_order: maps NHWC back to NCHW.
/// E.g. rank=4, spatial_rank=2 -> (0, 3, 1, 2) for NHWC -> NCHW.
fn channels_last_inverse_order(rank: usize, spatial_rank: usize) -> Vec<usize> {
    let order = channels_last_order(rank, spatial_rank);
    let mut inverse = vec![0usize; rank];
    for (idx, &axis) in order.iter().enumerate() {
        inverse[axis] = idx;
    }
    inverse
}

/// Infer the initial spatial rank from tensor rank.
fn initial_spatial_rank(tensor_rank: usize) -> usize {
    if tensor_rank >= 4 {
        2
    } else if tensor_rank == 3 {
        1
    } else {
        0
    }
}

/// Check if a node is a non-parameter placeholder (i.e. a user input).
/// In ExportedProgram, parameter placeholders have names starting with "p_"
/// (parameters) or "b_" (buffers), while user inputs have names like "x",
/// "arg0_1", etc.
fn is_input_placeholder(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let op = FxGraphModule::node_op(node)?;
    if op != "placeholder" {
        return Ok(false);
    }
    let meta = node.getattr("meta")?;
    let val = match meta.call_method1("get", ("val",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(false),
    };
    // Skip SymInt values
    let type_name: String = val.get_type().getattr("__name__")?.extract()?;
    if type_name == "SymInt" {
        return Ok(false);
    }
    // Filter out parameter/buffer/constant placeholders by name convention.
    // In ExportedProgram: parameters start with "p_", buffers with "b_",
    // lifted tensor constants with "c_".
    let name = FxGraphModule::node_name(node)?;
    if name.starts_with("p_") || name.starts_with("b_") || name.starts_with("c_") {
        return Ok(false);
    }
    Ok(true)
}

impl FxPass for FxToTosaMemoryFormatPass {
    fn name(&self) -> &str {
        "FxToTosaMemoryFormat"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut modified = false;
        let all_nodes = gm.nodes()?;

        // Phase 1: Annotate tosa_spatial_rank on every node that has a val in meta.
        for node in &all_nodes {
            let meta = node.getattr("meta")?;

            // Skip nodes without a "val" tensor in meta
            let val = match meta.call_method1("get", ("val",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };

            // Skip SymInt values
            let val_type = val.get_type().getattr("__name__")?;
            let type_name: String = val_type.extract()?;
            if type_name == "SymInt" {
                continue;
            }

            // Get tensor rank
            let shape = match val.getattr("shape") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rank: usize = match py
                .import_bound("builtins")?
                .call_method1("len", (&shape,))
            {
                Ok(v) => v.extract().unwrap_or(0),
                Err(_) => continue,
            };

            let spatial_rank = initial_spatial_rank(rank);
            meta.set_item("tosa_spatial_rank", spatial_rank)?;
        }

        // Phase 2: Propagate spatial ranks backwards (from users to inputs).
        // This ensures earlier nodes learn about upcoming spatial requirements.
        let mut changed = true;
        while changed {
            changed = false;
            for node in all_nodes.iter().rev() {
                let meta = node.getattr("meta")?;

                let val = match meta.call_method1("get", ("val",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => continue,
                };
                let shape = match val.getattr("shape") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let rank: usize = match py
                    .import_bound("builtins")?
                    .call_method1("len", (&shape,))
                {
                    Ok(v) => v.extract().unwrap_or(0),
                    Err(_) => continue,
                };
                let limit = rank.saturating_sub(2);

                let current: usize = meta
                    .call_method1("get", ("tosa_spatial_rank",))?
                    .extract()
                    .unwrap_or(0);

                let users = FxGraphModule::node_users(node)?;
                let mut propagated = current;
                for user in &users {
                    let user_meta = user.getattr("meta")?;
                    let user_rank: usize = user_meta
                        .call_method1("get", ("tosa_spatial_rank",))?
                        .extract()
                        .unwrap_or(0);
                    propagated = propagated.max(user_rank.min(limit));
                }

                if propagated != current {
                    meta.set_item("tosa_spatial_rank", propagated)?;
                    changed = true;
                }
            }
        }

        // Phase 3: Annotate tosa_dim_order
        for node in &all_nodes {
            let meta = node.getattr("meta")?;

            let val = match meta.call_method1("get", ("val",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let shape = match val.getattr("shape") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rank: usize = match py
                .import_bound("builtins")?
                .call_method1("len", (&shape,))
            {
                Ok(v) => v.extract().unwrap_or(0),
                Err(_) => continue,
            };

            let spatial_rank: usize = meta
                .call_method1("get", ("tosa_spatial_rank",))?
                .extract()
                .unwrap_or(0);

            let op = FxGraphModule::node_op(node)?;

            let dim_order = if op == "placeholder" || op == "output" {
                // For placeholders/outputs, use identity order (original format)
                (0..rank).collect::<Vec<_>>()
            } else if rank >= 4 {
                channels_last_order(rank, spatial_rank)
            } else {
                (0..rank).collect()
            };

            let py_dim_order =
                pyo3::types::PyTuple::new_bound(py, dim_order.iter().map(|&d| d));
            meta.set_item("tosa_dim_order", py_dim_order)?;
            modified = true;
        }

        // ------------------------------------------------------------------
        // Phase 4: Insert TRANSPOSE nodes at NCHW/NHWC boundaries
        // ------------------------------------------------------------------
        let helpers = py.import_bound("fx_helpers")?;
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();

        // Re-snapshot nodes for transpose insertion.
        let all_nodes = gm.nodes()?;

        // 4a: After 4D+ input placeholders, insert NCHW->NHWC transpose.
        // The input comes in NCHW format; internal ops expect NHWC.
        // The permutation is channels_last_order: (0, 2, 3, 1).
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "placeholder" {
                continue;
            }
            if !is_input_placeholder(node)? {
                continue;
            }

            let meta = node.getattr("meta")?;
            let val = meta.call_method1("get", ("val",))?;
            let shape = match val.getattr("shape") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rank: usize = py
                .import_bound("builtins")?
                .call_method1("len", (&shape,))?
                .extract()
                .unwrap_or(0);

            // Only transpose 4D+ tensors (the ones that differ between NCHW/NHWC)
            if rank < 4 {
                continue;
            }

            let spatial_rank: usize = meta
                .call_method1("get", ("tosa_spatial_rank",))?
                .extract()
                .unwrap_or(2);

            // The dim_order for the placeholder is identity (NCHW).
            // We need channels_last_order to convert NCHW -> NHWC.
            let perm = channels_last_order(rank, spatial_rank);
            let perm_py: Vec<i64> = perm.iter().map(|&p| p as i64).collect();

            if debug {
                let name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[MemoryFormat] inserting NCHW->NHWC transpose after input '{}' (rank={}, perm={:?})",
                    name, rank, perm_py
                );
            }

            helpers.call_method1(
                "insert_transpose_after",
                (gm.gm_ref(), node, perm_py.clone()),
            )?;
            modified = true;
        }

        // 4b: Before the output node, insert NHWC->NCHW transpose for 4D+ outputs.
        // The output node's args are the tensors being returned; they're in NHWC
        // format internally but the caller expects NCHW.
        //
        // FX output node args pattern: args=((result_node1, result_node2, ...),)
        // So args[0] is a Python tuple of the actual output nodes.
        let all_nodes = gm.nodes()?;
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "output" {
                continue;
            }

            // output node's args[0] is typically a tuple of output nodes
            let args = FxGraphModule::node_args(node)?;

            // Collect output nodes: either direct nodes in args, or nodes
            // inside a tuple at args[0].
            let mut output_nodes: Vec<Bound<'_, PyAny>> = Vec::new();
            for arg in &args {
                if FxGraphModule::is_node(arg) {
                    output_nodes.push(arg.clone());
                } else {
                    // Try iterating as a tuple/list of nodes
                    if let Ok(iter) = arg.iter() {
                        for item in iter {
                            if let Ok(item) = item {
                                if FxGraphModule::is_node(&item) {
                                    output_nodes.push(item);
                                }
                            }
                        }
                    }
                }
            }

            for output_arg in &output_nodes {
                let arg_meta = output_arg.getattr("meta")?;
                let val = match arg_meta.call_method1("get", ("val",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => continue,
                };
                let shape = match val.getattr("shape") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let rank: usize = py
                    .import_bound("builtins")?
                    .call_method1("len", (&shape,))?
                    .extract()
                    .unwrap_or(0);

                if rank < 4 {
                    continue;
                }

                let spatial_rank: usize = arg_meta
                    .call_method1("get", ("tosa_spatial_rank",))?
                    .extract()
                    .unwrap_or(2);

                // Inverse of channels_last: converts NHWC -> NCHW
                let perm = channels_last_inverse_order(rank, spatial_rank);
                let perm_py: Vec<i64> = perm.iter().map(|&p| p as i64).collect();

                if debug {
                    let name = FxGraphModule::node_name(output_arg)?;
                    eprintln!(
                        "[MemoryFormat] inserting NHWC->NCHW transpose before output for '{}' (rank={}, perm={:?})",
                        name, rank, perm_py
                    );
                }

                helpers.call_method1(
                    "insert_transpose_before",
                    (gm.gm_ref(), node, output_arg, perm_py.clone()),
                )?;
                modified = true;
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
