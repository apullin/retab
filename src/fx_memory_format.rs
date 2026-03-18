use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `ToTosaMemoryFormatPass`.
///
/// Annotates each node with `tosa_dim_order` metadata indicating the
/// channels-last permutation, and `tosa_spatial_rank`.
///
/// For the prototype this pass only does the annotation phase (dim_order and
/// spatial_rank metadata).  The transpose-insertion phase (NCHW <-> NHWC) is
/// complex and depends on FakeTensor shapes that are easiest to handle from
/// Python, so we defer actual TRANSPOSE node creation to a follow-up.
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

        // Note: We intentionally skip transpose insertion in this prototype.
        // The annotation phase is sufficient for downstream consumers that
        // read tosa_dim_order from node.meta.

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
