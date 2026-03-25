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

/// Check whether a tensor shape would be laid out differently in channels-first
/// (NCHW) versus channels-last (NHWC) memory format. Returns false when the
/// formats are equivalent (e.g. C==1 or all spatial dims are 1).
fn memory_format_differs(shape: &[i64], rank: usize) -> bool {
    if rank <= 2 {
        return false;
    }
    let spatial_rank = initial_spatial_rank(rank);
    if spatial_rank == 0 {
        return false;
    }
    let channel_idx = rank.saturating_sub(spatial_rank + 1);
    let channel_dim = shape.get(channel_idx).copied().unwrap_or(1);
    let spatial_dims = &shape[channel_idx + 1..];
    if spatial_dims.is_empty() {
        return false;
    }
    channel_dim > 1 && spatial_dims.iter().any(|&d| d > 1)
}

/// Check whether a reshape touches the channel or batch dimensions (4D+/5D+ -> 4D+/5D+).
/// Returns true when the channel dimension or consolidated batch dimension changes.
fn is_channel_reshape(input_shape: &[i64], output_shape: &[i64]) -> bool {
    let valid = [4, 5, 6];
    if !valid.contains(&input_shape.len()) || !valid.contains(&output_shape.len()) {
        return false;
    }
    let in_sr = initial_spatial_rank(input_shape.len());
    let out_sr = initial_spatial_rank(output_shape.len());

    let in_ch_idx = input_shape.len().saturating_sub(in_sr + 1);
    let out_ch_idx = output_shape.len().saturating_sub(out_sr + 1);

    let c_old = input_shape.get(in_ch_idx).copied().unwrap_or(1);
    let c_new = output_shape.get(out_ch_idx).copied().unwrap_or(1);

    let n_old: i64 = input_shape[..in_ch_idx].iter().product();
    let n_new: i64 = output_shape[..out_ch_idx].iter().product();

    n_old != n_new || c_old != c_new
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
    let is_arm_constant_placeholder: bool = meta
        .call_method1("get", ("arm_constant_placeholder", false))?
        .extract()
        .unwrap_or(false);
    if is_arm_constant_placeholder {
        return Ok(false);
    }
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

/// Check whether a rank-4 tensor is consumed as the weight input to a
/// TRANSPOSE_CONV2D op. Python annotates these with a dedicated HWIO-like
/// order instead of generic channels-last.
fn is_transpose_conv2d_weight(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let users = FxGraphModule::node_users(node)?;
    for user in &users {
        let op = FxGraphModule::node_op(user)?;
        if op != "call_function" {
            continue;
        }
        let target = FxGraphModule::node_target_str(user)?.to_ascii_lowercase();
        if !(target.contains("transpose_conv2d") || target.contains("transposeconv2d")) {
            continue;
        }

        let args = FxGraphModule::node_args(user)?;
        if let Some(arg) = args.get(1) {
            if arg.is(node) {
                return Ok(true);
            }
        }
    }
    Ok(false)
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

            let dim_order = if is_input_placeholder(node)? || op == "output" {
                // True external inputs/outputs keep their original format.
                (0..rank).collect::<Vec<_>>()
            } else if rank == 4 && is_transpose_conv2d_weight(node)? {
                vec![1, 2, 3, 0]
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
        //
        // IMPORTANT: Process view_copy nodes BEFORE inserting placeholder
        // transposes.  This matches the Python ordering in
        // insert_tosa_transposes() which visits views first (line 336)
        // then inputs (line 350).  If we insert placeholder transposes
        // first, the view_copy sees NHWC shapes and memory_format_differs
        // returns false for cases where C=1 in the NHWC layout.
        let all_nodes = gm.nodes()?;

        // 4a: Around view_copy nodes that cross 4D+/<4D boundaries, insert
        // transposes so the reshape sees the data in the correct memory order.
        //   - >=4D -> <4D (NHWC territory -> NCHW): insert NHWC->NCHW before the view
        //   - <4D -> >=4D (NCHW -> NHWC territory): insert NCHW->NHWC after the view
        let all_nodes = gm.nodes()?;
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            // Match both aten.view_copy.default and aten.view.default
            if !(target.contains("view_copy") || target.contains("aten.view.")) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let input_node = match args.first() {
                Some(n) if FxGraphModule::is_node(n) => n,
                _ => continue,
            };

            // Get input rank
            let input_meta = input_node.getattr("meta")?;
            let input_val = match input_meta.call_method1("get", ("val",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let input_shape = match input_val.getattr("shape") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let input_rank: usize = py
                .import_bound("builtins")?
                .call_method1("len", (&input_shape,))?
                .extract()
                .unwrap_or(0);

            // Get output rank
            let node_meta = node.getattr("meta")?;
            let output_val = match node_meta.call_method1("get", ("val",)) {
                Ok(v) if !v.is_none() => v,
                _ => continue,
            };
            let output_shape = match output_val.getattr("shape") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let output_rank: usize = py
                .import_bound("builtins")?
                .call_method1("len", (&output_shape,))?
                .extract()
                .unwrap_or(0);

            let input_shape_vec: Vec<i64> = input_shape.extract().unwrap_or_default();
            let output_shape_vec: Vec<i64> = output_shape.extract().unwrap_or_default();

            let nhwc_to_nchw = input_rank >= 4 && output_rank < 4;
            let nchw_to_nhwc = input_rank < 4 && output_rank >= 4;
            let channel_reshape = is_channel_reshape(&input_shape_vec, &output_shape_vec);

            // Input transpose: NHWC -> NCHW before the view when crossing from
            // NHWC territory to NCHW, or when a channel reshape requires NCHW order.
            if (channel_reshape || nhwc_to_nchw)
                && memory_format_differs(&input_shape_vec, input_rank)
            {
                let input_spatial_rank: usize = input_meta
                    .call_method1("get", ("tosa_spatial_rank",))?
                    .extract()
                    .unwrap_or(2);

                // NHWC -> NCHW: inverse of channels_last
                let perm = channels_last_inverse_order(input_rank, input_spatial_rank);
                let perm_py: Vec<i64> = perm.iter().map(|&p| p as i64).collect();

                if debug {
                    let name = FxGraphModule::node_name(node)?;
                    eprintln!(
                        "[MemoryFormat] inserting NHWC->NCHW transpose before view '{}' (input_rank={}, perm={:?})",
                        name, input_rank, perm_py
                    );
                }

                helpers.call_method1(
                    "insert_transpose_before",
                    (gm.gm_ref(), node, input_node, perm_py.clone()),
                )?;
                modified = true;
            }

            // Output transpose: NCHW -> NHWC after the view when crossing into
            // NHWC territory, or when a channel reshape needs to return to NHWC.
            if (channel_reshape || nchw_to_nhwc)
                && memory_format_differs(&output_shape_vec, output_rank)
            {
                let output_spatial_rank: usize = node_meta
                    .call_method1("get", ("tosa_spatial_rank",))?
                    .extract()
                    .unwrap_or(2);

                // NCHW -> NHWC: channels_last_order
                let perm = channels_last_order(output_rank, output_spatial_rank);
                let perm_py: Vec<i64> = perm.iter().map(|&p| p as i64).collect();

                if debug {
                    let name = FxGraphModule::node_name(node)?;
                    eprintln!(
                        "[MemoryFormat] inserting NCHW->NHWC transpose after view '{}' (output_rank={}, perm={:?})",
                        name, output_rank, perm_py
                    );
                }

                helpers.call_method1(
                    "insert_transpose_after",
                    (gm.gm_ref(), node, perm_py.clone()),
                )?;
                modified = true;
            }
        }

        // 4b: After 4D+ input placeholders, insert NCHW->NHWC transpose.
        // The input comes in NCHW format; internal ops expect NHWC.
        // Runs AFTER view transposes (4a) so views see the original NCHW shapes.
        let all_nodes = gm.nodes()?;
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

            if rank < 4 {
                continue;
            }

            let spatial_rank: usize = meta
                .call_method1("get", ("tosa_spatial_rank",))?
                .extract()
                .unwrap_or(2);

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

        // 4c: Before the output node, insert NHWC->NCHW transpose for 4D+ outputs.
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

        // ------------------------------------------------------------------
        // Phase 4d: Remap permute_copy perm args from NCHW to NHWC domain.
        //
        // After phases 4a-4c, rank-4 data flows in NHWC format between the
        // input transpose and the output transpose.  Any explicit user
        // `permute_copy` node was authored in NCHW order, so its perm vector
        // is wrong for the now-NHWC data.  We detect such nodes by checking
        // whether their input comes from a TOSA TRANSPOSE (the NCHW->NHWC
        // we inserted in 4a), and remap:
        //
        //   P_nhwc[j] = F_inv[P_nchw[F[j]]]
        //
        // where F = NCHW->NHWC perm and F_inv = NHWC->NCHW perm.
        // ------------------------------------------------------------------
        let all_nodes = gm.nodes()?;
        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            // Match aten.permute_copy.default and aten.permute.default
            if !(target.contains("permute_copy") || target.contains("aten.permute.")) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.len() < 2 {
                continue;
            }
            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            // Get the input rank from FakeTensor metadata
            let input_rank: usize = {
                let input_meta = input_node.getattr("meta")?;
                let input_val = match input_meta.call_method1("get", ("val",)) {
                    Ok(v) if !v.is_none() => v,
                    _ => continue,
                };
                let input_shape = match input_val.getattr("shape") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                py.import_bound("builtins")?
                    .call_method1("len", (&input_shape,))?
                    .extract()
                    .unwrap_or(0)
            };
            if input_rank < 4 {
                continue;
            }

            // Check if the input was produced by the NCHW->NHWC transpose
            // we inserted earlier (target contains "tosa" and "TRANSPOSE"
            // or "transpose"). A more robust check: the input node has
            // tosa_dim_order == channels_last_order.
            let input_target = FxGraphModule::node_target_str(input_node).unwrap_or_default();
            let input_is_nhwc = {
                let input_meta = input_node.getattr("meta")?;
                let dim_order_val = input_meta.call_method1("get", ("tosa_dim_order",))?;
                if dim_order_val.is_none() {
                    // Fall back: check if input is a TOSA TRANSPOSE node
                    let input_op = FxGraphModule::node_op(input_node)?;
                    input_op == "call_function"
                        && (input_target.contains("TRANSPOSE")
                            || input_target.contains("transpose"))
                        && input_target.contains("tosa")
                } else {
                    // tosa_dim_order is channels_last when we've already
                    // transposed: e.g. (0, 2, 3, 1) for rank 4.
                    let spatial_rank: usize = input_meta
                        .call_method1("get", ("tosa_spatial_rank",))?
                        .extract()
                        .unwrap_or(2);
                    let expected_cl = channels_last_order(input_rank, spatial_rank);
                    let actual: Vec<usize> =
                        dim_order_val.extract().unwrap_or_else(|_| (0..input_rank).collect());
                    actual == expected_cl
                }
            };

            if !input_is_nhwc {
                continue;
            }

            // Extract the user's NCHW perm from args[1]
            let user_perm: Vec<i64> = match args[1].extract::<Vec<i64>>() {
                Ok(p) => p,
                Err(_) => continue,
            };
            if user_perm.len() != input_rank {
                continue;
            }

            let spatial_rank = {
                let input_meta = input_node.getattr("meta")?;
                input_meta
                    .call_method1("get", ("tosa_spatial_rank",))?
                    .extract::<usize>()
                    .unwrap_or(2)
            };

            // F = NCHW -> NHWC (channels_last_order)
            let f = channels_last_order(input_rank, spatial_rank);
            // F_inv = NHWC -> NCHW (channels_last_inverse_order)
            let f_inv = channels_last_inverse_order(input_rank, spatial_rank);

            // P_nhwc[j] = F_inv[P_nchw[F[j]]]
            //
            // Derivation: the overall data flow is:
            //   input(NCHW) -> T_in=F -> data(NHWC) -> P_nhwc -> result(NHWC) -> T_out=F_inv -> output(NCHW)
            // We need T_in[P_nhwc[T_out[k]]] = P_user[k], where T_out = F_inv, T_in = F.
            // Solving: P_nhwc[j] = F_inv[P_user[F[j]]].
            let mut p_nhwc = vec![0i64; input_rank];
            for j in 0..input_rank {
                let nchw_axis = f[j];                       // map NHWC pos j -> NCHW axis
                let user_mapped = user_perm[nchw_axis] as usize; // user's perm in NCHW
                p_nhwc[j] = f_inv[user_mapped] as i64;      // map result NCHW axis -> NHWC pos
            }

            // Skip if the remapped perm is identical to the original —
            // no change needed, and refresh_node_fake_tensor would corrupt
            // the metadata for identity-like permutations.
            if p_nhwc == user_perm {
                if debug {
                    let name = FxGraphModule::node_name(node)?;
                    eprintln!(
                        "[MemoryFormat] skipping permute '{}' — perm {:?} unchanged after remap",
                        name, user_perm
                    );
                }
                continue;
            }

            if debug {
                let name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[MemoryFormat] remapping permute '{}' perm {:?} -> {:?} (NCHW->NHWC)",
                    name, user_perm, p_nhwc
                );
            }

            // Update the node's args[1] with the remapped perm
            let new_perm_list = pyo3::types::PyList::new_bound(py, p_nhwc.iter().copied());
            let new_args = pyo3::types::PyTuple::new_bound(
                py,
                &[input_node.as_ref(), new_perm_list.as_ref()],
            );
            node.setattr("args", new_args)?;

            // Do NOT call refresh_node_fake_tensor — the remapped perm changes
            // the physical layout but the logical NCHW shape in meta.val must
            // stay unchanged for downstream passes and serialization.

            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
