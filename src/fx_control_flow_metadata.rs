//! FX control-flow metadata pass.
//!
//! Rust port of the 7 simplest metadata-stamping functions from
//! `python/retab/_control_flow.py` and `python/retab/_cf_ctx.py`.
//! None of these perform graph surgery -- they only read/write `node.meta`.
//!
//! Also contains the cleanup pass (`FxControlFlowCleanupPass`) which ports
//! graph-surgery functions from `_cf_rewrites.py` and `_cf_ctx.py`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_graph::{meta_get, meta_val, py_attr, FxGraphModule};
use crate::fx_pass::FxPass;
use crate::runner_logging::log_line;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get all (name, submodule, cf_node) triples from the Python helper.
fn get_cf_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let arm_utils =
        py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
    let result = arm_utils.call_method1("get_cond_while_submodules_nested", (gm,))?;
    let mut out = Vec::new();
    for item in result.iter()? {
        let item = item?;
        let tup = item.downcast::<PyTuple>()?;
        let name: String = tup.get_item(0)?.extract()?;
        let submodule = tup.get_item(1)?;
        let cf_node = tup.get_item(2)?;
        out.push((name, submodule.clone(), cf_node.clone()));
    }
    Ok(out)
}

/// Get all nodes from a submodule's graph.
fn submodule_nodes<'py>(
    submodule: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let graph = submodule.getattr("graph")?;
    let nodes = graph.getattr("nodes")?;
    let mut out = Vec::new();
    for n in nodes.iter()? {
        out.push(n?);
    }
    Ok(out)
}

/// Get placeholder nodes from a submodule.
fn submodule_placeholders<'py>(
    submodule: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut out = Vec::new();
    for node in submodule_nodes(submodule)? {
        let op: String = node.getattr("op")?.extract()?;
        if op == "placeholder" {
            out.push(node);
        }
    }
    Ok(out)
}

/// Get the output node from a submodule.
fn submodule_output_node<'py>(
    submodule: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    for node in submodule_nodes(submodule)? {
        let op: String = node.getattr("op")?.extract()?;
        if op == "output" {
            return Ok(Some(node));
        }
    }
    Ok(None)
}

/// Recompile a submodule.
fn recompile(submodule: &Bound<'_, PyAny>) -> PyResult<()> {
    submodule.call_method0("recompile")?;
    Ok(())
}

/// `node.users` keys as a sorted list (matches _ordered_control_flow_users).
fn ordered_cf_users<'py>(node: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let users = node.getattr("users")?;
    let keys = users.call_method0("keys")?;
    let mut result = Vec::new();
    for k in keys.iter()? {
        result.push(k?);
    }
    // Sort by (is_getitem, getitem_idx, name) to match Python's _ordered_control_flow_users
    let py = node.py();
    let builtins = py.import_bound("builtins")?;
    let sorted_fn = builtins.getattr("sorted")?;
    // For simplicity, just use the Python sorted with a key matching the Python code
    let sort_key = py.eval_bound(
        r#"lambda user: (
            (0, user.args[1], getattr(user, 'name', ''))
            if 'getitem' in str(getattr(user, 'target', '')) and len(getattr(user, 'args', ())) > 1 and isinstance(user.args[1], int)
            else (1, 0, getattr(user, 'name', ''))
        )"#,
        None,
        None,
    )?;
    let py_list = pyo3::types::PyList::new_bound(py, &result);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("key", &sort_key)?;
    let sorted_list = sorted_fn.call((py_list,), Some(&kwargs))?;
    let mut out = Vec::new();
    for item in sorted_list.iter()? {
        out.push(item?);
    }
    Ok(out)
}

/// Is node a quantize_per_tensor call_function?
fn is_quantize_node(node: &Bound<'_, PyAny>) -> bool {
    let op: String = py_attr(node, &["op"], String::new());
    if op != "call_function" {
        return false;
    }
    let target = node
        .getattr("target")
        .and_then(|t| t.str().map(|s| s.to_string()))
        .unwrap_or_default();
    target.contains("quantize_per_tensor") && !target.contains("dequantize_per_tensor")
}

// ---------------------------------------------------------------------------
// 1. _regularize_control_flow_submodules
//    Syncs placeholder fake tensors with parent args, stamps metadata.
// ---------------------------------------------------------------------------

/// Resolve parent args for a control-flow node (cond or while_loop).
fn resolve_cf_parent_args<'py>(
    cf_node: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let target_str = cf_node
        .getattr("target")
        .and_then(|t| t.str().map(|s| s.to_string()))
        .unwrap_or_default();

    let all_args = FxGraphModule::node_args(cf_node)?;
    if target_str.contains("cond") {
        // cond: last arg is the list of operands
        if let Some(last) = all_args.last() {
            let mut v = Vec::new();
            if let Ok(iter) = last.iter() {
                for item in iter {
                    v.push(item?);
                }
            }
            Ok(v)
        } else {
            Ok(Vec::new())
        }
    } else if target_str.contains("while_loop") {
        // while_loop: args[-2] + args[-1]
        let len = all_args.len();
        if len >= 2 {
            let mut v = Vec::new();
            for list_arg in &all_args[len - 2..] {
                if let Ok(iter) = list_arg.iter() {
                    for item in iter {
                        v.push(item?);
                    }
                }
            }
            Ok(v)
        } else {
            Ok(Vec::new())
        }
    } else {
        Ok(Vec::new())
    }
}

/// Structural regularization: stamps placeholder markers, syncs fake tensors
/// and dim_order, but does NOT propagate qparams. Run before parent passes.
fn regularize_control_flow_submodules_structural<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_ctx = py.import_bound("retab._cf_ctx")?;
    let get_matching_fake_tensor = cf_ctx.getattr("_get_matching_fake_tensor")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut regularized: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let placeholders = submodule_placeholders(submodule)?;
        for ph in &placeholders {
            let meta = ph.getattr("meta")?;
            meta.set_item("is_input", true)?;
            meta.set_item("arm_control_flow_placeholder", true)?;
        }

        let parent_args = resolve_cf_parent_args(cf_node)?;

        for (ph, parent_arg) in placeholders.iter().zip(parent_args.iter()) {
            let fake = get_matching_fake_tensor.call1((parent_arg,))?;
            let meta = ph.getattr("meta")?;
            meta.set_item("val", &fake)?;

            let parent_meta = parent_arg.getattr("meta")?;
            if meta_val(&parent_meta, "tosa_dim_order").is_some() {
                let dim_order = parent_meta.get_item("tosa_dim_order")?;
                let builtins = py.import_bound("builtins")?;
                let py_tuple = builtins.call_method1("tuple", (&dim_order,))?;
                meta.set_item("tosa_dim_order", &py_tuple)?;
            }
            // No qparam propagation here — that happens post-parent-passes
        }

        regularized += 1;
    }

    Ok(regularized)
}

/// Full regularization including qparam propagation. Used by the legacy path.
fn regularize_control_flow_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let count = regularize_control_flow_submodules_structural(py, gm)?;
    propagate_parent_qparams_to_placeholders(py, gm)?;
    Ok(count)
}

/// Copy input_qparams and output_qparams from parent args to submodule
/// placeholders. Run AFTER parent passes so placeholders see post-fold qparams.
pub(crate) fn propagate_parent_qparams_to_placeholders<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut propagated: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        let placeholders = submodule_placeholders(submodule)?;
        let parent_args = resolve_cf_parent_args(cf_node)?;

        for (ph, parent_arg) in placeholders.iter().zip(parent_args.iter()) {
            let meta = ph.getattr("meta")?;
            let parent_meta = parent_arg.getattr("meta")?;

            for qp_key in &["input_qparams", "output_qparams"] {
                if let Some(parent_qp) = meta_val(&parent_meta, qp_key) {
                    let copied = parent_qp.call_method0("copy").unwrap_or(parent_qp.clone());
                    meta.set_item(*qp_key, &copied)?;
                    propagated += 1;
                }
            }
        }
    }

    if propagated > 0 {
        log_line(format_args!(
            "[propagate-boundary-qparams] propagated {} qparam(s) from parent args to submodule placeholders",
            propagated
        ));
    }

    Ok(propagated)
}

// ---------------------------------------------------------------------------
// 2. _annotate_control_flow_parent_rank4_dim_order
//    Stamps NHWC dim_order on parent CF node users (getitem nodes).
// ---------------------------------------------------------------------------

fn annotate_cf_parent_rank4_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let desired = PyTuple::new_bound(py, [0i64, 2, 3, 1]);
    let submodules = get_cf_submodules(py, gm)?;
    let mut annotated: u32 = 0;

    for (_name, _submodule, cf_node) in &submodules {
        let users = ordered_cf_users(cf_node)?;
        for user in &users {
            let meta = user.getattr("meta")?;
            let val = match meta_val(&meta, "val") {
                Some(v) => v,
                None => continue,
            };
            let dim: i64 = match val.call_method0("dim") {
                Ok(d) => d.extract().unwrap_or(0),
                Err(_) => continue,
            };
            if dim != 4 {
                continue;
            }
            meta.set_item("arm_control_flow_parent_output_user", true)?;

            // Check if already correct
            let current: Vec<i64> = meta_get(&meta, "tosa_dim_order", Vec::new());
            if current == [0, 2, 3, 1] {
                continue;
            }
            meta.set_item("tosa_dim_order", &desired)?;
            annotated += 1;
        }
    }

    Ok(annotated)
}

// ---------------------------------------------------------------------------
// 3. _annotate_control_flow_rank4_dim_order
//    Stamps dim_order on rank-4 call_function nodes in submodules.
// ---------------------------------------------------------------------------

fn annotate_cf_rank4_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let desired = PyTuple::new_bound(py, [0i64, 2, 3, 1]);
    let submodules = get_cf_submodules(py, gm)?;
    let mut annotated: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let mut modified = false;
        for node in submodule_nodes(&submodule)? {
            let op: String = node.getattr("op")?.extract()?;
            if op != "call_function" {
                continue;
            }
            let meta = node.getattr("meta")?;
            if meta_val(&meta, "tosa_dim_order").is_some() {
                continue;
            }
            let val = match meta_val(&meta, "val") {
                Some(v) => v,
                None => continue,
            };
            let dim: i64 = match val.call_method0("dim") {
                Ok(d) => d.extract().unwrap_or(0),
                Err(_) => continue,
            };
            if dim != 4 {
                continue;
            }
            meta.set_item("tosa_dim_order", &desired)?;
            annotated += 1;
            modified = true;
        }
        if modified {
            recompile(&submodule)?;
        }
    }

    Ok(annotated)
}

// ---------------------------------------------------------------------------
// 4. _force_control_flow_rank4_output_dim_order
//    Stamps dim_order on output-feeding nodes with rank-4 tensors.
// ---------------------------------------------------------------------------

fn force_cf_rank4_output_dim_order<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let desired = PyTuple::new_bound(py, [0i64, 2, 3, 1]);
    let submodules = get_cf_submodules(py, gm)?;
    let mut forced: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let output_node = match submodule_output_node(&submodule)? {
            Some(n) => n,
            None => continue,
        };
        let args = FxGraphModule::node_args(&output_node)?;
        if args.is_empty() {
            continue;
        }

        // output_node.args[0] can be a single node or a list of nodes
        let output_args: Vec<Bound<'py, PyAny>> = if FxGraphModule::is_node(&args[0]) {
            vec![args[0].clone()]
        } else if let Ok(iter) = args[0].iter() {
            let mut v = Vec::new();
            for item in iter {
                v.push(item?);
            }
            v
        } else {
            continue;
        };

        let mut modified = false;
        for output_arg in &output_args {
            if !FxGraphModule::is_node(output_arg) {
                continue;
            }
            let meta = output_arg.getattr("meta")?;
            let val = match meta_val(&meta, "val") {
                Some(v) => v,
                None => continue,
            };
            let dim: i64 = match val.call_method0("dim") {
                Ok(d) => d.extract().unwrap_or(0),
                Err(_) => continue,
            };
            if dim != 4 {
                continue;
            }
            let current: Vec<i64> = meta_get(&meta, "tosa_dim_order", Vec::new());
            if current == [0, 2, 3, 1] {
                continue;
            }
            meta.set_item("tosa_dim_order", &desired)?;
            forced += 1;
            modified = true;
        }
        if modified {
            recompile(&submodule)?;
        }
    }

    Ok(forced)
}

// ---------------------------------------------------------------------------
// 5. _restamp_control_flow_placeholder_dtypes
//    Updates placeholder val dtype to match qparam dtype.
// ---------------------------------------------------------------------------

fn restamp_cf_placeholder_dtypes<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_mod = py.import_bound("retab._control_flow")?;
    let branch_local_fn =
        cf_mod.getattr("_branch_local_control_flow_output_qparam")?;
    let extract_fn = cf_mod.getattr("_extract_single_qparam_entry")?;

    let torch = py.import_bound("torch")?;
    let int8 = torch.getattr("int8")?;
    let int16 = torch.getattr("int16")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut restamped: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let mut updated = false;
        let placeholders = submodule_placeholders(&submodule)?;

        for ph in &placeholders {
            // Skip if any user is a quantize node
            let users = FxGraphModule::node_users(ph)?;
            let has_quantize_user = users.iter().any(|u| is_quantize_node(u));
            if has_quantize_user {
                continue;
            }

            // Get qparam
            let qparam = branch_local_fn.call1((ph,))?;

            // Check if qparam.dtype is int8 or int16; if not, try input_qparams
            let qparam = if qparam.is_none() {
                py.None().into_bound(py)
            } else {
                let dtype = qparam.getattr("dtype");
                let is_int8_or_int16 = match &dtype {
                    Ok(dt) => dt.eq(&int8).unwrap_or(false) || dt.eq(&int16).unwrap_or(false),
                    Err(_) => false,
                };
                if !is_int8_or_int16 {
                    let meta = ph.getattr("meta")?;
                    let input_qparams = meta.call_method1("get", ("input_qparams",))?;
                    let extracted = extract_fn.call1((&input_qparams,))?;
                    if extracted.is_none() {
                        qparam
                    } else {
                        extracted
                    }
                } else {
                    qparam
                }
            };

            if qparam.is_none() {
                continue;
            }

            let meta = ph.getattr("meta")?;
            let fake_val = match meta_val(&meta, "val") {
                Some(v) => v,
                None => continue,
            };
            if !fake_val.hasattr("to")? {
                continue;
            }

            // Check if dtype already matches
            let fake_dtype = fake_val.getattr("dtype").ok();
            let qparam_dtype = qparam.getattr("dtype").ok();
            match (&fake_dtype, &qparam_dtype) {
                (Some(fd), Some(qd)) if fd.eq(qd).unwrap_or(false) => continue,
                _ => {}
            }

            // Restamp
            if let Some(qd) = qparam_dtype {
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("dtype", &qd)?;
                let new_val = fake_val.call_method("to", (), Some(&kwargs))?;
                meta.set_item("val", &new_val)?;
                updated = true;
                restamped += 1;
            }
        }

        if updated {
            let graph = submodule.getattr("graph")?;
            graph.call_method0("lint")?;
            recompile(&submodule)?;
        }
    }

    Ok(restamped)
}

// ---------------------------------------------------------------------------
// 6. _snapshot_control_flow_table_qparams
//    Saves qparam metadata on TABLE-eligible nodes.
// ---------------------------------------------------------------------------

fn snapshot_cf_table_qparams<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let table_ops_mod =
        py.import_bound("executorch.backends.arm._passes.insert_table_ops")?;
    let table_ops_cls = table_ops_mod.getattr("TableOps")?;
    let included_ops = table_ops_cls.call_method0("included_ops")?;
    let table_targets: Vec<Bound<'py, PyAny>> = included_ops.iter()?.filter_map(|r| r.ok()).collect();

    let quant_args_mod =
        py.import_bound("executorch.backends.arm._passes.quant_args")?;
    let quant_args_cls = quant_args_mod.getattr("QuantArgs")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut snapshotted: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        for node in submodule_nodes(&submodule)? {
            let op: String = node.getattr("op")?.extract()?;
            if op != "call_function" {
                continue;
            }
            let target = node.getattr("target")?;

            // Check if target is in table_targets
            let is_table = table_targets.iter().any(|t| target.eq(t).unwrap_or(false));
            if !is_table {
                continue;
            }

            let meta = node.getattr("meta")?;
            let args = FxGraphModule::node_args(&node)?;

            // Input qparam: from first arg
            if !args.is_empty() && FxGraphModule::is_node(&args[0]) {
                let input_node = &args[0];
                let input_target = input_node.getattr("target")?;
                let input_args = FxGraphModule::node_args(input_node)?;
                let input_args_tuple = PyTuple::new_bound(py, &input_args);
                match quant_args_cls.call_method1("from_operator", (&input_target, &input_args_tuple)) {
                    Ok(qparam) => {
                        meta.set_item("arm_saved_table_input_qparam", &qparam)?;
                        snapshotted += 1;
                    }
                    Err(_) => {}
                }
            }

            // Output qparam: from first user
            let users = FxGraphModule::node_users(&node)?;
            for user in &users {
                let user_target = user.getattr("target")?;
                let user_args = FxGraphModule::node_args(user)?;
                let user_args_tuple = PyTuple::new_bound(py, &user_args);
                match quant_args_cls.call_method1("from_operator", (&user_target, &user_args_tuple)) {
                    Ok(qparam) => {
                        meta.set_item("arm_saved_table_output_qparam", &qparam)?;
                        snapshotted += 1;
                        break;
                    }
                    Err(_) => continue,
                }
            }
        }
    }

    Ok(snapshotted)
}

// ---------------------------------------------------------------------------
// 7. _refresh_tosa_dim_order_fake_tensors
//    Re-sync fake tensors with tosa_dim_order metadata.
// ---------------------------------------------------------------------------

pub(crate) fn refresh_tosa_dim_order_fake_tensors<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let cf_ctx = py.import_bound("retab._cf_ctx")?;
    let get_matching_fake_tensor = cf_ctx.getattr("_get_matching_fake_tensor")?;

    // Collect all modules: parent + submodules
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }

    for module in &modules {
        let mut updated = false;
        for node in submodule_nodes(module)? {
            let meta = node.getattr("meta")?;
            if meta_val(&meta, "val").is_none() {
                continue;
            }
            if meta_val(&meta, "tosa_dim_order").is_none() {
                continue;
            }
            let new_val = get_matching_fake_tensor.call1((&node,))?;
            meta.set_item("val", &new_val)?;
            updated = true;
        }
        if updated {
            recompile(module)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Combined FxPass
// ---------------------------------------------------------------------------

pub struct FxControlFlowMetadataPass;

impl FxPass for FxControlFlowMetadataPass {
    fn name(&self) -> &str {
        "FxControlFlowMetadata"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        // Check if there are any cond/while_loop nodes at all.
        // If not, skip everything.
        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        let mut any_modified = false;

        // 1. Regularize (sync placeholders with parent args)
        let count = regularize_control_flow_submodules(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] regularized {} control-flow submodule(s)",
                count
            ));
            any_modified = true;
        }

        // 2. Annotate parent rank-4 dim_order
        let count = annotate_cf_parent_rank4_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] annotated NHWC dim-order on {} parent CF output user(s)",
                count
            ));
            any_modified = true;
        }

        // 3. Annotate rank-4 dim_order in submodules
        let count = annotate_cf_rank4_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] annotated NHWC dim-order on {} interior rank-4 node(s)",
                count
            ));
            any_modified = true;
        }

        // 4. Force output dim_order
        let count = force_cf_rank4_output_dim_order(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] forced NHWC dim-order on {} rank-4 output(s)",
                count
            ));
            any_modified = true;
        }

        // 5. Restamp placeholder dtypes
        let count = restamp_cf_placeholder_dtypes(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] restamped {} placeholder dtype(s)",
                count
            ));
            any_modified = true;
        }

        // 6. Snapshot table qparams
        let count = snapshot_cf_table_qparams(py, gm_ref)?;
        if count > 0 {
            log_line(format_args!(
                "[FxControlFlowMetadata] snapshotted {} table qparam attachment(s)",
                count
            ));
            any_modified = true;
        }

        // 7. Refresh fake tensors with dim_order
        refresh_tosa_dim_order_fake_tensors(py, gm_ref)?;

        // Recompile the parent graph module if we made changes
        if any_modified {
            gm.recompile()?;
        }

        Ok(any_modified)
    }
}

// ===========================================================================
// Structural-only pre-pass (no qparam propagation)
// ===========================================================================

/// Pre-pass that stamps structural metadata on control-flow submodules WITHOUT
/// propagating qparams. Run before parent passes.
pub struct FxControlFlowStructuralPass;

impl FxPass for FxControlFlowStructuralPass {
    fn name(&self) -> &str {
        "FxControlFlowStructural"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        let mut any_modified = false;

        // 1. Structural regularize (no qparams)
        let count = regularize_control_flow_submodules_structural(py, gm_ref)?;
        if count > 0 {
            any_modified = true;
        }

        // 2-4. Dim order annotations
        annotate_cf_parent_rank4_dim_order(py, gm_ref)?;
        annotate_cf_rank4_dim_order(py, gm_ref)?;
        force_cf_rank4_output_dim_order(py, gm_ref)?;

        // 5. Restamp placeholder dtypes
        restamp_cf_placeholder_dtypes(py, gm_ref)?;

        // 6. Snapshot table qparams
        snapshot_cf_table_qparams(py, gm_ref)?;

        // 7. Refresh fake tensors with dim_order
        refresh_tosa_dim_order_fake_tensors(py, gm_ref)?;

        if any_modified {
            gm.recompile()?;
        }

        Ok(any_modified)
    }
}

// ===========================================================================
// QParam propagation post-pass
// ===========================================================================

/// Post-pass that copies parent post-pass qparams onto submodule placeholders.
/// Run AFTER parent passes so placeholders see correct post-fold qparams.
pub struct FxControlFlowQParamPropagationPass;

impl FxPass for FxControlFlowQParamPropagationPass {
    fn name(&self) -> &str {
        "FxControlFlowQParamPropagation"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        // Propagate qparams from parent args to submodule placeholders
        let count = propagate_parent_qparams_to_placeholders(py, gm_ref)?;

        // Restamp placeholder dtypes now that qparams are present
        restamp_cf_placeholder_dtypes(py, gm_ref)?;

        // Re-snapshot table qparams with correct qparams
        snapshot_cf_table_qparams(py, gm_ref)?;

        Ok(count > 0)
    }
}

// ===========================================================================
// Cleanup pass: graph surgery functions ported from _cf_rewrites.py / _cf_ctx.py
// ===========================================================================

// ---------------------------------------------------------------------------
// Helpers for the cleanup pass
// ---------------------------------------------------------------------------

/// Check if a node is a call_function whose target string contains any of the given markers.
fn is_call_function_target(node: &Bound<'_, PyAny>, markers: &[&str]) -> bool {
    let op: String = py_attr(node, &["op"], String::new());
    if op != "call_function" {
        return false;
    }
    let target_str = node
        .getattr("target")
        .and_then(|t| t.str().map(|s| s.to_string()))
        .unwrap_or_default();
    markers.iter().any(|m| target_str.contains(m))
}

/// Check if a node is a TOSA RESCALE call_function.
fn is_tosa_rescale_node(node: &Bound<'_, PyAny>) -> bool {
    is_call_function_target(
        node,
        &["tosa_RESCALE", "tosa.RESCALE", "TosaRescale"],
    )
}

/// Check if a node is a TOSA TRANSPOSE call_function.
fn is_tosa_transpose_node(node: &Bound<'_, PyAny>) -> bool {
    is_call_function_target(
        node,
        &["tosa.TRANSPOSE", "tosa_TRANSPOSE", "TosaTranspose"],
    )
}

/// Cleanup graph: eliminate dead code + lint.
fn cleanup_graph(graph: &Bound<'_, PyAny>) -> PyResult<()> {
    graph.call_method0("eliminate_dead_code")?;
    graph.call_method0("lint")?;
    Ok(())
}

/// Get the rescale signature: canonicalized args[1:5].
fn rescale_signature<'py>(node: &Bound<'py, PyAny>) -> Option<Vec<String>> {
    if !is_tosa_rescale_node(node) {
        return None;
    }
    let args = FxGraphModule::node_args(node).ok()?;
    if args.len() < 5 {
        return None;
    }
    let mut sig = Vec::new();
    for arg in &args[1..5] {
        sig.push(format!("{}", arg));
    }
    Some(sig)
}

/// Extract a single qparam entry from a qparams dict.
/// Mirrors Python `_extract_single_qparam_entry`.
fn extract_single_qparam_entry<'py>(
    py: Python<'py>,
    qparams: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    if qparams.is_none() {
        return None;
    }
    let is_truthy = qparams.is_truthy().unwrap_or(false);
    if !is_truthy {
        return None;
    }
    // Try dict with key 0
    if let Ok(val) = qparams.get_item(0i64) {
        if !val.is_none() {
            return Some(val);
        }
    }
    // Try dict with len==1 -> return first value
    if let Ok(len) = qparams.len() {
        if len == 1 {
            if let Ok(values) = qparams.call_method0("values") {
                if let Ok(mut iter) = values.iter() {
                    if let Some(Ok(val)) = iter.next() {
                        return Some(val);
                    }
                }
            }
        }
    }
    // Return qparams itself as fallback
    let _ = py;
    Some(qparams.clone())
}

/// Extract the effective quantization parameters for a node by walking
/// producer edges. Pure Rust port of Python `_extract_effective_node_qparam`.
fn extract_effective_node_qparam<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let mut current = node.clone();
    let mut seen = std::collections::HashSet::new();

    loop {
        if current.getattr("op").is_err() {
            break;
        }
        let node_id: usize = current
            .call_method0("__hash__")
            .and_then(|h| h.extract())
            .unwrap_or(0);
        if !seen.insert(node_id) {
            break;
        }

        let meta = current.getattr("meta")?;

        // Check output_qparams first
        if let Ok(oqp) = meta.get_item("output_qparams") {
            if let Some(qp) = extract_single_qparam_entry(py, &oqp) {
                return Ok(Some(qp));
            }
        }

        // Check input_qparams (prefer_input_qparams=True)
        if let Ok(iqp) = meta.get_item("input_qparams") {
            if let Some(qp) = extract_single_qparam_entry(py, &iqp) {
                return Ok(Some(qp));
            }
        }

        // RESCALE reverse-engineering: derive qparam from source + multiplier
        let args = FxGraphModule::node_args(&current)?;
        if let Some(qp) = try_reverse_engineer_rescale_qparam(py, &current, &args)? {
            return Ok(Some(qp));
        }

        // Walk to producer (args[0])
        if args.is_empty() || !FxGraphModule::is_node(&args[0]) {
            break;
        }
        current = args[0].clone();
    }
    Ok(None)
}

/// Try to reverse-engineer a QuantArgs from a TOSA RESCALE node's args.
fn try_reverse_engineer_rescale_qparam<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    args: &[Bound<'py, PyAny>],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !is_tosa_rescale_node(node) || args.len() < 5 {
        return Ok(None);
    }
    let source = &args[0];
    if source.getattr("op").is_err() {
        return Ok(None);
    }
    let source_qp = match extract_effective_node_qparam(py, source)? {
        Some(qp) => qp,
        None => return Ok(None),
    };
    if source_qp.getattr("per_channel")?.extract::<bool>().unwrap_or(false) {
        return Ok(None);
    }
    let torch_dtype_cls = py.import_bound("torch")?.getattr("dtype")?;
    if !args[1].is_instance(&torch_dtype_cls).unwrap_or(false) {
        return Ok(None);
    }
    if args[4].extract::<i64>().is_err() {
        return Ok(None);
    }

    // Multiplier must be a single-element list/tuple with nonzero value
    let mult_len = args[2].len().unwrap_or(0);
    if mult_len != 1 {
        return Ok(None);
    }
    let mult_val: f64 = args[2].get_item(0)?.extract().unwrap_or(0.0);
    if mult_val == 0.0 {
        return Ok(None);
    }

    let torch_mod = py.import_bound("torch")?;
    let qinfo = match torch_mod.call_method1("iinfo", (&args[1],)) {
        Ok(qi) => qi,
        Err(_) => return Ok(None),
    };

    let source_scale: f64 = source_qp.call_method0("get_scale_per_tensor")?.extract()?;
    let output_zp: i64 = args[4].extract()?;
    let qmin: i64 = qinfo.getattr("min")?.extract()?;
    let qmax: i64 = qinfo.getattr("max")?.extract()?;

    let quant_args_cls = py
        .import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("scale", source_scale / mult_val)?;
    kwargs.set_item("zp", output_zp)?;
    kwargs.set_item("qmin", qmin)?;
    kwargs.set_item("qmax", qmax)?;
    kwargs.set_item("dtype", &args[1])?;
    Ok(Some(quant_args_cls.call((), Some(&kwargs))?))
}

/// Stamp single qparams on a node (input_qparam and/or output_qparam).
fn stamp_single_qparams<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    input_qparam: Option<&Bound<'py, PyAny>>,
    output_qparam: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let meta = node.getattr("meta")?;
    if let Some(iqp) = input_qparam {
        let d = PyDict::new_bound(py);
        d.set_item(0i64, iqp)?;
        meta.set_item("input_qparams", &d)?;
    }
    if let Some(oqp) = output_qparam {
        let d = PyDict::new_bound(py);
        d.set_item(0i64, oqp)?;
        meta.set_item("output_qparams", &d)?;
    }
    Ok(())
}

/// Copy output contract meta from src to dst.
fn copy_output_contract_meta<'py>(
    dst: &Bound<'py, PyAny>,
    src: &Bound<'py, PyAny>,
    copy_output_qparams: bool,
) -> PyResult<()> {
    let dst_meta = dst.getattr("meta")?;
    let src_meta = src.getattr("meta")?;
    // val
    if let Some(val) = meta_val(&src_meta, "val") {
        dst_meta.set_item("val", &val)?;
    }
    // tosa_dim_order
    if let Some(tdo) = meta_val(&src_meta, "tosa_dim_order") {
        let py = dst.py();
        let builtins = py.import_bound("builtins")?;
        let t = builtins.call_method1("tuple", (&tdo,))?;
        dst_meta.set_item("tosa_dim_order", &t)?;
    }
    // output_qparams
    if copy_output_qparams {
        if let Some(oqp) = meta_val(&src_meta, "output_qparams") {
            let py = dst.py();
            let builtins = py.import_bound("builtins")?;
            let d = builtins.call_method1("dict", (&oqp,))?;
            dst_meta.set_item("output_qparams", &d)?;
        }
    }
    Ok(())
}

/// Get all modules (parent + submodules) for iteration.
fn get_all_modules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }
    Ok(modules)
}

/// Get `_actual_shape_from_meta` for a node. Returns None if not available.
fn actual_shape_from_meta<'py>(node: &Bound<'py, PyAny>) -> PyResult<Option<Vec<i64>>> {
    let meta = node.getattr("meta")?;
    let fake_tensor = match meta_val(&meta, "val") {
        Some(v) => v,
        None => return Ok(None),
    };
    let shape_obj = match fake_tensor.getattr("shape") {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let shape: Vec<i64> = match shape_obj.extract() {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let dim_order = meta_val(&meta, "tosa_dim_order");
    if let Some(tdo) = dim_order {
        let tdo_vec: Vec<usize> = match tdo.extract() {
            Ok(v) => v,
            Err(_) => return Ok(Some(shape)),
        };
        if tdo_vec.len() == shape.len() {
            let identity: Vec<usize> = (0..shape.len()).collect();
            if tdo_vec != identity {
                let permuted: Vec<i64> = tdo_vec.iter().map(|&idx| shape[idx]).collect();
                return Ok(Some(permuted));
            }
        }
    }
    Ok(Some(shape))
}

// ---------------------------------------------------------------------------
// 1. _remove_control_flow_noop_passthroughs
// ---------------------------------------------------------------------------

pub(crate) fn remove_control_flow_noop_passthroughs<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    let mut rewritten: u32 = 0;

    let noop_markers = &[
        "dim_order_ops._clone_dim_order",
        "aten.clone",
        "aten.alias_copy",
        "aten.detach_copy",
    ];

    for (_name, submodule, _cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;
        let mut updated = false;

        // Collect nodes into a list first (we'll be modifying the graph)
        let nodes = submodule_nodes(&submodule)?;

        for node in &nodes {
            if !is_call_function_target(node, noop_markers) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.is_empty() || !FxGraphModule::is_node(&args[0]) {
                continue;
            }

            let source = &args[0];

            // Check dtype match
            let node_meta = node.getattr("meta")?;
            let source_meta = source.getattr("meta")?;
            let node_val = meta_val(&node_meta, "val");
            let source_val = meta_val(&source_meta, "val");

            // dtype check
            if let (Some(ref nv), Some(ref sv)) = (&node_val, &source_val) {
                let nd = nv.getattr("dtype").ok();
                let sd = sv.getattr("dtype").ok();
                if let (Some(ref nd), Some(ref sd)) = (&nd, &sd) {
                    if !nd.is_none() && !sd.is_none() && !nd.eq(sd).unwrap_or(true) {
                        continue;
                    }
                }
            }

            // shape check
            if let (Some(ref nv), Some(ref sv)) = (&node_val, &source_val) {
                let ns = nv.getattr("shape").ok();
                let ss = sv.getattr("shape").ok();
                if let (Some(ref ns), Some(ref ss)) = (&ns, &ss) {
                    let ns_t: Result<Vec<i64>, _> = ns.extract();
                    let ss_t: Result<Vec<i64>, _> = ss.extract();
                    if let (Ok(ns_t), Ok(ss_t)) = (ns_t, ss_t) {
                        if ns_t != ss_t {
                            continue;
                        }
                    }
                }
            }

            // dim_order check
            let node_dim_order = meta_val(&node_meta, "tosa_dim_order");
            let source_dim_order = meta_val(&source_meta, "tosa_dim_order");
            if let (Some(ref ndo), Some(ref sdo)) = (&node_dim_order, &source_dim_order) {
                let ndo_v: Result<Vec<i64>, _> = ndo.extract();
                let sdo_v: Result<Vec<i64>, _> = sdo.extract();
                if let (Ok(ndo_v), Ok(sdo_v)) = (ndo_v, sdo_v) {
                    if ndo_v != sdo_v {
                        continue;
                    }
                }
            }

            // Propagate metadata from node to source if source is missing it
            if meta_val(&source_meta, "tosa_dim_order").is_none() {
                if let Some(ref ndo) = node_dim_order {
                    let builtins = py.import_bound("builtins")?;
                    let t = builtins.call_method1("tuple", (ndo,))?;
                    source_meta.set_item("tosa_dim_order", &t)?;
                }
            }
            if meta_val(&source_meta, "input_qparams").is_none() {
                if let Some(iqp) = meta_val(&node_meta, "input_qparams") {
                    if iqp.is_truthy().unwrap_or(false) {
                        let builtins = py.import_bound("builtins")?;
                        let d = builtins.call_method1("dict", (&iqp,))?;
                        source_meta.set_item("input_qparams", &d)?;
                    }
                }
            }
            if meta_val(&source_meta, "output_qparams").is_none() {
                if let Some(oqp) = meta_val(&node_meta, "output_qparams") {
                    if oqp.is_truthy().unwrap_or(false) {
                        let builtins = py.import_bound("builtins")?;
                        let d = builtins.call_method1("dict", (&oqp,))?;
                        source_meta.set_item("output_qparams", &d)?;
                    }
                }
            }

            // Replace all uses and erase
            node.call_method1("replace_all_uses_with", (source,))?;
            graph.call_method1("erase_node", (node,))?;
            rewritten += 1;
            updated = true;
        }

        if updated {
            cleanup_graph(&graph)?;
            recompile(&submodule)?;
        }
    }

    Ok(rewritten)
}

// ---------------------------------------------------------------------------
// 2. _prune_duplicate_terminal_rescales
// ---------------------------------------------------------------------------

pub(crate) fn prune_duplicate_terminal_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let mut pruned: u32 = 0;

    // Include parent + submodules (matches Python include_parent=True)
    let mut modules: Vec<Bound<'py, PyAny>> = vec![gm.clone()];
    let submodules = get_cf_submodules(py, gm)?;
    for (_name, submodule, _cf_node) in &submodules {
        modules.push(submodule.clone());
    }

    for module in &modules {
        let graph = module.getattr("graph")?;
        let mut modified = false;

        let nodes = submodule_nodes(module)?;

        for node in &nodes {
            if !is_tosa_rescale_node(node) {
                continue;
            }
            // Check single user
            let users = FxGraphModule::node_users(node)?;
            if users.len() != 1 {
                continue;
            }
            let user = &users[0];

            if !is_tosa_rescale_node(user) {
                continue;
            }

            // user.args[0] is node
            let user_args = FxGraphModule::node_args(user)?;
            if user_args.is_empty() {
                continue;
            }
            let user_arg0_is_node = user_args[0].is(node);
            if !user_arg0_is_node {
                continue;
            }

            // Compare rescale signatures
            let node_sig = rescale_signature(node);
            let user_sig = rescale_signature(user);
            match (&node_sig, &user_sig) {
                (Some(ns), Some(us)) if ns == us => {}
                _ => continue,
            }

            // Check qparams: first has no output_qparams, user has output_qparams
            let node_meta = node.getattr("meta")?;
            let user_meta = user.getattr("meta")?;

            let first_output_qparam = {
                let oqp = meta_val(&node_meta, "output_qparams");
                match oqp {
                    Some(ref v) => extract_single_qparam_entry(py, v),
                    None => None,
                }
            };
            let final_output_qparam = {
                let oqp = meta_val(&user_meta, "output_qparams");
                match oqp {
                    Some(ref v) => extract_single_qparam_entry(py, v),
                    None => None,
                }
            };

            if first_output_qparam.is_some() {
                continue;
            }
            if final_output_qparam.is_none() {
                continue;
            }
            let final_output_qparam = final_output_qparam.unwrap();

            // Rewrite: node.args = (source, user.args[1..5])
            let node_args = FxGraphModule::node_args(node)?;
            let source = &node_args[0];

            let new_args = PyTuple::new_bound(
                py,
                &[
                    source.clone(),
                    user_args[1].clone(),
                    user_args[2].clone(),
                    user_args[3].clone(),
                    user_args[4].clone(),
                ],
            );
            node.setattr("args", &new_args)?;

            // Copy output contract meta (without output_qparams)
            copy_output_contract_meta(node, user, false)?;

            // If node has no input_qparams, try to get from source
            if meta_val(&node_meta, "input_qparams").is_none()
                || !node_meta
                    .get_item("input_qparams")
                    .map(|v| v.is_truthy().unwrap_or(false))
                    .unwrap_or(false)
            {
                let source_meta = source.getattr("meta")?;
                if let Some(soqp) = meta_val(&source_meta, "output_qparams") {
                    if let Some(source_qparam) = extract_single_qparam_entry(py, &soqp) {
                        stamp_single_qparams(py, node, Some(&source_qparam), None)?;
                    }
                }
            }

            // Stamp output_qparam
            stamp_single_qparams(py, node, None, Some(&final_output_qparam))?;

            // Replace user with node and erase user
            user.call_method1("replace_all_uses_with", (node,))?;
            graph.call_method1("erase_node", (user,))?;
            pruned += 1;
            modified = true;
        }

        if modified {
            cleanup_graph(&graph)?;
            recompile(module)?;
        }
    }

    Ok(pruned)
}

// ---------------------------------------------------------------------------
// 3. _restore_missing_control_flow_rescale_qparams
// ---------------------------------------------------------------------------

pub(crate) fn restore_missing_control_flow_rescale_qparams<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let torch = py.import_bound("torch")?;
    let int8 = torch.getattr("int8")?;
    let int16 = torch.getattr("int16")?;
    let int32 = torch.getattr("int32")?;
    let quant_args_mod =
        py.import_bound("executorch.backends.arm._passes.quant_args")?;
    let quant_args_cls = quant_args_mod.getattr("QuantArgs")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut restored: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;
        let mut modified = false;

        for node in submodule_nodes(&submodule)? {
            if !is_tosa_rescale_node(&node) {
                continue;
            }
            let args = FxGraphModule::node_args(&node)?;
            if args.len() < 5 {
                continue;
            }

            let meta = node.getattr("meta")?;
            let has_input = meta_val(&meta, "input_qparams")
                .map(|v| v.is_truthy().unwrap_or(false))
                .unwrap_or(false);
            let has_output = meta_val(&meta, "output_qparams")
                .map(|v| v.is_truthy().unwrap_or(false))
                .unwrap_or(false);
            if has_input && has_output {
                continue;
            }

            // source = args[0]
            let source = &args[0];
            if !FxGraphModule::is_node(source) {
                continue;
            }

            let source_qparam = match extract_effective_node_qparam(py, source)? {
                Some(qp) => qp,
                None => continue,
            };

            // output_dtype = args[1]
            let output_dtype = &args[1];
            let is_valid_dtype = output_dtype.eq(&int8).unwrap_or(false)
                || output_dtype.eq(&int16).unwrap_or(false)
                || output_dtype.eq(&int32).unwrap_or(false);
            if !is_valid_dtype {
                continue;
            }

            // scale_arg = args[2], must be list/tuple of length 1, non-zero
            let scale_arg = &args[2];
            let scale_list: Vec<f64> = match scale_arg.extract() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if scale_list.len() != 1 || scale_list[0] == 0.0 {
                continue;
            }

            // Build output qparam
            let source_scale = source_qparam.call_method0("get_scale_per_tensor")?;
            let source_scale_f: f64 = match source_scale.extract() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let output_scale = source_scale_f / scale_list[0];

            let zp_val: i64 = match args[4].extract() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let qinfo = torch.call_method1("iinfo", (output_dtype,))?;
            let qmin: i64 = qinfo.getattr("min")?.extract()?;
            let qmax: i64 = qinfo.getattr("max")?.extract()?;

            let output_qparam = match quant_args_cls.call1((
                output_scale,
                zp_val,
                qmin,
                qmax,
                output_dtype,
            )) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if !has_input {
                stamp_single_qparams(py, &node, Some(&source_qparam), None)?;
            }
            if !has_output {
                stamp_single_qparams(py, &node, None, Some(&output_qparam))?;
            }
            restored += 1;
            modified = true;
        }

        if modified {
            recompile(&submodule)?;
        }
        let _ = graph;
    }

    Ok(restored)
}

// ---------------------------------------------------------------------------
// 4. _restore_control_flow_output_dequantizes
//    Delegates to Python because it depends on complex boundary contract infra.
// ---------------------------------------------------------------------------

fn restore_control_flow_output_dequantizes<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_rewrites = py.import_bound("retab._cf_rewrites")?;
    let func = cf_rewrites.getattr("_restore_control_flow_output_dequantizes")?;
    func.call1((gm,))?;
    // Python function doesn't return a count, but we can estimate from log.
    // For simplicity, return 0 and let the Python side log.
    Ok(0)
}

// ---------------------------------------------------------------------------
// 5. _restore_missing_control_flow_input_rescales
//    Delegates to Python because it depends on complex boundary contract infra.
// ---------------------------------------------------------------------------

fn restore_missing_control_flow_input_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_rewrites = py.import_bound("retab._cf_rewrites")?;
    let func = cf_rewrites.getattr("_restore_missing_control_flow_input_rescales")?;
    func.call1((gm,))?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// 6. _refresh_tosa_transpose_fake_tensors
// ---------------------------------------------------------------------------

pub(crate) fn refresh_tosa_transpose_fake_tensors<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let modules = get_all_modules(py, gm)?;
    let mut refreshed: u32 = 0;

    for module in &modules {
        let mut updated = false;

        for node in submodule_nodes(module)? {
            if !is_tosa_transpose_node(&node) {
                continue;
            }

            let args = FxGraphModule::node_args(&node)?;
            if args.len() < 2 {
                continue;
            }

            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            let input_meta = input_node.getattr("meta")?;
            let input_fake = match meta_val(&input_meta, "val") {
                Some(v) => v,
                None => continue,
            };

            if !input_fake.hasattr("new_empty")? {
                continue;
            }

            let input_shape = match actual_shape_from_meta(input_node)? {
                Some(s) => s,
                None => continue,
            };

            // Parse perm from args[1]
            let perm_obj = &args[1];
            let perm: Vec<usize> = match perm_obj.extract() {
                Ok(v) => v,
                Err(_) => continue,
            };

            if perm.len() != input_shape.len() {
                continue;
            }

            // Compute new shape
            let new_shape: Vec<i64> = perm.iter().map(|&idx| input_shape[idx]).collect();

            let node_meta = node.getattr("meta")?;
            let node_fake = meta_val(&node_meta, "val").unwrap_or(input_fake.clone());

            let base_tensor = if node_fake.hasattr("new_empty")? {
                node_fake.clone()
            } else {
                input_fake.clone()
            };

            // Get dtype
            let dtype = node_fake
                .getattr("dtype")
                .or_else(|_| input_fake.getattr("dtype"))?;

            // Create new tensor: base_tensor.new_empty(new_shape, dtype=dtype)
            let shape_list = PyList::new_bound(py, &new_shape);
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", &dtype)?;
            let new_val = base_tensor.call_method("new_empty", (&shape_list,), Some(&kwargs))?;

            node_meta.set_item("val", &new_val)?;

            // Set tosa_dim_order to identity
            let identity: Vec<i64> = (0..new_shape.len() as i64).collect();
            let identity_tuple = PyTuple::new_bound(py, &identity);
            node_meta.set_item("tosa_dim_order", &identity_tuple)?;

            updated = true;
            refreshed += 1;
        }

        if updated {
            recompile(module)?;
        }
    }

    Ok(refreshed)
}

// NOTE: _refresh_tosa_dim_order_fake_tensors (item 7) is already implemented
// above as `refresh_tosa_dim_order_fake_tensors`.

/// Check if a node is a dequantize_per_tensor call_function.
fn is_dequantize_node(node: &Bound<'_, PyAny>) -> bool {
    is_call_function_target(node, &["dequantize_per_tensor"])
}

/// Check if a node is a compare op (eq, ge, gt, lt, le).
fn is_compare_user(node: &Bound<'_, PyAny>) -> bool {
    is_call_function_target(
        node,
        &[
            "aten.gt.",
            "aten.ge.",
            "aten.lt.",
            "aten.le.",
            "aten.eq.",
            "tosa.GREATER",
            "tosa.GREATER_EQUAL",
            "tosa.EQUAL",
            "TosaGreater",
            "TosaGreaterEqual",
            "TosaEqual",
        ],
    )
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Q/DQ node helpers (ported from Python)
// ---------------------------------------------------------------------------

/// Parse scale/zp/qmin/qmax/dtype from a Q or DQ node's literal args.
/// Returns a Python QuantArgs object or None.
fn parse_per_tensor_qparam<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 6 {
        return Ok(None);
    }
    let torch_dtype_cls = py.import_bound("torch")?.getattr("dtype")?;
    if !args[5].is_instance(&torch_dtype_cls).unwrap_or(false) {
        return Ok(None);
    }
    let quant_args_cls = py
        .import_bound("executorch.backends.arm._passes.quant_args")?
        .getattr("QuantArgs")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("scale", args[1].extract::<f64>().unwrap_or(1.0))?;
    kwargs.set_item("zp", args[2].extract::<i64>().unwrap_or(0))?;
    kwargs.set_item("qmin", args[3].extract::<i64>().unwrap_or(-128))?;
    kwargs.set_item("qmax", args[4].extract::<i64>().unwrap_or(127))?;
    kwargs.set_item("dtype", &args[5])?;
    kwargs.set_item("axis", 0i64)?;
    kwargs.set_item("per_channel", false)?;
    Ok(Some(quant_args_cls.call((), Some(&kwargs))?))
}

/// Compare two QuantArgs for equality (scale, zp, dtype, optionally range).
fn same_control_flow_qparam(
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    include_range: bool,
) -> bool {
    let lhs_scale: f64 = lhs.getattr("scale").and_then(|v| v.extract()).unwrap_or(1.0);
    let rhs_scale: f64 = rhs.getattr("scale").and_then(|v| v.extract()).unwrap_or(1.0);
    let lhs_zp: i64 = lhs.getattr("zp").and_then(|v| v.extract()).unwrap_or(0);
    let rhs_zp: i64 = rhs.getattr("zp").and_then(|v| v.extract()).unwrap_or(0);
    if lhs_scale != rhs_scale || lhs_zp != rhs_zp {
        return false;
    }
    let lhs_dtype = lhs.getattr("dtype").ok().map(|v| format!("{}", v));
    let rhs_dtype = rhs.getattr("dtype").ok().map(|v| format!("{}", v));
    if lhs_dtype != rhs_dtype {
        return false;
    }
    if include_range {
        let lhs_qmin: i64 = lhs.getattr("qmin").and_then(|v| v.extract()).unwrap_or(0);
        let rhs_qmin: i64 = rhs.getattr("qmin").and_then(|v| v.extract()).unwrap_or(0);
        let lhs_qmax: i64 = lhs.getattr("qmax").and_then(|v| v.extract()).unwrap_or(0);
        let rhs_qmax: i64 = rhs.getattr("qmax").and_then(|v| v.extract()).unwrap_or(0);
        if lhs_qmin != rhs_qmin || lhs_qmax != rhs_qmax {
            return false;
        }
    }
    true
}

/// Check if a node is a zero-input constant source (get_attr or no input args).
fn is_zero_input_const_source(node: &Bound<'_, PyAny>) -> bool {
    let op: String = py_attr(node, &["op"], String::new());
    if op == "get_attr" {
        return true;
    }
    if op != "call_function" {
        return false;
    }
    let args = FxGraphModule::node_args(node).unwrap_or_default();
    args.iter().all(|a| !FxGraphModule::is_node(a))
}

/// Collapse matching Q->DQ pairs in a submodule graph.
/// Rust port of Python `_collapse_control_flow_qdq_pairs`.
fn collapse_control_flow_qdq_pairs<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    quant_count: &mut u32,
) -> PyResult<bool> {
    let mut modified = false;
    let nodes: Vec<Bound<'py, PyAny>> = graph
        .getattr("nodes")?
        .iter()?
        .filter_map(|n| n.ok())
        .collect();

    for node in &nodes {
        if !is_dequantize_node(node) {
            continue;
        }
        let args = FxGraphModule::node_args(node)?;
        if args.is_empty() || !FxGraphModule::is_node(&args[0]) {
            continue;
        }
        let quant_node = &args[0];
        if !is_quantize_node(quant_node) {
            continue;
        }
        let quant_qp = match parse_per_tensor_qparam(py, quant_node)? {
            Some(qp) => qp,
            None => continue,
        };
        let dequant_qp = match parse_per_tensor_qparam(py, node)? {
            Some(qp) => qp,
            None => continue,
        };
        if !same_control_flow_qparam(&quant_qp, &dequant_qp, true) {
            continue;
        }
        let q_args = FxGraphModule::node_args(quant_node)?;
        if q_args.is_empty() || !FxGraphModule::is_node(&q_args[0]) {
            continue;
        }
        let float_source = &q_args[0];

        // Skip if zero-input const feeding a scalar binary
        if is_zero_input_const_source(float_source) {
            // Check if any user of the DQ node is a binary op (simplified check)
            let users = node.getattr("users")?;
            let user_count: usize = users.len().unwrap_or(0);
            if user_count > 0 {
                let mut feeds_binary = false;
                for user in users.iter()? {
                    let user = user?;
                    let target = FxGraphModule::node_target_str(&user)?;
                    if target.contains("add.") || target.contains("sub.")
                        || target.contains("mul.") {
                        feeds_binary = true;
                        break;
                    }
                }
                if feeds_binary {
                    continue;
                }
            }
        }

        // Bypass: replace DQ with float source
        node.call_method1("replace_all_uses_with", (float_source,))?;
        graph.call_method1("erase_node", (node,))?;
        // Erase Q if no remaining users
        let q_users = quant_node.getattr("users")?;
        if q_users.len().unwrap_or(1) == 0 {
            graph.call_method1("erase_node", (quant_node,))?;
        }
        *quant_count += 1;
        modified = true;
    }
    Ok(modified)
}

// ---------------------------------------------------------------------------
// _rewrite_control_flow_quantization_boundaries
//
// Iterates control-flow submodules and rewrites quantize/dequantize nodes:
//   1. Collapse matching Q->DQ pairs (bypass to float source)
//   2. Rewrite standalone quantize nodes into rescale or const materializations
//   3. Rewrite standalone dequantize nodes into int32 lifts or direct edges
//
// The inner sub-functions rely on heavy Python infra (_ControlFlowRescaleEdit,
// _materialize_quantized_constant_node, _same_control_flow_qparam, etc.).
// Following the same pattern as items 4+5 in the cleanup pass, we delegate
// the per-submodule visitor to Python helpers and orchestrate from Rust.
// ---------------------------------------------------------------------------

pub(crate) fn rewrite_control_flow_quantization_boundaries<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    if submodules.is_empty() {
        return Ok(0);
    }

    // Import Python helpers (rewrite_quant and rewrite_dequant still delegate)
    let cf_rewrites = py.import_bound("retab._cf_rewrites")?;
    let rewrite_quant_fn = cf_rewrites.getattr("_rewrite_control_flow_quantize_nodes")?;
    let rewrite_dequant_fn = cf_rewrites.getattr("_rewrite_control_flow_dequantize_nodes")?;
    let cf_ctx_mod = py.import_bound("retab._cf_ctx")?;
    let ctx_cls = cf_ctx_mod.getattr("CfSubmoduleCtx")?;
    let quant_args_mod =
        py.import_bound("executorch.backends.arm._passes.quant_args")?;
    let quant_args_cls = quant_args_mod.getattr("QuantArgs")?;

    let mut total_rewrites: u32 = 0;
    let mut quant_count: u32 = 0;

    // Stats dict for Python sub-functions
    let stats = PyDict::new_bound(py);
    stats.set_item("quant", 0i64)?;
    stats.set_item("dequant", 0i64)?;

    for (_name, submodule, cf_node) in &submodules {
        let ctx = ctx_cls.call_method1("from_submodule", (submodule, cf_node))?;
        let graph = ctx.getattr("graph")?;

        // Step 1: Rust-native Q/DQ collapse
        let m1 = collapse_control_flow_qdq_pairs(py, &graph, &mut quant_count)?;
        // Steps 2-3: Python delegates (rewrite lone Q and DQ nodes)
        let m2: bool = rewrite_quant_fn.call1((&ctx, &stats, &quant_args_cls))?.extract()?;
        let m3: bool = rewrite_dequant_fn.call1((&ctx, &stats))?.extract()?;

        if m1 || m2 || m3 {
            cleanup_graph(&graph)?;
            recompile(submodule)?;
            total_rewrites += 1;
        }
    }

    let py_quant_count: i64 = stats.get_item("quant")?.map(|v| v.extract().unwrap_or(0)).unwrap_or(0);
    quant_count += py_quant_count as u32;
    let dequant_count: i64 = stats.get_item("dequant")?.map(|v| v.extract().unwrap_or(0)).unwrap_or(0);

    if quant_count > 0 {
        log_line(format_args!(
            "[cf-rewrite-quant-boundaries] rewrote {} control-flow quantize node(s)",
            quant_count
        ));
    }
    if dequant_count > 0 {
        log_line(format_args!(
            "[cf-rewrite-quant-boundaries] rewrote {} control-flow dequantize node(s)",
            dequant_count
        ));
    }

    Ok(total_rewrites + quant_count + dequant_count as u32)
}

// ---------------------------------------------------------------------------
// _restore_control_flow_float_compare_inputs
//
// Finds compare ops (eq, ge, gt, lt, le) in submodules where one input is
// quantized (int8/int16/int32) and the other is float. Inserts a
// dequantize_per_tensor node to restore the float input for the compare.
// ---------------------------------------------------------------------------

pub(crate) fn restore_control_flow_float_compare_inputs<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    if submodules.is_empty() {
        return Ok(0);
    }

    let torch = py.import_bound("torch")?;
    let int8 = torch.getattr("int8")?;
    let int16 = torch.getattr("int16")?;
    let int32 = torch.getattr("int32")?;
    let float16 = torch.getattr("float16")?;
    let float32 = torch.getattr("float32")?;
    let float64 = torch.getattr("float64")?;

    let arm_utils =
        py.import_bound("executorch.backends.arm._passes.arm_pass_utils")?;
    let create_node_fn = arm_utils.getattr("create_node")?;
    let exir_ops = py
        .import_bound("executorch.exir.dialects._ops")?
        .getattr("ops")?;
    let dq_op = exir_ops
        .getattr("edge")?
        .getattr("quantized_decomposed")?
        .getattr("dequantize_per_tensor")?
        .getattr("default")?;

    let cf_mod = py.import_bound("retab._control_flow")?;
    let cleanup_graph_fn = cf_mod.getattr("_cleanup_graph")?;

    let compare_markers = &[
        "aten.gt.",
        "aten.ge.",
        "aten.lt.",
        "aten.le.",
        "aten.eq.",
    ];

    let mut restored: u32 = 0;

    for (_name, submodule, _cf_node) in &submodules {
        let graph = submodule.getattr("graph")?;
        let mut updated = false;

        let nodes = submodule_nodes(&submodule)?;

        for node in &nodes {
            if !is_call_function_target(node, compare_markers) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.len() != 2 {
                continue;
            }

            // Try each arg index
            let mut did_restore = false;
            for idx in 0..2usize {
                let arg = &args[idx];
                let other = &args[1 - idx];

                if !FxGraphModule::is_node(arg) || !FxGraphModule::is_node(other) {
                    continue;
                }

                let arg_meta = arg.getattr("meta")?;
                let other_meta = other.getattr("meta")?;
                let arg_val = meta_val(&arg_meta, "val");
                let other_val = meta_val(&other_meta, "val");

                let arg_dtype = arg_val.as_ref().and_then(|v| v.getattr("dtype").ok());
                let other_dtype = other_val.as_ref().and_then(|v| v.getattr("dtype").ok());

                // arg must be int8/int16/int32
                let arg_is_int = match &arg_dtype {
                    Some(dt) => {
                        dt.eq(&int8).unwrap_or(false)
                            || dt.eq(&int16).unwrap_or(false)
                            || dt.eq(&int32).unwrap_or(false)
                    }
                    None => false,
                };

                // other must be float
                let other_is_float = match &other_dtype {
                    Some(dt) => {
                        dt.eq(&float16).unwrap_or(false)
                            || dt.eq(&float32).unwrap_or(false)
                            || dt.eq(&float64).unwrap_or(false)
                    }
                    None => false,
                };

                if !arg_is_int || !other_is_float {
                    continue;
                }

                // Get qparam for the int arg
                let qparam = match extract_effective_node_qparam(py, &arg)? {
                    Some(qp) => qp,
                    None => continue,
                };
                // Check per_channel
                let per_channel: bool = qparam
                    .getattr("per_channel")
                    .and_then(|v| v.extract())
                    .unwrap_or(false);
                if per_channel {
                    continue;
                }

                let scale: f64 = qparam.getattr("scale")?.extract().unwrap_or(1.0);
                let zp: i64 = qparam.getattr("zp")?.extract().unwrap_or(0);
                let qmin: i64 = qparam.getattr("qmin")?.extract().unwrap_or(-128);
                let qmax: i64 = qparam.getattr("qmax")?.extract().unwrap_or(127);
                let qparam_dtype = qparam.getattr("dtype")?;

                // graph.inserting_before(node) -- use context manager
                let inserting_ctx = graph.call_method1("inserting_before", (node,))?;
                inserting_ctx.call_method0("__enter__")?;

                // create_node(graph=ctx.graph, op_target=dq_op, args=(...), from_node=node)
                let dq_args = PyTuple::new_bound(
                    py,
                    &[
                        arg.clone(),
                        scale.into_py(py).into_bound(py),
                        zp.into_py(py).into_bound(py),
                        qmin.into_py(py).into_bound(py),
                        qmax.into_py(py).into_bound(py),
                        qparam_dtype.clone(),
                    ],
                );
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("graph", &graph)?;
                kwargs.set_item("op_target", &dq_op)?;
                kwargs.set_item("args", &dq_args)?;
                kwargs.set_item("from_node", node)?;
                let dq = create_node_fn.call((), Some(&kwargs))?;

                // Exit context manager
                let none = py.None().into_bound(py);
                inserting_ctx.call_method1("__exit__", (&none, &none, &none))?;

                // Set meta on the dq node
                let dq_meta = dq.getattr("meta")?;
                if let Some(ref av) = arg_val {
                    if av.hasattr("to").unwrap_or(false) {
                        if let Some(ref odt) = other_dtype {
                            let to_kwargs = PyDict::new_bound(py);
                            to_kwargs.set_item("dtype", odt)?;
                            let new_val = av.call_method("to", (), Some(&to_kwargs))?;
                            dq_meta.set_item("val", &new_val)?;
                        }
                    }
                }
                // input_qparams = {0: qparam}
                let iqp = PyDict::new_bound(py);
                iqp.set_item(0i64, &qparam)?;
                dq_meta.set_item("input_qparams", &iqp)?;
                // output_qparams = {}
                let oqp = PyDict::new_bound(py);
                dq_meta.set_item("output_qparams", &oqp)?;

                // node.update_arg(idx, dq)
                node.call_method1("update_arg", (idx, &dq))?;
                restored += 1;
                updated = true;
                did_restore = true;
                break;
            }

            if did_restore {
                // Only break the inner arg loop, continue to next node
            }
        }

        if updated {
            cleanup_graph_fn.call1((&graph,))?;
            recompile(&submodule)?;
        }
    }

    if restored > 0 {
        log_line(format_args!(
            "[cf-restore-float-compare] restored {} control-flow float-compare dequantize input(s)",
            restored
        ));
    }

    Ok(restored)
}

// ---------------------------------------------------------------------------
// _normalize_control_flow_output_layouts
//
// Wrapper that delegates to Python _normalize_control_flow_output_contracts
// with normalize_outputs=True.
// ---------------------------------------------------------------------------

pub(crate) fn normalize_control_flow_output_layouts<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let submodules = get_cf_submodules(py, gm)?;
    if submodules.is_empty() {
        return Ok(0);
    }
    drop(submodules);

    let cf_rewrites = py.import_bound("retab._cf_rewrites")?;
    let func = cf_rewrites.getattr("_normalize_control_flow_output_contracts")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("normalize_outputs", true)?;
    func.call((gm,), Some(&kwargs))?;
    // Python side does its own logging; return 1 to signal "ran".
    Ok(1)
}

// ---------------------------------------------------------------------------
// Combined Cleanup Pass
// ---------------------------------------------------------------------------

// ===========================================================================
// Additional control-flow functions (delegate to Python via PyO3)
// ===========================================================================

// ---------------------------------------------------------------------------
// _reapply_control_flow_table_ops
//
// Re-applies TABLE lowering in control-flow submodules after earlier passes
// have mutated the graph. Delegates to the Python implementation.
// ---------------------------------------------------------------------------

pub(crate) fn reapply_control_flow_table_ops<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    edge_program: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_module = py.import_bound("retab._control_flow")?;
    cf_module.call_method1("_reapply_control_flow_table_ops", (gm, edge_program))?;
    Ok(1)
}

// ---------------------------------------------------------------------------
// _reapply_control_flow_boundary_rescales
//
// Inserts RESCALE nodes at submodule input/output boundaries.
// Delegates to the Python implementation.
// ---------------------------------------------------------------------------

pub(crate) fn reapply_control_flow_boundary_rescales<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    include_inputs: bool,
    include_outputs: bool,
) -> PyResult<u32> {
    let cf_module = py.import_bound("retab._control_flow")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("include_inputs", include_inputs)?;
    kwargs.set_item("include_outputs", include_outputs)?;
    cf_module.call_method(
        "_reapply_control_flow_boundary_rescales",
        (gm,),
        Some(&kwargs),
    )?;
    Ok(1)
}

// ---------------------------------------------------------------------------
// _restore_control_flow_output_dequantizes (pub(crate) wrapper)
//
// Inserts dequantize nodes before submodule outputs that need them.
// Delegates to the Python implementation.
// ---------------------------------------------------------------------------

pub(crate) fn restore_control_flow_output_dequantizes_pub<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_module = py.import_bound("retab._cf_rewrites")?;
    cf_module.call_method1("_restore_control_flow_output_dequantizes", (gm,))?;
    Ok(1)
}

// ---------------------------------------------------------------------------
// _restore_missing_control_flow_input_rescales (pub(crate) wrapper)
//
// Inserts RESCALE nodes between placeholders and binary ops.
// Delegates to the Python implementation.
// ---------------------------------------------------------------------------

pub(crate) fn restore_missing_control_flow_input_rescales_pub<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_module = py.import_bound("retab._cf_rewrites")?;
    cf_module.call_method1("_restore_missing_control_flow_input_rescales", (gm,))?;
    Ok(1)
}

// ---------------------------------------------------------------------------
// _quantize_constant_only_control_flow_outputs
//
// Inserts quantize ops for constant-only submodule outputs.
// Delegates to the Python implementation.
// ---------------------------------------------------------------------------

pub(crate) fn quantize_constant_only_control_flow_outputs<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_module = py.import_bound("retab._control_flow")?;
    cf_module.call_method1("_quantize_constant_only_control_flow_outputs", (gm,))?;
    Ok(1)
}

// ===========================================================================
// Combined Cleanup Pass
// ===========================================================================

pub struct FxControlFlowCleanupPass;

impl FxPass for FxControlFlowCleanupPass {
    fn name(&self) -> &str {
        "FxControlFlowCleanup"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let gm_ref = gm.gm_ref();

        // Check if there are any cond/while_loop nodes at all.
        let submodules = get_cf_submodules(py, gm_ref)?;
        if submodules.is_empty() {
            return Ok(false);
        }
        drop(submodules);

        // 1. Remove no-op passthroughs
        let c1 = remove_control_flow_noop_passthroughs(py, gm_ref)?;
        if c1 > 0 {
            log_line(format_args!(
                "[FxControlFlowCleanup] removed {} no-op passthrough node(s)",
                c1
            ));
        }

        // 2. Prune duplicate terminal rescales
        let c2 = prune_duplicate_terminal_rescales(py, gm_ref)?;
        if c2 > 0 {
            log_line(format_args!(
                "[FxControlFlowCleanup] pruned {} duplicate terminal rescale node(s)",
                c2
            ));
        }

        // 3. Restore missing rescale qparams
        let c3 = restore_missing_control_flow_rescale_qparams(py, gm_ref)?;
        if c3 > 0 {
            log_line(format_args!(
                "[FxControlFlowCleanup] restored qparams on {} rescale node(s)",
                c3
            ));
        }

        // 4. Restore output dequantizes (delegates to Python)
        restore_control_flow_output_dequantizes(py, gm_ref)?;

        // 5. Restore missing input rescales (delegates to Python)
        restore_missing_control_flow_input_rescales(py, gm_ref)?;

        // 6. Refresh TOSA transpose fake tensors
        let c6 = refresh_tosa_transpose_fake_tensors(py, gm_ref)?;
        if c6 > 0 {
            log_line(format_args!(
                "[FxControlFlowCleanup] refreshed {} TOSA transpose fake tensor(s)",
                c6
            ));
        }

        // 7. Refresh TOSA dim_order fake tensors (reuse existing function)
        refresh_tosa_dim_order_fake_tensors(py, gm_ref)?;

        // Recompile the parent graph module -- always recompile since
        // steps 4+5 delegate to Python and may have modified the graph.
        gm.recompile()?;

        Ok(true)
    }
}

// ===========================================================================
// Combined Output Normalize Phase
// ===========================================================================

/// Runs the full output normalize phase in a single Rust call.
/// Replaces 11 individual Python->Rust round-trips with one call.
pub(crate) fn run_output_normalize_phase<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // Steps 2 (restore_missing_rescale_qparams) and 5 (quantize_constant_only_outputs)
    // confirmed dead with correct qparam propagation.
    normalize_control_flow_output_layouts(py, gm)?;
    reapply_control_flow_boundary_rescales(py, gm, false, true)?;
    rewrite_control_flow_quantization_boundaries(py, gm)?;
    remove_control_flow_noop_passthroughs(py, gm)?;
    {
        let gm_wrapper = crate::fx_graph::FxGraphModule::new(gm);
        crate::fx_pass::FxPass::run(&FxControlFlowMetadataPass, &gm_wrapper)?;
    }
    refresh_tosa_transpose_fake_tensors(py, gm)?;
    refresh_tosa_dim_order_fake_tensors(py, gm)?;
    prune_duplicate_terminal_rescales(py, gm)?;
    restore_control_flow_float_compare_inputs(py, gm)?;
    Ok(())
}

// ===========================================================================
// Combined Boundary Phase
// ===========================================================================

/// Runs the post-Rust control-flow boundary phase in a single Rust call.
/// Includes boundary rescale repair and the 813-line binary input rescale refresh.
pub(crate) fn run_boundary_phase<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    _tosa_spec: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // 1. Rewrite quantization boundaries
    rewrite_control_flow_quantization_boundaries(py, gm)?;
    // 2. Reapply boundary rescales (inputs only)
    reapply_control_flow_boundary_rescales(py, gm, true, false)?;
    // DEAD with correct propagation: InsertRescaleInt32Pass, restore_missing_input_rescales
    // 3. Stamp control-flow metadata
    {
        let gm_wrapper = crate::fx_graph::FxGraphModule::new(gm);
        crate::fx_pass::FxPass::run(&FxControlFlowMetadataPass, &gm_wrapper)?;
    }
    // 4. Refresh binary input rescales (still needed for while_loop captured buffers)
    {
        let cf_mod = py.import_bound("retab._control_flow")?;
        cf_mod.call_method1("_refresh_control_flow_binary_input_rescales", (gm,))?;
    }
    Ok(())
}

// ===========================================================================
// Minimal Output Phase (for correct-propagation pipeline)
// ===========================================================================

/// Minimal output phase for the fixed pipeline where qparams are propagated
/// before submodule passes. Boundary cleanup, layout normalization, and metadata.
pub(crate) fn run_minimal_output_phase<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // 1. Rewrite Q/DQ boundaries (collapse pairs, rewrite lone Q/DQ)
    rewrite_control_flow_quantization_boundaries(py, gm)?;
    // 2. Reapply boundary rescales (both inputs and outputs)
    reapply_control_flow_boundary_rescales(py, gm, true, true)?;
    // 3. Restore missing rescale qparams
    restore_missing_control_flow_rescale_qparams(py, gm)?;
    // 4. Quantize constant-only outputs
    quantize_constant_only_control_flow_outputs(py, gm)?;
    // 5. Normalize output layouts (layout bypass — still needed)
    normalize_control_flow_output_layouts(py, gm)?;
    // 6. Remove noop passthroughs
    remove_control_flow_noop_passthroughs(py, gm)?;
    // 7. Re-stamp metadata after mutations
    {
        let gm_wrapper = crate::fx_graph::FxGraphModule::new(gm);
        crate::fx_pass::FxPass::run(&FxControlFlowMetadataPass, &gm_wrapper)?;
    }
    // 8. Refresh TOSA transpose fake tensors
    refresh_tosa_transpose_fake_tensors(py, gm)?;
    // 9. Refresh TOSA dim_order fake tensors
    refresh_tosa_dim_order_fake_tensors(py, gm)?;
    // 10. Prune duplicate terminal rescales
    prune_duplicate_terminal_rescales(py, gm)?;
    // 11. Restore float compare inputs
    restore_control_flow_float_compare_inputs(py, gm)?;
    Ok(())
}
