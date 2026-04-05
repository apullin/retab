//! FX control-flow metadata pass.
//!
//! Rust port of the 7 simplest metadata-stamping functions from
//! `python/retab/_control_flow.py` and `python/retab/_cf_ctx.py`.
//! None of these perform graph surgery -- they only read/write `node.meta`.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

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

fn regularize_control_flow_submodules<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
) -> PyResult<u32> {
    let cf_ctx = py.import_bound("retab._cf_ctx")?;
    let get_matching_fake_tensor = cf_ctx.getattr("_get_matching_fake_tensor")?;

    let submodules = get_cf_submodules(py, gm)?;
    let mut regularized: u32 = 0;

    for (_name, submodule, cf_node) in &submodules {
        // Mark placeholders
        let placeholders = submodule_placeholders(submodule)?;
        for ph in &placeholders {
            let meta = ph.getattr("meta")?;
            meta.set_item("is_input", true)?;
            meta.set_item("arm_control_flow_placeholder", true)?;
        }

        // Determine parent args based on target
        let target_str = cf_node
            .getattr("target")
            .and_then(|t| t.str().map(|s| s.to_string()))
            .unwrap_or_default();

        let all_args = FxGraphModule::node_args(cf_node)?;
        let parent_args: Vec<Bound<'py, PyAny>> = if target_str.contains("cond") {
            // cond: last arg is the list of operands
            if let Some(last) = all_args.last() {
                let mut v = Vec::new();
                if let Ok(iter) = last.iter() {
                    for item in iter {
                        v.push(item?);
                    }
                }
                v
            } else {
                Vec::new()
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
                v
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Sync placeholder fake tensors with parent args
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

            // Propagate qparams
            for qp_key in &["input_qparams", "output_qparams"] {
                if let Some(parent_qp) = meta_val(&parent_meta, qp_key) {
                    let copied = parent_qp.call_method0("copy").unwrap_or(parent_qp.clone());
                    meta.set_item(*qp_key, &copied)?;
                }
            }
        }

        regularized += 1;
    }

    Ok(regularized)
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

fn refresh_tosa_dim_order_fake_tensors<'py>(
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
