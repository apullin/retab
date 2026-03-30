use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of ops that TOSA cannot represent directly.
///
/// Each op is decomposed into primitive ATen ops via PyO3 calls to the FX graph
/// API. No Python helper functions are used for graph manipulation -- all node
/// creation goes through `graph.call_function`.
///
/// Supported decompositions:
///   - atanh(x)       -> 0.5 * log((1+x)/(1-x))
///   - hardsigmoid(x) -> clamp((x + 3) / 6, 0, 1)
///   - stack          -> unsqueeze each + cat
///   - unbind         -> slice + squeeze for each element
///
/// Note: `masked_fill` is handled separately by `FxDecomposeMaskedFillPass`
/// (runs after fold_qdq so the fill value can be quantized correctly).
pub struct FxDecomposeOpsPass;

fn is_atanh(target: &str) -> bool {
    target.contains("atanh.default")
}

fn is_hardsigmoid(target: &str) -> bool {
    target.contains("hardsigmoid.default")
}

fn is_stack(target: &str) -> bool {
    target.contains("stack.default")
}

fn is_unbind(target: &str) -> bool {
    target.contains("unbind.int") || target.contains("unbind_copy.int")
}

fn is_dequantize(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_quantize(target: &str) -> bool {
    target.contains("quantize_per_tensor") || target.contains("quantize_per_channel")
}

fn has_quantized_envelope(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    let Some(input) = args.first() else {
        return Ok(false);
    };
    if !FxGraphModule::is_node(input) {
        return Ok(false);
    }
    let input_target = FxGraphModule::node_target_str(input)?;
    if !is_dequantize(&input_target) {
        return Ok(false);
    }

    let users = FxGraphModule::node_users(node)?;
    Ok(users.iter().any(|user| {
        FxGraphModule::node_target_str(user)
            .map(|target| is_quantize(&target))
            .unwrap_or(false)
    }))
}

enum DecompKind {
    Atanh,
    Hardsigmoid,
    Stack,
    Unbind,
}

/// Resolve an ATen op target, trying exir_ops.edge.aten first, then torch.ops.aten.
pub fn resolve_aten_op<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let parts: Vec<&str> = name.split('.').collect();
    let op_name = parts[0];
    let overload = if parts.len() > 1 { parts[1] } else { "default" };

    let result = py
        .import_bound("executorch.exir.dialects._ops")
        .and_then(|m| m.getattr("ops"))
        .and_then(|ops| ops.getattr("edge"))
        .and_then(|edge| edge.getattr("aten"))
        .and_then(|aten| aten.getattr(op_name))
        .and_then(|op| op.getattr(overload));

    match result {
        Ok(op) => Ok(op),
        Err(_) => py
            .import_bound("torch")?
            .getattr("ops")?
            .getattr("aten")?
            .getattr(op_name)?
            .getattr(overload),
    }
}

/// Convert a Rust f64 into a Bound Python float.
fn py_float<'py>(py: Python<'py>, v: f64) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import_bound("builtins")?;
    builtins.call_method1("float", (v,))
}

/// Convert a Rust i64 into a Bound Python int.
fn py_int<'py>(py: Python<'py>, v: i64) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import_bound("builtins")?;
    builtins.call_method1("int", (v,))
}

/// Helper: insert a call_function node before `ref_node` using graph.inserting_before.
/// Copies meta from `meta_src` and refreshes the FakeTensor if possible.
pub fn insert_call_before<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    ref_node: &Bound<'py, PyAny>,
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyAny>,
    meta_src: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inserting = graph.call_method1("inserting_before", (ref_node,))?;
    inserting.call_method0("__enter__")?;
    let new_node = graph.call_method1("call_function", (target, args))?;
    // Copy meta from source node, then clear stale qparams
    let copy_mod = py.import_bound("copy")?;
    let meta_copy = copy_mod.call_method1("copy", (meta_src.getattr("meta")?,))?;
    // Clear qparams to avoid stale references on intermediate nodes
    let _ = meta_copy.call_method1("__setitem__", ("input_qparams", PyDict::new_bound(py)));
    let _ = meta_copy.call_method1("__setitem__", ("output_qparams", PyDict::new_bound(py)));
    new_node.setattr("meta", &meta_copy)?;
    // Try to refresh FakeTensor metadata
    let _ = refresh_fake_tensor(py, &new_node);
    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;
    Ok(new_node)
}

/// Propagate qparams from a source node to a target node.
/// Sets input_qparams and output_qparams on the target's meta.
/// For intermediate nodes in a decomposition, use the source's qparams.
pub fn propagate_qparams<'py>(src: &Bound<'py, PyAny>, dst: &Bound<'py, PyAny>) -> PyResult<()> {
    let src_meta = src.getattr("meta")?;
    let dst_meta = dst.getattr("meta")?;

    // Copy input_qparams if available
    if let Ok(iqp) = src_meta.call_method1("get", ("input_qparams",)) {
        if !iqp.is_none() {
            dst_meta.set_item("input_qparams", &iqp)?;
        }
    }
    // Copy output_qparams if available
    if let Ok(oqp) = src_meta.call_method1("get", ("output_qparams",)) {
        if !oqp.is_none() {
            dst_meta.set_item("output_qparams", &oqp)?;
        }
    }
    Ok(())
}

/// Try to refresh a node's meta["val"] from its target and args.
fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    helpers.call_method1("refresh_node_fake_tensor", (node,))?;
    Ok(())
}

impl FxPass for FxDecomposeOpsPass {
    fn name(&self) -> &str {
        "FxDecomposeOps"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<(Bound<'_, PyAny>, DecompKind)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_atanh(&target) {
                if has_quantized_envelope(node)? {
                    continue;
                }
                candidates.push((node.clone(), DecompKind::Atanh));
            } else if is_hardsigmoid(&target) {
                // Skip hardsigmoid decomposition for quantized graphs —
                // InsertTableOpsPass handles it via TABLE lookup instead.
                // The arithmetic decomposition (x+3)/6 causes INT32/FP32
                // dtype mismatches in the quantized pipeline.
            } else if is_stack(&target) {
                candidates.push((node.clone(), DecompKind::Stack));
            } else if is_unbind(&target) {
                candidates.push((node.clone(), DecompKind::Unbind));
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for (node, kind) in &candidates {
            match kind {
                DecompKind::Atanh => {
                    decompose_atanh(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                DecompKind::Hardsigmoid => {
                    decompose_hardsigmoid(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                DecompKind::Stack => {
                    decompose_stack(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                DecompKind::Unbind => {
                    decompose_unbind(py, &graph, gm, node, debug)?;
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

/// atanh(x) = 0.5 * log((1+x) / (1-x))
fn decompose_atanh<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }
    let x = &args[0];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeOps] decomposing atanh: {}", name);
    }

    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let sub_op = resolve_aten_op(py, "sub.Tensor")?;
    let div_op = resolve_aten_op(py, "div.Tensor")?;
    let log_op = resolve_aten_op(py, "log.default")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // Get input shape for scalar constants
    let input_shape = FxGraphModule::node_shape(x).unwrap_or_else(|_| vec![1]);
    let shape_tuple = PyTuple::new_bound(py, input_shape.iter().copied());

    // Create constant 1.0
    let one_val = py_float(py, 1.0)?;
    let one_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), one_val.as_ref()]);
    let one = insert_call_before(py, graph, node, &full_op, one_args.as_ref(), node)?;

    // Create constant 0.5
    let half_val = py_float(py, 0.5)?;
    let half_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), half_val.as_ref()]);
    let half = insert_call_before(py, graph, node, &full_op, half_args.as_ref(), node)?;

    // one_plus_x = add(1, x) = 1 + x
    let one_plus_args = PyTuple::new_bound(py, &[one.as_ref(), x.as_ref()]);
    let one_plus_x = insert_call_before(py, graph, node, &add_op, one_plus_args.as_ref(), node)?;

    // one_minus_x = sub(1, x) = 1 - x
    let one_minus_args = PyTuple::new_bound(py, &[one.as_ref(), x.as_ref()]);
    let one_minus_x = insert_call_before(py, graph, node, &sub_op, one_minus_args.as_ref(), node)?;

    // ratio = div(one_plus_x, one_minus_x)
    let div_args = PyTuple::new_bound(py, &[one_plus_x.as_ref(), one_minus_x.as_ref()]);
    let ratio = insert_call_before(py, graph, node, &div_op, div_args.as_ref(), node)?;

    // log_ratio = log(ratio)
    let log_args = PyTuple::new_bound(py, &[ratio.as_ref()]);
    let log_ratio = insert_call_before(py, graph, node, &log_op, log_args.as_ref(), node)?;

    // result = mul(0.5, log_ratio)
    let mul_args = PyTuple::new_bound(py, &[half.as_ref(), log_ratio.as_ref()]);
    let result = insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?;

    // Copy original node's meta (with qparams) onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)
fn decompose_hardsigmoid<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }
    let x = &args[0];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeOps] decomposing hardsigmoid: {}", name);
    }

    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let div_op = resolve_aten_op(py, "div.Tensor")?;
    let clamp_op = resolve_aten_op(py, "clamp.default")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    let input_shape = FxGraphModule::node_shape(x).unwrap_or_else(|_| vec![1]);
    let shape_tuple = PyTuple::new_bound(py, input_shape.iter().copied());

    // Create constant 3.0
    let three_val = py_float(py, 3.0)?;
    let three_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), three_val.as_ref()]);
    let three = insert_call_before(py, graph, node, &full_op, three_args.as_ref(), node)?;

    // Create constant 6.0
    let six_val = py_float(py, 6.0)?;
    let six_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), six_val.as_ref()]);
    let six = insert_call_before(py, graph, node, &full_op, six_args.as_ref(), node)?;

    // x_plus_3 = add(x, 3)
    let add_args = PyTuple::new_bound(py, &[x.as_ref(), three.as_ref()]);
    let x_plus_3 = insert_call_before(py, graph, node, &add_op, add_args.as_ref(), node)?;

    // divided = div(x_plus_3, 6)
    let div_args = PyTuple::new_bound(py, &[x_plus_3.as_ref(), six.as_ref()]);
    let divided = insert_call_before(py, graph, node, &div_op, div_args.as_ref(), node)?;

    // result = clamp(divided, 0.0, 1.0)
    let zero_val = py_float(py, 0.0)?;
    let one_val = py_float(py, 1.0)?;
    let clamp_args =
        PyTuple::new_bound(py, &[divided.as_ref(), zero_val.as_ref(), one_val.as_ref()]);
    let result = insert_call_before(py, graph, node, &clamp_op, clamp_args.as_ref(), node)?;

    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// stack(tensors, dim) -> unsqueeze each tensor at dim, then cat along dim
fn decompose_stack<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }

    // args[0] is a list/tuple of tensors, args[1] is dim (default 0)
    let tensors_obj = &args[0];
    let dim: i64 = if args.len() > 1 {
        args[1].extract().unwrap_or(0)
    } else {
        0
    };

    // Extract individual tensor nodes from the list
    let tensors: Vec<Bound<'_, PyAny>> = tensors_obj.iter()?.filter_map(|t| t.ok()).collect();
    if tensors.is_empty() {
        return Ok(());
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeOps] decomposing stack: {} ({} tensors, dim={})",
            name,
            tensors.len(),
            dim
        );
    }

    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")
        .or_else(|_| resolve_aten_op(py, "unsqueeze.default"))?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    let dim_py = py_int(py, dim)?;

    // Unsqueeze each tensor at the stack dimension
    let mut unsqueezed: Vec<Bound<'_, PyAny>> = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        let unsqueeze_args = PyTuple::new_bound(py, &[tensor.as_ref(), dim_py.as_ref()]);
        let unsq = insert_call_before(
            py,
            graph,
            node,
            &unsqueeze_op,
            unsqueeze_args.as_ref(),
            node,
        )?;
        unsqueezed.push(unsq);
    }

    // Cat the unsqueezed tensors along dim
    let unsqueezed_list = PyList::new_bound(py, &unsqueezed);
    let cat_args = PyTuple::new_bound(py, &[unsqueezed_list.as_ref(), dim_py.as_ref()]);
    let result = insert_call_before(py, graph, node, &cat_op, cat_args.as_ref(), node)?;

    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// unbind(x, dim) -> for each index: slice(x, dim, i, i+1) + squeeze(dim)
///
/// unbind returns a tuple of tensors. In the FX graph, the unbind node is
/// consumed by getitem nodes. We replace each getitem with the corresponding
/// slice+squeeze chain.
fn decompose_unbind<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }
    let x = &args[0];
    let dim: i64 = if args.len() > 1 {
        args[1].extract().unwrap_or(0)
    } else {
        0
    };

    let input_shape = match FxGraphModule::node_shape(x) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let rank = input_shape.len() as i64;
    let norm_dim = if dim < 0 { dim + rank } else { dim } as usize;

    if debug {
        let name = FxGraphModule::node_name(node)?;
        let num_slices = input_shape.get(norm_dim).copied().unwrap_or(0);
        eprintln!(
            "[DecomposeOps] decomposing unbind: {} (dim={}, slices={})",
            name, dim, num_slices
        );
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")
        .or_else(|_| resolve_aten_op(py, "slice.Tensor"))?;
    let squeeze_op =
        resolve_aten_op(py, "squeeze_copy.dims").or_else(|_| resolve_aten_op(py, "squeeze.dim"))?;

    // Collect getitem users of the unbind node
    let users = FxGraphModule::node_users(node)?;
    let mut getitem_users: Vec<(Bound<'_, PyAny>, i64)> = Vec::new();
    for user in &users {
        let user_op = FxGraphModule::node_op(user)?;
        if user_op == "call_function" {
            let user_target = FxGraphModule::node_target_str(user)?;
            if user_target.contains("getitem") {
                let user_args = FxGraphModule::node_args(user)?;
                if user_args.len() >= 2 {
                    if let Ok(idx) = user_args[1].extract::<i64>() {
                        getitem_users.push((user.clone(), idx));
                    }
                }
            }
        }
    }

    let copy_mod = py.import_bound("copy")?;
    let dim_py = py_int(py, norm_dim as i64)?;

    for (getitem_node, idx) in &getitem_users {
        let idx_py = py_int(py, *idx)?;
        let idx_plus_one_py = py_int(py, idx + 1)?;

        // slice_copy(x, dim, idx, idx+1)
        let slice_args = PyTuple::new_bound(
            py,
            &[
                x.as_ref(),
                dim_py.as_ref(),
                idx_py.as_ref(),
                idx_plus_one_py.as_ref(),
            ],
        );
        let sliced = insert_call_before(
            py,
            graph,
            getitem_node,
            &slice_op,
            slice_args.as_ref(),
            node,
        )?;

        // squeeze_copy(sliced, [dim])
        let dims_list = PyList::new_bound(py, &[norm_dim as i64]);
        let squeeze_args = PyTuple::new_bound(py, &[sliced.as_ref(), dims_list.as_ref()]);
        let squeezed = insert_call_before(
            py,
            graph,
            getitem_node,
            &squeeze_op,
            squeeze_args.as_ref(),
            node,
        )?;

        // Copy getitem's meta to the squeezed node for correct shape
        let gi_meta = copy_mod.call_method1("copy", (getitem_node.getattr("meta")?,))?;
        squeezed.setattr("meta", &gi_meta)?;

        FxGraphModule::replace_all_uses_with(getitem_node, &squeezed)?;
        gm.erase_node(getitem_node)?;
    }

    // If the unbind node has no remaining users, erase it
    let remaining_users = FxGraphModule::node_users(node)?;
    if remaining_users.is_empty() {
        gm.erase_node(node)?;
    }

    Ok(())
}
