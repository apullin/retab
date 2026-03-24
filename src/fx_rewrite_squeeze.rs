use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct rewrite of squeeze_copy and unsqueeze_copy to view_copy.
///
/// The serializer emits squeeze/unsqueeze as RESHAPE using `node.meta.val.shape`,
/// but that shape can be stale after memory format changes. By explicitly
/// rewriting to `view_copy` with a computed shape, we ensure the RESHAPE
/// output shape is always correct.
///
/// Also rewrites `transpose_copy` to `permute_copy` (full permutation).
pub struct FxRewriteSqueezePass;

fn is_squeeze(target: &str) -> bool {
    // Match squeeze_copy.default, squeeze_copy.dim, squeeze_copy.dims,
    // squeeze.default, squeeze.dim, squeeze.dims — but NOT unsqueeze variants.
    if target.contains("unsqueeze") {
        return false;
    }
    target.contains("squeeze_copy") || target.contains("squeeze.dim") || target.contains("squeeze.default") || target.contains("squeeze.dims")
}

fn is_unsqueeze(target: &str) -> bool {
    target.contains("unsqueeze_copy") || target.contains("unsqueeze.default")
}

fn is_transpose(target: &str) -> bool {
    target.contains("transpose_copy") || target.contains("transpose.int")
}

/// Compute the output shape for squeeze given input shape and dims.
/// Removes size-1 dims at the specified positions.
/// If dims is empty, removes all size-1 dims.
fn compute_squeeze_shape(input_shape: &[i64], dims: &[i64], rank: usize) -> Vec<i64> {
    if dims.is_empty() {
        // Remove all size-1 dims
        return input_shape.iter().copied().filter(|&d| d != 1).collect();
    }

    let mut result = Vec::with_capacity(input_shape.len());
    for (i, &dim_size) in input_shape.iter().enumerate() {
        let mut should_remove = false;
        for &d in dims {
            let normalized = if d < 0 { d + rank as i64 } else { d };
            if normalized == i as i64 && dim_size == 1 {
                should_remove = true;
                break;
            }
        }
        if !should_remove {
            result.push(dim_size);
        }
    }
    result
}

/// Compute the output shape for unsqueeze given input shape and dim.
/// Inserts a size-1 dim at the specified position.
fn compute_unsqueeze_shape(input_shape: &[i64], dim: i64, rank: usize) -> Vec<i64> {
    // For unsqueeze, the rank of the OUTPUT is input_rank + 1
    let out_rank = rank + 1;
    let normalized = if dim < 0 {
        dim + out_rank as i64
    } else {
        dim
    };
    let pos = normalized as usize;

    let mut result = Vec::with_capacity(input_shape.len() + 1);
    for (i, &dim_size) in input_shape.iter().enumerate() {
        if i == pos {
            result.push(1);
        }
        result.push(dim_size);
    }
    // If inserting at the end
    if pos >= input_shape.len() {
        result.push(1);
    }
    result
}

impl FxPass for FxRewriteSqueezePass {
    fn name(&self) -> &str {
        "FxRewriteSqueeze"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        // Collect candidates to avoid mutation during iteration
        enum RewriteKind {
            Squeeze,
            Unsqueeze,
            Transpose,
        }
        let mut candidates: Vec<(Bound<'_, PyAny>, RewriteKind)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_squeeze(&target) {
                candidates.push((node.clone(), RewriteKind::Squeeze));
            } else if is_unsqueeze(&target) {
                candidates.push((node.clone(), RewriteKind::Unsqueeze));
            } else if is_transpose(&target) {
                candidates.push((node.clone(), RewriteKind::Transpose));
            }
        }

        // Resolve view_copy and permute_copy op targets
        let view_copy_op = py.eval_bound(
            "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'edge'), 'aten'), 'view_copy').default",
            None,
            None,
        ).or_else(|_| {
            py.eval_bound(
                "__import__('torch').ops.aten.view_copy.default",
                None,
                None,
            )
        })?;

        let permute_copy_op = py.eval_bound(
            "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'edge'), 'aten'), 'permute_copy').default",
            None,
            None,
        ).or_else(|_| {
            py.eval_bound(
                "__import__('torch').ops.aten.permute_copy.default",
                None,
                None,
            )
        })?;

        let graph = gm.gm_ref().getattr("graph")?;
        let copy_mod = py.import_bound("copy")?;

        for (node, kind) in &candidates {
            let args = FxGraphModule::node_args(node)?;
            if args.is_empty() {
                continue;
            }

            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            let input_shape = match FxGraphModule::node_shape(input_node) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let input_rank = input_shape.len();

            match kind {
                RewriteKind::Squeeze => {
                    // Extract dims: could be args[1] as int or list of ints
                    let dims: Vec<i64> = if args.len() > 1 {
                        // Try as list first, then single int
                        args[1]
                            .extract::<Vec<i64>>()
                            .or_else(|_| args[1].extract::<i64>().map(|d| vec![d]))
                            .unwrap_or_default()
                    } else {
                        Vec::new() // squeeze all size-1 dims
                    };

                    let output_shape = compute_squeeze_shape(&input_shape, &dims, input_rank);

                    if debug {
                        let name = FxGraphModule::node_name(node)?;
                        eprintln!(
                            "[RewriteSqueeze] {} squeeze dims={:?} shape {:?} -> {:?}",
                            name, dims, input_shape, output_shape
                        );
                    }

                    // Create view_copy node via graph.inserting_before context manager
                    let shape_list =
                        pyo3::types::PyList::new_bound(py, output_shape.iter().copied());
                    let inserting =
                        graph.call_method1("inserting_before", (node,))?;
                    inserting.call_method0("__enter__")?;
                    let new_args =
                        pyo3::types::PyTuple::new_bound(py, &[input_node.as_ref(), shape_list.as_ref()]);
                    let new_node = graph.call_method1(
                        "call_function",
                        (&view_copy_op, new_args),
                    )?;
                    // Copy meta from original node
                    let orig_meta = node.getattr("meta")?;
                    let meta_copy = copy_mod.call_method1("copy", (&orig_meta,))?;
                    new_node.setattr("meta", &meta_copy)?;
                    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;

                    FxGraphModule::replace_all_uses_with(node, &new_node)?;
                    gm.erase_node(node)?;
                    modified = true;
                }
                RewriteKind::Unsqueeze => {
                    // Extract dim from args[1]
                    let dim: i64 = if args.len() > 1 {
                        match args[1].extract::<i64>() {
                            Ok(d) => d,
                            Err(_) => continue,
                        }
                    } else {
                        continue;
                    };

                    let output_shape = compute_unsqueeze_shape(&input_shape, dim, input_rank);

                    if debug {
                        let name = FxGraphModule::node_name(node)?;
                        eprintln!(
                            "[RewriteSqueeze] {} unsqueeze dim={} shape {:?} -> {:?}",
                            name, dim, input_shape, output_shape
                        );
                    }

                    let shape_list =
                        pyo3::types::PyList::new_bound(py, output_shape.iter().copied());
                    let inserting =
                        graph.call_method1("inserting_before", (node,))?;
                    inserting.call_method0("__enter__")?;
                    let new_args =
                        pyo3::types::PyTuple::new_bound(py, &[input_node.as_ref(), shape_list.as_ref()]);
                    let new_node = graph.call_method1(
                        "call_function",
                        (&view_copy_op, new_args),
                    )?;
                    let orig_meta = node.getattr("meta")?;
                    let meta_copy = copy_mod.call_method1("copy", (&orig_meta,))?;
                    new_node.setattr("meta", &meta_copy)?;
                    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;

                    FxGraphModule::replace_all_uses_with(node, &new_node)?;
                    gm.erase_node(node)?;
                    modified = true;
                }
                RewriteKind::Transpose => {
                    // transpose_copy(input, dim0, dim1) -> permute_copy(input, perm)
                    if args.len() < 3 {
                        continue;
                    }
                    let dim0: i64 = match args[1].extract::<i64>() {
                        Ok(d) => d,
                        Err(_) => continue,
                    };
                    let dim1: i64 = match args[2].extract::<i64>() {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    // Normalize negative dims
                    let rank = input_rank as i64;
                    let d0 = if dim0 < 0 { dim0 + rank } else { dim0 } as usize;
                    let d1 = if dim1 < 0 { dim1 + rank } else { dim1 } as usize;

                    // Build identity permutation, then swap
                    let mut perm: Vec<i64> = (0..input_rank as i64).collect();
                    perm.swap(d0, d1);

                    if debug {
                        let name = FxGraphModule::node_name(node)?;
                        eprintln!(
                            "[RewriteSqueeze] {} transpose({},{}) -> permute {:?}",
                            name, dim0, dim1, perm
                        );
                    }

                    let perm_list =
                        pyo3::types::PyList::new_bound(py, perm.iter().copied());
                    let inserting =
                        graph.call_method1("inserting_before", (node,))?;
                    inserting.call_method0("__enter__")?;
                    let new_args =
                        pyo3::types::PyTuple::new_bound(py, &[input_node.as_ref(), perm_list.as_ref()]);
                    let new_node = graph.call_method1(
                        "call_function",
                        (&permute_copy_op, new_args),
                    )?;
                    let orig_meta = node.getattr("meta")?;
                    let meta_copy = copy_mod.call_method1("copy", (&orig_meta,))?;
                    new_node.setattr("meta", &meta_copy)?;
                    inserting.call_method("__exit__", (py.None(), py.None(), py.None()), None)?;

                    FxGraphModule::replace_all_uses_with(node, &new_node)?;
                    gm.erase_node(node)?;
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
