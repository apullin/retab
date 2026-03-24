use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of `aten.mean.dim` — mirrors the Python
/// `DecomposeMeanDimPass` exactly:
///
///   h, w dims  → avg_pool2d (kernel = reduced dim size, stride = 1)
///   n, c dims  → sum(keepdim=True) + mul(1/N)
///   final view → reshape to target output shape if keepdim=False
///
/// For rank < 4, the input is padded to 4D first (prepend 1s).
pub struct FxDecomposeMeanPass;

fn is_mean_dim(target: &str) -> bool {
    target.contains("mean.dim") || target.contains("mean.default")
}

impl FxPass for FxDecomposeMeanPass {
    fn name(&self) -> &str {
        "FxDecomposeMean"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            if is_mean_dim(&target) {
                candidates.push(node.clone());
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_mean(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_mean<'py>(
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

    let input_shape = FxGraphModule::node_shape(x).unwrap_or_default();
    let input_rank = input_shape.len() as i64;
    let output_shape = FxGraphModule::node_shape(node).unwrap_or_else(|_| vec![1]);

    // Extract dims
    let dims: Vec<i64> = if args.len() > 1 {
        let dims_obj = &args[1];
        if dims_obj.is_none() {
            (0..input_rank).collect()
        } else {
            match dims_obj.extract::<Vec<i64>>() {
                Ok(d) => d,
                Err(_) => match dims_obj.extract::<i64>() {
                    Ok(d) => vec![d],
                    Err(_) => (0..input_rank).collect(),
                },
            }
        }
    } else {
        (0..input_rank).collect()
    };

    // Normalize negative dims
    let mut dims: Vec<i64> = dims
        .into_iter()
        .map(|d| if d < 0 { d + input_rank } else { d })
        .collect();

    // Filter out dims where input_shape[dim] == 1 (matches Python)
    dims.retain(|&d| input_shape.get(d as usize).copied().unwrap_or(1) != 1);

    if dims.is_empty() {
        FxGraphModule::replace_all_uses_with(node, x)?;
        gm.erase_node(node)?;
        return Ok(());
    }

    let keepdim: bool = if args.len() > 2 {
        args[2].extract().unwrap_or(false)
    } else {
        false
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeMean] {} dims={:?} keepdim={} input_shape={:?} output_shape={:?}",
            name, dims, keepdim, input_shape, output_shape
        );
    }

    let mut current = x.clone();
    let mut current_shape = input_shape.clone();

    // ---- Step 1: Pad to 4D if needed (prepend 1s) ----
    let pad_count = if current_shape.len() < 4 {
        let pad = 4 - current_shape.len();
        let mut new_shape: Vec<i64> = vec![1; pad];
        new_shape.extend_from_slice(&current_shape);
        // Adjust dims
        dims = dims.iter().map(|&d| d + pad as i64).collect();

        let view_op = resolve_aten_op(py, "view_copy.default")?;
        let shape_list = pyo3::types::PyList::new_bound(py, new_shape.iter().copied());
        let view_args = PyTuple::new_bound(py, &[current.as_ref(), shape_list.as_ref()]);
        current = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), node)?;
        current_shape = new_shape;
        pad
    } else {
        0
    };

    // ---- Step 2: Reduce spatial dims (>= 2) with avg_pool2d ----
    let spatial_dims: Vec<i64> = dims.iter().filter(|&&d| d >= 2).copied().collect();
    let nonspatial_dims: Vec<i64> = dims.iter().filter(|&&d| d < 2).copied().collect();

    if !spatial_dims.is_empty() {
        let h_size = current_shape.get(2).copied().unwrap_or(1);
        let w_size = current_shape.get(3).copied().unwrap_or(1);

        // Determine kernel size based on which spatial dims to reduce
        let (kh, kw) = if spatial_dims.contains(&2) && spatial_dims.contains(&3) {
            (h_size, w_size)
        } else if spatial_dims.contains(&3) {
            (1, w_size)
        } else if spatial_dims.contains(&2) {
            (h_size, 1)
        } else {
            (1, 1) // shouldn't happen
        };

        if debug {
            eprintln!(
                "[DecomposeMean] avg_pool2d kernel=[{}, {}] on shape {:?}",
                kh, kw, current_shape
            );
        }

        let avgpool_op = resolve_aten_op(py, "avg_pool2d.default")?;
        let kernel = pyo3::types::PyList::new_bound(py, [kh, kw].iter().copied());
        let stride = pyo3::types::PyList::new_bound(py, [1i64, 1].iter().copied());
        let pool_args =
            PyTuple::new_bound(py, &[current.as_ref(), kernel.as_ref(), stride.as_ref()]);
        current = insert_call_before(py, graph, node, &avgpool_op, pool_args.as_ref(), node)?;

        // Update current_shape after avg_pool
        // avg_pool2d with stride=1 reduces each kernel dim to 1
        let mut new_shape = current_shape.clone();
        if spatial_dims.contains(&2) {
            new_shape[2] = 1;
        }
        if spatial_dims.contains(&3) {
            new_shape[3] = 1;
        }
        current_shape = new_shape;
    }

    // ---- Step 3: Reduce non-spatial dims (< 2) with sum + mul(1/N) ----
    if !nonspatial_dims.is_empty() {
        let count: f64 = nonspatial_dims
            .iter()
            .map(|&d| current_shape.get(d as usize).copied().unwrap_or(1) as f64)
            .product();

        if count > 0.0 && count != 1.0 {
            let sum_op = resolve_aten_op(py, "sum.dim_IntList")?;
            let mul_op = resolve_aten_op(py, "mul.Tensor")?;
            let full_op = resolve_aten_op(py, "full.default")?;

            // sum(x, nonspatial_dims, keepdim=True) — always keepdim=True here
            let dims_py =
                pyo3::types::PyList::new_bound(py, nonspatial_dims.iter().copied());
            let keepdim_true = true.into_py(py).into_bound(py);
            let sum_args = PyTuple::new_bound(
                py,
                &[current.as_ref(), dims_py.as_ref(), keepdim_true.as_ref()],
            );
            current =
                insert_call_before(py, graph, node, &sum_op, sum_args.as_ref(), node)?;

            // full([1]*rank, 1/count)
            let rank = current_shape.len();
            let ones_shape =
                PyTuple::new_bound(py, vec![1i64; rank].iter().copied());
            let builtins = py.import_bound("builtins")?;
            let inv_count = builtins.call_method1("float", (1.0 / count,))?;
            let full_args =
                PyTuple::new_bound(py, &[ones_shape.as_ref(), inv_count.as_ref()]);
            let count_node =
                insert_call_before(py, graph, node, &full_op, full_args.as_ref(), node)?;

            // mul(sum, 1/count)
            let mul_args =
                PyTuple::new_bound(py, &[current.as_ref(), count_node.as_ref()]);
            current =
                insert_call_before(py, graph, node, &mul_op, mul_args.as_ref(), node)?;
        }
    }

    // ---- Step 4: Reshape to target output shape ----
    // The intermediates are kept at 4D (keepdim=True for sum, avg_pool keeps rank).
    // Now reshape to the actual target output shape.
    let current_final_shape = FxGraphModule::node_shape(&current).unwrap_or_default();
    if current_final_shape != output_shape {
        let view_op = resolve_aten_op(py, "view_copy.default")?;
        let shape_list = pyo3::types::PyList::new_bound(py, output_shape.iter().copied());
        let view_args = PyTuple::new_bound(py, &[current.as_ref(), shape_list.as_ref()]);
        current = insert_call_before(py, graph, node, &view_op, view_args.as_ref(), node)?;
    }

    // Copy original node's meta onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    current.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &current)?;
    gm.erase_node(node)?;
    Ok(())
}
