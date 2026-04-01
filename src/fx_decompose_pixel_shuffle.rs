use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;
use crate::fx_utils;

/// FX-graph-direct decomposition of `pixel_shuffle` and `pixel_unshuffle`
/// into reshape + permute + reshape chains.
///
/// pixel_shuffle(x, r):
///   x: (N, C*r*r, H, W) -> view(N, C, r, r, H, W) -> permute(0,1,4,2,5,3) -> view(N, C, H*r, W*r)
///
/// pixel_unshuffle(x, r):
///   x: (N, C, H, W) -> view(N, C, H//r, r, W//r, r) -> permute(0,1,3,5,2,4) -> view(N, C*r*r, H//r, W//r)
pub struct FxDecomposePixelShufflePass;

fn is_pixel_shuffle(target: &str) -> bool {
    target.contains("pixel_shuffle") && !target.contains("pixel_unshuffle")
}

fn is_pixel_unshuffle(target: &str) -> bool {
    target.contains("pixel_unshuffle")
}

enum PixelKind {
    Shuffle,
    Unshuffle,
}

impl FxPass for FxDecomposePixelShufflePass {
    fn name(&self) -> &str {
        "FxDecomposePixelShuffle"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = FxGraphModule::debug();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<(Bound<'_, PyAny>, PixelKind)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_pixel_unshuffle(&target) {
                candidates.push((node.clone(), PixelKind::Unshuffle));
            } else if is_pixel_shuffle(&target) {
                candidates.push((node.clone(), PixelKind::Shuffle));
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for (node, kind) in &candidates {
            match kind {
                PixelKind::Shuffle => {
                    decompose_pixel_shuffle(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                PixelKind::Unshuffle => {
                    decompose_pixel_unshuffle(py, &graph, gm, node, debug)?;
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

/// pixel_shuffle(x, r):
///   x: (N, C*r*r, H, W)
///   -> view(N, C, r, r, H, W)
///   -> permute(0, 1, 4, 2, 5, 3)
///   -> view(N, C, H*r, W*r)
fn decompose_pixel_shuffle<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 2 {
        return Ok(());
    }
    let x = &args[0];
    let r: i64 = args[1].extract()?;

    let input_shape = FxGraphModule::node_shape(x)?;
    if input_shape.len() != 4 {
        return Ok(());
    }
    let n = input_shape[0];
    let c_rr = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let c = c_rr / (r * r);

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposePixelShuffle] {} r={} shape={:?} -> ({},{},{},{})",
            name,
            r,
            input_shape,
            n,
            c,
            h * r,
            w * r
        );
    }

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let permute_op = resolve_aten_op(py, "permute_copy.default")?;

    // view(x, (N, C, r, r, H, W))
    let shape1 = pyo3::types::PyList::new_bound(py, [n, c, r, r, h, w]);
    let view1_args = PyTuple::new_bound(py, &[x.as_ref(), shape1.as_ref()]);
    let view1 = insert_call_before(py, graph, node, &view_op, view1_args.as_ref(), node)?;

    // permute(view1, (0, 1, 4, 2, 5, 3))
    let perm = pyo3::types::PyList::new_bound(py, [0i64, 1, 4, 2, 5, 3]);
    let perm_args = PyTuple::new_bound(py, &[view1.as_ref(), perm.as_ref()]);
    let permuted = insert_call_before(py, graph, node, &permute_op, perm_args.as_ref(), node)?;

    // view(permuted, (N, C, H*r, W*r))
    let shape2 = pyo3::types::PyList::new_bound(py, [n, c, h * r, w * r]);
    let view2_args = PyTuple::new_bound(py, &[permuted.as_ref(), shape2.as_ref()]);
    let result = insert_call_before(py, graph, node, &view_op, view2_args.as_ref(), node)?;

    // Copy original node's meta onto the final result
    fx_utils::copy_meta(py, node, &result)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// pixel_unshuffle(x, r):
///   x: (N, C, H, W) where H,W divisible by r
///   -> view(N, C, H//r, r, W//r, r)
///   -> permute(0, 1, 3, 5, 2, 4)
///   -> view(N, C*r*r, H//r, W//r)
fn decompose_pixel_unshuffle<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() < 2 {
        return Ok(());
    }
    let x = &args[0];
    let r: i64 = args[1].extract()?;

    let input_shape = FxGraphModule::node_shape(x)?;
    if input_shape.len() != 4 {
        return Ok(());
    }
    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let h_out = h / r;
    let w_out = w / r;

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposePixelUnshuffle] {} r={} shape={:?} -> ({},{},{},{})",
            name,
            r,
            input_shape,
            n,
            c * r * r,
            h_out,
            w_out
        );
    }

    let view_op = resolve_aten_op(py, "view_copy.default")?;
    let permute_op = resolve_aten_op(py, "permute_copy.default")?;

    // view(x, (N, C, H//r, r, W//r, r))
    let shape1 = pyo3::types::PyList::new_bound(py, [n, c, h_out, r, w_out, r]);
    let view1_args = PyTuple::new_bound(py, &[x.as_ref(), shape1.as_ref()]);
    let view1 = insert_call_before(py, graph, node, &view_op, view1_args.as_ref(), node)?;

    // permute(view1, (0, 1, 3, 5, 2, 4))
    let perm = pyo3::types::PyList::new_bound(py, [0i64, 1, 3, 5, 2, 4]);
    let perm_args = PyTuple::new_bound(py, &[view1.as_ref(), perm.as_ref()]);
    let permuted = insert_call_before(py, graph, node, &permute_op, perm_args.as_ref(), node)?;

    // view(permuted, (N, C*r*r, H//r, W//r))
    let shape2 = pyo3::types::PyList::new_bound(py, [n, c * r * r, h_out, w_out]);
    let view2_args = PyTuple::new_bound(py, &[permuted.as_ref(), shape2.as_ref()]);
    let result = insert_call_before(py, graph, node, &view_op, view2_args.as_ref(), node)?;

    // Copy original node's meta onto the final result
    fx_utils::copy_meta(py, node, &result)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}
