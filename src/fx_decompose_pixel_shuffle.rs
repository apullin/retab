use pyo3::prelude::*;

use crate::fx_decompose_ops::FxDecomposer;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

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

        let mut candidates: Vec<(Bound<'_, PyAny>, PixelKind)> = Vec::new();

        for (node, target) in gm.call_functions()? {
            if is_pixel_unshuffle(&target) {
                candidates.push((node, PixelKind::Unshuffle));
            } else if is_pixel_shuffle(&target) {
                candidates.push((node, PixelKind::Shuffle));
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
    gm: &'py FxGraphModule<'py>,
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
            name, r, input_shape, n, c, h * r, w * r
        );
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    let view1 = d.view(x, &[n, c, r, r, h, w])?;
    let permuted = d.permute(&view1, &[0, 1, 4, 2, 5, 3])?;
    let result = d.view(&permuted, &[n, c, h * r, w * r])?;

    d.replace(&result)?;
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
    gm: &'py FxGraphModule<'py>,
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
            name, r, input_shape, n, c * r * r, h_out, w_out
        );
    }

    let d = FxDecomposer::new(py, graph, gm, node);

    let view1 = d.view(x, &[n, c, h_out, r, w_out, r])?;
    let permuted = d.permute(&view1, &[0, 1, 3, 5, 2, 4])?;
    let result = d.view(&permuted, &[n, c * r * r, h_out, w_out])?;

    d.replace(&result)?;
    Ok(())
}
