use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// Decompose LayerNorm, Softmax, and BatchNorm into simpler primitive ops.
///
/// The heavy lifting (graph insertion, metadata propagation) is done in
/// Python helper functions (`fx_decompose_helpers.py`).  Rust handles
/// pattern detection, iteration, and orchestration.
///
/// Note: quantized models that go through `torch.export` typically already
/// have these ops decomposed into primitives by PyTorch.  This pass handles
/// models that export the composite ops as-is.
pub struct FxDecomposePass;

// ---------------------------------------------------------------------------
// Target-name matching
// ---------------------------------------------------------------------------

fn is_layer_norm(target: &str) -> bool {
    target.contains("layer_norm")
}

fn is_softmax(target: &str) -> bool {
    target.contains("softmax")
}

fn is_batch_norm(target: &str) -> bool {
    target.contains("batch_norm") || target.contains("native_batch_norm")
}

/// Which decomposition to apply.
enum DecomposeKind {
    LayerNorm,
    Softmax,
    BatchNorm,
}

impl DecomposeKind {
    /// Name of the Python helper function in `fx_decompose_helpers`.
    fn helper_name(&self) -> &'static str {
        match self {
            Self::LayerNorm => "decompose_layernorm",
            Self::Softmax => "decompose_softmax",
            Self::BatchNorm => "decompose_batchnorm",
        }
    }
}

impl FxPass for FxDecomposePass {
    fn name(&self) -> &str {
        "FxDecompose"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();

        // Import the Python helper module.
        // Ensure the project root (where fx_decompose_helpers.py lives) is on sys.path.
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        let project_root = std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        // Only add if not already present, to avoid unbounded growth.
        let contains: bool = path.call_method1("__contains__", (&project_root,))?.extract()?;
        if !contains {
            path.call_method1("insert", (0, &project_root))?;
        }
        let helpers = py.import_bound("fx_decompose_helpers")?;

        let mut modified = false;

        // Snapshot nodes up-front; can't mutate the graph while iterating.
        let nodes = gm.nodes()?;
        let mut to_decompose: Vec<(Bound<'_, PyAny>, DecomposeKind)> = Vec::new();

        for node in &nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_layer_norm(&target) {
                to_decompose.push((node.clone(), DecomposeKind::LayerNorm));
            } else if is_softmax(&target) {
                to_decompose.push((node.clone(), DecomposeKind::Softmax));
            } else if is_batch_norm(&target) {
                to_decompose.push((node.clone(), DecomposeKind::BatchNorm));
            }
        }

        // Call the Python helper for each matched node.
        // The helpers derive the Graph from `node.graph` so we only need to
        // pass the node itself — no need to access the private `gm` field.
        for (node, kind) in &to_decompose {
            helpers.call_method1(kind.helper_name(), (node,))?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python; Rust unit tests require a live
    // Python FX graph which is impractical to construct here.
}
