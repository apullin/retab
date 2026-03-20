use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of Arm's Conv1dUnsqueezePass.
///
/// TOSA has no native 1D convolution. The reference Arm pass rewrites conv1d
/// into:
///   view_copy(input[..., W] -> [..., W, 1])
///   view_copy(weight[..., KW] -> [..., KW, 1])
///   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
///   view_copy(output[..., W_out, 1] -> [..., W_out])
///
/// This pass mirrors that shape legalization before `FxRewriteConvPass`.
pub struct FxConv1dUnsqueezePass;

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("conv1d.default")
}

impl FxPass for FxConv1dUnsqueezePass {
    fn name(&self) -> &str {
        "FxConv1dUnsqueeze"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let builtins = py.import_bound("builtins")?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<Bound<'_, PyAny>> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }

            let target = FxGraphModule::node_target_str(node)?;
            if !is_convolution(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let Some(stride) = args.get(3) else {
                continue;
            };
            let stride_len: usize = builtins
                .call_method1("len", (stride,))?
                .extract()
                .unwrap_or(0);
            if stride_len == 1 {
                candidates.push(node.clone());
            }
        }

        for node in &candidates {
            helpers.call_method1("unsqueeze_conv1d_node", (gm.gm_ref(), node))?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
