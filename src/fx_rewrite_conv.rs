use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of conv rewriting.
///
/// Detects `aten.convolution.default` nodes and rewrites them to TOSA
/// `CONV2D` or `DEPTHWISE_CONV2D` ops via the Python helper. This matches
/// the Python `RewriteConvPass` behavior:
///   - Changes target from aten.convolution to tosa.CONV2D/DEPTHWISE_CONV2D
///   - Rearranges args (pad format, bias handling)
///   - Attaches conv_info metadata with quantization zero-points
pub struct FxRewriteConvPass;

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default")
}

impl FxPass for FxRewriteConvPass {
    fn name(&self) -> &str {
        "FxRewriteConv"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;
        let all_nodes = gm.nodes()?;

        // Collect candidates to avoid mutating during iteration (the helper
        // erases nodes and inserts new ones).
        let mut candidates: Vec<(Bound<'_, PyAny>, bool, i64, i64, i64, i64, i64)> = Vec::new();

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
            // aten.convolution args:
            // [input, weight, bias, stride, padding, dilation, transposed, output_padding, groups]
            if args.len() < 9 {
                continue;
            }

            let input_node = &args[0];
            let weight_node = &args[1];
            let groups: i64 = args[8].extract().unwrap_or(1);

            // Weight shape for conv: [out_channels, in_channels/groups, kH, kW]
            let weight_shape = match FxGraphModule::node_shape(weight_node) {
                Ok(s) => s,
                Err(_) => continue,
            };

            if weight_shape.len() < 4 {
                continue;
            }

            let out_channels = weight_shape[0];

            // Get in_channels from the input tensor shape [N, C, H, W]
            let in_channels = match FxGraphModule::node_shape(input_node) {
                Ok(s) if s.len() >= 2 => s[1],
                _ => 0,
            };

            // Match Python reference: depthwise when in_channels == groups and
            // out_channels is a multiple of in_channels.
            let is_depthwise = in_channels > 0
                && groups == in_channels
                && out_channels % in_channels == 0;

            // Get quantization info from meta if available
            let meta = node.getattr("meta")?;
            let input_qparams = meta.call_method1("get", ("input_qparams",))?;

            let mut input_zp: i64 = 0;
            let mut weight_zp: i64 = 0;

            if !input_qparams.is_none() {
                if let Ok(qp0) = input_qparams.get_item(0_usize) {
                    input_zp = qp0
                        .getattr("zp")
                        .and_then(|v| v.extract())
                        .or_else(|_| {
                            qp0.call_method1("get", ("zero_point",))
                                .and_then(|v| v.extract())
                        })
                        .unwrap_or(0);
                }
                if let Ok(qp1) = input_qparams.get_item(1_usize) {
                    weight_zp = qp1
                        .getattr("zp")
                        .and_then(|v| v.extract())
                        .or_else(|_| {
                            qp1.call_method1("get", ("zero_point",))
                                .and_then(|v| v.extract())
                        })
                        .unwrap_or(0);
                }
            }

            // Store conv metadata for downstream passes (rescale, serialization)
            let conv_meta = pyo3::types::PyDict::new_bound(py);
            conv_meta.set_item("is_depthwise", is_depthwise)?;
            conv_meta.set_item("input_zp", input_zp)?;
            conv_meta.set_item("weight_zp", weight_zp)?;
            conv_meta.set_item("groups", groups)?;
            conv_meta.set_item("in_channels", in_channels)?;
            conv_meta.set_item("out_channels", out_channels)?;
            FxGraphModule::set_meta(node, "conv_info", conv_meta.as_ref())?;

            candidates.push((
                node.clone(),
                is_depthwise,
                input_zp,
                weight_zp,
                in_channels,
                out_channels,
                groups,
            ));
        }

        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();

        for (node, is_depthwise, input_zp, weight_zp, in_channels, out_channels, groups) in
            &candidates
        {
            if debug {
                let node_name = FxGraphModule::node_name(node)?;
                eprintln!(
                    "[RewriteConv] {}: in_ch={}, out_ch={}, groups={}, depthwise={}",
                    node_name, in_channels, out_channels, groups, is_depthwise
                );
            }

            // Delegate the actual target swap and arg rearrangement to Python.
            // This avoids needing to resolve TOSA op overloads from Rust.
            helpers.call_method1(
                "rewrite_conv_to_tosa",
                (
                    gm.gm_ref(),
                    node,
                    *is_depthwise,
                    *input_zp,
                    *weight_zp,
                    *in_channels,
                ),
            )?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}
