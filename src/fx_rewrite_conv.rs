use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of conv rewriting.
///
/// Detects `aten.convolution.default` and raw `aten.conv2d.default` nodes and
/// rewrites them to TOSA
/// `CONV2D` or `DEPTHWISE_CONV2D` ops via the Python helper. This matches
/// the Python `RewriteConvPass` behavior:
///   - Changes target from aten.convolution to tosa.CONV2D/DEPTHWISE_CONV2D
///   - Rearranges args (pad format, bias handling)
///   - Attaches conv_info metadata with quantization zero-points
pub struct FxRewriteConvPass;

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("conv2d.default")
}

fn infer_in_channels(
    input_node: &Bound<'_, PyAny>,
    input_shape: &[i64],
    weight_shape: &[i64],
    groups: i64,
) -> PyResult<i64> {
    // The direct path may already have converted activations into channels-last
    // layout before conv rewriting. Respect `tosa_dim_order` when present so
    // we read the logical C axis, not just shape[1].
    let meta = input_node.getattr("meta")?;
    let dim_order = meta.call_method1("get", ("tosa_dim_order",))?;
    if !dim_order.is_none() {
        let order: Vec<usize> = dim_order.extract().unwrap_or_default();
        if let Some(channel_axis) = order.iter().position(|&axis| axis == 1) {
            if let Some(&channels) = input_shape.get(channel_axis) {
                if channels > 0 {
                    return Ok(channels);
                }
            }
        }
    }

    if let Some(&channels) = input_shape.get(1) {
        if channels > 0 {
            return Ok(channels);
        }
    }

    if weight_shape.len() >= 2 {
        let channels_per_group = weight_shape[1];
        if channels_per_group > 0 && groups > 0 {
            return Ok(channels_per_group * groups);
        }
    }

    Ok(0)
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
            let is_raw_conv2d = target.contains("conv2d.default");

            // aten.convolution args:
            // [input, weight, bias, stride, padding, dilation, transposed,
            //  output_padding, groups]
            //
            // aten.conv2d args can appear in compact form after earlier FX
            // normalization:
            // [input, weight, bias, stride, padding]
            // [input, weight, bias, stride, padding, dilation]
            // [input, weight, bias, stride, padding, dilation, groups]
            // aten.convolution.default needs 9 args; aten.conv2d.default needs
            // at least 2 (input, weight).  After fold_qdq, conv2d may have
            // only 2-3 args if bias/stride/padding were simplified.
            if !is_raw_conv2d && args.len() < 9 {
                continue;
            }
            if is_raw_conv2d && args.len() < 2 {
                continue;
            }

            let input_node = &args[0];
            let weight_node = &args[1];
            let groups: i64 = if is_raw_conv2d {
                args.get(6).and_then(|arg| arg.extract().ok()).unwrap_or(1)
            } else {
                args[8].extract().unwrap_or(1)
            };

            // Weight shape for conv: [out_channels, in_channels/groups, kH, kW]
            // Raw fast-lane graphs can arrive with partially-normalized
            // metadata; use best-effort shape recovery and only require exact
            // channel inference when deciding whether the op is depthwise.
            let weight_shape = FxGraphModule::node_shape(weight_node).unwrap_or_default();
            let out_channels = weight_shape.first().copied().unwrap_or(0);
            let input_shape = FxGraphModule::node_shape(input_node).unwrap_or_default();
            let in_channels = infer_in_channels(input_node, &input_shape, &weight_shape, groups)?;

            // Match Python reference: depthwise when in_channels == groups and
            // out_channels is a multiple of in_channels.
            let is_depthwise = weight_shape.len() >= 4
                && groups > 1
                && in_channels > 0
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

        let debug = FxGraphModule::debug();

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
