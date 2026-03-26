use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::{insert_call_before, propagate_qparams, resolve_aten_op};
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of grouped convolutions into
/// `slice_copy -> conv2d (groups=1) -> cat`.
///
/// TOSA does not support grouped convolutions (groups > 1) that are not
/// depthwise. This pass splits them into per-group convolutions using
/// slice_copy on input/weight/bias, then concatenates the results.
///
/// Matches Python `DecomposeGroupedConvPass`.
pub struct FxDecomposeGroupedConvPass;

fn is_convolution(target: &str) -> bool {
    target.contains("convolution.default") || target.contains("conv2d.default")
}

impl FxPass for FxDecomposeGroupedConvPass {
    fn name(&self) -> &str {
        "FxDecomposeGroupedConv"
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
            if !is_convolution(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            let is_raw_conv2d = target.contains("conv2d.default");

            // Need enough args to extract groups
            if !is_raw_conv2d && args.len() < 9 {
                continue;
            }
            if is_raw_conv2d && args.len() < 2 {
                continue;
            }

            let groups: i64 = if is_raw_conv2d {
                args.get(6).and_then(|a| a.extract().ok()).unwrap_or(1)
            } else {
                args[8].extract().unwrap_or(1)
            };

            if groups <= 1 {
                continue;
            }

            // Check if depthwise (in_channels == groups) — skip those
            let input_node = &args[0];
            let in_channels = FxGraphModule::node_shape(input_node)
                .ok()
                .and_then(|s| s.get(1).copied())
                .unwrap_or(0);

            let transposed: bool = if is_raw_conv2d {
                false
            } else {
                args[6].extract().unwrap_or(false)
            };

            if !transposed && in_channels == groups {
                // Depthwise — handled by FxRewriteConvPass
                continue;
            }

            candidates.push(node.clone());
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for node in &candidates {
            decompose_grouped_conv(py, &graph, gm, node, debug)?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

fn decompose_grouped_conv<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    let target = FxGraphModule::node_target_str(node)?;
    let is_raw_conv2d = target.contains("conv2d.default");

    let input_node = &args[0];
    let weight_node = &args[1];
    let bias_node = &args[2];

    let groups: i64 = if is_raw_conv2d {
        args.get(6).and_then(|a| a.extract().ok()).unwrap_or(1)
    } else {
        args[8].extract().unwrap_or(1)
    };

    let transposed: bool = if is_raw_conv2d {
        false
    } else {
        args[6].extract().unwrap_or(false)
    };

    let weight_shape = FxGraphModule::node_shape(weight_node).unwrap_or_default();

    // Compute slice sizes from weight shape
    // weight shape: [out_channels, in_channels/groups, kH, kW]
    // For transposed conv: [in_channels, out_channels/groups, kH, kW]
    let (input_slice_size, output_slice_size) = if transposed {
        let input_slice = weight_shape.first().copied().unwrap_or(0) / groups;
        let output_slice = weight_shape.get(1).copied().unwrap_or(0);
        (input_slice, output_slice)
    } else {
        let input_slice = weight_shape.get(1).copied().unwrap_or(0);
        let output_slice = weight_shape.first().copied().unwrap_or(0) / groups;
        (input_slice, output_slice)
    };

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeGroupedConv] {}: groups={}, in_slice={}, out_slice={}, transposed={}",
            name, groups, input_slice_size, output_slice_size, transposed
        );
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    // Resolve the conv op to use for the per-group convolutions
    let conv_op = if is_raw_conv2d {
        resolve_aten_op(py, "conv2d.default")?
    } else {
        resolve_aten_op(py, "convolution.default")?
    };

    let mut output_slices: Vec<Bound<'py, PyAny>> = Vec::new();

    for i in 0..groups {
        // Slice input along channel dim (dim=1)
        let in_start = i * input_slice_size;
        let in_stop = (i + 1) * input_slice_size;
        let input_slice_args = PyTuple::new_bound(
            py,
            &[
                input_node.as_ref(),
                1i64.into_py(py).into_bound(py).as_ref(),
                in_start.into_py(py).into_bound(py).as_ref(),
                in_stop.into_py(py).into_bound(py).as_ref(),
            ],
        );
        let input_slice =
            insert_call_before(py, graph, node, &slice_op, input_slice_args.as_ref(), input_node)?;
        propagate_qparams(input_node, &input_slice)?;

        // Slice weight along dim 0 (output channels for normal, input channels for transposed)
        let (w_start, w_stop) = if transposed {
            (i * input_slice_size, (i + 1) * input_slice_size)
        } else {
            (i * output_slice_size, (i + 1) * output_slice_size)
        };
        let weight_slice_args = PyTuple::new_bound(
            py,
            &[
                weight_node.as_ref(),
                0i64.into_py(py).into_bound(py).as_ref(),
                w_start.into_py(py).into_bound(py).as_ref(),
                w_stop.into_py(py).into_bound(py).as_ref(),
            ],
        );
        let weight_slice =
            insert_call_before(py, graph, node, &slice_op, weight_slice_args.as_ref(), weight_node)?;
        propagate_qparams(weight_node, &weight_slice)?;

        // Slice bias along dim 0 (output channels)
        let bias_slice = if bias_node.is_none() {
            py.None().into_bound(py)
        } else {
            let b_start = i * output_slice_size;
            let b_stop = (i + 1) * output_slice_size;
            let bias_slice_args = PyTuple::new_bound(
                py,
                &[
                    bias_node.as_ref(),
                    0i64.into_py(py).into_bound(py).as_ref(),
                    b_start.into_py(py).into_bound(py).as_ref(),
                    b_stop.into_py(py).into_bound(py).as_ref(),
                ],
            );
            let bias_slice = insert_call_before(
                py,
                graph,
                node,
                &slice_op,
                bias_slice_args.as_ref(),
                bias_node,
            )?;
            propagate_qparams(bias_node, &bias_slice)?;
            bias_slice
        };

        // Build conv args with groups=1
        let conv_args = if is_raw_conv2d {
            // conv2d args: (input, weight, bias, stride, padding, dilation, groups)
            let mut conv_arg_vec: Vec<Bound<'py, PyAny>> = vec![
                input_slice.clone(),
                weight_slice.clone(),
                bias_slice.clone(),
            ];
            // Copy stride, padding, dilation from original args (indices 3..6)
            for idx in 3..6 {
                if let Some(a) = args.get(idx) {
                    conv_arg_vec.push(a.clone());
                }
            }
            // groups=1
            conv_arg_vec.push(1i64.into_py(py).into_bound(py));
            PyTuple::new_bound(py, &conv_arg_vec)
        } else {
            // convolution args: (input, weight, bias, stride, padding, dilation,
            //                    transposed, output_padding, groups)
            let mut conv_arg_vec: Vec<Bound<'py, PyAny>> = vec![
                input_slice.clone(),
                weight_slice.clone(),
                bias_slice.clone(),
            ];
            // Copy stride, padding, dilation, transposed, output_padding (indices 3..8)
            for idx in 3..8 {
                conv_arg_vec.push(args[idx].clone());
            }
            // groups=1
            conv_arg_vec.push(1i64.into_py(py).into_bound(py));
            PyTuple::new_bound(py, &conv_arg_vec)
        };

        let conv_node =
            insert_call_before(py, graph, node, &conv_op, conv_args.as_ref(), node)?;
        // Propagate qparams so InsertRescalePass handles this conv correctly
        propagate_qparams(node, &conv_node)?;
        output_slices.push(conv_node);
    }

    // Cat all outputs along dim 1
    let slices_list = pyo3::types::PyList::new_bound(py, &output_slices);
    let cat_args = PyTuple::new_bound(
        py,
        &[
            slices_list.as_ref(),
            1i64.into_py(py).into_bound(py).as_ref(),
        ],
    );
    let cat_node = insert_call_before(py, graph, node, &cat_op, cat_args.as_ref(), node)?;

    // Copy original node's meta onto the cat result (preserves qparams)
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    cat_node.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &cat_node)?;
    gm.erase_node(node)?;
    Ok(())
}
