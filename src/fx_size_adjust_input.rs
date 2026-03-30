use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::fx_decompose_ops::resolve_aten_op;
use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of SizeAdjustInputPass.
///
/// For conv2d, max_pool2d, and avg_pool2d, checks if input spatial dimensions
/// yield exact integer output dimensions. If not, slices the input to remove
/// the remainder elements.
///
/// This ensures TOSA operators get inputs with exact integer output dimensions.
pub struct FxSizeAdjustInputPass;

fn is_target(target: &str) -> bool {
    let lowered = target.to_ascii_lowercase();
    if lowered.contains("transpose_conv2d") || lowered.contains("transposeconv2d") {
        return false;
    }
    lowered.contains("convolution.default")
        || lowered.contains("conv2d.default")
        || lowered.contains("conv3d.default")
        || lowered.contains("max_pool2d")
        || lowered.contains("avg_pool2d")
        || lowered.contains("conv2d")
        || lowered.contains("conv3d")
        || lowered.contains("max_pool2d")
        || lowered.contains("avg_pool2d")
}

impl FxPass for FxSizeAdjustInputPass {
    fn name(&self) -> &str {
        "FxSizeAdjustInput"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let graph = gm.gm_ref().getattr("graph")?;
        let slice_op = resolve_aten_op(py, "slice_copy.Tensor")?;

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            if let Ok(meta) = node.getattr("meta") {
                if let Ok(skip) = meta.call_method1("get", ("skip_size_adjust", false)) {
                    if skip.extract::<bool>().unwrap_or(false) {
                        continue;
                    }
                }
            }
            let target = FxGraphModule::node_target_str(node)?;
            if !is_target(&target) {
                continue;
            }

            let args = FxGraphModule::node_args(node)?;
            if args.is_empty() {
                continue;
            }
            let input_node = &args[0];
            if !FxGraphModule::is_node(input_node) {
                continue;
            }

            let input_shape = FxGraphModule::node_shape(input_node)?;
            if input_shape.len() < 4 {
                continue;
            }

            // Determine kernel, stride, padding, dilation based on op type
            let is_conv = target.contains("conv") || target.contains("CONV");
            let is_pool = target.contains("pool") || target.contains("POOL");

            if is_conv {
                // Conv: args = [input, weight, bias, stride, pad, dilation, ...]
                let weight_node = args.get(1);
                let weight_shape = weight_node
                    .and_then(|w| FxGraphModule::node_shape(w).ok())
                    .unwrap_or_default();
                if weight_shape.len() < input_shape.len() {
                    continue;
                }
                let spatial_rank = input_shape.len().saturating_sub(2);
                if spatial_rank == 0 {
                    continue;
                }

                let stride: Vec<i64> = args
                    .get(3)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| vec![1; spatial_rank]);
                let raw_pad: Vec<i64> = args
                    .get(4)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| vec![0; spatial_rank]);
                let dilation: Vec<i64> = args
                    .get(5)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| vec![1; spatial_rank]);

                // Handle both ATen pad [pad_d, pad_h, pad_w] and TOSA pad
                // [front, back, top, bottom, left, right] for arbitrary spatial rank.
                let pad_per_dim: Vec<(i64, i64)> = if raw_pad.len() == spatial_rank * 2 {
                    (0..spatial_rank)
                        .map(|i| (raw_pad[i * 2], raw_pad[i * 2 + 1]))
                        .collect()
                } else if raw_pad.len() == spatial_rank {
                    raw_pad.iter().copied().map(|p| (p, p)).collect()
                } else {
                    vec![(0, 0); spatial_rank]
                };

                let mut slices = Vec::new();
                for i in 0..spatial_rank {
                    let dim = i + 2;
                    let s = stride.get(i).copied().unwrap_or(1);
                    let (pad_before, pad_after) = pad_per_dim.get(i).copied().unwrap_or((0, 0));
                    let total_pad = pad_before + pad_after;
                    let d = dilation.get(i).copied().unwrap_or(1);
                    let k = weight_shape.get(dim).copied().unwrap_or(1);
                    let input_len = input_shape.get(dim).copied().unwrap_or(1);

                    let remainder = (input_len + total_pad - d * (k - 1) - 1) % s;
                    // Match Python Arm SizeAdjustInputPass: only trim beyond
                    // what the existing padding already accounts for.
                    if remainder > pad_before {
                        let adjustment = remainder - pad_before;
                        slices.push((dim as i64, 0i64, input_len - adjustment));
                    }
                }

                if !slices.is_empty() {
                    let mut current_input = input_node.clone();
                    for (dim, start, end) in &slices {
                        let inserting = graph.call_method1("inserting_before", (node,))?;
                        inserting.call_method0("__enter__")?;
                        let slice_args = PyTuple::new_bound(
                            py,
                            &[
                                current_input.as_ref(),
                                dim.into_py(py).into_bound(py).as_ref(),
                                start.into_py(py).into_bound(py).as_ref(),
                                end.into_py(py).into_bound(py).as_ref(),
                            ],
                        );
                        let slice_node =
                            graph.call_method1("call_function", (&slice_op, &slice_args))?;
                        // Copy meta
                        let copy_mod = py.import_bound("copy")?;
                        let meta = copy_mod.call_method1("copy", (input_node.getattr("meta")?,))?;
                        slice_node.setattr("meta", &meta)?;
                        inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

                        // Replace input of the conv/pool node
                        node.call_method1("replace_input_with", (&current_input, &slice_node))?;
                        current_input = slice_node;

                        if debug {
                            let name = FxGraphModule::node_name(node)?;
                            eprintln!(
                                "[SizeAdjust] {} dim={} slice [{}:{}]",
                                name, dim, start, end
                            );
                        }
                    }
                    modified = true;
                }
            } else if is_pool {
                // Pool: args = [input, kernel_size, stride, padding, ...]
                let kernel: Vec<i64> = args
                    .get(1)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| vec![2, 2]);
                let stride: Vec<i64> = args
                    .get(2)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| kernel.clone());
                let pad: Vec<i64> = args
                    .get(3)
                    .and_then(|a| a.extract().ok())
                    .unwrap_or_else(|| vec![0, 0]);

                let mut slices = Vec::new();
                for i in 0..2 {
                    let dim = i + 2;
                    let k = kernel.get(i).copied().unwrap_or(2);
                    let s = stride.get(i).copied().unwrap_or(k);
                    let p = pad.get(i).copied().unwrap_or(0);
                    let input_len = input_shape.get(dim).copied().unwrap_or(1);

                    let remainder = (input_len + 2 * p - k) % s;
                    if remainder > p {
                        let adjustment = remainder - p;
                        slices.push((dim as i64, 0i64, input_len - adjustment));
                    }
                }

                if !slices.is_empty() {
                    let mut current_input = input_node.clone();
                    for (dim, start, end) in &slices {
                        let inserting = graph.call_method1("inserting_before", (node,))?;
                        inserting.call_method0("__enter__")?;
                        let slice_args = PyTuple::new_bound(
                            py,
                            &[
                                current_input.as_ref(),
                                dim.into_py(py).into_bound(py).as_ref(),
                                start.into_py(py).into_bound(py).as_ref(),
                                end.into_py(py).into_bound(py).as_ref(),
                            ],
                        );
                        let slice_node =
                            graph.call_method1("call_function", (&slice_op, &slice_args))?;
                        let copy_mod = py.import_bound("copy")?;
                        let meta = copy_mod.call_method1("copy", (input_node.getattr("meta")?,))?;
                        slice_node.setattr("meta", &meta)?;
                        inserting.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

                        node.call_method1("replace_input_with", (&current_input, &slice_node))?;
                        current_input = slice_node;

                        if debug {
                            let name = FxGraphModule::node_name(node)?;
                            eprintln!(
                                "[SizeAdjust] {} dim={} slice [{}:{}]",
                                name, dim, start, end
                            );
                        }
                    }
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
