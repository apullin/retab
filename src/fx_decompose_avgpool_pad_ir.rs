//! Pure Rust IrPass: avg_pool2d count_include_pad / divisor_override handling.
//!
//! TOSA AVG_POOL2D averages over valid (non-pad) elements only, i.e.
//! count_include_pad=False. Port of upstream DecomposeAvgPool2dPass: when
//! count_include_pad is set (the aten default) with nonzero padding, pad the
//! input explicitly with zero constants via cat so the zeros count as valid
//! elements, then pool without padding; a divisor_override becomes a constant
//! multiply by kernel_size / divisor.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass};
use crate::fx_target::is_plain_avg_pool2d_target;
use crate::runner_logging::log_line;

/// Post padding needed so TOSA's explicit pre/post padding reproduces aten's
/// symmetric pad + ceil/floor output-size rounding.
fn adjust_pooling_pad_if_needed(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad: i64,
    ceil_mode: bool,
) -> i64 {
    let numerator = input_size - kernel_size + 2 * pad;
    let output_size = if ceil_mode {
        (numerator + stride - 1).div_euclid(stride) + 1
    } else {
        numerator.div_euclid(stride) + 1
    };
    (output_size - 1) * stride - input_size + kernel_size - pad
}

fn compute_post_pad(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad: i64,
    ceil_mode: bool,
    divisor_override: Option<i64>,
) -> i64 {
    if pad == 0 {
        return pad;
    }
    if ceil_mode && divisor_override.is_none() {
        return pad;
    }

    let adjusted = adjust_pooling_pad_if_needed(input_size, kernel_size, stride, pad, ceil_mode);
    if adjusted > 0 {
        adjusted
    } else {
        pad
    }
}

pub struct IrDecomposeAvgPoolPadPass;

impl IrPass for IrDecomposeAvgPoolPadPass {
    fn name(&self) -> &str {
        "IrDecomposeAvgPoolPad"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_plain_avg_pool2d_target(&graph.nodes[id].target) {
                continue;
            }

            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let node = &graph.nodes[id];
            let Some((kernel_h, kernel_w)) = node.arg_int_pair_or_scalar(1) else {
                continue;
            };
            let (stride_h, stride_w) = node
                .arg_int_pair_or_scalar(2)
                .unwrap_or((kernel_h, kernel_w));
            let (pad_h, pad_w) = node.arg_int_pair_or_scalar(3).unwrap_or((0, 0));
            let ceil_mode = node.arg_bool_or(4, false);
            let mut count_include_pad = node.arg_bool_or(5, true);
            let divisor_override = node.arg_int(6);

            if divisor_override.is_some() {
                count_include_pad = true;
            }
            let pads_needed = count_include_pad && (pad_h > 0 || pad_w > 0);
            let multiply_needed =
                divisor_override.is_some_and(|d| d != 0 && d != kernel_h * kernel_w);
            if !pads_needed && !multiply_needed {
                continue;
            }

            let input_shape = match graph.node_shape(x_id) {
                Some(shape) if shape.len() == 4 => shape.to_vec(),
                _ => continue,
            };
            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let input_meta = x_meta.tensor_stamp_same_dtype_or_float32();
            let output_meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);
            let dtype = input_meta.dtype();
            let (n, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );

            let mut pooled_in = x_id;
            let (mut new_pad_h, mut new_pad_w) = (pad_h, pad_w);
            let mut post_pad_w = 0_i64;

            let result = {
                let mut cursor = graph.rewrite_cursor(id);

                if count_include_pad && pad_w > 0 {
                    post_pad_w =
                        compute_post_pad(w, kernel_w, stride_w, pad_w, ceil_mode, divisor_override);
                    let pre_shape = vec![n, c, h, pad_w];
                    let pre = cursor.full_float(
                        format!("{}_pre_pad_w", name),
                        &pre_shape,
                        0.0,
                        dtype,
                        input_meta.drop_qparams(pre_shape.clone()),
                    );
                    let mut cat_inputs = vec![IrArg::Node(pre), IrArg::Node(pooled_in)];
                    if post_pad_w > 0 {
                        let post_shape = vec![n, c, h, post_pad_w];
                        let post = cursor.full_float(
                            format!("{}_post_pad_w", name),
                            &post_shape,
                            0.0,
                            dtype,
                            input_meta.drop_qparams(post_shape.clone()),
                        );
                        cat_inputs.push(IrArg::Node(post));
                    }
                    let padded_w = w + pad_w + post_pad_w;
                    let padded_shape = vec![n, c, h, padded_w];
                    pooled_in = cursor.call_function(
                        format!("{}_pad_w_cat", name),
                        "aten.cat.default",
                        vec![IrArg::List(cat_inputs), IrArg::Int(3)],
                        input_meta.preserve_qparams(padded_shape),
                    );
                    new_pad_w = 0;
                }

                if count_include_pad && pad_h > 0 {
                    let post_pad_h =
                        compute_post_pad(h, kernel_h, stride_h, pad_h, ceil_mode, divisor_override);
                    let cur_w = w + pad_w + post_pad_w;
                    let pre_shape = vec![n, c, pad_h, cur_w];
                    let pre = cursor.full_float(
                        format!("{}_pre_pad_h", name),
                        &pre_shape,
                        0.0,
                        dtype,
                        input_meta.drop_qparams(pre_shape.clone()),
                    );
                    let mut cat_inputs = vec![IrArg::Node(pre), IrArg::Node(pooled_in)];
                    if post_pad_h > 0 {
                        let post_shape = vec![n, c, post_pad_h, cur_w];
                        let post = cursor.full_float(
                            format!("{}_post_pad_h", name),
                            &post_shape,
                            0.0,
                            dtype,
                            input_meta.drop_qparams(post_shape.clone()),
                        );
                        cat_inputs.push(IrArg::Node(post));
                    }
                    let padded_h = h + pad_h + post_pad_h;
                    let padded_shape = vec![n, c, padded_h, cur_w];
                    pooled_in = cursor.call_function(
                        format!("{}_pad_h_cat", name),
                        "aten.cat.default",
                        vec![IrArg::List(cat_inputs), IrArg::Int(2)],
                        input_meta.preserve_qparams(padded_shape),
                    );
                    new_pad_h = 0;
                }

                let pooled = cursor.call_function(
                    format!("{}_unpadded", name),
                    "aten.avg_pool2d.default",
                    vec![
                        IrArg::Node(pooled_in),
                        i64_list(&[kernel_h, kernel_w]),
                        i64_list(&[stride_h, stride_w]),
                        i64_list(&[new_pad_h, new_pad_w]),
                        IrArg::Bool(ceil_mode),
                        IrArg::Bool(false),
                    ],
                    orig_meta.clone(),
                );

                if multiply_needed {
                    let divisor = divisor_override.unwrap();
                    let factor = cursor.full_float(
                        format!("{}_divisor_factor", name),
                        &[1, 1, 1, 1],
                        (kernel_h * kernel_w) as f64 / divisor as f64,
                        dtype,
                        output_meta.drop_qparams(vec![1, 1, 1, 1]),
                    );
                    cursor.call_function(
                        format!("{}_divisor_mul", name),
                        "aten.mul.Tensor",
                        vec![IrArg::Node(pooled), IrArg::Node(factor)],
                        orig_meta.clone(),
                    )
                } else {
                    pooled
                }
            };

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeAvgPoolPad] decomposed padded avg_pool2d ops"
            ));
        }
        modified
    }
}
