//! Pure Rust IrPass: dilated max_pool2d → pad + space-to-batch + pool + batch-to-space + slice.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId};
use crate::fx_target::{is_max_pool2d_target, is_max_pool2d_with_indices_target};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

fn pack_dimension(dil: i64, pad: i64, orig: i64, kernel: i64, stride: i64) -> (i64, i64, i64) {
    let padded = orig + pad * 2;
    let extra = (dil - (padded % dil)) % dil;
    let packed = (padded + extra) / dil;
    let eff_k = dil * (kernel - 1) + 1;
    let out = if padded < eff_k {
        0
    } else {
        1 + (padded - eff_k) / stride
    };
    (packed, pad + extra, out)
}

fn pool_output_size(input: i64, kernel: i64, stride: i64, pad: i64, ceil_mode: bool) -> i64 {
    let mut output = if ceil_mode {
        (input + 2 * pad - kernel + stride - 1) / stride + 1
    } else {
        (input + 2 * pad - kernel) / stride + 1
    };
    if ceil_mode && (output - 1) * stride >= input + pad {
        output -= 1;
    }
    output
}

fn pad_value_for_dtype(dtype: &str) -> IrArg {
    if DType::torch_str_is_float_like(dtype) {
        IrArg::Float(f64::NEG_INFINITY)
    } else if let Some((qmin, _)) = DType::quantized_integer_bounds_from_torch_str(dtype) {
        IrArg::Int(qmin)
    } else {
        IrArg::Int(i32::MIN as i64)
    }
}

pub struct IrDecomposeMaxPoolDilationPass;

impl IrPass for IrDecomposeMaxPoolDilationPass {
    fn name(&self) -> &str {
        "IrDecomposeMaxPoolDilation"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            let t = &graph.nodes[id].target;
            if !is_max_pool2d_target(t) {
                continue;
            }
            let is_with_indices = is_max_pool2d_with_indices_target(t);

            let value_getitem = if is_with_indices {
                let mut value_user = None;
                let mut unsupported = false;
                for &user_id in graph.users(id) {
                    let user = &graph.nodes[user_id];
                    if !user.alive {
                        continue;
                    }
                    if !user.is_getitem() {
                        unsupported = true;
                        break;
                    }
                    match user.args.get(1).and_then(|arg| arg.as_int()) {
                        Some(0) => value_user = Some(user_id),
                        Some(_) => {
                            if !graph.live_user_ids(user_id).is_empty() {
                                unsupported = true;
                                break;
                            }
                        }
                        None => {
                            unsupported = true;
                            break;
                        }
                    }
                }
                if unsupported {
                    continue;
                }
                match value_user {
                    Some(user_id) => Some(user_id),
                    None => continue,
                }
            } else {
                None
            };

            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.node_shape(x_id) {
                Some(s) if s.len() >= 4 => s.to_vec(),
                _ => continue,
            };
            let (n, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );
            let node = &graph.nodes[id];

            let (k_h, k_w) = node.arg_int_pair_or_scalar(1).unwrap_or((1, 1));
            let (s_h, s_w) = node.arg_int_pair_or_scalar(2).unwrap_or((k_h, k_w));
            let (pad_h, pad_w) = node.arg_int_pair_or_scalar(3).unwrap_or((0, 0));
            let (d_h, d_w) = node.arg_int_pair_or_scalar(4).unwrap_or((1, 1));
            let ceil_mode = node.arg_bool_or(5, false);

            if d_h == 1 && d_w == 1 {
                continue;
            }

            let (h_pack, bottom_pad, h_final) = pack_dimension(d_h, pad_h, h, k_h, s_h);
            let (w_pack, right_pad, w_final) = pack_dimension(d_w, pad_w, w, k_w, s_w);

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = value_getitem
                .map(|user_id| graph.nodes[user_id].meta.clone())
                .unwrap_or_else(|| graph.nodes[id].meta.clone());
            let x_meta = graph.nodes[x_id].meta.clone();
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);
            let dtype = meta.dtype();

            let result = {
                let mut cursor = graph.rewrite_cursor(id);

                // 1. constant_pad_nd(x, [pad_w, right_pad, pad_h, bottom_pad, 0,0,0,0], pad_val)
                let x_pad = cursor.call_function(
                    format!("{}_pad", node_name),
                    "aten.constant_pad_nd.default",
                    vec![
                        IrArg::Node(x_id),
                        i64_list(&[pad_w, right_pad, pad_h, bottom_pad, 0, 0, 0, 0]),
                        pad_value_for_dtype(dtype),
                    ],
                    meta.preserve_qparams(vec![
                        n,
                        c,
                        h + pad_h + bottom_pad,
                        w + pad_w + right_pad,
                    ]),
                );

                // 2. view -> [N, C, h_pack, d_h, w_pack, d_w]
                let x_view = cursor.view_copy(
                    format!("{}_pack_view", node_name),
                    x_pad,
                    &[n, c, h_pack, d_h, w_pack, d_w],
                    meta.preserve_qparams(vec![n, c, h_pack, d_h, w_pack, d_w]),
                );

                // 3. permute [3,5,0,1,2,4]
                let x_perm = cursor.permute_copy(
                    format!("{}_pack_perm", node_name),
                    x_view,
                    &[3, 5, 0, 1, 2, 4],
                    meta.preserve_qparams(vec![d_h, d_w, n, c, h_pack, w_pack]),
                );

                // 4. view -> [N*d_h*d_w, C, h_pack, w_pack]
                let x_batch = cursor.view_copy(
                    format!("{}_batch", node_name),
                    x_perm,
                    &[n * d_h * d_w, c, h_pack, w_pack],
                    meta.preserve_qparams(vec![n * d_h * d_w, c, h_pack, w_pack]),
                );

                // 5. max_pool2d (non-dilated, zero padding)
                let h_out = pool_output_size(h_pack, k_h, s_h, 0, ceil_mode);
                let w_out = pool_output_size(w_pack, k_w, s_w, 0, ceil_mode);
                let pooled = cursor.call_function(
                    format!("{}_pool", node_name),
                    "aten.max_pool2d.default",
                    vec![
                        IrArg::Node(x_batch),
                        i64_list(&[k_h, k_w]),
                        i64_list(&[s_h, s_w]),
                        i64_list(&[0, 0]),
                        IrArg::Int(1), // dilation=1
                        IrArg::Bool(ceil_mode),
                    ],
                    meta.preserve_qparams(vec![n * d_h * d_w, c, h_out, w_out]),
                );

                // 6. view -> [d_h, d_w, N, C, h_out, w_out]
                let unpack = cursor.view_copy(
                    format!("{}_unpack", node_name),
                    pooled,
                    &[d_h, d_w, n, c, h_out, w_out],
                    meta.preserve_qparams(vec![d_h, d_w, n, c, h_out, w_out]),
                );

                // 7. permute [2,3,4,0,5,1]
                let unperm = cursor.permute_copy(
                    format!("{}_unperm", node_name),
                    unpack,
                    &[2, 3, 4, 0, 5, 1],
                    meta.preserve_qparams(vec![n, c, h_out, d_h, w_out, d_w]),
                );

                // 8. view -> [N, C, h_out*d_h, w_out*d_w]
                let out_view = cursor.view_copy(
                    format!("{}_out_view", node_name),
                    unperm,
                    &[n, c, h_out * d_h, w_out * d_w],
                    meta.preserve_qparams(vec![n, c, h_out * d_h, w_out * d_w]),
                );

                // 9. slice H to h_final
                let slice_h = cursor.slice_copy(
                    format!("{}_slice_h", node_name),
                    out_view,
                    2,
                    0,
                    h_final,
                    None,
                    meta.preserve_qparams(vec![n, c, h_final, w_out * d_w]),
                );

                // 10. slice W to w_final
                cursor.slice_copy(
                    format!("{}_slice_w", node_name),
                    slice_h,
                    3,
                    0,
                    w_final,
                    None,
                    meta.preserve_qparams(vec![n, c, h_final, w_final]),
                )
            };

            if let Some(value_getitem) = value_getitem {
                graph.replace_and_erase(value_getitem, result);
                let extra_getitems: Vec<NodeId> = graph
                    .users(id)
                    .iter()
                    .copied()
                    .filter(|&user_id| {
                        user_id != value_getitem
                            && graph.nodes[user_id].alive
                            && graph.nodes[user_id].is_getitem()
                    })
                    .collect();
                for user_id in extra_getitems {
                    graph.erase(user_id);
                }
                graph.erase(id);
            } else {
                graph.replace_and_erase(id, result);
            }
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeMaxPoolDilation] decomposed dilated maxpool ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_output_drops_window_starting_inside_padding() {
        assert_eq!(pool_output_size(4, 1, 2, 0, true), 2);
    }

    #[test]
    fn floor_output_size_unchanged() {
        assert_eq!(pool_output_size(8, 2, 2, 0, false), 4);
    }
}
