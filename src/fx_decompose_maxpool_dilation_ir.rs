//! Pure Rust IrPass: dilated max_pool2d → pad + space-to-batch + pool + batch-to-space + slice.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId, NodeOp};
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
    if ceil_mode {
        (input + 2 * pad - kernel + stride - 1) / stride + 1
    } else {
        (input + 2 * pad - kernel) / stride + 1
    }
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
            let dtype = graph.node_dtype_or_float32(x_id).to_string();

            // 1. constant_pad_nd(x, [pad_w, right_pad, pad_h, bottom_pad, 0,0,0,0], pad_val)
            let x_pad = graph.add_anchored_call_function(
                id,
                format!("{}_pad", node_name),
                "aten.constant_pad_nd.default",
                vec![
                    IrArg::Node(x_id),
                    i64_list(&[pad_w, right_pad, pad_h, bottom_pad, 0, 0, 0, 0]),
                    pad_value_for_dtype(&dtype),
                ],
                orig_meta.cloned_with_val(
                    vec![n, c, h + pad_h + bottom_pad, w + pad_w + right_pad],
                    &dtype,
                ),
            );

            // 2. view → [N, C, h_pack, d_h, w_pack, d_w]
            let x_view = graph.add_anchored_view_copy(
                id,
                format!("{}_pack_view", node_name),
                x_pad,
                &[n, c, h_pack, d_h, w_pack, d_w],
                orig_meta.cloned_with_val(vec![n, c, h_pack, d_h, w_pack, d_w], &dtype),
            );

            // 3. permute [3,5,0,1,2,4]
            let x_perm = graph.add_anchored_call_function(
                id,
                format!("{}_pack_perm", node_name),
                "aten.permute_copy.default",
                vec![IrArg::Node(x_view), i64_list(&[3, 5, 0, 1, 2, 4])],
                orig_meta.cloned_with_val(vec![d_h, d_w, n, c, h_pack, w_pack], &dtype),
            );

            // 4. view → [N*d_h*d_w, C, h_pack, w_pack]
            let x_batch = graph.add_anchored_view_copy(
                id,
                format!("{}_batch", node_name),
                x_perm,
                &[n * d_h * d_w, c, h_pack, w_pack],
                orig_meta.cloned_with_val(vec![n * d_h * d_w, c, h_pack, w_pack], &dtype),
            );

            // 5. max_pool2d (non-dilated, zero padding)
            let h_out = pool_output_size(h_pack, k_h, s_h, 0, ceil_mode);
            let w_out = pool_output_size(w_pack, k_w, s_w, 0, ceil_mode);
            let pooled = graph.add_anchored_call_function(
                id,
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
                orig_meta.cloned_with_val(vec![n * d_h * d_w, c, h_out, w_out], &dtype),
            );

            // 6. view → [d_h, d_w, N, C, h_out, w_out]
            let unpack = graph.add_anchored_view_copy(
                id,
                format!("{}_unpack", node_name),
                pooled,
                &[d_h, d_w, n, c, h_out, w_out],
                orig_meta.cloned_with_val(vec![d_h, d_w, n, c, h_out, w_out], &dtype),
            );

            // 7. permute [2,3,4,0,5,1]
            let unperm = graph.add_anchored_call_function(
                id,
                format!("{}_unperm", node_name),
                "aten.permute_copy.default",
                vec![IrArg::Node(unpack), i64_list(&[2, 3, 4, 0, 5, 1])],
                orig_meta.cloned_with_val(vec![n, c, h_out, d_h, w_out, d_w], &dtype),
            );

            // 8. view → [N, C, h_out*d_h, w_out*d_w]
            let out_view = graph.add_anchored_view_copy(
                id,
                format!("{}_out_view", node_name),
                unperm,
                &[n, c, h_out * d_h, w_out * d_w],
                orig_meta.cloned_with_val(vec![n, c, h_out * d_h, w_out * d_w], &dtype),
            );

            // 9. slice H to h_final
            let slice_h = graph.add_anchored_call_function(
                id,
                format!("{}_slice_h", node_name),
                "aten.slice_copy.Tensor",
                vec![
                    IrArg::Node(out_view),
                    IrArg::Int(2),
                    IrArg::Int(0),
                    IrArg::Int(h_final),
                ],
                orig_meta.cloned_with_val(vec![n, c, h_final, w_out * d_w], &dtype),
            );

            // 10. slice W to w_final
            let result = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_slice_w", node_name),
                "aten.slice_copy.Tensor".to_string(),
                vec![
                    IrArg::Node(slice_h),
                    IrArg::Int(3),
                    IrArg::Int(0),
                    IrArg::Int(w_final),
                ],
                orig_meta.cloned_with_val(vec![n, c, h_final, w_final], &dtype),
            );

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
