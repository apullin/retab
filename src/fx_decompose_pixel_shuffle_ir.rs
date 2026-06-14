//! Pure Rust IrPass version of FxDecomposePixelShufflePass.
//!
//! Decomposes `pixel_shuffle` and `pixel_unshuffle` into
//! view_copy + permute_copy + view_copy chains.
//!
//! pixel_shuffle(x, r):
//!   x: (N, C*r*r, H, W)
//!   -> view_copy(N, C, r, r, H, W)
//!   -> permute_copy(0, 1, 4, 2, 5, 3)
//!   -> view_copy(N, C, H*r, W*r)
//!
//! pixel_unshuffle(x, r):
//!   x: (N, C, H, W)
//!   -> view_copy(N, C, H//r, r, W//r, r)
//!   -> permute_copy(0, 1, 3, 5, 2, 4)
//!   -> view_copy(N, C*r*r, H//r, W//r)

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::{is_pixel_shuffle_target, is_pixel_unshuffle_target};
use crate::runner_logging::log_line;

/// Helper: create a view_copy node with the given output shape.
fn add_view_copy(
    graph: &mut IrGraph,
    input_id: NodeId,
    shape: &[i64],
    name: String,
    base_meta: &NodeMeta,
    dtype: &str,
) -> NodeId {
    graph.add_node(
        NodeOp::CallFunction,
        name,
        "aten.view_copy.default".to_string(),
        vec![
            IrArg::Node(input_id),
            IrArg::List(shape.iter().map(|&d| IrArg::Int(d)).collect()),
        ],
        base_meta.cloned_with_val(shape.to_vec(), dtype),
    )
}

/// Helper: create a permute_copy node with the given permutation.
fn add_permute_copy(
    graph: &mut IrGraph,
    input_id: NodeId,
    input_shape: &[i64],
    perm: &[usize],
    name: String,
    base_meta: &NodeMeta,
    dtype: &str,
) -> NodeId {
    let permuted_shape: Vec<i64> = perm.iter().map(|&p| input_shape[p]).collect();
    graph.add_node(
        NodeOp::CallFunction,
        name,
        "aten.permute_copy.default".to_string(),
        vec![
            IrArg::Node(input_id),
            IrArg::List(perm.iter().map(|&p| IrArg::Int(p as i64)).collect()),
        ],
        base_meta.cloned_with_val(permuted_shape, dtype),
    )
}

pub struct IrDecomposePixelShufflePass;

impl IrPass for IrDecomposePixelShufflePass {
    fn name(&self) -> &str {
        "IrDecomposePixelShuffle"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            let target = &graph.nodes[id].target;
            let is_ps = is_pixel_shuffle_target(target);
            let is_pus = is_pixel_unshuffle_target(target);
            if !is_ps && !is_pus {
                continue;
            }

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(nid) => nid,
                None => continue,
            };
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            if input_shape.len() != 4 {
                continue;
            }

            // Extract upscale_factor (second arg)
            let r = match graph.nodes[id].args.get(1).and_then(|a| a.as_int()) {
                Some(v) => v,
                None => continue,
            };

            let orig_dtype = graph.nodes[id].meta.val_dtype_or_float32().to_string();
            let node_name = graph.nodes[id].name.clone();
            let base_meta = graph.nodes[id].meta.clone();

            if is_ps {
                let n = input_shape[0];
                let c_rr = input_shape[1];
                let h = input_shape[2];
                let w = input_shape[3];
                let c = c_rr / (r * r);

                // Step 1: view_copy -> (N, C, r, r, H, W)
                let view1_shape = [n, c, r, r, h, w];
                let view1_id = add_view_copy(
                    graph,
                    input_id,
                    &view1_shape,
                    format!("{}_view1", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                // Step 2: permute_copy(0,1,4,2,5,3) -> (N, C, H, r, W, r)
                let perm = [0, 1, 4, 2, 5, 3];
                let permute_id = add_permute_copy(
                    graph,
                    view1_id,
                    &view1_shape,
                    &perm,
                    format!("{}_permute", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                // Step 3: view_copy -> (N, C, H*r, W*r)
                let final_shape = [n, c, h * r, w * r];
                let result_id = add_view_copy(
                    graph,
                    permute_id,
                    &final_shape,
                    format!("{}_view2", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                graph.replace_and_erase(id, result_id);
                modified = true;
            } else {
                // pixel_unshuffle
                let n = input_shape[0];
                let c = input_shape[1];
                let h = input_shape[2];
                let w = input_shape[3];
                let h_out = h / r;
                let w_out = w / r;

                // Step 1: view_copy -> (N, C, H//r, r, W//r, r)
                let view1_shape = [n, c, h_out, r, w_out, r];
                let view1_id = add_view_copy(
                    graph,
                    input_id,
                    &view1_shape,
                    format!("{}_view1", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                // Step 2: permute_copy(0,1,3,5,2,4) -> (N, C, r, r, H//r, W//r)
                let perm = [0, 1, 3, 5, 2, 4];
                let permute_id = add_permute_copy(
                    graph,
                    view1_id,
                    &view1_shape,
                    &perm,
                    format!("{}_permute", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                // Step 3: view_copy -> (N, C*r*r, H//r, W//r)
                let final_shape = [n, c * r * r, h_out, w_out];
                let result_id = add_view_copy(
                    graph,
                    permute_id,
                    &final_shape,
                    format!("{}_view2", node_name),
                    &base_meta,
                    &orig_dtype,
                );

                graph.replace_and_erase(id, result_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposePixelShuffle] decomposed pixel_shuffle/pixel_unshuffle ops"
            ));
        }
        modified
    }
}
