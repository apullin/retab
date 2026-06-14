//! Pure Rust IrPass version of FxConvertSplitToSlicePass.
//!
//! Converts split_with_sizes_copy / split_copy into multiple slice_copy ops.
//! Each getitem user of the split node is replaced with a slice_copy that
//! extracts the corresponding chunk along the split dimension.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::fx_target::is_split_target;
use crate::runner_logging::log_line;

/// Compute the per-chunk split lengths from the second arg of the split node.
///
/// - If it is a list of ints, use those directly (split_with_sizes).
/// - If it is a single int, compute equal chunks with a possible remainder
///   (split_copy with a chunk size).
fn resolve_split_lengths(arg: &IrArg, dim_size: i64) -> Option<Vec<i64>> {
    match arg {
        IrArg::List(items) => {
            let lengths: Option<Vec<i64>> = items.iter().map(|a| a.as_int()).collect();
            lengths
        }
        IrArg::Int(chunk_size) => {
            let chunk_size = *chunk_size;
            if chunk_size <= 0 {
                return None;
            }
            let n_full = dim_size / chunk_size;
            let remainder = dim_size % chunk_size;
            let mut lengths = vec![chunk_size; n_full as usize];
            if remainder > 0 {
                lengths.push(remainder);
            }
            Some(lengths)
        }
        _ => None,
    }
}

pub struct IrConvertSplitToSlicePass;

impl IrPass for IrConvertSplitToSlicePass {
    fn name(&self) -> &str {
        "IrConvertSplitToSlice"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for split_id in graph.live_call_function_ids() {
            if !is_split_target(&graph.nodes[split_id].target) {
                continue;
            }

            // --- Extract split parameters ---

            let input_id = match graph.nodes[split_id].first_node_arg() {
                Some(id) => id,
                None => continue,
            };
            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };
            let rank = input_shape.len() as i64;
            if rank == 0 {
                continue;
            }

            // args[2] = dim (default 0)
            let dim_raw: i64 = graph.nodes[split_id]
                .args
                .get(2)
                .and_then(|a| a.as_int())
                .unwrap_or(0);
            let dim = ((dim_raw % rank) + rank) % rank;
            let dim_size = input_shape[dim as usize];

            // args[1] = split_lengths (list) or chunk_size (int)
            let split_lengths = match graph.nodes[split_id].args.get(1) {
                Some(arg) => match resolve_split_lengths(arg, dim_size) {
                    Some(l) => l,
                    None => continue,
                },
                None => continue,
            };

            // Compute start/end offsets for each chunk
            let mut starts = Vec::with_capacity(split_lengths.len());
            let mut ends = Vec::with_capacity(split_lengths.len());
            let mut offset = 0i64;
            for &length in &split_lengths {
                starts.push(offset);
                ends.push(offset + length);
                offset += length;
            }

            // Grab the dtype from the input for building output metas
            let dtype = graph.node_dtype_or_float32(input_id).to_string();

            let split_name = graph.nodes[split_id].name.clone();

            // --- Find getitem users and replace each with a slice_copy ---

            // Collect user info before mutating the graph.  We scan all alive
            // nodes for those whose first arg is this split node and whose
            // target contains "getitem".
            let getitem_users: Vec<(NodeId, usize)> = graph
                .live_getitem_users(split_id)
                .into_iter()
                .map(|(node_id, index)| (node_id, index as usize))
                .collect();

            let mut replaced_any = false;

            for (user_id, index) in &getitem_users {
                let index = *index;
                if index >= starts.len() {
                    continue;
                }

                // Build output shape: same as input but with the split dim
                // narrowed to the chunk length.
                let mut out_shape = input_shape.clone();
                out_shape[dim as usize] = split_lengths[index];

                // Build meta for the slice_copy node.
                // Inherit output_qparams from the getitem user (not the split).
                let mut slice_meta = graph.nodes[split_id]
                    .meta
                    .cloned_with_val(out_shape, &dtype);
                slice_meta.copy_output_qparams_from(&graph.nodes[*user_id].meta);

                // slice_copy.Tensor(input, dim, start, end)
                let new_id = graph.add_slice_copy(
                    format!("{}_slice_{}", split_name, index),
                    input_id,
                    dim,
                    starts[index],
                    ends[index],
                    None,
                    slice_meta,
                );

                // Replace the getitem user with the new slice node
                graph.replace_and_erase(*user_id, new_id);
                replaced_any = true;
            }

            // The split node itself is now dead (all getitem users replaced).
            if replaced_any {
                graph.erase(split_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrConvertSplitToSlice] converted split ops to slice_copy"
            ));
        }
        modified
    }
}
