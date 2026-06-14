//! Pure Rust IrPass version of FxDecomposeAsStridedCopyPass.
//!
//! Rewrites as_strided_copy → view_copy when the stride is contiguous
//! and storage_offset is zero.

use crate::fx_index_linear::contiguous_strides;
use crate::fx_ir::{IrGraph, IrPass};
use crate::fx_target::is_as_strided_copy_target;
use crate::runner_logging::log_line;

/// Check whether the given stride matches contiguous layout for the shape.
fn stride_matches(shape: &[i64], stride: &[i64]) -> bool {
    if shape.len() != stride.len() {
        return false;
    }
    let expected = contiguous_strides(shape);
    shape
        .iter()
        .zip(stride.iter())
        .zip(expected.iter())
        .enumerate()
        .all(|(idx, ((&dim, &actual), &expected))| {
            if actual < 0 {
                return false;
            }
            if idx == shape.len() - 1 {
                return actual >= expected;
            }
            if dim == 1 || expected == 0 {
                return true;
            }
            actual == expected
        })
}

pub struct IrDecomposeAsStridedCopyPass;

impl IrPass for IrDecomposeAsStridedCopyPass {
    fn name(&self) -> &str {
        "IrDecomposeAsStridedCopy"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_as_strided_copy_target(&graph.nodes[id].target) {
                return false;
            }

            // args: [input, size, stride, storage_offset?]
            // Need at least 3 args: input, size, stride
            if graph.nodes[id].args.len() < 3 {
                return false;
            }
            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => return false,
            };

            // Extract size (args[1]) as int list
            let size = match graph.nodes[id].args[1].as_int_list() {
                Some(v) => v,
                None => return false,
            };

            // Extract stride (args[2]) as int list
            let stride = match graph.nodes[id].args[2].as_int_list() {
                Some(v) => v,
                None => return false,
            };

            // Extract storage_offset (args[3] if present, else 0)
            let storage_offset = if graph.nodes[id].args.len() > 3 {
                match graph.nodes[id].args[3].as_int() {
                    Some(v) => v,
                    None => 0,
                }
            } else {
                0
            };

            // Only decompose when offset is zero and strides are contiguous
            if storage_offset != 0 || !stride_matches(&size, &stride) {
                return false;
            }

            // Build output meta: shape from size, dtype from input
            let dtype = graph.nodes[input_id]
                .meta
                .val_dtype_or_float32()
                .to_string();
            let out_meta = graph.nodes[id].meta.cloned_with_val(size.clone(), dtype);

            // Replace as_strided_copy with view_copy(input, size)
            graph.replace_with_view_copy(
                id,
                format!("{}_view_copy", graph.nodes[id].name),
                input_id,
                &size,
                out_meta,
            );
            true
        });

        if modified {
            log_line(format_args!(
                "[IrDecomposeAsStridedCopy] rewrote as_strided_copy → view_copy"
            ));
        }
        modified
    }
}
