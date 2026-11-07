//! Pure Rust IrPass version of FxDecomposeRepeatInterleavePass.
//!
//! Decomposes `repeat_interleave.self_int(x, repeats, dim)` into:
//!   1. unsqueeze_copy(x, dim + 1)
//!   2. repeat(unsqueezed, multiples)  — `multiples[dim+1] = repeats`
//!   3. view_copy(repeated, final_shape) — `final_shape[dim] *= repeats`

use crate::fx_ir::{i64_list_arg as i64_list, normalize_dim, IrArg, IrGraph, IrPass};
use crate::fx_target::is_repeat_interleave_self_int_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeRepeatInterleavePass;

impl IrPass for IrDecomposeRepeatInterleavePass {
    fn name(&self) -> &str {
        "IrDecomposeRepeatInterleave"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_repeat_interleave_self_int_target(&graph.nodes[id].target) {
                return false;
            }

            // args: (x, repeats, dim)
            if graph.nodes[id].args.len() < 3 {
                return false;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => return false,
            };
            let repeats = match graph.nodes[id].args[1].as_int() {
                Some(r) if r >= 0 => r,
                _ => return false,
            };
            let dim_raw = match &graph.nodes[id].args[2] {
                IrArg::None => return false,
                IrArg::Int(d) => *d,
                _ => return false,
            };

            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => return false,
            };
            let Some(dim) = normalize_dim(dim_raw, input_shape.len()) else {
                return false;
            };

            let node_name = graph.nodes[id].name.clone();
            let base_meta = graph.nodes[id].meta.clone();
            let mut cursor = graph.rewrite_cursor(id);

            // 1. unsqueeze_copy(x, dim + 1)
            let unsqueeze_dim = dim + 1;
            let mut unsqueeze_shape = Vec::with_capacity(input_shape.len() + 1);
            for (i, &d) in input_shape.iter().enumerate() {
                if i == unsqueeze_dim {
                    unsqueeze_shape.push(1);
                }
                unsqueeze_shape.push(d);
            }
            if unsqueeze_dim >= input_shape.len() {
                unsqueeze_shape.push(1);
            }

            let unsqueeze_meta =
                base_meta.cloned_with_same_dtype_or_float32(unsqueeze_shape.clone());
            let unsqueezed = cursor.call_function(
                format!("{}_unsqueeze", node_name),
                "aten.unsqueeze_copy.default",
                vec![IrArg::Node(input_id), IrArg::Int(unsqueeze_dim as i64)],
                unsqueeze_meta,
            );

            // 2. repeat(unsqueezed, multiples) — multiples[unsqueeze_dim] = repeats
            let repeat_rank = input_shape.len() + 1;
            let mut multiples = vec![1i64; repeat_rank];
            multiples[unsqueeze_dim] = repeats;

            let mut repeated_shape = unsqueeze_shape;
            repeated_shape[unsqueeze_dim] = repeats;

            let repeat_meta = base_meta.cloned_with_same_dtype_or_float32(repeated_shape);
            let repeated = cursor.call_function(
                format!("{}_repeat", node_name),
                "aten.repeat.default",
                vec![IrArg::Node(unsqueezed), i64_list(&multiples)],
                repeat_meta,
            );

            // 3. view_copy(repeated, final_shape) — final_shape[dim] *= repeats
            let mut final_shape = input_shape;
            final_shape[dim] *= repeats;

            let view_meta = base_meta.cloned_with_same_dtype_or_float32(final_shape.clone());
            let result = cursor.view_copy(
                format!("{}_view", node_name),
                repeated,
                &final_shape,
                view_meta,
            );
            cursor.replace_anchor_with(result);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrDecomposeRepeatInterleave] decomposed repeat_interleave ops"
            ));
        }
        modified
    }
}
