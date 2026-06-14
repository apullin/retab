//! Pure Rust IrPass: truncation semantics for fp32 -> int32 casts.
//!
//! Torch float-to-int casts truncate toward zero while TOSA CAST rounds to
//! nearest even, which shifts values like T5's relative-position bucket
//! indices by one. Port of upstream DecorateFp32toInt32CastingPass: rewrite
//! the cast input to where(x >= 0, floor(x), ceil(x)) first.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

fn is_fp_to_int_cast_target(target: &str) -> bool {
    target.contains("_to_dim_order_copy") || target.contains("aten._to_copy")
}

pub struct IrDecorateFpToIntCastPass;

impl IrPass for IrDecorateFpToIntCastPass {
    fn name(&self) -> &str {
        "IrDecorateFpToIntCast"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_fp_to_int_cast_target(&graph.nodes[id].target) {
                continue;
            }
            let x_id = match graph.nodes[id].args.first().and_then(IrArg::as_node) {
                Some(n) => n,
                None => continue,
            };
            if graph.nodes[x_id].meta.val_dtype() != Some("torch.float32")
                || graph.nodes[id].meta.val_dtype() != Some("torch.int32")
            {
                continue;
            }
            if graph.nodes[x_id].meta.output_qparam().is_some() {
                continue;
            }
            let input_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(shape) => shape.to_vec(),
                None => continue,
            };

            let name = graph.nodes[id].name.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let scalar_shape = vec![1_i64; input_shape.len().max(1)];

            let zero = graph.add_anchored_full_float(
                id,
                format!("{}_trunc_zero", name),
                &scalar_shape,
                0.0,
                "torch.float32",
                x_meta.cloned_without_qparams(scalar_shape.clone(), "torch.float32"),
            );
            let is_non_negative = graph.add_anchored_call_function(
                id,
                format!("{}_trunc_ge", name),
                "aten.ge.Tensor".to_string(),
                vec![IrArg::Node(x_id), IrArg::Node(zero)],
                x_meta.cloned_without_qparams(input_shape.clone(), "torch.bool"),
            );
            let floor_x = graph.add_anchored_call_function(
                id,
                format!("{}_trunc_floor", name),
                "aten.floor.default".to_string(),
                vec![IrArg::Node(x_id)],
                x_meta.cloned_with_val(input_shape.clone(), "torch.float32"),
            );
            let ceil_x = graph.add_anchored_call_function(
                id,
                format!("{}_trunc_ceil", name),
                "aten.ceil.default".to_string(),
                vec![IrArg::Node(x_id)],
                x_meta.cloned_with_val(input_shape.clone(), "torch.float32"),
            );
            let truncated = graph.add_anchored_call_function(
                id,
                format!("{}_trunc_where", name),
                "aten.where.self".to_string(),
                vec![
                    IrArg::Node(is_non_negative),
                    IrArg::Node(floor_x),
                    IrArg::Node(ceil_x),
                ],
                x_meta.cloned_with_val(input_shape, "torch.float32"),
            );
            graph.set_arg(id, 0, IrArg::Node(truncated));
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecorateFpToIntCast] decorated fp32->int32 casts with truncation"
            ));
        }
        modified
    }
}
