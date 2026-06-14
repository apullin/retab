//! Pure Rust IrPass: truncation semantics for fp32 -> int32 casts.
//!
//! Torch float-to-int casts truncate toward zero while TOSA CAST rounds to
//! nearest even, which shifts values like T5's relative-position bucket
//! indices by one. Port of upstream DecorateFp32toInt32CastingPass: rewrite
//! the cast input to where(x >= 0, floor(x), ceil(x)) first.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::{noop_target, NoopTarget};
use crate::runner_logging::log_line;

fn is_fp_to_int_cast_target(target: &str) -> bool {
    noop_target(target) == Some(NoopTarget::DtypeCheckedPassthrough)
}

pub struct IrDecorateFpToIntCastPass;

impl IrPass for IrDecorateFpToIntCastPass {
    fn name(&self) -> &str {
        "IrDecorateFpToIntCast"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_fp_to_int_cast_target(&graph.nodes[id].target) {
                return false;
            }
            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => return false,
            };
            if graph.node_dtype(x_id) != Some("torch.float32")
                || graph.node_dtype(id) != Some("torch.int32")
            {
                return false;
            }
            if graph.nodes[x_id].meta.output_qparam().is_some() {
                return false;
            }
            let input_shape = match graph.node_shape_vec(x_id) {
                Some(shape) => shape,
                None => return false,
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
            true
        });

        if modified {
            log_line(format_args!(
                "[IrDecorateFpToIntCast] decorated fp32->int32 casts with truncation"
            ));
        }
        modified
    }
}
