//! Pure Rust IrPass: truncation semantics for fp32 -> int32 casts.
//!
//! Torch float-to-int casts and copies truncate toward zero while TOSA CAST rounds to
//! nearest even, which shifts values like T5's relative-position bucket
//! indices by one. Port of upstream DecorateFp32toInt32CastingPass: rewrite
//! the cast input to where(x >= 0, floor(x), ceil(x)) first.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::{noop_target, NoopTarget};
use crate::runner_logging::log_line;

pub struct IrDecorateFpToIntCastPass;

impl IrPass for IrDecorateFpToIntCastPass {
    fn name(&self) -> &str {
        "IrDecorateFpToIntCast"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let input_arg = match noop_target(&graph.nodes[id].target) {
                Some(NoopTarget::DtypeCheckedPassthrough) => 0,
                Some(NoopTarget::CopySourceArg) => 1,
                _ => return false,
            };
            let x_id = match graph.nodes[id].arg_node(input_arg) {
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
            let fp_meta = x_meta.tensor_stamp("torch.float32");
            let bool_meta = x_meta.tensor_stamp("torch.bool");
            let scalar_shape = vec![1_i64; input_shape.len().max(1)];

            let mut cursor = graph.rewrite_cursor(id);
            let zero = cursor.full_float(
                format!("{}_trunc_zero", name),
                &scalar_shape,
                0.0,
                "torch.float32",
                fp_meta.drop_qparams(scalar_shape.clone()),
            );
            let is_non_negative = cursor.call_function(
                format!("{}_trunc_ge", name),
                "aten.ge.Tensor",
                vec![IrArg::Node(x_id), IrArg::Node(zero)],
                bool_meta.drop_qparams(input_shape.clone()),
            );
            let floor_x = cursor.call_function(
                format!("{}_trunc_floor", name),
                "aten.floor.default",
                vec![IrArg::Node(x_id)],
                fp_meta.preserve_qparams(input_shape.clone()),
            );
            let ceil_x = cursor.call_function(
                format!("{}_trunc_ceil", name),
                "aten.ceil.default",
                vec![IrArg::Node(x_id)],
                fp_meta.preserve_qparams(input_shape.clone()),
            );
            let truncated = cursor.call_function(
                format!("{}_trunc_where", name),
                "aten.where.self",
                vec![
                    IrArg::Node(is_non_negative),
                    IrArg::Node(floor_x),
                    IrArg::Node(ceil_x),
                ],
                fp_meta.preserve_qparams(input_shape),
            );
            graph.set_arg(id, input_arg, IrArg::Node(truncated));
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
