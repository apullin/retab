//! Pure Rust IrPass: rescale a quantized `aten.mm`/`aten.bmm` product to its
//! output quantization, as rc4's RewriteMatmulPass does when it retargets a
//! bmm to TOSA MATMUL (`backends/arm/_passes/rewrite_matmul.py:39-100`).
//!
//! RETAB keeps the aten target, which the serializer lowers to MATMUL, so the
//! rescaled node is RETAB's MATMUL. In explicit layout mode (APU-135 P2) the
//! pass also writes MATMUL's dtype into its val: int32 for int8 and int16
//! inputs, from rc4's fake kernel (`backends/arm/tosa/dialect/ops/matmul.py:28-76`),
//! and marks an int16 product INT48, as rc4 marks its MATMUL
//! (`rewrite_matmul.py:102-104`). The layout passes read that mark: a RESCALE
//! of INT48 data is not elementwise, so no view crosses it onto the packed
//! INT48 accumulator, and no permute moves over it
//! (`TosaPropagationOverrides`, APU-135 P3).
//! Otherwise RETAB leaves a quantized compute node's val in float32 (design
//! F2). `layout::dtype::value_dtype` reads that val, so the explicit layout
//! passes see an int32 accumulator before the RESCALE, as rc4's
//! MoveDataMovementOpsToSmallerDtypePass does. Ingest types the accumulator
//! int32 in either mode (`is_accumulator_op`, `ingest/finalize.rs`).

use crate::fx_ir::{IrGraph, IrPass, NodeId, NodeOp, RescaleRewrite};
use crate::layout::dtype;
use crate::pipeline_config::LayoutMode;
use crate::runner_logging::log_line;

fn insert_matmul_rescale(graph: &mut IrGraph, matmul_id: NodeId, layout: LayoutMode) -> bool {
    let node = &graph.nodes[matmul_id];
    if !node.alive || node.op != NodeOp::CallFunction || !node.is_mm_or_bmm() {
        return false;
    }

    let Some(qp_a) = node.meta.input_qparam_at(0) else {
        return false;
    };
    let Some(qp_b) = node.meta.input_qparam_at(1) else {
        return false;
    };
    let Some(out_qp) = node.meta.output_qparam() else {
        return false;
    };

    let rescale_scale = (qp_a.scale * qp_b.scale) / out_qp.scale;
    let rescale_meta = node.meta.rescale_output_meta(out_qp.clone());
    let rescale_name = format!("{}_output_rescale", node.name);
    let output_dtype = out_qp.dtype_str.clone();
    let output_zp = out_qp.zp;
    let rescale = {
        let mut cursor = graph.rewrite_cursor_after(matmul_id, matmul_id);
        cursor.rescale(RescaleRewrite::new(
            rescale_name,
            matmul_id,
            output_dtype,
            vec![rescale_scale],
            0,
            output_zp,
            rescale_meta,
        ))
    };
    if layout == LayoutMode::Explicit {
        let input = dtype::operand_dtype(graph, matmul_id, 0);
        let accumulator = input.as_deref().and_then(dtype::matmul_output_dtype);
        if let Some(accumulator) = accumulator {
            let meta = &mut graph.node_mut(matmul_id).meta;
            meta.set_val_dtype_for_existing_shape(accumulator);
            // rc4 rescales an int32 product to an int8 or int16 output, and marks
            // the MATMUL INT48 when its first input is int16.
            if input.as_deref() == Some("torch.int16") && accumulator == "torch.int32" {
                dtype::set_special_dtype(meta, dtype::SpecialDtype::Int48);
            }
        }
    }
    graph.replace_live_uses_except(matmul_id, rescale, rescale)
}

/// The layout mode decides whether the accumulator's val records its dtype;
/// see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct IrRewriteMatmulRescalePass {
    pub layout: LayoutMode,
}

impl IrPass for IrRewriteMatmulRescalePass {
    fn name(&self) -> &str {
        "IrRewriteMatmulRescale"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph
            .rewrite_live_call_functions(|graph, id| insert_matmul_rescale(graph, id, self.layout));
        if modified {
            log_line(format_args!(
                "[IrRewriteMatmulRescale] inserted matmul output rescale ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, QParam};
    use crate::test_ir::{call, call_with_meta, f32_call, f32_placeholder, output, placeholder};

    use super::*;

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, dtype.to_string())
    }

    #[test]
    fn inserts_output_rescale_and_rewires_existing_users() {
        let mut graph = IrGraph::empty();
        let lhs = placeholder(&mut graph, "lhs", [2, 3], "torch.int8");
        let rhs = placeholder(&mut graph, "rhs", [3, 4], "torch.int8");
        let mut matmul_meta = NodeMeta::tensor([2, 4], "torch.int32");
        matmul_meta.set_binary_input_output_qparams(
            qparam(0.5, 3, "torch.int8"),
            qparam(0.25, -4, "torch.int8"),
            qparam(0.125, 7, "torch.int8"),
        );
        let matmul = call_with_meta(
            &mut graph,
            "mm",
            "aten.mm.default",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            matmul_meta,
        );
        let user = call(
            &mut graph,
            "consumer",
            "aten.add.Tensor",
            vec![IrArg::Node(matmul), IrArg::Node(matmul)],
            [2, 4],
            "torch.int8",
        );
        output(&mut graph, user);

        assert!(IrRewriteMatmulRescalePass::default().run(&mut graph));
        let rescale = graph.by_name("mm_output_rescale").unwrap();
        assert_eq!(rescale.target, "backend.tosa.RESCALE.default");
        assert!(matches!(rescale.args[0], IrArg::Node(id) if id == matmul));
        assert!(matches!(rescale.args[1], IrArg::Dtype(ref dtype) if dtype == "torch.int8"));
        assert!(
            matches!(rescale.args[2], IrArg::List(ref scales) if matches!(scales.as_slice(), [IrArg::Float(scale)] if (*scale - 1.0).abs() < 1e-12))
        );
        assert!(matches!(rescale.args[3], IrArg::Int(0)));
        assert!(matches!(rescale.args[4], IrArg::Int(7)));
        assert_eq!(rescale.meta.val_dtype(), Some("torch.int8"));

        let user = graph.by_name("consumer").unwrap();
        let rescale_id = rescale.id;
        assert!(matches!(user.args[0], IrArg::Node(id) if id == rescale_id));
        assert!(matches!(user.args[1], IrArg::Node(id) if id == rescale_id));
    }

    #[test]
    fn skips_matmul_without_quantization_metadata() {
        let mut graph = IrGraph::empty();
        let lhs = f32_placeholder(&mut graph, "lhs", [2, 3]);
        let rhs = f32_placeholder(&mut graph, "rhs", [3, 4]);
        f32_call(
            &mut graph,
            "mm",
            "aten.mm.default",
            vec![IrArg::Node(lhs), IrArg::Node(rhs)],
            [2, 4],
        );

        assert!(!IrRewriteMatmulRescalePass::default().run(&mut graph));
    }

    /// A quantized `aten.bmm` as RETAB's IR holds it after IrFoldQdq: a
    /// float32 val with integer maps (design F2; RETAB_DEBUG_NODE on a
    /// quantized `torch.bmm` for TOSA-1.0+INT shows `dtype=torch.float32
    /// out_qparam=(.., int8)` before and after this pass).
    fn quantized_bmm(graph: &mut IrGraph, dtype: &str) -> NodeId {
        let a = placeholder(graph, "a", [1, 4, 8], dtype);
        let b = placeholder(graph, "b", [1, 8, 3], dtype);
        let mut meta = NodeMeta::tensor([1, 4, 3], "torch.float32");
        meta.set_binary_input_output_qparams(
            qparam(0.1, 0, dtype),
            qparam(0.2, 0, dtype),
            qparam(0.3, 4, dtype),
        );
        let bmm = call_with_meta(
            graph,
            "bmm",
            "aten.bmm.default",
            vec![IrArg::Node(a), IrArg::Node(b)],
            meta,
        );
        output(graph, bmm);
        bmm
    }

    /// rc4's RewriteMatmulPass in .venv-et16, on a quantized `torch.bmm`:
    /// `tosa_matmul_default :: int32[1, 4, 3] iq {0:int8,1:int8} oq {0:int8}`
    /// for TOSA-1.0+INT and `int32[1, 4, 3] iq {0:int16,1:int16} oq {0:int16}`
    /// with A16 on +int16, each followed by a RESCALE to the output dtype.
    /// Explicit mode gives RETAB's accumulator that dtype and keeps its maps;
    /// dim-order mode leaves its val alone.
    #[test]
    fn explicit_mode_gives_the_rescaled_accumulator_rc4s_matmul_dtype() {
        for dtype in ["torch.int8", "torch.int16"] {
            for (layout, expected) in [
                (LayoutMode::Explicit, "torch.int32"),
                (LayoutMode::DimOrder, "torch.float32"),
            ] {
                let mut graph = IrGraph::empty();
                let bmm = quantized_bmm(&mut graph, dtype);
                assert!(IrRewriteMatmulRescalePass { layout }.run(&mut graph));
                let meta = &graph.node(bmm).meta;
                assert_eq!(meta.val_dtype(), Some(expected), "{dtype} {layout:?}");
                assert_eq!(meta.val_shape(), Some(&[1, 4, 3][..]));
                assert_eq!(
                    meta.output_qparam().map(|qparam| qparam.dtype_str.as_str()),
                    Some(dtype)
                );
                let rescale = graph.by_name("bmm_output_rescale").unwrap();
                assert_eq!(rescale.meta.val_dtype(), Some(dtype));
                // rc4 marks an int16 MATMUL INT48 (rewrite_matmul.py:102-104).
                let int48 = layout == LayoutMode::Explicit && dtype == "torch.int16";
                assert_eq!(
                    meta.extra_str(crate::layout::dtype::SPECIAL_DTYPE_KEY),
                    int48.then_some("INT48"),
                    "{dtype} {layout:?}"
                );
            }
        }
    }

    /// The main and region pipelines build the pass for their layout mode.
    #[test]
    fn pipelines_rescale_matmuls_as_their_layout_mode_says() {
        use crate::fx_ir::PipelineStep;
        use crate::fx_pipeline::{
            build_ir_cond_submodule_pipeline, build_ir_while_submodule_pipeline,
            build_main_pipeline,
        };
        use crate::pipeline_config::PipelineConfig;

        for (layout, expected) in [
            (LayoutMode::DimOrder, "torch.float32"),
            (LayoutMode::Explicit, "torch.int32"),
        ] {
            let config = PipelineConfig {
                layout,
                ..PipelineConfig::from_tosa_spec("TOSA-1.0+INT")
            };
            for steps in [
                build_main_pipeline(&config),
                build_ir_cond_submodule_pipeline(&config),
                build_ir_while_submodule_pipeline(&config),
            ] {
                let pass = steps
                    .into_iter()
                    .find_map(|step| match step {
                        PipelineStep::Ir(pass) if pass.name() == "IrRewriteMatmulRescale" => {
                            Some(pass)
                        }
                        _ => None,
                    })
                    .expect("the pipeline rescales matmuls");
                let mut graph = IrGraph::empty();
                let bmm = quantized_bmm(&mut graph, "torch.int8");
                assert!(pass.run(&mut graph));
                assert_eq!(
                    graph.node(bmm).meta.val_dtype(),
                    Some(expected),
                    "{layout:?}"
                );
            }
        }
    }
}
