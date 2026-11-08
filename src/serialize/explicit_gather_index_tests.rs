//! rc4's TopKGather delegate (APU-167) through RETAB's parent pipeline, ingest and
//! serialization. rc4's partitioner delegates the int32-to-int64 cast in front of
//! the gather, and its PrepareGatherIndicesPass (`arm_pass_manager.py:548`) points
//! the gather back at the int32 indices. Without that port the cast's INT64 tensor
//! reached the serializer, which refuses it; with it the delegate serializes to
//! the operators stock emits for the case on TOSA-1.0+INT and Ethos-U85.

use super::collection::{collect_regions, SerializedGraphItems};
use super::tosa_types::{DType, Op};
use super::visitors;
use crate::fx_ir::{IrArg, IrGraph, MetaValue, NodeId, NodeMeta, NodeOp, PipelineStep};
use crate::graph::Graph;
use crate::layout::shape::mark_contiguous;
use crate::pipeline_config::PipelineConfig;
use crate::store::TensorStore;
use crate::test_ir::{call_with_meta, output, placeholder};

const INT8: &str = "torch.int8";
const INT32: &str = "torch.int32";
const INT64: &str = "torch.int64";
const FLOAT32: &str = "torch.float32";

/// The specs whose stock lowering of the case this mirrors: generic INT and U85's
/// (`EthosUCompileSpec._tosa_spec_for_target`).
const SPECS: [&str; 2] = ["TOSA-1.0+INT", "TOSA-1.0+INT+int16+int4+cf"];

fn value(shape: &[i64], dtype: &str) -> NodeMeta {
    let mut meta = NodeMeta::tensor(shape.to_vec(), dtype);
    mark_contiguous(&mut meta);
    meta
}

fn qdq(graph: &mut IrGraph, name: &str, target: &str, input: NodeId, meta: NodeMeta) -> NodeId {
    call_with_meta(
        graph,
        name,
        target,
        vec![
            IrArg::Node(input),
            IrArg::Float(0.05),
            IrArg::Int(-3),
            IrArg::Int(-128),
            IrArg::Int(127),
            IrArg::Dtype(INT8.to_owned()),
        ],
        meta,
    )
}

/// The delegate as rc4's partitioner cuts the quantized TopKGather of
/// test_convert_int64_output_ops_to_int32.py: the quantized features and the
/// topk's int32 indices come in (the topk is FP-only and stays on the CPU), the
/// indices are unsqueezed, expanded and cast to int64, and the dequantized
/// features are gathered and requantized.
fn topk_gather_delegate() -> IrGraph {
    let mut graph = IrGraph::empty();
    let features = placeholder(&mut graph, "features", vec![2, 8, 2], INT8);
    mark_contiguous(&mut graph.node_mut(features).meta);
    let indices = placeholder(&mut graph, "indices", vec![2, 3], INT32);
    mark_contiguous(&mut graph.node_mut(indices).meta);
    let dequantized = qdq(
        &mut graph,
        "dequantize",
        "quantized_decomposed.dequantize_per_tensor.default",
        features,
        value(&[2, 8, 2], FLOAT32),
    );
    let unsqueeze = call_with_meta(
        &mut graph,
        "unsqueeze",
        "aten.unsqueeze_copy.default",
        vec![IrArg::Node(indices), IrArg::Int(-1)],
        value(&[2, 3, 1], INT32),
    );
    let expand = call_with_meta(
        &mut graph,
        "expand",
        "aten.expand_copy.default",
        vec![
            IrArg::Node(unsqueeze),
            IrArg::List(vec![IrArg::Int(-1), IrArg::Int(-1), IrArg::Int(2)]),
        ],
        value(&[2, 3, 2], INT32),
    );
    let cast = graph.add_node_with_kwargs(
        NodeOp::CallFunction,
        "cast".to_owned(),
        "dim_order_ops._to_dim_order_copy.default".to_owned(),
        vec![IrArg::Node(expand)],
        std::collections::HashMap::from([("dtype".to_owned(), IrArg::Dtype(INT64.to_owned()))]),
        value(&[2, 3, 2], INT64),
    );
    // The quantizer annotates the gather, which is what IrFoldQdq folds.
    let mut gather_meta = value(&[2, 3, 2], FLOAT32);
    gather_meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));
    let gather = call_with_meta(
        &mut graph,
        "gather",
        "aten.gather.default",
        vec![IrArg::Node(dequantized), IrArg::Int(1), IrArg::Node(cast)],
        gather_meta,
    );
    let quantized = qdq(
        &mut graph,
        "quantize",
        "quantized_decomposed.quantize_per_tensor.default",
        gather,
        value(&[2, 3, 2], INT8),
    );
    output(&mut graph, quantized);
    graph
}

/// Run the parent pipeline for `spec`, without the steps `skip` names, then ingest
/// and serialize the graph as `ingest_and_serialize` does.
fn lower(spec: &str, skip: &[&str]) -> Result<SerializedGraphItems, String> {
    let config = PipelineConfig::from_tosa_spec(spec);
    let mut graph = topk_gather_delegate();
    crate::stock_fallback::take();
    for step in crate::fx_pipeline::build_main_pipeline(&config) {
        match step {
            PipelineStep::Ir(pass) if !skip.contains(&pass.name()) => {
                pass.run(&mut graph);
            }
            PipelineStep::Effectful(pass) if !skip.contains(&pass.name()) => {
                pass.run(&mut graph);
            }
            _ => {}
        }
    }
    if let Some(reason) = crate::stock_fallback::take() {
        return Err(format!("stock fallback: {reason}"));
    }
    let mut store = TensorStore::new();
    let mut converted: Graph = crate::ingest::from_ir::graph_from_ir(&graph, &mut store);
    crate::ingest::finalize::finalize_converted_graph(&mut converted);
    converted.rebuild_user_lists();
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    let mut regions = collect_regions(&[("main".to_string(), converted)], &store)?;
    Ok(regions.remove(0).items)
}

fn tensor_dtype(items: &SerializedGraphItems, name: &str) -> (DType, Vec<i32>) {
    let tensor = items
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor named {name}"));
    (tensor.dtype, tensor.shape.clone())
}

/// The delegate serializes with no INT64 tensor, and with stock's operators for the
/// case: RESHAPE and TILE for the unsqueeze and expand, and CanonicalizeGather's
/// rank-3 form, two TRANSPOSE and RESHAPE pairs into GATHER and one back out.
/// GATHER reads the INT32 indices, flattened to [N*C, W] = [4, 3].
#[test]
fn topk_gather_serializes_with_int32_indices() {
    for spec in SPECS {
        let items = lower(spec, &[]).unwrap_or_else(|error| panic!("{spec}: {error}"));
        let mut ops: Vec<Op> = items
            .operators
            .iter()
            .map(|op| op.op)
            .filter(|&op| op != Op::ConstShape)
            .collect();
        ops.sort_by_key(|&op| op as u32);
        let mut expected = vec![
            Op::Reshape,
            Op::Tile,
            Op::Transpose,
            Op::Reshape,
            Op::Transpose,
            Op::Reshape,
            Op::Gather,
            Op::Reshape,
            Op::Transpose,
        ];
        expected.sort_by_key(|&op| op as u32);
        assert_eq!(ops, expected, "{spec}");

        for tensor in &items.tensors {
            assert!(
                matches!(tensor.dtype, DType::Int8 | DType::Int32 | DType::Shape),
                "{spec}: {} is {:?}",
                tensor.name,
                tensor.dtype
            );
        }
        let gather = items
            .operators
            .iter()
            .find(|op| op.op == Op::Gather)
            .expect("a GATHER");
        assert_eq!(
            tensor_dtype(&items, &gather.inputs[0]),
            (DType::Int8, vec![4, 8, 1]),
            "{spec}"
        );
        assert_eq!(
            tensor_dtype(&items, &gather.inputs[1]),
            (DType::Int32, vec![4, 3]),
            "{spec}"
        );
        assert_eq!(items.input_names, ["features", "indices"], "{spec}");
        assert_eq!(
            tensor_dtype(&items, "indices"),
            (DType::Int32, vec![2, 3]),
            "{spec}"
        );
        let output = &items.output_names[0];
        assert_eq!(
            tensor_dtype(&items, output),
            (DType::Int8, vec![2, 3, 2]),
            "{spec}"
        );
    }
}

/// Without IrPrepareGatherIndices the int64 cast reaches the serializer, which
/// refuses it: the failure APU-167 reported.
#[test]
fn without_the_port_the_int64_cast_reaches_the_serializer() {
    for spec in SPECS {
        let error = lower(spec, &["IrPrepareGatherIndices"])
            .err()
            .unwrap_or_else(|| panic!("{spec}: lowered without the port"));
        assert!(
            error.contains("TOSA 1.0 does not support INT64 tensor"),
            "{spec}: {error}"
        );
    }
}
