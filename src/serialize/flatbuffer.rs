use crate::pipeline_config::PipelineConfig;
use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, Vector, WIPOffset};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::attribute_flatbuffer::build_attribute;
use super::collection::SerializedRegion;
use super::tosa_types::{TosaOperator, TosaShape, TosaTensor};

pub(super) fn serialize_regions_to_tosa_py(
    py: Python<'_>,
    regions: &[SerializedRegion],
    estimated_capacity: usize,
    config: &PipelineConfig,
) -> Py<PyBytes> {
    let mut builder = FlatBufferBuilder::with_capacity(estimated_capacity.max(16 * 1024 * 1024));
    let regions: Vec<_> = regions
        .iter()
        .map(|region| build_region(&mut builder, region, config))
        .collect();
    finish_tosa_graph_py(py, &mut builder, &regions)
}

fn build_region<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    region: &SerializedRegion,
    config: &PipelineConfig,
) -> WIPOffset<TableFinishedWIPOffset> {
    if crate::config::retab_debug_enabled() {
        eprintln!("[Serialize] building fb tensors...");
    }
    let fb_tensors = build_tensors(builder, &region.items.tensors);
    if crate::config::retab_debug_enabled() {
        eprintln!("[Serialize] building fb shapes...");
    }
    let fb_shapes = build_shapes(builder, &region.items.shapes);
    if crate::config::retab_debug_enabled() {
        eprintln!("[Serialize] building fb operators...");
    }
    let fb_operators = build_operators(builder, &region.items.operators, config.is_u55_subset);
    if crate::config::retab_debug_enabled() {
        eprintln!("[Serialize] assembling graph...");
    }

    let fb_input_strs: Vec<_> = region
        .items
        .input_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_output_strs: Vec<_> = region
        .items
        .output_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_inputs = builder.create_vector(&fb_input_strs);
    let fb_outputs = builder.create_vector(&fb_output_strs);

    let fb_tensors_vec = builder.create_vector(&fb_tensors);
    let fb_shapes_vec = builder.create_vector(&fb_shapes);
    let fb_operators_vec = builder.create_vector(&fb_operators);

    let block_name = builder.create_string(&region.name);
    let block = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, block_name);
        builder.push_slot_always::<WIPOffset<_>>(6, fb_operators_vec);
        builder.push_slot_always::<WIPOffset<_>>(8, fb_tensors_vec);
        builder.push_slot_always::<WIPOffset<_>>(10, fb_inputs);
        builder.push_slot_always::<WIPOffset<_>>(12, fb_outputs);
        builder.push_slot_always::<WIPOffset<_>>(14, fb_shapes_vec);
        builder.end_table(start)
    };

    let blocks_vec = builder.create_vector(&[block]);
    let region_name = builder.create_string(&region.name);
    let start = builder.start_table();
    builder.push_slot_always::<WIPOffset<_>>(4, region_name);
    builder.push_slot_always::<WIPOffset<_>>(6, blocks_vec);
    builder.end_table(start)
}

fn finalize_tosa_graph(
    builder: &mut FlatBufferBuilder<'_>,
    regions: &[WIPOffset<TableFinishedWIPOffset>],
) {
    let modifier_str = builder.create_string("retab");
    let sw_version = {
        let start = builder.start_table();
        builder.push_slot::<i32>(4, 0, -1);
        builder.push_slot::<i32>(6, 1, -1);
        builder.push_slot::<i32>(8, 0, -1);
        builder.push_slot_always::<WIPOffset<_>>(10, modifier_str);
        builder.end_table(start)
    };

    let version = {
        let start = builder.start_table();
        builder.push_slot::<i32>(4, 1, -1);
        builder.push_slot::<i32>(6, 0, -1);
        builder.push_slot::<i32>(8, 0, -1);
        builder.push_slot::<bool>(10, false, true);
        builder.end_table(start)
    };

    let regions_vec = builder.create_vector(regions);
    let tosa_graph = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, version);
        builder.push_slot_always::<WIPOffset<_>>(6, regions_vec);
        builder.push_slot_always::<WIPOffset<_>>(8, sw_version);
        builder.end_table(start)
    };

    builder.finish(tosa_graph, Some("TOSA"));
}

fn finish_tosa_graph_py(
    py: Python<'_>,
    builder: &mut FlatBufferBuilder<'_>,
    regions: &[WIPOffset<TableFinishedWIPOffset>],
) -> Py<PyBytes> {
    finalize_tosa_graph(builder, regions);
    PyBytes::new(py, builder.finished_data()).unbind()
}

fn create_force_aligned_u8_vector<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    data: &[u8],
) -> WIPOffset<Vector<'fbb, u8>> {
    // The flatbuffers Rust runtime does not expose schema-level vector
    // `force_align` directly. Treating each `u8` as an 8-byte-aligned element
    // corrupts the payload, so dense TOSA tensor buffers use a normal vector.
    builder.create_vector(data)
}

fn build_tensors(
    builder: &mut FlatBufferBuilder<'_>,
    tensors: &[TosaTensor],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        assert_ne!(
            t.dtype,
            super::tosa_types::DType::Int64,
            "TOSA 1.0 does not support INT64 tensor '{}'",
            t.name
        );
        let name = builder.create_string(&t.name);
        let shape = builder.create_vector(&t.shape);
        let data = t
            .data
            .as_ref()
            .filter(|d| !d.is_empty())
            .map(|d| create_force_aligned_u8_vector(builder, d));
        let var_name = t
            .variable_name
            .as_ref()
            .filter(|s| !s.is_empty())
            .map(|s| builder.create_string(s));

        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, name);
        builder.push_slot_always::<WIPOffset<_>>(6, shape);
        builder.push_slot::<u32>(8, t.dtype as u32, 0);
        if let Some(d) = data {
            builder.push_slot_always::<WIPOffset<_>>(10, d);
        }
        builder.push_slot::<bool>(12, t.variable, false);
        builder.push_slot::<bool>(14, t.is_unranked, false);
        if let Some(vn) = var_name {
            builder.push_slot_always::<WIPOffset<_>>(16, vn);
        }
        offsets.push(builder.end_table(start));
    }
    offsets
}

fn build_shapes(
    builder: &mut FlatBufferBuilder<'_>,
    shapes: &[TosaShape],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(shapes.len());
    for s in shapes {
        let name = builder.create_string(&s.name);
        let data = if !s.data.is_empty() {
            Some(create_force_aligned_u8_vector(builder, &s.data))
        } else {
            None
        };

        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, name);
        builder.push_slot::<u32>(6, s.rank, 0);
        if let Some(d) = data {
            builder.push_slot_always::<WIPOffset<_>>(8, d);
        }
        offsets.push(builder.end_table(start));
    }
    offsets
}

fn build_operators(
    builder: &mut FlatBufferBuilder<'_>,
    operators: &[TosaOperator],
    is_u55_subset: bool,
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(operators.len());
    for op in operators {
        let input_strs: Vec<_> = op.inputs.iter().map(|s| builder.create_string(s)).collect();
        let output_strs: Vec<_> = op
            .outputs
            .iter()
            .map(|s| builder.create_string(s))
            .collect();
        let inputs = builder.create_vector(&input_strs);
        let outputs = builder.create_vector(&output_strs);

        let (attr_type, attr_offset) = build_attribute(builder, &op.attribute, is_u55_subset);

        let start = builder.start_table();
        builder.push_slot::<u32>(4, op.op as u32, 0);
        builder.push_slot::<u8>(6, attr_type, 0);
        if let Some(attr) = attr_offset {
            builder.push_slot_always::<WIPOffset<_>>(8, attr);
        }
        builder.push_slot_always::<WIPOffset<_>>(10, inputs);
        builder.push_slot_always::<WIPOffset<_>>(12, outputs);
        offsets.push(builder.end_table(start));
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::tosa_types::DType;

    #[test]
    #[should_panic(expected = "INT64")]
    fn regression_int64_tensor_cannot_reach_tosa_1_0_flatbuffer() {
        let tensor = TosaTensor {
            name: "int64_tensor".into(),
            shape: vec![1],
            dtype: DType::Int64,
            data: Some(1i64.to_le_bytes().to_vec()),
            ..Default::default()
        };
        let mut builder = FlatBufferBuilder::new();
        build_tensors(&mut builder, &[tensor]);
    }

    #[test]
    fn int64_constants_and_runtime_inputs_return_serialization_errors() {
        use crate::graph::ir::{Arg, DType as IrDType, NodeMeta, OpTarget, OpType, TensorInfo};
        use crate::graph::Graph;
        use crate::store::{TensorData, TensorStore};
        use smallvec::smallvec;

        for constant in [false, true] {
            let mut graph = Graph::new();
            let input = graph.create_node(
                OpType::Placeholder,
                OpTarget::Other("x".into()),
                smallvec![],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[1], IrDType::Int64)),
                "x".into(),
            );
            graph.create_node(
                OpType::Output,
                OpTarget::Other("output".into()),
                smallvec![Arg::Node(input)],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[1], IrDType::Int64)),
                "output".into(),
            );
            let mut store = TensorStore::default();
            if constant {
                store.insert(
                    "x".into(),
                    TensorData {
                        data: 1i64.to_le_bytes().to_vec(),
                        shape: vec![1],
                        dtype: IrDType::Int64,
                    },
                );
            }
            let error = match super::super::collection::collect_regions(
                &[("main".into(), graph)],
                &store,
            ) {
                Err(error) => error,
                Ok(_) => panic!("accepted an INT64 tensor"),
            };
            assert!(error.contains("TOSA 1.0 does not support INT64 tensor 'x'"));
        }
    }

    #[test]
    fn u55_shift_rounding_reaches_every_serialized_region() {
        use super::super::collection::SerializedGraphItems;
        use super::super::tosa_types::{ArithmeticRightShiftAttribute, Attribute, Op};

        for spec in ["TOSA-1.0+INT+u55", "TOSA-1.0+INT+cf", "TOSA-1.0+INT"] {
            let config = PipelineConfig::from_tosa_spec(spec);
            let regions: Vec<_> = ["main", "branch", "loop_body"]
                .into_iter()
                .map(|name| SerializedRegion {
                    name: name.into(),
                    items: SerializedGraphItems {
                        tensors: vec![],
                        shapes: vec![],
                        input_names: vec![],
                        output_names: vec![],
                        operators: vec![TosaOperator::new(
                            Op::ArithmeticRightShift,
                            Attribute::ArithmeticRightShift(ArithmeticRightShiftAttribute {}),
                            vec!["x".into(), "shift".into()],
                            vec!["out".into()],
                        )],
                    },
                })
                .collect();
            let mut builder = FlatBufferBuilder::new();
            let regions: Vec<_> = regions
                .iter()
                .map(|region| build_region(&mut builder, region, &config))
                .collect();
            finalize_tosa_graph(&mut builder, &regions);
            let graph = tosa_rs::generated::root_as_tosa_graph(builder.finished_data()).unwrap();
            for region in graph.regions().unwrap() {
                let block = region.blocks().unwrap().get(0);
                let op = block.operators().unwrap().get(0);
                let attr = op.attribute_as_arithmetic_right_shift_attribute().unwrap();
                assert_eq!(attr.round(), config.is_u55_subset);
            }
        }
    }
}
