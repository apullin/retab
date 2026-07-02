use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, Vector, WIPOffset};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::attribute_flatbuffer::build_attribute;
use super::collection::SerializedRegion;
use super::tosa_generated::{TosaOperator, TosaShape, TosaTensor};

fn log_flatbuffer_profile_phase(profile: bool, phase: &str, start: std::time::Instant) {
    if profile {
        eprintln!(
            "[RETAB][flatbuffer-profile] phase={phase} time_s={:.6}",
            start.elapsed().as_secs_f64()
        );
    }
}

pub(super) fn serialize_regions_to_tosa_py(
    py: Python<'_>,
    regions: &[SerializedRegion],
    estimated_capacity: usize,
) -> Py<PyBytes> {
    let profile = crate::config::profile_ingest_serialize_enabled();
    let phase_start = std::time::Instant::now();
    let mut builder = FlatBufferBuilder::with_capacity(estimated_capacity.max(16 * 1024 * 1024));
    log_flatbuffer_profile_phase(profile, "init_builder", phase_start);
    let phase_start = std::time::Instant::now();
    let regions: Vec<_> = regions
        .iter()
        .map(|region| build_region(&mut builder, region))
        .collect();
    log_flatbuffer_profile_phase(profile, "build_regions", phase_start);
    let phase_start = std::time::Instant::now();
    let py_bytes = finish_tosa_graph_py(py, &mut builder, &regions);
    log_flatbuffer_profile_phase(profile, "finish_tosa_graph_py", phase_start);
    py_bytes
}

fn build_region<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    region: &SerializedRegion,
) -> WIPOffset<TableFinishedWIPOffset> {
    let profile = crate::config::profile_ingest_serialize_enabled();
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!("[Serialize] building fb tensors...");
    }
    let phase_start = std::time::Instant::now();
    let fb_tensors = build_tensors(builder, &region.items.tensors);
    log_flatbuffer_profile_phase(profile, "build_tensors", phase_start);
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!("[Serialize] building fb shapes...");
    }
    let phase_start = std::time::Instant::now();
    let fb_shapes = build_shapes(builder, &region.items.shapes);
    log_flatbuffer_profile_phase(profile, "build_shapes", phase_start);
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!("[Serialize] building fb operators...");
    }
    let phase_start = std::time::Instant::now();
    let fb_operators = build_operators(builder, &region.items.operators);
    log_flatbuffer_profile_phase(profile, "build_operators", phase_start);
    if crate::config::tosa_engine_debug_enabled() {
        eprintln!("[Serialize] assembling graph...");
    }

    let phase_start = std::time::Instant::now();
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
    log_flatbuffer_profile_phase(profile, "build_region_vectors", phase_start);

    let phase_start = std::time::Instant::now();
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
    let region_offset = builder.end_table(start);
    log_flatbuffer_profile_phase(profile, "assemble_region", phase_start);
    region_offset
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
    PyBytes::new_bound(py, builder.finished_data()).unbind()
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
        let name = builder.create_string(&t.name);
        let shape = builder.create_vector(&t.shape);
        let data = t
            .data
            .as_ref()
            .filter(|data| !data.is_empty())
            .map(|data| create_force_aligned_u8_vector(builder, data));
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

        let (attr_type, attr_offset) = build_attribute(builder, &op.attribute);

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
