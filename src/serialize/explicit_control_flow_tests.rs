//! Serializer tests for explicit layout mode's control flow (APU-135 P2 design S8). Each
//! test takes a hand-built parent and its regions through the boundary contracts,
//! ingest and serialization, as `ingest_and_serialize` does.

use std::collections::BTreeMap;

use super::collection::{check_explicit_layout_invariants, collect_regions, SerializedRegion};
use super::tosa_types::{Op, TosaTensor};
use super::visitors;
use crate::control_flow::fixtures::{rank4_hierarchy, RANK4};
use crate::fx_ir::IrGraph;
use crate::lowering_effects::LoweringEffectBatch;
use crate::pipeline_config::LayoutMode;
use crate::store::TensorStore;
use crate::StashedPipeline;

/// Serialize `parent` and its regions in `layout` as `ingest_and_serialize` does. In
/// explicit mode, also check D1 on every graph: no dim order, and every node serializes
/// at its val's shape.
fn serialize_hierarchy(
    (mut parent, mut submodules): (IrGraph, BTreeMap<String, StashedPipeline>),
    layout: LayoutMode,
) -> Result<Vec<SerializedRegion>, String> {
    crate::control_flow::apply_boundary_contracts(&mut parent, &mut submodules, layout);
    serialize_contracted(&parent, &submodules, layout)
}

/// [`serialize_hierarchy`] for a hierarchy whose boundary contracts have run.
fn serialize_contracted(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    layout: LayoutMode,
) -> Result<Vec<SerializedRegion>, String> {
    let mut store = TensorStore::new();
    let mut main = crate::ingest::from_ir::graph_from_ir(parent, &mut store);
    main.layout = layout;
    crate::ingest::from_ir::apply_effects(&mut main, &mut store, &LoweringEffectBatch::default());
    crate::ingest::finalize::finalize_converted_graph(&mut main);
    let mut graphs = vec![("main".to_string(), main)];
    crate::control_flow::try_append_subgraphs_from_ir(
        &mut graphs,
        &mut store,
        parent,
        submodules,
        layout,
    )?;
    for (_, graph) in &mut graphs {
        graph.rebuild_user_lists();
    }
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    if layout == LayoutMode::Explicit {
        for (name, graph) in &graphs {
            check_explicit_layout_invariants(name, graph, &store)?;
        }
    }
    collect_regions(&graphs, &store)
}

fn tensor<'a>(region: &'a SerializedRegion, name: &str) -> &'a TosaTensor {
    region
        .items
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor named {name} in {}", region.name))
}

fn count(region: &SerializedRegion, op: Op) -> usize {
    region.items.operators.iter().filter(|o| o.op == op).count()
}

/// S8 (design §4.3.1): a rank-4 cond or while output keeps its val in explicit mode, as
/// in rc4, where a region's IO takes the parent's vals. The boundary contracts used to
/// give it the NHWC dim order of dim-order mode's channels-last regions in either mode,
/// and explicit serialization sent every such graph to stock (`reject_dim_orders`).
/// Dim-order mode is unchanged: there the output is NHWC, and the main graph TRANSPOSEs
/// it back to the user's layout.
#[test]
fn rank4_control_flow_outputs_keep_their_vals_in_explicit_mode() {
    let logical: Vec<i32> = RANK4.iter().map(|&dim| dim as i32).collect();
    for while_loop in [false, true] {
        let op = if while_loop {
            Op::WhileLoop
        } else {
            Op::CondIf
        };

        let regions = serialize_hierarchy(rank4_hierarchy(while_loop), LayoutMode::Explicit)
            .unwrap_or_else(|error| panic!("{op:?}: {error}"));
        let main = &regions[0];
        assert_eq!(main.name, "main");
        assert_eq!(count(main, op), 1, "{op:?}");
        assert_eq!(tensor(main, "getitem").shape, logical, "{op:?}");
        assert_eq!(main.items.output_names, ["getitem"], "{op:?}");
        assert_eq!(regions.len(), 3, "{op:?}");
        for region in &regions {
            assert_eq!(count(region, Op::Transpose), 0, "{op:?} {}", region.name);
        }
        for region in &regions[1..] {
            let input = &region.items.input_names[0];
            assert_eq!(
                tensor(region, input).shape,
                logical,
                "{op:?} {}",
                region.name
            );
            assert_eq!(
                region.items.output_names,
                std::slice::from_ref(input),
                "{op:?}"
            );
        }

        let regions = serialize_hierarchy(rank4_hierarchy(while_loop), LayoutMode::DimOrder)
            .unwrap_or_else(|error| panic!("{op:?}: {error}"));
        let main = &regions[0];
        assert_eq!(tensor(main, "getitem").shape, [2, 2, 2, 3], "{op:?}");
        assert_eq!(count(main, Op::Transpose), 1, "{op:?}");
        assert_eq!(
            tensor(main, &main.items.output_names[0]).shape,
            logical,
            "{op:?}"
        );
    }
}

/// Critique B2 and design S8: a cond whose branches run an avg pool, then a conv whose
/// weight and bias they capture from the parent. rc4's RewriteAvgPool2dPass makes its
/// zero points constants at the graph head. Made as placeholders in a region, they
/// would become runtime inputs and shift the positional pairing of the branch inputs
/// with the cond's operands, which the boundary contracts and captured_conv rely on.
/// The region passes keep the branch inputs (the zero points are `get_attr`
/// constants); captured_conv then resolves each conv's weight and bias to the
/// parent's `w` and `b`; and each branch serializes with the cond's three operands as
/// its only inputs.
#[test]
fn a_region_avg_pool_and_captured_conv_keep_the_branch_inputs_aligned() {
    use crate::fx_constant_tensor_eval::permute_tensor_data;
    use crate::fx_ir::{i64_list_arg, IrArg, IrPass, NodeMeta, NodeOp};
    use crate::graph::ir::DType;
    use crate::layout::constants::IrMaterializeConstantsPass;
    use crate::layout::rewrite_avg_pool2d::IrRewriteAvgPool2dPass;
    use crate::layout::rewrite_conv::IrRewriteConvPass;
    use crate::store::TensorData;
    use crate::test_ir::{buffer, call, call_with_meta, live_node, output, placeholder, qparam};

    const I8: &str = "torch.int8";
    const F32: &str = "torch.float32";
    const QUANTIZE: &str = "quantized_decomposed.quantize_per_tensor.default";
    const BRANCHES: [&str; 2] = ["true_graph", "false_graph"];
    let int8 = |shape: &[i64], values: Vec<i8>| TensorData {
        data: values.into_iter().map(|value| value as u8).collect(),
        shape: shape.to_vec(),
        dtype: DType::Int8,
    };
    let weight = int8(
        &[4, 2, 3, 3],
        (0..72).map(|v| (v % 23 - 11) as i8).collect(),
    );
    let bias = int8(&[4], vec![13, -7, 3, 0]);

    // A branch after IrFoldQdq: the captured weight and bias read through their
    // boundary quantizes, at the conv's own weight and bias qparams.
    let branch = || {
        let mut region = IrGraph::empty();
        let x = placeholder(&mut region, "x_ph", [1, 2, 8, 8], I8);
        region
            .node_mut(x)
            .meta
            .set_single_output_qparam(qparam(0.1, -3, I8));
        let w = placeholder(&mut region, "w_ph", [4, 2, 3, 3], I8);
        let b = placeholder(&mut region, "b_ph", [4], I8);
        let mut pool_meta = NodeMeta::tensor(vec![1, 2, 4, 4], F32);
        pool_meta.set_single_input_qparam(qparam(0.1, -3, I8));
        pool_meta.set_single_output_qparam(qparam(0.2, 5, I8));
        let pool = call_with_meta(
            &mut region,
            "avg_pool",
            "aten.avg_pool2d.default",
            vec![IrArg::Node(x), i64_list_arg(&[2, 2]), i64_list_arg(&[2, 2])],
            pool_meta,
        );
        let mut quantize = |name: &str, input, shape: &[i64], scale| {
            call(
                &mut region,
                name,
                QUANTIZE,
                vec![
                    IrArg::Node(input),
                    IrArg::Float(scale),
                    IrArg::Int(0),
                    IrArg::Int(-128),
                    IrArg::Int(127),
                    IrArg::Dtype(I8.into()),
                ],
                shape.to_vec(),
                I8,
            )
        };
        let w_q = quantize("w_ph_quantize", w, &[4, 2, 3, 3], 0.04);
        let b_q = quantize("b_ph_quantize", b, &[4], 0.01);
        let mut conv_meta = NodeMeta::tensor(vec![1, 4, 4, 4], F32);
        conv_meta.set_input_qparams_from([
            (0, qparam(0.2, 5, I8)),
            (1, qparam(0.04, 0, I8)),
            (2, qparam(0.01, 0, I8)),
        ]);
        conv_meta.set_single_output_qparam(qparam(0.3, -2, I8));
        let conv = call_with_meta(
            &mut region,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(pool),
                IrArg::Node(w_q),
                IrArg::Node(b_q),
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            conv_meta,
        );
        output(&mut region, conv);
        region
    };
    let rewrite_conv = IrRewriteConvPass {
        is_u55_subset: false,
        in_region: true,
        u55_split_permutes: Default::default(),
    };
    let mut submodules: BTreeMap<String, StashedPipeline> = BTreeMap::new();
    for name in BRANCHES {
        let mut region = branch();
        let before = region.live_placeholder_names();
        crate::stock_fallback::take();
        assert!(IrRewriteAvgPool2dPass.run(&mut region));
        assert!(IrMaterializeConstantsPass { in_region: true }.run(&mut region));
        assert!(rewrite_conv.run(&mut region));
        assert_eq!(crate::stock_fallback::take(), None);
        assert_eq!(
            region.live_placeholder_names(),
            before,
            "the branch inputs stay"
        );
        let pool = region
            .live_nodes()
            .find(|node| node.target == "backend.tosa.AVG_POOL2D.default")
            .expect("a TOSA avg pool");
        for zero_point in [1, 2] {
            let constant = region.node(pool.arg_node(zero_point).expect("a zero point"));
            assert_eq!(constant.op, NodeOp::GetAttr, "{}", constant.name);
            assert!(constant.tensor.is_some(), "{}", constant.name);
        }
        submodules.insert(name.to_string(), (region, LoweringEffectBatch::default()));
    }

    // The parent: a cond over `x` and the captured constants, which the quantizer
    // shares one operand qparam between.
    let mut parent = IrGraph::empty();
    let parent_x = placeholder(&mut parent, "x", [1, 2, 8, 8], I8);
    let parent_w = buffer(&mut parent, "w", weight.clone());
    parent.node_mut(parent_w).meta = NodeMeta::tensor(vec![4, 2, 3, 3], I8);
    let parent_b = buffer(&mut parent, "b", bias);
    parent.node_mut(parent_b).meta = NodeMeta::tensor(vec![4], I8);
    let mut cond_meta = NodeMeta::default();
    cond_meta.set_input_qparams_from([(3, qparam(0.05, 0, I8))]);
    let cond = call_with_meta(
        &mut parent,
        "cond",
        "torch.ops.higher_order.cond",
        vec![
            IrArg::Int(1),
            IrArg::Str(BRANCHES[0].into()),
            IrArg::Str(BRANCHES[1].into()),
            IrArg::List(vec![
                IrArg::Node(parent_x),
                IrArg::Node(parent_w),
                IrArg::Node(parent_b),
            ]),
        ],
        cond_meta,
    );
    let getitem = call(
        &mut parent,
        "getitem",
        "<built-in function getitem>",
        vec![IrArg::Node(cond), IrArg::Int(0)],
        [1, 4, 4, 4],
        I8,
    );
    output(&mut parent, getitem);
    assert_eq!(
        crate::apply_control_flow_contracts(&mut parent, &mut submodules, LayoutMode::Explicit),
        Ok(())
    );

    let ohwi = permute_tensor_data(&weight, &[0, 2, 3, 1]).expect("a permutable weight");
    assert_eq!(ohwi.shape, [4, 3, 3, 2]);
    // The bias, at 0.05 through the boundary, requantized at s_in * s_w = 0.2 * 0.04.
    let int32_bias: Vec<i32> = [13, -7, 3, 0]
        .iter()
        .map(|&q| ((q as f32 * 0.05f32) / (0.2f32 * 0.04f32)).round_ties_even() as i32)
        .collect();
    for name in BRANCHES {
        // The weight: the parent's `w` in OHWI, then its boundary RESCALE (0.05 to
        // 0.04).
        let region = &submodules[name].0;
        let conv = live_node(region, "conv_tosa");
        let rescale = region.node(conv.arg_node(1).expect("a weight"));
        assert!(rescale.is_rescale(), "{name}: {}", rescale.name);
        assert_eq!(
            format!("{:?}", rescale.args[2]),
            format!("{:?}", IrArg::List(vec![IrArg::Float(0.05 / 0.04)])),
            "{name}"
        );
        let constant = region.node(rescale.first_node_arg().expect("an input"));
        assert_eq!(constant.op, NodeOp::GetAttr, "{name}");
        let data = &constant.tensor.as_ref().expect("constant data").data;
        assert_eq!(
            (&data.shape, data.dtype, &data.data),
            (&ohwi.shape, ohwi.dtype, &ohwi.data),
            "{name}"
        );
        let bias = region.node(conv.arg_node(2).expect("a bias"));
        assert_eq!(bias.op, NodeOp::GetAttr, "{name}");
        let values: Vec<i32> = bias
            .tensor
            .as_ref()
            .expect("int32 data")
            .data
            .data
            .chunks(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(values, int32_bias, "{name}");
    }

    let regions = serialize_contracted(&parent, &submodules, LayoutMode::Explicit)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(regions.len(), 3);
    for branch in &regions[1..] {
        let label = &branch.name;
        assert_eq!(
            branch.items.input_names.len(),
            3,
            "{label}: {:?}",
            branch.items.input_names
        );
        assert_eq!(count(branch, Op::AvgPool2d), 1, "{label}");
        assert_eq!(count(branch, Op::Conv2d), 1, "{label}");
        let pool = branch
            .items
            .operators
            .iter()
            .find(|operator| operator.op == Op::AvgPool2d)
            .expect("an AVG_POOL2D");
        for zero_point in &pool.inputs[1..3] {
            assert!(
                tensor(branch, zero_point).data.is_some(),
                "{label}: {zero_point}"
            );
            assert!(
                !branch.items.input_names.contains(zero_point),
                "{label}: {zero_point}"
            );
        }
    }
}
