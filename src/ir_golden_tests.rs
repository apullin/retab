use crate::fx_convert_sum_ir::IrConvertSumPass;
use crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass;
use crate::fx_ir::{IrArg, IrGraph, NodeId};
use crate::fx_memory_format_ir::IrToTosaMemoryFormatPass;
use crate::test_ir::{
    assert_live_signature, f32_tensor, live_node, meta, parameter, qparam, IrFixture,
};

#[test]
fn convert_sum_decomposes_multi_axis_reduction_with_qparam_contract() {
    let mut fixture = IrFixture::new();
    let input = fixture.f32_placeholder("x", [2, 3, 4]);
    let sum = fixture.f32_call(
        "sum",
        "aten.sum.dim_IntList",
        vec![
            IrArg::Node(input),
            IrArg::List(vec![IrArg::Int(0), IrArg::Int(2)]),
            IrArg::Bool(false),
        ],
        [3],
    );
    fixture
        .graph_mut()
        .node_mut(sum)
        .meta
        .set_single_input_qparam(qparam(0.25, 3, "torch.int8"));
    fixture
        .graph_mut()
        .node_mut(sum)
        .meta
        .set_single_output_qparam(qparam(0.5, -7, "torch.int8"));
    fixture.output(sum);

    assert!(fixture.run(&IrConvertSumPass));

    assert_live_signature(
        fixture.graph(),
        &[
            "x|Placeholder|x|args=[]|shape=[2,3,4]|dtype=torch.float32|iqp=-|oqp=-",
            "sum_reduce_dim2|CallFunction|aten.sum.dim_IntList|args=[%x,[2],false]|shape=[2,3]|dtype=torch.float32|iqp={0:torch.int8@0.25:3}|oqp={0:torch.int8@0.5:-7}",
            "sum_reduce_dim0|CallFunction|aten.sum.dim_IntList|args=[%sum_reduce_dim2,[0],false]|shape=[3]|dtype=torch.float32|iqp={0:torch.int8@0.25:3}|oqp={0:torch.int8@0.5:-7}",
            "output|Output|output|args=[%sum_reduce_dim0]|shape=-|dtype=-|iqp=-|oqp=-",
        ],
    );
}

#[test]
fn qdq_rescale_rewrite_has_stable_boundary_shape() {
    let mut fixture = IrFixture::new();
    let input = fixture.placeholder("input", [1, 3, 4, 4], "torch.int8");
    let dq = fixture.f32_call(
        "dq",
        "quantized_decomposed.dequantize_per_tensor.default",
        vec![
            IrArg::Node(input),
            IrArg::Float(0.5),
            IrArg::Int(3),
            IrArg::Int(-128),
            IrArg::Int(127),
            IrArg::Dtype("torch.int8".to_string()),
        ],
        [1, 3, 4, 4],
    );
    let q = fixture.call(
        "q",
        "quantized_decomposed.quantize_per_tensor.default",
        vec![
            IrArg::Node(dq),
            IrArg::Float(0.25),
            IrArg::Int(-7),
            IrArg::Int(-128),
            IrArg::Int(127),
            IrArg::Dtype("torch.int8".to_string()),
        ],
        [1, 3, 4, 4],
        "torch.int8",
    );
    let user = fixture.call(
        "user",
        "aten.view_copy.default",
        vec![IrArg::Node(q)],
        [1, 48],
        "torch.int8",
    );
    fixture.output(user);

    assert!(fixture.run(&IrInsertQdqRescalePass));

    assert_live_signature(
        fixture.graph(),
        &[
            "input|Placeholder|input|args=[]|shape=[1,3,4,4]|dtype=torch.int8|iqp=-|oqp=-",
            "q_rescale|CallFunction|backend.tosa.RESCALE.default|args=[%input,dtype(torch.int8),[2],3,-7]|shape=[1,3,4,4]|dtype=torch.int8|iqp={0:torch.int8@0.5:3}|oqp={0:torch.int8@0.25:-7}",
            "user|CallFunction|aten.view_copy.default|args=[%q_rescale]|shape=[1,48]|dtype=torch.int8|iqp=-|oqp=-",
            "output|Output|output|args=[%user]|shape=-|dtype=-|iqp=-|oqp=-",
        ],
    );
}

// ---------------------------------------------------------------------------
// Layout: which rank>=4 tensors end up channels-last, and where the permutes land.
//
// RETAB has to reproduce stock ExecuTorch's choice. Stock keeps the delegate interior
// logical and wraps each layout anchor (convolution, pooling, resize) in a local
// channels-last island; an elementwise run merges into a neighbouring island only when
// bounded by an anchor on every edge, and a run reaching the delegate boundary is
// rejected outright.
// ---------------------------------------------------------------------------

const NHWC: [usize; 4] = [0, 2, 3, 1];
const LOGICAL: [usize; 4] = [0, 1, 2, 3];

fn dim_order(graph: &IrGraph, name: &str) -> Vec<usize> {
    live_node(graph, name)
        .meta
        .cloned_tosa_dim_order()
        .unwrap_or_else(|| panic!("{name} carries no tosa dim order"))
}

/// Every inserted layout permute, as (transposed node name, permutation).
fn transposes(graph: &IrGraph) -> Vec<(String, Vec<i64>)> {
    graph
        .nodes
        .iter()
        .filter(|node| node.alive && node.target == "backend.tosa.TRANSPOSE.default")
        .map(|node| {
            let source = node
                .first_node_arg()
                .map(|id| graph.nodes[id].name.clone())
                .unwrap_or_default();
            let perm = node
                .args
                .get(1)
                .and_then(|arg| arg.as_int_list())
                .unwrap_or_default();
            (source, perm)
        })
        .collect()
}

fn rank4_param(fixture: &mut IrFixture, name: &str, shape: [i64; 4]) -> NodeId {
    let count = shape.iter().product::<i64>() as usize;
    let id = parameter(
        fixture.graph_mut(),
        name,
        f32_tensor(shape.to_vec(), &vec![1.0; count]),
    );
    fixture.graph_mut().node_mut(id).meta = meta(shape.to_vec(), "torch.float32");
    id
}

/// `conv -> mul -> output`: the run after the convolution reaches the delegate output, so
/// stock leaves it -- and its rank-4 constant operand -- in logical order and puts the
/// channels-last exit right after the convolution. This is the LayerNorm affine defect in
/// miniature: annotating the run channels-last is what used to bake an NHWC-permuted
/// payload into that constant.
#[test]
fn span_reaching_the_graph_output_stays_logical() {
    let mut fixture = IrFixture::new();
    let x = fixture.f32_placeholder("x", [1, 3, 8, 8]);
    let weight = rank4_param(&mut fixture, "weight", [4, 3, 3, 3]);
    let conv = fixture.f32_call(
        "conv",
        "aten.convolution.default",
        vec![IrArg::Node(x), IrArg::Node(weight)],
        [1, 4, 8, 8],
    );
    let gamma = rank4_param(&mut fixture, "gamma", [1, 4, 8, 8]);
    let scaled = fixture.f32_call(
        "scaled",
        "aten.mul.Tensor",
        vec![IrArg::Node(conv), IrArg::Node(gamma)],
        [1, 4, 8, 8],
    );
    fixture.output(scaled);

    assert!(fixture.run(&IrToTosaMemoryFormatPass));
    let graph = fixture.graph();

    assert_eq!(dim_order(graph, "conv"), NHWC, "the anchor is channels-last");
    assert_eq!(
        dim_order(graph, "scaled"),
        LOGICAL,
        "a run that reaches the delegate output stays logical"
    );
    assert_eq!(
        dim_order(graph, "gamma"),
        LOGICAL,
        "the constant follows the run, so its payload is never permuted"
    );
    assert_eq!(
        dim_order(graph, "weight"),
        NHWC,
        "the convolution's own weight still goes channels-last"
    );
    assert_eq!(
        transposes(graph),
        vec![
            ("x".to_string(), vec![0, 2, 3, 1]),
            ("conv".to_string(), vec![0, 3, 1, 2]),
        ],
        "channels-last is entered before the convolution and left right after it"
    );
}

/// The same run bounded by a second convolution instead of the output: now both edges
/// touch an anchor, so it merges into the island and no permute sits between them.
#[test]
fn span_bounded_by_anchors_stays_channels_last() {
    let mut fixture = IrFixture::new();
    let x = fixture.f32_placeholder("x", [1, 3, 8, 8]);
    let weight = rank4_param(&mut fixture, "weight", [4, 3, 3, 3]);
    let conv = fixture.f32_call(
        "conv",
        "aten.convolution.default",
        vec![IrArg::Node(x), IrArg::Node(weight)],
        [1, 4, 8, 8],
    );
    let gamma = rank4_param(&mut fixture, "gamma", [1, 4, 8, 8]);
    let scaled = fixture.f32_call(
        "scaled",
        "aten.mul.Tensor",
        vec![IrArg::Node(conv), IrArg::Node(gamma)],
        [1, 4, 8, 8],
    );
    let weight2 = rank4_param(&mut fixture, "weight2", [4, 4, 3, 3]);
    let conv2 = fixture.f32_call(
        "conv2",
        "aten.convolution.default",
        vec![IrArg::Node(scaled), IrArg::Node(weight2)],
        [1, 4, 8, 8],
    );
    fixture.output(conv2);

    assert!(fixture.run(&IrToTosaMemoryFormatPass));
    let graph = fixture.graph();

    assert_eq!(
        dim_order(graph, "scaled"),
        NHWC,
        "a run bounded by anchors on both edges merges into the island"
    );
    assert_eq!(dim_order(graph, "gamma"), NHWC, "its constant follows it");
    assert_eq!(
        transposes(graph),
        vec![
            ("x".to_string(), vec![0, 2, 3, 1]),
            ("conv2".to_string(), vec![0, 3, 1, 2]),
        ],
        "one permute in at the start, one out at the end, none in between"
    );
}

/// A reshape that only moves unit dims -- `[1, 6] -> [1, 6, 1, 1]`, which is how a
/// decomposed `linear` feeds its convolution -- is layout-transparent: channels-last is a
/// free relabel there. Annotating it logical would make the convolution read the operand
/// as NHWC and compute a bogus kernel extent, and inserting a permute for it would put an
/// op in the graph that stock does not emit.
#[test]
fn unit_dim_reshape_into_an_anchor_needs_no_permute() {
    let mut fixture = IrFixture::new();
    let x = fixture.f32_placeholder("x", [1, 6]);
    let input_view = fixture.f32_call(
        "input_view",
        "aten.view_copy.default",
        vec![
            IrArg::Node(x),
            IrArg::List(vec![
                IrArg::Int(1),
                IrArg::Int(6),
                IrArg::Int(1),
                IrArg::Int(1),
            ]),
        ],
        [1, 6, 1, 1],
    );
    let weight = rank4_param(&mut fixture, "weight", [5, 6, 1, 1]);
    let conv = fixture.f32_call(
        "conv",
        "aten.convolution.default",
        vec![IrArg::Node(input_view), IrArg::Node(weight)],
        [1, 5, 1, 1],
    );
    let output_view = fixture.f32_call(
        "output_view",
        "aten.view_copy.default",
        vec![
            IrArg::Node(conv),
            IrArg::List(vec![IrArg::Int(1), IrArg::Int(5)]),
        ],
        [1, 5],
    );
    fixture.output(output_view);

    assert!(fixture.run(&IrToTosaMemoryFormatPass));
    let graph = fixture.graph();

    assert_eq!(
        dim_order(graph, "input_view"),
        NHWC,
        "the reshape is relabelled channels-last for the convolution, not permuted"
    );
    assert_eq!(
        dim_order(graph, "weight"),
        NHWC,
        "so the convolution sees operand and kernel in the same order"
    );
    assert!(
        transposes(graph).is_empty(),
        "no data moves anywhere in this graph, so no permute is emitted: {:?}",
        transposes(graph)
    );
}
