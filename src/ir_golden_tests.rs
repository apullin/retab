use crate::fx_convert_sum_ir::IrConvertSumPass;
use crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass;
use crate::fx_ir::IrArg;
use crate::test_ir::{assert_live_signature, qparam, IrFixture};

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
            "sum_reduce_dim2|CallFunction|aten.sum.dim_IntList|args=[%x,[2],true]|shape=[2,3,1]|dtype=torch.float32|iqp={0:torch.int8@0.25:3}|oqp={0:torch.int8@0.5:-7}",
            "sum_reduce_dim0|CallFunction|aten.sum.dim_IntList|args=[%sum_reduce_dim2,[0],true]|shape=[1,3,1]|dtype=torch.float32|iqp={0:torch.int8@0.5:-7}|oqp={0:torch.int8@0.5:-7}",
            "sum|CallFunction|aten.view_copy.default|args=[%sum_reduce_dim0,[3]]|shape=[3]|dtype=torch.float32|iqp={0:torch.int8@0.25:3}|oqp={0:torch.int8@0.5:-7}",
            "output|Output|output|args=[%sum]|shape=-|dtype=-|iqp=-|oqp=-",
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
