import torch


def _make_per_channel_constant_quant_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    scale = graph.get_attr("scale")
    zero_point = graph.get_attr("zero_point")
    quantized = graph.call_function(
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        (weight, scale, zero_point, 0, -128, 127, torch.int8),
    )
    dequantized = graph.call_function(
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
        (quantized, scale, zero_point, 0, -128, 127, torch.int8),
    )
    summed = graph.call_function(torch.ops.aten.add.Tensor, (x, dequantized))
    graph.output(summed)

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(3, 4))
            self.register_buffer("scale", torch.ones(3) * 0.125)
            self.register_buffer("zero_point", torch.zeros(3, dtype=torch.int64))

    gm = torch.fx.GraphModule(Holder(), graph)
    graph.lint()
    gm.recompile()
    return gm


def test_pt2e_fastpath_installs_and_restores_constant_fold():
    import torchao.quantization.pt2e.quantize_pt2e as quantize_pt2e

    from retab.integration import torchao_pt2e

    original = quantize_pt2e.constant_fold
    torchao_pt2e.install()
    try:
        assert quantize_pt2e.constant_fold is not original
        assert getattr(quantize_pt2e.constant_fold, "__retab_patched__", False)
    finally:
        torchao_pt2e.uninstall()

    assert quantize_pt2e.constant_fold is original


def test_pt2e_fastpath_folds_per_channel_constant_quant_like_torchao():
    import torchao.quantization.pt2e.quantize_pt2e as quantize_pt2e

    from retab.integration import torchao_pt2e

    original = quantize_pt2e.constant_fold
    fast_gm = _make_per_channel_constant_quant_graph()
    reference_gm = _make_per_channel_constant_quant_graph()
    reference_gm.load_state_dict(fast_gm.state_dict())

    original(reference_gm, quantize_pt2e._quant_node_constraint)

    torchao_pt2e.install()
    try:
        quantize_pt2e.constant_fold(
            fast_gm, quantize_pt2e._quant_node_constraint
        )
    finally:
        torchao_pt2e.uninstall()

    assert fast_gm.meta["_retab_fast_quant_constant_fold_count"] == 1
    assert all(
        node.target
        is not torch.ops.quantized_decomposed.quantize_per_channel.default
        for node in fast_gm.graph.nodes
    )
    input_tensor = torch.randn(3, 4)
    torch.testing.assert_close(fast_gm(input_tensor), reference_gm(input_tensor))
