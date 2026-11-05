"""Placeholder import and input-spec order regressions."""

import pytest
import torch

pytestmark = pytest.mark.native


@pytest.mark.parametrize(
    "placeholder_order",
    [
        ("input", "constant", "buffer", "parameter"),
        ("constant", "parameter", "input", "buffer"),
    ],
)
def test_placeholder_tensor_bindings_follow_graph_signature(placeholder_order):
    import retab
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from retab.preprocess import _build_pipeline_config
    from torch.export.exported_program import ExportGraphSignature
    from tosa.TosaGraph import TosaGraph

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
            self.register_buffer("offset", torch.tensor([3.0, 4.0]))
            self.constant = torch.tensor([5.0, 6.0])

        def forward(self, x):
            return x + self.weight + self.offset + self.constant

    ep = torch.export.export(Module(), (torch.ones(2),))
    signature = ep.graph_signature
    names = {
        "parameter": next(iter(signature.inputs_to_parameters)),
        "buffer": next(iter(signature.inputs_to_buffers)),
        "constant": next(iter(signature.inputs_to_lifted_tensor_constants)),
        "input": signature.user_inputs[0],
    }
    spec_by_name = {spec.arg.name: spec for spec in signature.input_specs}
    expected_data = {
        name: ep.state_dict[target].detach().numpy().tobytes()
        for mapping in [signature.inputs_to_parameters, signature.inputs_to_buffers]
        for name, target in mapping.items()
    }
    expected_data.update(
        {
            name: ep.constants[target].numpy().tobytes()
            for name, target in signature.inputs_to_lifted_tensor_constants.items()
        }
    )
    graph = ep.graph_module.graph
    placeholders = {node.name: node for node in graph.nodes if node.op == "placeholder"}
    first_call = next(node for node in graph.nodes if node.op == "call_function")
    for kind in placeholder_order:
        first_call.prepend(placeholders[names[kind]])
    ep.graph_module.recompile()
    placeholder_names = [node.name for node in graph.nodes if node.op == "placeholder"]
    ep._graph_signature = ExportGraphSignature(
        [spec_by_name[name] for name in placeholder_names], signature.output_specs
    )

    ordered_names = [names[kind] for kind in placeholder_order]
    assert [spec.arg.name for spec in ep.graph_signature.input_specs] == ordered_names
    assert all(
        spec is spec_by_name[spec.arg.name] for spec in ep.graph_signature.input_specs
    )
    assert ep.graph_signature.user_inputs == (names["input"],)

    retab.run_fx_passes(
        ep.graph_module,
        ep,
        pipeline_config=_build_pipeline_config(TosaCompileSpec("TOSA-1.0+INT")),
    )
    tosa_bytes = retab.ingest_and_serialize(
        ep.graph_module, dict(ep.state_dict), ep.graph_signature, ep
    )
    block = TosaGraph.GetRootAs(tosa_bytes, 0).Regions(0).Blocks(0)
    assert [block.Inputs(i).decode() for i in range(block.InputsLength())] == [
        names["input"]
    ]
    tensors = {
        block.Tensors(i).Name().decode(): block.Tensors(i)
        for i in range(block.TensorsLength())
    }
    for name, expected in expected_data.items():
        tensor = tensors[name]
        assert bytes(tensor.Data(i) for i in range(tensor.DataLength())) == expected
