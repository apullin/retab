"""End-to-end smoke test: lower a small quantized CNN for Ethos-U55 with RETAB active."""

import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.executorch]


class TinyCnn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def _quantized_program():
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    torch.manual_seed(0)
    model, example = TinyCnn().eval(), torch.randn(1, 1, 28, 28)
    compile_spec = EthosUCompileSpec(
        target="ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    qmodule = prepare_pt2e(torch.export.export(model, (example,), strict=True).module(), quantizer)
    with torch.no_grad():
        for _ in range(3):
            qmodule(torch.randn_like(example))
    qmodule = convert_pt2e(qmodule)
    return torch.export.export(qmodule, (example,), strict=True), compile_spec


def test_tiny_cnn_lowers_through_retab():
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    from retab.integration.executorch_arm import activated
    from retab.preprocess import preprocess_stats

    quantized_ep, compile_spec = _quantized_program()
    before = preprocess_stats()
    with activated():
        edge = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[EthosUPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    after = preprocess_stats()
    pte = edge.to_executorch().buffer

    assert after["retab"] > before["retab"], "RETAB's preprocess never ran"
    assert after["stock_fallback"] == before["stock_fallback"], "the graph fell back to stock"
    assert len(pte) > 0
