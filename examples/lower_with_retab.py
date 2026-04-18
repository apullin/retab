#!/usr/bin/env python3
"""Lower a small quantized model through ExecuTorch Arm using RETAB."""

import argparse
import time
from pathlib import Path

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from retab.integration.executorch_arm import activated


class TinyCnn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(model_name: str) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    if model_name != "tiny_cnn":
        raise ValueError(f"Unsupported model '{model_name}'")

    torch.manual_seed(0)
    model = TinyCnn().eval()
    example_inputs = (torch.randn(1, 1, 28, 28),)
    return model, example_inputs


def prepare_quantized_ep(
    model_name: str,
    calibration_runs: int,
) -> tuple[torch.export.ExportedProgram, EthosUCompileSpec]:
    model, example_inputs = build_model(model_name)
    exported = torch.export.export(model, example_inputs, strict=True)

    compile_spec = EthosUCompileSpec(
        target="ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    quantized_module = prepare_pt2e(exported.module(), quantizer)
    with torch.no_grad():
        for _ in range(calibration_runs):
            quantized_module(*example_inputs)
    quantized_module = convert_pt2e(quantized_module)

    quantized_ep = torch.export.export(quantized_module, example_inputs, strict=True)
    return quantized_ep, compile_spec


def lower_with_retab(
    quantized_ep: torch.export.ExportedProgram,
    compile_spec: EthosUCompileSpec,
) -> tuple[bytes, float]:
    partitioner = EthosUPartitioner(compile_spec)

    with activated():
        start = time.perf_counter()
        edge = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        lower_s = time.perf_counter() - start

    return edge.to_executorch().buffer, lower_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lower a small quantized model through ExecuTorch Arm using RETAB."
    )
    parser.add_argument("--model", default="tiny_cnn", choices=["tiny_cnn"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calibration-runs", type=int, default=5)
    args = parser.parse_args()

    quantized_ep, compile_spec = prepare_quantized_ep(
        args.model, args.calibration_runs
    )
    pte, lower_s = lower_with_retab(quantized_ep, compile_spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pte)

    print(f"model={args.model}")
    print(f"lower_s={lower_s:.3f}")
    print(f"pte_bytes={len(pte)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
