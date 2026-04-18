#!/usr/bin/env python3
"""Benchmark stock ExecuTorch Arm lowering against RETAB.

This is the product-facing comparison script:
  - same prepared quantized ExportedProgram
  - same Ethos-U compile spec
  - stock ExecuTorch Arm lowering
  - RETAB-activated lowering via the supported integration surface

It reports median lower time and the final `.pte` artifact hashes so the
comparison is not just a speed number.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tosa_serializer as ts
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision import models  # type: ignore[import-untyped]

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from retab.integration.executorch_arm import activated, uninstall


@dataclass
class RunSummary:
    label: str
    times_s: list[float]
    pte_hash: str
    pte_bytes: int

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)


def require_native_stock_serializer() -> None:
    serializer_path = Path(ts.__file__).resolve()
    if serializer_path.name == "tosa_serializer.py":
        raise SystemExit(
            "error: stock ExecuTorch Arm benchmark requires a native "
            "tosa_serializer install. The current environment is using the "
            "source-checkout shim for RETAB fast-path bootstrap."
        )


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


def build_model(model_name: str) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)

    if model_name == "tiny_cnn":
        return TinyCnn().eval(), torch.randn(1, 1, 28, 28)
    if model_name == "resnet18":
        return models.resnet18(weights=None).eval(), torch.randn(1, 3, 224, 224)
    if model_name == "mobilenet_v3_small":
        return (
            models.mobilenet_v3_small(weights=None).eval(),
            torch.randn(1, 3, 232, 232),
        )

    raise ValueError(f"unsupported model {model_name!r}")


def prepare_quantized_ep(
    model_name: str, calibration_runs: int
) -> tuple[torch.export.ExportedProgram, EthosUCompileSpec]:
    model, example_input = build_model(model_name)
    exported = torch.export.export(model, (example_input,), strict=True)

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
            quantized_module(*((example_input,)))
    quantized_module = convert_pt2e(quantized_module)

    quantized_ep = torch.export.export(quantized_module, (example_input,), strict=True)
    return quantized_ep, compile_spec


def lower_once(
    quantized_ep: torch.export.ExportedProgram,
    compile_spec: EthosUCompileSpec,
    *,
    use_retab: bool,
) -> tuple[bytes, float]:
    ep = copy.deepcopy(quantized_ep)
    partitioner = EthosUPartitioner(compile_spec)

    start = time.perf_counter()
    if use_retab:
        with activated():
            edge = to_edge_transform_and_lower(
                ep,
                partitioner=[partitioner],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
    else:
        uninstall()
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    elapsed_s = time.perf_counter() - start
    return edge.to_executorch().buffer, elapsed_s


def run_benchmark(
    label: str,
    quantized_ep: torch.export.ExportedProgram,
    compile_spec: EthosUCompileSpec,
    *,
    use_retab: bool,
    warmup: int,
    runs: int,
) -> RunSummary:
    warm_hash = ""
    warm_size = 0
    for _ in range(warmup):
        pte, _ = lower_once(quantized_ep, compile_spec, use_retab=use_retab)
        warm_hash = hashlib.sha256(pte).hexdigest()
        warm_size = len(pte)

    times_s: list[float] = []
    pte_hash = warm_hash
    pte_bytes = warm_size
    for _ in range(runs):
        pte, elapsed_s = lower_once(quantized_ep, compile_spec, use_retab=use_retab)
        times_s.append(elapsed_s)
        if not pte_hash:
            pte_hash = hashlib.sha256(pte).hexdigest()
            pte_bytes = len(pte)

    return RunSummary(
        label=label,
        times_s=times_s,
        pte_hash=pte_hash,
        pte_bytes=pte_bytes,
    )


def format_times(times_s: list[float]) -> str:
    return ", ".join(f"{t:.3f}" for t in times_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stock ExecuTorch Arm lowering against RETAB."
    )
    parser.add_argument(
        "--model",
        default="mobilenet_v3_small",
        choices=["tiny_cnn", "resnet18", "mobilenet_v3_small"],
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--calibration-runs", type=int, default=5)
    parser.add_argument("--require-identical-artifact", action="store_true")
    args = parser.parse_args()

    require_native_stock_serializer()

    quantized_ep, compile_spec = prepare_quantized_ep(
        args.model, args.calibration_runs
    )

    stock = run_benchmark(
        "stock",
        quantized_ep,
        compile_spec,
        use_retab=False,
        warmup=args.warmup,
        runs=args.runs,
    )
    fast = run_benchmark(
        "retab",
        quantized_ep,
        compile_spec,
        use_retab=True,
        warmup=args.warmup,
        runs=args.runs,
    )

    same_artifact = stock.pte_hash == fast.pte_hash and stock.pte_bytes == fast.pte_bytes
    speedup = stock.median_s / fast.median_s if fast.median_s > 0 else float("inf")

    print(f"model={args.model}")
    print(f"runs={args.runs}")
    print(f"warmup={args.warmup}")
    print()
    print(f"stock_times_s={format_times(stock.times_s)}")
    print(f"stock_median_s={stock.median_s:.3f}")
    print(f"stock_pte_bytes={stock.pte_bytes}")
    print(f"stock_pte_sha256={stock.pte_hash}")
    print()
    print(f"retab_times_s={format_times(fast.times_s)}")
    print(f"retab_median_s={fast.median_s:.3f}")
    print(f"retab_pte_bytes={fast.pte_bytes}")
    print(f"retab_pte_sha256={fast.pte_hash}")
    print()
    print(f"speedup_x={speedup:.2f}")
    print(f"artifact_match={'yes' if same_artifact else 'no'}")

    if args.require_identical_artifact and not same_artifact:
        raise SystemExit("error: stock and RETAB artifacts differ")


if __name__ == "__main__":
    main()
