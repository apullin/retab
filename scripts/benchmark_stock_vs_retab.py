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


@dataclass(frozen=True)
class ModelEntry:
    name: str
    factory: callable
    input_shape: tuple[int, ...]


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


MODEL_ENTRIES: dict[str, ModelEntry] = {
    "tiny_cnn": ModelEntry("tiny_cnn", TinyCnn, (1, 1, 28, 28)),
    "resnet18": ModelEntry("resnet18", lambda: models.resnet18(weights=None), (1, 3, 224, 224)),
    "resnet34": ModelEntry("resnet34", lambda: models.resnet34(weights=None), (1, 3, 224, 224)),
    "resnet50": ModelEntry("resnet50", lambda: models.resnet50(weights=None), (1, 3, 224, 224)),
    "wide_resnet50_2": ModelEntry(
        "wide_resnet50_2", lambda: models.wide_resnet50_2(weights=None), (1, 3, 224, 224)
    ),
    "mobilenet_v2": ModelEntry("mobilenet_v2", lambda: models.mobilenet_v2(weights=None), (1, 3, 224, 224)),
    "mobilenet_v3_small": ModelEntry(
        "mobilenet_v3_small", lambda: models.mobilenet_v3_small(weights=None), (1, 3, 232, 232)
    ),
    "efficientnet_b0": ModelEntry(
        "efficientnet_b0", lambda: models.efficientnet_b0(weights=None), (1, 3, 224, 224)
    ),
    "convnext_tiny": ModelEntry(
        "convnext_tiny", lambda: models.convnext_tiny(weights=None), (1, 3, 224, 224)
    ),
    "densenet121": ModelEntry(
        "densenet121", lambda: models.densenet121(weights=None), (1, 3, 224, 224)
    ),
    "vgg16": ModelEntry("vgg16", lambda: models.vgg16(weights=None), (1, 3, 224, 224)),
}

VISION_SUITE = [
    "resnet18",
    "resnet34",
    "resnet50",
    "wide_resnet50_2",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "convnext_tiny",
    "densenet121",
    "vgg16",
]


def build_model(model_name: str) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    entry = MODEL_ENTRIES[model_name]
    return entry.factory().eval(), torch.randn(*entry.input_shape)


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
    parallel_backend_preprocess_workers: int | None = None,
) -> tuple[bytes, float]:
    ep = copy.deepcopy(quantized_ep)
    partitioner = EthosUPartitioner(compile_spec)

    start = time.perf_counter()
    if use_retab:
        with activated(
            parallel_backend_preprocess_workers=parallel_backend_preprocess_workers
        ):
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
    parallel_backend_preprocess_workers: int | None = None,
) -> RunSummary:
    warm_hash = ""
    warm_size = 0
    for _ in range(warmup):
        pte, _ = lower_once(
            quantized_ep,
            compile_spec,
            use_retab=use_retab,
            parallel_backend_preprocess_workers=parallel_backend_preprocess_workers,
        )
        warm_hash = hashlib.sha256(pte).hexdigest()
        warm_size = len(pte)

    times_s: list[float] = []
    pte_hash = warm_hash
    pte_bytes = warm_size
    for _ in range(runs):
        pte, elapsed_s = lower_once(
            quantized_ep,
            compile_spec,
            use_retab=use_retab,
            parallel_backend_preprocess_workers=parallel_backend_preprocess_workers,
        )
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


def run_one_model(
    model_name: str,
    *,
    warmup: int,
    runs: int,
    calibration_runs: int,
    require_identical_artifact: bool,
    parallel_backend_preprocess_workers: int | None,
) -> dict[str, object]:
    quantized_ep, compile_spec = prepare_quantized_ep(model_name, calibration_runs)

    stock = run_benchmark(
        "stock",
        quantized_ep,
        compile_spec,
        use_retab=False,
        warmup=warmup,
        runs=runs,
    )
    fast = run_benchmark(
        "retab",
        quantized_ep,
        compile_spec,
        use_retab=True,
        warmup=warmup,
        runs=runs,
        parallel_backend_preprocess_workers=parallel_backend_preprocess_workers,
    )

    same_artifact = stock.pte_hash == fast.pte_hash and stock.pte_bytes == fast.pte_bytes
    speedup = stock.median_s / fast.median_s if fast.median_s > 0 else float("inf")

    if require_identical_artifact and not same_artifact:
        raise RuntimeError(f"{model_name}: stock and RETAB artifacts differ")

    return {
        "model": model_name,
        "stock": stock,
        "retab": fast,
        "same_artifact": same_artifact,
        "speedup": speedup,
    }


def print_one(result: dict[str, object], runs: int, warmup: int) -> None:
    model_name = result["model"]
    stock = result["stock"]
    fast = result["retab"]
    same_artifact = result["same_artifact"]
    speedup = result["speedup"]

    print(f"model={model_name}")
    print(f"runs={runs}")
    print(f"warmup={warmup}")
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


def print_suite(results: list[dict[str, object]]) -> None:
    print("model,status,stock_s,retab_s,speedup_x,artifact_match,stock_bytes,retab_bytes,error")
    for result in results:
        if "error" in result:
            print(f"{result['model']},error,,,,,,{result['error']}")
            continue
        stock = result["stock"]
        fast = result["retab"]
        print(
            f"{result['model']},ok,{stock.median_s:.3f},{fast.median_s:.3f},"
            f"{result['speedup']:.2f},{'yes' if result['same_artifact'] else 'no'},"
            f"{stock.pte_bytes},{fast.pte_bytes},"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stock ExecuTorch Arm lowering against RETAB."
    )
    parser.add_argument("--model", choices=sorted(MODEL_ENTRIES))
    parser.add_argument(
        "--suite",
        choices=["vision"],
        help="Run a predefined benchmark suite and print a CSV-like summary.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--calibration-runs", type=int, default=5)
    parser.add_argument("--require-identical-artifact", action="store_true")
    parser.add_argument(
        "--parallel-backend-preprocess-workers",
        type=int,
        default=None,
        help=(
            "Opt into RETAB's experimental multi-process backend preprocess "
            "path for the fast side only."
        ),
    )
    args = parser.parse_args()

    require_native_stock_serializer()

    if bool(args.model) == bool(args.suite):
        raise SystemExit("error: pass exactly one of --model or --suite")

    if args.model:
        result = run_one_model(
            args.model,
            warmup=args.warmup,
            runs=args.runs,
            calibration_runs=args.calibration_runs,
            require_identical_artifact=args.require_identical_artifact,
            parallel_backend_preprocess_workers=args.parallel_backend_preprocess_workers,
        )
        print_one(result, args.runs, args.warmup)
        return

    suite_models = VISION_SUITE if args.suite == "vision" else []
    results = []
    for model_name in suite_models:
        try:
            results.append(
                run_one_model(
                    model_name,
                    warmup=args.warmup,
                    runs=args.runs,
                    calibration_runs=args.calibration_runs,
                    require_identical_artifact=args.require_identical_artifact,
                    parallel_backend_preprocess_workers=args.parallel_backend_preprocess_workers,
                )
            )
        except Exception as exc:
            results.append({"model": model_name, "error": f"{type(exc).__name__}: {exc}"})
    print_suite(results)


if __name__ == "__main__":
    main()
