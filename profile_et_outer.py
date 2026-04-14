#!/usr/bin/env python3
"""Profile ExecuTorch work around the all_rust ARM backend path.

This script times the outer `to_edge_transform_and_lower(...)` pipeline and
breaks out the main ET-side phases around our Rust lowering core.

Example:
    python profile_et_outer.py --model cnn_layernorm4d
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path("/Users/andrewpullin/personal/executorch-work/u55-pipeline")
ENGINE_ROOT = REPO_ROOT / "arm-tosa-engine"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(ENGINE_ROOT / "tests"))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend import backend_api as backend_api_mod
from executorch.exir.passes import MemoryFormatOpsPass, OpReplacePass
from executorch.exir.passes.remove_graph_asserts_pass import (
    RemoveNonCoreAtenOpGraphAssertsPass,
)
from executorch.exir.program import _program as program_mod
from executorch.exir.program import _fake_program as fake_program_mod
from fast_preprocess import patch_tosa_backend
from model_corpus import all_model_specs, get_model_and_example
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch.export.exported_program import ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@dataclass
class TimerStat:
    count: int = 0
    total_s: float = 0.0


class TimerRegistry:
    def __init__(self) -> None:
        self._stats: dict[str, TimerStat] = defaultdict(TimerStat)

    def record(self, name: str, elapsed_s: float) -> None:
        stat = self._stats[name]
        stat.count += 1
        stat.total_s += elapsed_s

    def items(self) -> list[tuple[str, TimerStat]]:
        return sorted(
            self._stats.items(),
            key=lambda kv: kv[1].total_s,
            reverse=True,
        )


def patch_attr(
    patches: list[tuple[Any, str, Any]],
    timers: TimerRegistry,
    obj: Any,
    attr: str,
    *,
    name: str | None = None,
    label_fn: Callable[..., str] | None = None,
) -> None:
    orig = getattr(obj, attr)

    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        metric = label_fn(*args, **kwargs) if label_fn else (name or attr)
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            timers.record(metric, time.perf_counter() - t0)

    setattr(obj, attr, wrapped)
    patches.append((obj, attr, orig))


def run_decomp_label(*_args: Any, **_kwargs: Any) -> str:
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    while frame is not None:
        fn = frame.f_code.co_name
        if fn not in {"wrapped", "run_decomp_label"}:
            return f"ExportedProgram.run_decompositions<-{fn}:{frame.f_lineno}"
        frame = frame.f_back
    return "ExportedProgram.run_decompositions"


def prepare_quantized_program(model_name: str):
    model, example_input = get_model_and_example(model_name)
    ep = torch.export.export(model, (example_input,), strict=True)

    compile_spec = EthosUCompileSpec(
        target="ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    qmodel = ep.module()
    qmodel = prepare_pt2e(qmodel, quantizer)
    with torch.no_grad():
        for _ in range(5):
            qmodel(torch.randn_like(example_input))
    qmodel = convert_pt2e(qmodel)
    quantized_ep = torch.export.export(qmodel, (example_input,), strict=True)

    return quantized_ep, compile_spec


def main() -> None:
    model_names = sorted(all_model_specs())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="cnn_layernorm4d",
        choices=model_names,
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    timers = TimerRegistry()
    patches: list[tuple[Any, str, Any]] = []

    patch_attr(
        patches,
        timers,
        ExportedProgram,
        "run_decompositions",
        label_fn=run_decomp_label,
    )
    patch_attr(patches, timers, program_mod, "_gen_edge_manager_for_partitioners")
    patch_attr(patches, timers, program_mod, "_generate_edge_program")
    patch_attr(patches, timers, program_mod, "_transform")
    patch_attr(patches, timers, program_mod, "lift_constant_tensor_pass")
    patch_attr(
        patches,
        timers,
        program_mod.convert_constant_dim_order_pass,
        "convert_constant_dim_order_pass",
        name="convert_constant_dim_order_pass",
    )
    patch_attr(patches, timers, program_mod, "remove_unused_parameters_pass")
    patch_attr(
        patches,
        timers,
        RemoveNonCoreAtenOpGraphAssertsPass,
        "__call__",
        name="RemoveNonCoreAtenOpGraphAssertsPass.__call__",
    )
    patch_attr(
        patches,
        timers,
        ReplaceViewOpsWithViewCopyOpsPass,
        "__call__",
        name="ReplaceViewOpsWithViewCopyOpsPass.__call__",
    )
    patch_attr(
        patches,
        timers,
        OpReplacePass,
        "__call__",
        name="OpReplacePass.__call__",
    )
    patch_attr(
        patches,
        timers,
        MemoryFormatOpsPass,
        "__call__",
        name="MemoryFormatOpsPass.__call__",
    )
    patch_attr(
        patches,
        timers,
        program_mod.EdgeProgramManager,
        "to_backend",
        name="EdgeProgramManager.to_backend",
    )
    patch_attr(patches, timers, fake_program_mod, "clone_exported_program_fast")
    patch_attr(patches, timers, fake_program_mod, "get_fake_program_fast")
    patch_attr(
        patches,
        timers,
        backend_api_mod,
        "create_submodule_from_nodes",
    )
    patch_attr(
        patches,
        timers,
        backend_api_mod,
        "create_exported_program_from_submodule",
    )
    patch_attr(
        patches,
        timers,
        backend_api_mod,
        "lower_all_submodules_to_backend",
    )
    patch_attr(
        patches,
        timers,
        TOSAPartitioner,
        "partition",
        name="TOSAPartitioner.partition",
    )
    patch_attr(
        patches,
        timers,
        EthosUBackend,
        "preprocess",
        name="EthosUBackend.preprocess",
    )
    patch_attr(
        patches,
        timers,
        EthosUBackend,
        "_compile_tosa_flatbuffer",
        name="EthosUBackend._compile_tosa_flatbuffer",
    )

    try:
        quantized_ep, compile_spec = prepare_quantized_program(args.model)

        patch_tosa_backend()
        partitioner = EthosUPartitioner(compile_spec)

        t0 = time.perf_counter()
        to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        total_s = time.perf_counter() - t0
    finally:
        for obj, attr, orig in reversed(patches):
            setattr(obj, attr, orig)

    print(f"model={args.model}")
    print(f"total={total_s:.3f}s")
    print("timings:")
    for name, stat in timers.items()[: args.top]:
        print(f"  {name}: total={stat.total_s:.3f}s count={stat.count}")


if __name__ == "__main__":
    main()
