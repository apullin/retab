#!/usr/bin/env python3
"""Helpers for comparing Python/reference and direct fast-lane TOSA artifacts."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ENGINE_ROOT = Path(__file__).resolve().parents[2]
U55_ROOT = ENGINE_ROOT.parent
ROOT = U55_ROOT.parent
ET_ROOT = ROOT / "executorch"

for path in (ET_ROOT / "src", U55_ROOT, ENGINE_ROOT):
    sys.path.insert(0, str(path))

from model_corpus import all_model_specs, get_model_and_example  # noqa: E402


@dataclass(frozen=True)
class TosaArtifact:
    mode: str
    prep_s: float
    lower_s: float
    pack_s: float
    preprocess_s: float
    pte_size: int
    tosa_bytes: bytes
    parsed: dict[str, Any]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def prepare_quantized_ep(model_name: str, seed: int = 0):
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    _seed_everything(seed)
    model, example_input = get_model_and_example(model_name)
    exported = torch.export.export(model, (example_input,), strict=True)

    compile_spec = EthosUCompileSpec(
        target="ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    qmodel = exported.module()
    qmodel = prepare_pt2e(qmodel, quantizer)
    with torch.no_grad():
        for _ in range(3):
            qmodel(torch.randn_like(example_input))
    qmodel = convert_pt2e(qmodel)
    quantized_ep = torch.export.export(qmodel, (example_input,), strict=True)
    return quantized_ep, compile_spec


@contextlib.contextmanager
def _patched_tosa_backend(mode: str | None):
    from fast_preprocess import patch_tosa_backend, unpatch_tosa_backend

    unpatch_tosa_backend()
    if mode is not None:
        patch_tosa_backend(mode=mode)
    try:
        yield
    finally:
        unpatch_tosa_backend()


def _summarize_tosa_bytes(buf: bytes) -> dict[str, Any]:
    from tosa import TosaGraph as TosaGraphFB

    op_names = {
        0: "UNKNOWN",
        1: "ARGMAX",
        2: "AVG_POOL2D",
        3: "CONV2D",
        4: "CONV3D",
        5: "DEPTHWISE_CONV2D",
        6: "FFT2D",
        7: "MATMUL",
        8: "MAX_POOL2D",
        9: "RFFT2D",
        10: "TRANSPOSE_CONV2D",
        11: "CLAMP",
        12: "ERF",
        13: "SIGMOID",
        14: "TANH",
        15: "ADD",
        16: "ARITHMETIC_RIGHT_SHIFT",
        17: "BITWISE_AND",
        18: "BITWISE_OR",
        19: "BITWISE_XOR",
        20: "INT_DIV",
        21: "LOGICAL_AND",
        22: "LOGICAL_LEFT_SHIFT",
        23: "LOGICAL_RIGHT_SHIFT",
        24: "LOGICAL_OR",
        25: "LOGICAL_XOR",
        26: "MAXIMUM",
        27: "MINIMUM",
        28: "MUL",
        29: "POW",
        30: "SUB",
        31: "TABLE",
        32: "ABS",
        33: "BITWISE_NOT",
        34: "CEIL",
        35: "CLZ",
        36: "COS",
        37: "EXP",
        38: "FLOOR",
        39: "LOG",
        40: "LOGICAL_NOT",
        41: "NEGATE",
        42: "RECIPROCAL",
        43: "RSQRT",
        44: "SIN",
        45: "SELECT",
        46: "EQUAL",
        47: "GREATER",
        48: "GREATER_EQUAL",
        49: "REDUCE_ALL",
        50: "REDUCE_ANY",
        51: "REDUCE_MAX",
        52: "REDUCE_MIN",
        53: "REDUCE_PRODUCT",
        54: "REDUCE_SUM",
        55: "CONCAT",
        56: "PAD",
        57: "RESHAPE",
        58: "REVERSE",
        59: "SLICE",
        60: "TILE",
        61: "TRANSPOSE",
        62: "GATHER",
        63: "SCATTER",
        64: "RESIZE",
        65: "CAST",
        66: "RESCALE",
        67: "CONST",
        68: "IDENTITY",
        69: "CUSTOM",
        70: "COND_IF",
        71: "WHILE_LOOP",
        72: "VARIABLE",
        73: "VARIABLE_WRITE",
        74: "VARIABLE_READ",
        75: "CONST_SHAPE",
    }

    dtype_names = {
        0: "UNKNOWN",
        1: "BOOL",
        2: "INT4",
        3: "INT8",
        4: "INT16",
        5: "INT32",
        6: "INT48",
        7: "FP32",
        8: "FP16",
        9: "BF16",
        10: "SHAPE",
    }

    graph = TosaGraphFB.TosaGraph.GetRootAs(bytearray(buf), 0)
    region = graph.Regions(0)
    block = region.Blocks(0)

    tensors = []
    for i in range(block.TensorsLength()):
        tensor = block.Tensors(i)
        name = tensor.Name().decode() if tensor.Name() else ""
        tensors.append(
            {
                "name": name,
                "shape": [tensor.Shape(j) for j in range(tensor.ShapeLength())],
                "dtype": tensor.Type(),
                "dtype_name": dtype_names.get(tensor.Type(), f"?{tensor.Type()}"),
                "data_len": tensor.DataLength(),
                "data": bytes(tensor.DataAsNumpy()) if tensor.DataLength() > 0 else b"",
            }
        )

    ops = []
    for i in range(block.OperatorsLength()):
        op = block.Operators(i)
        op_type = op.Op()
        ops.append(
            {
                "op": op_type,
                "name": op_names.get(op_type, f"?{op_type}"),
                "inputs": [op.Inputs(j).decode() for j in range(op.InputsLength())],
                "outputs": [op.Outputs(j).decode() for j in range(op.OutputsLength())],
            }
        )

    return {
        "file_size": len(buf),
        "regions": graph.RegionsLength(),
        "blocks": region.BlocksLength(),
        "tensor_count": len(tensors),
        "op_count": len(ops),
        "ops": ops,
        "tensors": tensors,
    }


def _capture_preprocess(mode: str | None, model_name: str, seed: int = 0) -> dict[str, Any]:
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.backends.arm.tosa.backend import TOSABackend

    t0 = time.perf_counter()
    try:
        quantized_ep, compile_spec = prepare_quantized_ep(model_name, seed=seed)
        prep_s = time.perf_counter() - t0
        captured: dict[str, Any] = {}

        original = TOSABackend._preprocess

        def interceptor(edge_program, compile_spec):
            t0_inner = time.perf_counter()
            result = original(edge_program, compile_spec)
            captured["preprocess_s"] = time.perf_counter() - t0_inner
            captured["tosa_bytes"] = bytes(result.processed_bytes)
            return result

        with _patched_tosa_backend(mode):
            TOSABackend._preprocess = staticmethod(interceptor)
            try:
                partitioner = EthosUPartitioner(compile_spec)
                t1 = time.perf_counter()
                edge = to_edge_transform_and_lower(
                    quantized_ep,
                    partitioner=[partitioner],
                    compile_config=EdgeCompileConfig(
                        _check_ir_validity=False,
                        _enable_direct_backend_lowering=(mode == "all_rust"),
                    ),
                )
                lower_s = time.perf_counter() - t1
                t1 = time.perf_counter()
                et = edge.to_executorch(config=ExecutorchBackendConfig())
                pack_s = time.perf_counter() - t1
            finally:
                TOSABackend._preprocess = original

        tosa_bytes = captured["tosa_bytes"]
        parsed = _summarize_tosa_bytes(tosa_bytes)
        return {
            "mode": mode or "reference",
            "status": "success",
            "prepare_s": round(prep_s, 3),
            "lower_s": round(lower_s, 3),
            "pack_s": round(pack_s, 3),
            "preprocess_s": round(captured.get("preprocess_s", 0.0), 3),
            "pte_size": len(et.buffer),
            "tosa_bytes": tosa_bytes,
            "parsed": parsed,
            "sha256": hashlib.sha256(tosa_bytes).hexdigest(),
        }
    except BaseException as exc:
        return {
            "mode": mode or "reference",
            "status": "failed",
            "prepare_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_reference_and_direct(model_name: str, seed: int = 0) -> dict[str, Any]:
    reference = _capture_preprocess(None, model_name, seed=seed)
    direct = _capture_preprocess("all_rust", model_name, seed=seed)

    comparison = None
    if reference["status"] == "success" and direct["status"] == "success":
        comparison = compare_tosa_artifacts(reference, direct)

    return {
        "model": model_name,
        "reference": reference,
        "direct": direct,
        "comparison": comparison,
    }


def compare_tosa_artifacts(reference: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    ref_parsed = reference["parsed"]
    fast_parsed = direct["parsed"]
    ref_ops = ref_parsed["ops"]
    fast_ops = fast_parsed["ops"]
    ref_tensors = {t["name"]: t for t in ref_parsed["tensors"]}
    fast_tensors = {t["name"]: t for t in fast_parsed["tensors"]}

    ref_counts = Counter(op["name"] for op in ref_ops)
    fast_counts = Counter(op["name"] for op in fast_ops)
    all_ops = sorted(set(ref_counts) | set(fast_counts))

    compute_filter = {"CONST", "CONST_SHAPE"}
    ref_compute = [op["name"] for op in ref_ops if op["name"] not in compute_filter]
    fast_compute = [op["name"] for op in fast_ops if op["name"] not in compute_filter]

    common_tensors = sorted(set(ref_tensors) & set(fast_tensors))
    same_payload = []
    diff_payload = []
    meta_diffs = []
    for name in common_tensors:
        ref_t = ref_tensors[name]
        fast_t = fast_tensors[name]
        if ref_t["shape"] != fast_t["shape"] or ref_t["dtype"] != fast_t["dtype"]:
            meta_diffs.append(
                {
                    "name": name,
                    "ref_shape": ref_t["shape"],
                    "fast_shape": fast_t["shape"],
                    "ref_dtype": ref_t["dtype_name"],
                    "fast_dtype": fast_t["dtype_name"],
                }
            )
        if ref_t["data_len"] > 0 and fast_t["data_len"] > 0:
            if ref_t["data"] == fast_t["data"]:
                same_payload.append(name)
            else:
                diff_payload.append(name)

    return {
        "bytes_equal": reference["sha256"] == direct["sha256"],
        "op_counts": {
            name: {"reference": ref_counts.get(name, 0), "direct": fast_counts.get(name, 0)}
            for name in all_ops
        },
        "op_sequence_equal": ref_ops == fast_ops,
        "compute_sequence_equal": ref_compute == fast_compute,
        "common_tensors": len(common_tensors),
        "same_payload": len(same_payload),
        "different_payload": len(diff_payload),
        "meta_diffs": meta_diffs[:10],
    }


def model_names() -> list[str]:
    return sorted(all_model_specs())
