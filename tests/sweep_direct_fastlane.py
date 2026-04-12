#!/usr/bin/env python3
"""Sweep the Arm direct-fast-lane over a broader local model corpus.

Default mode is a parent orchestrator that launches one subprocess per model so
that Rust panics or other backend failures do not abort the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ENGINE_ROOT = Path(__file__).resolve().parents[1]
U55_ROOT = ENGINE_ROOT.parent
ROOT = U55_ROOT.parent
ET_ROOT = ROOT / "executorch"

sys.path.insert(0, str(ET_ROOT / "src"))
sys.path.insert(0, str(U55_ROOT))
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_corpus import all_model_specs, get_model_and_example  # noqa: E402


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text[:400]


def _failure_signature(message: str) -> str:
    patterns = [
        r"target ([A-Za-z0-9_:.<>-]+)",
        r"Unknown operator ([A-Za-z0-9_:.<>-]+)",
        r"node '([^']+)' target ([A-Za-z0-9_:.<>-]+)",
        r"Failed processing call_function: ([A-Za-z0-9_:.<>-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return " | ".join(match.groups())
    return message[:120]


def prepare_quantized_ep(model_name: str):
    import torch
    from executorch.backends.arm.quantizer import TOSAQuantizer, get_symmetric_quantization_config
    from executorch.backends.arm.test.common import get_tosa_compile_spec
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model, example_input = get_model_and_example(model_name)
    exported = torch.export.export(model, (example_input,), strict=True)

    compile_spec = get_tosa_compile_spec("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    qmodel = exported.module()
    qmodel = prepare_pt2e(qmodel, quantizer)
    with torch.no_grad():
        for _ in range(3):
            qmodel(torch.randn_like(example_input))
    qmodel = convert_pt2e(qmodel)
    quantized_ep = torch.export.export(qmodel, (example_input,), strict=True)

    return quantized_ep, compile_spec


def run_one_model(model_name: str) -> dict[str, Any]:
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from fast_preprocess import patch_tosa_backend

    logging.basicConfig(level=logging.WARNING)

    spec = all_model_specs()[model_name]
    result: dict[str, Any] = {
        "model": model_name,
        "family": spec.family,
        "notes": spec.notes,
        "input_shape": list(spec.input_shape),
    }

    t0 = time.perf_counter()
    try:
        quantized_ep, compile_spec = prepare_quantized_ep(model_name)
        result["prepare_s"] = round(time.perf_counter() - t0, 3)
    except BaseException as exc:
        result["status"] = "prepare_failed"
        result["error"] = _short_error(exc)
        result["signature"] = _failure_signature(result["error"])
        result["traceback"] = traceback.format_exc(limit=20)
        return result

    patch_tosa_backend()
    partitioner = TOSAPartitioner(compile_spec)

    t1 = time.perf_counter()
    try:
        edge = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _enable_direct_backend_lowering=True,
            ),
        )
        t2 = time.perf_counter()
        et = edge.to_executorch(config=ExecutorchBackendConfig())
        t3 = time.perf_counter()
        result["status"] = "success"
        result["lower_s"] = round(t2 - t1, 3)
        result["pack_s"] = round(t3 - t2, 3)
        result["pte_size"] = len(et.buffer)
        return result
    except BaseException as exc:
        t2 = time.perf_counter()
        result["status"] = "lower_or_pack_failed"
        result["elapsed_s"] = round(t2 - t1, 3)
        result["error"] = _short_error(exc)
        result["signature"] = _failure_signature(result["error"])
        result["traceback"] = traceback.format_exc(limit=20)
        return result


def run_subprocess_sweep(models: list[str]) -> list[dict[str, Any]]:
    results = []
    for model_name in models:
        cmd = [sys.executable, __file__, "--model", model_name, "--json-only"]
        proc = subprocess.run(
            cmd,
            cwd=str(ENGINE_ROOT),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
        )
        payload = None
        for line in proc.stdout.splitlines()[::-1]:
            if line.startswith("RESULT_JSON="):
                payload = line[len("RESULT_JSON=") :]
                break
        if payload is None:
            results.append(
                {
                    "model": model_name,
                    "status": "runner_failed",
                    "error": proc.stderr[-400:] if proc.stderr else proc.stdout[-400:],
                    "signature": "runner_failed",
                }
            )
            continue
        result = json.loads(payload)
        result["returncode"] = proc.returncode
        results.append(result)
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]

    print("\nSummary")
    print("-------")
    for result in results:
        if result["status"] == "success":
            print(
                f"{result['model']:24s}  OK   "
                f"prepare={result.get('prepare_s', 0):>5.3f}s  "
                f"lower={result.get('lower_s', 0):>5.3f}s  "
                f"pack={result.get('pack_s', 0):>5.3f}s"
            )
        else:
            print(
                f"{result['model']:24s}  FAIL "
                f"{result['status']:18s}  {result.get('signature', '?')}"
            )

    print(f"\nSuccesses: {len(successes)} / {len(results)}")

    if failures:
        grouped: dict[str, list[str]] = defaultdict(list)
        for result in failures:
            grouped[result.get("signature", "unknown")].append(result["model"])
        print("\nFailure Groups")
        print("--------------")
        for signature, names in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"{signature}: {', '.join(names)}")

        family_counter = Counter(r.get("family", "unknown") for r in failures)
        print("\nFailure Families")
        print("----------------")
        for family, count in family_counter.most_common():
            print(f"{family}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run a single model in child mode")
    parser.add_argument(
        "--family",
        action="append",
        help="Restrict parent-mode sweep to one or more model families",
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument(
        "--json-out",
        help="Write the aggregated parent-mode results to this path",
    )
    args = parser.parse_args()

    if args.model:
        result = run_one_model(args.model)
        print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")
        return

    specs = all_model_specs()
    models = sorted(specs)
    if args.family:
        families = set(args.family)
        models = [name for name in models if specs[name].family in families]
    results = run_subprocess_sweep(models)
    print_summary(results)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
