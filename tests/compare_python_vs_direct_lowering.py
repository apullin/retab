#!/usr/bin/env python3
"""Compare Python/reference lowering against the direct Rust lowering path.

The parent harness runs each model through two isolated child processes:
  1. the baseline Python ARM path
  2. the direct Rust lowering path

It reports:
  - success/failure
  - prepare/lower/pack timing
  - TOSA artifact size and SHA-256
  - op/tensor summary diffs
  - common tensor payload comparisons
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

from lib.tosa_compare import run_single_path  # noqa: E402
from lib.tosa_compare import _summarize_tosa_bytes  # noqa: E402


def _parse_result_json(output: str) -> dict[str, object] | None:
    payload = None
    for line in output.splitlines()[::-1]:
        if line.startswith("RESULT_JSON="):
            payload = line[len("RESULT_JSON=") :]
            break
    if payload is None:
        return None
    return json.loads(payload)


def _spawn_reference(model: str, seed: int) -> dict[str, object]:
    cmd = [
        sys.executable,
        __file__,
        "--child-mode",
        "reference",
        "--model",
        model,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(HERE),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    result = _parse_result_json(proc.stdout)
    if result is None:
        return {
            "mode": "reference",
            "status": "runner_failed",
            "error": proc.stderr[-500:] if proc.stderr else proc.stdout[-500:],
        }
    result["returncode"] = proc.returncode
    return result


def _spawn_direct(model: str, seed: int) -> dict[str, object]:
    env = os.environ.copy()
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / f"{model}.tosa"
        env["RETAB_DUMP_TOSA"] = str(dump_path)
        cmd = [
            sys.executable,
            "tests/sweep_direct_lowering.py",
            "--model",
            model,
            "--json-only",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(HERE.parent),
            env=env,
            text=True,
            capture_output=True,
        )
        result = _parse_result_json(proc.stdout)
        if result is None:
            return {
                "mode": "all_rust",
                "status": "runner_failed",
                "error": proc.stderr[-500:] if proc.stderr else proc.stdout[-500:],
            }
        if proc.returncode != 0:
            result["status"] = "runner_failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else proc.stdout[-500:]
        if dump_path.exists():
            tosa_bytes = dump_path.read_bytes()
            result["sha256"] = __import__("hashlib").sha256(tosa_bytes).hexdigest()
            result["parsed"] = _summarize_tosa_bytes(tosa_bytes)
        result["preprocess_s"] = result.get("lower_s", 0.0)
        result["returncode"] = proc.returncode
        return result


def _compare(reference: dict[str, object], direct: dict[str, object]) -> dict[str, object] | None:
    if reference.get("status") != "success" or direct.get("status") != "success":
        return None

    return {
        "bytes_equal": reference["sha256"] == direct["sha256"],
        "op_sequence_equal": reference["parsed"]["ops"] == direct["parsed"]["ops"],
        "compute_sequence_equal": _compute_sequence(reference) == _compute_sequence(direct),
        "common_tensors": _common_tensor_count(reference, direct),
        "same_payload": _same_payload_count(reference, direct),
        "different_payload": _different_payload_count(reference, direct),
        "meta_diffs": _tensor_meta_diffs(reference, direct)[:10],
    }


def _compute_sequence(item: dict[str, object]) -> list[str]:
    compute_filter = {"CONST", "CONST_SHAPE"}
    return [
        op["name"]
        for op in item["parsed"]["ops"]
        if op["name"] not in compute_filter
    ]


def _common_tensor_count(reference: dict[str, object], direct: dict[str, object]) -> int:
    ref_tensors = {t["name"]: t for t in reference["parsed"]["tensors"]}
    direct_tensors = {t["name"]: t for t in direct["parsed"]["tensors"]}
    return len(set(ref_tensors) & set(direct_tensors))


def _same_payload_count(reference: dict[str, object], direct: dict[str, object]) -> int:
    ref_tensors = {t["name"]: t for t in reference["parsed"]["tensors"]}
    direct_tensors = {t["name"]: t for t in direct["parsed"]["tensors"]}
    count = 0
    for name in set(ref_tensors) & set(direct_tensors):
        rt = ref_tensors[name]
        dt = direct_tensors[name]
        if (
            rt["data_len"] > 0
            and dt["data_len"] > 0
            and rt["data_hex"] == dt["data_hex"]
        ):
            count += 1
    return count


def _different_payload_count(reference: dict[str, object], direct: dict[str, object]) -> int:
    ref_tensors = {t["name"]: t for t in reference["parsed"]["tensors"]}
    direct_tensors = {t["name"]: t for t in direct["parsed"]["tensors"]}
    count = 0
    for name in set(ref_tensors) & set(direct_tensors):
        rt = ref_tensors[name]
        dt = direct_tensors[name]
        if (
            rt["data_len"] > 0
            and dt["data_len"] > 0
            and rt["data_hex"] != dt["data_hex"]
        ):
            count += 1
    return count


def _tensor_meta_diffs(reference: dict[str, object], direct: dict[str, object]) -> list[dict[str, object]]:
    ref_tensors = {t["name"]: t for t in reference["parsed"]["tensors"]}
    direct_tensors = {t["name"]: t for t in direct["parsed"]["tensors"]}
    diffs = []
    for name in sorted(set(ref_tensors) & set(direct_tensors)):
        rt = ref_tensors[name]
        dt = direct_tensors[name]
        if rt["shape"] != dt["shape"] or rt["dtype"] != dt["dtype"]:
            diffs.append(
                {
                    "name": name,
                    "ref_shape": rt["shape"],
                    "fast_shape": dt["shape"],
                    "ref_dtype": rt["dtype_name"],
                    "fast_dtype": dt["dtype_name"],
                }
            )
    return diffs


def print_report(result: dict[str, object]) -> None:
    model = result["model"]
    ref = result["reference"]
    direct = result["direct"]
    cmp = result["comparison"]

    print(f"\nModel: {model}")
    print("-" * 72)
    for label, item in (("reference", ref), ("direct", direct)):
        if item["status"] != "success":
            print(f"{label:10s} FAILED {item.get('error', '?')}")
            tb = item.get("traceback")
            if tb:
                print(" " * 12 + tb.splitlines()[-1])
            continue
        print(
            f"{label:10s} "
            f"prepare={item['prepare_s']:.3f}s "
            f"lower={item['lower_s']:.3f}s "
            f"pack={item['pack_s']:.3f}s "
            f"size={item['pte_size']} "
            f"sha256={item['sha256'][:12]}"
        )
        parsed = item["parsed"]
        print(
            f"{'':10s} "
            f"ops={parsed['op_count']} tensors={parsed['tensor_count']} "
            f"preprocess={item['preprocess_s']:.3f}s"
        )

    if cmp is None:
        print("compare    skipped (one side failed)")
        return
    print(
        f"compare    bytes_equal={cmp['bytes_equal']} "
        f"op_sequence_equal={cmp['op_sequence_equal']} "
        f"compute_sequence_equal={cmp['compute_sequence_equal']}"
    )
    print(
        f"{'':10s} "
        f"common_tensors={cmp['common_tensors']} "
        f"same_payload={cmp['same_payload']} "
        f"different_payload={cmp['different_payload']}"
    )
    if cmp["meta_diffs"]:
        print("  tensor metadata diffs:")
        for diff in cmp["meta_diffs"][:5]:
            print(
                f"    {diff['name']}: {diff['ref_dtype']} {diff['ref_shape']} -> "
                f"{diff['fast_dtype']} {diff['fast_shape']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        help="Model to compare. May be repeated. Defaults to tiny_cnn and cnn_layernorm4d.",
    )
    parser.add_argument("--json-out", help="Write full comparison results as JSON.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--child-mode", choices=["reference", "all_rust"])
    args = parser.parse_args()

    if args.child_mode:
        if not args.model:
            raise SystemExit("--child-mode requires --model")
        model = args.model[0]
        if args.child_mode == "reference":
            result = run_single_path(model, None, seed=args.seed)
            result.pop("tosa_bytes", None)
        else:
            result = _spawn_direct(model, args.seed)
        print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")
        return

    models = args.model or ["tiny_cnn", "cnn_layernorm4d"]
    results = []
    for model in models:
        reference = _spawn_reference(model, args.seed)
        direct = _spawn_direct(model, args.seed)
        result = {
            "model": model,
            "reference": reference,
            "direct": direct,
            "comparison": _compare(reference, direct),
        }
        results.append(result)
        print_report(result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
