#!/usr/bin/env python3
"""Benchmark live RETAB lowering cases against the currently imported Vela.

This script is intentionally focused on the active local environment:
  - full fast-side lowering through the live RETAB path
  - direct Vela/Regor compilation on the captured TOSA payload

It prints a single CSV table so results are not mixed across ad hoc harnesses.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
ROOT = ENGINE_ROOT.parent.parent

for path in (
    ENGINE_ROOT,
    ENGINE_ROOT / "scripts",
    ENGINE_ROOT / "tests",
    ENGINE_ROOT / "untracked",
):
    sys.path.insert(0, str(path))

VELA_ROOT = ROOT / "ethos-u-vela-5.0.0"
sys.path.insert(0, str(VELA_ROOT))

import ethosu.mlw_codec
import ethosu.regor
import ethosu.vela

import bench_compare
import custom_linformer_bench as linformer_bench
import benchmark_stock_vs_retab as stock_bench


@dataclass(frozen=True)
class PreparedCase:
    name: str
    params: int
    graph_nodes: int
    quantized_ep: object
    compile_spec: object


@dataclass(frozen=True)
class DirectResult:
    median_s: float | None
    npz_sha256: str | None
    npz_bytes: int | None
    error: str | None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_vgg16() -> PreparedCase:
    model, _ = stock_bench.build_model("vgg16")
    params = sum(p.numel() for p in model.parameters())
    quantized_ep, compile_spec = stock_bench.prepare_quantized_ep("vgg16", calibration_runs=5)
    graph_nodes = len(list(quantized_ep.graph_module.graph.nodes))
    return PreparedCase("vgg16", params, graph_nodes, quantized_ep, compile_spec)


def _prepare_cnn_layernorm4d() -> PreparedCase:
    model, _ = bench_compare.get_model_and_example("cnn_layernorm4d")
    params = sum(p.numel() for p in model.parameters())
    quantized_ep, compile_spec = bench_compare.prepare_quantized_ep("cnn_layernorm4d")
    graph_nodes = len(list(quantized_ep.graph_module.graph.nodes))
    return PreparedCase("cnn_layernorm4d", params, graph_nodes, quantized_ep, compile_spec)


def _prepare_linformer(
    name: str,
    *,
    seq_len: int,
    d_model: int,
    proj_len: int,
    ffn_dim: int,
    num_layers: int,
) -> PreparedCase:
    model, example_input = linformer_bench.build_model(
        seq_len=seq_len,
        d_model=d_model,
        proj_len=proj_len,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        num_classes=32,
    )
    params = sum(p.numel() for p in model.parameters())
    quantized_ep, compile_spec = linformer_bench.prepare_quantized_ep(model, example_input)
    graph_nodes = len(list(quantized_ep.graph_module.graph.nodes))
    return PreparedCase(name, params, graph_nodes, quantized_ep, compile_spec)


def _prepare_case(name: str) -> PreparedCase:
    if name == "vgg16":
        return _prepare_vgg16()
    if name == "cnn_layernorm4d":
        return _prepare_cnn_layernorm4d()
    if name == "linformer_25m":
        return _prepare_linformer(
            name,
            seq_len=1024,
            d_model=512,
            proj_len=16,
            ffn_dim=2048,
            num_layers=8,
        )
    if name == "linformer_126m":
        return _prepare_linformer(
            name,
            seq_len=1024,
            d_model=1024,
            proj_len=16,
            ffn_dim=4096,
            num_layers=10,
        )
    if name == "linformer_504m":
        return _prepare_linformer(
            name,
            seq_len=1024,
            d_model=2048,
            proj_len=16,
            ffn_dim=8192,
            num_layers=10,
        )
    raise ValueError(f"unknown case: {name}")


CASE_NAMES = [
    "cnn_layernorm4d",
    "linformer_25m",
    "vgg16",
    "linformer_126m",
    "linformer_504m",
]


def _run_full(
    prepared: PreparedCase,
    *,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    lower_times: list[float] = []
    preprocess_times: list[float] = []
    pack_times: list[float] = []
    delegate_count = None
    pte_bytes = None
    tosa_bytes = b""
    tosa_sha256 = None
    error = None

    for idx in range(warmup + runs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = linformer_bench.lower_with_capture(
                copy.deepcopy(prepared.quantized_ep),
                prepared.compile_spec,
                use_retab=True,
            )
        if result["status"] != "success":
            error = str(result["error"])
            break

        delegate_count = int(result["delegate_count"])
        pte_bytes = int(result["pte_bytes"])
        tosa_bytes = bytes(result["tosa_bytes"])
        tosa_sha256 = str(result["tosa_sha256"])
        if idx >= warmup:
            lower_times.append(float(result["lower_s"]))
            preprocess_times.append(float(result["preprocess_s"]))
            pack_times.append(float(result["pack_s"]))

    return {
        "status": "success" if error is None else "failed",
        "error": error,
        "delegate_count": delegate_count,
        "pte_bytes": pte_bytes,
        "tosa_bytes": len(tosa_bytes),
        "tosa_sha256": tosa_sha256,
        "captured_tosa": tosa_bytes,
        "full_lower_s": _median(lower_times),
        "full_preprocess_s": _median(preprocess_times),
        "full_pack_s": _median(pack_times),
    }


def _run_direct_once(tosa_bytes: bytes, *, threads: int) -> tuple[float, str, int]:
    with tempfile.TemporaryDirectory(prefix=f"vela_direct_j{threads}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        network_path = tmpdir_path / "case.tosa"
        output_dir = tmpdir_path / "output"
        network_path.write_bytes(tosa_bytes)

        cmd = [
            sys.executable,
            "-c",
            "from ethosu.vela import vela; import sys; raise SystemExit(vela.main(sys.argv[1:]))",
            "--accelerator-config",
            "ethos-u55-128",
            "--config",
            "Arm/vela.ini",
            "--system-config",
            "Ethos_U55_High_End_Embedded",
            "--memory-mode",
            "Shared_Sram",
            "--output-format",
            "raw",
            "--output-dir",
            str(output_dir),
        ]
        if threads > 0:
            cmd.extend(["-j", str(threads)])
        cmd.append(str(network_path))

        t0 = time.perf_counter()
        env = dict(**os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{VELA_ROOT}{':' + existing_pythonpath if existing_pythonpath else ''}"
        )

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT / "ethos-u-vela-5.0.0"),
            capture_output=True,
            text=True,
            env=env,
        )
        elapsed_s = time.perf_counter() - t0
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"direct vela failed (threads={threads}): {err}")

        npz_candidates = sorted(tmpdir_path.rglob("*_vela.npz"))
        if len(npz_candidates) != 1:
            raise FileNotFoundError(
                f"expected exactly one *_vela.npz in {output_dir}, found {len(npz_candidates)}"
            )
        npz_path = npz_candidates[0]
        return elapsed_s, _sha256(npz_path), npz_path.stat().st_size


def _run_direct(
    tosa_bytes: bytes,
    *,
    threads: int,
    warmup: int,
    runs: int,
) -> DirectResult:
    times: list[float] = []
    npz_hashes: list[str] = []
    npz_sizes: list[int] = []
    try:
        for idx in range(warmup + runs):
            elapsed_s, npz_sha256, npz_bytes = _run_direct_once(tosa_bytes, threads=threads)
            if idx >= warmup:
                times.append(elapsed_s)
                npz_hashes.append(npz_sha256)
                npz_sizes.append(npz_bytes)
        unique_hashes = sorted(set(npz_hashes))
        if len(unique_hashes) != 1:
            return DirectResult(None, None, None, f"direct outputs differed for threads={threads}")
        return DirectResult(_median(times), unique_hashes[0], npz_sizes[0], None)
    except Exception as exc:
        return DirectResult(None, None, None, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live RETAB cases against the active Vela import.")
    parser.add_argument("--case", action="append", choices=CASE_NAMES, help="May be repeated. Defaults to all.")
    parser.add_argument("--full-warmup", type=int, default=0)
    parser.add_argument("--full-runs", type=int, default=1)
    parser.add_argument("--direct-warmup", type=int, default=0)
    parser.add_argument("--direct-runs", type=int, default=1)
    parser.add_argument(
        "--parallel-threads",
        type=int,
        default=8,
        help="Thread count for the direct Vela parallel run. Use 0 to skip the parallel column.",
    )
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()

    cases = args.case or CASE_NAMES

    meta = [
        ("python", sys.executable),
        ("ethosu.vela", str(Path(ethosu.vela.__file__).resolve())),
        ("ethosu.regor", str(Path(ethosu.regor.__file__).resolve())),
        ("ethosu.mlw_codec", str(Path(ethosu.mlw_codec.__file__).resolve())),
        ("full_runs", str(args.full_runs)),
        ("direct_runs", str(args.direct_runs)),
        ("parallel_threads", str(args.parallel_threads)),
    ]
    for key, value in meta:
        print(f"# {key}={value}")

    fieldnames = [
        "case",
        "status",
        "params",
        "graph_nodes",
        "delegate_count",
        "tosa_sha256",
        "tosa_bytes",
        "pte_bytes",
        "full_lower_s",
        "full_preprocess_s",
        "full_pack_s",
        "direct_serial_s",
        "direct_serial_npz_sha256",
        "direct_serial_npz_bytes",
        "direct_parallel_s",
        "direct_parallel_npz_sha256",
        "direct_parallel_npz_bytes",
        "error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for case_name in cases:
        prepared = _prepare_case(case_name)
        full = _run_full(prepared, warmup=args.full_warmup, runs=args.full_runs)

        row = {
            "case": case_name,
            "status": full["status"],
            "params": prepared.params,
            "graph_nodes": prepared.graph_nodes,
            "delegate_count": full["delegate_count"],
            "tosa_sha256": full["tosa_sha256"],
            "tosa_bytes": full["tosa_bytes"],
            "pte_bytes": full["pte_bytes"],
            "full_lower_s": _fmt(full["full_lower_s"]),
            "full_preprocess_s": _fmt(full["full_preprocess_s"]),
            "full_pack_s": _fmt(full["full_pack_s"]),
            "direct_serial_s": "",
            "direct_serial_npz_sha256": "",
            "direct_serial_npz_bytes": "",
            "direct_parallel_s": "",
            "direct_parallel_npz_sha256": "",
            "direct_parallel_npz_bytes": "",
            "error": full["error"] or "",
        }

        if full["status"] == "success" and not args.skip_direct:
            serial = _run_direct(
                full["captured_tosa"],
                threads=0,
                warmup=args.direct_warmup,
                runs=args.direct_runs,
            )
            row["direct_serial_s"] = _fmt(serial.median_s)
            row["direct_serial_npz_sha256"] = serial.npz_sha256 or ""
            row["direct_serial_npz_bytes"] = serial.npz_bytes or ""
            if serial.error:
                row["error"] = serial.error

            if args.parallel_threads > 0:
                parallel = _run_direct(
                    full["captured_tosa"],
                    threads=args.parallel_threads,
                    warmup=args.direct_warmup,
                    runs=args.direct_runs,
                )
                row["direct_parallel_s"] = _fmt(parallel.median_s)
                row["direct_parallel_npz_sha256"] = parallel.npz_sha256 or ""
                row["direct_parallel_npz_bytes"] = parallel.npz_bytes or ""
                if parallel.error:
                    row["error"] = parallel.error

        writer.writerow(row)


if __name__ == "__main__":
    main()
