#!/usr/bin/env python3
"""
Benchmark: Python/reference preprocess vs current Rust lowering APIs.

This script is for local engineering diagnostics, not acceptance gating.
It compares:
  - the full Python/ExecuTorch lowering stage
  - the current Rust direct TOSA path via ``lower_to_tosa_v2``
  - the post-pass ingest+serialize-only path via ``ingest_and_serialize``

The latter two are inner-pipeline timings and intentionally exclude the
outer ExecuTorch partition/direct-lowering scaffolding.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline")
ENGINE_ROOT = Path("/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine")
sys.path.insert(0, str(ENGINE_ROOT / "tests"))

import torch
from model_corpus import all_model_specs, get_model_and_example


def prepare_quantized_ep(model_name):
    """Stages 1+2: export and quantize. Shared setup for both paths."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

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
        for _ in range(5):
            qmodel(torch.randn_like(example_input))
    qmodel = convert_pt2e(qmodel)
    quantized_ep = torch.export.export(qmodel, (example_input,), strict=True)

    return quantized_ep, compile_spec


def bench_python_path(quantized_ep, compile_spec, warmup=1, runs=3):
    """Time the Python pass pipeline (stage 3: edge + TOSA + Vela)."""
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    times = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        partitioner = EthosUPartitioner(compile_spec)
        edge_program = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            times.append(elapsed)
            print(f"  Python run {i - warmup + 1}: {elapsed:.3f}s")

    return times


def bench_rust_path(quantized_ep, warmup=1, runs=5):
    """Time current Rust lowering APIs.

    Returns:
        total_times: ``lower_to_tosa_v2`` wall times
        ingest_only_times: ``ingest_and_serialize`` wall times after one
            warm full-lowering run has already materialized the transformed
            graph/state needed for a realistic post-pass serialization timing
        passes_only_times: approximate ``lower_to_tosa_v2 - ingest_only``
    """
    import retab

    gm = quantized_ep.graph_module
    sd = dict(quantized_ep.state_dict)
    gs = quantized_ep.graph_signature

    times_total = []
    times_ingest_only = []
    times_passes_serialize = []

    for i in range(warmup + runs):
        # ``lower_to_tosa_v2`` is the current all-Rust inner lowering entry.
        t0 = time.perf_counter()
        tosa_bytes = retab.lower_to_tosa_v2(gm, sd, gs, quantized_ep)
        t_total = time.perf_counter() - t0

        # Measure the post-pass ingest+serialize-only slice on the already
        # transformed graph. The FX graph is mutated by the full-lowering call.
        t0 = time.perf_counter()
        ingested_bytes = retab.ingest_and_serialize(gm, sd, gs, quantized_ep)
        t_ingest_only = time.perf_counter() - t0

        t_passes_serialize = t_total - t_ingest_only

        if i >= warmup:
            times_total.append(t_total)
            times_ingest_only.append(t_ingest_only)
            times_passes_serialize.append(t_passes_serialize)
            print(
                f"  Rust run {i - warmup + 1}: "
                f"total={t_total*1000:.1f}ms "
                f"(ingest+serialize-only={t_ingest_only*1000:.1f}ms + approx_prepass={t_passes_serialize*1000:.1f}ms) "
                f"output={len(tosa_bytes)} bytes / ingest_only={len(ingested_bytes)} bytes"
            )

    return times_total, times_ingest_only, times_passes_serialize


def main():
    model_names = sorted(all_model_specs())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_layernorm4d",
                        choices=model_names)
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip the slow Python path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"{'='*60}")

    print("\nPreparing quantized model (shared setup)...")
    t0 = time.perf_counter()
    quantized_ep, compile_spec = prepare_quantized_ep(args.model)
    print(f"  Setup took {time.perf_counter() - t0:.1f}s")

    node_count = len(list(quantized_ep.graph_module.graph.nodes))
    param_count = len(quantized_ep.state_dict)
    print(f"  Graph: {node_count} nodes, {param_count} state_dict entries")

    # Rust path
    print(f"\n{'='*60}")
    print("RUST PATH (retab)")
    print(f"{'='*60}")
    rust_total, rust_ingest_only, rust_lower = bench_rust_path(
        quantized_ep, warmup=1, runs=5
    )

    avg_rust = sum(rust_total) / len(rust_total)
    avg_ingest = sum(rust_ingest_only) / len(rust_ingest_only)
    avg_lower = sum(rust_lower) / len(rust_lower)

    print(f"\n  Average: {avg_rust*1000:.1f}ms total")
    print(f"    Ingest+Serialize-only: {avg_ingest*1000:.1f}ms")
    print(f"    Approx pre-pass work:  {avg_lower*1000:.1f}ms")

    # Python path
    if not args.skip_python:
        print(f"\n{'='*60}")
        print("PYTHON PATH (ExecuTorch ARM backend)")
        print(f"{'='*60}")
        print("  (includes TOSA passes + Vela compilation)")
        python_times = bench_python_path(quantized_ep, compile_spec, warmup=0, runs=1)

        avg_python = sum(python_times) / len(python_times)
        print(f"\n  Average: {avg_python:.3f}s")

        # Speedup
        print(f"\n{'='*60}")
        print("SPEEDUP")
        print(f"{'='*60}")
        print(f"  Python: {avg_python:.3f}s")
        print(f"  Rust:   {avg_rust*1000:.1f}ms")
        print(f"  Ratio:  {avg_python / avg_rust:.0f}x faster")
        print(f"  Note: Python includes Vela; Rust inner path here does not.")
    else:
        print("\n  (Python path skipped)")

    print()


if __name__ == "__main__":
    main()
