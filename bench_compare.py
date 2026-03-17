#!/usr/bin/env python3
"""
Benchmark: Python pass pipeline vs Rust arm-tosa-engine.

Compares the inner operation (post-quantization graph → TOSA flatbuffer)
to isolate the pass pipeline speedup from export/quantization overhead.
"""

import sys
import time
import argparse

sys.path.insert(0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline")

import torch
from model import get_model


def prepare_quantized_ep(model_name):
    """Stages 1+2: export and quantize. Shared setup for both paths."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model, example_input = get_model(model_name)
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
    """Time the Rust pass pipeline (ingest + passes + serialize)."""
    import arm_tosa_engine

    gm = quantized_ep.graph_module
    sd = dict(quantized_ep.state_dict)
    gs = quantized_ep.graph_signature

    times_total = []
    times_ingest = []
    times_lower = []

    for i in range(warmup + runs):
        # Time ingestion separately
        t0 = time.perf_counter()
        ingest_result = arm_tosa_engine.ingest_graph(gm, sd, gs)
        t_ingest = time.perf_counter() - t0

        # Time full lower_to_tosa (includes ingest + passes + serialize)
        t0 = time.perf_counter()
        tosa_bytes = arm_tosa_engine.lower_to_tosa(gm, sd, gs)
        t_total = time.perf_counter() - t0

        t_passes_serialize = t_total - t_ingest  # approximate

        if i >= warmup:
            times_total.append(t_total)
            times_ingest.append(t_ingest)
            times_lower.append(t_passes_serialize)
            print(
                f"  Rust run {i - warmup + 1}: "
                f"total={t_total*1000:.1f}ms "
                f"(ingest={t_ingest*1000:.1f}ms + passes+serialize={t_passes_serialize*1000:.1f}ms) "
                f"output={len(tosa_bytes)} bytes"
            )

    return times_total, times_ingest, times_lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_layernorm4d",
                        choices=["tiny_cnn", "cnn_layernorm4d", "cnn_raw_layernorm"])
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
    print("RUST PATH (arm-tosa-engine)")
    print(f"{'='*60}")
    rust_total, rust_ingest, rust_lower = bench_rust_path(quantized_ep, warmup=1, runs=5)

    avg_rust = sum(rust_total) / len(rust_total)
    avg_ingest = sum(rust_ingest) / len(rust_ingest)
    avg_lower = sum(rust_lower) / len(rust_lower)

    print(f"\n  Average: {avg_rust*1000:.1f}ms total")
    print(f"    Ingest:           {avg_ingest*1000:.1f}ms")
    print(f"    Passes+Serialize: {avg_lower*1000:.1f}ms")

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
        print(f"  Note: Python includes Vela; Rust does not (yet)")
    else:
        print("\n  (Python path skipped)")

    print()


if __name__ == "__main__":
    main()
