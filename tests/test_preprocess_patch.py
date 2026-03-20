#!/usr/bin/env python3
"""Test the dedicated all_rust TOSABackend patch in the full edge flow.

The EthosU pipeline is:
  EthosUBackend.preprocess
    -> TOSABackend._preprocess  (ARM passes + TOSA serialize -- the slow part)
    -> Vela compilation          (fast, ~2s)

We patch TOSABackend._preprocess to use our Rust engine, then run the full
to_edge_transform_and_lower flow and compare results.

Usage:
    bash build.sh && python tests/test_preprocess_patch.py
    bash build.sh && python tests/test_preprocess_patch.py --skip-unpatched
    bash build.sh && python tests/test_preprocess_patch.py --model tiny_cnn
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline")
sys.path.insert(
    0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine"
)

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prepare_quantized_ep(model_name):
    """Export + quantize the model. Shared across all paths."""
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from model import get_model
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


def run_edge_lower(quantized_ep, compile_spec):
    """Run to_edge_transform_and_lower, return (elapsed_seconds, edge_program)."""
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    partitioner = EthosUPartitioner(compile_spec)
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        quantized_ep,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    elapsed = time.perf_counter() - t0
    return elapsed, edge


def main():
    parser = argparse.ArgumentParser(description="Test fast_preprocess integration")
    parser.add_argument(
        "--model",
        default="tiny_cnn",
        choices=["tiny_cnn", "cnn_layernorm4d", "cnn_raw_layernorm"],
    )
    parser.add_argument(
        "--skip-unpatched",
        action="store_true",
        help="Skip the slow unpatched baseline",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("=" * 70)

    # Shared setup
    print("\nPreparing quantized model...")
    t0 = time.perf_counter()
    quantized_ep, compile_spec = prepare_quantized_ep(args.model)
    setup_time = time.perf_counter() - t0
    print(f"  Setup: {setup_time:.1f}s")

    node_count = len(list(quantized_ep.graph_module.graph.nodes))
    print(f"  Graph: {node_count} nodes")

    results = {}

    # --- Unpatched baseline ---
    if not args.skip_unpatched:
        print(f"\n{'=' * 70}")
        print("UNPATCHED (original Python backend + Vela)")
        print("=" * 70)
        try:
            from fast_preprocess import unpatch_tosa_backend

            unpatch_tosa_backend()
            elapsed, edge = run_edge_lower(quantized_ep, compile_spec)
            print(f"  to_edge_transform_and_lower: {elapsed:.3f}s")
            results["unpatched"] = {"time": elapsed}
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            results["unpatched"] = {"time": None, "error": str(e)}

    # --- Dedicated all_rust path ---
    from fast_preprocess import patch_tosa_backend, unpatch_tosa_backend

    print(f"\n{'=' * 70}")
    print("PATCHED: all_rust (dedicated Rust TOSA generation + Vela)")
    print("=" * 70)

    unpatch_tosa_backend()
    patch_tosa_backend()

    try:
        elapsed, edge = run_edge_lower(quantized_ep, compile_spec)
        print(f"  to_edge_transform_and_lower: {elapsed:.3f}s")
        results["all_rust"] = {"time": elapsed}
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["all_rust"] = {"time": None, "error": str(e)}

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        t = res.get("time")
        if t is not None:
            print(f"  {name:20s}: {t:.3f}s")
        else:
            print(f"  {name:20s}: FAILED ({res.get('error', '?')})")

    baseline = results.get("unpatched", {}).get("time")
    patched_time = results.get("all_rust", {}).get("time")
    if baseline and patched_time:
        speedup = baseline / patched_time
        print(f"\n  all_rust vs unpatched: {speedup:.1f}x faster")
        # Note: both include Vela time (~2s), so speedup on the TOSA
        # generation step alone is higher.

    print()


if __name__ == "__main__":
    main()
