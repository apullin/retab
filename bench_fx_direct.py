"""Benchmark: Rust-on-FX fold_qdq vs Python fold_qdq.

Run after building:
    bash build.sh
    python bench_fx_direct.py
"""
import copy
import sys
import time

import torch
import torch.nn as nn

# ---- tiny_cnn model (same as the rest of the benchmarks) --------------------

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def get_quantized_ep():
    """Export + quantize TinyCNN, return the ExportedProgram."""
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    model = TinyCNN().eval()
    example = (torch.randn(1, 1, 28, 28),)

    ep = torch.export.export(model, example)
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config()
    )
    prepared = prepare_pt2e(ep, quantizer)
    prepared(*example)  # calibrate
    quantized = convert_pt2e(prepared)
    return quantized


def count_qdq_nodes(gm):
    """Count Q/DQ nodes in a GraphModule."""
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_str = str(node.target)
            if ("quantize_per_tensor" in target_str
                    or "dequantize_per_tensor" in target_str
                    or "dequantize_per_channel" in target_str):
                count += 1
    return count


def main():
    print("Preparing quantized model...")
    ep = get_quantized_ep()
    gm = ep.module()

    total_nodes = len(list(gm.graph.nodes))
    qdq_count = count_qdq_nodes(gm)
    print(f"Graph has {total_nodes} nodes, {qdq_count} Q/DQ nodes")

    # ---- Rust-on-FX benchmark ----
    import retab

    # Warm up
    gm_rust = copy.deepcopy(gm)
    retab.fold_qdq_fx(gm_rust)

    # Summary after fold
    remaining_qdq = count_qdq_nodes(gm_rust)
    remaining_total = len(list(gm_rust.graph.nodes))
    print(f"\nAfter Rust fold_qdq_fx: {remaining_total} nodes, {remaining_qdq} Q/DQ remaining")

    # Quick sanity: print summary
    summary = retab.fx_graph_summary(gm_rust)
    print(f"  Summary has {len(summary)} entries")
    for name, op, target in summary[:5]:
        print(f"    {name}: op={op}, target={target}")
    if len(summary) > 5:
        print(f"    ... ({len(summary) - 5} more)")

    # Timed runs
    N = 20
    times_rust = []
    for _ in range(N):
        gm_copy = copy.deepcopy(gm)
        t0 = time.perf_counter()
        retab.fold_qdq_fx(gm_copy)
        elapsed = time.perf_counter() - t0
        times_rust.append(elapsed)

    avg_rust = sum(times_rust) / len(times_rust)
    min_rust = min(times_rust)
    print(f"\nRust fold_qdq_fx: avg={avg_rust*1000:.2f}ms, min={min_rust*1000:.2f}ms (n={N})")

    # ---- Check for meta annotations ----
    gm_check = copy.deepcopy(gm)
    retab.fold_qdq_fx(gm_check)
    annotated = 0
    for node in gm_check.graph.nodes:
        if node.op == "call_function":
            meta = node.meta
            if meta.get("input_qparams") or meta.get("output_qparams"):
                annotated += 1
    print(f"\nNodes with qparam annotations: {annotated}")

    print("\nDone.")


if __name__ == "__main__":
    main()
