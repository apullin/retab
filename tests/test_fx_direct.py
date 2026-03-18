"""Tests for the FX-graph-direct prototype.

Verifies that:
  1. Rust can read FX graph nodes via PyO3 (fx_graph_summary)
  2. Rust can call FX mutation methods (fold_qdq_fx)
  3. Q/DQ nodes are removed and qparams are annotated
"""
import copy

import torch
import torch.nn as nn


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


def _get_quantized_gm():
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
    prepared(*example)
    quantized = convert_pt2e(prepared)
    return quantized.module()


def _count_qdq(gm):
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            t = str(node.target)
            if "quantize_per_tensor" in t or "dequantize_per_tensor" in t or "dequantize_per_channel" in t:
                count += 1
    return count


def test_fx_graph_summary():
    """Rust can iterate FX graph nodes and read op/target/name."""
    import arm_tosa_engine

    gm = _get_quantized_gm()
    summary = arm_tosa_engine.fx_graph_summary(gm)
    assert len(summary) > 0, "Summary should be non-empty"

    # Each entry is (name, op, target)
    names = {name for name, _, _ in summary}
    ops = {op for _, op, _ in summary}
    assert "placeholder" in ops, f"Expected 'placeholder' op, got {ops}"
    assert "output" in ops, f"Expected 'output' op, got {ops}"
    assert "call_function" in ops, f"Expected 'call_function' op, got {ops}"
    print(f"  fx_graph_summary: {len(summary)} nodes, ops={ops}")


def test_fold_qdq_fx_removes_qdq_nodes():
    """Rust fold_qdq_fx removes all Q/DQ nodes."""
    import arm_tosa_engine

    gm = _get_quantized_gm()
    before = _count_qdq(gm)
    assert before > 0, "Quantized model should have Q/DQ nodes"
    before_total = len(list(gm.graph.nodes))

    modified = arm_tosa_engine.fold_qdq_fx(gm)
    assert modified, "Pass should report modified=True"

    after = _count_qdq(gm)
    after_total = len(list(gm.graph.nodes))
    assert after == 0, f"Expected 0 Q/DQ nodes after fold, got {after}"
    assert after_total < before_total, "Node count should decrease"
    print(f"  fold_qdq_fx: {before_total} -> {after_total} nodes, removed {before} Q/DQ")


def test_fold_qdq_fx_annotates_qparams():
    """After fold, at least some nodes have input_qparams or output_qparams in meta."""
    import arm_tosa_engine

    gm = _get_quantized_gm()
    arm_tosa_engine.fold_qdq_fx(gm)

    annotated = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            iq = node.meta.get("input_qparams")
            oq = node.meta.get("output_qparams")
            if iq or oq:
                annotated += 1
    assert annotated > 0, "Expected some nodes to have qparam annotations"
    print(f"  qparam annotations: {annotated} nodes annotated")


def test_fold_qdq_fx_idempotent():
    """Running fold_qdq_fx twice should be safe (second run is a no-op)."""
    import arm_tosa_engine

    gm = _get_quantized_gm()
    arm_tosa_engine.fold_qdq_fx(gm)
    count_after_first = len(list(gm.graph.nodes))

    modified = arm_tosa_engine.fold_qdq_fx(gm)
    count_after_second = len(list(gm.graph.nodes))
    assert not modified, "Second run should report modified=False"
    assert count_after_first == count_after_second


if __name__ == "__main__":
    print("test_fx_graph_summary...")
    test_fx_graph_summary()
    print("PASSED\n")

    print("test_fold_qdq_fx_removes_qdq_nodes...")
    test_fold_qdq_fx_removes_qdq_nodes()
    print("PASSED\n")

    print("test_fold_qdq_fx_annotates_qparams...")
    test_fold_qdq_fx_annotates_qparams()
    print("PASSED\n")

    print("test_fold_qdq_fx_idempotent...")
    test_fold_qdq_fx_idempotent()
    print("PASSED\n")

    print("All tests passed!")
