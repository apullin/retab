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
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model = TinyCNN().eval()
    example = (torch.randn(1, 1, 28, 28),)
    gm = torch.export.export(model, example).module()
    compile_spec = EthosUCompileSpec(target="ethos-u55-128")
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(gm, quantizer)
    prepared(*example)
    quantized = convert_pt2e(prepared)
    return quantized


def _count_qdq(gm):
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            t = str(node.target)
            if "quantize_per_tensor" in t or "dequantize_per_tensor" in t or "dequantize_per_channel" in t:
                count += 1
    return count


def _qdq_nodes(gm):
    nodes = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if (
            "quantize_per_tensor" in target
            or "dequantize_per_tensor" in target
            or "dequantize_per_channel" in target
        ):
            nodes.append(node)
    return nodes


def _is_boundary_qdq(node):
    target = str(node.target)
    source = node.args[0] if node.args else None

    if "dequantize_per_tensor" in target or "dequantize_per_channel" in target:
        return any(user.op == "output" for user in node.users)

    if "quantize_per_tensor" in target or "quantize_per_channel" in target:
        return isinstance(source, torch.fx.Node) and source.op == "placeholder"

    return False


def _count_interior_qdq(gm):
    return sum(1 for node in _qdq_nodes(gm) if not _is_boundary_qdq(node))


def test_fx_graph_summary():
    """Rust can iterate FX graph nodes and read op/target/name."""
    import retab

    gm = _get_quantized_gm()
    summary = retab.fx_graph_summary(gm)
    assert len(summary) > 0, "Summary should be non-empty"

    # Each entry is (name, op, target)
    names = {name for name, _, _ in summary}
    ops = {op for _, op, _ in summary}
    assert "placeholder" in ops, f"Expected 'placeholder' op, got {ops}"
    assert "output" in ops, f"Expected 'output' op, got {ops}"
    assert "call_function" in ops, f"Expected 'call_function' op, got {ops}"
    print(f"  fx_graph_summary: {len(summary)} nodes, ops={ops}")


def test_fold_qdq_fx_removes_qdq_nodes():
    """Rust fold_qdq_fx removes interior Q/DQ nodes and keeps only graph boundaries."""
    import retab

    gm = _get_quantized_gm()
    before = _count_qdq(gm)
    assert before > 0, "Quantized model should have Q/DQ nodes"
    before_total = len(list(gm.graph.nodes))

    modified = retab.fold_qdq_fx(gm)
    assert modified, "Pass should report modified=True"

    after_nodes = _qdq_nodes(gm)
    boundary_after = [node for node in after_nodes if _is_boundary_qdq(node)]
    interior_after = [node for node in after_nodes if not _is_boundary_qdq(node)]
    after_total = len(list(gm.graph.nodes))
    assert not interior_after, (
        "Expected all interior Q/DQ nodes to be folded, got "
        f"{[node.name for node in interior_after]}"
    )
    assert len(boundary_after) <= 2, (
        "Expected at most input/output boundary Q/DQ nodes to remain, got "
        f"{[node.name for node in boundary_after]}"
    )
    assert after_total < before_total, "Node count should decrease"
    print(
        "  fold_qdq_fx: "
        f"{before_total} -> {after_total} nodes, "
        f"boundary_qdq={len(boundary_after)}"
    )


def test_fold_qdq_fx_annotates_qparams():
    """After fold, at least some nodes have input_qparams or output_qparams in meta."""
    import retab

    gm = _get_quantized_gm()
    retab.fold_qdq_fx(gm)

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
    """Running fold_qdq_fx twice should be safe and not reintroduce interior Q/DQ."""
    import retab

    gm = _get_quantized_gm()
    retab.fold_qdq_fx(gm)
    count_after_first = len(list(gm.graph.nodes))
    qdq_after_first = _count_qdq(gm)

    modified = retab.fold_qdq_fx(gm)
    count_after_second = len(list(gm.graph.nodes))
    qdq_after_second = _count_qdq(gm)
    assert _count_interior_qdq(gm) == 0, "Second run should not reintroduce interior Q/DQ"
    assert count_after_second <= count_after_first
    assert qdq_after_second <= qdq_after_first
    print(
        "  fold_qdq_fx second run: "
        f"modified={modified}, nodes={count_after_first}->{count_after_second}, "
        f"qdq={qdq_after_first}->{qdq_after_second}"
    )


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
