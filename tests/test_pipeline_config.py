"""Compile-spec settings at the Python/Rust pipeline boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("executorch.backends.arm.ethosu.compile_spec")
pytest.importorskip("retab._native")

from executorch.backends.arm.common.pipeline_config import (
    LeakyReLULoweringConfig,
    QuantizeInfConfig,
    SDPASafeSoftmaxGuardPolicy,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec
from retab import _native
from retab import preprocess as rp

pytestmark = [pytest.mark.native, pytest.mark.executorch]


@pytest.mark.parametrize(
    "target,tosa_spec,softmax",
    [
        ("ethos-u55-128", "TOSA-1.0+INT+int16+int4+u55", "STABLE"),
        ("ethos-u85-256", "TOSA-1.0+INT+int16+int4+cf", "MASKED"),
    ],
)
def test_pipeline_config_from_ethosu_compile_spec(monkeypatch, target, tosa_spec, softmax):
    monkeypatch.delenv("RETAB_LAYOUT", raising=False)
    compile_spec = EthosUCompileSpec(target)
    assert rp._build_pipeline_config(compile_spec) == {
        "tosa_spec": tosa_spec,
        "tosa_dev_mode": False,
        "softmax": softmax,
        "leaky_relu": "DECOMPOSE",
        "quantize_inf": {"neg_inf": -256.0, "pos_inf": 255.0},
        "sdpa_safe_softmax_guard": "PRESERVE",
        "layout": "dim_order",
    }
    assert not rp._has_non_default_pipeline_config(compile_spec)
    compile_spec._set_tosa_dev_mode(True)
    assert rp._build_pipeline_config(compile_spec)["tosa_dev_mode"] is True


@pytest.mark.parametrize("layout", ["dim_order", "explicit"])
def test_retab_layout_sets_the_layout_without_sending_graphs_to_stock(monkeypatch, layout):
    monkeypatch.setenv("RETAB_LAYOUT", layout)
    compile_spec = EthosUCompileSpec("ethos-u55-128")
    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    assert rp._build_pipeline_config(compile_spec)["layout"] == layout
    assert rp.stock_fallback_reason(program, compile_spec) is None


@pytest.mark.parametrize("value", ["", "DIM_ORDER", "Explicit", "dim-order", " explicit"])
def test_unknown_retab_layout_is_an_error(monkeypatch, value):
    monkeypatch.setenv("RETAB_LAYOUT", value)
    with pytest.raises(ValueError, match="RETAB_LAYOUT must be 'dim_order' or 'explicit'"):
        rp._build_pipeline_config(EthosUCompileSpec("ethos-u55-128"))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("softmax", SoftmaxDecompositionConfig.MASKED, "non-default pass pipeline config"),
        ("leaky_relu", LeakyReLULoweringConfig.TABLE, "non-default pass pipeline config"),
        (
            "quantize_inf",
            QuantizeInfConfig(neg_inf=-100.0, pos_inf=200.0),
            "non-default pass pipeline config",
        ),
        (
            "sdpa_safe_softmax_guard",
            SDPASafeSoftmaxGuardPolicy.REMOVE,
            "sdpa_safe_softmax_guard=REMOVE is not supported by RETAB",
        ),
        (
            "sdpa_safe_softmax_guard",
            SDPASafeSoftmaxGuardPolicy.AUTO,
            "sdpa_safe_softmax_guard=AUTO is not supported by RETAB",
        ),
    ],
)
def test_non_default_configs_hand_the_original_program_to_stock(
    monkeypatch, field, value, reason
):
    compile_spec = EthosUCompileSpec("ethos-u55-128")
    compile_spec.set_pass_pipeline_config(
        replace(compile_spec._create_default_pipeline_config(), **{field: value})
    )
    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    calls = []
    result = object()

    def stock_preprocess(edge_program, spec):
        calls.append((edge_program, spec))
        return result

    monkeypatch.setattr(rp, "stock_preprocess", stock_preprocess)
    assert rp.stock_fallback_reason(program, compile_spec) == reason
    assert rp._all_rust_preprocess(program, compile_spec) is result
    assert calls == [(program, compile_spec)]


@pytest.mark.parametrize(
    "entry_point",
    ["run_fx_passes", "run_fx_cond_passes", "run_fx_submodule_passes"],
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("sdpa_safe_softmax_guard", "REMOVE"),
        ("sdpa_safe_softmax_guard", "AUTO"),
        ("leaky_relu", "TABLE"),
    ],
)
def test_native_entry_points_refuse_unported_policies(entry_point, field, value):
    config = rp._build_pipeline_config(EthosUCompileSpec("ethos-u55-128"))
    config[field] = value
    with pytest.raises(ValueError, match=f"{field}={value}.*use stock preprocess"):
        getattr(_native, entry_point)(None, pipeline_config=config)


@pytest.mark.parametrize(
    "entry_point",
    ["run_fx_passes", "run_fx_cond_passes", "run_fx_submodule_passes"],
)
def test_native_entry_points_refuse_unknown_layouts(entry_point):
    config = rp._build_pipeline_config(EthosUCompileSpec("ethos-u55-128"))
    config["layout"] = "nhwc"
    with pytest.raises(ValueError, match='unknown layout config: "nhwc"'):
        getattr(_native, entry_point)(None, pipeline_config=config)


def _main_block_ops(tosa: bytes) -> list[int]:
    from tosa import TosaGraph

    block = TosaGraph.TosaGraph.GetRootAs(bytearray(tosa), 0).Regions(0).Blocks(0)
    return [block.Operators(i).Op() for i in range(block.OperatorsLength())]


def test_missing_layout_means_dim_order_and_explicit_mode_rewrites_the_conv():
    """A missing key means dim_order. Explicit mode builds rc4's pipeline (APU-157): the
    conv becomes a channels-last TOSA CONV2D on a pre-permuted weight between explicit
    TRANSPOSEs, so nothing reaches the serializer's aten-conv guard (P2 design X5) and
    the graph stays with RETAB."""
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from tosa import Op

    # Frozen parameters, as a lowered program has them: the importer reads the weight's
    # data, which IrRewriteConv pre-permutes.
    module = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.ReLU()).requires_grad_(False)
    program = torch.export.export(module, (torch.ones(1, 2, 8, 8),)).run_decompositions()
    config = rp._build_pipeline_config(TosaCompileSpec("TOSA-1.0+INT"))
    del config["layout"]
    tosa = []
    for layout in [None, "dim_order", "explicit"]:
        layout_config = config if layout is None else {**config, "layout": layout}
        _native.run_fx_passes(program.graph_module, program, pipeline_config=layout_config)
        tosa.append(_native.ingest_and_serialize())
    assert tosa[0] and tosa[0] == tosa[1]

    ops = _main_block_ops(tosa[2])
    assert tosa[2] != tosa[0]
    assert ops.count(Op.Op.CONV2D) == 1
    # The input permute to NHWC and the output permute back; the weight is a constant.
    assert ops.count(Op.Op.TRANSPOSE) == 2
    conv = ops.index(Op.Op.CONV2D)
    assert ops.index(Op.Op.TRANSPOSE) < conv < len(ops) - 1 - ops[::-1].index(Op.Op.TRANSPOSE)


class _CondModule(torch.nn.Module):
    def forward(self, x):
        def true_branch(value):
            return value + 1

        def false_branch(value):
            return value - 1

        return torch.cond(x.sum() > 0, true_branch, false_branch, [x])


def test_region_entry_points_reject_a_layout_mode_the_parent_did_not_run():
    """P2 design §4.1: a delegate's parent pipeline and its regions run one layout mode.
    `_all_rust_preprocess` hands every entry point the same config, so a region in the
    other mode means a caller passed two configs: a hard error, not a stock fallback."""
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.exir.graph_module import get_cond_while_submodules

    program = torch.export.export(_CondModule(), (torch.ones(2, 2),)).run_decompositions()
    config = rp._build_pipeline_config(TosaCompileSpec("TOSA-1.0+INT+cf"))
    explicit = {**config, "layout": "explicit"}
    dim_order = {**config, "layout": "dim_order"}
    (name, branch, _), *_ = get_cond_while_submodules(program.graph_module)

    for parent, region in [(explicit, dim_order), (dim_order, explicit)]:
        _native.run_fx_passes(program.graph_module, program, pipeline_config=parent)
        with pytest.raises(RuntimeError, match="must share one pipeline config") as error:
            _native.run_fx_cond_passes(branch, program, name, pipeline_config=region)
        assert rp._native_fallback_reason(error.value) is None, error.value
        with pytest.raises(RuntimeError, match="must share one pipeline config"):
            _native.run_fx_submodule_passes(branch, program, name, pipeline_config=region)
        _native.run_fx_cond_passes(branch, program, name, pipeline_config=parent)


@pytest.mark.parametrize("recursive", [False, True])
def test_main_and_nested_regions_receive_the_same_pipeline_config(monkeypatch, recursive):
    from executorch.exir import graph_module

    compile_spec = EthosUCompileSpec("ethos-u55-128")
    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    cond_branch, loop_body, nested_branch = [
        torch.fx.symbolic_trace(torch.nn.ReLU()) for _ in range(3)
    ]
    cond, loop = SimpleNamespace(target="cond"), SimpleNamespace(target="while_loop")
    regions = {
        program.graph_module: [("true_graph", cond_branch, cond)],
        cond_branch: [("body_graph", loop_body, loop)],
        loop_body: [("true_graph", nested_branch, cond)],
        nested_branch: [],
    }
    monkeypatch.setattr(graph_module, "get_cond_while_submodules", lambda gm: regions[gm])
    monkeypatch.setenv("RETAB_FORCE_CF_RECURSIVE_PASSES", "1" if recursive else "0")
    monkeypatch.setattr(rp, "stock_fallback_reason", lambda ep, spec: None)
    calls = []

    def record(kind):
        def run(gm, ep, region_name=None, *, pipeline_config, u55_transpose_probe=None):
            calls.append((kind, gm, ep, region_name, pipeline_config))

        return run

    for kind in ["run_fx_passes", "run_fx_cond_passes", "run_fx_submodule_passes"]:
        monkeypatch.setattr(_native, kind, record(kind))
    monkeypatch.setattr(_native, "ingest_and_serialize", lambda *args: b"tosa")

    result = rp._all_rust_preprocess(program, compile_spec)

    assert result.processed_bytes == b"tosa"
    assert calls[0][:4] == ("run_fx_passes", program.graph_module, program, None)
    config = calls[0][4]
    assert config == {**rp._build_pipeline_config(compile_spec), "u55_split_permutes": []}
    assert all(call[2] is program and call[4] is config for call in calls)
    assert [(kind, region) for kind, _, _, region, _ in calls if region is not None] == [
        ("run_fx_cond_passes", "true_graph"),
        ("run_fx_submodule_passes", "true_graph__body_graph"),
        ("run_fx_cond_passes", "true_graph__body_graph__true_graph"),
    ]
    assert sum(call[0] == "run_fx_passes" for call in calls) == (4 if recursive else 1)


@pytest.mark.parametrize("layout", ["dim_order", "explicit"])
def test_explicit_mode_hands_the_u55_split_rc4s_exact_check(monkeypatch, layout):
    """Explicit mode on U55 gives run_fx_passes rc4's exact TRANSPOSE check, which the
    U55 split asks about the permutes alive at its slot, and no edge names; dim-order
    mode gives it the edge names and no check (APU-160). The check runs Vela in a
    lowering context: a batch-1 NCHW-to-NHWC transpose of a 320x320 image passes, which
    rc4's check called outside one would reject."""
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec

    monkeypatch.setenv("RETAB_LAYOUT", layout)
    monkeypatch.setattr(rp, "stock_fallback_reason", lambda ep, spec: None)
    monkeypatch.setattr(rp, "_u55_permutes_to_split", lambda ep, spec: ["edge_permute"])
    calls = []

    def run(gm, ep, *, pipeline_config, u55_transpose_probe=None):
        calls.append((pipeline_config["u55_split_permutes"], u55_transpose_probe))

    monkeypatch.setattr(_native, "run_fx_passes", run)
    monkeypatch.setattr(_native, "ingest_and_serialize", lambda: b"tosa")
    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    for spec in ["TOSA-1.0+INT+int16+int4+u55", "TOSA-1.0+INT+int16+int4+cf"]:
        rp._all_rust_preprocess(program, TosaCompileSpec(spec))

    (u55_names, probe), (u85_names, u85_probe) = calls
    assert u85_names == ["edge_permute"] and u85_probe is None
    if layout == "dim_order":
        assert u55_names == ["edge_permute"] and probe is None
        return
    assert u55_names == [] and probe is not None
    pytest.importorskip("ethosu.vela")
    assert probe([2, 2, 257, 257], [0, 2, 3, 1], "torch.int8") is True
    assert probe([1, 3, 320, 320], [0, 2, 3, 1], "torch.int8") is False
    assert probe([1, 3, 320, 320], [0, 2, 3, 1], "torch.int32") is True
    assert probe([1, 3, 320, 320], [0, 2, 3, 1], "") is True
