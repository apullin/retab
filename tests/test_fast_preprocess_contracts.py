"""Unit tests for fast_preprocess contract/edit helpers."""

from dataclasses import dataclass, field

import pytest
import torch

from fast_preprocess_cf_rewrites import _ControlFlowOutputEdit
from fast_preprocess_contracts import (
    _ControlFlowBinaryOperandSpec,
    _ControlFlowBoundarySite,
    _ControlFlowInputBoundaryContract,
    _ControlFlowOutputBoundaryContract,
    _ControlFlowOutputContract,
)
from fast_preprocess_control_flow import _ControlFlowRescaleEdit
from fast_preprocess_control_flow import (
    _reapply_control_flow_submodule_boundary_contracts,
    _reconstruct_control_flow_rescale_source_qparam,
    _restore_input_boundary_contracts,
    _retune_rescale_qparams,
)


@dataclass
class _DummyNode:
    name: str
    op: str = "call_function"
    target: object = "dummy.target"
    args: tuple = ()
    meta: dict = field(default_factory=dict)
    users: set = field(default_factory=set)

    __hash__ = object.__hash__

    @property
    def all_input_nodes(self):
        return tuple(arg for arg in self.args if hasattr(arg, "op"))

    def update_arg(self, idx, value):
        args = list(self.args)
        args[idx] = value
        self.args = tuple(args)

    def replace_input_with(self, old, new):
        self.args = tuple(new if arg is old else arg for arg in self.args)


def _qparam(*, scale, zp, dtype):
    from executorch.backends.arm._passes.quant_args import QuantArgs

    qinfo = torch.iinfo(dtype)
    return QuantArgs(
        scale=scale,
        zp=zp,
        qmin=int(qinfo.min),
        qmax=int(qinfo.max),
        dtype=dtype,
    )


def test_boundary_site_factories_capture_expected_fields():
    submodule = object()
    placeholder = _DummyNode("ph", op="placeholder")
    output_node = _DummyNode("out", op="output")
    cf_node = _DummyNode("cf")

    input_site = _ControlFlowBoundarySite.for_input(
        submodule=submodule,
        index=3,
        placeholder=placeholder,
        cf_node=cf_node,
    )
    output_site = _ControlFlowBoundarySite.for_output(
        submodule=submodule,
        index=1,
        output_node=output_node,
        output_arg=placeholder,
        cf_node=cf_node,
    )

    assert input_site.boundary_kind == "input"
    assert input_site.placeholder is placeholder
    assert input_site.cf_node is cf_node
    assert output_site.boundary_kind == "output"
    assert output_site.output_node is output_node
    assert output_site.output_arg is placeholder


def test_rescale_edit_factory_binds_input_boundary_contract():
    placeholder = _DummyNode("ph", op="placeholder")
    contract = _ControlFlowInputBoundaryContract.from_site(
        _ControlFlowBoundarySite.for_input(
            submodule="submodule",
            index=0,
            placeholder=placeholder,
        ),
        input_qparam="in_q",
        output_qparam="out_q",
    )

    edit = _ControlFlowRescaleEdit.from_input_boundary_contract(
        contract,
        current_node="current_boundary",
        preserve_boundary=True,
    )

    assert edit.submodule == "submodule"
    assert edit.anchor is placeholder
    assert edit.source_node is placeholder
    assert edit.input_qparam == "in_q"
    assert edit.output_qparam == "out_q"
    assert edit.current_node == "current_boundary"
    assert edit.after is True
    assert edit.preserve_boundary is True


def test_rescale_edit_factory_binds_binary_operand_spec():
    source_node = _DummyNode("source")
    operand_spec = _ControlFlowBinaryOperandSpec(
        arg_idx=1,
        arg=_DummyNode("arg"),
        source_node=source_node,
        qparam="source_q",
    )

    edit = _ControlFlowRescaleEdit.from_binary_operand_spec(
        submodule="submodule",
        anchor="anchor",
        operand_spec=operand_spec,
        output_qparam="dest_q",
        current_node="existing",
    )

    assert edit.submodule == "submodule"
    assert edit.anchor == "anchor"
    assert edit.source_node is source_node
    assert edit.input_qparam == "source_q"
    assert edit.output_qparam == "dest_q"
    assert edit.current_node == "existing"


def test_rescale_edit_factory_binds_output_boundary_contract():
    output_arg = _DummyNode("output_arg")
    output_node = _DummyNode("output_node", op="output", args=((output_arg,),))
    contract = _ControlFlowOutputBoundaryContract.from_site(
        _ControlFlowBoundarySite.for_output(
            submodule="submodule",
            index=0,
            output_node=output_node,
            output_arg=output_arg,
        ),
        target_qparam="dest_q",
    )

    edit = _ControlFlowRescaleEdit.from_output_boundary_contract(
        contract,
        source_node="source",
        input_qparam="source_q",
        current_node="existing",
    )

    assert edit.submodule == "submodule"
    assert edit.anchor is output_arg
    assert edit.source_node == "source"
    assert edit.input_qparam == "source_q"
    assert edit.output_qparam == "dest_q"
    assert edit.current_node == "existing"


def test_output_edit_from_contract_retargets_layout_wrapper():
    source = _DummyNode(
        "source",
        meta={"val": torch.randn(1, 4, 4, 8), "tosa_dim_order": (0, 2, 3, 1)},
    )
    layout_wrapper = _DummyNode(
        "layout_wrapper",
        target="tosa.TRANSPOSE",
        args=(source,),
        meta={"val": torch.randn(1, 4, 4, 8), "tosa_dim_order": (0, 2, 3, 1)},
    )
    output = _DummyNode(
        "output_rescale",
        target="tosa.RESCALE",
        args=(layout_wrapper,),
        meta={"val": torch.randn(1, 4, 4, 8)},
    )
    contract = _ControlFlowOutputContract(
        submodule_output=output,
        submodule_user=_DummyNode("parent_user"),
        desired_fake_tensor=torch.randn(1, 4, 4, 8),
        desired_dim_order=(0, 2, 3, 1),
    )

    edit = _ControlFlowOutputEdit.from_contract(contract)

    assert edit is not None
    assert edit.output_value is output
    assert edit.contract_node is output
    assert edit.rewire_source is source


def test_output_edit_from_contract_returns_metadata_only_edit():
    output = _DummyNode(
        "plain_output",
        target="aten.add.Tensor",
        meta={"val": torch.randn(2, 3, dtype=torch.float32)},
    )
    contract = _ControlFlowOutputContract(
        submodule_output=output,
        submodule_user=_DummyNode("parent_user"),
        desired_fake_tensor=torch.randint(-4, 4, (2, 3), dtype=torch.int8),
        desired_dim_order=None,
    )

    edit = _ControlFlowOutputEdit.from_contract(contract)

    assert edit is not None
    assert edit.output_value is output
    assert edit.contract_node is output
    assert edit.structural is False


def test_restore_input_boundary_contracts_reuses_matching_boundary(monkeypatch):
    placeholder = _DummyNode("ph", op="placeholder")
    target_user = _DummyNode("target", args=(placeholder,))
    boundary = _DummyNode(
        "boundary",
        target="tosa.RESCALE",
        args=(placeholder, torch.int8, [0.5], 0, 0),
        meta={},
        users={target_user},
    )
    placeholder.users = {boundary, target_user}
    input_qparam = _qparam(scale=0.25, zp=3, dtype=torch.int8)
    output_qparam = _qparam(scale=0.5, zp=7, dtype=torch.int8)
    boundary.meta["input_qparams"] = {0: input_qparam}
    boundary.meta["output_qparams"] = {0: output_qparam}
    boundary.meta["arm_preserve_control_flow_quant_boundary"] = True
    contract = _ControlFlowInputBoundaryContract.from_site(
        _ControlFlowBoundarySite.for_input(
            submodule="submodule",
            index=0,
            placeholder=placeholder,
        ),
        input_qparam=input_qparam,
        output_qparam=output_qparam,
    )

    def _fake_ensure(self):
        return self.current_node, False

    monkeypatch.setattr(_ControlFlowRescaleEdit, "ensure", _fake_ensure)

    applied, modified = _restore_input_boundary_contracts(
        [contract],
        user_selector=lambda current: [target_user],
        preserve_boundary=True,
    )

    assert modified is True
    assert len(applied) == 1
    assert applied[0][1] is boundary
    assert target_user.args == (boundary,)


def test_reapply_output_boundary_contracts_use_shared_output_apply_path():
    output_arg = _DummyNode("out_arg")
    output_node = _DummyNode("output", op="output", args=(output_arg,))
    submodule = type(
        "_DummySubmodule",
        (),
        {"graph": type("_DummyGraph", (), {"output_node": lambda self: output_node})()},
    )()
    target_qparam = _qparam(scale=0.75, zp=5, dtype=torch.int8)
    contract = _ControlFlowOutputBoundaryContract.from_site(
        _ControlFlowBoundarySite.for_output(
            submodule=submodule,
            index=0,
            output_node=output_node,
            output_arg=output_arg,
        ),
        target_qparam=target_qparam,
    )

    class _DummyRescalePass:
        def __init__(self):
            self.calls = []

        def _rescale_submodule_outputs(self, submodule_arg, output_qparams_map):
            self.calls.append((submodule_arg, output_qparams_map))
            return True

    rescale_pass = _DummyRescalePass()

    modified = _reapply_control_flow_submodule_boundary_contracts(
        submodule,
        rescale_pass=rescale_pass,
        output_contracts=[contract],
        include_inputs=False,
        include_outputs=True,
    )

    assert modified is True
    assert rescale_pass.calls == [(submodule, {0: target_qparam})]


def test_reconstruct_control_flow_rescale_source_qparam_from_binary_input_qparams():
    source = _DummyNode(
        "source",
        meta={"val": torch.zeros(2, dtype=torch.int8)},
    )
    rescale = _DummyNode(
        "lift",
        target="tosa.RESCALE",
        args=(source, torch.int32, [0.25], 6, 0),
        meta={},
    )
    output_qparam = _qparam(scale=2.0, zp=0, dtype=torch.int32)
    binary = _DummyNode(
        "binary",
        meta={"input_qparams": {0: output_qparam}},
    )

    reconstructed = _reconstruct_control_flow_rescale_source_qparam(binary, 0, rescale)

    assert reconstructed is not None
    assert reconstructed.dtype == torch.int8
    assert float(reconstructed.scale) == pytest.approx(0.5)
    assert int(reconstructed.zp) == 6
    assert source.meta["output_qparams"][0] == reconstructed


def test_retune_rescale_qparams_updates_int32_lift_signature():
    input_qparam = _qparam(scale=0.125, zp=0, dtype=torch.int8)
    output_qparam = _qparam(scale=0.5, zp=0, dtype=torch.int32)
    rescale = _DummyNode(
        "lift",
        target="tosa.RESCALE",
        args=("src", torch.int32, [1.0], 4, 9),
        meta={},
    )

    _retune_rescale_qparams(
        rescale,
        input_qparam=input_qparam,
        output_qparam=output_qparam,
    )

    assert rescale.args[1] == torch.int32
    assert rescale.args[2] == [0.25]
    assert rescale.args[3] == 0
    assert rescale.args[4] == 0
