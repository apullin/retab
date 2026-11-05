"""RETAB's TOSABackend patch must round-trip in place and hand out-of-scope graphs to stock."""

import pytest

pytest.importorskip("executorch.backends.arm.tosa.backend")

from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import BackendDetails

import retab.preprocess as rp


def _tosa_backend_classes() -> int:
    return sum(cls.__name__ == "TOSABackend" for cls in BackendDetails.__subclasses__())


def test_patch_then_unpatch_restores_stock_attributes():
    stock = {name: TOSABackend.__dict__[name] for name in ("preprocess", "_preprocess")}
    rp.patch_tosa_backend()
    try:
        assert TOSABackend.__dict__["_preprocess"].__func__ is rp._all_rust_preprocess
        assert TOSABackend.__dict__["preprocess"] is stock["preprocess"]
    finally:
        rp.unpatch_tosa_backend()
    assert all(TOSABackend.__dict__[name] is value for name, value in stock.items())
    assert _tosa_backend_classes() == 1


def test_stock_preprocess_survives_a_direct_patch():
    original = TOSABackend.__dict__["_preprocess"]
    TOSABackend._preprocess = staticmethod(rp._all_rust_preprocess)
    try:
        assert rp._stock_tosa_backend_attrs["_preprocess"] is original
    finally:
        TOSABackend._preprocess = original


@pytest.mark.parametrize(
    "spec, reason",
    [
        ("TOSA-1.0+INT+FP", "float profile (TOSA-1.0+INT+FP)"),
        ("TOSA-1.0+FP", "float profile (TOSA-1.0+FP)"),
        ("TOSA-1.1+INT", "TOSA version 1.1"),
    ],
)
def test_out_of_scope_specs_fall_back_to_stock(spec, reason):
    assert rp.stock_fallback_reason(None, TosaCompileSpec(spec)) == reason


def test_dev_mode_falls_back_to_stock():
    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    compile_spec._set_tosa_dev_mode(True)
    assert rp.stock_fallback_reason(None, compile_spec) == "tosa_dev_mode"
