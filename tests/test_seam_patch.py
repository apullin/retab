"""RETAB's TOSABackend patch must round-trip in place and hand out-of-scope graphs to stock."""

import os

import pytest

pytest.importorskip("executorch.backends.arm.tosa.backend")
pytest.importorskip("retab._native")

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


def test_fast_paths_are_refused_until_rederived_on_rc4(monkeypatch):
    """fast_paths=True raises before anything is patched (Linear APU-117): the
    edge-manager fast path predates rc4 and skips its pre-decomposition transforms."""
    from retab import _decomp_hooks
    from retab.integration import executorch_arm

    monkeypatch.delenv("RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS", raising=False)
    stock = TOSABackend.__dict__["_preprocess"]
    with pytest.raises(NotImplementedError, match="APU-117"):
        with executorch_arm.activated(fast_paths=True, parallel_backend_preprocess_workers=2):
            pass
    with pytest.raises(NotImplementedError, match="APU-117"):
        rp.patch_tosa_backend(fast_paths=True)
    assert TOSABackend.__dict__["_preprocess"] is stock
    assert not _decomp_hooks._ORIGINAL_PATCHES
    assert "RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS" not in os.environ

    with executorch_arm.activated():
        assert not _decomp_hooks._ORIGINAL_PATCHES
        # A refused nested request leaves the outer install as it was.
        with pytest.raises(NotImplementedError):
            executorch_arm.install(fast_paths=True)
        assert rp._is_retab_callable(TOSABackend.__dict__["_preprocess"])
    assert TOSABackend.__dict__["_preprocess"] is stock
    assert executorch_arm._install_depth == 0


def test_uninstall_restores_fast_paths_installed_through_the_private_helpers():
    """The helpers stay for the rc4 re-derivation and deliberate measurements."""
    from retab import _decomp_hooks
    from retab.integration.executorch_arm import activated

    with activated():
        _decomp_hooks._install_edge_decomp_fast_path()
        _decomp_hooks._install_run_decomp_fast_path()
        assert "gen_edge_manager_for_partitioners" in _decomp_hooks._ORIGINAL_PATCHES
    assert not _decomp_hooks._ORIGINAL_PATCHES


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
