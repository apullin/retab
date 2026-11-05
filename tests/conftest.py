"""Skip tests whose optional dependencies are missing instead of failing at import.

Markers:
  native      needs the compiled retab._native extension
  executorch  needs ExecuTorch with the Arm backend
  refmodel    needs tosa_reference_model (from tosa-tools)
"""

from __future__ import annotations

import importlib.util

import pytest

_REQUIREMENTS = {
    "native": ("retab._native", "the retab._native extension is not built"),
    "executorch": ("executorch.backends.arm", "ExecuTorch's Arm backend is not installed"),
    "refmodel": ("tosa_reference_model", "tosa_reference_model is not installed"),
}


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def pytest_configure(config):
    for name, (_, reason) in _REQUIREMENTS.items():
        config.addinivalue_line("markers", f"{name}: skipped when {reason}")


def pytest_collection_modifyitems(config, items):
    missing = {name: reason for name, (module, reason) in _REQUIREMENTS.items() if not _available(module)}
    for item in items:
        for name, reason in missing.items():
            if item.get_closest_marker(name):
                item.add_marker(pytest.mark.skip(reason=reason))
