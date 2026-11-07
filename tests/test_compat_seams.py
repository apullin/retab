"""Every ExecuTorch seam compat/executorch.toml lists must exist in the installed ExecuTorch."""

import importlib
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+
pytest.importorskip("executorch.backends.arm.tosa.backend")

COMPAT = Path(__file__).resolve().parents[1] / "compat" / "executorch.toml"


def _seams():
    executorch = tomllib.loads(COMPAT.read_text())["executorch"]
    return [
        pytest.param(path, id=f"{table}.{name}")
        for table in ("required_seams", "opt_in_seams")
        for name, path in executorch[table].items()
    ]


def _resolve(path: str):
    """The object at a dotted path: its longest importable module, then attributes."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"no importable module in {path}")


def test_the_inventory_lists_the_preprocess_seam():
    paths = [param.values[0] for param in _seams()]
    assert "executorch.backends.arm.tosa.backend.TOSABackend._preprocess" in paths


@pytest.mark.parametrize("path", _seams())
def test_seam_exists_in_the_installed_executorch(path):
    assert _resolve(path) is not None
