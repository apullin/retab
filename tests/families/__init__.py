from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules


def load_all_families() -> None:
    package = import_module(__name__)
    for module_info in sorted(iter_modules(package.__path__), key=lambda info: info.name):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")
