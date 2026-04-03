"""Public fast-preprocess shim.

This module keeps the historical import path stable while the implementation
is split into smaller modules. The current implementation lives in
`fast_preprocess_pipeline.py`.
"""

from fast_preprocess_pipeline import (  # noqa: F401
    FastTOSABackend,
    _all_rust_preprocess,
    fast_preprocess,
    patch_tosa_backend,
    unpatch_tosa_backend,
)

__all__ = [
    "FastTOSABackend",
    "_all_rust_preprocess",
    "fast_preprocess",
    "patch_tosa_backend",
    "unpatch_tosa_backend",
]
