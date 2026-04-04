"""Backward-compatibility shim.

Redirects the old ``from fast_preprocess import ...`` import path to the
new ``retab`` package.  Remove once all external callers have migrated.
"""

from retab import (  # noqa: F401
    FastTOSABackend,
    _all_rust_preprocess,
    fast_preprocess,
    patch_tosa_backend,
    unpatch_tosa_backend,
)
