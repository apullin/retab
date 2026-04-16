"""Backward-compatibility shim."""

from retab.integration.executorch_arm import (  # noqa: F401
    FastTOSABackend,
    _all_rust_preprocess,
    activated,
    fast_preprocess,
    install,
    patch_tosa_backend,
    unpatch_tosa_backend,
    uninstall,
)
