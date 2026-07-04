"""RETAB: a Rust fast path for ExecuTorch's ARM backend preprocess stage.

Usage:
    from retab import patch_tosa_backend, unpatch_tosa_backend
    patch_tosa_backend()  # installs the ExecuTorch Arm fast path
"""

# Re-export the Rust native module functions so that existing code like
# ``import retab; retab.run_fx_passes(...)`` keeps working.
from retab._native import *  # noqa: F401,F403

_EXECUTORCH_ARM_EXPORTS = {
    "FastTOSABackend",
    "_all_rust_preprocess",
    "activated",
    "fast_preprocess",
    "install",
    "patch_tosa_backend",
    "uninstall",
    "unpatch_tosa_backend",
}

_PT2E_FASTPATH_EXPORTS = {
    "activated_pt2e_fastpaths",
    "install_pt2e_fastpaths",
    "uninstall_pt2e_fastpaths",
}


def __getattr__(name):
    if name in _PT2E_FASTPATH_EXPORTS:
        from retab.integration import __dict__ as integration_api

        return integration_api[name]
    if name in _EXECUTORCH_ARM_EXPORTS:
        from retab.integration.executorch_arm import __dict__ as integration_api

        return integration_api[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _EXECUTORCH_ARM_EXPORTS | _PT2E_FASTPATH_EXPORTS)
