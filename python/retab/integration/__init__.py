"""Integration surfaces for external hosts."""

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
from retab.integration.torchao_pt2e import (  # noqa: F401
    activated as activated_pt2e_fastpaths,
    install as install_pt2e_fastpaths,
    uninstall as uninstall_pt2e_fastpaths,
)
