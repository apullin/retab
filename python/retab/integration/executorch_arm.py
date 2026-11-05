"""Stable integration surface for ExecuTorch Arm backends."""

import os
from contextlib import contextmanager

from retab.preprocess import (
    FastTOSABackend,
    _all_rust_preprocess,
    fast_preprocess,
    patch_tosa_backend as _patch_tosa_backend,
    unpatch_tosa_backend as _unpatch_tosa_backend,
)

_PARALLEL_BACKEND_PREPROCESS_ENV = "RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS"
_NO_ENV_OVERRIDE = object()
_install_depth = 0
_parallel_backend_preprocess_env_stack: list[object | str | None] = []


def _normalize_parallel_backend_preprocess_workers(workers: int | None) -> int | None:
    if workers is None:
        return None
    if not isinstance(workers, int):
        raise TypeError(
            "parallel_backend_preprocess_workers must be an int or None"
        )
    if workers < 0:
        raise ValueError("parallel_backend_preprocess_workers must be >= 0")
    return workers


def _push_parallel_backend_preprocess_workers_override(
    workers: int | None,
) -> object | str | None:
    workers = _normalize_parallel_backend_preprocess_workers(workers)
    if workers is None:
        return _NO_ENV_OVERRIDE
    previous = os.environ.get(_PARALLEL_BACKEND_PREPROCESS_ENV)
    os.environ[_PARALLEL_BACKEND_PREPROCESS_ENV] = str(workers)
    return previous


def _restore_parallel_backend_preprocess_workers_override(
    previous: object | str | None,
) -> None:
    if previous is _NO_ENV_OVERRIDE:
        return
    if previous is None:
        os.environ.pop(_PARALLEL_BACKEND_PREPROCESS_ENV, None)
        return
    os.environ[_PARALLEL_BACKEND_PREPROCESS_ENV] = previous


def install(
    *, fast_paths: bool = False, parallel_backend_preprocess_workers: int | None = None
) -> None:
    """Install RETAB's preprocess into ExecuTorch Arm lowering.

    `fast_paths` also installs RETAB's edge-manager and run_decompositions fast
    paths (off by default). `parallel_backend_preprocess_workers` is an opt-in
    escape hatch for highly fragmented lowers that fan out into many delegates.
    Leave it unset for the default serial behavior. Nested installs keep the
    options of the outermost one.
    """
    global _install_depth

    previous = _push_parallel_backend_preprocess_workers_override(
        parallel_backend_preprocess_workers
    )
    try:
        if _install_depth == 0:
            _patch_tosa_backend(fast_paths=fast_paths)
    except Exception:
        _restore_parallel_backend_preprocess_workers_override(previous)
        raise

    _parallel_backend_preprocess_env_stack.append(previous)
    _install_depth += 1


def uninstall() -> None:
    """Restore the default ExecuTorch Arm lowering path."""
    global _install_depth

    previous = (
        _parallel_backend_preprocess_env_stack.pop()
        if _parallel_backend_preprocess_env_stack
        else _NO_ENV_OVERRIDE
    )
    try:
        if _install_depth > 0:
            _install_depth -= 1
            if _install_depth == 0:
                _unpatch_tosa_backend()
    finally:
        _restore_parallel_backend_preprocess_workers_override(previous)


patch_tosa_backend = install
unpatch_tosa_backend = uninstall


@contextmanager
def activated(
    *, fast_paths: bool = False, parallel_backend_preprocess_workers: int | None = None
):
    """Temporarily install RETAB's preprocess into ExecuTorch Arm lowering.

    See install() for `fast_paths` and `parallel_backend_preprocess_workers`.
    """
    install(
        fast_paths=fast_paths,
        parallel_backend_preprocess_workers=parallel_backend_preprocess_workers,
    )
    try:
        yield
    finally:
        uninstall()
