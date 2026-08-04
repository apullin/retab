"""Drop-in replacement for ExecuTorch Arm TOSA preprocess using the Rust engine.

Hooks into the real ARM backend pipeline at preprocess() time, where the FX
graph already has ARM annotations from to_edge_transform_and_lower. The
dedicated path is:

  1. FX-direct Rust passes on the live Python FX graph
  2. Rust ingestion of the transformed graph
  3. Rust TOSA flatbuffer serialization

The call chain is:
  EthosUBackend.preprocess(ep, compile_specs)
    -> TOSABackend._preprocess(ep, tosa_compile_spec)  <-- override or patch
    -> EthosUBackend._compile_tosa_flatbuffer(tosa_bytes, ...)  (Vela)

By overriding the TOSA preprocess stage, we only swap the TOSA generation
step. Vela compilation still happens normally downstream.
"""

import logging
import os
import time
from collections.abc import Callable
from typing import List

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

from retab._cf_ctx import _collect_delegate_info_meta
from retab._decomp_hooks import (
    _install_edge_decomp_fast_path,
    _install_run_decomp_fast_path,
    _uninstall_edge_decomp_fast_path,
    _uninstall_run_decomp_fast_path,
)

logger = logging.getLogger(__name__)

_PARALLEL_BACKEND_MULTIMETHOD_ATTR = "preprocess_multimethod"
_parallel_backend_multimethod_originals: dict[type, tuple[bool, object | None]] = {}
_parallel_backend_preprocess_jobs = None


def _env_get(name: str, *aliases: str, default: str | None = None) -> str | None:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return default


def _run_control_flow_submodule_rust_passes(
    gm, edge_program, retab, region_prefix: str | None = None
) -> None:
    try:
        from executorch.backends.arm._passes.arm_pass_utils import (
            get_cond_while_submodules_nested,
        )
    except ImportError:
        # This ExecuTorch build does not expose the control-flow submodule
        # helper. RETAB's recursive cond/while handling is only needed for
        # models that contain control flow; models without it (the common case,
        # e.g. CNN encoders) lower correctly without this step.
        logger.warning(
            "[fast_preprocess] get_cond_while_submodules_nested unavailable; "
            "skipping control-flow submodule passes (cond/while not supported "
            "on this ExecuTorch build)"
        )
        return

    if (
        _env_get(
            "RETAB_FORCE_CF_RECURSIVE_PASSES",
            "ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES",
        )
        == "1"
    ):
        for name, submodule, _ in get_cond_while_submodules_nested(gm):
            retab.run_fx_passes(submodule, edge_program)
            logger.info(
                "[fast_preprocess] ran recursive fx pass pipeline on control-flow submodule %s",
                name,
            )

    for name, submodule, cf_node in get_cond_while_submodules_nested(gm):
        region_name = f"{region_prefix}::{name}" if region_prefix else name
        cf_target = str(getattr(cf_node, "target", ""))
        if "cond" in cf_target:
            retab.run_fx_cond_passes(submodule, edge_program, region_name)
        else:
            retab.run_fx_submodule_passes(submodule, edge_program, region_name)
        _run_control_flow_submodule_rust_passes(
            submodule, edge_program, retab, region_name
        )


def _merge_preprocess_state_dict(original_state_dict, edge_program):
    import copy

    merged_state_dict = original_state_dict
    for key, value in dict(edge_program.state_dict).items():
        if key not in merged_state_dict:
            merged_state_dict[key] = copy.deepcopy(value)
    return merged_state_dict


def _sync_input_specs_to_placeholder_order(edge_program: ExportedProgram) -> None:
    """Keep ExportGraphSignature input_specs aligned with FX placeholder order."""
    graph_signature = edge_program.graph_signature
    placeholder_names = [
        node.name
        for node in edge_program.graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    spec_by_name = {spec.arg.name: spec for spec in graph_signature.input_specs}
    ordered_specs = [
        spec_by_name[name] for name in placeholder_names if name in spec_by_name
    ]
    if len(ordered_specs) != len(graph_signature.input_specs):
        return
    if [spec.arg.name for spec in ordered_specs] == [
        spec.arg.name for spec in graph_signature.input_specs
    ]:
        return

    from torch.export.exported_program import ExportGraphSignature

    edge_program._graph_signature = ExportGraphSignature(
        ordered_specs, graph_signature.output_specs
    )


def _maybe_dump_tosa_artifacts(edge_program, compile_spec, tosa_bytes) -> None:
    import json
    from pathlib import Path

    from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
    from executorch.backends.arm.common.debug import debug_tosa_dump
    from executorch.backends.arm.tosa.backend import arm_get_first_delegation_tag

    artifact_path = compile_spec._get_intermediate_path()
    if artifact_path:
        tag = arm_get_first_delegation_tag(edge_program.graph_module)
        suffix = "{}".format(f"_{tag}" if tag else "") + f"_{compile_spec.tosa_spec}"
        debug_tosa_dump(tosa_bytes, artifact_path, suffix=suffix)

        if compile_spec.tosa_debug_mode == ArmCompileSpec.DebugMode.JSON:
            debug_json_path = Path(artifact_path) / "debug.json"
            debug_json_path.parent.mkdir(parents=True, exist_ok=True)
            debug_json_path.write_text(
                json.dumps(
                    {
                        "tosa_spec": str(compile_spec.tosa_spec),
                        "num_bytes": len(tosa_bytes),
                        "delegate_tag": tag,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    dump_path = _env_get("RETAB_DUMP_TOSA", "ARM_TOSA_ENGINE_DUMP_TOSA")
    if dump_path:
        dump_file = Path(dump_path)
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_bytes(tosa_bytes)
        logger.info("[all_rust] dumped TOSA flatbuffer to %s", dump_file)


def _parallel_backend_preprocess_workers() -> int:
    raw = _env_get(
        "RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS",
        "ARM_TOSA_ENGINE_PARALLEL_BACKEND_PREPROCESS_WORKERS",
        default="0",
    )
    try:
        workers = int(raw)
    except ValueError:
        logger.warning(
            "[fast_preprocess] ignoring invalid RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS=%r",
            raw,
        )
        return 0
    return max(0, workers)


def _serial_backend_preprocess_multimethod(
    backend_cls,
    edge_programs,
    compile_specs,
):
    preprocess_results = {}
    for method_name, programs in edge_programs.items():
        assert (
            method_name in compile_specs
        ), f"Error: missing compile specs for {method_name}"
        compile_specs_for_method = compile_specs[method_name]
        assert len(compile_specs_for_method) == len(
            programs
        ), f"Error: method {method_name} has {len(programs)} partitions but only {len(compile_specs_for_method)}"
        results_for_method = []
        for program, compile_spec_for_program in zip(programs, compile_specs_for_method):
            preprocess_result = backend_cls.preprocess(program, compile_spec_for_program)
            results_for_method.append(preprocess_result)
        preprocess_results[method_name] = results_for_method
    return preprocess_results


def _compile_specs_safe_for_parallel(
    parse_compile_spec: Callable[[list[CompileSpec]], object],
    compile_specs,
) -> bool:
    if _env_get("RETAB_DUMP_TOSA", "ARM_TOSA_ENGINE_DUMP_TOSA"):
        return False
    for method_specs in compile_specs.values():
        for specs_for_program in method_specs:
            compile_spec = parse_compile_spec(specs_for_program)
            if compile_spec._get_intermediate_path() is not None:
                return False
    return True


def _run_parallel_backend_preprocess_job(task):
    import importlib

    backend_module, backend_name, job_index = task
    assert _parallel_backend_preprocess_jobs is not None
    method_name, method_index, edge_program, compile_specs = (
        _parallel_backend_preprocess_jobs[job_index]
    )
    backend_cls = getattr(importlib.import_module(backend_module), backend_name)
    return (
        method_name,
        method_index,
        backend_cls.preprocess(edge_program, compile_specs),
    )


def _make_parallel_backend_preprocess_multimethod(
    parse_compile_spec: Callable[[list[CompileSpec]], object],
):
    def _parallel_preprocess_multimethod(
        backend_cls,
        edge_programs,
        compile_specs,
    ):
        import multiprocessing as mp

        workers = _parallel_backend_preprocess_workers()
        total_jobs = sum(len(programs) for programs in edge_programs.values())
        if workers <= 1 or total_jobs <= 1:
            return _serial_backend_preprocess_multimethod(
                backend_cls, edge_programs, compile_specs
            )
        if not _compile_specs_safe_for_parallel(parse_compile_spec, compile_specs):
            logger.info(
                "[fast_preprocess] %s preprocess_multimethod falling back to serial due to debug artifact settings",
                backend_cls.__name__,
            )
            return _serial_backend_preprocess_multimethod(
                backend_cls, edge_programs, compile_specs
            )

        max_workers = min(workers, total_jobs)
        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            logger.info(
                "[fast_preprocess] %s preprocess_multimethod falling back to serial because fork multiprocessing is unavailable",
                backend_cls.__name__,
            )
            return _serial_backend_preprocess_multimethod(
                backend_cls, edge_programs, compile_specs
            )

        logger.info(
            "[fast_preprocess] %s preprocess_multimethod running %d jobs on %d workers",
            backend_cls.__name__,
            total_jobs,
            max_workers,
        )
        jobs = []
        method_to_sizes = {}
        for method_name, programs in edge_programs.items():
            assert (
                method_name in compile_specs
            ), f"Error: missing compile specs for {method_name}"
            compile_specs_for_method = compile_specs[method_name]
            assert len(compile_specs_for_method) == len(
                programs
            ), f"Error: method {method_name} has {len(programs)} partitions but only {len(compile_specs_for_method)}"
            method_to_sizes[method_name] = len(programs)
            for method_index, (program, specs_for_program) in enumerate(
                zip(programs, compile_specs_for_method)
            ):
                jobs.append((method_name, method_index, program, specs_for_program))

        global _parallel_backend_preprocess_jobs
        _parallel_backend_preprocess_jobs = jobs
        try:
            with mp_ctx.Pool(processes=max_workers) as pool:
                completed = pool.map(
                    _run_parallel_backend_preprocess_job,
                    [
                        (backend_cls.__module__, backend_cls.__name__, job_index)
                        for job_index in range(len(jobs))
                    ],
                )
        finally:
            _parallel_backend_preprocess_jobs = None

        preprocess_results = {
            method_name: [None] * size for method_name, size in method_to_sizes.items()
        }
        for method_name, method_index, preprocess_result in completed:
            preprocess_results[method_name][method_index] = preprocess_result
        return preprocess_results

    return classmethod(_parallel_preprocess_multimethod)


def _iter_parallel_multimethod_backends():
    from executorch.backends.arm.ethosu.backend import EthosUBackend
    from executorch.backends.arm.ethosu.compile_spec import EthosUCompileSpec

    yield EthosUBackend, EthosUCompileSpec.from_list

    try:
        from executorch.backends.arm.vgf.backend import VgfBackend
        from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec
    except ImportError:
        return

    yield VgfBackend, VgfCompileSpec.from_list


def _install_parallel_backend_preprocess_multimethods() -> bool:
    workers = _parallel_backend_preprocess_workers()
    if workers <= 1:
        return False

    patched = []
    for backend_cls, parse_compile_spec in _iter_parallel_multimethod_backends():
        if backend_cls in _parallel_backend_multimethod_originals:
            continue
        had_local_attr = _PARALLEL_BACKEND_MULTIMETHOD_ATTR in backend_cls.__dict__
        original = backend_cls.__dict__.get(_PARALLEL_BACKEND_MULTIMETHOD_ATTR)
        _parallel_backend_multimethod_originals[backend_cls] = (
            had_local_attr,
            original,
        )
        setattr(
            backend_cls,
            _PARALLEL_BACKEND_MULTIMETHOD_ATTR,
            _make_parallel_backend_preprocess_multimethod(parse_compile_spec),
        )
        patched.append(backend_cls.__name__)

    if patched:
        logger.info(
            "[fast_preprocess] Installed parallel preprocess_multimethod override for %s with %d workers",
            ", ".join(patched),
            workers,
        )
    return bool(patched)


def _restore_parallel_backend_preprocess_multimethods() -> bool:
    if not _parallel_backend_multimethod_originals:
        return False

    for backend_cls, (had_local_attr, original) in list(
        _parallel_backend_multimethod_originals.items()
    ):
        if had_local_attr:
            setattr(backend_cls, _PARALLEL_BACKEND_MULTIMETHOD_ATTR, original)
        else:
            delattr(backend_cls, _PARALLEL_BACKEND_MULTIMETHOD_ATTR)
    _parallel_backend_multimethod_originals.clear()
    return True



# -- Internal implementations ------------------------------------------------


def _all_rust_preprocess(
    edge_program: ExportedProgram,
    _compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Dedicated path: FX-direct Rust passes + Rust TOSA serialization.

    IMPORTANT: After run_fx_passes() transforms the FX graph, we must use
    ingest_and_serialize() (ingest + serialize only) rather than lower_to_tosa()
    (which would re-run the full IR pass pipeline on top of the already-
    transformed graph, causing double rescales, double transposes, etc.).
    """
    import copy
    t0 = time.perf_counter()
    gm = edge_program.graph_module

    # Deep-copy the state_dict BEFORE passes run, because some FX-direct
    # passes (e.g. RewriteConv) may mutate ep.state_dict in-place.
    # We want the original (un-mutated) weight data for pre-existing tensors,
    # but we still need to ingest any new constant placeholders created by the
    # FX passes later in the pipeline.
    original_state_dict = copy.deepcopy(dict(edge_program.state_dict))

    from retab import _native as _rust

    _rust.run_fx_passes(gm, edge_program)
    _run_control_flow_submodule_rust_passes(gm, edge_program, _rust)
    t_passes = time.perf_counter() - t0
    _sync_input_specs_to_placeholder_order(edge_program)

    merged_state_dict = _merge_preprocess_state_dict(original_state_dict, edge_program)
    tosa_bytes = _rust.ingest_and_serialize(
        gm, merged_state_dict, edge_program.graph_signature, edge_program
    )

    _maybe_dump_tosa_artifacts(edge_program, _compile_spec, tosa_bytes)

    t_total = time.perf_counter() - t0
    logger.info(
        f"[all_rust] passes={t_passes:.3f}s, serialize={t_total-t_passes:.3f}s, total={t_total:.3f}s"
    )

    return PreprocessResult(
        processed_bytes=tosa_bytes,
        _delegate_info_meta=_collect_delegate_info_meta(gm),
    )


# -- Public API: backend override and compatibility fallback -----------------


def fast_preprocess(
    edge_program: ExportedProgram, compile_specs: List[CompileSpec]
) -> PreprocessResult:
    """Drop-in for TOSABackend.preprocess using the dedicated Rust path."""
    compile_spec = TosaCompileSpec._from_list(compile_specs)
    return _all_rust_preprocess(edge_program, compile_spec)


def _install_backend_preprocess_override() -> bool:
    from executorch.backends.arm.tosa.backend import TOSABackend

    install_override = getattr(TOSABackend, "set_preprocess_override", None)
    if callable(install_override):
        install_override(_all_rust_preprocess)
        return True
    return False


def _restore_backend_preprocess_override() -> bool:
    from executorch.backends.arm.tosa.backend import TOSABackend

    clear_override = getattr(TOSABackend, "clear_preprocess_override", None)
    if callable(clear_override):
        clear_override()
        return True
    return False


def patch_tosa_backend():
    """Install the dedicated Rust path into ExecuTorch Arm lowering.

    Prefer the native ExecuTorch override seam when available. Fall back to the
    legacy monkey-patch for older trees that do not expose it yet.
    """
    from executorch.backends.arm.tosa.backend import TOSABackend

    _install_edge_decomp_fast_path()
    _install_run_decomp_fast_path()

    if _install_backend_preprocess_override():
        _install_parallel_backend_preprocess_multimethods()
        logger.info(
            "[fast_preprocess] Installed TOSABackend preprocess override (all_rust mode)"
        )
        return

    TOSABackend._preprocess = staticmethod(_all_rust_preprocess)
    TOSABackend.preprocess = staticmethod(fast_preprocess)
    _install_parallel_backend_preprocess_multimethods()
    logger.info("[fast_preprocess] Patched TOSABackend (legacy all_rust mode)")


def unpatch_tosa_backend():
    """Restore the original ExecuTorch Arm lowering path."""

    _uninstall_run_decomp_fast_path()
    _uninstall_edge_decomp_fast_path()
    _restore_parallel_backend_preprocess_multimethods()

    if _restore_backend_preprocess_override():
        logger.info("[fast_preprocess] Cleared TOSABackend preprocess override")
        return

    import importlib

    import executorch.backends.arm.tosa.backend as mod

    importlib.reload(mod)
    logger.info("[fast_preprocess] Restored original TOSABackend")


class FastTOSABackend:
    """Subclass-style alternative to install-time backend overrides."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return fast_preprocess(edge_program, compile_specs)

    @staticmethod
    def _preprocess(
        edge_program: ExportedProgram,
        compile_spec: TosaCompileSpec,
    ) -> PreprocessResult:
        return _all_rust_preprocess(edge_program, compile_spec)
