"""Drop-in replacement for ExecuTorch Arm TOSA preprocess using the Rust engine.

Hooks into the real ARM backend pipeline at preprocess() time, where the FX
graph already has ARM annotations from to_edge_transform_and_lower. The
dedicated path is:

  1. FX-direct Rust passes on the live Python FX graph
  2. Rust ingestion of the transformed graph
  3. Rust TOSA flatbuffer serialization

The call chain is:
  EthosUBackend.preprocess(ep, compile_specs)
    -> TOSABackend._preprocess(ep, tosa_compile_spec)  <-- RETAB replaces this
    -> EthosUBackend._compile_tosa_flatbuffer(tosa_bytes, ...)  (Vela)

By overriding the TOSA preprocess stage, we only swap the TOSA generation
step. Vela compilation still happens normally downstream.
"""

import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import List

from executorch.backends.arm.common.pipeline_config import SDPASafeSoftmaxGuardPolicy
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

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
    gm, edge_program, retab, pipeline_config, region_prefix: str | None = None
) -> None:
    from executorch.exir.graph_module import get_cond_while_submodules

    if (
        _env_get(
            "RETAB_FORCE_CF_RECURSIVE_PASSES",
            "ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES",
        )
        == "1"
    ):
        for name, submodule, _ in get_cond_while_submodules(gm):
            retab.run_fx_passes(
                submodule, edge_program, pipeline_config=pipeline_config
            )
            logger.info(
                "[retab] ran recursive fx pass pipeline on control-flow submodule %s",
                name,
            )

    for name, submodule, cf_node in get_cond_while_submodules(gm):
        region_name = f"{region_prefix}__{name}" if region_prefix else name
        cf_target = str(getattr(cf_node, "target", ""))
        if "cond" in cf_target:
            retab.run_fx_cond_passes(
                submodule, edge_program, region_name, pipeline_config=pipeline_config
            )
        else:
            retab.run_fx_submodule_passes(
                submodule, edge_program, region_name, pipeline_config=pipeline_config
            )
        _run_control_flow_submodule_rust_passes(
            submodule, edge_program, retab, pipeline_config, region_name
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
        logger.info("[retab] dumped TOSA flatbuffer to %s", dump_file)


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
            "[retab] ignoring invalid RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS=%r",
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


def _init_parallel_backend_preprocess_worker(jobs) -> None:
    # Runs in each forked worker: the job list arrives with this pool's fork, so a
    # concurrent call in the parent can't swap in its own jobs before the fork.
    global _parallel_backend_preprocess_jobs
    _parallel_backend_preprocess_jobs = jobs


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
                "[retab] %s preprocess_multimethod falling back to serial due to debug artifact settings",
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
                "[retab] %s preprocess_multimethod falling back to serial because fork multiprocessing is unavailable",
                backend_cls.__name__,
            )
            return _serial_backend_preprocess_multimethod(
                backend_cls, edge_programs, compile_specs
            )

        logger.info(
            "[retab] %s preprocess_multimethod running %d jobs on %d workers",
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

        with mp_ctx.Pool(
            processes=max_workers,
            initializer=_init_parallel_backend_preprocess_worker,
            initargs=(jobs,),
        ) as pool:
            completed = pool.map(
                _run_parallel_backend_preprocess_job,
                [
                    (backend_cls.__module__, backend_cls.__name__, job_index)
                    for job_index in range(len(jobs))
                ],
            )

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

    yield EthosUBackend, EthosUCompileSpec._from_list

    try:
        from executorch.backends.arm.vgf.backend import VgfBackend
        from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec
    except ImportError:
        return

    yield VgfBackend, VgfCompileSpec._from_list


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
            "[retab] Installed parallel preprocess_multimethod override for %s with %d workers",
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



# -- Stock originals, scope check and counters --------------------------------

# Class-dict values of TOSABackend attributes as stock ExecuTorch defined them. They are
# captured once, before any RETAB patch, and serve both the stock fallback and unpatching.
_stock_tosa_backend_attrs: dict[str, object] = {}
_stock_ethos_u_preprocess: dict[str, object] = {}
# Set while EthosUBackend.preprocess runs, so RETAB can apply its unused-input check.
_ethos_u_context = threading.local()
_UNUSED_INPUT_MESSAGE = (
    "A delegate input is not used after lowering, for example an "
    "input that only gives its shape to zeros_like or full_like, or "
    "that copy_ overwrites. Vela removes unused inputs, but the "
    "Ethos-U runtime binds delegate arguments by position, so this "
    "delegate would compute on the wrong tensors. Unused: {}"
)
_preprocess_counts: Counter = Counter()
_stock_fallback_counts: Counter = Counter()


def _is_retab_callable(value) -> bool:
    fn = value.__func__ if isinstance(value, staticmethod) else value
    return bool(getattr(fn, "_retab_override", False))


def _capture_stock_tosa_backend() -> None:
    from executorch.backends.arm.tosa.backend import TOSABackend

    for name in ("preprocess", "_preprocess"):
        value = TOSABackend.__dict__.get(name)
        if name not in _stock_tosa_backend_attrs and value is not None and not _is_retab_callable(value):
            _stock_tosa_backend_attrs[name] = value


def stock_preprocess(
    edge_program: ExportedProgram, compile_spec: TosaCompileSpec
) -> PreprocessResult:
    """Run unmodified ExecuTorch TOSA preprocess, whether or not RETAB is patched in."""
    _capture_stock_tosa_backend()
    original = _stock_tosa_backend_attrs.get("_preprocess")
    if original is None:
        raise RuntimeError("stock TOSABackend._preprocess was replaced before RETAB could capture it")
    fn = original.__func__ if isinstance(original, staticmethod) else original
    return fn(edge_program, compile_spec)


def _has_registered_pass_insertions() -> bool:
    try:
        from executorch.backends.arm._passes.arm_pass_manager import (
            _registered_pass_insertions,
        )
    except ImportError:
        return False
    return any(
        insertions.before_passes or insertions.after_passes
        for insertions in _registered_pass_insertions.values()
    )


def _build_pipeline_config(compile_spec: TosaCompileSpec) -> dict:
    """The compile spec's pipeline settings."""
    return {
        "tosa_spec": str(compile_spec.tosa_spec),
        "tosa_dev_mode": bool(compile_spec.tosa_dev_mode),
        **compile_spec._get_pass_pipeline_config().to_dict(),
    }


def _u55_transpose_probe(
    compile_spec: TosaCompileSpec,
) -> Callable[[list[int], list[int], str], bool] | None:
    """rc4's exact U55 TRANSPOSE check, for RETAB's IrDecomposePermuteForU55Pass.

    rc4's DecomposePermuteForU55Pass asks it about every permute alive at its slot whose
    input breaks U55's worst-case rule. RETAB's passes create most of those permutes, so
    the Rust pass calls this under the GIL with the permute's input shape, its dims and
    the torch name of the TRANSPOSE's dtype, once per question. The check builds its
    probe TOSA for the lowering context's TOSA version, so this enters a context, as
    stock's pass manager does.
    """
    if not compile_spec.tosa_spec.is_U55_subset:
        return None
    import torch
    from executorch.backends.arm._passes.decompose_permute_for_u55_pass import (
        DecomposePermuteForU55Pass,
    )
    from executorch.backends.arm.tosa.specification import TosaLoweringContext

    def violates(shape: list[int], permutation: list[int], dtype: str) -> bool:
        torch_dtype = getattr(torch, dtype.removeprefix("torch."), None)
        if not isinstance(torch_dtype, torch.dtype):
            return True  # rc4's check rejects every dtype but int8, int16 and bool
        with TosaLoweringContext(compile_spec.tosa_spec):
            return DecomposePermuteForU55Pass._violates_exact_constraint(
                tuple(shape), tuple(permutation), torch_dtype
            )

    return violates


def _has_non_default_pipeline_config(compile_spec) -> bool:
    config = getattr(compile_spec, "_pipeline_config", None)
    if config is None:
        return False
    return config != compile_spec._create_default_pipeline_config()


def _is_symbolic_val(val) -> bool:
    if isinstance(val, (list, tuple)):
        return any(_is_symbolic_val(item) for item in val)
    shape = getattr(val, "shape", None)
    return shape is not None and any(not isinstance(dim, int) for dim in shape)


def _has_symbolic_shapes(graph_module) -> bool:
    """Whether any node, including tuple outputs and cond/while regions, has a symbolic shape."""
    from executorch.exir.graph_module import get_cond_while_submodules

    if any(_is_symbolic_val(node.meta.get("val")) for node in graph_module.graph.nodes):
        return True
    return any(
        _has_symbolic_shapes(submodule)
        for _, submodule, _ in get_cond_while_submodules(graph_module)
    )


def stock_fallback_reason(
    edge_program: ExportedProgram, compile_spec: TosaCompileSpec
) -> str | None:
    """Why this graph must go to stock preprocess, or None when RETAB handles it.

    RETAB implements TOSA 1.0 INT lowering with the target's default pass pipeline.
    """
    spec = compile_spec.tosa_spec
    if spec.support_float():
        return f"float profile ({spec})"
    if (spec.version.major, spec.version.minor) != (1, 0):
        return f"TOSA version {spec.version}"
    if compile_spec.tosa_dev_mode:
        return "tosa_dev_mode"
    guard = compile_spec._get_pass_pipeline_config().sdpa_safe_softmax_guard
    if guard is not SDPASafeSoftmaxGuardPolicy.PRESERVE:
        return f"sdpa_safe_softmax_guard={guard.name} is not supported by RETAB"
    if _has_non_default_pipeline_config(compile_spec):
        return "non-default pass pipeline config"
    if _has_registered_pass_insertions():
        return "registered pass insertions"
    if edge_program.graph_signature.buffers_to_mutate:
        return "delegate-owned mutable buffers"
    if _has_symbolic_shapes(edge_program.graph_module):
        return "symbolic shapes"
    return None


def preprocess_stats() -> dict:
    """RETAB preprocess calls and stock fallbacks (by reason) in this process."""
    return {
        "retab": _preprocess_counts["retab"],
        "stock_fallback": dict(_stock_fallback_counts),
    }


# -- Internal implementations ------------------------------------------------


def _native_fallback_reason(exc: Exception) -> str | None:
    """The reason in a `RETAB_STOCK_FALLBACK: <reason>: <details>` error from Rust."""
    _, marker, rest = str(exc).partition("RETAB_STOCK_FALLBACK:")
    if not marker:
        return None
    return rest.strip().split(":", 1)[0]


def _all_rust_preprocess(
    edge_program: ExportedProgram,
    _compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Lower the edge program with RETAB's Rust pipeline and serialize it to TOSA.

    run_fx_passes() reads the FX graph and state_dict without modifying them and
    stashes the lowered IR (plus one IR per cond/while region); ingest_and_serialize()
    serializes that stash. Graphs outside RETAB's scope go to stock _preprocess.
    """
    reason = stock_fallback_reason(edge_program, _compile_spec)
    if reason is not None:
        _stock_fallback_counts[reason] += 1
        logger.info("[retab] stock preprocess fallback: %s", reason)
        return stock_preprocess(edge_program, _compile_spec)

    t0 = time.perf_counter()
    gm = edge_program.graph_module

    from retab import _native as _rust

    pipeline_config = _build_pipeline_config(_compile_spec)
    # The U55 split judges the permutes alive at its slot, as rc4 does.
    u55_transpose_probe = _u55_transpose_probe(_compile_spec)
    try:
        _rust.run_fx_passes(
            gm,
            edge_program,
            pipeline_config=pipeline_config,
            u55_transpose_probe=u55_transpose_probe,
        )
        _run_control_flow_submodule_rust_passes(gm, edge_program, _rust, pipeline_config)
    except Exception as exc:  # noqa: BLE001 - Rust requests stock via message
        reason = _native_fallback_reason(exc)
        if reason is None:
            raise
        _stock_fallback_counts[reason] += 1
        logger.info("[retab] stock preprocess fallback: %s", exc)
        return stock_preprocess(edge_program, _compile_spec)
    t_passes = time.perf_counter() - t0

    try:
        tosa_bytes = _rust.ingest_and_serialize()
    except Exception as exc:  # noqa: BLE001 - Rust requests stock via message
        reason = _native_fallback_reason(exc)
        if reason is None:
            raise
        _stock_fallback_counts[reason] += 1
        logger.info("[retab] stock preprocess fallback: %s", exc)
        return stock_preprocess(edge_program, _compile_spec)
    _preprocess_counts["retab"] += 1
    if getattr(_ethos_u_context, "depth", 0):
        unused = _rust.last_unused_inputs()
        if unused:
            raise RuntimeError(_UNUSED_INPUT_MESSAGE.format(unused))

    _maybe_dump_tosa_artifacts(edge_program, _compile_spec, tosa_bytes)

    t_total = time.perf_counter() - t0
    logger.info(
        f"[retab] passes={t_passes:.3f}s, serialize={t_total-t_passes:.3f}s, total={t_total:.3f}s"
    )

    # Like stock's _preprocess: the TOSA bytes and nothing else.
    return PreprocessResult(processed_bytes=tosa_bytes)


_all_rust_preprocess._retab_override = True


# -- Public API: backend override and compatibility fallback -----------------


def fast_preprocess(
    edge_program: ExportedProgram, compile_specs: List[CompileSpec]
) -> PreprocessResult:
    """Drop-in for TOSABackend.preprocess using the dedicated Rust path."""
    compile_spec = TosaCompileSpec._from_list(compile_specs)
    return _all_rust_preprocess(edge_program, compile_spec)


fast_preprocess._retab_override = True


def _install_ethos_u_unused_input_guard() -> None:
    """Mark EthosUBackend.preprocess calls for RETAB's unused-input check.

    Stock EthosUBackend rejects a delegate whose user input has no users after
    TOSABackend._preprocess, judging by the edge graph that stock's passes rewrite
    in place. RETAB leaves the edge graph untouched, so it checks its own lowered
    graph instead, while this marker is set.
    """
    try:
        from executorch.backends.arm.ethosu.backend import EthosUBackend
    except ImportError:
        return
    current = EthosUBackend.__dict__.get("preprocess")
    if current is None or _is_retab_callable(current):
        return
    _stock_ethos_u_preprocess["preprocess"] = current
    original = current.__func__ if isinstance(current, staticmethod) else current

    def preprocess(edge_program, compile_specs):
        depth = getattr(_ethos_u_context, "depth", 0)
        _ethos_u_context.depth = depth + 1
        try:
            return original(edge_program, compile_specs)
        finally:
            _ethos_u_context.depth = depth

    preprocess.__name__ = original.__name__
    preprocess.__qualname__ = original.__qualname__
    preprocess._retab_override = True
    EthosUBackend.preprocess = staticmethod(preprocess)


def _restore_ethos_u_unused_input_guard() -> None:
    original = _stock_ethos_u_preprocess.pop("preprocess", None)
    if original is None:
        return
    from executorch.backends.arm.ethosu.backend import EthosUBackend

    EthosUBackend.preprocess = original


def patch_tosa_backend(*, fast_paths: bool = False):
    """Install the dedicated Rust path into ExecuTorch Arm lowering.

    Stock ExecuTorch has no preprocess override hook, so this replaces
    TOSABackend._preprocess in place. `fast_paths` also replaces parts of
    ExecuTorch's edge-manager and run_decompositions flow; those paths have no
    test coverage on rc4, so they are opt-in.
    """
    from executorch.backends.arm.tosa.backend import TOSABackend

    _capture_stock_tosa_backend()
    _install_ethos_u_unused_input_guard()
    if fast_paths:
        _install_edge_decomp_fast_path()
        _install_run_decomp_fast_path()

    # Stock TOSABackend.preprocess and EthosUBackend both call TOSABackend._preprocess,
    # so swapping that one attribute (keeping its name) routes every TOSA lowering here.
    TOSABackend._preprocess = staticmethod(_all_rust_preprocess)
    _install_parallel_backend_preprocess_multimethods()
    logger.info("[retab] Patched TOSABackend._preprocess (all_rust mode)")


def unpatch_tosa_backend():
    """Restore the original ExecuTorch Arm lowering path."""

    _uninstall_run_decomp_fast_path()
    _uninstall_edge_decomp_fast_path()
    _restore_parallel_backend_preprocess_multimethods()
    _restore_ethos_u_unused_input_guard()

    # Restore the stock attributes in place. importlib.reload would create a second
    # TOSABackend class while other modules keep the patched one.
    from executorch.backends.arm.tosa.backend import TOSABackend

    for name, original in _stock_tosa_backend_attrs.items():
        setattr(TOSABackend, name, original)
    logger.info("[retab] Restored stock TOSABackend attributes")


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


try:
    _capture_stock_tosa_backend()
except ImportError as exc:  # pragma: no cover - ExecuTorch without the Arm backend
    logger.debug("[retab] could not capture stock TOSABackend: %s", exc)
