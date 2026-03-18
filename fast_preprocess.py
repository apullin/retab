"""Drop-in replacement for TOSABackend._preprocess() using the Rust engine.

Hooks into the real ARM backend pipeline at preprocess() time, where the FX
graph already has ARM annotations from to_edge_transform_and_lower. Replaces
the slow Python pass infrastructure + Python TOSA serializer with:

  1. FX-direct Rust passes (fold Q/DQ, fuse activations, rewrite conv, etc.)
  2. Targeted Python passes only where needed (rescale insertion)
  3. Rust TOSA flatbuffer serialization

The call chain is:
  EthosUBackend.preprocess(ep, compile_specs)
    -> TOSABackend._preprocess(ep, tosa_compile_spec)  <-- we replace this
    -> EthosUBackend._compile_tosa_flatbuffer(tosa_bytes, ...)  (Vela)

By replacing _preprocess, we only swap the TOSA generation step. Vela
compilation still happens normally downstream.
"""

import logging
import time
from typing import List

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

logger = logging.getLogger(__name__)


# -- Internal implementations ------------------------------------------------


def _fast_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Lower an exported program to TOSA using the Rust engine.

    Replaces TOSABackend._preprocess. Same signature: (ExportedProgram,
    TosaCompileSpec) -> PreprocessResult.

    The edge_program graph has ARM annotations from partitioning. We run our
    hybrid pipeline: FX-direct Rust passes on the live Python graph, then
    ingest the result into Rust IR for TOSA serialization.
    """
    import arm_tosa_engine

    t0 = time.perf_counter()

    graph_module = edge_program.graph_module
    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature

    # lower_to_tosa_v2 runs the full hybrid pipeline:
    #   1. FX-direct passes on live Python graph (Rust fold Q/DQ, Python
    #      rescales, table ops, fuse duplicates, clamp, conv, memory format)
    #   2. Ingest post-pass graph into Rust IR
    #   3. Serialize to TOSA flatbuffer
    tosa_bytes = arm_tosa_engine.lower_to_tosa_v2(
        graph_module,
        state_dict,
        graph_signature,
        edge_program,
    )

    t_total = time.perf_counter() - t0
    logger.info(f"[fast_preprocess] TOSA generation: {t_total:.3f}s")

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


def _full_python_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Run full Python ArmPassManager passes + Rust serialization.

    Correctness baseline: Python passes are known-correct, and we only
    replace the TOSA serialization with Rust. If this produces correct
    output, any remaining issues are in our Rust passes.
    """
    import arm_tosa_engine

    t0 = time.perf_counter()

    graph_module = edge_program.graph_module

    from executorch.backends.arm._passes import ArmPassManager

    pass_manager = ArmPassManager(compile_spec)
    graph_module = pass_manager.transform_to_backend_pipeline(
        exported_program=edge_program, graph_module=graph_module
    )

    t_passes = time.perf_counter() - t0

    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature
    tosa_bytes = arm_tosa_engine.lower_to_tosa(
        graph_module, state_dict, graph_signature
    )

    t_total = time.perf_counter() - t0
    logger.info(
        f"[python_passes_preprocess] TOSA generation: {t_total:.3f}s "
        f"(passes={t_passes:.3f}s, serialize={t_total - t_passes:.3f}s)"
    )

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


def _fast_python_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Run Python ArmPassManager passes with smart no-op skipping + Rust serialization.

    Same as python_passes mode but installs the smart pass skipping optimization
    which scans graph ops before each pass and skips passes whose target ops are
    absent. Saves ~35-40s by avoiding ExportPass retrace on ~70 no-op passes.
    """
    import arm_tosa_engine
    from fast_pass_manager import install_smart_skipping, uninstall_smart_skipping

    t0 = time.perf_counter()

    install_smart_skipping()
    try:
        graph_module = edge_program.graph_module

        from executorch.backends.arm._passes import ArmPassManager

        pass_manager = ArmPassManager(compile_spec)
        graph_module = pass_manager.transform_to_backend_pipeline(
            exported_program=edge_program, graph_module=graph_module
        )
    finally:
        uninstall_smart_skipping()

    t_passes = time.perf_counter() - t0

    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature
    tosa_bytes = arm_tosa_engine.lower_to_tosa(
        graph_module, state_dict, graph_signature
    )

    t_total = time.perf_counter() - t0
    logger.info(
        f"[fast_python_preprocess] TOSA generation: {t_total:.3f}s "
        f"(passes={t_passes:.3f}s, serialize={t_total - t_passes:.3f}s)"
    )

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


# -- Public API: patching and subclass alternatives --------------------------


def fast_preprocess(
    edge_program: ExportedProgram, compile_specs: List[CompileSpec]
) -> PreprocessResult:
    """Drop-in for TOSABackend.preprocess (public method).

    Parses compile_specs then delegates to _fast_preprocess.
    """
    compile_spec = TosaCompileSpec.from_list(compile_specs)
    return _fast_preprocess(edge_program, compile_spec)


def patch_tosa_backend(mode="fast"):
    """Monkey-patch TOSABackend._preprocess with our replacement.

    This is the effective hook: EthosUBackend.preprocess calls
    TOSABackend._preprocess to generate the TOSA flatbuffer, then feeds
    it to Vela. By patching _preprocess, we replace TOSA generation
    while Vela compilation continues normally.

    Args:
        mode: "fast" uses Rust passes + Rust serialization (default).
              "python_passes" uses Python passes + Rust serialization
              (correctness baseline).
              "python_passes_fast" uses Python passes with smart no-op
              skipping + Rust serialization (~40s savings).
    """
    from executorch.backends.arm.tosa.backend import TOSABackend

    if mode == "fast":
        TOSABackend._preprocess = staticmethod(_fast_preprocess)
        TOSABackend.preprocess = staticmethod(fast_preprocess)
        logger.info("[fast_preprocess] Patched TOSABackend (fast mode)")
    elif mode == "python_passes":
        TOSABackend._preprocess = staticmethod(_full_python_preprocess)
        logger.info("[fast_preprocess] Patched TOSABackend (python_passes mode)")
    elif mode == "python_passes_fast":
        TOSABackend._preprocess = staticmethod(_fast_python_preprocess)
        logger.info(
            "[fast_preprocess] Patched TOSABackend (python_passes_fast mode)"
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Use 'fast', 'python_passes', or 'python_passes_fast'."
        )


def unpatch_tosa_backend():
    """Restore the original TOSABackend._preprocess."""
    import importlib

    import executorch.backends.arm.tosa.backend as mod

    importlib.reload(mod)
    logger.info("[fast_preprocess] Restored original TOSABackend")


class FastTOSABackend:
    """Subclass-style alternative to monkey-patching.

    Can be used if a backend swapping mechanism is available.
    """

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
        return _fast_preprocess(edge_program, compile_spec)
