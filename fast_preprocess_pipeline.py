"""Drop-in replacement for TOSABackend._preprocess() using the Rust engine.

Hooks into the real ARM backend pipeline at preprocess() time, where the FX
graph already has ARM annotations from to_edge_transform_and_lower. The
dedicated path is:

  1. FX-direct Rust passes on the live Python FX graph
  2. Rust ingestion of the transformed graph
  3. Rust TOSA flatbuffer serialization

The call chain is:
  EthosUBackend.preprocess(ep, compile_specs)
    -> TOSABackend._preprocess(ep, tosa_compile_spec)  <-- we replace this
    -> EthosUBackend._compile_tosa_flatbuffer(tosa_bytes, ...)  (Vela)

By replacing _preprocess, we only swap the TOSA generation step. Vela
compilation still happens normally downstream.
"""

import logging
import os
import time
from typing import List

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

from fast_preprocess_cf_ctx import (
    _collect_delegate_info_meta,
    _refresh_tosa_dim_order_fake_tensors,
    _refresh_tosa_transpose_fake_tensors,
)
from fast_preprocess_cf_rewrites import (
    _debug_binary_rescales,
    _dump_control_flow_submodules,
    _normalize_control_flow_output_layouts,
    _prune_duplicate_terminal_rescales,
    _remove_control_flow_noop_passthroughs,
    _restore_control_flow_float_compare_inputs,
    _restore_control_flow_output_dequantizes,
    _restore_missing_control_flow_input_rescales,
    _restore_missing_control_flow_rescale_qparams,
    _rewrite_control_flow_quantization_boundaries,
)
from fast_preprocess_decomp_hooks import (
    _install_edge_decomp_fast_path,
    _install_run_decomp_fast_path,
    _uninstall_edge_decomp_fast_path,
    _uninstall_run_decomp_fast_path,
)
from fast_preprocess_control_flow import (
    _annotate_control_flow_parent_rank4_dim_order,
    _annotate_control_flow_rank4_dim_order,
    _canonicalize_control_flow_pool_binary_rescales,
    _force_control_flow_rank4_output_dim_order,
    _prepare_control_flow_qparams,
    _prune_duplicate_control_flow_rescales,
    _quantize_constant_only_control_flow_outputs,
    _reapply_control_flow_boundary_rescales,
    _reapply_control_flow_table_ops,
    _refresh_control_flow_binary_input_rescales,
    _regularize_control_flow_submodules,
    _repair_control_flow_binary_rescales,
    _restamp_control_flow_placeholder_dtypes,
    _restore_control_flow_pool_quant_boundaries,
    _restore_control_flow_unary_quant_boundaries,
    _snapshot_control_flow_table_qparams,
)

logger = logging.getLogger(__name__)


def _run_pre_rust_control_flow_phase(gm, edge_program) -> None:
    from fx_helpers import collapse_decomposed_round_to_table

    _regularize_control_flow_submodules(gm)
    _annotate_control_flow_parent_rank4_dim_order(gm)
    _prepare_control_flow_qparams(gm, edge_program)
    _snapshot_control_flow_table_qparams(gm)
    _reapply_control_flow_table_ops(gm, edge_program)
    collapse_decomposed_round_to_table(gm)


def _run_control_flow_submodule_rust_passes(gm, edge_program, arm_tosa_engine) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    if os.environ.get("ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES") == "1":
        for name, submodule, _ in get_cond_while_submodules_nested(gm):
            arm_tosa_engine.run_fx_passes(submodule, edge_program)
            logger.info(
                "[fast_preprocess] ran recursive fx pass pipeline on control-flow submodule %s",
                name,
            )

    for _, submodule, cf_node in get_cond_while_submodules_nested(gm):
        cf_target = str(getattr(cf_node, "target", ""))
        if "cond" in cf_target:
            arm_tosa_engine.run_fx_cond_passes(submodule, edge_program)
        else:
            arm_tosa_engine.run_fx_submodule_passes(submodule)


def _run_post_rust_control_flow_boundary_phase(gm, edge_program, compile_spec) -> None:
    _reapply_control_flow_table_ops(gm, edge_program)
    _rewrite_control_flow_quantization_boundaries(gm)
    _canonicalize_control_flow_pool_binary_rescales(gm)
    _reapply_control_flow_boundary_rescales(gm, include_inputs=True, include_outputs=False)
    _prune_duplicate_control_flow_rescales(gm)
    _repair_control_flow_binary_rescales(gm, compile_spec.tosa_spec)
    _restore_control_flow_output_dequantizes(gm)

    _regularize_control_flow_submodules(gm)
    _restore_missing_control_flow_input_rescales(gm)
    _restore_control_flow_unary_quant_boundaries(gm)
    _restore_control_flow_pool_quant_boundaries(gm)
    _refresh_control_flow_binary_input_rescales(gm)


def _run_output_normalize_phase(gm) -> None:
    _normalize_control_flow_output_layouts(gm)
    _restamp_control_flow_placeholder_dtypes(gm)
    _restore_missing_control_flow_rescale_qparams(gm)
    _reapply_control_flow_boundary_rescales(
        gm, include_inputs=False, include_outputs=True
    )
    _rewrite_control_flow_quantization_boundaries(gm)
    _quantize_constant_only_control_flow_outputs(gm)
    _regularize_control_flow_submodules(gm, include_inputs=False, include_outputs=True)
    _remove_control_flow_noop_passthroughs(gm)
    _annotate_control_flow_parent_rank4_dim_order(gm)
    _annotate_control_flow_rank4_dim_order(gm)
    _force_control_flow_rank4_output_dim_order(gm)
    _refresh_tosa_transpose_fake_tensors(gm)
    _refresh_tosa_dim_order_fake_tensors(gm)
    _prune_duplicate_terminal_rescales(gm)
    _restamp_control_flow_placeholder_dtypes(gm)
    _restore_control_flow_float_compare_inputs(gm)


def _merge_preprocess_state_dict(original_state_dict, edge_program):
    import copy

    merged_state_dict = original_state_dict
    for key, value in dict(edge_program.state_dict).items():
        if key not in merged_state_dict:
            merged_state_dict[key] = copy.deepcopy(value)
    return merged_state_dict


def _maybe_dump_tosa_artifacts(edge_program, compile_spec, tosa_bytes) -> None:
    import json
    from pathlib import Path

    from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
    from executorch.backends.arm.common.debug import debug_tosa_dump
    from executorch.backends.arm.tosa.backend import arm_get_first_delegation_tag

    artifact_path = compile_spec.get_intermediate_path()
    if artifact_path:
        tag = arm_get_first_delegation_tag(edge_program.graph_module)
        suffix = "{}".format(f"_{tag}" if tag else "") + f"_{compile_spec.tosa_spec}"
        debug_tosa_dump(bytes(tosa_bytes), artifact_path, suffix=suffix)

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

    dump_path = os.environ.get("ARM_TOSA_ENGINE_DUMP_TOSA")
    if dump_path:
        dump_file = Path(dump_path)
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_bytes(bytes(tosa_bytes))
        logger.info("[all_rust] dumped TOSA flatbuffer to %s", dump_file)





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

    import arm_tosa_engine

    _run_pre_rust_control_flow_phase(gm, edge_program)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
    _debug_binary_rescales(gm, "after_run_fx_passes")
    _run_control_flow_submodule_rust_passes(gm, edge_program, arm_tosa_engine)
    _run_post_rust_control_flow_boundary_phase(gm, edge_program, _compile_spec)
    t_passes = time.perf_counter() - t0
    _dump_control_flow_submodules(gm, "after_rust_passes")
    _run_output_normalize_phase(gm)
    _debug_binary_rescales(gm, "before_ingest_and_serialize")
    _dump_control_flow_submodules(gm, "after_control_flow_normalize")

    merged_state_dict = _merge_preprocess_state_dict(original_state_dict, edge_program)
    tosa_bytes = arm_tosa_engine.ingest_and_serialize(
        gm, merged_state_dict, edge_program.graph_signature
    )

    _maybe_dump_tosa_artifacts(edge_program, _compile_spec, tosa_bytes)

    t_total = time.perf_counter() - t0
    logger.info(
        f"[all_rust] passes={t_passes:.3f}s, serialize={t_total-t_passes:.3f}s, total={t_total:.3f}s"
    )

    return PreprocessResult(
        processed_bytes=bytes(tosa_bytes),
        _delegate_info_meta=_collect_delegate_info_meta(gm),
    )


# -- Public API: patching and subclass alternatives --------------------------


def fast_preprocess(
    edge_program: ExportedProgram, compile_specs: List[CompileSpec]
) -> PreprocessResult:
    """Drop-in for TOSABackend.preprocess using the dedicated Rust path."""
    compile_spec = TosaCompileSpec.from_list(compile_specs)
    return _all_rust_preprocess(edge_program, compile_spec)


def patch_tosa_backend():
    """Monkey-patch TOSABackend._preprocess with the dedicated Rust path.

    This is the effective hook: EthosUBackend.preprocess calls
    TOSABackend._preprocess to generate the TOSA flatbuffer, then feeds
    it to Vela. By patching _preprocess, we replace TOSA generation
    while Vela compilation continues normally.
    """
    from executorch.backends.arm.tosa.backend import TOSABackend

    _install_edge_decomp_fast_path()
    _install_run_decomp_fast_path()

    TOSABackend._preprocess = staticmethod(_all_rust_preprocess)
    TOSABackend.preprocess = staticmethod(fast_preprocess)
    logger.info("[fast_preprocess] Patched TOSABackend (all_rust mode)")


def unpatch_tosa_backend():
    """Restore the original TOSABackend._preprocess."""
    import importlib

    import executorch.backends.arm.tosa.backend as mod

    _uninstall_run_decomp_fast_path()
    _uninstall_edge_decomp_fast_path()
    importlib.reload(mod)
    logger.info("[fast_preprocess] Restored original TOSABackend")


class FastTOSABackend:
    """Subclass-style alternative to monkey-patching."""

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
