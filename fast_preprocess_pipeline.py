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
    _normalize_control_flow_output_contracts,
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
    _prune_duplicate_control_flow_rescales,
    _quantize_constant_only_control_flow_outputs,
    _reapply_control_flow_boundary_rescales,
    _reapply_control_flow_table_ops,
    _refresh_control_flow_binary_input_rescales,
    _regularize_control_flow_submodules,
    _repair_control_flow_binary_rescales,
    _restore_control_flow_quant_boundaries,
    _snapshot_control_flow_table_qparams,
)

logger = logging.getLogger(__name__)





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
    import json
    import os
    from pathlib import Path

    from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
    from executorch.backends.arm.common.debug import debug_tosa_dump
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm.tosa.backend import arm_get_first_delegation_tag
    t0 = time.perf_counter()
    gm = edge_program.graph_module

    # Deep-copy the state_dict BEFORE passes run, because some FX-direct
    # passes (e.g. RewriteConv) may mutate ep.state_dict in-place.
    # We want the original (un-mutated) weight data for pre-existing tensors,
    # but we still need to ingest any new constant placeholders created by the
    # FX passes later in the pipeline.
    original_state_dict = copy.deepcopy(dict(edge_program.state_dict))

    import arm_tosa_engine

    # Match the control-flow submodule fake tensors to the parent graph before
    # the Rust passes inspect placeholders and outputs.
    _regularize_control_flow_submodules(gm)
    _normalize_control_flow_output_contracts(gm, annotate_parent_users=True)
    _snapshot_control_flow_table_qparams(gm)
    # Parse, don't validate: control-flow table ops still have their original
    # q/dq scaffolding at this point. Reapply TABLE lowering before the
    # recursive FX-direct pass stack strips that local qparam context away,
    # then keep the later repair pass as a safety net for any surviving raw
    # table-capable ops.
    # TESTING: Rust submodule pass handles this
    _reapply_control_flow_table_ops(gm, edge_program)

    # Collapse decomposed banker's-rounding into a TABLE before Rust passes.
    # Python DecomposeRoundPass runs during quantization and the 12-op
    # decomposition loses precision at the 0.5 tie-break; a TABLE is exact.
    from fx_helpers import collapse_decomposed_round_to_table
    collapse_decomposed_round_to_table(gm)

    # Run the full FX-direct pipeline (all passes in Rust via PyO3)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
    _debug_binary_rescales(gm, "after_run_fx_passes")
    # The top-level graph still needs the full FX-direct pass stack, but
    # recursively replaying that whole stack on control-flow submodules has
    # proven too aggressive: it decomposes branch-local q/dq boundaries into
    # large quant-arithmetic islands that diverge from the generic Arm path and
    # destabilize TABLE/control-flow numerics. The explicit control-flow repair
    # passes around this point already normalize branch signatures, qparams,
    # table lowering, and boundary rescales, so keep branch graphs at that
    # higher-level form by default. Set ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES=1
    # to restore the old behavior for debugging.
    if os.environ.get("ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES") == "1":
        for name, submodule, _ in get_cond_while_submodules_nested(gm):
            arm_tosa_engine.run_fx_passes(submodule, edge_program)
            logger.info(
                "[fast_preprocess] ran recursive fx pass pipeline on control-flow submodule %s",
                name,
            )
    # Run safe pass subset on control-flow submodules.
    # Cond branches get fold_qdq (safe); while loops don't (breaks carried state).
    for name, submodule, cf_node in get_cond_while_submodules_nested(gm):
        cf_target = str(getattr(cf_node, "target", ""))
        if "cond" in cf_target:
            arm_tosa_engine.run_fx_cond_passes(submodule, edge_program)
        else:
            arm_tosa_engine.run_fx_submodule_passes(submodule)

    _reapply_control_flow_table_ops(gm, edge_program)
    _rewrite_control_flow_quantization_boundaries(gm)
    _restore_control_flow_quant_boundaries(
        gm, mode="pool", canonicalize_pool_binary_sources=True
    )
    _reapply_control_flow_boundary_rescales(gm, include_inputs=True, include_outputs=False)
    _prune_duplicate_control_flow_rescales(gm)
    _repair_control_flow_binary_rescales(gm, _compile_spec.tosa_spec)
    _restore_control_flow_output_dequantizes(gm)
    t_passes = time.perf_counter() - t0
    _dump_control_flow_submodules(gm, "after_rust_passes")

    # Re-apply the same normalization after passes so serializer ingest sees
    # branch signatures that still match the parent control-flow nodes.
    _regularize_control_flow_submodules(gm)
    _restore_missing_control_flow_input_rescales(gm)
    _restore_control_flow_quant_boundaries(gm, mode="unary")
    _restore_control_flow_quant_boundaries(gm, mode="pool")
    _refresh_control_flow_binary_input_rescales(gm)
    # Covered by Rust submodule passes — MUST KEEP for while-loop correctness
    _normalize_control_flow_output_contracts(
        gm,
        normalize_outputs=True,
        restamp_placeholder_dtypes=True,
    )
    _restore_missing_control_flow_rescale_qparams(gm)
    _reapply_control_flow_boundary_rescales(
        gm, include_inputs=False, include_outputs=True
    )
    _rewrite_control_flow_quantization_boundaries(gm)
    _quantize_constant_only_control_flow_outputs(gm)
    _regularize_control_flow_submodules(gm, include_inputs=False, include_outputs=True)
    # Covered by Rust submodule passes — MUST KEEP for while-loop correctness
    _remove_control_flow_noop_passthroughs(gm)
    _normalize_control_flow_output_contracts(
        gm,
        annotate_parent_users=True,
        normalize_outputs=True,
        annotate_rank4_nodes=True,
        restamp_placeholder_dtypes=True,
    )
    _refresh_tosa_transpose_fake_tensors(gm)
    _refresh_tosa_dim_order_fake_tensors(gm)
    _prune_duplicate_terminal_rescales(gm)
    _restore_control_flow_float_compare_inputs(gm)
    _debug_binary_rescales(gm, "before_ingest_and_serialize")
    _dump_control_flow_submodules(gm, "after_control_flow_normalize")

    # Merge in any pass-created constant placeholders added to the live
    # ExportedProgram after the pre-pass snapshot. Keep original entries for
    # existing keys so we do not pick up in-place mutations to legacy weights.
    merged_state_dict = original_state_dict
    for key, value in dict(edge_program.state_dict).items():
        if key not in merged_state_dict:
            merged_state_dict[key] = copy.deepcopy(value)

    # Ingest + serialize only -- do NOT re-run IR passes (they already ran
    # as FX-direct passes above). lower_to_tosa() would run
    # build_phase2_pipeline() again, corrupting the graph with duplicate
    # transformations.
    tosa_bytes = arm_tosa_engine.ingest_and_serialize(
        gm, merged_state_dict, edge_program.graph_signature
    )

    artifact_path = _compile_spec.get_intermediate_path()
    if artifact_path:
        tag = arm_get_first_delegation_tag(edge_program.graph_module)
        suffix = "{}".format(f"_{tag}" if tag else "") + f"_{_compile_spec.tosa_spec}"
        debug_tosa_dump(bytes(tosa_bytes), artifact_path, suffix=suffix)

        if _compile_spec.tosa_debug_mode == ArmCompileSpec.DebugMode.JSON:
            debug_json_path = Path(artifact_path) / "debug.json"
            debug_json_path.parent.mkdir(parents=True, exist_ok=True)
            debug_json_path.write_text(
                json.dumps(
                    {
                        "tosa_spec": str(_compile_spec.tosa_spec),
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
        logger.info(f"[all_rust] dumped TOSA flatbuffer to {dump_file}")

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
