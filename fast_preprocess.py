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
import time
from typing import List

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

logger = logging.getLogger(__name__)

_ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = None
_ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS = None
_ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS = None
_ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS = None


def _program_contains_targets(program: ExportedProgram, targets: set) -> bool:
    """Return True if the program or control-flow submodules contain any target."""
    from executorch.exir.program import _program as program_mod

    for node in program.graph.nodes:
        if node.op == "call_function" and node.target in targets:
            return True

    for _, submod, _ in program_mod.get_control_flow_submodules(program.graph_module):
        for node in submod.graph.nodes:
            if node.op == "call_function" and node.target in targets:
                return True

    return False


def _install_edge_decomp_fast_path() -> None:
    """Skip redundant EDGE_DO_NOT_DECOMP cleanup decompositions.

    ExecuTorch's EDGE_DO_NOT_DECOMP path always runs a second
    `run_decompositions(default_table)` pass after the preserve-op rewrite
    phase. That second pass is only needed if the first pass actually left any
    preserve-candidate ops in the graph. When none remain, the second pass is a
    no-op that still costs multiple seconds on large models.
    """
    global _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS

    from collections import defaultdict
    import copy

    from executorch.exir.program import _program as program_mod

    current = program_mod._gen_edge_manager_for_partitioners
    if getattr(current, "__arm_tosa_engine_patched__", False):
        return

    _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = current

    def optimized_gen_edge_manager_for_partitioners(
        partitioner,
        aten_programs,
        config,
        constant_methods,
        generate_etrecord=False,
    ):
        ops_set_to_not_decompose_by_program = defaultdict(list)
        edge_programs = {}
        skipped_second_passes = 0

        for name, program in aten_programs.items():
            # Functionalize program before asking partitioners to preserve ops.
            program = program.run_decompositions({})

            if partitioner is not None:
                partitioners_for_program = partitioner.get(name, [])
                final_ops_to_preserve = set()

                if not partitioners_for_program:
                    table = program_mod._default_decomposition_table()
                    for op in config.preserve_ops:
                        table.pop(op, None)
                    program = program.run_decompositions(table)

                for curr_partitioner in partitioners_for_program:
                    curr_ops_no_decomp, _ = curr_partitioner.ops_to_not_decompose(
                        program
                    )

                    can_skip_using_edge_do_not_decomp = (
                        program_mod._can_skip_using_EDGE_DO_NOT_DECOMP(
                            curr_partitioner, program
                        )
                    )

                    if can_skip_using_edge_do_not_decomp:
                        table = program_mod._default_decomposition_table()
                        ops_needing_preservation = []

                        for op in curr_ops_no_decomp:
                            if table.pop(op, None) is not None:
                                ops_needing_preservation.append(op)

                        program = program.run_decompositions(table)
                        final_ops_to_preserve.update(ops_needing_preservation)
                    else:
                        curr_ops_no_decomp = (
                            program_mod._remove_invalid_ops_for_not_decompose(
                                curr_ops_no_decomp
                            )
                        )

                        table = program_mod._default_decomposition_table()
                        for op in curr_ops_no_decomp:
                            table.pop(op, None)

                        # First pass: preserve this partitioner's candidate ops
                        # from nested decompositions.
                        program = program.run_decompositions(table)

                        if not _program_contains_targets(
                            program, set(curr_ops_no_decomp)
                        ):
                            skipped_second_passes += 1
                            continue

                        temp_partitioner_dict = {name: [curr_partitioner]}
                        preserved_ops = (
                            program_mod._replace_aten_ops_with_transformed_ops(
                                name, program, temp_partitioner_dict
                            )
                            or []
                        )
                        final_ops_to_preserve.update(preserved_ops)

                        # Second pass is only needed when preserve-candidate ops
                        # actually survive the first pass.
                        program = program.run_decompositions(
                            program_mod._default_decomposition_table()
                        )
                        program_mod._restore_transformed_ops_to_aten_ops(program)

                ops_set_to_not_decompose_by_program[name].extend(final_ops_to_preserve)

            edge_programs[name] = program_mod._generate_edge_program(
                config,
                program,
                preserve_ops=ops_set_to_not_decompose_by_program.get(name, []),
            )

        all_ops_not_to_decompose = list(
            set().union(*ops_set_to_not_decompose_by_program.values())
        )
        config._core_aten_ops_exception_list = list(
            set(config._core_aten_ops_exception_list) | set(all_ops_not_to_decompose)
        )

        edge_manager = program_mod.EdgeProgramManager(
            edge_programs,
            constant_methods,
            config,
            all_ops_not_to_decompose,
        )

        if generate_etrecord:
            etrecord = program_mod._create_empty_etrecord()
            etrecord.add_exported_program(aten_programs)
            etrecord.add_edge_dialect_program(copy.deepcopy(edge_manager))
            edge_manager._etrecord = etrecord

        if skipped_second_passes:
            logger.info(
                "[fast_preprocess] skipped %d redundant EDGE_DO_NOT_DECOMP cleanup passes",
                skipped_second_passes,
            )

        return edge_manager

    optimized_gen_edge_manager_for_partitioners.__arm_tosa_engine_patched__ = True
    optimized_gen_edge_manager_for_partitioners.__wrapped__ = current
    program_mod._gen_edge_manager_for_partitioners = (
        optimized_gen_edge_manager_for_partitioners
    )


def _graph_node_count(graph_module) -> int:
    import torch

    count = 0
    for module in graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            count += sum(1 for _ in module.graph.nodes)
    return count


def _install_run_decomp_fast_path() -> None:
    """Trim no-op recompiles in torch export decomposition helpers."""
    global _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS
    global _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS
    global _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS

    import functools

    import torch
    import torch.export._trace as trace_mod
    import torch.export.exported_program as exported_program_mod
    from torch._export.utils import _get_shape_env_from_gm
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )
    from torch._functorch._aot_autograd.input_output_analysis import (
        _graph_output_names,
    )
    from torch.export.exported_program import pytree

    try:
        runtime_assert_globals = trace_mod.apply_runtime_assertion_pass.__globals__
        insert_deferred_runtime_asserts = runtime_assert_globals[
            "insert_deferred_runtime_asserts"
        ]
        insert_aten_to_metadata_assert_pass = runtime_assert_globals[
            "_insert_aten_to_metadata_assert_pass"
        ]
        first_call_function_nn_module_stack = runtime_assert_globals[
            "first_call_function_nn_module_stack"
        ]
    except (AttributeError, KeyError) as exc:
        logger.warning(
            "[fast_preprocess] skipping run_decomp fast path; torch internals differ: %s",
            exc,
        )
        return

    if _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS is None:
        _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS = (
            exported_program_mod._register_constants_as_buffers
        )

        def fast_register_constants_as_buffers(
            mod, state_dict, non_persistent_buffers
        ):
            from torch.export.unflatten import _assign_attr, _AttrKind

            temp_registered_constants = set()

            for node in mod.graph.nodes:
                if node.op != "get_attr":
                    continue
                target = torch.fx.graph_module._get_attr(mod, node.target)
                if not isinstance(target, torch.Tensor):
                    continue
                if (node.target in state_dict) or (
                    node.target in non_persistent_buffers
                ):
                    continue

                torch.fx.graph_module._del_attr(mod, node.target)
                _assign_attr(target, mod, node.target, _AttrKind.BUFFER, False)
                temp_registered_constants.add(node.target)

            if temp_registered_constants:
                mod.recompile()

            return temp_registered_constants

        exported_program_mod._register_constants_as_buffers = (
            fast_register_constants_as_buffers
        )

    if _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS is None:
        _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS = trace_mod.apply_runtime_assertion_pass

        def fast_apply_runtime_assertion_pass(gm, graph_signature):
            modified = False

            if not torch._dynamo.config.do_not_emit_runtime_asserts:
                stack_trace = (
                    'File "torch/fx/passes/runtime_assert.py", line 24, '
                    "in insert_deferred_runtime_asserts"
                )
                before_nodes = _graph_node_count(gm)

                with _set_node_metadata_hook(
                    gm,
                    functools.partial(
                        _node_metadata_hook,
                        metadata={"stack_trace": stack_trace},
                    ),
                ):
                    shape_env = _get_shape_env_from_gm(gm)
                    if shape_env:
                        insert_deferred_runtime_asserts(
                            gm,
                            shape_env,
                            "exported program: "
                            f"{first_call_function_nn_module_stack(gm.graph)}",
                            export=True,
                        )

                insert_aten_to_metadata_assert_pass(gm)
                modified = _graph_node_count(gm) != before_nodes

            if modified:
                gm.recompile()

            graph_signature.user_outputs = _graph_output_names(gm)
            return gm, graph_signature

        trace_mod.apply_runtime_assertion_pass = fast_apply_runtime_assertion_pass

    if _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS is None:
        _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS = (
            exported_program_mod._remove_unnecessary_copy_op_pass
        )

        def fast_remove_unnecessary_copy_op_pass(gm, new_graph_signature):
            modified = False

            with gm._set_replace_hook(new_graph_signature.get_replace_hook()):
                for node in gm.graph.nodes:
                    if node.op != "output":
                        continue

                    args, _ = pytree.tree_flatten(node.args)
                    for out in args:
                        if not isinstance(out, torch.fx.Node):
                            continue
                        if (
                            out.name not in new_graph_signature.buffers_to_mutate
                            and out.name
                            not in new_graph_signature.parameters_to_mutate
                        ):
                            continue
                        if (
                            out.op == "call_function"
                            and out.target is torch.ops.aten.copy.default
                        ):
                            out.replace_all_uses_with(out.args[1])  # type: ignore[arg-type]
                            gm.graph.erase_node(out)
                            modified = True

                if modified:
                    gm.recompile()

            return gm, new_graph_signature

        exported_program_mod._remove_unnecessary_copy_op_pass = (
            fast_remove_unnecessary_copy_op_pass
        )


def _uninstall_edge_decomp_fast_path() -> None:
    global _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS

    if _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS is None:
        return

    from executorch.exir.program import _program as program_mod

    program_mod._gen_edge_manager_for_partitioners = (
        _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS
    )
    _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = None


def _uninstall_run_decomp_fast_path() -> None:
    global _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS
    global _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS
    global _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS

    import torch.export._trace as trace_mod
    import torch.export.exported_program as exported_program_mod

    if _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS is not None:
        exported_program_mod._register_constants_as_buffers = (
            _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS
        )
        _ORIGINAL_REGISTER_CONSTANTS_AS_BUFFERS = None

    if _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS is not None:
        trace_mod.apply_runtime_assertion_pass = (
            _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS
        )
        _ORIGINAL_APPLY_RUNTIME_ASSERTION_PASS = None

    if _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS is not None:
        exported_program_mod._remove_unnecessary_copy_op_pass = (
            _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS
        )
        _ORIGINAL_REMOVE_UNNECESSARY_COPY_OP_PASS = None


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
    import os
    from pathlib import Path

    t0 = time.perf_counter()
    gm = edge_program.graph_module

    # Deep-copy the state_dict BEFORE passes run, because some FX-direct
    # passes (e.g. RewriteConv) may mutate ep.state_dict in-place.
    # We want the original (un-mutated) weight data for pre-existing tensors,
    # but we still need to ingest any new constant placeholders created by the
    # FX passes later in the pipeline.
    original_state_dict = copy.deepcopy(dict(edge_program.state_dict))

    import arm_tosa_engine

    # Run the full FX-direct pipeline (all passes in Rust via PyO3)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
    t_passes = time.perf_counter() - t0

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

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


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
