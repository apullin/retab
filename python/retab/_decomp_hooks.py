"""RETAB's opt-in edge-manager and torch run_decompositions fast paths.

Each fast path monkeypatches a private ExecuTorch or torch function and restores it
on uninstall.
"""

import logging

from torch.export.exported_program import ExportedProgram
from executorch.exir.tracer import _default_decomposition_table

logger = logging.getLogger(__name__)

_ORIGINAL_PATCHES = {}
_RUN_DECOMP_DISABLE_UNLIFT_GUARDS = False

# Sticky, process-wide kill switch for the nn_module_stack fast path.
#
# The fast path swaps export's ModuleStackTracer for the cheaper PythonKeyTracer
# (torch passes `store_orig_mod=_record_nn_module_stack`). Nodes that survive the
# retrace inherit `nn_module_stack` from their pre-decomposition counterpart, but
# nodes decomposition *creates* have nothing to inherit from and end up bare --
# which `_verify_nn_module_stack` rejects outright, since it requires the metadata
# on every top-level call_function/get_attr node.
#
# Whether decomposition creates such a node cannot be decided from the
# pre-decomposition graph that `_uses_straight_line_fast_path_graph_module`
# inspects, so this is detected on failure and latched off for the rest of the
# process instead of predicted up front.
_NN_MODULE_STACK_FAST_PATH_UNSAFE = False


def _install_patch_once(slot: str, module, attr: str, wrap):
    if slot in _ORIGINAL_PATCHES:
        return

    # These are optional performance fast-paths that monkeypatch ExecuTorch/torch
    # internals, and those internals move between ExecuTorch and torch versions.
    # The core RETAB TOSA override installs separately, so skipping an absent
    # target costs speed but preserves correctness.
    original = getattr(module, attr, None)
    if original is None:
        logger.warning(
            "[retab] skipping patch %s: %s.%s not found in this build",
            slot,
            getattr(module, "__name__", module),
            attr,
        )
        return
    _ORIGINAL_PATCHES[slot] = original
    setattr(module, attr, wrap(original))


def _restore_patch(slot: str, module, attr: str) -> None:
    original = _ORIGINAL_PATCHES.pop(slot, None)
    if original is not None:
        setattr(module, attr, original)


def _iter_graph_modules(gm):
    import torch

    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule):
            yield module


def _graph_module_contains_targets(gm, targets: set) -> bool:
    for module in _iter_graph_modules(gm):
        for node in module.graph.nodes:
            if node.op == "call_function" and node.target in targets:
                return True
    return False


def _program_contains_targets(program: ExportedProgram, targets: set) -> bool:
    return _graph_module_contains_targets(program.graph_module, targets)


def _uses_straight_line_fast_path_program(program: ExportedProgram) -> bool:
    return _uses_straight_line_fast_path_graph_module(program.graph_module)


def _graph_module_has_mutating_op(gm) -> bool:
    for module in _iter_graph_modules(gm):
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue
            schema = getattr(node.target, "_schema", None)
            if schema is None:
                continue
            if any(
                arg.alias_info is not None and arg.alias_info.is_write
                for arg in schema.arguments
            ):
                return True
    return False


def _uses_straight_line_fast_path_graph_module(gm) -> bool:
    import torch

    if _NN_MODULE_STACK_FAST_PATH_UNSAFE:
        return False
    hop_targets = {
        torch.ops.higher_order.cond,
        torch.ops.higher_order.while_loop,
    }
    if _graph_module_contains_targets(gm, hop_targets):
        return False
    # In-place mutations (silu_, ...) — and their functionalized aten.copy nodes —
    # need the upfront nn_module_stack recording that the fast path disables; without
    # it the exporter's _verify_nn_module_stack rejects the bare copy node. Check both
    # forms since the graph is pre-functionalization (silu_) at the recording site and
    # post-functionalization (copy) later.
    if _graph_module_contains_targets(gm, {torch.ops.aten.copy.default}):
        return False
    return not _graph_module_has_mutating_op(gm)


def _latch_off_nn_module_stack_fast_path(exc: Exception) -> None:
    """Disable the nn_module_stack fast path for the rest of the process.

    Latched rather than per-call so a model that needs the full ModuleStackTracer
    pays the doubled decomposition once, not on every `run_decompositions`.
    """
    global _NN_MODULE_STACK_FAST_PATH_UNSAFE
    _NN_MODULE_STACK_FAST_PATH_UNSAFE = True
    logger.warning(
        "[retab] decomposition produced a node without nn_module_stack "
        "(%s); disabling the nn_module_stack fast path and retracing with "
        "ModuleStackTracer. Lowering is correct but slower.",
        exc,
    )


def _has_trivial_module_call_graph(module_call_graph) -> bool:
    if len(module_call_graph) != 1:
        return False

    signature = module_call_graph[0].signature
    if signature is None:
        return True

    return not signature.inputs and not signature.outputs


def _can_skip_initial_functionalize(
    program: ExportedProgram, partitioners_for_program
) -> bool:
    """Skip the upfront empty decomposition for the ARM TOSA path."""
    if not partitioners_for_program:
        return True

    try:
        from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
        from executorch.exir.program import _program as program_mod
    except Exception:
        return False

    if not all(isinstance(p, TOSAPartitioner) for p in partitioners_for_program):
        return False

    return _uses_straight_line_fast_path_program(program)


def _install_edge_decomp_fast_path() -> None:
    """Skip redundant EDGE_DO_NOT_DECOMP cleanup decompositions.

    ExecuTorch's EDGE_DO_NOT_DECOMP path always runs a second
    `run_decompositions(default_table)` pass after the preserve-op rewrite
    phase. That second pass is only needed if the first pass actually left any
    preserve-candidate ops in the graph. When none remain, the second pass is a
    no-op that still costs multiple seconds on large models.
    """
    from collections import defaultdict
    import copy

    from executorch.exir.program import _program as program_mod

    def wrap_gen_edge_manager_for_partitioners(current):
        if getattr(current, "__retab_patched__", False):
            return current

        def optimized_gen_edge_manager_for_partitioners(
            partitioner,
            aten_programs,
            config,
            constant_methods,
            generate_etrecord=False,
        ):
            for name, program in aten_programs.items():
                if not _can_skip_initial_functionalize(
                    program, partitioner.get(name, [])
                ):
                    return current(
                        partitioner,
                        aten_programs,
                        config,
                        constant_methods,
                        generate_etrecord,
                    )

            ops_set_to_not_decompose_by_program = defaultdict(list)
            edge_programs = {}
            skipped_initial_functionalize = 0
            skipped_second_passes = 0

            for name, program in aten_programs.items():
                if partitioner is not None:
                    partitioners_for_program = partitioner.get(name, [])
                    if _can_skip_initial_functionalize(
                        program, partitioners_for_program
                    ):
                        skipped_initial_functionalize += 1
                    else:
                        program = program.run_decompositions({})
                    final_ops_to_preserve = set()

                    if not partitioners_for_program:
                        table = _default_decomposition_table()
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
                            table = _default_decomposition_table()
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

                            table = _default_decomposition_table()
                            for op in curr_ops_no_decomp:
                                table.pop(op, None)

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

                            default_table = _default_decomposition_table()
                            if not any(
                                op
                                in {n.target for n in program.graph_module.graph.nodes if n.op == "call_function"}
                                for op in default_table
                            ):
                                skipped_second_passes += 1
                            else:
                                program = program.run_decompositions(default_table)
                            program_mod._restore_transformed_ops_to_aten_ops(program)

                    ops_set_to_not_decompose_by_program[name].extend(
                        final_ops_to_preserve
                    )

                edge_programs[name] = program_mod._generate_edge_program(
                    config,
                    program,
                    preserve_ops=ops_set_to_not_decompose_by_program.get(name, []),
                )

            all_ops_not_to_decompose = list(
                set().union(*ops_set_to_not_decompose_by_program.values())
            )
            config._core_aten_ops_exception_list = list(
                set(config._core_aten_ops_exception_list)
                | set(all_ops_not_to_decompose)
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
                    "[retab] skipped %d redundant EDGE_DO_NOT_DECOMP cleanup passes",
                    skipped_second_passes,
                )
            if skipped_initial_functionalize:
                logger.info(
                    "[retab] skipped %d upfront empty decomposition passes",
                    skipped_initial_functionalize,
                )

            return edge_manager

        optimized_gen_edge_manager_for_partitioners.__retab_patched__ = True
        optimized_gen_edge_manager_for_partitioners.__wrapped__ = current
        return optimized_gen_edge_manager_for_partitioners

    _install_patch_once(
        "gen_edge_manager_for_partitioners",
        program_mod,
        "_gen_edge_manager_for_partitioners",
        wrap_gen_edge_manager_for_partitioners,
    )


def _graph_node_count(graph_module) -> int:
    import torch

    count = 0
    for module in graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            count += sum(1 for _ in module.graph.nodes)
    return count


def _install_trace_run_decomp_fast_path_patches(
    trace_mod,
    torch,
    functools,
    _get_shape_env_from_gm,
    _node_metadata_hook,
    _set_node_metadata_hook,
    _graph_output_names,
    insert_deferred_runtime_asserts,
    insert_aten_to_metadata_assert_pass,
    first_call_function_nn_module_stack,
) -> None:
    def wrap_aot_export_joint_with_descriptors(original):
        def fast_aot_export_joint_with_descriptors(*args, **kwargs):
            mod = args[1] if len(args) > 1 else None
            if (
                kwargs.get("_record_nn_module_stack", False)
                and isinstance(mod, torch.fx.GraphModule)
                and _uses_straight_line_fast_path_graph_module(mod)
            ):
                kwargs = dict(kwargs)
                kwargs["_record_nn_module_stack"] = False

            return original(*args, **kwargs)

        return fast_aot_export_joint_with_descriptors

    _install_patch_once(
        "aot_export_joint_with_descriptors",
        trace_mod,
        "_aot_export_joint_with_descriptors",
        wrap_aot_export_joint_with_descriptors,
    )

    def wrap_apply_runtime_assertion_pass(original):
        def fast_apply_runtime_assertion_pass(gm, graph_signature):
            if not _uses_straight_line_fast_path_graph_module(gm):
                return original(gm, graph_signature)

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

        return fast_apply_runtime_assertion_pass

    _install_patch_once(
        "apply_runtime_assertion_pass",
        trace_mod,
        "apply_runtime_assertion_pass",
        wrap_apply_runtime_assertion_pass,
    )


def _install_exported_program_run_decomp_fast_path_patches(
    exported_program_mod,
    torch,
    pytree,
) -> None:
    global _RUN_DECOMP_DISABLE_UNLIFT_GUARDS

    def wrap_placeholder_naming_pass(original):
        def fast_placeholder_naming_pass(
            gm,
            export_graph_signature,
            mod,
            fake_args,
            fake_kwargs,
            fake_params_buffers,
            constants,
        ):
            if not _uses_straight_line_fast_path_graph_module(gm):
                return original(
                    gm,
                    export_graph_signature,
                    mod,
                    fake_args,
                    fake_kwargs,
                    fake_params_buffers,
                    constants,
                )

            original_recompile = torch.fx.GraphModule.recompile

            def noop_recompile(self, *args, **kwargs):
                return None

            torch.fx.GraphModule.recompile = noop_recompile
            try:
                return original(
                    gm,
                    export_graph_signature,
                    mod,
                    fake_args,
                    fake_kwargs,
                    fake_params_buffers,
                    constants,
                )
            finally:
                torch.fx.GraphModule.recompile = original_recompile

        return fast_placeholder_naming_pass

    _install_patch_once(
        "placeholder_naming_pass",
        exported_program_mod,
        "placeholder_naming_pass",
        wrap_placeholder_naming_pass,
    )

    def wrap_register_constants_as_buffers(original):
        def fast_register_constants_as_buffers(mod, state_dict, non_persistent_buffers):
            if not _uses_straight_line_fast_path_graph_module(mod):
                return original(mod, state_dict, non_persistent_buffers)

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

        return fast_register_constants_as_buffers

    _install_patch_once(
        "register_constants_as_buffers",
        exported_program_mod,
        "_register_constants_as_buffers",
        wrap_register_constants_as_buffers,
    )

    def wrap_remove_unnecessary_copy_op_pass(original):
        def fast_remove_unnecessary_copy_op_pass(gm, new_graph_signature):
            if not _uses_straight_line_fast_path_graph_module(gm):
                return original(gm, new_graph_signature)

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

        return fast_remove_unnecessary_copy_op_pass

    _install_patch_once(
        "remove_unnecessary_copy_op_pass",
        exported_program_mod,
        "_remove_unnecessary_copy_op_pass",
        wrap_remove_unnecessary_copy_op_pass,
    )

    def wrap_exported_program_module(original):
        def fast_exported_program_module(self, check_guards=True):
            if _RUN_DECOMP_DISABLE_UNLIFT_GUARDS:
                check_guards = False
            return original(self, check_guards=check_guards)

        return fast_exported_program_module

    _install_patch_once(
        "exported_program_module",
        exported_program_mod.ExportedProgram,
        "module",
        wrap_exported_program_module,
    )

    def wrap_decompose_and_get_gm_with_new_signature_constants(original):
        def fast_decompose_and_get_gm_with_new_signature_constants(
            ep,
            *,
            cia_to_decomp,
            python_decomp_table,
            joint_loss_index,
            decompose_custom_triton_ops,
        ):
            global _RUN_DECOMP_DISABLE_UNLIFT_GUARDS
            from torch._export.verifier import SpecViolationError

            use_fast_unlift = _uses_straight_line_fast_path_program(ep)
            previous = _RUN_DECOMP_DISABLE_UNLIFT_GUARDS

            def call_original():
                return original(
                    ep,
                    cia_to_decomp=cia_to_decomp,
                    python_decomp_table=python_decomp_table,
                    joint_loss_index=joint_loss_index,
                    decompose_custom_triton_ops=decompose_custom_triton_ops,
                )

            try:
                if use_fast_unlift:
                    _RUN_DECOMP_DISABLE_UNLIFT_GUARDS = True
                try:
                    return call_original()
                except SpecViolationError as exc:
                    if not use_fast_unlift or "nn_module_stack" not in str(exc):
                        raise
                    # Retrying is sound: the callee derives a fresh module via
                    # `ep.module()` on every call and leaves `ep` untouched, so the
                    # second attempt reproduces the stock graph exactly.
                    _latch_off_nn_module_stack_fast_path(exc)
                    _RUN_DECOMP_DISABLE_UNLIFT_GUARDS = False
                    return call_original()
            finally:
                _RUN_DECOMP_DISABLE_UNLIFT_GUARDS = previous

        return fast_decompose_and_get_gm_with_new_signature_constants

    _install_patch_once(
        "decompose_and_get_gm_with_new_signature_constants",
        exported_program_mod,
        "_decompose_and_get_gm_with_new_signature_constants",
        wrap_decompose_and_get_gm_with_new_signature_constants,
    )

    def wrap_get_updated_module_call_graph(original):
        def fast_get_updated_module_call_graph(
            old_gm,
            old_graph_signature,
            gm,
            graph_signature,
            old_module_call_graph,
        ):
            if (
                _has_trivial_module_call_graph(old_module_call_graph)
                and _uses_straight_line_fast_path_graph_module(old_gm)
                and _uses_straight_line_fast_path_graph_module(gm)
            ):
                return old_module_call_graph

            return original(
                old_gm,
                old_graph_signature,
                gm,
                graph_signature,
                old_module_call_graph,
            )

        return fast_get_updated_module_call_graph

    _install_patch_once(
        "get_updated_module_call_graph",
        exported_program_mod,
        "_get_updated_module_call_graph",
        wrap_get_updated_module_call_graph,
    )

    def wrap_get_updated_range_constraints(original):
        def fast_get_updated_range_constraints(
            gm,
            old_range_constraints=None,
        ):
            if not old_range_constraints:
                return old_range_constraints or {}
            return original(gm, old_range_constraints)

        return fast_get_updated_range_constraints

    _install_patch_once(
        "get_updated_range_constraints",
        exported_program_mod,
        "_get_updated_range_constraints",
        wrap_get_updated_range_constraints,
    )


def _install_run_decomp_fast_path() -> None:
    """Trim no-op recompiles in torch export decomposition helpers."""
    global _RUN_DECOMP_DISABLE_UNLIFT_GUARDS

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
            "[retab] skipping run_decomp fast path; torch internals differ: %s",
            exc,
        )
        return

    _install_trace_run_decomp_fast_path_patches(
        trace_mod,
        torch,
        functools,
        _get_shape_env_from_gm,
        _node_metadata_hook,
        _set_node_metadata_hook,
        _graph_output_names,
        insert_deferred_runtime_asserts,
        insert_aten_to_metadata_assert_pass,
        first_call_function_nn_module_stack,
    )
    _install_exported_program_run_decomp_fast_path_patches(
        exported_program_mod,
        torch,
        pytree,
    )


def _uninstall_edge_decomp_fast_path() -> None:
    from executorch.exir.program import _program as program_mod

    _restore_patch(
        "gen_edge_manager_for_partitioners",
        program_mod,
        "_gen_edge_manager_for_partitioners",
    )


def _uninstall_run_decomp_fast_path() -> None:
    global _RUN_DECOMP_DISABLE_UNLIFT_GUARDS

    import torch.export._trace as trace_mod
    import torch.export.exported_program as exported_program_mod

    _restore_patch(
        "aot_export_joint_with_descriptors",
        trace_mod,
        "_aot_export_joint_with_descriptors",
    )
    _restore_patch(
        "apply_runtime_assertion_pass",
        trace_mod,
        "apply_runtime_assertion_pass",
    )
    _restore_patch(
        "register_constants_as_buffers",
        exported_program_mod,
        "_register_constants_as_buffers",
    )
    _restore_patch(
        "placeholder_naming_pass",
        exported_program_mod,
        "placeholder_naming_pass",
    )
    _restore_patch(
        "remove_unnecessary_copy_op_pass",
        exported_program_mod,
        "_remove_unnecessary_copy_op_pass",
    )
    _restore_patch(
        "exported_program_module",
        exported_program_mod.ExportedProgram,
        "module",
    )
    _restore_patch(
        "decompose_and_get_gm_with_new_signature_constants",
        exported_program_mod,
        "_decompose_and_get_gm_with_new_signature_constants",
    )
    _restore_patch(
        "get_updated_module_call_graph",
        exported_program_mod,
        "_get_updated_module_call_graph",
    )
    _restore_patch(
        "get_updated_range_constraints",
        exported_program_mod,
        "_get_updated_range_constraints",
    )

    _RUN_DECOMP_DISABLE_UNLIFT_GUARDS = False
