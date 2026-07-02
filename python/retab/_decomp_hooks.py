"""Decomposition hook installation helpers for fast_preprocess_pipeline."""

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator

from torch.export.exported_program import ExportedProgram
from executorch.exir.tracer import _default_decomposition_table

logger = logging.getLogger(__name__)

_ORIGINAL_PATCHES = {}
_RUN_DECOMP_DISABLE_UNLIFT_GUARDS = False


@contextmanager
def _profile_scope(name: str, **fields: Any) -> Generator[None, None, None]:
    if os.environ.get("ET_LOWERING_PROFILE") != "1":
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        suffix = "".join(f" {key}={value}" for key, value in fields.items())
        print(
            f"[et-lower-profile] {elapsed:9.3f}s retab.{name}{suffix}",
            file=sys.stderr,
            flush=True,
        )


def _env_get(name: str, *aliases: str) -> str | None:
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return None


def _install_patch_once(slot: str, module, attr: str, wrap):
    if slot in _ORIGINAL_PATCHES:
        return

    original = getattr(module, attr)
    _ORIGINAL_PATCHES[slot] = original
    setattr(module, attr, wrap(original))


def _restore_patch(slot: str, module, attr: str) -> None:
    original = _ORIGINAL_PATCHES.pop(slot, None)
    if original is not None:
        setattr(module, attr, original)


def _install_override_once(slot: str, getter, setter, value) -> None:
    if slot in _ORIGINAL_PATCHES:
        return

    _ORIGINAL_PATCHES[slot] = getter()
    setter(value)


def _restore_override(slot: str, setter) -> None:
    if slot not in _ORIGINAL_PATCHES:
        return

    setter(_ORIGINAL_PATCHES.pop(slot))


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


def _functionalize_trivial_inplace_ops(program: ExportedProgram) -> int:
    import torch

    replacements = {
        torch.ops.aten.relu_.default: torch.ops.aten.relu.default,
    }
    rewrite_count = 0

    for module in _iter_graph_modules(program.graph_module):
        module_modified = False
        for node in module.graph.nodes:
            if node.op != "call_function" or node.target not in replacements:
                continue
            if not node.args or not isinstance(node.args[0], torch.fx.Node):
                continue

            input_node = node.args[0]
            if any(user is not node for user in input_node.users):
                continue

            node.target = replacements[node.target]
            module_modified = True
            rewrite_count += 1

        if module_modified:
            module.recompile()

    return rewrite_count


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
            with _profile_scope(
                "edge_decomp.optimized_gen_edge_manager", methods=len(aten_programs)
            ):
                for name, program in aten_programs.items():
                    if os.environ.get("RETAB_DISABLE_TRIVIAL_INPLACE_FAST_PATH") != "1":
                        with _profile_scope(
                            "edge_decomp.functionalize_trivial_inplace", method=name
                        ):
                            _functionalize_trivial_inplace_ops(program)
                    with _profile_scope(
                        "edge_decomp.can_skip_initial_functionalize", method=name
                    ):
                        can_skip = _can_skip_initial_functionalize(
                            program, partitioner.get(name, [])
                        )
                    if not can_skip:
                        with _profile_scope("edge_decomp.fallback_current", method=name):
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
                        with _profile_scope(
                            "edge_decomp.can_skip_initial_functionalize",
                            method=name,
                        ):
                            can_skip = _can_skip_initial_functionalize(
                                program, partitioners_for_program
                            )
                        if can_skip:
                            skipped_initial_functionalize += 1
                        else:
                            with _profile_scope(
                                "edge_decomp.initial_empty_decomposition", method=name
                            ):
                                program = program.run_decompositions({})
                        final_ops_to_preserve = set()

                        if not partitioners_for_program:
                            table = _default_decomposition_table()
                            for op in config.preserve_ops:
                                table.pop(op, None)
                            with _profile_scope(
                                "edge_decomp.default_decomposition", method=name
                            ):
                                program = program.run_decompositions(table)

                        for curr_partitioner in partitioners_for_program:
                            with _profile_scope(
                                "edge_decomp.ops_to_not_decompose",
                                method=name,
                                partitioner=curr_partitioner.__class__.__name__,
                            ):
                                curr_ops_no_decomp, _ = (
                                    curr_partitioner.ops_to_not_decompose(program)
                                )

                            with _profile_scope(
                                "edge_decomp.can_skip_edge_do_not_decomp",
                                method=name,
                                partitioner=curr_partitioner.__class__.__name__,
                            ):
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

                                with _profile_scope(
                                    "edge_decomp.preserve_decomposition",
                                    method=name,
                                    partitioner=curr_partitioner.__class__.__name__,
                                ):
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

                                with _profile_scope(
                                    "edge_decomp.edge_do_not_decomp_first_decomposition",
                                    method=name,
                                    partitioner=curr_partitioner.__class__.__name__,
                                ):
                                    program = program.run_decompositions(table)

                                with _profile_scope(
                                    "edge_decomp.program_contains_preserve_targets",
                                    method=name,
                                    partitioner=curr_partitioner.__class__.__name__,
                                ):
                                    contains_preserve_targets = _program_contains_targets(
                                        program, set(curr_ops_no_decomp)
                                    )
                                if not contains_preserve_targets:
                                    skipped_second_passes += 1
                                    continue

                                temp_partitioner_dict = {name: [curr_partitioner]}
                                with _profile_scope(
                                    "edge_decomp.replace_aten_ops_with_transformed_ops",
                                    method=name,
                                    partitioner=curr_partitioner.__class__.__name__,
                                ):
                                    preserved_ops = (
                                        program_mod._replace_aten_ops_with_transformed_ops(
                                            name, program, temp_partitioner_dict
                                        )
                                        or []
                                    )
                                final_ops_to_preserve.update(preserved_ops)

                                default_table = _default_decomposition_table()
                                with _profile_scope(
                                    "edge_decomp.scan_default_table_targets",
                                    method=name,
                                    partitioner=curr_partitioner.__class__.__name__,
                                ):
                                    call_targets = {
                                        n.target
                                        for n in program.graph_module.graph.nodes
                                        if n.op == "call_function"
                                    }
                                    has_default_targets = any(
                                        op in call_targets for op in default_table
                                    )
                                if not has_default_targets:
                                    skipped_second_passes += 1
                                else:
                                    with _profile_scope(
                                        "edge_decomp.second_default_decomposition",
                                        method=name,
                                        partitioner=curr_partitioner.__class__.__name__,
                                    ):
                                        program = program.run_decompositions(
                                            default_table
                                        )
                                with _profile_scope(
                                    "edge_decomp.restore_transformed_ops", method=name
                                ):
                                    program_mod._restore_transformed_ops_to_aten_ops(
                                        program
                                    )

                        ops_set_to_not_decompose_by_program[name].extend(
                            final_ops_to_preserve
                        )

                    with _profile_scope("edge_decomp.generate_edge_program", method=name):
                        edge_program = program_mod._generate_edge_program(
                            config,
                            program,
                            preserve_ops=ops_set_to_not_decompose_by_program.get(
                                name, []
                            )
                        )
                    if os.environ.get("ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING") == "1":
                        try:
                            from executorch.backends.arm.tosa.partitioner import (
                                _sync_direct_delegate_output_meta,
                            )

                            with _profile_scope(
                                "edge_decomp.sync_direct_delegate_output_meta",
                                method=name,
                            ):
                                _sync_direct_delegate_output_meta(edge_program)
                        except Exception as exc:
                            logger.info(
                                "[fast_preprocess] direct delegate output-meta sync failed for %s: %s",
                                name,
                                exc,
                            )

                    edge_programs[name] = edge_program

                all_ops_not_to_decompose = list(
                    set().union(*ops_set_to_not_decompose_by_program.values())
                )
                config._core_aten_ops_exception_list = list(
                    set(config._core_aten_ops_exception_list)
                    | set(all_ops_not_to_decompose)
                )

                with _profile_scope("edge_decomp.construct_edge_manager"):
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
                if skipped_initial_functionalize:
                    logger.info(
                        "[fast_preprocess] skipped %d upfront empty decomposition passes",
                        skipped_initial_functionalize,
                    )

                return edge_manager

        optimized_gen_edge_manager_for_partitioners.__retab_patched__ = True
        optimized_gen_edge_manager_for_partitioners.__wrapped__ = current
        return optimized_gen_edge_manager_for_partitioners

    def override_gen_edge_manager_for_partitioners(
        current,
        partitioner,
        aten_programs,
        config,
        constant_methods,
        generate_etrecord=False,
    ):
        optimized = wrap_gen_edge_manager_for_partitioners(current)
        return optimized(
            partitioner,
            aten_programs,
            config,
            constant_methods,
            generate_etrecord,
        )

    set_override = getattr(
        program_mod, "set_edge_manager_for_partitioners_override", None
    )
    get_override = getattr(
        program_mod, "get_edge_manager_for_partitioners_override", None
    )
    if callable(set_override) and callable(get_override):
        _install_override_once(
            "edge_manager_for_partitioners_override",
            get_override,
            set_override,
            override_gen_edge_manager_for_partitioners,
        )
        return

    _install_patch_once(
        "gen_edge_manager_for_partitioners",
        program_mod,
        "_gen_edge_manager_for_partitioners",
        wrap_gen_edge_manager_for_partitioners,
    )


def _install_op_replace_fast_path() -> None:
    """Replace ATen targets with Edge targets by rewriting FX nodes in place."""
    from executorch.exir.passes import replace_aten_with_edge_pass as replace_mod
    from torch.fx.passes.infra.pass_base import PassResult

    def wrap_call(original):
        if getattr(original, "__retab_patched__", False):
            return original

        def fast_call(self, graph_module):
            modified = False
            with _profile_scope("edge_decomp.op_replace_fast_call"):
                for module in _iter_graph_modules(graph_module):
                    module_modified = False
                    for node in module.graph.nodes:
                        if node.op != "call_function":
                            continue
                        if not replace_mod.should_lower_to_edge(node.target):
                            continue

                        edge_target = replace_mod.aten_to_edge(node.target)
                        if edge_target == node.target:
                            continue

                        node.target = edge_target
                        module_modified = True

                    if module_modified:
                        module.recompile()
                        modified = True

            return PassResult(graph_module, modified)

        fast_call.__retab_patched__ = True
        fast_call.__wrapped__ = original
        return fast_call

    _install_patch_once(
        "op_replace_call",
        replace_mod.OpReplacePass,
        "call",
        wrap_call,
    )


def _install_memory_format_fast_path() -> None:
    """Convert memory_format kwargs to dim_order kwargs without ExportPass retracing."""
    import torch
    from executorch.exir.dim_order_utils import get_dim_order
    from executorch.exir.passes import memory_format_ops_pass as memory_format_mod
    from executorch.exir.passes.dim_order_ops_registry import DimOrderOpsMap
    from torch.fx.passes.infra.pass_base import PassResult

    def arg_value(arg):
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val")
        return arg

    def rank_and_dim_order(arg):
        value = arg_value(arg)
        if isinstance(value, torch.Tensor):
            return value.dim(), [int(d) for d in value.dim_order()]
        if isinstance(arg, torch.fx.immutable_collections.immutable_list):
            return len(arg), None
        if isinstance(arg, (tuple, list)):
            return len(arg), None
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return len(value), None
        return None

    def rewritten_kwargs(node):
        if not node.args:
            return None

        nkwargs = dict(node.kwargs)
        mem_format = nkwargs.pop("memory_format", torch.preserve_format)
        input_info = rank_and_dim_order(node.args[0])
        if input_info is None:
            return None

        ndim, inherited_dim_order = input_info
        if mem_format in (None, torch.preserve_format):
            dim_order = (
                inherited_dim_order
                if inherited_dim_order is not None
                else list(range(ndim))
            )
        else:
            dim_order = get_dim_order(mem_format, ndim)
            if dim_order is None:
                return None

        nkwargs["dim_order"] = dim_order
        return nkwargs

    def wrap_call(original):
        if getattr(original, "__retab_patched__", False):
            return original

        def fast_call(self, graph_module):
            updates = []
            with _profile_scope("edge_decomp.memory_format_fast_scan"):
                for module in _iter_graph_modules(graph_module):
                    for node in module.graph.nodes:
                        if (
                            node.op != "call_function"
                            or node.target not in DimOrderOpsMap
                        ):
                            continue
                        nkwargs = rewritten_kwargs(node)
                        if nkwargs is None:
                            return original(self, graph_module)
                        updates.append(
                            (module, node, DimOrderOpsMap[node.target], nkwargs)
                        )

            if not updates:
                return PassResult(graph_module, False)

            touched_modules = set()
            with _profile_scope(
                "edge_decomp.memory_format_fast_apply", rewrites=len(updates)
            ):
                for module, node, target, nkwargs in updates:
                    node.target = target
                    node.kwargs = nkwargs
                    touched_modules.add(module)

                for module in touched_modules:
                    module.recompile()

            return PassResult(graph_module, True)

        fast_call.__retab_patched__ = True
        fast_call.__wrapped__ = original
        return fast_call

    _install_patch_once(
        "memory_format_call",
        memory_format_mod.MemoryFormatOpsPass,
        "call",
        wrap_call,
    )


def _graph_node_count(graph_module) -> int:
    import torch

    count = 0
    for module in graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            count += sum(1 for _ in module.graph.nodes)
    return count


def _direct_lower_node_limit() -> int:
    raw = _env_get(
        "RETAB_DIRECT_LOWER_NODE_LIMIT",
        "ARM_TOSA_ENGINE_DIRECT_LOWER_NODE_LIMIT",
    )
    if raw is None:
        # Direct lowering is best for small graphs; on larger graphs it can
        # speculatively preprocess many fragments before falling back.
        return 128
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[fast_preprocess] ignoring invalid RETAB_DIRECT_LOWER_NODE_LIMIT=%r",
            raw,
        )
        return 128


def _install_direct_lower_size_guard() -> None:
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner

    def wrap_direct_lower(original):
        if getattr(original, "__retab_patched__", False):
            return original

        def guarded_direct_lower(self, exported_program, compile_config):
            limit = _direct_lower_node_limit()
            if (
                limit > 0
                and getattr(compile_config, "_enable_direct_backend_lowering", False)
                and _graph_node_count(exported_program.graph_module) > limit
            ):
                return None
            return original(self, exported_program, compile_config)

        guarded_direct_lower.__retab_patched__ = True
        guarded_direct_lower.__wrapped__ = original
        return guarded_direct_lower

    _install_patch_once(
        "tosa_partitioner_direct_lower",
        TOSAPartitioner,
        "direct_lower",
        wrap_direct_lower,
    )


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
            use_fast_unlift = _uses_straight_line_fast_path_program(ep)
            previous = _RUN_DECOMP_DISABLE_UNLIFT_GUARDS
            try:
                if use_fast_unlift:
                    _RUN_DECOMP_DISABLE_UNLIFT_GUARDS = True
                return original(
                    ep,
                    cia_to_decomp=cia_to_decomp,
                    python_decomp_table=python_decomp_table,
                    joint_loss_index=joint_loss_index,
                    decompose_custom_triton_ops=decompose_custom_triton_ops,
                )
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

    _install_direct_lower_size_guard()

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

    set_override = getattr(
        program_mod, "set_edge_manager_for_partitioners_override", None
    )
    if callable(set_override):
        _restore_override("edge_manager_for_partitioners_override", set_override)

    _restore_patch(
        "gen_edge_manager_for_partitioners",
        program_mod,
        "_gen_edge_manager_for_partitioners",
    )


def _uninstall_op_replace_fast_path() -> None:
    from executorch.exir.passes import replace_aten_with_edge_pass as replace_mod

    _restore_patch(
        "op_replace_call",
        replace_mod.OpReplacePass,
        "call",
    )


def _uninstall_memory_format_fast_path() -> None:
    from executorch.exir.passes import memory_format_ops_pass as memory_format_mod

    _restore_patch(
        "memory_format_call",
        memory_format_mod.MemoryFormatOpsPass,
        "call",
    )


def _uninstall_run_decomp_fast_path() -> None:
    global _RUN_DECOMP_DISABLE_UNLIFT_GUARDS

    import torch.export._trace as trace_mod
    import torch.export.exported_program as exported_program_mod
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner

    _restore_patch(
        "tosa_partitioner_direct_lower",
        TOSAPartitioner,
        "direct_lower",
    )
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
