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
from typing import List, cast

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

logger = logging.getLogger(__name__)


def _flatten_output_nodes(output_arg):
    if isinstance(output_arg, (list, tuple)):
        for item in output_arg:
            yield from _flatten_output_nodes(item)
    elif hasattr(output_arg, "op"):
        yield output_arg


def _collect_delegate_info_meta(graph_module):
    output_node = graph_module.graph.output_node()
    output_leaves = list(_flatten_output_nodes(output_node.args[0]))
    output_meta = []
    for node in output_leaves:
        fake_val = node.meta.get("val")
        output_meta.append(
            {
                "fake_val": fake_val,
                "tosa_dim_order": tuple(node.meta.get("tosa_dim_order", ())),
                "node_name": node.name,
            }
        )
    return {"output_meta": output_meta}


def _get_matching_fake_tensor(node):
    """Return a fake tensor matching a node's value and Arm dim-order metadata."""
    from executorch.exir.dim_order_utils import get_memory_format

    fake_tensor = node.meta["val"]
    desired_dim_order = node.meta.get("tosa_dim_order")
    if desired_dim_order is None:
        return fake_tensor
    return fake_tensor.to(memory_format=get_memory_format(list(desired_dim_order)))


def _ordered_control_flow_users(node):
    def _user_sort_key(user):
        target_str = str(getattr(user, "target", ""))
        if "getitem" in target_str and len(getattr(user, "args", ())) > 1:
            idx = user.args[1]
            if isinstance(idx, int):
                return (0, idx, getattr(user, "name", ""))
        return (1, 0, getattr(user, "name", ""))

    return sorted(list(node.users), key=_user_sort_key)


def _control_flow_debug_enabled() -> bool:
    return bool(os.environ.get("ARM_TOSA_ENGINE_DEBUG_CONTROL_FLOW"))


def _log_control_flow_debug(message: str, *args) -> None:
    if _control_flow_debug_enabled():
        logger.info(message, *args)


def _extract_effective_node_qparam(node):
    """Follow producer edges until the node's effective quantization is known."""
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    current = node
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        output_qparams = current.meta.get("output_qparams")
        if output_qparams:
            if isinstance(output_qparams, dict):
                if 0 in output_qparams:
                    return output_qparams[0]
                if len(output_qparams) == 1:
                    return next(iter(output_qparams.values()))
            return output_qparams

        input_qparams = current.meta.get("input_qparams")
        if input_qparams:
            if isinstance(input_qparams, dict):
                if 0 in input_qparams:
                    return input_qparams[0]
                if len(input_qparams) == 1:
                    return next(iter(input_qparams.values()))
            return input_qparams

        if _is_tosa_rescale_node(current) and len(getattr(current, "args", ())) >= 5:
            source_node = current.args[0] if getattr(current, "args", None) else None
            source_qparam = (
                _extract_effective_node_qparam(source_node)
                if hasattr(source_node, "op")
                else None
            )
            if (
                source_qparam is not None
                and not source_qparam.per_channel
                and isinstance(current.args[1], torch.dtype)
                and isinstance(current.args[4], int)
            ):
                multiplier_arg = current.args[2]
                if (
                    isinstance(multiplier_arg, (list, tuple))
                    and len(multiplier_arg) == 1
                    and multiplier_arg[0] not in (0, 0.0)
                ):
                    try:
                        qinfo = torch.iinfo(current.args[1])
                    except TypeError:
                        qinfo = None
                    if qinfo is not None:
                        return QuantArgs(
                            scale=source_qparam.get_scale_per_tensor()
                            / float(multiplier_arg[0]),
                            zp=int(current.args[4]),
                            qmin=int(qinfo.min),
                            qmax=int(qinfo.max),
                            dtype=current.args[1],
                        )

        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return None


def _regularize_control_flow_submodule(submodule, submodule_node) -> None:
    """Mirror the Python Arm backend's submodule metadata normalization.

    Control-flow submodules need their placeholder/output fake tensors aligned
    with the parent cond/while node contract before the Rust passes run.
    """
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    submodule_inputs = []
    for node in submodule.graph.nodes:
        if node.op == "placeholder":
            node.meta["is_input"] = True
            # Control-flow branch placeholders are fed by the parent graph,
            # not by the external program boundary. The Rust memory-format
            # pass must not treat them as fresh NCHW inputs and insert another
            # input transpose.
            node.meta["arm_control_flow_placeholder"] = True
            submodule_inputs.append(node)

    match submodule_node.target:
        case torch.ops.higher_order.cond:
            args = cast(list, submodule_node.args[-1])
        case torch.ops.higher_order.while_loop:
            args = cast(list, submodule_node.args[-2]) + cast(list, submodule_node.args[-1])
        case _:
            raise RuntimeError(
                f"Unexpected control flow target: {submodule_node.target}"
            )

    for submodule_input, submodule_arg in zip(submodule_inputs, args, strict=True):
        submodule_input.meta["val"] = _get_matching_fake_tensor(submodule_arg)
        if submodule_arg.meta.get("tosa_dim_order") is not None:
            submodule_input.meta["tosa_dim_order"] = tuple(
                submodule_arg.meta["tosa_dim_order"]
            )
        output_qparams = submodule_input.meta.get("output_qparams")
        if isinstance(output_qparams, dict) and 0 in output_qparams:
            submodule_input.meta["arm_control_flow_saved_output_qparams"] = {
                0: output_qparams[0]
            }

    output_node = submodule.graph.output_node()
    if hasattr(output_node.args[0], "op"):
        output_node.update_arg(0, [output_node.args[0]])
    output_args = cast(list, output_node.args[0])
    output_dequant_qparams = {}

    for idx, submodule_output in enumerate(output_args):
        if not _is_dequantize_node(submodule_output):
            continue
        try:
            output_dequant_qparams[idx] = QuantArgs.from_operator(
                submodule_output.target, submodule_output.args
            )
        except (NotImplementedError, IndexError, TypeError, ValueError):
            continue

    if output_dequant_qparams:
        output_node.meta["arm_control_flow_output_dequant_qparams"] = (
            output_dequant_qparams
        )
    output_node.meta["arm_control_flow_output_fake_tensors"] = {
        idx: _get_matching_fake_tensor(submodule_user)
        for idx, submodule_user in enumerate(_ordered_control_flow_users(submodule_node))
    }

    for submodule_output, submodule_user in zip(
        output_args, _ordered_control_flow_users(submodule_node)
    ):
        submodule_output.meta["val"] = _get_matching_fake_tensor(submodule_user)
        if submodule_user.meta.get("tosa_dim_order") is not None:
            submodule_output.meta["tosa_dim_order"] = tuple(
                submodule_user.meta["tosa_dim_order"]
            )


def _regularize_control_flow_submodules(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    regularized = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        _regularize_control_flow_submodule(submodule, control_flow_node)
        regularized += 1

    if regularized:
        logger.info(
            "[fast_preprocess] regularized %d control-flow submodule(s)",
            regularized,
        )


def _prepare_control_flow_qparams(graph_module, exported_program) -> None:
    import torch

    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        FoldAndAnnotateQParamsPass,
    )
    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertControlFlowRescalesPass,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _topologically_rebuild_graph_module(module) -> None:
        new_graph = torch.fx.Graph()
        env = {}

        for old_node in module.graph.nodes:
            if old_node.op not in ("placeholder", "get_attr"):
                continue
            new_node = new_graph.node_copy(old_node, lambda old: env[old])
            new_node.meta = dict(old_node.meta)
            env[old_node] = new_node

        def _copy_node(old_node):
            if old_node in env:
                return env[old_node]
            for dep in old_node.all_input_nodes:
                _copy_node(dep)
            new_node = new_graph.node_copy(old_node, lambda old: env[old])
            new_node.meta = dict(old_node.meta)
            env[old_node] = new_node
            return new_node

        for old_node in module.graph.nodes:
            if old_node.op == "output":
                _copy_node(old_node)

        module.graph = new_graph
        module.recompile()

    def _extract_boundary_qparam(node):
        current = node
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))

            output_qparams = current.meta.get("output_qparams")
            if output_qparams:
                if isinstance(output_qparams, dict):
                    if 0 in output_qparams:
                        return output_qparams[0]
                    if len(output_qparams) == 1:
                        return next(iter(output_qparams.values()))
                return output_qparams

            input_qparams = current.meta.get("input_qparams")
            if input_qparams:
                if isinstance(input_qparams, dict):
                    if 0 in input_qparams:
                        return input_qparams[0]
                    if len(input_qparams) == 1:
                        return next(iter(input_qparams.values()))
                return input_qparams

            try:
                return QuantArgs.from_operator(current.target, current.args)
            except Exception:
                pass

            next_arg = None
            if getattr(current, "args", None):
                candidate = current.args[0]
                if hasattr(candidate, "op"):
                    next_arg = candidate
            if next_arg is None:
                break
            current = next_arg
        return None

    def _synthesize_control_flow_input_qparams(node) -> None:
        if node.target == torch.ops.higher_order.cond:
            input_index = 3
        elif node.target == torch.ops.higher_order.while_loop:
            input_index = 2
        else:
            return

        if len(node.args) <= input_index:
            return
        input_list = node.args[input_index]
        if not isinstance(input_list, (list, tuple)):
            return

        detailed_qparams = {}
        for idx, candidate in enumerate(input_list):
            qparam = _extract_boundary_qparam(candidate)
            _log_control_flow_debug(
                "[fast_preprocess] synth control-flow input qparams %s idx=%d candidate=%s qparam=%s",
                getattr(node, "name", "?"),
                idx,
                _describe_node_brief(candidate) if hasattr(candidate, "meta") else repr(candidate),
                qparam,
            )
            if qparam is None:
                logger.info(
                    "[fast_preprocess] control-flow input candidate type=%s value=%s",
                    type(candidate),
                    candidate,
                )
                if hasattr(candidate, "meta"):
                    logger.info(
                        "[fast_preprocess] control-flow input candidate meta input_qparams=%s output_qparams=%s",
                        candidate.meta.get("input_qparams"),
                        candidate.meta.get("output_qparams"),
                    )
                    _log_node_chain(candidate, limit=6)
                continue
            detailed_qparams[idx] = qparam

        if not detailed_qparams:
            logger.info(
                "[fast_preprocess] failed to synthesize control-flow input_qparams for %s",
                getattr(node, "name", "?"),
            )
            return

        node.meta["arm_control_flow_input_qparams_map"] = detailed_qparams
        if len(input_list) == 1:
            node.meta["input_qparams"] = {input_index: detailed_qparams[0]}
        logger.info(
            "[fast_preprocess] synthesized control-flow input_qparams for %s: %s",
            getattr(node, "name", "?"),
            detailed_qparams,
        )

    def _log_control_flow_qparams(node, owner_graph_module) -> None:
        logger.info(
            "[fast_preprocess] control-flow node %s input_qparams=%s output_qparams=%s",
            getattr(node, "name", "?"),
            node.meta.get("input_qparams"),
            node.meta.get("output_qparams"),
        )
        try:
            input_index = None
            if node.target == torch.ops.higher_order.cond:
                input_index = 3
            elif node.target == torch.ops.higher_order.while_loop:
                input_index = 2
            if input_index is not None and len(node.args) > input_index:
                inputs = node.args[input_index]
                logger.info(
                    "[fast_preprocess] control-flow node %s raw input list=%s",
                    getattr(node, "name", "?"),
                    inputs,
                )
                if isinstance(inputs, (list, tuple)):
                    for idx, inp in enumerate(inputs):
                        logger.info(
                            "[fast_preprocess] control-flow input[%d] type=%s value=%s",
                            idx,
                            type(inp),
                            inp,
                        )
                        if hasattr(inp, "meta"):
                            logger.info(
                                "[fast_preprocess] control-flow input[%d] meta input_qparams=%s output_qparams=%s",
                                idx,
                                inp.meta.get("input_qparams"),
                                inp.meta.get("output_qparams"),
                            )
        except Exception as exc:
            logger.info("[fast_preprocess] control-flow input logging failed: %s", exc)
        try:
            if node.target == torch.ops.higher_order.cond:
                submods = [owner_graph_module.get_submodule(node.args[1].target), owner_graph_module.get_submodule(node.args[2].target)]
            elif node.target == torch.ops.higher_order.while_loop:
                submods = [owner_graph_module.get_submodule(node.args[0].target), owner_graph_module.get_submodule(node.args[1].target)]
            else:
                submods = []
            for submod in submods:
                for subnode in submod.graph.nodes:
                    if subnode.op == "placeholder":
                        logger.info(
                            "[fast_preprocess] submodule input %s output_qparams=%s input_qparams=%s",
                            getattr(subnode, "name", "?"),
                            subnode.meta.get("output_qparams"),
                            subnode.meta.get("input_qparams"),
                        )
                        break
                out = submod.graph.output_node()
                logger.info(
                    "[fast_preprocess] submodule output meta input_qparams=%s output_qparams=%s",
                    out.meta.get("input_qparams"),
                    out.meta.get("output_qparams"),
                )
        except Exception as exc:
            logger.info("[fast_preprocess] control-flow qparam logging failed: %s", exc)

    def _snapshot_submodule_input_qparams(owner_graph_module, node) -> None:
        try:
            if node.target == torch.ops.higher_order.cond:
                submodules = [
                    owner_graph_module.get_submodule(node.args[1].target),
                    owner_graph_module.get_submodule(node.args[2].target),
                ]
            elif node.target == torch.ops.higher_order.while_loop:
                submodules = [
                    owner_graph_module.get_submodule(node.args[0].target),
                    owner_graph_module.get_submodule(node.args[1].target),
                ]
            else:
                submodules = []
        except Exception as exc:
            logger.info(
                "[fast_preprocess] failed to snapshot control-flow submodule qparams: %s",
                exc,
            )
            return

        for submodule in submodules:
            for subnode in submodule.graph.nodes:
                if subnode.op != "placeholder":
                    continue
                output_qparams = subnode.meta.get("output_qparams")
                if isinstance(output_qparams, dict) and 0 in output_qparams:
                    subnode.meta["arm_control_flow_saved_output_qparams"] = {
                        0: output_qparams[0]
                    }
                    logger.info(
                        "[fast_preprocess] snapshotted control-flow placeholder qparams %s -> %s",
                        getattr(subnode, "name", "?"),
                        output_qparams[0],
                    )

    fold_pass = FoldAndAnnotateQParamsPass(exported_program)
    rescale_pass = InsertControlFlowRescalesPass()
    prepared = 0
    modified = False

    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.higher_order.cond:
            fold_pass._handle_control_flow_node(node, graph_module)
            _snapshot_submodule_input_qparams(graph_module, node)
            _synthesize_control_flow_input_qparams(node)
            _log_control_flow_qparams(node, graph_module)
            modified |= rescale_pass._rescale_cond_submodules(node, graph_module)
            prepared += 1
        elif node.target == torch.ops.higher_order.while_loop:
            fold_pass._handle_control_flow_node(node, graph_module)
            _snapshot_submodule_input_qparams(graph_module, node)
            _synthesize_control_flow_input_qparams(node)
            _log_control_flow_qparams(node, graph_module)
            modified |= rescale_pass._rescale_while_submodules(node, graph_module)
            prepared += 1

    if not prepared:
        return

    graph_module.graph.eliminate_dead_code()
    _topologically_rebuild_graph_module(graph_module)
    graph_module.graph.lint()
    graph_module.recompile()

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        submodule.graph.eliminate_dead_code()
        _topologically_rebuild_graph_module(submodule)
        submodule.graph.lint()
        submodule.recompile()

    logger.info(
        "[fast_preprocess] prepared %d control-flow node(s) with Python qparam/rescale helpers (modified=%s)",
        prepared,
        modified,
    )


def _reapply_control_flow_table_ops(graph_module, exported_program) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_table_ops import (
        TableOps,
    )
    import fx_helpers

    table_targets = list(TableOps.included_ops())
    repaired = 0

    def _table_spec(node):
        target_str = str(node.target)
        scalar_arg = None

        kind_map = [
            ("ceil.default", "ceil"),
            ("floor.default", "floor"),
            ("rsqrt.default", "rsqrt"),
            ("reciprocal.default", "reciprocal"),
            ("exp.default", "exp"),
            ("expm1.default", "expm1"),
            ("log.default", "log"),
            ("log1p.default", "log1p"),
            ("hardsigmoid.default", "hardsigmoid"),
            ("sigmoid.default", "sigmoid"),
            ("atanh.default", "atanh"),
            ("asinh.default", "asinh"),
            ("acosh.default", "acosh"),
            ("atan.default", "atan"),
            ("asin.default", "asin"),
            ("acos.default", "acos"),
            ("tanh.default", "tanh"),
            ("sin.default", "sin"),
            ("cos.default", "cos"),
            ("tan.default", "tan"),
            ("sinh.default", "sinh"),
            ("cosh.default", "cosh"),
            ("gelu.default", "gelu"),
            ("silu.default", "silu"),
            ("erf.default", "erf"),
            ("hardswish.default", "hardswish"),
        ]
        for suffix, kind in kind_map:
            if suffix in target_str:
                return kind, scalar_arg

        if "pow.Tensor_Scalar" in target_str:
            if len(node.args) > 1 and isinstance(node.args[1], (int, float)):
                scalar_arg = float(node.args[1])
            return "pow", scalar_arg
        return None, None

    def _extract_producer_qparam(value_node):
        current = value_node
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))
            output_qparams = current.meta.get("output_qparams")
            if output_qparams:
                if isinstance(output_qparams, dict):
                    if 0 in output_qparams:
                        return output_qparams[0]
                    if len(output_qparams) == 1:
                        return next(iter(output_qparams.values()))
                return output_qparams

            input_qparams = current.meta.get("input_qparams")
            if input_qparams:
                if isinstance(input_qparams, dict):
                    if 0 in input_qparams:
                        return input_qparams[0]
                    if len(input_qparams) == 1:
                        return next(iter(input_qparams.values()))
                return input_qparams

            if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
                break
            current = current.args[0]
        return None

    def _find_input_qparam_for_user(user, value_node):
        input_qparams = user.meta.get("input_qparams")
        if not input_qparams:
            return None

        matches = []
        for idx, arg in enumerate(getattr(user, "args", ())):
            if arg is value_node and idx in input_qparams:
                matches.append(input_qparams[idx])

        if 0 in input_qparams and len(input_qparams) == 1 and not matches:
            return input_qparams[0]
        if not matches:
            return None

        first = matches[0]
        if any(qp != first for qp in matches[1:]):
            logger.info(
                "[fast_preprocess] conflicting consumer input_qparams for %s via %s: %s",
                getattr(value_node, "name", "?"),
                getattr(user, "name", "?"),
                matches,
            )
            return None
        return first

    def _extract_consumer_qparam(node, seen=None):
        if seen is None:
            seen = set()
        if id(node) in seen:
            return None
        seen.add(id(node))

        found = []
        for user in list(getattr(node, "users", ())):
            qp = _find_input_qparam_for_user(user, node)
            if qp is not None:
                found.append(qp)
                continue

            qp = _extract_consumer_qparam(user, seen)
            if qp is not None:
                found.append(qp)

        if not found:
            return None

        first = found[0]
        if any(qp != first for qp in found[1:]):
            logger.info(
                "[fast_preprocess] conflicting downstream qparams for %s: %s",
                getattr(node, "name", "?"),
                found,
            )
            return None
        return first

    def _synthesize_control_flow_table_qparams(submodule) -> int:
        updated = 0
        for node in submodule.graph.nodes:
            if node.op != "call_function" or node.target not in table_targets:
                continue

            if not node.meta.get("input_qparams"):
                qparam = None
                if getattr(node, "args", None) and hasattr(node.args[0], "op"):
                    qparam = _extract_producer_qparam(node.args[0])
                if qparam is not None:
                    node.meta["input_qparams"] = {0: qparam}
                    updated += 1
                    logger.info(
                        "[fast_preprocess] synthesized table input_qparams for %s: %s",
                        getattr(node, "name", "?"),
                        node.meta["input_qparams"],
                    )

            if not node.meta.get("output_qparams"):
                qparam = _extract_consumer_qparam(node)
                if qparam is not None:
                    node.meta["output_qparams"] = {0: qparam}
                    updated += 1
                    logger.info(
                        "[fast_preprocess] synthesized table output_qparams for %s: %s",
                        getattr(node, "name", "?"),
                        node.meta["output_qparams"],
                    )
        return updated

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        canonicalized = 0
        for node in submodule.graph.nodes:
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            for edge_target in table_targets:
                edge_target_str = str(edge_target)
                if "aten." not in edge_target_str:
                    continue
                suffix = edge_target_str.split("aten.", 1)[1].split(">", 1)[0]
                if f"aten.{suffix}" in target_str and node.target is not edge_target:
                    node.target = edge_target
                    canonicalized += 1
                    break

        needs_table_repair = any(
            node.op == "call_function" and node.target in table_targets
            for node in submodule.graph.nodes
        )
        if not needs_table_repair:
            continue

        synthesized = _synthesize_control_flow_table_qparams(submodule)
        if synthesized:
            logger.info(
                "[fast_preprocess] synthesized %d table qparam attachment(s) in control-flow submodule",
                synthesized,
            )

        modified = False
        for node in list(submodule.graph.nodes):
            if node.op != "call_function" or node.target not in table_targets:
                continue
            if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
                continue
            kind, scalar_arg = _table_spec(node)
            if kind is None:
                continue
            fx_helpers.insert_table_op(submodule, node, kind, scalar_arg)
            modified = True

        if modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()
            repaired += 1

    if repaired:
        logger.info(
            "[fast_preprocess] reapplied table-op lowering on %d control-flow submodule(s)",
            repaired,
        )


def _repair_control_flow_binary_rescales(graph_module, tosa_spec) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertRescaleInt32Pass,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs
    from executorch.backends.arm.tosa.specification import TosaLoweringContext

    def _replace_submodule(root_module, target: str, replacement) -> None:
        atoms = target.split(".")
        parent = root_module
        for atom in atoms[:-1]:
            parent = parent.get_submodule(atom)
        setattr(parent, atoms[-1], replacement)

    def _is_tosa_rescale_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "tosa.RESCALE" in target_str or "tosa::rescale" in target_str

    def _promote_binary_input(arg, sibling):
        candidate = sibling
        seen = set()
        while hasattr(candidate, "op") and id(candidate) not in seen:
            seen.add(id(candidate))
            if _is_tosa_rescale_node(candidate):
                return candidate
            next_arg = None
            for maybe_arg in getattr(candidate, "args", ()):
                if hasattr(maybe_arg, "op"):
                    next_arg = maybe_arg
                    break
            candidate = next_arg
        return None

    def _extract_qparam(node):
        def _consumer_qparam(value_node):
            for user in list(getattr(value_node, "users", ())):
                input_qparams = user.meta.get("input_qparams") or {}
                for idx, arg in enumerate(getattr(user, "args", ())):
                    if arg is value_node and idx in input_qparams:
                        return input_qparams[idx]
                if 0 in input_qparams and len(input_qparams) == 1:
                    return input_qparams[0]
            return None

        current = node
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))
            output_qparams = current.meta.get("output_qparams")
            if output_qparams:
                if isinstance(output_qparams, dict):
                    if 0 in output_qparams:
                        return output_qparams[0]
                    if len(output_qparams) == 1:
                        return next(iter(output_qparams.values()))
                return output_qparams

            if _is_tosa_rescale_node(current) and len(getattr(current, "args", ())) >= 5:
                source_node = current.args[0] if getattr(current, "args", None) else None
                source_qparam = (
                    _extract_qparam(source_node) if hasattr(source_node, "op") else None
                )
                if (
                    source_qparam is not None
                    and not source_qparam.per_channel
                    and isinstance(current.args[1], torch.dtype)
                    and isinstance(current.args[4], int)
                ):
                    multiplier_arg = current.args[2]
                    if (
                        isinstance(multiplier_arg, (list, tuple))
                        and len(multiplier_arg) == 1
                        and multiplier_arg[0] not in (0, 0.0)
                    ):
                        try:
                            qinfo = torch.iinfo(current.args[1])
                        except TypeError:
                            qinfo = None
                        if qinfo is not None:
                            return QuantArgs(
                                scale=source_qparam.get_scale_per_tensor()
                                / float(multiplier_arg[0]),
                                zp=int(current.args[4]),
                                qmin=int(qinfo.min),
                                qmax=int(qinfo.max),
                                dtype=current.args[1],
                            )

            try:
                return QuantArgs.from_operator(current.target, current.args)
            except Exception:
                pass

            target_str = str(getattr(current, "target", ""))
            if "dequantize_per_tensor" in target_str and len(getattr(current, "args", ())) >= 6:
                try:
                    return QuantArgs(
                        scale=float(current.args[1]),
                        zp=int(current.args[2]),
                        qmin=int(current.args[3]),
                        qmax=int(current.args[4]),
                        dtype=current.args[5],
                    )
                except Exception:
                    pass

            input_qparams = current.meta.get("input_qparams")
            if input_qparams:
                if isinstance(input_qparams, dict):
                    if 0 in input_qparams:
                        return input_qparams[0]
                    if len(input_qparams) == 1:
                        return next(iter(input_qparams.values()))
                return input_qparams

            consumer_qparam = _consumer_qparam(current)
            if consumer_qparam is not None:
                return consumer_qparam

            if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
                break
            current = current.args[0]
        return None

    def _reconstruct_rescale_source_qparam(binary_node, arg_idx, rescale_node):
        if not _is_tosa_rescale_node(rescale_node):
            return None
        if len(getattr(rescale_node, "args", ())) < 5:
            return None
        source_node = rescale_node.args[0]
        if not hasattr(source_node, "op"):
            return None
        if source_node.meta.get("output_qparams"):
            return _extract_qparam(source_node)

        binary_input_qparams = binary_node.meta.get("input_qparams") or {}
        output_qparam = binary_input_qparams.get(arg_idx)
        if output_qparam is None:
            output_qparam = _extract_qparam(rescale_node)
        if output_qparam is None:
            return None

        scale_arg = rescale_node.args[2]
        if not (
            isinstance(scale_arg, (list, tuple))
            and len(scale_arg) == 1
            and scale_arg[0] not in (0, 0.0)
        ):
            return None

        source_dtype = getattr(source_node.meta.get("val"), "dtype", None)
        if source_dtype not in (torch.int8, torch.int16):
            return None

        try:
            qinfo = torch.iinfo(source_dtype)
        except TypeError:
            return None

        try:
            source_qparam = QuantArgs(
                scale=float(scale_arg[0]) * output_qparam.get_scale_per_tensor(),
                zp=int(rescale_node.args[3]),
                qmin=int(qinfo.min),
                qmax=int(qinfo.max),
                dtype=source_dtype,
            )
        except Exception:
            return None

        source_node.meta["output_qparams"] = {0: source_qparam}
        source_node.meta["input_qparams"] = {0: source_qparam}
        logger.info(
            "[fast_preprocess] reconstructed control-flow binary source qparams %s arg%d -> %s",
            getattr(binary_node, "name", "?"),
            arg_idx,
            source_qparam,
        )
        return source_qparam

    def _reconstruct_binary_output_qparam(binary_node, rescale_node):
        if not _is_tosa_rescale_node(rescale_node):
            return None
        if len(getattr(rescale_node, "args", ())) < 5:
            return None

        input_qparams = binary_node.meta.get("output_qparams") or {}
        source_qparam = input_qparams.get(0)
        if source_qparam is None:
            source_qparam = _extract_qparam(binary_node)
        if source_qparam is None:
            return None

        scale_arg = rescale_node.args[2]
        if not (
            isinstance(scale_arg, (list, tuple))
            and len(scale_arg) == 1
            and scale_arg[0] not in (0, 0.0)
        ):
            return None

        output_dtype = rescale_node.args[1]
        if output_dtype not in (torch.int8, torch.int16):
            return None

        try:
            qinfo = torch.iinfo(output_dtype)
            output_qparam = QuantArgs(
                scale=source_qparam.get_scale_per_tensor() / float(scale_arg[0]),
                zp=int(rescale_node.args[4]),
                qmin=int(qinfo.min),
                qmax=int(qinfo.max),
                dtype=output_dtype,
            )
        except Exception:
            return None

        binary_node.meta["output_qparams"] = {0: output_qparam}
        logger.info(
            "[fast_preprocess] reconstructed control-flow binary output qparams %s -> %s",
            getattr(binary_node, "name", "?"),
            output_qparam,
        )
        return output_qparam

    repaired = 0
    synthesized = 0
    for name, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        updated_graph = False
        control_flow_args = []
        match control_flow_node.target:
            case torch.ops.higher_order.cond:
                control_flow_args = list(control_flow_node.args[-1])
            case torch.ops.higher_order.while_loop:
                control_flow_args = list(control_flow_node.args[-2]) + list(
                    control_flow_node.args[-1]
                )
            case _:
                control_flow_args = []

        placeholders = [node for node in submodule.graph.nodes if node.op == "placeholder"]
        placeholder_qparams = dict(
            control_flow_node.meta.get("arm_control_flow_input_qparams_map", {})
        )
        placeholder_index_by_name = {
            getattr(node, "name", f"placeholder_{idx}"): idx
            for idx, node in enumerate(placeholders)
        }

        def _lookup_control_flow_input_qparam(arg):
            current = arg
            seen = set()
            while hasattr(current, "op") and id(current) not in seen:
                seen.add(id(current))
                idx = placeholder_index_by_name.get(getattr(current, "name", ""))
                if idx is not None and idx in placeholder_qparams:
                    return placeholder_qparams[idx]
                if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
                    break
                current = current.args[0]
            return None

        for idx, placeholder in enumerate(placeholders):
            if idx >= len(control_flow_args):
                continue
            if placeholder.meta.get("output_qparams"):
                continue
            parent_qparam = _extract_qparam(control_flow_args[idx])
            if parent_qparam is None:
                continue
            placeholder.meta["output_qparams"] = {0: parent_qparam}
            placeholder.meta["input_qparams"] = {0: parent_qparam}
            updated_graph = True
            logger.info(
                "[fast_preprocess] restored control-flow placeholder qparams %s idx=%d -> %s",
                getattr(placeholder, "name", "?"),
                idx,
                parent_qparam,
            )

        for node in list(submodule.graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op"):
                    continue
                if _extract_qparam(arg) is not None:
                    continue
                sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                if not hasattr(sibling, "op"):
                    continue
                promoted = None
                sibling_input = None
                if getattr(sibling, "args", None) and hasattr(sibling.args[0], "op"):
                    sibling_input = sibling.args[0]
                if sibling_input is not None and _extract_qparam(sibling_input) is not None:
                    promoted = sibling_input
                else:
                    promoted = _promote_binary_input(arg, sibling)
                if promoted is None or promoted is arg:
                    continue
                node.replace_input_with(arg, promoted)
                promoted.meta["arm_promoted_control_flow_binary_input"] = True
                updated_graph = True
                logger.info(
                    "[fast_preprocess] promoted control-flow binary input %s arg%d -> %s",
                    getattr(node, "name", "?"),
                    idx,
                    getattr(promoted, "name", "?"),
                )

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op") or not _is_tosa_rescale_node(arg):
                    continue
                if arg.meta.get("arm_promoted_control_flow_binary_input"):
                    continue
                source = arg.args[0] if getattr(arg, "args", None) else None
                if not hasattr(source, "op"):
                    continue
                _reconstruct_rescale_source_qparam(node, idx, arg)
                node.replace_input_with(arg, source)
                updated_graph = True
                logger.info(
                    "[fast_preprocess] stripped pre-existing control-flow binary input rescale %s from %s arg%d",
                    getattr(arg, "name", "?"),
                    getattr(node, "name", "?"),
                    idx,
                )

            for user in list(node.users):
                if not _is_tosa_rescale_node(user):
                    continue
                _reconstruct_binary_output_qparam(node, user)
                for grand_user in list(user.users):
                    grand_user.replace_input_with(user, node)
                updated_graph = True
                logger.info(
                    "[fast_preprocess] stripped pre-existing control-flow binary output rescale %s from %s",
                    getattr(user, "name", "?"),
                    getattr(node, "name", "?"),
                )

            input_qparams = {}

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op"):
                    continue

                producer_qparam = _extract_qparam(arg)
                if producer_qparam is None:
                    producer_qparam = _lookup_control_flow_input_qparam(arg)
                    if producer_qparam is not None:
                        logger.info(
                            "[fast_preprocess] recovered control-flow input qparams for %s arg%d from parent boundary: %s",
                            getattr(node, "name", "?"),
                            idx,
                            producer_qparam,
                        )
                sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                sibling_qparam = _extract_qparam(sibling) if hasattr(sibling, "op") else None
                if (
                    producer_qparam is not None
                    and getattr(producer_qparam, "dtype", None) == torch.int32
                    and sibling_qparam is not None
                ):
                    arg_input_qparams = arg.meta.get("input_qparams") or {}
                    if isinstance(arg_input_qparams, dict):
                        preferred_qparam = None
                        for candidate in arg_input_qparams.values():
                            if getattr(candidate, "dtype", None) != sibling_qparam.dtype:
                                continue
                            if (
                                float(candidate.scale) == float(sibling_qparam.scale)
                                and int(candidate.zp) == int(sibling_qparam.zp)
                            ):
                                preferred_qparam = candidate
                                break
                            if preferred_qparam is None:
                                preferred_qparam = candidate
                        if preferred_qparam is not None:
                            producer_qparam = preferred_qparam
                            logger.info(
                                "[fast_preprocess] preferred binary operand qparams for %s arg%d from nested input qparams: %s",
                                getattr(node, "name", "?"),
                                idx,
                                preferred_qparam,
                            )
                if producer_qparam is None:
                    if hasattr(sibling, "op"):
                        sibling_input_qparams = sibling.meta.get("input_qparams")
                        if sibling_input_qparams:
                            if isinstance(sibling_input_qparams, dict):
                                producer_qparam = sibling_input_qparams.get(0)
                                if producer_qparam is None and len(sibling_input_qparams) == 1:
                                    producer_qparam = next(
                                        iter(sibling_input_qparams.values())
                                    )
                            else:
                                producer_qparam = sibling_input_qparams
                    if (
                        producer_qparam is None
                        and hasattr(sibling, "op")
                        and getattr(sibling, "args", None)
                        and hasattr(sibling.args[0], "op")
                    ):
                        producer_qparam = _extract_qparam(sibling.args[0])
                if producer_qparam is None:
                    continue

                input_qparams[idx] = producer_qparam
                synthesized += 1
                updated_graph = True
                logger.info(
                    "[fast_preprocess] synthesized control-flow binary input_qparams for %s arg%d: %s",
                    getattr(node, "name", "?"),
                    idx,
                    producer_qparam,
                )

            if len(input_qparams) == 1:
                missing_idx = 0 if 0 not in input_qparams else 1
                present_idx = 1 - missing_idx
                missing_arg = node.args[missing_idx] if len(node.args) > missing_idx else None
                present_arg = node.args[present_idx] if len(node.args) > present_idx else None
                present_qparam = input_qparams.get(present_idx)
                missing_dtype = getattr(
                    getattr(missing_arg, "meta", {}).get("val"), "dtype", None
                )
                present_dtype = getattr(
                    getattr(present_arg, "meta", {}).get("val"), "dtype", None
                )
                if (
                    present_qparam is not None
                    and missing_dtype is not None
                    and missing_dtype == present_dtype == present_qparam.dtype
                ):
                    input_qparams[missing_idx] = present_qparam
                    updated_graph = True
                    logger.info(
                        "[fast_preprocess] mirrored control-flow binary input_qparams for %s arg%d from arg%d: %s",
                        getattr(node, "name", "?"),
                        missing_idx,
                        present_idx,
                        present_qparam,
                    )

            node.meta["input_qparams"] = input_qparams
            if len(input_qparams) != 2:
                _log_control_flow_debug(
                    "[fast_preprocess] incomplete control-flow binary qparams for %s %s",
                    _describe_node_brief(node),
                    {
                        idx: _describe_node_brief(arg)
                        for idx, arg in enumerate(node.args[:2])
                        if hasattr(arg, "op")
                    },
                )

        if updated_graph:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

        for node in list(submodule.graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue
            _log_control_flow_debug(
                "[fast_preprocess] pre-InsertRescaleInt32Pass binary state %s",
                _describe_node_brief(node),
            )
            for idx, arg in enumerate(node.args[:2]):
                if hasattr(arg, "op"):
                    _log_control_flow_debug(
                        "[fast_preprocess] pre-InsertRescaleInt32Pass arg%d %s",
                        idx,
                        _describe_node_brief(arg),
                    )
                    _log_control_flow_debug(
                        "[fast_preprocess] pre-InsertRescaleInt32Pass arg%d_raw args=%s kwargs=%s",
                        idx,
                        repr(getattr(arg, "args", None)),
                        repr(getattr(arg, "kwargs", None)),
                    )

        with TosaLoweringContext(tosa_spec):
            result = InsertRescaleInt32Pass().call(submodule)
        if result.modified:
            repaired += 1
            _replace_submodule(graph_module, name, result.graph_module)

    if synthesized:
        logger.info(
            "[fast_preprocess] synthesized %d control-flow binary operand qparam attachment(s)",
            synthesized,
        )
    if repaired:
        logger.info(
            "[fast_preprocess] reapplied InsertRescaleInt32Pass on %d control-flow submodule(s)",
            repaired,
        )


def _canonicalize_control_flow_pool_binary_rescales(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _topologically_rebuild_graph_module(module) -> None:
        new_graph = torch.fx.Graph()
        env = {}

        for old_node in module.graph.nodes:
            if old_node.op not in ("placeholder", "get_attr"):
                continue
            new_node = new_graph.node_copy(old_node, lambda old: env[old])
            new_node.meta = dict(old_node.meta)
            env[old_node] = new_node

        def _copy_node(old_node):
            if old_node in env:
                return env[old_node]
            for dep in old_node.all_input_nodes:
                _copy_node(dep)
            new_node = new_graph.node_copy(old_node, lambda old: env[old])
            new_node.meta = dict(old_node.meta)
            env[old_node] = new_node
            return new_node

        for old_node in module.graph.nodes:
            if old_node.op == "output":
                _copy_node(old_node)

        module.graph = new_graph
        module.recompile()

    def _is_control_flow_quant_input_node(node) -> bool:
        if _is_dequantize_node(node):
            return True
        if getattr(node, "op", None) != "placeholder":
            return False
        if not node.meta.get("is_input"):
            return False
        return (
            node.meta.get("arm_control_flow_placeholder", False)
            or "dequantize_per_tensor" in getattr(node, "name", "")
        )

    def _is_avg_pool_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "AVG_POOL2D" in target_str or "avg_pool2d" in target_str

    def _is_binary_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "aten.add.Tensor" in target_str or "aten.sub.Tensor" in target_str

    def _is_reshape_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "RESHAPE" in target_str or "view_copy" in target_str

    def _downstream_non_reshape_users(node):
        stack = list(getattr(node, "users", ()))
        seen = set()
        while stack:
            user = stack.pop()
            if id(user) in seen:
                continue
            seen.add(id(user))
            if _is_reshape_node(user):
                stack.extend(list(getattr(user, "users", ())))
                continue
            yield user

    canonicalized = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        updated = False
        for node in list(submodule.graph.nodes):
            if not _is_control_flow_quant_input_node(node):
                continue

            rescale_users = [
                user
                for user in list(getattr(node, "users", ()))
                if _is_tosa_rescale_node(user)
            ]
            if len(rescale_users) != 3:
                continue

            avg_pool_source = None
            binary_sources = []
            for rescale in rescale_users:
                downstream = list(_downstream_non_reshape_users(rescale))
                if any(_is_avg_pool_node(user) for user in downstream):
                    avg_pool_source = rescale
                if any(_is_binary_node(user) for user in downstream):
                    binary_sources.append(rescale)

            if avg_pool_source is None or len(binary_sources) != 2:
                continue

            for binary_source in binary_sources:
                if binary_source is avg_pool_source:
                    continue
                if not getattr(binary_source, "args", None) or binary_source.args[0] is not node:
                    continue
                binary_source.update_arg(0, avg_pool_source)
                updated = True
                canonicalized += 1
                logger.info(
                    "[fast_preprocess] rewired control-flow binary rescale %s to share avg-pool source %s",
                    getattr(binary_source, "name", "?"),
                    getattr(avg_pool_source, "name", "?"),
                )

        if updated:
            _topologically_rebuild_graph_module(submodule)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if canonicalized:
        logger.info(
            "[fast_preprocess] canonicalized %d control-flow pool/binary rescale edge(s)",
            canonicalized,
        )


def _refresh_control_flow_binary_input_rescales(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    refreshed = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        updated = False
        for node in list(submodule.graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op") or not _is_tosa_rescale_node(arg):
                    continue
                if len(getattr(arg, "args", ())) < 5 or arg.args[1] != torch.int32:
                    continue
                source_node = arg.args[0]
                if not hasattr(source_node, "op"):
                    continue

                source_qparam = _extract_effective_node_qparam(source_node)
                if source_qparam is None:
                    continue
                if source_qparam.dtype not in (torch.int8, torch.int16):
                    continue
                desired_input_zp = source_qparam.get_zp_per_tensor()
                current_input_zp = arg.args[3] if len(arg.args) > 3 else None

                if current_input_zp == desired_input_zp:
                    continue

                arg.update_arg(3, desired_input_zp)
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] refreshed control-flow binary input rescale %s arg%d -> in_zp=%s",
                    getattr(node, "name", "?"),
                    idx,
                    desired_input_zp,
                )

        if updated:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if refreshed:
        logger.info(
            "[fast_preprocess] refreshed %d control-flow binary input rescale(s)",
            refreshed,
        )


def _reapply_control_flow_boundary_rescales(
    graph_module, *, include_inputs: bool = True, include_outputs: bool = True
) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertControlFlowRescalesPass,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs

    def _restore_submodule_input_qparams(
        submodule,
        input_qparams_map,
    ) -> None:
        if not input_qparams_map:
            return
        placeholders = [node for node in submodule.graph.nodes if node.op == "placeholder"]
        for idx, qparam in input_qparams_map.items():
            if idx >= len(placeholders):
                continue
            placeholder = placeholders[idx]
            output_qparams = placeholder.meta.get("output_qparams")
            saved_output_qparams = placeholder.meta.get(
                "arm_control_flow_saved_output_qparams"
            )
            _log_control_flow_debug(
                "[fast_preprocess] restore submodule input qparams %s %s idx=%d incoming=%s output_qparams=%s saved_output_qparams=%s",
                getattr(submodule, "__class__", type(submodule)).__name__,
                getattr(placeholder, "name", "?"),
                idx,
                qparam,
                output_qparams,
                saved_output_qparams,
            )
            effective_qparam = qparam
            if isinstance(saved_output_qparams, dict) and 0 in saved_output_qparams:
                placeholder.meta["output_qparams"] = {0: saved_output_qparams[0]}
                effective_qparam = saved_output_qparams[0]
            elif isinstance(output_qparams, dict) and 0 in output_qparams:
                effective_qparam = output_qparams[0]
            elif not output_qparams:
                placeholder.meta["output_qparams"] = {0: qparam}
            placeholder.meta["input_qparams"] = {0: qparam}
            fake_val = placeholder.meta.get("val")
            if (
                hasattr(fake_val, "to")
                and getattr(fake_val, "dtype", None) != effective_qparam.dtype
            ):
                placeholder.meta["val"] = fake_val.to(dtype=effective_qparam.dtype)

    def _annotate_submodule_input_rescales(submodule, input_qparams_map) -> None:
        if not input_qparams_map:
            return
        placeholders = [node for node in submodule.graph.nodes if node.op == "placeholder"]
        for idx, input_qparam in input_qparams_map.items():
            if idx >= len(placeholders):
                continue
            placeholder = placeholders[idx]
            saved_output_qparams = placeholder.meta.get(
                "arm_control_flow_saved_output_qparams"
            )
            output_qparams = placeholder.meta.get("output_qparams")
            effective_output_qparam = None
            if isinstance(saved_output_qparams, dict) and 0 in saved_output_qparams:
                effective_output_qparam = saved_output_qparams[0]
            elif isinstance(output_qparams, dict) and 0 in output_qparams:
                effective_output_qparam = output_qparams[0]
            if effective_output_qparam is None:
                continue

            for user in list(getattr(placeholder, "users", ())):
                if not _is_tosa_rescale_node(user):
                    continue
                user.meta["input_qparams"] = {0: input_qparam}
                user.meta["output_qparams"] = {0: effective_output_qparam}
                logger.info(
                    "[fast_preprocess] annotated control-flow input rescale %s idx=%d input=%s output=%s",
                    getattr(user, "name", "?"),
                    idx,
                    input_qparam,
                    effective_output_qparam,
                )

    def _infer_control_flow_input_qparams_map(node, input_index: int) -> dict[int, QuantArgs]:
        if len(node.args) <= input_index:
            return {}
        inputs = node.args[input_index]
        if not isinstance(inputs, (list, tuple)):
            return {}

        input_qparams_map: dict[int, QuantArgs] = {}
        for idx, arg in enumerate(inputs):
            if not hasattr(arg, "op"):
                continue
            qparam = _extract_effective_node_qparam(arg)
            if qparam is not None:
                input_qparams_map[idx] = qparam

        return input_qparams_map

    def _refresh_submodule_output_input_qparams(submodule) -> None:
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        input_qparams_map = {}
        for idx, output_arg in enumerate(output_args):
            if not hasattr(output_arg, "op"):
                continue
            qparam = _extract_effective_node_qparam(output_arg)
            if qparam is not None:
                input_qparams_map[idx] = qparam
        if input_qparams_map:
            output_node.meta["input_qparams"] = input_qparams_map

    rescale_pass = InsertControlFlowRescalesPass()
    modified = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.higher_order.cond:
            if_graph = graph_module.get_submodule(node.args[1].target)
            else_graph = graph_module.get_submodule(node.args[2].target)
            input_qparams_map = node.meta.get("arm_control_flow_input_qparams_map", {})
            if not input_qparams_map:
                input_qparams_map = _infer_control_flow_input_qparams_map(node, 3)
            if not input_qparams_map:
                input_qparams_map = rescale_pass._get_input_qparams_map(node, 3)
            _log_control_flow_debug(
                "[fast_preprocess] reapply cond input qparams %s %s",
                getattr(node, "name", "?"),
                input_qparams_map,
            )
            if include_inputs and input_qparams_map:
                _restore_submodule_input_qparams(if_graph, input_qparams_map)
                _restore_submodule_input_qparams(else_graph, input_qparams_map)
                modified |= rescale_pass._rescale_submodule_inputs(
                    if_graph, input_qparams_map
                )
                modified |= rescale_pass._rescale_submodule_inputs(
                    else_graph, input_qparams_map
                )
                _annotate_submodule_input_rescales(if_graph, input_qparams_map)
                _annotate_submodule_input_rescales(else_graph, input_qparams_map)

            output_qparams_map = _infer_control_flow_output_qparams_map(node)
            if include_outputs and output_qparams_map:
                _refresh_submodule_output_input_qparams(if_graph)
                _refresh_submodule_output_input_qparams(else_graph)
                modified |= rescale_pass._rescale_submodule_outputs(
                    if_graph, output_qparams_map
                )
                modified |= rescale_pass._rescale_submodule_outputs(
                    else_graph, output_qparams_map
                )
        elif node.target == torch.ops.higher_order.while_loop:
            cond_graph = graph_module.get_submodule(node.args[0].target)
            body_graph = graph_module.get_submodule(node.args[1].target)
            input_qparams_map = node.meta.get("arm_control_flow_input_qparams_map", {})
            if not input_qparams_map:
                input_qparams_map = _infer_control_flow_input_qparams_map(node, 2)
            if not input_qparams_map:
                input_qparams_map = rescale_pass._get_input_qparams_map(node, 2)
            if include_inputs and input_qparams_map:
                _restore_submodule_input_qparams(cond_graph, input_qparams_map)
                _restore_submodule_input_qparams(body_graph, input_qparams_map)
                modified |= rescale_pass._rescale_submodule_inputs(
                    cond_graph, input_qparams_map
                )
                modified |= rescale_pass._rescale_submodule_inputs(
                    body_graph, input_qparams_map
                )
                _annotate_submodule_input_rescales(cond_graph, input_qparams_map)
                _annotate_submodule_input_rescales(body_graph, input_qparams_map)

            output_qparams_map = _infer_control_flow_output_qparams_map(node)
            if include_outputs and output_qparams_map:
                _refresh_submodule_output_input_qparams(body_graph)
                modified |= rescale_pass._rescale_submodule_outputs(
                    body_graph, output_qparams_map
                )

    if not modified:
        return

    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        submodule.graph.eliminate_dead_code()
        submodule.graph.lint()
        submodule.recompile()

        logger.info(
            "[fast_preprocess] reapplied control-flow boundary rescales after rust passes"
        )


def _prune_duplicate_control_flow_rescales(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _rescale_signature(node):
        if not _is_tosa_rescale_node(node):
            return None
        args = getattr(node, "args", ())
        if len(args) < 5:
            return None
        return tuple(repr(arg) for arg in args[1:5])

    pruned = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        updated = False
        for node in list(submodule.graph.nodes):
            if not _is_tosa_rescale_node(node):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue
            source = node.args[0]
            if not _is_tosa_rescale_node(source):
                continue
            if _rescale_signature(node) != _rescale_signature(source):
                continue

            source.meta["val"] = node.meta.get("val", source.meta.get("val"))
            if node.meta.get("tosa_dim_order") is not None:
                source.meta["tosa_dim_order"] = tuple(node.meta["tosa_dim_order"])
            if node.meta.get("output_qparams"):
                source.meta["output_qparams"] = dict(node.meta["output_qparams"])

            node.replace_all_uses_with(source)
            submodule.graph.erase_node(node)
            pruned += 1
            updated = True
            logger.info(
                "[fast_preprocess] pruned duplicate control-flow rescale %s -> %s",
                getattr(node, "name", "?"),
                getattr(source, "name", "?"),
            )

        if updated:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if pruned:
        logger.info(
            "[fast_preprocess] pruned %d duplicate control-flow rescale node(s)",
            pruned,
        )


def _infer_control_flow_output_qparams_map(control_flow_node):
    from executorch.backends.arm._passes.quant_args import QuantArgs

    output_qparams_map: dict[int, QuantArgs] = {}
    for getitem_node in _ordered_control_flow_users(control_flow_node):
        if len(getattr(getitem_node, "args", ())) < 2:
            continue
        idx = int(getitem_node.args[1])

        qparam_meta = getitem_node.meta.get("output_qparams")
        if qparam_meta and 0 in qparam_meta:
            output_qparams_map[idx] = qparam_meta[0]
            continue

        for user in list(getattr(getitem_node, "users", ())):
            input_qparams = user.meta.get("input_qparams") or {}
            for arg_idx, arg in enumerate(getattr(user, "args", ())):
                if arg is getitem_node and arg_idx in input_qparams:
                    output_qparams_map[idx] = input_qparams[arg_idx]
                    break
            if idx in output_qparams_map:
                break

            target_str = str(getattr(user, "target", ""))
            if "dequantize_per_tensor" in target_str and len(getattr(user, "args", ())) >= 6:
                output_qparams_map[idx] = QuantArgs(
                    scale=float(user.args[1]),
                    zp=int(user.args[2]),
                    qmin=int(user.args[3]),
                    qmax=int(user.args[4]),
                    dtype=user.args[5],
                )
                break

    return output_qparams_map


def _quantize_constant_only_control_flow_outputs(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _quantize_tensor(value: torch.Tensor, qparam) -> torch.Tensor | None:
        if qparam.per_channel:
            return None
        scale = qparam.get_scale_per_tensor()
        if scale == 0:
            return None
        quantized = torch.round(value.detach().to(torch.float32) / scale) + int(qparam.zp)
        quantized = torch.clamp(quantized, int(qparam.qmin), int(qparam.qmax))
        return quantized.to(qparam.dtype)

    rewritten = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        placeholders = [node for node in submodule.graph.nodes if node.op == "placeholder"]
        if placeholders:
            continue

        output_qparams_map = _infer_control_flow_output_qparams_map(control_flow_node)
        if not output_qparams_map:
            continue

        output_node = submodule.graph.output_node()
        output_arg0 = output_node.args[0]
        wrapped_single = hasattr(output_arg0, "op")
        output_args = [output_arg0] if wrapped_single else list(output_arg0)
        updated = False

        for idx, output_arg in enumerate(output_args):
            qparam = output_qparams_map.get(idx)
            if qparam is None:
                continue

            source = output_arg
            if (
                _is_tosa_rescale_node(output_arg)
                and getattr(output_arg, "args", None)
                and hasattr(output_arg.args[0], "op")
                and output_arg.args[0].op == "get_attr"
            ):
                source = output_arg.args[0]

            if getattr(source, "op", None) != "get_attr":
                continue

            constant_value = getattr(submodule, source.target, None)
            if not isinstance(constant_value, torch.Tensor):
                continue
            if not constant_value.is_floating_point():
                continue

            quantized_value = _quantize_tensor(constant_value, qparam)
            if quantized_value is None:
                continue

            setattr(submodule, source.target, quantized_value)
            source.meta["val"] = quantized_value
            source.meta["output_qparams"] = {0: qparam}
            output_args[idx] = source
            updated = True
            rewritten += 1
            logger.info(
                "[fast_preprocess] quantized constant-only control-flow output %s for output index %d",
                getattr(source, "name", "?"),
                idx,
            )

        if updated:
            replacement = output_args[0] if wrapped_single else tuple(output_args)
            output_node.update_arg(0, replacement)
            output_node.meta["input_qparams"] = {
                idx: qparam
                for idx, qparam in output_qparams_map.items()
                if idx < len(output_args)
            }
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] quantized %d constant-only control-flow output(s)",
            rewritten,
        )


def _is_dim_order_copy_node(node) -> bool:
    return (
        getattr(node, "op", None) == "call_function"
        and "_to_dim_order_copy" in str(getattr(node, "target", ""))
    )


def _is_layout_output_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return (
        "_to_dim_order_copy" in target_str
        or "tosa.TRANSPOSE" in target_str
        or "tosa_TRANSPOSE" in target_str
        or "TosaTranspose" in target_str
    )


def _is_tosa_rescale_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return (
        "tosa_RESCALE" in target_str
        or "tosa.RESCALE" in target_str
        or "TosaRescale" in target_str
    )


def _is_dequantize_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    return "dequantize_per_tensor" in str(getattr(node, "target", ""))


def _is_control_flow_input_passthrough_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return "dequantize_per_tensor" in target_str


def _describe_node_brief(node) -> str:
    fake_val = node.meta.get("val")
    dtype = getattr(fake_val, "dtype", None)
    shape = tuple(fake_val.shape) if hasattr(fake_val, "shape") else None
    return (
        f"name={getattr(node, 'name', '?')} "
        f"target={getattr(node, 'target', '?')} "
        f"dtype={dtype} shape={shape} "
        f"tosa_dim_order={node.meta.get('tosa_dim_order')} "
        f"input_qparams={node.meta.get('input_qparams')} "
        f"output_qparams={node.meta.get('output_qparams')}"
    )


def _get_control_flow_output_fake_tensor(submodule_user):
    fake_tensor = _get_matching_fake_tensor(submodule_user)
    output_qparams = submodule_user.meta.get("output_qparams")
    if output_qparams:
        return fake_tensor.to(dtype=output_qparams[0].dtype)

    for user in submodule_user.users:
        user_fake_val = user.meta.get("val")
        if getattr(user_fake_val, "dtype", None) is not None and getattr(
            user_fake_val, "dtype", None
        ) != getattr(fake_tensor, "dtype", None):
            return _get_matching_fake_tensor(user)
    return fake_tensor


def _log_node_chain(node, limit: int = 8) -> None:
    current = node
    depth = 0
    while hasattr(current, "op") and depth < limit:
        logger.info("[fast_preprocess] chain[%d] %s", depth, _describe_node_brief(current))
        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
        depth += 1


def _select_control_flow_output_value(node, submodule_user):
    desired_fake_tensor = _get_control_flow_output_fake_tensor(submodule_user)
    desired_dtype = getattr(desired_fake_tensor, "dtype", None)
    desired_dim_order = submodule_user.meta.get("tosa_dim_order")
    current = node
    candidate = None
    preferred_candidate = None

    while hasattr(current, "op"):
        fake_val = current.meta.get("val")
        current_dtype = getattr(fake_val, "dtype", None)
        current_dim_order = current.meta.get("tosa_dim_order")

        if current_dtype == desired_dtype and (
            desired_dim_order is None or current_dim_order == tuple(desired_dim_order)
        ):
            candidate = current
            target_str = str(getattr(current, "target", ""))
            if "tosa.RESCALE" in target_str or "tosa_RESCALE" in target_str:
                preferred_candidate = current
                break

        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]

    return preferred_candidate or candidate


def _is_trivial_control_flow_output_chain(node) -> bool:
    current = node
    seen = set()
    saw_layout = False
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        if not _is_layout_output_node(current):
            return saw_layout
        saw_layout = True
        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return saw_layout


def _normalize_control_flow_output_layouts(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    rewritten = 0
    for name, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        output_node = submodule.graph.output_node()
        output_arg0 = output_node.args[0]
        wrapped_single = hasattr(output_arg0, "op")
        output_args = [output_arg0] if wrapped_single else list(output_arg0)
        ordered_users = _ordered_control_flow_users(control_flow_node)
        _log_control_flow_debug(
            "[fast_preprocess] normalize outputs name=%s output_count=%d user_count=%d",
            name,
            len(output_args),
            len(ordered_users),
        )
        updated = False

        for idx, submodule_output in enumerate(output_args):
            submodule_user = (
                ordered_users[idx] if idx < len(ordered_users) else control_flow_node
            )
            _log_control_flow_debug(
                "[fast_preprocess] inspect output idx=%d node=%s target=%s user=%s user_target=%s",
                idx,
                getattr(submodule_output, "name", "?"),
                getattr(submodule_output, "target", "?"),
                getattr(submodule_user, "name", "?"),
                getattr(submodule_user, "target", "?"),
            )
            logger.info(
                "[fast_preprocess] inspect control-flow output idx=%d name=%s target=%s user=%s user_target=%s",
                idx,
                getattr(submodule_output, "name", "?"),
                getattr(submodule_output, "target", "?"),
                getattr(submodule_user, "name", "?"),
                getattr(submodule_user, "target", "?"),
            )
            if _is_layout_output_node(submodule_output):
                source_input = (
                    submodule_output.args[0]
                    if getattr(submodule_output, "args", None)
                    and hasattr(submodule_output.args[0], "op")
                    else None
                )
                if source_input is not None:
                    logger.info(
                        "[fast_preprocess] retargeting control-flow output layout %s -> %s",
                        getattr(submodule_output, "name", "?"),
                        getattr(source_input, "name", "?"),
                    )
                    source_input.meta["val"] = _get_control_flow_output_fake_tensor(
                        submodule_user
                    )
                    if submodule_user.meta.get("tosa_dim_order") is not None:
                        source_input.meta["tosa_dim_order"] = tuple(
                            submodule_user.meta["tosa_dim_order"]
                        )
                    output_args[idx] = source_input
                    updated = True
                    rewritten += 1
                    continue
            if _is_dequantize_node(submodule_output):
                layout_input = submodule_output.args[0] if getattr(
                    submodule_output, "args", None
                ) else None
                source_input = (
                    layout_input.args[0]
                    if hasattr(layout_input, "op")
                    and _is_layout_output_node(layout_input)
                    and getattr(layout_input, "args", None)
                    and hasattr(layout_input.args[0], "op")
                    else None
                )
                if source_input is not None:
                    logger.info(
                        "[fast_preprocess] retargeting control-flow output dequantize %s through %s -> %s",
                        getattr(submodule_output, "name", "?"),
                        getattr(layout_input, "name", "?"),
                        getattr(source_input, "name", "?"),
                    )
                    submodule_output.update_arg(0, source_input)
                    submodule_output.meta["val"] = _get_control_flow_output_fake_tensor(
                        submodule_user
                    )
                    if submodule_user.meta.get("tosa_dim_order") is not None:
                        submodule_output.meta["tosa_dim_order"] = tuple(
                            submodule_user.meta["tosa_dim_order"]
                        )
                    updated = True
                    rewritten += 1
                    continue

            if _is_tosa_rescale_node(submodule_output):
                layout_input = submodule_output.args[0] if getattr(
                    submodule_output, "args", None
                ) else None
                _log_control_flow_debug(
                    "[fast_preprocess] inspect rescale output node=%s input=%s input_target=%s is_layout=%s",
                    getattr(submodule_output, "name", "?"),
                    getattr(layout_input, "name", "?"),
                    getattr(layout_input, "target", "?"),
                    _is_layout_output_node(layout_input)
                    if hasattr(layout_input, "op")
                    else False,
                )
                logger.info(
                    "[fast_preprocess] rescale output candidate %s input=%s input_target=%s is_layout=%s",
                    getattr(submodule_output, "name", "?"),
                    getattr(layout_input, "name", "?"),
                    getattr(layout_input, "target", "?"),
                    _is_layout_output_node(layout_input) if hasattr(layout_input, "op") else False,
                )
                source_input = (
                    layout_input.args[0]
                    if hasattr(layout_input, "op")
                    and _is_layout_output_node(layout_input)
                    and getattr(layout_input, "args", None)
                    and hasattr(layout_input.args[0], "op")
                    else None
                )
                if source_input is not None:
                    logger.info(
                        "[fast_preprocess] retargeting control-flow output rescale %s through %s -> %s",
                        getattr(submodule_output, "name", "?"),
                        getattr(layout_input, "name", "?"),
                        getattr(source_input, "name", "?"),
                    )
                    submodule_output.update_arg(0, source_input)
                    submodule_output.meta["val"] = _get_control_flow_output_fake_tensor(
                        submodule_user
                    )
                    if submodule_user.meta.get("tosa_dim_order") is not None:
                        submodule_output.meta["tosa_dim_order"] = tuple(
                            submodule_user.meta["tosa_dim_order"]
                        )
                    updated = True
                    rewritten += 1
                    continue

            if not _is_trivial_control_flow_output_chain(submodule_output):
                continue
            logger.info(
                "[fast_preprocess] control-flow output candidate name=%s target=%s user=%s user_target=%s",
                getattr(submodule_output, "name", "?"),
                getattr(submodule_output, "target", "?"),
                getattr(submodule_user, "name", "?"),
                getattr(submodule_user, "target", "?"),
            )
            _log_node_chain(submodule_output)

            replacement = _select_control_flow_output_value(submodule_output, submodule_user)
            if replacement is None:
                continue

            logger.info(
                "[fast_preprocess] selected control-flow replacement %s",
                _describe_node_brief(replacement),
            )
            replacement.meta["val"] = _get_control_flow_output_fake_tensor(submodule_user)
            if submodule_user.meta.get("tosa_dim_order") is not None:
                replacement.meta["tosa_dim_order"] = tuple(
                    submodule_user.meta["tosa_dim_order"]
                )
            output_args[idx] = replacement
            updated = True
            rewritten += 1

        if updated:
            replacement = output_args[0] if wrapped_single else tuple(output_args)
            output_node.update_arg(0, replacement)
            final_outputs = output_node.args[0]
            if hasattr(final_outputs, "op"):
                final_outputs = [final_outputs]
            logger.info(
                "[fast_preprocess] final control-flow outputs %s",
                [getattr(n, "name", "?") for n in final_outputs],
            )
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow output layout node(s)",
            rewritten,
        )


def _restore_control_flow_output_dequantizes(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertControlFlowRescalesPass,
    )

    def _refresh_submodule_output_input_qparams(submodule) -> None:
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        input_qparams_map = {}
        for idx, output_arg in enumerate(output_args):
            if not hasattr(output_arg, "op"):
                continue
            qparam = _extract_effective_node_qparam(output_arg)
            if qparam is not None:
                input_qparams_map[idx] = qparam
        if input_qparams_map:
            output_node.meta["input_qparams"] = input_qparams_map

    rescale_pass = InsertControlFlowRescalesPass()
    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        output_qparams_map = submodule.graph.output_node().meta.get(
            "arm_control_flow_output_dequant_qparams", {}
        )
        if not output_qparams_map:
            continue
        _refresh_submodule_output_input_qparams(submodule)
        if rescale_pass._rescale_submodule_outputs(submodule, output_qparams_map):
            restored += len(output_qparams_map)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow branch output rescale(s)",
            restored,
        )


def _normalize_control_flow_input_layouts(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    rewritten = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        updated = False
        for node in list(submodule.graph.nodes):
            if node.op != "placeholder":
                continue

            anchor = node
            users = list(anchor.users)
            while len(users) == 1:
                next_user = users[0]
                if _is_layout_output_node(next_user):
                    break
                if not _is_control_flow_input_passthrough_node(next_user):
                    next_user = None
                    break
                anchor = next_user
                users = list(anchor.users)

            if len(users) != 1:
                continue
            layout_user = users[0]
            if not _is_layout_output_node(layout_user):
                continue

            logger.info(
                "[fast_preprocess] control-flow input candidate placeholder=%s anchor=%s user=%s target=%s",
                getattr(node, "name", "?"),
                getattr(anchor, "name", "?"),
                getattr(layout_user, "name", "?"),
                getattr(layout_user, "target", "?"),
            )
            _log_node_chain(layout_user)

            anchor.meta["val"] = layout_user.meta.get("val", anchor.meta.get("val"))
            if layout_user.meta.get("tosa_dim_order") is not None:
                anchor.meta["tosa_dim_order"] = tuple(layout_user.meta["tosa_dim_order"])
            layout_user.replace_all_uses_with(anchor)
            if len(layout_user.users) == 0:
                submodule.graph.erase_node(layout_user)
            rewritten += 1
            updated = True

        if updated:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow input layout node(s)",
            rewritten,
        )


def _dump_control_flow_submodules(graph_module, label: str) -> None:
    if not _control_flow_debug_enabled():
        return

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    for name, submodule, _ in get_cond_while_submodules_nested(graph_module):
        try:
            readable = submodule.print_readable(print_output=False)
        except TypeError:
            readable = submodule.print_readable()
        logger.info(
            "[fast_preprocess] control-flow submodule dump (%s:%s)\n%s",
            label,
            name,
            readable,
        )


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
    import json
    import os
    from pathlib import Path

    from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
    from executorch.backends.arm.common.debug import debug_tosa_dump
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
    _prepare_control_flow_qparams(gm, edge_program)

    # Run the full FX-direct pipeline (all passes in Rust via PyO3)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
    _reapply_control_flow_table_ops(gm, edge_program)
    _repair_control_flow_binary_rescales(gm, _compile_spec.tosa_spec)
    _canonicalize_control_flow_pool_binary_rescales(gm)
    _reapply_control_flow_boundary_rescales(gm, include_inputs=True, include_outputs=False)
    _prune_duplicate_control_flow_rescales(gm)
    _repair_control_flow_binary_rescales(gm, _compile_spec.tosa_spec)
    _restore_control_flow_output_dequantizes(gm)
    t_passes = time.perf_counter() - t0
    _dump_control_flow_submodules(gm, "after_rust_passes")

    # Re-apply the same normalization after passes so serializer ingest sees
    # branch signatures that still match the parent control-flow nodes.
    _regularize_control_flow_submodules(gm)
    _normalize_control_flow_input_layouts(gm)
    _normalize_control_flow_output_layouts(gm)
    _reapply_control_flow_boundary_rescales(
        gm, include_inputs=False, include_outputs=True
    )
    _prune_duplicate_control_flow_rescales(gm)
    _quantize_constant_only_control_flow_outputs(gm)
    _regularize_control_flow_submodules(gm)
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
