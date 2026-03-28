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
    import torch

    from executorch.exir.dim_order_utils import get_memory_format

    fake_tensor = node.meta["val"]
    desired_dim_order = node.meta.get("tosa_dim_order")
    if desired_dim_order is None:
        return fake_tensor

    # Some control-flow rewrites preserve a 4D dim-order annotation while the
    # effective value has already collapsed to a lower-rank tensor. In that
    # case the stale memory-format hint is no longer applicable; keep the fake
    # tensor as-is instead of crashing on a rank/layout mismatch.
    if fake_tensor.dim() != len(desired_dim_order):
        return fake_tensor

    try:
        memory_format = get_memory_format(list(desired_dim_order))
    except AssertionError:
        # Some TOSA-only tensor orders, like transpose-conv weights, are
        # metadata-only layouts and do not map to a torch.memory_format.
        return fake_tensor
    if memory_format == torch.channels_last and fake_tensor.dim() != 4:
        return fake_tensor

    return fake_tensor.to(memory_format=memory_format)


def _refresh_tosa_dim_order_fake_tensors(graph_module) -> None:
    """Re-sync fake tensors with `tosa_dim_order` after FX-direct rewrites.

    Several Rust/Python passes update `meta["tosa_dim_order"]` without forcing
    a matching FakeTensor memory-format refresh. The runner later decodes TOSA
    outputs using the FakeTensor's dim_order, so stale contiguous fake tensors
    can turn a logically correct NHWC result into a numerically scrambled NCHW
    tensor. Metadata is part of the program here; keep the fake tensors aligned.
    """
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    modules = [graph_module]
    modules.extend(submodule for _, submodule, _ in get_cond_while_submodules_nested(graph_module))

    for module in modules:
        updated = False
        for node in module.graph.nodes:
            if "val" not in node.meta or node.meta.get("tosa_dim_order") is None:
                continue
            node.meta["val"] = _get_matching_fake_tensor(node)
            updated = True
        if updated:
            module.recompile()


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


def _regularize_control_flow_submodule(
    submodule,
    submodule_node,
    *,
    include_inputs: bool = True,
    include_outputs: bool = True,
) -> None:
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

    if include_inputs:
        for submodule_input, submodule_arg in zip(submodule_inputs, args, strict=True):
            submodule_input.meta["val"] = _get_matching_fake_tensor(submodule_arg)
            if submodule_arg.meta.get("tosa_dim_order") is not None:
                submodule_input.meta["tosa_dim_order"] = tuple(
                    submodule_arg.meta["tosa_dim_order"]
                )
            saved_output_qparam = _branch_local_control_flow_output_qparam(
                submodule_input
            )
            if saved_output_qparam is not None:
                submodule_input.meta["arm_control_flow_saved_output_qparams"] = {
                    0: saved_output_qparam
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
    if include_outputs:
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


def _regularize_control_flow_submodules(
    graph_module,
    *,
    include_inputs: bool = True,
    include_outputs: bool = True,
) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    regularized = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        _regularize_control_flow_submodule(
            submodule,
            control_flow_node,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
        )
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


def _extract_single_qparam_entry(qparams):
    if not qparams:
        return None
    if isinstance(qparams, dict):
        if 0 in qparams:
            return qparams[0]
        if len(qparams) == 1:
            return next(iter(qparams.values()))
    return qparams


def _saved_control_flow_placeholder_output_qparam(placeholder):
    return _extract_single_qparam_entry(
        placeholder.meta.get("arm_control_flow_saved_output_qparams")
    )


def _downstream_control_flow_table_input_qparam(placeholder):
    pending = list(getattr(placeholder, "users", ()))
    seen = {id(placeholder)}
    while pending:
        user = pending.pop(0)
        if not hasattr(user, "op") or id(user) in seen:
            continue
        seen.add(id(user))

        table_qparam = user.meta.get("arm_saved_table_input_qparam")
        if table_qparam is not None:
            return table_qparam

        pending.extend(list(getattr(user, "users", ())))

    return None

def _branch_local_control_flow_output_qparam(placeholder):
    import torch

    # Parse, don't validate: the control-flow placeholder boundary should keep
    # the saved branch-local quantized contract when it exists, not rediscover
    # it from a later accumulator-domain rescale chain.
    saved_qparam = _saved_control_flow_placeholder_output_qparam(placeholder)
    if saved_qparam is not None:
        return saved_qparam

    table_qparam = _downstream_control_flow_table_input_qparam(placeholder)
    if table_qparam is not None:
        return table_qparam

    placeholder_output_qparam = _extract_single_qparam_entry(
        placeholder.meta.get("output_qparams")
    )
    if getattr(placeholder_output_qparam, "dtype", None) in (torch.int8, torch.int16):
        return placeholder_output_qparam

    return _infer_control_flow_placeholder_output_qparam(placeholder)


def _placeholder_immediately_lifts_to_int32(placeholder) -> bool:
    import torch

    users = [user for user in list(getattr(placeholder, "users", ())) if hasattr(user, "op")]
    if not users:
        return False
    return all(
        _is_tosa_rescale_node(user)
        and len(getattr(user, "args", ())) >= 2
        and user.args[1] == torch.int32
        for user in users
    )


def _placeholder_feeds_binary_directly(placeholder) -> bool:
    users = [user for user in list(getattr(placeholder, "users", ())) if hasattr(user, "op")]
    if not users:
        return False
    return all(
        getattr(user, "op", None) == "call_function"
        and any(
            target in str(getattr(user, "target", ""))
            for target in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor")
        )
        and not _binary_peer_is_scalar_like(user, placeholder)
        for user in users
    )


def _placeholder_int32_lift_feeds_scalar_binary(placeholder) -> bool:
    import torch

    lift_nodes = [user for user in list(getattr(placeholder, "users", ())) if hasattr(user, "op")]
    if not lift_nodes:
        return False
    if not all(
        _is_tosa_rescale_node(user)
        and len(getattr(user, "args", ())) >= 2
        and user.args[1] == torch.int32
        for user in lift_nodes
    ):
        return False

    saw_binary_user = False
    for lift_node in lift_nodes:
        downstream_users = [user for user in list(getattr(lift_node, "users", ())) if hasattr(user, "op")]
        if not downstream_users:
            return False
        for user in downstream_users:
            target_str = str(getattr(user, "target", ""))
            if getattr(user, "op", None) != "call_function" or not any(
                target in target_str
                for target in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor")
            ):
                return False
            if not _binary_peer_is_scalar_like(user, lift_node):
                return False
            saw_binary_user = True
    return saw_binary_user


def _unwrap_passthrough_input(node):
    current = node
    seen = set()
    passthrough_markers = (
        "quantized_decomposed.quantize_per_tensor",
        "quantized_decomposed.dequantize_per_tensor",
        "backend._ops.tosa.RESCALE",
        "aten.view_copy",
        "aten.clone",
    )
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "op", None) != "call_function":
            break
        if not any(marker in str(getattr(current, "target", "")) for marker in passthrough_markers):
            break
        args = tuple(getattr(current, "args", ()))
        if not args or not hasattr(args[0], "op"):
            break
        current = args[0]
    return current


def _binary_peer_is_scalar_like(binary_user, placeholder) -> bool:
    peers = [
        arg
        for arg in tuple(getattr(binary_user, "args", ()))
        if hasattr(arg, "op") and arg is not placeholder
    ]
    if not peers:
        return False
    for peer in peers:
        source = _unwrap_passthrough_input(peer)
        fake_val = getattr(source, "meta", {}).get("val")
        if hasattr(fake_val, "numel") and fake_val.numel() == 1:
            return True
    return False


def _infer_control_flow_placeholder_output_qparam(placeholder):
    import torch

    # Parse, don't validate: the branch-local contract is defined by the
    # placeholder's downstream passthrough path, not by whichever qparams
    # happened to survive on the placeholder node after intermediate rewrites.
    saved_qparam = _saved_control_flow_placeholder_output_qparam(placeholder)
    if saved_qparam is not None:
        return saved_qparam

    pending = list(getattr(placeholder, "users", ()))
    seen = {id(placeholder)}
    fallback_qparam = None
    while pending:
        user = pending.pop(0)
        if not hasattr(user, "op") or id(user) in seen:
            continue
        seen.add(id(user))

        qparam = _extract_effective_node_qparam(user)
        if qparam is not None:
            if getattr(qparam, "dtype", None) in (torch.int8, torch.int16):
                return qparam
            if fallback_qparam is None:
                fallback_qparam = qparam

        target_str = str(getattr(user, "target", ""))
        if (
            "quantize_per_tensor" in target_str
            or "dequantize_per_tensor" in target_str
            or "tosa_RESCALE" in target_str
            or "tosa.RESCALE" in target_str
            or "TosaRescale" in target_str
        ):
            pending.extend(list(getattr(user, "users", ())))

    if fallback_qparam is not None:
        return fallback_qparam

    output_qparam = _extract_single_qparam_entry(placeholder.meta.get("output_qparams"))
    if output_qparam is not None:
        return output_qparam

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

    def _infer_parent_output_qparams_map(node) -> dict[int, QuantArgs]:
        output_qparams_map: dict[int, QuantArgs] = {}
        for getitem_node in _ordered_control_flow_users(node):
            if len(getattr(getitem_node, "args", ())) < 2:
                continue
            idx = int(getitem_node.args[1])

            qparam = _extract_single_qparam_entry(getitem_node.meta.get("output_qparams"))
            if qparam is not None:
                output_qparams_map[idx] = qparam
                continue

            for user in list(getattr(getitem_node, "users", ())):
                input_qparams = user.meta.get("input_qparams") or {}
                for arg_idx, arg in enumerate(getattr(user, "args", ())):
                    if arg is getitem_node and arg_idx in input_qparams:
                        output_qparams_map[idx] = input_qparams[arg_idx]
                        break
                if idx in output_qparams_map:
                    break

        return output_qparams_map

    def _seed_constant_only_control_flow_output_qparams(owner_graph_module, node) -> None:
        output_qparams_map = _infer_parent_output_qparams_map(node)
        if not output_qparams_map:
            return

        if node.target == torch.ops.higher_order.cond:
            submodules = [
                owner_graph_module.get_submodule(node.args[1].target),
                owner_graph_module.get_submodule(node.args[2].target),
            ]
        elif node.target == torch.ops.higher_order.while_loop:
            submodules = [owner_graph_module.get_submodule(node.args[1].target)]
        else:
            submodules = []

        for submodule in submodules:
            placeholders = [
                subnode for subnode in submodule.graph.nodes if subnode.op == "placeholder"
            ]
            if placeholders:
                continue

            output_node = submodule.graph.output_node()
            if output_node.meta.get("input_qparams"):
                continue

            output_node.meta["input_qparams"] = dict(output_qparams_map)
            logger.info(
                "[fast_preprocess] seeded constant-only control-flow output qparams for %s: %s",
                getattr(submodule, "__class__", type(submodule)).__name__,
                output_qparams_map,
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
            _seed_constant_only_control_flow_output_qparams(graph_module, node)
            _log_control_flow_qparams(node, graph_module)
            cond_inputs = cast(list, node.args[-1]) if node.args else []
            if cond_inputs:
                modified |= rescale_pass._rescale_cond_submodules(node, graph_module)
            prepared += 1
        elif node.target == torch.ops.higher_order.while_loop:
            fold_pass._handle_control_flow_node(node, graph_module)
            _snapshot_submodule_input_qparams(graph_module, node)
            _synthesize_control_flow_input_qparams(node)
            _seed_constant_only_control_flow_output_qparams(graph_module, node)
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

            # Parse, don't validate: dequantize/quantize nodes already carry
            # the exact local qparam domain we need for table lowering. If we
            # skip straight to their producer, we silently recover the parent
            # boundary domain instead and build the TABLE against the wrong LUT
            # input scale/zero-point.
            try:
                from executorch.backends.arm._passes.quant_args import QuantArgs

                return QuantArgs.from_operator(current.target, current.args)
            except Exception:
                pass

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
                qparam = node.meta.get("arm_saved_table_input_qparam")
                if qparam is None and getattr(node, "args", None) and hasattr(node.args[0], "op"):
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
                qparam = node.meta.get("arm_saved_table_output_qparam")
                if qparam is None:
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


def _snapshot_control_flow_table_qparams(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_table_ops import (
        TableOps,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs

    table_targets = list(TableOps.included_ops())
    snapshotted = 0

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        for node in submodule.graph.nodes:
            if node.op != "call_function" or node.target not in table_targets:
                continue

            input_qparam = None
            if getattr(node, "args", None) and hasattr(node.args[0], "op"):
                try:
                    input_qparam = QuantArgs.from_operator(
                        node.args[0].target, node.args[0].args
                    )
                except Exception:
                    input_qparam = None

            output_qparam = None
            for user in list(getattr(node, "users", ())):
                try:
                    output_qparam = QuantArgs.from_operator(user.target, user.args)
                    break
                except Exception:
                    continue

            if input_qparam is not None:
                node.meta["arm_saved_table_input_qparam"] = input_qparam
                snapshotted += 1
            if output_qparam is not None:
                node.meta["arm_saved_table_output_qparam"] = output_qparam
                snapshotted += 1

    if snapshotted:
        logger.info(
            "[fast_preprocess] snapshotted %d control-flow table qparam attachment(s)",
            snapshotted,
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
    import fx_helpers

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

    def _is_tosa_table_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "tosa.TABLE" in target_str or "tosa::table" in target_str

    def _rescale_output_qparam(node):
        if not _is_tosa_rescale_node(node):
            return _extract_qparam(node)
        if len(getattr(node, "args", ())) < 5:
            return _extract_qparam(node)

        output_dtype = node.args[1]
        if output_dtype not in (torch.int8, torch.int16, torch.int32):
            return _extract_qparam(node)

        source_node = node.args[0]
        if not hasattr(source_node, "op"):
            return _extract_qparam(node)
        source_qparam = _extract_qparam(source_node)
        scale_arg = node.args[2]
        if (
            source_qparam is None
            or not isinstance(scale_arg, (list, tuple))
            or len(scale_arg) != 1
            or scale_arg[0] in (0, 0.0)
        ):
            return _extract_qparam(node)

        try:
            qinfo = torch.iinfo(output_dtype)
            return QuantArgs(
                scale=source_qparam.get_scale_per_tensor() / float(scale_arg[0]),
                zp=int(node.args[4]),
                qmin=int(qinfo.min),
                qmax=int(qinfo.max),
                dtype=output_dtype,
            )
        except Exception:
            return _extract_qparam(node)

    def _chain_nodes(node):
        current = node
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))
            yield current
            next_arg = None
            for maybe_arg in getattr(current, "args", ()):
                if hasattr(maybe_arg, "op"):
                    next_arg = maybe_arg
                    break
            current = next_arg

    def _placeholder_dependencies(node):
        deps = set()
        stack = [node]
        seen = set()
        while stack:
            current = stack.pop()
            if not hasattr(current, "op") or id(current) in seen:
                continue
            seen.add(id(current))
            if getattr(current, "op", None) == "placeholder":
                deps.add(current)
                continue
            stack.extend(
                maybe_arg
                for maybe_arg in getattr(current, "args", ())
                if hasattr(maybe_arg, "op")
            )
        return deps

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

    def _is_quantized_binary_result(node):
        if not _is_tosa_rescale_node(node):
            return False
        args = getattr(node, "args", ())
        if not args or not hasattr(args[0], "op"):
            return False
        source = args[0]
        if getattr(source, "op", None) != "call_function":
            return False
        target_str = str(getattr(source, "target", ""))
        return "aten.add.Tensor" in target_str or "aten.sub.Tensor" in target_str

    def _preferred_quantized_binary_output_rescale(source_node):
        if getattr(source_node, "op", None) != "call_function":
            return None
        target_str = str(getattr(source_node, "target", ""))
        if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
            return None

        best = None
        best_score = -1
        for user in list(getattr(source_node, "users", ())):
            if not _is_tosa_rescale_node(user):
                continue
            if len(getattr(user, "args", ())) < 5:
                continue
            if user.args[1] not in (torch.int8, torch.int16):
                continue
            output_qparam = _rescale_output_qparam(user)
            if output_qparam is None:
                continue

            score = 0
            for grand_user in list(getattr(user, "users", ())):
                if getattr(grand_user, "op", None) == "output":
                    score += 4
                elif _is_tosa_rescale_node(grand_user):
                    score += 2
            if score > best_score:
                best = (user, output_qparam)
                best_score = score

        return best

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

    def _int32_accumulator_qparam(scale: float):
        qinfo = torch.iinfo(torch.int32)
        return QuantArgs(
            scale=scale,
            zp=0,
            qmin=int(qinfo.min),
            qmax=int(qinfo.max),
            dtype=torch.int32,
        )

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

        # Parse, don't validate: a captured control-flow tensor can get
        # accidentally replaced by a dequantized TABLE result before the
        # int32-rescale repair runs. That leaves one branch placeholder dead,
        # and the later serializer drops the missing input entirely. If a
        # binary op only depends on one placeholder even though another branch
        # placeholder is available, prefer that unused placeholder over a
        # TABLE-derived dequantize chain so InsertRescaleInt32Pass can rebuild
        # the correct sibling path.
        for node in list(submodule.graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue

            unused_placeholders = [
                placeholder
                for placeholder in placeholders
                if placeholder not in _placeholder_dependencies(node)
            ]
            if len(unused_placeholders) != 1:
                continue
            rescued_placeholder = unused_placeholders[0]

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op"):
                    continue
                chain = list(_chain_nodes(arg))
                if not any(_is_dequantize_node(candidate) for candidate in chain):
                    continue
                if not any(_is_tosa_table_node(candidate) for candidate in chain):
                    continue
                if not _is_tosa_rescale_node(arg):
                    continue
                if getattr(arg, "args", None):
                    arg.update_arg(0, rescued_placeholder)
                    rescued_qparam = _extract_qparam(rescued_placeholder)
                    if rescued_qparam is not None:
                        arg.meta["input_qparams"] = {0: rescued_qparam}
                    updated_graph = True
                    logger.info(
                        "[fast_preprocess] rescued control-flow binary input %s arg%d -> %s",
                        getattr(node, "name", "?"),
                        idx,
                        getattr(rescued_placeholder, "name", "?"),
                    )
                    break

        for node in list(submodule.graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op"):
                    continue
                candidate_source = arg
                if _is_tosa_rescale_node(arg) and getattr(arg, "args", None):
                    source_arg = arg.args[0]
                    if hasattr(source_arg, "op"):
                        candidate_source = source_arg
                preferred = _preferred_quantized_binary_output_rescale(candidate_source)
                if preferred is not None:
                    quant_rescale, output_qparam = preferred
                    cloned = fx_helpers.insert_rescale_node(
                        submodule,
                        node,
                        candidate_source,
                        quant_rescale.args[1],
                        list(quant_rescale.args[2]),
                        int(quant_rescale.args[3]),
                        int(quant_rescale.args[4]),
                        after=False,
                    )
                    source_qparam = _extract_qparam(candidate_source)
                    if source_qparam is not None:
                        cloned.meta["input_qparams"] = {0: source_qparam}
                    cloned.meta["output_qparams"] = {0: output_qparam}
                    node.replace_input_with(arg, cloned)
                    updated_graph = True
                    logger.info(
                        "[fast_preprocess] cloned control-flow binary output rescale %s for %s arg%d via %s",
                        getattr(cloned, "name", "?"),
                        getattr(node, "name", "?"),
                        idx,
                        getattr(quant_rescale, "name", "?"),
                    )
                    arg = cloned
                if _extract_effective_node_qparam(arg) is not None:
                    continue
                sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                if not hasattr(sibling, "op"):
                    continue
                promoted = None
                sibling_input = None
                if getattr(sibling, "args", None) and hasattr(sibling.args[0], "op"):
                    sibling_input = sibling.args[0]
                if (
                    sibling_input is not None
                    and _extract_effective_node_qparam(sibling_input) is not None
                ):
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
                if (
                    len(getattr(user, "args", ())) >= 2
                    and user.args[1] in (torch.int8, torch.int16)
                ):
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

                producer_qparam = _extract_effective_node_qparam(arg)
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
                sibling_qparam = (
                    _extract_effective_node_qparam(sibling) if hasattr(sibling, "op") else None
                )
                sibling_binary_output_qparam = None
                if hasattr(sibling, "op"):
                    preferred_sibling_quantized = _preferred_quantized_binary_output_rescale(
                        sibling
                    )
                    if preferred_sibling_quantized is not None:
                        _, sibling_binary_output_qparam = preferred_sibling_quantized
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
                if (
                    producer_qparam is not None
                    and (
                        sibling_qparam is not None or sibling_binary_output_qparam is not None
                    )
                    and getattr(producer_qparam, "dtype", None)
                    in (torch.int8, torch.int16)
                    and getattr(
                        sibling_binary_output_qparam or sibling_qparam, "dtype", None
                    )
                    in (torch.int8, torch.int16)
                    and (
                        (hasattr(sibling, "op") and _is_quantized_binary_result(sibling))
                        or sibling_binary_output_qparam is not None
                    )
                    and (
                        float(producer_qparam.scale)
                        != float((sibling_binary_output_qparam or sibling_qparam).scale)
                        or int(producer_qparam.zp)
                        != int((sibling_binary_output_qparam or sibling_qparam).zp)
                    )
                ):
                    producer_qparam = sibling_binary_output_qparam or sibling_qparam
                    logger.info(
                        "[fast_preprocess] aligned binary operand qparams for %s arg%d to sibling quantized result: %s",
                        getattr(node, "name", "?"),
                        idx,
                        producer_qparam,
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
                        producer_qparam = _extract_effective_node_qparam(sibling.args[0])
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
            if len(input_qparams) == 2:
                lhs_qparam = input_qparams.get(0)
                rhs_qparam = input_qparams.get(1)
                if {
                    getattr(lhs_qparam, "dtype", None),
                    getattr(rhs_qparam, "dtype", None),
                } <= {torch.int8, torch.int16, torch.int32} and (
                    getattr(lhs_qparam, "dtype", None)
                    != getattr(rhs_qparam, "dtype", None)
                ):
                    for mixed_idx in (0, 1):
                        mixed_qparam = input_qparams.get(mixed_idx)
                        sibling_qparam = input_qparams.get(1 - mixed_idx)
                        if (
                            getattr(mixed_qparam, "dtype", None) != torch.int32
                            or getattr(sibling_qparam, "dtype", None)
                            not in (torch.int8, torch.int16)
                        ):
                            continue

                        mixed_arg = node.args[mixed_idx] if len(node.args) > mixed_idx else None
                        recovered_qparam = None
                        if hasattr(mixed_arg, "op"):
                            if _is_tosa_rescale_node(mixed_arg) and getattr(
                                mixed_arg, "args", None
                            ):
                                source_arg = mixed_arg.args[0]
                                if hasattr(source_arg, "op"):
                                    recovered_qparam = _extract_effective_node_qparam(
                                        source_arg
                                    )
                                    if recovered_qparam is None:
                                        recovered_qparam = _reconstruct_rescale_source_qparam(
                                            node, mixed_idx, mixed_arg
                                        )
                            if recovered_qparam is None:
                                preferred_quantized = _preferred_quantized_binary_output_rescale(
                                    mixed_arg
                                )
                                if preferred_quantized is not None:
                                    _, recovered_qparam = preferred_quantized
                            if recovered_qparam is None and getattr(
                                mixed_arg, "op", None
                            ) == "call_function":
                                mixed_target = str(getattr(mixed_arg, "target", ""))
                                if (
                                    "aten.add.Tensor" in mixed_target
                                    or "aten.sub.Tensor" in mixed_target
                                    or "aten.mul.Tensor" in mixed_target
                                ):
                                    for source_arg in mixed_arg.args[:2]:
                                        if not hasattr(source_arg, "op"):
                                            continue
                                        source_qparam = _extract_effective_node_qparam(
                                            source_arg
                                        )
                                        if (
                                            getattr(source_qparam, "dtype", None)
                                            == getattr(sibling_qparam, "dtype", None)
                                        ):
                                            recovered_qparam = source_qparam
                                            break

                        if (
                            getattr(recovered_qparam, "dtype", None)
                            == getattr(sibling_qparam, "dtype", None)
                        ):
                            input_qparams[mixed_idx] = recovered_qparam
                            lhs_qparam = input_qparams.get(0)
                            rhs_qparam = input_qparams.get(1)
                            updated_graph = True
                            logger.info(
                                "[fast_preprocess] recovered mixed control-flow binary input_qparams for %s arg%d -> %s",
                                getattr(node, "name", "?"),
                                mixed_idx,
                                recovered_qparam,
                            )
                            break

                if (
                    getattr(lhs_qparam, "dtype", None)
                    == getattr(rhs_qparam, "dtype", None)
                    and getattr(lhs_qparam, "dtype", None) in (torch.int8, torch.int16)
                ):
                    shift_bits = 12 if lhs_qparam.dtype == torch.int16 else 20
                    accumulator_scale = (
                        2.0
                        * max(
                            float(lhs_qparam.get_scale_per_tensor()),
                            float(rhs_qparam.get_scale_per_tensor()),
                        )
                        / float(1 << shift_bits)
                    )
                    accumulator_qparam = _int32_accumulator_qparam(accumulator_scale)
                    node.meta["input_qparams"] = {
                        0: accumulator_qparam,
                        1: accumulator_qparam,
                    }
                    node.meta["output_qparams"] = {0: accumulator_qparam}
                    updated_graph = True
                    logger.info(
                        "[fast_preprocess] recomputed control-flow binary accumulator qparams for %s -> %s",
                        getattr(node, "name", "?"),
                        accumulator_qparam,
                    )

                    for user in list(node.users):
                        if not _is_tosa_rescale_node(user):
                            continue
                        if len(getattr(user, "args", ())) < 5:
                            continue
                        if user.args[1] not in (torch.int8, torch.int16):
                            continue
                        user_output_qparam = _rescale_output_qparam(user)
                        if user_output_qparam is None:
                            continue
                        user.update_arg(
                            2,
                            [
                                accumulator_qparam.get_scale_per_tensor()
                                / user_output_qparam.get_scale_per_tensor()
                            ],
                        )
                        user.update_arg(3, accumulator_qparam.get_zp_per_tensor())
                        user.update_arg(4, user_output_qparam.get_zp_per_tensor())
                        user.meta["input_qparams"] = {0: accumulator_qparam}
                        user.meta["output_qparams"] = {0: user_output_qparam}
                        updated_graph = True
                        logger.info(
                            "[fast_preprocess] realigned pre-pass control-flow binary output rescale %s from %s",
                            getattr(user, "name", "?"),
                            getattr(node, "name", "?"),
                        )
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

        try:
            with TosaLoweringContext(tosa_spec):
                result = InsertRescaleInt32Pass().call(submodule)
        except Exception:
            if _control_flow_debug_enabled():
                print(f"[fast_preprocess] InsertRescaleInt32Pass failed in {name}")
                for subnode in list(submodule.graph.nodes):
                    if getattr(subnode, "op", None) != "call_function":
                        continue
                    target_str = str(getattr(subnode, "target", ""))
                    if (
                        "aten.add.Tensor" not in target_str
                        and "aten.sub.Tensor" not in target_str
                        and "aten.mul.Tensor" not in target_str
                    ):
                        continue
                    print(
                        "[fast_preprocess] failing binary",
                        _describe_node_brief(subnode),
                        "input_qparams=",
                        repr(subnode.meta.get("input_qparams")),
                        "output_qparams=",
                        repr(subnode.meta.get("output_qparams")),
                    )
                    for idx, arg in enumerate(subnode.args[:2]):
                        if not hasattr(arg, "op"):
                            continue
                        print(
                            f"[fast_preprocess] failing binary arg{idx}",
                            _describe_node_brief(arg),
                            "input_qparams=",
                            repr(arg.meta.get("input_qparams")),
                            "output_qparams=",
                            repr(arg.meta.get("output_qparams")),
                        )
            raise
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


def _restore_control_flow_pool_quant_boundaries(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    import fx_helpers

    def _is_avg_pool_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "AVG_POOL2D" in target_str or "avg_pool2d" in target_str

    def _is_view_like(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "view_copy" in target_str or "RESHAPE" in target_str

    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        modified = False
        for placeholder in list(submodule.graph.nodes):
            if placeholder.op != "placeholder":
                continue

            input_qparam = _extract_single_qparam_entry(placeholder.meta.get("input_qparams"))
            if getattr(input_qparam, "dtype", None) not in (torch.int8, torch.int16):
                continue

            for avg_pool in list(getattr(placeholder, "users", ())):
                if not _is_avg_pool_node(avg_pool):
                    continue

                existing_boundary = None
                for user in list(getattr(placeholder, "users", ())):
                    if not _is_tosa_rescale_node(user):
                        continue
                    if not getattr(user, "meta", {}).get(
                        "arm_preserve_control_flow_quant_boundary", False
                    ):
                        continue
                    if avg_pool in getattr(user, "users", ()):
                        existing_boundary = user
                        break

                quant_boundary = existing_boundary
                if quant_boundary is None:
                    quant_boundary = fx_helpers.insert_rescale_node(
                        submodule,
                        placeholder,
                        placeholder,
                        input_qparam.dtype,
                        [1.0],
                        input_qparam.get_zp_per_tensor(),
                        input_qparam.get_zp_per_tensor(),
                        after=True,
                    )
                    quant_boundary.meta["input_qparams"] = {0: input_qparam}
                    quant_boundary.meta["output_qparams"] = {0: input_qparam}
                    quant_boundary.meta["arm_preserve_control_flow_quant_boundary"] = True
                    restored += 1
                    modified = True
                    logger.info(
                        "[fast_preprocess] restored control-flow pool quant boundary %s -> %s",
                        getattr(placeholder, "name", "?"),
                        getattr(quant_boundary, "name", "?"),
                    )

                avg_pool.replace_input_with(placeholder, quant_boundary)
                avg_pool.meta["input_qparams"] = {0: input_qparam}
                avg_pool.meta["output_qparams"] = {0: input_qparam}
                fx_helpers.refresh_node_fake_tensor(avg_pool)

                avg_pool_users = [
                    user for user in list(getattr(avg_pool, "users", ())) if hasattr(user, "op")
                ]
                if not avg_pool_users or not all(_is_view_like(user) for user in avg_pool_users):
                    continue

                for view_user in avg_pool_users:
                    view_user.meta["input_qparams"] = {0: input_qparam}
                    view_user.meta["output_qparams"] = {0: input_qparam}
                    fx_helpers.refresh_node_fake_tensor(view_user)

                modified = True
                logger.info(
                    "[fast_preprocess] refreshed control-flow pool/view quant metadata for %s",
                    getattr(avg_pool, "name", "?"),
                )

        if modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow pool quant boundary rescale(s)",
            restored,
        )


def _restore_control_flow_unary_quant_boundaries(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    import fx_helpers

    def _is_relu_like(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return "aten.relu.default" in target_str or "aten.clamp.default" in target_str

    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        modified = False
        for placeholder in list(submodule.graph.nodes):
            if placeholder.op != "placeholder":
                continue

            input_qparam = _extract_single_qparam_entry(placeholder.meta.get("input_qparams"))
            if getattr(input_qparam, "dtype", None) not in (torch.int8, torch.int16):
                continue

            for unary_node in list(getattr(placeholder, "users", ())):
                if not _is_relu_like(unary_node):
                    continue

                existing_boundary = None
                for user in list(getattr(placeholder, "users", ())):
                    if not _is_tosa_rescale_node(user):
                        continue
                    if not getattr(user, "meta", {}).get(
                        "arm_preserve_control_flow_quant_boundary", False
                    ):
                        continue
                    if unary_node in getattr(user, "users", ()):
                        existing_boundary = user
                        break

                quant_boundary = existing_boundary
                if quant_boundary is None:
                    quant_boundary = fx_helpers.insert_rescale_node(
                        submodule,
                        placeholder,
                        placeholder,
                        input_qparam.dtype,
                        [1.0],
                        input_qparam.get_zp_per_tensor(),
                        input_qparam.get_zp_per_tensor(),
                        after=True,
                    )
                    quant_boundary.meta["input_qparams"] = {0: input_qparam}
                    quant_boundary.meta["output_qparams"] = {0: input_qparam}
                    quant_boundary.meta["arm_preserve_control_flow_quant_boundary"] = True
                    restored += 1
                    modified = True
                    logger.info(
                        "[fast_preprocess] restored control-flow unary quant boundary %s -> %s",
                        getattr(placeholder, "name", "?"),
                        getattr(quant_boundary, "name", "?"),
                    )

                unary_node.replace_input_with(placeholder, quant_boundary)
                unary_node.meta["input_qparams"] = {0: input_qparam}
                unary_node.meta["output_qparams"] = {0: input_qparam}
                fx_helpers.refresh_node_fake_tensor(unary_node)
                modified = True
                logger.info(
                    "[fast_preprocess] refreshed control-flow unary quant metadata for %s",
                    getattr(unary_node, "name", "?"),
                )

        if modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow unary quant boundary rescale(s)",
            restored,
        )


def _refresh_control_flow_binary_input_rescales(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs
    import fx_helpers

    def _rescale_output_qparam(node):
        if not _is_tosa_rescale_node(node):
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))
        if len(getattr(node, "args", ())) < 5:
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))
        output_dtype = node.args[1]
        if output_dtype not in (torch.int8, torch.int16, torch.int32):
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))

        source_node = node.args[0]
        if not hasattr(source_node, "op"):
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))
        source_qparam = _extract_effective_node_qparam(source_node)
        scale_arg = node.args[2]
        if (
            source_qparam is None
            or not isinstance(scale_arg, (list, tuple))
            or len(scale_arg) != 1
            or scale_arg[0] in (0, 0.0)
        ):
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))

        try:
            qinfo = torch.iinfo(output_dtype)
            return QuantArgs(
                scale=source_qparam.get_scale_per_tensor() / float(scale_arg[0]),
                zp=int(node.args[4]),
                qmin=int(qinfo.min),
                qmax=int(qinfo.max),
                dtype=output_dtype,
            )
        except Exception:
            return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))

    def _preferred_quantized_binary_output_rescale(source_node):
        if getattr(source_node, "op", None) != "call_function":
            return None
        target_str = str(getattr(source_node, "target", ""))
        if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
            return None

        best = None
        best_score = -1
        for user in list(getattr(source_node, "users", ())):
            if not _is_tosa_rescale_node(user):
                continue
            if len(getattr(user, "args", ())) < 5:
                continue
            out_dtype = user.args[1]
            if out_dtype not in (torch.int8, torch.int16):
                continue
            output_qparam = _rescale_output_qparam(user)
            if output_qparam is None:
                continue

            # Parse, don't validate: when a branch-local binary result is both
            # returned and reused by another binary op, the quantized branch
            # output contract is the stable source of truth. Prefer the rescale
            # edge that is already headed toward a branch output/dequantize path.
            score = 0
            for grand_user in list(getattr(user, "users", ())):
                if getattr(grand_user, "op", None) == "output":
                    score += 4
                elif _is_tosa_rescale_node(grand_user):
                    score += 2
            if score > best_score:
                best = (user, output_qparam)
                best_score = score

        return best

    def _clone_quantized_binary_output_rescale(submodule, ref_node, quant_rescale, source_node):
        if len(getattr(quant_rescale, "args", ())) < 5:
            return None

        cloned = fx_helpers.insert_rescale_node(
            submodule,
            ref_node,
            source_node,
            quant_rescale.args[1],
            list(quant_rescale.args[2]),
            int(quant_rescale.args[3]),
            int(quant_rescale.args[4]),
            after=False,
        )
        cloned.meta["input_qparams"] = dict(quant_rescale.meta.get("input_qparams") or {})
        reconstructed_qparam = _rescale_output_qparam(cloned)
        if reconstructed_qparam is not None:
            cloned.meta["output_qparams"] = {0: reconstructed_qparam}
        else:
            cloned.meta["output_qparams"] = dict(quant_rescale.meta.get("output_qparams") or {})
        return cloned

    def _int32_accumulator_qparam(scale: float):
        qinfo = torch.iinfo(torch.int32)
        return QuantArgs(
            scale=scale,
            zp=0,
            qmin=int(qinfo.min),
            qmax=int(qinfo.max),
            dtype=torch.int32,
        )

    def _binary_operand_qparam(arg):
        if not hasattr(arg, "op"):
            return None
        if (
            _is_tosa_rescale_node(arg)
            and len(getattr(arg, "args", ())) >= 5
            and arg.args[1] == torch.int32
            and hasattr(arg.args[0], "op")
        ):
            source_qparam = _rescale_output_qparam(arg.args[0])
            if source_qparam is None:
                source_qparam = _extract_effective_node_qparam(arg.args[0])
            if getattr(source_qparam, "dtype", None) in (torch.int8, torch.int16):
                return source_qparam
        source_qparam = _extract_effective_node_qparam(arg)
        if getattr(source_qparam, "dtype", None) in (torch.int8, torch.int16):
            return source_qparam
        return None

    def _same_qparam(lhs, rhs):
        if lhs is None or rhs is None:
            return False
        return (
            float(lhs.scale) == float(rhs.scale)
            and int(lhs.zp) == int(rhs.zp)
            and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
        )

    def _matching_quantized_rescale(source_node, desired_qparam):
        if not hasattr(source_node, "op") or desired_qparam is None:
            return None
        for user in list(getattr(source_node, "users", ())):
            if not _is_tosa_rescale_node(user):
                continue
            if len(getattr(user, "args", ())) < 5:
                continue
            if user.args[1] not in (torch.int8, torch.int16):
                continue
            user_qparam = _rescale_output_qparam(user)
            if _same_qparam(user_qparam, desired_qparam):
                return user
        return None

    refreshed = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        updated = False
        output_qparams_map = _infer_control_flow_output_qparams_map(control_flow_node)
        output_node = submodule.graph.output_node()
        raw_outputs = output_node.args[0]
        output_args = [raw_outputs] if hasattr(raw_outputs, "op") else list(raw_outputs)

        def _branch_output_target_qparam(candidate):
            for idx, output_arg in enumerate(output_args):
                current = output_arg
                seen = set()
                while hasattr(current, "op") and current not in seen:
                    if current is candidate:
                        return output_qparams_map.get(idx)
                    seen.add(current)
                    if (
                        _is_tosa_rescale_node(current)
                        and getattr(current, "args", None)
                        and hasattr(current.args[0], "op")
                    ):
                        current = current.args[0]
                        continue
                    break
            return None

        def _terminal_branch_output_qparam(candidate):
            target_qparam = _branch_output_target_qparam(candidate)
            if target_qparam is None:
                return None
            for user in list(getattr(candidate, "users", ())):
                if (
                    _is_tosa_rescale_node(user)
                    and len(getattr(user, "args", ())) >= 2
                    and user.args[1] in (torch.int8, torch.int16)
                ):
                    return None
            return target_qparam

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

                output_qparam = _extract_single_qparam_entry(arg.meta.get("output_qparams"))
                if output_qparam is None:
                    node_input_qparams = node.meta.get("input_qparams") or {}
                    output_qparam = _extract_single_qparam_entry(
                        {0: node_input_qparams.get(idx)}
                    )

                source_fake_val = source_node.meta.get("val")
                if getattr(source_fake_val, "dtype", None) == torch.int32:
                    preferred = _preferred_quantized_binary_output_rescale(source_node)
                    if preferred is not None:
                        quant_rescale, quant_qparam = preferred
                        sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                        sibling_source = sibling
                        if (
                            hasattr(sibling, "op")
                            and _is_tosa_rescale_node(sibling)
                            and len(getattr(sibling, "args", ())) >= 2
                            and sibling.args[1] == torch.int32
                            and hasattr(sibling.args[0], "op")
                        ):
                            sibling_source = sibling.args[0]
                        sibling_source_qparam = (
                            _extract_effective_node_qparam(sibling_source)
                            if hasattr(sibling_source, "op")
                            else None
                        )
                        cloned_quant_rescale = _clone_quantized_binary_output_rescale(
                            submodule, arg, quant_rescale, source_node
                        )
                        if cloned_quant_rescale is not None:
                            cloned_quant_rescale.meta["output_qparams"] = {0: quant_qparam}
                            for quant_user in list(getattr(quant_rescale, "users", ())):
                                quant_user.replace_input_with(
                                    quant_rescale, cloned_quant_rescale
                                )
                            if len(getattr(quant_rescale, "users", ())) == 0:
                                submodule.graph.erase_node(quant_rescale)
                            arg.update_arg(0, cloned_quant_rescale)
                            arg.meta["input_qparams"] = {0: quant_qparam}
                            source_node = cloned_quant_rescale
                            source_fake_val = source_node.meta.get("val")
                            if output_qparam is not None:
                                arg.update_arg(
                                    2,
                                    [
                                        quant_qparam.get_scale_per_tensor()
                                        / output_qparam.get_scale_per_tensor()
                                    ],
                                )
                                arg.update_arg(3, quant_qparam.get_zp_per_tensor())
                                arg.update_arg(4, output_qparam.get_zp_per_tensor())
                            updated = True
                            refreshed += 1
                            logger.info(
                                "[fast_preprocess] cloned quantized control-flow binary output %s for %s arg%d via %s",
                                getattr(source_node, "name", "?"),
                                getattr(node, "name", "?"),
                                idx,
                                getattr(quant_rescale, "name", "?"),
                            )
                            source_fake_val = source_node.meta.get("val")
                source_fake_val = source_node.meta.get("val")
                source_qparam = None
                sibling_align_qparam = None
                sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                if hasattr(source_node, "op"):
                    source_qparam = _extract_single_qparam_entry(
                        getattr(source_node, "meta", {}).get("input_qparams")
                    )
                    if source_qparam is None:
                        source_qparam = _extract_effective_node_qparam(source_node)
                if hasattr(sibling, "op"):
                    sibling_align_qparam = _extract_single_qparam_entry(
                        getattr(sibling, "meta", {}).get("input_qparams")
                    )
                current_output_qparam = _rescale_output_qparam(arg)
                if (
                    hasattr(source_node, "op")
                    and getattr(source_node, "op", None) == "placeholder"
                    and source_qparam is not None
                    and sibling_align_qparam is not None
                    and current_output_qparam is not None
                    and getattr(source_qparam, "dtype", None)
                    in (torch.int8, torch.int16)
                    and getattr(sibling_align_qparam, "dtype", None)
                    == getattr(source_qparam, "dtype", None)
                    and (
                        float(source_qparam.scale)
                        != float(sibling_align_qparam.scale)
                        or int(source_qparam.zp) != int(sibling_align_qparam.zp)
                    )
                ):
                    aligned_source = fx_helpers.insert_rescale_node(
                        submodule,
                        arg,
                        source_node,
                        sibling_align_qparam.dtype,
                        [
                            source_qparam.get_scale_per_tensor()
                            / sibling_align_qparam.get_scale_per_tensor()
                        ],
                        source_qparam.get_zp_per_tensor(),
                        sibling_align_qparam.get_zp_per_tensor(),
                        after=False,
                    )
                    aligned_source.meta["input_qparams"] = {0: source_qparam}
                    aligned_source.meta["output_qparams"] = {0: sibling_align_qparam}
                    arg.update_arg(0, aligned_source)
                    arg.update_arg(
                        2,
                        [
                            sibling_align_qparam.get_scale_per_tensor()
                            / current_output_qparam.get_scale_per_tensor()
                        ],
                    )
                    arg.update_arg(3, sibling_align_qparam.get_zp_per_tensor())
                    source_node = aligned_source
                    source_fake_val = source_node.meta.get("val")
                    source_qparam = sibling_align_qparam
                    refreshed += 1
                    updated = True
                    logger.info(
                        "[fast_preprocess] aligned control-flow binary input %s arg%d through sibling domain %s",
                        getattr(node, "name", "?"),
                        idx,
                        sibling_align_qparam,
                    )
                if getattr(source_fake_val, "dtype", None) == torch.int32:
                    desired_input_zp = 0
                else:
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

        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            if "aten.add.Tensor" not in str(getattr(node, "target", "")):
                continue
            if len(getattr(node, "args", ())) < 2:
                continue
            lhs, rhs = node.args[:2]
            if not (_is_tosa_rescale_node(lhs) and _is_tosa_rescale_node(rhs)):
                continue
            if len(getattr(lhs, "args", ())) < 5 or lhs.args[1] != torch.int32:
                continue
            lhs_source = lhs.args[0]
            if not hasattr(lhs_source, "op") or getattr(lhs_source, "op", None) != "placeholder":
                continue
            lhs_source_qparam = _extract_single_qparam_entry(
                getattr(lhs_source, "meta", {}).get("input_qparams")
            )
            rhs_input_qparam = _extract_single_qparam_entry(
                getattr(rhs, "meta", {}).get("input_qparams")
            )
            lhs_output_qparam = _rescale_output_qparam(lhs)
            if (
                lhs_source_qparam is None
                or rhs_input_qparam is None
                or lhs_output_qparam is None
                or getattr(lhs_source_qparam, "dtype", None)
                not in (torch.int8, torch.int16)
                or getattr(rhs_input_qparam, "dtype", None)
                != getattr(lhs_source_qparam, "dtype", None)
                or (
                    float(lhs_source_qparam.scale) == float(rhs_input_qparam.scale)
                    and int(lhs_source_qparam.zp) == int(rhs_input_qparam.zp)
                )
            ):
                continue

            aligned_source = fx_helpers.insert_rescale_node(
                submodule,
                lhs,
                lhs_source,
                rhs_input_qparam.dtype,
                [
                    lhs_source_qparam.get_scale_per_tensor()
                    / rhs_input_qparam.get_scale_per_tensor()
                ],
                lhs_source_qparam.get_zp_per_tensor(),
                rhs_input_qparam.get_zp_per_tensor(),
                after=False,
            )
            aligned_source.meta["input_qparams"] = {0: lhs_source_qparam}
            aligned_source.meta["output_qparams"] = {0: rhs_input_qparam}
            lhs.update_arg(0, aligned_source)
            lhs.update_arg(
                2,
                [
                    rhs_input_qparam.get_scale_per_tensor()
                    / lhs_output_qparam.get_scale_per_tensor()
                ],
            )
            lhs.update_arg(3, rhs_input_qparam.get_zp_per_tensor())
            refreshed += 1
            updated = True
            logger.info(
                    "[fast_preprocess] explicitly aligned add input %s through %s",
                    getattr(node, "name", "?"),
                    rhs_input_qparam,
                )

        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            target_str = str(getattr(node, "target", ""))
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue
            desired_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
            desired_output_qparam = _extract_single_qparam_entry(
                getattr(node, "meta", {}).get("output_qparams")
            )
            if not (
                getattr(desired_input_qparams.get(0), "dtype", None) == torch.int32
                and getattr(desired_input_qparams.get(1), "dtype", None) == torch.int32
                and getattr(desired_output_qparam, "dtype", None) == torch.int32
            ):
                continue

            for idx, arg in enumerate(node.args[:2]):
                source_node = arg
                source_qparam = _binary_operand_qparam(arg)
                if getattr(source_qparam, "dtype", None) not in (torch.int8, torch.int16):
                    continue
                desired_input_qparam = desired_input_qparams[idx]
                current_arg = node.args[idx]
                if (
                    _is_tosa_rescale_node(current_arg)
                    and len(getattr(current_arg, "args", ())) >= 5
                    and current_arg.args[1] == torch.int32
                    and hasattr(current_arg.args[0], "op")
                ):
                    lift = current_arg
                    source_node = current_arg.args[0]
                else:
                    lift = fx_helpers.insert_rescale_node(
                        submodule,
                        node,
                        source_node,
                        torch.int32,
                        [
                            source_qparam.get_scale_per_tensor()
                            / desired_input_qparam.get_scale_per_tensor()
                        ],
                        source_qparam.get_zp_per_tensor(),
                        desired_input_qparam.get_zp_per_tensor(),
                        after=False,
                    )
                    node.update_arg(idx, lift)
                    refreshed += 1
                    updated = True
                lift.update_arg(0, source_node)
                lift.update_arg(
                    2,
                    [
                        source_qparam.get_scale_per_tensor()
                        / desired_input_qparam.get_scale_per_tensor()
                    ],
                )
                lift.update_arg(3, source_qparam.get_zp_per_tensor())
                lift.update_arg(4, desired_input_qparam.get_zp_per_tensor())
                lift.meta["input_qparams"] = {0: source_qparam}
                lift.meta["output_qparams"] = {0: desired_input_qparam}

        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            target_str = str(getattr(node, "target", ""))
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue
            operand_qparams = {}
            for idx, arg in enumerate(node.args[:2]):
                operand_qparam = _binary_operand_qparam(arg)
                if operand_qparam is not None:
                    operand_qparams[idx] = operand_qparam
            if len(operand_qparams) != 2:
                continue
            lhs_qparam = operand_qparams[0]
            rhs_qparam = operand_qparams[1]
            if getattr(lhs_qparam, "dtype", None) != getattr(rhs_qparam, "dtype", None):
                continue
            if getattr(lhs_qparam, "dtype", None) not in (torch.int8, torch.int16):
                continue
            shift_bits = 12 if lhs_qparam.dtype == torch.int16 else 20
            accumulator_scale = (
                2.0
                * max(
                    float(lhs_qparam.get_scale_per_tensor()),
                    float(rhs_qparam.get_scale_per_tensor()),
                )
                / float(1 << shift_bits)
            )
            accumulator_qparam = _int32_accumulator_qparam(accumulator_scale)
            current_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
            current_output_qparam = _extract_single_qparam_entry(
                getattr(node, "meta", {}).get("output_qparams")
            )
            if (
                _same_qparam(current_input_qparams.get(0), accumulator_qparam)
                and _same_qparam(current_input_qparams.get(1), accumulator_qparam)
                and _same_qparam(current_output_qparam, accumulator_qparam)
            ):
                continue
            node.meta["input_qparams"] = {0: accumulator_qparam, 1: accumulator_qparam}
            node.meta["output_qparams"] = {0: accumulator_qparam}
            refreshed += 1
            updated = True
            logger.info(
                "[fast_preprocess] recomputed control-flow binary accumulator qparams for %s -> %s",
                getattr(node, "name", "?"),
                accumulator_qparam,
            )

            for user in list(getattr(node, "users", ())):
                if not _is_tosa_rescale_node(user):
                    continue
                if len(getattr(user, "args", ())) < 5:
                    continue
                if user.args[1] not in (torch.int8, torch.int16):
                    continue
                user_output_qparam = _branch_output_target_qparam(user)
                if user_output_qparam is None:
                    user_output_qparam = _extract_effective_node_qparam(user)
                if user_output_qparam is None:
                    user_output_qparam = _rescale_output_qparam(user)
                if getattr(user_output_qparam, "dtype", None) != user.args[1]:
                    continue
                user.update_arg(
                    2,
                    [
                        accumulator_qparam.get_scale_per_tensor()
                        / user_output_qparam.get_scale_per_tensor()
                    ],
                )
                user.update_arg(3, accumulator_qparam.get_zp_per_tensor())
                user.update_arg(4, user_output_qparam.get_zp_per_tensor())
                user.meta["input_qparams"] = {0: accumulator_qparam}
                user.meta["output_qparams"] = {0: user_output_qparam}
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] realigned control-flow binary output rescale %s from %s",
                    getattr(user, "name", "?"),
                    getattr(node, "name", "?"),
                )

        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            if "aten.mul.Tensor" not in str(getattr(node, "target", "")):
                continue

            operand_specs = {}
            for idx, arg in enumerate(node.args[:2]):
                source_node = arg
                source_qparam = None

                if (
                    _is_tosa_rescale_node(arg)
                    and len(getattr(arg, "args", ())) >= 5
                    and arg.args[1] == torch.int32
                    and hasattr(arg.args[0], "op")
                ):
                    source_node = arg.args[0]
                    source_qparam = _rescale_output_qparam(source_node)
                    if source_qparam is None:
                        source_qparam = _extract_effective_node_qparam(source_node)
                elif _is_tosa_rescale_node(arg) and len(getattr(arg, "args", ())) >= 5:
                    source_qparam = _rescale_output_qparam(arg)
                else:
                    if getattr(arg, "op", None) == "placeholder":
                        branch_qparam = _branch_local_control_flow_output_qparam(arg)
                        branch_source = _matching_quantized_rescale(arg, branch_qparam)
                        if branch_source is not None:
                            source_node = branch_source
                            source_qparam = branch_qparam
                    if source_qparam is None:
                        source_qparam = _extract_effective_node_qparam(arg)

                if getattr(source_qparam, "dtype", None) not in (torch.int8, torch.int16):
                    continue
                operand_specs[idx] = (source_node, source_qparam)

            if len(operand_specs) != 2:
                continue

            lhs_source, lhs_qparam = operand_specs[0]
            rhs_source, rhs_qparam = operand_specs[1]
            if getattr(lhs_qparam, "dtype", None) != getattr(rhs_qparam, "dtype", None):
                continue

            lhs_int32_qparam = _int32_accumulator_qparam(lhs_qparam.get_scale_per_tensor())
            rhs_int32_qparam = _int32_accumulator_qparam(rhs_qparam.get_scale_per_tensor())
            output_qparam = _int32_accumulator_qparam(
                lhs_qparam.get_scale_per_tensor() * rhs_qparam.get_scale_per_tensor()
            )
            desired_input_qparams = {0: lhs_int32_qparam, 1: rhs_int32_qparam}

            current_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
            current_output_qparam = _extract_single_qparam_entry(
                getattr(node, "meta", {}).get("output_qparams")
            )
            if not (
                _same_qparam(current_input_qparams.get(0), lhs_int32_qparam)
                and _same_qparam(current_input_qparams.get(1), rhs_int32_qparam)
                and _same_qparam(current_output_qparam, output_qparam)
            ):
                node.meta["input_qparams"] = desired_input_qparams
                node.meta["output_qparams"] = {0: output_qparam}
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] recomputed control-flow mul qparams for %s -> lhs=%s rhs=%s out=%s",
                    getattr(node, "name", "?"),
                    lhs_int32_qparam,
                    rhs_int32_qparam,
                    output_qparam,
                )

            for idx, (source_node, source_qparam) in operand_specs.items():
                current_arg = node.args[idx]
                desired_input_qparam = desired_input_qparams[idx]
                if (
                    _is_tosa_rescale_node(current_arg)
                    and len(getattr(current_arg, "args", ())) >= 5
                    and current_arg.args[1] == torch.int32
                ):
                    lift = current_arg
                    lift.update_arg(0, source_node)
                else:
                    lift = fx_helpers.insert_rescale_node(
                        submodule,
                        node,
                        source_node,
                        torch.int32,
                        [
                            source_qparam.get_scale_per_tensor()
                            / desired_input_qparam.get_scale_per_tensor()
                        ],
                        source_qparam.get_zp_per_tensor(),
                        desired_input_qparam.get_zp_per_tensor(),
                        after=False,
                    )
                    node.update_arg(idx, lift)
                    refreshed += 1
                    updated = True
                lift.update_arg(
                    2,
                    [
                        source_qparam.get_scale_per_tensor()
                        / desired_input_qparam.get_scale_per_tensor()
                    ],
                )
                lift.update_arg(3, source_qparam.get_zp_per_tensor())
                lift.update_arg(4, desired_input_qparam.get_zp_per_tensor())
                lift.meta["input_qparams"] = {0: source_qparam}
                lift.meta["output_qparams"] = {0: desired_input_qparam}

            for user in list(getattr(node, "users", ())):
                if not _is_tosa_rescale_node(user):
                    continue
                if len(getattr(user, "args", ())) < 5:
                    continue
                if user.args[1] not in (torch.int8, torch.int16):
                    continue
                user_output_qparam = _terminal_branch_output_qparam(user)
                if user_output_qparam is None:
                    user_output_qparam = _extract_effective_node_qparam(user)
                if user_output_qparam is None:
                    user_output_qparam = _rescale_output_qparam(user)
                if getattr(user_output_qparam, "dtype", None) != user.args[1]:
                    continue
                user.update_arg(
                    2,
                    [
                        output_qparam.get_scale_per_tensor()
                        / user_output_qparam.get_scale_per_tensor()
                    ],
                )
                user.update_arg(3, output_qparam.get_zp_per_tensor())
                user.update_arg(4, user_output_qparam.get_zp_per_tensor())
                user.meta["input_qparams"] = {0: output_qparam}
                user.meta["output_qparams"] = {0: user_output_qparam}
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] realigned control-flow mul output rescale %s from %s",
                    getattr(user, "name", "?"),
                    getattr(node, "name", "?"),
                )

        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            target_str = str(getattr(node, "target", ""))
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue
            node_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
            for idx, arg in enumerate(node.args[:2]):
                if not _is_tosa_rescale_node(arg):
                    continue
                if len(getattr(arg, "args", ())) < 5 or arg.args[1] != torch.int32:
                    continue
                desired_output_qparam = _extract_single_qparam_entry({0: node_input_qparams.get(idx)})
                source_node = arg.args[0]
                if not hasattr(source_node, "op"):
                    continue
                if (
                    _is_tosa_rescale_node(source_node)
                    and len(getattr(source_node, "args", ())) >= 5
                    and source_node.args[1] in (torch.int8, torch.int16)
                    and hasattr(source_node.args[0], "op")
                    and getattr(source_node.args[0], "op", None) == "placeholder"
                ):
                    placeholder_source = source_node.args[0]
                    align_qparam = _rescale_output_qparam(source_node)
                    branch_qparam = _branch_local_control_flow_output_qparam(
                        placeholder_source
                    )
                    if (
                        align_qparam is not None
                        and branch_qparam is not None
                        and (
                            float(align_qparam.scale) != float(branch_qparam.scale)
                            or int(align_qparam.zp) != int(branch_qparam.zp)
                        )
                    ):
                        arg.update_arg(0, placeholder_source)
                        if len(getattr(source_node, "users", ())) == 0:
                            submodule.graph.erase_node(source_node)
                        source_node = placeholder_source
                        updated = True
                        logger.info(
                            "[fast_preprocess] collapsed non-branch-local binary aligner %s before %s arg%d",
                            getattr(source_node, "name", "?"),
                            getattr(node, "name", "?"),
                            idx,
                        )
                source_qparam = _extract_effective_node_qparam(source_node)
                if (
                    source_qparam is None
                    or desired_output_qparam is None
                    or getattr(source_qparam, "dtype", None)
                    not in (torch.int8, torch.int16)
                    or getattr(desired_output_qparam, "dtype", None) != torch.int32
                ):
                    continue
                desired_scale = (
                    source_qparam.get_scale_per_tensor()
                    / desired_output_qparam.get_scale_per_tensor()
                )
                current_scale = arg.args[2] if len(arg.args) > 2 else None
                current_input_zp = arg.args[3] if len(arg.args) > 3 else None
                current_output_zp = arg.args[4] if len(arg.args) > 4 else None
                if (
                    current_scale == [desired_scale]
                    and current_input_zp == source_qparam.get_zp_per_tensor()
                    and current_output_zp == desired_output_qparam.get_zp_per_tensor()
                ):
                    continue
                arg.update_arg(2, [desired_scale])
                arg.update_arg(3, source_qparam.get_zp_per_tensor())
                arg.update_arg(4, desired_output_qparam.get_zp_per_tensor())
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] realigned control-flow binary lift %s arg%d to node input qparams %s",
                    getattr(node, "name", "?"),
                    idx,
                    desired_output_qparam,
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
            immediate_int32_lift = _placeholder_immediately_lifts_to_int32(placeholder)
            scalar_binary_after_lift = _placeholder_int32_lift_feeds_scalar_binary(
                placeholder
            )
            feeds_binary_directly = _placeholder_feeds_binary_directly(placeholder)
            if _control_flow_debug_enabled():
                print(
                    "[fast_preprocess] restore input users",
                    getattr(placeholder, "name", "?"),
                    "users=",
                    [
                        (
                            getattr(user, "name", "?"),
                            str(getattr(user, "target", "?")),
                            getattr(user, "args", (None, None))[1]
                            if len(getattr(user, "args", ())) >= 2
                            else None,
                        )
                        for user in list(getattr(placeholder, "users", ()))
                        if hasattr(user, "op")
                    ],
                    "immediate_int32_lift=",
                    immediate_int32_lift,
                    "scalar_binary_after_lift=",
                    scalar_binary_after_lift,
                    "feeds_binary_directly=",
                    feeds_binary_directly,
                )
            effective_output_qparam = _branch_local_control_flow_output_qparam(
                placeholder
            )
            if (
                effective_output_qparam is None
                and feeds_binary_directly
            ):
                effective_output_qparam = qparam
            _log_control_flow_debug(
                "[fast_preprocess] restore submodule input qparams %s %s idx=%d incoming=%s branch_output_qparams=%s",
                getattr(submodule, "__class__", type(submodule)).__name__,
                getattr(placeholder, "name", "?"),
                idx,
                qparam,
                effective_output_qparam,
            )
            if _control_flow_debug_enabled():
                print(
                    "[fast_preprocess] restore input qparams",
                    getattr(placeholder, "name", "?"),
                    "idx=",
                    idx,
                    "incoming=",
                    qparam,
                    "branch_output=",
                    effective_output_qparam,
                    "existing_output=",
                    placeholder.meta.get("output_qparams"),
                )
            # Parse, don't validate: the placeholder boundary straddles two
            # domains. Its incoming edge lives in the parent graph's qparam
            # domain, while its outgoing branch-local uses still need the
            # saved submodule qparams so InsertControlFlowRescalesPass can
            # materialize the explicit entry RESCALE between them. If we stamp
            # both sides with the parent qparams here, that pass sees
            # in_qparams == out_qparams and silently skips the branch-entry
            # rescale.
            if effective_output_qparam is None:
                effective_output_qparam = qparam

            placeholder.meta["input_qparams"] = {0: qparam}
            placeholder.meta["output_qparams"] = {0: effective_output_qparam}
            placeholder.meta["arm_control_flow_saved_output_qparams"] = {
                0: effective_output_qparam
            }
            fake_val = placeholder.meta.get("val")
            preserves_float_boundary = any(
                _is_quantize_node(user) for user in list(getattr(placeholder, "users", ()))
            )
            if (
                not preserves_float_boundary
                and
                hasattr(fake_val, "to")
                and getattr(fake_val, "dtype", None) != effective_output_qparam.dtype
            ):
                placeholder.meta["val"] = fake_val.to(dtype=effective_output_qparam.dtype)

    def _annotate_submodule_input_rescales(submodule, input_qparams_map) -> None:
        if not input_qparams_map:
            return
        placeholders = [node for node in submodule.graph.nodes if node.op == "placeholder"]
        for idx, input_qparam in input_qparams_map.items():
            if idx >= len(placeholders):
                continue
            placeholder = placeholders[idx]
            effective_output_qparam = _branch_local_control_flow_output_qparam(
                placeholder
            )
            if effective_output_qparam is None:
                continue

            for user in list(getattr(placeholder, "users", ())):
                if not _is_tosa_rescale_node(user):
                    continue
                if (
                    len(getattr(user, "args", ())) >= 2
                    and user.args[1] != effective_output_qparam.dtype
                ):
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

    def _has_submodule_output_input_qparams(submodule) -> bool:
        output_node = submodule.graph.output_node()
        input_qparams = output_node.meta.get("input_qparams")
        return bool(input_qparams)

    def _same_qparam(lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        return (
            float(lhs.scale) == float(rhs.scale)
            and int(lhs.zp) == int(rhs.zp)
            and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
        )

    def _quantized_rescale_chain_depth(node) -> int:
        depth = 0
        current = node
        seen = set()
        while hasattr(current, "op") and current not in seen:
            if not _is_tosa_rescale_node(current):
                break
            if len(getattr(current, "args", ())) < 2:
                break
            if current.args[1] not in (torch.int8, torch.int16):
                break
            seen.add(current)
            depth += 1
            if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
                break
            current = current.args[0]
        return depth

    def _existing_output_rescale_qparams(submodule) -> dict[int, QuantArgs]:
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        existing_qparams: dict[int, QuantArgs] = {}
        for idx, output_arg in enumerate(output_args):
            qparam = _extract_effective_node_qparam(output_arg)
            if qparam is not None:
                existing_qparams[idx] = qparam
        return existing_qparams

    def _filter_output_qparams_map(submodule, output_qparams_map):
        if not output_qparams_map:
            return {}
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        existing_qparams = _existing_output_rescale_qparams(submodule)
        filtered: dict[int, QuantArgs] = {}
        for idx, target_qparam in output_qparams_map.items():
            if idx >= len(output_args):
                continue
            output_arg = output_args[idx]
            current_qparam = existing_qparams.get(idx)
            if (
                _quantized_rescale_chain_depth(output_arg) >= 2
                and current_qparam is not None
                and _same_qparam(current_qparam, target_qparam)
            ):
                continue
            filtered[idx] = target_qparam
        return filtered

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
                if_output_qparams_map = _filter_output_qparams_map(
                    if_graph, output_qparams_map
                )
                else_output_qparams_map = _filter_output_qparams_map(
                    else_graph, output_qparams_map
                )
                if _has_submodule_output_input_qparams(if_graph) and if_output_qparams_map:
                    modified |= rescale_pass._rescale_submodule_outputs(
                        if_graph, if_output_qparams_map
                    )
                if (
                    _has_submodule_output_input_qparams(else_graph)
                    and else_output_qparams_map
                ):
                    modified |= rescale_pass._rescale_submodule_outputs(
                        else_graph, else_output_qparams_map
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
                body_output_qparams_map = _filter_output_qparams_map(
                    body_graph, output_qparams_map
                )
                if _has_submodule_output_input_qparams(body_graph) and body_output_qparams_map:
                    modified |= rescale_pass._rescale_submodule_outputs(
                        body_graph, body_output_qparams_map
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
    import math
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _canonicalize_rescale_arg(arg):
        if isinstance(arg, float):
            return round(arg, 12)
        if isinstance(arg, (list, tuple)):
            return tuple(_canonicalize_rescale_arg(item) for item in arg)
        return repr(arg)

    def _rescale_signature(node):
        if not _is_tosa_rescale_node(node):
            return None
        args = getattr(node, "args", ())
        if len(args) < 5:
            return None
        return tuple(_canonicalize_rescale_arg(arg) for arg in args[1:5])

    def _is_identity_rescale(node):
        if not _is_tosa_rescale_node(node):
            return False
        if node.meta.get("arm_preserve_control_flow_quant_boundary", False):
            return False
        args = getattr(node, "args", ())
        if len(args) < 5:
            return False

        dtype = args[1]
        if dtype not in (torch.int8, torch.int16):
            return False

        scale_arg = args[2]
        scales = scale_arg if isinstance(scale_arg, (list, tuple)) else [scale_arg]
        try:
            scale_is_identity = all(
                math.isclose(float(scale), 1.0, rel_tol=0.0, abs_tol=1e-12)
                for scale in scales
            )
            input_zp = int(args[3])
            output_zp = int(args[4])
        except (TypeError, ValueError):
            return False

        if not scale_is_identity or input_zp != output_zp:
            return False

        input_qparam = _extract_single_qparam_entry(node.meta.get("input_qparams"))
        output_qparam = _extract_single_qparam_entry(node.meta.get("output_qparams"))
        if input_qparam is not None and output_qparam is not None:
            return _same_control_flow_qparam(input_qparam, output_qparam)
        return True

    pruned = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        updated = False
        for node in list(submodule.graph.nodes):
            if not _is_tosa_rescale_node(node):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue
            source = node.args[0]
            if _is_identity_rescale(node):
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
                    "[fast_preprocess] pruned identity control-flow rescale %s -> %s",
                    getattr(node, "name", "?"),
                    getattr(source, "name", "?"),
                )
                continue
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

    def _constant_tensor_from_node(submodule, source) -> torch.Tensor | None:
        if getattr(source, "op", None) == "get_attr":
            constant_value = getattr(submodule, source.target, None)
            if not isinstance(constant_value, torch.Tensor):
                return None
            if constant_value.is_floating_point():
                return constant_value

            source_qparam = _extract_single_qparam_entry(
                source.meta.get("output_qparams")
            ) or _extract_single_qparam_entry(source.meta.get("input_qparams"))
            if source_qparam is None or source_qparam.per_channel:
                return None

            scale = source_qparam.get_scale_per_tensor()
            if scale == 0:
                return None
            return (constant_value.detach().to(torch.float32) - int(source_qparam.zp)) * float(
                scale
            )

        target_str = str(getattr(source, "target", ""))
        if "aten.full.default" not in target_str:
            return None

        if len(getattr(source, "args", ())) < 2:
            return None

        size_arg = source.args[0]
        fill_value = source.args[1]
        if not isinstance(size_arg, (list, tuple)):
            return None

        try:
            shape = tuple(int(dim) for dim in size_arg)
            fill_scalar = float(fill_value)
        except (TypeError, ValueError):
            return None

        return torch.full(shape, fill_scalar, dtype=torch.float32)

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
            ):
                source = output_arg.args[0]

            constant_value = _constant_tensor_from_node(submodule, source)
            if constant_value is None:
                continue

            quantized_value = _quantize_tensor(constant_value, qparam)
            if quantized_value is None:
                continue

            const_name = getattr(source, "target", None)
            if getattr(source, "op", None) != "get_attr":
                const_name = f"_arm_cf_const_output_{idx}"
                suffix = idx
                while hasattr(submodule, const_name):
                    suffix += 1
                    const_name = f"_arm_cf_const_output_{suffix}"
                submodule.register_buffer(const_name, quantized_value)
                with submodule.graph.inserting_before(output_node):
                    const_node = submodule.graph.get_attr(const_name)
                const_node.meta["val"] = quantized_value
                if source.meta.get("tosa_dim_order") is not None:
                    const_node.meta["tosa_dim_order"] = tuple(source.meta["tosa_dim_order"])
                source = const_node
            else:
                submodule.register_buffer(const_name, quantized_value)
                source.meta["val"] = quantized_value

            source.meta["input_qparams"] = {0: qparam}
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


def _fold_zero_input_full_in_control_flow_submodules(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _build_full_const(node) -> torch.Tensor | None:
        if "aten.full.default" not in str(getattr(node, "target", "")):
            return None
        args = tuple(getattr(node, "args", ()))
        if len(args) < 2:
            return None
        shape_arg, fill_value = args[0], args[1]
        if not isinstance(shape_arg, (list, tuple)):
            return None
        try:
            shape = tuple(int(dim) for dim in shape_arg)
            fill_scalar = float(fill_value)
        except (TypeError, ValueError):
            return None

        fake_val = getattr(node, "meta", {}).get("val")
        dtype = getattr(fake_val, "dtype", torch.float32)
        return torch.full(shape, fill_scalar, dtype=dtype)

    def _infer_control_flow_full_qparam(node):
        # Parse, don't validate: scalar FULL nodes inside cond branches usually
        # participate in a binary op with the captured branch tensor. When the
        # FULL node lost its own qparam annotation during earlier rewrites,
        # recover the branch-local quantized contract from nearby quantized
        # structure instead of freezing an FP32 const and hoping later
        # serializer logic repairs it.
        for user in list(getattr(node, "users", ())):
            qparam = _extract_effective_node_qparam(user)
            if qparam is not None and not getattr(qparam, "per_channel", False):
                return qparam

            if getattr(user, "op", None) != "call_function":
                continue
            target_str = str(getattr(user, "target", ""))
            if "aten.add.Tensor" not in target_str and "aten.sub.Tensor" not in target_str:
                continue
            for arg in user.args[:2]:
                if arg is node or not hasattr(arg, "op"):
                    continue
                qparam = _extract_effective_node_qparam(arg)
                if qparam is not None and not getattr(qparam, "per_channel", False):
                    return qparam
        return None

    def _feeds_binary_downstream(node) -> bool:
        stack = list(getattr(node, "users", ()))
        seen = set()
        while stack:
            user = stack.pop()
            if id(user) in seen:
                continue
            seen.add(id(user))
            target_str = str(getattr(user, "target", ""))
            if "aten.add.Tensor" in target_str or "aten.sub.Tensor" in target_str:
                return True
            if (
                _is_quantize_node(user)
                or _is_dequantize_node(user)
                or _is_tosa_rescale_node(user)
            ):
                stack.extend(list(getattr(user, "users", ())))
        return False

    rewritten = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        placeholders = [n for n in submodule.graph.nodes if n.op == "placeholder"]
        output_qparams_map = _infer_control_flow_output_qparams_map(control_flow_node)
        output_node = submodule.graph.output_node()
        output_arg0 = output_node.args[0]
        wrapped_single = hasattr(output_arg0, "op")
        output_args = [output_arg0] if wrapped_single else list(output_arg0)
        updated = False
        for node in list(submodule.graph.nodes):
            if getattr(node, "op", None) != "call_function":
                continue
            if list(getattr(node, "all_input_nodes", ())):
                continue

            const_value = _build_full_const(node)
            if const_value is None:
                continue
            if _feeds_binary_downstream(node):
                continue

            output_qparams = None
            if not placeholders and output_qparams_map:
                for idx, output_arg in enumerate(output_args):
                    if output_arg is node:
                        output_qparams = output_qparams_map.get(idx)
                        break
            if output_qparams is None:
                output_qparams = _extract_single_qparam_entry(node.meta.get("output_qparams"))
            if output_qparams is None:
                output_qparams = _infer_control_flow_full_qparam(node)
            if output_qparams is not None and not getattr(output_qparams, "per_channel", False):
                const_value = output_qparams.quantize_value(const_value)

            const_name = f"{getattr(node, 'name', 'full')}_fused_const"
            suffix = 0
            while hasattr(submodule, const_name):
                suffix += 1
                const_name = f"{getattr(node, 'name', 'full')}_fused_const_{suffix}"

            submodule.register_buffer(const_name, const_value)
            with submodule.graph.inserting_before(node):
                const_node = submodule.graph.get_attr(const_name)

            const_node.meta = node.meta.copy()
            const_node.meta["val"] = const_value
            if output_qparams is not None:
                const_node.meta["input_qparams"] = {0: output_qparams}
                const_node.meta["output_qparams"] = {0: output_qparams}
            node.replace_all_uses_with(const_node)
            submodule.graph.erase_node(node)
            updated = True
            rewritten += 1

        if updated:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] folded %d zero-input FULL node(s) inside control-flow submodules",
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


def _is_quantize_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return "quantize_per_tensor" in target_str and "dequantize_per_tensor" not in target_str


def _is_control_flow_input_passthrough_node(node) -> bool:
    import torch

    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    if (
        "tosa_RESCALE" in target_str
        or "tosa.RESCALE" in target_str
        or "TosaRescale" in target_str
    ):
        if len(getattr(node, "args", ())) >= 2 and node.args[1] == torch.int32:
            return False
        input_qparam = _extract_single_qparam_entry(node.meta.get("input_qparams"))
        output_qparam = _extract_single_qparam_entry(node.meta.get("output_qparams"))
        if input_qparam is not None and output_qparam is not None:
            input_zp = int(input_qparam.get_zp_per_tensor())
            output_zp = int(output_qparam.get_zp_per_tensor())
            input_scale = float(input_qparam.get_scale_per_tensor())
            output_scale = float(output_qparam.get_scale_per_tensor())
            input_dtype = getattr(input_qparam, "dtype", None)
            output_dtype = getattr(output_qparam, "dtype", None)
            if (
                input_zp != output_zp
                or input_scale != output_scale
                or input_dtype != output_dtype
            ):
                return False
        return True
    return (
        "dequantize_per_tensor" in target_str
    )


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

    def _infer_output_dim_order(node):
        current = node
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))
            dim_order = current.meta.get("tosa_dim_order")
            if dim_order is not None:
                return tuple(dim_order)
            next_arg = None
            for maybe_arg in getattr(current, "args", ()):
                if hasattr(maybe_arg, "op"):
                    next_arg = maybe_arg
                    break
            current = next_arg

        fake_val = node.meta.get("val")
        if hasattr(fake_val, "dim") and fake_val.dim() == 4:
            return (0, 2, 3, 1)
        return None

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
            desired_fake_tensor = _get_control_flow_output_fake_tensor(submodule_user)
            desired_dim_order = submodule_user.meta.get("tosa_dim_order")
            if desired_dim_order is None:
                desired_dim_order = _infer_output_dim_order(submodule_output)
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

            current_fake_tensor = submodule_output.meta.get("val")
            current_dim_order = submodule_output.meta.get("tosa_dim_order")
            if desired_dim_order is not None and tuple(desired_dim_order) != tuple(
                current_dim_order or ()
            ):
                submodule_output.meta["tosa_dim_order"] = tuple(desired_dim_order)
                current_dim_order = submodule_output.meta.get("tosa_dim_order")
                updated = True
                print(
                    "[fast_preprocess] stamped control-flow output dim_order",
                    getattr(submodule_output, "name", "?"),
                    current_dim_order,
                )
            if desired_fake_tensor is not None and (
                getattr(current_fake_tensor, "shape", None)
                != getattr(desired_fake_tensor, "shape", None)
                or getattr(current_fake_tensor, "dtype", None)
                != getattr(desired_fake_tensor, "dtype", None)
                or (
                    desired_dim_order is not None
                    and tuple(desired_dim_order) != tuple(current_dim_order or ())
                )
            ):
                submodule_output.meta["val"] = desired_fake_tensor
                if desired_dim_order is not None:
                    submodule_output.meta["tosa_dim_order"] = tuple(desired_dim_order)
                updated = True
                logger.info(
                    "[fast_preprocess] restamped control-flow output %s -> shape=%s dtype=%s dim_order=%s",
                    getattr(submodule_output, "name", "?"),
                    tuple(desired_fake_tensor.shape),
                    getattr(desired_fake_tensor, "dtype", None),
                    desired_dim_order,
                )

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
    import torch

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

    def _has_submodule_output_input_qparams(submodule) -> bool:
        output_node = submodule.graph.output_node()
        return bool(output_node.meta.get("input_qparams"))

    def _current_output_already_quantized(submodule, idx: int) -> bool:
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        if idx >= len(output_args):
            return False
        output_arg = output_args[idx]
        return (
            _is_tosa_rescale_node(output_arg)
            and len(getattr(output_arg, "args", ())) >= 2
            and output_arg.args[1] in (torch.int8, torch.int16)
        )

    rescale_pass = InsertControlFlowRescalesPass()
    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        output_qparams_map = submodule.graph.output_node().meta.get(
            "arm_control_flow_output_dequant_qparams", {}
        )
        if not output_qparams_map:
            continue
        output_qparams_map = {
            idx: qparam
            for idx, qparam in output_qparams_map.items()
            if not _current_output_already_quantized(submodule, idx)
        }
        if not output_qparams_map:
            continue
        _refresh_submodule_output_input_qparams(submodule)
        if not _has_submodule_output_input_qparams(submodule):
            continue
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


def _force_control_flow_rank4_output_dim_order(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    desired_dim_order = (0, 2, 3, 1)
    forced = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        output_node = submodule.graph.output_node()
        outputs = output_node.args[0]
        output_args = [outputs] if hasattr(outputs, "op") else list(outputs)
        for output_arg in output_args:
            fake_val = output_arg.meta.get("val")
            if not hasattr(fake_val, "dim") or fake_val.dim() != 4:
                continue
            current_dim_order = output_arg.meta.get("tosa_dim_order")
            if tuple(current_dim_order or ()) == desired_dim_order:
                continue
            output_arg.meta["tosa_dim_order"] = desired_dim_order
            forced += 1
            print(
                "[fast_preprocess] forced control-flow output dim_order",
                getattr(output_arg, "name", "?"),
                output_arg.meta["tosa_dim_order"],
            )
        if forced:
            submodule.recompile()

    if forced:
        logger.info(
            "[fast_preprocess] forced NHWC dim-order on %d control-flow rank-4 output(s)",
            forced,
        )


def _annotate_control_flow_rank4_dim_order(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    desired_dim_order = (0, 2, 3, 1)
    annotated = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        submodule_modified = False
        for node in submodule.graph.nodes:
            if node.op != "call_function":
                continue
            if node.meta.get("tosa_dim_order") is not None:
                continue
            fake_val = node.meta.get("val")
            if not hasattr(fake_val, "dim") or fake_val.dim() != 4:
                continue
            node.meta["tosa_dim_order"] = desired_dim_order
            annotated += 1
            submodule_modified = True
        if submodule_modified:
            submodule.recompile()

    if annotated:
        logger.info(
            "[fast_preprocess] annotated NHWC dim-order on %d control-flow interior rank-4 node(s)",
            annotated,
        )


def _same_control_flow_qparam(lhs, rhs) -> bool:
    if lhs is None or rhs is None:
        return False
    return (
        float(lhs.get_scale_per_tensor()) == float(rhs.get_scale_per_tensor())
        and int(lhs.get_zp_per_tensor()) == int(rhs.get_zp_per_tensor())
        and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
    )


def _restore_missing_control_flow_input_rescales(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    import fx_helpers

    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        submodule_modified = False
        for placeholder in list(submodule.graph.nodes):
            if placeholder.op != "placeholder":
                continue

            input_qparam = _extract_single_qparam_entry(placeholder.meta.get("input_qparams"))
            output_qparam = _branch_local_control_flow_output_qparam(placeholder)
            if os.environ.get("TOSA_ENGINE_DEBUG"):
                print(
                    "[cf-entry-rescale-check]",
                    getattr(placeholder, "name", "?"),
                    "input_qparam=",
                    input_qparam,
                    "output_qparam=",
                    output_qparam,
                    "users=",
                    [getattr(user, "name", "?") for user in list(getattr(placeholder, "users", ()))],
                )
            if input_qparam is None or output_qparam is None:
                continue
            if _same_control_flow_qparam(input_qparam, output_qparam):
                continue

            users = [user for user in list(placeholder.users) if hasattr(user, "op")]
            if not users:
                continue

            has_matching_entry_rescale = False
            for user in users:
                if not _is_tosa_rescale_node(user):
                    continue
                if len(getattr(user, "args", ())) >= 2 and user.args[1] != output_qparam.dtype:
                    continue
                user_input_qparam = _extract_single_qparam_entry(
                    user.meta.get("input_qparams")
                )
                user_output_qparam = _extract_single_qparam_entry(
                    user.meta.get("output_qparams")
                )
                if _same_control_flow_qparam(user_input_qparam, input_qparam) and _same_control_flow_qparam(
                    user_output_qparam, output_qparam
                ):
                    has_matching_entry_rescale = True
                    break
            if has_matching_entry_rescale:
                continue

            if os.environ.get("TOSA_ENGINE_DEBUG"):
                print(
                    "[cf-entry-rescale-insert]",
                    getattr(placeholder, "name", "?"),
                    "from",
                    input_qparam,
                    "to",
                    output_qparam,
                )
            rescale = fx_helpers.insert_rescale_node(
                submodule,
                placeholder,
                placeholder,
                output_qparam.dtype,
                [
                    input_qparam.get_scale_per_tensor()
                    / output_qparam.get_scale_per_tensor()
                ],
                input_qparam.get_zp_per_tensor(),
                output_qparam.get_zp_per_tensor(),
                after=True,
            )
            rescale.meta["input_qparams"] = {0: input_qparam}
            rescale.meta["output_qparams"] = {0: output_qparam}

            for user in users:
                if user is rescale:
                    continue
                user.replace_input_with(placeholder, rescale)

            restored += 1
            submodule_modified = True

        if submodule_modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if restored:
        logger.info(
            "[fast_preprocess] restored %d missing control-flow input boundary rescale(s)",
            restored,
        )


def _rewrite_control_flow_quantize_nodes(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs

    import fx_helpers

    def _scale_value(qparam) -> float:
        scale = getattr(qparam, "scale", 1.0)
        if isinstance(scale, (list, tuple)):
            return float(scale[0]) if scale else 1.0
        return float(scale)

    def _zp_value(qparam) -> int:
        zp = getattr(qparam, "zp", getattr(qparam, "zero_point", 0))
        if isinstance(zp, (list, tuple)):
            return int(zp[0]) if zp else 0
        return int(zp)

    def _same_qparam(lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        return (
            _scale_value(lhs) == _scale_value(rhs)
            and _zp_value(lhs) == _zp_value(rhs)
            and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
            and getattr(lhs, "qmin", None) == getattr(rhs, "qmin", None)
            and getattr(lhs, "qmax", None) == getattr(rhs, "qmax", None)
        )

    def _single_qparam_entry(qparams):
        if not qparams:
            return None
        if isinstance(qparams, dict):
            if 0 in qparams:
                return qparams[0]
            if len(qparams) == 1:
                return next(iter(qparams.values()))
        return qparams

    def _parse_per_tensor_qparam(node):
        if not getattr(node, "args", None) or len(node.args) < 6:
            return None
        if not isinstance(node.args[5], torch.dtype):
            return None
        return QuantArgs(
            scale=float(node.args[1]),
            zp=int(node.args[2]),
            qmin=int(node.args[3]),
            qmax=int(node.args[4]),
            dtype=node.args[5],
            axis=0,
            per_channel=False,
        )

    def _build_full_const(node) -> torch.Tensor | None:
        if "aten.full.default" not in str(getattr(node, "target", "")):
            return None
        args = tuple(getattr(node, "args", ()))
        if len(args) < 2:
            return None
        shape_arg, fill_value = args[0], args[1]
        if not isinstance(shape_arg, (list, tuple)):
            return None
        try:
            shape = tuple(int(dim) for dim in shape_arg)
            fill_scalar = float(fill_value)
        except (TypeError, ValueError):
            return None

        fake_val = getattr(node, "meta", {}).get("val")
        dtype = getattr(fake_val, "dtype", torch.float32)
        return torch.full(shape, fill_scalar, dtype=dtype)

    def _is_zero_input_const_source(node) -> bool:
        if not hasattr(node, "op"):
            return False
        if node.op == "get_attr":
            return True
        if node.op != "call_function":
            return False
        return not list(getattr(node, "all_input_nodes", ()))

    def _feeds_scalar_binary(node) -> bool:
        stack = list(getattr(node, "users", ()))
        seen = set()
        while stack:
            user = stack.pop()
            if id(user) in seen:
                continue
            seen.add(id(user))
            target_str = str(getattr(user, "target", ""))
            if "aten.add.Tensor" in target_str or "aten.sub.Tensor" in target_str:
                return True
            if _is_tosa_rescale_node(user):
                stack.extend(list(getattr(user, "users", ())))
        return False

    rewritten = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        modified = False
        for node in list(submodule.graph.nodes):
            if _is_dequantize_node(node):
                if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                    continue
                quant_node = node.args[0]
                if not _is_quantize_node(quant_node):
                    continue
                quant_qparam = _parse_per_tensor_qparam(quant_node)
                dequant_qparam = _parse_per_tensor_qparam(node)
                if os.environ.get("TOSA_ENGINE_DEBUG"):
                    print(
                        "[cf-qdq-collapse] candidate",
                        getattr(node, "name", "?"),
                        "quant=",
                        getattr(quant_node, "name", "?"),
                        "quant_qparam=",
                        quant_qparam,
                        "dequant_qparam=",
                        dequant_qparam,
                    )
                if not _same_qparam(quant_qparam, dequant_qparam):
                    continue
                float_source = quant_node.args[0] if getattr(quant_node, "args", None) else None
                if not hasattr(float_source, "op"):
                    continue
                # Parse, don't validate: branch-local zero-input constants need
                # to keep their quantized boundary when they feed a binary op.
                # If we collapse q->dq to the raw FULL/get_attr here, later
                # control-flow binary repair sees a missing-qparam operand and
                # can incorrectly substitute the sibling activation path,
                # turning `x +/- c` into `x +/- x`.
                if _is_zero_input_const_source(float_source) and _feeds_scalar_binary(node):
                    continue
                if os.environ.get("TOSA_ENGINE_DEBUG"):
                    print(
                        "[cf-qdq-collapse] bypass",
                        getattr(node, "name", "?"),
                        "->",
                        getattr(float_source, "name", "?"),
                    )
                node.replace_all_uses_with(float_source)
                submodule.graph.erase_node(node)
                if len(getattr(quant_node, "users", ())) == 0:
                    submodule.graph.erase_node(quant_node)
                rewritten += 1
                modified = True
                continue

            if not _is_quantize_node(node):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue

            source_node = node.args[0]
            source_fake = source_node.meta.get("val")
            source_dtype = getattr(source_fake, "dtype", None)
            source_qparam = _extract_effective_node_qparam(source_node)
            if source_qparam is None:
                source_qparam = _single_qparam_entry(node.meta.get("input_qparams"))
            if os.environ.get("TOSA_ENGINE_DEBUG"):
                print(
                    "[cf-quantize-rewrite] candidate",
                    getattr(node, "name", "?"),
                    "source=",
                    getattr(source_node, "name", "?"),
                    "source_dtype=",
                    source_dtype,
                    "input_qparams=",
                    node.meta.get("input_qparams"),
                    "output_qparams=",
                    node.meta.get("output_qparams"),
                )

            try:
                output_qparam = QuantArgs.from_operator(node.target, node.args)
            except Exception:
                output_qparam = _parse_per_tensor_qparam(node)
                if output_qparam is None:
                    continue

            if (
                source_dtype not in (torch.int8, torch.int16, torch.int32)
                and source_qparam is None
                and not _is_zero_input_const_source(source_node)
            ):
                continue

            if (
                source_qparam is None
                or source_qparam.per_channel
                or output_qparam.per_channel
            ):
                if source_qparam is None and _is_zero_input_const_source(source_node):
                    const_value = _build_full_const(source_node)
                    if const_value is not None:
                        quantized_value = output_qparam.quantize_value(const_value)
                        const_name = (
                            f"{getattr(source_node, 'name', 'const')}_quantized_const"
                        )
                        suffix = 0
                        while hasattr(submodule, const_name):
                            suffix += 1
                            const_name = (
                                f"{getattr(source_node, 'name', 'const')}_quantized_const_{suffix}"
                            )

                        submodule.register_buffer(const_name, quantized_value)
                        with submodule.graph.inserting_before(node):
                            const_node = submodule.graph.get_attr(const_name)

                        const_node.meta = source_node.meta.copy()
                        const_node.meta["val"] = quantized_value
                        const_node.meta["input_qparams"] = {0: output_qparam}
                        const_node.meta["output_qparams"] = {0: output_qparam}
                        node.replace_all_uses_with(const_node)
                        submodule.graph.erase_node(node)
                        if len(getattr(source_node, "users", ())) == 0:
                            submodule.graph.erase_node(source_node)
                        rewritten += 1
                        modified = True
                        if os.environ.get("TOSA_ENGINE_DEBUG"):
                            print(
                                "[cf-quantize-rewrite] materialized const",
                                getattr(node, "name", "?"),
                                "->",
                                getattr(const_node, "name", "?"),
                            )
                        continue
                if os.environ.get("TOSA_ENGINE_DEBUG"):
                    print(
                        "[cf-quantize-rewrite] skip",
                        getattr(node, "name", "?"),
                        "source_qparam=",
                        source_qparam,
                        "output_qparam=",
                        output_qparam,
                    )
                continue

            if _same_qparam(source_qparam, output_qparam):
                if os.environ.get("TOSA_ENGINE_DEBUG"):
                    print(
                        "[cf-quantize-rewrite] bypass",
                        getattr(node, "name", "?"),
                    )
                node.replace_all_uses_with(source_node)
                submodule.graph.erase_node(node)
                rewritten += 1
                modified = True
                continue

            rescale = fx_helpers.insert_rescale_node(
                submodule,
                node,
                source_node,
                output_qparam.dtype,
                [_scale_value(source_qparam) / _scale_value(output_qparam)],
                _zp_value(source_qparam),
                _zp_value(output_qparam),
                after=False,
            )
            rescale.meta["input_qparams"] = {0: source_qparam}
            rescale.meta["output_qparams"] = {0: output_qparam}
            if os.environ.get("TOSA_ENGINE_DEBUG"):
                print(
                    "[cf-quantize-rewrite] rescale",
                    getattr(node, "name", "?"),
                    "from",
                    source_qparam,
                    "to",
                    output_qparam,
                )
            node.replace_all_uses_with(rescale)
            submodule.graph.erase_node(node)
            rewritten += 1
            modified = True

        if modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow quantize node(s) into explicit rescale/no-op edges",
            rewritten,
        )


def _rewrite_control_flow_dequantize_nodes(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    import fx_helpers

    def _single_qparam_entry(qparams):
        if not qparams:
            return None
        if isinstance(qparams, dict):
            if 0 in qparams:
                return qparams[0]
            if len(qparams) == 1:
                return next(iter(qparams.values()))
        return qparams

    def _same_qparam(lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        return (
            float(lhs.scale) == float(rhs.scale)
            and int(lhs.zp) == int(rhs.zp)
            and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
        )

    def _binary_input_qparam(user, value_node):
        input_qparams = getattr(user, "meta", {}).get("input_qparams") or {}
        for idx, arg in enumerate(getattr(user, "args", ())):
            if arg is value_node and idx in input_qparams:
                return input_qparams[idx]
        return _single_qparam_entry(input_qparams)

    rewritten = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        modified = False
        for node in list(submodule.graph.nodes):
            if not _is_dequantize_node(node):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue

            source_node = node.args[0]
            users = list(getattr(node, "users", ()))
            if not users:
                continue

            source_qparam = _extract_effective_node_qparam(source_node)
            if getattr(source_qparam, "dtype", None) in (torch.int8, torch.int16):
                desired_qparam = None
                binary_users = []
                for user in users:
                    if getattr(user, "op", None) != "call_function":
                        binary_users = []
                        break
                    target_str = str(getattr(user, "target", ""))
                    if (
                        "aten.add.Tensor" not in target_str
                        and "aten.sub.Tensor" not in target_str
                        and "aten.mul.Tensor" not in target_str
                    ):
                        binary_users = []
                        break
                    user_qparam = _binary_input_qparam(user, node)
                    if getattr(user_qparam, "dtype", None) != torch.int32:
                        binary_users = []
                        break
                    if desired_qparam is None:
                        desired_qparam = user_qparam
                    elif not _same_qparam(desired_qparam, user_qparam):
                        binary_users = []
                        break
                    binary_users.append(user)

                if binary_users and desired_qparam is not None:
                    lifted = fx_helpers.insert_rescale_node(
                        submodule,
                        binary_users[0],
                        source_node,
                        torch.int32,
                        [
                            source_qparam.get_scale_per_tensor()
                            / desired_qparam.get_scale_per_tensor()
                        ],
                        source_qparam.get_zp_per_tensor(),
                        desired_qparam.get_zp_per_tensor(),
                        after=False,
                    )
                    lifted.meta["input_qparams"] = {0: source_qparam}
                    lifted.meta["output_qparams"] = {0: desired_qparam}
                    node.replace_all_uses_with(lifted)
                    submodule.graph.erase_node(node)
                    rewritten += 1
                    modified = True
                    logger.info(
                        "[fast_preprocess] rewrote control-flow dequantize %s into int32 lift %s",
                        getattr(node, "name", "?"),
                        getattr(lifted, "name", "?"),
                    )
                    continue

            if not all(
                _is_tosa_rescale_node(user) or getattr(user, "op", None) == "output"
                for user in users
            ):
                continue

            if os.environ.get("TOSA_ENGINE_DEBUG"):
                print(
                    "[cf-dequantize-rewrite] bypass",
                    getattr(node, "name", "?"),
                    "->",
                    getattr(source_node, "name", "?"),
                )

            node.replace_all_uses_with(source_node)
            submodule.graph.erase_node(node)
            rewritten += 1
            modified = True

        if modified:
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()
            submodule.recompile()

    if rewritten:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow dequantize node(s) into direct quant-domain edges",
            rewritten,
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

            # Parse, don't validate: the submodule input boundary is the
            # placeholder itself, not any interior dequant/rescale helper that
            # happened to survive the FX-direct pass stack. Collapse the full
            # placeholder -> passthrough* -> layout chain back to the
            # placeholder so all cond branches expose the same TOSA-visible
            # input signature.
            replacement = node
            replacement.meta["val"] = layout_user.meta.get(
                "val", anchor.meta.get("val", replacement.meta.get("val"))
            )
            if layout_user.meta.get("tosa_dim_order") is not None:
                replacement.meta["tosa_dim_order"] = tuple(layout_user.meta["tosa_dim_order"])
            if (
                replacement.meta.get("input_qparams") is None
                and anchor.meta.get("input_qparams") is not None
            ):
                replacement.meta["input_qparams"] = dict(anchor.meta["input_qparams"])
            branch_output_qparam = _branch_local_control_flow_output_qparam(
                replacement
            )
            if branch_output_qparam is not None:
                replacement.meta["output_qparams"] = {0: branch_output_qparam}
            elif (
                replacement.meta.get("output_qparams") is None
                and anchor.meta.get("output_qparams") is not None
            ):
                replacement.meta["output_qparams"] = dict(anchor.meta["output_qparams"])
            layout_user.replace_all_uses_with(replacement)
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


def _restore_missing_control_flow_rescale_qparams(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs

    restored = 0
    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        submodule_modified = False
        for node in submodule.graph.nodes:
            if not _is_tosa_rescale_node(node):
                continue
            if len(getattr(node, "args", ())) < 5:
                continue
            if node.meta.get("input_qparams") and node.meta.get("output_qparams"):
                continue

            source = node.args[0]
            if not hasattr(source, "op"):
                continue
            source_qparam = _extract_effective_node_qparam(source)
            if source_qparam is None:
                continue

            output_dtype = node.args[1]
            scale_arg = node.args[2]
            if output_dtype not in (torch.int8, torch.int16, torch.int32):
                continue
            if not (
                isinstance(scale_arg, (list, tuple))
                and len(scale_arg) == 1
                and scale_arg[0] not in (0, 0.0)
            ):
                continue

            try:
                qinfo = torch.iinfo(output_dtype)
                output_qparam = QuantArgs(
                    scale=source_qparam.get_scale_per_tensor() / float(scale_arg[0]),
                    zp=int(node.args[4]),
                    qmin=int(qinfo.min),
                    qmax=int(qinfo.max),
                    dtype=output_dtype,
                )
            except Exception:
                continue

            if not node.meta.get("input_qparams"):
                node.meta["input_qparams"] = {0: source_qparam}
            if not node.meta.get("output_qparams"):
                node.meta["output_qparams"] = {0: output_qparam}
            restored += 1
            submodule_modified = True

        if submodule_modified:
            submodule.recompile()

    if restored:
        logger.info(
            "[fast_preprocess] restored qparams on %d control-flow rescale node(s)",
            restored,
        )


def _prune_duplicate_terminal_rescales(graph_module) -> None:
    import math
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _modules():
        yield graph_module
        for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
            yield submodule

    def _single_qparam_entry(qparams):
        if not qparams:
            return None
        if isinstance(qparams, dict):
            if 0 in qparams:
                return qparams[0]
            if len(qparams) == 1:
                return next(iter(qparams.values()))
        return qparams

    def _normalize_scale_arg(scale_arg):
        if isinstance(scale_arg, (list, tuple)):
            try:
                return [float(value) for value in scale_arg]
            except (TypeError, ValueError):
                return None
        try:
            return [float(scale_arg)]
        except (TypeError, ValueError):
            return None

    def _same_scale_list(lhs, rhs):
        lhs_vals = _normalize_scale_arg(lhs)
        rhs_vals = _normalize_scale_arg(rhs)
        if lhs_vals is None or rhs_vals is None or len(lhs_vals) != len(rhs_vals):
            return False
        return all(
            math.isclose(lval, rval, rel_tol=1e-5, abs_tol=1e-9)
            for lval, rval in zip(lhs_vals, rhs_vals, strict=True)
        )

    pruned = 0
    for module in _modules():
        modified = False
        for node in list(module.graph.nodes):
            if not _is_tosa_rescale_node(node):
                continue
            if len(getattr(node, "users", ())) != 1:
                continue

            user = next(iter(node.users))
            if not _is_tosa_rescale_node(user):
                continue

            if len(getattr(node, "args", ())) < 5 or len(getattr(user, "args", ())) < 5:
                continue
            if user.args[0] is not node:
                continue
            if node.args[1] != user.args[1]:
                continue
            if node.args[1] not in (torch.int8, torch.int16):
                continue
            if node.args[3] != user.args[3] or node.args[4] != user.args[4]:
                continue
            if not _same_scale_list(node.args[2], user.args[2]):
                continue

            first_output_qparam = _single_qparam_entry(node.meta.get("output_qparams"))
            final_output_qparam = _single_qparam_entry(user.meta.get("output_qparams"))
            if first_output_qparam is not None:
                continue
            if final_output_qparam is None:
                continue

            source = node.args[0]
            node.args = (
                source,
                user.args[1],
                user.args[2],
                user.args[3],
                user.args[4],
            )
            node.meta["val"] = user.meta.get("val", node.meta.get("val"))
            if node.meta.get("input_qparams") is None:
                source_qparam = _single_qparam_entry(
                    getattr(source, "meta", {}).get("output_qparams")
                )
                if source_qparam is not None:
                    node.meta["input_qparams"] = {0: source_qparam}
            node.meta["output_qparams"] = {0: final_output_qparam}
            if user.meta.get("tosa_dim_order") is not None:
                node.meta["tosa_dim_order"] = tuple(user.meta["tosa_dim_order"])

            user.replace_all_uses_with(node)
            module.graph.erase_node(user)
            pruned += 1
            modified = True

        if modified:
            module.graph.eliminate_dead_code()
            module.graph.lint()
            module.recompile()

    if pruned:
        logger.info(
            "[fast_preprocess] pruned %d duplicate terminal rescale node(s)",
            pruned,
        )


def _dump_control_flow_submodules(graph_module, label: str) -> None:
    if not _control_flow_debug_enabled():
        return

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _node_line(node) -> str:
        meta = getattr(node, "meta", {})
        fake_val = meta.get("val")
        shape = tuple(fake_val.shape) if hasattr(fake_val, "shape") else None
        dtype = getattr(fake_val, "dtype", None)
        return (
            f"{node.name} | {node.op} | {node.target} | "
            f"shape={shape} dtype={dtype} "
            f"qin={meta.get('input_qparams')} "
            f"qout={meta.get('output_qparams')} "
            f"saved_qout={meta.get('arm_control_flow_saved_output_qparams')}"
        )

    for name, submodule, _ in get_cond_while_submodules_nested(graph_module):
        try:
            readable = submodule.print_readable(print_output=False)
        except TypeError:
            readable = submodule.print_readable()
        node_dump = "\n".join(_node_line(node) for node in submodule.graph.nodes)
        print(
            f"[fast_preprocess] control-flow submodule dump ({label}:{name})\n{readable}\n[node-meta]\n{node_dump}"
        )
        logger.info(
            "[fast_preprocess] control-flow submodule dump (%s:%s)\n%s\n[node-meta]\n%s",
            label,
            name,
            readable,
            node_dump,
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

            edge_program = program_mod._generate_edge_program(
                config,
                program,
                preserve_ops=ops_set_to_not_decompose_by_program.get(name, []),
            )
            if os.environ.get("ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING") == "1":
                try:
                    from executorch.backends.arm.tosa.partitioner import (
                        _sync_direct_delegate_output_meta,
                    )

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
    _prepare_control_flow_qparams(gm, edge_program)
    _snapshot_control_flow_table_qparams(gm)
    # Parse, don't validate: control-flow table ops still have their original
    # q/dq scaffolding at this point. Reapply TABLE lowering before the
    # recursive FX-direct pass stack strips that local qparam context away,
    # then keep the later repair pass as a safety net for any surviving raw
    # table-capable ops.
    _reapply_control_flow_table_ops(gm, edge_program)

    # Run the full FX-direct pipeline (all passes in Rust via PyO3)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
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
    _reapply_control_flow_table_ops(gm, edge_program)
    _rewrite_control_flow_quantize_nodes(gm)
    _rewrite_control_flow_dequantize_nodes(gm)
    _canonicalize_control_flow_pool_binary_rescales(gm)
    _reapply_control_flow_boundary_rescales(gm, include_inputs=True, include_outputs=False)
    _rewrite_control_flow_quantize_nodes(gm)
    _rewrite_control_flow_dequantize_nodes(gm)
    _prune_duplicate_control_flow_rescales(gm)
    _repair_control_flow_binary_rescales(gm, _compile_spec.tosa_spec)
    _restore_control_flow_output_dequantizes(gm)
    t_passes = time.perf_counter() - t0
    _dump_control_flow_submodules(gm, "after_rust_passes")

    # Re-apply the same normalization after passes so serializer ingest sees
    # branch signatures that still match the parent control-flow nodes.
    _regularize_control_flow_submodules(gm)
    _normalize_control_flow_input_layouts(gm)
    _restore_missing_control_flow_input_rescales(gm)
    _restore_control_flow_unary_quant_boundaries(gm)
    _restore_control_flow_pool_quant_boundaries(gm)
    _refresh_control_flow_binary_input_rescales(gm)
    _normalize_control_flow_output_layouts(gm)
    _restore_missing_control_flow_rescale_qparams(gm)
    _reapply_control_flow_boundary_rescales(
        gm, include_inputs=False, include_outputs=True
    )
    _rewrite_control_flow_quantize_nodes(gm)
    _rewrite_control_flow_dequantize_nodes(gm)
    _prune_duplicate_control_flow_rescales(gm)
    _fold_zero_input_full_in_control_flow_submodules(gm)
    _quantize_constant_only_control_flow_outputs(gm)
    _regularize_control_flow_submodules(gm, include_inputs=False, include_outputs=True)
    _annotate_control_flow_rank4_dim_order(gm)
    _force_control_flow_rank4_output_dim_order(gm)
    _refresh_tosa_dim_order_fake_tensors(gm)
    _prune_duplicate_terminal_rescales(gm)
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
