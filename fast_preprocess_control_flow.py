"""Control-flow helpers and repair passes for fast_preprocess_pipeline."""

from dataclasses import dataclass
import logging
import os
from typing import cast

from fast_preprocess_cf_ctx import (
    _for_each_cf_submodule,
    _get_matching_fake_tensor,
)

logger = logging.getLogger(__name__)


def _node_target_str(node) -> str:
    return str(getattr(node, "target", ""))


def _is_call_function_target(node, *markers: str) -> bool:
    return getattr(node, "op", None) == "call_function" and any(
        marker in _node_target_str(node) for marker in markers
    )


def _is_control_flow_quant_input_node(node) -> bool:
    if _is_dequantize_node(node):
        return True
    if getattr(node, "op", None) != "placeholder":
        return False
    if not node.meta.get("is_input"):
        return False
    return node.meta.get("arm_control_flow_placeholder", False) or (
        "dequantize_per_tensor" in getattr(node, "name", "")
    )


def _is_control_flow_view_like(node) -> bool:
    return _is_call_function_target(node, "view_copy", "RESHAPE")


def _is_control_flow_binary_tensor_op(node) -> bool:
    return _is_call_function_target(node, "aten.add.Tensor", "aten.sub.Tensor")


def _is_control_flow_mul_tensor_op(node) -> bool:
    return _is_call_function_target(node, "aten.mul.Tensor")


def _is_control_flow_binary_or_mul_tensor_op(node) -> bool:
    return _is_control_flow_binary_tensor_op(node) or _is_control_flow_mul_tensor_op(node)


def _downstream_non_view_like_users(node):
    stack = list(getattr(node, "users", ()))
    seen = set()
    while stack:
        user = stack.pop()
        if id(user) in seen:
            continue
        seen.add(id(user))
        if _is_control_flow_view_like(user):
            stack.extend(list(getattr(user, "users", ())))
            continue
        yield user


def _ordered_control_flow_users(node):
    def _user_sort_key(user):
        target_str = _node_target_str(user)
        if "getitem" in target_str and len(getattr(user, "args", ())) > 1:
            idx = user.args[1]
            if isinstance(idx, int):
                return (0, idx, getattr(user, "name", ""))
        return (1, 0, getattr(user, "name", ""))

    return sorted(list(node.users), key=_user_sort_key)


def _first_node_arg(node):
    return (
        node.args[0]
        if getattr(node, "args", None) and hasattr(node.args[0], "op")
        else None
    )


def _control_flow_debug_enabled() -> bool:
    return bool(os.environ.get("ARM_TOSA_ENGINE_DEBUG_CONTROL_FLOW"))


def _log_control_flow_debug(message: str, *args) -> None:
    if _control_flow_debug_enabled():
        logger.info(message, *args)


def _cleanup_graph(graph) -> None:
    graph.eliminate_dead_code()
    graph.lint()


def _cleanup_and_recompile(module) -> None:
    _cleanup_graph(module.graph)
    module.recompile()


@dataclass(frozen=True)
class _ControlFlowInputBoundaryContract:
    submodule: object
    index: int
    placeholder: object
    input_qparam: object
    output_qparam: object


@dataclass(frozen=True)
class _ControlFlowOutputBoundaryContract:
    submodule: object
    index: int
    target_qparam: object


@dataclass(frozen=True)
class _ControlFlowBinaryQparamContract:
    input_qparams: dict[int, object]
    output_qparam: object


@dataclass(frozen=True)
class _ControlFlowBinaryAlignmentEnv:
    parent_args: tuple[object, ...]
    placeholders: tuple[object, ...]
    placeholder_qparams: dict[int, object]
    placeholder_index_by_name: dict[str, int]
    output_args: tuple[object, ...]
    output_qparams_map: dict[int, object]


@dataclass(frozen=True)
class _ControlFlowRescaleEdit:
    submodule: object
    anchor: object
    source_node: object
    input_qparam: object
    output_qparam: object
    current_node: object = None
    after: bool = False
    preserve_boundary: bool = False

    @classmethod
    def from_input_boundary_contract(
        cls,
        contract,
        *,
        current_node=None,
        preserve_boundary: bool = False,
    ):
        return cls(
            submodule=contract.submodule,
            anchor=contract.placeholder,
            source_node=contract.placeholder,
            input_qparam=contract.input_qparam,
            output_qparam=contract.output_qparam,
            current_node=current_node,
            after=True,
            preserve_boundary=preserve_boundary,
        )

    def ensure(self):
        return _ensure_control_flow_rescale_node(
            self.submodule,
            self.anchor,
            self.source_node,
            input_qparam=self.input_qparam,
            output_qparam=self.output_qparam,
            current_node=self.current_node,
            after=self.after,
            preserve_boundary=self.preserve_boundary,
        )


def _extract_effective_node_qparam(
    node,
    *,
    parse_operator_qparams: bool = False,
    fallback_to_consumer_qparams: bool = False,
    prefer_input_qparams: bool = True,
):
    """Follow producer edges until the node's effective quantization is known."""
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    current = node
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        output_qparam = _extract_single_qparam_entry(current.meta.get("output_qparams"))
        if output_qparam is not None:
            return output_qparam

        input_qparam = _extract_single_qparam_entry(current.meta.get("input_qparams"))
        if prefer_input_qparams and input_qparam is not None:
            return input_qparam

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

        if parse_operator_qparams:
            try:
                return QuantArgs.from_operator(current.target, current.args)
            except Exception:
                pass

        if not prefer_input_qparams and input_qparam is not None:
            return input_qparam

        if fallback_to_consumer_qparams:
            consumer_qparam = _extract_consumer_qparam(current)
            if consumer_qparam is not None:
                return consumer_qparam

        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return None


def _topologically_rebuild_graph_module(module) -> None:
    import torch

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
        seeded_output_fake_tensors = {}
        for idx, (submodule_output, submodule_user) in enumerate(
            zip(output_args, _ordered_control_flow_users(submodule_node))
        ):
            # Control-flow outputs need the parent-visible boundary contract,
            # not the raw getitem fake tensor. For while-loop carried values in
            # particular, the parent getitem can still look float while the
            # loop body output has already been repaired back into its carried
            # quantized dtype.
            desired_fake_tensor = _get_control_flow_output_fake_tensor(submodule_user)
            current_fake_tensor = submodule_output.meta.get("val")
            seeded_fake_tensor = (
                current_fake_tensor
                if _preserve_control_flow_output_meta(
                    current_fake_tensor, desired_fake_tensor
                )
                else desired_fake_tensor
            )
            seeded_output_fake_tensors[idx] = seeded_fake_tensor
            submodule_output.meta["val"] = seeded_fake_tensor
            if (
                seeded_fake_tensor is current_fake_tensor
                and submodule_output.meta.get("tosa_dim_order") is None
            ):
                continue
            if submodule_user.meta.get("tosa_dim_order") is not None:
                submodule_output.meta["tosa_dim_order"] = tuple(
                    submodule_user.meta["tosa_dim_order"]
                )
        output_node.meta["arm_control_flow_output_fake_tensors"] = (
            seeded_output_fake_tensors
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


def _extract_single_qparam_entry(qparams):
    if not qparams:
        return None
    if isinstance(qparams, dict):
        if 0 in qparams:
            return qparams[0]
        if len(qparams) == 1:
            return next(iter(qparams.values()))
    return qparams


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
        qparam = _find_input_qparam_for_user(user, node)
        if qparam is not None:
            found.append(qparam)
            continue

        qparam = _extract_consumer_qparam(user, seen)
        if qparam is not None:
            found.append(qparam)

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


def _branch_local_control_flow_output_qparam(placeholder):
    import torch

    # Parse, don't validate: the control-flow placeholder boundary should keep
    # the saved branch-local quantized contract when it exists, not rediscover
    # it from a later accumulator-domain rescale chain.
    saved_qparam = _extract_single_qparam_entry(
        placeholder.meta.get("arm_control_flow_saved_output_qparams")
    )
    if saved_qparam is not None:
        return saved_qparam

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

    placeholder_output_qparam = _extract_single_qparam_entry(
        placeholder.meta.get("output_qparams")
    )
    if getattr(placeholder_output_qparam, "dtype", None) in (torch.int8, torch.int16):
        return placeholder_output_qparam

    return _infer_control_flow_placeholder_output_qparam(placeholder)


def _placeholder_feeds_binary_directly(placeholder) -> bool:
    users = [user for user in list(getattr(placeholder, "users", ())) if hasattr(user, "op")]
    if not users:
        return False
    return all(
        _is_control_flow_binary_or_mul_tensor_op(user)
        and not _binary_peer_is_scalar_like(user, placeholder)
        for user in users
    )


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
    saved_qparam = _extract_single_qparam_entry(
        placeholder.meta.get("arm_control_flow_saved_output_qparams")
    )
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


_CONTROL_FLOW_TABLE_KIND_SUFFIXES = (
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
)


def _normalize_qparam_dtype(dtype):
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return {
            "torch.int8": torch.int8,
            "int8": torch.int8,
            "torch.int16": torch.int16,
            "int16": torch.int16,
            "torch.int32": torch.int32,
            "int32": torch.int32,
        }.get(dtype)
    return None


def _iter_control_flow_table_nodes(submodule, table_targets):
    for node in submodule.graph.nodes:
        if node.op == "call_function" and node.target in table_targets:
            yield node


def _iter_control_flow_call_function_nodes(submodule, predicate):
    for node in list(submodule.graph.nodes):
        if getattr(node, "op", None) == "call_function" and predicate(node):
            yield node


def _iter_control_flow_binary_nodes(submodule):
    yield from _iter_control_flow_call_function_nodes(
        submodule, _is_control_flow_binary_tensor_op
    )


def _iter_control_flow_binary_rescale_args(node, *, rescale_dtype=None):
    for idx, arg in enumerate(node.args[:2]):
        if not _is_tosa_rescale_node(arg) or len(getattr(arg, "args", ())) < 5:
            continue
        if rescale_dtype is not None and arg.args[1] != rescale_dtype:
            continue
        source_node = arg.args[0]
        if not hasattr(source_node, "op"):
            continue
        yield idx, arg, source_node


def _canonicalize_control_flow_table_targets(submodule, table_targets) -> int:
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
    return canonicalized


def _table_spec(node):
    target_str = str(node.target)
    for suffix, kind in _CONTROL_FLOW_TABLE_KIND_SUFFIXES:
        if suffix in target_str:
            return kind, None
    if "pow.Tensor_Scalar" in target_str:
        scalar_arg = None
        if len(node.args) > 1 and isinstance(node.args[1], (int, float)):
            scalar_arg = float(node.args[1])
        return "pow", scalar_arg
    return None, None


def _synthesize_control_flow_table_qparams_for_submodule(submodule, table_targets) -> int:
    updated = 0
    for node in _iter_control_flow_table_nodes(submodule, table_targets):
        if not node.meta.get("input_qparams"):
            qparam = node.meta.get("arm_saved_table_input_qparam")
            if qparam is None and getattr(node, "args", None) and hasattr(node.args[0], "op"):
                qparam = _extract_effective_node_qparam(
                    node.args[0],
                    parse_operator_qparams=True,
                )
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


def _restamp_control_flow_table_meta(submodule) -> int:
    import torch

    updated = 0
    for node in submodule.graph.nodes:
        if not _is_tosa_table_node(node):
            continue
        input_node = _first_node_arg(node)
        if not hasattr(input_node, "meta"):
            continue

        input_val = input_node.meta.get("val")
        if input_val is None or not hasattr(input_val, "new_empty"):
            continue

        input_qparam = _extract_single_qparam_entry(node.meta.get("input_qparams"))
        input_dtype = _normalize_qparam_dtype(getattr(input_qparam, "dtype", None))
        if input_dtype is None and isinstance(input_qparam, dict):
            input_dtype = _normalize_qparam_dtype(input_qparam.get("dtype"))
        if input_dtype is None:
            input_dtype = getattr(input_val, "dtype", None)

        desired_dtype = torch.int32 if input_dtype == torch.int16 else input_dtype
        current_val = node.meta.get("val")
        if (
            tuple(getattr(current_val, "shape", ())) == tuple(input_val.shape)
            and getattr(current_val, "dtype", None) == desired_dtype
        ):
            continue

        node.meta["val"] = input_val.new_empty(input_val.shape, dtype=desired_dtype)
        if input_node.meta.get("tosa_dim_order") is not None:
            node.meta["tosa_dim_order"] = tuple(input_node.meta["tosa_dim_order"])
        updated += 1
        logger.info(
            "[fast_preprocess] restamped control-flow table meta %s -> shape=%s dtype=%s",
            getattr(node, "name", "?"),
            tuple(input_val.shape),
            desired_dtype,
        )
    return updated


def _reapply_control_flow_table_ops(graph_module, exported_program) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_table_ops import (
        TableOps,
    )
    import fx_helpers

    del exported_program
    table_targets = list(TableOps.included_ops())
    repaired = 0

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        _canonicalize_control_flow_table_targets(submodule, table_targets)
        needs_table_repair = any(_iter_control_flow_table_nodes(submodule, table_targets))
        if not needs_table_repair:
            continue

        synthesized = _synthesize_control_flow_table_qparams_for_submodule(
            submodule, table_targets
        )
        if synthesized:
            logger.info(
                "[fast_preprocess] synthesized %d table qparam attachment(s) in control-flow submodule",
                synthesized,
            )

        modified = False
        for node in list(_iter_control_flow_table_nodes(submodule, table_targets)):
            if not node.meta.get("input_qparams") or not node.meta.get("output_qparams"):
                continue
            kind, scalar_arg = _table_spec(node)
            if kind is None:
                continue
            fx_helpers.insert_table_op(submodule, node, kind, scalar_arg)
            modified = True

        restamped = _restamp_control_flow_table_meta(submodule)

        if modified or restamped:
            _cleanup_and_recompile(submodule)
            repaired += int(modified)

    if repaired:
        logger.info(
            "[fast_preprocess] reapplied table-op lowering on %d control-flow submodule(s)",
            repaired,
        )


def _snapshot_control_flow_table_qparams(graph_module) -> None:
    from executorch.backends.arm._passes.insert_table_ops import (
        TableOps,
    )
    from executorch.backends.arm._passes.quant_args import QuantArgs

    table_targets = list(TableOps.included_ops())
    snapshotted = 0

    def _visit(ctx):
        nonlocal snapshotted
        for node in ctx.graph.nodes:
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
        return False  # metadata-only, no graph structure changes

    _for_each_cf_submodule(graph_module, _visit)

    if snapshotted:
        logger.info(
            "[fast_preprocess] snapshotted %d control-flow table qparam attachment(s)",
            snapshotted,
        )


def _collect_control_flow_binary_alignment_env(submodule, control_flow_node):
    import torch

    if control_flow_node.target == torch.ops.higher_order.cond:
        parent_args = tuple(control_flow_node.args[-1])
    elif control_flow_node.target == torch.ops.higher_order.while_loop:
        parent_args = tuple(control_flow_node.args[-2]) + tuple(control_flow_node.args[-1])
    else:
        parent_args = ()

    placeholders = tuple(_placeholder_nodes(submodule))
    output_node = submodule.graph.output_node()
    output_args = tuple(_output_node_args(output_node))
    return _ControlFlowBinaryAlignmentEnv(
        parent_args=parent_args,
        placeholders=placeholders,
        placeholder_qparams=dict(
            control_flow_node.meta.get("arm_control_flow_input_qparams_map", {})
        ),
        placeholder_index_by_name={
            getattr(node, "name", f"placeholder_{idx}"): idx
            for idx, node in enumerate(placeholders)
        },
        output_args=output_args,
        output_qparams_map=_infer_control_flow_output_qparams_map(control_flow_node),
    )


def _repair_control_flow_binary_rescales(graph_module, tosa_spec) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )
    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertRescaleInt32Pass,
    )
    from executorch.backends.arm.tosa.specification import TosaLoweringContext

    repaired = 0
    synthesized = 0
    for name, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        env = _collect_control_flow_binary_alignment_env(submodule, control_flow_node)
        updated_graph = False

        for idx, placeholder in enumerate(env.placeholders):
            if idx >= len(env.parent_args):
                continue
            if placeholder.meta.get("output_qparams"):
                continue
            parent_qparam = _repair_control_flow_qparam(env.parent_args[idx])
            if parent_qparam is None:
                continue
            _stamp_single_qparams(placeholder, parent_qparam, parent_qparam)
            updated_graph = True
            logger.info(
                "[fast_preprocess] restored control-flow placeholder qparams %s idx=%d -> %s",
                getattr(placeholder, "name", "?"),
                idx,
                parent_qparam,
            )

        for node in _iter_control_flow_binary_nodes(submodule):
            # Prefer an unused branch placeholder over a TABLE-derived dequantize
            # chain so InsertRescaleInt32Pass can rebuild the missing sibling path.
            unused_placeholders = [
                placeholder
                for placeholder in env.placeholders
                if placeholder not in _placeholder_dependencies(node)
            ]
            if len(unused_placeholders) == 1:
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
                        rescued_qparam = _repair_control_flow_qparam(rescued_placeholder)
                        if rescued_qparam is not None:
                            _stamp_single_qparams(arg, input_qparam=rescued_qparam)
                        updated_graph = True
                        logger.info(
                            "[fast_preprocess] rescued control-flow binary input %s arg%d -> %s",
                            getattr(node, "name", "?"),
                            idx,
                            getattr(rescued_placeholder, "name", "?"),
                        )
                        break

            for idx, arg in enumerate(node.args[:2]):
                if not hasattr(arg, "op"):
                    continue
                candidate_source = arg
                if _is_tosa_rescale_node(arg) and getattr(arg, "args", None):
                    source_arg = arg.args[0]
                    if hasattr(source_arg, "op"):
                        candidate_source = source_arg
                cloned_preferred = (
                    _maybe_clone_preferred_control_flow_quantized_binary_output_rescale(
                        submodule,
                        node,
                        candidate_source,
                        rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
                    )
                )
                if cloned_preferred is not None:
                    cloned, quant_rescale, output_qparam = cloned_preferred
                    source_qparam = _repair_control_flow_qparam(candidate_source)
                    _stamp_single_qparams(cloned, source_qparam, output_qparam)
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

            for idx, arg, source in _iter_control_flow_binary_rescale_args(node):
                if arg.meta.get("arm_promoted_control_flow_binary_input"):
                    continue
                _reconstruct_control_flow_rescale_source_qparam(node, idx, arg)
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
                _reconstruct_control_flow_binary_output_qparam(node, user)
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
                producer_qparam = _resolve_repaired_control_flow_binary_input_qparam(
                    node,
                    idx,
                    arg,
                    env=env,
                )
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
                        recovered_qparam = _recover_mixed_control_flow_binary_operand_qparam(
                            node,
                            mixed_idx,
                            mixed_arg,
                            sibling_qparam,
                        )

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
                    contract = _additive_binary_qparam_contract(lhs_qparam, rhs_qparam)
                    node_updated, _ = _enforce_control_flow_binary_contract(
                        submodule,
                        node,
                        {},
                        contract,
                        env=env,
                        terminal_outputs_only=False,
                        qparam_log_message="[fast_preprocess] recomputed control-flow binary accumulator qparams for %s -> %s",
                        qparam_log_value=contract.output_qparam,
                        output_rescale_log_message="[fast_preprocess] realigned %d pre-pass control-flow binary output rescale(s) from %s",
                        insert_operand_lifts=False,
                    )
                    updated_graph |= node_updated
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
            _cleanup_and_recompile(submodule)

        for node in _iter_control_flow_binary_nodes(submodule):
            _log_control_flow_binary_debug_state(
                node,
                prefix="[fast_preprocess] pre-InsertRescaleInt32Pass",
            )

        try:
            with TosaLoweringContext(tosa_spec):
                result = InsertRescaleInt32Pass().call(submodule)
        except Exception:
            _log_control_flow_debug(
                "[fast_preprocess] InsertRescaleInt32Pass failed in %s",
                name,
            )
            for subnode in list(submodule.graph.nodes):
                if not _is_control_flow_binary_or_mul_tensor_op(subnode):
                    continue
                _log_control_flow_binary_debug_state(
                    subnode,
                    prefix="[fast_preprocess] failing binary",
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


def _restore_control_flow_quant_boundaries(
    graph_module, *, mode: str, canonicalize_pool_binary_sources: bool = False
) -> None:
    """Restore control-flow quant boundaries for pool or unary ops."""
    import torch

    import fx_helpers

    def _is_target_op(node) -> bool:
        if mode == "pool":
            return _is_call_function_target(node, "AVG_POOL2D", "avg_pool2d")
        return _is_call_function_target(node, "aten.relu.default", "aten.clamp.default")

    canonicalized = 0
    if mode == "pool" and canonicalize_pool_binary_sources:
        def _canonicalize_pool_binary_sources(ctx):
            nonlocal canonicalized
            updated = False
            for node in list(ctx.graph.nodes):
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
                    downstream = list(_downstream_non_view_like_users(rescale))
                    if any(_is_target_op(user) for user in downstream):
                        avg_pool_source = rescale
                    if any(_is_control_flow_binary_tensor_op(user) for user in downstream):
                        binary_sources.append(rescale)

                if avg_pool_source is None or len(binary_sources) != 2:
                    continue

                for binary_source in binary_sources:
                    if binary_source is avg_pool_source:
                        continue
                    if (
                        not getattr(binary_source, "args", None)
                        or binary_source.args[0] is not node
                    ):
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
                _topologically_rebuild_graph_module(ctx.submodule)
                _cleanup_graph(ctx.submodule.graph)
            return updated

        _for_each_cf_submodule(graph_module, _canonicalize_pool_binary_sources)

        if canonicalized:
            logger.info(
                "[fast_preprocess] canonicalized %d control-flow pool/binary rescale edge(s)",
                canonicalized,
            )

    restored = 0

    def _visit(ctx):
        nonlocal restored
        modified = False
        for placeholder in list(ctx.placeholders):

            input_qparam = _extract_single_qparam_entry(placeholder.meta.get("input_qparams"))
            if getattr(input_qparam, "dtype", None) not in (torch.int8, torch.int16):
                continue

            for target_node in list(getattr(placeholder, "users", ())):
                if not _is_target_op(target_node):
                    continue

                contract = _ControlFlowInputBoundaryContract(
                    submodule=ctx.submodule,
                    index=-1,
                    placeholder=placeholder,
                    input_qparam=input_qparam,
                    output_qparam=input_qparam,
                )
                quant_boundary, boundary_modified = _apply_control_flow_input_boundary_contract(
                    contract,
                    [target_node],
                    preserve_boundary=True,
                    require_preserve_boundary=True,
                )
                if boundary_modified:
                    restored += 1
                    modified = True
                    logger.info(
                        "[fast_preprocess] restored control-flow %s quant boundary %s -> %s",
                        mode,
                        getattr(placeholder, "name", "?"),
                        getattr(quant_boundary, "name", "?"),
                    )

                target_node.replace_input_with(placeholder, quant_boundary)
                _stamp_single_qparams(target_node, input_qparam, input_qparam)
                fx_helpers.refresh_node_fake_tensor(target_node)

                if mode == "pool":
                    target_users = [
                        user
                        for user in list(getattr(target_node, "users", ()))
                        if hasattr(user, "op")
                    ]
                    if target_users and all(
                        _is_control_flow_view_like(user) for user in target_users
                    ):
                        for view_user in target_users:
                            _stamp_single_qparams(
                                view_user, input_qparam, input_qparam
                            )
                            fx_helpers.refresh_node_fake_tensor(view_user)
                        modified = True
                        logger.info(
                            "[fast_preprocess] refreshed control-flow pool/view quant metadata for %s",
                            getattr(target_node, "name", "?"),
                        )
                else:
                    modified = True
                    logger.info(
                        "[fast_preprocess] refreshed control-flow unary quant metadata for %s",
                        getattr(target_node, "name", "?"),
                    )

        if modified:
            _cleanup_graph(ctx.graph)
        return modified

    _for_each_cf_submodule(graph_module, _visit)

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow %s quant boundary rescale(s)",
            restored,
            mode,
        )


def _refresh_control_flow_binary_input_rescales(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    refreshed = 0
    for _, submodule, control_flow_node in get_cond_while_submodules_nested(graph_module):
        env = _collect_control_flow_binary_alignment_env(submodule, control_flow_node)
        updated = False

        for node in _iter_control_flow_binary_nodes(submodule):

            for idx, arg, source_node in _iter_control_flow_binary_rescale_args(
                node, rescale_dtype=torch.int32
            ):
                output_qparam = _extract_single_qparam_entry(arg.meta.get("output_qparams"))
                if output_qparam is None:
                    node_input_qparams = node.meta.get("input_qparams") or {}
                    output_qparam = _extract_single_qparam_entry(
                        {0: node_input_qparams.get(idx)}
                    )

                if getattr(source_node.meta.get("val"), "dtype", None) == torch.int32:
                    cloned_preferred = (
                        _maybe_clone_preferred_control_flow_quantized_binary_output_rescale(
                            submodule,
                            arg,
                            source_node,
                            rescale_output_qparam_fn=_binary_rescale_output_qparam,
                        )
                    )
                    if cloned_preferred is not None:
                        cloned_quant_rescale, quant_rescale, quant_qparam = cloned_preferred
                        cloned_quant_rescale.meta["output_qparams"] = {0: quant_qparam}
                        for quant_user in list(getattr(quant_rescale, "users", ())):
                            quant_user.replace_input_with(
                                quant_rescale, cloned_quant_rescale
                            )
                        if len(getattr(quant_rescale, "users", ())) == 0:
                            submodule.graph.erase_node(quant_rescale)
                        arg.update_arg(0, cloned_quant_rescale)
                        source_node = cloned_quant_rescale
                        if output_qparam is not None:
                            _retune_rescale_qparams(
                                arg,
                                input_qparam=quant_qparam,
                                output_qparam=output_qparam,
                            )
                        else:
                            _stamp_single_qparams(arg, input_qparam=quant_qparam)
                        updated = True
                        refreshed += 1
                        logger.info(
                            "[fast_preprocess] cloned quantized control-flow binary output %s for %s arg%d via %s",
                            getattr(source_node, "name", "?"),
                            getattr(node, "name", "?"),
                            idx,
                            getattr(quant_rescale, "name", "?"),
                        )
                sibling_align_qparam = None
                sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
                source_spec = _resolve_control_flow_binary_operand_spec(
                    arg,
                    rescale_output_qparam_fn=_binary_rescale_output_qparam,
                )
                source_qparam = (
                    source_spec[1] if source_spec is not None else _extract_effective_node_qparam(source_node)
                )
                if hasattr(sibling, "op"):
                    sibling_align_qparam = _extract_single_qparam_entry(
                        getattr(sibling, "meta", {}).get("input_qparams")
                    )
                current_output_qparam = _binary_rescale_output_qparam(arg)
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
                    aligned_source, _ = _ControlFlowRescaleEdit(
                        submodule=submodule,
                        anchor=arg,
                        source_node=source_node,
                        input_qparam=source_qparam,
                        output_qparam=sibling_align_qparam,
                    ).ensure()
                    arg.update_arg(0, aligned_source)
                    _retune_rescale_qparams(
                        arg,
                        input_qparam=sibling_align_qparam,
                        output_qparam=current_output_qparam,
                    )
                    source_node = aligned_source
                    source_qparam = sibling_align_qparam
                    refreshed += 1
                    updated = True
                    logger.info(
                        "[fast_preprocess] aligned control-flow binary input %s arg%d through sibling domain %s",
                        getattr(node, "name", "?"),
                        idx,
                        sibling_align_qparam,
                    )
                source_fake_val = source_node.meta.get("val")
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

        for node in _iter_control_flow_binary_nodes(submodule):
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
                operand_spec = _resolve_control_flow_binary_operand_spec(
                    arg,
                    rescale_output_qparam_fn=_binary_rescale_output_qparam,
                )
                if operand_spec is None:
                    continue
                source_node, source_qparam = operand_spec
                desired_input_qparam = desired_input_qparams[idx]
                current_arg = node.args[idx]
                lift, inserted = _ControlFlowRescaleEdit(
                    submodule=submodule,
                    anchor=node,
                    source_node=source_node,
                    input_qparam=source_qparam,
                    output_qparam=desired_input_qparam,
                    current_node=current_arg,
                ).ensure()
                if inserted:
                    node.update_arg(idx, lift)
                    refreshed += 1
                    updated = True

        for refresh_cfg in (
            dict(
                predicate=_is_control_flow_binary_tensor_op,
                allow_branch_local_rescale=False,
                accept_qparams=lambda lhs_qparam, _rhs_qparam: getattr(
                    lhs_qparam, "dtype", None
                )
                in (torch.int8, torch.int16),
                contract_fn=_additive_binary_qparam_contract,
                terminal_outputs_only=False,
                qparam_log_message="[fast_preprocess] recomputed control-flow binary accumulator qparams for %s -> %s",
                qparam_log_value_fn=lambda contract: contract.output_qparam,
                output_rescale_log_message="[fast_preprocess] realigned %d control-flow binary output rescale(s) from %s",
            ),
            dict(
                predicate=_is_control_flow_mul_tensor_op,
                allow_branch_local_rescale=True,
                accept_qparams=lambda _lhs_qparam, _rhs_qparam: True,
                contract_fn=_mul_binary_qparam_contract,
                terminal_outputs_only=True,
                qparam_log_message="[fast_preprocess] recomputed control-flow mul qparams for %s -> %s",
                qparam_log_value_fn=lambda contract: (
                    contract.input_qparams[0],
                    contract.input_qparams[1],
                    contract.output_qparam,
                ),
                output_rescale_log_message="[fast_preprocess] realigned %d control-flow mul output rescale(s) from %s",
            ),
        ):
            node_updated, node_refreshed = _refresh_control_flow_binary_contract_nodes(
                submodule,
                env,
                **refresh_cfg,
            )
            refreshed += node_refreshed
            updated |= node_updated

        for node in _iter_control_flow_binary_nodes(submodule):
            node_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
            for idx, arg, source_node in _iter_control_flow_binary_rescale_args(
                node, rescale_dtype=torch.int32
            ):
                desired_output_qparam = _extract_single_qparam_entry({0: node_input_qparams.get(idx)})
                if (
                    _is_tosa_rescale_node(source_node)
                    and len(getattr(source_node, "args", ())) >= 5
                    and source_node.args[1] in (torch.int8, torch.int16)
                    and hasattr(source_node.args[0], "op")
                    and getattr(source_node.args[0], "op", None) == "placeholder"
                ):
                    placeholder_source = source_node.args[0]
                    align_qparam = _binary_rescale_output_qparam(source_node)
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
                _retune_rescale_qparams(
                    arg,
                    input_qparam=source_qparam,
                    output_qparam=desired_output_qparam,
                )
                refreshed += 1
                updated = True
                logger.info(
                    "[fast_preprocess] realigned control-flow binary lift %s arg%d to node input qparams %s",
                    getattr(node, "name", "?"),
                    idx,
                    desired_output_qparam,
                )

        if updated:
            _cleanup_and_recompile(submodule)

    if refreshed:
        logger.info(
            "[fast_preprocess] refreshed %d control-flow binary input rescale(s)",
            refreshed,
        )


def _build_control_flow_input_boundary_contracts(
    submodule, input_qparams_map=None, *, current_only: bool = False
):
    contracts = []
    for idx, placeholder in enumerate(_placeholder_nodes(submodule)):
        input_qparam = (input_qparams_map or {}).get(idx)
        if input_qparam is None:
            input_qparam = _extract_single_qparam_entry(placeholder.meta.get("input_qparams"))
        if input_qparam is None and not current_only:
            input_qparam = _branch_local_control_flow_output_qparam(placeholder)
        if input_qparam is None:
            continue
        output_qparam = _branch_local_control_flow_output_qparam(placeholder)
        if output_qparam is None and current_only:
            continue
        if output_qparam is None and _placeholder_feeds_binary_directly(placeholder):
            output_qparam = input_qparam
        if output_qparam is None:
            output_qparam = input_qparam
        contracts.append(
            _ControlFlowInputBoundaryContract(
                submodule=submodule,
                index=idx,
                placeholder=placeholder,
                input_qparam=input_qparam,
                output_qparam=output_qparam,
            )
        )
    return contracts


def _build_control_flow_output_boundary_contracts(
    submodule,
    output_qparams_map,
    *,
    allow_placeholder_fallback: bool,
):
    if not output_qparams_map:
        return []

    _refresh_submodule_output_input_qparams(
        submodule,
        allow_placeholder_fallback=allow_placeholder_fallback,
    )
    if not submodule.graph.output_node().meta.get("input_qparams"):
        return []

    return [
        _ControlFlowOutputBoundaryContract(
            submodule=submodule,
            index=idx,
            target_qparam=target_qparam,
        )
        for idx, target_qparam in _filter_output_qparams_map(
            submodule, output_qparams_map
        ).items()
    ]


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

    rescale_pass = InsertControlFlowRescalesPass()
    modified = False
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.higher_order.cond:
            if_graph = graph_module.get_submodule(node.args[1].target)
            else_graph = graph_module.get_submodule(node.args[2].target)
            seed_input_qparams_map = _seed_control_flow_input_qparams_map(
                node, 3, rescale_pass
            )
            _log_control_flow_debug(
                "[fast_preprocess] reapply cond input qparams %s %s",
                getattr(node, "name", "?"),
                seed_input_qparams_map,
            )
            if include_inputs:
                for submodule in (if_graph, else_graph):
                    contracts = _build_control_flow_input_boundary_contracts(
                        submodule, seed_input_qparams_map
                    )
                    if not contracts:
                        continue
                    modified |= _reapply_control_flow_input_boundary_rescales(
                        contracts,
                        rescale_pass=rescale_pass,
                    )

            output_qparams_map = _infer_control_flow_output_qparams_map(node)
            if include_outputs and output_qparams_map:
                for submodule in (if_graph, else_graph):
                    modified |= _reapply_control_flow_output_boundary_rescales(
                        _build_control_flow_output_boundary_contracts(
                            submodule,
                            output_qparams_map,
                            allow_placeholder_fallback=True,
                        ),
                        rescale_pass=rescale_pass,
                    )
        elif node.target == torch.ops.higher_order.while_loop:
            cond_graph = graph_module.get_submodule(node.args[0].target)
            body_graph = graph_module.get_submodule(node.args[1].target)
            seed_input_qparams_map = _seed_control_flow_input_qparams_map(
                node, 2, rescale_pass
            )
            if include_inputs:
                for submodule in (cond_graph, body_graph):
                    contracts = _build_control_flow_input_boundary_contracts(
                        submodule, seed_input_qparams_map
                    )
                    if not contracts:
                        continue
                    modified |= _reapply_control_flow_input_boundary_rescales(
                        contracts,
                        rescale_pass=rescale_pass,
                    )

            # While body outputs are loop-carried, so rescale them to the
            # carried-input qparams rather than the post-loop getitem dtype.
            body_output_contracts = _build_control_flow_output_boundary_contracts(
                body_graph,
                seed_input_qparams_map,
                allow_placeholder_fallback=True,
            )
            if include_outputs and body_output_contracts:
                modified |= _reapply_control_flow_output_boundary_rescales(
                    body_output_contracts,
                    rescale_pass=rescale_pass,
                )

    if not modified:
        return

    _cleanup_and_recompile(graph_module)

    for _, submodule, _ in get_cond_while_submodules_nested(graph_module):
        _cleanup_and_recompile(submodule)

        logger.info(
            "[fast_preprocess] reapplied control-flow boundary rescales after rust passes"
        )


def _prune_duplicate_control_flow_rescales(graph_module) -> None:
    pruned = 0

    def _visit(ctx):
        nonlocal pruned
        updated = False
        for node in list(ctx.graph.nodes):
            if not _is_tosa_rescale_node(node):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue
            source = node.args[0]
            if _is_identity_rescale(node):
                _copy_output_contract_meta(source, node)

                node.replace_all_uses_with(source)
                ctx.graph.erase_node(node)
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

            _copy_output_contract_meta(source, node)

            node.replace_all_uses_with(source)
            ctx.graph.erase_node(node)
            pruned += 1
            updated = True
            logger.info(
                "[fast_preprocess] pruned duplicate control-flow rescale %s -> %s",
                getattr(node, "name", "?"),
                getattr(source, "name", "?"),
            )

        if updated:
            _cleanup_graph(ctx.graph)
        return updated

    _for_each_cf_submodule(graph_module, _visit)

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


def _control_flow_output_target_qparam(candidate, output_args, output_qparams_map):
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


def _terminal_control_flow_output_target_qparam(candidate, output_args, output_qparams_map):
    import torch

    target_qparam = _control_flow_output_target_qparam(
        candidate, output_args, output_qparams_map
    )
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


def _additive_binary_qparam_contract(lhs_qparam, rhs_qparam):
    import torch

    shift_bits = 12 if lhs_qparam.dtype == torch.int16 else 20
    accumulator_scale = (
        2.0
        * max(
            float(lhs_qparam.get_scale_per_tensor()),
            float(rhs_qparam.get_scale_per_tensor()),
        )
        / float(1 << shift_bits)
    )
    accumulator_qparam = _control_flow_int32_accumulator_qparam(accumulator_scale)
    return _ControlFlowBinaryQparamContract(
        input_qparams={0: accumulator_qparam, 1: accumulator_qparam},
        output_qparam=accumulator_qparam,
    )


def _mul_binary_qparam_contract(lhs_qparam, rhs_qparam):
    lhs_int32_qparam = _control_flow_int32_accumulator_qparam(
        lhs_qparam.get_scale_per_tensor()
    )
    rhs_int32_qparam = _control_flow_int32_accumulator_qparam(
        rhs_qparam.get_scale_per_tensor()
    )
    output_qparam = _control_flow_int32_accumulator_qparam(
        lhs_qparam.get_scale_per_tensor() * rhs_qparam.get_scale_per_tensor()
    )
    return _ControlFlowBinaryQparamContract(
        input_qparams={0: lhs_int32_qparam, 1: rhs_int32_qparam},
        output_qparam=output_qparam,
    )


def _apply_control_flow_binary_qparam_contract(
    node,
    contract,
    *,
    output_args,
    output_qparams_map,
    terminal_outputs_only: bool,
):
    current_input_qparams = getattr(node, "meta", {}).get("input_qparams") or {}
    current_output_qparam = _extract_single_qparam_entry(
        getattr(node, "meta", {}).get("output_qparams")
    )
    updated = False
    if not (
        all(
            _same_control_flow_qparam(current_input_qparams.get(idx), qparam)
            for idx, qparam in contract.input_qparams.items()
        )
        and _same_control_flow_qparam(current_output_qparam, contract.output_qparam)
    ):
        node.meta["input_qparams"] = dict(contract.input_qparams)
        node.meta["output_qparams"] = {0: contract.output_qparam}
        updated = True

    target_qparam_fn = (
        _terminal_control_flow_output_target_qparam
        if terminal_outputs_only
        else _control_flow_output_target_qparam
    )
    retuned = _retune_control_flow_quantized_output_users(
        node,
        input_qparam=contract.output_qparam,
        resolve_output_qparam=lambda user: (
            target_qparam_fn(user, output_args, output_qparams_map)
            or _extract_effective_node_qparam(user)
            or _binary_rescale_output_qparam(user)
        ),
    )
    return updated, retuned


def _enforce_control_flow_binary_contract(
    submodule,
    node,
    operand_specs,
    contract,
    *,
    env,
    terminal_outputs_only: bool,
    qparam_log_message: str,
    qparam_log_value,
    output_rescale_log_message: str,
    insert_operand_lifts: bool = True,
):
    refreshed = 0
    updated = False
    qparams_updated, retuned = _apply_control_flow_binary_qparam_contract(
        node,
        contract,
        output_args=env.output_args,
        output_qparams_map=env.output_qparams_map,
        terminal_outputs_only=terminal_outputs_only,
    )
    if qparams_updated:
        refreshed += 1
        updated = True
        logger.info(qparam_log_message, getattr(node, "name", "?"), qparam_log_value)

    if insert_operand_lifts:
        for idx, (source_node, source_qparam) in operand_specs.items():
            desired_input_qparam = contract.input_qparams.get(idx)
            if desired_input_qparam is None:
                continue
            lift, inserted = _ControlFlowRescaleEdit(
                submodule=submodule,
                anchor=node,
                source_node=source_node,
                input_qparam=source_qparam,
                output_qparam=desired_input_qparam,
                current_node=node.args[idx],
            ).ensure()
            if inserted:
                node.update_arg(idx, lift)
                refreshed += 1
                updated = True

    if retuned:
        refreshed += retuned
        updated = True
        logger.info(output_rescale_log_message, retuned, getattr(node, "name", "?"))

    return updated, refreshed


def _refresh_control_flow_binary_contract_nodes(
    submodule,
    env,
    *,
    predicate,
    allow_branch_local_rescale: bool,
    accept_qparams,
    contract_fn,
    terminal_outputs_only: bool,
    qparam_log_message: str,
    qparam_log_value_fn,
    output_rescale_log_message: str,
):
    refreshed = 0
    updated = False
    for node in _iter_control_flow_call_function_nodes(submodule, predicate):
        operand_specs = _collect_control_flow_binary_operand_specs(
            node,
            rescale_output_qparam_fn=_binary_rescale_output_qparam,
            allow_branch_local_rescale=allow_branch_local_rescale,
        )
        if len(operand_specs) != 2:
            continue
        lhs_qparam = operand_specs[0][1]
        rhs_qparam = operand_specs[1][1]
        if getattr(lhs_qparam, "dtype", None) != getattr(rhs_qparam, "dtype", None):
            continue
        if not accept_qparams(lhs_qparam, rhs_qparam):
            continue
        contract = contract_fn(lhs_qparam, rhs_qparam)
        node_updated, node_refreshed = _enforce_control_flow_binary_contract(
            submodule,
            node,
            operand_specs,
            contract,
            env=env,
            terminal_outputs_only=terminal_outputs_only,
            qparam_log_message=qparam_log_message,
            qparam_log_value=qparam_log_value_fn(contract),
            output_rescale_log_message=output_rescale_log_message,
        )
        refreshed += node_refreshed
        updated |= node_updated
    return updated, refreshed


def _log_control_flow_binary_debug_state(node, *, prefix: str) -> None:
    _log_control_flow_debug(
        "%s %s input_qparams=%r output_qparams=%r",
        prefix,
        _describe_node_brief(node),
        node.meta.get("input_qparams"),
        node.meta.get("output_qparams"),
    )
    for idx, arg in enumerate(node.args[:2]):
        if not hasattr(arg, "op"):
            continue
        _log_control_flow_debug(
            "%s arg%d %s input_qparams=%r output_qparams=%r args=%r kwargs=%r",
            prefix,
            idx,
            _describe_node_brief(arg),
            arg.meta.get("input_qparams"),
            arg.meta.get("output_qparams"),
            getattr(arg, "args", None),
            getattr(arg, "kwargs", None),
        )


def _quantize_constant_only_control_flow_outputs(graph_module) -> None:
    rewritten = 0

    def _visit(ctx):
        nonlocal rewritten
        if ctx.placeholders:
            return False

        if ctx.cf_node is None:
            return False

        output_qparams_map = _infer_control_flow_output_qparams_map(ctx.cf_node)
        if not output_qparams_map:
            return False

        output_node = ctx.output_node
        wrapped_single = hasattr(output_node.args[0], "op")
        output_args = _output_node_args(output_node)
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

            const_node = _materialize_quantized_constant_node(
                ctx.submodule,
                ctx.graph,
                output_node,
                source,
                qparam,
                const_name_prefix=f"{getattr(source, 'name', 'const')}_quantized_output_{idx}",
            )
            if const_node is None:
                continue

            output_args[idx] = const_node
            updated = True
            rewritten += 1
            logger.info(
                "[fast_preprocess] quantized constant-only control-flow output %s for output index %d",
                getattr(const_node, "name", "?"),
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
            _cleanup_graph(ctx.graph)
        return updated

    _for_each_cf_submodule(graph_module, _visit)

    if rewritten:
        logger.info(
            "[fast_preprocess] quantized %d constant-only control-flow output(s)",
            rewritten,
        )


def _quantize_tensor(value, qparam):
    import torch

    if qparam.per_channel:
        return None
    scale = _qparam_scale_value(qparam)
    if scale == 0:
        return None
    quantized = torch.round(value.detach().to(torch.float32) / scale) + _qparam_zp_value(
        qparam
    )
    quantized = torch.clamp(quantized, int(qparam.qmin), int(qparam.qmax))
    return quantized.to(qparam.dtype)


def _materialize_quantized_constant_node(
    submodule,
    graph,
    anchor_node,
    source_node,
    qparam,
    *,
    const_name_prefix: str,
    include_get_attr: bool = True,
):
    constant_value = _constant_tensor_from_node(
        source_node,
        submodule=submodule,
        include_get_attr=include_get_attr,
    )
    if constant_value is None:
        return None

    quantized_value = _quantize_tensor(constant_value, qparam)
    if quantized_value is None:
        return None

    const_name = const_name_prefix
    suffix = 0
    while hasattr(submodule, const_name):
        suffix += 1
        const_name = f"{const_name_prefix}_{suffix}"
    submodule.register_buffer(const_name, quantized_value)
    with graph.inserting_before(anchor_node):
        const_node = graph.get_attr(const_name)

    const_node.meta = source_node.meta.copy()
    const_node.meta["val"] = quantized_value
    if source_node.meta.get("tosa_dim_order") is not None:
        const_node.meta["tosa_dim_order"] = tuple(source_node.meta["tosa_dim_order"])
    _stamp_single_qparams(const_node, qparam, qparam)
    return const_node


def _constant_tensor_from_node(source, *, submodule=None, include_get_attr: bool = True):
    import torch

    if include_get_attr and getattr(source, "op", None) == "get_attr":
        if submodule is None:
            return None
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

        scale = _qparam_scale_value(source_qparam)
        if scale == 0:
            return None
        return (
            constant_value.detach().to(torch.float32) - _qparam_zp_value(source_qparam)
        ) * scale

    if not _is_call_function_target(source, "aten.full.default"):
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

    fake_val = getattr(source, "meta", {}).get("val")
    dtype = getattr(fake_val, "dtype", torch.float32)
    return torch.full(shape, fill_scalar, dtype=dtype)

def _is_tosa_rescale_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return (
        "tosa_RESCALE" in target_str
        or "tosa.RESCALE" in target_str
        or "TosaRescale" in target_str
    )


def _replace_submodule(root_module, target: str, replacement) -> None:
    atoms = target.split(".")
    parent = root_module
    for atom in atoms[:-1]:
        parent = parent.get_submodule(atom)
    setattr(parent, atoms[-1], replacement)


def _is_tosa_table_node(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    target_str = str(getattr(node, "target", ""))
    return "tosa.TABLE" in target_str or "tosa::table" in target_str


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
    del arg
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
    return _is_control_flow_binary_tensor_op(source)


def _is_dequantize_node(node) -> bool:
    return _is_call_function_target(node, "dequantize_per_tensor")


def _is_quantize_node(node) -> bool:
    return _is_call_function_target(
        node, "quantize_per_tensor"
    ) and not _is_call_function_target(node, "dequantize_per_tensor")

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


def _preserve_control_flow_output_meta(current_fake_tensor, desired_fake_tensor) -> bool:
    current_dtype = getattr(current_fake_tensor, "dtype", None)
    desired_dtype = getattr(desired_fake_tensor, "dtype", None)
    return current_dtype is not None and str(current_dtype) == "torch.bool" and (
        desired_dtype is None or str(desired_dtype) != "torch.bool"
    )


def _qparam_scale_value(qparam) -> float:
    if qparam is None:
        return 1.0
    getter = getattr(qparam, "get_scale_per_tensor", None)
    if callable(getter):
        return float(getter())
    scale = getattr(qparam, "scale", 1.0)
    if isinstance(scale, (list, tuple)):
        return float(scale[0]) if scale else 1.0
    return float(scale)


def _qparam_zp_value(qparam) -> int:
    if qparam is None:
        return 0
    getter = getattr(qparam, "get_zp_per_tensor", None)
    if callable(getter):
        return int(getter())
    zp = getattr(qparam, "zp", getattr(qparam, "zero_point", 0))
    if isinstance(zp, (list, tuple)):
        return int(zp[0]) if zp else 0
    return int(zp)


def _same_control_flow_qparam(lhs, rhs, *, include_range: bool = False) -> bool:
    if lhs is None or rhs is None:
        return False
    same = (
        _qparam_scale_value(lhs) == _qparam_scale_value(rhs)
        and _qparam_zp_value(lhs) == _qparam_zp_value(rhs)
        and getattr(lhs, "dtype", None) == getattr(rhs, "dtype", None)
    )
    if include_range:
        same = (
            same
            and getattr(lhs, "qmin", None) == getattr(rhs, "qmin", None)
            and getattr(lhs, "qmax", None) == getattr(rhs, "qmax", None)
        )
    return same


def _meta_output_qparam(node):
    return _extract_single_qparam_entry(getattr(node, "meta", {}).get("output_qparams"))


def _stamp_single_qparams(node, input_qparam=None, output_qparam=None) -> None:
    if input_qparam is not None:
        node.meta["input_qparams"] = {0: input_qparam}
    if output_qparam is not None:
        node.meta["output_qparams"] = {0: output_qparam}


def _copy_output_contract_meta(dst, src, *, copy_output_qparams: bool = True) -> None:
    dst.meta["val"] = src.meta.get("val", dst.meta.get("val"))
    if src.meta.get("tosa_dim_order") is not None:
        dst.meta["tosa_dim_order"] = tuple(src.meta["tosa_dim_order"])
    if copy_output_qparams and src.meta.get("output_qparams"):
        dst.meta["output_qparams"] = dict(src.meta["output_qparams"])


def _normalize_rescale_scale_arg(scale_arg):
    if isinstance(scale_arg, (list, tuple)):
        try:
            return [float(value) for value in scale_arg]
        except (TypeError, ValueError):
            return None
    try:
        return [float(scale_arg)]
    except (TypeError, ValueError):
        return None


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
    import math
    import torch

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

    scales = _normalize_rescale_scale_arg(args[2])
    if scales is None:
        return False
    try:
        scale_is_identity = all(
            math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12) for scale in scales
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


def _single_rescale_scale(node):
    if len(getattr(node, "args", ())) < 3:
        return None
    scale_arg = node.args[2]
    if (
        not isinstance(scale_arg, (list, tuple))
        or len(scale_arg) != 1
        or scale_arg[0] in (0, 0.0)
    ):
        return None
    try:
        return float(scale_arg[0])
    except (TypeError, ValueError):
        return None


def _rescale_ratio(input_qparam, output_qparam) -> list[float]:
    return [_qparam_scale_value(input_qparam) / _qparam_scale_value(output_qparam)]


def _retune_rescale_qparams(node, *, input_qparam, output_qparam) -> None:
    node.update_arg(2, _rescale_ratio(input_qparam, output_qparam))
    node.update_arg(3, _qparam_zp_value(input_qparam))
    node.update_arg(4, _qparam_zp_value(output_qparam))
    _stamp_single_qparams(node, input_qparam, output_qparam)


def _ensure_control_flow_rescale_node(
    submodule,
    anchor,
    source_node,
    *,
    input_qparam,
    output_qparam,
    current_node=None,
    after: bool = False,
    preserve_boundary: bool = False,
):
    import fx_helpers

    if (
        _is_tosa_rescale_node(current_node)
        and len(getattr(current_node, "args", ())) >= 5
        and current_node.args[1] == getattr(output_qparam, "dtype", None)
    ):
        current_node.update_arg(0, source_node)
        _retune_rescale_qparams(
            current_node, input_qparam=input_qparam, output_qparam=output_qparam
        )
        if preserve_boundary:
            current_node.meta["arm_preserve_control_flow_quant_boundary"] = True
        return current_node, False

    rescale = fx_helpers.insert_rescale_node(
        submodule,
        anchor,
        source_node,
        output_qparam.dtype,
        _rescale_ratio(input_qparam, output_qparam),
        _qparam_zp_value(input_qparam),
        _qparam_zp_value(output_qparam),
        after=after,
    )
    _stamp_single_qparams(rescale, input_qparam, output_qparam)
    if preserve_boundary:
        rescale.meta["arm_preserve_control_flow_quant_boundary"] = True
    return (
        rescale,
        True,
    )


def _apply_control_flow_input_boundary_contract(
    contract,
    users,
    *,
    preserve_boundary: bool,
    require_preserve_boundary: bool,
):
    placeholder = contract.placeholder
    target_user = users[0] if len(users) == 1 else None
    boundary = None
    for user in list(getattr(placeholder, "users", ())):
        if not _is_tosa_rescale_node(user):
            continue
        if (
            require_preserve_boundary
            and not user.meta.get("arm_preserve_control_flow_quant_boundary", False)
        ):
            continue
        if (
            len(getattr(user, "args", ())) >= 2
            and user.args[1] != contract.output_qparam.dtype
        ):
            continue
        if not _same_control_flow_qparam(
            _extract_single_qparam_entry(user.meta.get("input_qparams")),
            contract.input_qparam,
        ) or not _same_control_flow_qparam(
            _extract_single_qparam_entry(user.meta.get("output_qparams")),
            contract.output_qparam,
        ):
            continue
        if target_user is not None and target_user not in getattr(user, "users", ()):
            continue
        boundary = user
        break
    rescale, inserted = _ControlFlowRescaleEdit.from_input_boundary_contract(
        contract,
        current_node=boundary,
        preserve_boundary=preserve_boundary,
    ).ensure()
    modified = inserted
    for user in users:
        if user is rescale or placeholder not in getattr(user, "all_input_nodes", ()):
            continue
        user.replace_input_with(placeholder, rescale)
        modified = True
    return rescale, modified


def _retune_control_flow_quantized_output_users(
    source_node,
    *,
    input_qparam,
    resolve_output_qparam,
):
    import torch

    retuned = 0
    for user in list(getattr(source_node, "users", ())):
        if not _is_tosa_rescale_node(user):
            continue
        if len(getattr(user, "args", ())) < 5:
            continue
        if user.args[1] not in (torch.int8, torch.int16):
            continue
        output_qparam = resolve_output_qparam(user)
        if output_qparam is None:
            continue
        if getattr(output_qparam, "dtype", None) != user.args[1]:
            continue
        _retune_rescale_qparams(
            user, input_qparam=input_qparam, output_qparam=output_qparam
        )
        retuned += 1
    return retuned


def _repair_control_flow_qparam(node):
    return _extract_effective_node_qparam(
        node,
        parse_operator_qparams=True,
        fallback_to_consumer_qparams=True,
        prefer_input_qparams=False,
    )


def _repair_control_flow_rescale_output_qparam(node):
    return _binary_rescale_output_qparam(
        node,
        fallback_qparam_fn=_repair_control_flow_qparam,
        source_qparam_fn=_repair_control_flow_qparam,
    )


def _lookup_control_flow_input_qparam(arg, placeholder_qparams, placeholder_index_by_name):
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


def _reconstruct_control_flow_rescale_source_qparam(binary_node, arg_idx, rescale_node):
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    if not _is_tosa_rescale_node(rescale_node):
        return None
    if len(getattr(rescale_node, "args", ())) < 5:
        return None
    source_node = rescale_node.args[0]
    if not hasattr(source_node, "op"):
        return None
    if source_node.meta.get("output_qparams"):
        return _repair_control_flow_qparam(source_node)

    binary_input_qparams = binary_node.meta.get("input_qparams") or {}
    output_qparam = binary_input_qparams.get(arg_idx)
    if output_qparam is None:
        output_qparam = _repair_control_flow_qparam(rescale_node)
    if output_qparam is None:
        return None

    rescale_scale = _single_rescale_scale(rescale_node)
    if rescale_scale is None:
        return None

    source_dtype = getattr(source_node.meta.get("val"), "dtype", None)
    if source_dtype not in (torch.int8, torch.int16):
        return None

    try:
        qinfo = torch.iinfo(source_dtype)
        source_qparam = QuantArgs(
            scale=rescale_scale * output_qparam.get_scale_per_tensor(),
            zp=int(rescale_node.args[3]),
            qmin=int(qinfo.min),
            qmax=int(qinfo.max),
            dtype=source_dtype,
        )
    except Exception:
        return None

    _stamp_single_qparams(source_node, source_qparam, source_qparam)
    logger.info(
        "[fast_preprocess] reconstructed control-flow binary source qparams %s arg%d -> %s",
        getattr(binary_node, "name", "?"),
        arg_idx,
        source_qparam,
    )
    return source_qparam


def _reconstruct_control_flow_binary_output_qparam(binary_node, rescale_node):
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    if not _is_tosa_rescale_node(rescale_node):
        return None
    if len(getattr(rescale_node, "args", ())) < 5:
        return None

    source_qparam = _extract_single_qparam_entry(
        binary_node.meta.get("output_qparams") or {}
    )
    if source_qparam is None:
        source_qparam = _repair_control_flow_qparam(binary_node)
    if source_qparam is None:
        return None

    rescale_scale = _single_rescale_scale(rescale_node)
    if rescale_scale is None:
        return None

    output_dtype = rescale_node.args[1]
    if output_dtype not in (torch.int8, torch.int16):
        return None

    try:
        qinfo = torch.iinfo(output_dtype)
        output_qparam = QuantArgs(
            scale=source_qparam.get_scale_per_tensor() / rescale_scale,
            zp=int(rescale_node.args[4]),
            qmin=int(qinfo.min),
            qmax=int(qinfo.max),
            dtype=output_dtype,
        )
    except Exception:
        return None

    _stamp_single_qparams(binary_node, output_qparam=output_qparam)
    logger.info(
        "[fast_preprocess] reconstructed control-flow binary output qparams %s -> %s",
        getattr(binary_node, "name", "?"),
        output_qparam,
    )
    return output_qparam


def _binary_rescale_output_qparam(
    node, *, fallback_qparam_fn=None, source_qparam_fn=None
):
    import torch
    from executorch.backends.arm._passes.quant_args import QuantArgs

    fallback_qparam_fn = fallback_qparam_fn or _meta_output_qparam
    source_qparam_fn = source_qparam_fn or _extract_effective_node_qparam

    if not _is_tosa_rescale_node(node):
        return fallback_qparam_fn(node)
    if len(getattr(node, "args", ())) < 5:
        return fallback_qparam_fn(node)

    output_dtype = node.args[1]
    if output_dtype not in (torch.int8, torch.int16, torch.int32):
        return fallback_qparam_fn(node)

    source_node = node.args[0]
    if not hasattr(source_node, "op"):
        return fallback_qparam_fn(node)

    source_qparam = source_qparam_fn(source_node)
    rescale_scale = _single_rescale_scale(node)
    if source_qparam is None or rescale_scale is None:
        return fallback_qparam_fn(node)

    try:
        qinfo = torch.iinfo(output_dtype)
        return QuantArgs(
            scale=source_qparam.get_scale_per_tensor() / rescale_scale,
            zp=int(node.args[4]),
            qmin=int(qinfo.min),
            qmax=int(qinfo.max),
            dtype=output_dtype,
        )
    except Exception:
        return fallback_qparam_fn(node)


def _preferred_control_flow_quantized_binary_output_rescale(
    source_node, *, rescale_output_qparam_fn
):
    import torch

    if not _is_control_flow_binary_tensor_op(source_node):
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
        output_qparam = rescale_output_qparam_fn(user)
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


def _control_flow_int32_accumulator_qparam(scale: float):
    import torch
    from executorch.backends.arm._passes.quant_args import QuantArgs

    qinfo = torch.iinfo(torch.int32)
    return QuantArgs(
        scale=scale,
        zp=0,
        qmin=int(qinfo.min),
        qmax=int(qinfo.max),
        dtype=torch.int32,
    )


def _resolve_control_flow_binary_operand_spec(
    arg,
    *,
    rescale_output_qparam_fn,
    allow_branch_local_rescale: bool = False,
):
    import torch

    if not hasattr(arg, "op"):
        return None
    source_node = arg
    source_qparam = None
    if (
        _is_tosa_rescale_node(arg)
        and len(getattr(arg, "args", ())) >= 5
        and arg.args[1] == torch.int32
        and hasattr(arg.args[0], "op")
    ):
        source_node = arg.args[0]
        source_qparam = rescale_output_qparam_fn(source_node)
        if source_qparam is None:
            source_qparam = _extract_effective_node_qparam(source_node)
    elif _is_tosa_rescale_node(arg) and len(getattr(arg, "args", ())) >= 5:
        source_qparam = rescale_output_qparam_fn(arg)

    if (
        source_qparam is None
        and allow_branch_local_rescale
        and getattr(arg, "op", None) == "placeholder"
    ):
        branch_qparam = _branch_local_control_flow_output_qparam(arg)
        if branch_qparam is not None:
            for user in list(getattr(arg, "users", ())):
                if not _is_tosa_rescale_node(user):
                    continue
                if len(getattr(user, "args", ())) < 5:
                    continue
                if user.args[1] not in (torch.int8, torch.int16):
                    continue
                user_qparam = rescale_output_qparam_fn(user)
                if _same_control_flow_qparam(user_qparam, branch_qparam):
                    source_node = user
                    source_qparam = branch_qparam
                    break

    if source_qparam is None:
        source_qparam = _extract_effective_node_qparam(arg)
    if getattr(source_qparam, "dtype", None) in (torch.int8, torch.int16):
        return source_node, source_qparam
    return None


def _collect_control_flow_binary_operand_specs(
    node,
    *,
    rescale_output_qparam_fn,
    allow_branch_local_rescale: bool = False,
):
    operand_specs = {}
    for idx, arg in enumerate(node.args[:2]):
        operand_spec = _resolve_control_flow_binary_operand_spec(
            arg,
            rescale_output_qparam_fn=rescale_output_qparam_fn,
            allow_branch_local_rescale=allow_branch_local_rescale,
        )
        if operand_spec is not None:
            operand_specs[idx] = operand_spec
    return operand_specs


def _resolve_control_flow_binary_operand_qparam(
    arg,
    *,
    rescale_output_qparam_fn,
    allow_branch_local_rescale: bool = False,
    placeholder_qparams=None,
    placeholder_index_by_name=None,
):
    operand_spec = _resolve_control_flow_binary_operand_spec(
        arg,
        rescale_output_qparam_fn=rescale_output_qparam_fn,
        allow_branch_local_rescale=allow_branch_local_rescale,
    )
    if operand_spec is not None:
        return operand_spec[1], False
    if placeholder_qparams is None or placeholder_index_by_name is None:
        return None, False
    qparam = _lookup_control_flow_input_qparam(
        arg,
        placeholder_qparams,
        placeholder_index_by_name,
    )
    return qparam, qparam is not None


def _maybe_clone_preferred_control_flow_quantized_binary_output_rescale(
    submodule,
    ref_node,
    source_node,
    *,
    rescale_output_qparam_fn,
):
    preferred = _preferred_control_flow_quantized_binary_output_rescale(
        source_node,
        rescale_output_qparam_fn=rescale_output_qparam_fn,
    )
    if preferred is None:
        return None
    quant_rescale, output_qparam = preferred
    cloned = _clone_quantized_binary_output_rescale(
        submodule,
        ref_node,
        quant_rescale,
        source_node,
        rescale_output_qparam_fn=rescale_output_qparam_fn,
    )
    if cloned is None:
        return None
    return cloned, quant_rescale, output_qparam


def _resolve_repaired_control_flow_binary_input_qparam(
    node,
    idx,
    arg,
    *,
    env,
):
    import torch

    if not hasattr(arg, "op"):
        return None

    producer_qparam, recovered_from_boundary = _resolve_control_flow_binary_operand_qparam(
        arg,
        rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
        allow_branch_local_rescale=True,
        placeholder_qparams=env.placeholder_qparams,
        placeholder_index_by_name=env.placeholder_index_by_name,
    )
    if recovered_from_boundary:
        logger.info(
            "[fast_preprocess] recovered control-flow input qparams for %s arg%d from parent boundary: %s",
            getattr(node, "name", "?"),
            idx,
            producer_qparam,
        )

    sibling = node.args[1 - idx] if len(node.args) > 1 - idx else None
    sibling_qparam = None
    sibling_binary_output_qparam = None
    if hasattr(sibling, "op"):
        sibling_qparam, _ = _resolve_control_flow_binary_operand_qparam(
            sibling,
            rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
            allow_branch_local_rescale=True,
            placeholder_qparams=env.placeholder_qparams,
            placeholder_index_by_name=env.placeholder_index_by_name,
        )
        preferred_sibling_quantized = _preferred_control_flow_quantized_binary_output_rescale(
            sibling,
            rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
        )
        if preferred_sibling_quantized is not None:
            sibling_binary_output_qparam = preferred_sibling_quantized[1]
    sibling_target_qparam = sibling_binary_output_qparam or sibling_qparam

    if (
        producer_qparam is not None
        and getattr(producer_qparam, "dtype", None) == torch.int32
        and sibling_qparam is not None
    ):
        arg_input_qparams = arg.meta.get("input_qparams") or {}
        if isinstance(arg_input_qparams, dict):
            typed_candidates = [
                candidate
                for candidate in arg_input_qparams.values()
                if getattr(candidate, "dtype", None)
                == getattr(sibling_qparam, "dtype", None)
            ]
            preferred_qparam = next(
                (
                    candidate
                    for candidate in typed_candidates
                    if _same_control_flow_qparam(candidate, sibling_qparam)
                ),
                typed_candidates[0] if typed_candidates else None,
            )
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
        and sibling_target_qparam is not None
        and getattr(producer_qparam, "dtype", None) in (torch.int8, torch.int16)
        and getattr(sibling_target_qparam, "dtype", None) in (torch.int8, torch.int16)
        and (
            (hasattr(sibling, "op") and _is_quantized_binary_result(sibling))
            or sibling_binary_output_qparam is not None
        )
        and not _same_control_flow_qparam(
            producer_qparam, sibling_target_qparam
        )
    ):
        producer_qparam = sibling_target_qparam
        logger.info(
            "[fast_preprocess] aligned binary operand qparams for %s arg%d to sibling quantized result: %s",
            getattr(node, "name", "?"),
            idx,
            producer_qparam,
        )

    if producer_qparam is not None:
        return producer_qparam
    if not hasattr(sibling, "op"):
        return None
    return _extract_single_qparam_entry(
        sibling.meta.get("input_qparams")
    ) or (
        _extract_effective_node_qparam(sibling.args[0])
        if getattr(sibling, "args", None) and hasattr(sibling.args[0], "op")
        else None
    )


def _recover_mixed_control_flow_binary_operand_qparam(
    binary_node,
    mixed_idx,
    mixed_arg,
    sibling_qparam,
):
    if not hasattr(mixed_arg, "op"):
        return None
    if _is_tosa_rescale_node(mixed_arg) and getattr(mixed_arg, "args", None):
        source_arg = mixed_arg.args[0]
        if hasattr(source_arg, "op"):
            recovered_qparam = _extract_effective_node_qparam(source_arg)
            if recovered_qparam is not None:
                return recovered_qparam
            reconstructed_qparam = _reconstruct_control_flow_rescale_source_qparam(
                binary_node,
                mixed_idx,
                mixed_arg,
            )
            if reconstructed_qparam is not None:
                return reconstructed_qparam
    recovered_qparam, _ = _resolve_control_flow_binary_operand_qparam(
        mixed_arg,
        rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
        allow_branch_local_rescale=True,
    )
    if getattr(recovered_qparam, "dtype", None) == getattr(sibling_qparam, "dtype", None):
        return recovered_qparam
    if not _is_control_flow_binary_or_mul_tensor_op(mixed_arg):
        return None
    for _, source_qparam in _collect_control_flow_binary_operand_specs(
        mixed_arg,
        rescale_output_qparam_fn=_repair_control_flow_rescale_output_qparam,
        allow_branch_local_rescale=True,
    ).values():
        if getattr(source_qparam, "dtype", None) == getattr(sibling_qparam, "dtype", None):
            return source_qparam
    return None


def _clone_quantized_binary_output_rescale(
    submodule,
    ref_node,
    quant_rescale,
    source_node,
    *,
    rescale_output_qparam_fn,
):
    import fx_helpers

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
    reconstructed_qparam = rescale_output_qparam_fn(cloned)
    if reconstructed_qparam is not None:
        cloned.meta["output_qparams"] = {0: reconstructed_qparam}
    else:
        cloned.meta["output_qparams"] = dict(quant_rescale.meta.get("output_qparams") or {})
    return cloned


def _effective_submodule_output_qparam(output_arg, *, allow_placeholder_fallback: bool) -> object:
    current = output_arg
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        qparam = _extract_effective_node_qparam(current)
        if qparam is not None:
            return qparam
        if allow_placeholder_fallback and getattr(current, "op", None) == "placeholder":
            return _branch_local_control_flow_output_qparam(current)
        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return None


def _refresh_submodule_output_input_qparams(
    submodule, *, allow_placeholder_fallback: bool
) -> None:
    output_node = submodule.graph.output_node()
    output_args = _output_node_args(output_node)
    input_qparams_map = {}
    for idx, output_arg in enumerate(output_args):
        if not hasattr(output_arg, "op"):
            continue
        qparam = _effective_submodule_output_qparam(
            output_arg,
            allow_placeholder_fallback=allow_placeholder_fallback,
        )
        if qparam is not None:
            input_qparams_map[idx] = qparam
    if input_qparams_map:
        output_node.meta["input_qparams"] = input_qparams_map


def _placeholder_nodes(submodule):
    return [node for node in submodule.graph.nodes if node.op == "placeholder"]


def _output_node_args(output_node):
    outputs = output_node.args[0]
    return [outputs] if hasattr(outputs, "op") else list(outputs)


def _seed_control_flow_input_qparams_map(node, input_index: int, rescale_pass):
    input_qparams_map = dict(
        getattr(node, "meta", {}).get("arm_control_flow_input_qparams_map", {}) or {}
    )
    if not input_qparams_map and len(node.args) > input_index:
        inputs = node.args[input_index]
        if isinstance(inputs, (list, tuple)):
            for idx, arg in enumerate(inputs):
                if not hasattr(arg, "op"):
                    continue
                qparam = _extract_effective_node_qparam(arg)
                if qparam is not None:
                    input_qparams_map[idx] = qparam
    if not input_qparams_map:
        input_qparams_map = rescale_pass._get_input_qparams_map(node, input_index)
    return input_qparams_map


def _quantized_rescale_chain_depth(node) -> int:
    import torch

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


def _filter_output_qparams_map(submodule, output_qparams_map):
    if not output_qparams_map:
        return {}
    output_node = submodule.graph.output_node()
    output_args = _output_node_args(output_node)
    filtered = {}
    for idx, target_qparam in output_qparams_map.items():
        if idx >= len(output_args):
            continue
        output_arg = output_args[idx]
        current_qparam = _extract_effective_node_qparam(output_arg)
        if (
            _quantized_rescale_chain_depth(output_arg) >= 2
            and current_qparam is not None
            and _same_control_flow_qparam(current_qparam, target_qparam)
        ):
            continue
        filtered[idx] = target_qparam
    return filtered


def _reapply_control_flow_input_boundary_rescales(
    contracts,
    *,
    rescale_pass,
):
    if not contracts:
        return False

    submodule = contracts[0].submodule
    input_qparams_map = {contract.index: contract.input_qparam for contract in contracts}
    for contract in contracts:
        placeholder = contract.placeholder
        _log_control_flow_debug(
            "[fast_preprocess] restore submodule input qparams %s %s idx=%d incoming=%s branch_output_qparams=%s",
            getattr(contract.submodule, "__class__", type(contract.submodule)).__name__,
            getattr(placeholder, "name", "?"),
            contract.index,
            contract.input_qparam,
            contract.output_qparam,
        )
        _stamp_single_qparams(
            placeholder,
            contract.input_qparam,
            contract.output_qparam,
        )
        placeholder.meta["arm_control_flow_saved_output_qparams"] = {
            0: contract.output_qparam
        }
        fake_val = placeholder.meta.get("val")
        preserves_float_boundary = any(
            _is_quantize_node(user) for user in list(getattr(placeholder, "users", ()))
        )
        if (
            not preserves_float_boundary
            and hasattr(fake_val, "to")
            and getattr(fake_val, "dtype", None) != contract.output_qparam.dtype
        ):
            placeholder.meta["val"] = fake_val.to(dtype=contract.output_qparam.dtype)

    modified = rescale_pass._rescale_submodule_inputs(submodule, input_qparams_map)
    for contract in contracts:
        for user in list(getattr(contract.placeholder, "users", ())):
            if not _is_tosa_rescale_node(user):
                continue
            if (
                len(getattr(user, "args", ())) >= 2
                and user.args[1] != contract.output_qparam.dtype
            ):
                continue
            _stamp_single_qparams(
                user,
                contract.input_qparam,
                contract.output_qparam,
            )
            logger.info(
                "[fast_preprocess] annotated control-flow input rescale %s idx=%d input=%s output=%s",
                getattr(user, "name", "?"),
                contract.index,
                contract.input_qparam,
                contract.output_qparam,
            )
    return modified


def _reapply_control_flow_output_boundary_rescales(
    contracts,
    *,
    rescale_pass,
):
    if not contracts:
        return False

    return rescale_pass._rescale_submodule_outputs(
        contracts[0].submodule,
        {contract.index: contract.target_qparam for contract in contracts},
    )
