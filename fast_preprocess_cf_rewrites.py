"""Control-flow normalization and q/dq rewrite helpers for fast_preprocess_pipeline."""

import logging
from dataclasses import dataclass

from fast_preprocess_cf_ctx import (
    _for_each_cf_submodule,
    _get_matching_fake_tensor,
)
from fast_preprocess_contracts import _ControlFlowOutputContract
from fast_preprocess_control_flow import (
    _ControlFlowRescaleEdit,
    _restore_input_boundary_contracts,
    _build_control_flow_input_boundary_contracts,
    _build_control_flow_output_boundary_contracts,
    _branch_local_control_flow_output_qparam,
    _cleanup_graph,
    _control_flow_debug_enabled,
    _copy_output_contract_meta,
    _describe_node_brief,
    _extract_effective_node_qparam,
    _extract_single_qparam_entry,
    _first_node_arg,
    _get_control_flow_output_fake_tensor,
    _is_call_function_target,
    _is_control_flow_binary_or_mul_tensor_op,
    _is_control_flow_binary_tensor_op,
    _is_dequantize_node,
    _is_quantize_node,
    _materialize_quantized_constant_node,
    _rescale_signature,
    _is_tosa_rescale_node,
    _log_control_flow_debug,
    _output_node_args,
    _ordered_control_flow_users,
    _preserve_control_flow_output_meta,
    _reapply_control_flow_submodule_boundary_contracts,
    _same_control_flow_qparam,
    _stamp_single_qparams,
)

logger = logging.getLogger(__name__)

_COMPARE_USER_MARKERS = (
    "aten.gt.",
    "aten.ge.",
    "aten.lt.",
    "aten.le.",
    "aten.eq.",
    "tosa.GREATER",
    "tosa.GREATER_EQUAL",
    "tosa.EQUAL",
    "TosaGreater",
    "TosaGreaterEqual",
    "TosaEqual",
)
_CONTROL_FLOW_RANK4_DIM_ORDER = (0, 2, 3, 1)


@dataclass(frozen=True)
class _ControlFlowOutputEdit:
    output_value: object
    contract_node: object
    desired_fake_tensor: object
    desired_dim_order: object
    structural: bool = True
    rewire_source: object = None

    @classmethod
    def from_contract(cls, contract):
        submodule_output = contract.submodule_output
        submodule_user = contract.submodule_user
        desired_fake_tensor = contract.desired_fake_tensor
        desired_dim_order = contract.desired_dim_order
        direct_source = _first_node_arg(submodule_output)
        layout_wrapper = None
        source_input = None
        if _is_layout_output_node(submodule_output):
            source_input = direct_source
        elif (
            _is_dequantize_node(submodule_output) or _is_tosa_rescale_node(submodule_output)
        ) and hasattr(direct_source, "op") and _is_layout_output_node(direct_source):
            source_input = _first_node_arg(direct_source)
            layout_wrapper = direct_source
        if source_input is None:
            if _is_trivial_control_flow_output_chain(submodule_output):
                replacement = _select_control_flow_output_value(
                    submodule_output, submodule_user
                )
                if replacement is not None:
                    _log_control_flow_debug(
                        "[fast_preprocess] selected control-flow replacement %s",
                        _describe_node_brief(replacement),
                    )
                    return cls(
                        output_value=replacement,
                        contract_node=replacement,
                        desired_fake_tensor=desired_fake_tensor,
                        desired_dim_order=desired_dim_order,
                    )
            current_fake_tensor = submodule_output.meta.get("val")
            current_dim_order = submodule_output.meta.get("tosa_dim_order")
            dim_order_changed = desired_dim_order is not None and tuple(
                desired_dim_order
            ) != tuple(current_dim_order or ())
            tensor_changed = desired_fake_tensor is not None and (
                getattr(current_fake_tensor, "shape", None)
                != getattr(desired_fake_tensor, "shape", None)
                or getattr(current_fake_tensor, "dtype", None)
                != getattr(desired_fake_tensor, "dtype", None)
                or dim_order_changed
            )
            if dim_order_changed or tensor_changed:
                return cls(
                    output_value=submodule_output,
                    contract_node=submodule_output,
                    desired_fake_tensor=desired_fake_tensor,
                    desired_dim_order=desired_dim_order,
                    structural=False,
                )
            return None
        if layout_wrapper is None:
            logger.info(
                "[fast_preprocess] retargeting control-flow output layout %s -> %s",
                getattr(submodule_output, "name", "?"),
                getattr(source_input, "name", "?"),
            )
            return cls(
                output_value=source_input,
                contract_node=source_input,
                desired_fake_tensor=desired_fake_tensor,
                desired_dim_order=desired_dim_order,
            )

        logger.info(
            "[fast_preprocess] retargeting control-flow output %s through %s -> %s",
            getattr(submodule_output, "name", "?"),
            getattr(layout_wrapper, "name", "?"),
            getattr(source_input, "name", "?"),
        )
        return cls(
            output_value=submodule_output,
            contract_node=submodule_output,
            desired_fake_tensor=desired_fake_tensor,
            desired_dim_order=desired_dim_order,
            rewire_source=source_input,
        )

    def apply(self, output_args, idx) -> None:
        if self.rewire_source is not None:
            self.contract_node.update_arg(0, self.rewire_source)
        self.contract_node.meta["val"] = self.desired_fake_tensor
        if self.desired_dim_order is not None:
            self.contract_node.meta["tosa_dim_order"] = tuple(self.desired_dim_order)
        output_args[idx] = self.output_value


def _parse_per_tensor_qparam(node):
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

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


def _is_zero_input_const_source(node) -> bool:
    return hasattr(node, "op") and (
        node.op == "get_attr"
        or (node.op == "call_function" and not list(getattr(node, "all_input_nodes", ())))
    )


def _feeds_scalar_binary(node) -> bool:
    stack = list(getattr(node, "users", ()))
    seen = set()
    while stack:
        user = stack.pop()
        if id(user) in seen:
            continue
        seen.add(id(user))
        if _is_control_flow_binary_tensor_op(user):
            return True
        if _is_tosa_rescale_node(user):
            stack.extend(list(getattr(user, "users", ())))
    return False


def _binary_input_qparam(user, value_node):
    input_qparams = getattr(user, "meta", {}).get("input_qparams") or {}
    for idx, arg in enumerate(getattr(user, "args", ())):
        if arg is value_node and idx in input_qparams:
            return input_qparams[idx]
    return _extract_single_qparam_entry(input_qparams)


def _is_compare_user(user) -> bool:
    return _is_call_function_target(user, *_COMPARE_USER_MARKERS)


def _matching_int32_user_cluster(users, value_node, *, user_predicate):
    import torch

    desired_qparam = None
    matched_users = []
    for user in users:
        if not user_predicate(user):
            return None
        user_qparam = _binary_input_qparam(user, value_node)
        if getattr(user_qparam, "dtype", None) != torch.int32:
            return None
        if desired_qparam is None:
            desired_qparam = user_qparam
        elif not _same_control_flow_qparam(desired_qparam, user_qparam):
            return None
        matched_users.append(user)

    if not matched_users or desired_qparam is None:
        return None
    return matched_users, desired_qparam


def _rewrite_dequantize_int32_lift(
    ctx,
    node,
    source_node,
    source_qparam,
    cluster,
    stats,
    *,
    log_message: str,
) -> bool:
    if cluster is None:
        return False
    users, desired_qparam = cluster
    lifted, _ = _ControlFlowRescaleEdit.from_source(
        submodule=ctx.submodule,
        anchor=users[0],
        source_node=source_node,
        input_qparam=source_qparam,
        output_qparam=desired_qparam,
    ).ensure()
    node.replace_all_uses_with(lifted)
    ctx.graph.erase_node(node)
    stats["dequant"] += 1
    logger.info(log_message, getattr(node, "name", "?"), getattr(lifted, "name", "?"))
    return True


def _resolve_quantize_output_qparam(node, QuantArgs):
    try:
        return QuantArgs.from_operator(node.target, node.args)
    except Exception:
        return _parse_per_tensor_qparam(node)


def _try_rewrite_dequantize_to_int32_cluster(
    ctx,
    node,
    source_node,
    source_qparam,
    users,
    stats,
) -> bool:
    cluster_specs = (
        (
            _is_control_flow_binary_or_mul_tensor_op,
            "[fast_preprocess] rewrote control-flow dequantize %s into int32 lift %s",
        ),
        (
            _is_compare_user,
            "[fast_preprocess] rewrote control-flow compare dequantize %s into int32 lift %s",
        ),
    )
    for user_predicate, log_message in cluster_specs:
        cluster = _matching_int32_user_cluster(
            users,
            node,
            user_predicate=user_predicate,
        )
        if _rewrite_dequantize_int32_lift(
            ctx,
            node,
            source_node,
            source_qparam,
            cluster,
            stats,
            log_message=log_message,
        ):
            return True
    return False


def _collapse_control_flow_qdq_pairs(ctx, stats) -> bool:
    modified = False
    for node in list(ctx.graph.nodes):
        if not _is_dequantize_node(node):
            continue
        if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
            continue
        quant_node = node.args[0]
        if not _is_quantize_node(quant_node):
            continue
        quant_qparam = _parse_per_tensor_qparam(quant_node)
        dequant_qparam = _parse_per_tensor_qparam(node)
        _log_control_flow_debug(
            "[cf-qdq-collapse] candidate %s quant=%s quant_qparam=%s dequant_qparam=%s",
            getattr(node, "name", "?"),
            getattr(quant_node, "name", "?"),
            quant_qparam,
            dequant_qparam,
        )
        if not _same_control_flow_qparam(
            quant_qparam, dequant_qparam, include_range=True
        ):
            continue
        float_source = quant_node.args[0] if getattr(quant_node, "args", None) else None
        if not hasattr(float_source, "op"):
            continue
        if _is_zero_input_const_source(float_source) and _feeds_scalar_binary(node):
            continue
        _log_control_flow_debug(
            "[cf-qdq-collapse] bypass %s -> %s",
            getattr(node, "name", "?"),
            getattr(float_source, "name", "?"),
        )
        node.replace_all_uses_with(float_source)
        ctx.graph.erase_node(node)
        if len(getattr(quant_node, "users", ())) == 0:
            ctx.graph.erase_node(quant_node)
        stats["quant"] += 1
        modified = True
    return modified


def _rewrite_control_flow_quantize_nodes(ctx, stats, QuantArgs) -> bool:
    import torch

    modified = False
    for node in list(ctx.graph.nodes):
        if not _is_quantize_node(node):
            continue
        if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
            continue

        source_node = node.args[0]
        source_fake = source_node.meta.get("val")
        source_dtype = getattr(source_fake, "dtype", None)
        source_qparam = _extract_effective_node_qparam(source_node)
        if source_qparam is None:
            source_qparam = _extract_single_qparam_entry(node.meta.get("input_qparams"))

        _log_control_flow_debug(
            "[cf-quantize-rewrite] candidate %s source=%s source_dtype=%s input_qparams=%s output_qparams=%s",
            getattr(node, "name", "?"),
            getattr(source_node, "name", "?"),
            source_dtype,
            node.meta.get("input_qparams"),
            node.meta.get("output_qparams"),
        )

        output_qparam = _resolve_quantize_output_qparam(node, QuantArgs)
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
                const_node = _materialize_quantized_constant_node(
                    ctx.submodule,
                    ctx.graph,
                    node,
                    source_node,
                    output_qparam,
                    const_name_prefix=(
                        f"{getattr(source_node, 'name', 'const')}_quantized_const"
                    ),
                    include_get_attr=False,
                )
                if const_node is not None:
                    node.replace_all_uses_with(const_node)
                    ctx.graph.erase_node(node)
                    if len(getattr(source_node, "users", ())) == 0:
                        ctx.graph.erase_node(source_node)
                    stats["quant"] += 1
                    modified = True
                    _log_control_flow_debug(
                        "[cf-quantize-rewrite] materialized const %s -> %s",
                        getattr(node, "name", "?"),
                        getattr(const_node, "name", "?"),
                    )
                    continue
            _log_control_flow_debug(
                "[cf-quantize-rewrite] skip %s source_qparam=%s output_qparam=%s",
                getattr(node, "name", "?"),
                source_qparam,
                output_qparam,
            )
            continue

        if _same_control_flow_qparam(
            source_qparam, output_qparam, include_range=True
        ):
            _log_control_flow_debug(
                "[cf-quantize-rewrite] bypass %s",
                getattr(node, "name", "?"),
            )
            node.replace_all_uses_with(source_node)
            ctx.graph.erase_node(node)
            stats["quant"] += 1
            modified = True
            continue

        rescale, _ = _ControlFlowRescaleEdit.from_source(
            submodule=ctx.submodule,
            anchor=node,
            source_node=source_node,
            input_qparam=source_qparam,
            output_qparam=output_qparam,
        ).ensure()
        logger.info(
            "[fast_preprocess] quantize rewrite rescale %s via %s from %s to %s",
            getattr(node, "name", "?"),
            getattr(rescale, "name", "?"),
            source_qparam,
            output_qparam,
        )
        _log_control_flow_debug(
            "[cf-quantize-rewrite] rescale %s from %s to %s",
            getattr(node, "name", "?"),
            source_qparam,
            output_qparam,
        )
        node.replace_all_uses_with(rescale)
        ctx.graph.erase_node(node)
        stats["quant"] += 1
        modified = True
    return modified


def _rewrite_control_flow_dequantize_nodes(ctx, stats) -> bool:
    import torch

    modified = False
    for node in list(ctx.graph.nodes):
        if not _is_dequantize_node(node):
            continue
        if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
            continue

        source_node = node.args[0]
        users = list(getattr(node, "users", ()))
        if not users:
            continue

        source_qparam = _extract_effective_node_qparam(source_node)
        if getattr(source_qparam, "dtype", None) in (torch.int8, torch.int16) and (
            _try_rewrite_dequantize_to_int32_cluster(
                ctx,
                node,
                source_node,
                source_qparam,
                users,
                stats,
            )
        ):
            modified = True
            continue

        if not all(
            _is_tosa_rescale_node(user) or getattr(user, "op", None) == "output"
            for user in users
        ):
            continue

        _log_control_flow_debug(
            "[cf-dequantize-rewrite] bypass %s -> %s",
            getattr(node, "name", "?"),
            getattr(source_node, "name", "?"),
        )

        node.replace_all_uses_with(source_node)
        ctx.graph.erase_node(node)
        stats["dequant"] += 1
        modified = True
    return modified

def _remove_control_flow_noop_passthroughs(graph_module) -> None:
    rewritten = 0

    def _visit(ctx):
        nonlocal rewritten
        updated = False
        for node in list(ctx.graph.nodes):
            if not _is_call_function_target(
                node,
                "dim_order_ops._clone_dim_order",
                "aten.clone",
                "aten.alias_copy",
                "aten.detach_copy",
            ):
                continue
            if not getattr(node, "args", None) or not hasattr(node.args[0], "op"):
                continue

            source = node.args[0]
            node_val = node.meta.get("val")
            source_val = source.meta.get("val")
            node_dtype = getattr(node_val, "dtype", None)
            source_dtype = getattr(source_val, "dtype", None)
            if (
                node_dtype is not None
                and source_dtype is not None
                and node_dtype != source_dtype
            ):
                continue

            node_shape = getattr(node_val, "shape", None)
            source_shape = getattr(source_val, "shape", None)
            if (
                node_shape is not None
                and source_shape is not None
                and tuple(node_shape) != tuple(source_shape)
            ):
                continue

            node_dim_order = node.meta.get("tosa_dim_order")
            source_dim_order = source.meta.get("tosa_dim_order")
            if (
                node_dim_order is not None
                and source_dim_order is not None
                and tuple(node_dim_order) != tuple(source_dim_order)
            ):
                continue

            if source.meta.get("tosa_dim_order") is None and node_dim_order is not None:
                source.meta["tosa_dim_order"] = tuple(node_dim_order)
            if source.meta.get("input_qparams") is None and node.meta.get("input_qparams"):
                source.meta["input_qparams"] = dict(node.meta["input_qparams"])
            if source.meta.get("output_qparams") is None and node.meta.get("output_qparams"):
                source.meta["output_qparams"] = dict(node.meta["output_qparams"])

            node.replace_all_uses_with(source)
            ctx.graph.erase_node(node)
            rewritten += 1
            updated = True

        if updated:
            _cleanup_graph(ctx.graph)
        return updated

    _for_each_cf_submodule(graph_module, _visit)

    if rewritten:
        logger.info(
            "[fast_preprocess] removed %d control-flow no-op passthrough node(s)",
            rewritten,
        )


def _restore_control_flow_float_compare_inputs(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.arm_pass_utils import (
        create_node,
    )
    from executorch.exir.dialects._ops import ops as exir_ops

    def _is_float_dtype(dtype) -> bool:
        return dtype in (torch.float16, torch.float32, torch.float64)

    restored = 0

    def _visit(ctx):
        nonlocal restored
        updated = False
        for node in list(ctx.graph.nodes):
            if not _is_call_function_target(
                node, "aten.gt.", "aten.ge.", "aten.lt.", "aten.le.", "aten.eq."
            ):
                continue
            args = list(getattr(node, "args", ()))
            if len(args) != 2:
                continue

            for idx, arg in enumerate(args):
                if not hasattr(arg, "op"):
                    continue
                other = args[1 - idx]
                if not hasattr(other, "op"):
                    continue

                arg_val = arg.meta.get("val")
                other_val = other.meta.get("val")
                arg_dtype = getattr(arg_val, "dtype", None)
                other_dtype = getattr(other_val, "dtype", None)
                if not (
                    arg_dtype in (torch.int8, torch.int16, torch.int32)
                    and _is_float_dtype(other_dtype)
                ):
                    continue

                qparam = _extract_effective_node_qparam(arg)
                if qparam is None or qparam.per_channel:
                    continue

                with ctx.graph.inserting_before(node):
                    dq = create_node(
                        graph=ctx.graph,
                        op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                        args=(
                            arg,
                            float(qparam.scale),
                            int(qparam.zp),
                            int(qparam.qmin),
                            int(qparam.qmax),
                            qparam.dtype,
                        ),
                        from_node=node,
                    )
                if hasattr(arg_val, "to"):
                    dq.meta["val"] = arg_val.to(dtype=other_dtype)
                dq.meta["input_qparams"] = {0: qparam}
                dq.meta["output_qparams"] = {}
                node.update_arg(idx, dq)
                restored += 1
                updated = True
                break

        if updated:
            _cleanup_graph(ctx.graph)
        return updated

    _for_each_cf_submodule(graph_module, _visit)

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow float-compare dequantize input(s)",
            restored,
        )


def _is_layout_output_node(node) -> bool:
    return _is_call_function_target(
        node,
        "_to_dim_order_copy",
        "tosa.TRANSPOSE",
        "tosa_TRANSPOSE",
        "TosaTranspose",
    )

def _preferred_quantized_rescale_user(source_node):
    import torch

    if not hasattr(source_node, "op"):
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

        output_qparam = _extract_single_qparam_entry(user.meta.get("output_qparams"))
        if output_qparam is None:
            output_qparam = _extract_effective_node_qparam(user)
        if getattr(output_qparam, "dtype", None) != user.args[1]:
            continue

        score = 0
        for grand_user in list(getattr(user, "users", ())):
            if _is_tosa_rescale_node(grand_user):
                score += 2
            elif _is_dequantize_node(grand_user):
                score += 1
            elif getattr(grand_user, "op", None) == "output":
                score += 4
        if score > best_score:
            best = (user, output_qparam)
            best_score = score

    return best


def _debug_binary_rescales(graph_module, stage: str) -> None:
    import os

    if os.environ.get("ARM_TOSA_ENGINE_DEBUG_BINARY_RESCALES") != "1":
        return

    logger.info("[fast_preprocess] binary rescale trace stage=%s", stage)
    for node in list(graph_module.graph.nodes):
        if not _is_control_flow_binary_tensor_op(node):
            continue

        node_meta = getattr(node, "meta", {})
        logger.info(
            "[fast_preprocess] binary node=%s target=%s input_qparams=%s output_qparams=%s val_dtype=%s",
            node.name,
            getattr(node, "target", None),
            node_meta.get("input_qparams"),
            node_meta.get("output_qparams"),
            getattr(node_meta.get("val"), "dtype", None),
        )
        for idx, arg in enumerate(node.args[:2]):
            if not hasattr(arg, "name"):
                logger.info(
                    "[fast_preprocess]   arg[%d] literal=%r type=%s",
                    idx,
                    arg,
                    type(arg).__name__,
                )
                continue

            arg_meta = getattr(arg, "meta", {})
            logger.info(
                "[fast_preprocess]   arg[%d]=%s target=%s val_dtype=%s rescale=%s qparams=%s",
                idx,
                arg.name,
                getattr(arg, "target", None),
                getattr(arg_meta.get("val"), "dtype", None),
                _is_tosa_rescale_node(arg),
                _extract_effective_node_qparam(arg),
            )
            if not _is_tosa_rescale_node(arg):
                continue

            source_node = arg.args[0] if len(getattr(arg, "args", ())) > 0 else None
            logger.info(
                "[fast_preprocess]     rescale dtype_arg=%s scales=%s input_zp=%s output_zp=%s",
                arg.args[1] if len(arg.args) > 1 else None,
                arg.args[2] if len(arg.args) > 2 else None,
                arg.args[3] if len(arg.args) > 3 else None,
                arg.args[4] if len(arg.args) > 4 else None,
            )
            if not hasattr(source_node, "name"):
                logger.info(
                    "[fast_preprocess]     source literal=%r type=%s",
                    source_node,
                    type(source_node).__name__,
                )
                continue

            preferred = _preferred_quantized_rescale_user(source_node)
            source_meta = getattr(source_node, "meta", {})
            logger.info(
                "[fast_preprocess]     source=%s target=%s val_dtype=%s input_qparams=%s output_qparams=%s qparams=%s preferred=%s",
                source_node.name,
                getattr(source_node, "target", None),
                getattr(source_meta.get("val"), "dtype", None),
                source_meta.get("input_qparams"),
                source_meta.get("output_qparams"),
                _extract_effective_node_qparam(source_node),
                (
                    preferred[0].name,
                    preferred[1],
                )
                if preferred is not None
                else None,
            )
            if hasattr(source_node, "users"):
                for user in list(source_node.users):
                    user_meta = getattr(user, "meta", {})
                    logger.info(
                        "[fast_preprocess]       source user=%s target=%s val_dtype=%s rescale=%s qparams=%s",
                        user.name,
                        getattr(user, "target", None),
                        getattr(user_meta.get("val"), "dtype", None),
                        _is_tosa_rescale_node(user),
                        _extract_effective_node_qparam(user),
                    )

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


def _trace_control_flow_output_source(node):
    current = node
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "op", None) == "placeholder":
            return current
        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return None


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


def _infer_control_flow_output_dim_order(node):
    current = node
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        dim_order = current.meta.get("tosa_dim_order")
        if dim_order is not None:
            return tuple(dim_order)
        current = next(
            (arg for arg in getattr(current, "args", ()) if hasattr(arg, "op")),
            None,
        )

    fake_val = node.meta.get("val")
    if hasattr(fake_val, "dim") and fake_val.dim() == 4:
        return _CONTROL_FLOW_RANK4_DIM_ORDER
    return None


def _annotate_parent_control_flow_output_users(ctx, stats) -> None:
    if ctx.cf_node is None:
        return
    for user in _ordered_control_flow_users(ctx.cf_node):
        fake_val = user.meta.get("val")
        if not hasattr(fake_val, "dim") or fake_val.dim() != 4:
            continue
        user.meta["arm_control_flow_parent_output_user"] = True
        if tuple(user.meta.get("tosa_dim_order") or ()) == _CONTROL_FLOW_RANK4_DIM_ORDER:
            continue
        user.meta["tosa_dim_order"] = _CONTROL_FLOW_RANK4_DIM_ORDER
        _log_control_flow_debug(
            "[fast_preprocess] parent control-flow user %s target=%s shape=%s dim_order=%s",
            getattr(user, "name", "?"),
            getattr(user, "target", "?"),
            tuple(fake_val.shape),
            user.meta["tosa_dim_order"],
        )
        stats["annotated_parent_users"] += 1


def _restamp_control_flow_placeholder_dtypes(ctx, stats) -> bool:
    updated = False
    for node in ctx.placeholders:
        if any(_is_quantize_node(user) for user in list(getattr(node, "users", ()))):
            continue
        qparam = _extract_single_qparam_entry(node.meta.get("input_qparams"))
        if qparam is None:
            qparam = _branch_local_control_flow_output_qparam(node)
        fake_val = node.meta.get("val")
        if (
            qparam is None
            or not hasattr(fake_val, "to")
            or getattr(fake_val, "dtype", None) == getattr(qparam, "dtype", None)
        ):
            continue
        node.meta["val"] = fake_val.to(dtype=qparam.dtype)
        stats["restamped_placeholders"] += 1
        updated = True
    return updated


def _annotate_control_flow_rank4_nodes(ctx, stats) -> bool:
    updated = False
    for node in ctx.graph.nodes:
        if node.op != "call_function":
            continue
        if node.meta.get("tosa_dim_order") is not None:
            continue
        fake_val = node.meta.get("val")
        if not hasattr(fake_val, "dim") or fake_val.dim() != 4:
            continue
        node.meta["tosa_dim_order"] = _CONTROL_FLOW_RANK4_DIM_ORDER
        stats["annotated_rank4_nodes"] += 1
        updated = True
    return updated


def _normalize_control_flow_outputs_for_ctx(ctx, stats) -> bool:
    if ctx.output_node is None or ctx.cf_node is None:
        return False

    wrapped_single = hasattr(ctx.output_node.args[0], "op")
    output_args = _output_node_args(ctx.output_node)
    ordered_users = _ordered_control_flow_users(ctx.cf_node)
    _log_control_flow_debug(
        "[fast_preprocess] normalize outputs node=%s output_count=%d user_count=%d",
        getattr(ctx.cf_node, "name", "?"),
        len(output_args),
        len(ordered_users),
    )

    updated = False
    for idx, submodule_output in enumerate(output_args):
        if idx < len(ordered_users):
            submodule_user = ordered_users[idx]
            desired_fake_tensor = _get_control_flow_output_fake_tensor(submodule_user)
            desired_dim_order = submodule_user.meta.get("tosa_dim_order")
        else:
            fallback_source = _trace_control_flow_output_source(submodule_output)
            submodule_user = fallback_source or ctx.cf_node
            desired_fake_tensor = (
                _get_matching_fake_tensor(fallback_source)
                if fallback_source is not None
                else _get_matching_fake_tensor(ctx.cf_node)
            )
            desired_dim_order = (
                fallback_source.meta.get("tosa_dim_order")
                if fallback_source is not None
                else ctx.cf_node.meta.get("tosa_dim_order")
            )
        if desired_dim_order is None:
            desired_dim_order = _infer_control_flow_output_dim_order(submodule_output)
        contract = _ControlFlowOutputContract(
            submodule_output=submodule_output,
            submodule_user=submodule_user,
            desired_fake_tensor=desired_fake_tensor,
            desired_dim_order=desired_dim_order,
        )
        desired_fake_tensor = contract.desired_fake_tensor
        desired_dim_order = contract.desired_dim_order

        current_fake_tensor = submodule_output.meta.get("val")
        if _preserve_control_flow_output_meta(
            current_fake_tensor, desired_fake_tensor
        ):
            logger.info(
                "[fast_preprocess] preserving boolean control-flow output %s shape=%s dtype=%s over desired shape=%s dtype=%s",
                getattr(submodule_output, "name", "?"),
                tuple(current_fake_tensor.shape)
                if hasattr(current_fake_tensor, "shape")
                else None,
                getattr(current_fake_tensor, "dtype", None),
                tuple(desired_fake_tensor.shape)
                if hasattr(desired_fake_tensor, "shape")
                else None,
                getattr(desired_fake_tensor, "dtype", None),
            )
            continue

        edit = _ControlFlowOutputEdit.from_contract(contract)
        if edit is not None:
            edit.apply(output_args, idx)
            updated = True
            if edit.structural:
                stats["rewritten_outputs"] += 1
            else:
                logger.info(
                    "[fast_preprocess] restamped control-flow output %s -> shape=%s dtype=%s dim_order=%s",
                    getattr(submodule_output, "name", "?"),
                    tuple(desired_fake_tensor.shape),
                    getattr(desired_fake_tensor, "dtype", None),
                    desired_dim_order,
                )

    if not updated:
        return False

    replacement = output_args[0] if wrapped_single else tuple(output_args)
    ctx.output_node.update_arg(0, replacement)
    ctx.graph.eliminate_dead_code()
    return True


def _normalize_control_flow_output_contracts(
    graph_module,
    *,
    annotate_parent_users: bool = False,
    normalize_outputs: bool = False,
    annotate_rank4_nodes: bool = False,
    restamp_placeholder_dtypes: bool = False,
) -> None:
    stats = {
        "rewritten_outputs": 0,
        "restamped_placeholders": 0,
        "annotated_rank4_nodes": 0,
        "annotated_parent_users": 0,
    }

    def _visit(ctx):
        if annotate_parent_users:
            _annotate_parent_control_flow_output_users(ctx, stats)
        updated = False
        if restamp_placeholder_dtypes:
            updated |= _restamp_control_flow_placeholder_dtypes(ctx, stats)
        if annotate_rank4_nodes:
            updated |= _annotate_control_flow_rank4_nodes(ctx, stats)
        if normalize_outputs:
            updated |= _normalize_control_flow_outputs_for_ctx(ctx, stats)
        if updated:
            ctx.graph.lint()
        return updated

    _for_each_cf_submodule(graph_module, _visit)

    if stats["rewritten_outputs"]:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow output contract node(s)",
            stats["rewritten_outputs"],
        )
    if stats["restamped_placeholders"]:
        logger.info(
            "[fast_preprocess] restamped %d control-flow placeholder dtype(s)",
            stats["restamped_placeholders"],
        )
    if stats["annotated_rank4_nodes"]:
        logger.info(
            "[fast_preprocess] annotated NHWC dim-order on %d control-flow interior rank-4 node(s)",
            stats["annotated_rank4_nodes"],
        )
    if stats["annotated_parent_users"]:
        logger.info(
            "[fast_preprocess] annotated NHWC dim-order on %d parent control-flow output user(s)",
            stats["annotated_parent_users"],
        )


def _restore_control_flow_output_dequantizes(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.insert_rescales_pass import (
        InsertControlFlowRescalesPass,
    )

    rescale_pass = InsertControlFlowRescalesPass()
    restored = 0
    seen_while_submodules = {}

    def _visit(ctx):
        nonlocal restored
        if ctx.cf_node is not None and ctx.is_while:
            while_key = id(ctx.cf_node)
            while_idx = seen_while_submodules.get(while_key, 0)
            seen_while_submodules[while_key] = while_idx + 1
            if while_idx == 1:
                return False
        output_qparams_map = ctx.output_node.meta.get(
            "arm_control_flow_output_dequant_qparams", {}
        )
        if not output_qparams_map:
            return False
        output_args = _output_node_args(ctx.submodule.graph.output_node())
        output_qparams_map = {
            idx: qparam
            for idx, qparam in output_qparams_map.items()
            if not (
                idx < len(output_args)
                and _is_tosa_rescale_node(output_args[idx])
                and len(getattr(output_args[idx], "args", ())) >= 2
                and output_args[idx].args[1] in (torch.int8, torch.int16)
            )
        }
        contracts = _build_control_flow_output_boundary_contracts(
            ctx.submodule,
            output_qparams_map,
            allow_placeholder_fallback=False,
        )
        if not contracts:
            return False
        if _reapply_control_flow_submodule_boundary_contracts(
            ctx.submodule,
            rescale_pass=rescale_pass,
            output_contracts=contracts,
            include_inputs=False,
            include_outputs=True,
        ):
            restored += len(contracts)
            _cleanup_graph(ctx.graph)
            return True
        return False

    _for_each_cf_submodule(graph_module, _visit)

    if restored:
        logger.info(
            "[fast_preprocess] restored %d control-flow branch output rescale(s)",
            restored,
        )


def _restore_missing_control_flow_input_rescales(graph_module) -> None:
    restored = 0

    def _visit(ctx):
        nonlocal restored
        submodule_modified = False
        pending_contracts = []
        for contract in _build_control_flow_input_boundary_contracts(ctx.submodule, current_only=True):
            placeholder = contract.placeholder
            users = [user for user in list(placeholder.users) if hasattr(user, "op")]
            placeholder_dtype = getattr(placeholder.meta.get("val"), "dtype", None)
            same_qparam = _same_control_flow_qparam(
                contract.input_qparam, contract.output_qparam
            )
            needs_noop_quant_boundary = (
                same_qparam
                and placeholder_dtype != getattr(contract.output_qparam, "dtype", None)
                and bool(users)
            )
            _log_control_flow_debug(
                "[cf-entry-rescale-check] %s input_qparam=%s output_qparam=%s users=%s placeholder_dtype=%s needs_noop_quant_boundary=%s",
                getattr(placeholder, "name", "?"),
                contract.input_qparam,
                contract.output_qparam,
                [getattr(user, "name", "?") for user in users],
                placeholder_dtype,
                needs_noop_quant_boundary,
            )
            if same_qparam and not needs_noop_quant_boundary:
                continue
            if not users:
                continue
            pending_contracts.append(contract)

        applied, _ = _restore_input_boundary_contracts(
            pending_contracts,
            user_selector=lambda contract: list(getattr(contract.placeholder, "users", ())),
            preserve_boundary=False,
        )
        if applied:
            restored += len(applied)
            submodule_modified = True

        if submodule_modified:
            _cleanup_graph(ctx.graph)
        return submodule_modified

    _for_each_cf_submodule(graph_module, _visit)

    if restored:
        logger.info(
            "[fast_preprocess] restored %d missing control-flow input boundary rescale(s)",
            restored,
        )


def _rewrite_control_flow_quantization_boundaries(graph_module) -> None:
    from executorch.backends.arm._passes.quant_args import QuantArgs

    stats = {"quant": 0, "dequant": 0}

    def _visit(ctx):
        modified = _collapse_control_flow_qdq_pairs(ctx, stats)
        modified |= _rewrite_control_flow_quantize_nodes(ctx, stats, QuantArgs)
        modified |= _rewrite_control_flow_dequantize_nodes(ctx, stats)

        if modified:
            _cleanup_graph(ctx.graph)
        return modified

    _for_each_cf_submodule(graph_module, _visit)

    if stats["quant"]:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow quantize node(s) into explicit rescale/no-op edges",
            stats["quant"],
        )
    if stats["dequant"]:
        logger.info(
            "[fast_preprocess] rewrote %d control-flow dequantize node(s) into direct quant-domain edges",
            stats["dequant"],
        )


def _restore_missing_control_flow_rescale_qparams(graph_module) -> None:
    import torch

    from executorch.backends.arm._passes.quant_args import QuantArgs

    restored = 0

    def _visit(ctx):
        nonlocal restored
        modified = False
        for node in ctx.graph.nodes:
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
                _stamp_single_qparams(node, input_qparam=source_qparam)
            if not node.meta.get("output_qparams"):
                _stamp_single_qparams(node, output_qparam=output_qparam)
            restored += 1
            modified = True
        return modified

    _for_each_cf_submodule(graph_module, _visit)

    if restored:
        logger.info(
            "[fast_preprocess] restored qparams on %d control-flow rescale node(s)",
            restored,
        )


def _prune_duplicate_terminal_rescales(graph_module) -> None:
    pruned = 0

    def _visit(ctx):
        nonlocal pruned
        modified = False
        for node in list(ctx.graph.nodes):
            if not _is_tosa_rescale_node(node):
                continue
            if len(getattr(node, "users", ())) != 1:
                continue

            user = next(iter(node.users))
            if not _is_tosa_rescale_node(user):
                continue
            if user.args[0] is not node:
                continue
            if _rescale_signature(node) != _rescale_signature(user):
                continue

            first_output_qparam = _extract_single_qparam_entry(
                node.meta.get("output_qparams")
            )
            final_output_qparam = _extract_single_qparam_entry(
                user.meta.get("output_qparams")
            )
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
            _copy_output_contract_meta(node, user, copy_output_qparams=False)
            if node.meta.get("input_qparams") is None:
                source_qparam = _extract_single_qparam_entry(
                    getattr(source, "meta", {}).get("output_qparams")
                )
                if source_qparam is not None:
                    _stamp_single_qparams(node, input_qparam=source_qparam)
            _stamp_single_qparams(node, output_qparam=final_output_qparam)

            user.replace_all_uses_with(node)
            ctx.graph.erase_node(user)
            pruned += 1
            modified = True

        if modified:
            _cleanup_graph(ctx.graph)
        return modified

    _for_each_cf_submodule(graph_module, _visit, include_parent=True)

    if pruned:
        logger.info(
            "[fast_preprocess] pruned %d duplicate terminal rescale node(s)",
            pruned,
        )


def _dump_control_flow_submodules(graph_module, label: str) -> None:
    if not _control_flow_debug_enabled():
        return

    def _visit(ctx):
        name = getattr(ctx.cf_node, "name", "<unknown>")
        try:
            readable = ctx.submodule.print_readable(print_output=False)
        except TypeError:
            readable = ctx.submodule.print_readable()
        node_dump = "\n".join(
            (
                f"{node.name} | {node.op} | {node.target} | "
                f"shape={tuple(meta.get('val').shape) if hasattr(meta.get('val'), 'shape') else None} "
                f"dtype={getattr(meta.get('val'), 'dtype', None)} "
                f"qin={meta.get('input_qparams')} "
                f"qout={meta.get('output_qparams')} "
                f"saved_qout={meta.get('arm_control_flow_saved_output_qparams')}"
            )
            for node in ctx.graph.nodes
            for meta in [getattr(node, "meta", {})]
        )
        logger.info(
            "[fast_preprocess] control-flow submodule dump (%s:%s)\n%s\n[node-meta]\n%s",
            label,
            name,
            readable,
            node_dump,
        )
        return False

    _for_each_cf_submodule(graph_module, _visit)


# ---------------------------------------------------------------------------
# Compatibility wrappers -- original pipeline call ordering (pre-consolidation)
# ---------------------------------------------------------------------------


def _normalize_control_flow_output_layouts(graph_module) -> None:
    _normalize_control_flow_output_contracts(graph_module, normalize_outputs=True)
