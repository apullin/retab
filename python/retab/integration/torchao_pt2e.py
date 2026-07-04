"""Optional TorchAO PT2E fast paths used by RETAB export scripts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch

_install_depth = 0
_original_constant_fold = None
_patched_constant_fold = None


def _get_attr(gm: torch.fx.GraphModule, target: str):
    value = gm
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _delete_attr(gm: torch.fx.GraphModule, target: str) -> bool:
    atoms = target.split(".")
    value = gm
    for atom in atoms[:-1]:
        if not hasattr(value, atom):
            return False
        value = getattr(value, atom)
    if not hasattr(value, atoms[-1]):
        return False
    delattr(value, atoms[-1])
    return True


def _is_get_attr(value: object) -> bool:
    return isinstance(value, torch.fx.Node) and value.op == "get_attr"


def _constraint_allows(
    node: torch.fx.Node, constraint_fn: Callable[[torch.fx.Node], bool] | None
) -> bool:
    return constraint_fn is None or constraint_fn(node)


def _is_supported_per_channel_constant_quant(
    node: torch.fx.Node,
    constraint_fn: Callable[[torch.fx.Node], bool] | None,
) -> bool:
    quantize_per_channel = torch.ops.quantized_decomposed.quantize_per_channel.default
    return (
        node.op == "call_function"
        and node.target is quantize_per_channel
        and _constraint_allows(node, constraint_fn)
        and len(node.args) >= 7
        and _is_get_attr(node.args[0])
        and _is_get_attr(node.args[1])
        and _is_get_attr(node.args[2])
    )


def _is_const_tree(
    node: torch.fx.Node,
    memo: dict[torch.fx.Node, bool],
    visiting: set[torch.fx.Node],
) -> bool:
    if node in memo:
        return memo[node]
    if node in visiting:
        return False
    if node.op == "get_attr":
        memo[node] = True
        return True
    if node.op != "call_function":
        memo[node] = False
        return False

    visiting.add(node)
    is_const = all(
        _is_const_tree(input_node, memo, visiting)
        for input_node in node.all_input_nodes
    )
    visiting.remove(node)
    memo[node] = is_const
    return is_const


def _needs_original_constant_folder(
    gm: torch.fx.GraphModule,
    constraint_fn: Callable[[torch.fx.Node], bool] | None,
) -> bool:
    memo: dict[torch.fx.Node, bool] = {}
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or not _constraint_allows(node, constraint_fn)
            or not str(node.target).startswith("quantized_decomposed.quantize_")
        ):
            continue
        if _is_supported_per_channel_constant_quant(node, constraint_fn):
            continue
        if _is_const_tree(node, memo, set()):
            return True
    return False


def _fast_quant_constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Callable[[torch.fx.Node], bool] | None = None,
) -> int:
    from torchao.quantization.pt2e.constant_fold import (
        replace_node_with_constant,
    )

    folded = 0
    with torch.utils._python_dispatch._disable_current_modes():
        for node in list(gm.graph.nodes):
            if not _is_supported_per_channel_constant_quant(node, constraint_fn):
                continue

            tensor_node, scale_node, zp_node, axis, qmin, qmax, dtype = node.args[:7]
            assert isinstance(tensor_node.target, str)
            assert isinstance(scale_node.target, str)
            assert isinstance(zp_node.target, str)
            constant = node.target(
                _get_attr(gm, tensor_node.target),
                _get_attr(gm, scale_node.target),
                _get_attr(gm, zp_node.target),
                axis,
                qmin,
                qmax,
                dtype,
            )
            replace_node_with_constant(gm, node, constant)
            folded += 1

        erased_params = []
        for node in gm.graph.find_nodes(op="get_attr"):
            if len(node.users) == 0:
                target = node.target
                if isinstance(target, str):
                    _delete_attr(gm, target)
                erased_params.append(node)

        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    gm.meta["_retab_fast_quant_constant_fold_count"] = folded
    return folded


def _make_patched_constant_fold(original):
    def retab_constant_fold(
        gm: torch.fx.GraphModule,
        constraint_fn: Callable[[torch.fx.Node], bool] | None = None,
    ) -> None:
        if _needs_original_constant_folder(gm, constraint_fn):
            original(gm, constraint_fn)
            gm.meta["_retab_fast_quant_constant_fold_count"] = None
            return
        _fast_quant_constant_fold(gm, constraint_fn)

    retab_constant_fold.__name__ = "retab_constant_fold"
    retab_constant_fold.__retab_patched__ = True
    retab_constant_fold.__retab_original__ = original
    return retab_constant_fold


def install() -> None:
    """Patch TorchAO ``convert_pt2e`` with RETAB's narrow constant-fold fast path."""
    global _install_depth, _original_constant_fold, _patched_constant_fold

    import torchao.quantization.pt2e.quantize_pt2e as quantize_pt2e

    if _install_depth == 0:
        _original_constant_fold = quantize_pt2e.constant_fold
        _patched_constant_fold = _make_patched_constant_fold(
            _original_constant_fold
        )
        quantize_pt2e.constant_fold = _patched_constant_fold
    _install_depth += 1


def uninstall() -> None:
    """Restore TorchAO's original PT2E constant folder."""
    global _install_depth, _original_constant_fold, _patched_constant_fold

    if _install_depth == 0:
        return
    _install_depth -= 1
    if _install_depth != 0:
        return

    import torchao.quantization.pt2e.quantize_pt2e as quantize_pt2e

    if _original_constant_fold is not None:
        quantize_pt2e.constant_fold = _original_constant_fold
    _original_constant_fold = None
    _patched_constant_fold = None


@contextmanager
def activated():
    """Temporarily enable RETAB PT2E front-end fast paths."""
    install()
    try:
        yield
    finally:
        uninstall()
