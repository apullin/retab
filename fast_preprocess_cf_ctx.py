"""Control-flow context and graph metadata helpers for fast preprocess."""

import dataclasses
import logging
from typing import Any

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

    if fake_tensor.dim() != len(desired_dim_order):
        return fake_tensor

    try:
        memory_format = get_memory_format(list(desired_dim_order))
    except AssertionError:
        return fake_tensor
    if memory_format == torch.channels_last and fake_tensor.dim() != 4:
        return fake_tensor

    return fake_tensor.to(memory_format=memory_format)


def _actual_shape_from_meta(node) -> list[int] | None:
    fake_tensor = node.meta.get("val")
    if not hasattr(fake_tensor, "shape"):
        return None
    shape = list(fake_tensor.shape)
    dim_order = node.meta.get("tosa_dim_order")
    if (
        dim_order is not None
        and len(dim_order) == len(shape)
        and tuple(dim_order) != tuple(range(len(shape)))
    ):
        return [shape[idx] for idx in dim_order]
    return shape


@dataclasses.dataclass
class CfSubmoduleCtx:
    """Pre-computed context for a control-flow submodule."""

    submodule: Any
    cf_node: Any
    graph: Any
    placeholders: list
    output_node: Any
    is_cond: bool
    is_while: bool

    @classmethod
    def from_submodule(cls, submodule, cf_node):
        graph = submodule.graph
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        output_nodes = [n for n in graph.nodes if n.op == "output"]
        output_node = output_nodes[0] if output_nodes else None
        cf_target = str(getattr(cf_node, "target", "")) if cf_node else ""
        return cls(
            submodule=submodule,
            cf_node=cf_node,
            graph=graph,
            placeholders=placeholders,
            output_node=output_node,
            is_cond="cond" in cf_target,
            is_while="while" in cf_target,
        )

    def call_function_nodes(self):
        return [n for n in self.graph.nodes if n.op == "call_function"]

    def recompile(self):
        self.submodule.recompile()


def _for_each_cf_submodule(graph_module, visitor, *, include_parent=False):
    """Iterate control-flow submodules, applying `visitor(ctx)` to each."""
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    modules = []
    if include_parent:
        modules.append((graph_module, None))
    modules.extend(
        (sm, cf) for _, sm, cf in get_cond_while_submodules_nested(graph_module)
    )

    modified = False
    for submodule, cf_node in modules:
        ctx = CfSubmoduleCtx.from_submodule(submodule, cf_node)
        if visitor(ctx):
            ctx.recompile()
            modified = True
    return modified


def _refresh_tosa_dim_order_fake_tensors(graph_module) -> None:
    """Re-sync fake tensors with `tosa_dim_order` after FX-direct rewrites."""
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    modules = [graph_module]
    modules.extend(
        submodule for _, submodule, _ in get_cond_while_submodules_nested(graph_module)
    )

    for module in modules:
        updated = False
        for node in module.graph.nodes:
            if "val" not in node.meta or node.meta.get("tosa_dim_order") is None:
                continue
            node.meta["val"] = _get_matching_fake_tensor(node)
            updated = True
        if updated:
            module.recompile()


def _refresh_tosa_transpose_fake_tensors(graph_module) -> None:
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_cond_while_submodules_nested,
    )

    def _is_tosa_transpose_node(node) -> bool:
        if getattr(node, "op", None) != "call_function":
            return False
        target_str = str(getattr(node, "target", ""))
        return (
            "tosa.TRANSPOSE" in target_str
            or "tosa_TRANSPOSE" in target_str
            or "TosaTranspose" in target_str
        )

    modules = [graph_module]
    modules.extend(
        submodule for _, submodule, _ in get_cond_while_submodules_nested(graph_module)
    )

    for module in modules:
        updated = False
        for node in module.graph.nodes:
            if not _is_tosa_transpose_node(node):
                continue
            if not getattr(node, "args", None) or len(node.args) < 2:
                continue
            input_node = node.args[0]
            if not hasattr(input_node, "op"):
                continue
            input_fake = input_node.meta.get("val")
            if not hasattr(input_fake, "new_empty"):
                continue
            input_shape = _actual_shape_from_meta(input_node)
            if input_shape is None:
                continue
            try:
                perm = [int(idx) for idx in node.args[1]]
            except TypeError:
                continue
            if len(perm) != len(input_shape):
                continue
            new_shape = [input_shape[idx] for idx in perm]
            node_fake = node.meta.get("val", input_fake)
            base_tensor = node_fake if hasattr(node_fake, "new_empty") else input_fake
            dtype = getattr(node_fake, "dtype", None) or getattr(
                input_fake, "dtype", None
            )
            node.meta["val"] = base_tensor.new_empty(new_shape, dtype=dtype)
            node.meta["tosa_dim_order"] = tuple(range(len(new_shape)))
            updated = True
        if updated:
            module.recompile()

