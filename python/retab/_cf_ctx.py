"""Control-flow context helpers for fast preprocess."""

import logging

logger = logging.getLogger(__name__)


def _flatten_output_nodes(output_arg):
    if isinstance(output_arg, (list, tuple)):
        for item in output_arg:
            yield from _flatten_output_nodes(item)
    elif hasattr(output_arg, "op"):
        yield output_arg


def _fake_with_shape(fake_val, shape):
    if fake_val is None:
        return None
    return fake_val.new_empty(tuple(int(dim) for dim in shape), dtype=fake_val.dtype)


def _apply_dim_order(shape, dim_order):
    if dim_order is None or len(dim_order) != len(shape):
        return tuple(shape)
    return tuple(shape[i] for i in dim_order)


def _target_contains(node, text):
    return text in str(getattr(node, "target", ""))


def _is_tosa_transpose(node):
    return _target_contains(node, "tosa.TRANSPOSE")


def _is_tosa_table(node):
    return _target_contains(node, "tosa.TABLE")


def _is_tosa_rescale(node):
    return _target_contains(node, "tosa.RESCALE")


def _is_table_rescale(node):
    if not _is_tosa_rescale(node) or not getattr(node, "args", ()):
        return False
    return _is_tosa_table(node.args[0])


def _output_layout_permutation(node):
    target = str(getattr(node, "target", ""))
    if (
        node.op == "call_function"
        and "tosa.TRANSPOSE" in target
        and str(getattr(node, "name", "")).endswith("_output_layout")
        and len(getattr(node, "args", ())) >= 2
    ):
        return tuple(int(axis) for axis in node.args[1])
    return None


def _permute_shape(shape, perm):
    if len(shape) != len(perm):
        return None
    return tuple(shape[i] for i in perm)


def _actual_tosa_shape(node):
    fake_val = getattr(node, "meta", {}).get("val")
    if fake_val is None:
        return None

    if _is_tosa_table(node):
        input_node = node.args[0] if getattr(node, "args", ()) else None
        if input_node is not None:
            return _actual_tosa_shape(input_node)
        return tuple(fake_val.shape)

    if _is_table_rescale(node):
        return _actual_tosa_shape(node.args[0])

    shape = tuple(fake_val.shape)
    if _is_tosa_transpose(node):
        return shape
    return _apply_dim_order(shape, getattr(node, "meta", {}).get("tosa_dim_order"))


def _delegate_output_contract(node):
    fake_val = node.meta.get("val")
    tosa_dim_order = tuple(node.meta.get("tosa_dim_order", ()))
    perm = _output_layout_permutation(node)
    if perm is None:
        return fake_val, tosa_dim_order

    input_node = node.args[0]
    input_fake = getattr(input_node, "meta", {}).get("val")
    if input_fake is None:
        return fake_val, tosa_dim_order

    if not (_is_tosa_table(input_node) or _is_table_rescale(input_node)):
        return fake_val, tosa_dim_order

    input_shape = _actual_tosa_shape(input_node)
    if input_shape is None:
        input_shape = tuple(input_fake.shape)
    logical_shape = _permute_shape(input_shape, perm)
    if logical_shape is None:
        return fake_val, tosa_dim_order

    base_fake = fake_val if fake_val is not None else input_fake
    return _fake_with_shape(base_fake, logical_shape), tuple(range(len(logical_shape)))


def _collect_delegate_info_meta(graph_module):
    output_node = graph_module.graph.output_node()
    output_leaves = list(_flatten_output_nodes(output_node.args[0]))
    output_meta = []
    for node in output_leaves:
        fake_val, tosa_dim_order = _delegate_output_contract(node)
        output_meta.append(
            {
                "fake_val": fake_val,
                "tosa_dim_order": tosa_dim_order,
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

    try:
        if memory_format == torch.channels_last:
            if fake_tensor.is_contiguous(memory_format=torch.channels_last):
                return fake_tensor
        elif fake_tensor.is_contiguous():
            return fake_tensor
    except Exception:
        pass

    return fake_tensor.to(memory_format=memory_format)

def _extract_single_qparam_entry(qparams):
    if not qparams:
        return None
    if isinstance(qparams, dict):
        if 0 in qparams:
            return qparams[0]
        if len(qparams) == 1:
            return next(iter(qparams.values()))
    return qparams


def _branch_local_control_flow_output_qparam(placeholder):
    import torch

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
