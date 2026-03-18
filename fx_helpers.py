"""Python helpers for FX-direct Rust passes that need to create new nodes.

The Rust code calls these via PyO3 when it needs to insert new FX nodes
(RESCALE, TRANSPOSE, clamp, etc.) — creating FX nodes purely from Rust is
fragile because we need proper meta["val"] FakeTensors and the correct
torch op overload objects.
"""

from copy import copy

import torch
from torch.fx import Graph, Node


def insert_node_after(
    graph_module,
    ref_node: Node,
    target,
    args: tuple,
    kwargs: dict | None = None,
    name: str | None = None,
) -> Node:
    """Insert a new call_function node after ref_node."""
    graph: Graph = graph_module.graph
    with graph.inserting_after(ref_node):
        new_node = graph.call_function(target, args=tuple(args), kwargs=kwargs or {})
        if name:
            new_node.name = name
        new_node.meta = copy(ref_node.meta)
        if "input_qparams" in new_node.meta:
            new_node.meta["input_qparams"] = {}
        if "output_qparams" in new_node.meta:
            new_node.meta["output_qparams"] = {}
    return new_node


def insert_node_before(
    graph_module,
    ref_node: Node,
    target,
    args: tuple,
    kwargs: dict | None = None,
    name: str | None = None,
) -> Node:
    """Insert a new call_function node before ref_node."""
    graph: Graph = graph_module.graph
    with graph.inserting_before(ref_node):
        new_node = graph.call_function(target, args=tuple(args), kwargs=kwargs or {})
        if name:
            new_node.name = name
        new_node.meta = copy(ref_node.meta)
        if "input_qparams" in new_node.meta:
            new_node.meta["input_qparams"] = {}
        if "output_qparams" in new_node.meta:
            new_node.meta["output_qparams"] = {}
    return new_node


def get_tosa_rescale_op():
    """Return the tosa.rescale op target, importing as needed."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.RESCALE.default
    except Exception:
        # Fallback: try to find it via torch.ops
        try:
            return torch.ops.tosa.rescale.default
        except Exception:
            return "tosa::rescale"


def get_tosa_transpose_op():
    """Return the tosa.transpose op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.TRANSPOSE.default
    except Exception:
        try:
            return torch.ops.tosa.transpose.default
        except Exception:
            return "tosa::transpose"


def get_aten_clamp_op():
    """Return the aten.clamp.default op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.edge.aten.clamp.default
    except Exception:
        return torch.ops.aten.clamp.default


def insert_rescale_node(
    graph_module,
    ref_node: Node,
    input_node: Node,
    out_dtype,
    scales: list,
    input_zp: int,
    output_zp: int,
    after: bool = True,
) -> Node:
    """Insert a TOSA RESCALE node before or after ref_node.

    Args:
        graph_module: the GraphModule
        ref_node: node to insert relative to
        input_node: the actual input to the rescale
        out_dtype: output dtype (torch.int8 or torch.int32)
        scales: list of float scale values
        input_zp: input zero point
        output_zp: output zero point
        after: if True insert after ref_node, else before
    """
    target = get_tosa_rescale_op()
    args = (input_node, out_dtype, scales, input_zp, output_zp)
    if after:
        return insert_node_after(graph_module, ref_node, target, args)
    else:
        return insert_node_before(graph_module, ref_node, target, args)


def insert_transpose_after(
    graph_module,
    node: Node,
    perm: list[int],
) -> Node:
    """Insert a TOSA TRANSPOSE node after the given node.

    Creates a transpose with the given permutation, updates meta to reflect
    the permuted shape, and rewires all users of `node` to use the transpose.

    Args:
        graph_module: the GraphModule
        node: the node to insert after
        perm: permutation indices, e.g. [0,2,3,1] for NCHW->NHWC
    """
    target = get_tosa_transpose_op()
    args = (node, list(perm))
    transpose = insert_node_after(graph_module, node, target, args)

    # Update the val shape in meta for the permuted output
    if "val" in node.meta:
        old_val = node.meta["val"]
        old_shape = list(old_val.shape)
        new_shape = [old_shape[p] for p in perm]
        transpose.meta["val"] = old_val.new_empty(new_shape)

    # Rewire: all users of node (except transpose itself) should use transpose
    users = [u for u in node.users if u is not transpose]
    for user in users:
        user.replace_input_with(node, transpose)

    return transpose


def insert_transpose_before(
    graph_module,
    ref_node: Node,
    input_node: Node,
    perm: list[int],
) -> Node:
    """Insert a TOSA TRANSPOSE node before ref_node.

    Creates a transpose of input_node with the given permutation and replaces
    ref_node's use of input_node with the transpose.

    Args:
        graph_module: the GraphModule
        ref_node: the node to insert before
        input_node: the input to transpose
        perm: permutation indices, e.g. [0,3,1,2] for NHWC->NCHW
    """
    target = get_tosa_transpose_op()
    args = (input_node, list(perm))
    transpose = insert_node_before(graph_module, ref_node, target, args)

    # Update the val shape in meta for the permuted output
    if "val" in input_node.meta:
        old_val = input_node.meta["val"]
        old_shape = list(old_val.shape)
        new_shape = [old_shape[p] for p in perm]
        transpose.meta["val"] = old_val.new_empty(new_shape)

    ref_node.replace_input_with(input_node, transpose)
    return transpose


def replace_target_with_clamp(
    graph_module,
    node: Node,
    min_val,
    max_val,
) -> None:
    """Replace a relu/hardtanh node target with aten.clamp.default."""
    target = get_aten_clamp_op()
    input_node = node.args[0]
    node.target = target
    node.args = (input_node, min_val, max_val)


# ---------------------------------------------------------------------------
# TABLE op helpers (InsertTableOps pass)
# ---------------------------------------------------------------------------

def get_tosa_table_op():
    """Return the tosa.TABLE op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.TABLE.default
    except Exception:
        try:
            return torch.ops.tosa.table.default
        except Exception:
            return "tosa::table"


import math
import numpy as np


# Map of supported math functions for table generation.
_TABLE_MATH_FNS = {
    "rsqrt": lambda x: 1.0 / math.sqrt(max(x, 1e-10)),
    "exp": math.exp,
    "log": lambda x: math.log(max(x, 1e-10)),
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "tanh": math.tanh,
}


def _compute_table(
    kind: str,
    input_scale: float,
    input_zp: int,
    input_qmin: int,
    input_qmax: int,
    output_scale: float,
    output_zp: int,
    output_qmin: int,
    output_qmax: int,
) -> list[int]:
    """Compute a 513-entry lookup table for a unary math function.

    For each integer i in [qmin, qmax+1]:
        x_float = (i - input_zp) * input_scale
        y_float = math_fn(x_float)
        y_quant = clamp(round(y_float / output_scale + output_zp), output_qmin, output_qmax)
        table[i - qmin] = y_quant   (as int16)

    Returns a list of 513 int16 values.
    """
    fn = _TABLE_MATH_FNS[kind]
    table = []
    for i in range(input_qmin, input_qmax + 2):  # +2 to get 513 entries (qmin..=qmax+1)
        x_float = (i - input_zp) * input_scale
        y_float = fn(x_float)
        y_quant = int(round(y_float / output_scale + output_zp))
        y_quant = max(output_qmin, min(output_qmax, y_quant))
        table.append(y_quant)
    return table


def insert_table_op(
    graph_module,
    node: Node,
    kind: str,
) -> None:
    """Replace a unary math node with a TOSA TABLE op.

    Reads input_qparams[0] and output_qparams[0] from node.meta,
    computes the lookup table, creates a const tensor node for the table
    data, and replaces the original node with a TABLE call_function.

    Args:
        graph_module: the GraphModule
        node: the unary math node to replace (e.g. rsqrt)
        kind: one of "rsqrt", "exp", "log", "sigmoid", "tanh"
    """
    graph = graph_module.graph
    meta = node.meta

    # Extract qparams
    input_qp = meta["input_qparams"][0]
    output_qp = meta["output_qparams"][0]

    input_scale = float(input_qp["scale"])
    input_zp = int(input_qp["zero_point"])
    input_qmin = int(input_qp.get("qmin", -128))
    input_qmax = int(input_qp.get("qmax", 127))

    output_scale = float(output_qp["scale"])
    output_zp = int(output_qp["zero_point"])
    output_qmin = int(output_qp.get("qmin", -128))
    output_qmax = int(output_qp.get("qmax", 127))

    # Compute the lookup table
    table_data = _compute_table(
        kind,
        input_scale, input_zp, input_qmin, input_qmax,
        output_scale, output_zp, output_qmin, output_qmax,
    )

    # Create a tensor constant for the table
    table_tensor = torch.tensor(table_data, dtype=torch.int16)

    # Register the table tensor as a buffer on the graph module so it
    # persists and can be referenced by a get_attr node.
    table_name = f"{node.name}_table"
    # Ensure unique name
    counter = 0
    attr_name = table_name
    while hasattr(graph_module, attr_name):
        counter += 1
        attr_name = f"{table_name}_{counter}"

    graph_module.register_buffer(attr_name, table_tensor)

    # Create a get_attr node for the table constant
    input_node = node.args[0]
    with graph.inserting_before(node):
        table_node = graph.get_attr(attr_name)
        table_node.meta = {"val": table_tensor}

    # Create the TABLE call_function node
    target = get_tosa_table_op()
    with graph.inserting_before(node):
        table_op_node = graph.call_function(target, args=(input_node, table_node))
        table_op_node.name = f"{node.name}_tosa_table"
        # Carry over meta from original node
        from copy import copy
        table_op_node.meta = copy(node.meta)
        # Clear qparams on the TABLE node (it's already quantized)
        table_op_node.meta.pop("input_qparams", None)
        table_op_node.meta.pop("output_qparams", None)

    # Replace all uses of the original node and erase it
    node.replace_all_uses_with(table_op_node)
    graph.erase_node(node)


# ---------------------------------------------------------------------------
# REDUCE_SUM helpers (ConvertSum pass)
# ---------------------------------------------------------------------------

def get_tosa_reduce_sum_op():
    """Return an op target for single-axis reduce_sum.
    Uses aten.sum since TOSA ops aren't registered in PyTorch's op registry.
    The serializer handles the ATen→TOSA REDUCE_SUM mapping."""
    return torch.ops.aten.sum.dim_IntList


def convert_sum_to_reduce_sum(
    graph_module,
    node: Node,
    input_node: Node,
    dims: list[int],
    keepdim: bool,
) -> None:
    """Replace an aten.sum.dim_IntList node with chained TOSA REDUCE_SUM ops.

    Each REDUCE_SUM reduces a single axis.  Dims are processed in
    descending order so earlier reductions don't shift later indices.

    Args:
        graph_module: the GraphModule
        node: the aten.sum node to replace
        input_node: the input tensor node
        dims: list of dimensions to reduce
        keepdim: whether to keep the reduced dimension
    """
    from copy import copy
    graph = graph_module.graph
    target = get_tosa_reduce_sum_op()

    # Sort dims descending to keep indices stable during chained reductions.
    sorted_dims = sorted(dims, reverse=True)

    current_input = input_node
    last_reduce = None

    for dim in sorted_dims:
        with graph.inserting_before(node):
            reduce_node = graph.call_function(
                target, args=(current_input, [dim], keepdim)
            )
            reduce_node.name = f"{node.name}_reduce_dim{dim}"
            reduce_node.meta = copy(node.meta)
            # Clear stale qparams (the reduce inherits from the sum node)
            if "input_qparams" in reduce_node.meta:
                reduce_node.meta["input_qparams"] = {}
            if "output_qparams" in reduce_node.meta:
                reduce_node.meta["output_qparams"] = {}
        current_input = reduce_node
        last_reduce = reduce_node

    # Copy qparams from original node onto the final reduce
    if last_reduce is not None:
        if "input_qparams" in node.meta:
            last_reduce.meta["input_qparams"] = node.meta["input_qparams"]
        if "output_qparams" in node.meta:
            last_reduce.meta["output_qparams"] = node.meta["output_qparams"]

        node.replace_all_uses_with(last_reduce)
        graph.erase_node(node)
