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

    # Extract qparams -- supports both QuantArgs namedtuples (from Python fold pass)
    # and plain dicts (legacy Rust fold pass).
    input_qp = meta["input_qparams"][0]
    output_qp = meta["output_qparams"][0]

    def _get(qp, attr, default=None):
        """Read a field from either a QuantArgs namedtuple or a dict."""
        if hasattr(qp, attr):
            return getattr(qp, attr)
        if isinstance(qp, dict):
            return qp.get(attr, default)
        return default

    def _scalar(val):
        """Ensure we get a scalar float from scale (which may be a list)."""
        if isinstance(val, (list, tuple)):
            return val[0] if val else 1.0
        return val

    input_scale = float(_scalar(_get(input_qp, "scale", 1.0)))
    input_zp = int(_scalar(_get(input_qp, "zp", _get(input_qp, "zero_point", 0))))
    input_qmin = int(_get(input_qp, "qmin", -128))
    input_qmax = int(_get(input_qp, "qmax", 127))

    output_scale = float(_scalar(_get(output_qp, "scale", 1.0)))
    output_zp = int(_scalar(_get(output_qp, "zp", _get(output_qp, "zero_point", 0))))
    output_qmin = int(_get(output_qp, "qmin", -128))
    output_qmax = int(_get(output_qp, "qmax", 127))

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


# ---------------------------------------------------------------------------
# DecomposeLinear helpers
# ---------------------------------------------------------------------------

def _get_first_fake_tensor(node: Node):
    """Return the FakeTensor from node.meta['val'], handling list/tuple."""
    val = node.meta["val"]
    if isinstance(val, (list, tuple)):
        return val[0]
    return val


def _get_edge_op(name: str):
    """Resolve an edge op target by dotted name, e.g. 'aten.view_copy.default'."""
    from executorch.exir.dialects._ops import ops as exir_ops
    parts = name.split(".")
    obj = exir_ops.edge
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _create_node(graph, op_target, args, kwargs=None, from_node=None, inherit_qparams=False):
    """Simplified version of arm_pass_utils.create_node for FX-direct passes."""
    node = graph.create_node("call_function", op_target, args=args, kwargs=kwargs or {})
    new_meta = {}
    if from_node is not None:
        for key in from_node.meta.keys():
            new_meta[key] = from_node.meta[key]
        if not inherit_qparams:
            if "input_qparams" in new_meta:
                new_meta["input_qparams"] = {}
            if "output_qparams" in new_meta:
                new_meta["output_qparams"] = {}
    node.meta = new_meta
    return node


def decompose_linear_node(graph_module, node: Node) -> None:
    """Decompose a single aten.linear.default node into view+conv2d+view.

    Mirrors the Python DecomposeLinearPass.call() logic for one node.

    Args:
        graph_module: the GraphModule
        node: the linear node to decompose
    """
    graph = graph_module.graph
    args = node.args
    input_node = args[0]
    weights_node = args[1]
    bias_node = args[2] if len(args) > 2 else None

    output_shape = list(_get_first_fake_tensor(node).shape)
    input_shape = list(_get_first_fake_tensor(input_node).shape)
    weights_shape = list(_get_first_fake_tensor(weights_node).shape)

    # Compute batch dimension: product of all dims except last
    batches = 1
    for dim in input_shape[:-1]:
        batches *= dim

    # input has shape (..., Ci) -> reshape to (N, Ci, 1, 1)
    input_reshaped_shape = [batches, input_shape[-1], 1, 1]
    # weights have shape (Co, Ci) -> reshape to (Co, Ci, 1, 1)
    weights_reshaped_shape = [weights_shape[0], weights_shape[1], 1, 1]

    view_copy_op = _get_edge_op("aten.view_copy.default")
    conv_op = _get_edge_op("aten.convolution.default")

    with graph.inserting_before(node):
        # Reshape input to 4D
        input_reshaped = _create_node(
            graph, view_copy_op,
            args=(input_node, input_reshaped_shape),
            from_node=node, inherit_qparams=False,
        )

        # Reshape weights to 4D
        weights_reshaped = _create_node(
            graph, view_copy_op,
            args=(weights_node, weights_reshaped_shape),
            from_node=node, inherit_qparams=False,
        )

        # Conv2d with 1x1 kernel
        conv = _create_node(
            graph, conv_op,
            args=(
                input_reshaped,
                weights_reshaped,
                bias_node,
                [1, 1],   # strides
                [0, 0],   # padding
                [1, 1],   # dilation
                False,    # transposed
                [0, 0],   # output padding
                1,        # groups
            ),
            from_node=node, inherit_qparams=True,
        )

    with graph.inserting_after(conv):
        # Reshape output back to original shape
        output = _create_node(
            graph, view_copy_op,
            args=(conv, output_shape),
            from_node=node, inherit_qparams=False,
        )

    node.replace_all_uses_with(output)
    graph.erase_node(node)
    graph.eliminate_dead_code()


# ---------------------------------------------------------------------------
# DecomposeQuantNodes helpers
# ---------------------------------------------------------------------------

def decompose_quant_node(graph_module, node: Node, is_quantize: bool) -> None:
    """Decompose a single Q or DQ node into primitive arithmetic.

    For quantize_per_tensor (is_quantize=True):
        scaled = mul(x, 1/scale)
        rounded = round(scaled)
        shifted = add(rounded, zp)
        clamped = clamp(shifted, qmin, qmax)
        output = to_dtype(clamped, target_dtype)

    For dequantize_per_tensor (is_quantize=False):
        casted = to_dtype(x, int32)
        shifted = sub(casted, zp)
        casted_float = to_dtype(shifted, fp_dtype)
        output = mul(casted_float, scale)

    Args:
        graph_module: the GraphModule
        node: the Q or DQ node to decompose
        is_quantize: True for quantize, False for dequantize
    """
    graph = graph_module.graph
    args = node.args
    x = args[0]
    scale = float(args[1])
    zero_point = int(args[2])
    qmin = int(args[3])
    qmax = int(args[4])
    dtype = args[5]

    input_rank = x.meta["val"].ndim
    input_dtype = x.meta["val"].dtype
    output_dtype = node.meta["val"].dtype
    fp_dtype = output_dtype if not is_quantize else input_dtype

    # For quantization, use 1/scale
    effective_scale = (1.0 / scale) if is_quantize else scale

    mul_op = _get_edge_op("aten.mul.Tensor")
    add_op = _get_edge_op("aten.add.Tensor")
    sub_op = _get_edge_op("aten.sub.Tensor")
    round_op = _get_edge_op("aten.round.default")
    clamp_op = _get_edge_op("aten.clamp.default")
    full_op = _get_edge_op("aten.full.default")
    to_copy_op = _get_edge_op("dim_order_ops._to_dim_order_copy.default")

    with graph.inserting_before(node):
        # Create scale constant
        scale_const = _create_node(
            graph, full_op,
            args=((1,) * input_rank, effective_scale),
            kwargs={"dtype": fp_dtype},
        )

        # Create zero_point constant
        zp_dtype = fp_dtype if is_quantize else torch.int32
        zp_const = _create_node(
            graph, full_op,
            args=((1,) * input_rank, zero_point),
            kwargs={"dtype": zp_dtype},
        )

        if is_quantize:
            scaled = _create_node(
                graph, mul_op,
                args=(x, scale_const), from_node=node,
            )
            rounded = _create_node(
                graph, round_op,
                args=(scaled,), from_node=node,
            )
            shifted = _create_node(
                graph, add_op,
                args=(rounded, zp_const), from_node=node,
            )
            clamped = _create_node(
                graph, clamp_op,
                args=(shifted, float(qmin), float(qmax)), from_node=node,
            )
            output = _create_node(
                graph, to_copy_op,
                args=(clamped,), kwargs={"dtype": dtype}, from_node=node,
            )
        else:
            # Dequantize
            input_casted = _create_node(
                graph, to_copy_op,
                args=(x,), kwargs={"dtype": torch.int32}, from_node=node,
            )
            shifted = _create_node(
                graph, sub_op,
                args=(input_casted, zp_const), from_node=node,
            )
            casted_float = _create_node(
                graph, to_copy_op,
                args=(shifted,), kwargs={"dtype": fp_dtype}, from_node=node,
            )
            output = _create_node(
                graph, mul_op,
                args=(casted_float, scale_const), from_node=node,
            )

    node.replace_all_uses_with(output)
    graph.erase_node(node)


# ---------------------------------------------------------------------------
# FuseConstantArgs helpers
# ---------------------------------------------------------------------------

def _get_param_tensor(exported_program, node: Node):
    """Resolve the actual tensor data for a constant placeholder node."""
    import torch._export.utils
    if torch._export.utils.is_param(exported_program, node):
        fqn = exported_program.graph_signature.inputs_to_parameters.get(node.name, None)
        if fqn:
            return exported_program.state_dict[fqn]
    if torch._export.utils.is_buffer(exported_program, node):
        fqn = exported_program.graph_signature.inputs_to_buffers.get(node.name, None)
        if fqn:
            parts = fqn.split(".")
            obj = exported_program.graph_module
            for p in parts:
                obj = getattr(obj, p)
            return obj
    if torch._export.utils.is_lifted_tensor_constant(exported_program, node):
        fqn = exported_program.graph_signature.inputs_to_lifted_tensor_constants.get(
            node.name, None
        )
        if fqn:
            parts = fqn.split(".")
            obj = exported_program.graph_module
            for p in parts:
                obj = getattr(obj, p)
            return obj
    raise ValueError(f"Cannot resolve param tensor for {node.name}")


def fuse_constant_node(graph_module, exported_program, node: Node) -> bool:
    """Evaluate a call_function with all-constant inputs and replace it with
    a single pre-computed constant placeholder.

    Returns True if the node was fused, False if skipped.
    """
    from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
    from executorch.backends.transforms.utils import create_constant_placeholder
    from torch.export.graph_signature import InputKind

    # Skip if special dtype is SHAPE
    if node.meta.get(TosaSpecialDtype.meta_key(), None) == TosaSpecialDtype.SHAPE:
        return False

    input_nodes = list(node.all_input_nodes)
    qparams = node.meta.get("input_qparams", None)

    def resolve_arg(arg):
        if isinstance(arg, Node) and arg in input_nodes:
            idx = input_nodes.index(arg)
            t = _get_param_tensor(exported_program, arg)
            if qparams and idx in qparams.keys():
                t = qparams[idx].dequantize_value(t)
            return t
        if isinstance(arg, tuple):
            return tuple(resolve_arg(x) for x in arg)
        if isinstance(arg, list):
            return [resolve_arg(x) for x in arg]
        return arg

    new_args = tuple(resolve_arg(a) for a in node.args)
    new_kwargs = {k: resolve_arg(v) for k, v in node.kwargs.items()}

    data = node.target(*new_args, **new_kwargs)

    # Only fuse if the tensor does not get bigger.
    first_fake = node.meta["val"]
    if isinstance(first_fake, (list, tuple)):
        first_fake = first_fake[0]
    if data.numel() > first_fake.numel():
        return False

    if "output_qparams" in node.meta and len(node.meta["output_qparams"]) > 0:
        q_params = next(iter(node.meta["output_qparams"].values()))
        data = q_params.quantize_value(data)

    insert_pos = input_nodes[0]

    # Determine the kind from the first constant input
    from executorch.backends.arm._passes.arm_pass_utils import (
        get_constant_placeholder_kind,
        is_persistent_buffer,
    )
    input_kind = get_constant_placeholder_kind(exported_program, insert_pos)
    persistent_buffer = is_persistent_buffer(exported_program, insert_pos)

    with node.graph.inserting_before(insert_pos):
        const_node = create_constant_placeholder(
            exp_program=exported_program,
            graph=node.graph,
            kind=input_kind,
            name=node.name + "_fused_const",
            data=data,
            persistent_buffer=persistent_buffer,
        )

    # Propagate special dtype if present
    special_dtypes = set()
    for inp in input_nodes:
        special_type = inp.meta.get(TosaSpecialDtype.meta_key(), None)
        if special_type and special_type != TosaSpecialDtype.SHAPE:
            special_dtypes.add(special_type)
    if len(special_dtypes) == 1:
        special_dtype = list(special_dtypes)[0]
        if data.abs().max() <= special_dtype.max():
            const_node.meta[TosaSpecialDtype.meta_key()] = special_dtype

    node.replace_all_uses_with(const_node)
    return True


def delete_constant_placeholder(exported_program, node: Node) -> None:
    """Remove an unused constant placeholder from the ExportedProgram."""
    try:
        from executorch.backends.transforms.utils import (
            delete_constant_placeholder as _delete,
        )
        _delete(exported_program, node)
    except Exception:
        # Best-effort: if the utility isn't available, just erase the node.
        try:
            node.graph.erase_node(node)
        except Exception:
            pass
