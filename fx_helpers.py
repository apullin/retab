"""Python helpers for FX-direct Rust passes that need to create new nodes.

The Rust code calls these via PyO3 when it needs to insert new FX nodes
(RESCALE, TRANSPOSE, clamp, etc.) — creating FX nodes purely from Rust is
fragile because we need proper meta["val"] FakeTensors and the correct
torch op overload objects.
"""

import os
from copy import copy

import torch
from torch.fx import Graph, Node


def _apply_fake_tensor_dim_order(fake_tensor, dim_order):
    """Return a fake tensor with the requested Arm dim-order memory format."""
    if fake_tensor is None or dim_order is None:
        return fake_tensor
    from executorch.exir.dim_order_utils import get_memory_format

    try:
        memory_format = get_memory_format(list(dim_order))
    except AssertionError:
        # Some Arm-internal dim orders are metadata-only tensor layouts, not
        # representable torch.memory_format values. Preserve the tensor and let
        # the explicit tosa_dim_order metadata carry the contract.
        return fake_tensor
    if memory_format == torch.channels_last and fake_tensor.dim() != 4:
        return fake_tensor
    return fake_tensor.to(memory_format=memory_format)


def _effective_tosa_dim_order(node: Node):
    current = node
    seen = set()
    while hasattr(current, "op") and id(current) not in seen:
        seen.add(id(current))
        dim_order = current.meta.get("tosa_dim_order") if hasattr(current, "meta") else None
        if dim_order is not None:
            return tuple(dim_order)
        if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
            break
        current = current.args[0]
    return None


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
        rescale = insert_node_after(graph_module, ref_node, target, args)
    else:
        rescale = insert_node_before(graph_module, ref_node, target, args)

    restamp_rescale_output_meta(rescale, input_node, out_dtype)

    return rescale


def restamp_rescale_output_meta(
    rescale_node: Node,
    input_node: Node,
    out_dtype,
) -> None:
    """Refresh a RESCALE node's FakeTensor metadata from its real output dtype."""
    # RESCALE materializes a new tensor in the requested output dtype; it
    # should not inherit accumulator-only special dtypes like INT48 from the
    # producer node.
    rescale_node.meta.pop("tosa_special_dtype", None)
    input_val = input_node.meta.get("val")
    if isinstance(input_val, (list, tuple)):
        input_val = input_val[0] if input_val else None
    if input_val is not None and hasattr(input_val, "new_empty"):
        rescale_node.meta["val"] = _apply_fake_tensor_dim_order(
            input_val.new_empty(input_val.shape, dtype=out_dtype),
            input_node.meta.get("tosa_dim_order"),
        )


def restamp_node_output_meta(
    node: Node,
    source_node: Node,
    out_dtype,
) -> None:
    """Retype an existing node's FakeTensor metadata without inserting an op."""
    node.meta.pop("tosa_special_dtype", None)
    source_val = source_node.meta.get("val")
    if isinstance(source_val, (list, tuple)):
        source_val = source_val[0] if source_val else None
    if source_val is not None and hasattr(source_val, "new_empty"):
        node.meta["val"] = _apply_fake_tensor_dim_order(
            source_val.new_empty(source_val.shape, dtype=out_dtype),
            source_node.meta.get("tosa_dim_order"),
        )


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
        old_shape = _node_actual_shape(node) or list(old_val.shape)
        new_shape = [old_shape[p] for p in perm]
        transpose.meta["val"] = old_val.new_empty(new_shape)
        transpose.meta["tosa_dim_order"] = tuple(perm)
        node.meta["tosa_dim_order"] = tuple(range(len(old_shape)))
        if "tosa_spatial_rank" in node.meta:
            transpose.meta["tosa_spatial_rank"] = node.meta["tosa_spatial_rank"]

    # Rewire: all users of node (except transpose itself) should use transpose
    users = [u for u in node.users if u is not transpose]
    for user in users:
        user.replace_input_with(node, transpose)

    return transpose


def _node_actual_shape(node: Node) -> list[int] | None:
    if "val" not in node.meta:
        return None
    old_shape = list(node.meta["val"].shape)
    dim_order = node.meta.get("tosa_dim_order")
    if (
        dim_order is not None
        and len(dim_order) == len(old_shape)
        and tuple(dim_order) != tuple(range(len(old_shape)))
    ):
        return [old_shape[p] for p in dim_order]
    return old_shape


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
        old_shape = _node_actual_shape(input_node) or list(old_val.shape)
        new_shape = [old_shape[p] for p in perm]
        transpose.meta["val"] = old_val.new_empty(new_shape)
        transpose.meta["tosa_dim_order"] = tuple(range(len(old_shape)))
        if "tosa_spatial_rank" in input_node.meta:
            transpose.meta["tosa_spatial_rank"] = input_node.meta["tosa_spatial_rank"]

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


_TABLE_TORCH_FNS = {
    "ceil": torch.ceil,
    "floor": torch.floor,
    "round": torch.round,
    "rsqrt": torch.rsqrt,
    "reciprocal": torch.reciprocal,
    "exp": torch.exp,
    "log": torch.log,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": torch.erf,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tan": torch.tan,
    "acos": torch.acos,
    "asin": torch.asin,
    "atan": torch.atan,
    "expm1": torch.expm1,
    "acosh": torch.acosh,
    "asinh": torch.asinh,
    "atanh": torch.atanh,
    "erfinv": torch.erfinv,
    "log1p": torch.log1p,
    "sqrt": torch.sqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "hardswish": torch.nn.functional.hardswish,
    "silu": torch.nn.functional.silu,
    "gelu": lambda x: torch.nn.functional.gelu(x, approximate="none"),
}


def _table_torch_fn(kind: str, scalar_arg: float | None = None):
    if kind == "pow":
        if scalar_arg is None:
            raise ValueError("pow table op requires a scalar exponent")
        return lambda x: torch.pow(x, float(scalar_arg))

    if kind == "elu":
        alpha = scalar_arg if scalar_arg is not None else 1.0
        return lambda x: torch.nn.functional.elu(x, alpha=alpha)

    if kind in _TABLE_TORCH_FNS:
        return _TABLE_TORCH_FNS[kind]

    raise ValueError(f"Unknown table op kind: {kind}")


def _compute_table(
    kind: str,
    input_scale: float,
    input_zp: int,
    input_qmin: int,
    input_qmax: int,
    input_dtype,
    output_scale: float,
    output_zp: int,
    output_qmin: int,
    output_qmax: int,
    output_dtype,
    scalar_arg: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute TABLE contents by mirroring Arm's Python InsertTableOps pass.

    The important part is semantic fidelity, not "helpful" scalar guarding.
    Domain-sensitive ops like acos/asin rely on the exact torch behavior here;
    clamping invalid inputs changes the table bytes and therefore the model.
    """
    from executorch.backends.arm._passes.quant_args import QuantArgs

    torch_op = _table_torch_fn(kind, scalar_arg)
    in_quantargs = QuantArgs(
        scale=input_scale,
        zp=input_zp,
        qmin=input_qmin,
        qmax=input_qmax,
        dtype=input_dtype,
    )
    out_quantargs = QuantArgs(
        scale=output_scale,
        zp=output_zp,
        qmin=output_qmin,
        qmax=output_qmax,
        dtype=output_dtype,
    )

    if output_dtype == torch.int8:
        quantized_inputs = torch.linspace(
            start=input_qmin,
            end=input_qmax,
            steps=256,
            dtype=torch.int8,
        )
        dequantized_inputs = in_quantargs.dequantize_value(quantized_inputs)
        return out_quantargs.quantize_value(torch_op(dequantized_inputs)).to(
            dtype=torch.int8
        ), 0

    if output_dtype in (torch.int16, torch.int32):
        quantized_inputs = torch.linspace(
            start=input_qmin,
            end=input_qmax + 1,
            steps=513,
            dtype=torch.int32,
        )
        dequantized_inputs = in_quantargs.dequantize_value(
            (quantized_inputs.clamp(input_qmin, input_qmax).to(dtype=input_dtype) & ~0x7F)
        )
        lut_values = out_quantargs.quantize_value(torch_op(dequantized_inputs))
        rshift = int(torch.ceil(torch.log2(lut_values.abs().max())).item()) + 1 - 16
        rescale_lshift = rshift - 7
        return (lut_values >> rshift).to(dtype=torch.int16), rescale_lshift

    raise ValueError(f"Unsupported output dtype for table: {output_dtype}")


def insert_table_op(
    graph_module,
    node: Node,
    kind: str,
    scalar_arg: float | None = None,
) -> None:
    """Replace a unary math node with a TOSA TABLE op.

    Reads input_qparams[0] and output_qparams[0] from node.meta,
    computes the lookup table, creates a const tensor node for the table
    data, and replaces the original node with a TABLE call_function.

    Args:
        graph_module: the GraphModule
        node: the unary math node to replace (e.g. rsqrt)
        kind: one of "rsqrt", "reciprocal", "pow", "exp", "log", "sigmoid",
            "tanh", "erf", "silu", "gelu"
        scalar_arg: embedded scalar exponent for `pow.Tensor_Scalar`
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

    def _torch_dtype(val, default=torch.int8):
        """Normalize a QuantArgs dtype field to a torch.dtype."""
        if isinstance(val, torch.dtype):
            return val
        if isinstance(val, str):
            return {
                "torch.int8": torch.int8,
                "int8": torch.int8,
                "torch.int16": torch.int16,
                "int16": torch.int16,
                "torch.int32": torch.int32,
                "int32": torch.int32,
            }.get(val, default)
        return default

    def _quant_args_from_operator(current: Node):
        try:
            from executorch.backends.arm._passes.quant_args import QuantArgs

            return QuantArgs.from_operator(current.target, current.args)
        except Exception:
            return None

    def _is_dequantize_node(current: Node) -> bool:
        target_str = str(getattr(current, "target", ""))
        return "dequantize_per_tensor" in target_str

    def _is_quantize_node(current: Node) -> bool:
        target_str = str(getattr(current, "target", ""))
        return "quantize_per_tensor" in target_str

    # Prefer the quantization contract encoded by the adjacent quantize /
    # dequantize operators. Direct lowering can preserve stale node.meta
    # qparams across rewrites, while the operator args remain authoritative.
    input_node = node.args[0] if getattr(node, "args", None) else None
    if hasattr(input_node, "op") and _is_dequantize_node(input_node):
        derived_input_qp = _quant_args_from_operator(input_node)
        if derived_input_qp is not None:
            input_qp = derived_input_qp

    original_users = list(node.users)
    for user in original_users:
        if not hasattr(user, "op") or user.op != "call_function" or not _is_quantize_node(user):
            continue
        derived_output_qp = _quant_args_from_operator(user)
        if derived_output_qp is not None:
            output_qp = derived_output_qp
            break

    input_scale = float(_scalar(_get(input_qp, "scale", 1.0)))
    input_zp = int(_scalar(_get(input_qp, "zp", _get(input_qp, "zero_point", 0))))
    input_qmin = int(_get(input_qp, "qmin", -128))
    input_qmax = int(_get(input_qp, "qmax", 127))
    input_dtype = _torch_dtype(_get(input_qp, "dtype", torch.int8))

    output_scale = float(_scalar(_get(output_qp, "scale", 1.0)))
    output_zp = int(_scalar(_get(output_qp, "zp", _get(output_qp, "zero_point", 0))))
    output_qmin = int(_get(output_qp, "qmin", -128))
    output_qmax = int(_get(output_qp, "qmax", 127))
    output_dtype = _torch_dtype(_get(output_qp, "dtype", torch.int8))

    if os.environ.get("TOSA_ENGINE_DEBUG"):
        print(
            "[insert_table_op:qparams]",
            node.name,
            "kind=",
            kind,
            "input=",
            {
                "scale": input_scale,
                "zp": input_zp,
                "qmin": input_qmin,
                "qmax": input_qmax,
                "dtype": str(input_dtype),
            },
            "output=",
            {
                "scale": output_scale,
                "zp": output_zp,
                "qmin": output_qmin,
                "qmax": output_qmax,
                "dtype": str(output_dtype),
            },
        )

    # Compute the lookup table
    table_tensor, rescale_lshift = _compute_table(
        kind,
        input_scale, input_zp, input_qmin, input_qmax, input_dtype,
        output_scale, output_zp, output_qmin, output_qmax,
        output_dtype,
        scalar_arg,
    )

    if os.environ.get("TOSA_ENGINE_DEBUG"):
        print(
            "[insert_table_op:table]",
            node.name,
            "min=",
            int(table_tensor.min().item()),
            "max=",
            int(table_tensor.max().item()),
            "head=",
            [int(v) for v in table_tensor.flatten()[:8]],
        )

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
    def _extract_node_qparam(current: Node):
        seen = set()
        while hasattr(current, "op") and id(current) not in seen:
            seen.add(id(current))

            output_qparams = current.meta.get("output_qparams")
            if isinstance(output_qparams, dict):
                if 0 in output_qparams:
                    return output_qparams[0]
                if len(output_qparams) == 1:
                    return next(iter(output_qparams.values()))
            elif output_qparams:
                return output_qparams

            target_str = str(getattr(current, "target", ""))
            if (
                "tosa.RESCALE" in target_str
                and len(getattr(current, "args", ())) >= 5
                and isinstance(current.args[1], torch.dtype)
                and hasattr(current.args[0], "op")
            ):
                source_qparam = _extract_node_qparam(current.args[0])
                if source_qparam is not None:
                    from executorch.backends.arm._passes.quant_args import QuantArgs

                    multiplier_arg = current.args[2]
                    output_dtype = current.args[1]
                    output_zp = int(current.args[4])
                    qinfo = torch.iinfo(output_dtype)
                    if getattr(source_qparam, "per_channel", False):
                        source_scales = [float(v) for v in source_qparam.get_scale_per_channel()]
                        if isinstance(multiplier_arg, (list, tuple)):
                            if len(multiplier_arg) == 1:
                                multiplier_values = [float(multiplier_arg[0])] * len(source_scales)
                            elif len(multiplier_arg) == len(source_scales):
                                multiplier_values = [float(v) for v in multiplier_arg]
                            else:
                                multiplier_values = None
                        else:
                            multiplier_values = [float(multiplier_arg)] * len(source_scales)
                        if multiplier_values and all(v not in (0.0,) for v in multiplier_values):
                            return QuantArgs(
                                scale=[
                                    src / mult
                                    for src, mult in zip(source_scales, multiplier_values)
                                ],
                                zp=[output_zp] * len(source_scales),
                                qmin=int(qinfo.min),
                                qmax=int(qinfo.max),
                                dtype=output_dtype,
                                axis=getattr(source_qparam, "axis", 0),
                                per_channel=True,
                            )
                    else:
                        source_scale = float(_scalar(_get(source_qparam, "scale", 1.0)))
                        if isinstance(multiplier_arg, (list, tuple)):
                            if len(multiplier_arg) != 1 or float(multiplier_arg[0]) == 0.0:
                                multiplier_value = None
                            else:
                                multiplier_value = float(multiplier_arg[0])
                        else:
                            multiplier_value = float(multiplier_arg)
                        if multiplier_value not in (None, 0.0):
                            return QuantArgs(
                                scale=source_scale / multiplier_value,
                                zp=output_zp,
                                qmin=int(qinfo.min),
                                qmax=int(qinfo.max),
                                dtype=output_dtype,
                                axis=0,
                                per_channel=False,
                            )

            input_qparams = current.meta.get("input_qparams")
            if isinstance(input_qparams, dict):
                if 0 in input_qparams:
                    return input_qparams[0]
                if len(input_qparams) == 1:
                    return next(iter(input_qparams.values()))
            elif input_qparams:
                return input_qparams

            try:
                from executorch.backends.arm._passes.quant_args import QuantArgs

                return QuantArgs.from_operator(current.target, current.args)
            except Exception:
                pass

            if not getattr(current, "args", None) or not hasattr(current.args[0], "op"):
                break
            current = current.args[0]
        return None

    def _same_qparam(lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return False
        return (
            float(_scalar(_get(lhs, "scale", 1.0)))
            == float(_scalar(_get(rhs, "scale", 1.0)))
            and int(_scalar(_get(lhs, "zp", _get(lhs, "zero_point", 0))))
            == int(_scalar(_get(rhs, "zp", _get(rhs, "zero_point", 0))))
            and int(_get(lhs, "qmin", -128)) == int(_get(rhs, "qmin", -128))
            and int(_get(lhs, "qmax", 127)) == int(_get(rhs, "qmax", 127))
            and _torch_dtype(_get(lhs, "dtype", torch.int8))
            == _torch_dtype(_get(rhs, "dtype", torch.int8))
        )

    input_node = node.args[0]
    producer_qp = _extract_node_qparam(input_node) if hasattr(input_node, "meta") else None

    # Parse, don't validate: TOSA TABLE is a quantized op. If the source is a
    # dequantize node, lower the TABLE against the quantized producer instead
    # of preserving the transient FP32 view. That matches the generic Arm path
    # and lets any required branch-local domain fixup happen as an explicit
    # RESCALE before TABLE.
    if (
        hasattr(input_node, "op")
        and _is_dequantize_node(input_node)
        and getattr(input_node, "args", None)
        and hasattr(input_node.args[0], "meta")
    ):
        quantized_source = input_node.args[0]
        quantized_source_qp = producer_qp or _extract_node_qparam(quantized_source)
        if quantized_source_qp is not None:
            input_node = quantized_source
            producer_qp = quantized_source_qp

    if (
        producer_qp is not None
        and (
            float(_scalar(_get(producer_qp, "scale", 1.0))) != input_scale
            or int(_scalar(_get(producer_qp, "zp", _get(producer_qp, "zero_point", 0))))
            != input_zp
            or _torch_dtype(_get(producer_qp, "dtype", torch.int8)) != input_dtype
        )
    ):
        input_node = insert_rescale_node(
            graph_module,
            node,
            input_node,
            input_dtype,
            [float(_scalar(_get(producer_qp, "scale", 1.0))) / input_scale],
            int(_scalar(_get(producer_qp, "zp", _get(producer_qp, "zero_point", 0)))),
            input_zp,
            after=False,
        )
        input_node.meta["input_qparams"] = {0: producer_qp}
        input_node.meta["output_qparams"] = {0: input_qp}

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
        old_val = node.meta.get("val")
        input_val = (
            input_node.meta.get("val")
            if hasattr(input_node, "meta")
            else None
        )
        input_dim_order = _effective_tosa_dim_order(input_node)
        if os.environ.get("TOSA_ENGINE_DEBUG"):
            print(
                "[table-meta]",
                node.name,
                "node_dim_order=",
                node.meta.get("tosa_dim_order"),
                "input_dim_order=",
                input_dim_order,
                "input_meta_dim_order=",
                input_node.meta.get("tosa_dim_order") if hasattr(input_node, "meta") else None,
                "node_shape=",
                tuple(old_val.shape) if old_val is not None and hasattr(old_val, "shape") else None,
                "input_shape=",
                tuple(input_val.shape) if input_val is not None and hasattr(input_val, "shape") else None,
            )
        if input_dim_order is not None:
            table_op_node.meta["tosa_dim_order"] = input_dim_order
        source_val = input_val if input_val is not None else old_val
        if source_val is not None and hasattr(source_val, "new_empty"):
            # TOSA TABLE's output dtype is determined by the TABLE input dtype,
            # not by the eventual quantized result dtype after an optional
            # RESCALE. INT8 TABLE returns INT8, while INT16 TABLE returns INT32.
            table_output_dtype = torch.int32 if input_dtype == torch.int16 else input_dtype
            table_op_node.meta["val"] = source_val.new_empty(
                source_val.shape, dtype=table_output_dtype
            )

    output_node = table_op_node
    if rescale_lshift != 0:
        output_node = insert_rescale_node(
            graph_module,
            table_op_node,
            table_op_node,
            output_dtype,
            [2.0**rescale_lshift],
            0,
            0,
            after=True,
        )

    # Replace all uses of the original node and erase it
    node.replace_all_uses_with(output_node)

    # Absorb a trailing quantize node when TABLE already materializes the same
    # quantized domain. This keeps the branch graph in the same higher-level
    # contract as the generic Arm path instead of leaving a now-invalid
    # quantize(INT8) wrapper around the TABLE result.
    for user in original_users:
        if not hasattr(user, "op") or user.op != "call_function" or not _is_quantize_node(user):
            continue
        quant_qp = _extract_node_qparam(user)
        if not _same_qparam(quant_qp, output_qp):
            continue
        user.replace_all_uses_with(output_node)
        graph.erase_node(user)

    graph.erase_node(node)


# ---------------------------------------------------------------------------
# Collapse decomposed round back to TABLE
# ---------------------------------------------------------------------------

def _skip_passthrough(node):
    """Follow arg[0] through dequantize/quantize/RESCALE/Identity wrappers."""
    for _ in range(8):
        if not isinstance(node, Node) or node.op != "call_function":
            break
        t = str(node.target)
        if "dequantize" in t or "quantize" in t or "RESCALE" in t or "rescale" in t or "Identity" in t:
            node = node.args[0] if node.args else node
        else:
            break
    return node


def collapse_decomposed_round_to_table(graph_module) -> bool:
    """Replace the decomposed banker's-rounding subgraph with a single TABLE.

    Must run BEFORE q/dq folding (on the pre-folded graph where math ops sit
    between dequantize/quantize pairs).  The TABLE takes INT8 input, outputs
    INT8, and a trailing dequantize converts back to float so the rest of the
    pre-folding graph is happy.
    """
    graph = graph_module.graph
    modified = False
    debug = os.environ.get("TOSA_ENGINE_DEBUG")

    # Phase 1: find candidates.
    candidates = []  # list of (final_where_node, quantized_input_node, scale, zp, qmin, qmax)

    for node in graph.nodes:
        if node.op != "call_function" or "where" not in str(node.target):
            continue
        args = node.args
        if len(args) < 3:
            continue

        # where(is_tie, tie_result, round_up) — condition is arg[0]
        is_tie = args[0]
        if not isinstance(is_tie, Node) or is_tie.op != "call_function":
            continue
        if "eq" not in str(is_tie.target):
            continue
        if len(is_tie.args) < 2:
            continue

        # eq(frac, half_const) — frac is arg[0], might be through dq
        frac_raw = is_tie.args[0]
        frac_node = frac_raw
        # Skip one dequantize wrapper
        if isinstance(frac_node, Node) and "dequantize" in str(getattr(frac_node, 'target', '')):
            frac_node = frac_node.args[0] if frac_node.args else frac_node
        # Skip one quantize wrapper
        if isinstance(frac_node, Node) and "quantize" in str(getattr(frac_node, 'target', '')):
            frac_node = frac_node.args[0] if frac_node.args else frac_node

        if not isinstance(frac_node, Node) or frac_node.op != "call_function":
            continue
        if "sub" not in str(frac_node.target):
            continue

        # sub.args[0] is the round input (possibly through dq)
        round_input_float = frac_node.args[0]
        if not isinstance(round_input_float, Node):
            continue

        # Find the quantized (INT8) version: if round_input is a dequantize,
        # its arg[0] is the quantized tensor.
        quantized_input = None
        scale = zp = qmin = qmax = None
        if "dequantize" in str(getattr(round_input_float, 'target', '')):
            dq_args = round_input_float.args
            quantized_input = dq_args[0] if dq_args else None
            if len(dq_args) >= 5:
                try:
                    scale = float(dq_args[1])
                    zp = int(dq_args[2])
                    qmin = int(dq_args[3])
                    qmax = int(dq_args[4])
                except (TypeError, ValueError):
                    continue

        if quantized_input is None or scale is None:
            continue

        # Verify floor(x) is the second arg of sub
        floor_raw = frac_node.args[1] if len(frac_node.args) > 1 else None
        # Walk through possible dq/q wrappers
        floor_check = floor_raw
        for _ in range(3):
            if isinstance(floor_check, Node) and floor_check.op == "call_function":
                t = str(floor_check.target)
                if "dequantize" in t or "quantize" in t:
                    floor_check = floor_check.args[0] if floor_check.args else floor_check
                else:
                    break
            else:
                break
        if not isinstance(floor_check, Node) or "floor" not in str(getattr(floor_check, 'target', '')):
            continue

        # Extract output qparams from the quantize node after where_1
        out_scale, out_zp, out_qmin, out_qmax = scale, zp, qmin, qmax
        for user in node.users:
            if isinstance(user, Node) and user.op == "call_function":
                ut = str(user.target)
                if "quantize" in ut and "dequantize" not in ut:
                    u_args = user.args
                    if len(u_args) >= 5:
                        try:
                            out_scale = float(u_args[1])
                            out_zp = int(u_args[2])
                            out_qmin = int(u_args[3])
                            out_qmax = int(u_args[4])
                        except (TypeError, ValueError):
                            pass
                    break

        if debug:
            print(f"[collapse_round] matched: where={node.name} q_input={quantized_input.name} in_scale={scale} in_zp={zp} out_scale={out_scale} out_zp={out_zp}")
        candidates.append((node, quantized_input, round_input_float, scale, zp, qmin, qmax, out_scale, out_zp, out_qmin, out_qmax))

    # Phase 2: replace each candidate with TABLE + dequantize.
    for final_where, quantized_input, float_input, scale, zp, qmin, qmax, out_scale, out_zp, out_qmin, out_qmax in candidates:
        # Compute the round TABLE: for each INT8 value, dequant → round → requant
        # Input uses (scale, zp), output uses (out_scale, out_zp).
        table_entries = []
        for q in range(qmin, qmax + 1):
            float_val = (q - zp) * scale
            rounded = round(float_val)  # Python round = banker's rounding
            requant = int(max(out_qmin, min(out_qmax, round(rounded / out_scale + out_zp))))
            table_entries.append(requant)
        table_tensor = torch.tensor(table_entries, dtype=torch.int8)

        if debug:
            print(f"[collapse_round] table head={table_entries[:8]} tail={table_entries[-4:]}")

        # Register table as buffer
        attr_name = f"{final_where.name}_round_table"
        counter = 0
        while hasattr(graph_module, attr_name):
            counter += 1
            attr_name = f"{final_where.name}_round_table_{counter}"
        graph_module.register_buffer(attr_name, table_tensor)

        # Create nodes: get_attr → TABLE → dequantize
        with graph.inserting_before(final_where):
            table_const = graph.get_attr(attr_name)
            table_const.meta["val"] = table_tensor

            table_target = get_tosa_table_op()
            table_node = graph.call_function(table_target, args=(quantized_input, table_const))

            # TABLE output is INT8 with same qparams as input
            input_val = float_input.meta.get("val")
            if input_val is not None and hasattr(input_val, "new_empty"):
                table_node.meta["val"] = input_val.new_empty(input_val.shape, dtype=torch.int8)
            else:
                table_node.meta["val"] = table_tensor.new_empty([])

            # Dequantize TABLE output back to float for the pre-folding graph
            # Use OUTPUT qparams since the TABLE output is requantized with those.
            dq_target = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            dq_node = graph.call_function(
                dq_target,
                args=(table_node, out_scale, out_zp, out_qmin, out_qmax, torch.int8),
            )
            if input_val is not None:
                dq_node.meta["val"] = input_val.clone() if hasattr(input_val, 'clone') else input_val

        # Replace all uses of final_where with dq_node
        final_where.replace_all_uses_with(dq_node)
        # Fix: replace_all_uses also changed dq_node's own args
        dq_node.args = (table_node, out_scale, out_zp, out_qmin, out_qmax, torch.int8)

        graph.erase_node(final_where)
        modified = True

        if debug:
            print(f"[collapse_round] replaced {final_where.name} with TABLE → dequantize")

    if modified:
        # Clean up dead decomposition nodes
        changed = True
        while changed:
            changed = False
            for node in reversed(list(graph.nodes)):
                if node.op == "call_function" and len(node.users) == 0:
                    try:
                        graph.erase_node(node)
                        changed = True
                    except Exception:
                        pass
        graph_module.recompile()
    return modified


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
    current_input_qparams = copy(node.meta.get("input_qparams", {}))
    current_output_qparams = copy(node.meta.get("output_qparams", {}))

    for dim in sorted_dims:
        with graph.inserting_before(node):
            reduce_node = graph.call_function(
                target, args=(current_input, [dim], keepdim)
            )
            reduce_node.name = f"{node.name}_reduce_dim{dim}"
            reduce_node.meta = copy(node.meta)
            reduce_node.meta["input_qparams"] = copy(current_input_qparams)
            reduce_node.meta["output_qparams"] = copy(current_output_qparams)
            input_val = current_input.meta.get("val")
            if isinstance(input_val, (list, tuple)):
                input_val = input_val[0] if input_val else None
            if input_val is not None and hasattr(input_val, "new_empty"):
                input_shape = list(input_val.shape)
                rank = len(input_shape)
                normalized_dim = dim % rank
                output_shape = list(input_shape)
                if keepdim:
                    output_shape[normalized_dim] = 1
                else:
                    del output_shape[normalized_dim]
                reduce_node.meta["val"] = input_val.new_empty(output_shape)
        current_input = reduce_node
        current_input_qparams = copy(current_output_qparams)
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


def _resolve_fake_arg(arg):
    """Resolve FX Node arguments to their fake/meta values recursively."""
    if isinstance(arg, Node):
        return _get_first_fake_tensor(arg)
    if isinstance(arg, tuple):
        return tuple(_resolve_fake_arg(x) for x in arg)
    if isinstance(arg, list):
        return [_resolve_fake_arg(x) for x in arg]
    return arg


def refresh_node_fake_tensor(node: Node) -> None:
    """Recompute node.meta['val'] from the current FX args/kwargs when possible."""
    op_target = node.target
    try:
        fake_args = tuple(_resolve_fake_arg(arg) for arg in node.args)
        fake_kwargs = {
            key: _resolve_fake_arg(value) for key, value in dict(node.kwargs).items()
        }
        node.meta["val"] = _apply_fake_tensor_dim_order(
            op_target(*fake_args, **fake_kwargs),
            node.meta.get("tosa_dim_order"),
        )
    except Exception:
        pass


_SAFE_META_REFRESH_TARGETS = (
    "aten.permute_copy",
    "aten.avg_pool2d",
    "aten.max_pool2d_with_indices",
    "aten.clamp.default",
    "aten.view_copy",
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "sum.dim_IntList",
    "tosa.TRANSPOSE",
    "tosa.RESCALE",
    "rescale.default",
)


def refresh_safe_quantized_fake_tensors(graph_module) -> None:
    """Refresh FakeTensor metadata for ops whose output follows current inputs."""
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if any(key in target for key in _SAFE_META_REFRESH_TARGETS):
            refresh_node_fake_tensor(node)


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
    refresh_node_fake_tensor(node)
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

    # Parse, don't validate: keep the same logical decomposition as the
    # Python Arm pass, then let the memory-format pass legalize layout later.
    #
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
# Conv1d legalization helpers
# ---------------------------------------------------------------------------

def unsqueeze_conv1d_node(graph_module, node: Node) -> None:
    """Convert a 1D convolution into a 2D convolution wrapped in view_copy ops."""
    graph = graph_module.graph
    args = node.args
    target_str = str(node.target)

    if "conv1d.default" in target_str:
        # aten.conv1d.default signature:
        #   conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # Minimum args: 2 (input, weight). Optional args get defaults.
        if len(args) < 2:
            raise RuntimeError(
                f"Expected aten.conv1d.default to have at least 2 args, got {len(args)}"
            )
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else (1,)
        pad = args[4] if len(args) > 4 else (0,)
        dilation = args[5] if len(args) > 5 else (1,)
        groups = args[6] if len(args) > 6 else 1
        transposed = False
        output_padding = (0,)
    else:
        if len(args) < 9:
            raise RuntimeError(
                f"Expected aten.convolution.default to have 9 args, got {len(args)}"
            )
        (
            x,
            weight,
            bias,
            stride,
            pad,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = args[:9]

    stride_tuple = tuple(stride)
    if len(stride_tuple) != 1:
        return

    pad_tuple = tuple(pad)
    dilation_tuple = tuple(dilation)
    output_padding_tuple = tuple(output_padding)

    view_copy_op = _get_edge_op("aten.view_copy.default")
    conv_op = _get_edge_op("aten.convolution.default")

    with graph.inserting_before(node):
        x_unsqueezed = _create_node(
            graph,
            view_copy_op,
            args=(x, list(_get_first_fake_tensor(x).shape) + [1]),
            from_node=node,
            inherit_qparams=False,
        )
        x_unsqueezed.meta["input_qparams"] = {}
        x_unsqueezed.meta["output_qparams"] = {}
        refresh_node_fake_tensor(x_unsqueezed)

        w_unsqueezed = _create_node(
            graph,
            view_copy_op,
            args=(weight, list(_get_first_fake_tensor(weight).shape) + [1]),
            from_node=node,
            inherit_qparams=False,
        )
        w_unsqueezed.meta["input_qparams"] = {}
        w_unsqueezed.meta["output_qparams"] = {}
        refresh_node_fake_tensor(w_unsqueezed)

        conv2d = _create_node(
            graph,
            conv_op,
            args=(
                x_unsqueezed,
                w_unsqueezed,
                bias,
                stride_tuple + (1,),
                pad_tuple + (0,),
                dilation_tuple + (1,),
                transposed,
                output_padding_tuple + (0,),
                groups,
            ),
            kwargs=dict(node.kwargs),
            from_node=node,
            inherit_qparams=True,
        )

    with graph.inserting_after(conv2d):
        squeezed = _create_node(
            graph,
            view_copy_op,
            args=(conv2d, list(_get_first_fake_tensor(conv2d).shape)[:-1]),
            from_node=node,
            inherit_qparams=False,
        )
        squeezed.meta["input_qparams"] = {}
        squeezed.meta["output_qparams"] = {}
        refresh_node_fake_tensor(squeezed)

    node.replace_all_uses_with(squeezed)
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
                args=(shifted, qmin, qmax), from_node=node,
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
    try:
        from executorch.backends.arm._passes.arm_pass_utils import (
            get_param_tensor as _arm_get_param_tensor,
        )

        tensor = _arm_get_param_tensor(exported_program, node)
        if tensor is not None:
            return tensor
    except Exception:
        pass

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

    const_node.meta["arm_constant_placeholder"] = True

    node.replace_all_uses_with(const_node)
    node.graph.erase_node(node)
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


# ---------------------------------------------------------------------------
# RewriteConv helpers
# ---------------------------------------------------------------------------

def get_tosa_conv2d_op():
    """Return the tosa.CONV2D op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.CONV2D.default
    except Exception:
        try:
            return torch.ops.tosa.conv2d.default
        except Exception:
            return "tosa::conv2d"


def get_tosa_conv3d_op():
    """Return the tosa.CONV3D op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops

        return exir_ops.backend.tosa.CONV3D.default
    except Exception:
        try:
            return torch.ops.tosa.conv3d.default
        except Exception:
            return "tosa::conv3d"


def get_tosa_depthwise_conv2d_op():
    """Return the tosa.DEPTHWISE_CONV2D op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.DEPTHWISE_CONV2D.default
    except Exception:
        try:
            return torch.ops.tosa.depthwise_conv2d.default
        except Exception:
            return "tosa::depthwise_conv2d"


def get_tosa_transpose_conv2d_op():
    """Return the tosa.TRANSPOSE_CONV2D op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops
        return exir_ops.backend.tosa.TRANSPOSE_CONV2D.default
    except Exception:
        try:
            return torch.ops.tosa.transpose_conv2d.default
        except Exception:
            return "tosa::transpose_conv2d"


def _expand_spatial_param(param, spatial_rank: int) -> list[int]:
    """Expand scalar/1-element spatial params to the requested rank."""
    if isinstance(param, int):
        return [param] * spatial_rank
    param_list = list(param)
    if len(param_list) == 1 and spatial_rank > 1:
        param_list *= spatial_rank
    return param_list


def _adjust_pad_if_needed(
    input_size: int,
    kernel_size: int,
    stride: int,
    pad: int,
    dilation: int,
) -> int:
    """Mirror Arm RewriteConvPass trailing-pad legality adjustment."""
    effective_kernel = (kernel_size - 1) * dilation + 1
    remainder = (input_size + 2 * pad - effective_kernel) % stride
    if remainder > pad:
        return pad
    return pad - remainder


def get_tosa_scatter_op():
    """Return the tosa.SCATTER op target."""
    try:
        from executorch.exir.dialects._ops import ops as exir_ops

        return exir_ops.backend.tosa.SCATTER.default
    except Exception:
        try:
            return torch.ops.tosa.scatter.default
        except Exception:
            return "tosa::scatter"


def rewrite_conv_to_tosa(
    graph_module,
    node: Node,
    is_depthwise: bool,
    input_zp: int,
    weight_zp: int,
    in_channels: int,
) -> None:
    """Rewrite an aten.convolution.default or aten.conv2d.default node to TOSA.

    Mirrors the Python RewriteConvPass: changes the target, rearranges args
    (stride/pad/dilation formatting), handles bias creation, and reshapes
    weights for depthwise convolution.

    Args:
        graph_module: the GraphModule
        node: the convolution node
        is_depthwise: True for depthwise_conv2d, False for conv2d
        input_zp: input zero point from qparams
        weight_zp: weight zero point from qparams
        in_channels: number of input channels
    """
    graph = graph_module.graph
    args = node.args
    target_str = str(node.target)
    if "conv2d.default" in target_str:
        if len(args) < 2:
            raise RuntimeError(
                f"Expected aten.conv2d.default to have at least 2 args, got {len(args)}"
            )
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else (1, 1)
        pad = args[4] if len(args) > 4 else (0, 0)
        dilation = args[5] if len(args) > 5 else (1, 1)
        groups = args[6] if len(args) > 6 else 1
        transposed = False
        output_padding = (0, 0)
    else:
        if len(args) < 9:
            raise RuntimeError(
                f"Expected aten.convolution.default to have 9 args, got {len(args)}"
            )
        (
            x,
            weight,
            bias,
            stride,
            pad,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = args[:9]

    input_val = _get_first_fake_tensor(x)
    spatial_rank = max(len(getattr(input_val, "shape", ())) - 2, 0)
    stride_list = _expand_spatial_param(stride, spatial_rank)
    dilation_list = _expand_spatial_param(dilation, spatial_rank)
    pad_list = _expand_spatial_param(pad, spatial_rank)
    output_padding_list = _expand_spatial_param(output_padding, spatial_rank)

    stride_tuple = tuple(stride_list)
    dilation_tuple = tuple(dilation_list)
    weight_val = _get_first_fake_tensor(weight)
    input_shape = list(getattr(input_val, "shape", ()))
    weight_shape = list(getattr(weight_val, "shape", ()))
    input_qparams = node.meta.get("input_qparams") or {}
    if isinstance(input_qparams, dict):
        input_qparam = input_qparams.get(0)
        if input_qparam is None and len(input_qparams) == 1:
            input_qparam = next(iter(input_qparams.values()))
    else:
        input_qparam = None
    input_dtype = getattr(_get_first_fake_tensor(x), "dtype", None)
    input_contract_dtype = getattr(input_qparam, "dtype", input_dtype)

    # Handle missing bias: create a zero bias placeholder
    if bias is None:
        output_val = node.meta.get("val")
        if output_val is not None:
            output_channels = output_val.shape[1]
        else:
            output_channels = weight.meta["val"].shape[0]

        has_qparams = "output_qparams" in node.meta and len(node.meta.get("output_qparams", {})) > 0
        if has_qparams:
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.int32)
        else:
            output_dtype = node.meta["val"].dtype if output_val is not None else torch.float32
            bias_data = torch.zeros(size=(output_channels,), dtype=output_dtype)

        bias_name = f"{node.name}_bias"
        counter = 0
        attr_name = bias_name
        while hasattr(graph_module, attr_name):
            counter += 1
            attr_name = f"{bias_name}_{counter}"
        graph_module.register_buffer(attr_name, bias_data)

        with graph.inserting_before(node):
            bias = graph.get_attr(attr_name)
            bias.meta = {"val": bias_data}
    if bias is not None and input_contract_dtype == torch.int16:
        from executorch.backends.arm.tosa.mapping import TosaSpecialDtype

        if not hasattr(bias, "meta") or bias.meta is None:
            bias.meta = {}
        bias.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48

    # Select target op
    if transposed:
        if spatial_rank != 2:
            raise RuntimeError(
                "Only 2D transpose convolutions are supported in the Arm backend."
            )
        if any(d != 1 for d in dilation_list):
            raise RuntimeError(
                "Transpose convolutions with dilation are not supported in the Arm backend."
            )
        target_op = get_tosa_transpose_conv2d_op()
        conv_args = (
            x,
            weight,
            bias,
            [
                -pad_list[0],
                -pad_list[0] + output_padding_list[0],
                -pad_list[1],
                -pad_list[1] + output_padding_list[1],
            ],
            stride_tuple,
        )
        conv_kwargs = {"input_zp": input_zp, "weight_zp": weight_zp}
    elif is_depthwise:
        tosa_pad = []
        for axis_index, p in enumerate(pad_list):
            tosa_pad.extend([p, p])
            trailing_index = axis_index * 2 + 1
            tosa_pad[trailing_index] = _adjust_pad_if_needed(
                input_shape[axis_index + 2],
                weight_shape[axis_index + 2],
                stride_list[axis_index],
                tosa_pad[trailing_index],
                dilation_list[axis_index],
            )
        target_op = get_tosa_depthwise_conv2d_op()
        conv_args = (x, weight, bias, stride_tuple, tosa_pad, dilation_tuple)
        conv_kwargs = None
    elif spatial_rank == 3:
        if groups != 1:
            raise RuntimeError(
                "CONV3D with groups != 1 is not supported in the Arm backend."
            )
        tosa_pad = []
        for axis_index, p in enumerate(pad_list):
            tosa_pad.extend([p, p])
            trailing_index = axis_index * 2 + 1
            tosa_pad[trailing_index] = _adjust_pad_if_needed(
                input_shape[axis_index + 2],
                weight_shape[axis_index + 2],
                stride_list[axis_index],
                tosa_pad[trailing_index],
                dilation_list[axis_index],
            )
        target_op = get_tosa_conv3d_op()
        conv_args = (x, weight, bias, stride_tuple, tosa_pad, dilation_tuple)
        conv_kwargs = None
    else:
        tosa_pad = []
        for axis_index, p in enumerate(pad_list):
            tosa_pad.extend([p, p])
            trailing_index = axis_index * 2 + 1
            tosa_pad[trailing_index] = _adjust_pad_if_needed(
                input_shape[axis_index + 2],
                weight_shape[axis_index + 2],
                stride_list[axis_index],
                tosa_pad[trailing_index],
                dilation_list[axis_index],
            )
        target_op = get_tosa_conv2d_op()
        conv_args = (x, weight, bias, stride_tuple, tosa_pad, dilation_tuple)
        conv_kwargs = None

    with graph.inserting_after(node):
        tosa_op = _create_node(
            graph, target_op,
            args=conv_args,
            kwargs=conv_kwargs,
            from_node=node,
            inherit_qparams=True,
        )

    input_fake = _get_first_fake_tensor(x)
    output_fake = _get_first_fake_tensor(node)
    bias_fake = _get_first_fake_tensor(bias) if bias is not None else None
    try:
        tosa_op.meta["val"] = target_op(
            input_fake,
            _get_first_fake_tensor(weight),
            bias_fake,
            *conv_args[3:],
        )
    except Exception:
        if output_fake is not None and getattr(input_fake, "dtype", None) in (
            torch.int8,
            torch.int16,
        ):
            tosa_op.meta["val"] = output_fake.new_empty(
                output_fake.shape, dtype=torch.int32
            )
            if input_fake.dtype == torch.int16:
                from executorch.backends.arm.tosa.mapping import TosaSpecialDtype

                tosa_op.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48

    replacement = tosa_op
    if spatial_rank == 3 and output_fake is not None and input_fake is not None:
        def _qparam_get(qparams, idx):
            if isinstance(qparams, dict):
                return qparams.get(idx)
            if isinstance(qparams, (list, tuple)):
                return qparams[idx] if idx < len(qparams) else None
            return None

        def _field(qparam, attr, default=None):
            if qparam is None:
                return default
            if hasattr(qparam, attr):
                return getattr(qparam, attr)
            if isinstance(qparam, dict):
                return qparam.get(attr, default)
            getter = getattr(qparam, "get", None)
            if getter is None:
                return default
            try:
                return getter(attr, default)
            except TypeError:
                value = getter(attr)
                return default if value is None else value

        input_qp = _qparam_get(node.meta.get("input_qparams", {}), 0)
        weight_qp = _qparam_get(node.meta.get("input_qparams", {}), 1)
        output_qp = _qparam_get(node.meta.get("output_qparams", {}), 0)

        if (
            input_fake.dtype == torch.int8
            and getattr(tosa_op.meta.get("val"), "dtype", None) == torch.int32
            and input_qp is not None
            and weight_qp is not None
            and output_qp is not None
        ):
            input_scale = _field(input_qp, "scale", 1.0)
            output_scale = _field(output_qp, "scale", 1.0)
            output_zp = int(_field(output_qp, "zp", _field(output_qp, "zero_point", 0)) or 0)
            output_dtype = _field(output_qp, "dtype", output_fake.dtype)
            per_channel = bool(_field(weight_qp, "per_channel", False))
            if per_channel:
                weight_scales = _field(weight_qp, "scale", [1.0])
            else:
                weight_scales = [_field(weight_qp, "scale", 1.0)]
            if not isinstance(weight_scales, (list, tuple)):
                weight_scales = [weight_scales]
            input_scale_scalar = input_scale[0] if isinstance(input_scale, (list, tuple)) else input_scale
            output_scale_scalar = output_scale[0] if isinstance(output_scale, (list, tuple)) else output_scale
            if output_scale_scalar:
                post_conv_scale = [
                    (float(input_scale_scalar) * float(weight_scale)) / float(output_scale_scalar)
                    for weight_scale in weight_scales
                ]
                rescale = insert_rescale_node(
                    graph_module,
                    tosa_op,
                    tosa_op,
                    output_dtype,
                    post_conv_scale,
                    0,
                    output_zp,
                    after=True,
                )
                rescale.meta["input_qparams"] = {0: input_qp}
                rescale.meta["output_qparams"] = {0: output_qp}
                replacement = rescale

    node.replace_all_uses_with(replacement)
    graph.erase_node(node)


def _calculate_tosa_scatter_dims(
    index_shape: list[int], source_shape: list[int], num_index_dims: int
) -> tuple[int, int, int, int]:
    n, k, w, c = 1, 1, 1, 1
    w = math.prod(index_shape) if index_shape else 1
    for i, dim in enumerate(source_shape):
        if i < num_index_dims:
            k *= dim
    total_vals = math.prod(source_shape)
    c = int(total_vals / k) if k else 1
    return n, k, w, c


def _calculate_values_stride(source_shape: list[int]) -> list[int]:
    values_strides: list[int] = []
    stride = 1
    for dim in reversed(source_shape):
        values_strides.insert(0, stride)
        stride *= dim
    return values_strides


def rewrite_index_put_to_tosa(graph_module, node: Node) -> None:
    """Rewrite aten.index_put into view+scatter+view.

    This mirrors the Python RewriteIndexPutPass shape:
      1. Expand None indices to explicit arange tensors
      2. Flatten indices into [N, W]
      3. Reshape source to [N, K, C], values to [N, W, C]
      4. Emit tosa.SCATTER
      5. Reshape back to the original source shape
    """
    graph = graph_module.graph
    args = node.args
    if len(args) < 3:
        return

    source = args[0]
    indices = args[1]
    values = args[2]
    accumulate = bool(args[3]) if len(args) > 3 else False
    if accumulate or not isinstance(indices, (list, tuple)):
        return

    source_shape = list(_get_first_fake_tensor(source).shape)
    if not source_shape:
        return

    index_dtype = None
    for idx in indices:
        if isinstance(idx, Node):
            index_dtype = _get_first_fake_tensor(idx).dtype
            break
    if index_dtype is None:
        return

    arange_op = _get_edge_op("aten.arange.start_step")
    view_copy_op = _get_edge_op("aten.view_copy.default")
    add_op = _get_edge_op("aten.add.Tensor")
    mul_op = _get_edge_op("aten.mul.Tensor")
    full_op = _get_edge_op("aten.full.default")
    scatter_op = get_tosa_scatter_op()

    processed_indices: list[Node] = []
    with graph.inserting_before(node):
        for dim_idx, idx in enumerate(indices):
            if idx is None:
                expanded = _create_node(
                    graph,
                    arange_op,
                    args=(0, int(source_shape[dim_idx]), 1),
                    kwargs={"dtype": index_dtype},
                    from_node=node,
                    inherit_qparams=False,
                )
                processed_indices.append(expanded)
            elif isinstance(idx, Node):
                processed_indices.append(idx)
            else:
                return

    if not processed_indices:
        return

    index_shapes = [tuple(_get_first_fake_tensor(idx).shape) for idx in processed_indices]
    try:
        broadcast_shape = list(torch.broadcast_shapes(*index_shapes))
    except Exception:
        return

    n, k, w, c = _calculate_tosa_scatter_dims(
        broadcast_shape, source_shape, len(processed_indices)
    )
    if not (n > 0 and k > 0 and w > 0 and c > 0):
        return

    values_strides = _calculate_values_stride(source_shape)

    with graph.inserting_before(node):
        flat_indices = None
        for dim_idx, index_node in enumerate(processed_indices):
            scale_val = int(values_strides[dim_idx] / c)
            mul_const = _create_node(
                graph,
                full_op,
                args=((1,), scale_val),
                kwargs={"dtype": index_dtype},
                from_node=node,
                inherit_qparams=False,
            )
            scaled_index = _create_node(
                graph,
                mul_op,
                args=(index_node, mul_const),
                from_node=node,
                inherit_qparams=False,
            )
            flat_indices = (
                scaled_index
                if flat_indices is None
                else _create_node(
                    graph,
                    add_op,
                    args=(scaled_index, flat_indices),
                    from_node=node,
                    inherit_qparams=False,
                )
            )

        if flat_indices is None:
            return

        reshape_indices = _create_node(
            graph,
            view_copy_op,
            args=(flat_indices, [n, w]),
            from_node=node,
            inherit_qparams=False,
        )
        reshape_source = _create_node(
            graph,
            view_copy_op,
            args=(source, [n, k, c]),
            from_node=node,
            inherit_qparams=False,
        )
        reshape_values = _create_node(
            graph,
            view_copy_op,
            args=(values, [n, w, c]),
            from_node=node,
            inherit_qparams=False,
        )
        scatter_node = _create_node(
            graph,
            scatter_op,
            args=(reshape_source, reshape_indices, reshape_values),
            from_node=node,
            inherit_qparams=False,
        )
        output = _create_node(
            graph,
            view_copy_op,
            args=(scatter_node, source_shape),
            from_node=node,
            inherit_qparams=False,
        )

    node.replace_all_uses_with(output)
    graph.erase_node(node)
    graph.eliminate_dead_code()
