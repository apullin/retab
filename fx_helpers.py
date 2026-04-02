"""Python helpers for FX-direct Rust passes that need to create new nodes.

The Rust code calls these via PyO3 when it needs to insert new FX nodes
(RESCALE, TRANSPOSE, clamp, etc.) — creating FX nodes purely from Rust is
fragile because we need proper meta["val"] FakeTensors and the correct
torch op overload objects.
"""

import dataclasses
import os
from copy import copy

import torch
from torch.fx import Graph, Node


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    """Quantization parameters for a single tensor."""

    scale: float
    zp: int
    qmin: int
    qmax: int
    dtype: object  # torch.dtype


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
    input_qspec: QuantSpec,
    output_qspec: QuantSpec,
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
        scale=input_qspec.scale,
        zp=input_qspec.zp,
        qmin=input_qspec.qmin,
        qmax=input_qspec.qmax,
        dtype=input_qspec.dtype,
    )
    out_quantargs = QuantArgs(
        scale=output_qspec.scale,
        zp=output_qspec.zp,
        qmin=output_qspec.qmin,
        qmax=output_qspec.qmax,
        dtype=output_qspec.dtype,
    )

    if output_qspec.dtype == torch.int8:
        quantized_inputs = torch.linspace(
            start=input_qspec.qmin,
            end=input_qspec.qmax,
            steps=256,
            dtype=torch.int8,
        )
        dequantized_inputs = in_quantargs.dequantize_value(quantized_inputs)
        return out_quantargs.quantize_value(torch_op(dequantized_inputs)).to(
            dtype=torch.int8
        ), 0

    if output_qspec.dtype in (torch.int16, torch.int32):
        quantized_inputs = torch.linspace(
            start=input_qspec.qmin,
            end=input_qspec.qmax + 1,
            steps=513,
            dtype=torch.int32,
        )
        dequantized_inputs = in_quantargs.dequantize_value(
            (quantized_inputs.clamp(input_qspec.qmin, input_qspec.qmax).to(dtype=input_qspec.dtype) & ~0x7F)
        )
        lut_values = out_quantargs.quantize_value(torch_op(dequantized_inputs))
        rshift = int(torch.ceil(torch.log2(lut_values.abs().max())).item()) + 1 - 16
        rescale_lshift = rshift - 7
        return (lut_values >> rshift).to(dtype=torch.int16), rescale_lshift

    raise ValueError(f"Unsupported output dtype for table: {output_qspec.dtype}")


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
    in_qspec = QuantSpec(input_scale, input_zp, input_qmin, input_qmax, input_dtype)
    out_qspec = QuantSpec(output_scale, output_zp, output_qmin, output_qmax, output_dtype)
    table_tensor, rescale_lshift = _compute_table(kind, in_qspec, out_qspec, scalar_arg)

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


