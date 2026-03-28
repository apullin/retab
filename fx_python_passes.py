"""Run the real Python ARM backend fold_qdq and rescale passes.

Instead of reimplementing the complex qparams/rescale logic in Rust, we call
the battle-tested Python implementations directly. The Rust side handles
the simpler transforms (table ops, conv rewriting, memory format, etc.).
"""

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.insert_rescales_pass import (
    InsertRescaleInt32Pass,
    InsertRescalePass,
)
from executorch.backends.arm._passes.decompose_int16_activation_conv_pass import (
    DecomposeConvWithInt16ActivationPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir import ExportedProgram
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)


def _extract_qparam_dtype(qparam):
    if qparam is None:
        return None
    dtype = getattr(qparam, "dtype", None)
    if dtype is not None:
        return dtype
    getter = getattr(qparam, "get", None)
    if getter is None:
        return None
    try:
        return getter("dtype", None)
    except TypeError:
        return getter("dtype")


def _is_convolution_like_target(target):
    target_str = str(target)
    return (
        target == exir_ops.edge.aten.convolution.default
        or target == exir_ops.edge.aten.conv2d.default
        or "aten.convolution.default" in target_str
        or "aten.conv2d.default" in target_str
    )


def _is_raw_conv2d_target(target):
    target_str = str(target)
    return target == exir_ops.edge.aten.conv2d.default or "aten.conv2d.default" in target_str


def _node_has_int16_activation_conv_bias(node) -> bool:
    if node.op != "call_function":
        return False
    if not _is_convolution_like_target(node.target):
        return False
    if len(node.args) < 3 or node.args[2] is None:
        return False

    input_qparams = node.meta.get("input_qparams", {})
    qparam = input_qparams.get(0) if hasattr(input_qparams, "get") else None
    return _extract_qparam_dtype(qparam) == torch.int16


def _has_int16_activation_conv(graph_module) -> bool:
    for node in graph_module.graph.nodes:
        if _node_has_int16_activation_conv_bias(node):
            return True
    return False


def _normalize_raw_conv2d_to_convolution_for_int16_bias(graph_module):
    """Rewrite raw `aten.conv2d.default` into `aten.convolution.default`.

    The upstream Python int16-activation conv decomposition only matches
    `aten.convolution.default`. Direct lowering often still carries raw
    `aten.conv2d.default` nodes at this point, so normalize only the exact
    int16+bias subset the pass expects.
    """
    graph = graph_module.graph
    modified = False

    for node in list(graph.nodes):
        if not _node_has_int16_activation_conv_bias(node):
            continue
        if not _is_raw_conv2d_target(node.target):
            continue

        if len(node.args) < 3:
            continue

        input_node, weight_node, bias_node = node.args[:3]
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        stride = node.args[3] if len(node.args) > 3 else kwargs.pop("stride", (1, 1))
        padding = node.args[4] if len(node.args) > 4 else kwargs.pop("padding", (0, 0))
        dilation = node.args[5] if len(node.args) > 5 else kwargs.pop("dilation", (1, 1))
        groups = node.args[6] if len(node.args) > 6 else kwargs.pop("groups", 1)
        convolution_args = (
            input_node,
            weight_node,
            bias_node,
            stride,
            padding,
            dilation,
            False,
            (0, 0),
            groups,
        )

        with graph.inserting_before(node):
            new_node = graph.call_function(
                exir_ops.edge.aten.convolution.default,
                convolution_args,
                kwargs,
            )

        new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        modified = True

    if modified:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

    return graph_module


def run_python_fold_and_rescale(graph_module, exported_program=None):
    """Run Python's fold_qdq + rescale passes on the graph module.

    Args:
        graph_module: The FX GraphModule to transform.
        exported_program: Optional ExportedProgram, needed by FoldAndAnnotateQParamsPass
            to resolve parameter tensor nodes (per-channel scales/zps stored as graph params).

    Returns:
        The transformed GraphModule (may be the same object or a new one).
    """
    gm = graph_module

    # 1. Fold Q/DQ and annotate qparams as QuantArgs namedtuples in node.meta.
    #    The ExportedProgram is needed when DQ scale/zp args are parameter nodes
    #    rather than literal constants (happens with per-channel quantization).
    fold_pass = FoldAndAnnotateQParamsPass(exported_program)
    result = fold_pass(gm)
    if result is not None and hasattr(result, "graph_module"):
        gm = result.graph_module

    # 2. Replace surviving DQ->Q pairs with TOSA RESCALE ops.
    result = InsertRescalePass()(gm)
    if result is not None and hasattr(result, "graph_module"):
        gm = result.graph_module

    # 3. Insert INT32 rescales around elementwise ops (add, mul, sub, etc.).
    result = InsertRescaleInt32Pass()(gm)
    if result is not None and hasattr(result, "graph_module"):
        gm = result.graph_module

    return gm


def run_python_rescale_only(graph_module):
    """Run only the Python rescale passes (fold was already done by Rust).

    The Rust FxFoldAndAnnotateQParamsPass stores real QuantArgs namedtuples
    in node.meta["input_qparams"] and node.meta["output_qparams"]. This
    function runs only the rescale insertion passes which read those QuantArgs.

    Args:
        graph_module: The FX GraphModule (already Q/DQ-folded with QuantArgs).

    Returns:
        The transformed GraphModule (may be the same object or a new one).
    """
    gm = graph_module

    # 1. Replace surviving DQ->Q pairs with TOSA RESCALE ops.
    result = InsertRescalePass()(gm)
    if result is not None and hasattr(result, "graph_module"):
        gm = result.graph_module

    # 2. Insert INT32 rescales around elementwise ops (add, mul, sub, etc.).
    result = InsertRescaleInt32Pass()(gm)
    if result is not None and hasattr(result, "graph_module"):
        gm = result.graph_module

    return gm


def run_python_decompose_int16_activation_conv(graph_module):
    """Run Python's int16-activation conv decomposition on the graph module.

    This mirrors the Arm backend pass that rewrites conv+bias with int16
    activations into no-bias conv + bias add, avoiding illegal direct
    serialization of int16 conv bias.
    """
    gm = graph_module
    if not _has_int16_activation_conv(gm):
        return gm
    gm = _normalize_raw_conv2d_to_convolution_for_int16_bias(gm)
    int16_spec = TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
    with TosaLoweringContext(int16_spec):
        result = DecomposeConvWithInt16ActivationPass()(gm)
        if result is not None and hasattr(result, "graph_module"):
            gm = result.graph_module
    return gm
