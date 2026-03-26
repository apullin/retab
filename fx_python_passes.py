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
from executorch.exir import ExportedProgram
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)


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
    int16_spec = TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
    with TosaLoweringContext(int16_spec):
        result = DecomposeConvWithInt16ActivationPass()(gm)
        if result is not None and hasattr(result, "graph_module"):
            gm = result.graph_module
    return gm
