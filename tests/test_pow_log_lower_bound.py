"""rc4's POW LOG lower-bound check (InsertTableOpsPass._validate_pow_log_positive_lower_bound).

On an INT-only spec, rc4's quantizer rewrites pow(base, exp) as exp(exp * log(base)) when
the graph proves the base positive, and stamps the proven lower bound on the LOG, in
meta["custom"]. Stock raises when that bound does not stay positive after the LOG input's
quantization. RETAB hands such a graph to stock, which raises its own error.
"""

import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.executorch]

REASON = "POW LOG lower bound not positive after quantization"


class PositiveBasePow(torch.nn.Module):
    """rc4 test_pow.py's Pow_TensorTensorPositiveBase: abs(x) + 0.25 proves the base positive."""

    def forward(self, x, y):
        return torch.pow(torch.abs(x) + 0.25, y)


def _quantize_and_lower(inputs):
    from executorch.backends.arm.quantizer import (
        TOSAQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from retab.integration.executorch_arm import activated

    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    module = prepare_pt2e(
        torch.export.export(PositiveBasePow(), inputs, strict=True).module(), quantizer
    )
    module(*inputs)
    program = torch.export.export(convert_pt2e(module), inputs, strict=True)
    with activated():
        to_edge_transform_and_lower(
            program,
            partitioner=[TOSAPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )


def test_lower_bound_quantized_to_zero_goes_to_stock_which_raises():
    """rc4's test_pow_tensor_tensor_int_rejects_base_lower_bound_quantized_to_zero.

    Calibrating x on [0, 128] gives the LOG input an INT8 scale that maps 0.25 to the
    zero code.
    """
    from retab.preprocess import preprocess_stats

    before = preprocess_stats()["stock_fallback"].get(REASON, 0)
    with pytest.raises(Exception, match="InsertTableOpsPass") as exc_info:
        _quantize_and_lower((torch.tensor([[0.0, 128.0]]), torch.full((1, 2), -1.0)))

    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert "Unsafe Tensor/Tensor POW INT decomposition" in str(cause)
    assert preprocess_stats()["stock_fallback"].get(REASON, 0) == before + 1


def test_lower_bound_that_stays_positive_lowers_on_retab():
    from retab.preprocess import preprocess_stats

    base = torch.linspace(0.25, 1.75, 24).reshape(2, 3, 4)
    before = preprocess_stats()
    _quantize_and_lower((base, base.flip(-1)))
    after = preprocess_stats()

    assert after["retab"] > before["retab"], "RETAB's preprocess never ran"
    assert after["stock_fallback"] == before["stock_fallback"], "the graph fell back to stock"
