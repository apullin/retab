import numpy as np
import pytest
import torch
from torch.export.graph_signature import InputKind

pytest.importorskip("executorch.backends.arm.tosa.compile_spec")
pytest.importorskip("retab._native")

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from retab.preprocess import _all_rust_preprocess
from tosa import DType, Op, TosaGraph

try:
    from tosa_tools.tosa_reference_model import tosa_reference_model as reference_model
except ImportError:
    reference_model = pytest.importorskip("tosa_reference_model").tosa_reference_model


def _export(module, inputs):
    program = torch.export.export(module, inputs, strict=True)
    graph = program.graph_module.graph
    for node in list(graph.nodes):
        if node.target == torch.ops.aten._assert_tensor_metadata.default:
            graph.erase_node(node)
    program.graph_module.recompile()
    return program


def _lower_and_compare(module, inputs):
    program = _export(module, inputs)
    result = _all_rust_preprocess(program, TosaCompileSpec("TOSA-1.0+INT+cf"))
    data = bytes(result.processed_bytes)
    graph = TosaGraph.TosaGraph.GetRootAs(bytearray(data), 0)
    main = graph.Regions(0).Blocks(0)
    tensors = {
        main.Tensors(i).Name().decode(): main.Tensors(i)
        for i in range(main.TensorsLength())
    }
    arguments = {
        spec.arg.name: value.numpy()
        for spec, value in zip(
            (
                spec
                for spec in program.graph_signature.input_specs
                if spec.kind == InputKind.USER_INPUT
            ),
            inputs,
        )
    }
    arrays = []
    for i in range(main.InputsLength()):
        name = main.Inputs(i).decode()
        tensor = tensors[name]
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        arrays.append(arguments[name].reshape(shape))
    outputs, status = reference_model.run(
        data, arrays, initialize_variable_tensor_from_numpy=True
    )
    assert status == reference_model.GraphStatus.TOSA_VALID
    assert len(outputs) == 1, status
    expected = module(*inputs).numpy()
    np.testing.assert_array_equal(outputs[0], expected.reshape(outputs[0].shape))
    return graph


class CondFlipRoll(torch.nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.nested = nested

    def forward(self, predicate, inner_predicate, value):
        def true_branch(arg):
            return torch.roll(torch.flip(arg, [0, 1]), [1, -1], [0, 1])

        def false_branch(arg):
            return torch.flip(torch.roll(arg, [-1, 1], [0, 1]), [0, 1])

        def outer_true(arg):
            return torch.cond(inner_predicate, true_branch, false_branch, [arg])

        if self.nested:
            return torch.cond(predicate, outer_true, false_branch, [value])
        return torch.cond(predicate, true_branch, false_branch, [value])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("predicate,inner_predicate", [(True, True), (True, False), (False, True)])
def test_cond_flip_roll_tosa_INT(nested, predicate, inner_predicate):
    graph = _lower_and_compare(
        CondFlipRoll(nested),
        (
            torch.tensor(predicate),
            torch.tensor(inner_predicate),
            torch.arange(6, dtype=torch.int8).reshape(2, 3),
        ),
    )
    transformed_regions = 0
    for i in range(1, graph.RegionsLength()):
        block = graph.Regions(i).Blocks(0)
        ops = [block.Operators(j).Op() for j in range(block.OperatorsLength())]
        if Op.Op.REVERSE in ops:
            assert ops.count(Op.Op.REVERSE) == 2
            assert ops.count(Op.Op.SLICE) == 4
            assert ops.count(Op.Op.CONCAT) == 2
            transformed_regions += 1
    assert transformed_regions == (3 if nested else 2)
    if nested:
        assert any(
            "__" in graph.Regions(i).Name().decode()
            for i in range(graph.RegionsLength())
        )


class WhileFlipRoll(torch.nn.Module):
    def forward(self, value, count):
        def cond_fn(arg, iteration):
            return (iteration < 2).squeeze()

        def body_fn(arg, iteration):
            return (
                torch.roll(torch.flip(arg, [0, 1]), [1, -1], [0, 1]),
                iteration + 1,
            )

        return torch.while_loop(cond_fn, body_fn, (value, count))[0]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_while_flip_roll_tosa_INT(count):
    graph = _lower_and_compare(
        WhileFlipRoll(),
        (torch.arange(6, dtype=torch.int8).reshape(2, 3), torch.tensor([count], dtype=torch.int32)),
    )
    body = next(
        graph.Regions(i).Blocks(0)
        for i in range(graph.RegionsLength())
        if "body" in graph.Regions(i).Name().decode()
    )
    ops = [body.Operators(i).Op() for i in range(body.OperatorsLength())]
    assert ops.count(Op.Op.REVERSE) == 2
    assert ops.count(Op.Op.SLICE) == 4
    assert ops.count(Op.Op.CONCAT) == 2


class WhileRank4Reduction(torch.nn.Module):
    def forward(self, value):
        def cond_fn(arg):
            return (arg.sum(dim=(0, 1, 2, 3), dtype=torch.int32) < 300).squeeze()

        def body_fn(arg):
            return (arg + 1,)

        return torch.while_loop(cond_fn, body_fn, (value,))[0]


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_while_rank4_reduction_tosa_INT(dtype):
    graph = _lower_and_compare(
        WhileRank4Reduction(),
        ((torch.arange(120) % 3).to(dtype).reshape(2, 3, 4, 5),),
    )
    condition = next(
        graph.Regions(i).Blocks(0)
        for i in range(graph.RegionsLength())
        if "cond" in graph.Regions(i).Name().decode()
    )
    assert sum(
        condition.Operators(i).Op() == Op.Op.REDUCE_SUM
        for i in range(condition.OperatorsLength())
    ) == 4


class WhileRawIntegerReduction(torch.nn.Module):
    def forward(self, value, accumulator, count):
        def cond_fn(arg, total, iteration):
            return (iteration < 2).squeeze()

        def body_fn(arg, total, iteration):
            return (
                arg + 1,
                total + arg.sum(dim=(0, 1), keepdim=True, dtype=torch.int32),
                iteration + 1,
            )

        return torch.while_loop(cond_fn, body_fn, (value, accumulator, count))[1]


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16])
@pytest.mark.parametrize("count", [0, 2])
def test_while_raw_integer_arithmetic_and_reduction_tosa_INT(dtype, count):
    limits = torch.iinfo(dtype)
    graph = _lower_and_compare(
        WhileRawIntegerReduction(),
        (
            torch.tensor([[limits.max, limits.min, 120], [-2, 3, -4]], dtype=dtype),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.tensor([count], dtype=torch.int32),
        ),
    )
    body = next(
        graph.Regions(i).Blocks(0)
        for i in range(graph.RegionsLength())
        if "body" in graph.Regions(i).Name().decode()
    )
    tensors = {
        body.Tensors(i).Name(): body.Tensors(i)
        for i in range(body.TensorsLength())
    }
    ops = [body.Operators(i) for i in range(body.OperatorsLength())]
    assert sum(op.Op() == Op.Op.REDUCE_SUM for op in ops) == 2
    for op in ops:
        if op.Op() in (Op.Op.ADD, Op.Op.REDUCE_SUM):
            assert tensors[op.Inputs(0)].Type() == DType.DType.INT32
            assert tensors[op.Outputs(0)].Type() == DType.DType.INT32
            if op.Op() == Op.Op.ADD:
                assert tensors[op.Inputs(1)].Type() == DType.DType.INT32


class RawIntegerArithmeticReduction(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return (value - 3).sum(dim=(0, 1), dtype=self.dtype)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_raw_integer_arithmetic_and_requested_sum_dtype_tosa_INT(dtype):
    _lower_and_compare(
        RawIntegerArithmeticReduction(dtype),
        (torch.tensor([[127, -128, 120], [-2, 3, -4]], dtype=torch.int8),),
    )


def test_raw_integer_sum_int64_rejected_tosa_INT():
    program = _export(
        RawIntegerArithmeticReduction(torch.int64),
        (torch.tensor([[127, -128, 120], [-2, 3, -4]], dtype=torch.int8),),
    )
    with pytest.raises(RuntimeError, match="TOSA 1.0 does not support INT64"):
        _all_rust_preprocess(program, TosaCompileSpec("TOSA-1.0+INT"))


class ConstantIndexSelect(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.int64))

    def forward(self, value):
        return torch.index_select(value, -2, self.indices)


@pytest.mark.parametrize("indices", [[1, 2, 3], [3, 1, 3], [2]])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.bool])
def test_constant_index_select_tosa_INT(indices, dtype):
    graph = _lower_and_compare(
        ConstantIndexSelect(indices),
        ((torch.arange(30).reshape(2, 5, 3) % 7 - 3).to(dtype),),
    )
    block = graph.Regions(0).Blocks(0)
    ops = [block.Operators(i).Op() for i in range(block.OperatorsLength())]
    assert Op.Op.SLICE in ops
    assert Op.Op.GATHER not in ops
    assert (Op.Op.CONCAT in ops) == (indices == [3, 1, 3])


class BoolScalarComparison(torch.nn.Module):
    """`value == scalar`, or `value != scalar`, on a bool tensor."""

    def __init__(self, scalar, negate=False):
        super().__init__()
        self.scalar = scalar
        self.negate = negate

    def forward(self, value):
        if self.negate:
            return value != self.scalar
        return value == self.scalar


def _bool_input(shape):
    return torch.arange(int(np.prod(shape))).reshape(shape) % 3 == 0


def _main_ops(graph):
    """The main block's ops without its CONSTs."""
    block = graph.Regions(0).Blocks(0)
    ops = [block.Operators(i).Op() for i in range(block.OperatorsLength())]
    return [op for op in ops if op != Op.Op.CONST]


@pytest.mark.parametrize(
    "shape,scalar", [((6,), True), ((2, 3, 4), True), ((3, 4), False), ((2, 5), 1)]
)
def test_bool_eq_scalar_lowers_as_logical_xor_then_not_tosa_INT(shape, scalar):
    """TOSA's EQUAL takes no bool inputs. As rc4's ReplaceScalarWithTensorByProfilePass
    does (APM:656), IrReplaceScalarWithTensor makes a bool eq.Scalar an eq.Tensor with a
    bool `full`, which IrDecomposeBoolEquality splits (APU-123)."""
    graph = _lower_and_compare(BoolScalarComparison(scalar), (_bool_input(shape),))
    assert _main_ops(graph) == [Op.Op.LOGICAL_XOR, Op.Op.LOGICAL_NOT]


def test_bool_ne_scalar_lowers_as_logical_xor_then_two_nots_tosa_INT():
    graph = _lower_and_compare(
        BoolScalarComparison(True, negate=True), (_bool_input((3, 4)),)
    )
    assert _main_ops(graph) == [Op.Op.LOGICAL_XOR, Op.Op.LOGICAL_NOT, Op.Op.LOGICAL_NOT]


class BoolEqualityOfConstants(torch.nn.Module):
    """`logical_and(value, lhs == rhs)` with rc4's test_eq.py `eq_bool` operands as
    bool buffers."""

    def __init__(self):
        super().__init__()
        self.register_buffer("lhs", torch.tensor([[True], [False], [True]]))
        self.register_buffer("rhs", torch.tensor([[True, False, True]]))

    def forward(self, value):
        return torch.logical_and(value, self.lhs == self.rhs)


def test_bool_eq_of_constants_folds_to_one_constant_tosa_INT():
    """rc4's FuseConstantArgs (APM:700) folds the LOGICAL_XOR and LOGICAL_NOT that a
    bool eq of two constants becomes, and so does RETAB's constant planner (APU-123)."""
    graph = _lower_and_compare(BoolEqualityOfConstants(), (_bool_input((3, 3)),))
    assert _main_ops(graph) == [Op.Op.LOGICAL_AND]
    block = graph.Regions(0).Blocks(0)
    tensors = {
        block.Tensors(i).Name(): block.Tensors(i) for i in range(block.TensorsLength())
    }
    (logical_and,) = (
        block.Operators(i)
        for i in range(block.OperatorsLength())
        if block.Operators(i).Op() == Op.Op.LOGICAL_AND
    )
    inputs = [tensors[logical_and.Inputs(i)] for i in range(logical_and.InputsLength())]
    (constant,) = (tensor for tensor in inputs if tensor.DataLength() > 0)
    assert constant.Type() == DType.DType.BOOL
    assert constant.DataAsNumpy().tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1]


class NegativePad(torch.nn.Module):
    def __init__(self, constant_pad_nd):
        super().__init__()
        self.constant_pad_nd = constant_pad_nd

    def forward(self, value):
        if self.constant_pad_nd:
            return torch.ops.aten.constant_pad_nd.default(value, [-1, 1], 0)
        return torch.ops.aten.pad.default(value, [-1, 1], "constant", 0.0)


@pytest.mark.parametrize("constant_pad_nd", [False, True])
def test_negative_pad_tosa_INT(constant_pad_nd):
    graph = _lower_and_compare(
        NegativePad(constant_pad_nd),
        (torch.arange(16, dtype=torch.int8).reshape(1, 1, 4, 4),),
    )
    block = graph.Regions(0).Blocks(0)
    ops = [block.Operators(i) for i in range(block.OperatorsLength())]
    assert sum(op.Op() == Op.Op.SLICE for op in ops) == 1
    pad = next(op for op in ops if op.Op() == Op.Op.PAD)
    shape = next(
        block.Shapes(i)
        for i in range(block.ShapesLength())
        if block.Shapes(i).Name() == pad.Inputs(1)
    )
    assert np.all(np.frombuffer(shape.DataAsNumpy().tobytes(), dtype=np.int64) >= 0)


class ArgmaxConsumer(torch.nn.Module):
    def __init__(self, reshape):
        super().__init__()
        self.reshape = reshape

    def forward(self, value, scalar):
        result = torch.argmax(value, dim=0).to(torch.int32)
        if self.reshape:
            return result.reshape(1, 1)
        return result + scalar


@pytest.mark.parametrize("reshape", [False, True])
def test_argmax_consumer_tosa_INT(reshape):
    graph = _lower_and_compare(
        ArgmaxConsumer(reshape),
        (torch.tensor([0, 5, 2, 3, 4], dtype=torch.int8), torch.tensor(7, dtype=torch.int32)),
    )
    block = graph.Regions(0).Blocks(0)
    tensors = {
        block.Tensors(i).Name(): block.Tensors(i)
        for i in range(block.TensorsLength())
    }
    ops = [block.Operators(i) for i in range(block.OperatorsLength())]
    # The rank-0 argmax result adds to the rank-0 scalar input without a RESHAPE.
    assert any(op.Op() == Op.Op.RESHAPE for op in ops) == reshape
    for op in ops:
        if op.Op() == Op.Op.ADD:
            assert tensors[op.Inputs(0)].ShapeLength() == tensors[op.Inputs(1)].ShapeLength()


class ArgmaxWithoutDim(torch.nn.Module):
    def forward(self, value):
        return torch.argmax(value).to(torch.int32)


def test_argmax_without_dim_tosa_INT():
    program = _export(
        ArgmaxWithoutDim(), (torch.arange(6, dtype=torch.int8).reshape(2, 3),)
    )
    with pytest.raises(RuntimeError, match="argmax.*explicit dim"):
        _all_rust_preprocess(program, TosaCompileSpec("TOSA-1.0+INT"))


def _main_block_ops(tosa: bytes) -> list[int]:
    block = TosaGraph.TosaGraph.GetRootAs(bytearray(tosa), 0).Regions(0).Blocks(0)
    return [block.Operators(i).Op() for i in range(block.OperatorsLength())]


def test_conv_becomes_a_channels_last_conv2d_tosa_INT():
    """RETAB builds rc4's pipeline: the conv becomes a channels-last TOSA CONV2D on a
    pre-permuted weight between explicit TRANSPOSEs, so nothing reaches the serializer's
    aten-conv guard (P2 design X5) and the graph stays with RETAB."""
    from retab import _native
    from retab import preprocess as rp

    # Frozen parameters, as a lowered program has them: the importer reads the weight's
    # data, which IrRewriteConv pre-permutes.
    module = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.ReLU()).requires_grad_(False)
    program = torch.export.export(module, (torch.ones(1, 2, 8, 8),)).run_decompositions()
    config = rp._build_pipeline_config(TosaCompileSpec("TOSA-1.0+INT"))
    _native.run_fx_passes(program.graph_module, program, pipeline_config=config)
    ops = _main_block_ops(_native.ingest_and_serialize())
    assert ops.count(Op.Op.CONV2D) == 1
    # The input permute to NHWC and the output permute back; the weight is a constant.
    assert ops.count(Op.Op.TRANSPOSE) == 2
    conv = ops.index(Op.Op.CONV2D)
    assert ops.index(Op.Op.TRANSPOSE) < conv < len(ops) - 1 - ops[::-1].index(Op.Op.TRANSPOSE)


class CopyOverUnusedInput(torch.nn.Module):
    def forward(self, x, y):
        return y.clone().copy_(x) + x


def test_unused_input_is_rejected_only_for_ethos_u():
    import retab.preprocess as rp

    inputs = (torch.ones(1, 2, 3, 4, dtype=torch.int8), torch.zeros(1, 2, 3, 4, dtype=torch.int8))
    result = _all_rust_preprocess(_export(CopyOverUnusedInput(), inputs), TosaCompileSpec("TOSA-1.0+INT"))
    block = TosaGraph.TosaGraph.GetRootAs(bytearray(bytes(result.processed_bytes)), 0).Regions(0).Blocks(0)
    assert [block.Inputs(i).decode() for i in range(block.InputsLength())] == ["x", "y"]

    rp._ethos_u_context.depth = 1
    try:
        with pytest.raises(RuntimeError, match=r"Unused: \['y'\]"):
            _all_rust_preprocess(_export(CopyOverUnusedInput(), inputs), TosaCompileSpec("TOSA-1.0+INT"))
    finally:
        rp._ethos_u_context.depth = 0


def test_ethos_u_preprocess_marker_is_installed_and_restored():
    from executorch.backends.arm.ethosu.backend import EthosUBackend
    from retab.integration.executorch_arm import activated
    import retab.preprocess as rp

    stock = EthosUBackend.__dict__["preprocess"]
    with activated():
        wrapped = EthosUBackend.__dict__["preprocess"]
        assert wrapped is not stock and wrapped.__func__.__name__ == "preprocess"
    assert EthosUBackend.__dict__["preprocess"] is stock
    assert getattr(rp._ethos_u_context, "depth", 0) == 0
