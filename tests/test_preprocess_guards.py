"""Guards in RETAB's preprocess seam: per-call parallel job lists and the symbolic-shape fallback."""

import types

import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.executorch]


class _ProgramTagBackend:
    """Stands in for an Arm backend: 'lowers' a program by returning its tag."""

    @staticmethod
    def preprocess(edge_program, compile_specs):
        return edge_program


class _ForkSnapshotPool:
    """In-process Pool that mimics fork semantics: each task sees the state the pool's
    initializer set up, even if another call ran in between (as it would in a real child)."""

    nested_call = None

    def __init__(self, processes, initializer=None, initargs=()):
        self.initializer, self.initargs = initializer, initargs
        if _ForkSnapshotPool.nested_call is not None:
            call, _ForkSnapshotPool.nested_call = _ForkSnapshotPool.nested_call, None
            call()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, tasks):
        results = []
        for task in tasks:
            if self.initializer is not None:
                self.initializer(*self.initargs)
            results.append(fn(task))
        return results


def test_overlapping_parallel_calls_each_lower_their_own_programs(monkeypatch):
    import multiprocessing

    import retab.preprocess as rp

    monkeypatch.setenv("RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS", "2")
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: types.SimpleNamespace(Pool=_ForkSnapshotPool)
    )
    spec = types.SimpleNamespace(_get_intermediate_path=lambda: None)
    multimethod = rp._make_parallel_backend_preprocess_multimethod(lambda specs: spec).__get__(
        None, _ProgramTagBackend
    )
    specs = {"forward": [[], []]}
    results = {}
    # Call B runs after call A has built its job list but before A's pool forks.
    _ForkSnapshotPool.nested_call = lambda: results.setdefault(
        "b", multimethod({"forward": ["b0", "b1"]}, specs)
    )
    results["a"] = multimethod({"forward": ["a0", "a1"]}, specs)

    assert results["a"] == {"forward": ["a0", "a1"]}
    assert results["b"] == {"forward": ["b0", "b1"]}


def _symbolic_dim():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return ShapeEnv().create_unbacked_symint()


def _graph_module(meta_vals):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = meta_vals[0]
    out = graph.call_function(torch.ops.aten.relu.default, (x,))
    out.meta["val"] = meta_vals[1]
    graph.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_symbolic_shapes_are_found_in_tuple_metadata():
    from retab.preprocess import _has_symbolic_shapes

    static = types.SimpleNamespace(shape=(2, 3))
    symbolic = types.SimpleNamespace(shape=(_symbolic_dim(), 3))
    assert not _has_symbolic_shapes(_graph_module([static, static]))
    assert _has_symbolic_shapes(_graph_module([static, (static, symbolic)]))


def test_symbolic_shapes_are_found_inside_cond_branches():
    from retab.preprocess import _has_symbolic_shapes

    static = types.SimpleNamespace(shape=(2, 3))
    branch = _graph_module([static, types.SimpleNamespace(shape=(_symbolic_dim(), 3))])
    other = _graph_module([static, static])

    root = torch.nn.Module()
    root.true_graph_0, root.false_graph_0 = branch, other
    graph = torch.fx.Graph()
    pred, x = graph.placeholder("pred"), graph.placeholder("x")
    pred.meta["val"], x.meta["val"] = types.SimpleNamespace(shape=()), static
    true_gm, false_gm = graph.get_attr("true_graph_0"), graph.get_attr("false_graph_0")
    cond = graph.call_function(torch.ops.higher_order.cond, (pred, true_gm, false_gm, (x,)))
    cond.meta["val"] = (static,)
    graph.output(cond)

    assert _has_symbolic_shapes(torch.fx.GraphModule(root, graph))


def _qdq_graph(compute_user: bool):
    from executorch.exir.dialects._ops import ops as exir_ops

    qd = exir_ops.edge.quantized_decomposed
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    dq = graph.call_function(qd.dequantize_per_tensor.default, (x, 0.5, 0, -(2**31), 2**31 - 1, torch.int32))
    if compute_user:
        out = graph.call_function(exir_ops.edge.aten.add.Tensor, (dq, dq))
    else:
        out = graph.call_function(qd.quantize_per_tensor.default, (dq, 0.25, 0, -128, 127, torch.int8))
    graph.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _int_only_spec():
    from retab.preprocess import SDPASafeSoftmaxGuardPolicy

    version = types.SimpleNamespace(major=1, minor=0)
    tosa_spec = types.SimpleNamespace(support_float=lambda: False, version=version)
    pipeline_config = types.SimpleNamespace(sdpa_safe_softmax_guard=SDPASafeSoftmaxGuardPolicy.PRESERVE)
    return types.SimpleNamespace(
        tosa_spec=tosa_spec,
        tosa_dev_mode=False,
        _pipeline_config=None,
        _get_pass_pipeline_config=lambda: pipeline_config,
    )


def _edge_program(graph_module):
    signature = types.SimpleNamespace(buffers_to_mutate={})
    return types.SimpleNamespace(graph_module=graph_module, graph_signature=signature)


@pytest.mark.parametrize("layout", ["dim_order", "explicit"])
def test_dynamic_shape_export_goes_to_stock_in_every_layout_mode(monkeypatch, layout):
    """A batch/H/W-dynamic upsample, as in rc4's tosa_INT_dynamic_* op tests, stays on stock.

    RETAB's IR is static. Without this fallback it lowers the graph with the example
    input's sizes baked in: INT8 [2, 4, 8, 3] where stock emits [-1, 4, -1, -1]. The op
    tests cannot see that, because they skip the output comparison for these cases.
    """
    import retab.preprocess as rp

    batch = torch.export.Dim("batch", min=1, max=2)
    height = torch.export.Dim("height", min=1, max=8)
    width = torch.export.Dim("width", min=1, max=8)
    program = torch.export.export(
        torch.nn.Upsample(scale_factor=2.0, mode="nearest"),
        (torch.rand(2, 4, 8, 3),),
        dynamic_shapes=({0: batch, 2: height, 3: width},),
    )
    stock_result = object()
    monkeypatch.setenv("RETAB_LAYOUT", layout)
    monkeypatch.setattr(rp, "stock_preprocess", lambda edge_program, compile_spec: stock_result)
    before = rp._stock_fallback_counts["symbolic shapes"]

    assert rp.stock_fallback_reason(program, _int_only_spec()) == "symbolic shapes"
    assert rp._all_rust_preprocess(program, _int_only_spec()) is stock_result
    assert rp._stock_fallback_counts["symbolic shapes"] == before + 1


def test_int32_quantized_compute_lowers_without_fallback():
    import retab.preprocess as rp

    assert not hasattr(rp, "_has_int32_quantized_compute")
    # Both int32-domain compute and plain int32 requantize lower natively
    # (RESCALE-based, no float tensors), so neither falls back to stock.
    assert (
        rp.stock_fallback_reason(
            _edge_program(_qdq_graph(compute_user=True)), _int_only_spec()
        )
        is None
    )
    assert (
        rp.stock_fallback_reason(
            _edge_program(_qdq_graph(compute_user=False)), _int_only_spec()
        )
        is None
    )


def test_native_fallback_reason_parses_rust_requests():
    from retab.preprocess import _native_fallback_reason

    table = RuntimeError(
        "RETAB_STOCK_FALLBACK: unsupported table op: sigmoid_1 would emit integer TOSA SIGMOID"
    )
    assert _native_fallback_reason(table) == "unsupported table op"
    dim_order = RuntimeError(
        "RETAB_STOCK_FALLBACK: quantized _to_dim_order_copy without unit qparams: copy_2"
    )
    assert _native_fallback_reason(dim_order) == "quantized _to_dim_order_copy without unit qparams"
    assert _native_fallback_reason(RuntimeError("index out of range")) is None
