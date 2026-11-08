"""A panic in RETAB's Rust code sends the graph to stock preprocess.

PyO3 raises a panic as PanicException, a BaseException that the seam's `except Exception`
would let through to the user. So each Rust entry point turns a panic into a
`RETAB_STOCK_FALLBACK: Rust panic: ...` error (src/stock_fallback.rs: catch_panic), unless
RETAB_DEBUG is set. RETAB_TEST_PANIC names an entry point that panics on entry.
"""

import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("executorch.backends.arm.tosa.compile_spec")
_native = pytest.importorskip("retab._native")

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from retab import preprocess as rp

pytestmark = [pytest.mark.native, pytest.mark.executorch]

# Each in-process test needs panics caught, which RETAB_DEBUG turns off for the process.
catching = pytest.mark.skipif(
    "RETAB_DEBUG" in os.environ, reason="RETAB_DEBUG lets Rust panics propagate"
)

# The order in which _all_rust_preprocess calls the entry points, for a graph with a cond
# region and then a while region, lowered for Ethos-U.
CALL_ORDER = [
    "run_fx_passes",
    "run_fx_cond_passes",
    "run_fx_submodule_passes",
    "ingest_and_serialize",
    "last_unused_inputs",
]


def test_python_calls_every_entry_point():
    """CALL_ORDER names every entry point, so the fallback tests below cover a new one."""
    assert sorted(CALL_ORDER) == sorted(_native.__all__)


def _call_with_placeholder_args(name):
    """Call an entry point with arguments PyO3 accepts. It panics before reading them."""
    args, kwargs = [], {}
    for param in inspect.signature(getattr(_native, name)).parameters.values():
        if param.default is not param.empty:
            continue
        if param.kind is param.KEYWORD_ONLY:
            kwargs[param.name] = {}
        else:
            args.append(None)
    return getattr(_native, name)(*args, **kwargs)


@catching
@pytest.mark.parametrize("entry_point", sorted(_native.__all__))
def test_a_panic_in_an_entry_point_raises_a_stock_fallback_error(monkeypatch, entry_point):
    monkeypatch.setenv("RETAB_TEST_PANIC", entry_point)

    with pytest.raises(RuntimeError) as raised:
        _call_with_placeholder_args(entry_point)

    assert str(raised.value).startswith(
        f"RETAB_STOCK_FALLBACK: Rust panic: RETAB_TEST_PANIC={entry_point} (at src/lib.rs:"
    )
    assert rp._native_fallback_reason(raised.value) == "Rust panic"


@catching
@pytest.mark.parametrize("entry_point", CALL_ORDER)
def test_a_rust_panic_lowers_the_graph_with_stock(monkeypatch, caplog, entry_point):
    """Whichever entry point panics, the delegate gets stock's result, the fallback counts
    under "Rust panic", and a warning gives the panic's message and location."""
    from executorch.exir import graph_module

    program = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    cond_branch, loop_body = (torch.fx.symbolic_trace(torch.nn.ReLU()) for _ in range(2))
    regions = {
        program.graph_module: [
            ("true_graph", cond_branch, SimpleNamespace(target="cond")),
            ("body_graph", loop_body, SimpleNamespace(target="while_loop")),
        ],
        cond_branch: [],
        loop_body: [],
    }
    monkeypatch.setattr(graph_module, "get_cond_while_submodules", lambda gm: regions[gm])
    monkeypatch.setattr(rp, "stock_fallback_reason", lambda ep, spec: None)
    # The entry points called before the one that panics succeed without lowering.
    stubs = {"ingest_and_serialize": lambda: b"tosa", "last_unused_inputs": lambda: []}
    for name in CALL_ORDER[: CALL_ORDER.index(entry_point)]:
        monkeypatch.setattr(_native, name, stubs.get(name, lambda *args, **kwargs: False))
    # Inside EthosUBackend.preprocess, which asks for the unused inputs.
    monkeypatch.setattr(rp._ethos_u_context, "depth", 1, raising=False)
    stock_result = object()
    monkeypatch.setattr(rp, "stock_preprocess", lambda ep, spec: stock_result)
    monkeypatch.setenv("RETAB_TEST_PANIC", entry_point)
    caplog.set_level(logging.INFO, logger=rp.__name__)
    before = rp.preprocess_stats()

    assert rp._all_rust_preprocess(program, TosaCompileSpec("TOSA-1.0+INT")) is stock_result

    after = rp.preprocess_stats()
    assert after["retab"] == before["retab"]
    panics = after["stock_fallback"].get("Rust panic", 0)
    assert panics == before["stock_fallback"].get("Rust panic", 0) + 1
    [record] = [
        r for r in caplog.records if r.name == rp.__name__ and "fallback" in r.getMessage()
    ]
    assert record.levelno == logging.WARNING
    assert f"Rust panic: RETAB_TEST_PANIC={entry_point} (at src/lib.rs:" in record.getMessage()


class _SelectRows(torch.nn.Module):
    """Selects rows 1 and 5 of a five-row int32 input, so row 5 is out of range."""

    def __init__(self):
        super().__init__()
        self.register_buffer("rows", torch.tensor([1, 5], dtype=torch.int32))

    def forward(self, x):
        return torch.index_select(x, 0, self.rows) + 1


@catching
def test_a_graph_retab_panics_on_gets_stocks_own_error():
    """IrDecomposeIndexSelect panics on an out-of-range constant index, where rc4's
    DecomposeIndexSelectToGatherPass raises. The delegate goes to stock, so the user gets
    stock's error, an ordinary Exception, where v0.4.2 raised a PanicException."""
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from retab.integration.executorch_arm import activated

    program = torch.export.export(_SelectRows(), (torch.zeros(5, 3, dtype=torch.int32),))
    before = rp.preprocess_stats()

    with activated(), pytest.raises(Exception) as raised:
        to_edge_transform_and_lower(
            program,
            partitioner=[TOSAPartitioner(TosaCompileSpec("TOSA-1.0+INT"))],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

    after = rp.preprocess_stats()
    assert "DecomposeIndexSelectToGatherPass" in str(raised.value)
    assert after["retab"] == before["retab"]
    assert sum(after["stock_fallback"].values()) == sum(before["stock_fallback"].values()) + 1


def test_retab_debug_lets_the_panic_propagate():
    """With RETAB_DEBUG set, a panic reaches Python as PyO3's PanicException, as before."""
    script = (
        "from retab import _native\n"
        "try:\n"
        "    _native.ingest_and_serialize()\n"
        "except BaseException as exc:\n"
        "    print(type(exc).__name__, exc)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(_native.__file__).parents[1]),
        "RETAB_DEBUG": "1",
        "RETAB_TEST_PANIC": "ingest_and_serialize",
    }
    done = subprocess.run(
        [sys.executable, "-P", "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.stdout.strip() == "PanicException RETAB_TEST_PANIC=ingest_and_serialize", done.stderr
