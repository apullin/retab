"""Record the graphs rc4's RemovePermutesAroundElementwiseOps tests hand the
pass, for src/layout/remove_permutes_upstream_tests.rs.

Runs rc4's two test files for the pass,

    backends/arm/test/passes/test_remove_permutes_around_elementwise_tosa_ops.py
    backends/transforms/test/test_permute_optimization_passes.py (its
        RemovePermutes*/LayoutPermuteVisibility tests)

under pytest with `RemovePermutesAroundElementwiseOps.call` instrumented: every
call becomes a case named `<test>#<n>` (n counts the test's calls) in the
layout replay format (see tests/fixtures/layout_replay/gen_layout_replay.py,
whose import also pins rc4's address-ordered edge sets to graph order, as the
replay fixture does). The TOSA subclass's graph-signature constants are marked
`kind=param`. One Arm test builds call nodes without vals; those are printed
with the val fake execution gives them, which does not change what the pass
does there.

The source tree is rc4 with the release/1.6 backports of #24315 and of
#24469's mixed-rank broadcast guard, and the guard's TABLE exemption, which
add six tests and the fixes RETAB's port follows (see
tests/fixtures/layout_replay/gen_layout_replay.py's docstring). The pass must
come from the tree whose tests run, so put its `src` on PYTHONPATH. Usage,
with the rc4 environment:

    cd /tmp && PYTHONPATH=<checkout>/src <venv>/bin/python -P <retab>/tests/fixtures/remove_permutes_upstream/record_rc4_test_calls.py <checkout>
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    "backends/arm/test/passes/test_remove_permutes_around_elementwise_tosa_ops.py",
    "backends/transforms/test/test_permute_optimization_passes.py",
]
SELECT = "RemovePermutes or remove_permutes or LayoutPermute or constant_input_cache or extra_permutable"

CONFTEST = '''
import sys
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, {layout_replay!r})
import gen_layout_replay as G  # noqa: E402  pins the edge sets

from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (  # noqa: E402
    RemovePermutesAroundElementwiseOps as R,
)

_PASS_FILE = Path(sys.modules[R.__module__].__file__).resolve()
assert _PASS_FILE.is_relative_to(Path({source!r}).resolve()), (
    f"the pass comes from {{_PASS_FILE}}; put the tree whose tests run on PYTHONPATH"
)

RECORDS = []
STATE = {{"test": None, "calls": 0}}
ORIGINAL_CALL = R.call


def _dump(gm, kinds):
    missing = []
    vals = {{}}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        val = node.meta.get("val")
        if val is None and node.op == "call_function":
            args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), lambda n: vals[n])
            with FakeTensorMode(allow_non_fake_inputs=True):
                val = node.target(*args, **kwargs)
            node.meta["val"] = val
            missing.append(node)
        vals[node] = val
    try:
        return G.dump(gm, kinds)
    finally:
        for node in missing:
            del node.meta["val"]


def recording_call(self, graph_module):
    kinds = {{name: "param" for name in getattr(self, "_constant_input_names", set())}}
    before = _dump(graph_module, kinds)
    name = f"{{STATE['test']}}#{{STATE['calls']}}"
    STATE["calls"] += 1
    try:
        result = ORIGINAL_CALL(self, graph_module)
    except Exception as error:
        RECORDS.append([f"case {{name}}", "before", *before, *G.raised(error), "end"])
        raise
    after = _dump(result.graph_module, kinds)
    lines = [f"case {{name}}", "before", *before, f"after modified={{str(bool(result.modified)).lower()}}"]
    if after == before:
        lines[-1] += " same"
    else:
        lines += after
    lines.append("end")
    RECORDS.append(lines)
    return result


R.call = recording_call


@pytest.fixture(autouse=True)
def _name_calls(request):
    STATE["test"] = request.node.name
    STATE["calls"] = 0
    yield


def pytest_sessionfinish(session, exitstatus):
    with open({out!r}, "w") as f:
        for lines in RECORDS:
            f.write("\\n".join(lines) + "\\n")
'''


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    out = HERE / "rc4_test_calls.txt"
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for test in TESTS:
            shutil.copy(source / test, work / Path(test).name)
        (work / "conftest.py").write_text(
            CONFTEST.format(
                layout_replay=str(HERE.parent / "layout_replay"),
                out=str(work / "calls.txt"),
                source=str(source),
            )
        )
        command = [
            sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--rootdir", str(work), *[Path(t).name for t in TESTS], "-k", SELECT,
        ]
        subprocess.run(command, cwd=work, check=True)
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        header = [
            "# The graphs rc4's RemovePermutesAroundElementwiseOps tests hand the pass, and what it returns.",
            "# Recorded by tests/fixtures/remove_permutes_upstream/record_rc4_test_calls.py; don't edit by hand.",
            f"# executorch sources: git {commit or 'unknown'}",
        ]
        out.write_text("\n".join(header) + "\n" + (work / "calls.txt").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
