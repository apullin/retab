"""Record the graphs rc4's propagation tests hand PropagateViewCopyPermute*Pass,
and what each call returns, for src/layout/propagate_upstream_tests.rs.

Runs rc4's test files for the passes,

    backends/transforms/test/test_propagate_view_copy_permute_pass.py
    backends/arm/test/passes/test_propagate_permutes_views_pass.py
    backends/arm/test/passes/test_rank_match_after_propagation.py

copied next to the installed Arm test conftest (so its TEST_SEED fixture
applies and imports come from the installed ExecuTorch), under pytest with
`PropagateViewCopyPermutePass.call` instrumented: every call becomes a case
in the layout replay format (see tests/fixtures/layout_replay/gen_layout_replay.py,
whose import also pins RemovePermutes' address-ordered edge sets to graph
order, as the replay fixtures do). A case is named
`<test>#<n>|<config>`, where `n` counts the test's calls and `config` says
how the pass was built:

    arm.down / arm.up        Arm's pass (TosaPropagationOverrides)
    shared.down / shared.up  the shared pass in backends/transforms
    +ep                      constructed with an exported program
    ~<Class>                 a test subclass that overrides blocks_moving

Placeholders the exported program lists as parameters, buffers or lifted
constants are marked `kind=`.

Usage, with the rc4 environment and the rc4 source tree:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_upstream/record_rc4_test_calls.py <executorch-1.6>
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    "backends/transforms/test/test_propagate_view_copy_permute_pass.py",
    "backends/arm/test/passes/test_propagate_permutes_views_pass.py",
    "backends/arm/test/passes/test_rank_match_after_propagation.py",
]

PLUGIN = '''
import sys

import pytest

sys.path.insert(0, {layout_replay!r})
import gen_layout_replay as G  # noqa: E402  pins the edge sets

from executorch.backends.transforms import propagate_view_copy_permute_pass as shared  # noqa: E402
from executorch.backends.arm._passes import propagate_view_copy_permute_pass as arm  # noqa: E402

RECORDS = []
STATE = {{"test": None, "calls": 0}}
ORIGINAL_CALL = shared.PropagateViewCopyPermutePass.call


def _config(self):
    direction = "up" if isinstance(self, shared.PropagateViewCopyPermuteUpPass) else "down"
    family = "arm" if isinstance(self, arm.TosaPropagationOverrides) else "shared"
    config = f"{{family}}.{{direction}}"
    if self.exported_program is not None:
        config += "+ep"
    cls = type(self)
    if cls.__module__ not in (shared.__name__, arm.__name__) and "blocks_moving" in cls.__dict__:
        config += f"~{{cls.__name__}}"
    return config


def recording_call(self, graph_module):
    kinds = G.placeholder_kinds(self.exported_program) if self.exported_program is not None else {{}}
    try:
        before = G.dump(graph_module, kinds)
    except Exception as error:  # noqa: BLE001  a graph the format can't print
        before = None
        print(f"record: cannot dump {{STATE['test']}}: {{error}}", file=sys.stderr)
    name = f"{{STATE['test']}}#{{STATE['calls']}}|{{_config(self)}}"
    STATE["calls"] += 1
    try:
        result = ORIGINAL_CALL(self, graph_module)
    except Exception as error:
        if before is not None:
            message = " ".join(str(error).split())[:300]
            RECORDS.append([f"case {{name}}", "before", *before, f"after raises={{type(error).__name__}}", f"# rc4: {{message}}", "end"])
        raise
    if before is not None:
        after = G.dump(result.graph_module, kinds)
        lines = [f"case {{name}}", "before", *before, f"after modified={{str(bool(result.modified)).lower()}}"]
        if after == before:
            lines[-1] += " same"
        else:
            lines += after
        lines.append("end")
        RECORDS.append(lines)
    return result


shared.PropagateViewCopyPermutePass.call = recording_call


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
    import os
    import shutil

    import executorch.backends.arm.test as arm_test

    source = Path(sys.argv[1]).resolve()
    out = HERE / "rc4_test_calls.txt"
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for test in TESTS:
            shutil.copy(source / test, work / Path(test).name)
        shutil.copy(Path(list(arm_test.__path__)[0]) / "conftest.py", work / "conftest.py")
        plugin = work / "record_propagation_calls.py"
        plugin.write_text(
            PLUGIN.format(layout_replay=str(HERE.parent / "layout_replay"), out=str(work / "calls.txt"))
        )
        command = [
            sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-p", "record_propagation_calls", "-rA", "--rootdir", str(work),
            *[Path(test).name for test in TESTS],
        ]
        env = {**os.environ, "PYTHONPATH": str(work)}
        completed = subprocess.run(command, cwd=work, env=env)
        print(f"pytest exited {completed.returncode}")
        header = [
            "# The graphs rc4's propagation tests hand PropagateViewCopyPermute*Pass, and what it returns.",
            "# Recorded by tests/fixtures/propagate_upstream/record_rc4_test_calls.py; don't edit by hand.",
        ]
        out.write_text("\n".join(header) + "\n" + (work / "calls.txt").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
