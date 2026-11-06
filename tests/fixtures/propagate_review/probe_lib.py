"""Shared helpers for the propagation review probes (src/layout/propagate_review_tests.rs).

They run rc4's Arm propagation passes, built as arm_pass_manager.py:726-727
builds them (with the exported program), on graphs made with the replay
generator's builder: the Down pass, the Up pass, or both in sequence
(the generator's `PropagationPass`), and write fixture-format cases to `$RETAB_REVIEW_DIR`
(default /tmp/retab_review) as down.txt, up.txt and down_up.txt, which
`propagate_review_external` and `propagate_review_order_only` replay.

Run with the rc4 environment from a neutral directory, for example:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_review/fuzz_t16.py down_up 2000 5
"""
import logging
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "layout_replay"))
import gen_layout_replay as G  # noqa: E402

from executorch.backends.arm._passes import (  # noqa: E402
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)

logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)


SPECS = {
    "down": G.PassSpec("down", lambda ep: G.PropagationPass(ep, PropagateViewCopyPermuteDownPass), [], None, None, tosa_spec=G.PROPAGATE_TOSA_SPEC),
    "up": G.PassSpec("up", lambda ep: G.PropagationPass(ep, PropagateViewCopyPermuteUpPass), [], None, None, tosa_spec=G.PROPAGATE_TOSA_SPEC),
    "down_up": G.PassSpec(
        "down_up",
        lambda ep: G.PropagationPass(ep, PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass),
        [],
        None,
        None,
        tosa_spec=G.PROPAGATE_TOSA_SPEC,
    ),
}


def out_path(which):
    directory = os.environ.get("RETAB_REVIEW_DIR", "/tmp/retab_review")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{which}.txt")


def emit(cases, which, out):
    lines = []
    for name, build in cases.items():
        try:
            body = G.run_case(SPECS[which], build)
        except G.BuildError as error:
            print(f"{which}/{name}: rc4 result not representable: {error}", file=sys.stderr)
            continue
        lines += [f"case {name}", *body, "end"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {sum(1 for l in lines if l.startswith('case '))} cases", file=sys.stderr)
