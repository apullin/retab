"""rc4 ground truth for the generated swap cases, run through the whole pass.

view_permute: for each input shape I, view shape S (factorization of numel(I), rank <= 4)
and permutation P of S's rank:
  pair:  x -> view(S) -> permute(P)
  chain: x -> view(S) -> permute(P) -> flat = view([numel])
permute_view: for each I, permutation P of I's rank, output shape T:
  pair:  x -> permute(P) -> view(T)
  chain: x -> permute(P) -> view(T) -> rev = permute(reversed range(len(T)))
A line is "<form><I><S|P><P|T>: " then the surviving calls in graph order as
"<label> = <kind>(<input label>, <int list>) -> <shape>" and "output(<labels>)",
joined by "; ". The digest is FNV-1a 64 over every line followed by a newline.

Regenerate the digests that src/layout/canonicalize_upstream_tests.rs checks,
from a neutral directory with the rc4 venv:
  cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/canonicalize_swap_digests/gen_swap_digests.py view_permute
  cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/canonicalize_swap_digests/gen_swap_digests.py permute_view
Each prints the line count, the modified count and the digest; the rendered
lines go to gen_<which>.txt in the current directory.
"""

import sys
import time
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_lib import run  # noqa: E402

INPUT_SHAPES = [[2, 3], [2, 4], [2, 3, 4], [2, 2, 3], [1, 2, 3], [2, 1, 3], [2, 3, 1], [2, 2, 2, 3]]


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def factorizations(n, max_rank):
    shapes = []

    def rec(rem, rank, shape):
        if rank == 0:
            if rem == 1:
                shapes.append(list(shape))
            return
        for d in range(1, rem + 1):
            if rem % d == 0:
                shape.append(d)
                rec(rem // d, rank - 1, shape)
                shape.pop()

    for r in range(1, max_rank + 1):
        rec(n, r, [])
    return shapes


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def line(form, a, b, c, modified, survivors, out):
    rows = [f"{label} = {kind}({src}, {arg}) -> {shape}" for label, kind, src, arg, shape in survivors]
    rows.append(f"output({', '.join(out)})")
    return f"{form}{a}{b}{c}: {'; '.join(rows)}", modified


def view_permute_cases():
    for i in INPUT_SHAPES:
        for s in factorizations(numel(i), 4):
            for p in permutations(range(len(s))):
                yield i, s, list(p)


def permute_view_cases():
    for i in INPUT_SHAPES:
        outputs = factorizations(numel(i), 4)
        for p in permutations(range(len(i))):
            for t in outputs:
                yield i, list(p), t


def main(which):
    lines = []
    modified_count = 0
    start = time.time()
    if which == "view_permute":
        for i, s, p in view_permute_cases():
            for form in ("pair", "chain"):
                ops = [("view", "view", "x", s), ("permute", "permute", "view", p)]
                out = ["permute"]
                if form == "chain":
                    ops.append(("flat", "view", "permute", [numel(i)]))
                    out = ["flat"]
                ln, m = line(form, i, s, p, *run([("x", i)], ops, out))
                lines.append(ln)
                modified_count += m
    else:
        for i, p, t in permute_view_cases():
            for form in ("pair", "chain"):
                ops = [("permute", "permute", "x", p), ("view", "view", "permute", t)]
                out = ["view"]
                if form == "chain":
                    ops.append(("rev", "permute", "view", list(range(len(t)))[::-1]))
                    out = ["rev"]
                ln, m = line(form, i, p, t, *run([("x", i)], ops, out))
                lines.append(ln)
                modified_count += m
    blob = "".join(ln + "\n" for ln in lines).encode()
    with open(f"gen_{which}.txt", "w") as f:
        f.write(blob.decode())
    print(f"{which}: lines={len(lines)} modified={modified_count} digest={fnv1a(blob):#018x} "
          f"({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
