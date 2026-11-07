"""The committed generator's random propagation graphs, recorded as Down then
Up (the per-pass fixtures only ever give the Up pass an original graph).
Usage: `fuzz_down_up.py <count> <seed>`; replay with propagate_review_external."""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402

G = P.G
count, seed = int(sys.argv[1]), int(sys.argv[2])
cases = {f"random_{i}": G.propagation_random_case(seed * 100_000 + i) for i in range(count)}
P.emit(cases, "down_up", P.out_path("down_up"))
