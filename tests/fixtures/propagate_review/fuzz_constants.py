"""Random propagation graphs whose second operands are often constants
(parameters, buffers, lifted constants), which RemovePermutesAroundElementwiseTosaOps
treats differently (the committed random graphs have none). Records the
Down pass, the Up pass and Down then Up. Usage:
`fuzz_constants.py down|up|down_up <count> <seed>`."""
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402

G = P.G


class ConstGraph(G.PropagationGraph):
    def partner(self, x):
        rng = self.rng
        if rng.random() < 0.55:
            shape = list(x.meta["val"].shape)
            roll = rng.random()
            if roll < 0.45:
                other = shape
            elif roll < 0.65:
                other = [1] * len(shape)
            elif roll < 0.85:
                other = [1 if rng.random() < 0.5 else d for d in shape]
            else:
                keep = rng.randint(0, len(shape))
                other = [1 if rng.random() < 0.4 else d for d in shape[len(shape) - keep:]]
            name = f"c{self.inputs}"
            self.inputs += 1
            kind = rng.choice(["param", "buffer", "constant"])
            node = self.b.constant(kind, name, other, x.meta["val"].dtype)
            return self.keep(node)
        return super().partner(x)


def build_case(seed):
    def build(b):
        rng = random.Random(seed)
        graph = ConstGraph(rng, b)
        for _ in range(rng.randint(1, 3)):
            graph.fresh_float()
        for _ in range(rng.randint(4, 24)):
            graph.step()
        graph.finish()

    return build


which, count, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cases = {f"const_{i}": build_case(seed * 100_000 + i) for i in range(count)}
P.emit(cases, which, P.out_path(which))
