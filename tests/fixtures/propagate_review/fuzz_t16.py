"""Random propagation graphs weighted towards 16-bit TABLE ops, recorded per
pass and as Down then Up. Usage: `fuzz_t16.py down|up|down_up <count> <seed>`.
Where rc4 keeps a TABLE apart from its RESCALE, the fixture holds RETAB's
split form (`layout::pending`)."""
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402

G = P.G
A = G.A

T16_QPARAMS = [
    (G.qargs(1 / 1024, dtype=torch.int16), G.qargs(1 / 32768, dtype=torch.int16)),
    (G.qargs(1 / 256, dtype=torch.int16), G.qargs(1 / 4096, dtype=torch.int16)),
    (G.qargs(1 / 2048, dtype=torch.int16), G.qargs(1 / 2048, dtype=torch.int16)),
]
T16_TARGETS = [(A.sigmoid.default, (), {}), (A.tanh.default, (), {}), (A.exp.default, (), {})]


class Graph16(G.PropagationGraph):
    WEIGHTS = {**G.PropagationGraph.WEIGHTS, "table16": 2.5, "int16_chain": 1.5}

    def step(self):
        rng, b = self.rng, self.b
        kind = rng.choices(list(self.WEIGHTS), list(self.WEIGHTS.values()))[0]
        if kind == "table16":
            x = self.pick(torch.int16)
            roll = rng.random()
            if roll < 0.35:
                x = b.permute(x, self.permutation(x.meta["val"].dim()))
            elif roll < 0.7:
                x = b.view(x, self.view_shape(list(x.meta["val"].shape)))
            target, args, kwargs = rng.choice(T16_TARGETS)
            qin, qout = rng.choice(T16_QPARAMS)
            t = G.table_op(b, target, x, qin, qout, *args, **kwargs)
            self.keep(t)
            if rng.random() < 0.5:
                self.keep(self.transform(t))
        elif kind == "int16_chain":
            x = self.pick(torch.int16)
            value = x
            for _ in range(rng.randint(1, 3)):
                step = rng.choice(["permute", "view", "abs", "neg"])
                if step == "permute":
                    value = b.permute(value, self.permutation(value.meta["val"].dim()))
                elif step == "view":
                    value = b.view(value, self.view_shape(list(value.meta["val"].shape)))
                else:
                    value = b.call(getattr(A, step).default, value)
            self.keep(value)
        else:
            self.WEIGHTS = G.PropagationGraph.WEIGHTS
            try:
                super().step()
            finally:
                del self.WEIGHTS


def build_case(seed):
    def build(b):
        rng = random.Random(seed)
        graph = Graph16(rng, b)
        for _ in range(rng.randint(1, 2)):
            graph.fresh_float()
        graph.keep(graph.new_input(graph.shape(rng.randint(1, 4)), torch.int16))
        for _ in range(rng.randint(3, 14)):
            graph.step()
        graph.finish()

    return build


count = int(sys.argv[2])
seed0 = int(sys.argv[3])
cases = {f"t16_{i}": build_case(seed0 * 100_000 + i) for i in range(count)}
P.emit(cases, sys.argv[1], P.out_path(sys.argv[1]))
