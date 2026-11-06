"""Targeted search with review-shaped motifs: an entry permute P read by a
binary op directly and by a unary chain through a copy that a walk can move,
so the walk appends a user to P after the binary op."""
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import user_order_search as S  # noqa: E402

G = S.G
A = G.A
UN = [A.relu.default, A.neg.default, A.abs.default]
BIN = [A.add.Tensor, A.sub.Tensor, A.mul.Tensor]


def make(seed):
    def build(b):
        rng = random.Random(seed)
        shape = rng.choice([(4, 4), (3, 3), (2, 2)])
        pool = [b.input(f"x{i}", shape) for i in range(rng.randint(1, 3))]

        def pick():
            return pool[-1 - min(int(rng.expovariate(0.5)), len(pool) - 1)]

        for _ in range(rng.randint(1, 4)):
            src = pick()
            p = b.permute(src, [1, 0])
            mid = p
            for _ in range(rng.randint(0, 2)):
                mid = b.call(rng.choice(UN), mid)
            c = b.permute(mid, [1, 0]) if rng.random() < 0.6 else b.view(mid, list(shape))
            u1 = c
            for _ in range(rng.randint(1, 2)):
                u1 = b.call(rng.choice(UN), u1)
            if rng.random() < 0.4:
                u1 = b.permute(u1, [1, 0])
            args = (p, u1) if rng.random() < 0.5 else (u1, p)
            u2 = b.call(rng.choice(BIN), *args)
            if rng.random() < 0.3:
                u2 = b.call(rng.choice(BIN), u2, rng.choice(pool))
            e2 = b.permute(u2, [1, 0]) if rng.random() < 0.7 else u2
            pool += [p, u1, u2, e2]
            if rng.random() < 0.3:
                pool.append(b.call(rng.choice(UN), rng.choice(pool)))
        outs = [n for n in pool if n.op == "call_function" and not n.users] or [pool[-1]]
        b.output(outs)
    return build


count, seed = int(sys.argv[1]), int(sys.argv[2])
builders = [(f"m{i}", make(seed * 100_000 + i)) for i in range(count)]
found = S.search(builders, "motif")
print(f"done: {len(found)} order-dependent of {count}")
