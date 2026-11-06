"""Probe: two 16-bit tables behind a shared permute, one reached through a
rank-changing view (the minimal form of fuzz_t16.py's t16_1377). Usage:
`probe_t16_extra_transpose.py down|up|down_up` writes that stage's cases."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402

G = P.G
A = G.A
Q1 = (G.qargs(1 / 256, dtype=torch.int16), G.qargs(1 / 4096, dtype=torch.int16))
Q2 = (G.qargs(1 / 1024, dtype=torch.int16), G.qargs(1 / 32768, dtype=torch.int16))
Q3 = (G.qargs(1 / 2048, dtype=torch.int16), G.qargs(1 / 2048, dtype=torch.int16))

cases = {}


def case(fn):
    cases[fn.__name__] = fn
    return fn


@case
def t16_1377_original(b):
    x1 = b.input("x1", (2, 1, 3), dtype=torch.int16)
    G.set_qparams(x1, inputs=[0], outputs=[0])
    p8 = b.permute(x1, [2, 1, 0])
    p9 = b.permute(p8, [1, 0, 2])
    p15 = b.permute(p9, [0, 2, 1])
    t16 = G.table_op(b, A.tanh.default, p15, *Q1)
    p17 = b.permute(t16, [1, 0, 2])
    p20 = b.permute(p17, [0, 2, 1])
    s21 = G.table_op(b, A.sigmoid.default, p20, *Q2)
    v22 = b.view(p9, [1, 6])
    s23 = G.table_op(b, A.sigmoid.default, v22, *Q3)
    b.output([s21, s23])


@case
def t16_1377_no_placeholder_qparams(b):
    x1 = b.input("x1", (2, 1, 3), dtype=torch.int16)
    p8 = b.permute(x1, [2, 1, 0])
    p9 = b.permute(p8, [1, 0, 2])
    p15 = b.permute(p9, [0, 2, 1])
    t16 = G.table_op(b, A.tanh.default, p15, *Q1)
    p17 = b.permute(t16, [1, 0, 2])
    p20 = b.permute(p17, [0, 2, 1])
    s21 = G.table_op(b, A.sigmoid.default, p20, *Q2)
    v22 = b.view(p9, [1, 6])
    s23 = G.table_op(b, A.sigmoid.default, v22, *Q3)
    b.output([s21, s23])


@case
def t16_1377_minimal(b):
    x1 = b.input("x1", (2, 1, 3), dtype=torch.int16)
    p9 = b.permute(x1, [1, 2, 0])
    p15 = b.permute(p9, [0, 2, 1])
    t16 = G.table_op(b, A.tanh.default, p15, *Q1)
    p20 = b.permute(t16, [1, 2, 0])
    s21 = G.table_op(b, A.sigmoid.default, p20, *Q2)
    v22 = b.view(p9, [1, 6])
    s23 = G.table_op(b, A.sigmoid.default, v22, *Q3)
    b.output([s21, s23])


which = sys.argv[1]
P.emit(cases, which, P.out_path(which))
