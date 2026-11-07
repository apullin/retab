"""Generate torch.fx node-order vectors for src/layout/ctx.rs (LayoutCtx).

Run from a neutral directory with the ExecuTorch v1.6.0-rc4 environment:

    python -P tests/fixtures/torch_fx_insert_order.py > tests/fixtures/torch_fx_insert_order.txt

Each scenario starts from argument-free call_function nodes and replays an op
script through torch.fx's own insertion machinery (torch/fx/graph.py
Graph.inserting_before/inserting_after/create_node, torch/fx/node.py
Node.prepend/append):

    before A n..   with g.inserting_before(A): create n..
    after A n..    with g.inserting_after(A): create n..
    start n..      with g.inserting_before(None): create n..
    end n..        create n.. at the default insert point (Graph._root.prepend)
    prepend A X    A.prepend(X), moving an existing node X
    append A X     A.append(X)
    erase X        g.erase_node(X)
    check ..       the expected g.nodes order at this point

With --embedded it instead prints one line per nested-scope case and per rc4
rewrite sequence (data edges included); ctx.rs embeds those orders in its tests.
"""

import random
import sys

import torch
import torch.fx as fx


def new_graph(names):
    g = fx.Graph()
    nodes = {}
    for name in names:
        create(g, nodes, name)
    return g, nodes


def create(g, nodes, name):
    n = g.create_node("call_function", torch.relu, (), name=name)
    assert n.name == name, (n.name, name)
    nodes[name] = n


def order(g):
    return [n.name for n in g.nodes]


def apply(g, nodes, line):
    op, *rest = line.split()
    if op == "before":
        with g.inserting_before(nodes[rest[0]]):
            for name in rest[1:]:
                create(g, nodes, name)
    elif op == "after":
        with g.inserting_after(nodes[rest[0]]):
            for name in rest[1:]:
                create(g, nodes, name)
    elif op == "start":
        with g.inserting_before(None):
            for name in rest:
                create(g, nodes, name)
    elif op == "end":
        for name in rest:
            create(g, nodes, name)
    elif op == "prepend":
        nodes[rest[0]].prepend(nodes[rest[1]])
    elif op == "append":
        nodes[rest[0]].append(nodes[rest[1]])
    elif op == "erase":
        g.erase_node(nodes.pop(rest[0]))
    else:
        raise ValueError(line)


def fixed_script(base, ops):
    g, nodes = new_graph(base)
    lines = []
    for line in ops:
        apply(g, nodes, line)
        lines += [line, "check " + " ".join(order(g))]
    return lines


def random_script(seed, steps, base, check_every):
    rng = random.Random(seed)
    g, nodes = new_graph(base)
    lines = []
    fresh = 0
    kinds = ["before", "after", "start", "end", "prepend", "append", "erase"]
    for step in range(steps):
        live = order(g)
        weights = [5, 5, 1, 1, 6, 6, 4 if len(live) > 24 else 1]
        kind = rng.choices(kinds, weights)[0]
        if kind in ("before", "after", "start", "end"):
            count = rng.choice([1, 1, 2, 3])
            names = [f"m{fresh + i}" for i in range(count)]
            fresh += count
            anchor = [rng.choice(live)] if kind in ("before", "after") else []
            line = " ".join([kind, *anchor, *names])
        elif kind in ("prepend", "append"):
            anchor = rng.choice(live)
            # Mostly distinct nodes; sometimes the anchor itself or a neighbour,
            # which exercises FX's no-op cases.
            pick = rng.random()
            if pick < 0.1:
                node = anchor
            elif pick < 0.2 and nodes[anchor].next.op != "root":
                node = nodes[anchor].next.name
            elif pick < 0.3 and nodes[anchor].prev.op != "root":
                node = nodes[anchor].prev.name
            else:
                node = rng.choice(live)
            line = f"{kind} {anchor} {node}"
        else:
            line = f"erase {rng.choice(live)}"
        apply(g, nodes, line)
        lines.append(line)
        if step % check_every == check_every - 1 or step == steps - 1:
            lines.append("check " + " ".join(order(g)))
    return lines


def nested():
    """Nested insertion scopes and insert points whose anchors move."""
    out = []
    g, nodes = new_graph(["a", "b", "c", "d"])
    with g.inserting_after(nodes["a"]):
        create(g, nodes, "p0")
        with g.inserting_before(nodes["c"]):
            create(g, nodes, "q0")
            create(g, nodes, "q1")
        create(g, nodes, "p1")
    out.append("nested_scopes " + " ".join(order(g)))

    g, nodes = new_graph(["a", "b", "c", "d"])
    with g.inserting_after(nodes["a"]):
        create(g, nodes, "p0")
        nodes["a"].append(nodes["c"])
        create(g, nodes, "p1")
    out.append("after_anchor_next_moves " + " ".join(order(g)))

    g, nodes = new_graph(["a", "b", "c", "d"])
    with g.inserting_before(nodes["b"]):
        create(g, nodes, "p0")
        nodes["d"].append(nodes["b"])
        create(g, nodes, "p1")
    out.append("before_anchor_moves " + " ".join(order(g)))

    g, nodes = new_graph(["a", "b", "c", "d"])
    g.inserting_after(nodes["b"])  # permanent insert point, no `with`
    create(g, nodes, "p0")
    create(g, nodes, "p1")
    out.append("permanent_insert_point " + " ".join(order(g)))
    return out


def graph_from(defs):
    """Build a graph from (name, args) pairs; args name earlier nodes, and a
    tuple arg becomes a tuple of nodes. Argument-free names are placeholders."""
    g = fx.Graph()
    nodes = {}

    def resolve(arg):
        if isinstance(arg, (list, tuple)):
            return type(arg)(nodes[a] for a in arg)
        return nodes[arg]

    for name, args in defs:
        resolved = tuple(resolve(a) for a in args)
        if name == "output":
            nodes[name] = g.output(resolved[0])
        elif not args:
            nodes[name] = g.placeholder(name)
        else:
            nodes[name] = g.create_node("call_function", torch.relu, resolved, name=name)
    return g, nodes


def up_move_node(frontier_is_placeholder):
    """PropagateViewCopyPermuteUpPass._move_node
    (backends/transforms/propagate_view_copy_permute_pass.py:797-822), moving
    p up past b (frontier b) or past b and a (frontier x, a placeholder)."""
    g, n = graph_from([("x", []), ("a", ["x"]), ("b", ["a"]), ("p", ["b"]), ("output", ["p"])])
    node, original_input = n["p"], n["b"]
    if frontier_is_placeholder:
        producer, frontier_user = n["x"], n["a"]
    else:
        producer, frontier_user = n["a"], n["b"]
    node.replace_input_with(original_input, producer)
    frontier_user.replace_input_with(producer, node)
    for user in list(node.users):
        if user is not frontier_user:
            user.replace_input_with(node, original_input)
    frontier_user.prepend(node)
    return order(g)


def down_move_node(frontier_is_output):
    """PropagateViewCopyPermuteDownPass._move_node
    (backends/transforms/propagate_view_copy_permute_pass.py:1393-1418), moving
    p down past a (frontier a) or past a and b (frontier output)."""
    g, n = graph_from([("x", []), ("p", ["x"]), ("a", ["p"]), ("b", ["a"]), ("output", ["b"])])
    node = n["p"]
    original_user = next(iter(node.users))
    producer = node.all_input_nodes[0]
    if frontier_is_output:
        frontier, frontier_input = n["output"], n["b"]
    else:
        frontier, frontier_input = n["a"], n["a"]
    frontier_users = list(frontier_input.users)
    original_user.replace_input_with(node, producer)
    node.replace_input_with(producer, frontier_input)
    for user in frontier_users:
        if user is not node:
            user.replace_input_with(frontier_input, node)
    if frontier.op == "output":
        frontier.prepend(node)
    else:
        frontier.append(node)
    return order(g)


def down_fork_split():
    """PropagateViewCopyPermuteDownPass._apply_fork_split with frontier is node
    (backends/transforms/propagate_view_copy_permute_pass.py:1338-1391)."""
    g, n = graph_from([("x", []), ("p", ["x"]), ("a", ["p"]), ("b", ["p"]), ("output", [("a", "b")])])
    node, producer = n["p"], n["x"]
    for next_node in (n["a"], n["b"]):
        next_node.replace_input_with(node, producer)
        with g.inserting_after(next_node):
            branch = g.create_node("call_function", torch.permute, (next_node,), name="t_" + next_node.name)
        for user in list(next_node.users):
            if user is not branch:
                user.replace_input_with(next_node, branch)
    if not node.users:
        g.erase_node(node)
    return order(g)


def up_cat_fanout():
    """PropagateViewCopyPermuteUpPass._maybe_split_upwards_cat_fanout
    (backends/transforms/propagate_view_copy_permute_pass.py:824-890)."""
    g, n = graph_from([("x1", []), ("x2", []), ("c", [["x1", "x2"]]), ("p", ["c"]), ("output", ["p"])])
    next_node = n["c"]
    cat_users = list(next_node.users)
    new_inputs = []
    for input_node in list(next_node.all_input_nodes):
        with g.inserting_before(next_node):
            permute = g.create_node("call_function", torch.permute, (input_node,), name="q_" + input_node.name)
        new_inputs.append(permute)
    next_node.args = (new_inputs, *next_node.args[1:])
    for cat_user in cat_users:
        cat_user.replace_all_uses_with(next_node)
    for cat_user in cat_users:
        if len(cat_user.users) == 0:
            g.erase_node(cat_user)
    return order(g)


def move_before_rescale():
    """MoveDataMovementOpsToSmallerDtypePass._move_before_rescale
    (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:100-118)."""
    g, n = graph_from([("x", []), ("r", ["x"]), ("p", ["r"]), ("u", ["p"]), ("output", ["u"])])
    data_movement, rescale = n["p"], n["r"]
    producer = rescale.all_input_nodes[0]
    movement_users = list(data_movement.users)
    data_movement.replace_input_with(rescale, producer)
    rescale.replace_input_with(producer, data_movement)
    for user in movement_users:
        user.replace_input_with(data_movement, rescale)
    data_movement.append(rescale)
    return order(g)


def move_after_rescale():
    """MoveDataMovementOpsToSmallerDtypePass._move_after_rescale
    (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:120-137)."""
    g, n = graph_from([("x", []), ("p", ["x"]), ("r", ["p"]), ("u", ["r"]), ("output", ["u"])])
    data_movement, rescale = n["p"], n["r"]
    producer = data_movement.all_input_nodes[0]
    rescale_users = list(rescale.users)
    rescale.replace_input_with(data_movement, producer)
    data_movement.replace_input_with(producer, rescale)
    for user in rescale_users:
        user.replace_input_with(rescale, data_movement)
    rescale.append(data_movement)
    return order(g)


def split_after_rescale():
    """MoveDataMovementOpsToSmallerDtypePass._split_after_rescale
    (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:139-158)."""
    g, n = graph_from(
        [("x", []), ("p", ["x"]), ("r", ["p"]), ("v", ["p"]), ("u", ["r"]), ("output", [("u", "v")])]
    )
    data_movement, rescale = n["p"], n["r"]
    producer = data_movement.all_input_nodes[0]
    rescale_users = list(rescale.users)
    rescale.replace_input_with(data_movement, producer)
    with g.inserting_after(rescale):
        branch = g.create_node("call_function", torch.permute, (rescale,), name="branch")
    for user in rescale_users:
        user.replace_input_with(rescale, branch)
    if not data_movement.users:
        g.erase_node(data_movement)
    return order(g)


def embedded():
    lines = nested()
    for name, result in [
        ("up_move_node_frontier_b", up_move_node(False)),
        ("up_move_node_frontier_placeholder", up_move_node(True)),
        ("down_move_node_frontier_a", down_move_node(False)),
        ("down_move_node_frontier_output", down_move_node(True)),
        ("down_fork_split", down_fork_split()),
        ("up_cat_fanout", up_cat_fanout()),
        ("move_before_rescale", move_before_rescale()),
        ("move_after_rescale", move_after_rescale()),
        ("split_after_rescale", split_after_rescale()),
    ]:
        lines.append(name + " " + " ".join(result))
    return lines


def main():
    print(f"# torch.fx insertion-order vectors (torch {torch.__version__}) for src/layout/ctx.rs.")
    print("# Generated by tests/fixtures/torch_fx_insert_order.py; see its docstring for the op format.")
    print("scenario fixed")
    base = ["a", "b", "c", "d", "e"]
    print("base " + " ".join(base))
    for line in fixed_script(
        base,
        [
            "before c x0 x1 x2",
            "after b y0 y1 y2",
            "start s0 s1",
            "end z0 z1",
            "prepend b d",
            "append a e",
            "prepend c c",
            "append a a",
            "append d y2",
            "erase x1",
            "after z1 w0 w1",
            "before s1 v0",
            "prepend s1 z1",
        ],
    ):
        print(line)
    seed, steps = 20261105, 120
    print(f"scenario random seed={seed} steps={steps}")
    base = [f"n{i}" for i in range(12)]
    print("base " + " ".join(base))
    for line in random_script(seed, steps, base, check_every=20):
        print(line)


if __name__ == "__main__":
    if sys.argv[1:] == ["--embedded"]:
        print(f"# torch {torch.__version__}")
        for line in embedded():
            print(line)
    else:
        main()
