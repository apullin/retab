"""Generate torch.fx user-order vectors for src/layout/ctx.rs (LayoutCtx::users_in_fx_order).

Run from a neutral directory with the ExecuTorch v1.6.0-rc4 environment:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/torch_fx_user_order.py > <retab>/tests/fixtures/torch_fx_user_order.txt

torch.fx keeps `node.users` in insertion order, and every args or kwargs
update re-appends the updated node to all its inputs' users
(`Node._update_args_kwargs`, C++ in torch 2.15). Each scenario builds a graph
and replays an edit script through torch.fx itself, recording every node's
users at checkpoints:

    graph SPEC           the starting graph: `;`-separated items, `name` for a
                         placeholder, `name=ARG..` for a call, `output=ARG`
    create N P ARG..     create call N at insert point P: `end`, `start`,
                         `before=X` (with g.inserting_before(X)) or `after=X`
    set_args N ARG..     N.args = (ARG..)
    set_arg N I ARG      N.update_arg(I, ARG)
    set_kwarg N K ARG    N.update_kwarg(K, ARG)
    replace N OLD NEW    N.replace_input_with(OLD, NEW)
    rauw OLD NEW         OLD.replace_all_uses_with(NEW)
    erase N              g.erase_node(N)
    prepend A N          A.prepend(N): move N directly before A
    append A N           A.append(N): move N directly after A
    retrace              copy every node, in graph order, into a fresh graph,
                         which is what ExportPass.call's retrace does to users
    check N>U,U ..       each node that has users, with its users in order

An ARG is a node name, `[a,b]` (a list of nodes), an integer, or, for create
and graph items, `key=VALUE` (a kwarg). Edits keep the graph topological, so
the retrace can copy it in order.
"""

import random
import sys

import torch
import torch.fx as fx

STEPS = 70
RANDOM_SCENARIOS = 30
SEED = 158


def target(*args, **kwargs):  # the calls are never executed
    return None


def parse_value(token, nodes):
    if token.startswith("["):
        inner = token[1:-1]
        return [nodes[name] for name in inner.split(",") if name]
    if token.lstrip("-").isdigit():
        return int(token)
    return nodes[token]


def parse_args(tokens, nodes):
    args, kwargs = [], {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and not token.startswith("["):
            kwargs[key] = parse_value(value, nodes)
        else:
            args.append(parse_value(token, nodes))
    return tuple(args), kwargs


def build(spec):
    g = fx.Graph()
    nodes = {}
    for item in spec.split(";"):
        words = item.split()
        name, _, first = words[0].partition("=")
        tokens = ([first] if first else []) + words[1:]
        if name == "output":
            nodes[name] = g.output(parse_value(tokens[0], nodes))
        elif not tokens:
            nodes[name] = g.placeholder(name)
        else:
            args, kwargs = parse_args(tokens, nodes)
            nodes[name] = g.create_node("call_function", target, args, kwargs, name=name)
        assert nodes[name].name == name, (nodes[name].name, name)
    return g, nodes


def retrace(g):
    g2 = fx.Graph()
    env = {}
    for node in g.nodes:
        env[node] = g2.node_copy(node, lambda n: env[n])
        assert env[node].name == node.name
    return g2, {node.name: copy for node, copy in env.items()}


def apply(g, nodes, line):
    op, *rest = line.split()
    if op == "create":
        name, point, *tokens = rest
        args, kwargs = parse_args(tokens, nodes)
        if point == "end":
            node = g.create_node("call_function", target, args, kwargs, name=name)
        elif point == "start":
            with g.inserting_before(None):
                node = g.create_node("call_function", target, args, kwargs, name=name)
        else:
            where, anchor = point.split("=")
            scope = g.inserting_before if where == "before" else g.inserting_after
            with scope(nodes[anchor]):
                node = g.create_node("call_function", target, args, kwargs, name=name)
        assert node.name == name
        nodes[name] = node
    elif op == "set_args":
        nodes[rest[0]].args = parse_args(rest[1:], nodes)[0]
    elif op == "set_arg":
        nodes[rest[0]].update_arg(int(rest[1]), parse_value(rest[2], nodes))
    elif op == "set_kwarg":
        nodes[rest[0]].update_kwarg(rest[1], parse_value(rest[2], nodes))
    elif op == "replace":
        nodes[rest[0]].replace_input_with(nodes[rest[1]], nodes[rest[2]])
    elif op == "rauw":
        nodes[rest[0]].replace_all_uses_with(nodes[rest[1]])
    elif op == "erase":
        g.erase_node(nodes.pop(rest[0]))
    elif op == "prepend":
        nodes[rest[0]].prepend(nodes[rest[1]])
    elif op == "append":
        nodes[rest[0]].append(nodes[rest[1]])
    elif op == "retrace":
        g, nodes = retrace(g)
    else:
        raise ValueError(line)
    return g, nodes


def check(g):
    items = [
        f"{node.name}>{','.join(user.name for user in node.users)}"
        for node in g.nodes
        if node.users
    ]
    return "check " + " ".join(items)


def fixed():
    """Hand-picked scripts for each FX rule (the comments give the expected users)."""
    return [
        (
            "args_updates_reappend",
            "x; y; a=x; b=x y; c=y x; d=[x,y] k=x",
            [
                "set_arg b 1 y",  # x>a,c,d,b: an unchanged update re-appends b
                "set_args a x",  # x>c,d,b,a
                "set_kwarg d k y",  # d moves to the end of x and y
                "replace c y x",  # c reads x twice: one user entry
                "replace a y x",  # a does not read y: still re-appended
                "replace b x x",  # old == new: still re-appended
            ],
        ),
        (
            "replace_all_uses",
            "x; y; z; a=x y; b=y; c=z y; d=y x",
            [
                "rauw y z",  # z>a,b,c,d: y's users in y's order; c, which read z too, moves as well
                "rauw z z",  # each user re-appended in z's order
                "rauw x z",
            ],
        ),
        (
            "create_erase_move_retrace",
            "x; y; a=x; b=y x; output=[a,b]",
            [
                "create m0 before=a x y",
                "create m1 after=x x",
                "create m2 start",
                "set_arg a 0 m0",
                "erase m2",
                "append b m1",  # moving changes no user list
                "prepend a m1",
                "retrace",  # back to graph order
                "set_args output [b,a,m0]",
            ],
        ),
    ]


class Scenario:
    def __init__(self, rng):
        self.rng = rng
        self.fresh = 0
        self.spec = self.random_spec()
        self.g, self.nodes = build(self.spec)

    def name(self):
        self.fresh += 1
        return f"m{self.fresh - 1}"

    def random_spec(self):
        rng = self.rng
        items, names = [], []
        for index in range(rng.randint(2, 4)):
            names.append(f"p{index}")
            items.append(names[-1])
        for index in range(rng.randint(5, 10)):
            tokens = [self.random_arg(names) for _ in range(rng.randint(1, 3))]
            if rng.random() < 0.2:
                tokens.append(f"k={rng.choice(names)}")
            name = f"n{index}"
            items.append(f"{name}=" + " ".join(tokens))
            names.append(name)
        if rng.random() < 0.6:
            outs = rng.sample(names[-4:], rng.randint(1, 2))
            items.append("output=[" + ",".join(outs) + "]")
        return "; ".join(items)

    def random_arg(self, candidates):
        rng = self.rng
        roll = rng.random()
        if roll < 0.12 or not candidates:
            return str(rng.randint(-2, 9))
        if roll < 0.25:
            picks = [rng.choice(candidates) for _ in range(rng.randint(0, 3))]
            return "[" + ",".join(picks) + "]"
        return rng.choice(candidates)

    # Graph queries ---------------------------------------------------------

    def order(self):
        return list(self.g.nodes)

    def calls(self):
        return [n for n in self.g.nodes if n.op == "call_function"]

    def output(self):
        outs = [n for n in self.g.nodes if n.op == "output"]
        return outs[0] if outs else None

    def before(self, node):
        """Names of the non-output nodes before `node`."""
        result = []
        for n in self.g.nodes:
            if n is node:
                break
            if n.op != "output":
                result.append(n.name)
        return result

    def topological(self, order):
        position = {n: i for i, n in enumerate(order)}
        return all(
            position[i] < position[n] for n in order for i in n.all_input_nodes
        )

    # Steps -----------------------------------------------------------------

    def step(self):
        rng = self.rng
        kinds = ["create", "set_args", "set_arg", "set_kwarg", "replace", "rauw", "erase", "move", "retrace"]
        weights = [3, 2, 4, 1.5, 5, 2, 1.5, 2.5, 0.6]
        while True:
            line = getattr(self, "op_" + rng.choices(kinds, weights)[0])()
            if line is not None:
                return line

    def op_create(self):
        rng = self.rng
        out = self.output()
        live = [n for n in self.g.nodes if n.op != "output"]
        roll = rng.random()
        if roll < 0.08:
            point, candidates = "start", []
        elif roll < 0.5 and live:
            anchor = rng.choice(live + ([out] if out else []))
            point, candidates = f"before={anchor.name}", self.before(anchor)
        elif live:
            anchor = rng.choice(live)
            point, candidates = f"after={anchor.name}", self.before(anchor) + [anchor.name]
        else:
            return None
        if out is None and rng.random() < 0.3:
            point, candidates = "end", [n.name for n in live]
        tokens = [self.random_arg(candidates) for _ in range(rng.randint(0, 3))]
        if candidates and rng.random() < 0.2:
            tokens.append(f"w={rng.choice(candidates)}")
        return " ".join(["create", self.name(), point, *tokens])

    def editable(self):
        nodes = self.calls()
        out = self.output()
        if out is not None and self.rng.random() < 0.15:
            nodes.append(out)
        return nodes

    def op_set_args(self):
        node = self.rng.choice(self.editable())
        candidates = self.before(node)
        if node.op == "output":
            picks = self.rng.sample(candidates, min(len(candidates), self.rng.randint(1, 3)))
            return f"set_args output [{','.join(picks)}]"
        tokens = [self.random_arg(candidates) for _ in range(self.rng.randint(0, 3))]
        return " ".join(["set_args", node.name, *tokens])

    def op_set_arg(self):
        rng = self.rng
        nodes = [n for n in self.calls() if n.args]
        if not nodes:
            return None
        node = rng.choice(nodes)
        index = rng.randrange(len(node.args))
        current = node.args[index]
        if rng.random() < 0.2 and isinstance(current, fx.Node):
            value = current.name  # an update that changes nothing
        else:
            value = self.random_arg(self.before(node))
        return f"set_arg {node.name} {index} {value}"

    def op_set_kwarg(self):
        node = self.rng.choice(self.calls())
        candidates = self.before(node)
        value = self.rng.choice(candidates) if candidates and self.rng.random() < 0.8 else "3"
        return f"set_kwarg {node.name} {self.rng.choice(['k', 'w'])} {value}"

    def op_replace(self):
        rng = self.rng
        node = rng.choice(self.editable())
        candidates = self.before(node)
        inputs = [n.name for n in node.all_input_nodes]
        if not candidates:
            return None
        roll = rng.random()
        if roll < 0.75 and inputs:
            old = rng.choice(inputs)
        else:
            old = rng.choice([n.name for n in self.g.nodes if n.op != "output"])
        new = old if rng.random() < 0.1 else rng.choice(candidates)
        return f"replace {node.name} {old} {new}"

    def op_rauw(self):
        rng = self.rng
        old = rng.choice([n for n in self.g.nodes if n.op != "output"])
        users = list(old.users)
        if not users:
            return None if rng.random() < 0.7 else f"rauw {old.name} {old.name}"
        position = {n: i for i, n in enumerate(self.g.nodes)}
        first = min(position[u] for u in users)
        candidates = [n.name for n in self.g.nodes if position[n] < first and n.op != "output"]
        new = old.name if rng.random() < 0.15 or not candidates else rng.choice(candidates)
        return f"rauw {old.name} {new}"

    def op_erase(self):
        dead = [n for n in self.calls() if not n.users]
        return f"erase {self.rng.choice(dead).name}" if dead else None

    def op_move(self):
        rng = self.rng
        order = self.order()
        node = rng.choice([n for n in order if n.op == "call_function"])
        anchor = rng.choice([n for n in order if n.op != "output"] + ([self.output()] if self.output() else []))
        kind = rng.choice(["prepend", "append"])
        if kind == "append" and anchor.op == "output":
            return None
        trial = [n for n in order if n is not node]
        if node is anchor:
            trial = order
        else:
            at = trial.index(anchor) + (1 if kind == "append" else 0)
            trial.insert(at, node)
        if not self.topological(trial):
            return None
        return f"{kind} {anchor.name} {node.name}"

    def op_retrace(self):
        return "retrace"


def random_script(seed):
    rng = random.Random(seed)
    scenario = Scenario(rng)
    lines = [f"graph {scenario.spec}", check(scenario.g)]
    for _ in range(STEPS):
        line = scenario.step()
        scenario.g, scenario.nodes = apply(scenario.g, scenario.nodes, line)
        lines += [line, check(scenario.g)]
    return lines


def main():
    out = [
        f"# torch.fx user-order vectors (torch {torch.__version__}) for src/layout/ctx.rs.",
        "# Generated by tests/fixtures/torch_fx_user_order.py; see its docstring for the format.",
    ]
    for title, spec, ops in fixed():
        g, nodes = build(spec)
        out += [f"scenario {title}", f"graph {spec}", check(g)]
        for line in ops:
            g, nodes = apply(g, nodes, line)
            out += [line, check(g)]
    for index in range(RANDOM_SCENARIOS):
        out += [f"scenario random_{index}", *random_script(SEED * 1000 + index)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
