"""Write rc4 dim_maps results on seeded random cases as a TSV for src/layout/dim_maps.rs.

The Rust test `matches_rc4_python_vectors` replays every line against the port.
Regenerate with a Python that has ExecuTorch v1.6.0-rc4 installed, from a neutral
directory so a source checkout can't shadow the installed package:

    cd /tmp && python -P <retab>/tests/fixtures/dim_maps_rc4_vectors.py \
        > <retab>/tests/fixtures/dim_maps_rc4_vectors.tsv

The cases come from `random.Random(SEED)`, so the output is stable for a given
CPython `random` implementation (written with 3.12).
"""

import hashlib
import json
import math
import random
import sys
import types

from executorch.backends.transforms import dim_maps
from executorch.backends.transforms.dim_maps import (
    _extend_permutation,
    PermuteMap,
    ViewMap,
)

SEED = 135
VIEW_CASES = 200
PERMUTE_CASES = 80
EXTEND_CASES = 40
MAX_RANK = 6
DIMS = [1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16]
RANKS = [0, 1, 2, 2, 3, 3, 4, 4, 5, 6]
EXPECTED_EXCEPTIONS = (AssertionError, ValueError, ZeroDivisionError)
# normalize_view_shape corners the random views rarely reach: flooring `//` with a
# negative known dim, ZeroDivisionError, two `-1`s, rank 0, and the docstring view.
EDGE_VIEWS = [
    ([5], [-2, -1]),
    ([7, 3], [-4, -1]),
    ([0, 5], [0, -1]),
    ([4], [2, -1, -1]),
    ([1], [-1, 1]),
    ([], [-1]),
    ([4, 3, 10], [2, 2, 1, 5, 3, 2]),
]

rng = random.Random(SEED)
lines: list[str] = []


def dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def exception_text(error: Exception) -> str:
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def to_json(value):
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if value is not None and not isinstance(value, int):
        raise TypeError(f"unexpected result {value!r}")
    return value


def run(fn, *args):
    try:
        result = fn(*args)
    except EXPECTED_EXCEPTIONS as error:
        return exception_text(error)
    return to_json(result)


def emit(query: str, args: list, result) -> None:
    lines.append("\t".join([query, dumps(args), dumps(result)]))


def prime_factors(dim: int) -> list[int]:
    factors = []
    divisor = 2
    while divisor * divisor <= dim:
        while dim % divisor == 0:
            factors.append(divisor)
            dim //= divisor
        divisor += 1
    if dim > 1:
        factors.append(dim)
    return factors


def random_shape(rank: int) -> list[int]:
    return [rng.choice(DIMS) for _ in range(rank)]


def chunk(primes: list[int], rank: int) -> list[int]:
    """Split ordered primes into `rank` contiguous products; empty chunks are 1."""
    cuts = sorted(rng.randint(0, len(primes)) for _ in range(rank - 1))
    bounds = [0, *cuts, len(primes)]
    return [math.prod(primes[lo:hi]) for lo, hi in zip(bounds, bounds[1:])]


def random_target(source: list[int]) -> list[int]:
    primes = [prime for dim in source for prime in prime_factors(dim)]
    mode = rng.random()
    if mode < 0.06:
        return random_shape(rng.choice(RANKS))
    if mode < 0.09:
        return [0, -1]
    if mode < 0.19:
        target = list(source)
        rng.shuffle(target)
    else:
        if mode < 0.45:
            for _ in range(rng.randint(1, 3)):
                if len(primes) >= 2:
                    index = rng.randrange(len(primes) - 1)
                    primes[index], primes[index + 1] = primes[index + 1], primes[index]
        rank = rng.randint(1 if primes else 0, MAX_RANK)
        target = chunk(primes, rank) if rank else []
    inferred = rng.random()
    if target and inferred < 0.12:
        target[rng.randrange(len(target))] = -1
    elif len(target) >= 2 and inferred < 0.15:
        first, second = rng.sample(range(len(target)), 2)
        target[first] = target[second] = -1
    return target


def random_dims(rank: int, axis_to_groups=None) -> list[int]:
    """Random axes: whole groups when given, else any subset; sometimes aliased or invalid."""
    mode = rng.random()
    if rank == 0 or mode < 0.03:
        return [] if mode < 0.015 else [rng.randint(-3, 3)]
    if axis_to_groups is not None and mode < 0.55:
        groups = sorted({group for groups in axis_to_groups for group in groups})
        chosen = set(rng.sample(groups, rng.randint(1, len(groups)))) if groups else set()
        dims = [
            axis
            for axis, groups in enumerate(axis_to_groups)
            if (groups and groups[0] in chosen) or (not groups and rng.random() < 0.3)
        ]
        if not dims:
            dims = [rng.randrange(rank)]
    else:
        dims = sorted(rng.sample(range(rank), rng.randint(1, rank)))
    variant = rng.random()
    if variant < 0.15:
        dims = [dim - rank if rng.random() < 0.5 else dim for dim in dims]
    elif variant < 0.22:
        rng.shuffle(dims)
    elif variant < 0.26:
        dims.append(rng.choice([rank, -rank - 1]))
    elif variant < 0.29:
        dims.append(dims[0])
    return dims


def random_permutation(rank: int) -> list[int]:
    permutation = list(range(rank))
    rng.shuffle(permutation)
    return alias_or_break(permutation)


def block_permutation(axis_to_groups) -> list[int]:
    """Shuffle whole group blocks; singleton axes move on their own."""
    blocks: list[tuple[int | None, list[int]]] = []
    for axis, groups in enumerate(axis_to_groups):
        group = groups[0] if groups else None
        if group is not None and blocks and blocks[-1][0] == group:
            blocks[-1][1].append(axis)
        else:
            blocks.append((group, [axis]))
    rng.shuffle(blocks)
    return alias_or_break([axis for _, axes in blocks for axis in axes])


def alias_or_break(permutation: list[int]) -> list[int]:
    rank = len(permutation)
    variant = rng.random()
    if variant < 0.15:
        return [dim - rank if rng.random() < 0.5 else dim for dim in permutation]
    if rank >= 2 and variant < 0.19:
        permutation[0] = permutation[1]
    elif variant < 0.22:
        permutation = permutation + [rank]
    return permutation


def expanded_shape(source: list[int]) -> tuple[list[int], list[int]]:
    """A reduction or slice input for `source`: some singleton axes regrown."""
    if rng.random() < 0.05:
        return random_shape(max(0, len(source) + rng.choice([-1, 1]))), [0]
    shape = list(source)
    grown = [axis for axis, dim in enumerate(source) if dim == 1 and rng.random() < 0.6]
    for axis in grown:
        shape[axis] = rng.randint(2, 6)
    if shape and rng.random() < 0.15:
        shape[rng.randrange(len(shape))] = rng.choice(DIMS)
    if shape and rng.random() < 0.05:
        shape[rng.randrange(len(shape))] = 0
    dims = grown or random_dims(len(source))
    return shape, dims


def unit_slice_args(source: list[int], dropped_axis: int | None) -> list:
    rank = len(source)
    singletons = [axis for axis, dim in enumerate(source) if dim == 1]
    if dropped_axis is not None and rng.random() < 0.85:
        slice_dim = dropped_axis
    elif singletons and rng.random() < 0.85:
        slice_dim = rng.choice(singletons)
    else:
        slice_dim = rng.randrange(rank) if rank else 0
    producer = list(source)
    if rank:
        producer[slice_dim] = rng.randint(2, 6)
    size = producer[slice_dim] if rank else 1
    start = rng.randrange(size)
    end = start + 1
    step = 1
    variant = rng.random()
    if variant < 0.08:
        start, end = start - size, end - size
    elif variant < 0.12:
        end = start + 2
    elif variant < 0.15:
        step = 2
    elif variant < 0.18:
        start = None
    elif variant < 0.21:
        end = None
    elif variant < 0.25:
        slice_dim -= rank
    elif variant < 0.27:
        producer = producer[:-1]
    return [producer, slice_dim, start, end, step]


def random_view_shapes() -> tuple[list[int], list[int], int | None]:
    source = random_shape(rng.choice(RANKS))
    if source and rng.random() < 0.03:
        source[rng.randrange(len(source))] = 0
    dropped_axis = None
    if 0 < len(source) < MAX_RANK and rng.random() < 0.2:
        # Drop an inserted singleton, sometimes merging the next two axes: the
        # views remap_unit_slice moves. A trailing singleton has nothing to fold into.
        trailing = len(source) == 1 or rng.random() < 0.2
        dropped_axis = len(source) if trailing else rng.randint(1, len(source) - 1)
        source.insert(dropped_axis, 1)
        target = source[:dropped_axis] + source[dropped_axis + 1 :]
        if dropped_axis + 1 < len(target) and rng.random() < 0.5:
            merged = target[dropped_axis] * target[dropped_axis + 1]
            target[dropped_axis : dropped_axis + 2] = [merged]
    else:
        target = random_target(source)
    return source, target, dropped_axis


def view_case(source: list[int], target: list[int], dropped_axis: int | None) -> None:
    try:
        view_map = ViewMap.from_shapes(source, target)
    except EXPECTED_EXCEPTIONS as error:
        emit("view", [source, target], exception_text(error))
        return
    emit(
        "view",
        [source, target],
        [view_map.source_shape, view_map.target_shape, view_map.is_valid_map],
    )
    groups = view_map._groups
    source_groups = groups.source_axis_to_groups if groups else None
    target_groups = groups.target_axis_to_groups if groups else None
    source_rank, target_rank = view_map.source_rank, view_map.target_rank

    emit("source_to_target_axes", [], run(view_map.source_to_target_axes))
    for _ in range(2):
        dims = random_dims(source_rank, source_groups)
        emit("map_dim", [dims], run(view_map.map_dim, dims))
    for _ in range(2):
        dims = random_dims(target_rank, target_groups)
        emit("map_dim_inverse", [dims], run(view_map.map_dim_inverse, dims))
    for query, rank, axis_groups in [
        ("map_permutation", source_rank, source_groups),
        ("map_permutation_inverse", target_rank, target_groups),
    ]:
        method = getattr(view_map, query)
        permutations = [random_permutation(rank)]
        if axis_groups is not None:
            permutations.append(block_permutation(axis_groups))
        for permutation in permutations:
            emit(query, [permutation], run(method, permutation))
    shape, _ = expanded_shape(view_map.source_shape)
    emit("remap_target_shape", [shape], run(view_map.remap_target_shape, shape))
    shape, dims = expanded_shape(view_map.source_shape)
    emit(
        "map_reduction_after_view",
        [shape, dims],
        run(view_map.map_reduction_after_view, shape, dims),
    )
    dims = random_dims(target_rank, target_groups)
    emit("map_reduction_before_view", [dims], run(view_map.map_reduction_before_view, dims))
    dims = random_dims(source_rank)
    emit(
        "map_source_dims_to_target_axes",
        [dims],
        run(view_map.map_source_dims_to_target_axes, dims),
    )
    args = unit_slice_args(view_map.source_shape, dropped_axis)
    emit("remap_unit_slice", args, run(view_map.remap_unit_slice, *args))


def permute_case() -> None:
    dims = random_permutation(rng.choice(RANKS))
    if dims and rng.random() < 0.06:
        dims[rng.randrange(len(dims))] = len(dims) + rng.randint(0, 2)
    try:
        permute_map = PermuteMap(types.SimpleNamespace(args=(None, dims)))
    except EXPECTED_EXCEPTIONS as error:
        emit("permute", [dims], exception_text(error))
        return
    emit("permute", [dims], permute_map.permute_dims)
    rank = len(dims)
    for query in ["map_dims", "map_dims", "map_dims_inverse", "map_dims_inverse"]:
        mapped = random_dims(rank)
        emit(query, [mapped], run(getattr(permute_map, query), mapped))

    lifted_rank = rank + rng.randint(0, 3)
    if rng.random() < 0.5:
        axes = list(range(lifted_rank - rank, lifted_rank))
    else:
        axes = sorted(rng.sample(range(lifted_rank), rank))
    variant = rng.random()
    if variant < 0.1:
        rng.shuffle(axes)
    elif variant < 0.2:
        axes = [axis - lifted_rank for axis in axes]
    elif axes and variant < 0.25:
        axes[-1] = axes[0] if len(axes) > 1 else lifted_rank
    elif variant < 0.28:
        axes = axes + [0]
    emit("lift_to_rank", [axes, lifted_rank], run(permute_map.lift_to_rank, axes, lifted_rank))

    if rank and rng.random() < 0.5:
        axes = list(range(rng.randint(0, rank), rank))
    else:
        axes = random_dims(rank)
    preserved = rng.random() < 0.5
    emit(
        "project_to_axes",
        [axes, preserved],
        run(
            lambda: permute_map.project_to_axes(
                axes, require_other_axes_preserved=preserved
            )
        ),
    )


def extend_case() -> None:
    rank = rng.randint(0, MAX_RANK)
    dims = rng.sample(range(rank), rng.randint(0, rank))
    variant = rng.random()
    if dims and variant < 0.1:
        dims.append(dims[0])
    elif variant < 0.15:
        dims.append(rank + rng.randint(0, 2))
    emit("extend_permutation", [dims, rank], run(_extend_permutation, dims, rank))


def main() -> None:
    for source, target in EDGE_VIEWS:
        view_case(source, target, None)
    for _ in range(VIEW_CASES):
        view_case(*random_view_shapes())
    for _ in range(PERMUTE_CASES):
        permute_case()
    for _ in range(EXTEND_CASES):
        extend_case()
    with open(dim_maps.__file__, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    sys.stdout.write(
        "# rc4 dim_maps results; regenerate with dim_maps_rc4_vectors.py "
        f"(seed {SEED}).\n"
        f"# backends/transforms/dim_maps.py sha256 {digest}\n"
        "# query<TAB>JSON args<TAB>JSON result. `view` and `permute` lines open a\n"
        "# case that the following lines query. null is Python's None; a string\n"
        "# is the exception the call raised.\n"
    )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
