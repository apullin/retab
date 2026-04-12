#!/usr/bin/env python3
"""Run the direct FX pipeline for a model and inspect matching nodes."""

from __future__ import annotations

import argparse
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

from lib.tosa_compare import prepare_quantized_ep  # noqa: E402


def _shape_from_meta(node) -> object:
    val = node.meta.get("val")
    if val is None:
        return None
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    try:
        return list(shape)
    except TypeError:
        return shape


def _dtype_from_meta(node) -> object:
    val = node.meta.get("val")
    return getattr(val, "dtype", None) if val is not None else None


def _qparam_keys(node, key: str) -> object:
    value = node.meta.get(key)
    if value is None:
        return None
    if hasattr(value, "keys"):
        try:
            return list(value.keys())
        except Exception:
            return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qep, _ = prepare_quantized_ep(args.model, seed=args.seed)
    gm = qep.graph_module

    from retab import _native as rust

    try:
        rust.run_fx_passes(gm, qep)
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else ""
        print("PIPELINE_ERROR", type(exc).__name__, message)

    patterns = args.pattern or ["rescale", "fused_const"]
    for node in gm.graph.nodes:
        if not any(pattern in node.name for pattern in patterns):
            continue
        print("NODE", node.name)
        print("  op", node.op)
        print("  target", node.target)
        print("  meta_shape", _shape_from_meta(node))
        print("  meta_dtype", _dtype_from_meta(node))
        print("  input_qparams", _qparam_keys(node, "input_qparams"))
        print("  output_qparams", _qparam_keys(node, "output_qparams"))
        print("  args", node.args)
        if node.op == "get_attr":
            tensor = getattr(gm, node.target)
            print("  attr_shape", list(tensor.shape))
            print("  attr_dtype", tensor.dtype)
        users = list(node.users)
        print("  users", [user.name for user in users])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
