#!/usr/bin/env python3
"""Compare reference and direct lowering numerically for a named model."""

from __future__ import annotations

import argparse
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

from lib.registry import get_model_and_example  # noqa: E402
from lib.tosa_compare import run_single_path  # noqa: E402


def _load_ref_model():
    try:
        from tosa_tools.tosa_reference_model import tosa_reference_model as ref
    except ImportError:
        from tosa_reference_model import tosa_reference_model as ref
    return ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pattern")
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()

    _, example = get_model_and_example(args.model)
    input_np = np.clip(
        example.detach().cpu().numpy() * 127, -128, 127
    ).astype(np.int8)

    direct = run_single_path(args.model, "all_rust", seed=args.seed)
    print("direct_status", direct.get("status"))
    if direct.get("status") != "success":
        print("direct_error", direct.get("error"))
        return 3

    reference = None
    if not args.direct_only:
        reference = run_single_path(args.model, None, seed=args.seed)
        print("reference_status", reference.get("status"))
        if reference.get("status") != "success":
            print("reference_error", reference.get("error"))
            return 2

    if args.pattern:
        print("PATTERN", args.pattern)
        artifacts = [("direct", direct)]
        if reference is not None:
            artifacts.insert(0, ("reference", reference))
        for label, artifact in artifacts:
            print(f"{label}_tensors")
            for tensor in artifact["parsed"]["tensors"]:
                if args.pattern in tensor["name"]:
                    print(
                        "tensor",
                        tensor["name"],
                        tensor["shape"],
                        tensor["dtype_name"],
                        tensor["data_len"],
                    )
            print(f"{label}_ops")
            for op in artifact["parsed"]["ops"]:
                if any(args.pattern in name for name in op["inputs"] + op["outputs"]):
                    print("op", op["name"], op["inputs"], op["outputs"])

    if args.artifact_only:
        return 0

    ref = _load_ref_model()
    if reference is None:
        _, example = get_model_and_example(args.model)
        input_np = np.clip(
            example.detach().cpu().numpy() * 127, -128, 127
        ).astype(np.int8)
        dir_out, dir_status = ref.run(
            direct["tosa_bytes"],
            [input_np],
            initialize_variable_tensor_from_numpy=True,
        )
        print("direct_refmodel_status", dir_status)
        print("direct_output_count", len(dir_out))
        return 0

    ref_out, ref_status = ref.run(
        reference["tosa_bytes"],
        [input_np],
        initialize_variable_tensor_from_numpy=True,
    )
    dir_out, dir_status = ref.run(
        direct["tosa_bytes"],
        [input_np],
        initialize_variable_tensor_from_numpy=True,
    )

    print("reference_refmodel_status", ref_status)
    print("direct_refmodel_status", dir_status)
    if len(ref_out) != len(dir_out):
        print("output_count_mismatch", len(ref_out), len(dir_out))
        return 4

    max_diff = 0.0
    for ref_tensor, dir_tensor in zip(ref_out, dir_out):
        diff = float(
            np.max(
                np.abs(
                    ref_tensor.astype(np.float64) - dir_tensor.astype(np.float64)
                )
            )
        )
        max_diff = max(max_diff, diff)

    print("GLOBAL_MAX_DIFF", max_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
