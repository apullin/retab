#!/usr/bin/env python3
"""Print high-signal artifact differences for a named model."""

from __future__ import annotations

import argparse
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

from lib.tosa_compare import compare_tosa_artifacts, run_single_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference = run_single_path(args.model, None, seed=args.seed)
    direct = run_single_path(args.model, "all_rust", seed=args.seed)

    if reference.get("status") != "success":
        print("reference_failed", reference.get("error"))
        return 2
    if direct.get("status") != "success":
        print("direct_failed", direct.get("error"))
        return 3

    comparison = compare_tosa_artifacts(reference, direct)
    print("reference_ops", reference["parsed"]["op_count"])
    print("direct_ops", direct["parsed"]["op_count"])
    print("reference_tensors", reference["parsed"]["tensor_count"])
    print("direct_tensors", direct["parsed"]["tensor_count"])
    print("bytes_equal", comparison["bytes_equal"])
    print("op_sequence_equal", comparison["op_sequence_equal"])
    print("compute_sequence_equal", comparison["compute_sequence_equal"])
    print("common_tensors", comparison["common_tensors"])
    print("same_payload", comparison["same_payload"])
    print("different_payload", comparison["different_payload"])

    op_count_diffs = {
        name: counts
        for name, counts in comparison["op_counts"].items()
        if counts["reference"] != counts["direct"]
    }
    print("op_count_diffs")
    for name, counts in sorted(
        op_count_diffs.items(),
        key=lambda item: (
            abs(item[1]["reference"] - item[1]["direct"]),
            item[0],
        ),
        reverse=True,
    )[:20]:
        print(name, counts)

    ref_compute = [
        op["name"]
        for op in reference["parsed"]["ops"]
        if op["name"] not in {"CONST", "CONST_SHAPE"}
    ]
    dir_compute = [
        op["name"]
        for op in direct["parsed"]["ops"]
        if op["name"] not in {"CONST", "CONST_SHAPE"}
    ]
    mismatch_idx = None
    for idx, (ref_op, dir_op) in enumerate(zip(ref_compute, dir_compute)):
        if ref_op != dir_op:
            mismatch_idx = idx
            break
    if mismatch_idx is None and len(ref_compute) != len(dir_compute):
        mismatch_idx = min(len(ref_compute), len(dir_compute))
    print("first_compute_mismatch", mismatch_idx)
    if mismatch_idx is not None:
        start = max(0, mismatch_idx - 5)
        end = mismatch_idx + 6
        print("reference_window", ref_compute[start:end])
        print("direct_window", dir_compute[start:end])
        print("reference_window_detail")
        ref_ops = [
            op
            for op in reference["parsed"]["ops"]
            if op["name"] not in {"CONST", "CONST_SHAPE"}
        ]
        dir_ops = [
            op
            for op in direct["parsed"]["ops"]
            if op["name"] not in {"CONST", "CONST_SHAPE"}
        ]
        for idx in range(start, min(end, len(ref_ops))):
            print(
                idx,
                ref_ops[idx]["name"],
                ref_ops[idx]["inputs"],
                ref_ops[idx]["outputs"],
            )
        print("direct_window_detail")
        for idx in range(start, min(end, len(dir_ops))):
            print(
                idx,
                dir_ops[idx]["name"],
                dir_ops[idx]["inputs"],
                dir_ops[idx]["outputs"],
            )

    print("meta_diffs")
    for diff in comparison["meta_diffs"][:20]:
        print(diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
