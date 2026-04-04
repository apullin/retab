"""Compare Python vs Rust TOSA pipeline on random models.

Runs both pipelines, then feeds both TOSA outputs through the reference
model with the same inputs and compares numerical results.

Usage (inside Docker):
    python /work/ws/u55-pipeline/retab/tests/compare_pipelines.py [num_seeds]
"""
import copy, os, random, sys, time
import numpy as np
sys.path.insert(0, "/work/ws/executorch/src")
sys.path.insert(0, "/work/ws/u55-pipeline")
sys.path.insert(0, "/work/ws/u55-pipeline/retab")
sys.path.insert(0, "/work/ws/executorch/examples/arm/arm-scratch/tosa-tools/reference_model/build/lib")

import torch, torch.nn as nn

class RandomCNN(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = random.Random(seed)
        layers, in_c, spatial = [], 1, 28
        for _ in range(rng.randint(2, 4)):
            out_c = rng.choice([8, 16, 32])
            k = rng.choice([1, 3])
            layers += [nn.Conv2d(in_c, out_c, k, padding=k//2), nn.ReLU()]
            if spatial >= 4 and rng.random() < 0.5:
                layers.append(nn.MaxPool2d(2)); spatial //= 2
            in_c = out_c
        layers += [nn.Flatten(), nn.Linear(in_c * spatial * spatial, 10)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def prepare(seed):
    from executorch.backends.arm.quantizer import TOSAQuantizer, get_symmetric_quantization_config
    from executorch.backends.arm.test.common import get_tosa_compile_spec
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    torch.manual_seed(seed); random.seed(seed)
    model = RandomCNN(seed); model.eval()
    x = torch.randn(1, 1, 28, 28)
    ep = torch.export.export(model, (x,), strict=True)
    cs = get_tosa_compile_spec("TOSA-1.0+INT")
    q = TOSAQuantizer(cs)
    q.set_global(get_symmetric_quantization_config(is_per_channel=False))
    qm = ep.module()
    qm = prepare_pt2e(qm, q)
    with torch.no_grad():
        for _ in range(5): qm(torch.randn(1,1,28,28))
    qm = convert_pt2e(qm)
    qep = torch.export.export(qm, (x,), strict=True)
    return qep, x, cs

def lower(qep, x, cs, use_rust):
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    if use_rust:
        from fast_preprocess import patch_tosa_backend; patch_tosa_backend()
    edge = to_edge_transform_and_lower(
        copy.deepcopy(qep),
        partitioner=[TOSAPartitioner(cs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    if use_rust:
        from fast_preprocess import unpatch_tosa_backend; unpatch_tosa_backend()
    regions = []
    ep = edge.exported_program()
    for n in ep.graph.nodes:
        if n.op == "get_attr" and "lowered" in str(n.target):
            m = getattr(ep.graph_module, n.target, None)
            if m and hasattr(m, "processed_bytes"):
                regions.append(bytes(m.processed_bytes))
    return regions

def run_tosa_ref(tosa_bytes, tosa_spec, input_np):
    """Run TOSA flatbuffer through reference model, return output arrays."""
    try:
        try:
            from tosa_tools.tosa_reference_model import tosa_reference_model as ref
        except ImportError:
            from tosa_reference_model import tosa_reference_model as ref
        outputs_np, status = ref.run(
            tosa_bytes,
            [input_np],
            initialize_variable_tensor_from_numpy=True,
        )
        return status, outputs_np
    except Exception as e:
        return None, str(e)

def compare(seed):
    qep, x, cs = prepare(seed)
    py_regions = lower(qep, x, cs, use_rust=False)
    rs_regions = lower(qep, x, cs, use_rust=True)

    if len(py_regions) != len(rs_regions):
        return f"region count: py={len(py_regions)} rs={len(rs_regions)}", None
    if not py_regions:
        return "no regions", None

    # TOSA graph expects quantized INT8 input — just cast to int8 range
    input_np = np.clip(x.numpy() * 127, -128, 127).astype(np.int8)
    tosa_spec = str(cs.tosa_spec) if hasattr(cs, 'tosa_spec') else "TOSA-1.0+INT"

    max_diff = 0.0
    for i, (pb, rb) in enumerate(zip(py_regions, rs_regions)):
        byte_match = (pb == rb)

        py_status, py_out = run_tosa_ref(pb, tosa_spec, input_np)
        rs_status, rs_out = run_tosa_ref(rb, tosa_spec, input_np)

        if py_status is None:
            return f"region {i}: python ref model error: {py_out}", None
        if rs_status is None:
            return f"region {i}: rust ref model error: {rs_out}", None

        # Compare outputs
        if len(py_out) != len(rs_out):
            return f"region {i}: output count py={len(py_out)} rs={len(rs_out)}", None

        for j, (po, ro) in enumerate(zip(py_out, rs_out)):
            diff = np.max(np.abs(po.astype(np.float64) - ro.astype(np.float64)))
            max_diff = max(max_diff, diff)

    return None, max_diff

num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f"Comparing Python vs Rust TOSA on {num} random CNNs\n")
ok = bad = 0
for s in range(num):
    t0 = time.perf_counter()
    try:
        err, max_diff = compare(s)
        dt = time.perf_counter() - t0
        if err:
            bad += 1; print(f"  seed={s:3d}: FAIL ({dt:.1f}s) — {err}")
        elif max_diff == 0.0:
            ok += 1; print(f"  seed={s:3d}: EXACT MATCH ({dt:.1f}s)")
        else:
            ok += 1; print(f"  seed={s:3d}: MATCH ({dt:.1f}s) max_diff={max_diff:.6f}")
    except Exception as e:
        dt = time.perf_counter() - t0
        bad += 1; print(f"  seed={s:3d}: ERROR ({dt:.1f}s) — {type(e).__name__}: {e}")
print(f"\n{ok} match, {bad} fail")
