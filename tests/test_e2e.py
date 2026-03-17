#!/usr/bin/env python3
"""End-to-end test: compare Rust TOSA output with Python reference."""
import sys
sys.path.insert(0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline")

import torch
from model import get_model
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Prepare quantized model
model, example_input = get_model("tiny_cnn")
exported = torch.export.export(model, (example_input,), strict=True)
compile_spec = EthosUCompileSpec(target="ethos-u55-128")
quantizer = EthosUQuantizer(compile_spec)
quantizer.set_global(get_symmetric_quantization_config())
qmodel = exported.module()
qmodel = prepare_pt2e(qmodel, quantizer)
with torch.no_grad():
    for _ in range(5):
        qmodel(torch.randn_like(example_input))
qmodel = convert_pt2e(qmodel)
quantized_ep = torch.export.export(qmodel, (example_input,), strict=True)

# Run Rust path
import arm_tosa_engine
rust_tosa = arm_tosa_engine.lower_to_tosa(
    quantized_ep.graph_module,
    dict(quantized_ep.state_dict),
    quantized_ep.graph_signature,
)
print(f"Rust TOSA output: {len(rust_tosa)} bytes")

# Compare with reference
ref_path = "/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/test_data/tiny_cnn_reference.tosa"
try:
    with open(ref_path, "rb") as f:
        ref_tosa = f.read()
    print(f"Reference TOSA: {len(ref_tosa)} bytes")

    if rust_tosa == ref_tosa:
        print("PASS: Byte-identical!")
    else:
        print(f"DIFF: Outputs differ ({len(rust_tosa)} vs {len(ref_tosa)} bytes)")
        # Show first difference
        for i in range(min(len(rust_tosa), len(ref_tosa))):
            if rust_tosa[i] != ref_tosa[i]:
                print(f"  First diff at byte {i}: rust={rust_tosa[i]:02x} ref={ref_tosa[i]:02x}")
                break
except FileNotFoundError:
    print(f"Reference file not found at {ref_path}")
    print("Skipping byte comparison — writing Rust output for manual inspection.")
    import os
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    out_path = ref_path.replace("_reference.tosa", "_rust_output.tosa")
    with open(out_path, "wb") as f:
        f.write(rust_tosa)
    print(f"Wrote Rust output to {out_path}")

# Basic sanity: check TOSA file identifier
if len(rust_tosa) >= 8:
    ident = rust_tosa[4:8]
    if ident == b"TOSA":
        print("PASS: File identifier is 'TOSA'")
    else:
        print(f"FAIL: File identifier is {ident!r}, expected b'TOSA'")
else:
    print("FAIL: Output too short to contain file identifier")
