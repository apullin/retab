#!/usr/bin/env python3
"""Test that FX graph ingestion works."""
import sys
sys.path.insert(0, "/Users/andrewpullin/personal/executorch-work/u55-pipeline")
from model import get_model
import torch

# Export and quantize tiny_cnn (reuse lower_cnn.py logic)
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

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

# Now test ingestion
import retab
result = retab.ingest_graph(
    quantized_ep.graph_module,
    dict(quantized_ep.state_dict),
    quantized_ep.graph_signature,
)
print(result)
# Should print something like "Ingested 267 nodes, 68 tensors"
