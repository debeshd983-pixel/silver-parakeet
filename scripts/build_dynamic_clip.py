"""Builds the dynamically-quantized CLIP encoder at a chosen path.

Dynamic quantization cannot be produced by ``scripts/export_clip_onnx.py`` any more: that
script now defaults to static QDQ because dynamic quantization is batch-dependent (it
computes activation clipping ranges per run, so the same image embeds differently alone
than beside another -- measured cosine 0.65-0.89).

It is kept available because for this particular service dynamic INT8 is still the better
trade: the service serves exactly one image per request, so the batch-dependence never
bites, and dynamic INT8 scores AUC ~0.72 as served where every batch-safe static variant
scored 0.56-0.59. The head must be fitted with ``--embed-batch 1`` to match.

The single hard constraint: 83.5 MB fits inside PyPI's 100 MB per-file limit, while the
fp32 encoder at 329 MB does not and would have to be distributed as a GitHub Release
asset instead.

Build-time only.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

FP32 = Path("models/clip_encoder.onnx")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "models/clip_encoder.dyn.int8.onnx")

if not FP32.is_file():
    print(f"ERROR: {FP32} not found. Run scripts/export_clip_onnx.py --no-quantize first.")
    raise SystemExit(1)

print(f"quantizing {FP32} -> {OUT} (dynamic INT8, MatMul/Attention only)", flush=True)
# Conv is excluded: the patch-embedding ConvInteger has no CPU kernel, and including it
# produces a model that saves fine but fails to load.
quantize_dynamic(
    model_input=str(FP32),
    model_output=str(OUT),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    op_types_to_quantize=["MatMul", "Attention"],
    extra_options={"MatMulConstBOnly": True},
)
print(f"wrote {OUT} ({OUT.stat().st_size / 1048576:.1f} MB)", flush=True)

sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
o = sess.run(None, {inp: np.zeros((1, 3, 224, 224), np.float32)})[0]
print(f"loads OK, output shape {o.shape}", flush=True)
print("\nnext: python scripts/train_clip_probe.py --encoder "
      f"{OUT} --embed-batch 1", flush=True)