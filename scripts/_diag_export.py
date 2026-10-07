"""Step-by-step ONNX export diagnostic.

Each heavy import and model call is intentionally kept at its own STEP so a
failure points at the exact step that broke. That ordering is the point of the
script, hence the E402 waivers on the mid-file imports.
"""
import os
import traceback

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
REPO = "Organika/sdxl-detector"
REV = "657a8bf7e4daee1a067ffbb3e5a5937172088f12"

print("STEP1: importing torch/transformers", flush=True)
import torch  # noqa: E402
from transformers import AutoModelForImageClassification  # noqa: E402

print("  torch", torch.__version__, flush=True)

print("STEP2: loading model", flush=True)
model = AutoModelForImageClassification.from_pretrained(REPO, revision=REV)
model.eval()
print("  loaded; id2label", model.config.id2label, flush=True)

print("STEP3: forward pass", flush=True)
x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
with torch.no_grad():
    out = model(x)
print(
    "  logits shape",
    tuple(out.logits.shape),
    "sample",
    out.logits.flatten().tolist()[:4],
    flush=True,
)

print("STEP4: torch.onnx.export", flush=True)
try:
    torch.onnx.export(
        model,
        x,
        "models/_probe.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print("  EXPORT OK", os.path.getsize("models/_probe.onnx"), "bytes", flush=True)
except Exception:
    traceback.print_exc()
    print("  EXPORT FAILED via torch.onnx.export", flush=True)

print("DONE", flush=True)