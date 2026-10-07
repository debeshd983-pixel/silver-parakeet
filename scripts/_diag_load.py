import os, sys, faulthandler
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
faulthandler.enable()
faulthandler.dump_traceback_later(120, exit=False)
REPO = "Organika/sdxl-detector"
REV = "657a8bf7e4daee1a067ffbb3e5a5937172088f12"

print("STEP-A: snapshot_download", flush=True)
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id=REPO, revision=REV, local_dir="models/raw/classifier")
print("  downloaded to", local, flush=True)
print("  files:", os.listdir(local), flush=True)

print("STEP-B: importing torch/transformers", flush=True)
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
print("  torch", torch.__version__, flush=True)

print("STEP-C: image processor", flush=True)
ip = AutoImageProcessor.from_pretrained(local)
print("  ok", flush=True)

print("STEP-D: load model local_files_only", flush=True)
model = AutoModelForImageClassification.from_pretrained(local, local_files_only=True)
model.eval()
print("  loaded; id2label", model.config.id2label, flush=True)

print("STEP-E: forward", flush=True)
x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
with torch.no_grad():
    out = model(x)
print("  logits", tuple(out.logits.shape), out.logits.flatten().tolist()[:4], flush=True)
print("DONE", flush=True)
