"""Fetch the author-provided ONNX export of the S2 classifier via single-file
hf_hub_download (avoids snapshot_download, which crashes in this env), and write
classifier_meta.json so the runtime picks the correct AI class index."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import hf_hub_download

REPO = "Organika/sdxl-detector"
PINNED = "657a8bf7e4daee1a067ffbb3e5a5937172088f12"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

AI_WORDS = ("artificial", "ai", "fake", "synthetic", "generated", "deepfake", "machine", "non-photographic")


def detect_ai_index(id2label):
    for idx, name in (id2label or {}).items():
        if any(w in str(name).lower() for w in AI_WORDS):
            return int(idx)
    return 1


def try_download(filename, revisions):
    last_err = None
    for rev in revisions:
        try:
            p = hf_hub_download(repo_id=REPO, filename=filename, revision=rev)
            print(f"  {filename}: OK from revision {rev[:12]} -> {p}", flush=True)
            return p, rev
        except Exception as e:
            print(f"  {filename}: revision {str(rev)[:12]} FAILED: {type(e).__name__}: {e}", flush=True)
            last_err = e
    raise last_err


revisions = [PINNED, "main"]
print("Downloading S2 ONNX files...", flush=True)
onnx_path, used_rev = try_download("onnx/model.onnx", revisions)
cfg_path, _ = try_download("onnx/config.json", revisions)
prep_path, _ = try_download("onnx/preprocessor_config.json", revisions)

# Copy ONNX into models/classifier.onnx
import shutil
dst_onnx = os.path.join(MODEL_DIR, "classifier.onnx")
shutil.copyfile(onnx_path, dst_onnx)
print(f"  wrote {dst_onnx} ({os.path.getsize(dst_onnx)} bytes)", flush=True)

# Build classifier_meta.json from config id2label
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
ai_idx = detect_ai_index(id2label)
meta = {"id2label": id2label, "ai_class_index": ai_idx, "source_revision": used_rev}
meta_path = os.path.join(MODEL_DIR, "classifier_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"  wrote {meta_path}: {meta}", flush=True)

# Save preprocessor config for parity reference
shutil.copyfile(prep_path, os.path.join(MODEL_DIR, "classifier_preprocessor_config.json"))
print("DONE", flush=True)
