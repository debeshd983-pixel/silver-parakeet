"""Sanity check: run the real ClassifierService on 1 AI + 1 real image to
confirm the ONNX loads and the AI/real direction is correct."""
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, ".")

# Imported after the progress-bar env var and sys.path are set, hence the E402 waivers.
from app.config import get_settings  # noqa: E402
from app.services.classifier import ClassifierService  # noqa: E402
from app.services.preprocess import decode_image, preprocess_for_classifier  # noqa: E402

s = get_settings()
svc = ClassifierService(s)
svc.load()
print("ai_class_index:", svc.ai_class_index, "| session:", svc.session is not None, flush=True)

samples = [
    ("AI  ", r"assets\ai\images (1).jpg"),
    ("REAL", r"assets\rl\IMG-20260411-WA0029(1).jpg"),
]
for label, path in samples:
    with open(path, "rb") as f:
        data = f.read()
    img, warnings = decode_image(data)
    t = preprocess_for_classifier(img)
    p = svc.predict(t)
    print(f"  {label} {os.path.basename(path):28s} size={img.size} p_ai={p:.4f} warnings={warnings}", flush=True)
print("DONE", flush=True)
