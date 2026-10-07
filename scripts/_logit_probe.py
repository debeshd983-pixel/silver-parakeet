"""Inspect raw ONNX logits for 1 AI + 1 real image to determine the true AI class index."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, ".")

# Imported after the progress-bar env var is set, hence the E402 waivers.
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from app.services.preprocess import decode_image, preprocess_for_classifier  # noqa: E402

sess = ort.InferenceSession("models/classifier.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
print("meta:", json.load(open("models/classifier_meta.json")))


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


samples = [
    ("AI  ", r"assets\ai\images (1).jpg"),
    ("REAL", r"assets\rl\IMG-20260411-WA0029(1).jpg"),
]
for label, path in samples:
    with open(path, "rb") as f:
        data = f.read()
    img, _ = decode_image(data)
    t = preprocess_for_classifier(img)
    out = sess.run([oname], {iname: t})[0]
    logits = out[0]
    p = softmax(logits)
    print(f"\n{label} {os.path.basename(path)} size={img.size}")
    print(f"  logits = {logits.tolist()}")
    print(f"  probs   = {p.tolist()}  (sum={p.sum():.4f})")
    print(f"  P(class0=artificial)={p[0]:.4f}  P(class1=human)={p[1]:.4f}")