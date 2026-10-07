"""Inspect raw ONNX logits for 1 AI + 1 real image to determine the true AI class index."""
import os, sys, json
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, ".")
import numpy as np
import onnxruntime as ort
from app.services.preprocess import decode_image, preprocess_for_classifier

sess = ort.InferenceSession("models/classifier.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
print("meta:", json.load(open("models/classifier_meta.json")))

def softmax(l):
    e = np.exp(l - np.max(l))
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
    l = out[0]
    p = softmax(l)
    print(f"\n{label} {os.path.basename(path)} size={img.size}")
    print(f"  logits = {l.tolist()}")
    print(f"  probs   = {p.tolist()}  (sum={p.sum():.4f})")
    print(f"  P(class0=artificial)={p[0]:.4f}  P(class1=human)={p[1]:.4f}")
