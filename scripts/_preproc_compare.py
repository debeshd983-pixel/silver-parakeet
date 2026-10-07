"""Compare the service's squash-resize preprocessing vs transformers' actual
AutoImageProcessor on the ONNX model, to find whether a preprocessing mismatch
explains the real-image false positive."""
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, ".")
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor
from app.services.preprocess import decode_image, preprocess_for_classifier

sess = ort.InferenceSession("models/classifier.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name
proc = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
print("processor:", type(proc).__name__)
print("proc.size:", getattr(proc, "size", None), "| crop_size:", getattr(proc, "crop_size", None),
      "| do_center_crop:", getattr(proc, "do_center_crop", None), "| do_resize:", getattr(proc, "do_resize", None))

def softmax(l):
    e = np.exp(l - np.max(l)); return e / e.sum()

def onnx(t):
    out = sess.run([oname], {iname: t.astype(np.float32)})[0]
    return softmax(out[0])

samples = [
    ("AI  ", r"assets\ai\images (1).jpg"),
    ("REAL", r"assets\rl\IMG-20260411-WA0029(1).jpg"),
]
for label, path in samples:
    img = Image.open(path).convert("RGB")
    # service preprocessing (squash to 224x224)
    t_svc = preprocess_for_classifier(img)
    p_svc = onnx(t_svc)
    # correct HF preprocessing
    enc = proc(images=img, return_tensors="np")
    t_hf = enc["pixel_values"]
    p_hf = onnx(t_hf)
    print(f"\n{label} {os.path.basename(path)} size={img.size}")
    print(f"  service(squash) shape={t_svc.shape} P(artificial)={p_svc[0]:.4f} P(human)={p_svc[1]:.4f}")
    print(f"  HF(processor) shape={t_hf.shape}   P(artificial)={p_hf[0]:.4f} P(human)={p_hf[1]:.4f}")
