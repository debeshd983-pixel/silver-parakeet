"""Run the real /deep-guard/detect endpoint (in-process via TestClient) over the labeled
images in assets/ai (AI) and assets/rl (real), and emit a per-image table plus
accuracy/FPR/abstain metrics. Writes results to analysis_results.json."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from app.main import app

LABELS = {"ai": 1, "rl": 0}  # ai = AI-generated, rl = real-life photo

def iter_images(root):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            yield p

def mime_for(path):
    if path.lower().endswith(".png"):
        return "image/png"
    if path.lower().endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if path.lower().endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"

rows = []
with TestClient(app) as client:
    # readiness sanity
    ready = client.get("/deep-guard/readyz").json()
    ver = client.get("/deep-guard/version").json()
    print("readyz:", ready)
    print("version:", json.dumps(ver, indent=2))

    for folder, label in (("ai", 1), ("rl", 0)):
        root = os.path.join("assets", folder)
        if not os.path.isdir(root):
            continue
        for path in iter_images(root):
            with open(path, "rb") as f:
                data = f.read()
            files = {"file": (os.path.basename(path), data, mime_for(path))}
            r = client.post("/deep-guard/detect", files=files)
            rec = {
                "file": f"{folder}/{os.path.basename(path)}",
                "label": "ai" if label == 1 else "real",
                "label_int": label,
                "size_bytes": len(data),
                "status": r.status_code,
            }
            if r.status_code == 200:
                d = r.json()
                rec.update({
                    "verdict": d["verdict"],
                    "ai_probability": d["ai_probability"],
                    "confidence": d["confidence"],
                    "p_cls": d["signals"]["classifier"]["probability"],
                    "p_clip": d["signals"]["clip_probe"]["probability"],
                    "c2pa_present": d["signals"]["c2pa"]["present"],
                    "warnings": d["warnings"],
                    "latency_ms": d["latency_ms"],
                })
            else:
                rec["error"] = r.json()
            rows.append(rec)
            print(json.dumps(rec, indent=2))

with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "version": ver}, f, indent=2)
print("\nWrote analysis_results.json with", len(rows), "rows")
