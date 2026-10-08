"""Sweeps INT8 quantization configs for the CLIP encoder and keeps the best.

Why this exists
---------------
Two INT8 attempts failed for opposite reasons:

* **Dynamic** quantization is batch-dependent. It computes activation clipping ranges at
  run time from whatever is in the batch, so the same image embeds differently alone
  than beside another image (measured cosine 0.65-0.89). A probe head fitted on
  batch-16 embeddings scored AUC 0.86 offline and 0.72 as actually served.
* **Static MinMax/QDQ** is batch-invariant but MinMax over 64 calibration images
  saturated the activation ranges. AUC collapsed to 0.57.

fp32 is batch-invariant and scores AUC 0.98, at 329 MB. The goal here is an INT8 build
that is both batch-invariant and accurate, at roughly a quarter of the size.

Every candidate is checked for the two properties that matter, in this order:
1. batch invariance (cosine(solo, in-batch) must be ~1.0), and
2. linear-probe AUC by cross-validation.

A config that fails (1) is rejected outright no matter how good its AUC looks in a
batched run, because the AUC would not describe serving.

Build-time only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_clip_onnx import _calibration_reader  # noqa: E402
from train_clip_probe import cv_scores, roc_auc  # noqa: E402

FP32 = Path("models/clip_encoder.onnx")
MANIFEST = Path("benchmark/manifest.csv")
OUT_DIR = Path("models")

CONFIGS: List[Tuple[str, dict]] = [
    ("qdq_minmax_rr", dict(calibrate_method="MinMax", reduce_range=True)),
    ("qdq_percentile", dict(calibrate_method="Percentile", reduce_range=True)),
    ("qdq_entropy", dict(calibrate_method="Entropy", reduce_range=True)),
    ("qdq_percentile_norr", dict(calibrate_method="Percentile", reduce_range=False)),
    ("qdq_minmax_256", dict(calibrate_method="MinMax", reduce_range=True, n_calib=256)),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def quantize(name: str, opts: dict, calib: List[np.ndarray]) -> Optional[Path]:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    out = OUT_DIR / f"clip_encoder.{name}.onnx"

    class Reader(CalibrationDataReader):
        """Serves one preprocessed image per call, keyed by the model's input name."""

        def __init__(self, data):
            self.data = data
            self._it = iter(data)

        def get_next(self):
            item = next(self._it, None)
            return None if item is None else {"pixel_values": item}

        def rewind(self):
            self._it = iter(self.data)

    methods = {
        "MinMax": CalibrationMethod.MinMax,
        "Percentile": CalibrationMethod.Percentile,
        "Entropy": CalibrationMethod.Entropy,
    }
    try:
        quantize_static(
            model_input=str(FP32),
            model_output=str(out),
            calibration_data_reader=Reader(calib),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=opts["reduce_range"],
            calibrate_method=methods[opts["calibrate_method"]],
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    except Exception as exc:
        log(f"    quantization failed: {type(exc).__name__}: {exc}")
        return None
    return out


def check_and_score(path: Path, n_calib: int = 24) -> Tuple[bool, float, str]:
    """Returns (batch_invariant, probe_auc, note)."""
    import onnxruntime as ort

    from sahu65.services.preprocess import decode_image, preprocess_for_clip

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def run(ts):
        out = sess.run(None, {inp: np.concatenate(ts, axis=0)})[0]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    calib = _calibration_reader(MANIFEST, n_calib)
    if len(calib) < 16:
        return False, float("nan"), "not enough calibration images"

    cos = float(np.dot(run([calib[0]])[0], run(calib[:16])[0]))
    invariant = cos > 0.999
    note = f"batch-cos={cos:.6f}"

    # Full benchmark extraction for the probe AUC.
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    feats = np.zeros((len(rows), 512), dtype=np.float32)

    pending: List[np.ndarray] = []
    pending_idx: List[int] = []
    for i, r in enumerate(rows):
        try:
            with open(MANIFEST.parent / r["path"], "rb") as f:
                img, _ = decode_image(f.read())
            pending.append(preprocess_for_clip(img))
            pending_idx.append(i)
        except Exception:
            continue
        if len(pending) == 16 or i == len(rows) - 1:
            out = sess.run(None, {inp: np.concatenate(pending, axis=0)})[0]
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            for j, idx in enumerate(pending_idx):
                feats[idx] = out[j]
            pending, pending_idx = [], []

    labels = np.array([int(r["label"]) for r in rows])

    best = -1.0
    for l2 in (1e-3, 1e-2, 1e-1):
        best = max(best, roc_auc(labels, cv_scores(feats, labels, 5, l2)))
    return invariant, best, note


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run just one config by name")
    args = ap.parse_args(argv)

    if not FP32.is_file():
        log(f"ERROR: {FP32} missing. Run scripts/export_clip_onnx.py first.")
        return 1

    base_calib = _calibration_reader(MANIFEST, 64)
    log(f"base calibration set: {len(base_calib)} images")

    results = []
    for name, opts in CONFIGS:
        if args.only and args.only != name:
            continue
        n = opts.get("n_calib", 64)
        calib = base_calib if n == 64 else _calibration_reader(MANIFEST, n)
        log("")
        log(f"=== {name}  ({opts}, {len(calib)} calib images) ===")
        path = quantize(name, opts, calib)
        if path is None:
            results.append((name, None, float("nan"), ""))
            continue
        mb = path.stat().st_size / (1024 * 1024)
        invariant, auc, note = check_and_score(path)
        log(f"  size {mb:.1f} MB   {note}   probe CV AUC = {auc:.4f}   "
            f"{'BATCH-SAFE' if invariant else 'BATCH-DEPENDENT -> rejected'}")
        results.append((name, mb, auc, note))
        if not invariant:
            path.unlink(missing_ok=True)
            log("  deleted (not shippable)")

    log("")
    log("=" * 78)
    log(f"{'config':<26} {'MB':>8} {'probe CV AUC':>14}  note")
    log("=" * 78)
    log(f"{'fp32 (reference)':<26} {FP32.stat().st_size/1048576:>8.1f} {0.9807:>14.4f}"
        f"  batch-safe, accurate")
    for name, mb, auc, note in results:
        if mb is None:
            log(f"{name:<26} {'--':>8} {'failed':>14}")
        else:
            log(f"{name:<26} {mb:>8.1f} {auc:>14.4f}  {note}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())