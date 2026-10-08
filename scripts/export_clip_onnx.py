"""Exports the frozen CLIP ViT-B/16 image encoder to ONNX for the S3 signal.

Why this exists
---------------
``architecture.md`` 3.1 designs the ensemble around two signals that "fail on different
images". Only one exists today: the S2 classifier. The S3 CLIP probe was disabled
because its head had been fitted on ``np.random.randn`` (see MODEL_CARD 1) -- it
contributed noise.

This script produces the missing *encoder*. Fitting the head honestly is
``scripts/train_clip_probe.py``, which needs a real benchmark to fit on.

Why CLIP for the unseen-generator case
--------------------------------------
The shipped S2 checkpoint saturates on the families it was trained near (dalle3,
midjourney-v5, stable-diffusion-xl all score a constant 1.000) and collapses on the rest
(glide 5.9%, FLUX.1-dev 13.2%). A linear probe on frozen CLIP features (the UnivFD
recipe) is the standard technique for generalising to generators a detector never saw,
because it keys on semantic texture statistics rather than generator fingerprints. That is
precisely the axis S2 fails on, which is what makes it a genuine second signal rather
than a second copy of the first.

Build-time only. torch is deliberately absent from requirements.txt.

Usage
-----
    python scripts/export_clip_onnx.py
    python scripts/export_clip_onnx.py --quantize-int8
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from onnxruntime.quantization import CalibrationMethod

MODEL_REPO = "openai/clip-vit-base-patch16"
# Pinned by commit hash, never by `main`: architecture.md 7 requires a pinned revision.
MODEL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
# The Hub metadata declares no licence for this repo. Recorded verbatim rather than
# assumed; the OpenAI model card states MIT. Worth resolving before commercial release.
MODEL_LICENSE = "undeclared-on-hub (upstream model card states MIT)"

OUT_DIR = Path("models")
FP32_NAME = "clip_encoder.onnx"
INT8_NAME = "clip_encoder.int8.onnx"
# Expected by sahu65/services/clip_probe.py: one 512-dim unit-norm image embedding.
EMBED_DIM = 512


def log(msg: str) -> None:
    print(msg, flush=True)


def _calibration_reader(manifest: Path, limit: int):
    """Yields preprocessed images for static-quantization calibration.

    Uses the exact serve-time preprocessing (shortest side to 224, centre crop, CLIP
    mean/std), because activation ranges calibrated on a different input distribution
    do not describe what the model will actually see.
    """
    import numpy as np
    from PIL import Image

    from sahu65.services.preprocess import CLIP_MEAN, CLIP_STD

    root = manifest.parent
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    rows = [r for r in rows if r["perturbation"] == "original"][:limit]

    batch = []
    for r in rows:
        try:
            with Image.open(root / r["path"]) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = 224 / min(w, h)
                im = im.resize((max(int(round(w * scale)), 224),
                                max(int(round(h * scale)), 224)),
                               Image.Resampling.BICUBIC)
                left = (im.size[0] - 224) // 2
                top = (im.size[1] - 224) // 2
                im = im.crop((left, top, left + 224, top + 224))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
                batch.append(((arr - CLIP_MEAN) / CLIP_STD).astype(np.float32))
        except Exception:
            continue
    return batch


def static_quantize(fp32_path: Path, out_path: Path, manifest: Path,
                    n_calib: int = 64) -> bool:
    """Static (QDQ) quantization with calibrated activation ranges.

    This exists because *dynamic* quantization is unusable here. Dynamic quantization
    computes each activation's clipping range at run time from the observed tensor, so
    for a batch of N different images every image's embedding depends on which images it
    happened to share the batch with. Measured on this encoder: the same image embedded
    alone versus batched with a different image gives cosine 0.65-0.89.

    That silently invalidates any offline measurement. A benchmark scored in batches of 16
    measures a model that never serves an image that way: single-image requests all land
    at different embeddings. This repository hit exactly that -- a probe head fitted on
    batch-16 embeddings scored AUC 0.87 offline and 0.72 as actually served.

    Static quantization bakes the ranges into the graph, which makes the encoder
    deterministic and batch-invariant.
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    tensors = _calibration_reader(manifest, n_calib)
    if len(tensors) < 8:
        log(f"  not enough calibration images ({len(tensors)}); skipping static quant")
        return False
    log(f"  calibrating on {len(tensors)} images")

    class Reader(CalibrationDataReader):
        """Serves one image per call, keyed by the model's input name.

        ORT requires a dict here; handing it a bare array raises
        "truth value of an array is ambiguous" from deep inside the calibrator.
        """

        INPUT_NAME = "pixel_values"

        def __init__(self, data):
            self.data = data
            self._it = iter(data)

        def get_next(self):
            item = next(self._it, None)
            return None if item is None else {self.INPUT_NAME: item}

        def rewind(self):
            self._it = iter(self.data)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=Reader(tensors),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    # Verify the property we actually care about: batch invariance.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def run(ts):
        out = sess.run(None, {inp: np.concatenate(ts, axis=0)})[0]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    solo = run([tensors[0]])[0]
    batched = run(tensors[:16])[0]
    cos = float(np.dot(solo, batched))
    log(f"  batch-invariance check: cosine(solo, in-batch-16) = {cos:.6f}")
    if cos < 0.999:
        log(f"  WARNING: static model is still batch-dependent (cos {cos:.6f}).")
        log("           Do not trust any measurement made on it.")
        return False
    log(f"  wrote {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB), "
        f"batch-invariant")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--quantize-int8", action="store_true", default=True)
    ap.add_argument("--no-quantize", dest="quantize_int8", action="store_false")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--manifest", default="benchmark/manifest.csv",
                    help="images used to calibrate static activation ranges")
    ap.add_argument("--calibration-images", type=int, default=64)
    args = ap.parse_args(argv)

    try:
        import torch
        from transformers import CLIPVisionModelWithProjection
    except ImportError as exc:
        log(f"ERROR: torch/transformers required at build time ({exc})")
        log("  pip install -r requirements-build.txt")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / FP32_NAME

    log(f"loading {MODEL_REPO}@{MODEL_REVISION[:7]}")
    model = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=torch.float32)
    model.eval()

    config = model.config
    log(f"  image_size={config.image_size} hidden={config.hidden_size} "
        f"projection_dim={config.projection_dim}")
    if config.projection_dim != EMBED_DIM:
        log(f"  WARNING: projection_dim is {config.projection_dim}, but "
            f"sahu65/services/clip_probe.py expects {EMBED_DIM}. "
            f"Adjust EMBED_DIM before shipping.")

    dummy = torch.zeros(1, 3, config.image_size, config.image_size, dtype=torch.float32)

    log(f"exporting ONNX (opset {args.opset}) -> {fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    size_mb = fp32_path.stat().st_size / (1024 * 1024)
    log(f"  wrote fp32 ONNX ({size_mb:.1f} MB)")

    # Verify the export before anything depends on it. An unverified export is how a
    # silently-wrong embedding reaches a fitted head.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.run(None, {inp: dummy.numpy()})[0]
        if out.shape[-1] != EMBED_DIM:
            log(f"  FAIL: embedding dim {out.shape[-1]} != {EMBED_DIM}")
            return 1
        norms = np.linalg.norm(out, axis=-1)
        log(f"  verified: output {out.shape}, L2 norms {norms.round(4).tolist()}")
    except Exception as exc:
        log(f"  FAIL: could not verify the exported model: {type(exc).__name__}: {exc}")
        return 1

    if not args.quantize_int8:
        log("skipping INT8 (--no-quantize)")
        return 0

    log("static (QDQ) INT8 quantization -> " + str(out_dir / INT8_NAME))
    log("  dynamic quantization is NOT used: it computes activation ranges per run, which")
    log("  makes an image's embedding depend on its batch-mates (measured cosine 0.65-0.89")
    log("  for the same image alone vs batched). See static_quantize().")
    manifest = Path(args.manifest)
    ok = False
    if manifest.is_file():
        try:
            ok = static_quantize(fp32_path, out_dir / INT8_NAME, manifest,
                                 args.calibration_images)
        except Exception as exc:
            log(f"  static quantization failed: {type(exc).__name__}: {exc}")
    else:
        log(f"  {manifest} not found; run build_benchmark.py to calibrate static ranges")

    if not ok:
        bad = out_dir / INT8_NAME
        if bad.exists():
            bad.unlink()
            log("  removed the unusable INT8 file so it can never be shipped by accident")
        log("  fp32 remains usable and is batch-invariant, but is 329 MB.")
        log("  Train the head with: python scripts/train_clip_probe.py "
            "--encoder models/clip_encoder.onnx")

    log("")
    log("Next: python scripts/train_clip_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())