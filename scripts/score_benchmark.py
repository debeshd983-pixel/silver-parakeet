"""Scores every benchmark image with the real detector and caches the probabilities.

This is the step the old pipeline never actually did. ``fit_fusion.py`` used to
synthesise "model predictions" from ``np.random.beta()``, so the fitted thresholds
described a random number generator rather than this model.

Running the genuine INT8 ONNX classifier over the manifest once and caching the result
keeps fitting and evaluation cheap, reproducible, and honest: every number downstream
traces to a real inference on a real image.

Usage
-----
    python scripts/score_benchmark.py --manifest benchmark/manifest.csv
    python scripts/score_benchmark.py --workers 4          # more ONNX sessions
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def score_one(svc, root: Path, row: Dict[str, str], use_clip: bool) -> Optional[Dict[str, object]]:
    """Scores one manifest row. Returns None if the image cannot be decoded.

    The preprocessing call is the same one the service makes, deliberately: fitting
    thresholds against a different preprocessing path than production would measure a
    model nobody ships. Note that the two detectors want different geometry -- the CLIP
    probe is aspect-preserving with a centre crop, the retired classifier squashed to a
    square -- so this cannot be shared between them.
    """
    from sahu65.services.preprocess import (
        decode_image,
        preprocess_for_classifier,
        preprocess_for_clip,
    )

    path = root / row["path"]
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        img, _warnings = decode_image(data)
        tensor = preprocess_for_clip(img) if use_clip else preprocess_for_classifier(img)
    except Exception:
        return None

    try:
        p = svc.predict(tensor)
    except Exception:
        return None

    out = dict(row)
    out["p_detector"] = round(float(p), 6)
    out["ai_class_index"] = getattr(svc, "ai_class_index", None)
    out["model_id"] = svc.model_id
    return out


def score_all(manifest: Path, out_path: Path, workers: int, detector: str) -> int:
    root = manifest.parent
    from sahu65.config import get_settings

    if detector == "clip":
        from sahu65.services.clip_probe import ClipProbeService

        svc = ClipProbeService(get_settings())
        svc.load()
        svc.warmup()
        log(f"loaded {svc.model_id} ({svc.model_path})")
    else:
        from sahu65.services.classifier import ClassifierService

        svc = ClassifierService(get_settings())
        svc.load()
        svc.warmup()
        log(f"loaded {svc.model_id} (ai_class_index={svc.ai_class_index})")

    use_clip = detector == "clip"
    rows = load_manifest(manifest)
    log(f"scoring {len(rows)} images with the {detector} detector")

    start = time.time()
    results: List[Optional[Dict[str, object]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        mapped = pool.map(lambda r: score_one(svc, root, r, use_clip), rows)
        for i, res in enumerate(mapped, start=1):
            results.append(res)
            if i % 100 == 0 or i == len(rows):
                rate = i / max(time.time() - start, 1e-9)
                log(f"  {i}/{len(rows)} ({rate:.1f} img/s)")

    scored = [r for r in results if r is not None]
    failures = len(results) - len(scored)
    if not scored:
        log("ERROR: nothing scored; refusing to write an empty score file")
        return 1

    fields = list(rows[0].keys()) + ["p_detector", "ai_class_index", "model_id"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in scored:
            writer.writerow(r)

    elapsed = time.time() - start
    p = np.array([r["p_detector"] for r in scored], dtype=float)
    y = np.array([int(r["label"]) for r in scored], dtype=int)
    log("")
    log(f"scored {len(scored)} images in {elapsed:.1f}s ({failures} undecodable)")
    log(f"wrote {out_path}")
    log("")
    log("sanity check of the score distribution (not an accuracy claim):")
    log(f"  real (label 0) n={(y == 0).sum():5d}  mean={p[y == 0].mean():.4f}  "
        f"median={np.median(p[y == 0]):.4f}")
    log(f"  ai   (label 1) n={(y == 1).sum():5d}  mean={p[y == 1].mean():.4f}  "
        f"median={np.median(p[y == 1]):.4f}")

    # Separation is the single most informative thing to check before fitting anything.
    auc = roc_auc(y, p)
    log(f"  AUC over all rows = {auc:.4f}")
    if auc < 0.6:
        log("")
        log("  WARNING: AUC below 0.6 means these scores barely separate the classes.")
        log("  Do NOT fit thresholds on this. Fix the benchmark (labels, format")
        log("  leakage, or preprocessing) before trusting any fitted number.")
    return 0


def roc_auc(y: Sequence[int], p: Sequence[float]) -> float:
    """AUC via the rank-sum identity. Ties get average ranks, so it is exact."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="benchmark/manifest.csv")
    p.add_argument("--out", default=None,
                   help="default: <manifest dir>/scores.csv")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel ONNX sessions (each is memory-hungry: ~500 MB)")
    p.add_argument("--detector", default="clip", choices=["clip", "classifier"],
                   help="which detector to score with (clip = the shipped primary)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        log(f"ERROR: manifest not found at {manifest}")
        log("run scripts/build_benchmark.py first")
        return 1
    out = Path(args.out) if args.out else manifest.parent / "scores.csv"
    try:
        return score_all(manifest, out, args.workers, args.detector)
    except Exception as exc:
        log(f"score_benchmark failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())