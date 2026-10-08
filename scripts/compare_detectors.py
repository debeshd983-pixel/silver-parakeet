"""Scores alternative detector checkpoints on the benchmark so one can be chosen.

This is the comparison ``plan.md`` P1 asked for and that was never actually run: the
shipped checkpoint was adopted without ever being compared to the alternatives on real
data. The old ``MODEL_CARD.md`` metrics that "justified" the choice came from
``np.random.beta()``.

Why swap a checkpoint at all, when adding a second signal is the usual answer?
Because a swap is the cheap move. Every candidate here is the same shape as the shipped
one -- a small ViT/SigLIP image classifier exported to INT8 ONNX -- so swapping preserves
the entire runtime: same ONNX Runtime, same ~92 MB package budget, same latency, same
preprocessing contract, no new dependency. It can also fix licensing: the shipped
checkpoint is cc-by-nc-3.0 (non-commercial), which is a real problem for a package
published on PyPI.

Requires torch + transformers. Those are BUILD-TIME dependencies only and are
deliberately absent from requirements.txt; the runtime stays onnxruntime.

Usage
-----
    python scripts/compare_detectors.py --scores-only
    python scripts/compare_detectors.py --repo umm-maybe/AI-image-detector
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CANDIDATES: Dict[str, Dict[str, str]] = {
    "umm-maybe/AI-image-detector": {
        "revision": "c7e223baf11bc40528af364ba7bdea030ef42f9e",
        "license": "cc-by-4.0",
        "note": "commercial use permitted",
    },
    "prithivMLmods/Deep-Fake-Detector-v2-Model": {
        "revision": "3a99ae26f52c7ac7c3a53103b6cf3a8b617f7093",
        "license": "apache-2.0",
        "note": "commercial use permitted",
    },
    "prithivMLmods/Deep-Fake-Detector-Model": {
        "revision": "c5cb24c6a159dd2b57ca15c6a1065bd0ce8fa380",
        "license": "apache-2.0",
        "note": "v1; MODEL_CARD reported AUC 0.479 on documents, never on this benchmark",
    },
}

SHIPPED = {
    "Organika/sdxl-detector": {
        "revision": "b37fede8562cb72b89ec201c0987f96ba21b518a",
        "license": "cc-by-nc-3.0",
        "note": "NON-COMMERCIAL; currently shipped",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Metrics (kept local so this script has no import-time dependency on sahu65)
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if not mask.sum():
            continue
        total += (mask.sum() / len(p)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def best_operating_point(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """The FPR/recall trade-off curve, for comparing models on equal terms."""
    reals = np.sort(p[y == 0])
    ais = p[y == 1]
    out = []
    for t in np.unique(np.concatenate([p, [0.5, 0.9, 0.99]])):
        t = float(t)
        out.append({"T": t,
                    "fpr": float((reals >= t).mean()),
                    "recall": float((ais >= t).mean())})
    # Report the best achievable FPR at >=50% recall, a model-comparison-neutral point.
    eligible = [r for r in out if r["recall"] >= 0.50]
    best = min(eligible, key=lambda r: r["fpr"]) if eligible else min(
        out, key=lambda r: -r["recall"])
    return {"fpr_at_50pct_recall": best["fpr"], "t_at_50pct_recall": best["T"],
            "recall_there": best["recall"]}


def summarise(y: np.ndarray, p: np.ndarray, label: str) -> Dict[str, object]:
    res: Dict[str, object] = {
        "model": label,
        "n": len(p),
        "auc": roc_auc(y, p),
        "ece": ece(y, p),
        "mean_real": float(p[y == 0].mean()),
        "mean_ai": float(p[y == 1].mean()),
    }
    res.update(best_operating_point(y, p))
    # Worst-case generator recall, which is what exposed the shipped model's weakness.
    return res


def load_scores(path: Path) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["p_detector"]) for r in rows])
    return rows, y, p


def per_generator(rows: List[dict], y: np.ndarray, p: np.ndarray) -> List[dict]:
    by_gen: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        if r["label"] != "1":
            continue
        by_gen.setdefault(r["generator"], []).append(i)
    out = []
    for gen in sorted(by_gen):
        idx = np.array(by_gen[gen])
        det = int((p[idx] >= 0.5).sum())
        lo, hi = wilson_ci(det, len(idx))
        out.append({"generator": gen, "n": len(idx),
                    "recall_at_0.5": det / len(idx),
                    "ci": [lo, hi], "mean": float(p[idx].mean())})
    return out


# --------------------------------------------------------------------------- #
# Scoring a candidate with transformers (build-time only)
# --------------------------------------------------------------------------- #

def resolve_ai_index(config, id2label: Dict[int, str]) -> int:
    """Finds the 'AI' class index from the checkpoint's own labels.

    Never hardcoded: the shipped model shipped with an inverted-index bug that silently
    flipped every score. This is the same guard, applied to whatever we consider next.
    """
    tokens = ("artificial", "ai", "fake", "generated", "synthetic", "deepfake")
    matches = [(int(k), str(v)) for k, v in id2label.items()
               if any(t in str(v).strip().lower() for t in tokens)]
    if len(matches) != 1:
        raise ValueError(
            f"cannot resolve a unique AI class: labels={id2label}, matched={matches}")
    return matches[0][0]


def score_candidate(repo: str, revision: str, root: Path, rows: List[dict],
                    batch: int = 16) -> np.ndarray:
    """Runs a HF image-classification checkpoint over every benchmark image."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    log(f"  loading {repo}@{revision[:7]}")
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForImageClassification.from_pretrained(repo, revision=revision)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    ai_index = resolve_ai_index(model.config, id2label)
    n_labels = len(id2label)
    log(f"  labels={id2label}  -> AI index {ai_index}  (n_labels={n_labels})")

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    out = np.zeros(len(rows), dtype=float)
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            images, keep = [], []
            for i, r in enumerate(chunk):
                try:
                    with Image.open(root / r["path"]) as im:
                        images.append(im.convert("RGB"))
                    keep.append(start + i)
                except Exception:
                    out[start + i] = np.nan
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt")
            logits = model(**inputs).logits
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)[:, 0]
            else:
                probs = torch.softmax(logits, dim=-1)[:, ai_index]
            for j, idx in enumerate(keep):
                out[idx] = float(probs[j])
            if (start // batch) % 5 == 0:
                log(f"    {min(start + batch, len(rows))}/{len(rows)}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="benchmark/scores.csv")
    ap.add_argument("--repo", default=None, help="score only this checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="score only N rows (faster smoke run)")
    ap.add_argument("--out", default="benchmark/detector_comparison.csv")
    args = ap.parse_args(argv)

    scores = Path(args.scores)
    if not scores.is_file():
        log(f"ERROR: {scores} not found")
        return 1
    rows, y, p_shipped = load_scores(scores)
    if args.limit:
        rows = rows[:args.limit]
        y, p_shipped = y[:args.limit], p_shipped[:args.limit]
    root = scores.parent

    log(f"benchmark: n={len(rows)}  real={(y == 0).sum()}  ai={(y == 1).sum()}")
    log("")
    log("=" * 78)
    log("SHIPPED MODEL (baseline)")
    log("=" * 78)
    base = summarise(y, p_shipped, "Organika/sdxl-detector [SHIPPED]")
    base.update(SHIPPED["Organika/sdxl-detector"])
    log(f"  AUC                        {base['auc']:.4f}")
    log(f"  ECE                        {base['ece']:.4f}")
    log(f"  FPR at 50% recall          {base['fpr_at_50pct_recall']*100:.1f}%")
    log(f"  license                    {base['license']}  <-- NON-COMMERCIAL")
    log("")
    log("  recall per generator at score>=0.5:")
    for g in per_generator(rows, y, p_shipped):
        log(f"    {g['generator']:24s} n={g['n']:3d}  {g['recall_at_0.5']*100:5.1f}%  "
            f"mean={g['mean']:.3f}")

    results = [base]

    targets = {args.repo: CANDIDATES.get(args.repo, {})} if args.repo else CANDIDATES
    for repo, meta in targets.items():
        log("")
        log("=" * 78)
        log(f"CANDIDATE {repo}   license={meta.get('license')}  {meta.get('note', '')}")
        log("=" * 78)
        try:
            p_new = score_candidate(repo, meta["revision"], root, rows)
        except Exception as exc:
            log(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        finite = np.isfinite(p_new)
        if finite.sum() < 10:
            log("  too few usable scores; skipping")
            continue
        s = summarise(y[finite], p_new[finite], repo)
        s.update(meta)
        results.append(s)
        log(f"  AUC                        {s['auc']:.4f}   "
            f"(shipped {base['auc']:.4f}, "
            f"{'BETTER' if s['auc'] > base['auc'] else 'worse'})")
        log(f"  ECE                        {s['ece']:.4f}   (shipped {base['ece']:.4f})")
        log(f"  FPR at 50% recall          {s['fpr_at_50pct_recall']*100:.1f}%   "
            f"(shipped {base['fpr_at_50pct_recall']*100:.1f}%)")
        log(f"  license                    {s['license']}")
        log("")
        log("  recall per generator at score>=0.5:")
        for g in per_generator([r for r, f in zip(rows, finite) if f], y[finite], p_new[finite]):
            log(f"    {g['generator']:24s} n={g['n']:3d}  {g['recall_at_0.5']*100:5.1f}%  "
                f"mean={g['mean']:.3f}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in results for k in r}))
        w.writeheader()
        for r in results:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()})
    log("")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())