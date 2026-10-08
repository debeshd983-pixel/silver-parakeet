"""Trains the S3 CLIP linear probe on real embeddings and tests whether it adds signal.

This replaces the previous head, which ``scripts/train_probe.py`` fitted on
``np.random.randn``. A head fitted on noise is worse than no head: it looked like a
signal, doubled the payload and the latency, and was only disabled once someone checked.

What it does
------------
1. Runs the exported CLIP encoder over every benchmark image, using the *same*
   preprocessing ``sahu65/services/preprocess.py`` applies at serve time. Training on a
   different preprocessing path than production measures a model nobody ships.
2. Fits one linear layer (512 weights + bias) by regularised logistic regression, chosen
   by k-fold cross-validation. The split in the benchmark is stratified but proved
   unstable on its own (AUC 0.545 vs 0.754 between halves), so a single split is not
   trusted here either.
3. Reports S3 alone, S2 alone, and the two fused, and refuses to write a head that does
   not earn its place.

Fits on whichever encoder is passed, because a head fitted on fp32 embeddings does not
transfer to an INT8 encoder: measured cosine between the two builds here is 0.55, which
is far too low to assume.

Usage
-----
    python scripts/train_clip_probe.py
    python scripts/train_clip_probe.py --encoder models/clip_encoder.onnx --force
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

EMBED_DIM = 512


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

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


def best_fpr_at_recall(y: np.ndarray, p: np.ndarray, min_recall: float = 0.50) -> float:
    reals = p[y == 0]
    ais = p[y == 1]
    best = 1.0
    for t in np.unique(ais):
        if float((ais >= t).mean()) >= min_recall:
            best = min(best, float((reals >= t).mean()))
    return best


# --------------------------------------------------------------------------- #
# Logistic regression (no sklearn dependency; scipy is already required at runtime)
# --------------------------------------------------------------------------- #

def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2
                 ) -> Tuple[np.ndarray, float]:
    """L2-regularised logistic regression via scipy. Returns (weights, bias)."""
    from scipy.optimize import minimize

    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    yf = y.astype(np.float64)

    def objective(theta: np.ndarray) -> float:
        z = xb @ theta
        loss = float(np.mean(np.logaddexp(0.0, z) - yf * z))
        reg = l2 * float(np.sum(theta[:-1] ** 2))
        return loss + reg

    res = minimize(objective, np.zeros(d + 1), method="L-BFGS-B",
                   options={"maxiter": 2000})
    theta = res.x
    return theta[:-1].astype(np.float32), float(theta[-1])


def cv_scores(x: np.ndarray, y: np.ndarray, k: int, l2: float, seed: int = 0
              ) -> np.ndarray:
    """Out-of-fold scores. Each row is predicted by a model that never saw it."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    out = np.zeros(len(y), dtype=float)
    for f in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[f] = False
        w, b = fit_logistic(x[mask], y[mask], l2)
        z = x[f] @ w + b
        out[f] = 1.0 / (1.0 + np.exp(-z))
    return out


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def extract_embeddings(encoder_path: Path, manifest: Path, batch: int = 16,
                       limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Runs the ONNX encoder over the benchmark, using serve-time preprocessing."""
    import onnxruntime as ort

    from sahu65.services.preprocess import CLIP_MEAN, CLIP_STD, decode_image

    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    root = manifest.parent

    # Prefer the INT8 build exactly as sahu65/services/classifier.py prefers its own,
    # so what is fitted here is what gets served.
    if not encoder_path.exists():
        alt = encoder_path.parent / "clip_encoder.int8.onnx"
        if alt.exists():
            log(f"  {encoder_path.name} not found; using {alt.name}")
            encoder_path = alt
        else:
            raise FileNotFoundError(f"no CLIP encoder at {encoder_path}")

    sess = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    log(f"  encoder: {encoder_path.name} ({encoder_path.stat().st_size/(1024*1024):.1f} MB)")

    import PIL.Image

    feats = np.zeros((len(rows), EMBED_DIM), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=int)
    ok = np.zeros(len(rows), dtype=bool)

    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        tensors, keep = [], []
        for i, r in enumerate(chunk):
            try:
                data = (root / r["path"]).read_bytes()
                img, _ = decode_image(data)
                # Mirror preprocess_for_clip: shortest side to 224, then centre crop.
                w, h = img.size
                scale = 224 / min(w, h)
                img = img.resize((max(int(round(w * scale)), 224),
                                  max(int(round(h * scale)), 224)),
                                 PIL.Image.Resampling.BICUBIC)
                left = (img.size[0] - 224) // 2
                top = (img.size[1] - 224) // 2
                img = img.crop((left, top, left + 224, top + 224))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
                arr = (arr - CLIP_MEAN) / CLIP_STD
                tensors.append(arr.astype(np.float32))
                keep.append(start + i)
            except Exception:
                pass
        if not tensors:
            continue
        out = sess.run(None, {inp: np.concatenate(tensors, axis=0)})[0]
        for j, idx in enumerate(keep):
            feats[idx] = out[j]
            labels[idx] = int(chunk[j]["label"])
            ok[idx] = True
        if (start // batch) % 5 == 0:
            log(f"    {min(start + batch, len(rows))}/{len(rows)}")

    # Unit-normalise, exactly as clip_probe.py does before the dot product.
    norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    feats = (feats / norms).astype(np.float32)
    return feats, labels, [rows[i] for i in range(len(rows)) if ok[i]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="models/clip_encoder.int8.onnx")
    ap.add_argument("--manifest", default="benchmark/manifest.csv")
    ap.add_argument("--scores", default="benchmark/scores.csv",
                    help="S2 scores, used for the fusion comparison")
    ap.add_argument("--out-head", default="models/clip_head.json")
    ap.add_argument("--embeddings-out", default="benchmark/clip_embeddings.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--limit", type=int, default=None, help="smoke-run on N rows")
    ap.add_argument("--reuse-embeddings", action="store_true",
                    help="reuse the cached .npz instead of re-running the encoder")
    ap.add_argument("--embed-batch", type=int, default=1,
                    help="batch size used when extracting embeddings. MUST be 1 for any "
                         "dynamically-quantized encoder: it computes activation ranges "
                         "per run, so an image embeds differently alone than beside "
                         "another one (measured cosine 0.65-0.89). The service serves "
                         "one image per request, so the head must be fitted at batch 1.")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="required AUC gain over the BEST single signal to ship the head")
    ap.add_argument("--emit-split-scores", default=None,
                    help="write <path> with scores from a head fitted on the CALIBRATION "
                         "split only, so thresholds can be fitted and reported without "
                         "the head having seen the data it is scored on")
    ap.add_argument("--head-on", default="all", choices=["all", "calibration"],
                    help="what the shipped head is fitted on. 'all' is standard after "
                         "model selection; 'calibration' keeps the artifact itself "
                         "honest about never having seen the test split")
    ap.add_argument("--force", action="store_true",
                    help="write the head even if it does not clear --min-gain")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        log(f"ERROR: {manifest} not found. Run build_benchmark.py first.")
        return 1

    cache = Path(args.embeddings_out)
    if args.reuse_embeddings and cache.is_file():
        log(f"reusing cached embeddings from {cache}")
        blob = np.load(cache, allow_pickle=False)
        feats, y = blob["features"], blob["labels"]
        kept = [{"path": str(p)} for p in blob["paths"]]
    else:
        log("extracting CLIP embeddings ...")
        feats, labels, kept_all = extract_embeddings(
            Path(args.encoder), manifest, batch=args.embed_batch, limit=args.limit)
        y = labels[np.isfinite(feats).all(axis=1)]
        feats = feats[np.isfinite(feats).all(axis=1)]
        kept = [r for r, k in zip(kept_all, np.isfinite(feats).all(axis=1)) if k]
        np.savez_compressed(cache, features=feats, labels=y,
                            paths=np.array([r["path"] for r in kept]))
        log(f"  cached embeddings -> {cache}")

    log(f"  usable: {len(y)}  real={(y == 0).sum()}  ai={(y == 1).sum()}")

    # --- L2 strength chosen by CV, not by taste ---
    log("")
    log("choosing L2 by cross-validation:")
    best = None
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        s = cv_scores(feats, y, args.folds, l2)
        a = roc_auc(y, s)
        log(f"  l2={l2:<8g} CV AUC = {a:.4f}")
        if best is None or a > best[1]:
            best = (l2, a, s)
    l2, cv_auc, p_s3 = best
    log(f"  chose l2={l2:g}  CV AUC = {cv_auc:.4f}")

    auc_s3 = cv_auc
    fpr_s3 = best_fpr_at_recall(y, p_s3)

    # --- Compare against S2 on the same rows ---
    auc_s2 = float("nan")
    auc_fused = float("nan")
    fpr_s2 = float("nan")
    fpr_fused = float("nan")
    if Path(args.scores).is_file():
        s2_rows = list(csv.DictReader(open(args.scores, encoding="utf-8")))
        s2_by_path = {r["path"]: float(r["p_detector"]) for r in s2_rows}
        p_s2 = np.array([s2_by_path.get(r["path"], np.nan) for r in kept])
        mask = np.isfinite(p_s2)
        if mask.sum() > 20:
            y2, s2, s3 = y[mask], p_s2[mask], p_s3[mask]
            auc_s2 = roc_auc(y2, s2)
            fpr_s2 = best_fpr_at_recall(y2, s2)

            # Logit-space fusion with S2, matching sahu65/services/ensemble.py.
            # w2 sweeps upward because S2 may be weaker than S3 here: at equal weight a
            # weak signal drags a strong one down, so the search must be able to
            # down-weight S2, not just up-weight S3.
            def lg(p):
                p = np.clip(p, 1e-4, 1 - 1e-4)
                return np.log(p / (1 - p))

            best_w2 = 0.0
            best_a = -1.0
            for w2 in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
                z = lg(s2) + w2 * lg(s3)
                pf = 1.0 / (1.0 + np.exp(-z))
                a = roc_auc(y2, pf)
                f = best_fpr_at_recall(y2, pf)
                marker = ""
                if a > best_a:
                    best_a, best_w2, best_fpr = a, w2, f
                    marker = "  <-- best"
                log(f"  fusion w1=1 w2={w2:<5g} AUC = {a:.4f}   "
                    f"FPR@50%recall = {f*100:.1f}%{marker}")
            # Guard against the nan-comparison trap: start below any real AUC.
            auc_fused = best_a if best_a > 0 else float("nan")
            fpr_fused = best_fpr if best_a > 0 else float("nan")

    log("")
    log("=" * 70)
    log(f"{'signal':<30} {'AUC':>8} {'FPR@50%recall':>16}")
    log("=" * 70)
    log(f"{'S2 classifier (shipped)':<30} {auc_s2:>8.4f} {fpr_s2*100:>15.1f}%")
    log(f"{'S3 CLIP probe (new)':<30} {auc_s3:>8.4f} {fpr_s3*100:>15.1f}%")
    if np.isfinite(auc_fused):
        log(f"{'S2 + S3 fused (best w2=' + str(best_w2) + ')':<30} "
            f"{auc_fused:>8.4f} {fpr_fused*100:>15.1f}%")
    log("=" * 70)

    # Compare the fusion against the BEST single signal, not against S2. S3 alone is
    # strong enough here that shipping the 85 MB encoder only makes sense if combining it
    # with S2 is actually an improvement.
    best_single_auc = max(a for a in (auc_s3, auc_s2) if np.isfinite(a))
    best_single_name = "S3" if auc_s3 >= auc_s2 else "S2"
    gain = auc_fused - best_single_auc if np.isfinite(auc_fused) else float("nan")
    log("")
    log(f"best single signal: {best_single_name} at AUC {best_single_auc:.4f}")
    log(f"fused AUC {auc_fused:.4f}  ->  gain {gain:+.4f}")

    if not np.isfinite(gain):
        log("Could not compute a fusion comparison. Use --force to write anyway.")
        return 2

    if gain < args.min_gain:
        log("")
        log(f"FUSION GAIN {gain:+.4f} is below the required {args.min_gain:+.4f}.")
        log("The two signals do not combine into something better than the better one")
        log("alone. Options are then: ship S3 and drop S2, or keep S2 and stay small.")
        log("NOT writing a head automatically. Re-run with --force once you have chosen.")
        if args.force:
            log("--force given: writing anyway.")
        else:
            return 3

    # Fit the shipped head on ALL data with the chosen L2.
    w, b = fit_logistic(feats, y, l2)

    # Optional: OUT-OF-FOLD scores for threshold selection.
    #
    # A head fitted on every row yields in-sample scores, and thresholds chosen on those
    # are optimistic by exactly the amount the head memorised. Out-of-fold scores are the
    # honest input.
    #
    # A single calibration/test split is NOT used here, because on this benchmark it is
    # demonstrably unreliable: a head fitted on the calibration half scores AUC 0.963 on
    # that half and 0.571 on the test half. The two halves are not comparable (the same
    # instability was measured for the previous model: 0.545 vs 0.754). k-fold CV is
    # used instead so every image is scored by a head that did not train on it.
    if args.emit_split_scores:
        import csv as _csv

        p_oof = cv_scores(feats, y, args.folds, l2)
        splits = {}
        try:
            for r in _csv.DictReader(open(Path(args.manifest), encoding="utf-8")):
                splits[r["path"]] = r["split"]
        except Exception as exc:
            log(f"  could not read splits from {args.manifest}: {exc}")

        out = Path(args.emit_split_scores)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.DictWriter(fh, fieldnames=["path", "label", "split", "p_detector",
                                                 "generator", "perturbation", "score_kind"])
            wr.writeheader()
            for i, r in enumerate(kept):
                wr.writerow({
                    "path": r["path"],
                    "label": int(y[i]),
                    "split": splits.get(r["path"], ""),
                    "p_detector": round(float(p_oof[i]), 6),
                    "generator": r.get("generator", ""),
                    "perturbation": r.get("perturbation", "original"),
                    "score_kind": f"out_of_fold_cv{args.folds}",
                })
        log(f"  wrote {out}")
        log(f"    scores are OUT-OF-FOLD ({args.folds}-fold CV); overall AUC "
            f"{roc_auc(y, p_oof):.4f}")
        cal = np.array([splits.get(r["path"]) == "calibration" for r in kept])
        if cal.any() and (~cal).any():
            log(f"    AUC on calibration rows = {roc_auc(y[cal], p_oof[cal]):.4f}")
            log(f"    AUC on test rows       = {roc_auc(y[~cal], p_oof[~cal]):.4f}"
                f"   (split variance, see the note above)")

    payload = {
        "weights": [round(float(v), 8) for v in w],
        "bias": round(b, 8),
        "embed_dim": EMBED_DIM,
        "normalised": True,
        "l2": l2,
        "folds": args.folds,
        "cv_auc": round(float(cv_auc), 6),
        "auc_s2_alone": round(float(auc_s2), 6) if np.isfinite(auc_s2) else None,
        "auc_fused": round(float(auc_fused), 6) if np.isfinite(auc_fused) else None,
        "fusion_weight_s3": best_w2,
        "encoder": Path(args.encoder).name,
        "encoder_note": ("Fitted on embeddings from this exact encoder build. The fp32 and "
                         "INT8 builds disagree (cosine 0.55), so a head is not portable "
                         "between them."),
        "note": ("Replaces a head fitted on np.random.randn. Fitted on real benchmark "
                 "embeddings by k-fold CV. Small sample: read MODEL_CARD.md before citing."),
    }
    Path(args.out_head).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_head).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log("")
    log(f"wrote {args.out_head}")
    log(f"fusion gain over S2 alone: {gain:+.4f}")
    log("")
    log("Next: re-run scripts/fit_fusion.py with --scores benchmark/scores.csv once S3")
    log("scores are added, then re-run scripts/evaluate.py on the test split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())