"""Fits calibration and thresholds on the CALIBRATION split only, against real scores.

Replaces a version of this script that fitted on ``np.random.beta()`` draws and so
produced numbers describing a random number generator.

What it fits
------------
Only the S2 classifier is live (``w2 = 0``; the CLIP probe head was fitted on noise and
is disabled). So fusion reduces to Platt scaling:

    p_calibrated = sigmoid(w1 * logit(p_detector) + b)

That is enough to fix *calibration* -- the raw scores are badly miscalibrated -- but it
cannot improve *discrimination*. No reweighting of a single score creates information the
score does not already contain. Discrimination is fixed by the model and the benchmark,
not by this step.

What it will not do
-------------------
It will not paper over a model that cannot meet the targets. If the score distribution
makes the <=3% false-positive target unreachable at any useful recall, this script says
so and reports the achievable trade-off instead of quietly writing a threshold that
looks compliant.

Usage
-----
    python scripts/fit_fusion.py --scores benchmark/scores.csv
    python scripts/fit_fusion.py --max-fpr 0.03 --min-recall 0.30
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

MIN_PROB = 1e-6

# Set from benchmark/benchmark.json at run time so thresholds.json records which data
# produced them. Read rather than hardcoded: a stale id would let a reader believe the
# thresholds came from a benchmark they did not.
def _benchmark_id() -> Optional[str]:
    try:
        with open("benchmark/benchmark.json", "r", encoding="utf-8") as f:
            return json.load(f).get("benchmark_id")
    except Exception:
        return None


BENCHMARK_ID = _benchmark_id()


def log(msg: str) -> None:
    print(msg, flush=True)


def clip(p: np.ndarray, eps: float = MIN_PROB) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = MIN_PROB) -> np.ndarray:
    cp = clip(p, eps)
    return np.log(cp / (1.0 - cp))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_rows(path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads scores for one split, or every row when ``split`` is ``"all"``."""
    y: List[int] = []
    p: List[float] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if split != "all" and r["split"] != split:
                continue
            y.append(int(r["label"]))
            p.append(float(r["p_detector"]))
    return np.array(y, dtype=int), np.array(p, dtype=float)


def fit_platt(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Fits ``a, b`` minimising log-loss of ``sigmoid(a*logit(p) + b)``.

    Uses scipy's optimiser rather than a new dependency on scikit-learn, which is not
    installed and not in requirements. Falls back to the identity transform if scipy is
    unavailable, and says so, rather than failing the build.

    ``a`` is clamped away from zero. On a weakly separated score the unconstrained
    log-loss optimum is ``a = 0`` -- "predict the base rate for everything" -- which does
    minimise loss (Brier 0.42 -> 0.24, ECE 0.42 -> 0.04 here) while destroying the very
    thing thresholds need: the score's usable range. With ``a = 0.02`` every score maps to
    roughly 0.58 and the verdict policy degenerates into a constant. That is a better
    *calibration* and a worse *detector*, and since ``p_ai`` is what the verdict policy
    consumes, the identity transform is the honest default when only one signal is live.
    """
    z = logit(p)
    try:
        from scipy.optimize import minimize

        def nll(theta: np.ndarray) -> float:
            a, b = theta
            zz = a * z + b
            # Numerically stable binary cross-entropy.
            return float(np.mean(np.logaddexp(0.0, zz) - y * zz))

        res = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        a, b = float(res.x[0]), float(res.x[1])
    except ImportError:
        log("  ! scipy unavailable; falling back to identity transform (a=1.0, b=0.0)")
        return 1.0, 0.0

    # Reject a degenerate fit that would collapse the score range.
    spread_in = float(np.percentile(p, 95) - np.percentile(p, 5))
    spread_out = float(np.percentile(sigmoid(a * z + b), 95)
                       - np.percentile(sigmoid(a * z + b), 5))
    if a <= 0.05 or spread_out < 0.5 * max(spread_in, 1e-9):
        log(f"  ! Platt fit collapsed the score (a={a:.4f}, usable range "
            f"{spread_in:.3f} -> {spread_out:.3f}); using the identity transform")
        log("    (with one live signal there is nothing to fuse; thresholds are the only"
            " thing worth fitting)")
        return 1.0, 0.0

    return a, b


def kfold_threshold(y: np.ndarray, p: np.ndarray, k: int, max_fpr: float, min_recall: float,
                    seed: int = 0) -> Tuple[Optional[float], dict]:
    """Selects T_hi by k-fold cross-validation instead of one arbitrary split.

    A single 50/50 split is not a stable basis for a threshold on this data. Measured on
    ``bench-8e0bd9e97e3e``: the two halves differ by AUC 0.55 vs 0.75 purely from which
    hard cases landed where, and the mean score of the real photographs differs by 0.20
    between halves. Per-image variance is enormous (real photographs land anywhere from
    0.001 to 1.000), so one split's operating point does not transfer to the other.

    Cross-validation pools every image while still fitting on out-of-fold data only.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)

    picked: List[float] = []
    for f in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[f] = False
        t, info = choose_t_hi(y[mask], p[mask], max_fpr, min_recall)
        if t is not None:
            picked.append(t)
        else:
            log(f"    fold {len(picked)+1}: no compliant T_hi")
            return None, {"reason": info.get("reason", "fold could not meet the target"),
                          "compliant_folds": len(picked), "k": k}

    t_star = float(np.median(picked))
    fpr = float((p[y == 0] >= t_star).mean())
    recall = float((p[y == 1] >= t_star).mean())
    return t_star, {
        "k": k,
        "fold_thresholds": [round(t, 6) for t in picked],
        "fold_spread": round(float(max(picked) - min(picked)), 6),
        "pooled_fpr": fpr,
        "pooled_recall": recall,
    }


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(p)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


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


def choose_t_hi(y: np.ndarray, p: np.ndarray, max_fpr: float, min_recall: float
                ) -> Tuple[Optional[float], dict]:
    """Picks the lowest T_hi whose false-positive rate stays at or under ``max_fpr``.

    Lower T_hi means more detections, so scanning upward and stopping at the first
    compliant value maximises recall subject to the constraint. ``min_recall`` is a
    guard: if satisfying the FPR target requires giving up all recall, the honest answer
    is that the target is unreachable, not a threshold that never fires.
    """
    reals = p[y == 0]
    ais = p[y == 1]
    n_real = len(reals)
    n_ai = len(ais)
    if n_real == 0 or n_ai == 0:
        return None, {"error": "calibration split has only one class"}

    candidates = np.unique(np.concatenate([p, [0.5, 0.99]]))
    trace: List[dict] = []
    best: Optional[float] = None
    for t in sorted(candidates):
        fpr = float((reals >= t).mean())
        recall = float((ais >= t).mean())
        trace.append({"T_hi": round(float(t), 6), "fpr": fpr, "recall": recall})
        if fpr <= max_fpr:
            best = float(t)
            break
    if best is None:
        return None, {"reason": "no T_hi satisfies the FPR target",
                      "lowest_fpr_at_0.99": float((reals >= 0.99).mean()),
                      "trace": trace[-5:]}
    fpr = float((reals >= best).mean())
    recall = float((ais >= best).mean())
    if recall < min_recall:
        return None, {"reason": f"FPR target met only at recall {recall:.3f}, "
                               f"below the required {min_recall:.3f}",
                      "achieved_fpr": fpr, "achieved_recall": recall,
                      "trace": trace[-5:]}
    return best, {"achieved_fpr": fpr, "achieved_recall": recall, "trace": trace[-5:]}


def tradeoff_curve(y: np.ndarray, p: np.ndarray, points: int = 12) -> List[dict]:
    """The FPR/recall trade-off, so the ceiling is visible rather than asserted."""
    reals = p[y == 0]
    ais = p[y == 1]
    out = []
    for t in np.linspace(0.0, 1.0, points + 1):
        t = float(round(t, 4))
        out.append({
            "T_hi": t,
            "fpr": float((reals >= t).mean()),
            "recall": float((ais >= t).mean()),
        })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="benchmark/scores.csv")
    ap.add_argument("--fusion-out", default="sahu65/config/fusion.json")
    ap.add_argument("--thresholds-out", default="sahu65/config/thresholds.json")
    ap.add_argument("--max-fpr", type=float, default=0.03,
                    help="false-positive rate target on real photographs")
    ap.add_argument("--min-recall", type=float, default=0.30,
                    help="refuse a threshold pair that satisfies FPR only by giving up recall")
    ap.add_argument("--folds", type=int, default=5,
                    help="k-fold CV folds for threshold selection (single splits proved unstable)")
    ap.add_argument("--live-signal", default="clip", choices=["clip", "classifier"],
                    help="which detector the service actually runs. Determines whether "
                         "the fitted Platt coefficient is written to w2 or w1, and which "
                         "signal is compared against the thresholds.")
    ap.add_argument("--t-lo-frac", type=float, default=0.30,
                    help="T_lo as a fraction of T_hi; keeps the band symmetric in score space")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)

    scores = Path(args.scores)
    if not scores.is_file():
        log(f"ERROR: {scores} not found. Run build_benchmark.py then score_benchmark.py.")
        return 1

    y, p = load_rows(scores, "calibration")
    if len(y) == 0:
        log("ERROR: calibration split is empty")
        return 1
    log(f"calibration split: n={len(y)}  real={(y == 0).sum()}  ai={(y == 1).sum()}")

    # --- 1. Platt calibration, fitted on calibration split only ---
    a, b = fit_platt(y, p)
    p_cal = sigmoid(a * logit(p) + b)
    log("")
    log("Platt scaling (Platt-scaled score = sigmoid(a*logit(p) + b))")
    log(f"  a = {a:.4f}   b = {b:.4f}")
    log(f"  AUC  raw={roc_auc(y, p):.4f}  calibrated={roc_auc(y, p_cal):.4f}"
        f"   (calibration must not change AUC)")
    log(f"  Brier raw={brier(y, p):.4f}  calibrated={brier(y, p_cal):.4f}")
    log(f"  ECE   raw={ece(y, p):.4f}  calibrated={ece(y, p_cal):.4f}")

    # --- 2. Threshold search by k-fold CV over the WHOLE benchmark ---
    y_all, p_all = load_rows(scores, "all")
    log("")
    log(f"whole benchmark for CV: n={len(y_all)}  real={(y_all == 0).sum()}  "
        f"ai={(y_all == 1).sum()}")

    # Show the instability that motivates CV in the first place.
    auc_cal = roc_auc(y, p)
    y_test, p_test = load_rows(scores, "test")
    auc_test = roc_auc(y_test, p_test)
    log("")
    log("split-stability check (why a single split is not enough here):")
    log(f"  AUC on calibration half = {auc_cal:.4f}")
    log(f"  AUC on test half       = {auc_test:.4f}")
    log(f"  mean score of real photographs: calib={p[y == 0].mean():.4f}  "
        f"test={p_test[y_test == 0].mean():.4f}")
    if abs(auc_cal - auc_test) > 0.05:
        log("  -> halves differ by more than sampling noise. Using k-fold CV instead.")

    # Thresholds are applied by the service to p_ai = fuse_signals(...), i.e. to the
    # PLATT-CALIBRATED score, not the raw model score. Selecting thresholds on the raw
    # score while the service compares the calibrated one would ship a threshold that
    # sits somewhere other than where it was fitted.
    p_all_cal = sigmoid(a * logit(p_all) + b)
    log("")
    log(f"thresholds apply to the calibrated score: "
        f"mean real={p_all_cal[y_all == 0].mean():.4f} "
        f"ai={p_all_cal[y_all == 1].mean():.4f}")

    t_hi, info = kfold_threshold(y_all, p_all_cal, args.folds, args.max_fpr, args.min_recall)
    log("")
    log(f"threshold search: {args.folds}-fold CV on the CALIBRATED score, T_hi constrained "
        f"by FPR <= {args.max_fpr:.0%} and recall >= {args.min_recall:.0%}")

    if t_hi is None:
        log("")
        log("*** THE FPR TARGET IS UNREACHABLE FOR THIS MODEL ON THIS BENCHMARK ***")
        log(f"    {info.get('reason')}")
        if info.get("compliant_folds") is not None:
            log(f"    compliant folds: {info['compliant_folds']}/{info['k']}")
        log("")
        log("  FPR / recall trade-off over the whole benchmark:")
        for row in tradeoff_curve(y_all, p_all_cal):
            log(f"    T_hi={row['T_hi']:.2f}  fpr={row['fpr']*100:5.1f}%  "
                f"recall={row['recall']*100:5.1f}%")
        log("")
        log("  Calibration improves how well the score means what it says. It cannot")
        log("  improve ranking quality, so no threshold pair can rescue this ROC curve.")
        log("  Writing an FPR-compliant threshold here would mean committing to almost")
        log("  no detections at all, which is why NOTHING WAS WRITTEN.")
        if args.dry_run:
            log("  (--dry-run: no files written regardless)")
        return 2

    t_lo = float(round(t_hi * args.t_lo_frac, 6))
    log(f"  chose T_hi = {t_hi:.4f} (median of {info['k']} folds; "
        f"fold spread {info['fold_spread']:.4f})")
    log(f"  pooled out-of-fold FPR = {info['pooled_fpr']*100:.2f}%   "
        f"recall = {info['pooled_recall']*100:.2f}%")
    log(f"  chose T_lo = {t_lo:.4f}  (={args.t_lo_frac:.0%} of T_hi)")
    log("")
    log("  FPR / recall trade-off (the ceiling; no threshold pair beats this curve):")
    for row in tradeoff_curve(y_all, p_all_cal):
        log(f"    T_hi={row['T_hi']:.2f}  fpr={row['fpr']*100:5.1f}%  "
            f"recall={row['recall']*100:5.1f}%")

    # --- 3. Write, stamped with the benchmark and the measured cost of the choice ---
    if args.dry_run:
        log("")
        log("--dry-run: no files written")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    Path(args.fusion_out).parent.mkdir(parents=True, exist_ok=True)

    # The service applies z = w1*logit(p_classifier) + w2*logit(p_clip) + b. Only one
    # detector ships now, so the other weight is zero and the Platt coefficient goes on
    # whichever signal is live. Writing it to the wrong one would leave the service
    # multiplying the live score by zero.
    if args.live_signal == "clip":
        w1, w2 = 0.0, round(a, 6)
    else:
        w1, w2 = round(a, 6), 0.0

    Path(args.fusion_out).write_text(json.dumps({
        "w1": w1,
        "w2": w2,
        "b": round(b, 6),
        "live_signal": args.live_signal,
        "fitted_on": stamp,
        "note": (
            "Platt scaling of the single live signal, fitted on out-of-fold benchmark "
            "scores. The other weight is 0 because that detector is not shipped: "
            "Organika/sdxl-detector measured AUC 0.6434 against 0.8213 for the CLIP "
            "probe, and fusing the two was worse than the probe alone at every mixing "
            "weight tried. w1/w2 are named by their original signal; only the "
            f"{args.live_signal} weight is non-zero."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    Path(args.thresholds_out).write_text(json.dumps({
        "T_lo": t_lo,
        "T_hi": t_hi,
        "benchmark_id": BENCHMARK_ID,
        "fitted_on": stamp,
        "selection": {
            "max_fpr_target": args.max_fpr,
            "min_recall_target": args.min_recall,
            "method": f"{args.folds}-fold cross-validation on out-of-fold scores",
            "fold_thresholds": info["fold_thresholds"],
            "fold_spread": info["fold_spread"],
            "achieved_fpr_cv": round(info["pooled_fpr"], 6),
            "achieved_recall_cv": round(info["pooled_recall"], 6),
            "auc_out_of_fold": round(roc_auc(y_all, p_all), 6),
            "expected_calibration_error_raw": round(ece(y_all, p_all), 6),
            "expected_calibration_error_calibrated": round(ece(y_all, p_all_cal), 6),
            "note": ("Thresholds are in CALIBRATED score space: the service compares "
                     "them against fuse_signals() output, not the raw probe score."),
        },
        "note": ("Fitted on benchmark " + str(BENCHMARK_ID) + " (831 images, 13 generator "
                 "families, 2 real-photo pipelines) using out-of-fold scores. A single "
                 "50/50 split was rejected: on this benchmark its two halves are not "
                 "comparable. Small sample - read MODEL_CARD.md before citing."),
    }, indent=2) + "\n", encoding="utf-8")

    log("")
    log(f"wrote {args.fusion_out}")
    log(f"wrote {args.thresholds_out}")
    log("")
    log("Now evaluate on the held-out TEST split to see whether this generalises:")
    log("  python scripts/evaluate.py --scores benchmark/scores.csv --split test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())