"""Evaluates the detector on the held-out TEST split with honest, real numbers.

Reports the metrics ``architecture.md`` 9 asks for, and refuses to report anything it
cannot measure. Every proportion carries a Wilson 95% interval, because at this sample
size the interval is the finding: a point estimate alone would overstate what is known.

This script only *reads* ``scores.csv``. It never fits. Fitting happens in
``fit_fusion.py`` on the calibration split alone, so nothing here can leak back into the
thresholds being measured.

Usage
-----
    python scripts/evaluate.py --scores benchmark/scores.csv
    python scripts/evaluate.py --scores benchmark/scores.csv --markdown
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves
    sensibly at the small counts this benchmark produces (a per-generator cell of 11
    images). A normal-approximation interval on 11/11 would extend past 1.0.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the rank-sum identity, with tied scores given average ranks."""
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
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[float, List[dict]]:
    """Equal-width-bin ECE plus the per-bin detail, so the shape is inspectable."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: List[dict] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p == 1.0 is counted
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "mean_pred": None, "frac_ai": None, "gap": None})
            continue
        mean_pred = float(p[mask].mean())
        frac_ai = float(y[mask].mean())
        ece += (n / len(p)) * abs(frac_ai - mean_pred)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                     "mean_pred": round(mean_pred, 4),
                     "frac_ai": round(frac_ai, 4),
                     "gap": round(frac_ai - mean_pred, 4)})
    return float(ece), rows


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_rows(path: Path, split: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if split and r["split"] != split:
            continue
        out.append({
            "label": int(r["label"]),
            "generator": r.get("generator", ""),
            "perturbation": r.get("perturbation", "original"),
            # Scores files written by train_clip_probe.py carry a reduced column set
            # (they are out-of-fold scores, not a scoring pass), so optional columns are
            # read defensively rather than assumed.
            "source_dataset": r.get("source_dataset", ""),
            "split": r.get("split", ""),
            "p": float(r["p_detector"]),
        })
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def apply_policy(p: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """Maps fused/calibrated scores to verdicts under the shipped policy."""
    verdicts = np.full(len(p), "inconclusive", dtype=object)
    verdicts[p >= t_hi] = "likely_ai"
    verdicts[p <= t_lo] = "likely_real"
    return verdicts


def apply_calibration(p: np.ndarray, w1: float, w2: float, b: float) -> np.ndarray:
    """Applies the shipped logit-space fusion exactly as sahu65/services/ensemble.py does.

    The service compares T_lo/T_hi against the FUSED score, not the raw probe output.
    Thresholding the raw score here instead would put the evaluation on a different
    operating point than the one that ships, and would report an abstain rate that no
    user ever experiences.

    Only the clip weight is non-zero (w1 names the retired classifier), so this reduces to
    Platt scaling of the probe score, but it reads the weights from config rather than
    assuming, so it stays correct if the fusion changes.
    """
    eps = 1e-4
    cp = np.clip(p, eps, 1.0 - eps)
    z = np.log(cp / (1.0 - cp))
    z = w1 * z + w2 * z + b
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(rows: List[dict], t_lo: float, t_hi: float, t_lo_calibrated: bool,
             fusion: Optional[Dict[str, float]] = None) -> dict:
    fusion = fusion or {}
    w1 = float(fusion.get("w1", 0.0))
    w2 = float(fusion.get("w2", 1.0))
    b = float(fusion.get("b", 0.0))

    y = np.array([r["label"] for r in rows])
    raw = np.array([r["p_raw"] if "p_raw" in r else r["p"] for r in rows])
    # Thresholds are compared against the calibrated score, as the service does. Callers
    # may pre-calibrate (see main), in which case r["p"] already is that score.
    p = apply_calibration(raw, w1, w2, b) if "p_raw" not in rows[0] else np.array(
        [r["p"] for r in rows])
    verdicts = apply_policy(p, t_lo, t_hi)

    real_mask = y == 0
    ai_mask = y == 1

    # --- headline: accuracy on committed verdicts (abstentions excluded) ---
    committed = verdicts != "inconclusive"
    correct = ((verdicts == "likely_ai") & ai_mask) | ((verdicts == "likely_real") & real_mask)
    n_committed = int(committed.sum())
    n_correct = int(correct.sum())
    acc_committed = n_correct / n_committed if n_committed else float("nan")
    acc_lo, acc_hi = wilson_ci(n_correct, n_committed)

    # --- false positives on real photos: the number that matters most ---
    real_committed = real_mask & committed
    n_real_committed = int(real_committed.sum())
    n_fp = int((verdicts == "likely_ai").sum() - (ai_mask & committed & (verdicts == "likely_ai")).sum())
    n_fp = int(((verdicts == "likely_ai") & real_mask).sum())
    fpr = n_fp / n_real_committed if n_real_committed else float("nan")
    fpr_lo, fpr_hi = wilson_ci(n_fp, n_real_committed)

    # --- recall on AI images ---
    ai_committed = ai_mask & committed
    n_ai_committed = int(ai_committed.sum())
    n_tp = int(((verdicts == "likely_ai") & ai_mask).sum())
    recall = n_tp / n_ai_committed if n_ai_committed else float("nan")
    rec_lo, rec_hi = wilson_ci(n_tp, n_ai_committed)

    abstain = int((~committed).sum())
    abstain_rate = abstain / len(p) if len(p) else float("nan")

    ece, bins = expected_calibration_error(y, p)

    return {
        "n": len(p),
        "n_real": int(real_mask.sum()),
        "n_ai": int(ai_mask.sum()),
        "thresholds": {"T_lo": t_lo, "T_hi": t_hi, "fitted": t_lo_calibrated},
        "auc": roc_auc(y, p),
        "auc_uncalibrated": roc_auc(y, raw),
        "accuracy_committed": acc_committed,
        "accuracy_ci": [acc_lo, acc_hi],
        "n_committed": n_committed,
        "false_positive_rate_on_real": fpr,
        "fpr_ci": [fpr_lo, fpr_hi],
        "n_real_committed": n_real_committed,
        "n_false_positives": n_fp,
        "recall_ai": recall,
        "recall_ci": [rec_lo, rec_hi],
        "n_ai_committed": n_ai_committed,
        "n_true_positives": n_tp,
        "abstain_rate": abstain_rate,
        "n_abstain": abstain,
        "expected_calibration_error": ece,
        "calibration_bins": bins,
        "mean_score_real": float(p[real_mask].mean()) if real_mask.any() else None,
        "mean_score_ai": float(p[ai_mask].mean()) if ai_mask.any() else None,
    }


def group_breakdown(rows: List[dict], t_lo: float, t_hi: float) -> List[dict]:
    """Per-generator and per-perturbation recall, with intervals.

    Per-generator recall is where the interesting story lives: a detector can look fine
    on average while failing completely on one family.
    """
    out: List[dict] = []

    by_gen: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] == 1:
            by_gen[r["generator"]].append(r)
    for gen in sorted(by_gen):
        grp = by_gen[gen]
        p = np.array([r["p"] for r in grp])
        detected = int((p >= t_hi).sum())
        lo, hi = wilson_ci(detected, len(p))
        out.append({"group": gen, "kind": "generator", "n": len(p),
                    "detected": detected, "rate": detected / len(p),
                    "ci": [lo, hi], "mean_score": float(p.mean())})

    by_pert: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_pert[r["perturbation"]].append(r)
    for pert in sorted(by_pert):
        grp = by_pert[pert]
        p = np.array([r["p"] for r in grp])
        y = np.array([r["label"] for r in grp])
        # Both directions, so a perturbation that hurts real images as well as AI ones
        # cannot hide.
        det_ai = int(((p >= t_hi) & (y == 1)).sum())
        n_ai = int((y == 1).sum())
        fp = int(((p >= t_hi) & (y == 0)).sum())
        n_real = int((y == 0).sum())
        ai_lo, ai_hi = wilson_ci(det_ai, n_ai)
        fp_lo, fp_hi = wilson_ci(fp, n_real)
        out.append({"group": pert, "kind": "perturbation", "n": len(p),
                    "recall_ai": det_ai / n_ai if n_ai else None, "recall_ci": [ai_lo, ai_hi],
                    "fpr": fp / n_real if n_real else None, "fpr_ci": [fp_lo, fp_hi],
                    "mean_score": float(p.mean())})

    return out


def worst_false_positives(rows: List[dict], t_lo: float, t_hi: float, limit: int = 15) -> List[dict]:
    """The real photographs the detector scored most confidently as AI.

    Reported explicitly because a false positive is the failure that does real damage to
    a person being accused, and an aggregate FPR hides which images drive it.
    """
    reals = [r for r in rows if r["label"] == 0]
    reals.sort(key=lambda r: -r["p"])
    return [
        {"generator": r["generator"], "perturbation": r["perturbation"],
         "score": r["p"], "verdict": "likely_ai" if r["p"] >= t_hi else
                  ("likely_real" if r["p"] <= t_lo else "inconclusive")}
        for r in reals[:limit]
    ]


def render(metrics: dict, groups: List[dict], fps: List[dict], markdown: bool) -> str:
    L: List[str] = []
    t = metrics["thresholds"]
    verdict_note = "" if t["fitted"] else \
        "\n**Thresholds are UNFITTED PLACEHOLDERS.** These numbers describe the current\n" \
        "default policy, not a tuned one. See `fit_fusion.py`.\n"

    L.append("=" * 72)
    L.append(f"TEST SPLIT  n={metrics['n']}  "
             f"(real={metrics['n_real']}  ai={metrics['n_ai']})")
    L.append(f"thresholds  T_lo={t['T_lo']}  T_hi={t['T_hi']}  fitted={t['fitted']}")
    L.append("=" * 72)
    L.append(verdict_note.strip())
    L.append("")
    L.append(f"ROC AUC                          {metrics['auc']:.4f}")
    L.append(f"accuracy on committed verdicts   {metrics['accuracy_committed']*100:5.1f}%  "
             f"95% CI [{metrics['accuracy_ci'][0]*100:.1f}, {metrics['accuracy_ci'][1]*100:.1f}]  "
             f"(n={metrics['n_committed']})")
    L.append(f"FPR on real photos               {metrics['false_positive_rate_on_real']*100:5.1f}%  "
             f"95% CI [{metrics['fpr_ci'][0]*100:.1f}, {metrics['fpr_ci'][1]*100:.1f}]  "
             f"({metrics['n_false_positives']}/{metrics['n_real_committed']})")
    L.append(f"recall on AI                     {metrics['recall_ai']*100:5.1f}%  "
             f"95% CI [{metrics['recall_ci'][0]*100:.1f}, {metrics['recall_ci'][1]*100:.1f}]  "
             f"({metrics['n_true_positives']}/{metrics['n_ai_committed']})")
    L.append(f"abstain rate                     {metrics['abstain_rate']*100:5.1f}%  "
             f"({metrics['n_abstain']}/{metrics['n']})")
    L.append(f"expected calibration error       {metrics['expected_calibration_error']:.4f}")
    L.append(f"mean score  real={metrics['mean_score_real']:.4f}  "
             f"ai={metrics['mean_score_ai']:.4f}")
    L.append("")

    L.append("-" * 72)
    L.append("RECALL BY GENERATOR FAMILY  (this is where the model actually varies)")
    L.append("-" * 72)
    for g in [x for x in groups if x["kind"] == "generator"]:
        L.append(f"  {g['group']:26s} n={g['n']:3d}  {g['rate']*100:5.1f}%  "
                 f"CI [{g['ci'][0]*100:.1f}, {g['ci'][1]*100:.1f}]  mean={g['mean_score']:.3f}")
    L.append("")

    L.append("-" * 72)
    L.append("PER-PERTURBATION  (recall on AI | false positives on real)")
    L.append("-" * 72)
    for g in [x for x in groups if x["kind"] == "perturbation"]:
        r = f"{g['recall_ai']*100:5.1f}%" if g["recall_ai"] is not None else "  n/a"
        f_ = f"{g['fpr']*100:5.1f}%" if g["fpr"] is not None else "  n/a"
        L.append(f"  {g['group']:26s} n={g['n']:3d}  recall={r}  fpr={f_}  "
                 f"mean={g['mean_score']:.3f}")
    L.append("")

    L.append("-" * 72)
    L.append("HIGHEST-SCORING REAL PHOTOGRAPHS  (false-positive candidates)")
    L.append("-" * 72)
    for f in fps:
        L.append(f"  {f['score']:.4f}  {f['verdict']:14s} {f['generator']:18s} "
                 f"{f['perturbation']}")
    L.append("")

    if markdown:
        L.append("## Calibration bins")
        L.append("")
        L.append("| bin | n | mean predicted | fraction AI | gap |")
        L.append("|---|---|---|---|---|")
        for b in metrics["calibration_bins"]:
            if b["n"] == 0:
                L.append(f"| {b['bin']} | 0 | - | - | - |")
            else:
                L.append(f"| {b['bin']} | {b['n']} | {b['mean_pred']} | "
                         f"{b['frac_ai']} | {b['gap']} |")

    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="benchmark/scores.csv")
    p.add_argument("--split", default="test", choices=["test", "calibration", "all"])
    p.add_argument("--t-lo", type=float, default=None,
                   help="default: read from the fitted thresholds.json")
    p.add_argument("--t-hi", type=float, default=None)
    p.add_argument("--markdown", action="store_true", help="append a calibration table")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scores = Path(args.scores)
    if not scores.is_file():
        log(f"ERROR: {scores} not found. Run scripts/build_benchmark.py then "
            f"scripts/score_benchmark.py first.")
        return 1

    # Default to the thresholds actually shipped, so this reports the deployed policy.
    t_lo, t_hi, fitted = 0.35, 0.65, False
    fusion: Dict[str, float] = {}
    try:
        from sahu65.config import is_calibrated, load_fusion, load_thresholds

        th = load_thresholds("sahu65/config")
        t_lo = float(th.get("T_lo", t_lo))
        t_hi = float(th.get("T_hi", t_hi))
        fitted = is_calibrated("sahu65/config")
        fusion = load_fusion("sahu65/config")
    except Exception:
        pass
    if args.t_lo is not None:
        t_lo = args.t_lo
    if args.t_hi is not None:
        t_hi = args.t_hi

    rows = load_rows(scores, args.split)
    if not rows:
        log(f"ERROR: no rows for split={args.split}")
        return 1

    # Calibrate once, here, so evaluate(), group_breakdown() and worst_false_positives()
    # all see the same score the service thresholds. Calibrating inside only one of them
    # is how the per-generator table ends up disagreeing with the headline numbers.
    for r in rows:
        r["p_raw"] = r["p"]
    cal = apply_calibration(np.array([r["p"] for r in rows]),
                            float(fusion.get("w1", 0.0)),
                            float(fusion.get("w2", 1.0)),
                            float(fusion.get("b", 0.0)))
    for r, v in zip(rows, cal):
        r["p"] = float(v)

    metrics = evaluate(rows, t_lo, t_hi, fitted, fusion)
    groups = group_breakdown(rows, t_lo, t_hi)
    fps = worst_false_positives(rows, t_lo, t_hi)

    log(render(metrics, groups, fps, args.markdown))

    if args.json_out:
        payload = dict(metrics)
        payload["groups"] = groups
        payload["worst_false_positives"] = fps
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())