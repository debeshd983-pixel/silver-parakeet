"""Ensemble fusion, threshold application, verdict policy, and confidence calculation."""
import numpy as np
from typing import List, Literal, Tuple

VerdictType = Literal["likely_ai", "likely_real", "inconclusive", "ai_generated_verified"]
ConfidenceType = Literal["high", "medium", "low"]


def clip_prob(p: float, eps: float = 1e-4) -> float:
    """Clips probability to [eps, 1 - eps] to prevent logit explosion."""
    return float(np.clip(p, eps, 1.0 - eps))


def logit(p: float, eps: float = 1e-4) -> float:
    """Computes logit (log-odds) of probability."""
    cp = clip_prob(p, eps)
    return float(np.log(cp / (1.0 - cp)))


def sigmoid(z: float) -> float:
    """Computes sigmoid of log-odds."""
    return float(1.0 / (1.0 + np.exp(-z)))


def fuse_signals(p_cls: float, p_clip: float, w1: float = 1.0, w2: float = 1.0, b: float = 0.0) -> float:
    """Fuses S2 and S3 probabilities in logit space (Platt-style logistic regression).

    z = w1 * logit(p_cls) + w2 * logit(p_clip) + b
    p_ai = sigmoid(z)
    """
    z1 = logit(p_cls)
    z2 = logit(p_clip)
    z = (w1 * z1) + (w2 * z2) + b
    return float(np.clip(sigmoid(z), 0.0, 1.0))


def compute_verdict(
    p_ai: float,
    t_lo: float = 0.35,
    t_hi: float = 0.65,
    c2pa_ai_declared: bool = False,
) -> Tuple[VerdictType, float]:
    """Applies verdict policy based on C2PA manifest or fused probability and thresholds.

    Returns: (verdict, final_p_ai)
    """
    if c2pa_ai_declared:
        return "ai_generated_verified", 0.99

    if p_ai >= t_hi:
        return "likely_ai", p_ai
    elif p_ai <= t_lo:
        return "likely_real", p_ai
    else:
        return "inconclusive", p_ai


def compute_confidence(
    p_ai: float,
    p_cls: float,
    p_clip: float,
    verdict: VerdictType,
    t_lo: float = 0.35,
    t_hi: float = 0.65,
    warnings: List[str] = None,
) -> ConfidenceType:
    """Derives confidence from threshold distance and signal agreement.

    - High: signals agree (both >= 0.5 or both < 0.5) and fused prob is well beyond threshold.
    - Medium: signals agree, or moderate distance.
    - Low: signals disagree or in inconclusive band.
    """
    if warnings is None:
        warnings = []

    if verdict == "ai_generated_verified":
        return "high"

    if verdict == "inconclusive":
        return "low"

    # Check signal agreement: both indicate fake (>= 0.5) or both indicate real (< 0.5)
    signals_agree = (p_cls >= 0.5 and p_clip >= 0.5) or (p_cls < 0.5 and p_clip < 0.5)

    # Distance to threshold
    if verdict == "likely_ai":
        margin = p_ai - t_hi
    else:  # likely_real
        margin = t_lo - p_ai

    if signals_agree and margin >= 0.15:
        base_confidence = "high"
    elif signals_agree or margin >= 0.10:
        base_confidence = "medium"
    else:
        base_confidence = "low"

    # Downgrade if low_resolution or severe warning
    if "low_resolution" in warnings:
        if base_confidence == "high":
            base_confidence = "medium"
        elif base_confidence == "medium":
            base_confidence = "low"

    return base_confidence
